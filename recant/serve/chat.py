from __future__ import annotations
import json
import re
import uuid
from ..data.render import Renderer
TOOL_OPEN, TOOL_CLOSE, THINK_CLOSE = ('<tool_call>', '</tool_call>', '</think>')

class ChatRenderer:

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.r = Renderer(tokenizer)
        eot = tokenizer.token_to_id('<|endoftext|>')
        self.stop_ids = {self.r.IM_END} | ({eot} if eot is not None else set())

    def render(self, messages: list[dict], tools: list[dict] | None=None, enable_thinking: bool=True) -> list[int]:
        msgs = [dict(m, role='system') if m.get('role') == 'developer' else m for m in messages]
        tool_json = [json.dumps(t, ensure_ascii=False) for t in tools] if tools else None
        ids = self.r.render(msgs, tool_json, 'openhands').ids.tolist()
        r = self.r
        items = [r.IM_START, 'assistant\n', r.THINK, '\n'] if enable_thinking else [r.IM_START, 'assistant\n', r.THINK, '\n\n', r.THINK_END, '\n\n']
        return ids + r.encode_items(items)[0].tolist()

    def encode_text(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

class IncDecoder:

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read = (tok, [], 0, 0)

    def push(self, tid: int) -> str:
        self.ids.append(tid)
        old = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        new = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if len(new) > len(old) and (not new.endswith('�')):
            self.prefix, self.read = (self.read, len(self.ids))
            return new[len(old):]
        return ''

    def flush(self) -> str:
        new = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        old = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        self.prefix = self.read = len(self.ids)
        return new[len(old):]

def _partial(buf: str, markers) -> int:
    best = 0
    for m in markers:
        for k in range(min(len(m) - 1, len(buf)), best, -1):
            if buf.endswith(m[:k]):
                best = k
                break
    return best

class StreamParser:

    def __init__(self, thinking: bool, stop: list[str] | None=None):
        self.mode = 'reasoning' if thinking else 'content'
        self.stop = [s for s in stop or [] if s]
        self.buf = ''
        self.call_buf = ''
        self.in_call = False
        self.lead = not thinking
        self.calls: list[str] = []
        self.reasoning, self.content = ('', '')
        self.stopped = False

    def _emit(self, kind: str, text: str, out: list) -> None:
        if kind == 'content' and self.lead:
            text = text.lstrip('\n')
            if not text:
                return
            self.lead = False
        if text:
            out.append((kind, text))
            if kind == 'reasoning':
                self.reasoning += text
            else:
                self.content += text

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self.buf += text
        while not self.stopped:
            if self.in_call:
                self.call_buf += self.buf
                self.buf = ''
                i = self.call_buf.find(TOOL_CLOSE)
                if i < 0:
                    break
                self.calls.append(self.call_buf[:i + len(TOOL_CLOSE)])
                self.buf, self.call_buf, self.in_call, self.lead = (self.call_buf[i + len(TOOL_CLOSE):], '', False, True)
                continue
            markers = [THINK_CLOSE] if self.mode == 'reasoning' else [TOOL_OPEN] + self.stop
            hits = [(self.buf.find(m), m) for m in markers if self.buf.find(m) >= 0]
            if hits:
                i, m = min(hits)
                head = self.buf[:i]
                if self.mode == 'reasoning':
                    self._emit('reasoning', head.rstrip('\n'), out)
                    self.mode, self.lead, self.buf = ('content', True, self.buf[i + len(m):])
                elif m == TOOL_OPEN:
                    self._emit('content', head.rstrip('\n'), out)
                    self.in_call, self.buf = (True, self.buf[i:])
                else:
                    self._emit('content', head, out)
                    self.stopped, self.buf = (True, '')
                continue
            k = _partial(self.buf, markers)
            body = self.buf[:len(self.buf) - k]
            safe = body.rstrip('\n')
            self._emit(self.mode, safe, out)
            self.buf = body[len(safe):] + self.buf[len(body):]
            break
        return out

    def finish(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.stopped:
            return out
        rest = self.call_buf + self.buf if self.in_call else self.buf
        kind = 'content' if self.in_call else self.mode
        self._emit(kind, rest.rstrip('\n'), out)
        self.buf = self.call_buf = ''
        self.in_call = False
        return out
_FN = re.compile('<function=([^>\\n]+)>\\s*(.*?)\\s*</function>', re.S)
_PARAM = re.compile('<parameter=([^>\\n]+)>\\n?(.*?)\\n?</parameter>', re.S)

def _coerce(value: str, spec: dict | None):
    types = spec.get('type') if spec else None
    types = [types] if isinstance(types, str) else types or []
    if 'string' in types or (spec is not None and (not types)):
        return value
    try:
        if 'integer' in types:
            return int(value.strip())
        if 'number' in types:
            v = float(value.strip())
            return int(v) if v.is_integer() and '.' not in value else v
        if 'boolean' in types:
            return value.strip().lower() == 'true'
        if 'null' in types and value.strip() == 'null':
            return None
        if 'object' in types or 'array' in types:
            return json.loads(value)
    except ValueError:
        return value
    v = value.strip()
    if v[:1] in ('{', '['):
        try:
            return json.loads(v)
        except ValueError:
            pass
    return value

def parse_tool_call(raw: str, tools: list[dict] | None) -> dict | None:
    m = _FN.search(raw)
    if not m:
        return None
    name, body = (m.group(1).strip(), m.group(2))
    props = {}
    for t in tools or []:
        fn = t.get('function', t)
        if fn.get('name') == name:
            props = (fn.get('parameters') or {}).get('properties') or {}
    args = {k.strip(): _coerce(v, props.get(k.strip())) for k, v in _PARAM.findall(body)}
    return {'id': 'call_' + uuid.uuid4().hex[:24], 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}