from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
from .commit import classify_turn, parse_args
TOOLS_TAIL = '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>'
TURN_STRIDE = 8
T_START, T_CE_START, T_CE_END, T_CALL_START, T_CALL_END, T_ANCHOR, T_FLAGS, T_NCALLS = range(8)
F_COMMIT, F_FINISH, F_REASONING = (1, 2, 4)

class RenderError(ValueError):

@dataclass
class Rendered:
    ids: np.ndarray
    turns: np.ndarray
    unknown_tools: list[str]

class Renderer:

    def __init__(self, tokenizer, cache_size: int=64):
        self.tok = tokenizer
        tid = tokenizer.token_to_id
        self.IM_START, self.IM_END = (tid('<|im_start|>'), tid('<|im_end|>'))
        self.THINK, self.THINK_END = (tid('<think>'), tid('</think>'))
        self.CALL, self.CALL_END = (tid('<tool_call>'), tid('</tool_call>'))
        self.RESP, self.RESP_END = (tid('<tool_response>'), tid('</tool_response>'))
        for n in ('IM_START', 'IM_END', 'THINK', 'THINK_END', 'CALL', 'CALL_END', 'RESP', 'RESP_END'):
            if getattr(self, n) is None:
                raise RuntimeError(f'tokenizer lacks structural token for {n}')
        self._cache: dict[str, list[int]] = {}
        self._cache_size = cache_size

    def _encode_texts(self, texts: list[str]) -> list[list[int]]:
        out: list[list[int] | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            hit = self._cache.get(t) if len(t) > 1500 else None
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        if todo:
            encs = self.tok.encode_batch([texts[i] for i in todo], add_special_tokens=False)
            for i, e in zip(todo, encs):
                out[i] = e.ids
                if len(texts[i]) > 1500 and len(self._cache) < self._cache_size:
                    self._cache[texts[i]] = e.ids
        return out

    def encode_items(self, items: list) -> tuple[np.ndarray, np.ndarray]:
        texts = [(i, it) for i, it in enumerate(items) if isinstance(it, str)]
        enc = dict(zip((i for i, _ in texts), self._encode_texts([t for _, t in texts])))
        lens = np.fromiter((1 if isinstance(it, int) else len(enc[i]) for i, it in enumerate(items)), dtype=np.int64, count=len(items))
        cum = np.zeros(len(items) + 1, dtype=np.int64)
        np.cumsum(lens, out=cum[1:])
        ids = np.empty(cum[-1], dtype=np.uint32)
        for i, it in enumerate(items):
            if isinstance(it, int):
                ids[cum[i]] = it
            else:
                ids[cum[i]:cum[i + 1]] = enc[i]
        return (ids, cum)

    def render(self, messages: list[dict], tools: list[str] | None, harness: str) -> Rendered:
        items: list = []

        def sp(x: int) -> None:
            items.append(x)

        def tx(s: str) -> None:
            if not s:
                return
            if items and isinstance(items[-1], str):
                items[-1] += s
            else:
                items.append(s)
        if not messages:
            raise RenderError('no_messages')
        tool_defs = [json.loads(t) for t in tools] if tools else []
        first_is_system = messages[0]['role'] == 'system'
        sys_content = _content(messages[0]).strip() if first_is_system else ''
        if tool_defs:
            sp(self.IM_START)
            head = 'system\n# Tools\n\nYou have access to the following functions:\n\n<tools>'
            head += ''.join(('\n' + json.dumps(t, ensure_ascii=False) for t in tool_defs))
            head += '\n</tools>' + TOOLS_TAIL
            if sys_content:
                head += '\n\n' + sys_content
            tx(head)
            sp(self.IM_END)
            tx('\n')
        elif first_is_system:
            sp(self.IM_START)
            tx('system\n' + sys_content)
            sp(self.IM_END)
            tx('\n')
        last_query = _last_query_index(messages)
        turn_marks: list[dict] = []
        unknown: list[str] = []
        n = len(messages)
        for idx, m in enumerate(messages):
            role = m['role']
            content = _content(m).strip()
            if role == 'system':
                if idx != 0:
                    raise RenderError('system_not_first')
            elif role == 'user':
                sp(self.IM_START)
                tx('user\n' + content)
                sp(self.IM_END)
                tx('\n')
            elif role == 'assistant':
                turn_marks.append(self._assistant(items, m, content, idx > last_query, harness, unknown))
            elif role == 'tool':
                if idx > 0 and messages[idx - 1]['role'] != 'tool':
                    sp(self.IM_START)
                    tx('user')
                tx('\n')
                sp(self.RESP)
                tx('\n' + content + '\n')
                sp(self.RESP_END)
                if idx == n - 1 or messages[idx + 1]['role'] != 'tool':
                    sp(self.IM_END)
                    tx('\n')
            else:
                raise RenderError('bad_role')
        ids, cum = self.encode_items(items)
        turns = np.full((len(turn_marks), TURN_STRIDE), -1, dtype=np.int32)
        for k, mk in enumerate(turn_marks):
            turns[k, T_START] = cum[mk['start']]
            turns[k, T_CE_START] = cum[mk['ce_start']]
            turns[k, T_CE_END] = cum[mk['end'] + 1]
            if mk['call_first'] is not None:
                turns[k, T_CALL_START] = cum[mk['call_first']]
                turns[k, T_CALL_END] = cum[mk['call_last_end'] + 1]
                turns[k, T_ANCHOR] = turns[k, T_CALL_START] - 1
            turns[k, T_FLAGS] = mk['flags']
            turns[k, T_NCALLS] = mk['n_calls']
        return Rendered(ids=ids, turns=turns, unknown_tools=unknown)

    def _assistant(self, items: list, m: dict, content: str, with_think: bool, harness: str, unknown: list[str]) -> dict:

        def sp(x):
            items.append(x)

        def tx(s):
            if not s:
                return
            if items and isinstance(items[-1], str):
                items[-1] += s
            else:
                items.append(s)
        reasoning = m.get('reasoning_content')
        if not isinstance(reasoning, str):
            reasoning = ''
            if '</think>' in content:
                reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                content = content.split('</think>')[-1].lstrip('\n')
        reasoning = reasoning.strip()
        mk: dict = {'start': len(items), 'call_first': None, 'call_last_end': None, 'flags': 0}
        sp(self.IM_START)
        tx('assistant\n')
        if with_think:
            sp(self.THINK)
            mk['ce_start'] = len(items)
            tx('\n' + reasoning + '\n')
            sp(self.THINK_END)
            tx('\n\n' + content)
            if reasoning:
                mk['flags'] |= F_REASONING
        else:
            mk['ce_start'] = len(items)
            tx(content)
        calls = m.get('tool_calls') or []
        parsed: list[tuple[str, dict]] = []
        for j, tc in enumerate(calls):
            fn = tc.get('function', tc)
            args = parse_args(fn.get('arguments'))
            if args is None:
                raise RenderError('bad_tool_args')
            name = fn['name']
            parsed.append((name, args))
            tx('\n\n' if j == 0 and content else '\n' if j > 0 else '')
            if mk['call_first'] is None:
                mk['call_first'] = len(items)
            sp(self.CALL)
            body = '\n<function=' + name + '>\n'
            for k, v in args.items():
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(v, ensure_ascii=False)
                else:
                    v = str(v)
                body += '<parameter=' + k + '>\n' + v + '\n</parameter>\n'
            tx(body + '</function>\n')
            mk['call_last_end'] = len(items)
            sp(self.CALL_END)
        mk['n_calls'] = len(calls)
        if parsed:
            is_commit, is_finish, unk = classify_turn(harness, parsed)
            unknown.extend(unk)
            if is_commit:
                mk['flags'] |= F_COMMIT
            if is_finish:
                mk['flags'] |= F_FINISH
        mk['end'] = len(items)
        sp(self.IM_END)
        tx('\n')
        return mk

    def render_hint(self, hint_text: str) -> np.ndarray:
        items: list = [self.IM_START, 'user\n' + hint_text, self.IM_END, '\n']
        ids, _ = self.encode_items(items)
        return ids

def _content(m: dict) -> str:
    c = m.get('content')
    if c is None:
        return ''
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ''.join((p.get('text', '') for p in c if isinstance(p, dict)))
    raise RenderError('bad_content')

def _last_query_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m['role'] == 'user':
            c = _content(m).strip()
            if not (c.startswith('<tool_response>') and c.endswith('</tool_response>')):
                return i
    raise RenderError('no_user_query')

def verify_against_hf(renderer: Renderer, tokenizer_dir: str, rows: list[dict]) -> dict:
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(tokenizer_dir)
    ok = bad = hf_div = 0
    first_bad = None
    for r in rows:
        msgs = []
        for m in r['messages']:
            m2 = dict(m)
            if m2.get('tool_calls'):
                m2['tool_calls'] = [{'type': 'function', 'function': {'name': tc['function']['name'], 'arguments': parse_args(tc['function']['arguments']) or {}}} for tc in m2['tool_calls']]
            else:
                m2.pop('tool_calls', None)
            msgs.append(m2)
        text = hf.apply_chat_template(msgs, tools=[json.loads(t) for t in r['tools']], tokenize=False)
        want = renderer.tok.encode(text, add_special_tokens=False).ids
        if want != hf(text, add_special_tokens=False)['input_ids']:
            hf_div += 1
        got = renderer.render(r['messages'], r['tools'], r['_harness']).ids.tolist()
        if want == got:
            ok += 1
        else:
            bad += 1
            if first_bad is None:
                n = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
                first_bad = {'instance_id': r.get('instance_id'), 'first_diff_at': n, 'want': hf.decode(want[max(0, n - 8):n + 8]), 'got': hf.decode(got[max(0, n - 8):n + 8]), 'len': (len(want), len(got))}
    return {'ok': ok, 'mismatch': bad, 'hf_tokenizer_divergence': hf_div, 'first_mismatch': first_bad}