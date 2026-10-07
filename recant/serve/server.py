from __future__ import annotations
import hmac
import json
import threading
import time
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ..data.render import RenderError
from .chat import IncDecoder, StreamParser, parse_tool_call
from .engine import GenParams, Knobs

class ApiError(Exception):

    def __init__(self, status: int, message: str, kind: str='invalid_request_error'):
        super().__init__(message)
        self.status, self.kind = (status, kind)

def _num(body: dict, key: str, default, cast=float, lo=None, hi=None):
    v = body.get(key)
    if v is None:
        return default
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise ApiError(400, f"'{key}' must be a number")
    if lo is not None and v < lo or (hi is not None and v > hi):
        raise ApiError(400, f"'{key}' out of range")
    return v

class RecantServer:

    def __init__(self, engine, chat, api_key: str | None, model_name: str='recant'):
        self.engine, self.chat, self.api_key, self.model_name = (engine, chat, api_key, model_name)
        self.lock = threading.Lock()

    def _params(self, body: dict) -> tuple[GenParams, list[str]]:
        stop = body.get('stop') or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        mt = body.get('max_completion_tokens', body.get('max_tokens'))
        p = GenParams(max_tokens=int(mt) if mt is not None else 1024, temperature=_num(body, 'temperature', 0.7, lo=0.0), top_p=_num(body, 'top_p', 1.0, lo=0.0, hi=1.0), top_k=_num(body, 'top_k', 0, int, lo=0), seed=body.get('seed'), stop_ids=frozenset(self.chat.stop_ids))
        return (p, stop)

    def _knobs(self, body: dict) -> tuple[Knobs, bool]:
        over = dict(body.get('recant') or {})
        ret = bool(over.pop('return_gate', False))
        try:
            return (self.engine.knobs.merged(over), ret)
        except (ValueError, TypeError) as e:
            raise ApiError(400, str(e))

    def _check_len(self, prompt: list) -> None:
        if len(prompt) >= self.engine.max_ctx:
            raise ApiError(400, f'prompt has {len(prompt)} tokens; max_ctx is {self.engine.max_ctx}')

    def chat_completions(self, body: dict, send_json, stream_open):
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ApiError(400, "'messages' must be a non-empty list")
        tools = body.get('tools') or None
        kwargs = body.get('chat_template_kwargs') or {}
        thinking = bool(body.get('enable_thinking', kwargs.get('enable_thinking', True)))
        try:
            prompt = self.chat.render(messages, tools, thinking)
        except (RenderError, KeyError, TypeError) as e:
            raise ApiError(400, f'cannot render messages: {e}')
        params, stop = self._params(body)
        knobs, ret_gate = self._knobs(body)
        parser = StreamParser(thinking, stop)
        dec = IncDecoder(self.chat.tok)
        cid, created = ('chatcmpl-' + uuid.uuid4().hex[:24], int(time.time()))
        self._check_len(prompt)
        chunks = stream_open() if body.get('stream') else None

        def chunk(delta: dict, finish=None, extra=None):
            d = {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            if extra:
                d.update(extra)
            chunks(d)

        def emit(events):
            if chunks is None:
                return
            for kind, text in events:
                chunk({'reasoning_content': text} if kind == 'reasoning' else {'content': text})

        def on_token(tid, g, e):
            emit(parser.feed(dec.push(tid)))
            return parser.stopped
        if chunks is not None:
            chunk({'role': 'assistant', 'content': ''})
        try:
            res = self.engine.generate(prompt, params, knobs, on_token, record=ret_gate)
        except ValueError as e:
            raise ApiError(400, str(e))
        emit(parser.feed(dec.flush()))
        emit(parser.finish())
        calls = [c for c in (parse_tool_call(r, tools) for r in parser.calls) if c]
        reason = 'tool_calls' if calls else 'stop' if parser.stopped else res.finish_reason
        usage = {'prompt_tokens': res.prompt_tokens, 'completion_tokens': len(res.tokens), 'total_tokens': res.prompt_tokens + len(res.tokens), 'prompt_tokens_details': {'cached_tokens': res.cached_tokens}}
        rec = {'knobs': asdict(knobs), 'prefill_s': round(res.prefill_s, 3), 'decode_tok_s': round(len(res.tokens) / max(res.decode_s, 1e-09), 2)}
        if ret_gate:
            rec['gate'], rec['mag'] = (res.gates, res.mags)
        if chunks is not None:
            if calls:
                chunk({'tool_calls': [dict(c, index=i) for i, c in enumerate(calls)]})
            chunk({}, reason, {'usage': usage, 'recant': rec})
            chunks(None)
            return
        msg = {'role': 'assistant', 'content': parser.content.strip() or None, 'tool_calls': calls or None}
        if parser.reasoning.strip():
            msg['reasoning_content'] = parser.reasoning.strip()
        msg = {k: v for k, v in msg.items() if v is not None or k == 'content'}
        send_json(200, {'id': cid, 'object': 'chat.completion', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'message': msg, 'finish_reason': reason}], 'usage': usage, 'recant': rec})

    def completions(self, body: dict, send_json, stream_open):
        prompt = body.get('prompt')
        if isinstance(prompt, str):
            prompt = self.chat.encode_text(prompt)
        elif not (isinstance(prompt, list) and prompt and all((isinstance(t, int) for t in prompt))):
            raise ApiError(400, "'prompt' must be a string or a list of token ids")
        params, stop = self._params(body)
        knobs, ret_gate = self._knobs(body)
        dec, text, cid, created = (IncDecoder(self.chat.tok), [''], 'cmpl-' + uuid.uuid4().hex[:24], int(time.time()))
        self._check_len(prompt)
        chunks = stream_open() if body.get('stream') else None

        def piece(s, finish=None, extra=None):
            d = {'id': cid, 'object': 'text_completion', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'text': s, 'finish_reason': finish}]}
            d.update(extra or {})
            return d
        stopped = [False]

        def on_token(tid, g, e):
            s = dec.push(tid)
            text[0] += s
            for st in stop:
                i = text[0].find(st)
                if i >= 0:
                    keep = i - (len(text[0]) - len(s))
                    s = s[:max(keep, 0)]
                    text[0] = text[0][:i]
                    stopped[0] = True
                    break
            if chunks is not None and s:
                chunks(piece(s))
            return stopped[0]
        try:
            res = self.engine.generate(prompt, params, knobs, on_token, record=ret_gate)
        except ValueError as e:
            raise ApiError(400, str(e))
        if not stopped[0]:
            tail = dec.flush()
            text[0] += tail
            if chunks is not None and tail:
                chunks(piece(tail))
        reason = 'stop' if stopped[0] else res.finish_reason
        usage = {'prompt_tokens': res.prompt_tokens, 'completion_tokens': len(res.tokens), 'total_tokens': res.prompt_tokens + len(res.tokens)}
        rec = {'knobs': asdict(knobs), 'prefill_s': round(res.prefill_s, 3)}
        if ret_gate:
            rec['gate'], rec['mag'] = (res.gates, res.mags)
        if chunks is not None:
            chunks(piece('', reason, {'usage': usage, 'recant': rec}))
            chunks(None)
            return
        send_json(200, dict(piece(text[0], reason), usage=usage, recant=rec))

    def score(self, body: dict, send_json):
        if 'messages' in body:
            try:
                prompt = self.chat.render(body['messages'], body.get('tools') or None, bool(body.get('enable_thinking', True)))
            except (RenderError, KeyError, TypeError) as e:
                raise ApiError(400, f'cannot render messages: {e}')
        elif isinstance(body.get('prompt'), str):
            prompt = self.chat.encode_text(body['prompt'])
        else:
            raise ApiError(400, "give 'messages' or a text 'prompt'")
        comp = body.get('completion_ids')
        if comp is None:
            if not isinstance(body.get('completion'), str) or not body['completion']:
                raise ApiError(400, "give 'completion' (text) or 'completion_ids'")
            comp = self.chat.encode_text(body['completion'])
        try:
            sweep = [self.engine.knobs.merged(s) for s in body.get('sweep') or []]
            out = self.engine.score(prompt, list(comp), sweep)
        except (ValueError, TypeError) as e:
            raise ApiError(400, str(e))
        send_json(200, out)

    def config(self, body: dict | None, send_json):
        if body is not None:
            try:
                self.engine.knobs = self.engine.knobs.merged(body)
            except (ValueError, TypeError) as e:
                raise ApiError(400, str(e))
        send_json(200, self.engine.describe())

    def make_handler(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, fmt, *a):
                print(f'[serve] {self.address_string()} {fmt % a}', flush=True)

            def _send(self, status: int, obj: dict):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream_open(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                self._headers_sent = True

                def write(obj):
                    payload = 'data: [DONE]\n\n' if obj is None else 'data: ' + json.dumps(obj) + '\n\n'
                    self.wfile.write(payload.encode())
                    self.wfile.flush()
                return write

            def _auth(self) -> bool:
                if not srv.api_key:
                    return True
                got = self.headers.get('Authorization', '')
                got = got[7:] if got.lower().startswith('bearer ') else self.headers.get('x-api-key', '')
                return hmac.compare_digest(got.encode(), srv.api_key.encode())

            def _handle(self, method: str):
                path = self.path.split('?')[0].rstrip('/')
                try:
                    if path == '/health':
                        return self._send(200, {'status': 'ok'})
                    if not self._auth():
                        raise ApiError(401, 'invalid or missing API key', 'authentication_error')
                    body = None
                    if method == 'POST':
                        n = int(self.headers.get('Content-Length') or 0)
                        try:
                            body = json.loads(self.rfile.read(n) or b'{}')
                        except ValueError:
                            raise ApiError(400, 'body is not valid JSON')
                        if not isinstance(body, dict):
                            raise ApiError(400, 'body must be a JSON object')
                    if method == 'GET' and path == '/v1/models':
                        return self._send(200, {'object': 'list', 'data': [{'id': srv.model_name, 'object': 'model', 'created': 0, 'owned_by': 'recant'}]})
                    if path == '/v1/recant/config' and method in ('GET', 'POST'):
                        with srv.lock:
                            return srv.config(body, self._send)
                    routes = {'/v1/chat/completions': lambda: srv.chat_completions(body, self._send, self._stream_open), '/v1/completions': lambda: srv.completions(body, self._send, self._stream_open), '/v1/recant/score': lambda: srv.score(body, self._send)}
                    if method == 'POST' and path in routes:
                        with srv.lock:
                            return routes[path]()
                    raise ApiError(404, f'no route {method} {path}', 'not_found')
                except ApiError as e:
                    if not getattr(self, '_headers_sent', False):
                        self._send(e.status, {'error': {'message': str(e), 'type': e.kind}})
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    try:
                        self._send(500, {'error': {'message': f'{type(e).__name__}: {e}', 'type': 'server_error'}})
                    except OSError:
                        self.close_connection = True

            def do_GET(self):
                self._handle('GET')

            def do_POST(self):
                self._handle('POST')
        return Handler

    def serve(self, host: str, port: int) -> ThreadingHTTPServer:
        httpd = ThreadingHTTPServer((host, port), self.make_handler())
        httpd.daemon_threads = True
        return httpd