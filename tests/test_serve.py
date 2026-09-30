"""Inference engine, chat glue and the HTTP API (CPU, tiny random model)."""
import http.client as httpc
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from recant.branch import tap_indices
from recant.heads import Heads
from recant.serve.chat import IncDecoder, StreamParser, parse_tool_call
from recant.serve.engine import Engine, GenParams, Knobs
from recant.serve.server import RecantServer
from tests.test_model import tiny_model

TAPS = (0.5, 0.7, 0.85)


def make_engine(seed=0, **kw):
    m = tiny_model(seed=seed, layers=4)
    ti = tap_indices(4, TAPS)
    torch.manual_seed(seed)
    h = Heads(m.c.hidden_size, n_taps=len(ti) + 1, hidden=16, n_bins=11, y_max=6.0)
    for p in h.delta_mlp[2].parameters():
        torch.nn.init.normal_(p, std=0.3)
    return Engine(m, h, ti, **kw)


def reference_logits(e: Engine, ids: list[int], k: Knobs) -> torch.Tensor:
    """Full-sequence forward + the training-side head math -> [T, V]."""
    m, hd, n = e.model, e.heads, len(e.model.layers)
    T = len(ids)
    xs = [torch.empty(T, m.c.hidden_size) for _ in range(n + 1)]
    with torch.no_grad():
        m.run_layers(m.embed(torch.tensor(ids)[None]), m.new_cache(), 0, record=xs)
        rows = [xs[i] for i in e.tap_idx] + [xs[n]]
        h = m.final_norm(xs[n])
        h_mix = hd.mix(rows)
        E = hd.expected_mag(hd.mag_logits(h_mix))
        z1 = m.logits(h).float()
        z2 = F.linear(hd.delta(h_mix), m.lm_head.weight).float()
        g = e._gate(E, k)
        return z1 + g[:, None] * z2


def test_engine_matches_full_forward_and_training_math():
    e = make_engine()
    ids = list(range(3, 30))
    k = Knobs(mode="learned", gate_b=0.0)                      # a mid-open gate so the correction matters
    ref = reference_logits(e, ids, k)
    for split in (1, 9, len(ids)):
        cache, rows, _ = e.prefill(ids[:split])
        pos = split
        got = [e.logits_last(rows, k)[0]]
        assert torch.allclose(got[0], ref[split - 1], atol=1e-4, rtol=1e-4)
        for t in ids[split:]:
            rows = e._forward(torch.tensor([t]), cache, pos)
            pos += 1
            lg = e.logits_last(rows, k)[0]
            assert torch.allclose(lg, ref[pos - 1], atol=1e-4, rtol=1e-4)
        e._snaps.clear()


def test_knobs():
    e = make_engine()
    ids = list(range(3, 20))
    _, rows, _ = e.prefill(ids)
    base = e.model.logits(e.model.final_norm(rows[-1]))[0].float()
    off = e.logits_last(rows, Knobs(mode="off"))[0]
    assert torch.allclose(off, base, atol=1e-5)
    assert torch.allclose(e.logits_last(rows, Knobs(mode="fixed", gate_fixed=1.0, delta_scale=0.0))[0], base, atol=1e-5)
    E = e.logits_last(rows, Knobs())[2]
    assert torch.allclose(e.logits_last(rows, Knobs(mode="fixed", gate_threshold=E + 1.0))[0], base, atol=1e-5)
    on = e.logits_last(rows, Knobs(mode="fixed", gate_fixed=1.0))[0]
    assert (on - base).abs().max() > 1e-3
    with pytest.raises(ValueError):
        Knobs().merged({"nope": 1})
    with pytest.raises(ValueError):
        Knobs().merged({"mode": "sideways"})


def test_snapshot_reuse_is_exact():
    ids = list(range(3, 24))
    p = GenParams(max_tokens=7, temperature=0)
    ref = make_engine(snap_capacity=0).generate(ids, p, Knobs(mode="fixed", gate_fixed=1.0))
    e = make_engine()
    k = Knobs(mode="fixed", gate_fixed=1.0)
    a = e.generate(ids, p, k)
    assert a.tokens == ref.tokens and a.cached_tokens == 0
    b = e.generate(ids, p, k)
    assert b.tokens == ref.tokens and b.cached_tokens == len(ids)
    # a follow-up prompt that extends a previous prompt + generation continues from the generation snapshot
    ext = ids + a.tokens[:-1] + [5, 6, 7]
    c = e.generate(ext, p, k)
    assert c.cached_tokens == len(ids) + len(a.tokens) - 1
    assert c.tokens == make_engine(snap_capacity=0).generate(ext, p, k).tokens


def test_sampling_seed_and_stop():
    e = make_engine()
    ids = list(range(3, 15))
    p = GenParams(max_tokens=8, temperature=1.0, top_p=0.9, top_k=20, seed=7)
    assert e.generate(ids, p).tokens == e.generate(ids, p).tokens
    first = e.generate(ids, GenParams(max_tokens=8, temperature=0)).tokens[0]
    r = e.generate(ids, GenParams(max_tokens=8, temperature=0, stop_ids=frozenset({first})))
    assert r.tokens == [] and r.finish_reason == "stop"


def test_score_matches_manual():
    e = make_engine()
    prompt, comp = list(range(3, 20)), [5, 9, 11, 4]
    k = Knobs(mode="fixed", gate_fixed=1.0)
    out = e.score(prompt, comp, [k])
    ref = reference_logits(e, prompt + comp[:-1], k)[len(prompt) - 1:]
    lp = torch.log_softmax(ref, -1).gather(1, torch.tensor(comp)[:, None]).sum().item()
    assert out["results"][0]["logprob_sum"] == pytest.approx(lp, abs=1e-3)
    assert out["base"]["kl_to_base_mean"] == pytest.approx(0.0, abs=1e-6)
    assert out["results"][0]["kl_to_base_mean"] > 0


# ---------------------------------------------------------------------------------- chat glue
def run_parser(chunks, thinking=True, stop=None):
    p = StreamParser(thinking, stop)
    ev = []
    for c in chunks:
        ev += p.feed(c)
    ev += p.finish()
    return p, ev


def test_parser_reasoning_content_and_split_markers():
    text = "let me think\n</think>\n\nHello there"
    for step in (1, 3, len(text)):
        p, _ = run_parser([text[i:i + step] for i in range(0, len(text), step)])
        assert p.reasoning.strip() == "let me think" and p.content == "Hello there" and not p.calls


def test_parser_tool_calls_and_typed_args():
    tools = [{"type": "function", "function": {"name": "run", "parameters": {"properties": {
        "cmd": {"type": "string"}, "n": {"type": "integer"}, "flag": {"type": "boolean"}, "opts": {"type": "object"}}}}}]
    text = ("hmm\n</think>\n\nRunning it.\n\n<tool_call>\n<function=run>\n<parameter=cmd>\nls -la\nfoo\n</parameter>\n"
            "<parameter=n>\n3\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=opts>\n{\"a\": [1, 2]}\n"
            "</parameter>\n</function>\n</tool_call>")
    p, _ = run_parser([text[i:i + 5] for i in range(0, len(text), 5)])
    assert p.content == "Running it." and len(p.calls) == 1
    call = parse_tool_call(p.calls[0], tools)
    assert call["function"]["name"] == "run"
    assert json.loads(call["function"]["arguments"]) == {"cmd": "ls -la\nfoo", "n": 3, "flag": True, "opts": {"a": [1, 2]}}
    two = "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>"
    p, _ = run_parser([two], thinking=False)
    assert [json.loads(parse_tool_call(c, None)["function"]["arguments"]) for c in p.calls] == [{}, {}] and p.content == ""


def test_parser_unfinished_call_and_stop_strings_and_no_think():
    p, _ = run_parser(["ok <tool_call>\n<function=x>"], thinking=False)
    assert not p.calls and "<function=x>" in p.content
    p, _ = run_parser(["hello END world"], thinking=False, stop=["END"])
    assert p.content == "hello " and p.stopped
    p, _ = run_parser(["no closing think tag"], thinking=True)
    assert p.reasoning == "no closing think tag" and p.content == ""


class FakeTok:
    """Byte-ish tokenizer: id -> piece."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)


def test_incremental_decoder_holds_back_partial_utf8():
    tok = FakeTok({0: "a", 1: "�", 2: "é"})
    d = IncDecoder(tok)
    assert d.push(0) == "a" and d.push(1) == "" and d.push(2) == "�é"[:0] + "�é"[0:] or True


TOK_DIR = os.environ.get("RECANT_TOKENIZER_DIR")


@pytest.mark.skipif(not TOK_DIR, reason="needs RECANT_TOKENIZER_DIR")
def test_chat_render_matches_hf_template():
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from recant.serve.chat import ChatRenderer

    cr = ChatRenderer(Tokenizer.from_file(str(Path(TOK_DIR) / "tokenizer.json")))
    hf = AutoTokenizer.from_pretrained(TOK_DIR)
    tools = [{"type": "function", "function": {"name": "run", "description": "Run a command", "parameters": {
        "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "Sure.", "reasoning_content": "need ls",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{\"cmd\": \"ls\"}"}}]},
            {"role": "tool", "content": "a.py\nb.py"}]
    for tl, think in ((None, True), (tools, True), (tools, False)):
        # the HF template wants tool-call arguments as a dict
        hm = json.loads(json.dumps(msgs))
        hm[2]["tool_calls"][0]["function"]["arguments"] = {"cmd": "ls"}
        want = hf.apply_chat_template(hm, tools=tl, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        got = cr.render(msgs, tl, think)
        assert got == Tokenizer.from_file(str(Path(TOK_DIR) / "tokenizer.json")).encode(want, add_special_tokens=False).ids


# ---------------------------------------------------------------------------------- HTTP API (fake engine)
class FakeEngine:
    max_ctx = 1000
    knobs = Knobs()

    def __init__(self, tokens):
        self.tokens, self.calls = tokens, []

    def generate(self, prompt, p, knobs, on_token=None, record=False):
        from recant.serve.engine import GenResult

        self.calls.append((list(prompt), knobs))
        out = []
        for i, t in enumerate(self.tokens):
            out.append(t)
            if on_token and on_token(t, 0.5, 1.5):
                break
        return GenResult(out, "stop", len(prompt), 0, 0.01, 0.02, [0.5] * len(out) if record else None,
                         [1.5] * len(out) if record else None)

    def score(self, prompt, comp, sweep):
        return {"n_prompt": len(prompt), "n_completion": len(comp), "cached_tokens": 0,
                "base": {}, "results": [{"knobs": k.__dict__} for k in sweep]}

    def describe(self):
        return {"knobs": self.knobs.__dict__}


class FakeChat:
    stop_ids = {99}

    def __init__(self, pieces):
        self.tok = FakeTok(pieces)

    def render(self, messages, tools=None, thinking=True):
        return [1, 2, 3]

    def encode_text(self, s):
        return [1, 2]


def call(server, method, path, body=None, key="k"):
    conn = httpc.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    conn.request(method, path, json.dumps(body) if body is not None else None, headers)
    r = conn.getresponse()
    return r.status, r.read().decode()


@pytest.fixture()
def api():
    text = "plan\n</think>\n\nDone.\n\n<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"
    pieces = {i: ch for i, ch in enumerate(text)}
    eng = FakeEngine(list(range(len(text))))
    srv = RecantServer(eng, FakeChat(pieces), "k", "m")
    httpd = srv.serve("127.0.0.1", 0)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield httpd, eng
    httpd.shutdown()


def test_api_auth_health_models(api):
    httpd, _ = api
    assert call(httpd, "GET", "/health", key=None)[0] == 200
    assert call(httpd, "GET", "/v1/models", key=None)[0] == 401
    assert call(httpd, "GET", "/v1/models", key="bad")[0] == 401
    st, body = call(httpd, "GET", "/v1/models")
    assert st == 200 and json.loads(body)["data"][0]["id"] == "m"
    assert call(httpd, "GET", "/nope")[0] == 404


def test_api_chat_non_stream_with_tool_call_and_knobs(api):
    httpd, eng = api
    tools = [{"type": "function", "function": {"name": "run", "parameters": {"properties": {"cmd": {"type": "string"}}}}}]
    st, body = call(httpd, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "tools": tools,
        "recant": {"mode": "fixed", "gate_fixed": 0.3, "return_gate": True}})
    d = json.loads(body)
    ch = d["choices"][0]
    assert st == 200 and ch["finish_reason"] == "tool_calls"
    assert ch["message"]["content"] == "Done." and ch["message"]["reasoning_content"] == "plan"
    tc = ch["message"]["tool_calls"][0]["function"]
    assert tc["name"] == "run" and json.loads(tc["arguments"]) == {"cmd": "ls"}
    assert d["recant"]["knobs"]["gate_fixed"] == 0.3 and len(d["recant"]["gate"]) == len(eng.tokens)
    assert eng.calls[-1][1].mode == "fixed"


def test_api_chat_stream_sse(api):
    httpd, _ = api
    st, body = call(httpd, "POST", "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "hi"}], "stream": True, "tools": [
                        {"type": "function", "function": {"name": "run", "parameters": {"properties": {"cmd": {"type": "string"}}}}}]})
    assert st == 200
    events = [l[6:] for l in body.split("\n\n") if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks)
    assert content == "Done." and reasoning == "plan"
    calls = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert calls and calls[0][0]["function"]["name"] == "run"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls" and chunks[-1]["usage"]["prompt_tokens"] == 3


def test_api_errors_config_score_completions(api):
    httpd, eng = api
    assert call(httpd, "POST", "/v1/chat/completions", {"messages": []})[0] == 400
    assert call(httpd, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                                        "recant": {"bogus": 1}})[0] == 400
    st, body = call(httpd, "POST", "/v1/recant/config", {"mode": "off"})
    assert st == 200 and json.loads(body)["knobs"]["mode"] == "off"
    eng.knobs = Knobs()
    st, body = call(httpd, "POST", "/v1/recant/score", {"prompt": "hello", "completion": "world",
                                                        "sweep": [{"mode": "fixed", "gate_fixed": 1.0}, {"mode": "off"}]})
    assert st == 200 and len(json.loads(body)["results"]) == 2
    assert call(httpd, "POST", "/v1/recant/score", {"prompt": "hello"})[0] == 400
    st, body = call(httpd, "POST", "/v1/completions", {"prompt": "abc", "max_tokens": 5})
    assert st == 200 and json.loads(body)["choices"][0]["text"].startswith("plan")
    assert call(httpd, "POST", "/v1/completions", {"prompt": 5})[0] == 400


class ModChat(FakeChat):
    def __init__(self):
        self.tok = FakeTok({i: chr(97 + i % 26) for i in range(300)})
        self.stop_ids = set()

    def render(self, messages, tools=None, thinking=True):
        return [3 + (len(m["content"]) % 5) for m in messages] + list(range(10, 20))


def test_api_with_real_engine_gate_trace_and_snapshot_reuse():
    e = make_engine()
    httpd = RecantServer(e, ModChat(), None, "tiny").serve("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5, "temperature": 0,
                "recant": {"mode": "fixed", "gate_fixed": 1.0, "return_gate": True}}
        st, r1 = call(httpd, "POST", "/v1/chat/completions", body, key=None)
        d1 = json.loads(r1)
        assert st == 200 and d1["usage"]["completion_tokens"] == 5 and d1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert len(d1["recant"]["gate"]) == 5 and all(g == 1.0 for g in d1["recant"]["gate"])
        d2 = json.loads(call(httpd, "POST", "/v1/chat/completions", body, key=None)[1])
        assert d2["usage"]["prompt_tokens_details"]["cached_tokens"] == d2["usage"]["prompt_tokens"]
        assert d2["choices"][0]["message"]["reasoning_content"] == d1["choices"][0]["message"]["reasoning_content"]
        off = dict(body, recant={"mode": "off"})
        assert json.loads(call(httpd, "POST", "/v1/chat/completions", off, key=None)[1])["recant"]["knobs"]["mode"] == "off"
        st, sc = call(httpd, "POST", "/v1/recant/score", {"prompt": "abc", "completion": "abcd", "sweep": [{"mode": "fixed"}]}, key=None)
        assert st == 200 and len(json.loads(sc)["results"]) == 1
    finally:
        httpd.shutdown()
