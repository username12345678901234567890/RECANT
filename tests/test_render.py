import json
import os

import numpy as np
import pytest

tokenizers = pytest.importorskip("tokenizers")

TOK_DIR = os.environ.get("RECANT_TOKENIZER_DIR", "")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(TOK_DIR, "tokenizer.json")),
                                reason="set RECANT_TOKENIZER_DIR to a Qwen3.5 tokenizer dir")

from recant.data.render import (F_COMMIT, F_FINISH, F_REASONING, T_ANCHOR, T_CALL_END, T_CALL_START,  # noqa: E402
                                T_CE_END, T_CE_START, T_FLAGS, T_START, Renderer, RenderError,
                                verify_against_hf)

TOOLS = [json.dumps({"type": "function", "function": {"name": n, "description": f"{n} tool",
                     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}})
         for n in ("execute_bash", "str_replace_editor", "finish")]


def call(name, **args):
    return {"function": {"name": name, "arguments": json.dumps(args)}, "id": "c", "type": "function"}


def asst(reasoning, content="", calls=()):
    return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": list(calls)}


def traj():
    return [
        {"role": "system", "content": "You are a helpful agent.", "reasoning_content": None, "tool_calls": []},
        {"role": "user", "content": "Fix the bug in foo.py", "reasoning_content": None, "tool_calls": []},
        asst("Let me look.", calls=[call("str_replace_editor", command="view", path="/foo.py")]),
        {"role": "tool", "content": "def foo(): return 1", "reasoning_content": None, "tool_calls": []},
        asst("Edit it.", calls=[call("str_replace_editor", command="str_replace", path="/foo.py",
                                     old_str="return 1", new_str="return 2")]),
        {"role": "tool", "content": "edited", "reasoning_content": None, "tool_calls": []},
        {"role": "tool", "content": "second observation", "reasoning_content": None, "tool_calls": []},
        asst("", content="All done.", calls=[call("finish", message="fixed")]),
    ]


@pytest.fixture(scope="module")
def R():
    return Renderer(tokenizers.Tokenizer.from_file(os.path.join(TOK_DIR, "tokenizer.json")))


def test_turn_structure(R):
    x = R.render(traj(), TOOLS, "openhands")
    ids, turns = x.ids, x.turns
    assert len(turns) == 3
    for t in turns:
        assert ids[t[T_START]] == R.IM_START
        assert ids[t[T_CE_END] - 1] == R.IM_END
        assert t[T_CE_START] > t[T_START]
    view, edit, fin = turns
    assert not view[T_FLAGS] & F_COMMIT
    assert edit[T_FLAGS] & F_COMMIT and not edit[T_FLAGS] & F_FINISH
    assert fin[T_FLAGS] & F_COMMIT and fin[T_FLAGS] & F_FINISH
    assert edit[T_FLAGS] & F_REASONING and not fin[T_FLAGS] & F_REASONING
    # the commit span is exactly <tool_call> ... </tool_call>, anchor is the token before it
    for t in (edit, fin):
        assert ids[t[T_CALL_START]] == R.CALL and ids[t[T_CALL_END] - 1] == R.CALL_END
        assert t[T_ANCHOR] == t[T_CALL_START] - 1
    # exploration turns still record their call span for structure, but are not commits
    assert view[T_CALL_START] >= 0


def test_consecutive_tool_messages_share_one_user_turn(R):
    x = R.render(traj(), TOOLS, "openhands")
    ids = x.ids.tolist()
    # two tool observations back to back -> two <tool_response> blocks inside a single user turn
    edit_end = int(x.turns[1][T_CE_END])
    fin_start = int(x.turns[2][T_START])
    between = ids[edit_end:fin_start]
    assert between.count(R.RESP) == 2 and between.count(R.IM_START) == 1 and between.count(R.IM_END) == 1


def test_matches_hf_template(R):
    pytest.importorskip("transformers")
    rows = [{"messages": traj(), "tools": TOOLS, "_harness": "openhands", "instance_id": "synthetic"}]
    thai = traj()
    thai[1]["content"] = "Bug: `เอ`, as in the loanword วิตามินเอ, raises an out of range error."
    rows.append({"messages": thai, "tools": TOOLS, "_harness": "openhands", "instance_id": "thai"})
    res = verify_against_hf(R, TOK_DIR, rows)
    assert res["mismatch"] == 0, res


def test_render_hint_is_a_user_turn(R):
    ids = R.render_hint("Files: a.py")
    assert ids[0] == R.IM_START and ids[-2] == R.IM_END
    assert R.tok.decode(ids.tolist(), skip_special_tokens=False).startswith("<|im_start|>user\nFiles: a.py")


def test_bad_arguments_and_no_user_query(R):
    bad = traj()
    bad[2]["tool_calls"][0]["function"]["arguments"] = "{not json"
    with pytest.raises(RenderError, match="bad_tool_args"):
        R.render(bad, TOOLS, "openhands")
    with pytest.raises(RenderError):
        R.render([{"role": "system", "content": "x"}], TOOLS, "openhands")


def test_ids_dtype_and_alignment(R):
    x = R.render(traj(), TOOLS, "sweagent")
    assert x.ids.dtype == np.uint32 and x.turns.dtype == np.int32
    assert x.turns[:, T_CE_END].max() <= len(x.ids)
