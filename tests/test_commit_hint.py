import json

import pytest

from recant.data.commit import classify_call, classify_turn, parse_args
from recant.data.hint import MAX_FILES, MAX_SYMBOLS, build_hint, parse_patch

PATCH = """diff --git a/src/pkg/core.py b/src/pkg/core.py
index 111..222 100644
--- a/src/pkg/core.py
+++ b/src/pkg/core.py
@@ -10,6 +10,9 @@ class Engine(object):
     x = 1
-    y = 2
+    y = SECRET_ANSWER_LINE
@@ -50,3 +53,4 @@ def helper(a, b):
     pass
diff --git a/docs/notes.rst b/docs/notes.rst
--- a/docs/notes.rst
+++ b/docs/notes.rst
@@ -1 +1 @@ Notes
"""


def test_parse_patch_files_and_symbols():
    files, symbols = parse_patch(PATCH)
    assert files == ["src/pkg/core.py", "docs/notes.rst"]
    assert symbols == ["Engine", "helper"]


def test_hint_never_contains_diff_body():
    hint = build_hint(PATCH)
    assert "SECRET_ANSWER_LINE" not in hint
    assert "core.py" in hint and "Engine" in hint
    assert "\n+" not in hint and "\n-" not in hint


def test_hint_empty_patch_is_none():
    assert build_hint("") is None
    assert build_hint(None) is None


def test_hint_caps():
    patch = "".join(f"diff --git a/f{i}.py b/f{i}.py\n@@ -1 +1 @@ def fn{i}():\n" for i in range(40))
    files, symbols = parse_patch(patch)
    assert len(files) == MAX_FILES and len(symbols) == MAX_SYMBOLS


@pytest.mark.parametrize("harness,name,args,want", [
    ("openhands", "str_replace_editor", {"command": "view"}, "explore"),
    ("openhands", "str_replace_editor", {"command": "str_replace"}, "commit"),
    ("openhands", "str_replace_editor", {"command": "create"}, "commit"),
    ("openhands", "str_replace_editor", {"command": "insert"}, "commit"),
    ("openhands", "execute_bash", {"command": "pytest -x"}, "explore"),
    ("openhands", "think", {"thought": "hmm"}, "explore"),
    ("openhands", "finish", {"message": "done"}, "finish"),
    ("sweagent", "bash", {"command": "grep -r x ."}, "explore"),
    ("sweagent", "str_replace_editor", {"command": "str_replace"}, "commit"),
    ("sweagent", "submit", {}, "finish"),
    ("sweagent", "mystery_tool", {}, "unknown"),
])
def test_classify_call(harness, name, args, want):
    assert classify_call(harness, name, args) == want


def test_classify_turn_flags_and_unknown():
    commit, finish, unk = classify_turn("sweagent", [("bash", {}), ("str_replace_editor", {"command": "insert"})])
    assert commit and not finish and unk == []
    commit, finish, unk = classify_turn("openhands", [("finish", {"message": "x"})])
    assert commit and finish
    commit, finish, unk = classify_turn("openhands", [("weird", {})])
    assert not commit and unk == ["weird"]


def test_parse_args():
    assert parse_args('{"a": 1}') == {"a": 1}
    assert parse_args("") == {}
    assert parse_args(None) == {}
    assert parse_args("not json") is None
    assert parse_args("[1, 2]") is None
    assert parse_args({"a": 1}) == {"a": 1}
