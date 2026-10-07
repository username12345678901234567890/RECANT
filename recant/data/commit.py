from __future__ import annotations
import json
from dataclasses import dataclass
EDIT_COMMANDS = frozenset({'create', 'str_replace', 'insert', 'undo_edit'})

@dataclass(frozen=True)
class HarnessSpec:
    editor_tools: frozenset
    direct_edit_tools: frozenset
    finish_tools: frozenset
    explore_tools: frozenset
HARNESSES: dict[str, HarnessSpec] = {'openhands': HarnessSpec(editor_tools=frozenset({'str_replace_editor'}), direct_edit_tools=frozenset(), finish_tools=frozenset({'finish'}), explore_tools=frozenset({'execute_bash', 'think', 'browser', 'execute_ipython_cell'})), 'sweagent': HarnessSpec(editor_tools=frozenset({'str_replace_editor'}), direct_edit_tools=frozenset({'edit', 'create', 'insert', 'str_replace'}), finish_tools=frozenset({'submit'}), explore_tools=frozenset({'bash', 'find_file', 'search_dir', 'search_file', 'open', 'goto', 'scroll_up', 'scroll_down'}))}
COMMIT, FINISH, EXPLORE, UNKNOWN = ('commit', 'finish', 'explore', 'unknown')

def parse_args(arguments) -> dict | None:
    if isinstance(arguments, dict):
        return arguments
    if arguments is None or arguments == '':
        return {}
    try:
        obj = json.loads(arguments)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None

def classify_call(harness: str, name: str, args: dict) -> str:
    spec = HARNESSES[harness]
    if name in spec.finish_tools:
        return FINISH
    if name in spec.editor_tools:
        return COMMIT if args.get('command') in EDIT_COMMANDS else EXPLORE
    if name in spec.direct_edit_tools:
        return COMMIT
    if name in spec.explore_tools:
        return EXPLORE
    return UNKNOWN

def classify_turn(harness: str, calls: list[tuple[str, dict]]) -> tuple[bool, bool, list[str]]:
    kinds = [classify_call(harness, n, a) for n, a in calls]
    unknown = [n for (n, _), k in zip(calls, kinds) if k == UNKNOWN]
    is_finish = FINISH in kinds
    return (COMMIT in kinds or is_finish, is_finish, unknown)