from __future__ import annotations
import re
MAX_FILES = 8
MAX_SYMBOLS = 16
DEFAULT_TEMPLATE = 'So far you have been working through this task on your own. Suppose that, at this point, you learned where the correct fix lives:\n{lines}\nOnly locations are given, not the fix itself.'
_DIFF_GIT = re.compile('^diff --git a/(?P<a>\\S+) b/(?P<b>\\S+)\\s*$')
_PLUS = re.compile('^\\+\\+\\+ b/(?P<p>\\S+)')
_HUNK = re.compile('^@@ [^@]*@@(?P<tail>.*)$')
_SYMBOL = re.compile('\\b(?:def|class)\\s+([A-Za-z_]\\w*)')

def parse_patch(patch: str, max_files: int=MAX_FILES, max_symbols: int=MAX_SYMBOLS) -> tuple[list[str], list[str]]:
    files: list[str] = []
    symbols: list[str] = []
    seen_f: set[str] = set()
    seen_s: set[str] = set()
    for line in (patch or '').splitlines():
        m = _DIFF_GIT.match(line)
        if m:
            p = m['b']
        else:
            m = _PLUS.match(line)
            p = m['p'] if m else None
        if p and p not in seen_f:
            seen_f.add(p)
            files.append(p)
            continue
        h = _HUNK.match(line)
        if h:
            for s in _SYMBOL.findall(h['tail']):
                if s not in seen_s:
                    seen_s.add(s)
                    symbols.append(s)
    return (files[:max_files], symbols[:max_symbols])

def build_hint(patch: str, template: str=DEFAULT_TEMPLATE) -> str | None:
    files, symbols = parse_patch(patch)
    if not files:
        return None
    lines = ['Files: ' + ', '.join(files)]
    if symbols:
        lines.append('Functions/classes: ' + ', '.join(symbols))
    return template.format(lines='\n'.join(lines))