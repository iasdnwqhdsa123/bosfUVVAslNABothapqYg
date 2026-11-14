from __future__ import annotations
import ast
import os
import re
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_GAP = 45
OTHER = ['06:55', '18:55']

def forecast_slots() -> list[str]:
    with open(os.path.join(HERE, '..', 'pipeline', 'check.py'), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any((getattr(t, 'id', None) == 'SLOTS' for t in node.targets)):
            return sorted(ast.literal_eval(node.value))
    raise SystemExit('slots: no SLOTS in check.py')

def busy() -> list[int]:
    return sorted((int(s[:2]) * 60 + int(s[3:]) for s in forecast_slots() + OTHER))

def crons(path: str) -> list[str]:
    with open(path, encoding='utf-8') as f:
        return re.findall('cron:\\s*"([^"]+)"', f.read())

def times(cron: str) -> list[int]:
    m, h = cron.split()[:2]
    hs = range(24) if h == '*' else [int(x) for x in h.split(',')]
    return [hh * 60 + int(mm) for hh in hs for mm in m.split(',')]

def gap_min(cron: str) -> int:
    b = busy()
    return min((min(abs(t - s), 1440 - abs(t - s)) for t in times(cron) for s in b))

def windows() -> list[tuple[int, int]]:
    b = busy()
    out = []
    for s, n in zip(b, b[1:] + [b[0] + 1440]):
        if n - s >= 2 * MIN_GAP:
            out.append((s + MIN_GAP, n - MIN_GAP))
    return out
if __name__ == '__main__':
    for a, z in windows():
        print(f'{a // 60 % 24:02d}:{a % 60:02d}-{z // 60 % 24:02d}:{z % 60:02d} UTC')
    sys.exit(0)
