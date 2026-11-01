from __future__ import annotations
from datetime import date, timedelta
START = date(2021, 1, 1)
RECENT_DAYS, RECENT_LAG = (30, 3)
MAX_TRIES = 3
BITS = 12

def bitrev(i: int, bits: int=BITS) -> int:
    return int(f'{i:0{bits}b}'[::-1], 2)

def spread(days: list[str]) -> list[str]:
    return sorted(days, key=lambda d: (bitrev((date.fromisoformat(d) - START).days % (1 << BITS)), d))

def flying(d: date) -> bool:
    return d.month not in (11, 12, 1, 2) or d.weekday() == 5

def new_state(seed_days: list[str]) -> dict:
    return {'version': 1, 'seed_days': spread(sorted(set(seed_days))), 'done': {}, 'counts': {}, 'unmapped': {}, 'seen': [], 'weglide': {'requests': 0, 'status': {}}, 'runs': []}

def _open(state: dict, d: str) -> bool:
    x = state['done'].get(d)
    return x is None

def _retry(state: dict, d: str) -> bool:
    x = state['done'].get(d)
    return x is not None and x.get('status') == 'failed' and (x.get('tries', 0) < MAX_TRIES)

def next_days(state: dict, n: int, today: date, new_ok: bool=True) -> list[tuple[str, str]]:
    out = []
    seed = set(state['seed_days'])
    for d in state['seed_days']:
        if len(out) >= n:
            return out
        if _open(state, d):
            out.append((d, 'seed'))
    if new_ok:
        last = today - timedelta(days=RECENT_LAG)
        recent = [last - timedelta(days=k) for k in range(RECENT_DAYS)]
        for d in recent:
            s = d.isoformat()
            if len(out) >= n:
                return out
            if flying(d) and s not in seed and _open(state, s):
                out.append((s, 'new'))
        k = 0
        span = (last - START).days
        rest = spread([(START + timedelta(days=i)).isoformat() for i in range(span + 1)])
        for s in rest:
            if len(out) >= n:
                return out
            if flying(date.fromisoformat(s)) and s not in seed and _open(state, s):
                out.append((s, 'new'))
                k += 1
    for d in list(state['seed_days']) + sorted(state['done']):
        if len(out) >= n:
            break
        if _retry(state, d) and all((d != x for x, _ in out)):
            out.append((d, 'seed' if d in seed else 'new'))
    return out

def progress(state: dict, today: date) -> dict:
    done = state['done']
    seed_left = sum((1 for d in state['seed_days'] if d not in done))
    by = {}
    for x in done.values():
        by[x.get('status')] = by.get(x.get('status'), 0) + 1
    return {'seed_days': len(state['seed_days']), 'seed_left': seed_left, 'days_done': len(done), 'by_status': by}
