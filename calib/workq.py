from __future__ import annotations
from datetime import date, timedelta
import common
import weglide
START = date(2021, 1, 1)
EU_START = date.fromisoformat(common.EU_START)
RECENT_DAYS, RECENT_LAG = (30, 3)
MAX_TRIES = 3
REVISIT_SHARE = 1 / 3
BITS = 12

def bitrev(i: int, bits: int=BITS) -> int:
    return int(f'{i:0{bits}b}'[::-1], 2)

def spread(days: list[str]) -> list[str]:
    return sorted(days, key=lambda d: (bitrev((date.fromisoformat(common.day_of(d)) - START).days % (1 << BITS)), d))

def flying(d: date) -> bool:
    return d.month not in (11, 12, 1, 2) or d.weekday() == 5

def new_state(seed_days: list[str]) -> dict:
    return {'version': 1, 'seed_days': spread(sorted(set(seed_days))), 'done': {}, 'counts': {}, 'unmapped': {}, 'seen': [], 'weglide': {'requests': 0, 'status': {}}, 'runs': []}

def _open(state: dict, d: str) -> bool:
    x = state['done'].get(d)
    return x is None

def _retry(state: dict, d: str) -> bool:
    x = state['done'].get(d)
    return x is not None and x.get('status') in ('failed', 'pending') and (x.get('tries', 0) < MAX_TRIES)

def _redo(state: dict, d: str) -> bool:
    x = state['done'].get(d)
    if x is None or x.get('status') not in ('ok', 'partial') or x.get('redo_tries', 0) >= MAX_TRIES:
        return False
    return x.get('v', 1) < common.DAY_VERSION or (bool(x.get('grids')) and (not common.physics_matches(x.get('pv'), common.macro_of_key(d))))

def _rescore(state: dict, d: str, cv: str | None=None) -> bool:
    x = state['done'].get(d)
    if x is None or x.get('status') not in ('ok', 'partial') or (not x.get('grids')) or (x.get('rs_tries', 0) >= MAX_TRIES):
        return False
    if x.get('v', 1) < common.DAY_VERSION or not common.physics_matches(x.get('pv'), common.macro_of_key(d)):
        return False
    return x.get('sv') != common.SCORE_VERSION or x.get('cv') != cv

def _lead(state: dict, d: str) -> bool:
    x = state['done'].get(d)
    if x is None or x.get('status') not in ('ok', 'partial') or x.get('lv_tries', 0) >= MAX_TRIES:
        return False
    if x.get('v', 1) < common.DAY_VERSION or not common.physics_matches(x.get('pv'), common.macro_of_key(d)):
        return False
    want = common.lead_version(d)
    return want is not None and (not common.lead_version_matches(x.get('lv'), d))

def _revisit(state: dict, d: str, ac: str) -> bool:
    x = state['done'].get(d)
    if x is None or x.get('src') != 'new' or x.get('rv_tries', 0) >= MAX_TRIES:
        return False
    if x.get('status') == 'empty':
        return x.get('ac') != ac
    if x.get('status') not in ('ok', 'partial') or x.get('v', 1) < common.DAY_VERSION or x.get('flights', 0) >= weglide.MAX_PER_DAY:
        return False
    return x.get('ac') != ac or x.get('cap', OLD_CAP) < weglide.MAX_PER_DAY
OLD_CAP = 6

def _rv_pending(state: dict, d: str) -> bool:
    x = state['done'].get(d)
    return x is not None and x.get('src') == 'new' and (x.get('status') in ('ok', 'partial', 'empty')) and (x.get('rv_tries', 0) < MAX_TRIES)

def free_days(state: dict, n: int, pending=(), cv: str | None=None) -> list[tuple[str, str]]:
    out = []
    seed = set(state['seed_days'])
    for d in sorted(pending):
        if len(out) >= n:
            return out
        if _retry(state, d):
            out.append((d, 'seed' if d in seed else 'new'))
        elif _rv_pending(state, d):
            out.append((d, 'revisit'))
    for d in state['seed_days']:
        if len(out) >= n:
            return out
        if _open(state, d):
            out.append((d, 'seed'))
    for d in sorted(state['done']):
        if len(out) >= n:
            return out
        if _redo(state, d):
            out.append((d, 'redo'))
    for d in sorted(state['done']):
        if len(out) >= n:
            return out
        if _rescore(state, d, cv):
            out.append((d, 'rescore'))
    for d in sorted(state['done'], reverse=True):
        if len(out) >= n:
            return out
        if _lead(state, d) and (not _rescore(state, d, cv)):
            out.append((d, 'lead'))
    for d in state['seed_days']:
        if len(out) >= n:
            break
        if _retry(state, d) and d not in pending:
            out.append((d, 'seed'))
    return out

def _new_days(state: dict, today: date, seed: set, macro: str):
    start = START if macro == 'US' else EU_START
    last = today - timedelta(days=RECENT_LAG)
    recent = [(last - timedelta(days=k)).isoformat() for k in range(RECENT_DAYS) if last - timedelta(days=k) >= start]
    span = (last - start).days
    rest = spread([(start + timedelta(days=i)).isoformat() for i in range(span + 1)])
    for s in dict.fromkeys(recent + rest):
        k = common.day_key(s, macro)
        if flying(date.fromisoformat(s)) and k not in seed and _open(state, k):
            yield k

def _new_keys(state: dict, today: date, seed: set, eu_share: float | None=None):
    share = common.EU_SHARE if eu_share is None else eu_share
    us, eu = (_new_days(state, today, seed, 'US'), _new_days(state, today, seed, 'EU'))
    n_eu = k = 0
    while True:
        k += 1
        want_eu = share > 0 and n_eu + 1 <= share * k + 1e-09
        first, second = (eu, us) if want_eu else (us, eu)
        d = next(first, None)
        if d is None:
            d = next(second, None)
            if d is None:
                return
        n_eu += common.macro_of_key(d) == 'EU'
        yield d

def eu_first(state: dict) -> bool:
    return common.EU_FIRST > 0 and sum((1 for d in state['done'] if common.macro_of_key(d) == 'EU')) < common.EU_FIRST

def weglide_days(state: dict, n: int, today: date, pending=(), ac: str | None=None, listed=(), eu_share: float | None=None) -> list[tuple[str, str]]:
    out = []
    seed = set(state['seed_days'])
    for d in sorted(pending):
        if len(out) >= n:
            return out
        if _retry(state, d):
            out.append((d, 'seed' if d in seed else 'new'))
        elif _rv_pending(state, d):
            out.append((d, 'revisit'))
    ac = ac or common.aircraft_version()
    listed = set(listed)
    rv = [d for d in sorted(state['done']) if d not in pending and _revisit(state, d, ac)]
    rv.sort(key=lambda d: (d not in listed, d))
    rv = iter(rv)
    first = eu_share is None and eu_first(state)
    new = _new_keys(state, today, seed, 1.0 if first else eu_share)
    every = 10 ** 9 if first else max(2, round(1 / REVISIT_SHARE))
    k = 0
    while len(out) < n:
        k += 1
        d = next(rv, None) if k % every == 0 else None
        if d is not None:
            out.append((d, 'revisit'))
            continue
        s = next(new, None)
        if s is None:
            d = next(rv, None)
            if d is None:
                break
            out.append((d, 'revisit'))
            continue
        out.append((s, 'new'))
    for d in sorted(state['done']):
        if len(out) >= n:
            break
        if d not in seed and d not in pending and _retry(state, d) and all((d != x for x, _ in out)):
            out.append((d, 'new'))
    return out

def next_days(state: dict, n: int, today: date, new_ok: bool=True, pending=(), ac: str | None=None, listed=(), cv: str | None=None) -> list[tuple[str, str]]:
    out = weglide_days(state, n, today, pending, ac, listed) if new_ok else []
    done = {d for d, _ in out}
    return out + [x for x in free_days(state, n - len(out) + len(done & set(pending)), pending, cv) if x[0] not in done][:max(0, n - len(out))]

def progress(state: dict, today: date) -> dict:
    done = state['done']
    seed_left = sum((1 for d in state['seed_days'] if d not in done))
    by = {}
    for x in done.values():
        by[x.get('status')] = by.get(x.get('status'), 0) + 1
    redo = sum((1 for d in done if _redo(state, d)))
    rescore = sum((1 for d in done if _rescore(state, d, state.get('corr_v'))))
    ac = common.aircraft_version()
    revisit = sum((1 for d in done if _revisit(state, d, ac)))
    lead = sum((1 for d in done if _lead(state, d)))
    by_macro = {}
    for d in done:
        m = common.macro_of_key(d)
        by_macro[m] = by_macro.get(m, 0) + 1
    return {'seed_days': len(state['seed_days']), 'seed_left': seed_left, 'days_done': len(done), 'by_status': by, 'redo_left': redo, 'rescore_left': rescore, 'revisit_left': revisit, 'lead_left': lead, 'grids': sum((1 for x in done.values() if x.get('grids'))), 'by_macro': by_macro}
