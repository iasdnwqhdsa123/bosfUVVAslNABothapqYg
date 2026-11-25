from __future__ import annotations
import math
import numpy as np
from . import events as E
from . import guard, objects
WINDOW_V = 'w1'
CELL = 1.0
ON_AIR = 1.7
SINK_FALLBACK = 0.7
MIN_AC = 2
QLEAD_H = 0.5
SETTLED_D = 3

def air_of(c: dict) -> float:
    a = c.get('air')
    if isinstance(a, (int, float)) and math.isfinite(a):
        return float(a)
    r = c.get('rate')
    return float(r) + SINK_FALLBACK if isinstance(r, (int, float)) and math.isfinite(r) else float('nan')

def cell_of(la: float, lo: float) -> tuple[int, int]:
    return (math.floor(la / CELL), math.floor(lo / CELL))

def rows_of(ev: dict) -> list[dict]:
    by = {}
    for k in ('climbs', 'straight', 'failed'):
        for e in ev.get(k) or []:
            c = by.setdefault(cell_of(e['la0'], e['lo0']), {'all': set(), 'on': [], 't0': [], 't1': []})
            c['all'].add(e['ac'])
            c['t0'].append(e['lt'])
            c['t1'].append(e['lt'] + e['dur'] / 3600.0)
            if k == 'climbs' and air_of(e) >= ON_AIR:
                c['on'].append(e)
    out = []
    for (i, j), c in sorted(by.items()):
        if len(c['all']) < MIN_AC:
            continue
        fly0, fly1 = (min(c['t0']), max(c['t1']))
        on = c['on']
        first = min((e['lt'] for e in on)) if on else float('nan')
        last = max((e['lt'] + e['dur'] / 3600.0 for e in on)) if on else float('nan')
        out.append({'i': i, 'j': j, 'first': first, 'last': last, 'n': len(on), 'ac': len({e['ac'] for e in on}), 'all': len(c['all']), 'fly0': fly0, 'fly1': fly1, 'lead': first - fly0, 'tail': fly1 - last, 'best': max((air_of(e) for e in on), default=float('nan'))})
    return out

def noon_utc(sd: str, lon: float) -> int:
    from datetime import datetime, timezone
    d = datetime.strptime(sd, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return int(d.timestamp() + 12 * 3600 - lon * 240.0)

def table_of(sd: str, rows: list[dict]):
    import pairs as P
    n = len(rows)
    la = np.array([(r['i'] + 0.5) * CELL for r in rows], np.float32)
    lo = np.array([(r['j'] + 0.5) * CELL for r in rows], np.float32)
    f = lambda k: np.array([r[k] for r in rows], np.float32)
    return P.table(P.obs_schema('window'), {'kind': 'window', 'obs_id': np.array([f"w{r['i']}_{r['j']}" for r in rows], dtype=object), 't_valid': np.array([noon_utc(sd, float(x)) for x in lo], np.int64), 'lat': la, 'lon': lo, 'review': np.ones(n, bool), 'o_first': f('first'), 'o_last': f('last'), 'o_n': f('n'), 'q_ac': f('ac'), 'q_all': f('all'), 'q_fly0': f('fly0'), 'q_fly1': f('fly1'), 'q_lead': f('lead'), 'q_tail': f('tail'), 'q_best': f('best')}, n)

def input_of(sd: str, ev: dict) -> str:
    import hashlib
    files = sorted((f"{x.get('day')}|{x.get('v')}|{x.get('sha256')}" for x in ev.get('inputs') or []))
    return f'window|{WINDOW_V}|{sd}|' + hashlib.sha256('\n'.join(files).encode()).hexdigest()[:16]

def run(lk, days=None, cache=None, n=None, most: int | None=None, env=None, today=None) -> dict:
    from datetime import date, timedelta
    import pairs as P
    n = n if n is not None else {}
    try:
        sds = days or guard.days_for_learning(lk, {'events': E.EVENTS_V}, env)
    except guard.Refused:
        n['off'] = 1
        return n
    settled = ((today or date.today()) - timedelta(days=objects.GRACE_DAYS + SETTLED_D)).isoformat()
    done = 0
    for sd in sds:
        if not days and sd < settled and P.markers(lk, 'obs', 'window', day=sd, batch='all'):
            n['there'] = n.get('there', 0) + 1
            continue
        if most is not None and done >= most:
            n['left'] = n.get('left', 0) + 1
            continue
        try:
            ev = guard.read_for_learning(lk, 'events', E.EVENTS_V, sd, env)
        except guard.Refused:
            n['off'] = 1
            return n
        except LookupError:
            n['no_events'] = n.get('no_events', 0) + 1
            continue

        def build(_new, _all, ev=ev, sd=sd):
            rows = rows_of(ev)
            n['cells'] = n.get('cells', 0) + len(rows)
            return table_of(sd, rows)
        r = P.write_unit(lk, 'obs', 'window', sd, 'all', [input_of(sd, ev)], build)
        n[r['status']] = n.get(r['status'], 0) + 1
        done += r['status'] != 'seen'
    return n
