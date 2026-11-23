from __future__ import annotations
import hashlib
import math
from collections import Counter
from . import events as ev
from . import exposure as ex
from . import guard
from . import join as jn
from . import objects
K = 2
CELL = 2
REGION_DEG = 10.0
READS = {'events': ev.EVENTS_V, 'exposure': ex.EXPOSURE_V, 'join': jn.stamp()}

def version(own: str) -> str:
    return f'{ev.EVENTS_V}-{ex.EXPOSURE_V}-{jn.JOIN_V}-{own}'

def ecell(la: float, lo: float) -> tuple[int, int]:
    return (math.floor(la / ex.G), math.floor(lo / ex.G))

def lcell(la: float, lo: float) -> tuple[int, int]:
    i, j = ecell(la, lo)
    return (i // CELL, j // CELL)

def centre(I: int, J: int) -> tuple[float, float]:
    d = ex.G * CELL
    return (round((I + 0.5) * d, 4), round((J + 0.5) * d, 4))

def region(la: float, lo: float) -> tuple[int, int]:
    return (math.floor(la / REGION_DEG), math.floor(lo / REGION_DEG))

def hour_bin(lt: float, width: int) -> int:
    return int(min(23.999, max(0.0, lt)) // width) * width

class Day:

    def __init__(self, sd: str, e: dict, meta: dict, cols: dict, j: dict):
        self.sd, self.events, self.meta, self.cols, self.join = (sd, e, meta, cols, j)
        self.climbs, self.failed, self.thermals = (e['climbs'], e['failed'], j['thermals'])
        self.partial = {h for h, st, *_ in meta.get('hours', []) if st == 'partial'}
        self.counted = {h for h, *_ in meta.get('hours', [])}
        self.climb_ok = [True] * len(self.climbs)
        self.failed_ok = [True] * len(self.failed)
        self.thermal_ok = [True] * len(self.thermals)
        self.row_ok = None
        self.climb_cell_ok = [True] * len(self.climbs)
        self._src = None

    def source_of(self) -> list:
        if self._src is None:
            self._src = sources(self)
        return self._src

    def acday(self, c: dict) -> tuple:
        g = (1, int(c['t0']) // 86400) if c.get('v') == 1 else (c.get('v'),)
        return (self.sd, g, c['ac'])

def acdays(who) -> int:
    groups = {}
    for sd, g, ac in who:
        groups.setdefault((sd, g), set()).add(ac)
    best = {}
    for (sd, _), acs in groups.items():
        best[sd] = max(best.get(sd, 0), len(acs))
    return sum(best.values())

def sources(d: 'Day') -> list:
    keep = [t for t, ok in enumerate(d.thermal_ok) if ok]
    par = {t: t for t in keep}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    grid, c = ({}, 0.01)
    pts = {}
    for t in keep:
        f = d.climbs[d.thermals[t]['first']]
        pts[t] = (f['la0'], f['lo0'])
        grid.setdefault((math.floor(f['la0'] / c), math.floor(f['lo0'] / c)), []).append(t)
    for t in keep:
        la, lo = pts[t]
        gi, gj = (math.floor(la / c), math.floor(lo / c))
        for k in ((gi + a, gj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)):
            for u in grid.get(k, ()):
                if u > t and jn._dist(pts[t], pts[u]) <= jn.JOIN['r_m']:
                    a, b = (find(t), find(u))
                    if a != b:
                        par[max(a, b)] = min(a, b)
    out = [None] * len(d.thermals)
    for t in keep:
        out[t] = find(t)
    return out

def load(lk, days=None, env=None, n=None) -> list[Day]:
    n = n if n is not None else {}
    have = guard.days_for_learning(lk, READS, env)
    want = sorted(set(days) & set(have)) if days else have
    out = []
    for sd in want:
        e = guard.read_for_learning(lk, 'events', READS['events'], sd, env)
        meta, cols = guard.read_for_learning(lk, 'exposure', READS['exposure'], sd, env)
        j = guard.read_for_learning(lk, 'join', READS['join'], sd, env)
        same = e.get('missing') == j.get('missing') == meta.get('missing')
        if not same or len(j.get('climb_thermal', [])) != len(e['climbs']) or len(j.get('failed', [])) != len(e['failed']):
            n['stale'] = n.get('stale', 0) + 1
            continue
        out.append(Day(sd, e, meta, cols, j))
    return out

def digest(days: list[Day]) -> str:
    h = hashlib.sha256()
    for d in days:
        h.update(d.sd.encode())
        h.update(objects.dumps({k: v for k, v in d.events.items() if k != 'inputs'}))
        h.update(objects.pack_columns({k: v for k, v in d.meta.items() if k != 'inputs'}, d.cols))
        h.update(objects.dumps(d.join))
    return h.hexdigest()[:12]

def prepare(days: list[Day]) -> dict:
    seen, most = ({}, Counter())
    for d in days:
        for part in ('climbs', 'straight', 'failed'):
            for x in d.events.get(part) or []:
                seen.setdefault(lcell(x['la0'], x['lo0']), set()).add(d.acday(x))
        c = d.cols
        if len(c.get('h', [])):
            for i, j, n in zip((c['i'] // CELL).tolist(), (c['j'] // CELL).tolist(), c['n'].tolist()):
                if n > most[i, j]:
                    most[i, j] = n
    cells = set(seen) | set(most)
    ok = {k for k in cells if max(acdays(seen.get(k, ())), most.get(k, 0)) >= K}
    cnt = Counter(cells_kept=len(ok), cells_dropped=len(cells) - len(ok))
    import numpy as np
    for d in days:
        cl_cell = [lcell(c['la0'], c['lo0']) in ok for c in d.climbs]
        d.climb_cell_ok = cl_cell
        for t, th in enumerate(d.thermals):
            d.thermal_ok[t] = all((cl_cell[i] for i in th['climbs']))
        th_of = d.join['climb_thermal']
        d.climb_ok = [cl_cell[i] and d.thermal_ok[th_of[i]] for i in range(len(d.climbs))]
        attached = {f: t for f, t, _ in d.join.get('failed', [])}
        d.failed_ok = [lcell(f['la0'], f['lo0']) in ok and (attached.get(i) is None or d.thermal_ok[attached[i]]) for i, f in enumerate(d.failed)]
        c = d.cols
        if len(c.get('h', [])):
            d.row_ok = np.array([(i, j) in ok for i, j in zip((c['i'] // CELL).tolist(), (c['j'] // CELL).tolist())], dtype=bool)
        else:
            d.row_ok = np.zeros(0, dtype=bool)
        cnt['climbs_dropped'] += d.climb_ok.count(False)
        cnt['failed_dropped'] += d.failed_ok.count(False)
        cnt['thermals_dropped'] += d.thermal_ok.count(False)
        cnt['rows_dropped'] += int((~d.row_ok).sum())
    return dict(sorted(cnt.items()))

def r(x, nd=3):
    if x is None or not isinstance(x, (int, float)) or (not math.isfinite(x)):
        return None
    return round(float(x), nd)

def run(lk, days=None, env=None, held: int=7, baseline=None, baseline_v: str | None=None, n=None) -> dict:
    from . import cycle, movement, report, strength, trigger
    if baseline is not None and (not (isinstance(baseline_v, str) and baseline_v)):
        raise ValueError('thermalml: a baseline needs its version')
    n = n if n is not None else {}
    ds = load(lk, days, env, n)
    if not ds:
        n['no_days'] = 1
        return n
    kept = prepare(ds)
    first, through, dg = (ds[0].sd, ds[-1].sd, digest(ds))
    outs = {'trigger': trigger.learn(ds), 'cycle': cycle.learn(ds), 'movement': movement.learn(ds), 'strength': strength.learn(ds)}
    vers = {'trigger': trigger.TRIGGER_V, 'cycle': cycle.CYCLE_V, 'movement': movement.MOVEMENT_V, 'strength': strength.STRENGTH_V}
    outs['report'] = report.learn(ds, outs, kept, held=held, baseline=baseline)
    outs['report']['options'] = {'held': held, 'baseline': baseline_v if baseline is not None else None}
    opt = hashlib.sha256(f"{dg}|held={held}|baseline={outs['report']['options']['baseline']}".encode()).hexdigest()[:12]
    vers['report'] = ''.join((vers[x] for x in ('trigger', 'cycle', 'movement', 'strength'))) + '-' + report.REPORT_V
    for name in objects.LEARNERS:
        body = {'v': 1, 'learner': name, 'rule': vers[name], 'reads': dict(READS), 'k': K, 'first': first, 'through': through, 'days': len(ds), **outs[name]}
        objects.put(lk, objects.learned_key(name, version(vers[name]), first, through, opt if name == 'report' else dg), objects.dumps(body), n)
    n['days'] = len(ds)
    return n
