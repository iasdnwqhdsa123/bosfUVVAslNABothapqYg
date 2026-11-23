from __future__ import annotations
import bisect
import math
from collections import Counter
from . import events as ev
from . import objects
JOIN_V = 'j1'
JOIN = {'r_m': 600.0, 'gap_s': 900, 'reach_m': 500.0, 'cell_deg': 0.5}
M_DEG = 111320.0

def stamp() -> str:
    return f'{ev.EVENTS_V}-{JOIN_V}'

def _dist(a, b) -> float:
    k = math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot((a[0] - b[0]) * M_DEG, (a[1] - b[1]) * M_DEG * k)

class Column:

    def __init__(self, c: dict):
        tr = c.get('tr') or [[c['la0'], c['lo0'], c['z0'], c['t0']], [c['la'], c['lo'], c['zt'], c['t0'] + c['dur']]]
        span = max(c['zt'] - c['z0'], 1)
        pts = [(p[2], p[0], p[1], p[3] if p[3] is not None else c['t0'] + c['dur'] * (p[2] - c['z0']) / span) for p in tr]
        pts.sort()
        self.z = [p[0] for p in pts]
        self.p = pts
        self.z0, self.zt = (self.z[0], self.z[-1])
        self.t0, self.t1 = (c['t0'], c['t0'] + c['dur'])
        self.w = sorted(((p[2], p[0], p[1]) for p in c.get('w') or []))
        self.first = (c['la0'], c['lo0'], c['z0'], c['t0'])
        self.top = (c['la'], c['lo'], c['zt'], c['t0'] + c['dur'])

    def at(self, z: float, cap: bool=True):
        if cap and (z < self.z0 - JOIN['reach_m'] or z > self.zt + JOIN['reach_m']):
            return None
        if z <= self.z0:
            p = self.p[0]
            return (p[1], p[2], p[3])
        if z >= self.zt:
            p = self.p[-1]
            return (p[1], p[2], p[3])
        k = max(1, bisect.bisect_left(self.z, z))
        a, b = (self.p[k - 1], self.p[k])
        f = (z - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
        return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3]))

    def wind(self, z: float):
        if not self.w:
            return (0.0, 0.0)
        if z <= self.w[0][0]:
            return (self.w[0][1], self.w[0][2])
        if z >= self.w[-1][0]:
            return (self.w[-1][1], self.w[-1][2])
        k = bisect.bisect_left([x[0] for x in self.w], z)
        a, b = (self.w[k - 1], self.w[k])
        f = (z - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
        return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]))

    def reach(self, pt, cap: bool=True):
        c = self.at(pt[2], cap)
        if c is None:
            return None
        dt = pt[3] - c[2]
        u, v = self.wind(pt[2])
        moved = (c[0] + v * dt / M_DEG, c[1] + u * dt / (M_DEG * max(math.cos(math.radians(c[0])), 1e-06)))
        return (dt, _dist((c[0], c[1]), pt[:2]), _dist(moved, pt[:2]))

def link(A: Column, B: Column):
    if B.t0 > A.t1 + JOIN['gap_s']:
        return None
    cap = B.t0 > A.t1
    best = None
    for r in (A.reach(B.first, cap), B.reach(A.top, cap)):
        if r is not None and (best is None or min(r[1], r[2]) < min(best[1], best[2])):
            best = r
    if best is None or min(best[1], best[2]) > JOIN['r_m']:
        return None
    return best

class _Sets:

    def __init__(self, n: int):
        self.up = list(range(n))

    def find(self, x: int) -> int:
        while self.up[x] != x:
            self.up[x] = self.up[self.up[x]]
            x = self.up[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = (self.find(a), self.find(b))
        if a != b:
            self.up[max(a, b)] = min(a, b)

def _cell(la: float, lo: float):
    c = JOIN['cell_deg']
    return (math.floor(la / c), math.floor(lo / c))

def join(e: dict) -> dict:
    climbs, failed = (e['climbs'], e['failed'])
    if any((climbs[i]['t0'] > climbs[i + 1]['t0'] for i in range(len(climbs) - 1))):
        raise ValueError("thermalml: an events object's climbs are not in start order")
    cols = [Column(c) for c in climbs]
    n = len(cols)
    sets, links, cnt = (_Sets(n), [], Counter())
    by_cell, starts = ({}, {})
    for i, c in enumerate(climbs):
        k = _cell(c['la0'], c['lo0'])
        by_cell.setdefault(k, []).append(i)
        starts.setdefault(k, []).append(cols[i].t0)
    for a in range(n):
        A = cols[a]
        for b in near(by_cell, starts, climbs[a]['la0'], climbs[a]['lo0'], A.t0, A.t1 + JOIN['gap_s']):
            if b <= a:
                continue
            B = cols[b]
            cnt['pairs'] += 1
            r = link(A, B)
            if r is None:
                continue
            links.append([a, b, int(round(r[0])), int(round(r[1])), int(round(r[2]))])
            sets.union(a, b)
    roots = {}
    for i in range(n):
        roots.setdefault(sets.find(i), []).append(i)
    order = sorted(roots.values(), key=lambda m: m[0])
    th_of = [0] * n
    thermals = []
    for k, m in enumerate(order):
        for i in m:
            th_of[i] = k
        f = climbs[m[0]]
        thermals.append({'t0': f['t0'], 't1': max((climbs[i]['t0'] + climbs[i]['dur'] for i in m)), 'la': f['la0'], 'lo': f['lo0'], 'z0': f['z0'], 'zt': max((climbs[i]['zt'] for i in m)), 'n': len(m), 'na': len({climbs[i]['ac'] for i in m}), 'first': m[0], 'climbs': m, 'failed': []})
    fl, longest = ([], max([c['dur'] for c in climbs] or [0]))
    for fi, f in enumerate(failed):
        pt = (f['la0'], f['lo0'], f['z0'], f['t0'])
        best = None
        for b in near(by_cell, starts, f['la0'], f['lo0'], f['t0'] - JOIN['gap_s'] - longest, f['t0'] + JOIN['gap_s']):
            B = cols[b]
            if f['t0'] < B.t0 - JOIN['gap_s'] or f['t0'] > B.t1 + JOIN['gap_s']:
                continue
            r = B.reach(pt)
            if r is None:
                continue
            d = min(r[1], r[2])
            if d <= JOIN['r_m'] and (best is None or (d, th_of[b]) < best):
                best = (d, th_of[b])
        if best:
            thermals[best[1]]['failed'].append(fi)
            fl.append([fi, best[1], int(round(best[0]))])
        else:
            fl.append([fi, None, None])
    glides = []
    for i, c in enumerate(climbs):
        p = c.get('prev')
        if p is None or c.get('gl') is None:
            continue
        g = c['gl']
        glides.append({'climb': i, 'prev': p, 'from': th_of[p], 'to': th_of[i], 'same': th_of[p] == th_of[i], 'z': climbs[p]['zt'], **{k: g.get(k) for k in ev.GL_FIELDS}})
    cnt.update(climbs=n, thermals=len(thermals), links=len(links), multi=sum((1 for t in thermals if t['n'] > 1)), shared=sum((1 for t in thermals if t['na'] > 1)), failed_attached=sum((1 for x in fl if x[1] is not None)), failed_alone=sum((1 for x in fl if x[1] is None)), glides=len(glides))
    return {'thermals': thermals, 'links': links, 'failed': fl, 'glides': glides, 'climb_thermal': th_of, 'counts': dict(sorted(cnt.items()))}

def near(by_cell: dict, starts: dict, la: float, lo: float, t_from: float, t_to: float) -> list[int]:
    ci, cj = _cell(la, lo)
    out = []
    for k in ((ci + di, cj + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)):
        if k in by_cell:
            ts = starts[k]
            out += by_cell[k][bisect.bisect_left(ts, t_from):bisect.bisect_right(ts, t_to)]
    return sorted(out)

def build(lk, sd: str, raw: dict, cache: dict | None=None) -> dict:
    k = objects.key('events', ev.EVENTS_V, sd, objects.ready(sd, raw)[1])
    e = objects.loads(lk.get(k)) if lk.exists(k) else ev.build(lk, sd, raw, cache)
    j = join(e)
    return {'v': 1, 'rule': JOIN_V, 'events': ev.EVENTS_V, 'sd': sd, 'params': dict(JOIN), 'missing': e.get('missing', []), 'feeds': e.get('feeds', []), **j}

def run(lk, days=None, cache=None, n=None, most: int | None=None) -> dict:

    def make(sd, raw, missing):
        obj = build(lk, sd, raw, cache)
        return (objects.dumps(obj), {'thermals': len(obj['thermals'])})
    return objects.run_days(lk, 'join', stamp(), make, days, cache, n, most)
