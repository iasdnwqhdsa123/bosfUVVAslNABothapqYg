from __future__ import annotations
import math
from . import join as jn
from . import learn as lab
MOVEMENT_V = 'm1'
MOVE = {'tell_m': 300.0, 'pair_m': 2000.0, 'pair_s': 2700, 'min_dt_s': 120, 'min_wind': 2.0, 'r_range': [-1.0, 2.0], 'wind_ms': [0, 4, 8], 'iters': 300}

def mean_wind(cs: list) -> tuple[float, float] | None:
    pts = [p for c in cs for p in c.get('w') or []]
    if not pts:
        return None
    return (sum((p[0] for p in pts)) / len(pts), sum((p[1] for p in pts)) / len(pts))

def wind_class(U: float) -> str:
    edges = MOVE['wind_ms']
    for k in range(len(edges) - 1, -1, -1):
        if U >= edges[k]:
            return f'{edges[k]}+' if k == len(edges) - 1 else f'{edges[k]}-{edges[k + 1]}'
    return 'all'

def thermal_votes(d: lab.Day) -> list:
    votes = {}
    for a, b, dt, dfx, ddr in d.join.get('links', []):
        t = d.join['climb_thermal'][a]
        w = mean_wind([d.climbs[a], d.climbs[b]])
        U = math.hypot(*w) if w else 0.0
        v = votes.setdefault(t, [0, 0, []])
        v[2].append(U)
        if U * abs(dt) < MOVE['tell_m']:
            continue
        if dfx < ddr:
            v[0] += 1
        elif ddr < dfx:
            v[1] += 1
    out = []
    for t, th in enumerate(d.thermals):
        if not d.thermal_ok[t]:
            continue
        who = {d.acday(d.climbs[i]) for i in th['climbs']}
        if th['n'] < 2:
            out.append((t, 'single', None, who))
            continue
        an, dr, Us = votes.get(t, [0, 0, []])
        dec = 'anchored' if an > dr else 'drifting' if dr > an else 'undecided'
        out.append((t, dec, wind_class(sum(Us) / len(Us)) if Us else None, who))
    return out

def pairs(d: lab.Day) -> list:
    idx = [i for i, ok in enumerate(d.climb_ok) if ok]
    idx.sort(key=lambda i: (d.climbs[i]['t0'], i))
    cols = {i: jn.Column(d.climbs[i]) for i in idx}
    out = []
    for x, a in enumerate(idx):
        A, ca = (cols[a], d.climbs[a])
        for b in idx[x + 1:]:
            cb = d.climbs[b]
            if cb['t0'] - ca['t0'] > MOVE['pair_s']:
                break
            if jn._dist((ca['la0'], ca['lo0']), (cb['la0'], cb['lo0'])) > MOVE['pair_m']:
                continue
            B = cols[b]
            lo, hi = (max(A.z0, B.z0), min(A.zt, B.zt))
            if hi < lo:
                continue
            z = (lo + hi) / 2
            pa, pb = (A.at(z, cap=False), B.at(z, cap=False))
            dt = pb[2] - pa[2]
            if abs(dt) < MOVE['min_dt_s']:
                continue
            wa, wb = (A.wind(z), B.wind(z))
            u, v = ((wa[0] + wb[0]) / 2, (wa[1] + wb[1]) / 2)
            U = math.hypot(u, v)
            if U < MOVE['min_wind'] or not (A.w and B.w):
                continue
            k = math.cos(math.radians((pa[0] + pb[0]) / 2))
            dx, dy = ((pb[1] - pa[1]) * jn.M_DEG * k, (pb[0] - pa[0]) * jn.M_DEG)
            rr = (dx * u + dy * v) / (U * U * dt)
            if not MOVE['r_range'][0] <= rr <= MOVE['r_range'][1]:
                continue
            out.append((rr, (d.sd, d.join['climb_thermal'][a]), wind_class(U), {d.acday(ca), d.acday(cb)}))
    return out

def mixture(rs: list, ws: list) -> dict:
    import numpy as np
    r, w = (np.array(rs, float), np.array(ws, float))
    lo, hi = MOVE['r_range']
    pi, s = (np.array([0.4, 0.4, 0.2]), np.array([0.2, 0.2]))
    flat = 1.0 / (hi - lo)
    for _ in range(MOVE['iters']):
        p0 = pi[0] * np.exp(-0.5 * (r / s[0]) ** 2) / (s[0] * math.sqrt(2 * math.pi))
        p1 = pi[1] * np.exp(-0.5 * ((r - 1) / s[1]) ** 2) / (s[1] * math.sqrt(2 * math.pi))
        p2 = np.full_like(r, pi[2] * flat)
        tot = p0 + p1 + p2
        g = np.stack([p0, p1, p2]) / np.maximum(tot, 1e-300)
        gw = g * w
        new = gw.sum(axis=1) / w.sum()
        s = np.array([math.sqrt(max((gw[0] * r ** 2).sum() / max(gw[0].sum(), 1e-12), 0.03 ** 2)), math.sqrt(max((gw[1] * (r - 1) ** 2).sum() / max(gw[1].sum(), 1e-12), 0.03 ** 2))])
        s = np.minimum(s, 0.6)
        done = np.abs(new - pi).max() < 1e-10
        pi = new
        if done:
            break
    return {'anchored': lab.r(pi[0], 4), 'drifting': lab.r(pi[1], 4), 'background': lab.r(pi[2], 4), 's_anchored': lab.r(s[0], 4), 's_drifting': lab.r(s[1], 4)}

def learn(days: list) -> dict:
    dec, who_dec = ({}, {})
    prs = []
    for d in days:
        for t, k, wc, who in thermal_votes(d):
            for b in ('all', wc) if wc else ('all',):
                dec.setdefault(b, {}).setdefault(k, 0)
                dec[b][k] += 1
                who_dec.setdefault(b, set()).update(who)
        prs += pairs(d)
    thermals = {b: dict(sorted(v.items())) for b, v in sorted(dec.items()) if lab.acdays(who_dec[b]) >= lab.K}
    fits = {}
    for b in sorted({'all'} | {p[2] for p in prs}):
        mine = [p for p in prs if b == 'all' or p[2] == b]
        who = set().union(*(p[3] for p in mine)) if mine else set()
        if lab.acdays(who) < lab.K:
            continue
        per = {}
        for p in mine:
            per[p[1]] = per.get(p[1], 0) + 1
        fits[b] = {'pairs': len(mine), 'sources': len(per), 'acdays': lab.acdays(who), **mixture([p[0] for p in mine], [1.0 / per[p[1]] for p in mine])}
    return {'params': MOVE, 'thermals': thermals, 'mixture': fits}
