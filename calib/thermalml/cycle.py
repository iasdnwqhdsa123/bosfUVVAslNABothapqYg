from __future__ import annotations
import math
from . import join as jn
from . import learn as lab
CYCLE_V = 'c2'
CYCLE = {'hist_s': 300, 'hist_max_s': 3600, 'grid_min': [0, 15, 30, 45, 60, 90, 120, 180, 240], 'reuse_min': [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240], 'periods': [[0, 12], [12, 15], [15, 24]]}

def uses(d: lab.Day, climbs: list) -> list:
    out = []
    for i in sorted(climbs, key=lambda i: (d.climbs[i]['t0'], i)):
        c = d.climbs[i]
        a, b = (c['t0'], c['t0'] + c['dur'])
        if out and a < out[-1][1]:
            s, e, who = out[-1]
            out[-1] = (s, max(e, b), who | {d.acday(c)})
        else:
            out.append((a, b, {d.acday(c)}))
    return out

def period(lt: float) -> str:
    for a, b in CYCLE['periods']:
        if a <= lt < b:
            return f'{a:02d}-{b:02d}'
    return 'all'

def km(lives: list, grid=None) -> list:
    pts = sorted(lives)
    out, S, k = ([], 1.0, 0)
    n_at = len(pts)
    times = sorted({t for t, e in pts if e})
    surv = []
    for t in times:
        while k < len(pts) and pts[k][0] < t:
            n_at -= 1
            k += 1
        d = sum((1 for x, e in pts if x == t and e))
        if n_at > 0:
            S *= 1 - d / n_at
        surv.append((t, S))
    for m in grid or CYCLE['grid_min']:
        s = 1.0
        for t, v in surv:
            if t <= m * 60:
                s = v
        out.append([m, lab.r(s, 4)])
    return out

def summary(intervals: list, lives: list, who: set, open_: list) -> dict:
    import numpy as np
    out = {'acdays': lab.acdays(who), 'intervals': {'n': len(intervals), 'censored': len(open_)}, 'life': {'n': len(lives), 'ended': sum((1 for _, e in lives if e))}}
    if intervals:
        out['intervals']['km'] = km([(x, True) for x in intervals] + [(x, False) for x in open_], CYCLE['reuse_min'])
    if intervals:
        q = np.quantile(np.array(intervals, float), [0.25, 0.5, 0.75])
        hist = {}
        for x in intervals:
            b = min(int(x // CYCLE['hist_s']) * CYCLE['hist_s'], CYCLE['hist_max_s'])
            hist[b] = hist.get(b, 0) + 1
        out['intervals'].update(q25=lab.r(q[0], 0), q50=lab.r(q[1], 0), q75=lab.r(q[2], 0), hist=sorted(([k, v] for k, v in hist.items())))
    if lives:
        out['life']['km'] = km(lives)
    return out

def seen_until(d: lab.Day) -> dict:
    out = {}
    c = d.cols
    if len(c.get('h', [])):
        keep = d.row_ok & (c['part'] == 0) & (c['cov'] > 0)
        for h, i, j in zip(c['h'][keep].tolist(), (c['i'][keep] // lab.CELL).tolist(), (c['j'][keep] // lab.CELL).tolist()):
            t = int(h) * 3600 + 1800
            if t > out.get((i, j), 0):
                out[i, j] = t
    return out

def learn(days: list) -> dict:
    from . import trigger as tg
    acc, tstar = ({}, {})
    multi = sources = 0
    for d in days:
        seen = seen_until(d)
        grid, g = ({}, 0.01)
        for f, x in enumerate(d.failed):
            if d.failed_ok[f]:
                grid.setdefault((math.floor(x['la0'] / g), math.floor(x['lo0'] / g)), []).append(f)
        att = {}
        for t, x in enumerate(d.source_of()):
            if x is not None:
                att.setdefault(x, set()).update((f for f in d.thermals[t].get('failed', []) if d.failed_ok[f]))
        for s, cl in sorted(tg.source_climbs(d).items()):
            us = uses(d, cl)
            sources += 1
            if len(us) > 1:
                multi += 1
            c0 = d.climbs[cl[0]]
            there = set(att.get(s, ()))
            gi, gj = (math.floor(c0['la0'] / g), math.floor(c0['lo0'] / g))
            for k in ((gi + a, gj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)):
                for f in grid.get(k, ()):
                    if jn._dist((c0['la0'], c0['lo0']), (d.failed[f]['la0'], d.failed[f]['lo0'])) <= jn.JOIN['r_m']:
                        there.add(f)
            ends = sorted((d.failed[f]['t0'] for f in there))
            life = (us[-1][1] - us[0][0], any((f >= us[-1][1] for f in ends)))
            ivs = [us[k + 1][0] - us[k][0] for k in range(len(us) - 1)]
            last = us[-1][0]
            until = max([f for f in ends if f > last] + [seen.get(lab.lcell(c0['la0'], c0['lo0']), 0)])
            open_ = [until - last] if until > last else []
            who = set().union(*(u[2] for u in us))
            cond = (getattr(d, 'cond', None) or {}).get((*lab.lcell(c0['la0'], c0['lo0']), tg.hour_bin_of(c0['t0'] // 3600, lab.ecell(c0['la0'], c0['lo0'])[1])))
            ts = cond['zi'] / cond['wstar'] if cond and cond.get('zi') and cond.get('wstar') and (cond['wstar'] > 0.3) else None
            if ts and ivs:
                t = tstar.setdefault(cond['regime'], [[], set()])
                t[0].extend((x / ts for x in ivs))
                t[1].update(who)
            for b in ('all', period(c0['lt'])):
                a = acc.setdefault(b, ([], [], set(), []))
                a[0].extend(ivs)
                a[1].append(life)
                a[2].update(who)
                a[3].extend(open_)
    bins = {b: summary(*a) for b, a in sorted(acc.items()) if lab.acdays(a[2]) >= lab.K}
    ts = {}
    for reg, (xs, who) in sorted(tstar.items()):
        if lab.acdays(who) >= lab.K and xs:
            q = sorted(xs)
            at = lambda p: lab.r(q[min(len(q) - 1, int(p * len(q)))], 3)
            ts[reg] = {'n': len(q), 'p25': at(0.25), 'median': at(0.5), 'p75': at(0.75)}
    return {'params': CYCLE, 'sources': sources, 'multi_use': multi, 'bins': bins, 'reuse_tstar': ts}
