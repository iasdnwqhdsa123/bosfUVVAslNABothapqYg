from __future__ import annotations
from . import learn as lab
STRENGTH_V = 's4'
STRENGTH = {'cell_deg': 0.1, 'hour_bin': 2, 'band_m': 500, 'q': 0.8, 'shrink': 5}

def _key(d, la, lo, lt, z, y):
    import math
    cd = STRENGTH['cell_deg']
    return (d.sd, (math.floor(la / cd), math.floor(lo / cd)), lab.hour_bin(lt, STRENGTH['hour_bin']), int(z // STRENGTH['band_m']) * STRENGTH['band_m'], y)

def failed_samples(d: lab.Day):
    for i, f in enumerate(d.failed):
        if d.failed_ok[i]:
            yield (_key(d, f['la0'], f['lo0'], f['lt'], (f['z0'] + f['zt']) / 2, str(f.get('y') or 'glider')), d.acday(f))

def glide_samples(d: lab.Day):
    for g in d.join.get('glides', []):
        i, p = (g.get('climb'), g.get('prev'))
        if g.get('nm') is None or i is None or p is None or (not (d.climb_ok[i] and d.climb_ok[p])):
            continue
        a, b = (d.climbs[p], d.climbs[i])
        la, lo = ((a.get('la', a['la0']) + b['la0']) / 2, (a.get('lo', a['lo0']) + b['lo0']) / 2)
        lt = (b['lt'] - (g.get('t') or 0) / 7200.0) % 24
        yield (_key(d, la, lo, lt, (a['zt'] + b['z0']) / 2, str(b.get('y') or 'glider')), float(g['nm']), d.acday(b))

def offsets(samples: list) -> dict:
    by_key = {}
    for k, v, w in samples:
        a = by_key.setdefault(k, {})
        a.setdefault(w, []).append(v)
    dev = {}
    for k, per in by_key.items():
        if len(per) < 2:
            continue
        tot, n = (sum((sum(x) for x in per.values())), sum((len(x) for x in per.values())))
        for w, xs in per.items():
            others = (tot - sum(xs)) / (n - len(xs))
            a = dev.setdefault(w, [0.0, 0])
            a[0] += sum((x - others for x in xs))
            a[1] += len(xs)
    return {w: s / (n + STRENGTH['shrink']) for w, (s, n) in dev.items()}

def samples(d: lab.Day):
    import math
    for i, c in enumerate(d.climbs):
        if not d.climb_ok[i]:
            continue
        cd = STRENGTH['cell_deg']
        cell = (math.floor(c['la0'] / cd), math.floor(c['lo0'] / cd))
        hb = lab.hour_bin(c['lt'], STRENGTH['hour_bin'])
        vz = c.get('vz')
        got = []
        if vz and vz.get('r'):
            for k, v in enumerate(vz['r']):
                if v is not None:
                    got.append((vz['z0'] + 100 * k + 50, v))
        if not got and c.get('rate') is not None:
            got.append(((c['z0'] + c['zt']) / 2, c['rate']))
        air = c.get('air')
        mid = (c['z0'] + c['zt']) / 2
        y = str(c.get('y') or 'glider')
        for z, v in got:
            band = int(z // STRENGTH['band_m']) * STRENGTH['band_m']
            yield ((d.sd, cell, hb, band, y), v, None, d.acday(c), i)
        if air is not None:
            yield ((d.sd, cell, hb, int(mid // STRENGTH['band_m']) * STRENGTH['band_m'], 'air'), None, air, d.acday(c), i)

def learn(days: list) -> dict:
    import numpy as np
    acc, rs, ars, fails, glides = ({}, [], [], {}, {})
    for d in days:
        for key, v, air, who, i in samples(d):
            a = acc.setdefault(key, ([], [], set(), set(), set(), [], []))
            if v is not None:
                a[0].append(v)
                a[2].add(who)
                a[5].append(who)
                rs.append((key, v, who))
            if air is not None:
                a[1].append(air)
                a[3].add(who)
                a[6].append(who)
                ars.append((key, air, who))
            a[4].add((d.sd, i))
        for key, who in failed_samples(d):
            f = fails.setdefault(key, [0, set()])
            f[0] += 1
            f[1].add(who)
        for key, nm, who in glide_samples(d):
            g = glides.setdefault(key, [[], set()])
            g[0].append(nm)
            g[1].add(who)
    off_r, off_a = (offsets(rs), offsets(ars))
    rows, dropped = ([], 0)
    cd = STRENGTH['cell_deg']
    q = STRENGTH['q']
    mean_q = lambda xs: (lab.r(float(np.mean(xs)), 2), lab.r(float(np.quantile(xs, q)), 2)) if xs else (None, None)
    empty = ([], [], set(), set(), set(), [], [])
    for key in sorted(set(acc) | set(fails) | set(glides)):
        sd, (ci, cj), hb, band, y = key
        vs, airs, wv, wa, cl, wvs, was = acc.get(key, empty)
        vs, wvs = (vs, wvs) if lab.acdays(wv) >= lab.K else ([], [])
        airs, was = (airs, was) if lab.acdays(wa) >= lab.K else ([], [])
        f = fails.get(key)
        nf = f[0] if f and lab.acdays(f[1]) >= lab.K else 0
        g = glides.get(key)
        gs = g[0] if g and lab.acdays(g[1]) >= lab.K else []
        if not vs and (not airs) and (not nf) and (not gs):
            dropped += 1
            continue
        nc = len(cl) if vs or airs else 0
        rows.append([sd, round((ci + 0.5) * cd, 3), round((cj + 0.5) * cd, 3), hb, band, y, nc, *mean_q(vs), *mean_q(airs), *mean_q([v - off_r.get(w, 0.0) for v, w in zip(vs, wvs)]), *mean_q([v - off_a.get(w, 0.0) for v, w in zip(airs, was)]), nf, lab.r(nc / (nc + nf), 3) if y != 'air' and nc + nf else None, len(gs), lab.r(float(np.mean(gs)), 2) if gs else None])
    return {'params': STRENGTH, 'cols': ['sd', 'la', 'lo', 'hour_bin', 'band_m', 'y', 'climbs', 'rate_mean', 'rate_q', 'air_mean', 'air_q', 'rate_adj_mean', 'rate_adj_q', 'air_adj_mean', 'air_adj_q', 'failed', 'found', 'glides', 'glide_netto'], 'rows': rows, 'below_k': dropped, 'offsets': {'rate': len(off_r), 'air': len(off_a)}}
