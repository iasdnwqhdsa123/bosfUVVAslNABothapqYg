from __future__ import annotations
from . import learn as lab
STRENGTH_V = 's2'
STRENGTH = {'cell_deg': 0.1, 'hour_bin': 2, 'band_m': 500, 'q': 0.8}

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
    acc = {}
    for d in days:
        for key, v, air, who, i in samples(d):
            a = acc.setdefault(key, ([], [], set(), set(), set()))
            if v is not None:
                a[0].append(v)
                a[2].add(who)
            if air is not None:
                a[1].append(air)
                a[3].add(who)
            a[4].add((d.sd, i))
    rows, dropped = ([], 0)
    cd = STRENGTH['cell_deg']
    for (sd, (ci, cj), hb, band, y), (vs, airs, wv, wa, cl) in sorted(acc.items()):
        vs = vs if lab.acdays(wv) >= lab.K else []
        airs = airs if lab.acdays(wa) >= lab.K else []
        if not vs and (not airs):
            dropped += 1
            continue
        rows.append([sd, round((ci + 0.5) * cd, 3), round((cj + 0.5) * cd, 3), hb, band, y, len(cl), lab.r(float(np.mean(vs)), 2) if vs else None, lab.r(float(np.quantile(vs, STRENGTH['q'])), 2) if vs else None, lab.r(float(np.mean(airs)), 2) if airs else None, lab.r(float(np.quantile(airs, STRENGTH['q'])), 2) if airs else None])
    return {'params': STRENGTH, 'cols': ['sd', 'la', 'lo', 'hour_bin', 'band_m', 'y', 'climbs', 'rate_mean', 'rate_q', 'air_mean', 'air_q'], 'rows': rows, 'below_k': dropped}
