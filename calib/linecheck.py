from __future__ import annotations
import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))
import common
import ridgecheck as rc
import track
NETTO_MIN = 0.5
LINE_S = 120.0
WIN = 3
LEG_S = (10.0, 40.0)
MIN_VG = 18.0
TURN_MAX = 40.0
AGL_MIN = 300.0
TOP_PAD = 200.0
RIDGE_MAX = 0.3
EDGE_S, CLIMB_PAD, ENGINE_ON = (300.0, 30.0, 500.0)
CONV_LINE = (0.15, 0.3)
CONV_SCALE = {'icon': 0.5, 'icond2': 0.5, 'ecmwf': 0.5}
SEARCH_KM = 40.0
NEAR_KM = (2, 5, 10, 20)
FIELDS = ('conv', 'u_blavg', 'v_blavg', 'zi', 'elevation', 'ridge', 'wstar', 't2')

class Boxes:

    def __init__(self, root: str):
        self.root, self.cache, self.no_conv, self.used = (root, {}, set(), set())

    def get(self, day: str, model: str, bi: int):
        k = (day, model, bi)
        if k in self.cache:
            return self.cache[k]
        base = os.path.join(self.root, day, f'{model}-box{bi}')
        B = None
        if os.path.exists(os.path.join(base, 'latest.json')):
            m = common.read_json(os.path.join(base, 'latest.json'))
            fns = sorted(glob.glob(os.path.join(base, '_hours', '*.npz')))
            has = bool(fns) and 'conv' in m.get('fields', {})
            if has:
                with np.load(fns[0]) as z:
                    has = 'conv' in z.files
            (self.used if has else self.no_conv).add(k)
            if has:
                f = {n: [] for n in FIELDS}
                stamps = []
                for fn in fns:
                    with np.load(fn) as z:
                        if 'conv' not in z.files:
                            continue
                        stamps.append(datetime.strptime(os.path.basename(fn)[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp())
                        for n in FIELDS:
                            f[n].append(decode(m['fields'].get(n), z[n]) if n in z.files and n in m['fields'] else np.full(z['conv'].shape, np.nan, np.float32))
                if stamps:
                    B = {'g': m['grid'], 'stamps': np.array(stamps), 'f': {n: np.stack(v) for n, v in f.items()}}
        self.cache[k] = B
        return B

    def clear(self):
        self.cache.clear()

def decode(f: dict, q) -> np.ndarray:
    lo, hi = (f['min'], f['max'])
    q = np.asarray(q)
    if f.get('transform') == 'sqrt':
        a, b = (math.sqrt(max(lo, 0)), math.sqrt(hi))
        v = (a + q.astype(np.float32) / 254 * (b - a)) ** 2
    else:
        v = lo + q.astype(np.float32) / 254 * (hi - lo)
    return np.where(q == 255, np.nan, v).astype(np.float32)

def cell(B, lon, lat):
    g = B['g']
    ix, iy = (round((lon - g['west']) / g['step']), round((g['north'] - lat) / g['step']))
    return (iy, ix) if 0 <= ix < g['nx'] and 0 <= iy < g['ny'] else None

def hour(B, t) -> int | None:
    h = int(np.argmin(np.abs(B['stamps'] - t)))
    return h if abs(B['stamps'][h] - t) <= 3600 else None

def nearest_line(B, h: int, pts, thr: float) -> float | None:
    g = B['g']
    c = B['f']['conv'][h]
    lat0 = float(np.mean([p[1] for p in pts]))
    kx, ky = (111.32 * math.cos(math.radians(lat0)), 110.54)
    r = int(SEARCH_KM / (g['step'] * ky)) + 1
    best = None
    for lon, lat in pts:
        ij = cell(B, lon, lat)
        if ij is None:
            continue
        iy, ix = ij
        y0, y1, x0, x1 = (max(0, iy - r), min(g['ny'], iy + r + 1), max(0, ix - 2 * r), min(g['nx'], ix + 2 * r + 1))
        sub = c[y0:y1, x0:x1]
        yy, xx = np.nonzero(np.nan_to_num(sub, nan=-9) >= thr)
        if not len(yy):
            continue
        clat = g['north'] - (yy + y0) * g['step']
        clon = g['west'] + (xx + x0) * g['step']
        d = float(np.min(np.hypot((clon - lon) * kx, (clat - lat) * ky)))
        best = d if best is None else min(best, d)
    return None if best is None or best > SEARCH_KM else round(best, 1)

def runs(f: dict, B, cls: str) -> list[dict]:
    tr = [p for p in f.get('track') or [] if p[4] is not None]
    if len(tr) < WIN + 2:
        return []
    climbs = [(c['_t0'] - CLIMB_PAD, c['_t1'] + CLIMB_PAD) for c in track._climbs_sorted(f)]
    t_off, t_land = (tr[0][0] + EDGE_S, tr[-1][0] - EDGE_S)

    def air(p):
        h = hour(B, p[0])
        ij = cell(B, p[1], p[2])
        if h is None or ij is None:
            return None
        return {n: float(B['f'][n][h][ij]) for n in FIELDS}
    A = [air(p) for p in tr]

    def leg_ok(i):
        a, b = (tr[i], tr[i + 1])
        dt = b[0] - a[0]
        if not LEG_S[0] <= dt <= LEG_S[1] or a[0] < t_off or b[0] > t_land or (A[i] is None):
            return False
        if any((lo < b[0] and a[0] < hi for lo, hi in climbs)):
            return False
        if any((len(p) > 5 and p[5] is not None and (p[5] > ENGINE_ON) for p in (a, b))):
            return False
        x = A[i]
        top = (x['zi'] if math.isfinite(x['zi']) else 0) + (x['elevation'] if math.isfinite(x['elevation']) else a[4])
        if a[3] - a[4] < AGL_MIN or a[3] > top + TOP_PAD or (math.isfinite(x['ridge']) and x['ridge'] > RIDGE_MAX):
            return False
        return track.dist_km(a[1:3], b[1:3]) * 1000 / dt >= MIN_VG
    ok = [leg_ok(i) for i in range(len(tr) - 1)]
    hd = [track.bearing(tr[i][1:3], tr[i + 1][1:3]) for i in range(len(tr) - 1)]
    straight = [ok[i] and (i == 0 or not ok[i - 1] or abs((hd[i] - hd[i - 1] + 180) % 360 - 180) < TURN_MAX) for i in range(len(ok))]
    sign = [0] * len(ok)
    nets = [None] * len(ok)
    for i in range(len(ok) - WIN + 1):
        if not all(straight[i:i + WIN]):
            continue
        x = A[i + WIN // 2]
        u, v = (x['u_blavg'], x['v_blavg'])
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        n, _ = rc.netto([list(p[:4]) for p in tr[i:i + WIN + 1]], u, v, cls)
        nets[i] = n
        s = 1 if n >= NETTO_MIN else -1 if n <= -NETTO_MIN else 0
        for j in range(i, i + WIN):
            if s and sign[j] in (0, s):
                sign[j] = s
            elif s and sign[j] == -s:
                sign[j] = 2
    out, i = ([], 0)
    while i < len(ok):
        s = sign[i]
        if s not in (1, -1):
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and sign[j + 1] == s and straight[j + 1]:
            j += 1
        a, b = (tr[i], tr[j + 1])
        if b[0] - a[0] >= LINE_S:
            fx = [list(p[:4]) for p in tr[i:j + 2]]
            mid = tr[(i + j + 1) // 2]
            x = A[(i + j + 1) // 2]
            n, va = rc.netto(fx, x['u_blavg'], x['v_blavg'], cls)
            out.append({'kind': 'line' if s > 0 else 'sink', 't0': a[0], 't1': b[0], 'lon': mid[1], 'lat': mid[2], 'pts': [[p[1], p[2]] for p in tr[i:j + 2]], 'netto': round(n, 2), 'va': round(va, 1), 'km': round(sum((track.dist_km(tr[k][1:3], tr[k + 1][1:3]) for k in range(i, j + 1))), 1), 'agl': round(float(np.mean([p[3] - p[4] for p in tr[i:j + 2]]))), 'hdg': round(track.bearing(a[1:3], b[1:3]))})
        i = j + 1
    i = 0
    while i < len(ok):
        if not straight[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and straight[j + 1] and (tr[j + 1][0] - tr[i][0] < LINE_S):
            j += 1
        a, b = (tr[i], tr[j + 1])
        if b[0] - a[0] >= LINE_S:
            mid, x = (tr[(i + j + 1) // 2], A[(i + j + 1) // 2])
            if math.isfinite(x['u_blavg']) and math.isfinite(x['v_blavg']):
                n, va = rc.netto([list(p[:4]) for p in tr[i:j + 2]], x['u_blavg'], x['v_blavg'], cls)
                out.append({'kind': 'straight', 't0': a[0], 't1': b[0], 'lon': mid[1], 'lat': mid[2], 'pts': [[p[1], p[2]] for p in tr[i:j + 2]], 'netto': round(n, 2), 'va': round(va, 1), 'km': round(track.dist_km(a[1:3], b[1:3]), 1), 'agl': round(float(np.mean([p[3] - p[4] for p in tr[i:j + 2]]))), 'hdg': round(track.bearing(a[1:3], b[1:3]))})
        i = j + 1
    return out

def judge(r: dict, B, model: str | None=None) -> dict:
    sc = CONV_SCALE.get(model, 1.0)
    t = (r['t0'] + r['t1']) / 2
    h = hour(B, t)
    out = {k: v for k, v in r.items() if k != 'pts'}
    pts = r['pts'][::max(1, len(r['pts']) // 8)]
    conv = []
    for lon, lat in pts:
        ij = cell(B, lon, lat)
        if ij is not None and h is not None:
            conv.append(float(B['f']['conv'][h][ij]))
    out['conv'] = round(float(np.nanmax(conv)), 3) if conv and np.isfinite(conv).any() else None
    x = {n: float(B['f'][n][h][cell(B, r['lon'], r['lat'])]) if h is not None and cell(B, r['lon'], r['lat']) else None for n in FIELDS}
    out.update({'wind': None if x['u_blavg'] is None else round(math.hypot(x['u_blavg'], x['v_blavg']), 1), 'wdir': None if x['u_blavg'] is None else round((math.degrees(math.atan2(-x['u_blavg'], -x['v_blavg'])) + 360) % 360), 'ws': x['wstar'], 'zi': x['zi'], 'ground': x['elevation'], 't2': x['t2']})
    for thr in CONV_LINE:
        best = None
        for dh in (0, -1, 1):
            hh = None if h is None else h + dh
            if hh is None or not 0 <= hh < len(B['stamps']):
                continue
            d = nearest_line(B, hh, pts, thr * sc)
            if dh == 0:
                out[f'd{thr}'] = d
            if d is not None and (best is None or d < best[0] - 0.5):
                best = (d, dh)
        out[f'best{thr}'] = None if best is None else best[0]
        out[f'dh{thr}'] = None if best is None else best[1]
    return out

def run(args, log=print) -> None:
    import rows as rowsmod
    os.makedirs(args.out, exist_ok=True)
    cls_of = rc.classes()
    boxes = Boxes(args.grids)
    start, n_fl, n = (time.time(), 0, defaultdict(int))
    with open(os.path.join(args.out, 'lines.jsonl'), 'w') as out:
        for d in rowsmod.load_days(args.data):
            if (time.time() - start) / 60 > args.budget:
                log('out of time: the report has the days done')
                break
            m = rowsmod.model_of(d, None)
            gkey = d.get('key') or d['date']
            for f in d.get('flights') or []:
                bi = (f.get('model_' + m) or {}).get('box')
                if bi is None or not f.get('track'):
                    continue
                B = boxes.get(gkey, m, bi)
                if not B:
                    continue
                try:
                    rs = runs(f, B, cls_of.get(f.get('ac')) or '15m')
                except Exception as e:
                    print(f'a flight not read: {e!r}', file=sys.stderr)
                    continue
                n_fl += 1
                for r in rs:
                    j = judge(r, B, m)
                    j.update(day=gkey, model=m, fid=f.get('id'), region=f.get('region'), tz=f.get('tz'))
                    out.write(json.dumps(j) + '\n')
                    n[j['kind']] += 1
            boxes.clear()
    common.write_json(os.path.join(args.out, 'meta.json'), {'boxes': len(boxes.used), 'boxes_no_conv': len(boxes.no_conv), 'flights': n_fl})
    log(f"{n_fl} flights read: {n['line']} rising runs, {n['sink']} sink runs, {n['straight']} straight pieces; {len(boxes.used)} boxes, {len(boxes.no_conv)} saved without the convergence (before 4 Nov 2026)")

def _share(R, key, km):
    return sum((1 for r in R if r.get(key) is not None and r[key] <= km)) / len(R) if R else float('nan')

def summary(R: list, meta: dict | None=None) -> str:
    L = [r for r in R if r['kind'] == 'line']
    S = [r for r in R if r['kind'] == 'sink']
    thr = CONV_LINE[0]
    md = ['# Lines check', '', f"{len(L)} rising runs (straight flight, {NETTO_MIN} m/s or more up for {LINE_S:.0f} s and more, inside the thermal layer, {AGL_MIN:.0f} m or more up, no ridge lift) in {len({r['fid'] for r in L})} flights on {len({r['day'] for r in L})} days; {len(S)} sink runs as controls.", '', (f"Forecast boxes read: {meta['boxes']}; {meta['boxes_no_conv']} saved without the convergence (before 4 Nov 2026: their days come in as they're built again).\n\n" if meta else '') + f'A forecast line: convergence {thr} m/s or more (build.py conv; ' + ', '.join((f'{m} {thr * s:g}' for m, s in sorted(CONV_SCALE.items()))) + "). The nearest at the run's hour, and at the best of the hours either side.", '']
    if not L:
        return '\n'.join(md) + '\n'

    def row(name, X):
        if not X:
            return f'| {name} | 0 |' + ' - |' * (len(NEAR_KM) + 4)
        on = [r for r in X if (r.get('conv') or 0) >= thr * CONV_SCALE.get(r.get('model'), 1.0)]
        none = sum((1 for r in X if r.get(f'd{thr}') is None)) / len(X)
        dd = [r[f'd{thr}'] for r in X if r.get(f'd{thr}') is not None]
        return f'| {name} | {len(X)} | {len(on) / len(X):.0%} | ' + ' | '.join((f"{_share(X, f'd{thr}', k):.0%}" for k in NEAR_KM)) + f" | {none:.0%} | {(np.median(dd) if dd else float('nan')):.1f} | {_share(X, f'best{thr}', 5):.0%} |"
    T = [r for r in R if r['kind'] == 'straight']
    md += [f'| runs | n | on a line | ' + ' | '.join((f'within {k} km' for k in NEAR_KM)) + f' | none within {SEARCH_KM:.0f} km | median km | within 5 km, an hour either side |', '|---|---|---|' + '---|' * len(NEAR_KM) + '---|---|---|', row('rising runs', L), row('sink runs', S), row(f'all straight flight ({LINE_S:.0f} s pieces: the baseline)', T)]
    if T:
        md += ['', f"Does flying near a forecast line find lift? Of the straight pieces, the share in lift ({NETTO_MIN} m/s or more) and their mean netto, by the distance to the forecast's nearest line:", '', '| nearest line | pieces | in lift | mean netto |', '|---|---|---|---|']
        for lo, hi, name in ((0, 2, 'on it (under 2 km)'), (2, 5, '2-5 km'), (5, 10, '5-10 km'), (10, 20, '10-20 km'), (20, 1000000000.0, '20 km and more, or none')):
            X = [r for r in T if lo <= (r.get(f'd{thr}') if r.get(f'd{thr}') is not None else 100000000.0) < hi]
            if X:
                md.append(f"| {name} | {len(X)} | {sum((r['netto'] >= NETTO_MIN for r in X)) / len(X):.0%} | {np.mean([r['netto'] for r in X]):+.2f} |")
    md += ['', 'Rising runs by what the forecast had there:', '', f'| | n | on a line | ' + ' | '.join((f'within {k} km' for k in NEAR_KM)) + f' | none within {SEARCH_KM:.0f} km | median km | within 5 km, an hour either side |', '|---|---|---|' + '---|' * len(NEAR_KM) + '---|---|---|']
    for name, X in (('W* under 1.5 m/s', [r for r in L if (r.get('ws') or 0) < 1.5]), ('W* 1.5 m/s and more', [r for r in L if (r.get('ws') or 0) >= 1.5]), ('wind under 5 m/s', [r for r in L if (r.get('wind') or 0) < 5]), ('wind 5 m/s and more', [r for r in L if (r.get('wind') or 0) >= 5]), ('long (10 km and more)', [r for r in L if r['km'] >= 10]), ('strong (1.5 m/s and more)', [r for r in L if r['netto'] >= 1.5])):
        md.append(row(name, X))
    regs = defaultdict(list)
    for r in L:
        regs[r.get('region') or '?'].append(r)
    md += ['', 'By region:', '', '| region | n | on a line | ' + ' | '.join((f'within {k} km' for k in NEAR_KM)) + ' | none | median km | ±1 h within 5 km |', '|---|---|---|' + '---|' * len(NEAR_KM) + '---|---|---|']
    for k in sorted(regs, key=lambda k: -len(regs[k])):
        md.append(row(k, regs[k]))
    mods = defaultdict(list)
    for r in L:
        mods[r.get('model') or '?'].append(r)
    md += ['', 'By model (its own cut):', '', '| model | n | on a line | ' + ' | '.join((f'within {k} km' for k in NEAR_KM)) + ' | none | median km | ±1 h within 5 km |', '|---|---|---|' + '---|' * len(NEAR_KM) + '---|---|---|']
    for k in sorted(mods, key=lambda k: -len(mods[k])):
        md.append(row(k, mods[k]))
    md += ['', "The forecast's convergence on the straight pieces, and what they found:", '']
    for lo, hi in ((-9, 0.0), (0.0, 0.15), (0.15, 0.3), (0.3, 9)):
        X = [r for r in T if r.get('conv') is not None and lo <= r['conv'] < hi]
        if X:
            md.append(f"- conv [{lo:g}, {hi:g}): {len(X)} straight pieces, {sum((r['netto'] >= NETTO_MIN for r in X)) / len(X):.0%} in lift, mean netto {np.mean([r['netto'] for r in X]):+.2f}")
    dh = defaultdict(int)
    for r in L:
        if r.get(f'dh{thr}') is not None and r[f'best{thr}'] <= 5:
            dh[r[f'dh{thr}']] += 1
    if dh:
        md += ['', 'The hour that put a line within 5 km nearest: ' + ', '.join((f'{k:+d} h {v}' for k, v in sorted(dh.items())))]
    return '\n'.join(md) + '\n'

def report(args, log=print) -> None:
    fn = os.path.join(args.out, 'lines.jsonl')
    R = [json.loads(x) for x in open(fn)] if os.path.exists(fn) else []
    mfn = os.path.join(args.out, 'meta.json')
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write(summary(R, common.read_json(mfn) if os.path.exists(mfn) else None))
    log(f'{len(R)} runs in the report')

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--data', required=True)
    r.add_argument('--grids', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--budget', type=float, default=240)
    p = sub.add_parser('report')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    (run if a.cmd == 'run' else report)(a)
if __name__ == '__main__':
    main()
