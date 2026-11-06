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
STREET_S, FIX_GAP_S, STREET_KM_MIN, WAVE_PAD = (600.0, 60.0, 2.0, 500.0)
ALONG_DEG = 30.0
ST_BINS = (0.0, 0.3, 0.5, 0.7, 1.01)
CALM = 4.0
ROLL_FIELDS = ('wstar', 'zi', 'depth', 'u_blavg', 'v_blavg', 'u_sfc', 'v_sfc', 'u_bltop', 'v_bltop', 'u_g100', 'v_g100', 'dth_m200', 'twidth', 'blapse', 'cubase')
ROLL_FULL = ('u_sfc', 'v_sfc', 'u_bltop', 'v_bltop', 'dth_m200')

class Boxes:

    def __init__(self, root: str):
        self.root, self.cache, self.no_conv, self.used, self.full = (root, {}, set(), set(), set())

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
            if fns:
                (self.used if has else self.no_conv).add(k)
            names = tuple(dict.fromkeys(FIELDS + ROLL_FIELDS))
            f = {n: [] for n in names}
            stamps, full = ([], bool(fns))
            for fn in fns:
                with np.load(fn) as z:
                    if has and 'conv' not in z.files:
                        continue
                    shape = z[next(iter(z.files))].shape
                    stamps.append(datetime.strptime(os.path.basename(fn)[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp())
                    full = full and all((n in z.files and n in m['fields'] for n in ROLL_FULL))
                    for n in names:
                        f[n].append(decode(m['fields'].get(n), z[n]) if n in z.files and n in m['fields'] else np.full(shape, np.nan, np.float32))
            if stamps:
                B = {'g': m['grid'], 'stamps': np.array(stamps), 'f': {n: np.stack(v) for n, v in f.items()}, 'conv': has, 'full': full}
                if full:
                    self.full.add(k)
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
    streets = not getattr(args, 'no_streets', False)
    start, n_fl, n, n_st, st_fl = (time.time(), 0, defaultdict(int), 0, set())
    sout = open(os.path.join(args.out, 'streets.jsonl'), 'w') if streets else None
    with open(os.path.join(args.out, 'lines.jsonl'), 'w') as out:
        for d in rowsmod.load_days(args.data):
            if (time.time() - start) / 60 > args.budget:
                log('out of time: the report has the days done')
                break
            m = rowsmod.model_of(d, None)
            gkey = d.get('key') or d['date']
            pend = defaultdict(list)
            for f in d.get('flights') or []:
                bi = (f.get('model_' + m) or {}).get('box')
                if bi is None or not f.get('track'):
                    continue
                B = boxes.get(gkey, m, bi)
                if not B:
                    continue
                cls = cls_of.get(f.get('ac')) or '15m'
                if streets:
                    try:
                        for p in street_pieces(f, B, cls):
                            p.update(day=gkey, model=m, fid=f.get('id'), region=f.get('region'), cls=cls)
                            pend[bi].append(p)
                    except Exception as e:
                        print(f"a flight's street pieces not read: {e!r}", file=sys.stderr)
                if not B['conv']:
                    continue
                try:
                    rs = runs(f, B, cls)
                except Exception as e:
                    print(f'a flight not read: {e!r}', file=sys.stderr)
                    continue
                n_fl += 1
                for r in rs:
                    j = judge(r, B, m)
                    j.update(day=gkey, model=m, fid=f.get('id'), region=f.get('region'), tz=f.get('tz'))
                    out.write(json.dumps(j) + '\n')
                    n[j['kind']] += 1
            for bi, ps in pend.items():
                for p in judge_streets(ps, boxes.get(gkey, m, bi), log):
                    sout.write(json.dumps(p) + '\n')
                    n_st += 1
                    st_fl.add(p['fid'])
            boxes.clear()
    if sout:
        sout.close()
    common.write_json(os.path.join(args.out, 'meta.json'), {'boxes': len(boxes.used), 'boxes_no_conv': len(boxes.no_conv), 'flights': n_fl, 'boxes_full': len(boxes.full), 'street_pieces': n_st, 'street_flights': len(st_fl)})
    log(f"{n_fl} flights read: {n['line']} rising runs, {n['sink']} sink runs, {n['straight']} straight pieces; {len(boxes.used)} boxes, {len(boxes.no_conv)} saved without the convergence (before 4 Nov 2026)")
    if streets:
        log(f"streets: {n_st} pieces in {len(st_fl)} flights; {len(boxes.full)} of {len(boxes.used) + len(boxes.no_conv)} boxes with the roll model's winds and cap")

def _overlap(a: float, b: float, spans) -> float:
    return sum((max(0.0, min(b, hi) - max(a, lo)) for lo, hi in spans))

def street_pieces(f: dict, B, cls: str) -> list[dict]:
    tr = [p for p in f.get('track') or [] if p[4] is not None]
    if len(tr) < 10:
        return []
    cs = track._climbs_sorted(f)
    climbs = [(c['_t0'], c['_t1']) for c in cs]
    engon = [(c['_t0'], c['_t1']) for c in cs if c.get('engon')]
    t_eng = track.engine_start(f)
    t_off, t_end = (tr[0][0] + EDGE_S, min(tr[-1][0], t_eng if t_eng is not None else math.inf) - EDGE_S)
    out, i = ([], 0)
    while i < len(tr) - 1:
        if tr[i][0] < t_off:
            i += 1
            continue
        j = i
        while j + 1 < len(tr) and tr[j + 1][0] - tr[i][0] <= STREET_S and (tr[j + 1][0] - tr[j][0] <= FIX_GAP_S):
            j += 1
        if tr[j][0] > t_end:
            break
        if tr[j][0] - tr[i][0] < 0.9 * STREET_S:
            i = j + 1
            continue
        r = _street_piece(tr[i:j + 1], B, cls, climbs, engon)
        if r:
            out.append(r)
        i = j
    return out

def _street_piece(P: list, B, cls: str, climbs, engon) -> dict | None:
    t0, t1 = (P[0][0], P[-1][0])
    T = t1 - t0
    if any((len(p) > 5 and p[5] is not None and (p[5] > ENGINE_ON) for p in P)) or _overlap(t0, t1, engon) > 0:
        return None
    mid = P[len(P) // 2]
    h, ij = (hour(B, (t0 + t1) / 2), cell(B, mid[1], mid[2]))
    if h is None or ij is None:
        return None
    x = {n: float(B['f'][n][h][ij]) if n in B['f'] else math.nan for n in dict.fromkeys(FIELDS + ROLL_FIELDS)}
    u, v = (x['u_blavg'], x['v_blavg'])
    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(x['wstar'])):
        return None
    if math.isfinite(x['ridge']) and x['ridge'] > RIDGE_MAX:
        return None
    agl = float(np.mean([p[3] - p[4] for p in P]))
    zi = x['zi'] if math.isfinite(x['zi']) else x['depth']
    top = (zi if math.isfinite(zi) else 0) + (x['elevation'] if math.isfinite(x['elevation']) else mid[4])
    if agl < AGL_MIN or float(np.mean([p[3] for p in P])) > top + WAVE_PAD:
        return None
    km = track.dist_km(P[0][1:3], P[-1][1:3])
    if km < STREET_KM_MIN:
        return None
    flown = sum((track.dist_km(a[1:3], b[1:3]) for a, b in zip(P, P[1:])))
    hdg = track.bearing(P[0][1:3], P[-1][1:3])
    dx, dy = (km * 1000 * math.sin(math.radians(hdg)), km * 1000 * math.cos(math.radians(hdg)))
    xc_air = math.hypot(dx - u * T, dy - v * T) / T * 3.6
    circ = min(1.0, _overlap(t0, t1, climbs) / T)
    pad = [(lo - CLIMB_PAD, hi + CLIMB_PAD) for lo, hi in climbs]
    runs_, cur = ([], [])
    for a, b in zip(P, P[1:]):
        if _overlap(a[0], b[0], pad) > 0:
            if len(cur) >= 3:
                runs_.append(cur)
            cur = []
        else:
            cur = cur or [a]
            cur.append(b)
    if len(cur) >= 3:
        runs_.append(cur)
    g_km = g_dz = g_t = g_n = 0.0
    for R in runs_:
        k = sum((track.dist_km(a[1:3], b[1:3]) for a, b in zip(R, R[1:])))
        dt = R[-1][0] - R[0][0]
        if k <= 0 or dt <= 0:
            continue
        nt, _ = rc.netto([list(p[:4]) for p in R], u, v, cls)
        g_km, g_dz, g_t, g_n = (g_km + k, g_dz + (R[-1][3] - R[0][3]), g_t + dt, g_n + nt * dt)
    r1 = lambda q: None if q is None or not math.isfinite(q) else round(q, 1)
    return {'kind': 'street', 't0': t0, 't1': t1, 'lon': mid[1], 'lat': mid[2], 'hdg': round(hdg), 'km': round(km, 2), 'straight': round(km / flown, 3) if flown > 0 else None, 'circ': round(circ, 3), 'xc_air': r1(xc_air), 'xc_gs': r1(km / T * 3600), 'glide': round(g_t / T, 3), 'loss': r1(-g_dz / g_km) if g_km >= 0.5 else None, 'gnetto': round(g_n / g_t, 2) if g_t >= 60 else None, 'dz': round(P[-1][3] - P[0][3]), 'agl': round(agl), 'wind': round(math.hypot(u, v), 1), 'wto': round((math.degrees(math.atan2(u, v)) + 360) % 360), 'ws': round(x['wstar'], 2), 'zi': r1(zi), '_air': [None if not math.isfinite(x[n]) else round(x[n], 3) for n in ROLL_FIELDS]}

def axis_off(a: float, b: float) -> float:
    d = abs((a - b) % 180)
    return min(d, 180 - d)

def node_rolls(rows: list, hdgs: list, log=print) -> list | None:
    import subprocess
    import tempfile
    import satcheck
    script = satcheck.node_script()
    if not script:
        log('  streets: no node script')
        return None
    with tempfile.TemporaryDirectory() as d:
        fi, fo = (os.path.join(d, 'in.json'), os.path.join(d, 'out.json'))
        with open(fi, 'w') as w:
            json.dump({'fields': list(ROLL_FIELDS), 'rows': rows, 'hdg': hdgs}, w)
        r = subprocess.run(['node', script, 'rolls', fi, fo], capture_output=True, text=True, timeout=600)
        if r.returncode or not os.path.exists(fo):
            log(f"  streets: the planner's rolls failed: {r.stderr[-300:]}")
            return None
        with open(fo) as fh:
            return json.load(fh)['rolls']

def judge_streets(ps: list, B, log=print) -> list:
    if not ps:
        return []
    R = node_rolls([p['_air'] for p in ps], [p['hdg'] for p in ps], log) or [None] * len(ps)
    out = []
    for p, r in zip(ps, R):
        q = {k: v for k, v in p.items() if k != '_air'}
        q['offw'] = round(axis_off(p['hdg'], p['wto']), 1)
        if r is None:
            q.update(st=None, ang=None, sp=None, full=None, factor=None, off=None)
        else:
            q.update(st=r[0], ang=r[1], sp=round(r[2] / 1000, 2), full=bool(r[3]), factor=r[4], off=round(axis_off(p['hdg'], r[1]), 1))
        out.append(q)
    return out

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

def _med(X, k):
    v = [r[k] for r in X if r.get(k) is not None]
    return float(np.median(v)) if v else float('nan')

def _mean(X, k):
    v = [r[k] for r in X if r.get(k) is not None]
    return float(np.mean(v)) if v else float('nan')

def _ratio_ci(A: list, X: list, k: str='xc_air', n: int=400, seed: int=1):
    if not A or not X:
        return (float('nan'), float('nan'), float('nan'))
    r0 = _med(A, k) / _med(X, k)
    by = defaultdict(lambda: ([], []))
    for r in A:
        by[r.get('fid')][0].append(r[k])
    for r in X:
        by[r.get('fid')][1].append(r[k])
    fl = list(by)
    rng, out = (np.random.default_rng(seed), [])
    for _ in range(n):
        pick = rng.integers(0, len(fl), len(fl))
        a = [v for i in pick for v in by[fl[i]][0] if v is not None]
        x = [v for i in pick for v in by[fl[i]][1] if v is not None]
        if a and x:
            out.append(np.median(a) / np.median(x))
    lo, hi = np.percentile(out, [5, 95]) if out else (float('nan'), float('nan'))
    return (r0, float(lo), float(hi))

def street_bins(P: list) -> list:
    lo = ST_BINS[1]
    out = [(f'none (under {lo:g}), wind under {CALM:g} m/s', [r for r in P if r['st'] < lo and r['wind'] < CALM]), (f'none (under {lo:g}), wind {CALM:g} m/s and more', [r for r in P if r['st'] < lo and r['wind'] >= CALM])]
    for a, b in zip(ST_BINS[1:], ST_BINS[2:]):
        out.append((f'{a:g}-{min(b, 1):g}', [r for r in P if a <= r['st'] < b]))
    return out

def street_summary(S: list, meta: dict | None=None) -> str:
    P = [r for r in S if r.get('st') is not None and r.get('xc_air') is not None]
    md = ['', '## Streets', '', f"{len(P)} pieces of {STREET_S / 60:.0f} min of cross-country flight ({STREET_KM_MIN:g} km and more made good, {AGL_MIN:.0f} m and more up, the engine off, no ridge lift, not over the forecast's top: no wave) in {len({r['fid'] for r in P})} flights on {len({r['day'] for r in P})} days, against the planner's rolls at their middle (xctask plannerRoll: the strength, the rows' angle; streetFactor: the planner's thermal cruise on the piece's course over its XC speed). Along: the course within {ALONG_DEG:.0f}° of the rows' line; across: {90 - ALONG_DEG:.0f}° and more off it.", '']
    if meta and meta.get('boxes_full') is not None:
        nb = (meta.get('boxes') or 0) + (meta.get('boxes_no_conv') or 0)
        md += [f"""Boxes with the roll model's 10 m and top winds and cap: {meta['boxes_full']} of {nb} (saved since 5 Nov 2026); on the rest the rolls are the wind-only estimate (the 10 m wind from the layer's, no turning, shear or cap: "wind-only" below).""", '']
    if not P:
        return '\n'.join(md) + '\n'
    fp = sum((1 for r in P if r.get('full'))) / len(P)
    md += [f'{fp:.0%} of the pieces on boxes with the full roll model.', '', "XC through the air: the way made good less the forecast's wind drift in the time, km/h (median). Straight: made good / flown. Circling: the share of the time in the flight's climbs. Glides: height lost a km and netto (m/s) clear of the climbs. Planner: its factor on the course (mean).", '', '| rolls (strength) | course | pieces | flights | straight | circling | XC through the air | glides m/km | glide netto | planner |', '|---|---|---|---|---|---|---|---|---|---|']
    along = lambda X: [r for r in X if r['off'] <= ALONG_DEG]
    across = lambda X: [r for r in X if r['off'] >= 90 - ALONG_DEG]
    bins = street_bins(P)
    for name, X in bins:
        for cn, Y in (('along', along(X)), ('across', across(X))):
            if not Y:
                md.append(f'| {name} | {cn} | 0 | | | | | | | |')
                continue
            md.append(f"| {name} | {cn} | {len(Y)} | {len({r['fid'] for r in Y})} | {_med(Y, 'straight'):.2f} | {_mean(Y, 'circ'):.0%} | {_med(Y, 'xc_air'):.0f} | {_med(Y, 'loss'):.0f} | {_mean(Y, 'gnetto'):+.2f} | {_mean(Y, 'factor'):.2f} |")
    md += ['', "Along over across, by the strength (the XC speed through the air: the medians' ratio, 90 % interval with the flights resampled; the planner's the same of its factor; circling and straightness along less across):", '', '| rolls (strength) | along / across | flown | 90 % | planner | circling | straight | glides m/km |', '|---|---|---|---|---|---|---|---|']
    for name, X in bins:
        A, C = (along(X), across(X))
        r0, lo, hi = _ratio_ci(A, C)
        pl = _mean(A, 'factor') / _mean(C, 'factor') if A and C else float('nan')
        md.append(f"| {name} | {len(A)} / {len(C)} | {r0:.2f} | {lo:.2f}-{hi:.2f} | {pl:.2f} | {(_mean(A, 'circ') - _mean(C, 'circ')) * 100:+.0f} pts | {_med(A, 'straight') - _med(C, 'straight'):+.2f} | {_med(A, 'loss') - _med(C, 'loss'):+.0f} |")
    Q = [r for r in P if r['st'] >= 0.5]
    if Q:
        md += ['', f"Strength 0.5 and more ({len(Q)} pieces): the rows' angle against the wind's line (the rolls turned up to 15° toward the shear):", '', '| line | along | across | along / across |', '|---|---|---|---|']
        for nm, key in (("the rows'", 'off'), ("the wind's", 'offw')):
            A = [r for r in Q if r[key] <= ALONG_DEG]
            C = [r for r in Q if r[key] >= 90 - ALONG_DEG]
            md.append(f"| {nm} | {len(A)} | {len(C)} | {(_med(A, 'xc_air') / _med(C, 'xc_air') if A and C else float('nan')):.2f} |")
        md += ['', "The same by the roll model's inputs (full: the 10 m and top winds and cap; wind-only: boxes saved before 5 Nov 2026):", '', '| rolls | pieces | along / across | flown | planner |', '|---|---|---|---|---|']
        for nm, Y in (('full', [r for r in Q if r.get('full')]), ('wind-only', [r for r in Q if not r.get('full')])):
            A, C = (along(Y), across(Y))
            r0 = _med(A, 'xc_air') / _med(C, 'xc_air') if A and C else float('nan')
            pl = _mean(A, 'factor') / _mean(C, 'factor') if A and C else float('nan')
            md.append(f'| {nm} | {len(Y)} | {len(A)} / {len(C)} | {r0:.2f} | {pl:.2f} |')
    md += ['', "Read with care: pilots choose to fly along streets (the across pieces are mostly a task's crosswind legs), and a piece's XC speed through the air leans on the forecast's wind. The control (no streets, windy) says how much of an along / across difference the wind's line gives by itself."]
    return '\n'.join(md) + '\n'

def report(args, log=print) -> None:
    fn = os.path.join(args.out, 'lines.jsonl')
    R = [json.loads(x) for x in open(fn)] if os.path.exists(fn) else []
    sfn = os.path.join(args.out, 'streets.jsonl')
    S = [json.loads(x) for x in open(sfn)] if os.path.exists(sfn) else []
    mfn = os.path.join(args.out, 'meta.json')
    meta = common.read_json(mfn) if os.path.exists(mfn) else None
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write(summary(R, meta))
        if S or (meta and meta.get('street_pieces') is not None):
            f.write(street_summary(S, meta))
    log(f'{len(R)} runs and {len(S)} street pieces in the report')

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--data', required=True)
    r.add_argument('--grids', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--budget', type=float, default=240)
    r.add_argument('--no-streets', action='store_true', help='the lines only')
    p = sub.add_parser('report')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    (run if a.cmd == 'run' else report)(a)
if __name__ == '__main__':
    main()
