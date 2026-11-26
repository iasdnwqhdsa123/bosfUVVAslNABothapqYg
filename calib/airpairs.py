from __future__ import annotations
import argparse
import glob
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog
import common
import lake as lakem
import linecheck as lc
import track
import windpairs
PREFIX = 'derived/air/p3/'
VERSION = 3
OFF_MIN = 5
WIN = 3
LEG_S = (10.0, 40.0)
MIN_VG = 15.0
TURN_MAX = 45.0
AGL_MIN = 15.0
LOW_AGL = 600.0
TOP_PAD = 200.0
EDGE_S, CLIMB_PAD, ENGINE_ON = (300.0, 30.0, 500.0)
BANDS = {'low': 0, 'layer': 1, 'above': 2}
WIND_OF = {0: ('u_g300', 'v_g300'), 1: ('u_blavg', 'v_blavg'), 2: ('u_bltop', 'v_bltop')}
AIR_FIELDS = ('wstar', 'zi', 'depth', 'cubase', 'sun', 'cloud', 'cloudlow', 'twidth', 'blapse', 'nlow', 'dth_m200', 'dth_m500', 't2', 'heat', 'ridge', 'sink', 'conv', 'wblmax', 'elevation', 'u_blavg', 'v_blavg', 'u_g300', 'v_g300', 'u_g100', 'v_g100', 'u_sfc', 'v_sfc', 'u_bltop', 'v_bltop', *common.GROUPS['aloft'])
WIND_FT = (3000, 6000, 10000, 15000)
WAVE_FT = (6000, 8000, 10000, 12000, 14000, 18000)
LEVEL_REACH_FT = 2000.0
FT = 0.3048
LIFT_FIELDS = ('ridge', 'sink', 'conv', 'wblmax', 'wavemax')
FAR_KM = 60.0
DRIFT_MIN_S, DRIFT_DT, DRIFT_DZ, DRIFT_MAX = (180.0, 1800.0, 600.0, 40.0)
NUM = ('t0', 't1', 'lon', 'lat', 'netto', 'va', 'agl', 'km', 'hdg', 'ws', 'zi', 'depth', 'cb_agl', 'sun', 'cloud', 'cloudlow', 'twidth', 'blapse', 'nlow', 'dth200', 'dth500', 't2', 'heat', 'ridge_g', 'ridge_gmax', 'sink', 'sinkmax', 'conv', 'convmax', 'wblmax', 'line_km', 'ws_bl', 'wd_bl', 'ws_300', 'wd_300', 'ws_100', 'ws_sfc', 'ws_top', 'along', 'ground', 'relief_line', 'rise', 'upslope', 'ridge_site', 'zr', 'solar_h', 'month', 'alt', 'wv_at', 'wavemax', 'ws_at', 'off', 'wsrc', 'fc_u', 'fc_v', 'pre')
NODE_TIMEOUT = 600

def object_key(lk, r: dict) -> str:
    return PREFIX + lk.name(f"air|{r['key']}|{r.get('sha256') or ''}")

def box_from_unit(z: dict) -> dict | None:
    meta = z.get('meta') or {}
    if not meta.get('grids'):
        return None
    enc = meta.get('enc') or {}
    stamps = [datetime.strptime(s[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp() for s in meta.get('stamps') or []]
    shape = (len(stamps), meta['grid']['ny'], meta['grid']['nx'])
    f = {}
    for n in AIR_FIELDS:
        key = f"g.{common.FIELD_GROUP.get(n, '')}.{n}"
        e = enc.get(n)
        f[n] = lc.decode({'min': e[0], 'max': e[1], 'transform': e[2]}, z[key]) if key in z and e else np.full(shape, np.nan, np.float32)
    return {'g': meta['grid'], 'stamps': np.array(stamps), 'f': f}

def _turn(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)

def pieces(f: dict, pre: list | None=None) -> list[list]:
    tr = [p for p in f.get('track') or [] if p[4] is not None and p[3] is not None]
    if len(tr) < WIN + 2:
        return []
    cs = track._climbs_sorted(f)
    climbs = [(c['_t0'], c['_t1'] + CLIMB_PAD) for c in cs]
    before = [(c['_t0'] - CLIMB_PAD, c['_t0']) for c in cs]
    t_off, t_land = (tr[0][0] + EDGE_S, tr[-1][0] - EDGE_S)

    def leg_ok(i):
        a, b = (tr[i], tr[i + 1])
        dt = b[0] - a[0]
        if not LEG_S[0] <= dt <= LEG_S[1] or a[0] < t_off or b[0] > t_land:
            return False
        if any((lo < b[0] and a[0] < hi for lo, hi in climbs)):
            return False
        if any((len(p) > 5 and p[5] is not None and (p[5] > ENGINE_ON) for p in (a, b))):
            return False
        if min(a[3] - a[4], b[3] - b[4]) < AGL_MIN:
            return False
        return track.dist_km(a[1:3], b[1:3]) * 1000 / dt >= MIN_VG
    ok = [leg_ok(i) for i in range(len(tr) - 1)]
    near = [any((lo < tr[i + 1][0] and tr[i][0] < hi for lo, hi in before)) for i in range(len(tr) - 1)]
    hd = [track.bearing(tr[i][1:3], tr[i + 1][1:3]) for i in range(len(tr) - 1)]
    out, i = ([], 0)
    while i + WIN <= len(ok):
        if all(ok[i:i + WIN]) and all((_turn(hd[j], hd[j - 1]) < TURN_MAX for j in range(i + 1, i + WIN))):
            out.append([list(p) for p in tr[i:i + WIN + 1]])
            if pre is not None:
                pre.append(1 if any(near[i:i + WIN]) else 0)
            i += WIN
        else:
            i += 1
    return out

def drift_winds(f: dict) -> list[tuple]:
    tr = [p for p in f.get('track') or [] if p[3] is not None]
    if len(tr) < 2:
        return []
    ts = [p[0] for p in tr]
    import bisect
    out = []
    for c in track._climbs_sorted(f):
        if c['_t1'] - c['_t0'] < DRIFT_MIN_S or c.get('engon'):
            continue
        a, b = (bisect.bisect_left(ts, c['_t0']), bisect.bisect_right(ts, c['_t1']) - 1)
        if b - a < 3:
            continue
        p, q = (tr[a], tr[b])
        dt = q[0] - p[0]
        if dt < DRIFT_MIN_S:
            continue
        lat = math.radians((p[2] + q[2]) / 2)
        u = (q[1] - p[1]) * 111320.0 * math.cos(lat) / dt
        v = (q[2] - p[2]) * 110540.0 / dt
        if math.hypot(u, v) > DRIFT_MAX:
            continue
        out.append(((p[0] + q[0]) / 2, float(np.mean([x[3] for x in tr[a:b + 1]])), u, v))
    return out

def own_wind(winds: list, t: float, z: float):
    best = None
    for tw, zw, u, v in winds:
        if abs(tw - t) <= DRIFT_DT and abs(zw - z) <= DRIFT_DZ:
            d = abs(tw - t) / DRIFT_DT + abs(zw - z) / DRIFT_DZ
            if best is None or d < best[0]:
                best = (d, u, v)
    return None if best is None else (best[1], best[2])

def at_height(vals: dict, levels, ft: float) -> float:
    lv = [(f, vals[f]) for f in levels if math.isfinite(vals.get(f, float('nan')))]
    if not lv or not math.isfinite(ft):
        return float('nan')
    for (f0, v0), (f1, v1) in zip(lv, lv[1:]):
        if f0 <= ft <= f1:
            return v0 + (v1 - v0) * (ft - f0) / (f1 - f0)
    near = min(lv, key=lambda x: abs(x[0] - ft))
    return near[1] if abs(near[0] - ft) <= LEVEL_REACH_FT else float('nan')

def _speed_dir(u: float, v: float) -> tuple[float, float]:
    if not (math.isfinite(u) and math.isfinite(v)):
        return (float('nan'), float('nan'))
    return (math.hypot(u, v), (math.degrees(math.atan2(-u, -v)) + 360) % 360)

def judge(fx: list, B: dict, cls: str, model: str | None=None, winds=None) -> dict | None:
    import ridgecheck as rc
    t = (fx[0][0] + fx[-1][0]) / 2
    h = lc.hour(B, t)
    mid = fx[len(fx) // 2]
    ij = lc.cell(B, mid[1], mid[2])
    if h is None or ij is None:
        return None
    F = B['f']
    at = lambda n: float(F[n][h][ij])
    cells = [c for c in (lc.cell(B, p[1], p[2]) for p in fx) if c is not None]

    def along(n):
        v = np.array([F[n][h][c] for c in cells], np.float64)
        v = v[np.isfinite(v)]
        return (float(v.mean()), float(v.max())) if len(v) else (float('nan'), float('nan'))
    agl = float(np.mean([p[3] - p[4] for p in fx]))
    zi = at('zi')
    top = zi if math.isfinite(zi) else 0.0
    band = BANDS['low'] if agl < LOW_AGL else BANDS['above'] if agl > top + TOP_PAD else BANDS['layer']
    alt_ft = float(np.mean([p[3] for p in fx])) / FT
    ua = at_height({f: at(f'u_a{f}') for f in WIND_FT}, WIND_FT, alt_ft)
    va_ = at_height({f: at(f'v_a{f}') for f in WIND_FT}, WIND_FT, alt_ft)
    ku, kv = WIND_OF[band]
    u, v = (at(ku), at(kv))
    if band == BANDS['above'] and math.isfinite(ua) and math.isfinite(va_):
        u, v = (ua, va_)
    if not (math.isfinite(u) and math.isfinite(v)):
        u, v = (at('u_blavg'), at('v_blavg'))
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    fc_u, fc_v = (u, v)
    ow = own_wind(winds or [], t, float(np.mean([p[3] for p in fx])))
    if ow is not None:
        u, v = ow
    n, va = rc.netto([list(p[:4]) for p in fx], u, v, cls)
    hdg = track.bearing(fx[0][1:3], fx[-1][1:3])
    dist = track.dist_km(fx[0][1:3], fx[-1][1:3]) * 1000
    gr = [float(p[4]) for p in fx]
    c, s = (math.cos(math.radians(hdg)), math.sin(math.radians(hdg)))
    wind_along = u * s + v * c
    rise = (gr[-1] - gr[0]) / dist if dist > 1 else float('nan')
    out = {'t0': fx[0][0], 't1': fx[-1][0], 'lon': mid[1], 'lat': mid[2], 'netto': round(n, 3), 'va': round(va, 2), 'agl': round(agl, 1), 'km': round(dist / 1000, 3), 'hdg': round(hdg), 'band': band, 'wsrc': 1 if ow is not None else 0, 'fc_u': round(fc_u, 2), 'fc_v': round(fc_v, 2)}
    out.update({'ws': at('wstar'), 'zi': zi, 'depth': at('depth'), 'sun': at('sun'), 'cloud': at('cloud'), 'cloudlow': at('cloudlow'), 'twidth': at('twidth'), 'blapse': at('blapse'), 'nlow': at('nlow'), 'dth200': at('dth_m200'), 'dth500': at('dth_m500'), 't2': at('t2'), 'heat': at('heat')})
    cb = at('cubase')
    out['cb_agl'] = cb if math.isfinite(cb) else float('nan')
    out['ridge_g'], out['ridge_gmax'] = along('ridge')
    out['sink'], out['sinkmax'] = along('sink')
    out['conv'], out['convmax'] = along('conv')
    out['wblmax'] = along('wblmax')[1]
    sc = lc.CONV_SCALE.get(model, 1.0)
    d = lc.nearest_line(B, h, [[p[1], p[2]] for p in fx], lc.CONV_LINE[0] * sc) if math.isfinite(out['conv']) else None
    out['line_km'] = FAR_KM if d is None else d
    out['ws_bl'], out['wd_bl'] = _speed_dir(at('u_blavg'), at('v_blavg'))
    out['ws_300'], out['wd_300'] = _speed_dir(at('u_g300'), at('v_g300'))
    out['ws_100'] = _speed_dir(at('u_g100'), at('v_g100'))[0]
    out['ws_sfc'] = _speed_dir(at('u_sfc'), at('v_sfc'))[0]
    out['ws_top'] = _speed_dir(at('u_bltop'), at('v_bltop'))[0]
    out.update({'along': wind_along, 'ground': float(np.mean(gr)), 'relief_line': max(gr) - min(gr), 'rise': rise, 'upslope': rise * wind_along if math.isfinite(rise) else float('nan')})
    out['zr'] = agl / zi if math.isfinite(zi) and zi > 0 else float('nan')
    out['alt'] = alt_ft * FT
    out['wv_at'] = at_height({f: at(f'wv{f}') for f in WAVE_FT}, WAVE_FT, alt_ft)
    out['wavemax'] = along('wavemax')[1]
    out['ws_at'] = math.hypot(ua, va_) if math.isfinite(ua) and math.isfinite(va_) else float('nan')
    tt = datetime.fromtimestamp(t, timezone.utc)
    out['solar_h'] = ((tt.hour + tt.minute / 60 + mid[1] / 15) % 24 + 24) % 24
    out['month'] = tt.month
    out['_q'] = [mid[2], mid[1], at('u_g300'), at('v_g300'), at('nlow')]
    return out

def site_ridge(qs: list, step: float, node: str='node', script: str | None=None, log=print) -> list:
    if not qs:
        return []
    nan = [float('nan')] * len(qs)
    script = script or next(iter(sorted(glob.glob(os.path.join(HERE, 'air_eval.*')))), None)
    if script is None or shutil.which(node) is None:
        return nan
    with tempfile.TemporaryDirectory() as td:
        qi, qo = (os.path.join(td, 'q.json'), os.path.join(td, 'a.json'))
        clean = [[x if isinstance(x, (int, float)) and math.isfinite(x) else None for x in q] for q in qs]
        with open(qi, 'w') as f:
            json.dump({'step': step, 'q': clean}, f)
        try:
            subprocess.run([node, script, qi, qo], check=True, timeout=NODE_TIMEOUT, capture_output=True, env={**os.environ, 'CALIB_PUB': common.PUB})
            with open(qo) as f:
                a = json.load(f).get('ridge') or []
        except (subprocess.SubprocessError, OSError, ValueError):
            log("air: the site's ridge lift not read (node)")
            return nan
    if len(a) != len(qs):
        return nan
    return [float('nan') if x is None else float(x) for x in a]

def offsets(ps: list[dict]) -> None:
    by = {}
    for p in ps:
        if p.get('band') == 1 and p.get('netto') is not None and math.isfinite(p['netto']):
            by.setdefault(p['_f'], []).append(p['netto'])
    med = {f: float(np.median(v)) for f, v in by.items() if len(v) >= OFF_MIN}
    mid = float(np.median(list(med.values()))) if len(med) >= 2 else None
    for p in ps:
        f = p.pop('_f', None)
        p['off'] = med[f] - mid if mid is not None and f in med else 0.0

def unit_pieces(z: dict, flights: list, model: str, cls_of: dict, ridge=site_ridge, log=print) -> list[dict]:
    B = box_from_unit(z)
    if B is None:
        return []
    out = []
    for fi, f in enumerate(flights):
        if not f.get('track'):
            continue
        cls = cls_of.get(f.get('ac')) or '15m'
        try:
            flags = []
            ps = pieces(f, flags)
            winds = drift_winds(f)
        except Exception:
            continue
        for fx, pre in zip(ps, flags):
            try:
                j = judge(fx, B, cls, model, winds)
            except Exception:
                j = None
            if j is not None:
                j['pre'] = pre
                j['region'], j['cls'], j['_f'] = (f.get('region'), cls, fi)
                out.append(j)
    offsets(out)
    vals = ridge([p['_q'] for p in out], float(B['g']['step']), log=log) if out else []
    for p, r in zip(out, vals):
        p['ridge_site'] = r
        p.pop('_q', None)
    return out

def columns(ps: list[dict], model: str, step: float) -> dict:
    n = len(ps)
    cols = {k: np.array([np.nan if p.get(k) is None else float(p[k]) for p in ps], np.float64) for k in NUM}
    cols['band'] = np.array([int(p.get('band', -1)) for p in ps], np.int8)
    cols['region'] = np.array([str(p.get('region') or '') for p in ps])
    cols['cls'] = np.array([str(p.get('cls') or '') for p in ps])
    cols['model'] = np.full(n, model)
    cols['step'] = np.full(n, float(step))
    return cols

def pack(cols: dict) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, meta=np.array(json.dumps({'v': VERSION})), **cols)
    return bio.getvalue()

def unpack(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != 'meta'}

def load_all(lk, counts=None) -> dict:
    cols = {}
    for k in lk.list(PREFIX):
        c = unpack(lk.get(k))
        if counts is not None:
            counts['units'] = counts.get('units', 0) + 1
        n = len(c.get('netto', []))
        if not n:
            continue
        for name in NUM:
            c.setdefault(name, np.full(n, np.nan))
        for name, v in c.items():
            cols.setdefault(name, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items() if len(v) == len(cols.get('netto', []))} if cols else {}

def run(lk, minutes: float=55, read_unit=None, read_items=None, boxes=None, clock=None, ridge=site_ridge, log=print) -> dict:
    import time
    import convpairs
    clock = clock or time.monotonic
    if read_unit is None:
        import labbuild
        read_unit = labbuild.read_unit
    read_items = read_items or catalog.read_items
    if boxes is None:
        import day as dayb
        boxes = dayb.boxes
    cls_of = lc.rc.classes()
    snap = catalog.pinned(lk)
    raw = (snap or {}).get('raw') or {}
    t_end = clock() + minutes * 60
    counts = {}
    for r in convpairs.flight_rows(snap):
        if clock() > t_end:
            counts['left'] = counts.get('left', 0) + 1
            continue
        key, wkey = (object_key(lk, r), windpairs.object_key(lk, r))
        need_air, need_wind = (not lk.exists(key), not lk.exists(wkey))
        if not need_air and (not need_wind):
            counts['done_before'] = counts.get('done_before', 0) + 1
            continue
        parts = (raw.get(r['day']) or {}).get('parts') or []
        try:
            z = read_unit(lk.get(r['key']))
            items = [f for f in read_items(lk, parts) if f.get('kind') != 'collector']
        except (lakem.LakeMissing, ValueError, KeyError):
            counts['unreadable'] = counts.get('unreadable', 0) + 1
            continue
        Bx = next((b for b in boxes(items) if b.get('region') == r.get('box')), None)
        flights = [f for f in (Bx or {}).get('flights') or [] if (f.get('kind') or 'flight') == 'flight']
        meta = z.get('meta') or {}
        model, step = (str(r.get('model') or ''), float((meta.get('grid') or {}).get('step') or np.nan))
        if need_wind:
            wr = windpairs.unit_rows(box_from_unit(z), flights)
            lk.put(wkey, windpairs.pack(windpairs.columns(wr, model, step)), strict=False)
            counts['wind_units'] = counts.get('wind_units', 0) + 1
            counts['wind_rows'] = counts.get('wind_rows', 0) + len(wr)
        if not need_air:
            continue
        ps = unit_pieces(z, flights, model, cls_of, ridge, log)
        cols = columns(ps, model, step)
        lk.put(key, pack(cols), strict=False)
        counts['units'] = counts.get('units', 0) + 1
        counts['pieces'] = counts.get('pieces', 0) + len(ps)
        counts['site_ridge'] = counts.get('site_ridge', 0) + int(np.isfinite(cols['ridge_site']).sum())
    return counts

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The air-motion pairs: straight flight against the forecast's air")
    ap.add_argument('--minutes', type=float, default=55)
    ap.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    x = ap.parse_args(argv)
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('air: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    try:
        counts = run(lk, x.minutes)
    except lakem.LakeError:
        print('air: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('air: ' + (', '.join((f'{k} {v}' for k, v in sorted(counts.items()))) or 'nothing to do'), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
