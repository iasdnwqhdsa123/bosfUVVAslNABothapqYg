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
import time
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lake as lakem
import satcheck as sc
PREFIX = 'derived/streets/p1/'
VERSION = 1
BOXES = {'ohin': ([-86.0, 39.0, -82.5, 41.5], 'America/New_York'), 'pany': ([-80.0, 40.5, -76.5, 43.0], 'America/New_York'), 'iljowa': ([-92.0, 40.5, -88.5, 43.0], 'America/Chicago'), 'ksne': ([-100.0, 38.0, -96.5, 40.5], 'America/Chicago'), 'txpl': ([-102.5, 33.0, -99.0, 35.5], 'America/Chicago'), 'segc': ([-84.0, 33.0, -80.5, 35.5], 'America/New_York'), 'mich': ([-86.0, 42.0, -82.5, 44.5], 'America/Detroit'), 'neng': ([-73.5, 41.5, -70.0, 44.0], 'America/New_York'), 'copl': ([-104.8, 38.5, -101.3, 41.0], 'America/Denver'), 'mnwi': ([-95.0, 43.5, -91.5, 46.0], 'America/Chicago'), 'ncpd': ([-80.5, 34.5, -77.0, 37.0], 'America/New_York'), 'wpa': ([-81.5, 39.3, -78.0, 41.8], 'America/New_York'), 'cpa': ([-78.5, 39.5, -75.0, 42.0], 'America/New_York'), 'ohio': ([-84.5, 39.5, -81.0, 42.0], 'America/New_York')}
SCAN_SOLAR = (12.5, 14.0)
MIN_CU_BLOCKS = 20
DAYS_BACK = 6
BLOCK_KM = (24, 48)
EX_FIELDS = ('elevation', 'u_cuw', 'v_cuw', 'thml', 'cudepth', 'zi', 'n_g750', 'n_g1250', 'n_g1750', 'n_g2500', 'n_g3500', 'u_g1000', 'v_g1000', 'u_g1500', 'v_g1500', 'u_g2000', 'v_g2000', 'u_g2500', 'v_g2500', 'u_g3000', 'v_g3000', 'dth_m500', 'dth_m200', 'cucover', 'snowd')

def object_key(lk, day: str, box: str) -> str:
    return PREFIX + lk.name(f'streets|{day}|{box}')

def scan_times(day: str, bbox) -> list[datetime]:
    d = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    lon = (bbox[0] + bbox[2]) / 2
    return [d + timedelta(hours=h - lon / 15) for h in SCAN_SOLAR]

def screen_mask(px: dict, t: datetime):
    from scipy import ndimage
    cs = sc.sun_cos(t, px['lat'], px['lon'])
    R = px['refl'] / np.maximum(cs, 0.05)
    ok = np.isfinite(R) & (cs >= math.sin(math.radians(sc.SUN_MIN - 5)))
    Rs = np.where(ok, R, np.nan)[::4, ::4]
    g = ndimage.percentile_filter(np.nan_to_num(Rs, nan=0.3), 8, size=9)
    g = ndimage.zoom(g, (R.shape[0] / g.shape[0], R.shape[1] / g.shape[1]), order=1)[:R.shape[0], :R.shape[1]]
    if g.shape != R.shape:
        g = np.pad(g, ((0, R.shape[0] - g.shape[0]), (0, R.shape[1] - g.shape[1])), mode='edge')
    return (ok & (R - g > sc.RISE) & (R > sc.BRIGHT), ok)

def block_centres(bbox, km: float):
    dlat = km / 111.32
    la = bbox[1] + 0.3
    while la <= bbox[3] - 0.3:
        dlon = km / (111.32 * math.cos(math.radians(la)))
        lo = bbox[0] + 0.3
        while lo <= bbox[2] - 0.3:
            yield (la, lo, dlat, dlon)
            lo += dlon
        la += dlat

def cu_blocks(cloud, ok, lat, lon, bbox, km: float=24) -> int:
    n = 0
    for la, lo, dlat, dlon in block_centres(bbox, km):
        near = (np.abs(lat - la) < dlat * 0.6) & (np.abs(lon - lo) < dlon * 0.6)
        if near.any():
            s = sc.street_stats(sc.regular_patch(cloud[near], ok[near], lat[near], lon[near], la, lo, half_km=km / 2))
            n += bool(s and 'aniso' in s)
    return n

def screen(day: str, box: str, read=None) -> dict:
    bbox = BOXES[box][0]
    read = read or sc.read_box
    ns = []
    for t in scan_times(day, bbox):
        url = sc.scan_url(t)
        if not url:
            ns.append(None)
            continue
        try:
            px = read(url, bbox)
        except Exception:
            ns.append(None)
            continue
        cloud, ok = screen_mask(px, t)
        ns.append(cu_blocks(cloud, ok, px['lat'], px['lon'], bbox))
    return {'go': all((n is not None and n >= MIN_CU_BLOCKS for n in ns)), 'n': ns}

def read_bt(url: str, bbox) -> dict:
    import h5py
    with h5py.File(sc.RangeFile(url), 'r') as f:
        a, b, H, lon0 = sc._proj(f)
        X, Y = (f['x'], f['y'])
        xs = X[:] * sc._num(X.attrs['scale_factor']) + sc._num(X.attrs['add_offset'])
        ys = Y[:] * sc._num(Y.attrs['scale_factor']) + sc._num(Y.attrs['add_offset'])
        w, s, e, n = bbox
        lats = np.array([s, s, n, n, (s + n) / 2, (s + n) / 2, s, n])
        lons = np.array([w, e, w, e, w, e, (w + e) / 2, (w + e) / 2])
        px, py = sc.latlon_to_xy(lats, lons, a, b, H, lon0)
        i0, i1 = (max(0, np.searchsorted(xs, px.min()) - 2), np.searchsorted(xs, px.max()) + 2)
        j0, j1 = (max(0, np.searchsorted(-ys, -py.max()) - 2), np.searchsorted(-ys, -py.min()) + 2)
        R = f['Rad']
        raw = R[j0:j1, i0:i1].astype(np.float32)
        rad = raw * sc._num(R.attrs['scale_factor']) + sc._num(R.attrs['add_offset'])
        fill = R.attrs.get('_FillValue')
        if fill is not None:
            rad[raw == sc._num(fill)] = np.nan
        fk1, fk2, bc1, bc2 = (sc._num(f[k][()]) for k in ('planck_fk1', 'planck_fk2', 'planck_bc1', 'planck_bc2'))
        with np.errstate(all='ignore'):
            bt = (fk2 / np.log(fk1 / rad + 1) - bc1) / bc2
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
        lat, lon = sc.xy_to_latlon(gx, gy, a, b, H, lon0)
    return {'bt': bt.astype(np.float32), 'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)}

def scan_url_band(t: datetime, ch: str):
    old = sc.CHANNEL
    sc.CHANNEL = ch
    try:
        return sc.scan_url(t)
    finally:
        sc.CHANNEL = old

def scan_time(url: str) -> datetime:
    s = url.split('_s')[1][:13]
    return datetime(int(s[:4]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(s[4:7]) - 1, hours=int(s[7:9]), minutes=int(s[9:11]), seconds=int(s[11:13]))

def motion(A, B, ex, ey, exp_px, px):
    n = A.shape[0]
    m = (B.shape[0] - n) // 2
    best, rb = (None, -2.0)
    cross = int(round(3 / px))

    def ncc(dx, dy):
        if abs(dx) > m or abs(dy) > m:
            return None
        Bs = B[m + dy:m + dy + n, m + dx:m + dx + n]
        ok = np.isfinite(A) & np.isfinite(Bs)
        if ok.mean() < 0.6:
            return None
        a, b = (A[ok] - A[ok].mean(), Bs[ok] - Bs[ok].mean())
        d = math.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else None
    r0 = ncc(0, 0)
    for s in np.arange(-0.5 * exp_px, 2 * exp_px + 1e-09, 1.0):
        for c in range(-cross, cross + 1):
            r = ncc(int(round(s * ex - c * ey)), int(round(s * ey + c * ex)))
            if r is not None and r > rb:
                rb, best = (r, s)
    return (best, rb, r0)

def roll_cells(work: str) -> tuple[list, list, np.ndarray] | None:
    out = os.path.join(work, 'rollcells.bin')
    r = subprocess.run(['node', sc.node_script(), 'rollcells', work, out], capture_output=True, text=True, timeout=1800)
    if r.returncode or not os.path.exists(out):
        return None
    blob = open(out, 'rb').read()
    nl = blob.index(b'\n')
    head = json.loads(blob[:nl])
    a = np.frombuffer(blob[nl + 1:], np.float32).reshape(len(head['stamps']), len(head['planes']), head['ny'], head['nx'])
    return (head['stamps'], head['planes'], a)

def block_window(ex, name, j, i, rj, ri):
    v = ex.get(name)
    return None if v is None else v[max(0, j - rj):j + rj + 1, max(0, i - ri):i + ri + 1]

def table(day: str, box: str, work: str, log=print) -> dict | None:
    bbox, tz = BOXES[box]
    bdir = os.path.join(work, f'{box}_{day}')
    gdir = os.path.join(bdir, 'hrrr')
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(bdir, exist_ok=True)
    try:
        subprocess.run([sys.executable, os.path.join(HERE, 'build_day.py'), '--day', day, '--tz', tz, f"--bbox={','.join((f'{v:.2f}' for v in bbox))}", '--out', bdir, '--workers', '2'], cwd=os.path.join(HERE, '..', 'pipeline'), capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        pass
    if not os.path.exists(os.path.join(gdir, 'latest.json')):
        log('  a build failed')
        return None
    try:
        return _table(day, box, bbox, gdir, os.path.join(bdir, '_work'), log)
    finally:
        shutil.rmtree(bdir, ignore_errors=True)

def _table(day, box, bbox, gdir, work, log):
    json.dump({'bbox': bbox, 'tz': BOXES[box][1], 'model': 'hrrr', 'day': day, 'region': box}, open(os.path.join(gdir, 'meta.json'), 'w'))
    m = json.load(open(os.path.join(gdir, 'latest.json')))
    g = m['grid']
    west, north, step, nx, ny = (g['west'], g['north'], g['step'], g['nx'], g['ny'])
    hours = []
    cla, clo = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    for f in sorted(glob.glob(os.path.join(gdir, '_hours', '*.npz'))):
        st = os.path.basename(f)[:-4]
        t = datetime.strptime(st, '%Y%m%d%H').replace(tzinfo=timezone.utc)
        if sc.sun_cos(t, cla, clo) < math.sin(math.radians(sc.SUN_MIN - 3)):
            continue
        fc = sc.forecast_hour(m, f)
        u1 = sc.scan_url(t) if fc is not None else None
        if not u1:
            continue
        try:
            px = sc.read_box(u1, bbox)
        except Exception:
            continue
        u2 = sc.scan_url(scan_time(u1) + timedelta(minutes=20))
        px2 = None
        if u2:
            try:
                px2 = sc.read_box(u2, bbox)
                if px2['refl'].shape != px['refl'].shape:
                    px2 = None
            except Exception:
                px2 = None
        u3 = scan_url_band(t, 'C13')
        b13 = None
        if u3:
            try:
                b13 = read_bt(u3, bbox)
            except Exception:
                b13 = None
        with np.load(f) as z:
            ex = {k: sc.decode(m, z, k) for k in EX_FIELDS}
        hours.append(dict(t=t, st=st, fc=fc, px=px, px2=px2, dt2=(scan_time(u2) - scan_time(u1)).total_seconds() if px2 is not None else None, b13=b13, ex=ex))
    if len(hours) < 4:
        log(f'  {len(hours)} lit hours')
        return None
    shp = hours[0]['px']['refl'].shape
    hours = [h for h in hours if h['px']['refl'].shape == shp]
    coss = [sc.sun_cos(h['t'], h['px']['lat'], h['px']['lon']) for h in hours]
    masks = sc.cloud_masks([h['px']['refl'] for h in hours], coss)
    Rall = np.stack([h['px']['refl'] / np.maximum(c, 0.05) for h, c in zip(hours, coss)])
    with np.errstate(all='ignore'):
        lit = np.stack([c >= math.sin(math.radians(sc.SUN_MIN)) for c in coss])
        ground = np.nanpercentile(np.where(lit, Rall, np.nan), sc.SKY_PCT, axis=0)
    lat, lon = (hours[0]['px']['lat'], hours[0]['px']['lon'])
    sc.write_env(gdir, m, [(h['t'], h['fc'], h['px']) for h in hours], work)
    rc = roll_cells(work)
    if rc is None:
        log('  the roll model failed')
        return None
    stamps, PL, CA = rc
    sidx = {s: k for k, s in enumerate(stamps)}
    from zoneinfo import ZoneInfo
    tzi = ZoneInfo(BOXES[box][1])
    rows = []
    for h, (cloud, ok) in zip(hours, masks):
        k = sidx.get(h['st'])
        if k is None:
            continue
        fc, ex = (h['fc'], h['ex'])
        lt = h['t'].astimezone(tzi)
        lh = lt.hour + lt.minute / 60
        D1 = h['px']['refl'] / np.maximum(sc.sun_cos(h['t'], lat, lon), 0.05) - ground
        D2 = None
        if h['px2'] is not None:
            D2 = h['px2']['refl'] / np.maximum(sc.sun_cos(h['t'] + timedelta(seconds=h['dt2']), lat, lon), 0.05) - ground
        for km in BLOCK_KM:
            half = km / 2
            for la, lo_, dlat, dlon in block_centres(bbox, km):
                j, i = (int(round((north - la) / step)), int(round((lo_ - west) / step)))
                if not (0 <= j < ny and 0 <= i < nx and fc['clear_above'][j, i] and np.isfinite(fc['u'][j, i])):
                    continue
                near = (np.abs(lat - la) < dlat * 0.6) & (np.abs(lon - lo_) < dlon * 0.6)
                if not near.any():
                    continue
                gp = sc.regular_patch(cloud[near], ok[near], lat[near], lon[near], la, lo_, half_km=half)
                s = sc.street_stats(gp)
                if not (s and 'aniso' in s):
                    continue
                u, v = (float(fc['u'][j, i]), float(fc['v'][j, i]))
                rec = dict(km=km, lh=lh, la=la, lo=lo_, aniso=s['aniso'], axis=s['axis'], spacing=s['spacing'] or np.nan, cover=s['cover'], wind=math.hypot(u, v), wax=sc.wind_axis(u, v), zi_fc=float(fc['zi'][j, i]))
                for p, name in enumerate(PL):
                    rec['c_' + name] = float(CA[k, p, j, i])
                rj = max(1, int(round(half / 111.32 / step)))
                ri = max(1, int(round(half / (111.32 * math.cos(math.radians(la))) / step)))
                el = block_window(ex, 'elevation', j, i, rj, ri)
                rec['relief'] = float(np.nanmax(el) - np.nanmin(el)) if el is not None and np.isfinite(el).any() else np.nan
                sn = block_window(ex, 'snowd', j, i, rj, ri)
                rec['snow'] = float(np.nanmean(sn)) if sn is not None and np.isfinite(sn).any() else np.nan
                zi = ex['zi'][j, i] if ex['zi'] is not None else np.nan
                Ns = [ex[f'n_g{z}'][j, i] for z in (750, 1250, 1750, 2500, 3500) if ex[f'n_g{z}'] is not None and np.isfinite(zi) and (zi < z <= zi + 2000)]
                rec['Nabove'] = float(np.nanmax(Ns)) if Ns and np.isfinite(Ns).any() else np.nan
                W13 = [math.hypot(ex[f'u_g{z}'][j, i], ex[f'v_g{z}'][j, i]) for z in (1000, 1500, 2000, 2500, 3000) if ex[f'u_g{z}'] is not None]
                with np.errstate(all='ignore'):
                    rec['w13'] = float(np.nanmax(W13)) if W13 and np.isfinite(W13).any() else np.nan
                for q in ('dth_m500', 'dth_m200', 'cucover'):
                    rec['x_' + q] = float(ex[q][j, i]) if ex[q] is not None else np.nan
                th = ex['thml'][j, i] if ex['thml'] is not None else np.nan
                cud = ex['cudepth'][j, i] if ex['cudepth'] is not None else np.nan
                zs = ex['elevation'][j, i] if ex['elevation'] is not None else np.nan
                rec['cutopT'] = float(th - 0.0098 * (zs + zi) - 0.0065 * np.nan_to_num(cud)) if np.isfinite(th) and np.isfinite(zi) else np.nan
                rec['bt10'] = _bt10(h['b13'], gp, cloud[near], ok[near], lat[near], lon[near], la, lo_, dlat, dlon, half)
                rec['mv_shift'] = rec['mv_exp'] = rec['mv_r'] = rec['mv_r0'] = np.nan
                d_ax = sc.axis_diff(s['axis'], rec['wax'])
                if D2 is not None and s['aniso'] >= 0.25 and (d_ax <= 30):
                    _motion(rec, ex, j, i, u, v, h['dt2'], D1, D2, lat, lon, la, lo_, half)
                rows.append(rec)
    if not rows:
        return {}
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in sorted(rows[0])}

def _bt10(b13, gp, cloud, ok, lat, lon, la, lo_, dlat, dlon, half):
    if b13 is None:
        return np.nan
    nb = (np.abs(b13['lat'] - la) < dlat * 0.6) & (np.abs(b13['lon'] - lo_) < dlon * 0.6)
    if not nb.any():
        return np.nan
    bt = sc.regular_patch(b13['bt'][nb], np.isfinite(b13['bt'][nb]), b13['lat'][nb], b13['lon'][nb], la, lo_, half_km=half, px=2.0)
    n2 = bt.shape[0]
    f = gp.shape[0] / n2
    cs = np.full((n2, n2), np.nan)
    for a in range(n2):
        for b in range(n2):
            w = gp[int(a * f):int((a + 1) * f), int(b * f):int((b + 1) * f)]
            if np.isfinite(w).any():
                cs[a, b] = np.nanmean(w)
    sel = (cs >= 0.5) & np.isfinite(bt)
    return float(np.percentile(bt[sel], 10)) if sel.sum() >= 3 else np.nan

def _motion(rec, ex, j, i, u, v, dt2, D1, D2, lat, lon, la, lo_, half):
    uc = ex['u_cuw'][j, i] if ex['u_cuw'] is not None else np.nan
    vc = ex['v_cuw'][j, i] if ex['v_cuw'] is not None else np.nan
    if not (np.isfinite(uc) and np.isfinite(vc)):
        uc, vc = (u, v)
    Vc = math.hypot(uc, vc)
    if Vc <= 1:
        return
    exp_px = Vc * dt2 / 1000 / sc.PIX_KM
    mpx = int(math.ceil(2 * exp_px + 3 / sc.PIX_KM)) + 2
    bigh = half + mpx * sc.PIX_KM
    nr = (np.abs(lat - la) < bigh / 111.32 * 1.1) & (np.abs(lon - lo_) < bigh / (111.32 * math.cos(math.radians(la))) * 1.1)
    A = sc.regular_patch(D1[nr], np.isfinite(D1[nr]), lat[nr], lon[nr], la, lo_, half_km=half)
    Bp = sc.regular_patch(D2[nr], np.isfinite(D2[nr]), lat[nr], lon[nr], la, lo_, half_km=bigh)
    nA = A.shape[0]
    mm = (Bp.shape[0] - nA) // 2
    Bp = Bp[:nA + 2 * mm, :nA + 2 * mm]
    bs, rb, r0 = motion(A, Bp, uc / Vc, -vc / Vc, exp_px, sc.PIX_KM)
    rec['mv_shift'], rec['mv_exp'], rec['mv_r'], rec['mv_r0'] = (np.nan if bs is None else bs, exp_px, rb, np.nan if r0 is None else r0)

def pack(day: str, box: str, cols: dict, scr: dict, kind: str='') -> bytes:
    bio = io.BytesIO()
    meta = json.dumps({'v': VERSION, 'day': day, 'box': box, 'kind': kind, 'screen': scr.get('n')})
    np.savez_compressed(bio, meta=np.array(meta), **cols)
    return bio.getvalue()

def unpack(blob: bytes) -> tuple[dict, dict]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        return (meta, {k: z[k] for k in z.files if k != 'meta'})

def load_all(lk, counts=None) -> dict:
    cols = {}
    for k in lk.list(PREFIX):
        meta, c = unpack(lk.get(k))
        n = len(c.get('aniso', []))
        if counts is not None:
            counts['box_days'] = counts.get('box_days', 0) + 1
        if not n:
            continue
        c['day'] = np.full(n, meta['day'])
        c['box'] = np.full(n, meta['box'])
        c['kind'] = np.full(n, meta.get('kind') or '')
        for name, v in c.items():
            cols.setdefault(name, []).append(v)
    if not cols:
        return {}
    return {k: np.concatenate(v) for k, v in cols.items() if len(v) == len(cols['aniso'])}

def due_days(today: date, span: str | None=None) -> list[str]:
    if span:
        a, _, b = span.partition(':')
        d0, d1 = (date.fromisoformat(a), date.fromisoformat(b or a))
        if d1 < d0:
            d0, d1 = (d1, d0)
        return [(d1 - timedelta(days=k)).isoformat() for k in range((d1 - d0).days + 1)]
    return [(today - timedelta(days=k)).isoformat() for k in range(1, DAYS_BACK + 2)]

def run(lk, today: date, span: str | None=None, curated: list | None=None, minutes: float=55, work: str='/tmp/streets', screen_fn=None, table_fn=None, clock=time.monotonic, log=print) -> dict:
    screen_fn = screen_fn or screen
    table_fn = table_fn or (lambda d, b: table(d, b, work, log))
    t_end = clock() + minutes * 60
    counts = {}
    jobs = [(b, d, k) for b, d, k in curated] if curated else [(b, d, '') for d in due_days(today, span) for b in BOXES]
    for box, day, kind in jobs:
        if clock() > t_end:
            counts['left'] = counts.get('left', 0) + 1
            continue
        key = object_key(lk, day, box)
        if lk.exists(key):
            counts['done_before'] = counts.get('done_before', 0) + 1
            continue
        scr = {'go': True, 'n': None} if curated else screen_fn(day, box)
        cols = {}
        if scr['go']:
            cols = table_fn(day, box)
            if cols is None:
                counts['failed'] = counts.get('failed', 0) + 1
                continue
            counts['tables'] = counts.get('tables', 0) + 1
            counts['blocks'] = counts.get('blocks', 0) + len(cols.get('aniso', []))
        else:
            counts['screened_out'] = counts.get('screened_out', 0) + 1
        lk.put(key, pack(day, box, cols, scr, kind), strict=False)
    return counts

def curated_days(path: str | None=None) -> list:
    with open(path or os.path.join(HERE, 'streetdays.json'), encoding='utf-8') as f:
        d = json.load(f)
    return [(b, day, kind) for kind, boxes in d.items() for b, days in boxes.items() for day in days]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The street pairs: the satellite's lines against the site's rolls")
    ap.add_argument('--days', help='YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD (default: yesterday and the six days before)')
    ap.add_argument('--curated', action='store_true', help="the eye checks' box-days (streetdays.json), not screened")
    ap.add_argument('--minutes', type=float, default=55)
    ap.add_argument('--day', help='today, YYYY-MM-DD (default: today, UTC)')
    ap.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    x = ap.parse_args(argv)
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('streets: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
    try:
        counts = run(lk, today, x.days, curated_days() if x.curated else None, x.minutes, log=lambda s: print('streets:' + s, flush=True))
    except lakem.LakeError:
        print('streets: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('streets: ' + (', '.join((f'{k} {v}' for k, v in sorted(counts.items()))) or 'nothing to do'), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
