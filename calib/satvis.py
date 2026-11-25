from __future__ import annotations
import math
import os
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
VIS_V = 1
R_KM = 10.0
BG_KM = 30.0
BG_PCT = 5.0
BG_MAX = 0.3
CLOUD_MIN = 0.22
CLOUD_OVER = 0.08
WIN = 5
FULL = 0.9
SUN_MIN = 0.3
KM_PX = 0.5

def local_share(cloudy: np.ndarray, win: int=WIN) -> np.ndarray:
    c = np.pad(cloudy.astype(np.float64), win // 2, constant_values=np.nan)
    ok = np.isfinite(c)
    v, k = (np.where(ok, c, 0.0), ok.astype(np.float64))
    S = lambda a: np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sv, sk = (S(v), S(k))
    h, w = cloudy.shape
    box = lambda s: s[win:win + h, win:win + w] - s[:h, win:win + w] - s[win:win + h, :w] + s[:h, :w]
    with np.errstate(invalid='ignore', divide='ignore'):
        return box(sv) / box(sk)

def stats(rc: np.ndarray, dist_km: np.ndarray, sun: float) -> dict:
    out = {'cloud': math.nan, 'cu': math.nan, 'broken': math.nan, 'cv': math.nan, 'n': 0, 'sun': float(sun), 'bg': math.nan}
    near = np.isfinite(rc) & (dist_km <= BG_KM)
    inside = np.isfinite(rc) & (dist_km <= R_KM)
    out['n'] = int(inside.sum())
    if not inside.any() or not sun >= SUN_MIN:
        return out
    bg = float(np.percentile(rc[near], BG_PCT))
    out['bg'] = bg
    if bg > BG_MAX:
        return out
    cloudy = np.where(np.isfinite(rc), rc > max(CLOUD_MIN, bg + CLOUD_OVER), np.nan)
    share = local_share(cloudy)
    cl = (cloudy == 1) & inside
    cu = cl & (share < FULL)
    out['cloud'] = float(cl.sum() / inside.sum())
    out['cu'] = float(cu.sum() / inside.sum())
    if cl.any():
        out['broken'] = float(cu.sum() / cl.sum())
        m = float(rc[cl].mean())
        out['cv'] = float(rc[cl].std() / m) if m > 0 else math.nan
    return out

def scan_time(url: str) -> datetime:
    import re
    m = re.search('_s(\\d{4})(\\d{3})(\\d{2})(\\d{2})(\\d{2})', url)
    return datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1, hours=int(m[3]), minutes=int(m[4]), seconds=int(m[5]))

def fetch(url: str, path: str, tries: int=4) -> None:
    import time
    import satcheck
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=satcheck.UA), timeout=120) as r, open(path, 'wb') as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        return
                    f.write(b)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 + 5 * k)

def at_points(path: str, t_scan: datetime, points: list) -> list:
    import h5py
    import satcheck as C
    out = []
    with h5py.File(path, 'r') as f:
        a, b, H, lon0 = C._proj(f)
        X, Y = (f['x'], f['y'])
        xs = X[:] * C._num(X.attrs['scale_factor']) + C._num(X.attrs['add_offset'])
        ys = Y[:] * C._num(Y.attrs['scale_factor']) + C._num(Y.attrs['add_offset'])
        R = f['Rad']
        sf, ao, fill, k0 = (C._num(R.attrs['scale_factor']), C._num(R.attrs['add_offset']), R.attrs.get('_FillValue'), C._num(f['kappa0'][()]))
        half = int(math.ceil(BG_KM / KM_PX))
        for lat, lon in points:
            px, py = C.latlon_to_xy(np.array([lat]), np.array([lon]), a, b, H, lon0)
            i, j = (int(np.searchsorted(xs, px[0])), int(np.searchsorted(-ys, -py[0])))
            i0, i1, j0, j1 = (max(0, i - half), min(len(xs), i + half + 1), max(0, j - half), min(len(ys), j + half + 1))
            sun = float(C.sun_cos(t_scan, lat, lon))
            if not np.isfinite(px[0]) or i0 >= i1 or j0 >= j1 or (i in (0, len(xs))) or (j in (0, len(ys))):
                out.append(stats(np.full((1, 1), np.nan), np.full((1, 1), np.inf), sun))
                continue
            raw = R[j0:j1, i0:i1]
            rad = raw.astype(np.float32) * sf + ao
            if fill is not None:
                rad[raw == C._num(fill)] = np.nan
            gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
            la, lo = C.xy_to_latlon(gx, gy, a, b, H, lon0)
            d = 111.2 * np.hypot(la - lat, (lo - lon) * math.cos(math.radians(lat)))
            with np.errstate(invalid='ignore', divide='ignore'):
                rc = rad * k0 / max(sun, 0.001)
            out.append(stats(rc, np.where(np.isfinite(d), d, np.inf), sun))
    return out

def day_rows(day: str, points: list, oids: list, url_of=None, counts: dict | None=None) -> list:
    import satcheck
    counts = counts if counts is not None else {}
    url_of = url_of or satcheck.scan_url
    by = {}
    for (la, lo, t), oid in zip(points, oids):
        t = t if isinstance(t, datetime) else datetime.fromisoformat(str(t).replace('Z', '+00:00'))
        by.setdefault(t, []).append((oid, la, lo))
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for t, pts in sorted(by.items()):
            try:
                url = url_of(t)
                if not url:
                    counts['vis_no_scan'] = counts.get('vis_no_scan', 0) + 1
                    continue
                path = os.path.join(tmp, 'scan.nc')
                fetch(url, path)
                ts = scan_time(url)
                got = at_points(path, ts, [(la, lo) for _, la, lo in pts])
            except Exception:
                counts['vis_failed'] = counts.get('vis_failed', 0) + 1
                continue
            counts['vis_hours'] = counts.get('vis_hours', 0) + 1
            for (oid, la, lo), s in zip(pts, got):
                out.append((int(t.timestamp()), oid, la, lo, s, (ts - t).total_seconds()))
    return out

def table_of(rows: list):
    import pairs as P
    n = len(rows)
    f = lambda k: np.array([r[4][k] for r in rows], np.float32)
    return P.table(P.obs_schema('vis'), {'kind': 'vis', 'obs_id': np.array([str(r[1]) for r in rows], dtype=object), 't_valid': np.array([r[0] for r in rows], np.int64), 'lat': np.array([r[2] for r in rows], np.float32), 'lon': np.array([r[3] for r in rows], np.float32), 'review': np.zeros(n, bool), 'o_cloud': f('cloud'), 'o_cu': f('cu'), 'o_broken': f('broken'), 'o_cv': f('cv'), 'q_n': f('n'), 'q_sun': f('sun'), 'q_bg': f('bg'), 'q_dt': np.array([r[5] for r in rows], np.float32)}, n)
