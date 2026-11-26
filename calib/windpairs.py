from __future__ import annotations
import io
import json
import math
from datetime import datetime, timezone
import numpy as np
import linecheck as lc
import track
PREFIX = 'derived/wind/p1/'
VERSION = 1
LOW_TOP = 500.0
LOW_MIN_S = 90.0
MIN_FIX = 5
DRIFT_MAX = 40.0
FIELDS = ('u_sfc', 'v_sfc', 'u_g100', 'v_g100', 'u_g300', 'v_g300', 'u_blavg', 'v_blavg', 'wstar', 'zi', 'nlow', 'blapse', 'dth_m200', 'sun', 'heat', 'cloudlow', 'elevation')
NUM = ('t', 'lat', 'lon', 'u', 'v', 'agl', 'dur', 'n', 'q_res', 'ground', 'relief', 'solar_h', 'month', 'step') + FIELDS

def object_key(lk, r: dict) -> str:
    return PREFIX + lk.name(f"wind|{r['key']}|{r.get('sha256') or ''}")

def drift(fx: list) -> tuple[float, float, float] | None:
    t = np.array([p[0] for p in fx], np.float64)
    lon, lat = (np.array([p[1] for p in fx], np.float64), np.array([p[2] for p in fx], np.float64))
    k = math.cos(math.radians(float(lat.mean())))
    x, y = ((lon - lon.mean()) * 111320.0 * k, (lat - lat.mean()) * 110540.0)
    tt = t - t.mean()
    s = float((tt * tt).sum())
    if s <= 0:
        return None
    u, v = (float((tt * x).sum() / s), float((tt * y).sum() / s))
    res = float(np.sqrt(np.mean((x - u * tt) ** 2 + (y - v * tt) ** 2)))
    return (u, v, res)

def low_segments(f: dict) -> list[list]:
    tr = [p for p in f.get('track') or [] if p[3] is not None and len(p) > 4 and (p[4] is not None)]
    if len(tr) < MIN_FIX:
        return []
    ts = [p[0] for p in tr]
    import bisect
    out = []
    for c in track._climbs_sorted(f):
        if c.get('engon'):
            continue
        a, b = (bisect.bisect_left(ts, c['_t0']), bisect.bisect_right(ts, c['_t1']))
        seg = []
        for p in tr[a:b]:
            if p[3] - p[4] > LOW_TOP:
                break
            seg.append(p)
        if len(seg) >= MIN_FIX and seg[-1][0] - seg[0][0] >= LOW_MIN_S:
            out.append(seg)
    return out

def row(seg: list, B: dict, region: str | None) -> dict | None:
    d = drift(seg)
    if d is None or math.hypot(d[0], d[1]) > DRIFT_MAX:
        return None
    t = (seg[0][0] + seg[-1][0]) / 2
    mid = seg[len(seg) // 2]
    h, ij = (lc.hour(B, t), lc.cell(B, mid[1], mid[2]))
    if h is None or ij is None:
        return None
    F = B['f']
    gr = [float(p[4]) for p in seg]
    tt = datetime.fromtimestamp(t, timezone.utc)
    out = {'t': t, 'lat': mid[2], 'lon': mid[1], 'u': round(d[0], 2), 'v': round(d[1], 2), 'q_res': round(d[2], 1), 'agl': round(float(np.mean([p[3] - p[4] for p in seg])), 1), 'dur': round(seg[-1][0] - seg[0][0]), 'n': len(seg), 'ground': round(float(np.mean(gr)), 1), 'relief': round(max(gr) - min(gr), 1), 'solar_h': ((tt.hour + tt.minute / 60 + mid[1] / 15) % 24 + 24) % 24, 'month': tt.month, '_region': region}
    for n in FIELDS:
        out[n] = float(F[n][h][ij]) if n in F else float('nan')
    return out

def unit_rows(B: dict | None, flights: list) -> list[dict]:
    if B is None:
        return []
    out = []
    for f in flights:
        try:
            segs = low_segments(f)
        except Exception:
            continue
        for s in segs:
            try:
                r = row(s, B, f.get('region'))
            except Exception:
                r = None
            if r is not None:
                out.append(r)
    return out

def columns(rs: list[dict], model: str, step: float) -> dict:
    n = len(rs)
    cols = {k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in rs], np.float64) for k in NUM if k != 'step'}
    cols['step'] = np.full(n, float(step))
    cols['region'] = np.array([str(r.get('_region') or '') for r in rs])
    cols['model'] = np.full(n, model)
    return cols

def pack(cols: dict) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, meta=np.array(json.dumps({'v': VERSION})), **cols)
    return bio.getvalue()

def load_all(lk, counts=None) -> dict:
    cols = {}
    for k in lk.list(PREFIX):
        with np.load(io.BytesIO(lk.get(k)), allow_pickle=False) as z:
            c = {name: z[name] for name in z.files if name != 'meta'}
        if counts is not None:
            counts['units'] = counts.get('units', 0) + 1
        n = len(c.get('u', []))
        if not n:
            continue
        for name in NUM:
            c.setdefault(name, np.full(n, np.nan))
        for name, v in c.items():
            cols.setdefault(name, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items() if len(v) == len(cols.get('u', []))} if cols else {}
