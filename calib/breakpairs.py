from __future__ import annotations
import io
import json
import math
from datetime import datetime, timezone
import numpy as np
import linecheck as lc
import track
PREFIX = 'derived/break/p1/'
VERSION = 1
OUTS = ('climb', 'save', 'weak', 'home', 'away', 'engine', 'unknown')
FIELDS = ('wstar', 'u_blavg', 'v_blavg', 'zi', 'cubase')
NUM = ('t', 'lat', 'lon', 'search', 'out', 'dur', 'loss', 'agl', 'solar_h', 'month', 'step') + FIELDS

def object_key(lk, r: dict) -> str:
    return PREFIX + lk.name(f"break|{r['key']}|{r.get('sha256') or ''}")

def row(g: dict, B: dict, region: str | None) -> dict | None:
    lon, lat, t = (g.get('lon'), g.get('lat'), g.get('t'))
    if lon is None or lat is None or t is None:
        return None
    h, ij = (lc.hour(B, t), lc.cell(B, lon, lat))
    if h is None or ij is None:
        return None
    F = B['f']
    tt = datetime.fromtimestamp(t, timezone.utc)
    out = {'t': float(t), 'lat': float(lat), 'lon': float(lon), 'search': 1.0 if g.get('search') else 0.0, 'out': float(OUTS.index(g['out'])) if g.get('out') in OUTS else float('nan'), 'dur': g.get('dur'), 'loss': g.get('loss'), 'agl': g.get('agl'), 'solar_h': ((tt.hour + tt.minute / 60 + lon / 15) % 24 + 24) % 24, 'month': tt.month, '_region': region}
    for n in FIELDS:
        out[n] = float(F[n][h][ij]) if n in F else float('nan')
    return out

def unit_rows(B: dict | None, flights: list) -> list[dict]:
    if B is None:
        return []
    out = []
    for f in flights:
        try:
            gs = track.glides(f)
        except Exception:
            continue
        for g in gs:
            try:
                r = row(g, B, f.get('region'))
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
        n = len(c.get('search', []))
        if not n:
            continue
        for name in NUM:
            c.setdefault(name, np.full(n, np.nan))
        for name, v in c.items():
            cols.setdefault(name, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items() if len(v) == len(cols.get('search', []))} if cols else {}

def bs_rasp(ws, u, v) -> np.ndarray:
    ws = np.asarray(ws, np.float64)
    sp = np.maximum(np.hypot(np.asarray(u, np.float64), np.asarray(v, np.float64)), 0.5)
    return np.where(ws > 0, np.minimum(20.0, ws / (math.sqrt(0.003) * sp)), 0.0)
