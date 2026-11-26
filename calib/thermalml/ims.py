from __future__ import annotations
import gzip
import math
import os
import urllib.request
from datetime import date, timedelta
import numpy as np
URL = 'https://noaadata.apps.nsidc.org/NOAA/G02156/4km/{y}/ims{y}{doy:03d}_00UTC_4km_v1.3.asc.gz'
N = 6144
CELL_M = 4000.0
ORIGIN = -12288000.0
A, B_ = (6378137.0, 6356752.314245)
E = math.sqrt(1 - (B_ / A) ** 2)
LAT_TS, LON_0 = (60.0, -80.0)
LAND, SNOW = (2, 4)
TIMEOUT_S = 60

def _t(phi):
    s = np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - E * s) / (1 + E * s)) ** (E / 2)

def xy(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    phi, lam = (np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64) - LON_0))
    pc = math.radians(LAT_TS)
    mc = math.cos(pc) / math.sqrt(1 - (E * math.sin(pc)) ** 2)
    rho = A * mc * _t(phi) / _t(pc)
    return (rho * np.sin(lam), -rho * np.cos(lam))

def cells(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    x, y = xy(lat, lon)
    col = np.floor((x - ORIGIN) / CELL_M).astype(np.int64)
    row = np.floor((y - ORIGIN) / CELL_M).astype(np.int64)
    bad = (col < 0) | (col >= N) | (row < 0) | (row >= N) | (np.asarray(lat, np.float64) < 0)
    return (np.where(bad, -1, row), np.where(bad, -1, col))

def parse(blob: bytes) -> np.ndarray | None:
    try:
        text = gzip.decompress(blob)
    except (OSError, EOFError):
        return None
    lines = [ln.strip() for ln in text.splitlines()]
    data = [ln for ln in lines if len(ln) == N and ln.isdigit()]
    if len(data) != N:
        return None
    return (np.frombuffer(b''.join(data), np.uint8) - ord('0')).reshape(N, N)

def _get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return r.read()
    except (OSError, ValueError):
        return None

def chart(day: str, get=_get, cache: str | None=None) -> np.ndarray | None:
    d0 = date.fromisoformat(day)
    for d in (d0, d0 + timedelta(days=1)):
        url = URL.format(y=d.year, doy=d.timetuple().tm_yday)
        path = os.path.join(cache, os.path.basename(url)) if cache else None
        blob = None
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                blob = f.read()
        if blob is None:
            blob = get(url)
            if blob and path:
                os.makedirs(cache, exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(blob)
        g = parse(blob) if blob else None
        if g is not None:
            return g
    return None

def share_at(g: np.ndarray, lat, lon) -> list:
    rows, cols = cells(lat, lon)
    out = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r < 0:
            out.append(None)
            continue
        w = g[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        land = int(((w == LAND) | (w == SNOW)).sum())
        out.append(round(float((w == SNOW).sum()) / land, 3) if land else None)
    return out

def snow(points: dict, day: str, n: dict | None=None, get=_get, cache: str | None=None) -> dict:
    n = n if n is not None else {}
    keys = [k for k, (la, _) in points.items() if la >= 0]
    out = {k: None for k in points}
    if not keys:
        return out
    g = chart(day, get, cache)
    if g is None:
        n['ims_missing'] = n.get('ims_missing', 0) + 1
        return out
    n['ims_days'] = n.get('ims_days', 0) + 1
    got = share_at(g, [points[k][0] for k in keys], [points[k][1] for k in keys])
    out.update(dict(zip(keys, got)))
    return out
