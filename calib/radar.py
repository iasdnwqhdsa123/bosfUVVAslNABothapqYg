from __future__ import annotations
import gzip
import json
import math
import re
import urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
UTC = timezone.utc
BUCKET = 'https://noaa-mrms-pds.s3.amazonaws.com'
PRODUCT = 'CONUS/MergedReflectivityQCComposite_00.50'
RADAR_HOURS = (15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2)
R_KM = 10.0
STORM_DBZ = 45.0
ECHO_DBZ = 20.0
MAX_DT_S = 6 * 60
RADAR_V = 1
_NAME = re.compile('MRMS_MergedReflectivityQCComposite_00\\.50_(\\d{8})-(\\d{6})\\.grib2\\.gz$')

def _get(url: str, opener=None, tries: int=4) -> bytes:
    import time
    opener = opener or urllib.request.urlopen
    for i in range(tries):
        try:
            with opener(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError('radar: unreachable')

def day_keys(day: str, opener=None) -> dict:
    d = day.replace('-', '')
    out, token = ({}, None)
    while True:
        url = f'{BUCKET}/?list-type=2&prefix={PRODUCT}/{d}/&max-keys=1000' + (f'&continuation-token={urllib.request.quote(token)}' if token else '')
        text = _get(url, opener).decode('utf-8', 'replace')
        for k in re.findall('<Key>([^<]+)</Key>', text):
            m = _NAME.search(k)
            if m:
                out[datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').replace(tzinfo=UTC)] = k
        t = re.search('<NextContinuationToken>([^<]+)</NextContinuationToken>', text)
        if not t:
            return out
        token = t.group(1)

def nearest(keys: dict, t: datetime) -> tuple[str, float] | None:
    if not keys:
        return None
    best = min(keys, key=lambda k: abs((k - t).total_seconds()))
    dt = (best - t).total_seconds()
    return (keys[best], dt) if abs(dt) <= MAX_DT_S else None

def decode(blob: bytes) -> tuple[np.ndarray, dict]:
    import eccodes
    h = eccodes.codes_new_from_message(gzip.decompress(blob))
    try:
        nx, ny = (int(eccodes.codes_get(h, 'Ni')), int(eccodes.codes_get(h, 'Nj')))
        north = float(eccodes.codes_get(h, 'latitudeOfFirstGridPointInDegrees'))
        west = float(eccodes.codes_get(h, 'longitudeOfFirstGridPointInDegrees'))
        step = float(eccodes.codes_get(h, 'iDirectionIncrementInDegrees'))
        south_first = int(eccodes.codes_get(h, 'jScansPositively')) == 1
        v = np.asarray(eccodes.codes_get_values(h), np.float32).reshape(ny, nx)
    finally:
        eccodes.codes_release(h)
    if south_first:
        v = v[::-1]
        north = north + (ny - 1) * step
    west = west - 360.0 if west > 180 else west
    return (clean(v), {'north': north, 'west': west, 'step': step, 'ny': ny, 'nx': nx})

def clean(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return np.where(v <= -900, np.nan, np.maximum(v, 0.0)).astype(np.float32)

def at_points(dbz: np.ndarray, grid: dict, lat, lon, r_km: float=R_KM) -> dict:
    lat, lon = (np.asarray(lat, np.float64), np.asarray(lon, np.float64))
    step = grid['step']
    ry = int(math.ceil(r_km / (111.0 * step)))
    out_max, out_cov = (np.full(len(lat), np.nan), np.zeros(len(lat)))
    for k, (la, lo) in enumerate(zip(lat, lon)):
        y = int(round((grid['north'] - la) / step))
        x = int(round((lo - grid['west']) / step))
        rx = int(math.ceil(r_km / (111.0 * step * max(math.cos(math.radians(la)), 0.1))))
        y0, y1, x0, x1 = (max(0, y - ry), min(grid['ny'], y + ry + 1), max(0, x - rx), min(grid['nx'], x + rx + 1))
        if y0 >= y1 or x0 >= x1:
            continue
        b = dbz[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dy = (yy - y) * step * 111.0
        dx = (xx - x) * step * 111.0 * math.cos(math.radians(la))
        inside = dy * dy + dx * dx <= r_km * r_km
        cov = inside & np.isfinite(b)
        if inside.any():
            out_cov[k] = cov.sum() / inside.sum()
        if cov.any():
            out_max[k] = float(np.max(b[cov]))
    return {'dbz_max': out_max, 'cover': out_cov}

def truth(dbz_max: np.ndarray, cover: np.ndarray, min_cover: float=0.5) -> dict:
    ok = np.isfinite(dbz_max) & (np.asarray(cover) >= min_cover)
    return {'storm': np.where(ok, (dbz_max >= STORM_DBZ).astype(float), np.nan), 'echo': np.where(ok, (dbz_max >= ECHO_DBZ).astype(float), np.nan)}

def day_rows(day: str, stations: list, box, fetch=None, keys=None, counts: dict | None=None) -> list:
    counts = counts if counts is not None else {}
    w, s, e, n = box
    sts = [x for x in stations if s <= x[1] <= n and w <= x[2] <= e]
    if not sts:
        return []
    fetch = fetch or (lambda k: _get(f'{BUCKET}/{k}'))
    d0 = datetime.fromisoformat(day).replace(tzinfo=UTC)
    if keys is None:
        keys = {**day_keys(day), **day_keys((d0 + timedelta(days=1)).date().isoformat())}
    lat = np.array([x[1] for x in sts], np.float64)
    lon = np.array([x[2] for x in sts], np.float64)
    out = []
    for h in RADAR_HOURS:
        t = d0 + timedelta(hours=h) + (timedelta(days=1) if h < 12 else timedelta(0))
        got = nearest(keys, t)
        if got is None:
            counts['radar_no_file'] = counts.get('radar_no_file', 0) + 1
            continue
        key, dt = got
        try:
            dbz, grid = decode(fetch(key))
        except Exception:
            counts['radar_unreadable'] = counts.get('radar_unreadable', 0) + 1
            continue
        p = at_points(dbz, grid, lat, lon)
        tr = truth(p['dbz_max'], p['cover'])
        out.append((int(t.timestamp()), {'obs_id': [str(x[0]) for x in sts], 'lat': lat, 'lon': lon, 'elev': np.array([float(x[3]) for x in sts]), 'dbz_max': p['dbz_max'], 'storm': tr['storm'], 'echo': tr['echo'], 'cover': p['cover'], 'dt': np.full(len(sts), dt)}))
        counts['radar_hours'] = counts.get('radar_hours', 0) + 1
    return out

def input_id(day: str) -> str:
    return f'radar|v{RADAR_V}|{day}|' + json.dumps([R_KM, STORM_DBZ, ECHO_DBZ, list(RADAR_HOURS)], separators=(',', ':'))
