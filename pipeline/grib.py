from __future__ import annotations
import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
UA = {'User-Agent': 'wx-pipeline/1.0'}

def http_get(url, rng: tuple[int, int | None] | None=None, tries: int=6, timeout: float=90) -> bytes:
    urls = [url] if isinstance(url, str) else list(url)
    headers = dict(UA)
    if rng:
        headers['Range'] = f"bytes={rng[0]}-{('' if rng[1] is None else rng[1])}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(urls[attempt % len(urls)], headers=headers), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 409) or attempt == tries - 1:
                raise
            time.sleep((4 if e.code == 503 else 1) * 2 ** attempt + random.random())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError('unreachable')

def exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, headers=UA, method='HEAD')
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        return False

def parse_noaa_idx(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        p = line.split(':')
        if len(p) < 6:
            continue
        rows.append({'offset': int(p[1]), 'var': p[3], 'level': p[4], 'fcst': p[5]})
    for i, r in enumerate(rows):
        r['end'] = rows[i + 1]['offset'] - 1 if i + 1 < len(rows) else None
    return rows

def parse_ecmwf_idx(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        lev, kind = (j.get('levelist'), j.get('levtype', 'sfc'))
        rows.append({'offset': j['_offset'], 'end': j['_offset'] + j['_length'] - 1, 'var': j['param'], 'level': (f'{lev} mb' if kind == 'pl' else f'{kind} {lev}') if lev else kind})
    return rows

def select(rows: list[dict], wanted: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    out = {}
    for var, level in wanted:
        rx = re.compile(level)
        for r in rows:
            if r['var'] == var and (rx.fullmatch(r['level']) or ('fcst' in r and rx.fullmatch(f"{r['level']} {r['fcst']}"))):
                out[var, level] = r
                break
    return out

def _merge_ranges(rows: list[dict], gap: int=64 * 1024) -> list[tuple[int, int | None, list[dict]]]:
    rows = sorted(rows, key=lambda r: r['offset'])
    groups: list[list] = []
    for r in rows:
        if groups and groups[-1][1] is not None and (r['offset'] - groups[-1][1] <= gap):
            groups[-1][1] = r['end']
            groups[-1][2].append(r)
        else:
            groups.append([r['offset'], r['end'], [r]])
    return [(a, b, c) for a, b, c in groups]

def decode(msg: bytes):
    import eccodes
    import numpy as np
    gid = eccodes.codes_new_from_message(msg)
    try:
        ni = eccodes.codes_get(gid, 'Nx') if eccodes.codes_is_defined(gid, 'Nx') else eccodes.codes_get(gid, 'Ni')
        nj = eccodes.codes_get(gid, 'Ny') if eccodes.codes_is_defined(gid, 'Ny') else eccodes.codes_get(gid, 'Nj')
        vals = eccodes.codes_get_values(gid).astype(np.float32)
        miss = eccodes.codes_get(gid, 'missingValue')
        vals[vals == miss] = np.nan
        meta = {'grid': eccodes.codes_get(gid, 'gridType'), 'ni': ni, 'nj': nj}
        return (vals.reshape(nj, ni), meta)
    finally:
        eccodes.codes_release(gid)

def earth_rotation(msg: bytes):
    import math
    import eccodes
    gid = eccodes.codes_new_from_message(msg)
    try:
        g = lambda k: eccodes.codes_get(gid, k) if eccodes.codes_is_defined(gid, k) else None
        if g('gridType') != 'lambert' or not g('uvRelativeToGrid'):
            return None
        p1, p2 = (math.radians(g('Latin1InDegrees')), math.radians(g('Latin2InDegrees')))
        n = math.sin(p1) if abs(p1 - p2) < 1e-09 else math.log(math.cos(p1) / math.cos(p2)) / math.log(math.tan(math.pi / 4 + p2 / 2) / math.tan(math.pi / 4 + p1 / 2))
        return (n, float(g('LoVInDegrees')))
    finally:
        eccodes.codes_release(gid)

def earth_winds(u, v, lon, rot):
    import numpy as np
    if rot is None or u is None or v is None:
        return (u, v)
    n, lov = rot
    a = n * np.radians((lon - lov + 180) % 360 - 180)
    c, s = (np.cos(a), np.sin(a))
    return ((c * u + s * v).astype(np.float32), (c * v - s * u).astype(np.float32))

def latlons(msg: bytes):
    import eccodes
    import numpy as np
    gid = eccodes.codes_new_from_message(msg)
    try:
        ni = eccodes.codes_get(gid, 'Nx') if eccodes.codes_is_defined(gid, 'Nx') else eccodes.codes_get(gid, 'Ni')
        nj = eccodes.codes_get(gid, 'Ny') if eccodes.codes_is_defined(gid, 'Ny') else eccodes.codes_get(gid, 'Nj')
        lat = eccodes.codes_get_array(gid, 'latitudes').reshape(nj, ni)
        lon = eccodes.codes_get_array(gid, 'longitudes').reshape(nj, ni)
        lon = np.where(lon > 180, lon - 360, lon)
        return (lat, lon)
    finally:
        eccodes.codes_release(gid)

def fetch_messages(url, idx_rows: dict, workers: int=8, gap: int=64 * 1024, tries: int=6, timeout: float=90) -> dict:
    result = {}
    with ThreadPoolExecutor(workers) as ex:
        for f in submit_messages(ex, url, idx_rows, gap, tries, timeout):
            result.update(f.result())
    return result

def submit_messages(pool, url, idx_rows: dict, gap: int=64 * 1024, tries: int=6, timeout: float=90) -> list:
    by_offset = {r['offset']: k for k, r in idx_rows.items()}

    def get(group):
        a, b, rows = group
        blob = http_get(url, (a, b), tries, timeout)
        if b is not None and len(blob) != b - a + 1:
            raise OSError(f'range {a}-{b}: {len(blob)} bytes, not {b - a + 1}')
        out = {}
        for r in rows:
            start = r['offset'] - a
            end = None if r['end'] is None else r['end'] - a + 1
            out[by_offset[r['offset']]] = blob[start:end]
        return out
    futs = []
    for g in _merge_ranges(list(idx_rows.values()), gap):
        f = pool.submit(get, g)
        f.again = lambda g=g: get(g)
        futs.append(f)
    return futs
