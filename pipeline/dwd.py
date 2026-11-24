from __future__ import annotations
import bz2
from concurrent.futures import ThreadPoolExecutor
import grib
BASE = 'https://opendata.dwd.de/weather/nwp/icon-d2/grib'

def url(run, fh: int, name: str, p: int) -> str:
    return f'{BASE}/{run:%H}/{name}/icon-d2_germany_regular-lat-lon_pressure-level_{run:%Y%m%d%H}_{fh:03d}_{p}_{name}.grib2.bz2'

def message(u: str) -> bytes:
    return bz2.decompress(grib.http_get(u, tries=4, timeout=60))

def single_url(run, fh: int, name: str) -> str:
    return f'{BASE}/{run:%H}/{name}/icon-d2_germany_regular-lat-lon_single-level_{run:%Y%m%d%H}_{fh:03d}_2d_{name}.grib2.bz2'

def messages(u: str) -> list:
    b, out, i = (message(u), [], 0)
    while True:
        j = b.find(b'GRIB', i)
        if j < 0 or j + 16 > len(b):
            return out
        n = int.from_bytes(b[j + 8:j + 16], 'big')
        out.append(b[j:j + n])
        i = j + n

class Field:

    def __init__(self, grid, msg: bytes):
        self.grid, self.msg = (grid, msg)

    def read(self, region, boxes=None):
        import numpy as np
        i_n, j_w, ny, nx = self.grid.window(region)
        a, _ = grib.decode(self.msg)
        if a.shape != (self.grid.ny, self.grid.nx):
            raise ValueError(f'ICON-D2 message on a {a.shape} grid, not {(self.grid.ny, self.grid.nx)}')
        v = a[i_n - ny + 1:i_n + 1, j_w:j_w + nx][::-1]
        if not boxes:
            return np.ascontiguousarray(v, np.float32)
        out = np.full((ny, nx), np.nan, np.float32)
        for y0, y1, x0, x1 in boxes:
            out[y0:y1, x0:x1] = v[y0:y1, x0:x1]
        return out

def level_url(run, fh: int, name: str, level: int) -> str:
    return f'{BASE}/{run:%H}/{name}/icon-d2_germany_regular-lat-lon_model-level_{run:%Y%m%d%H}_{fh:03d}_{level}_{name}.grib2.bz2'

def half_level_url(run, level: int) -> str:
    return f'{BASE}/{run:%H}/hhl/icon-d2_germany_regular-lat-lon_time-invariant_{run:%Y%m%d%H}_000_{level}_hhl.grib2.bz2'
_HALF: dict = {}

def half_level(run, level: int) -> bytes:
    if level not in _HALF:
        _HALF[level] = message(half_level_url(run, level))
    return _HALF[level]

class Mean(Field):

    def __init__(self, grid, a: bytes, b: bytes):
        self.grid, self.a, self.b = (grid, a, b)

    def read(self, region, boxes=None):
        return (Field(self.grid, self.a).read(region, boxes) + Field(self.grid, self.b).read(region, boxes)) / 2

def submit_levels(pool: ThreadPoolExecutor, run, fh: int, keys, label: str) -> dict:
    out = {}
    for k in keys:
        level = int(k[1][:-len(label)])
        if k[0] == 'hhl':
            out[k] = pool.submit(lambda lv=level: (half_level(run, lv), half_level(run, lv + 1)))
        else:
            out[k] = pool.submit(message, level_url(run, fh, k[0], level))
    return out

def submit(pool: ThreadPoolExecutor, run, fh: int, keys, names: dict) -> dict:
    return {k: pool.submit(message, url(run, fh, names[k[0]], int(k[1][:-3]))) for k in keys}
