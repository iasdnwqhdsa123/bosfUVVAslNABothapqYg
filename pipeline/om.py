from __future__ import annotations
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import grib
PART = 16 << 20
K0 = 273.15
G = 9.80665
RD = 287.05

def head_size(url: str, tries: int=4) -> int:
    import time
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=grib.UA, method='HEAD')
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(r.headers['Content-Length'])
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) or attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError('unreachable')

def download(url: str, workers: int=8, part: int=PART, folder: str | None=None) -> str:
    size = head_size(url)
    fd, path = tempfile.mkstemp(suffix='.om', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.truncate(size)

        def get(a):
            b = grib.http_get(url, (a, min(a + part, size) - 1))
            if len(b) != min(part, size - a):
                raise IOError(f'short read at {a} of {url}')
            with open(path, 'r+b') as f:
                f.seek(a)
                f.write(b)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(get, range(0, size, part)))
    except BaseException:
        os.unlink(path)
        raise
    return path

def open_file(path: str, unlink: bool=True):
    import omfiles
    r = omfiles.OmFileReader.from_path(path)
    if unlink:
        os.unlink(path)
    return r

class Grid:

    def __init__(self, lat0, lon0, step, ny, nx):
        self.lat0, self.lon0, self.step, self.ny, self.nx = (lat0, lon0, step, ny, nx)

    def window(self, region):
        w, s, e, n, step = region[:5]
        if abs(step - self.step) > 1e-09:
            raise ValueError(f"output step {step}° is not the files' {self.step}°")
        i_n, j_w, i_s, j_e = ((n - self.lat0) / step, (w - self.lon0) / step, (s - self.lat0) / step, (e - self.lon0) / step)
        if any((abs(x - round(x)) > 1e-06 for x in (i_n, j_w, i_s, j_e))):
            raise ValueError(f"region {region} is not on the files' grid")
        i_n, j_w, i_s, j_e = (int(round(x)) for x in (i_n, j_w, i_s, j_e))
        if i_s < 0 or i_n >= self.ny or j_w < 0 or (j_e >= self.nx):
            raise ValueError(f"region {region} is outside the files' grid")
        return (i_n, j_w, i_n - i_s + 1, j_e - j_w + 1)

class Hour:

    def __init__(self, readers, grid: Grid, arrays=None):
        self.grid = grid
        self.names = dict(arrays or {})
        self._units = {}
        for r in readers:
            for i in range(r.num_children):
                c = r.get_child_by_index(i)
                if c.is_array:
                    self.names.setdefault(c.name, c)

    def has(self, name):
        return name in self.names

    def unit(self, name):
        if name not in self._units:
            c, u = (self.names[name], None)
            for j in range(c.num_children):
                a = c.get_child_by_index(j)
                if a.name == 'unit':
                    u = a.read_scalar()
            self._units[name] = u
        return self._units[name]

    def read(self, name, region, boxes=None):
        import numpy as np
        c = self.names[name]
        i_n, j_w, ny, nx = self.grid.window(region)
        out = np.full((ny, nx), np.nan, np.float32)
        for y0, y1, x0, x1 in boxes or [(0, ny, 0, nx)]:
            a = c.read_array((slice(i_n - y1 + 1, i_n - y0 + 1), slice(j_w + x0, j_w + x1)))
            out[y0:y1, x0:x1] = a[::-1]
        return out
SEA = -998.0

def _to_si(v, unit):
    if unit == '°C':
        return v + K0
    if unit == 'hPa':
        return v * 100
    return v

def _td2(get):
    import numpy as np
    t, rh = (get('temperature_2m') - K0, np.clip(get('relative_humidity_2m'), 1, 100))
    e = 6.112 * np.exp(17.67 * t / (t + 243.5)) * rh / 100
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln) + K0
PS_LEVELS = (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500)
PS_NEEDS = (1000, 850, 700, 500)

def _ps(get):
    import numpy as np
    zs, t2 = (_zs(get), get('temperature_2m'))
    ps = np.full(zs.shape, np.nan, np.float32)
    for p in reversed(PS_LEVELS):
        if not (get.has(f'geopotential_height_{p}hPa') and get.has(f'temperature_{p}hPa')):
            continue
        z, t = (get(f'geopotential_height_{p}hPa'), get(f'temperature_{p}hPa'))
        tm = 0.5 * (t + t2)
        ok = z >= zs
        ps = np.where(ok, p * 100 * np.exp(G * (z - zs) / (RD * tm)), ps)
    return ps.astype(np.float32)

def _zs(get):
    import numpy as np
    return np.maximum(get('HSURF'), 0)

def _lsm(get):
    h = get('HSURF')
    return (h > SEA).astype('float32')

def _ice(get):
    import numpy as np
    return ((get('HSURF') <= SEA) & (get('soil_temperature_0cm') < K0 - 1.8)).astype(np.float32)

def _ptype(get):
    import numpy as np
    pr = np.nan_to_num(get('precipitation'), nan=0)
    share = np.where(pr > 0, np.nan_to_num(get('snowfall_water_equivalent'), nan=0) / np.maximum(pr, 1e-06), 0)
    wc = np.nan_to_num(get('weather_code'), nan=0)
    code = np.select([np.isin(wc, (56, 57, 66, 67)), share >= 0.8, share >= 0.2], [3, 5, 7], 1)
    return np.where(np.isfinite(get('precipitation')), code, np.nan).astype(np.float32)
RECIPES = {'td2': (('temperature_2m', 'relative_humidity_2m'), _td2), 'ps': (('temperature_2m', 'HSURF') + tuple((f'{k}_{p}hPa' for p in PS_NEEDS for k in ('geopotential_height', 'temperature'))), _ps), 'zs': (('HSURF',), _zs), 'lsm': (('HSURF',), _lsm), 'ice': (('HSURF', 'soil_temperature_0cm'), _ice), 'sw': (('direct_radiation', 'diffuse_radiation'), lambda get: get('direct_radiation') + get('diffuse_radiation')), 'shf': (('sensible_heat_flux',), lambda get: -get('sensible_heat_flux')), 'ptype': (('precipitation', 'snowfall_water_equivalent', 'weather_code'), _ptype)}

class Field:

    def __init__(self, hour: Hour, key: str):
        self.hour, self.key = (hour, key)

    def needs(self):
        return RECIPES[self.key][0] if self.key in RECIPES else (self.key,)

    def available(self):
        return all((self.hour.has(n) for n in self.needs()))

    def read(self, region, boxes=None):
        memo = {}

        def get(name):
            if name not in memo:
                memo[name] = _to_si(self.hour.read(name, region, boxes), self.hour.unit(name))
            return memo[name]
        get.has = self.hour.has
        if self.key in RECIPES:
            return RECIPES[self.key][1](get)
        return get(self.key)
_STATIC: dict = {}

def static_reader(url: str):
    if url not in _STATIC:
        path = download(url, workers=1)
        _STATIC[url] = open_file(path)
    return _STATIC[url]

def om_name(key):
    name, level = key
    return f'{name}_{level[:-3]}hPa' if level.endswith(' mb') else name

def fetch_hour(urls, static_url, grid: Grid, keys, workers=8):
    readers = [open_file(download(u, workers=workers)) for u in urls]
    hour = Hour(readers, grid, arrays={'HSURF': static_reader(static_url)})
    out = {}
    for k in keys:
        f = Field(hour, om_name(k))
        if f.available():
            out[k] = f
    return out

def gaussian_rows(n=1280):
    import numpy as np
    x, _ = np.polynomial.legendre.leggauss(2 * n)
    lats = np.degrees(np.arcsin(x))[::-1]
    i = np.arange(1, 2 * n + 1)
    nlon = np.where(i <= n, 4 * i + 16, 4 * (2 * n + 1 - i) + 16)
    start = np.concatenate([[0], np.cumsum(nlon)[:-1]])
    return (lats, nlon, start)

def gaussian_index(lats, lons, rows):
    import numpy as np
    glat, nlon, start = rows
    j = np.searchsorted(-glat, -np.asarray(lats))
    j = np.clip(j, 1, len(glat) - 1)
    row = np.where(np.abs(glat[j - 1] - lats) <= np.abs(glat[j] - lats), j - 1, j)
    n = nlon[row][:, None]
    k = np.round(np.asarray(lons)[None, :] % 360 / 360 * n).astype(np.int64) % n
    return start[row][:, None] + k

def read_flat(path, names, take=None):
    import numpy as np
    r = open_file(path)
    out = {}
    for i in range(r.num_children):
        c = r.get_child_by_index(i)
        if c.is_array and c.name in names:
            unit = None
            for j in range(c.num_children):
                a = c.get_child_by_index(j)
                if a.name == 'unit':
                    unit = a.read_scalar()
            v = np.asarray(c.read_array((slice(0, c.shape[0]), slice(0, c.shape[1]))), np.float32).reshape(-1)
            out[c.name] = _to_si(v, unit) if take is None else take(_to_si(v, unit))
    return out
