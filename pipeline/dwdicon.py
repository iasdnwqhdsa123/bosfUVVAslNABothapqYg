from __future__ import annotations
import bz2
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import grib
BASE = 'https://opendata.dwd.de/weather/nwp/icon/grib'
EASY = 'https://opendata.dwd.de/weather/lib/cdo/ICON_GLOBAL2WORLD_0125_EASY.tar.bz2'
CELLS = 2949120
WORLD = (-90.0, -180.0, 0.125, 1441, 2879)
INDEX_FILE = 'dwd-icon-nn-0125.npy'
DOWNLOADS = 6
TRIES = 4
TIMEOUT = 60
KEEPS = timedelta(hours=26)
G = 9.80665
NOW, ACC, AVG = ('now', 'acc', 'avg')

def url(run, fh, var, kind='single-level', level=None) -> str:
    lv = '' if level is None else f'_{level}'
    return f'{BASE}/{run:%H}/{var}/icon_global_icosahedral_{kind}_{run:%Y%m%d%H}_{fh:03d}{lv}_{var.upper()}.grib2.bz2'

def static_url(run, var, level=None) -> str:
    lv = '' if level is None else f'_{level}'
    return f'{BASE}/{run:%H}/{var}/icon_global_icosahedral_time-invariant_{run:%Y%m%d%H}{lv}_{var.upper()}.grib2.bz2'

def has_run(run, now=None) -> bool:
    return os.environ.get('ICON_DWD', '1') != '0' and (now or datetime.now(timezone.utc)) - run < KEEPS

def landed(run, fh) -> bool:
    return has_run(run) and all((grib.exists(url(run, fh, v, 'model-level', 118)) for v in ('u', 'v')))

def _s(var, how=NOW):
    return ('single-level', var, None, how)

def _soil(var, level):
    return ('soil-level', var, level, NOW)

def _model(var, level):
    return ('model-level', var, level, NOW)

def _fixed(var):
    return ('time-invariant', var, None, NOW)

def _sum(a):
    return a[0] + a[1]

def _land(a):
    import numpy as np
    return (a[0] >= 0.5).astype(np.float32)

def _ptype(a):
    import om
    named = {'precipitation': a[0], 'snowfall_water_equivalent': a[1] + a[2], 'weather_code': a[3]}
    return om._ptype(named.__getitem__)
SURFACE = {'temperature_2m': ((_s('t_2m'),), None), 'td2': ((_s('td_2m'),), None), 'wind_u_component_10m': ((_s('u_10m'),), None), 'wind_v_component_10m': ((_s('v_10m'),), None), 'wind_u_component_80m': ((_model('u', 118),), None), 'wind_v_component_80m': ((_model('v', 118),), None), 'wind_gusts_10m': ((_s('vmax_10m'),), None), 'shf': ((_s('ashfl_s', AVG),), lambda a: -a[0]), 'lhf': ((_s('alhfl_s', AVG),), lambda a: -a[0]), 'sw': ((_s('aswdir_s', AVG), _s('aswdifd_s', AVG)), _sum), 'cape': ((_s('cape_ml'),), None), 'cloud_cover': ((_s('clct'),), None), 'cloud_cover_low': ((_s('clcl'),), None), 'cloud_cover_mid': ((_s('clcm'),), None), 'cloud_cover_high': ((_s('clch'),), None), 'precipitation': ((_s('tot_prec', ACC),), None), 'showers': ((_s('rain_con', ACC), _s('snow_con', ACC)), _sum), 'pressure_msl': ((_s('pmsl'),), None), 'soil_temperature_0cm': ((_soil('t_so', 0),), None), 'lsm': ((_fixed('fr_land'),), _land), 'ice': ((_s('fr_ice'),), None), 'snow_depth': ((_s('h_snow'),), None), 'freezing_level_height': ((_s('hzerocl'),), None), 'ptype': ((_s('tot_prec', ACC), _s('snow_gsp', ACC), _s('snow_con', ACC), _s('ww')), _ptype), 'soil_moisture_3_to_9cm': ((_soil('w_so', 3),), lambda a: a[0] / 60), 'convective_cloud_base': ((_s('hbas_con'),), None), 'convective_cloud_top': ((_s('htop_con'),), None), 'zs': ((_fixed('hsurf'),), None), 'ps': ((_s('ps'),), None)}
LEVEL = {'temperature': 't', 'relative_humidity': 'relhum', 'wind_u_component': 'u', 'wind_v_component': 'v', 'geopotential_height': 'fi'}
OWN = ' ml'

def _half(level):
    return ('time-invariant', 'hhl', level, NOW)

def _mean(a):
    return (a[0] + a[1]) / 2

def recipe(key):
    name, level = key
    if level.endswith(OWN):
        k = int(level[:-len(OWN)])
        return ((_half(k), _half(k + 1)), _mean) if name == 'hhl' else ((_model(name, k),), None)
    if level.endswith(' mb'):
        p = int(level[:-3])
        return ((('pressure-level', LEVEL[name], p, NOW),), (lambda a: a[0] / G) if name == 'geopotential_height' else None)
    if level:
        raise KeyError(key)
    return SURFACE[name]

class Index:

    def __init__(self, world, how):
        self.world, self.how, self._windows = (world, how, {})

    def window(self, region):
        import numpy as np
        import om
        key = tuple(region[:5])
        if key not in self._windows:
            i_n, j_w, ny, nx = om.Grid(*WORLD).window(region)
            self._windows[key] = np.ascontiguousarray(self.world[i_n - ny + 1:i_n + 1, j_w:j_w + nx][::-1])
        return self._windows[key]

def index_from_links(path, shape=WORLD[3:], cells=CELLS):
    import warnings
    import numpy as np
    from scipy.io import netcdf_file
    ny, nx = shape
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with netcdf_file(path, 'r', mmap=True) as f:
            method = f._attributes.get('map_method', b'')
            method = method.decode() if isinstance(method, bytes) else str(method)
            if 'nearest' not in method.lower() or f.dimensions.get('num_wgts') != 1:
                raise ValueError(f"not nearest-neighbour links ({method!r}, {f.dimensions.get('num_wgts')} weights)")
            if f.dimensions.get('src_grid_size') != cells or f.dimensions.get('dst_grid_size') != ny * nx:
                raise ValueError(f"links for {f.dimensions.get('src_grid_size')} → {f.dimensions.get('dst_grid_size')} cells, not {cells} → {ny * nx}")
            src = np.array(f.variables['src_address'][:], np.int64) - 1
            dst = np.array(f.variables['dst_address'][:], np.int64) - 1
    world = np.full(ny * nx, -1, np.int64)
    ok = (dst >= 0) & (dst < ny * nx)
    world[dst[ok]] = src[ok]
    if (world < 0).any() or world.max() >= cells:
        raise ValueError("links don't cover every cell")
    return world.astype(np.int32).reshape(ny, nx)

def index_from_package(path, **kw):
    import shutil
    import tarfile
    with tarfile.open(path, 'r:bz2') as tar:
        member = next((m for m in tar.getmembers() if m.isfile() and m.name.endswith('.nc')), None)
        if member is None:
            raise ValueError('no weights file in the package')
        with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
            shutil.copyfileobj(tar.extractfile(member), tmp)
            tmp.flush()
            return index_from_links(tmp.name, **kw)

def _xyz(lat, lon):
    import numpy as np
    la, lo = (np.radians(lat), np.radians(lon))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

def index_from_cells(clat, clon, grid=WORLD):
    import numpy as np
    from scipy.spatial import cKDTree
    lat0, lon0, step, ny, nx = grid
    tree = cKDTree(_xyz(np.asarray(clat, np.float64), np.asarray(clon, np.float64)))
    la, lo = np.meshgrid(lat0 + np.arange(ny) * step, lon0 + np.arange(nx) * step, indexing='ij')
    _, i = tree.query(_xyz(la, lo).reshape(-1, 3), workers=-1)
    return i.astype(np.int32).reshape(ny, nx)
_INDEX = None
_INDEX_LOCK = threading.Lock()

def index(cache_dir=None, run=None, log=print):
    global _INDEX
    import numpy as np
    with _INDEX_LOCK:
        if _INDEX is not None:
            return _INDEX
        path = os.path.join(cache_dir, INDEX_FILE) if cache_dir else None
        if path and os.path.exists(path):
            try:
                w = np.load(path)
                if w.shape == WORLD[3:] and w.dtype == np.int32 and (0 <= w.min()) and (w.max() < CELLS):
                    _INDEX = Index(w, "DWD's nearest-neighbour links (stored)")
                    return _INDEX
            except Exception as e:
                log(f'  ICON grid index in {path} not read ({e!r})')
        try:
            import om
            pkg = om.download(EASY, workers=4)
            try:
                w = index_from_package(pkg)
            finally:
                os.unlink(pkg)
            how = "DWD's nearest-neighbour links"
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(path + '.tmp.npy', w)
                os.replace(path + '.tmp.npy', path)
        except Exception as e:
            if run is None:
                raise
            log(f"  ICON grid: DWD's links not had ({e!r}): the nearest of the run's cell centres")
            w = index_from_cells(_static(run, 'clat'), _static(run, 'clon'))
            how = 'the nearest of the cell centres'
        _INDEX = Index(w, how)
        log(f'  ICON grid: {how}')
        return _INDEX
_SLOTS = threading.BoundedSemaphore(DOWNLOADS)

def message(u: str) -> bytes:
    with _SLOTS:
        b = grib.http_get(u, tries=TRIES, timeout=TIMEOUT)
    return bz2.decompress(b)

def values(msg: bytes, cells=CELLS):
    import eccodes
    import numpy as np
    gid = eccodes.codes_new_from_message(msg)
    try:
        n = eccodes.codes_get(gid, 'numberOfDataPoints')
        if n != cells:
            raise ValueError(f"a message on {n} cells, not ICON's {cells}")
        v = eccodes.codes_get_values(gid).astype(np.float32)
        if eccodes.codes_get(gid, 'bitmapPresent'):
            v[v == eccodes.codes_get(gid, 'missingValue')] = np.nan
        return v
    finally:
        eccodes.codes_release(gid)
_STATIC: dict = {}
_STATIC_LOCK = threading.Lock()

def _static(run, var, level=None):
    k = var if level is None else (var, level)
    with _STATIC_LOCK:
        if k not in _STATIC:
            _STATIC[k] = values(message(static_url(run, var, level)))
        return _STATIC[k]

class Store:

    def __init__(self, folder=None):
        self.f = tempfile.TemporaryFile(dir=folder)
        self.size, self.lock = (0, threading.Lock())

    def put(self, b: bytes):
        with self.lock:
            at, self.size = (self.size, self.size + len(b))
        os.pwrite(self.f.fileno(), b, at)
        return (at, len(b))

    def get(self, at, n) -> bytes:
        b = os.pread(self.f.fileno(), n, at)
        if len(b) != n:
            raise IOError(f"short read from the step's store: {len(b)} of {n} bytes")
        return b

    def __del__(self):
        f = getattr(self, 'f', None)
        if f is not None:
            f.close()

class Field:
    threads = True

    def __init__(self, idx: Index, parts, combine, fh, prev):
        self.idx, self.parts, self.combine, self.fh, self.prev = (idx, parts, combine, fh, prev)

    def _part(self, now, before, how):
        v = now()
        if before is None or how == NOW:
            return v
        if how == ACC:
            return v - before()
        return (v * self.fh - before() * self.prev) / (self.fh - self.prev)

    def read(self, region, boxes=None):
        import numpy as np
        a = [self._part(*p) for p in self.parts]
        v = self.combine(a) if self.combine else a[0]
        cells = self.idx.window(region)
        if not boxes:
            return np.asarray(v[cells], np.float32)
        out = np.full(cells.shape, np.nan, np.float32)
        for y0, y1, x0, x1 in boxes:
            out[y0:y1, x0:x1] = v[cells[y0:y1, x0:x1]]
        return out

def fetch_step(run, fh, prev, keys, cache_dir=None, workers=DOWNLOADS, log=print):
    recipes = {k: recipe(k) for k in keys}
    idx = index(cache_dir, run, log)
    files = set()
    for parts, _ in recipes.values():
        for kind, var, level, how in parts:
            if kind != 'time-invariant':
                files.add((kind, var, level, fh))
                if how != NOW and prev > 0:
                    files.add((kind, var, level, prev))
    for parts, _ in recipes.values():
        for kind, var, level, _ in parts:
            if kind == 'time-invariant':
                _static(run, var, level)
    store, where = (Store(), {})

    def get(f):
        kind, var, level, step = f
        where[f] = store.put(message(url(run, step, var, kind, level)))
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(get, f) for f in sorted(files, key=lambda f: (f[3], f[0], f[1], str(f[2])))]
        try:
            for fu in futs:
                fu.result()
        except BaseException:
            for fu in futs:
                fu.cancel()
            raise

    def reader(f):
        at, n = where[f]
        return lambda: values(store.get(at, n))

    def fixed(var, level=None):
        return lambda: _STATIC[var if level is None else (var, level)]
    out = {}
    for k, (parts, combine) in recipes.items():
        ps = []
        for kind, var, level, how in parts:
            if kind == 'time-invariant':
                ps.append((fixed(var, level), None, NOW))
            else:
                before = reader((kind, var, level, prev)) if how != NOW and prev > 0 else None
                ps.append((reader((kind, var, level, fh)), before, how))
        out[k] = Field(idx, ps, combine, fh, prev)
    return out
