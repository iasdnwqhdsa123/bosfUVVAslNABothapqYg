from __future__ import annotations
import io
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import numpy as np
WINDNINJA = os.environ.get('WINDNINJA', 'WindNinja_cli')
DEM_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
Z_DEM = 12
DEM_M = 60.0
MESH_M = 120.0
DOMAIN_KM, CORE_KM = (26.0, 16.0)
REF_H = 300.0
U_REF = 10.0
HEIGHTS = (10, 25, 50, 75, 100, 150, 200, 300, 450, 600)
WORLDCOVER = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif'
COVER = {10: (99, 70, 15.0), 20: (99, 30, 1.5), 30: (99, 50, 0.3), 40: (93, 0, 0.0), 50: (91, 0, 0.0), 60: (90, 0, 0.0), 70: (92, 0, 0.0), 80: (98, 0, 0.0), 90: (99, 50, 0.3), 95: (99, 70, 8.0), 100: (99, 50, 0.1)}
NO_COVER = 80
FOREST = (10, 95)
MAX_SLOPE, LEE_SLOPE, SHAPE, DECAY, RELIEF_LO, RELIEF_HI, BLOCK_MIN = (0.7, 0.4, 0.65, 2.5, 20.0, 50.0, 0.05)
RELIEF_R_M, SLOPE_M = (750.0, 125.0)
RIDGE_LEAN, RIDGE_SPREAD = (0.0, 0.4)
LEE_SHARE = 0.5
CONV_DEPTH = 500.0
SLOPE_FLOW, SLOPE_FULL, SLOPE_SIGMA_M, SLOPE_DEPTH, SLOPE_DEPTH_MAX = (0.5, 0.15, 1000.0, 0.3, 600.0)

def build_windninja(src: str, prefix: str) -> str:
    b = os.path.join(prefix, 'build')
    os.makedirs(b, exist_ok=True)
    flags = ['-DUSE_VCPKG_MANIFEST=OFF', '-DSUPRESS_WARNINGS=ON', '-DNINJA_GUI=OFF', '-DNINJAFOAM=OFF', '-DWITH_NOMADS_SUPPORT=OFF', '-DWITH_LCP_CLIENT=OFF', '-DPHONE_HOME_QUERIES=OFF', '-DDISABLE_THREDDS_UPDATE=ON', '-DENABLE_GMTED=OFF', '-DSKIP_GETDATE=ON', '-DOPENMP_SUPPORT=ON', '-DCMAKE_BUILD_TYPE=Release']
    subprocess.run(['cmake', src, *flags], cwd=b, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['make', f'-j{os.cpu_count() or 2}', 'WindNinja_cli'], cwd=b, check=True, stdout=subprocess.DEVNULL)
    return os.path.join(b, 'src', 'cli', 'WindNinja_cli')

def read_vtk(path: str):
    b = open(path, 'rb').read()
    nx, ny, nz = (int(v) for v in re.search(b'DIMENSIONS (\\d+) (\\d+) (\\d+)', b).groups())
    n = nx * ny * nz
    i = b.index(b'\n', b.index(b'POINTS')) + 1
    P = np.frombuffer(b, '>f8', n * 3, i).reshape(nz, ny, nx, 3)
    j = b.index(b'\n', b.index(b'VECTORS', i + n * 24)) + 1
    V = np.frombuffer(b, '>f8', n * 3, j).reshape(nz, ny, nx, 3)
    return (P.astype(np.float32), V.astype(np.float32))

def at_heights(P, V, heights=HEIGHTS):
    z = P[..., 2]
    agl = z - z[0]
    iy, ix = np.indices(z.shape[1:])
    out = np.empty((len(heights), 3, *z.shape[1:]), np.float32)
    for k, h in enumerate(heights):
        j = np.clip((agl < h).sum(0), 1, z.shape[0] - 1)
        a0, a1 = (agl[j - 1, iy, ix], agl[j, iy, ix])
        t = np.clip((h - a0) / np.maximum(a1 - a0, 1e-06), 0, 1)
        out[k] = np.moveaxis(V[j - 1, iy, ix] * (1 - t[..., None]) + V[j, iy, ix] * t[..., None], -1, 0)
    return out

def run(dem: str, direction: float, out: str, threads: int | None=None, landscape: bool | None=None) -> str:
    if landscape is None:
        landscape = os.path.basename(dem).startswith('landscape')
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.setdefault('WINDNINJA_DATA', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(WINDNINJA)))), 'data'))
    args = [WINDNINJA, '--num_threads', str(threads or os.cpu_count() or 2), '--elevation_file', dem, '--initialization_method', 'domainAverageInitialization', '--input_speed', str(U_REF), '--input_speed_units', 'mps', '--input_direction', str(direction), '--input_wind_height', str(REF_H), '--units_input_wind_height', 'm', '--output_wind_height', '10', '--units_output_wind_height', 'm', *(['--vegetation', 'grass'] if not landscape else []), '--mesh_resolution', str(MESH_M), '--units_mesh_resolution', 'm', '--output_speed_units', 'mps', '--write_vtk_output', 'true', '--write_ascii_output', 'false', '--output_path', out]
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1800)
    vtk = [f for f in os.listdir(out) if f.endswith('.vtk') and (not f.endswith('_surf.vtk'))]
    if r.returncode != 0 or not vtk:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        raise RuntimeError(f"WindNinja failed ({r.returncode}): {' | '.join(tail)}")
    return os.path.join(out, vtk[0])

def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) % 60 + 1

def utm_xy(lat, lon, zone: int, south: bool):
    a, f, k0 = (6378137.0, 1 / 298.257223563, 0.9996)
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n * n / 4 + n ** 4 / 64)
    al = (n / 2 - 2 / 3 * n * n + 5 / 16 * n ** 3, 13 / 48 * n * n - 3 / 5 * n ** 3, 61 / 240 * n ** 3)
    ph = np.radians(np.asarray(lat, np.float64))
    dl = np.radians(np.asarray(lon, np.float64) - (zone * 6 - 183))
    c = 2 * math.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(ph)) - c * np.arctanh(c * np.sin(ph)))
    xi, eta = (np.arctan2(t, np.cos(dl)), np.arctanh(np.sin(dl) / np.sqrt(1 + t * t)))
    E = xi * 0 + eta
    N = xi + 0
    for j, aj in enumerate(al, 1):
        E = E + aj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        N = N + aj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return (500000 + k0 * A * E, k0 * A * N + (10000000 if south else 0))

def _tile(z, x, y, cache):
    fn = os.path.join(cache, f'{z}_{x}_{y}.png') if cache else None
    if fn and os.path.exists(fn):
        b = open(fn, 'rb').read()
    else:
        b = None
        for _ in range(4):
            try:
                b = urllib.request.urlopen(DEM_URL.format(z=z, x=x, y=y), timeout=60).read()
                break
            except Exception:
                continue
        if b is None:
            raise RuntimeError(f'terrain tile {z}/{x}/{y} not fetched')
        if fn:
            os.makedirs(cache, exist_ok=True)
            open(fn, 'wb').write(b)
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(b)).convert('RGB'), np.float32)
    return np.maximum(a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768, 0)

def dem(lat: float, lon: float, km: float, work: str, cache: str | None=None, cover: bool=True):
    z = Z_DEM
    dlat, dlon = (km / 2 / 110.5 + 0.01, km / 2 / (111.3 * math.cos(math.radians(lat))) + 0.01)
    tx = lambda lo: (lo + 180) / 360 * 2 ** z
    ty = lambda la: (1 - math.log(math.tan(math.radians(la)) + 1 / math.cos(math.radians(la))) / math.pi) / 2 * 2 ** z
    x0, x1, y0, y1 = (int(tx(lon - dlon)), int(tx(lon + dlon)), int(ty(lat + dlat)), int(ty(lat - dlat)))
    A = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.float32)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            A[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = _tile(z, x, y, cache)
    R = 6378137.0
    res = 2 * math.pi * R / 256 / 2 ** z
    os.makedirs(work, exist_ok=True)
    raw = os.path.join(work, 'merc.raw')
    A.tofile(raw)
    merc = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs"]]'
    with open(os.path.join(work, 'merc.hdr'), 'w') as f:
        f.write(f'ENVI\nsamples = {A.shape[1]}\nlines = {A.shape[0]}\nbands = 1\nheader offset = 0\ndata type = 4\ninterleave = bsq\nbyte order = 0\nmap info = {{Pseudo Mercator, 1, 1, {x0 * 256 * res - math.pi * R}, {math.pi * R - y0 * 256 * res}, {res}, {res}}}\ncoordinate system string = {{{merc}}}\n')
    zone, south = (utm_zone(lon), lat < 0)
    cx, cy = utm_xy(lat, lon, zone, south)
    h = km * 500
    te = [str(round(v / DEM_M) * DEM_M) for v in (cx - h, cy - h, cx + h, cy + h)]
    tif = os.path.join(work, 'dem.tif')
    srs = f'EPSG:{(32700 if south else 32600) + zone}'
    subprocess.run(['gdalwarp', '-q', '-overwrite', '-t_srs', srs, '-te', *te, '-tr', str(DEM_M), str(DEM_M), '-r', 'bilinear', raw, tif], check=True)
    env = os.path.join(work, 'dem.env')
    subprocess.run(['gdal_translate', '-q', '-of', 'ENVI', '-ot', 'Float32', tif, env], check=True)
    n = round((float(te[2]) - float(te[0])) / DEM_M)
    Z = np.fromfile(env, np.float32).reshape(-1, n)
    out = {'z': Z, 'x0': float(te[0]) + DEM_M / 2, 'y0': float(te[3]) - DEM_M / 2, 'res': DEM_M, 'zone': zone, 'south': south}
    if cover:
        lc = landcover(lat, lon, dlat, dlon, srs, te, Z.shape, work)
        out['cover'] = lc
        tif = landscape(Z, lc, env, work)
    return (tif, out)

def _cover_tiles(lat, lon, dlat, dlon) -> list:
    out = []
    for la in range(math.floor((lat - dlat) / 3) * 3, math.floor((lat + dlat) / 3) * 3 + 1, 3):
        for lo in range(math.floor((lon - dlon) / 3) * 3, math.floor((lon + dlon) / 3) * 3 + 1, 3):
            name = f"{('N' if la >= 0 else 'S')}{abs(la):02d}{('E' if lo >= 0 else 'W')}{abs(lo):03d}"
            url = WORLDCOVER.format(tile=name)
            try:
                urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30)
                out.append(url)
            except Exception:
                continue
    return out

def landcover(lat, lon, dlat, dlon, srs, te, shape, work):
    urls = _cover_tiles(lat, lon, dlat, dlon)
    if not urls:
        return np.full(shape, NO_COVER, np.uint8)
    lt, le = (os.path.join(work, 'lc.tif'), os.path.join(work, 'lc.env'))
    subprocess.run(['gdalwarp', '-q', '-overwrite', '-t_srs', srs, '-te', *te, '-tr', str(DEM_M), str(DEM_M), '-r', 'mode', '-dstnodata', '0', *(f'/vsicurl/{u}' for u in urls), lt], check=True, env={**os.environ, 'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR'})
    subprocess.run(['gdal_translate', '-q', '-of', 'ENVI', lt, le], check=True)
    a = np.fromfile(le, np.uint8).reshape(shape)
    return np.where(a == 0, NO_COVER, a).astype(np.uint8)

def landscape_bands(Z, lc) -> np.ndarray:
    fuel, cov, ht = (np.zeros(Z.shape, np.float32) for _ in range(3))
    for c, (f, cv, h) in COVER.items():
        m = lc == c
        fuel[m], cov[m], ht[m] = (f, cv, h * 10)
    other = ~np.isin(lc, list(COVER))
    fuel[other], cov[other], ht[other] = (99, 50, 3)
    zero = np.zeros(Z.shape, np.float32)
    return np.stack([Z, zero, zero, fuel, cov, ht, zero, zero]).astype(np.float32)

def landscape(Z, lc, dem_env, work) -> str:
    raw = os.path.join(work, 'lcp.raw')
    landscape_bands(Z, lc).tofile(raw)
    hdr = open(dem_env[:-4] + '.hdr').read()
    hdr = re.sub('bands\\s*=\\s*\\d+', 'bands = 8', hdr)
    hdr = re.sub('interleave\\s*=\\s*\\w+', 'interleave = bsq', hdr)
    hdr = re.sub('band names\\s*=\\s*\\{[^}]*\\}\\n?', '', hdr)
    open(os.path.join(work, 'lcp.hdr'), 'w').write(hdr)
    tif = os.path.join(work, 'landscape.tif')
    subprocess.run(['gdal_translate', '-q', '-of', 'GTiff', raw, tif], check=True)
    return tif

def _bilinear(F, x0, y0, res, x, y):
    fx, fy = ((np.asarray(x) - x0) / res, (y0 - np.asarray(y)) / res)
    ny, nx = F.shape[-2:]
    i, j = (np.floor(fx).astype(int), np.floor(fy).astype(int))
    ok = (i >= 0) & (j >= 0) & (i < nx - 1) & (j < ny - 1)
    i, j = (np.clip(i, 0, nx - 2), np.clip(j, 0, ny - 2))
    tx, ty = (fx - i, fy - j)
    v = F[..., j, i] * (1 - tx) * (1 - ty) + F[..., j, i + 1] * tx * (1 - ty) + F[..., j + 1, i] * (1 - tx) * ty + F[..., j + 1, i + 1] * tx * ty
    return np.where(ok, v, np.nan)

class Domain:

    def __init__(self, lat: float, lon: float, work: str, cache: str | None=None, km: float=DOMAIN_KM, solve: bool=True, cover: bool=True):
        from scipy import ndimage
        self.lat, self.lon = (lat, lon)
        tif, d = dem(lat, lon, km, os.path.join(work, 'dem'), cache, cover)
        self.cover = d.get('cover')
        self.zone, self.south = (d['zone'], d['south'])
        self.cx, self.cy = utm_xy(lat, lon, self.zone, self.south)
        e, self.res, self.x0, self.y0 = (d['z'], d['res'], d['x0'], d['y0'])
        self.e = ndimage.median_filter(e, size=3)
        k = max(1, round(SLOPE_M / self.res))
        self.gx = np.zeros_like(self.e)
        self.gy = np.zeros_like(self.e)
        self.gx[:, k:-k] = (self.e[:, 2 * k:] - self.e[:, :-2 * k]) / (2 * k * self.res)
        self.gy[k:-k] = (self.e[:-2 * k] - self.e[2 * k:]) / (2 * k * self.res)
        r = max(1, round(RELIEF_R_M / self.res))
        self.hi = ndimage.maximum_filter(self.e, size=2 * r + 1)
        self.relief = self.hi - ndimage.minimum_filter(self.e, size=2 * r + 1)
        self.E = self.N = None
        if solve:
            out = []
            for name, frm in (('east', 270.0), ('north', 180.0)):
                P, V = read_vtk(run(tif, frm, os.path.join(work, name)))
                out.append(at_heights(P, V) / U_REF)
            self.wx0, self.wy0 = (float(P[0, 0, 0, 0]), float(P[0, 0, 0, 1]))
            self.wres = float(P[0, 0, 1, 0] - P[0, 0, 0, 0])
            self.E, self.N = (np.ascontiguousarray(a[..., ::-1, :]) for a in out)
            self.wy0 = float(P[0, -1, 0, 1])
            self.ground = np.ascontiguousarray(P[0, ::-1, :, 2])
        shutil.rmtree(work, ignore_errors=True)

    def xy(self, lat, lon):
        return utm_xy(lat, lon, self.zone, self.south)

    def in_core(self, x, y, core_km: float=CORE_KM) -> np.ndarray:
        h = core_km * 500
        return (np.abs(np.asarray(x) - self.cx) <= h) & (np.abs(np.asarray(y) - self.cy) <= h)

    def forest_at(self, x, y):
        if self.cover is None:
            return np.full(np.shape(x), np.nan)
        f = np.isin(self.cover, FOREST).astype(np.float32)
        i = np.round((np.asarray(x) - self.x0) / self.res).astype(int)
        j = np.round((self.y0 - np.asarray(y)) / self.res).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < f.shape[1]) & (j < f.shape[0])
        return np.where(ok, f[np.clip(j, 0, f.shape[0] - 1), np.clip(i, 0, f.shape[1] - 1)], np.nan)

    def ground_at(self, x, y):
        if self.E is not None:
            return _bilinear(self.ground, self.wx0, self.wy0, self.wres, x, y)
        return _bilinear(self.e, self.x0, self.y0, self.res, x, y)

    def wind(self, x, y, agl, u, v):
        x, y, agl = (np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(agl))
        H = np.asarray(HEIGHTS, np.float64)
        a = np.clip(agl, H[0], H[-1])
        k = np.clip(np.searchsorted(H, a) - 1, 0, len(H) - 2)
        t = (a - H[k]) / (H[k + 1] - H[k])
        E = _bilinear(self.E, self.wx0, self.wy0, self.wres, x, y)
        N = _bilinear(self.N, self.wx0, self.wy0, self.wres, x, y)
        idx = np.arange(len(x))
        F = lambda A: A[k, :, idx] * (1 - t)[:, None] + A[k + 1, :, idx] * t[:, None]
        R = np.asarray(u)[..., None] * F(E) + np.asarray(v)[..., None] * F(N)
        return (R[:, 0], R[:, 1], R[:, 2])

    def _div(self, F, res):
        du = np.full(F.shape[:2], np.nan)
        dv = np.full(F.shape[:2], np.nan)
        du[:, 1:-1] = (F[:, 2:, 0] - F[:, :-2, 0]) / (2 * res)
        dv[1:-1, :] = (F[:-2, :, 1] - F[2:, :, 1]) / (2 * res)
        return du + dv

    def convergence(self, x, y, u, v, ws, zi):
        from scipy import ndimage
        x, y = (np.atleast_1d(x).astype(np.float64), np.atleast_1d(y).astype(np.float64))
        if not hasattr(self, '_dE'):
            k = HEIGHTS.index(150)
            self._dE = self._div(self.E[k, :2].transpose(1, 2, 0), self.wres)
            self._dN = self._div(self.N[k, :2].transpose(1, 2, 0), self.wres)
            es = ndimage.gaussian_filter(self.e.astype(np.float64), SLOPE_SIGMA_M / self.res)
            gy, gx = np.gradient(es, self.res)
            gy = -gy
            sl = np.hypot(gx, gy)
            f = np.minimum(1.0, sl / SLOPE_FULL) / np.maximum(sl, 1e-06)
            self._dS = self._div(np.stack([gx * f, gy * f], axis=-1), self.res)
        dE = _bilinear(self._dE, self.wx0, self.wy0, self.wres, x, y)
        dN = _bilinear(self._dN, self.wx0, self.wy0, self.wres, x, y)
        dS = _bilinear(self._dS, self.x0, self.y0, self.res, x, y)
        u, v, ws, zi = (np.broadcast_to(np.asarray(a, np.float64), x.shape) for a in (u, v, ws, zi))
        wn = -(u * dE + v * dN) * CONV_DEPTH
        slope = -SLOPE_FLOW * np.maximum(ws, 0) * dS * np.minimum(SLOPE_DEPTH * np.maximum(zi, 0), SLOPE_DEPTH_MAX)
        return {'wn': wn, 'slope': slope}

    def layer(self, x, y, agl, u, v, n=None, lean=None, spread=None, shape=None, decay=None, lee=None):
        x, y, agl = (np.atleast_1d(x).astype(np.float64), np.atleast_1d(y).astype(np.float64), np.atleast_1d(agl).astype(np.float64))
        u, v = (np.broadcast_to(np.asarray(u, np.float64), x.shape), np.broadcast_to(np.asarray(v, np.float64), x.shape))
        sp = np.maximum(np.hypot(u, v), 1e-06)
        ux, uy = (u / sp, v / sp)
        h = np.maximum(agl, 0)
        lean = RIDGE_LEAN if lean is None else lean
        bx, by = (x + ux * h * lean, y + uy * h * lean)
        d = (RIDGE_SPREAD if spread is None else spread) * h
        G = lambda F: np.nanmean([_bilinear(F, self.x0, self.y0, self.res, bx + ox * d * ux - oy * d * uy, by + ox * d * uy + oy * d * ux) for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))], axis=0)
        gx, gy = (G(self.gx), G(self.gy))
        relief = _bilinear(self.relief, self.x0, self.y0, self.res, bx, by)
        s = np.hypot(gx, gy)
        w0 = u * gx + v * gy
        fr = np.clip((relief - RELIEF_LO) / (RELIEF_HI - RELIEF_LO), 0, 1)
        up = w0 > 0
        cap = np.where(up, MAX_SLOPE, LEE_SLOPE)
        k = np.where(s > cap, cap / np.maximum(s, 1e-09), 1.0)
        open_ = np.ones_like(s)
        if n is not None:
            nn = np.broadcast_to(np.asarray(n, np.float64), x.shape)
            blk = (nn > 0) & (relief > 0) & (s > 0) & up
            open_ = np.where(blk, np.clip(w0 / np.maximum(s, 1e-09) / np.maximum(nn * relief, 1e-09), BLOCK_MIN, 1), 1.0)
        L = (SHAPE if shape is None else shape) * np.maximum(relief, 0.001) * open_ / np.maximum(s * k, 1e-06)
        out = w0 * k * fr / (1 + h / L) ** (DECAY if decay is None else decay)
        out = np.where(out < 0, out * (LEE_SHARE if lee is None else lee), out)
        return np.where(np.isfinite(out) & (s > 0), out, np.nan)

def main(argv=None):
    a = (argv if argv is not None else sys.argv)[1:]
    lat, lon = (float(a[0]), float(a[1]))
    frm, spd = (float(a[2]) if len(a) > 2 else 270.0, float(a[3]) if len(a) > 3 else 10.0)
    work = tempfile.mkdtemp(prefix='tw')
    D = Domain(lat, lon, work)
    th = math.radians(frm)
    u, v = (-spd * math.sin(th), -spd * math.cos(th))
    ys, xs = np.mgrid[-6000:6001:1000, -6000:6001:1000]
    x, y = (D.cx + xs.ravel(), D.cy - ys.ravel())
    for h in (50, 150, 300):
        _, _, w = D.wind(x, y, np.full(x.shape, h), u, v)
        L = D.layer(x, y, np.full(x.shape, h), u, v)
        print(f"{h} m up: WindNinja w max {np.nanmax(w):.2f} min {np.nanmin(w):.2f}; the layer's max {np.nanmax(L):.2f} min {np.nanmin(L):.2f}")
if __name__ == '__main__':
    main()
