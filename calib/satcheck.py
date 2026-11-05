from __future__ import annotations
import argparse
import glob
import io
import json
import math
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))
EAST_SWITCH = '2025-04-07'
CHANNEL = 'C02'
UA = {'User-Agent': 'satcheck'}
SUN_MIN = 25.0
SKY_PCT = 15
RISE, BRIGHT = (0.08, 0.2)
MIDHIGH_MAX = 15.0
STREET_KM, PIX_KM = (24.0, 0.75)
COVER_RANGE = (0.08, 0.6)
ANISO = 0.35
SPACING_KM = (2.0, 12.0)
ALONG_DEG = 20.0
COVER_BINS = [0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.7, 1.01]
SPACING_BINS = [(0, 5), (5, 7), (7, 9), (9, 99)]
STRENGTH_BINS = [0, 0.3, 0.5, 0.7, 1.01]
FIELDS = ['wstar', 'cudepth', 'overdev', 'zi', 'cubase', 'depth', 'cloudlow', 'cloudmid', 'cloudhigh', 'rain', 'u_blavg', 'v_blavg']

def bucket(day: str) -> str:
    return 'noaa-goes19' if day >= EAST_SWITCH else 'noaa-goes16'

def _get(url: str, rng: tuple[int, int] | None=None, tries: int=4) -> bytes:
    h = dict(UA)
    if rng:
        h['Range'] = f'bytes={rng[0]}-{rng[1]}'
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
    return b''

def scan_url(t: datetime) -> str | None:
    b = bucket(t.strftime('%Y-%m-%d'))
    prefix = f'ABI-L1b-RadC/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/'
    xml = _get(f'https://{b}.s3.amazonaws.com/?list-type=2&prefix={prefix}')
    keys = [e.text for e in ET.fromstring(xml).iter() if e.tag.endswith('Key') and e.text and re.search(f'-M\\d{CHANNEL}_', e.text)]
    best, bd = (None, None)
    for k in keys:
        m = re.search('_s(\\d{4})(\\d{3})(\\d{2})(\\d{2})(\\d{2})', k)
        if not m:
            continue
        s = datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1, hours=int(m[3]), minutes=int(m[4]), seconds=int(m[5]))
        d = abs((s - t).total_seconds())
        if bd is None or d < bd:
            best, bd = (k, d)
    return f'https://{b}.s3.amazonaws.com/{best}' if best and bd <= 600 else None

class RangeFile(io.RawIOBase):
    BLOCK = 1 << 20

    def __init__(self, url: str):
        self.url, self.pos, self.cache = (url, 0, {})
        req = urllib.request.Request(url, headers=UA, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        b = self.cache.get(i)
        if b is None:
            lo = i * self.BLOCK
            b = self.cache[i] = _get(self.url, (lo, min(self.size, lo + self.BLOCK) - 1))
        return b

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = (bytearray(), self.pos)
        while len(out) < n:
            i = p // self.BLOCK
            blk = self._block(i)
            o = p - i * self.BLOCK
            take = blk[o:o + n - len(out)]
            out += take
            p += len(take)
        buf[:n] = out
        self.pos += n
        return n

def _proj(f):
    p = f['goes_imager_projection'].attrs
    g = lambda k: float(np.asarray(p[k]).ravel()[0])
    a, b = (g('semi_major_axis'), g('semi_minor_axis'))
    return (a, b, g('perspective_point_height') + a, math.radians(g('longitude_of_projection_origin')))

def latlon_to_xy(lat, lon, a, b, H, lon0):
    phi, lam = (np.radians(lat), np.radians(lon))
    e2 = (a * a - b * b) / (a * a)
    pc = np.arctan(b * b / (a * a) * np.tan(phi))
    rc = b / np.sqrt(1 - e2 * np.cos(pc) ** 2)
    sx = H - rc * np.cos(pc) * np.cos(lam - lon0)
    sy = -rc * np.cos(pc) * np.sin(lam - lon0)
    sz = rc * np.sin(pc)
    return (np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz)), np.arctan(sz / sx))

def xy_to_latlon(x, y, a, b, H, lon0):
    sx_, cx, sy_, cy = (np.sin(x), np.cos(x), np.sin(y), np.cos(y))
    A = sx_ ** 2 + cx ** 2 * (cy ** 2 + a * a / (b * b) * sy_ ** 2)
    B = -2 * H * cx * cy
    C = H * H - a * a
    with np.errstate(invalid='ignore'):
        rs = (-B - np.sqrt(B * B - 4 * A * C)) / (2 * A)
    sx, sy, sz = (rs * cx * cy, -rs * sx_, rs * cx * sy_)
    lat = np.degrees(np.arctan(a * a / (b * b) * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
    lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
    return (lat, lon)

def _num(v) -> float:
    return float(np.asarray(v).ravel()[0])

def read_box(url: str, bbox) -> dict:
    import h5py
    with h5py.File(RangeFile(url), 'r') as f:
        a, b, H, lon0 = _proj(f)
        X, Y = (f['x'], f['y'])
        xs = X[:] * _num(X.attrs['scale_factor']) + _num(X.attrs['add_offset'])
        ys = Y[:] * _num(Y.attrs['scale_factor']) + _num(Y.attrs['add_offset'])
        w, s, e, n = bbox
        lats = np.array([s, s, n, n, (s + n) / 2, (s + n) / 2, s, n])
        lons = np.array([w, e, w, e, w, e, (w + e) / 2, (w + e) / 2])
        px, py = latlon_to_xy(lats, lons, a, b, H, lon0)
        i0, i1 = (np.searchsorted(xs, px.min()) - 2, np.searchsorted(xs, px.max()) + 2)
        j0, j1 = (np.searchsorted(-ys, -py.max()) - 2, np.searchsorted(-ys, -py.min()) + 2)
        i0, j0 = (max(0, i0), max(0, j0))
        R = f['Rad']
        raw = R[j0:j1, i0:i1].astype(np.float32)
        fill = R.attrs.get('_FillValue')
        rad = raw * _num(R.attrs['scale_factor']) + _num(R.attrs['add_offset'])
        if fill is not None:
            rad[raw == _num(fill)] = np.nan
        k0 = _num(f['kappa0'][()])
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
        lat, lon = xy_to_latlon(gx, gy, a, b, H, lon0)
    return {'refl': rad * k0, 'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)}

def sun_cos(t: datetime, lat, lon):
    doy = t.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (t.hour - 12) / 24)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    eqt = 229.18 * (7.5e-05 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    mins = t.hour * 60 + t.minute + t.second / 60
    ha = np.radians((mins + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    return np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha)

def cloud_masks(refl: list, coss: list) -> list:
    R = np.stack([r / np.maximum(c, 0.05) for r, c in zip(refl, coss)])
    lit = np.stack([c >= math.sin(math.radians(SUN_MIN)) for c in coss]) & np.isfinite(R)
    Rl = np.where(lit, R, np.nan)
    with np.errstate(all='ignore'):
        ground = np.nanpercentile(Rl, SKY_PCT, axis=0)
    out = []
    for k in range(R.shape[0]):
        ok = lit[k] & np.isfinite(ground)
        cloud = ok & (R[k] - ground > RISE) & (R[k] > BRIGHT)
        out.append((cloud, ok))
    return out

def decode(m: dict, z, name: str):
    f = (m.get('fields') or {}).get(name)
    if f is None or name not in z.files:
        return None
    q = z[name].astype(np.float32)
    lo, hi = (f['min'], f['max'])
    if f.get('transform') == 'sqrt':
        a, b = (math.sqrt(max(lo, 0)), math.sqrt(hi))
        v = (a + q / 254 * (b - a)) ** 2
    else:
        v = lo + q / 254 * (hi - lo)
    v[q == 255] = np.nan
    return v

def forecast_hour(m: dict, fn: str) -> dict | None:
    import sky
    with np.load(fn) as z:
        v = {k: decode(m, z, k) for k in FIELDS}
    if any((v[k] is None for k in ('wstar', 'cloudlow', 'u_blavg', 'v_blavg'))):
        return None
    shape = v['wstar'].shape
    for k in FIELDS:
        if v[k] is None:
            v[k] = np.full(shape, np.nan, np.float32)
    d, od, low = (v['cudepth'], v['overdev'], v['cloudlow'])
    reach = sky.cu_reach(v['zi'], v['cubase'])
    cover = np.minimum(sky.COVER_MAX, np.maximum(np.where(np.isfinite(d), sky.cu_cover(d, np.nan_to_num(od, nan=0.0), reach), 0.0), np.nan_to_num(low, nan=0.0) / 100))
    cover = np.where(np.isfinite(d) | np.isfinite(low), cover, 0.0)
    w = v['wstar']
    clear_above = (np.nan_to_num(v['cloudmid']) < MIDHIGH_MAX) & (np.nan_to_num(v['cloudhigh']) < MIDHIGH_MAX) & ~(np.nan_to_num(v['rain']) >= 0.1)
    return {'cover': cover, 'streets': None, 'clear_above': clear_above, 'u': v['u_blavg'], 'v': v['v_blavg'], 'zi': v['zi'], 'wstar': w, 'cu': np.isfinite(d) & (np.nan_to_num(d) > 0)}

def regular_patch(cloud, ok, lat, lon, clat, clon, half_km=STREET_KM / 2, px=PIX_KM):
    kx = 111.32 * math.cos(math.radians(clat))
    X = (lon - clon) * kx
    Y = (clat - lat) * 111.32
    n = int(round(2 * half_km / px))
    ix = np.floor((X + half_km) / px).astype(int)
    iy = np.floor((Y + half_km) / px).astype(int)
    sel = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    g = np.full((n, n), np.nan, np.float32)
    g[iy[sel], ix[sel]] = np.where(ok[sel], cloud[sel].astype(np.float32), np.nan)
    return g

def street_stats(g: np.ndarray, px=PIX_KM) -> dict | None:
    ok = np.isfinite(g)
    if ok.mean() < 0.8:
        return None
    cover = float(np.nanmean(g))
    if not COVER_RANGE[0] <= cover <= COVER_RANGE[1]:
        return {'cover': cover}
    a = np.where(ok, g - cover, 0.0)
    n = a.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2
    k = np.fft.fftshift(np.fft.fftfreq(n, d=px))
    KX, KY = np.meshgrid(k, -k)
    kr = np.hypot(KX, KY)
    band = (kr >= 1 / SPACING_KM[1]) & (kr <= 1 / SPACING_KM[0])
    if not band.any() or P[band].sum() <= 0:
        return {'cover': cover}
    w = P * band
    ux, uy = (np.where(kr > 0, KX / np.maximum(kr, 1e-09), 0), np.where(kr > 0, KY / np.maximum(kr, 1e-09), 0))
    Sxx, Syy, Sxy = ((w * ux * ux).sum(), (w * uy * uy).sum(), (w * ux * uy).sum())
    tr = Sxx + Syy
    lam = math.sqrt(max(0.0, ((Sxx - Syy) / 2) ** 2 + Sxy ** 2))
    aniso = 2 * lam / tr if tr > 0 else 0.0
    kang = 0.5 * math.atan2(2 * Sxy, Sxx - Syy)
    axis = (math.degrees(math.pi / 2 - kang) + 90) % 180
    ang = np.arctan2(KY, KX)
    dang = np.abs((ang - kang + math.pi / 2) % math.pi - math.pi / 2)
    wedge = band & (dang <= math.radians(20))
    spacing = None
    if wedge.any():
        rs = kr[wedge]
        edges = np.linspace(1 / SPACING_KM[1], 1 / SPACING_KM[0], 12)
        idx = np.digitize(rs, edges)
        tot = np.bincount(idx, weights=P[wedge], minlength=len(edges) + 1)
        j = int(np.argmax(tot[1:len(edges)])) + 1
        spacing = 1 / ((edges[j - 1] + edges[min(j, len(edges) - 1)]) / 2)
    return {'cover': cover, 'aniso': round(aniso, 3), 'axis': round(axis, 1), 'spacing': round(spacing, 2) if spacing else None}

def wind_axis(u: float, v: float) -> float:
    return math.degrees(math.atan2(u, v)) % 180

def axis_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)
CHUNK_Z = 10
PLACED_GRID = 3
FSS_KM = (2.25, 6.0, 12.0)

def node_script() -> str | None:
    return next(iter(sorted(glob.glob(os.path.join(HERE, 'cuplace_eval.*')))), None)
_ENV_FIELDS = None

def env_fields() -> list:
    global _ENV_FIELDS
    if _ENV_FIELDS is None:
        import subprocess
        r = subprocess.run(['node', node_script(), 'fields'], capture_output=True, text=True, timeout=120)
        if r.returncode:
            raise RuntimeError(f'cuplace_eval fields: {r.stderr[-300:]}')
        _ENV_FIELDS = json.loads(r.stdout)
    return _ENV_FIELDS

def tile_at(lat: float, lon: float) -> tuple[int, int]:
    n, s = (2 ** CHUNK_Z, math.sin(math.radians(lat)))
    return (int((lon + 180) / 360 * n), int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n))

def tile_centre(X: int, Y: int) -> tuple[float, float, float]:
    n = 2 ** CHUNK_Z
    lon = (X + 0.5) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (Y + 0.5) / n))))
    return (lat, lon, math.pi * 6371008.8 * math.cos(math.radians(lat)) / n / 1000)

def box_tiles(bbox) -> list:
    w, s, e, n = bbox
    out = []
    for j in range(PLACED_GRID):
        for i in range(PLACED_GRID):
            la = s + 0.6 + (n - s - 1.2) * (j + 0.5) / PLACED_GRID
            lo = w + 0.6 + (e - w - 1.2) * (i + 0.5) / PLACED_GRID
            t = tile_at(la, lo)
            if t not in out:
                out.append(t)
    return out

def fss(f: np.ndarray, o: np.ndarray, k: int) -> float | None:

    def frac(a):
        c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    if f.shape[0] <= k:
        return None
    pf, po = (frac(f.astype(np.float64)), frac(o.astype(np.float64)))
    ref = (pf ** 2).mean() + (po ** 2).mean()
    return None if ref <= 0 else float(1 - ((pf - po) ** 2).mean() / ref)

def write_env(gdir: str, m: dict, hours: list, work: str) -> list | None:
    meta = json.load(open(os.path.join(gdir, 'meta.json')))
    names, planes = ([], [])
    zs = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(gdir, '_hours', '*.npz'))}
    stamps = [f'{t:%Y%m%d%H}' for t, *_ in hours if f'{t:%Y%m%d%H}' in zs]
    if not stamps:
        return None
    zl = [np.load(zs[st]) for st in stamps]
    for name in env_fields():
        vs = [decode(m, z, name) for z in zl]
        if all((v is None for v in vs)):
            continue
        names.append(name)
        planes.append(vs)
    ny, nx = (m['grid']['ny'], m['grid']['nx'])
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, 'env.bin'), 'wb') as f:
        for k in range(len(stamps)):
            for vs in planes:
                v = vs[k]
                f.write((v if v is not None else np.full((ny, nx), np.nan, np.float32)).astype(np.float32).tobytes())
    tiles = box_tiles([float(x) for x in meta['bbox']])
    with open(os.path.join(work, 'env.json'), 'w') as f:
        json.dump({'grid': m['grid'], 'stamps': stamps, 'fields': names, 'tiles': tiles, 'px': PIX_KM}, f)
    return stamps

def run_cells(work: str, log=print) -> dict | None:
    import subprocess
    out = os.path.join(work, 'cells.bin')
    r = subprocess.run(['node', node_script(), 'cells', work, out], capture_output=True, text=True, timeout=1800)
    if r.returncode or not os.path.exists(out):
        log(f'  the roll model failed: {r.stderr[-300:]}')
        return None
    with open(out, 'rb') as f:
        blob = f.read()
    nl = blob.index(b'\n')
    head = json.loads(blob[:nl])
    a = np.frombuffer(blob[nl + 1:], np.float32).reshape(len(head['stamps']), len(head['planes']), head['ny'], head['nx'])
    return {st: {p: a[k, i] for i, p in enumerate(head['planes'])} for k, st in enumerate(head['stamps'])}

def run_placed(work: str, log=print) -> dict | None:
    import subprocess
    out = os.path.join(work, 'placed.bin')
    r = subprocess.run(['node', node_script(), work, out], capture_output=True, text=True, timeout=1800)
    if r.returncode or not os.path.exists(out):
        log(f'  placing failed: {r.stderr[-300:]}')
        return None
    with open(out, 'rb') as f:
        blob = f.read()
    nl = blob.index(b'\n')
    head = json.loads(blob[:nl])
    res, off = ({}, nl + 1)
    for t in head['tiles']:
        sz = t['n'] * t['n']
        res[t['stamp'], t['X'], t['Y']] = (t, np.frombuffer(blob[off:off + sz], np.uint8).reshape(t['n'], t['n']))
        off += sz
    return res

def placed_rows(placed: dict, hours: list, masks: list, lat, lon, m: dict) -> list:
    g = m['grid']
    rows = []
    for (t, fc, _), (cloud, ok) in zip(hours, masks):
        st = f'{t:%Y%m%d%H}'
        for (s_, X, Y), (meta, a) in placed.items():
            if s_ != st:
                continue
            la, lo, half = (meta['lat'], meta['lon'], meta['half'])
            j, i = (int(round((g['north'] - la) / g['step'])), int(round((lo - g['west']) / g['step'])))
            if not (0 <= j < g['ny'] and 0 <= i < g['nx']) or not fc['clear_above'][j, i]:
                continue
            dlat, dlon = (half / 111.32 * 1.2, half / (111.32 * math.cos(math.radians(la))) * 1.2)
            near = (np.abs(lat - la) < dlat) & (np.abs(lon - lo) < dlon)
            sat = regular_patch(cloud[near], ok[near], lat[near], lon[near], la, lo, half_km=half, px=meta['px'])
            if sat.shape != a.shape or np.isfinite(sat).mean() < 0.8:
                continue
            have = np.isfinite(sat)
            o, f = (np.where(have, sat, 0).astype(np.uint8), np.where(have, a, 0))
            row = {'h': f'{t:%H}', 'sat': round(float(o[have].mean()), 3), 'placed': round(float(f[have].mean()), 3), 'fc_cover': round(float(fc['cover'][j, i]), 3), 'fc_streets': cell_streets(fc, j, i), 'clouds': meta['clouds']}
            if meta.get('alone') is not None:
                row['placed_alone'] = meta['alone']
            ts = meta.get('streets')
            if ts:
                row['t_st'], row['t_sp'], row['t_ax'] = (ts['strength'], round(ts['spacing'] / 1000, 2), round(ts['angle'] % 180, 1))
                row['t_streets'] = ts.get('kind') in ('streets', 'bluestreets')
            for km in FSS_KM:
                v = fss(f, o, max(1, int(round(km / meta['px']))))
                row[f'fss{km:g}'] = None if v is None else round(v, 3)
            c = a.shape[0] // 2
            w = int(round(STREET_KM / 2 / meta['px']))
            if c - w >= 0:
                ss, sp = (street_stats(np.where(have, sat, np.nan)[c - w:c + w, c - w:c + w], meta['px']), street_stats(f[c - w:c + w, c - w:c + w].astype(np.float32), meta['px']))
                u, v_ = (float(fc['u'][j, i]), float(fc['v'][j, i]))
                row['wind_axis'] = round(wind_axis(u, v_), 1)
                row['wind'] = round(math.hypot(u, v_), 1)
                for k, s2 in (('s', ss), ('p', sp)):
                    if s2 and 'aniso' in s2:
                        row[f'{k}_aniso'], row[f'{k}_axis'], row[f'{k}_spacing'] = (s2['aniso'], s2['axis'], s2['spacing'])
            rows.append(row)
    return rows

def cell_streets(fc: dict, j: int, i: int):
    st = fc.get('streets')
    return None if st is None else bool(st[j, i] == 1)

def cell_rolls(fc: dict, j: int, i: int) -> dict:
    if fc.get('strength') is None or not np.isfinite(fc['strength'][j, i]):
        return {'st': None, 'st_sp': None, 'st_ax': None}
    return {'st': round(float(fc['strength'][j, i]), 3), 'st_sp': round(float(fc['spacing'][j, i]) / 1000, 2), 'st_ax': round(float(fc['angle'][j, i]) % 180, 1)}

def box_day(gdir: str, log=print, work: str | None=None) -> dict | None:
    meta = json.load(open(os.path.join(gdir, 'meta.json')))
    m = json.load(open(os.path.join(gdir, 'latest.json')))
    bbox = [float(x) for x in meta['bbox']]
    if bbox[0] > -50:
        return None
    g = m['grid']
    west, north, step, nx, ny = (g['west'], g['north'], g['step'], g['nx'], g['ny'])
    hours = []
    for fn in sorted(glob.glob(os.path.join(gdir, '_hours', '*.npz'))):
        st = os.path.basename(fn)[:-4]
        t = datetime.strptime(st, '%Y%m%d%H').replace(tzinfo=timezone.utc)
        fc = forecast_hour(m, fn)
        if fc is None:
            continue
        url = scan_url(t)
        if not url:
            log(f'  {st}: no scan')
            continue
        try:
            px = read_box(url, bbox)
        except Exception as e:
            log(f'  {st}: scan not read ({type(e).__name__})')
            continue
        hours.append((t, fc, px))
    if len(hours) < 4:
        return None
    shp = hours[0][2]['refl'].shape
    hours = [h for h in hours if h[2]['refl'].shape == shp]
    masks = cloud_masks([h[2]['refl'] for h in hours], [sun_cos(h[0], h[2]['lat'], h[2]['lon']) for h in hours])
    lat, lon = (hours[0][2]['lat'], hours[0][2]['lon'])
    ix = np.round((lon - west) / step).astype(int)
    iy = np.round((north - lat) / step).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(lat)
    cell = np.where(inside, iy * nx + ix, -1)
    nb = len(COVER_BINS) - 1
    out = {'model': meta.get('model'), 'day': meta.get('day'), 'tz': meta.get('tz'), 'region': meta.get('region'), 'bins': {'fc': [0.0] * nb, 'obs': [0.0] * nb, 'n': [0] * nb}, 'by_hour': {}, 'blocks': [], 'cells': 0, 'placed': []}
    if work is None:
        import tempfile
        work = tempfile.mkdtemp(prefix='satcheck')
    try:
        if write_env(gdir, m, hours, work):
            cells = run_cells(work, log)
            for t, fc, _ in hours:
                c = (cells or {}).get(f'{t:%Y%m%d%H}')
                if c:
                    fc.update(strength=c['strength'], spacing=c['spacing'], angle=c['angle'], streets=c['streets'])
            placed = run_placed(work, log)
            if placed:
                out['placed'] = placed_rows(placed, hours, masks, lat, lon, m)
    except Exception as e:
        log(f"  the model's streets or placed cumulus not compared: {type(e).__name__}: {e}")
    for (t, fc, _), (cloud, ok) in zip(hours, masks):
        use = inside & ok
        cnt = np.bincount(cell[use], minlength=nx * ny).astype(np.float64)
        cld = np.bincount(cell[use], weights=cloud[use].astype(np.float64), minlength=nx * ny)
        have = cnt >= 20
        obs = np.where(have, cld / np.maximum(cnt, 1), np.nan).reshape(ny, nx)
        sel = np.isfinite(obs) & fc['clear_above']
        f, o = (fc['cover'][sel], obs[sel])
        b = np.clip(np.digitize(f, COVER_BINS) - 1, 0, nb - 1)
        for k in range(nb):
            mk = b == k
            out['bins']['fc'][k] += float(f[mk].sum())
            out['bins']['obs'][k] += float(o[mk].sum())
            out['bins']['n'][k] += int(mk.sum())
        hr = f'{t:%H}'
        hb = out['by_hour'].setdefault(hr, [0.0, 0.0, 0])
        hb[0] += float(f.sum())
        hb[1] += float(o.sum())
        hb[2] += int(f.size)
        out['cells'] += int(f.size)
        clat0, clat1 = (bbox[1] + 0.3, bbox[3] - 0.3)
        dlat = STREET_KM / 111.32
        la = clat0
        while la <= clat1:
            dlon = STREET_KM / (111.32 * math.cos(math.radians(la)))
            lo_ = bbox[0] + 0.3
            while lo_ <= bbox[2] - 0.3:
                j, i = (int(round((north - la) / step)), int(round((lo_ - west) / step)))
                if 0 <= j < ny and 0 <= i < nx and fc['clear_above'][j, i] and np.isfinite(fc['u'][j, i]):
                    near = (np.abs(lat - la) < dlat * 0.6) & (np.abs(lon - lo_) < dlon * 0.6)
                    if near.any():
                        gpatch = regular_patch(cloud[near], ok[near], lat[near], lon[near], la, lo_)
                        s = street_stats(gpatch)
                        if s and 'aniso' in s:
                            u, v = (float(fc['u'][j, i]), float(fc['v'][j, i]))
                            out['blocks'].append({'h': hr, 'aniso': s['aniso'], 'axis': s['axis'], 'spacing': s['spacing'], 'cover': round(s['cover'], 3), 'fc_streets': cell_streets(fc, j, i), **cell_rolls(fc, j, i), 'fc_cover': round(float(fc['cover'][j, i]), 3), 'wind': round(math.hypot(u, v), 1), 'wind_axis': round(wind_axis(u, v), 1), 'zi': None if not np.isfinite(fc['zi'][j, i]) else round(float(fc['zi'][j, i])), 'w': None if not np.isfinite(fc['wstar'][j, i]) else round(float(fc['wstar'][j, i]), 2)})
                lo_ += dlon
            la += dlat
    return out

def report(results: list[dict]) -> str:
    L = ["# Satellite check: forecast cumulus against GOES-East's visible picture", '']
    by_model = {}
    for r in results:
        by_model.setdefault(r['model'], []).append(r)
    L.append(f"{len(results)} box-days ({', '.join((f'{m}: {len(v)}' for m, v in sorted(by_model.items())))}).")
    L.append(f"Cloud: a 0.5 km pixel {RISE} over the day's clear ground ({SKY_PCT}th percentile) and at least {BRIGHT} bright, the sun {SUN_MIN}° up; cells with forecast mid or high cloud ≥ {MIDHIGH_MAX:g} % or rain left out.")
    for model, rs in sorted(by_model.items()):
        nb = len(COVER_BINS) - 1
        F, O, N = ([0.0] * nb, [0.0] * nb, [0] * nb)
        hours = {}
        for r in rs:
            for k in range(nb):
                F[k] += r['bins']['fc'][k]
                O[k] += r['bins']['obs'][k]
                N[k] += r['bins']['n'][k]
            for h, (f, o, n) in r['by_hour'].items():
                a = hours.setdefault(h, [0.0, 0.0, 0])
                a[0] += f
                a[1] += o
                a[2] += n
        L += ['', f'## {model}: cumulus cover', '', '| forecast cover | cells | forecast mean | satellite mean | ratio |', '|---|---|---|---|---|']
        for k in range(nb):
            if N[k]:
                fm, om = (F[k] / N[k], O[k] / N[k])
                L.append(f'| {COVER_BINS[k]:.2f}–{min(1, COVER_BINS[k + 1]):.2f} | {N[k]} | {fm:.3f} | {om:.3f} | {om / fm:.2f} |' if fm > 0.005 else f'| {COVER_BINS[k]:.2f}–{min(1, COVER_BINS[k + 1]):.2f} | {N[k]} | {fm:.3f} | {om:.3f} | – |')
        tn = sum(N)
        if tn:
            L.append(f'\nAll: forecast {sum(F) / tn:.3f}, satellite {sum(O) / tn:.3f} ({tn} cell-hours).')
        L += ['', '| hour (UTC) | forecast | satellite | cell-hours |', '|---|---|---|---|']
        for h in sorted(hours):
            f, o, n = hours[h]
            if n:
                L.append(f'| {h} | {f / n:.3f} | {o / n:.3f} | {n} |')
        L += streets_section(model, rs)
        L += placed_section(model, [p for r in rs for p in r.get('placed') or []])
    return '\n'.join(L) + '\n'

def is_line(b: dict, along: bool=False, k: str='') -> bool:
    a = b.get(f'{k}aniso')
    if a is None or a < ANISO:
        return False
    return not along or axis_diff(b[f'{k}axis'], b['wind_axis']) <= ALONG_DEG

def roc_auc(score, label) -> float | None:
    s, y = (np.asarray(score, float), np.asarray(label, bool))
    n1, n0 = (int(y.sum()), int((~y).sum()))
    if not n1 or not n0:
        return None
    order = np.argsort(s, kind='mergesort')
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        mk = s == v
        if mk.sum() > 1:
            r[mk] = r[mk].mean()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def two_by_two(B: list, along: bool) -> tuple:
    tp = sum((1 for b in B if b['fc_streets'] and is_line(b, along)))
    fp = sum((1 for b in B if b['fc_streets'] and (not is_line(b, along))))
    fn = sum((1 for b in B if not b['fc_streets'] and is_line(b, along)))
    return (tp, fp, fn, len(B) - tp - fp - fn)

def streets_section(model: str, rs: list) -> list:
    B = [b for r in rs for b in r['blocks'] if b.get('fc_streets') is not None]
    if not B:
        return []
    L = ['', f'## {model}: streets', '', f"""{len(B)} blocks of {STREET_KM:g} km with {COVER_RANGE[0]:.0%}–{COVER_RANGE[1]:.0%} cumulus. Lines: the spectrum leaning ≥ {ANISO} ("lean"), and the stricter "lean along": that with the lines within {ALONG_DEG:g}° of the layer's wind. The forecast's streets: the site's rule (thermalKind: the rolls' strength ≥ ROLLS.on, W* ≥ 1, no rain, overdevelopment or overcast)."""]
    for along, name in ((False, 'lean'), (True, 'lean along')):
        tp, fp, fn, tn = two_by_two(B, along)
        L += ['', f'| {name} | satellite: lines | satellite: no lines |', '|---|---|---|', f'| forecast: streets | {tp} | {fp} |', f'| forecast: no streets | {fn} | {tn} |', '', f"Caught {tp / max(1, tp + fn):.0%} of the satellite's lines; {fp / max(1, tp + fp):.0%} of the forecast's street blocks showed none."]
    S = [b for b in B if b.get('st') is not None]
    if S:
        st = np.array([b['st'] for b in S])
        L += ['', "By the rolls' strength (the site's roll model, the cell's):", '', '| strength | blocks | lines (lean) | lines (lean along) | mean lean | forecast streets |', '|---|---|---|---|---|---|']
        for k in range(len(STRENGTH_BINS) - 1):
            g = [b for b, v in zip(S, st) if STRENGTH_BINS[k] <= v < STRENGTH_BINS[k + 1]]
            if g:
                L.append(f"| {STRENGTH_BINS[k]:.1f}–{min(1, STRENGTH_BINS[k + 1]):.1f} | {len(g)} | {np.mean([is_line(b) for b in g]):.0%} | {np.mean([is_line(b, True) for b in g]):.0%} | {np.mean([b['aniso'] for b in g]):.3f} | {np.mean([bool(b['fc_streets']) for b in g]):.0%} |")
        F = [b for b in S if b['fc_streets'] and b.get('st_sp')]
        if F:
            L += ['', "The forecast's street blocks by the model's spacing (aspect × zi): wide rows read as lines less often in a 24 km block.", '', '| spacing | blocks | lines (lean) | lines (lean along) |', '|---|---|---|---|']
            for lo, hi in SPACING_BINS:
                g = [b for b in F if lo <= b['st_sp'] < hi]
                if g:
                    L.append(f'| {lo:g}–{hi:g} km | {len(g)} | {np.mean([is_line(b) for b in g]):.0%} | {np.mean([is_line(b, True) for b in g]):.0%} |')
        a1, a2 = (roc_auc(st, [is_line(b) for b in S]), roc_auc(st, [is_line(b, True) for b in S]))
        rho = float(np.corrcoef(np.argsort(np.argsort(st)), np.argsort(np.argsort([b['aniso'] for b in S])))[0, 1]) if len(S) > 2 else None
        L.append("\nThe strength's ROC AUC: " + ', '.join((f'{n} {a:.2f}' for n, a in (('lean', a1), ('lean along', a2)) if a is not None)) + (f'; its rank correlation with the lean {rho:.2f}.' if rho is not None else '.'))
    lines = [b for b in B if is_line(b)]
    al = [axis_diff(b['axis'], b['wind_axis']) for b in lines if b['wind'] >= 3]
    if al:
        L.append(f"Lines (lean) against the layer's wind (≥ 3 m/s): median {np.median(al):.0f}° apart, {np.mean(np.array(al) <= 20):.0%} within 20°.")
    am = [axis_diff(b['axis'], b['st_ax']) for b in lines if b.get('st_ax') is not None and b['wind'] >= 3]
    if am:
        L.append(f"… against the model's rows (rollParams' angle: the wind turned toward the shear): median {np.median(am):.0f}° apart, {np.mean(np.array(am) <= 20):.0%} within 20°.")
    along = [b for b in lines if axis_diff(b['axis'], b['wind_axis']) <= 30 and b['spacing']]
    sp = [b['spacing'] / (b['zi'] / 1000) for b in along if b['zi'] and b['zi'] > 300]
    if sp:
        L.append(f'Spacing over zi where they lie along the wind: median {np.median(sp):.2f}, quartiles {np.percentile(sp, 25):.2f}–{np.percentile(sp, 75):.2f}, {len(sp)} blocks.')
    sm = [b['spacing'] / b['st_sp'] for b in along if b.get('st_sp')]
    if sm:
        L.append(f"Spacing over the model's (aspect × zi): median {np.median(sm):.2f}, quartiles {np.percentile(sm, 25):.2f}–{np.percentile(sm, 75):.2f}.")
    wl, wn = ([b['wind'] for b in lines], [b['wind'] for b in B if not is_line(b)])
    if wl and wn:
        L.append(f'Layer wind: {np.median(wl):.1f} m/s median under lines, {np.median(wn):.1f} without.')
    L += ['', '| day | blocks | lines (lean) | lines (lean along) | forecast streets | mean strength | hits / false alarms (lean along) | AUC (lean along) |', '|---|---|---|---|---|---|---|---|']
    for day in sorted({r.get('day') for r in rs}, key=str):
        D = [b for r in rs if r.get('day') == day for b in r['blocks'] if b.get('fc_streets') is not None]
        if not D:
            continue
        tp, fp, fn, _ = two_by_two(D, True)
        Ds = [b for b in D if b.get('st') is not None]
        a = roc_auc([b['st'] for b in Ds], [is_line(b, True) for b in Ds]) if Ds else None
        L.append(f"| {day} | {len(D)} | {np.mean([is_line(b) for b in D]):.0%} | {np.mean([is_line(b, True) for b in D]):.0%} | {np.mean([bool(b['fc_streets']) for b in D]):.0%} | {(np.mean([b['st'] for b in Ds]) if Ds else float('nan')):.2f} | {tp} / {fp} (missed {fn}) | {('–' if a is None else f'{a:.2f}')} |")
    return L

def placed_section(model: str, P: list) -> list:
    if not P:
        return []
    sat, pl, fc = (np.array([p[k] for p in P]) for k in ('sat', 'placed', 'fc_cover'))
    L = ['', f'## {model}: the placed cumulus (as the map draws them zoomed in)', '', f"{len(P)} tile-hours ({PLACED_GRID}x{PLACED_GRID} tiles a box, ~30 km; no ground cover: the site's comes from imagery).", '', f"Cover: placed {pl.mean():.3f}, satellite {sat.mean():.3f}, the forecast's cover at the tile's centre {fc.mean():.3f}" + (f" (placed without the next tiles' clouds drifting in: {np.mean([p['placed_alone'] for p in P if 'placed_alone' in p]):.3f})." if any(('placed_alone' in p for p in P)) else '.')]
    if len(P) > 2 and sat.std() > 0 and (pl.std() > 0):
        L.append(f'Tile by tile: correlation {np.corrcoef(pl, sat)[0, 1]:.2f}; mean error {np.abs(pl - sat).mean():.3f}.')
    L += ['', '| scale | FSS mean | useful (over 0.5 + share/2) |', '|---|---|---|']
    for km in FSS_KM:
        v = [(p[f'fss{km:g}'], p['sat']) for p in P if p.get(f'fss{km:g}') is not None]
        if v:
            L.append(f'| {km:g} km | {np.mean([x for x, _ in v]):.2f} | {np.mean([x >= 0.5 + o / 2 for x, o in v]):.0%} |')
    S = [p for p in P if p.get('s_aniso') is not None]
    if S:
        lines = [p for p in S if p['s_aniso'] >= ANISO]
        both = [p for p in lines if (p.get('p_aniso') or 0) >= ANISO]
        fake = [p for p in S if p['s_aniso'] < ANISO and (p.get('p_aniso') or 0) >= ANISO]
        L += ['', f"Streets (the tile's centre {STREET_KM:g} km): the satellite's lines in {len(lines)} of {len(S)}; the placed clouds lined up in {len(both)} of those, and in {len(fake)} where the satellite had none."]
        ad = [axis_diff(p['p_axis'], p['s_axis']) for p in both if p.get('p_axis') is not None]
        if ad:
            L.append(f"Their axes {np.median(ad):.0f}° apart (median); the satellite's against the wind {np.median([axis_diff(p['s_axis'], p['wind_axis']) for p in both]):.0f}°.")
        sr = [p['p_spacing'] / p['s_spacing'] for p in both if p.get('p_spacing') and p.get('s_spacing')]
        if sr:
            L.append(f'Spacing placed over seen: median {np.median(sr):.2f} (quartiles {np.percentile(sr, 25):.2f}–{np.percentile(sr, 75):.2f}).')
        al = [p for p in S if is_line(p, True, 's_')]
        bo = [p for p in al if is_line(p, True, 'p_')]
        fa = [p for p in S if not is_line(p, True, 's_') and is_line(p, True, 'p_')]
        L.append(f"Lean along the wind (within {ALONG_DEG:g}°): the satellite's lines in {len(al)}; placed lined up in {len(bo)} of those, in {len(fa)} where the satellite had none.")
        T = [p for p in S if p.get('t_st') is not None]
        if T:
            ta = roc_auc([p['t_st'] for p in T], [is_line(p, True, 's_') for p in T])
            L.append(f"The rolls at the tiles' centres (after their ground, tileRolls): strength {np.mean([p['t_st'] for p in T]):.2f} mean, {np.mean([bool(p.get('t_streets')) for p in T]):.0%} street kind" + ('' if ta is None else f"; its AUC for the satellite's lines (lean along) {ta:.2f}") + '.')
            ma = [axis_diff(p['s_axis'], p['t_ax']) for p in T if is_line(p, False, 's_')]
            if ma:
                L.append(f"The satellite's lines against the model's rows: median {np.median(ma):.0f}° apart (against the wind {np.median([axis_diff(p['s_axis'], p['wind_axis']) for p in T if is_line(p, False, 's_')]):.0f}°).")
            ms = [p['s_spacing'] / p['t_sp'] for p in T if is_line(p, False, 's_') and p.get('s_spacing') and p.get('t_sp')]
            if ms:
                L.append(f"Their spacing over the model's (aspect × zi): median {np.median(ms):.2f} (quartiles {np.percentile(ms, 25):.2f}–{np.percentile(ms, 75):.2f}).")
    return L

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--grids', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--budget', type=float, default=300, help='minutes')
    r.add_argument('--done', help='results of a run before (results.jsonl), its box-days skipped')
    p = sub.add_parser('report')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, 'results.jsonl')
    if a.cmd == 'run':
        done = set()
        if a.done and os.path.exists(a.done):
            same = os.path.abspath(a.done) == os.path.abspath(res_path)
            with open(a.done) as f:
                lines = f.readlines()
            done = {json.loads(line)['key'] for line in lines if line.strip()}
            if not same:
                with open(res_path, 'a') as w:
                    w.writelines(lines)
        t0 = time.time()
        boxes = sorted(glob.glob(os.path.join(a.grids, '*', '*-box*')))
        n = 0
        with open(res_path, 'a') as w, open(os.path.join(a.out, 'log.txt'), 'a') as logf:
            log = lambda s: logf.write(s + '\n')
            for gdir in boxes:
                key = os.path.relpath(gdir, a.grids)
                if key in done or '@' in key:
                    continue
                if (time.time() - t0) / 60 > a.budget:
                    break
                log(key)
                try:
                    rec = box_day(gdir, log, work=os.path.join(a.out, '_work'))
                except Exception as e:
                    log(f'  failed: {type(e).__name__}: {e}')
                    continue
                if rec:
                    rec['key'] = key
                    w.write(json.dumps(rec) + '\n')
                    w.flush()
                    n += 1
        print(f'satellite check: {n} box-days this run')
    results = []
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = [json.loads(line) for line in f]
    with open(os.path.join(a.out, 'report.md'), 'w') as f:
        f.write(report(results))
    print(f'satellite check: report over {len(results)} box-days')
if __name__ == '__main__':
    main()
