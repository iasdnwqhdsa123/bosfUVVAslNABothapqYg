from __future__ import annotations
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import satcheck
import satread
R_KM = 5.0
CELL_DEG = 0.1
GEO_WIN_MIN, POLAR_WIN_MIN = (10.0, 30.0)
GEO_TRIES = 3
GEO_ZEN_MAX = 70.0
TOP_MAX_KM = 16.0
PAD_KM = R_KM + TOP_MAX_KM * math.tan(math.radians(GEO_ZEN_MAX))
LOW_TOP_M = 2500.0
CU_TOP_AGL = 3500.0
CU_HIDDEN = 0.3
LOOKUP_KM = 12.0
EARTH_KM = 6371.0
J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
JRR_DAYS = 90
JRR_SAFE_DAYS = 85
DISK_S = 600.0
DISK_SHORT_S = 300.0
ACHA_FROM = '2019-12-02'
WEST_SWITCH = '2023-01-04'
GEO = [('goes16', 'noaa-goes16', -75.2, '2017-12-18', satcheck.EAST_SWITCH), ('goes19', 'noaa-goes19', -75.0, satcheck.EAST_SWITCH, '9999'), ('goes17', 'noaa-goes17', -137.2, '2019-02-12', WEST_SWITCH), ('goes18', 'noaa-goes18', -137.0, WEST_SWITCH, '9999'), ('himawari8', 'noaa-himawari8', 140.7, '2015-07-07', satread.HIMAWARI_SWITCH), ('himawari9', 'noaa-himawari9', 140.7, satread.HIMAWARI_SWITCH, '9999')]
POLAR = [('n20', 'noaa-nesdis-n20-pds', 'j01'), ('n21', 'noaa-nesdis-n21-pds', 'j02')]
SECTOR_C = {'east': ((-0.101332, 0.038612), (0.044268, 0.128212)), 'west': ((-0.07, 0.07), (0.04424, 0.12824))}
ABI_CODE = {'mask': 'ACM', 'phase': 'ACTP', 'height': 'ACHA'}
ENT_CODE = {'mask': 'CMSK', 'phase': 'CPHS', 'height': 'CHGT'}
JRR_DIR = {'mask': 'CloudMask', 'phase': 'CloudPhase', 'height': 'CloudHeight', 'base': 'CloudBase'}
PRODUCTS = {'abi': ('mask', 'phase', 'height'), 'ahi': ('mask', 'phase', 'height'), 'jrr': ('mask', 'phase', 'height', 'base'), 'laads': ('mask', 'phase', 'height')}
LAADS = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData'
LAADS_HOST = 'ladsweb.modaps.eosdis.nasa.gov'
LAADS_PRODUCT = {'mask': 'CLDMSK_L2_VIIRS_NOAA20', 'phase': 'CLDPROP_L2_VIIRS_NOAA20', 'height': 'CLDPROP_L2_VIIRS_NOAA20'}
LAADS_SET = {'CLDMSK_L2_VIIRS_NOAA20': 5110, 'CLDPROP_L2_VIIRS_NOAA20': 5111}
LAADS_TOKEN_ENV = 'EARTHDATA_TOKEN'
ABI_HEIGHT_GOOD = (0, 1, 4)
HEIGHT_GOOD = (0, 1)
ABI_VARS = {'mask': ('ACM', 'DQF', 'zero'), 'phase': ('Phase', 'DQF', 'bit0'), 'height': ('HT', 'DQF', ABI_HEIGHT_GOOD)}
ENT_VARS = {'mask': ('CloudMask', 'CloudMaskQualFlag', 'zero'), 'phase': ('CloudPhase', 'CloudPhaseFlag', 'bit0'), 'height': ('CldTopHght', 'CloudHgtQF', HEIGHT_GOOD), 'base': ('CldBaseHght', 'SummaryQC_Cloud_Base', 'zero')}
LAADS_VARS = {'mask': 'geophysical_data/Integer_Cloud_Mask', 'phase': 'geophysical_data/Cloud_Phase_Cloud_Top_Properties', 'height': 'geophysical_data/Cloud_Top_Height'}
VIIRS_ALT_KM = 829.0

class NotConfigured(Exception):
    pass

class Unreachable(Exception):
    pass

def _utc(t: datetime) -> datetime:
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)

def _ecef(lat, lon, h=0.0, a=6378137.0, b=6356752.31414):
    phi, lam = (np.radians(lat), np.radians(lon))
    e2 = 1 - b * b / (a * a)
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    return ((n + h) * np.cos(phi) * np.cos(lam), (n + h) * np.cos(phi) * np.sin(lam), (n * (1 - e2) + h) * np.sin(phi))

def _geodetic(x, y, z, a=6378137.0, b=6356752.31414):
    e2, ep2 = (1 - b * b / (a * a), a * a / (b * b) - 1)
    p = np.hypot(x, y)
    th = np.arctan2(z * a, p * b)
    phi = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    return (np.degrees(phi), np.degrees(np.arctan2(y, x)), p / np.cos(phi) - n)

def geo_zenith(lat, lon, lon0: float, H: float=42164160.0, a=6378137.0, b=6356752.31414):
    px, py, pz = _ecef(lat, lon, 0.0, a, b)
    sx, sy = (H * math.cos(math.radians(lon0)), H * math.sin(math.radians(lon0)))
    phi, lam = (np.radians(lat), np.radians(lon))
    ux, uy, uz = (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))
    dx, dy, dz = (sx - px, sy - py, -pz)
    c = (dx * ux + dy * uy + dz * uz) / np.sqrt(dx * dx + dy * dy + dz * dz)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def parallax_geo(lat, lon, h, geo: dict):
    a, b = (geo['a'], geo['b'])
    h = np.nan_to_num(np.asarray(h, float), nan=0.0)
    px, py, pz = _ecef(lat, lon, 0.0, a, b)
    sx, sy, sz = (geo['H'] * math.cos(math.radians(geo['lon0'])), geo['H'] * math.sin(math.radians(geo['lon0'])), 0.0)
    dx, dy, dz = (px - sx, py - sy, pz - sz)
    A, B = (a + h, b + h)
    qa = (dx * dx + dy * dy) / (A * A) + dz * dz / (B * B)
    qb = 2 * ((sx * dx + sy * dy) / (A * A) + sz * dz / (B * B))
    qc = (sx * sx + sy * sy) / (A * A) + sz * sz / (B * B) - 1
    with np.errstate(invalid='ignore'):
        t = (-qb - np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    la, lo, _ = _geodetic(sx + t * dx, sy + t * dy, sz + t * dz, a, b)
    return (la, lo)

def parallax_azimuth(lat, lon, h, zen, az):
    d = np.nan_to_num(np.asarray(h, float), nan=0.0) * np.tan(np.radians(zen)) / 1000.0
    lat = np.asarray(lat, float)
    dlat = d * np.cos(np.radians(az)) / 111.32
    dlon = d * np.sin(np.radians(az)) / (111.32 * np.maximum(np.cos(np.radians(lat)), 0.01))
    return (lat + dlat, (np.asarray(lon, float) + dlon + 180) % 360 - 180)

def km_between(lat1, lon1, lat2, lon2):
    p1, p2 = (np.radians(lat1), np.radians(lat2))
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return EARTH_KM * np.arccos(np.clip(c, -1, 1))

def _unit(lat, lon):
    phi, lam = (np.radians(lat), np.radians(lon))
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)

def _chord(km: float) -> float:
    return 2 * math.sin(km / EARTH_KM / 2)

def _day(t: datetime) -> str:
    return t.strftime('%Y-%m-%d')

def geo_for(t: datetime) -> list:
    d = _day(t)
    return [(s, b, l0) for s, b, l0, d0, d1 in GEO if d0 <= d < d1]

def choose(lat: float, lon: float, t: datetime, now: datetime | None=None) -> list:
    t = _utc(t)
    best = None
    for s, b, l0 in geo_for(t):
        z = float(geo_zenith(lat, lon, l0))
        if z < GEO_ZEN_MAX and (best is None or z < best[0]):
            best = (z, s, b, l0)
    if best:
        _, s, b, l0 = best
        if s.startswith('goes'):
            side = 'east' if l0 > -100 else 'west'
            x, y = satcheck.latlon_to_xy(np.array(lat), np.array(lon), 6378137.0, 6356752.31414, 42164160.0, math.radians(l0))
            (x0, x1), (y0, y1) = SECTOR_C[side]
            sector = 'C' if x0 <= float(x) <= x1 and y0 <= float(y) <= y1 else 'F'
            return [{'sensor': s, 'bucket': b, 'kind': 'geo', 'family': 'abi', 'sector': sector}]
        return [{'sensor': s, 'bucket': b, 'kind': 'geo', 'family': 'ahi'}]
    now = _utc(now or datetime.now(timezone.utc))
    if t >= now - timedelta(days=JRR_SAFE_DAYS):
        start = {'n20': '2018-01-01', 'n21': '2023-03-01'}
        return [{'sensor': s, 'bucket': b, 'kind': 'polar', 'family': 'jrr', 'tag': tag} for s, b, tag in POLAR if _day(t) >= start[s]]
    return [{'sensor': 'n20', 'bucket': None, 'kind': 'polar', 'family': 'laads'}]

def _open(src):
    import h5py
    if isinstance(src, str) and src.startswith('http'):
        return h5py.File(satcheck.RangeFile(src), 'r')
    return h5py.File(src, 'r')

def _text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (bytes, np.bytes_)):
        return bytes(v).decode()
    if isinstance(v, str):
        return v
    v = np.asarray(v).ravel()[0]
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)

def _attr_str(f, name):
    return _text(f.attrs.get(name))

def _iso(s: str) -> datetime:
    return datetime.strptime(s.strip().rstrip('Z')[:19], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)

def _raw(V, sl=()):
    raw = np.asarray(V[sl])
    fill = V.attrs.get('_FillValue')
    fill = None if fill is None else np.asarray(fill).ravel()[:1]
    if raw.dtype.kind == 'i' and (_text(V.attrs.get('_Unsigned')) or '').strip().lower() == 'true':
        ut = np.dtype(raw.dtype.str.replace('i', 'u'))
        raw = raw.view(ut)
        fill = None if fill is None else fill.astype(raw.dtype.str.replace('u', 'i')).view(ut)
    return (raw, None if fill is None else fill[0].item())

def _scale(V):
    g = lambda k, d: satcheck._num(V.attrs[k]) if k in V.attrs else d
    return (g('scale_factor', 1.0), g('add_offset', 0.0))

def _good(q, how):
    q = np.asarray(q).astype(np.int32)
    if how == 'zero':
        return q == 0
    if how == 'bit0':
        return (q >= 0) & (q < 255) & (q & 1 == 0)
    return np.isin(q, how)

def _norm(product, raw, fill, scale=1.0, offset=0.0):
    raw = np.asarray(raw)
    none = raw == fill if fill is not None else np.zeros(raw.shape, bool)
    if product in ('height', 'base'):
        v = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
        v[none | (v < -500) | (v > 25000)] = np.nan
        return v
    v = raw.astype(np.int32)
    v[none | (v < 0) | (v > (3 if product == 'mask' else 5))] = -1
    return v.astype(np.int8)

def _row_times(t0: datetime, t1: datetime, rows, nrows: int):
    a, b = (t0.timestamp(), t1.timestamp())
    return a + (b - a) * (np.asarray(rows, float) + 0.5) / max(nrows, 1)

def _valid(product, val):
    return np.isfinite(val) if product in ('height', 'base') else val >= 0

def read_abi(src, product: str, bbox, sensor: str | None=None) -> dict | None:
    vname, qname, how = ABI_VARS[product]
    with _open(src) as f:
        a, b, H, lon0 = satcheck._proj(f)
        X, Y = (f['x'], f['y'])
        xs = X[:] * satcheck._num(X.attrs['scale_factor']) + satcheck._num(X.attrs['add_offset'])
        ys = Y[:] * satcheck._num(Y.attrs['scale_factor']) + satcheck._num(Y.attrs['add_offset'])
        w, s, e, n = bbox
        lats = np.array([s, s, n, n, (s + n) / 2, (s + n) / 2, s, n])
        lons = np.array([w, e, w, e, w, e, (w + e) / 2, (w + e) / 2])
        px, py = satcheck.latlon_to_xy(lats, lons, a, b, H, lon0)
        i0, i1 = (max(0, int(np.searchsorted(xs, np.nanmin(px))) - 2), int(np.searchsorted(xs, np.nanmax(px))) + 2)
        j0, j1 = (max(0, int(np.searchsorted(-ys, -np.nanmax(py))) - 2), int(np.searchsorted(-ys, -np.nanmin(py))) + 2)
        i1, j1 = (min(i1, xs.size), min(j1, ys.size))
        if i1 <= i0 or j1 <= j0:
            return None
        levels = 4
        if product == 'mask' and vname not in f and ('BCM' in f):
            vname, levels = ('BCM', 2)
        sl = (slice(j0, j1), slice(i0, i1))
        V = f[vname]
        raw, fill = _raw(V, sl)
        val = _norm(product, raw, fill, *_scale(V))
        if levels == 2:
            val = np.where(val == 1, 3, np.where(val == 0, 0, -1)).astype(np.int8)
        good = _good(_raw(f[qname], sl)[0], how) & _valid(product, val)
        tb = f['time_bounds'][()]
        t0, t1 = (J2000 + timedelta(seconds=float(x)) for x in tb)
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
        lat, lon = satcheck.xy_to_latlon(gx, gy, a, b, H, lon0)
        rows = np.broadcast_to(np.arange(j0, j1)[:, None], gx.shape)
        sensor = sensor or _abi_sensor(f, src)
    geo = {'lon0': math.degrees(lon0), 'a': a, 'b': b, 'H': H}
    return {'sensor': sensor, 'product': product, 'kind': 'geo', 't0': t0, 't1': t1, 'lat': lat, 'lon': lon, 't': _row_times(t0, t1, rows, ys.size), 'val': val, 'good': good, 'zen': geo_zenith(lat, lon, geo['lon0'], H, a, b), 'geo': geo, 'pc': None, 'az': None, 'levels': levels}

def _abi_sensor(f, src) -> str | None:
    pid = _attr_str(f, 'platform_ID')
    m = re.search('G(\\d\\d)', pid or '') or re.search('_G(\\d\\d)_', str(src))
    return f'goes{m[1]}' if m else None

def in_lons(lon, w: float, e: float):
    return np.mod(np.asarray(lon, float) - w, 360) <= e - w

def _ent_window(f, bbox, kind):
    w, s, e, n = bbox
    if kind == 'geo':
        r0 = int(np.asarray(f['StartRow'][()]).ravel()[0]) - 1 if 'StartRow' in f else 0
        c0 = int(np.asarray(f['StartColumn'][()]).ravel()[0]) - 1 if 'StartColumn' in f else 0
        ny, nx = f['Latitude'].shape
        la = np.array([s, s, n, n, (s + n) / 2, (s + n) / 2, s, n])
        lo = np.array([w, e, w, e, w, e, (w + e) / 2, (w + e) / 2])
        col, line = satread.hsd_colline(la, lo, satread._H20)
        if not np.isfinite(col).any():
            return None
        j0, j1 = (int(np.nanmin(line)) - 1 - r0 - 2, int(np.nanmax(line)) - r0 + 2)
        i0, i1 = (int(np.nanmin(col)) - 1 - c0 - 2, int(np.nanmax(col)) - c0 + 2)
        j0, i0, j1, i1 = (max(0, j0), max(0, i0), min(ny, j1), min(nx, i1))
        return None if j1 <= j0 or i1 <= i0 else (j0, j1, i0, i1)
    lat, lon = (f['Latitude'][()], f['Longitude'][()])
    inside = (lat >= s) & (lat <= n) & in_lons(lon, w, e)
    if not inside.any():
        return None
    rr, cc = np.nonzero(inside)
    return (max(0, rr.min() - 2), min(lat.shape[0], rr.max() + 3), max(0, cc.min() - 2), min(lat.shape[1], cc.max() + 3))

def _cross(a, b, c, d) -> bool:
    o = lambda p, q, r: (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0

def _inside(pt, poly) -> bool:
    x, y, hit = (pt[0], pt[1], False)
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit

def footprint_meets(wkt: str | None, bbox, pad: float=1.0) -> bool:
    try:
        poly = [tuple((float(v) for v in p.split())) for p in re.search('\\(\\((.*)\\)\\)', wkt).group(1).split(',')]
    except Exception:
        return True
    if len(poly) > 1 and poly[0] == poly[-1]:
        poly = poly[:-1]
    if len(poly) < 3:
        return True
    w, s, e, n = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    if max((p[0] for p in poly)) - min((p[0] for p in poly)) > 180 or e > 180:
        poly = [((x + 360) % 360, y) for x, y in poly]
        w, e = ((w + 360) % 360, (e + 360) % 360)
        if e < w:
            e += 360
    box = [(w, s), (e, s), (e, n), (w, n)]
    if any((w <= x <= e and s <= y <= n for x, y in poly)) or any((_inside(c, poly) for c in box + [((w + e) / 2, (s + n) / 2)])):
        return True
    return any((_cross(a, b, c, d) for a, b in zip(poly, poly[1:] + poly[:1]) for c, d in zip(box, box[1:] + box[:1])))

def box_meets(f, bbox, pad: float=1.0) -> bool:
    wkt = _attr_str(f, 'geospatial_bounds')
    if wkt:
        return footprint_meets(wkt, bbox, pad)
    try:
        la0, la1, lo0, lo1 = (float(_attr_str(f, f'geospatial_{k}')) for k in ('lat_min', 'lat_max', 'lon_min', 'lon_max'))
    except (TypeError, ValueError):
        return True
    w, s, e, n = bbox
    if la1 < s - pad or la0 > n + pad:
        return False
    if lo1 - lo0 > 180:
        return True
    return bool(in_lons(np.array([lo0, lo1]), w - pad, e + pad).any()) or bool(in_lons(np.array([w, e]), lo0 - pad, lo1 + pad).any())

def swath_zenith(lat, lon, clat, clon, alt_km: float=VIIRS_ALT_KM):
    g = km_between(lat, lon, clat, clon) / EARTH_KM
    ro = (EARTH_KM + alt_km) / EARTH_KM
    return np.degrees(np.arctan2(ro * np.sin(g), ro * np.cos(g) - 1))

def read_enterprise(src, product: str, bbox, sensor: str, kind: str) -> dict | None:
    vname, qname, how = ENT_VARS[product]
    with _open(src) as f:
        if kind == 'polar' and (not footprint_meets(_attr_str(f, 'geospatial_bounds'), bbox)):
            return None
        win = _ent_window(f, bbox, kind)
        if win is None:
            return None
        j0, j1, i0, i1 = win
        sl = (slice(j0, j1), slice(i0, i1))
        V = f[vname]
        raw, fill = _raw(V, sl)
        val = _norm(product, raw, fill, *_scale(V))
        q = f[qname]
        q = _raw(q, sl + ((0,) if q.ndim == 3 else ()))[0]
        good = _good(q, how) & _valid(product, val)
        lat, lon = (f['Latitude'][sl].astype(np.float64), f['Longitude'][sl].astype(np.float64))
        off_earth = (lat < -90) | (lon < -180)
        lat[off_earth], lon[off_earth] = (np.nan, np.nan)
        good &= ~off_earth
        pc = None
        if product == 'height' and 'Latitude_Pc' in f:
            plat, plon = (f['Latitude_Pc'][sl].astype(np.float64), f['Longitude_Pc'][sl].astype(np.float64))
            bad = (plat < -90) | (plon < -180)
            plat[bad], plon[bad] = (np.nan, np.nan)
            pc = (plat, plon)
        t0, t1 = (_iso(_attr_str(f, 'time_coverage_start')), _iso(_attr_str(f, 'time_coverage_end')))
        nrows, ncols = f['Latitude'].shape
        if kind == 'polar':
            mid = (slice(j0, j1), ncols // 2)
            clat, clon = (f['Latitude'][mid].astype(np.float64)[:, None], f['Longitude'][mid].astype(np.float64)[:, None])
        r0 = int(np.asarray(f['StartRow'][()]).ravel()[0]) - 1 if kind == 'geo' and 'StartRow' in f else 0
    rows = np.broadcast_to(np.arange(j0, j1)[:, None] + r0, val.shape)
    if kind == 'geo':
        h = satread._H20
        geo = {'lon0': h['sub_lon'], 'a': h['req'] * 1000, 'b': h['rpol'] * 1000, 'H': h['rs'] * 1000}
        if (t1 - t0).total_seconds() < DISK_SHORT_S:
            t1 = t0 + timedelta(seconds=DISK_S)
        t = _row_times(t0, t1, rows, satread._H20_LINES)
        zen = geo_zenith(lat, lon, geo['lon0'], geo['H'], geo['a'], geo['b'])
    else:
        geo, t = (None, _row_times(t0, t1, rows, nrows))
        zen = swath_zenith(lat, lon, np.where(clat < -90, np.nan, clat), np.where(clon < -180, np.nan, clon))
    return {'sensor': sensor, 'product': product, 'kind': kind, 't0': t0, 't1': t1, 'lat': lat, 'lon': lon, 't': t, 'val': val, 'good': good, 'zen': zen, 'geo': geo, 'pc': pc, 'az': None, 'levels': 4}
LAADS_MASK = {0: 3, 1: 2, 2: 1, 3: 0}
LAADS_PHASE = {1: 0, 2: 1, 3: 4, 4: 5}

def read_laads(src, product: str, bbox, sensor: str='n20') -> dict | None:
    with _open(src) as f:
        if not box_meets(f, bbox):
            return None
        g = f['geolocation_data']
        lat, lon = (g['latitude'][()].astype(np.float64), g['longitude'][()].astype(np.float64))
        w, s, e, n = bbox
        inside = (lat >= s) & (lat <= n) & in_lons(lon, w, e)
        if not inside.any():
            return None
        rr, cc = np.nonzero(inside)
        sl = (slice(max(0, rr.min() - 2), rr.max() + 3), slice(max(0, cc.min() - 2), cc.max() + 3))
        V = f[LAADS_VARS[product]]
        raw, fill = _raw(V, sl)
        if product == 'height':
            val = _norm('height', raw, fill, *_scale(V))
            good = np.isfinite(val)
        else:
            table = LAADS_MASK if product == 'mask' else LAADS_PHASE
            val = np.full(raw.shape, -1, np.int8)
            for k, v in table.items():
                val[raw == k] = v
            good = val >= 0
        angles = []
        for name in ('sensor_zenith', 'sensor_azimuth'):
            A, fill2 = _raw(g[name], sl)
            A = A.astype(np.float64)
            if fill2 is not None:
                A[A == fill2] = np.nan
            sc2, off2 = _scale(g[name])
            angles.append(A * sc2 + off2)
        zen, az = angles
        t0, t1 = (_iso(_attr_str(f, 'time_coverage_start')), _iso(_attr_str(f, 'time_coverage_end')))
        nrows = lat.shape[0]
        la, lo = (lat[sl], lon[sl])
    rows = np.broadcast_to(np.arange(sl[0].start, sl[0].start + la.shape[0])[:, None], la.shape)
    return {'sensor': sensor, 'product': product, 'kind': 'polar', 't0': t0, 't1': t1, 'lat': la, 'lon': lo, 't': _row_times(t0, t1, rows, nrows), 'val': val, 'good': good, 'zen': zen, 'geo': None, 'pc': None, 'az': az, 'levels': 4}

def shift_of(height: dict):
    h, ok = (height['val'], height['good'] & np.isfinite(height['val']))
    if height.get('pc') is not None:
        la, lo = height['pc']
        la, lo = (np.where(ok, la, np.nan), np.where(ok, lo, np.nan))
    elif height.get('geo') is not None:
        la, lo = parallax_geo(height['lat'], height['lon'], np.where(ok, h, 0.0), height['geo'])
        la, lo = (np.where(ok, la, np.nan), np.where(ok, lo, np.nan))
    elif height.get('az') is not None:
        la, lo = parallax_azimuth(height['lat'], height['lon'], np.where(ok, h, 0.0), height['zen'], height['az'])
        la, lo = (np.where(ok, la, np.nan), np.where(ok, lo, np.nan))
    else:
        la, lo = (np.where(ok, height['lat'], np.nan), np.where(ok, height['lon'], np.nan))
    return (la, lo)

def parallax_how(height: dict | None) -> str:
    if height is None:
        return 'none'
    if height.get('pc') is not None:
        return 'product'
    return 'geometry' if height.get('geo') is not None or height.get('az') is not None else 'none'

def _cloudy(layer: dict):
    v = layer['val']
    if layer['product'] == 'mask':
        return v >= 2
    if layer['product'] == 'phase':
        return v >= 1
    return np.isfinite(v)

def corrected(layer: dict, height: dict | None):
    lat, lon = (layer['lat'], layer['lon'])
    if height is None:
        return (lat, lon, np.full(lat.shape, np.nan))
    hla, hlo = shift_of(height)
    if layer is height:
        return (np.where(np.isfinite(hla), hla, lat), np.where(np.isfinite(hlo), hlo, lon), np.where(height['good'], height['val'], np.nan))
    from scipy.spatial import cKDTree
    hv = np.isfinite(hla) & np.isfinite(height['lat'])
    out_la, out_lo, out_h = (lat.astype(np.float64).copy(), lon.astype(np.float64).copy(), np.full(lat.shape, np.nan))
    cloudy = _cloudy(layer) & np.isfinite(lat)
    if not hv.any() or not cloudy.any():
        return (out_la, out_lo, out_h)
    tree = cKDTree(_unit(height['lat'][hv], height['lon'][hv]))
    d, k = tree.query(_unit(lat[cloudy], lon[cloudy]), distance_upper_bound=_chord(LOOKUP_KM))
    hit = np.isfinite(d)
    idx = np.nonzero(cloudy)
    sel = (idx[0][hit], idx[1][hit])
    dla = (hla[hv] - height['lat'][hv])[k[hit]]
    dlo = (hlo[hv] - height['lon'][hv])[k[hit]]
    out_la[sel] += dla
    out_lo[sel] = (out_lo[sel] + dlo + 180) % 360 - 180
    out_h[sel] = height['val'][hv][k[hit]]
    moved = np.zeros(lat.shape, bool)
    moved[sel] = np.abs(dla) + np.abs(dlo) > 0
    clear = ~cloudy & np.isfinite(lat)
    if moved.any() and clear.any():
        cover = cKDTree(_unit(out_la[moved], out_lo[moved]))
        d, _ = cover.query(_unit(lat[clear], lon[clear]), distance_upper_bound=_chord(0.6 * spacing_km(layer)))
        ci = np.nonzero(clear)
        gone = (ci[0][np.isfinite(d)], ci[1][np.isfinite(d)])
        out_la[gone], out_lo[gone] = (np.nan, np.nan)
    return (out_la, out_lo, out_h)

def spacing_km(layer: dict) -> float:
    la, lo = (layer['lat'], layer['lon'])
    ny, nx = la.shape
    j, i = (ny // 2, nx // 2)
    d = []
    if nx > 1:
        i0 = max(i - 1, 0)
        d.append(km_between(la[j, i0], lo[j, i0], la[j, i0 + 1], lo[j, i0 + 1]))
    if ny > 1:
        j0 = max(j - 1, 0)
        d.append(km_between(la[j0, i], lo[j0, i], la[j0 + 1, i], lo[j0 + 1, i]))
    d = [float(x) for x in d if np.isfinite(x)]
    return max(d) if d else 0.0

class Prepared:

    def __init__(self, L: dict, height: dict | None, r_km: float):
        from scipy.spatial import cKDTree
        la, lo, h = corrected(L, height)
        self.ok = ok = np.isfinite(la) & np.isfinite(lo)
        self.val, self.good, self.zen, self.h = (L['val'][ok], L['good'][ok], L['zen'][ok], h[ok])
        self.t = np.broadcast_to(np.asarray(L['t'], float), L['lat'].shape)[ok]
        self.shift = km_between(L['lat'][ok], L['lon'][ok], la[ok], lo[ok])
        self.tree = cKDTree(_unit(la[ok], lo[ok])) if ok.any() else None
        self.r = max(r_km, 0.75 * spacing_km(L))
        self.reach = self.r + 3.0 + (float(self.shift.max()) if self.shift.size else 0.0)
        self.how = parallax_how(height)
        self.L = L

    def time_at(self, u) -> float | None:
        if self.tree is None:
            return None
        d, k = self.tree.query(u, distance_upper_bound=_chord(self.reach))
        return float(self.t[k]) if np.isfinite(d) else None

    def has(self, u) -> bool:
        return self.tree is not None and bool(np.isfinite(self.tree.query(u, distance_upper_bound=_chord(self.r))[0]))

def _same_scan(layers: list, L: dict):
    for H in layers:
        if H['product'] == 'height' and H['sensor'] == L['sensor'] and (abs((H['t0'] - L['t0']).total_seconds()) <= 60):
            return H
    return None

def pick(preps: list, product: str, u, t_valid: datetime):
    tv = _utc(t_valid).timestamp()
    on, near = (None, None)
    for P in preps:
        if P.L['product'] != product:
            continue
        tp = P.time_at(u)
        if tp is None:
            continue
        dt = abs(tp - tv)
        if dt > 60 * (GEO_WIN_MIN if P.L['kind'] == 'geo' else POLAR_WIN_MIN):
            continue
        if P.has(u):
            if on is None or dt < on[2]:
                on = (P, tp, dt)
        elif near is None or dt < near[2]:
            near = (P, tp, dt)
    best = on or near
    return (best[0], best[1]) if best else (None, None)

def _when(ts):
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
EMPTY = {'cloud_frac': math.nan, 'n_pix': 0, 'n_good': 0, 't_mask': None, 'dt_mask': None, 'mask_levels': None, 'water_frac': math.nan, 'low_water_frac': math.nan, 'ice_frac': math.nan, 'n_phase': 0, 't_phase': None, 'cth_mean': math.nan, 'cth_max': math.nan, 'n_height': 0, 't_height': None, 'cbh_mean': math.nan, 'n_base': 0, 't_base': None, 'cu_cover': math.nan, 'sat_zen': math.nan, 'shift_km': math.nan, 'parallax': None}

def low_water(g, v, hts):
    water = g & ((v == 1) | (v == 2))
    known_h = np.isfinite(hts)
    low = water & known_h & (hts < LOW_TOP_M)
    known = g & ~(water & ~known_h)
    return (low, known, water, water & known_h)

def cu_water(g, v, hts, ground: float):
    water = g & ((v == 1) | (v == 2))
    known_h = np.isfinite(hts)
    return (water & known_h & (hts - ground <= CU_TOP_AGL), g & ~(water & ~known_h))

def truth_at(points: list, layers: list, r_km: float=R_KM, grounds=None) -> list:
    layers = [L for L in layers if L is not None]
    preps = [Prepared(L, L if L['product'] == 'height' else _same_scan(layers, L), r_km) for L in layers]
    rows = []
    for k_pt, (lat, lon, tv) in enumerate(points):
        ground = grounds[k_pt] if grounds is not None else None
        tv = _utc(tv)
        row = {'lat': lat, 'lon': lon, 't_valid': tv.strftime('%Y-%m-%dT%H:%M:%SZ'), 'status': 'no_scan', 'sensor': None, 'blocked': [], **EMPTY}
        u = _unit(lat, lon)
        in_time, zens, shifts, cloud_seen, how = (False, [], [], False, None)
        for product in ('mask', 'phase', 'height', 'base'):
            P, tp = pick(preps, product, u, tv)
            if P is None:
                continue
            in_time = True
            idx = P.tree.query_ball_point(u, _chord(P.r))
            if not idx:
                continue
            row['sensor'] = row['sensor'] or P.L['sensor']
            v, g = (P.val[idx], P.good[idx])
            zens.append(float(np.nanmean(P.zen[idx])))
            when, dt = (_when(tp), round(tp - tv.timestamp(), 1))
            if product == 'mask':
                cl = g & (v >= 2)
                row.update(n_pix=len(idx), n_good=int(g.sum()), t_mask=when, dt_mask=dt, mask_levels=P.L.get('levels', 4), cloud_frac=float(cl.sum() / g.sum()) if g.any() else math.nan)
                cloud_seen |= bool(cl.any())
                matched = cl & np.isfinite(P.h[idx])
                if matched.any():
                    shifts.append(float(np.mean(P.shift[idx][matched])))
                    how = how or P.how
            elif product == 'phase':
                low, known, water, water_h = low_water(g, v, P.h[idx])
                n, nk = (int(g.sum()), int(known.sum()))
                cloud_seen |= bool((g & (v >= 1)).any())
                row.update(n_phase=n, t_phase=when, water_frac=float(water.sum() / n) if n else math.nan, low_water_frac=float(low.sum() / nk) if nk and (not (water.any() and (not water_h.any()))) else math.nan, ice_frac=float((g & (v == 4)).sum() / n) if n else math.nan)
                if ground is not None and math.isfinite(ground) and nk and (not (water.any() and (not water_h.any()))) and (row['ice_frac'] <= CU_HIDDEN):
                    cu, _ = cu_water(g, v, P.h[idx], float(ground))
                    row['cu_cover'] = float(cu.sum() / nk)
            elif product == 'height':
                hv = g & np.isfinite(v)
                row.update(n_height=int(hv.sum()), t_height=when, cth_mean=float(v[hv].mean()) if hv.any() else math.nan, cth_max=float(v[hv].max()) if hv.any() else math.nan)
                if hv.any():
                    cloud_seen = True
                    if not shifts:
                        shifts.append(float(np.mean(P.shift[idx][hv])))
                        how = how or P.how
            else:
                hv = g & np.isfinite(v)
                row.update(n_base=int(hv.sum()), t_base=when, cbh_mean=float(v[hv].mean()) if hv.any() else math.nan)
        if zens:
            row['status'] = 'ok'
            row['sat_zen'] = round(float(np.mean(zens)), 2)
            if shifts:
                row['shift_km'], row['parallax'] = (round(float(np.mean(shifts)), 2), how)
            else:
                row['shift_km'], row['parallax'] = (math.nan if cloud_seen else 0.0, 'none')
        elif in_time:
            row['status'] = 'no_pixels'
        rows.append(row)
    return rows

def cells(layers: list, step: float=CELL_DEG, ground=None) -> list:
    by = {L['product']: L for L in layers if L is not None}
    if 'mask' not in by:
        return []
    H = by.get('height')

    def binned(L):
        la, lo, h = corrected(L, H)
        ok = np.isfinite(la) & np.isfinite(lo)
        return (ok, np.floor(la[ok] / step).astype(np.int64), np.floor(lo[ok] / step).astype(np.int64), h[ok])
    M = by['mask']
    ok, iy, ix, _ = binned(M)
    keys, inv = np.unique(iy * 100000 + ix, return_inverse=True)
    n = keys.size

    def cnt(w, at=inv):
        return np.bincount(at, weights=w, minlength=n)
    g, v = (M['good'][ok], M['val'][ok])
    n_pix, n_good, cloud = (cnt(None), cnt(g.astype(float)), cnt((g & (v >= 2)).astype(float)))
    zen = cnt(M['zen'][ok]) / n_pix
    t = cnt(np.broadcast_to(np.asarray(M['t'], float), M['lat'].shape)[ok]) / n_pix

    def onto(L):
        ok, a, b, h = binned(L)
        k = a * 100000 + b
        j = np.searchsorted(keys, k)
        hit = (j < n) & (keys[np.minimum(j, n - 1)] == k)
        return (ok, hit, j[hit], h[hit])
    lw, ice, cth, cuc = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    ky = np.floor_divide(keys + 50000, 100000)
    kx = keys - ky * 100000
    gc = None if ground is None else np.asarray(ground((ky + 0.5) * step, (kx + 0.5) * step), np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        if 'phase' in by:
            P = by['phase']
            ok, hit, j, h = onto(P)
            g, v = (P['good'][ok][hit], P['val'][ok][hit])
            low, known, water, water_h = low_water(g, v, h)
            npx = cnt(g.astype(float), j)
            lw = cnt(low.astype(float), j) / cnt(known.astype(float), j)
            unknown = (cnt(water.astype(float), j) > 0) & (cnt(water_h.astype(float), j) == 0)
            lw[unknown] = np.nan
            ice = cnt((g & (v == 4)).astype(float), j) / npx
            if gc is not None:
                cu, _ = cu_water(g, v, h, gc[j])
                cuc = cnt(cu.astype(float), j) / cnt(known.astype(float), j)
                cuc[unknown | ~(ice <= CU_HIDDEN) | ~np.isfinite(gc)] = np.nan
        if H is not None:
            ok, hit, j, h = onto(H)
            g = H['good'][ok][hit] & np.isfinite(h)
            cth = cnt(np.where(g, h, 0.0), j) / cnt(g.astype(float), j)
        cf = cloud / n_good
    return [{'lat': round(float((ky[k] + 0.5) * step), 4), 'lon': round(float((kx[k] + 0.5) * step), 4), 't': _when(float(t[k])), 'n_pix': int(n_pix[k]), 'cloud_frac': float(cf[k]), 'low_water_frac': float(lw[k]), 'ice_frac': float(ice[k]), 'cth_mean': float(cth[k]), 'cu_cover': float(cuc[k]), 'sat_zen': round(float(zen[k]), 2)} for k in range(n)]

def list_keys(bucket: str, prefix: str) -> list:
    keys, token = ([], None)
    while True:
        url = f'https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}'
        if token:
            url += '&continuation-token=' + urllib.parse.quote(token, safe='')
        root = ET.fromstring(satcheck._get(url))
        keys += [e.text for e in root.iter() if e.tag.endswith('Key') and e.text]
        nxt = [e.text for e in root.iter() if e.tag.endswith('NextContinuationToken')]
        if not nxt:
            return keys
        token = nxt[0]

def key_start(key: str) -> datetime | None:
    name = os.path.basename(key)
    m = re.search('_s(\\d{4})(\\d{3})(\\d{2})(\\d{2})(\\d{2})\\d_', name)
    if m and name.startswith('OR_'):
        return datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1, hours=int(m[3]), minutes=int(m[4]), seconds=int(m[5]))
    m = re.search('_s(\\d{4})(\\d{2})(\\d{2})(\\d{2})(\\d{2})(\\d{2})\\d', name)
    if m:
        return datetime(*(int(x) for x in m.groups()), tzinfo=timezone.utc)
    m = re.search('\\.A(\\d{4})(\\d{3})\\.(\\d{2})(\\d{2})\\.', name)
    if m:
        return datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1, hours=int(m[3]), minutes=int(m[4]))
    return None

def scan_offset(src: dict, lat: float | None, lon: float | None) -> float:
    if src['kind'] != 'geo':
        return 45.0
    if src['family'] == 'ahi' and lat is not None:
        _, line = satread.hsd_colline(np.array(lat), np.array(lon), satread._H20)
        if np.isfinite(line):
            return DISK_S * float(line) / satread._H20_LINES
    return 75.0 if src.get('sector') == 'C' else 300.0

def candidates(src: dict, product: str, t: datetime, lister=None, lat: float | None=None, lon: float | None=None) -> list:
    lister = lister or list_keys
    t = _utc(t)
    fam = src['family']
    if product not in PRODUCTS.get(fam, ()):
        return []
    if fam == 'abi' and product == 'height' and (_day(t) < ACHA_FROM):
        return []
    win = timedelta(minutes=GEO_WIN_MIN if src['kind'] == 'geo' else POLAR_WIN_MIN)
    lead = timedelta(seconds=scan_offset(src, lat, lon))
    lo, hi = (t - win - lead - timedelta(minutes=1), t + win)
    keys = []
    if fam == 'abi':
        code = f"ABI-L2-{ABI_CODE[product]}{src['sector']}"
        h = lo.replace(minute=0, second=0, microsecond=0)
        while h <= hi:
            keys += [k for k in lister(src['bucket'], f'{code}/{h:%Y}/{h.timetuple().tm_yday:03d}/{h:%H}/') if f"-{ABI_CODE[product]}{src['sector']}-" in k]
            h += timedelta(hours=1)
    elif fam == 'ahi':
        m = lo.replace(minute=lo.minute // 10 * 10, second=0, microsecond=0)
        while m <= hi:
            keys += [k for k in lister(src['bucket'], f'AHI-L2-FLDK-Clouds/{m:%Y/%m/%d/%H%M}/') if f'AHI-{ENT_CODE[product]}_' in k]
            m += timedelta(minutes=10)
    elif fam == 'jrr':
        for d in sorted({lo.date(), hi.date()}):
            keys += [k for k in lister(src['bucket'], f'VIIRS-JRR-{JRR_DIR[product]}/{d:%Y/%m/%d}/') if f"_{src['tag']}_" in k]
    else:
        return []
    out = set()
    for k in keys:
        s = key_start(k)
        if s is not None and lo <= s <= hi:
            out.add((s, f"https://{src['bucket']}.s3.amazonaws.com/{k}"))
    return sorted(out, key=lambda x: abs(x[0] + lead - t))

def laads_token() -> str | None:
    return os.environ.get(LAADS_TOKEN_ENV) or None

class _ArchiveOnly(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).hostname != LAADS_HOST:
            raise urllib.error.HTTPError(req.full_url, 403, 'redirected off the archive', headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)
_OPENER = urllib.request.build_opener(_ArchiveOnly)

def _laads_request(url: str, token: str, rng: tuple[int, int] | None=None, head: bool=False):
    req = urllib.request.Request(url, headers=dict(satcheck.UA), method='HEAD' if head else 'GET')
    req.add_unredirected_header('Authorization', f'Bearer {token}')
    if rng:
        req.add_header('Range', f'bytes={rng[0]}-{rng[1]}')
    return req

def _laads_get(url: str, token: str, rng: tuple[int, int] | None=None, head: bool=False):
    try:
        with _OPENER.open(_laads_request(url, token, rng, head), timeout=180) as r:
            return int(r.headers['Content-Length']) if head else r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise NotConfigured(f'LAADS: HTTP {e.code}') from None
        raise

class LaadsFile(satcheck.RangeFile):

    def __init__(self, url: str, token: str):
        self.url, self.pos, self.cache, self._token = (url, 0, {}, token)
        self.size = _laads_get(url, token, head=True)

    def _block(self, i):
        b = self.cache.get(i)
        if b is None:
            lo = i * self.BLOCK
            b = self.cache[i] = _laads_get(self.url, self._token, (lo, min(self.size, lo + self.BLOCK) - 1))
        return b

def laads_open(url: str):
    token = laads_token()
    if not token:
        raise NotConfigured('LAADS: no Earthdata token')
    return LaadsFile(url, token)

def laads_candidates(product: str, t: datetime, kept: dict | None=None) -> list:
    token = laads_token()
    if not token:
        raise NotConfigured('LAADS: no Earthdata token')
    t = _utc(t)
    kept = {} if kept is None else kept
    prod = LAADS_PRODUCT[product]
    out = []
    for d in sorted({(t + dd).date() for dd in (timedelta(minutes=-POLAR_WIN_MIN), timedelta(minutes=POLAR_WIN_MIN))}):
        doy = d.timetuple().tm_yday
        base = f'{LAADS}/{LAADS_SET[prod]}/{prod}/{d:%Y}/{doy:03d}'
        if base not in kept:
            try:
                kept[base] = json.loads(_laads_get(base + '.json', token))
            except NotConfigured:
                raise
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise Unreachable(f'LAADS: HTTP {e.code}') from None
                kept[base] = {}
            except Exception as e:
                raise Unreachable(f'LAADS: {type(e).__name__}') from None
        listing = kept[base]
        names = [x.get('name', '') for x in listing.get('content') or []] if isinstance(listing, dict) else []
        for nm in names:
            s = key_start(nm)
            if s is not None and abs(s - t) <= timedelta(minutes=POLAR_WIN_MIN + 6):
                out.append((s, f'{base}/{nm}'))
    return sorted(out, key=lambda x: abs(x[0] - t))

def _bbox(pts, pad_km: float=PAD_KM):
    la = np.array([p[0] for p in pts], float)
    lo = np.array([p[1] for p in pts], float)
    if lo.max() - lo.min() > 180:
        lo = lo % 360
    dla = pad_km / 111.32
    dlo = pad_km / (111.32 * max(0.1, math.cos(math.radians(min(89.0, float(np.max(np.abs(la))) + dla)))))
    return [float(lo.min() - dlo), float(la.min() - dla), float(lo.max() + dlo), float(la.max() + dla)]

def read_any(src: dict, product: str, where, bbox):
    fam = src['family']
    if fam == 'abi':
        return read_abi(where, product, bbox, src['sensor'])
    if fam == 'ahi':
        return read_enterprise(where, product, bbox, src['sensor'], 'geo')
    if fam == 'jrr':
        return read_enterprise(where, product, bbox, src['sensor'], 'polar')
    return read_laads(laads_open(where) if isinstance(where, str) and where.startswith('http') else where, product, bbox, src['sensor'])
_RANK = {'ok': 3, 'no_pixels': 2, 'not_configured': 1, 'no_scan': 0}

def truth(points: list, now: datetime | None=None, lister=None, reader=None, log=None, grounds=None) -> list:
    reader = reader or read_any
    say = log or (lambda m: None)
    kept, lkept = ({}, {})

    def listing(bucket, prefix):
        if (bucket, prefix) not in kept:
            kept[bucket, prefix] = (lister or list_keys)(bucket, prefix)
        return kept[bucket, prefix]
    points = [(lat, lon, _utc(tv)) for lat, lon, tv in points]
    plans, blocked = ([], [set() for _ in points])
    for i, (lat, lon, tv) in enumerate(points):
        plan = []
        for s in choose(lat, lon, tv, now):
            for product in PRODUCTS[s['family']]:
                try:
                    if s['family'] == 'laads':
                        found = laads_candidates(product, tv, lkept)
                    else:
                        found = candidates(s, product, tv, listing, lat, lon)
                except NotConfigured:
                    blocked[i].add(product)
                    found = []
                except Unreachable as e:
                    say(f"  {s['sensor']} {product} listing: {e}")
                    found = []
                except Exception as e:
                    say(f"  {s['sensor']} {product} listing: {type(e).__name__}")
                    found = []
                plan.append((s, product, [u for _, u in found]))
        plans.append(plan)
    rows = [None] * len(points)
    failed = set()
    tries = {i: 0 for i in range(len(points))}
    todo = list(range(len(points)))
    while todo:
        slices = {}
        for i in todo:
            key = tuple(((product, tuple(urls[tries[i]:tries[i] + 1] if s['kind'] == 'geo' else urls)) for s, product, urls in plans[i]))
            slices.setdefault(key, []).append(i)
        order = sorted(slices.items(), key=lambda kv: min((points[i][2] for i in kv[1])))
        again = []
        for key, idx in order:
            pts = [points[i] for i in idx]
            bbox = _bbox(pts)
            first = idx[0]
            layers, cut = ([], set())
            for (s, product, urls), (_, mine) in zip(plans[first], key):
                if s['kind'] == 'geo':
                    mine = urls[tries[first]:tries[first] + GEO_TRIES]
                for url in mine:
                    if (url, product) in failed:
                        continue
                    try:
                        L = reader(s, product, url, bbox)
                    except NotConfigured:
                        L = None
                        cut.add(product)
                    except Exception as e:
                        L = None
                        say(f'  {os.path.basename(url)}: {type(e).__name__}')
                    if L is None:
                        failed.add((url, product))
                        continue
                    layers.append(L)
                    if s['kind'] == 'geo':
                        break
            got = truth_at(pts, layers, grounds=[grounds[i] for i in idx] if grounds is not None else None)
            del layers
            for i, r in zip(idx, got):
                bl = sorted(blocked[i] | cut)
                r['blocked'] = bl
                if r['status'] != 'ok' and bl:
                    r['status'] = 'not_configured'
                    r['sensor'] = r['sensor'] or (plans[i][0][0]['sensor'] if plans[i] else None)
                if rows[i] is None or _RANK[r['status']] > _RANK[rows[i]['status']]:
                    rows[i] = r
                geo_urls = [len(u) for s, _, u in plans[i] if s['kind'] == 'geo']
                if r['status'] == 'no_scan' and geo_urls and (tries[i] + 1 < min(GEO_TRIES, max(geo_urls))):
                    tries[i] += 1
                    again.append(i)
        todo = again
    return rows

def cells_for(src: dict, t: datetime, bbox, lister=None, reader=None, log=None, ground=None) -> list:
    reader = reader or read_any
    t = _utc(t)
    lat, lon = ((bbox[1] + bbox[3]) / 2, ((bbox[0] + bbox[2]) / 2 + 180) % 360 - 180)
    layers = []
    for product in ('mask', 'phase', 'height'):
        if src['family'] == 'laads':
            try:
                found = laads_candidates(product, t)
            except (NotConfigured, Unreachable):
                found = []
        else:
            found = candidates(src, product, t, lister, lat, lon)
        for _, url in found[:GEO_TRIES]:
            try:
                L = reader(src, product, url, bbox)
            except Exception as e:
                L = None
                if log:
                    log(f'  {os.path.basename(url)}: {type(e).__name__}')
            if L is not None:
                layers.append(L)
                break
    return cells(layers, ground=ground)
