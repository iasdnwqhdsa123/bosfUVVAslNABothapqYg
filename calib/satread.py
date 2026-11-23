from __future__ import annotations
import bz2
import math
import os
import re
import struct
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
import numpy as np
GRID_DEG = 0.005
UA = {'User-Agent': 'satcheck'}
HIMAWARI_BAND, HIMAWARI_RES, HIMAWARI_SEGS = (3, 'R05', 10)
HIMAWARI_SWITCH = '2022-12-13'
HSD_FLAT, HSD_E2 = (0.993305616, 0.00669438444)

def satellite_for(lon: float) -> str | None:
    lon = (lon + 180) % 360 - 180
    if lon >= 80 or lon <= -160:
        return 'himawari'
    if -135 <= lon <= -30:
        return 'goes'
    return None

class Missing(Exception):
    pass

def _fetch(url: str, cache: str | None, tries: int=4, magic: bytes=b'BZh') -> bytes:
    fn = os.path.join(cache, os.path.basename(url)) if cache else None
    if fn and os.path.exists(fn) and os.path.getsize(fn):
        with open(fn, 'rb') as f:
            b = f.read()
        if b.startswith(magic):
            return b
        os.unlink(fn)
    import time
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                b = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise Missing(url) from e
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
    if not b.startswith(magic):
        raise Missing(url)
    if fn:
        os.makedirs(cache, exist_ok=True)
        with open(fn + '.part', 'wb') as f:
            f.write(b)
        os.replace(fn + '.part', fn)
    return b

def hsd_parse(b: bytes) -> dict:
    little = b[5] == 0
    e = '<' if little else '>'
    hlen = struct.unpack_from(e + 'I', b, 70)[0]
    out, off = ({}, 0)
    while off < hlen:
        num, ln = (b[off], struct.unpack_from(e + 'H', b, off + 1)[0])
        if num == 2:
            out['ncol'], out['nline'] = struct.unpack_from(e + 'HH', b, off + 5)
        elif num == 3:
            out['sub_lon'], out['cfac'], out['lfac'], out['coff'], out['loff'], out['rs'], out['req'], out['rpol'] = struct.unpack_from(e + 'dIIffddd', b, off + 3)
        elif num == 5:
            out['band'] = struct.unpack_from(e + 'H', b, off + 3)[0]
            out['bits'], out['err'], out['outside'], out['gain'], out['const'] = struct.unpack_from(e + 'HHHdd', b, off + 13)
            out['albedo'] = struct.unpack_from(e + 'd', b, off + 35)[0]
        elif num == 7:
            out['segs'], out['seg'], out['line0'] = struct.unpack_from(e + 'BBH', b, off + 3)
        if ln == 0:
            break
        off += ln
    out['counts'] = np.frombuffer(b, dtype=e + 'u2', count=out['ncol'] * out['nline'], offset=hlen).reshape(out['nline'], out['ncol'])
    return out

def hsd_colline(lat, lon, h: dict):
    phi, lam = (np.radians(lat), np.radians(lon - h['sub_lon']))
    c_lat = np.arctan(HSD_FLAT * np.tan(phi))
    rl = h['rpol'] / np.sqrt(1 - HSD_E2 * np.cos(c_lat) ** 2)
    r1 = h['rs'] - rl * np.cos(c_lat) * np.cos(lam)
    r2 = -rl * np.cos(c_lat) * np.sin(lam)
    r3 = rl * np.sin(c_lat)
    rn = np.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    x, y = (np.degrees(np.arctan(-r2 / r1)), np.degrees(np.arcsin(-r3 / rn)))
    seen = r1 * (r1 - h['rs']) + r2 * r2 + r3 * r3 / HSD_FLAT ** 2 < 0
    col = h['coff'] + x * h['cfac'] / 65536.0
    line = h['loff'] + y * h['lfac'] / 65536.0
    return (np.where(seen, col, np.nan), np.where(seen, line, np.nan))

def himawari_url(t: datetime, seg: int) -> str:
    sat = '9' if f'{t:%Y-%m-%d}' >= HIMAWARI_SWITCH else '8'
    return f'https://noaa-himawari{sat}.s3.amazonaws.com/AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/HS_H0{sat}_{t:%Y%m%d_%H%M}_B{HIMAWARI_BAND:02d}_FLDK_{HIMAWARI_RES}_S{seg:02d}{HIMAWARI_SEGS:02d}.DAT.bz2'
_H05 = {'sub_lon': 140.7, 'cfac': 81865099, 'lfac': 81865099, 'coff': 11000.5, 'loff': 11000.5, 'rs': 42164.0, 'req': 6378.137, 'rpol': 6356.7523}
_H05_LINES = 22000

def himawari_scan(t: datetime, bbox, cache: str | None=None) -> dict | None:
    t = t.replace(minute=t.minute // 10 * 10, second=0, microsecond=0)
    w, s, e, n = bbox
    lat = np.arange(n - GRID_DEG / 2, s, -GRID_DEG)
    lon = np.arange(w + GRID_DEG / 2, e, GRID_DEG)
    LON, LAT = np.meshgrid(lon, lat)
    col, line = hsd_colline(LAT, LON, _H05)
    if not np.isfinite(line).all():
        return None
    per = _H05_LINES // HIMAWARI_SEGS
    s0, s1 = (int((np.nanmin(line) - 1) // per) + 1, int((np.nanmax(line) - 1) // per) + 1)
    refl = np.full(LAT.shape, np.nan, np.float32)
    for seg in range(max(1, s0), min(HIMAWARI_SEGS, s1) + 1):
        try:
            h = hsd_parse(bz2.decompress(_fetch(himawari_url(t, seg), cache)))
        except Exception:
            return None
        c, l = hsd_colline(LAT, LON, h)
        ci, li = (np.rint(c).astype(int) - 1, np.rint(l).astype(int) - h['line0'])
        ok = (ci >= 0) & (ci < h['ncol']) & (li >= 0) & (li < h['nline'])
        cnt = h['counts'][li[ok], ci[ok]]
        v = (cnt.astype(np.float32) * h['gain'] + h['const']) * h['albedo']
        v[(cnt == h['err']) | (cnt == h['outside'])] = np.nan
        refl[ok] = v
    when = t + timedelta(seconds=600.0 * float(np.nanmean(line)) / _H05_LINES)
    return {'refl': refl, 'lat': lat, 'lon': lon, 't': when}

def goes_scan(t: datetime, bbox, cache: str | None=None) -> dict | None:
    import satcheck
    url = satcheck.scan_url(t)
    if not url:
        return None
    kept = os.path.join(cache, f"{os.path.basename(url)[:-3]}_{'_'.join((f'{x:g}' for x in bbox))}.npz") if cache else None
    if kept and os.path.exists(kept):
        with np.load(kept) as z:
            return {'refl': z['refl'], 'lat': z['lat'], 'lon': z['lon'], 't': datetime.fromtimestamp(float(z['t']), timezone.utc)}
    px = satcheck.read_box(url, bbox)
    w, s, e, n = bbox
    lat = np.arange(n - GRID_DEG / 2, s, -GRID_DEG)
    lon = np.arange(w + GRID_DEG / 2, e, GRID_DEG)
    iy = np.floor((n - px['lat']) / GRID_DEG).astype(int)
    ix = np.floor((px['lon'] - w) / GRID_DEG).astype(int)
    ok = np.isfinite(px['lat']) & (iy >= 0) & (iy < lat.size) & (ix >= 0) & (ix < lon.size)
    refl = np.full((lat.size, lon.size), np.nan, np.float32)
    refl[iy[ok], ix[ok]] = px['refl'][ok]
    for _ in range(2):
        hole = ~np.isfinite(refl)
        if not hole.any():
            break
        p = np.pad(refl, 1, constant_values=np.nan)
        nb = np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx] for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))])
        with np.errstate(all='ignore'):
            fill = np.nanmean(nb, axis=0)
        refl[hole] = fill[hole]
    m = __import__('re').search('_s(\\d{4})(\\d{3})(\\d{2})(\\d{2})(\\d{2})', url)
    when = datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1, hours=int(m[3]), minutes=int(m[4]), seconds=int(m[5]) + 150)
    if kept:
        os.makedirs(cache, exist_ok=True)
        np.savez_compressed(kept + '.part.npz', refl=refl, lat=lat, lon=lon, t=when.timestamp())
        os.replace(kept + '.part.npz', kept)
    return {'refl': refl, 'lat': lat, 'lon': lon, 't': when}

def scan(sat: str, t: datetime, bbox, cache: str | None=None) -> dict | None:
    return himawari_scan(t, bbox, cache) if sat == 'himawari' else goes_scan(t, bbox, cache) if sat == 'goes' else None
IR_GRID_DEG = 0.02
_H20 = dict(_H05, cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5)
_H20_LINES = 5500

def _hsd_ir(b: bytes) -> dict:
    h = hsd_parse(b)
    e = '<' if b[5] == 0 else '>'
    hlen, off = (struct.unpack_from(e + 'I', b, 70)[0], 0)
    while off < hlen:
        num, ln = (b[off], struct.unpack_from(e + 'H', b, off + 1)[0])
        if num == 5:
            h['cwl'] = struct.unpack_from(e + 'd', b, off + 5)[0]
            h['c0'], h['c1'], h['c2'] = struct.unpack_from(e + 'ddd', b, off + 35)
            h['c'], h['h'], h['k'] = struct.unpack_from(e + 'ddd', b, off + 83)
        if ln == 0:
            break
        off += ln
    return h

def hsd_bt(counts, h: dict):
    rad = counts.astype(np.float64) * h['gain'] + h['const']
    lam = h['cwl'] * 1e-06
    with np.errstate(all='ignore'):
        te = h['h'] * h['c'] / (h['k'] * lam) / np.log(1 + 2 * h['h'] * h['c'] ** 2 / (np.maximum(rad, 1e-09) * 1000000.0 * lam ** 5))
    tb = h['c0'] + h['c1'] * te + h['c2'] * te * te
    return np.where((counts == h['err']) | (counts == h['outside']) | (rad <= 0), np.nan, tb).astype(np.float32)

def himawari_ir_url(t: datetime, seg: int, band: int=13) -> str:
    sat = '9' if f'{t:%Y-%m-%d}' >= HIMAWARI_SWITCH else '8'
    return f'https://noaa-himawari{sat}.s3.amazonaws.com/AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/HS_H0{sat}_{t:%Y%m%d_%H%M}_B{band:02d}_FLDK_R20_S{seg:02d}{HIMAWARI_SEGS:02d}.DAT.bz2'

def himawari_ir_scan(t: datetime, bbox, cache: str | None=None, band: int=13) -> dict | None:
    t = t.replace(minute=t.minute // 10 * 10, second=0, microsecond=0)
    w, s, e, n = bbox
    lat = np.arange(n - IR_GRID_DEG / 2, s, -IR_GRID_DEG)
    lon = np.arange(w + IR_GRID_DEG / 2, e, IR_GRID_DEG)
    LON, LAT = np.meshgrid(lon, lat)
    col, line = hsd_colline(LAT, LON, _H20)
    if not np.isfinite(line).all():
        return None
    per = _H20_LINES // HIMAWARI_SEGS
    s0, s1 = (int((np.nanmin(line) - 1) // per) + 1, int((np.nanmax(line) - 1) // per) + 1)
    out = np.full(LAT.shape, np.nan, np.float32)
    for seg in range(max(1, s0), min(HIMAWARI_SEGS, s1) + 1):
        try:
            h = _hsd_ir(bz2.decompress(_fetch(himawari_ir_url(t, seg, band), cache)))
        except Exception:
            return None
        c, l = hsd_colline(LAT, LON, h)
        ci, li = (np.rint(c).astype(int) - 1, np.rint(l).astype(int) - h['line0'])
        ok = (ci >= 0) & (ci < h['ncol']) & (li >= 0) & (li < h['nline'])
        out[ok] = hsd_bt(h['counts'][li[ok], ci[ok]], h)
    return {'bt': out, 'lat': lat, 'lon': lon, 't': t + timedelta(seconds=600.0 * float(np.nanmean(line)) / _H20_LINES)}

def goes_ir_scan(t: datetime, bbox, cache: str | None=None, band: int=13) -> dict | None:
    import xml.etree.ElementTree as ET
    import h5py
    import satcheck
    b = satcheck.bucket(t.strftime('%Y-%m-%d'))
    prefix = f'ABI-L1b-RadC/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/'
    keys = [x.text for x in ET.fromstring(satcheck._get(f'https://{b}.s3.amazonaws.com/?list-type=2&prefix={prefix}')).iter() if x.tag.endswith('Key') and x.text and (f'-M6C{band:02d}_' in x.text) or (x.tag.endswith('Key') and x.text and (f'-M3C{band:02d}_' in x.text))]
    best, bd = (None, None)
    for k in keys:
        m = re.search('_s(\\d{4})(\\d{3})(\\d{2})(\\d{2})(\\d{2})', k)
        st = datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1, hours=int(m[3]), minutes=int(m[4]), seconds=int(m[5]))
        d = abs((st - t).total_seconds())
        if bd is None or d < bd:
            best, bd = ((k, st), d)
    if not best or bd > 300:
        return None
    w, s, e, n = bbox
    with h5py.File(satcheck.RangeFile(f'https://{b}.s3.amazonaws.com/{best[0]}'), 'r') as f:
        a, bb, H, lon0 = satcheck._proj(f)
        X, Y = (f['x'], f['y'])
        xs = X[:] * satcheck._num(X.attrs['scale_factor']) + satcheck._num(X.attrs['add_offset'])
        ys = Y[:] * satcheck._num(Y.attrs['scale_factor']) + satcheck._num(Y.attrs['add_offset'])
        px, py = satcheck.latlon_to_xy(np.array([s, s, n, n, (s + n) / 2, (s + n) / 2, s, n]), np.array([w, e, w, e, w, e, (w + e) / 2, (w + e) / 2]), a, bb, H, lon0)
        i0, i1 = (max(0, np.searchsorted(xs, px.min()) - 2), np.searchsorted(xs, px.max()) + 2)
        j0, j1 = (max(0, np.searchsorted(-ys, -py.max()) - 2), np.searchsorted(-ys, -py.min()) + 2)
        R = f['Rad']
        raw = R[j0:j1, i0:i1].astype(np.float32)
        rad = raw * satcheck._num(R.attrs['scale_factor']) + satcheck._num(R.attrs['add_offset'])
        fill = R.attrs.get('_FillValue')
        if fill is not None:
            rad[raw == satcheck._num(fill)] = np.nan
        fk1, fk2, bc1, bc2 = (satcheck._num(f[k][()]) for k in ('planck_fk1', 'planck_fk2', 'planck_bc1', 'planck_bc2'))
        with np.errstate(all='ignore'):
            bt = (fk2 / np.log(fk1 / np.maximum(rad, 1e-06) + 1) - bc1) / bc2
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
        plat, plon = satcheck.xy_to_latlon(gx, gy, a, bb, H, lon0)
    lat = np.arange(n - IR_GRID_DEG / 2, s, -IR_GRID_DEG)
    lon = np.arange(w + IR_GRID_DEG / 2, e, IR_GRID_DEG)
    iy, ix = (np.floor((n - plat) / IR_GRID_DEG).astype(int), np.floor((plon - w) / IR_GRID_DEG).astype(int))
    ok = np.isfinite(plat) & (iy >= 0) & (iy < lat.size) & (ix >= 0) & (ix < lon.size)
    out = np.full((lat.size, lon.size), np.nan, np.float32)
    out[iy[ok], ix[ok]] = bt[ok]
    hole = ~np.isfinite(out)
    if hole.any():
        p = np.pad(out, 1, constant_values=np.nan)
        nb = np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx] for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))])
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[hole] = np.nanmean(nb, axis=0)[hole]
    return {'bt': out, 'lat': lat, 'lon': lon, 't': best[1] + timedelta(seconds=150)}

def ir_scan(sat: str, t: datetime, bbox, cache: str | None=None, band: int=13) -> dict | None:
    return himawari_ir_scan(t, bbox, cache, band) if sat == 'himawari' else goes_ir_scan(t, bbox, cache, band) if sat == 'goes' else None

def sun_elevation(t: datetime, lat, lon):
    doy = t.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (t.hour - 12) / 24)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    eqt = 229.18 * (7.5e-05 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    mins = t.hour * 60 + t.minute + t.second / 60
    ha = np.radians((mins + eqt + 4 * np.asarray(lon, float)) / 4 - 180)
    la = np.radians(np.asarray(lat, float))
    return np.degrees(np.arcsin(np.clip(np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha), -1, 1)))
