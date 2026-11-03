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
COVER_BINS = [0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.7, 1.01]
STREET_WIND, STREET_W, STREET_RATIO = (5.5, 1.0, 0.25)
KIND_RAIN, KIND_OVERDEV, KIND_OVERCAST = (0.5, 30.0, 90.0)
STREET_ZI = 2.5
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
    wind = np.hypot(v['u_blavg'], v['v_blavg'])
    w = v['wstar']
    streets = (wind >= STREET_WIND) & (w >= STREET_W) & (w <= STREET_RATIO * wind) & ~(np.nan_to_num(v['rain']) >= KIND_RAIN) & ~(np.nan_to_num(od) >= KIND_OVERDEV) & ~(np.nan_to_num(low) >= KIND_OVERCAST)
    clear_above = (np.nan_to_num(v['cloudmid']) < MIDHIGH_MAX) & (np.nan_to_num(v['cloudhigh']) < MIDHIGH_MAX) & ~(np.nan_to_num(v['rain']) >= 0.1)
    return {'cover': cover, 'streets': streets, 'clear_above': clear_above, 'u': v['u_blavg'], 'v': v['v_blavg'], 'zi': v['zi'], 'wstar': w, 'cu': np.isfinite(d) & (np.nan_to_num(d) > 0)}

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

def box_day(gdir: str, log=print) -> dict | None:
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
    out = {'model': meta.get('model'), 'day': meta.get('day'), 'tz': meta.get('tz'), 'region': meta.get('region'), 'bins': {'fc': [0.0] * nb, 'obs': [0.0] * nb, 'n': [0] * nb}, 'by_hour': {}, 'blocks': [], 'cells': 0}
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
                            out['blocks'].append({'h': hr, 'aniso': s['aniso'], 'axis': s['axis'], 'spacing': s['spacing'], 'cover': round(s['cover'], 3), 'fc_streets': bool(fc['streets'][j, i]), 'fc_cover': round(float(fc['cover'][j, i]), 3), 'wind': round(math.hypot(u, v), 1), 'wind_axis': round(wind_axis(u, v), 1), 'zi': None if not np.isfinite(fc['zi'][j, i]) else round(float(fc['zi'][j, i])), 'w': None if not np.isfinite(fc['wstar'][j, i]) else round(float(fc['wstar'][j, i]), 2)})
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
        B = [b for r in rs for b in r['blocks']]
        if B:
            obs_st = [b for b in B if b['aniso'] >= ANISO]
            tp = sum((b['fc_streets'] for b in obs_st))
            fp = sum((b['fc_streets'] and b['aniso'] < ANISO for b in B))
            fn = len(obs_st) - tp
            tn_ = len(B) - len(obs_st) - fp
            L += ['', f'## {model}: streets', '', f'{len(B)} blocks of {STREET_KM:g} km with {COVER_RANGE[0]:.0%}–{COVER_RANGE[1]:.0%} cumulus; lines when the spectrum leans ≥ {ANISO}.', '', '| | satellite: lines | satellite: no lines |', '|---|---|---|', f'| forecast: streets | {tp} | {fp} |', f'| forecast: no streets | {fn} | {tn_} |']
            hit = tp / max(1, tp + fn)
            far = fp / max(1, tp + fp)
            L.append(f"\nCaught {hit:.0%} of the satellite's lines; {far:.0%} of the forecast's street blocks showed none.")
            al = [axis_diff(b['axis'], b['wind_axis']) for b in obs_st if b['wind'] >= 3]
            if al:
                L.append(f"Lines against the layer's wind (≥ 3 m/s): median {np.median(al):.0f}° apart, {np.mean(np.array(al) <= 20):.0%} within 20°.")
            sp = [b['spacing'] / (b['zi'] / 1000) for b in obs_st if b['spacing'] and b['zi'] and (b['zi'] > 300) and (axis_diff(b['axis'], b['wind_axis']) <= 30)]
            if sp:
                L.append(f'Spacing over zi where they lie along the wind: median {np.median(sp):.2f} (the placed cumulus: {STREET_ZI}), quartiles {np.percentile(sp, 25):.2f}–{np.percentile(sp, 75):.2f}, {len(sp)} blocks.')
            wl = [b['wind'] for b in obs_st]
            wn = [b['wind'] for b in B if b['aniso'] < ANISO]
            if wl and wn:
                L.append(f"Layer wind: {np.median(wl):.1f} m/s median under lines, {np.median(wn):.1f} without (the forecast's rule: ≥ {STREET_WIND}).")
    return '\n'.join(L) + '\n'

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
                    rec = box_day(gdir, log)
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
