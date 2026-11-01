from __future__ import annotations
import json
import math
import os
import numpy as np
from PIL import Image
RIDGE_COL_M = (600, 1000, 1500, 2000, 3000)
LEE_AGL = (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 7500.0, 9000.0)
NA = len(LEE_AGL)
RIDGE_LOW, RIDGE_LAYER, RIDGE_OVER = (300.0, 1750.0, 150.0)
WAVE_WIND = (9.0, 14.0)
LEE_MAX_W = 8.0
LEE_EX_C, LEE_EX_GAMMA = (0.397, 1.49)
LEE_BOOST = 0.15
RIDGE_XC_SCALE = 0.7
CELLS_A_CHUNK = 24000

def js_round(x):
    f = np.floor(x)
    return f + (x - f >= 0.5)

def _round1(x):
    f = math.floor(x)
    return f + (1 if x - f >= 0.5 else 0)

def _toward(u, v, dirs):
    th = np.fmod(np.arctan2(u, v) * 180.0 / math.pi + 360.0, 360.0)
    q = th / 360.0 * dirs
    fq = np.floor(q)
    d0 = fq.astype(np.int64) % dirs
    return (d0, (d0 + 1) % dirs, q - fq)

class _Area:

    def __init__(self, folder, name, a, tile, planes):
        g = a['grid']
        self.folder, self.name = (folder, name)
        self.west, self.north, self.step = (float(g['west']), float(g['north']), float(g['step']))
        self.nx, self.ny = (int(g['nx']), int(g['ny']))
        self.T = int(a['tile']) if a.get('tile') is not None else int(tile)
        self.P = planes
        self.ntx, self.nty = (-(-self.nx // self.T), -(-self.ny // self.T))
        self.listed = np.zeros((self.nty, self.ntx), bool)
        for k in a.get('tiles', []):
            tx, ty = (int(s) for s in k.split('_'))
            if 0 <= tx < self.ntx and 0 <= ty < self.nty:
                self.listed[ty, tx] = True
        self.loaded = np.zeros_like(self.listed)
        self.mosaic = None

    def _load(self, tx, ty):
        T, P = (self.T, self.P)
        path = os.path.join(self.folder, self.name, f'{tx}_{ty}.png')
        with Image.open(path) as im:
            if im.mode == 'P':
                im = im.convert('RGBA')
            elif im.mode not in ('L', 'LA', 'RGB', 'RGBA'):
                raise ValueError(f'{path}: unsupported PNG mode {im.mode}')
            data = np.asarray(im, np.uint8).reshape(-1)
        if data.size < P * T * T:
            raise ValueError(f'{path}: {data.size} bytes, {P * T * T} expected')
        c = data[:P * T * T].reshape(P, T, T).transpose(1, 2, 0)
        if self.mosaic is None:
            self.mosaic = np.zeros((self.ny, self.nx, P), np.uint8)
        y0, x0 = (ty * T, tx * T)
        y1, x1 = (min(self.ny, y0 + T), min(self.nx, x0 + T))
        self.mosaic[y0:y1, x0:x1] = c[:y1 - y0, :x1 - x0]
        self.loaded[ty, tx] = True

    def gather(self, iy, ix):
        ty, tx = (iy // self.T, ix // self.T)
        has = self.listed[ty, tx]
        need = has & ~self.loaded[ty, tx]
        if need.any():
            for t in np.unique(ty[need] * self.ntx + tx[need]):
                self._load(int(t % self.ntx), int(t // self.ntx))
        if self.mosaic is None:
            return (np.zeros((iy.size, self.P), np.uint8), has)
        return (self.mosaic[iy, ix], has)

class Exposure:

    def __init__(self, folder):
        with open(os.path.join(folder, 'index.json')) as fh:
            ix = json.load(fh)
        self.folder = folder
        self.dirs = int(ix['dirs'])
        self.scale = float(ix['scale'])
        self.tile = ix.get('tile')
        self.edges = ix.get('edges')
        self.nband = len(self.edges) - 1 if self.edges else 0
        self.top_step = float(ix['topStep']) if ix.get('topStep') is not None else 20.0
        self.relief_step = float(ix['reliefStep']) if ix.get('reliefStep') is not None else None
        planes = (self.nband * self.dirs if self.edges else self.dirs) + 1
        self.areas = [_Area(folder, name, a, self.tile, planes) for name, a in (ix.get('areas') or {}).items()]

    def locate(self, lat, lon):
        aid = np.full(lat.shape, -1, np.int64)
        fx = np.full(lat.shape, np.nan)
        fy = np.full(lat.shape, np.nan)
        for k, a in enumerate(self.areas):
            x = (lon - a.west) / a.step
            y = (a.north - lat) / a.step
            inside = (aid < 0) & (x >= -0.5) & (y >= -0.5) & (x <= a.nx - 0.5) & (y <= a.ny - 0.5)
            aid[inside], fx[inside], fy[inside] = (k, x[inside], y[inside])
        return (aid, fx, fy)

def _offsets(area, fx, fy, r):
    cx = js_round(fx).astype(np.int64)
    cy = js_round(fy).astype(np.int64)
    for dy in range(-r, r + 1):
        iy = cy + dy
        oky = (iy >= 0) & (iy < area.ny)
        for dx in range(-r, r + 1):
            ix = cx + dx
            ok = oky & (ix >= 0) & (ix < area.nx)
            yield (iy, ix, ok)

def _lee_boost(w):
    t = np.minimum(1.0, np.maximum(0.0, (w - 1.0) / 2.0))
    return w * (1.0 + LEE_BOOST * t * t * (3.0 - 2.0 * t))

def _lee(lee, lat, lon, step, lu, lv, bands, lok):
    n = lat.size
    top = np.full(n, np.nan)
    peak = np.zeros(n)
    kbest = np.zeros(n, np.int64)
    if lee is None:
        return (top, peak, kbest, np.zeros(n, bool))
    aid, fx, fy = lee.locate(lat, lon)
    D, NB = (lee.dirs, lee.nband)
    dec = (np.arange(256) / 254.0) ** 2 * lee.scale
    for k in np.unique(aid[aid >= 0]):
        a = lee.areas[k]
        sel = np.flatnonzero(aid == k)
        r = 0 if step <= a.step + 1e-09 else max(1, _round1((step / a.step - 1) / 2))
        wv = lok[sel] if lok is not None else np.zeros(sel.size, bool)
        if wv.any():
            u, v = (np.where(wv, lu[sel], 0.0), np.where(wv, lv[sel], 0.0))
            U = np.hypot(u, v)
            d0, d1, t = _toward(u, v, D)
            d0, d1, t = (np.where(U > 0, d0, 0), np.where(U > 0, d1, 1), np.where(U > 0, t, 0.0))
            B = bands if sel.size == n else bands[sel]
        tsum = np.zeros(sel.size)
        tn = np.zeros(sel.size, np.int64)
        bp = np.zeros(sel.size)
        bk = np.zeros(sel.size, np.int64)
        for iy, ix, ok in _offsets(a, fx[sel], fy[sel], r):
            w = np.flatnonzero(ok)
            if not w.size:
                continue
            c, has = a.gather(iy[w], ix[w])
            w, c = (w[has], c[has])
            if not w.size:
                continue
            tsum[w] += c[:, NB * D] * lee.top_step
            tn[w] += 1
            ww = wv[w]
            if not ww.any():
                continue
            w, c = (w[ww], c[ww])
            cb = c[:, :NB * D].reshape(-1, NB, D)
            c0 = np.take_along_axis(cb, d0[w, None, None], 2)[:, :, 0]
            c1 = np.take_along_axis(cb, d1[w, None, None], 2)[:, :, 0]
            t1 = t[w, None]
            m = dec[c0] * (1 - t1) + dec[c1] * t1
            some = m.any(axis=1)
            if not some.all():
                w, m = (w[some], m[some])
            if not w.size:
                continue
            Bt = B.transpose(1, 0, 2)[:, w]
            mt = np.ascontiguousarray(m.T)
            s = Bt[0] * mt[0, :, None]
            s *= s
            y = np.empty_like(s)
            for b in range(1, NB):
                np.multiply(Bt[b], mt[b, :, None], out=y)
                y *= y
                s += y
            col = np.minimum(LEE_MAX_W, _lee_boost(LEE_EX_C * s ** (LEE_EX_GAMMA / 2)))
            rows = np.arange(w.size)
            kk = np.argmax(col, axis=1)
            pk = col[rows, kk]
            better = pk > bp[w]
            bp[w[better]] = pk[better]
            bk[w[better]] = kk[better]
        got = tn > 0
        top[sel[got]] = tsum[got] / tn[got]
        peak[sel], kbest[sel] = (bp, bk)
    return (top, peak, kbest, aid >= 0)

def _ridge_wind(u3, v3, uwl, vwl, h, cols):
    nan = np.isnan
    have = np.zeros(h.shape, bool)
    for _, cu, cv in cols:
        have |= ~nan(cu) & ~nan(cv)
    br = have & ~nan(u3) & ~nan(v3) & (h > RIDGE_LOW)
    ur, vr = (u3.copy(), v3.copy())
    if br.any():
        az, au, av = (np.full(h.shape, RIDGE_LOW), u3.copy(), v3.copy())
        done = ~br
        for z, cu, cv in cols:
            p = ~nan(cu) & ~nan(cv)
            hit = ~done & p & (h <= z)
            t = (h - az) / np.maximum(1.0, z - az)
            ur = np.where(hit, au + t * (cu - au), ur)
            vr = np.where(hit, av + t * (cv - av), vr)
            done |= hit
            step = ~done & p
            az, au, av = (np.where(step, float(z), az), np.where(step, cu, au), np.where(step, cv, av))
        ur, vr = (np.where(~done, au, ur), np.where(~done, av, vr))
    rest = ~br & ~(nan(uwl) | nan(vwl) | ~(h > RIDGE_LOW))
    upper = rest & (nan(u3) | nan(v3))
    blend = rest & ~upper
    t = np.minimum(1.0, (h - RIDGE_LOW) / (RIDGE_LAYER - RIDGE_LOW))
    ur = np.where(upper, uwl, np.where(blend, u3 + t * (uwl - u3), ur))
    vr = np.where(upper, vwl, np.where(blend, v3 + t * (vwl - v3), vr))
    return (ur, vr)

def _ridge_lift(ridge, lat, lon, u, v, nlow, step):
    out = np.full(lat.size, np.nan)
    if ridge is None:
        return out
    ok = ~np.isnan(u) & ~np.isnan(v)
    aid, fx, fy = ridge.locate(lat, lon)
    aid = np.where(ok, aid, -1)
    U = np.hypot(u, v)
    out[aid >= 0] = 0.0
    live = (aid >= 0) & (U > 0.5)
    D = ridge.dirs
    fk = ridge.scale / 254 * RIDGE_XC_SCALE
    rs = ridge.relief_step
    for k in np.unique(aid[live]):
        a = ridge.areas[k]
        sel = np.flatnonzero(live & (aid == k))
        r = 0 if step <= 2 * a.step + 1e-09 else max(1, _round1((step / a.step - 1) / 2))
        Us, n = (U[sel], nlow[sel])
        d0, d1, t = _toward(u[sel], v[sel], D)
        nok = n > 0
        total = np.zeros(sel.size)
        for iy, ix, inside in _offsets(a, fx[sel], fy[sel], r):
            w = np.flatnonzero(inside)
            if not w.size:
                continue
            c, has = a.gather(iy[w], ix[w])
            rows = np.arange(w.size)
            tw = t[w]
            f = (c[rows, d0[w]] * (1 - tw) + c[rows, d1[w]] * tw) * fk
            H = c[:, D] * rs
            nn = n[w]
            fo = nok[w] & (H > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                fr = np.where(fo, np.maximum(0.05, np.minimum(1.0, Us[w] / (nn * H))), 1.0)
            add = has & (f != 0)
            total[w[add]] += (Us[w] * f * fr)[add]
        out[sel] = total / (2 * r + 1) ** 2
    return out

def _cells(f, lat, lon, step, lee, ridge, lat_bands):
    nan = np.full(lat.size, np.nan)
    g = lambda k: f.get(k, nan)
    zs = np.nan_to_num(g('elevation'), nan=0.0)
    lu = lv = B = lok = None
    if lat_bands is not None:
        lu, lv, B, lok = lat_bands
    top, peak, kbest, inside = _lee(lee, lat, lon, step, lu, lv, B, lok)
    u, v = (g('u_blavg'), g('v_blavg'))
    u3 = np.where(np.isnan(g('u_g300')), u, g('u_g300'))
    v3 = np.where(np.isnan(g('v_g300')), v, g('v_g300'))
    rh = np.where(np.isnan(top), RIDGE_LOW, top + RIDGE_OVER - zs)
    up = rh > RIDGE_LOW
    cols = []
    for m in RIDGE_COL_M:
        cu, cv = (g(f'u_g{m}'), g(f'v_g{m}'))
        cols.append((m, np.where(up, cu, np.nan), np.where(up, cv, np.nan)))
    ur, vr = _ridge_wind(u3, v3, g('u_wl'), g('v_wl'), rh, cols)
    real = _ridge_lift(ridge, lat, lon, ur, vr, g('nlow'), step)
    out = {'ridge': np.maximum(np.nan_to_num(real, nan=0.0), np.nan_to_num(g('ridge'), nan=0.0)), 'wave': np.zeros(lat.size), 'wave_agl': np.full(lat.size, np.nan), 'wave_top': np.full(lat.size, np.nan), 'crest': top - zs}
    if lat_bands is not None:
        on = lok & (peak > 0)
        U = np.hypot(lu, lv)
        ramp = np.maximum(0.0, np.minimum(1.0, (U - WAVE_WIND[0]) / (WAVE_WIND[1] - WAVE_WIND[0])))
        out['wave'] = np.where(on, peak * ramp, 0.0)
        alt = zs + np.asarray(LEE_AGL)[kbest]
        out['wave_agl'] = np.where(on, alt - zs, np.nan)
        out['wave_top'] = np.where(inside & lok, peak, np.nan)
    return out

def _grid(grid):
    get = grid.get if isinstance(grid, dict) else lambda k: getattr(grid, k)
    return (float(get('west')), float(get('north')), float(get('step')), int(get('nx')), int(get('ny')))

class _Rows:

    def __init__(self, fields, y0, y1):
        self.fields, self.y0, self.y1, self.done = (fields, y0, y1, {})

    def get(self, k, default=None):
        if k not in self.done:
            a = self.fields.get(k)
            self.done[k] = None if a is None else np.asarray(a[self.y0:self.y1], np.float64).reshape(-1)
        a = self.done[k]
        return default if a is None else a

def run_fields(fields, grid, lee, ridge, lat_bands=None, chunk=CELLS_A_CHUNK):
    west, north, step, nx, ny = _grid(grid)
    out = {k: np.empty((ny, nx)) for k in ('ridge', 'wave', 'wave_agl', 'wave_top', 'crest')}
    rows = max(1, chunk // max(1, nx))
    xs = np.arange(nx)
    for y0 in range(0, ny, rows):
        y1 = min(ny, y0 + rows)
        iy = np.arange(y0, y1)
        lat = np.repeat(north - iy * step, nx)
        lon = np.tile(west + xs * step, y1 - y0)
        f = _Rows(fields, y0, y1)
        lb = None
        if lat_bands is not None:
            lu, lv, B, ok = lat_bands(y0, y1) if callable(lat_bands) else (x[y0:y1] for x in lat_bands)
            n = (y1 - y0) * nx
            B = np.asarray(B, np.float64).reshape(n, -1)
            if lee is not None and B.shape[1] != lee.nband * NA:
                raise ValueError(f'lattice bands: {B.shape[1]} a cell, {lee.nband} × {NA} expected')
            lb = (np.asarray(lu, np.float64).reshape(-1), np.asarray(lv, np.float64).reshape(-1), B.reshape(n, -1, NA), np.asarray(ok, bool).reshape(-1))
        res = _cells(f, lat, lon, step, lee, ridge, lb)
        for k, a in res.items():
            out[k][y0:y1] = a.reshape(y1 - y0, nx)
    return out
