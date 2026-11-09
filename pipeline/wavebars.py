from __future__ import annotations
import math
import numpy as np
import sky
LEE_AGL = (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 7500.0, 9000.0)
CEIL_NONE_JS = 5990.0
WB = 14
CODE = sky.CLASS_CODE0 + WB - sky.CB
CLASSES = (sky.NONE, sky.BLUE, 2, 3, sky.SC, sky.ST)
LOW = 10.0
WSTAR = 1.6
TRAP = 0.5
LAM = (3000.0, 15000.0)
WIND = 7.0
LIFT = 0.12
CARRY, REACH, STEP = (90000.0, 150000.0, 10000.0)
M_DEG = 111320.0

def deck_top(ceiltop, ceil, cubase, cudepth, zi, zs):
    ct = np.where(np.isfinite(ceiltop) & np.isfinite(ceil) & (ceil < CEIL_NONE_JS), ceiltop, np.nan)
    cu = zs + cubase + np.nan_to_num(cudepth, nan=0.0)
    return np.where(np.isfinite(ct), ct, np.where(np.isfinite(cubase), cu, zs + zi))

def lift_at(col, agl):
    a = np.asarray(LEE_AGL)
    k = np.minimum(len(a) - 2, np.searchsorted(a[1:], np.nan_to_num(agl, nan=0.0), side='left'))
    t = np.clip((agl - a[k]) / (a[k + 1] - a[k]), 0.0, 1.0)
    rows = np.arange(len(k))
    return np.maximum(0.0, col[rows, k] * (1 - t) + col[rows, k + 1] * t)

def _js_round(x):
    return np.floor(x + 0.5)

def classes(cls, wstar):
    return np.where((cls == sky.CS) & (np.nan_to_num(wstar, nan=0.0) >= sky.CU_MIN_WS), sky.BLUE, cls)

def bars(cls, wstar, low, trap, lam, u, v, own, north, step):
    ny, nx = cls.shape
    tt = np.where(trap > 1, trap / 100.0, trap)
    U = np.hypot(u, v)
    with np.errstate(invalid='ignore'):
        cand = np.isin(cls, CLASSES) & ~(np.nan_to_num(wstar, nan=-1.0) >= WSTAR) & (tt >= TRAP) & (lam >= LAM[0]) & (lam <= LAM[1]) & (U >= WIND) & np.isfinite(own)
    iy, ix = np.nonzero(cand)
    if not iy.size:
        return cand
    c0 = cls[iy, ix]
    lo = low(iy, ix) if callable(low) else low[iy, ix]
    ok = ~np.isin(c0, (sky.NONE, sky.BLUE)) | (lo >= LOW)
    iy, ix = (iy[ok], ix[ok])
    best = own[iy, ix].copy()
    uu, vv, UU = (u[iy, ix], v[iy, ix], U[iy, ix])
    lat = north - iy * step
    kx = np.where(UU > 0, -uu / UU / (M_DEG * np.cos(lat * math.pi / 180) * step), 0.0)
    ky = np.where(UU > 0, vv / UU / (M_DEG * step), 0.0)
    s = STEP
    while s <= REACH:
        jx, jy = (_js_round(ix + kx * s).astype(np.int64), _js_round(iy + ky * s).astype(np.int64))
        inside = (jx >= 0) & (jy >= 0) & (jx < nx) & (jy < ny)
        w = np.where(inside, own[np.clip(jy, 0, ny - 1), np.clip(jx, 0, nx - 1)], 0.0)
        f = math.exp(-s / CARRY)
        k = np.isfinite(w)
        best[k] = np.maximum(best[k], w[k] * f)
        s += STEP
    out = np.zeros_like(cand)
    out[iy[best >= LIFT], ix[best >= LIFT]] = True
    return out

def median_at(a, iy, ix):
    p = np.pad(a, 1, constant_values=np.nan)
    st = np.stack([p[iy + dy, ix + dx] for dy in range(3) for dx in range(3)])
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            m = np.nanmedian(st, axis=0)
    return np.where(np.isfinite(a[iy, ix]), m, np.nan)

def apply(cuclass, dec, own, grid, fine):
    need = ('wtrap', 'wlam', 'u_wl', 'v_wl', 'wstar', 'cloudlow')
    if any((dec.get(n) is None for n in need)):
        return None
    c = np.asarray(cuclass, np.float64)
    cls = np.where(c < 255, np.where(c < sky.SCORE_CODES, np.floor(c / 50 + 0.5), sky.CB + c - sky.CLASS_CODE0), np.nan)
    ws = np.asarray(dec.get('wstar'), np.float64)
    lowf = np.asarray(dec.get('cloudlow'), np.float64)
    low = (lambda iy, ix: median_at(lowf, iy, ix)) if fine else lowf
    b = bars(classes(cls, ws), ws, low, np.asarray(dec.get('wtrap'), np.float64), np.asarray(dec.get('wlam'), np.float64), np.asarray(dec.get('u_wl'), np.float64), np.asarray(dec.get('v_wl'), np.float64), own, float(grid['north']), float(grid['step']))
    if not b.any():
        return None
    return np.where(b, CODE, cuclass).astype(np.uint8)
