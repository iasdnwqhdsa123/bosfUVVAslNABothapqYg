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
ML_LEVELS = (600, 1000, 1500, 2000)
CLOUD_K, CLOUD_X0, CLOUD_W, CLOUD_MAX = (1.5, -0.3, 0.7, 0.95)
SUN_LOSS = 0.5

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
    best = carried(own, u, v, iy, ix, north, step)
    out = np.zeros_like(cand)
    out[iy[best >= LIFT], ix[best >= LIFT]] = True
    return out

def riding(trap, lam, u, v, own):
    tt = np.where(trap > 1, trap / 100.0, trap)
    with np.errstate(invalid='ignore'):
        return (tt >= TRAP) & (lam >= LAM[0]) & (lam <= LAM[1]) & (np.hypot(u, v) >= WIND) & np.isfinite(own)

def carried(own, u, v, iy, ix, north, step):
    ny, nx = own.shape
    best = own[iy, ix].copy()
    uu, vv = (u[iy, ix], v[iy, ix])
    UU = np.hypot(uu, vv)
    lat = north - iy * step
    kx = np.where(UU > 0, -uu / np.where(UU > 0, UU, 1) / (M_DEG * np.cos(lat * math.pi / 180) * step), 0.0)
    ky = np.where(UU > 0, vv / np.where(UU > 0, UU, 1) / (M_DEG * step), 0.0)
    s = STEP
    while s <= REACH:
        jx, jy = (_js_round(ix + kx * s).astype(np.int64), _js_round(iy + ky * s).astype(np.int64))
        inside = (jx >= 0) & (jy >= 0) & (jx < nx) & (jy < ny)
        w = np.where(inside, own[np.clip(jy, 0, ny - 1), np.clip(jx, 0, nx - 1)], 0.0)
        f = math.exp(-s / CARRY)
        k = np.isfinite(w)
        best[k] = np.maximum(best[k], w[k] * f)
        s += STEP
    return best

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

def ml_spread(dpd, zi):
    zi = np.asarray(zi, np.float64)
    best, at = (np.full(zi.shape, np.nan), np.full(zi.shape, float(ML_LEVELS[0])))
    zz = np.nan_to_num(zi, nan=-np.inf)
    for m in ML_LEVELS:
        d = dpd.get(m)
        if d is None:
            continue
        d = np.asarray(d, np.float64)
        ok = np.isfinite(d) & (~np.isfinite(best) | (zz >= m))
        best, at = (np.where(ok, d, best), np.where(ok, float(m), at))
    with np.errstate(invalid='ignore'):
        return np.maximum(0.0, best - 8 * np.maximum(0.0, np.nan_to_num(zi, nan=0.0) - at) / 1000)

def cover(lift, spread):
    return CLOUD_MAX / (1 + np.exp(-(CLOUD_K * lift - spread - CLOUD_X0) / CLOUD_W))

def cloud(dec, own, grid):
    need = ('wtrap', 'wlam', 'u_wl', 'v_wl', 'cloudlow', 'zi')
    if any((dec.get(n) is None for n in need)):
        return None
    u, v = (np.asarray(dec.get('u_wl'), np.float64), np.asarray(dec.get('v_wl'), np.float64))
    ws = dec.get('wstar')
    with np.errstate(invalid='ignore'):
        cand = riding(np.asarray(dec.get('wtrap'), np.float64), np.asarray(dec.get('wlam'), np.float64), u, v, own)
        if ws is not None:
            cand &= ~(np.nan_to_num(np.asarray(ws, np.float64), nan=-1.0) >= WSTAR)
    low = np.asarray(dec.get('cloudlow'), np.float64)
    cand &= np.isfinite(low)
    iy, ix = np.nonzero(cand)
    if not iy.size:
        return None
    lift = carried(own, u, v, iy, ix, float(grid['north']), float(grid['step']))
    at = lambda a: None if a is None else np.asarray(a, np.float64)[iy, ix]
    spread = ml_spread({m: at(dec.get(f'dpd_g{m}')) for m in ML_LEVELS}, at(dec.get('zi')))
    with np.errstate(invalid='ignore'):
        c = np.where((lift >= LIFT) & np.isfinite(spread), 100 * cover(lift, np.nan_to_num(spread, nan=99.0)), 0.0)
        gain = c > low[iy, ix]
    if not gain.any():
        return None
    out = low.copy()
    out[iy[gain], ix[gain]] = c[gain]
    return out
