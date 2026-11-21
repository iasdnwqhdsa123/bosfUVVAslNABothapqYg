from __future__ import annotations
import math
import numpy as np
import sky
LEE_AGL = (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 7500.0, 9000.0)
CEIL_NONE_JS = 5990.0
WB = 14
CODE = sky.CLASS_CODE0 + WB - sky.CB
CLASSES = (sky.NONE, sky.BLUE, 2, 3, sky.SC, sky.ST)
DECK = 40.0
THIN = (10.0, 0.4, 4.5)
RAIN = 0.2
DEEP = (50.0, 50.0)
WET, WET_AGL = ((80.0, 1.0), 3000)
WSTAR = 1.6
BARS_ZI = 1000.0
TRAP = 0.5
LAM = (3000.0, 15000.0)
WIND = 7.0
LIFT = 0.12
OWN, LID = (0.15, 1.0)
CARRY, REACH, STEP = (90000.0, 150000.0, 10000.0)
M_DEG = 111320.0
ML_LEVELS = (600, 1000, 1500, 2000)
CLOUD_K, CLOUD_X0, CLOUD_W, CLOUD_MAX = (1.5, -0.3, 0.7, 0.95)
SUN_LOSS = 0.5
RES_STEP = 0.05
RES_HALF, RES_DS = (24.0, 3.0)
RES_LAM = (9.0, 10.5, 12.0, 13.5, 15.0, 17.0, 19.0, 21.5, 24.0)
RES_ACROSS = (-12.0, -6.0, 6.0, 12.0)
RES_AMP, RES_FRAC, RES_COH, RES_RUNS = (25.0, 0.5, 0.5, 2)
RES_BOX, RES_TRAIN = (5, 40)

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

def bars(cls, wstar, low, trap, lam, u, v, own, north, step, lid=None, zi=None, spread=None, rain=None, mid=None, high=None, dry=None):
    ny, nx = cls.shape
    tt = np.where(trap > 1, trap / 100.0, trap)
    U = np.hypot(u, v)
    with np.errstate(invalid='ignore'):
        trapped = (tt >= TRAP) & (lam >= LAM[0]) & (lam <= LAM[1])
        lidded = np.zeros_like(trapped) if lid is None else (np.nan_to_num(own, nan=0.0) >= OWN) & (np.nan_to_num(lid, nan=-np.inf) >= LID)
        cand = np.isin(cls, CLASSES) & ~(np.nan_to_num(wstar, nan=-1.0) >= WSTAR) & (U >= WIND) & np.isfinite(own) & (trapped | lidded)
        if zi is not None:
            cand &= ~(np.nan_to_num(zi, nan=-1.0) >= BARS_ZI)
    iy, ix = np.nonzero(cand)
    if not iy.size:
        return cand
    lo = low(iy, ix) if callable(low) else low[iy, ix]
    with np.errstate(invalid='ignore'):
        ok = lo >= THIN[0]
    iy, ix, lo = (iy[ok], ix[ok], lo[ok])
    best = carried(own, u, v, iy, ix, north, step)
    at = lambda a: a[iy, ix]
    with np.errstate(invalid='ignore'):
        ok = lo >= DECK
        if spread is not None:
            ok |= (best >= THIN[1]) & (at(spread) <= THIN[2])
        if rain is not None:
            ok &= ~(at(rain) >= RAIN)
        if mid is not None and high is not None:
            ok &= ~((at(mid) >= DEEP[0]) & (at(high) >= DEEP[1]))
        if mid is not None and dry is not None:
            ok &= ~((at(mid) >= WET[0]) & (at(dry) < WET[1]))
    yes = ok & (trapped[iy, ix] & (best >= LIFT) | lidded[iy, ix])
    out = np.zeros_like(cand)
    out[iy[yes], ix[yes]] = True
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

def _box_sum(a, r):
    ny, nx = a.shape
    t = np.zeros((ny + 1, nx + 1), np.float64)
    t[1:, 1:] = np.cumsum(np.cumsum(a, 0), 1)
    y0, y1 = (np.clip(np.arange(ny) - r, 0, ny), np.clip(np.arange(ny) + r + 1, 0, ny))
    x0, x1 = (np.clip(np.arange(nx) - r, 0, nx), np.clip(np.arange(nx) + r + 1, 0, nx))
    return t[y1][:, x1] - t[y0][:, x1] - t[y1][:, x0] + t[y0][:, x0]

def _box_range(a, r):
    from scipy.ndimage import maximum_filter, minimum_filter
    ok = np.isfinite(a)
    hi = maximum_filter(np.where(ok, a, -np.inf), size=2 * r + 1, mode='constant', cval=-np.inf)
    lo = minimum_filter(np.where(ok, a, np.inf), size=2 * r + 1, mode='constant', cval=np.inf)
    with np.errstate(invalid='ignore'):
        return np.where(np.isfinite(hi), hi - lo, -np.inf)

def resolved(low, u, v, north, step, chunk=100000):
    ny, nx = low.shape
    out = np.zeros((ny, nx), bool)
    if step > RES_STEP:
        return out
    U = np.hypot(u, v)
    with np.errstate(invalid='ignore'):
        cand = np.isfinite(low) & (U >= WIND) & (_box_range(low, RES_BOX) >= RES_AMP)
    iy_all, ix_all = np.nonzero(cand)
    if not iy_all.size:
        return out
    n = int(round(2 * RES_HALF / RES_DS)) + 1
    ss = np.array([-RES_HALF + RES_DS * k for k in range(n)])
    hann = np.array([0.5 - 0.5 * math.cos(2 * math.pi * k / (n - 1)) for k in range(n)])
    w2 = float(sum((h * h for h in hann))) * 0.5
    tt = float(sum((s * s for s in ss)))
    flag = np.zeros((ny, nx), bool)
    for c0 in range(0, iy_all.size, chunk):
        iy, ix = (iy_all[c0:c0 + chunk], ix_all[c0:c0 + chunk])
        uu, vv, UU = (u[iy, ix], v[iy, ix], U[iy, ix])
        ex, ey = (uu / UU, vv / UU)
        kx = 1000.0 / (M_DEG * np.cos((north - iy * step) * math.pi / 180) * step)
        ky = 1000.0 / (M_DEG * step)

        def prof(c):
            P = np.empty((n, iy.size))
            for k in range(n):
                jx = _js_round(ix + (ex * ss[k] - ey * c) * kx).astype(np.int64)
                jy = _js_round(iy - (ey * ss[k] + ex * c) * ky).astype(np.int64)
                inside = (jx >= 0) & (jy >= 0) & (jx < nx) & (jy < ny)
                P[k] = np.where(inside, low[np.clip(jy, 0, ny - 1), np.clip(jx, 0, nx - 1)], np.nan)
            ok = np.isfinite(P)
            cnt = ok.sum(0)
            tot = np.zeros(iy.size)
            for k in range(n):
                tot = tot + np.where(ok[k], P[k], 0.0)
            m = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
            return np.where(ok, P - m, 0.0)
        pa = prof(0.0)
        b = np.zeros(iy.size)
        for k in range(n):
            b = b + pa[k] * ss[k]
        b = b / tt
        for k in range(n):
            pa[k] = pa[k] - ss[k] * b
        ea = np.zeros(iy.size)
        for k in range(n):
            ea = ea + pa[k] * pa[k]
        ea = ea / n
        best = np.zeros(iy.size)
        for lam in RES_LAM:
            a, bb = (np.zeros(iy.size), np.zeros(iy.size))
            for k in range(n):
                a = a + pa[k] * (math.cos(2 * math.pi * ss[k] / lam) * hann[k])
                bb = bb + pa[k] * (math.sin(2 * math.pi * ss[k] / lam) * hann[k])
            best = np.maximum(best, (a * a + bb * bb) / w2)
        frac = best / np.maximum(ea, 1e-06)
        amp = np.sqrt(2 * ea)
        hi, lo = (pa > amp / 2, pa < -amp / 2)
        crests = hi[0].astype(int) + (hi[1:] & ~hi[:-1]).sum(0)
        gaps = lo[0].astype(int) + (lo[1:] & ~lo[:-1]).sum(0)
        keep = (amp >= RES_AMP) & (frac >= RES_FRAC) & (crests >= RES_RUNS) & (gaps >= RES_RUNS)
        if not keep.any():
            continue
        iy, ix, ex, ey, kx, pa = (iy[keep], ix[keep], ex[keep], ey[keep], kx[keep], pa[:, keep])
        coh = np.zeros(iy.size)
        sp = np.zeros(iy.size)
        for k in range(n):
            sp = sp + pa[k] * pa[k]
        for c in RES_ACROSS:
            q = prof(c)
            sq, spq = (np.zeros(iy.size), np.zeros(iy.size))
            for k in range(n):
                sq = sq + q[k] * q[k]
                spq = spq + pa[k] * q[k]
            coh = coh + spq / np.sqrt(np.maximum(sp * sq, 1e-06))
        coh = coh / len(RES_ACROSS)
        flag[iy, ix] = coh >= RES_COH
    return _box_sum(flag.astype(np.float64), RES_BOX) >= RES_TRAIN

def _deck(lo, at, rain, mid, high, dry):
    with np.errstate(invalid='ignore'):
        ok = lo >= DECK
        if rain is not None:
            ok &= ~(at(rain) >= RAIN)
        if mid is not None and high is not None:
            ok &= ~((at(mid) >= DEEP[0]) & (at(high) >= DEEP[1]))
        if mid is not None and dry is not None:
            ok &= ~((at(mid) >= WET[0]) & (at(dry) < WET[1]))
    return ok

def apply(cuclass, dec, own, grid, fine):
    need = ('wtrap', 'wlam', 'u_wl', 'v_wl', 'wstar', 'cloudlow')
    if any((dec.get(n) is None for n in need)):
        return None
    c = np.asarray(cuclass, np.float64)
    cls = np.where(c < 255, np.where(c < sky.SCORE_CODES, np.floor(c / 50 + 0.5), sky.CB + c - sky.CLASS_CODE0), np.nan)
    ws = np.asarray(dec.get('wstar'), np.float64)
    lowf = np.asarray(dec.get('cloudlow'), np.float64)
    low = (lambda iy, ix: median_at(lowf, iy, ix)) if fine else lowf
    arr = lambda a: None if a is None else np.asarray(a, np.float64)
    zi = arr(dec.get('zi'))
    spread = ml_spread({m: arr(dec.get(f'dpd_g{m}')) for m in ML_LEVELS}, np.full(lowf.shape, np.nan) if zi is None else zi)
    b = bars(classes(cls, ws), ws, low, np.asarray(dec.get('wtrap'), np.float64), np.asarray(dec.get('wlam'), np.float64), np.asarray(dec.get('u_wl'), np.float64), np.asarray(dec.get('v_wl'), np.float64), own, float(grid['north']), float(grid['step']), arr(dec.get('dth_m500')), zi, spread=spread, rain=arr(dec.get('rain')), mid=arr(dec.get('cloudmid')), high=arr(dec.get('cloudhigh')), dry=arr(dec.get(f'dpd_g{WET_AGL}')))
    uw, vw = (np.asarray(dec.get('u_wl'), np.float64), np.asarray(dec.get('v_wl'), np.float64))
    train = resolved(lowf, uw, vw, float(grid['north']), float(grid['step']))
    iy, ix = np.nonzero(train & np.isin(classes(cls, ws), CLASSES) & np.isfinite(uw) & np.isfinite(vw) & ~b)
    if iy.size:
        lo = low(iy, ix) if callable(low) else low[iy, ix]
        ok = _deck(lo, lambda a: a[iy, ix], arr(dec.get('rain')), arr(dec.get('cloudmid')), arr(dec.get('cloudhigh')), arr(dec.get(f'dpd_g{WET_AGL}')))
        b = b.copy()
        b[iy[ok], ix[ok]] = True
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
