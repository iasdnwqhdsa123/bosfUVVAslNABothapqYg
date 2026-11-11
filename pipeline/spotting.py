from __future__ import annotations
import numpy as np
LEN, VIR, HALO, FLU, MAM, CAV, SEA = (1, 2, 4, 8, 16, 32, 64)
ARC, PIL, CTR, VOL = (1, 2, 4, 8)
G = 9.81
LIFT_PER_K = 125.0
P_TOP = 250.0
LEN_WIND = 8.0
LEN_SPREAD = 5.0
LEN_N2 = 2e-05
LEN_COVER = (60.0, 40.0)
LEN_AGL = 1200.0
LEN_RELIEF = (250.0, 40.0)
LEN_NEAR_KM = 12.0
LEN_TERRAIN = 0.6
LEN_RAIN = 0.3
N_MIN = 0.006
LEN_GRID_WAVE, LEN_WAVE_MAX, LEN_COARSE_KM = (6.0, 60000.0, 8.0)
VIR_DBZ, VIR_LOW_DBZ, VIR_RAIN, VIR_SFC_SPREAD = (10.0, 5.0, 0.1, 3.0)
VIR_DECK = (70.0, 2500.0, -8.0, 10.0)
HALO_HIGH, HALO_SUN, HALO_LOW, HALO_MID, HALO_EL = (60.0, 0.5, 30.0, 40.0, 5.0)
FLU_RI, FLU_SHEAR, FLU_AGL = (0.6, 0.008, 1000.0)
FLU_COVER = (20.0, 95.0)
MAM_KM, MAM_ANVIL, MAM_T, MAM_DRY, MAM_RAIN = (60.0, 70.0, -20.0, 8.0, 2.0)
STORM_DBZ = 45.0
CAV_T = (-38.0, -12.0)
CAV_COVER, CAV_ABOVE, CAV_TRAFFIC = (60.0, 40.0, 0.05)
SEA_LOW, SEA_ABOVE, SEA_KM, SEA_MID, SEA_HIGH, SEA_CEIL = (85.0, 150.0, 25.0, 50.0, 70.0, 2500.0)
ARC_KM, ARC_GUST, ARC_SPREAD = (15.0, 15.0, 8.0)
PIL_OD, PIL_SPREAD, PIL_AGL, PIL_COVER = (40.0, 2.5, (3500.0, 9000.0), 60.0)
VOL_BOX, VOL_MONTHS = ((136.5, -20.5, 142.5, -14.0), (8, 9, 10, 11))
VOL_HOURS, VOL_SPREAD, VOL_WIND, VOL_LOW, VOL_INV = ((4.0, 9.0), 3.0, 6.0, 60.0, 1.0)
CTR_COVER = 0.03
SEEN_LOW = 80.0

def _theta(t, p):
    return (t + 273.15) * (1000.0 / p) ** 0.2854

def _near(a, cells, fn):
    return fn(a, size=2 * cells + 1, mode='nearest') if cells > 0 else a

def _cover_at(cc, p):
    if p in cc:
        return cc[p]
    ps = sorted(cc, key=lambda q: abs(q - p))[:2]
    if not ps:
        return None
    a = np.stack([cc[q] for q in ps])
    n = np.isfinite(a).sum(axis=0)
    return np.where(n > 0, np.nansum(a, axis=0) / np.maximum(n, 1), np.nan)

def spot(levels, lv, zs, s, out, relief, grid_km, el=None, traffic=None, contrail=None, lat=None, lon=None, valid=None):
    from scipy import ndimage
    order = [i for i in np.argsort(-np.asarray(levels, dtype=np.float64)) if levels[i] >= P_TOP]
    p = np.asarray([float(levels[i]) for i in order])
    z, t, td, u, v = (np.stack([np.asarray(lv[k][i], dtype=np.float32) for i in order]) for k in ('z', 't', 'td', 'u', 'v'))
    w = None if lv.get('w') is None else np.stack([np.nan_to_num(np.asarray(lv['w'][i], dtype=np.float32)) for i in order])
    shape = zs.shape
    zero = np.zeros(shape, np.float32)
    f = lambda a, d=0.0: np.full(shape, d, np.float32) if a is None else np.nan_to_num(np.asarray(a, dtype=np.float32), nan=d)
    rain, od, low, mid, high = (f(out.get('rain')), f(out.get('overdev')), f(out.get('cloudlow')), f(out.get('cloudmid')), f(out.get('cloudhigh')))
    inside = np.isfinite(s['t2'])
    cc_by = {q: out[f'cc{q}'] for q in (1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200) if out.get(f'cc{q}') is not None}
    cc = np.stack([f(_cover_at(cc_by, q)) for q in p]) if cc_by else np.zeros(z.shape, np.float32)
    agl = z - zs[None]
    ok = np.isfinite(agl) & (agl >= 300) & np.isfinite(t) & np.isfinite(td)
    dpd = np.where(ok, np.maximum(t - td, 0), np.inf)
    th = _theta(t, p[:, None, None])
    spd = np.hypot(u, v)
    dz = np.maximum(z[1:] - z[:-1], 50.0)
    n2 = G / (0.5 * (th[1:] + th[:-1])) * (th[1:] - th[:-1]) / dz
    sh = np.hypot(u[1:] - u[:-1], v[1:] - v[:-1]) / dz
    with np.errstate(divide='ignore', invalid='ignore'):
        ri = n2 / np.maximum(sh * sh, 1e-10)
    n2k = np.concatenate([n2[:1], 0.5 * (n2[1:] + n2[:-1]), n2[-1:]])
    seen = low < SEEN_LOW
    cells = lambda km: int(round(km / max(grid_km, 0.1)))
    L = len(p)
    dry = _near(rain, cells(LEN_NEAR_KM), ndimage.maximum_filter) < LEN_RAIN
    hills = _near(np.nan_to_num(relief), cells(LEN_RELIEF[1]), ndimage.maximum_filter)
    zi = f(out.get('zi'))
    n_low = np.sqrt(np.maximum(n2k[min(1, L - 1)], N_MIN ** 2))
    eta_ground = LEN_TERRAIN * np.minimum(_near(np.nan_to_num(relief), cells(LEN_NEAR_KM), ndimage.maximum_filter), spd[min(1, L - 1)] / n_low)
    lent = np.zeros(shape, bool)
    for k in range(L):
        n = np.sqrt(np.maximum(n2k[k], N_MIN ** 2))
        eta = eta_ground
        if w is not None:
            wind = np.maximum(spd[k], 5.0)
            lam = np.minimum(np.maximum(2 * np.pi * wind / n, LEN_GRID_WAVE * grid_km * 1000), LEN_WAVE_MAX)
            eta_w = _near(np.abs(w[k]), cells(LEN_NEAR_KM), ndimage.maximum_filter) * lam / (2 * np.pi * wind)
            eta = np.maximum(eta_w, eta_ground) if grid_km >= LEN_COARSE_KM else eta_w
        patchy = _near(cc[k], cells(LEN_COVER[1]), ndimage.uniform_filter) < LEN_COVER[0]
        here = ok[k] & (agl[k] >= np.maximum(LEN_AGL, zi + 300)) & (spd[k] >= LEN_WIND) & (n2k[k] >= LEN_N2) & patchy
        lent |= here & (dpd[k] <= LEN_SPREAD) & (eta >= LIFT_PER_K * dpd[k]) & (seen | (agl[k] < 2000))
    lent &= dry & (hills >= LEN_RELIEF[0])
    refc, refd1 = (s.get('refc'), s.get('refd1'))
    spread2 = np.nan_to_num(s['t2'] - s['td2'])
    if refc is not None:
        lowest = f(refd1) if refd1 is not None else zero
        virga = (f(refc) >= VIR_DBZ) & (lowest < VIR_LOW_DBZ) & (rain < VIR_RAIN) & (spread2 >= VIR_SFC_SPREAD)
    else:
        deck = ok & (cc >= VIR_DECK[0]) & (agl >= VIR_DECK[1])
        thick = (deck[1:] & deck[:-1]).any(axis=0) if L > 1 else np.zeros(shape, bool)
        cold = (deck & (t <= VIR_DECK[2])).any(axis=0)
        base = np.where(deck, agl, np.inf).min(axis=0)
        under = ok & (agl < base[None]) & (agl >= 500)
        driest = np.where(under, dpd, -np.inf).max(axis=0)
        virga = thick & cold & (driest >= VIR_DECK[3]) & (rain < VIR_RAIN)
    sun = f(out.get('sun'), 1.0)
    day = np.ones(shape, bool) if el is None else np.asarray(el) >= HALO_EL
    halo = day & (high >= HALO_HIGH) & (sun >= HALO_SUN) & (low < HALO_LOW) & (mid < HALO_MID) & (rain < VIR_RAIN)
    edge = lambda c: (c >= FLU_COVER[0]) & (c <= FLU_COVER[1])
    solid = (cc[1:] > FLU_COVER[1]) & (cc[:-1] > FLU_COVER[1])
    flu = (ok[1:] & ok[:-1] & (agl[:-1] >= FLU_AGL) & (n2 > 0) & (sh >= FLU_SHEAR) & (ri < FLU_RI) & (edge(cc[1:]) | edge(cc[:-1])) & ~solid).any(axis=0) if L > 1 else np.zeros(shape, bool)
    flu &= seen & (rain < LEN_RAIN)
    core = f(refc) >= STORM_DBZ if refc is not None else (od >= 70) & (rain >= 2.0)
    storm_near = _near(core.astype(np.uint8), cells(MAM_KM), ndimage.maximum_filter) > 0
    anvil = ok & (cc >= MAM_ANVIL) & (t <= MAM_T)
    anvil_base = np.where(anvil, agl, np.inf).min(axis=0)
    under = ok & (agl < anvil_base[None]) & (agl >= 2000)
    mam = storm_near & anvil.any(axis=0) & (np.where(under, dpd, -np.inf).max(axis=0) >= MAM_DRY) & (rain < MAM_RAIN)
    gust = f(s.get('gust'))
    wind10 = np.hypot(f(s.get('u10')), f(s.get('v10')))
    arc = (_near(core.astype(np.uint8), cells(ARC_KM), ndimage.maximum_filter) > 0) & ((gust >= ARC_GUST) | (gust - wind10 >= ARC_SPREAD))
    cav = np.zeros(shape, bool)
    if traffic is not None:
        for k in range(L):
            above = cc[k + 1] if k + 1 < L else zero
            cav |= ok[k] & (cc[k] >= CAV_COVER) & (t[k] >= CAV_T[0]) & (t[k] <= CAV_T[1]) & (above < CAV_ABOVE)
        cav &= (np.nan_to_num(traffic) >= CAV_TRAFFIC) & (rain < VIR_RAIN) & seen
    sea = np.zeros(shape, bool)
    top, ceil = (out.get('ceiltop'), out.get('ceil'))
    if top is not None and ceil is not None:
        deck = (low >= SEA_LOW) & np.isfinite(top) & (f(ceil, 1000000000.0) < SEA_CEIL) & (mid < SEA_MID) & (high < SEA_HIGH) & (rain < VIR_RAIN)
        r = cells(SEA_KM)
        deck_top = np.where(deck, np.nan_to_num(top), np.inf)
        peaks = _near(zs, r, ndimage.maximum_filter)
        sea = deck & (peaks >= deck_top + SEA_ABOVE) | (zs >= _near(deck_top, r, ndimage.minimum_filter) + SEA_ABOVE)
    moist = ok & (agl >= PIL_AGL[0]) & (agl <= PIL_AGL[1]) & (dpd <= PIL_SPREAD) & (cc < PIL_COVER) & (n2k > 0)
    pil = (od >= PIL_OD) & moist.any(axis=0)
    ctr = np.zeros(shape, bool) if contrail is None else (np.nan_to_num(contrail) >= CTR_COVER) & seen & (mid < SEEN_LOW)
    vol = np.zeros(shape, bool)
    if lat is not None and lon is not None and (valid is not None) and (valid.month in VOL_MONTHS):
        box = (lon >= VOL_BOX[0]) & (lon <= VOL_BOX[2]) & (lat >= VOL_BOX[1]) & (lat <= VOL_BOX[3])
        if box.any():
            solar = (valid.hour + valid.minute / 60 + lon / 15) % 24
            k1 = min(1, L - 1)
            warm = np.where(ok, t, -np.inf)[:k1 + 2].max(axis=0) - np.nan_to_num(s['t2'])
            vol = box & (solar >= VOL_HOURS[0]) & (solar <= VOL_HOURS[1]) & (spread2 <= VOL_SPREAD) & (wind10 <= VOL_WIND) & (low < VOL_LOW) & (rain < VIR_RAIN) & (warm >= VOL_INV)
    s1 = LEN * lent + VIR * virga + HALO * halo + FLU * flu + MAM * mam + CAV * cav + SEA * sea
    s2 = ARC * arc + PIL * pil + CTR * ctr + VOL * vol
    return {'spot1': np.where(inside, s1, np.nan).astype(np.float32), 'spot2': np.where(inside, s2, np.nan).astype(np.float32)}
