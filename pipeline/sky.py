from __future__ import annotations
import warnings
import numpy as np
NONE, BLUE, CB, SHOWERS, NS, FOG, ST, SC, AC, AS, CS = (0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13)
LOW_TOP = 2000.0
BROKEN, OVERCAST = (56.25, 85.0)
CU_MIN_WS = 0.5
FEW_CU = 25.0
SC_SPREAD = 75.0
ACCAS_MIN = 25.0
THICK_SUN, THIN_SUN = (0.45, 0.6)
CONV_DBZ, CONV_DBZ_RAIN, CONV_RAIN, SHOWERS_SHARE = (45.0, (40.0, 2.5), 6.0, 0.5)
HEAVY_RAIN = 10.0
CEIL_NONE_JS = 5990.0
SC_CELL_KM = 1.8
SCORE_CODES, CLASS_CODE0 = (225, 225)
COVER_MAX = 0.95
FIELDS = ('cuclass', 'cusky', 'cucover', 'cuwide', 'cueff')
INPUTS = ('wstar', 'cudepth', 'overdev', 'rain', 'refc', 'showers', 'sun', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ceil', 'vis', 'accas', 'depth', 'zi', 'cubase')
COVERS = ('cloud', 'cloudlow', 'cloudmid', 'cloudhigh')

def decode(code, lo, hi, transform):
    c = np.arange(255, dtype=np.float64)
    if transform == 'sqrt':
        a, b = (np.sqrt(max(lo, 0)), np.sqrt(hi))
        t = (a + c / 254 * (b - a)) ** 2
    else:
        t = lo + c / 254 * (hi - lo)
    lut = np.append(t.astype(np.float32), np.float32(np.nan)).astype(np.float64)
    return lut[np.asarray(code, dtype=np.uint8)]

def area_median(a):
    p = np.pad(a, 1, constant_values=np.nan)
    ny, nx = a.shape
    stack = np.stack([p[dy:dy + ny, dx:dx + nx] for dy in range(3) for dx in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        m = np.nanmedian(stack, axis=0)
    return np.where(np.isfinite(a), m, np.nan)
FINE_STEP = 0.09
CLASS_MEAN = 5
SMOOTHED = ('wstar', 'cudepth', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'sun')

def window_sum(a, k):
    r = k // 2
    p = np.pad(a, r)
    ny, nx = a.shape
    return sum((p[dy:dy + ny, dx:dx + nx] for dy in range(k) for dx in range(k)))

def area_mean(a, k=CLASS_MEAN):
    ok = np.isfinite(a)
    s, n = (window_sum(np.where(ok, a, 0.0), k), window_sum(ok.astype(np.float64), k))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ok, s / n, np.nan)
CU_SHARE = 0.25

def cu_area_depth(d, k=CLASS_MEAN):
    ok = np.isfinite(d)
    cu = ok & (d > 0)
    n, n_cu, s_cu = (window_sum(x, k) for x in (ok.astype(np.float64), cu.astype(np.float64), np.where(cu, d, 0.0)))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ok, np.where(n_cu >= CU_SHARE * n, s_cu / n_cu, 0.0), np.nan)

def js_round(x):
    return np.floor(x + 0.5)

def clamp01(x):
    return np.clip(x, 0, 1)

def cu_score(w, d, od):
    within = lambda f: 0.4 * (clamp01(f) - 0.5)
    s = np.where(d < 800, 2 + within(d / 800), np.where(d < 2500, 3 + within((d - 800) / 1700), 4 + within((d - 2500) / 2000)))
    s = np.where(d == 0, 1.0, s)
    s = np.where(w < CU_MIN_WS, w / CU_MIN_WS * 0.49, s)
    s = np.where(od >= 50, 4.5 + np.minimum(0.49, (od - 50) / 100), s)
    return np.where(np.isfinite(w) & np.isfinite(d) & np.isfinite(od), s, np.nan)

def sky_layers(low, mid, high, total, ceil, sun):
    split = np.isfinite(low) | np.isfinite(mid) | np.isfinite(high)
    c = np.where(np.isfinite(ceil) & (ceil < CEIL_NONE_JS), ceil, np.nan)
    tot = np.nan_to_num(total, nan=0.0)
    has_c = np.isfinite(c)
    lo = np.where(split, np.nan_to_num(low, nan=0.0), np.where(has_c & (c < LOW_TOP), tot, 0.0))
    mi = np.where(split, np.nan_to_num(mid, nan=0.0), np.where(has_c & (c >= LOW_TOP), tot, 0.0))
    up = split & has_c & (lo >= BROKEN) & (mi < BROKEN) & (c >= LOW_TOP * 1.25)
    dn = split & has_c & (mi >= BROKEN) & (lo < BROKEN) & (c < LOW_TOP) & ~up
    lo, mi = (np.where(dn, mi, np.where(up, 0.0, lo)), np.where(up, lo, np.where(dn, 0.0, mi)))
    hi = np.where(split, np.nan_to_num(high, nan=0.0), np.where(~has_c, tot, 0.0))
    thick = np.where(np.isfinite(sun), sun < THICK_SUN, (mi >= BROKEN) & (hi >= BROKEN))
    return (c, lo, mi, hi, thick)

def deck_of(mid, upper, thick, sun):
    aloft = 100 * (1 - (1 - clamp01(mid / 100)) * (1 - clamp01(upper / 100)))
    thin_sun = np.isfinite(sun) & (sun < THIN_SUN)
    out = np.zeros(mid.shape, np.int16)
    out = np.where((aloft >= BROKEN) & (mid < BROKEN) & (upper < BROKEN), np.where(thick, AS, np.where(mid >= upper, AC, CS)), out)
    out = np.where((upper >= OVERCAST) | (upper >= BROKEN) & (thick | thin_sun), np.where(thick, AS, CS), out)
    out = np.where(mid >= BROKEN, np.where(thick, AS, AC), out)
    out = np.where(mid >= OVERCAST, AS, out)
    return out

def convective_rain(overdev, cudepth, refc, rain, showers, total, low, mid, sun=None):
    m = np.nan_to_num(mid, nan=0.0)
    shield = np.zeros(np.shape(total), bool) if sun is None else (np.nan_to_num(total, nan=0.0) >= OVERCAST) & (m >= BROKEN) & np.isfinite(sun) & (np.nan_to_num(sun, nan=1.0) < THICK_SUN)
    no_deck = np.isfinite(total) & ~(np.nan_to_num(low, nan=0.0) >= OVERCAST) & ~(m >= OVERCAST) & ~shield
    r = np.nan_to_num(rain, nan=0.0)
    by_refc = (refc >= CONV_DBZ) | (refc >= CONV_DBZ_RAIN[0]) & (r >= CONV_DBZ_RAIN[1])
    out = np.where(np.isfinite(refc), by_refc, r >= CONV_RAIN)
    out = np.where(np.isfinite(showers), showers >= SHOWERS_SHARE, out)
    out = out | no_deck | (r >= HEAVY_RAIN)
    return out | (np.nan_to_num(overdev, nan=0.0) >= 30) | (np.nan_to_num(cudepth, nan=0.0) >= 2500)
CU_OVERSHOOT, CU_OVERSHOOT_MIN = (0.5, 500.0)
CU_COVER = (0.2, 0.25)

def cu_reach(zi, cubase):
    over = (cubase > 0) & (zi > 0) & (cubase > zi)
    with np.errstate(invalid='ignore', divide='ignore'):
        e = np.minimum(1, (cubase - zi) / np.maximum(CU_OVERSHOOT * zi, CU_OVERSHOOT_MIN))
    return np.where(over, 1 - e, 1.0)

def cu_cover(cudepth, overdev, reach=1.0):
    c = (CU_COVER[0] + CU_COVER[1] * clamp01(cudepth / 3000)) * clamp01(reach)
    c = np.where(overdev > 30, np.maximum(c, 0.4 + 0.3 * clamp01((overdev - 30) / 50)), c)
    return np.where(cudepth > 0, c, 0.0)
CU_ZI, CU_WIDE, CU_MIN = (1.0, 1.3, 1.0)

def cumulus_width(D, zi, m=1.0):
    ratio = np.where(D <= 0.5, 2.0, np.where(D <= 1, 2.5 - D, np.where(D <= 2, 2 - 0.5 * D, np.maximum(0.6, 1 - 0.15 * (D - 2)))))
    return np.maximum(CU_MIN, np.minimum(5, np.maximum(CU_ZI * zi * (0.6 + 0.5 * m), D * ratio * (1 + (CU_WIDE - 1) * np.clip((3.5 - D) / 1.5, 0, 1)))))

def classify(v):
    w, d, od = (v['wstar'], v['cudepth'], v['overdev'])
    low, mid, high, total, ceil, sun, vis = (v['cloudlow'], v['cloudmid'], v['cloudhigh'], v['cloud'], v['ceil'], v['sun'], v['vis'])
    score = cu_score(w, d, od)
    cu = js_round(np.nan_to_num(score, nan=0.0))
    c, lo, mi, hi, thick = sky_layers(low, mid, high, total, ceil, sun)
    upper = hi
    deck = deck_of(mi, upper, thick, sun)
    ws = np.nan_to_num(w, nan=0.0)
    stirred = ws >= 0.7
    has_c = np.isfinite(c)
    fog = np.isfinite(vis) & (vis < 1000) & ((lo >= BROKEN) | has_c & (c < 300)) & ~stirred | has_c & (c < 150) & (lo >= BROKEN)
    stratus = ~(stirred | (cu >= 2)) & np.where(has_c, c < 600, lo >= OVERCAST)
    st_sc = np.where(stratus, ST, SC)
    cu_shown = (cu >= 2) & ((cu >= 4) | (lo >= FEW_CU) & ~(thick & (np.maximum(mi, upper) >= BROKEN)))
    accas = np.nan_to_num(v['accas'], nan=-1.0)
    cls = cu.copy()
    cls = np.where((cu <= 0) & (lo >= FEW_CU) & (lo < BROKEN) & (~has_c | (c < LOW_TOP * 1.25)), SC, cls)
    cls = np.where((accas >= ACCAS_MIN) & (mi >= FEW_CU), AC, cls)
    cls = np.where(deck > 0, deck, cls)
    cls = np.where(cu_shown, cu, cls)
    cls = np.where((lo >= BROKEN) & ((cu <= 2) | (cu <= 3) & (lo >= SC_SPREAD)), st_sc, cls)
    cls = np.where((lo >= OVERCAST) & ~(cu >= 4), st_sc, cls)
    cls = np.where(fog, FOG, cls)
    cls = np.where(cu >= CB, CB, cls)
    cls = np.where(np.isfinite(score), cls, np.nan)
    r = np.nan_to_num(v['rain'], nan=0.0)
    conv = convective_rain(od, d, v['refc'], v['rain'], v['showers'], total, low, mid, sun)
    wet = np.where(conv, SHOWERS, np.where((r < 1) & has_c & (c < 600) & (mi < BROKEN), ST, NS))
    over = deck_of(mi, upper, thick, sun)
    some = np.isfinite(d) | np.isfinite(low)
    reach = cu_reach(v['zi'], v['cubase'])
    cover = np.minimum(COVER_MAX, np.maximum(np.where(np.isfinite(d), cu_cover(d, np.nan_to_num(od, nan=0.0), reach), 0.0), np.nan_to_num(low, nan=0.0) / 100))
    cover = np.where(some, cover, np.nan)
    zi = np.where(np.isfinite(v['depth']), v['depth'], 1200.0)
    wkm = np.where(np.isfinite(d), cumulus_width(np.maximum(0.2, d / 1000), np.maximum(0.3, zi / 1000), 1.0), np.nan)
    return (cls, score, wet, over, cover, wkm)
COVER_WEIGHTS = (1.0, 0.7, 0.4)
SUN_SLACK = 0.2

def effective_cover(low, mid, high, total, cu, ceil=None, sun=None):
    f = lambda c, k: 1 - k * clamp01(np.nan_to_num(c, nan=0.0) / 100)
    lay = np.isfinite(low) | np.isfinite(mid) | np.isfinite(high)
    lo = np.maximum(np.nan_to_num(low, nan=0.0), 100 * np.nan_to_num(cu, nan=0.0))
    mh = 1 - f(mid, COVER_WEIGHTS[1]) * f(high, COVER_WEIGHTS[2])
    if sun is not None:
        mh = np.where(np.isfinite(sun), np.minimum(mh, clamp01(1 - np.nan_to_num(sun, nan=1.0)) + SUN_SLACK), mh)
    eff = 100 * (1 - f(lo, COVER_WEIGHTS[0]) * (1 - mh))
    c = np.full(np.shape(total), np.nan) if ceil is None else ceil
    k = np.where(~np.isfinite(c) | (c >= CEIL_NONE_JS), COVER_WEIGHTS[2], np.where(c < LOW_TOP, COVER_WEIGHTS[0], COVER_WEIGHTS[1]))
    tot = 100 * (1 - (1 - clamp01(np.nan_to_num(cu, nan=0.0))) * f(total, k))
    return np.where(lay, eff, np.where(np.isfinite(total), tot, np.nan))

def encode(cls, score, wet, over):
    ok = np.isfinite(cls)
    c = np.nan_to_num(cls, nan=0.0).astype(np.int16)
    s = np.clip(js_round(np.nan_to_num(score, nan=0.0) * 50), c * 50 - 25, c * 50 + 24)
    code = np.where(c < CB, np.clip(s, 0, SCORE_CODES - 1), CLASS_CODE0 + c - CB)
    cuclass = np.where(ok, code, 255).astype(np.uint8)
    wi = np.select([wet == SHOWERS, wet == NS], [0, 1], 2)
    oi = np.select([over == AC, over == AS, over == CS], [1, 2, 3], 0)
    return (cuclass, (wi * 4 + oi).astype(np.uint8))

def _codes(a, lo, hi):
    return np.where(np.isfinite(a), np.round((np.clip(a, lo, hi) - lo) / (hi - lo) * 254), 255).astype(np.uint8)

def cloud_overview(q, enc, fine, smooth=True):
    shape = q['wstar'].shape if 'wstar' in q else None
    if shape is None or any((n not in q for n in ('wstar', 'cudepth', 'overdev'))):
        return {}
    v = {}
    for n in INPUTS:
        v[n] = decode(q[n], *enc[n]) if n in q else np.full(shape, np.nan)
    if fine:
        for n in COVERS:
            v[n] = area_median(v[n])
    cls, score, wet, over, cover, wkm = classify(v)
    if fine and smooth:
        cls, score, wet, _, _, _ = classify({**v, **{n: cu_area_depth(v[n]) if n == 'cudepth' else area_mean(v[n]) for n in SMOOTHED}})
    cuclass, cusky = encode(cls, score, wet, over)
    out = {'cuclass': cuclass, 'cusky': cusky, 'cucover': _codes(cover, *enc['cucover'][:2]), 'cuwide': _codes(wkm, *enc['cuwide'][:2])}
    if 'cueff' in enc:
        out['cueff'] = _codes(effective_cover(v['cloudlow'], v['cloudmid'], v['cloudhigh'], v['cloud'], cover, v['ceil'], v['sun']), *enc['cueff'][:2])
    off = q['wstar'] == 255
    for a in out.values():
        a[off] = 255
    return out
