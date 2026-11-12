from __future__ import annotations
import math
from datetime import timedelta
import numpy as np
R_EARTH = 6371000.0
BAND_MIN = 45.0
SUN_SET = -0.83
LOW_M, MID_M, HIGH_M = (1500.0, 4500.0, 9000.0)
LOW_BAND, MID_BAND, TOP_M = (2800.0, 6500.0, 11000.0)
STEP_KM = 25.0
VIEW_KM = (40.0, 60.0, 120.0)
BLOCK_KM = (40.0, 60.0, 100.0)
CLEAR = 0.22
LIT = (0.35, 0.85, 1.0)
HIDE = 0.9
HAZE = (1.0, 4.0, 0.6)
CU_VIEW_KM = 40.0
CU_TOP_M, CU_LIT = ((1500.0, 3000.0, 6000.0), (0.25, 0.55, 0.9))
CU_SCATTER = ((0.0, 0.1, 0.25, 0.5, 0.8, 1.0), (0.0, 0.6, 1.0, 1.0, 0.45, 0.25))
CU_BLOCK = 0.25

def sun_position(valid, lat, lon):
    doy = valid.timetuple().tm_yday
    b = 2 * math.pi * (doy - 81) / 364
    eot = (9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)) / 60
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    hour = valid.hour + valid.minute / 60 + valid.second / 3600 + lon / 15 + eot
    ha = np.radians(15 * (hour - 12))
    la = np.radians(lat)
    sin_el = np.clip(np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha), -1, 1)
    el = np.arcsin(sin_el)
    az = np.arctan2(-np.sin(ha) * math.cos(decl), np.cos(la) * math.sin(decl) - np.sin(la) * math.cos(decl) * np.cos(ha))
    return (np.degrees(el), np.degrees(az) % 360)

def band(valid, lat, lon):
    d = timedelta(minutes=BAND_MIN)
    e0, a0 = sun_position(valid - d, lat, lon)
    e1, a1 = sun_position(valid + d, lat, lon)
    mask = ((e0 - SUN_SET) * (e1 - SUN_SET) <= 0) & (e0 != e1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.clip((e0 - SUN_SET) / (e0 - e1), 0, 1)
    da = (a1 - a0 + 180) % 360 - 180
    return (mask, (a0 + np.nan_to_num(f) * da) % 360)

def glow(lat, lon, valid, low, mid, high, rain=None, spread=None, grid_km=3.0, cu=None, cutop=None):
    shape = lat.shape
    out = np.full(shape, np.nan, np.float32)
    mask, az = band(valid, lat, lon)
    mask &= np.isfinite(low) | np.isfinite(mid) | np.isfinite(high)
    if not mask.any():
        return out
    iy, ix = np.nonzero(mask)
    from scipy import ndimage
    r = max(1, int(round(STEP_KM / max(grid_km, 0.1))))
    cov = [ndimage.uniform_filter(np.nan_to_num(np.asarray(c, dtype=np.float32)) / 100.0, size=r, mode='nearest') for c in (low, mid, high)]
    cuc = cut = None
    if cu is not None:
        cuv = np.clip(np.nan_to_num(np.asarray(cu, dtype=np.float32)), 0, 1)
        top = np.zeros(shape, np.float32) if cutop is None else np.nan_to_num(np.asarray(cutop, dtype=np.float32))
        cuc = ndimage.uniform_filter(cuv, size=r, mode='nearest')
        cut = ndimage.uniform_filter(cuv * top, size=r, mode='nearest')
    deck = cov[0] if cuc is None else np.maximum(cov[0] - cuc, 0)
    block = cov if cuc is None else [deck + CU_BLOCK * np.minimum(cuc, cov[0] + cuc), cov[1], cov[2]]
    dlat = float(lat[1, 0] - lat[0, 0]) if shape[0] > 1 else -grid_km / 111.2
    dlon = float(lon[0, 1] - lon[0, 0]) if shape[1] > 1 else grid_km / 111.2
    la, a = (np.radians(lat[iy, ix]), np.radians(az[iy, ix]))
    sy = np.cos(a) / 111.2 / dlat
    sx = np.sin(a) / (111.2 * np.maximum(np.cos(la), 0.05)) / dlon

    def along(c, km):
        y = np.clip(np.rint(iy + sy * km).astype(np.int64), 0, shape[0] - 1)
        x = np.clip(np.rint(ix + sx * km).astype(np.int64), 0, shape[1] - 1)
        return c[y, x]

    def canvas(c, reach):
        n = int(reach // STEP_KM) + 1
        return sum((along(c, i * STEP_KM) for i in range(n))) / n

    def light(own, h):
        tangent = math.sqrt(2 * R_EARTH * h) / 1000.0
        far = tangent + math.sqrt(2 * R_EARTH * TOP_M) / 1000.0
        tau = 0.0
        d = STEP_KM
        while d <= far:
            zr = ((d - tangent) * 1000.0) ** 2 / (2 * R_EARTH)
            k = 0 if zr < LOW_BAND else 1 if zr < MID_BAND else 2
            if k < own or d > tangent:
                tau = tau + along(block[k], d) * STEP_KM / BLOCK_KM[k]
            d += STEP_KM
        return np.exp(-tau)
    seen = [canvas(c, VIEW_KM[k]) for k, c in enumerate((deck, cov[1], cov[2]))]
    lit = [seen[k] * light(k, h) * LIT[k] for k, h in enumerate((LOW_M, MID_M, HIGH_M))]
    seen_cu = cu_lit = 0.0
    if cuc is not None:
        seen_cu = canvas(cuc, CU_VIEW_KM)
        tops = canvas(cut, CU_VIEW_KM) / np.maximum(seen_cu, 0.001)
        w = np.clip((tops - CU_TOP_M[0]) / (CU_TOP_M[2] - CU_TOP_M[0]), 0, 1)
        reach = light(0, LOW_M) * (1 - w) + light(1, MID_M) * w
        cu_lit = np.interp(seen_cu, *CU_SCATTER) * np.interp(tops, CU_TOP_M, CU_LIT) * reach
    colour = 1 - (1 - lit[0]) * (1 - lit[1]) * (1 - lit[2])
    covered = 1 - (1 - seen[0]) * (1 - seen[1]) * (1 - seen[2]) * (1 - seen_cu)
    score = 1 - (1 - np.maximum(CLEAR * (1 - covered), colour)) * (1 - cu_lit)
    over = deck[iy, ix]
    score = score * (1 - HIDE * over) + lit[0] * HIDE * over
    if rain is not None:
        score = score * np.where(np.nan_to_num(np.asarray(rain)[iy, ix]) >= 0.5, 0.5, 1.0)
    if spread is not None:
        sp = np.nan_to_num(np.asarray(spread)[iy, ix], nan=HAZE[1])
        score = score * (HAZE[2] + (1 - HAZE[2]) * np.clip((sp - HAZE[0]) / (HAZE[1] - HAZE[0]), 0, 1))
    out[iy, ix] = np.clip(100.0 * score, 0, 100)
    return out
