from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
import numpy as np
from concurrent.futures import Future, ThreadPoolExecutor
from scipy.spatial import cKDTree
import fronts
import grib
import physics
import physics as P
import pack
import runfields
import decks
import sky
import spotting
import sunset
import tiles
from sources import OPEN_AHEAD_H, OPEN_SLACK_H, SOURCES, candidate_runs, latest_complete_run, needed_times
from tiles import downsample, downsample_at_peak, downsample_peak
REGIONS = {'hrrr': (-125.0, 24.0, -66.5, 50.0, 0.03), 'gfs': (-170.0, -54.75, 179.75, 70.0, 0.25), 'ecmwf': (-170.0, -54.9, 179.9, 70.0, 0.1), 'icon': (-170.0, -54.875, 179.75, 70.0, 0.125), 'icond2': (-3.94, 43.18, 20.34, 58.08, 0.02), 'arome': (-12.0, 37.5, 16.0, 55.4, 0.025)}
KEEP = {'gfs': ((-130.0, 20.25, -52.0, 56.0), (-168.0, 59.0, -130.25, 70.0), (-163.0, 54.5, -148.0, 59.0), (-137.0, 54.5, -130.25, 59.0), (-25.0, 34.0, 44.75, 70.0), (-13.0, 27.0, 12.0, 34.0), (-76.0, -54.75, -53.0, -16.0), (-53.0, -34.0, -34.0, -2.0), (-82.0, -16.0, -66.0, 12.5), (11.0, -35.0, 36.0, -16.0), (166.0, -48.0, 179.0, -34.0), (114.0, -35.5, 120.5, -28.5), (131.5, -26.0, 136.5, -21.5), (144.0, -27.0, 154.0, -18.5), (136.0, -39.5, 154.0, -27.0), (143.5, -44.0, 148.75, -39.5), (72.0, 26.0, 97.0, 37.0), (24.0, 16.0, 63.5, 34.0), (44.75, 34.0, 63.5, 40.0), (97.0, 21.0, 123.0, 42.0), (128.5, 30.0, 146.0, 46.0))}
KEEP['icon'] = KEEP['ecmwf'] = KEEP['gfs']
FRONTS_BOXES = {'gfs': ((-170.0, 15.0, 50.0, 70.0),)}
FRONTS_BOXES['ecmwf'] = FRONTS_BOXES['gfs']

def fronts_cover(boxes, fboxes):
    inside = lambda b, f: b[0] >= f[0] - 1e-06 and b[1] >= f[1] - 1e-06 and (b[2] <= f[2] + 1e-06) and (b[3] <= f[3] + 1e-06)
    return [any((inside(b, f) for f in fboxes)) for b in boxes]

def keep_mask(src_id, lat2d, lon2d):
    boxes = KEEP.get(src_id)
    if not boxes:
        return None
    m = np.zeros(lat2d.shape, bool)
    for w, s, e, n in boxes:
        m |= (lon2d >= w) & (lon2d <= e) & (lat2d >= s) & (lat2d <= n)
    return m

def published_mask(src_id, lat2d, lon2d, fields):
    keep, t2 = (keep_mask(src_id, lat2d, lon2d), fields.get('t2'))
    have = None if t2 is None else t2 != 255 if t2.dtype == np.uint8 else np.isfinite(t2)
    if have is None or have.all():
        return keep
    return have if keep is None else keep & have

def boxes_windows(region, boxes, halo=1):
    lons, lats = target_grid(region)
    out = []
    for w, s, e, n in boxes:
        xs = np.nonzero((lons >= w - 1e-06) & (lons <= e + 1e-06))[0]
        ys = np.nonzero((lats >= s - 1e-06) & (lats <= n + 1e-06))[0]
        if not len(xs) or not len(ys):
            continue
        y0, y1 = (max(0, ys[0] - halo), min(len(lats), ys[-1] + 1 + halo))
        x0, x1 = (max(0, xs[0] - halo), min(len(lons), xs[-1] + 1 + halo))
        out.append(((y0, y1, x0, x1), (ys[0] - y0, ys[-1] + 1 - y0, xs[0] - x0, xs[-1] + 1 - x0)))
    return out

def keep_windows(src_id, region, halo=1):
    boxes = KEEP.get(src_id)
    if not boxes:
        return None
    return boxes_windows(region, boxes, halo)

def _crop(a, y0, y1, x0, x1, shape):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and (a.shape[-2:] == shape):
        return np.ascontiguousarray(a[..., y0:y1, x0:x1])
    if isinstance(a, tuple):
        return tuple((_crop(x, y0, y1, x0, x1, shape) for x in a))
    if isinstance(a, list):
        return [_crop(x, y0, y1, x0, x1, shape) for x in a]
    if isinstance(a, dict):
        return {k: _crop(x, y0, y1, x0, x1, shape) for k, x in a.items()}
    return a

def paste(windows, grids, shape):
    if windows is None:
        return grids[0]
    out = None
    for ((y0, y1, x0, x1), (iy0, iy1, ix0, ix1)), v in zip(windows, grids):
        if v is None:
            continue
        if out is None:
            out = np.full(shape, np.nan, np.float32)
        out[y0 + iy0:y0 + iy1, x0 + ix0:x0 + ix1] = v[iy0:iy1, ix0:ix1]
    return out

def analyse_parts(parts, coords, grid_step, valid, windows, shape):
    done = [analyse(s, lv, levels, grid_step, lat2d, lon2d, valid) for (s, lv, levels), (lat2d, lon2d) in zip(parts, coords)]
    names = dict.fromkeys((k for part in done for k in part))
    return {k: paste(windows, [part.get(k) for part in done], shape) for k in names}

def paste_codes(windows, grids, shape):
    if windows is None:
        return grids[0]
    out = None
    for ((y0, y1, x0, x1), (iy0, iy1, ix0, ix1)), v in zip(windows, grids):
        if v is None:
            continue
        if out is None:
            out = np.full(shape, 255, np.uint8)
        out[y0 + iy0:y0 + iy1, x0 + ix0:x0 + ix1] = v[iy0:iy1, ix0:ix1]
    return out

def analyse_codes(parts, coords, grid_step, valid, windows, shape):
    done = []
    for (s, lv, levels), (lat2d, lon2d) in zip(parts, coords):
        d = analyse(s, lv, levels, grid_step, lat2d, lon2d, valid)
        done.append({k: quantize(v, *ENC[k]) for k, v in d.items() if v is not None and k in ENC})
        del d
    names = dict.fromkeys((k for part in done for k in part))
    return {k: paste_codes(windows, [part.get(k) for part in done], shape) for k in names}

def part_coords(regrids):
    return [np.meshgrid(rg.lats, rg.lons, indexing='ij') for rg in regrids]

def analyse_kept(s, lv, levels, grid_step, lat2d, lon2d, valid, windows):
    if windows is None:
        return analyse(s, lv, levels, grid_step, lat2d, lon2d, valid)
    shape = lat2d.shape
    cut = [lambda a, w=w: _crop(a, *w, shape) for w, _ in windows]
    return analyse_parts([(c(s), c(lv), levels) for c in cut], [(c(lat2d), c(lon2d)) for c in cut], grid_step, valid, windows, shape)
TILE = {'hrrr': 100, 'gfs': 100, 'ecmwf': 250, 'icon': 200, 'icond2': 100, 'arome': 100}
OVERVIEW = {'hrrr': (2, 4, 8), 'icond2': (2, 4, 8), 'arome': (2, 4, 8)}
LEVEL_TILE = 200
DECODE_THREADS = 4
CLASS_FIELDS = {'ptype', 'fltcat', 'sinkby', 'cuclass', 'cusky', 'spot1', 'spot2'}
WIND_ALTS_FT = [2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 18000, 24000, 30000]
WAVE_ALTS_FT = [8000, 10000, 12000, 14000, 18000, 24000]
SAMPLED_FIELDS = {f'wv{ft}' for ft in WAVE_ALTS_FT} | {'wlow', 'vis', 'cucover', 'cuwide', 'cueff'}
PEAK_FIELDS = {'overdev', 'refc', 'showers', 'wrun', 'wtop'}
AT_PEAK = {'wrunz': 'wrun'}
ENC = {'wstar': (0, 6, None), 'zi': (0, 6000, None), 'depth': (0, 6000, None), 'cubase': (0, 7000, None), 'cudepth': (0, 12000, 'sqrt'), 'blapse': (0, 15, None), 'twidth': (0, 1000, None), 'overdev': (0, 100, None), 'bs': (0, P.BS_MAX, None), 'cape': (0, 6000, 'sqrt'), 'rain': (0, 50, 'sqrt'), 'showers': (0, 1, None), 'sun': (0, 1.2, None), 'cloud': (0, 100, None), 'cloudlow': (0, 100, None), 'cloudmid': (0, 100, None), 'cloudhigh': (0, 100, None), 't2': (-40, 50, None), 'td2': (-45, 35, None), 'sst': (-2, 36, None), 'snowd': (0, 3, 'sqrt'), 'soilm': (0, 1, None), 'mslp': (950, 1060, None), 'heat': (0, 800, None), 'densalt': (-1000, 7000, None), 'gust': (0, 50, None), 'wavemax': (0, 8, None), 'wavemaxz': (0, 12000, None), 'nlow': (0, 0.04, None), 'wlow': (-4, 4, None), 'wtrap': (0, 100, None), 'wlam': (0, 60000, None), 'conv': (-2, 2, None), 'sink': (0, 6, None), 'sinkth': (0, 3, None), 'sinkby': (0, 254, None), 'ridge': (0, 4, None), 'bumpy': (0, 1, None), 'pm25': (0, 300, 'sqrt'), 'elevation': (-100, 6000, None), 'ceil': (0, P.CEIL_NONE, 'sqrt'), 'vis': (0, 16100, 'sqrt'), 'fltcat': (0, 254, None), 'ptype': (0, 254, None), 'slr': (P.SLR_RANGE[0], P.SLR_RANGE[1], None), 'snowshare': (0, 1, None), 'frzl': (-100, 6500, None), 'icing': (0, 100, None), 'icebase': (0, 8000, None), 'icetop': (0, 8000, None), 'turb': (0, 25, None), 'turbz': (0, 11000, None), 'turbbase': (0, 11000, None), 'turbtop': (0, 11000, None), 'llws': (0, 40, None), 'ctop': (0, 16000, None), 'refc': (0, 75, None), 'lowbase': (0, 6000, None), 'rowstr': (0, 1, None), 'rowdir': (0, 180, None), 'rowgap': (0, 40, None), 'wdeck': (0, 3, None), 'dtop': (0, 3000, None), 'spot1': (0, 254, None), 'spot2': (0, 254, None), 'glow': (0, 100, None), 'ceiltop': (0, 16000, None), 'accas': (0, 100, None), 'acscore': (0, 100, None), 'acbase': (0, 8000, None), 'accape': (0, 2000, 'sqrt'), 'accin': (0, 300, None), 'cuclass': (0, 254, None), 'cusky': (0, 254, None), 'cucover': (0, 1, None), 'cuwide': (0, 5, None), 'cueff': (0, 100, None), 'u_cuw': (-60, 60, None), 'v_cuw': (-60, 60, None), 'curh': (0, 1, None), 'u_wl': (-60, 60, None), 'v_wl': (-60, 60, None), 'u_twt': (-60, 60, None), 'v_twt': (-60, 60, None), 'u_twa': (-60, 60, None), 'v_twa': (-60, 60, None), 'rrun': (0, 8, 'sqrt'), 'wrun': (0, 8, 'sqrt'), 'wrunz': (0, 12700, None), 'crest': (-2540, 2540, None), 'wtop': (0, (254 * 0.26 ** 0.5 / 45.5) ** 2, 'sqrt'), 'rrunf': (-0.5, 0.5, None), 'wrunf': (-0.5, 0.5, None), 'cshade': (0, 1, None), 'cthin': (0, 1, None), 'csun': (0, 1.2, None)}
for ft in WAVE_ALTS_FT:
    ENC[f'wv{ft}'] = (-8, 8, None)
    ENC[f'dpd{ft}'] = (0, 40, None)
    ENC[f't{ft}'] = (-70, 40, None)
ML_TOP_DZ = (200, 500, 1000, 1800, 3000)
TW_ABOVE = 1000.0
ENC['thml'] = (250, 350, None)
ENC['thx'] = (0, 5, None)
for dz in ML_TOP_DZ:
    ENC[f'dth_m{dz}'] = (-3, 22, None)
    ENC[f'q_m{dz}'] = (0, 20, None)
for p in P.CC_LEVELS:
    ENC[f'cc{p}'] = (0, 100, None)
WIND_AGL_M = [100, 300]
COL_AGL_M = [600, 1000, 1500, 2000, 3000]
COL_WIND_M = sorted(COL_AGL_M + [2500])
STAB_AGL_M = [500, 1000, 1500, 2000, 3000, 4000, 6000]
STAB_NAMES = [f'n_g{(a + b) // 2}' for a, b in zip(STAB_AGL_M, STAB_AGL_M[1:])]
for name in STAB_NAMES:
    ENC[name] = (0, 0.04, None)
for m in COL_AGL_M:
    ENC[f'dpd_g{m}'] = (0, 40, None)
WIND_ENC, WIND_ENC_ALOFT, WIND_ALOFT_FT = ((-60, 60, None), (-100, 100, None), 12000)
for key in ['sfc', 'blavg', 'bltop'] + [f'a{ft}' for ft in WIND_ALTS_FT] + [f'g{m}' for m in WIND_AGL_M + COL_WIND_M]:
    enc = WIND_ENC_ALOFT if key.startswith('a') and int(key[1:]) >= WIND_ALOFT_FT else WIND_ENC
    ENC[f'u_{key}'] = enc
    ENC[f'v_{key}'] = enc
LEVEL_NONE = {'ceil': 254}
LEVEL_SKIP = {f'cc{p}' for p in P.CC_LEVELS} | {'thx'} | {f'q_m{dz}' for dz in ML_TOP_DZ} | {'rrunf', 'wrunf'}

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)
runfields.LOG = log

def _fmt_dur(sec):
    sec = int(round(sec))
    return f'{sec // 60}m{sec % 60:02d}s' if sec >= 60 else f'{sec}s'

class Progress:

    def __init__(self, label, total):
        self.label, self.total, self.done, self.t0 = (label, total, 0, time.time())
        log(f'{label}: {total} hours to compute')

    def tick(self):
        self.done += 1
        el = time.time() - self.t0
        left = el / self.done * (self.total - self.done)
        eta = time.strftime('%H:%M', time.localtime(time.time() + left))
        log(f'{self.label}: [{self.done}/{self.total}] {_fmt_dur(el)} elapsed' + (f', ~{_fmt_dur(left)} left (done ≈ {eta})' if self.done < self.total else ', all hours done'))

def target_grid(region):
    w, s, e, n, step = region[:5]
    return (np.arange(w, e + step / 2, step), np.arange(n, s - step / 2, -step))

class Regridder:

    def __init__(self, src_lat, src_lon, region):
        w, s, e, n, step = region
        self.lons, self.lats = target_grid(region)
        glon, glat = np.meshgrid(self.lons, self.lats)
        lat0 = math.radians((s + n) / 2)
        src = np.column_stack([src_lat.ravel(), src_lon.ravel() * math.cos(lat0)])
        dst = np.column_stack([glat.ravel(), glon.ravel() * math.cos(lat0)])
        dist, self.index = cKDTree(src).query(dst, workers=-1)
        src_step = np.median(np.abs(np.diff(src_lat[:, src_lat.shape[1] // 2]))) if src_lat.ndim == 2 else step
        self.outside = dist > max(step, float(src_step)) * 1.5
        self.shape = glat.shape
        ix = np.round((src_lon - w) / step).astype(np.int64)
        iy = np.round((n - src_lat) / step).astype(np.int64)
        ny, nx = self.shape
        self.cell = np.where((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny), iy * nx + ix, -1).ravel()
        self._plan = None

    def __call__(self, field):
        out = field.ravel()[self.index].reshape(self.shape).astype(np.float32)
        out[self.outside.reshape(self.shape)] = np.nan
        return out

    def peak(self, field):
        out = self(field)
        if self._plan is None:
            ok = np.nonzero(self.cell >= 0)[0]
            order = ok[np.argsort(self.cell[ok], kind='stable')]
            tgt = self.cell[order]
            starts = np.r_[0, np.nonzero(np.diff(tgt))[0] + 1] if len(tgt) else np.zeros(0, np.int64)
            self._plan = (order, starts, tgt[starts])
        order, starts, tgt = self._plan
        if len(order):
            flat = out.reshape(-1)
            vals = np.fmax.reduceat(np.asarray(field, np.float32).ravel()[order], starts)
            keep = np.isfinite(vals) & ~self.outside.reshape(-1)[tgt]
            flat[tgt[keep]] = vals[keep]
        return out

    @classmethod
    def regular(cls, src_lat, src_lon, region):
        r = object.__new__(cls)
        w, s, e, n, step = region[:5]
        r.lons, r.lats = target_grid(region)
        r.shape = (len(r.lats), len(r.lons))
        lat1, lon1 = (src_lat[:, 0], src_lon[0, :])
        dlat, lon0 = (float(lat1[0] - lat1[1]), float(lon1[0]))
        dlon = float((lon1[1] - lon1[0]) % 360)
        ny, nx = (len(lat1), len(lon1))
        iy = np.round((lat1[0] - r.lats) / dlat).astype(np.int64)
        ix = np.round((r.lons - lon0) % 360 / dlon).astype(np.int64) % nx
        bad = (iy < 0) | (iy >= ny)
        r.index = (np.clip(iy, 0, ny - 1)[:, None] * nx + ix[None, :]).ravel()
        r.outside = np.repeat(bad, len(r.lons))
        cx = np.round((src_lon - w) / step).astype(np.int64)
        cy = np.round((n - src_lat) / step).astype(np.int64)
        r.cell = np.where((cx >= 0) & (cx < r.shape[1]) & (cy >= 0) & (cy < r.shape[0]), cy * r.shape[1] + cx, -1).ravel()
        r._plan = None
        return r

    @classmethod
    def identity(cls, region):
        r = object.__new__(cls)
        r.lons, r.lats = target_grid(region)
        r.shape = (len(r.lats), len(r.lons))
        n = r.shape[0] * r.shape[1]
        r.index = np.arange(n)
        r.outside = np.zeros(n, bool)
        r.cell = np.arange(n)
        r._plan = None
        return r

    def part(self, y0, y1, x0, x1):
        p = object.__new__(Regridder)
        p.lats, p.lons = (self.lats[y0:y1], self.lons[x0:x1])
        p.shape = (y1 - y0, x1 - x0)
        p.index = self.index.reshape(self.shape)[y0:y1, x0:x1].ravel()
        p.outside = self.outside.reshape(self.shape)[y0:y1, x0:x1].ravel()
        nx = self.shape[1]
        iy, ix = (self.cell // nx, self.cell % nx)
        inside = (self.cell >= 0) & (iy >= y0) & (iy < y1) & (ix >= x0) & (ix < x1)
        p.cell = np.where(inside, (iy - y0) * (x1 - x0) + (ix - x0), -1)
        p._plan = None
        return p

def quantize(arr, lo, hi, transform):
    a = np.asarray(arr, dtype=np.float32)
    if transform == 'sqrt':
        a, lo, hi = (np.sqrt(np.maximum(a, 0)), math.sqrt(max(lo, 0)), math.sqrt(hi))
    q = np.round((np.clip(a, lo, hi) - lo) / (hi - lo) * 254)
    return np.where(np.isfinite(a), q, 255).astype(np.uint8)

def equation_of_time(doy):
    b = 2 * math.pi * (doy - 81) / 364
    return (9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)) / 60

def solar_estimate(valid, lat, lon, cloud_pct):
    doy = valid.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    hour = valid.hour + valid.minute / 60 + lon / 15 + equation_of_time(doy)
    ha = np.radians(15 * (hour - 12))
    la = np.radians(lat)
    cosz = np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha)
    ghi = 1098 * np.maximum(cosz, 0) * np.exp(-0.057 / np.maximum(cosz, 0.05))
    return ghi * (1 - 0.75 * (np.clip(cloud_pct, 0, 100) / 100) ** 3.4)
SUN_MIN_ELEV = 15.0

def sunshine(sw, valid, lat, lon):
    if sw is None or valid is None or lon is None:
        return None
    clear = solar_estimate(valid, lat, lon, 0)
    return np.where(sun_elevation(valid, lat, lon) >= SUN_MIN_ELEV, np.clip(np.nan_to_num(sw, nan=0) / np.maximum(clear, 1), 0, 1.2), np.nan)

def instant_factor(valid, hours, lat, lon, end=None):
    now = solar_estimate(valid, lat, lon, 0)
    n = 12
    end = end or valid
    mean = sum((solar_estimate(end - timedelta(hours=hours * k / n), lat, lon, 0) * (0.5 if k in (0, n) else 1) for k in range(n + 1))) / n
    return np.where(mean > 20, np.clip(now / np.maximum(mean, 0.001), 0, 2.5), np.where(now > 1, 1.0, 0.0))

def heat_from_sunshine(ghi, t2, td2, land=None):
    frac = np.clip(0.09 + 0.0203 * np.minimum(np.maximum(t2 - td2, 0), 16), 0.05, 0.42)
    heat = frac * np.maximum(ghi, 0)
    return heat if land is None else heat * (land >= 0.5)
HEAT_RN = (0.4105, 0.0139, -0.767)
HEAT_RN_SHARE = (0.03, 0.7)

def heat_from_net_radiation(rn, t2, td2, soil, land=None):
    a, b, c = HEAT_RN
    spread = np.clip(np.nan_to_num(t2 - td2, nan=0), 0, 25)
    w = 0.25 if soil is None else np.nan_to_num(soil, nan=0.25)
    heat = np.nan_to_num(rn, nan=0) * np.clip(a + b * spread + c * w, *HEAT_RN_SHARE)
    return (heat if land is None else np.where(land >= 0.5, heat, 0)).astype(np.float32)

def water_temp(s):
    t, land = (s.get('tsfc'), s.get('lsm'))
    if t is None or land is None:
        return None
    water = land < 0.5
    if s.get('ice') is not None:
        water &= ~(np.nan_to_num(s['ice'], nan=0) >= 0.5)
    return np.where(water, t, np.nan)

def sun_elevation(valid, lat, lon):
    doy = valid.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    hour = valid.hour + valid.minute / 60 + lon / 15 + equation_of_time(doy)
    ha = np.radians(15 * (hour - 12))
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.clip(np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha), -1, 1)))
SUN_GATE = (1.0, 8.0)
HPBL_SEASON = (1.15, 1.15, 1.08, 1.0, 1.0, 1.0, 1.0, 1.0, 1.08, 1.12, 1.15, 1.15)
HPBL_SEASON_MID = (15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349)
HPBL_SEASON_WEST = (-105.0, -102.0)
HPBL_SEASON_SE = (36.0, -85.0)

def hpbl_season(valid, lat, lon):
    mid = np.r_[HPBL_SEASON_MID[-1] - 365, HPBL_SEASON_MID, HPBL_SEASON_MID[0] + 365]
    f = float(np.interp(valid.timetuple().tm_yday, mid, np.r_[HPBL_SEASON[-1], HPBL_SEASON, HPBL_SEASON[0]]))
    w0, w1 = HPBL_SEASON_WEST
    east = np.clip((lon - w0) / (w1 - w0), 0, 1)
    se_lat, se_lon = HPBL_SEASON_SE
    se = (1 - np.clip(lat - se_lat, 0, 1)) * (1 - np.clip(se_lon - lon, 0, 1))
    return 1 + (f - 1) * east * (1 - se)

def sun_gate(valid, lat, lon):
    lo, hi = SUN_GATE
    return np.clip((sun_elevation(valid, lat, lon) - lo) / (hi - lo), 0, 1)
CU_MIN_LOW_CLOUD = 5.0
CU_MARGIN_NO_CLOUD = 200.0
CU_LOW_BROKEN = 56.25
CU_OVERSHOOT = 0.5
CU_OVERSHOOT_MIN = 500.0
CU_FLAT = 150.0
BLUE_OVER_LCL = 500.0
CU_WIND_DEPTH = 1500.0
CU_NEAR_HPA = 75.0
DALR = 9.8
BLOCK_LAYER = 500.0
WAVE_BAND_M = (300.0, 600.0, 900.0)
WAVE_BAND_MAX_STEP = 0.09
BLUE_WIDTH = 0.85

def thermal_width(zi, blapse, tilt, blue=False):
    d0 = 2 * np.maximum(10, 0.102 * 0.6 ** (1 / 3) * (1 - 0.25 * 0.6) * zi)
    stab = np.clip(1.25 - 0.15 * (DALR - np.nan_to_num(blapse, nan=DALR - 5 / 3)), 0.7, 1.15)
    shear = np.clip(0.55 + 0.09 * np.nan_to_num(tilt, nan=5), 0.6, 1.0)
    return np.where(zi > 200, d0 * stab * shear * np.where(blue, BLUE_WIDTH, 1.0), np.nan)
STORM_DBZ = 50.0
GLACIATE_T = -20.0
WET_CAPE = (100.0, 300.0)

def storm_says(s, rain, refc):
    if s.get('cprate') is not None:
        return np.nan_to_num(s['cprate'], nan=0) >= SHOWERS_SHARE * rain
    if s.get('refmax') is not None:
        return np.maximum(np.nan_to_num(s['refmax'], nan=0), 0 if refc is None else refc) >= STORM_DBZ - 10
    return np.ones(np.shape(rain), bool)

def cumulus_forms(ws, zi, lcl, cover=None):
    reach = (ws > 0.5) & (lcl < zi)
    if cover is None or np.all(np.isnan(cover)):
        return reach & (zi - lcl > CU_MARGIN_NO_CLOUD)
    c = np.nan_to_num(cover, nan=0)
    near = (ws > 0.5) & (lcl >= zi) & (lcl < zi + np.maximum(CU_OVERSHOOT * zi, CU_OVERSHOOT_MIN)) & (c < CU_LOW_BROKEN)
    return (reach | near) & (c >= CU_MIN_LOW_CLOUD)
LOW_AGL, MID_AGL = (2000.0, 7000.0)

def bands_agl(bands, cc, h_agl, ps, tops, cc_levels=P.CC_LEVELS, rh=None):
    lcc, mcc = bands[:2]
    hcc = bands[2] if len(bands) > 2 else None
    if lcc is None or mcc is None:
        return None
    lo_t, mid_t = P.band_tops(tops, ps)
    p = np.asarray(cc_levels, np.float32)[:, None, None]
    ok = np.isfinite(cc) & np.isfinite(h_agl)
    if rh is not None:
        with np.errstate(invalid='ignore'):
            ok &= ~((h_agl < P.CEIL_NEAR_GROUND) & (np.nan_to_num(rh, nan=0) < P.CEIL_NEAR_RH))
    ccz = np.where(ok, cc, 0)

    def peak_height(inband):
        m = inband & ok
        k = np.where(m, ccz, -1).argmax(axis=0)
        h = np.take_along_axis(h_agl, k[None], axis=0)[0]
        return np.where(np.where(m, ccz, 0).max(axis=0) >= 1, h, np.nan)
    h_low = peak_height(p >= lo_t[None])
    h_mid = peak_height((p < lo_t[None]) & (p >= mid_t[None]))
    L, M = (np.nan_to_num(lcc, nan=0), np.nan_to_num(mcc, nan=0))
    H = np.zeros_like(L) if hcc is None else np.nan_to_num(hcc, nan=0)
    low_is_low = ~(h_low >= LOW_AGL) & ~(np.isnan(h_low) & (h_mid >= LOW_AGL) & (h_mid < MID_AGL))
    mid_is_low = h_mid < LOW_AGL
    mid_is_high = h_mid >= MID_AGL
    within = lambda lo, hi: np.where(ok & (h_agl >= lo) & (h_agl < hi) & (p >= mid_t[None]), ccz, 0).max(axis=0)
    low = np.maximum.reduce([np.where(low_is_low, L, 0), np.where(mid_is_low, M, 0), within(-np.inf, LOW_AGL)])
    mid = np.maximum.reduce([np.where(low_is_low, 0, L), np.where(mid_is_low | mid_is_high, 0, M), within(LOW_AGL, MID_AGL)])
    high = np.maximum.reduce([H, np.where(mid_is_high, M, 0), within(MID_AGL, np.inf)])
    inside = np.isfinite(lcc)
    return (np.where(inside, low, np.nan), np.where(inside, mid, np.nan), np.where(inside, high, np.nan))

def cumulus_cover(p_base, ps, tops, bands, cc=None, cc_levels=P.CC_LEVELS):
    if bands[0] is None:
        return None
    lo_t, mid_t = P.band_tops(tops, ps)
    z = np.zeros(np.shape(p_base), np.float32)
    lcc, mcc, hcc = (z if b is None else np.nan_to_num(b, nan=0) for b in bands)
    cover = np.where(p_base >= lo_t, lcc, np.where(p_base >= mid_t, mcc, hcc))
    if cc is None:
        return cover
    p = np.asarray(cc_levels, np.float32)[:, None, None]
    near = np.abs(p - p_base[None]) <= CU_NEAR_HPA
    at = np.where(near, np.nan_to_num(cc, nan=0), 0).max(axis=0)
    elsewhere = near.any(axis=0) & (at < CU_MIN_LOW_CLOUD)
    return np.where(elsewhere, np.minimum(cover, at), cover)
SHOWERS_SHARE = 0.5
OVERCAST = 85.0

def no_deck(tcc, lcc=None, mcc=None, sun=None):
    if tcc is None:
        return False
    over = lambda c: np.zeros(np.shape(tcc), bool) if c is None else np.nan_to_num(c, nan=0) >= OVERCAST
    shield = np.zeros(np.shape(tcc), bool) if sun is None or mcc is None else (np.nan_to_num(tcc, nan=0) >= OVERCAST) & (np.nan_to_num(mcc, nan=0) >= sky.BROKEN) & (np.nan_to_num(sun, nan=1) < sky.THICK_SUN)
    return np.isfinite(tcc) & ~over(lcc) & ~over(mcc) & ~shield
CONV_RELIEF = (400.0, 800.0)

def convective_rain(overdev, cudepth, refc, rain, showers=None, tcc=None, lcc=None, mcc=None, sun=None):
    conv = (overdev >= 30) | (cudepth >= 2500) | no_deck(tcc, lcc, mcc, sun) | (np.nan_to_num(rain, nan=0) >= sky.HEAVY_RAIN)
    if showers is not None:
        return conv | (np.nan_to_num(showers, nan=0) >= SHOWERS_SHARE)
    if refc is None:
        return conv | (rain >= 6.0)
    return conv | (refc >= 45.0) | (refc >= 40.0) & (rain >= 2.5)
RAIN_BOX_KM, RAIN_CV, RAIN_CV_FLOOR = (21.0, 0.8, 0.05)

def rain_texture(rain, grid_step):
    k = max(3, int(round(RAIN_BOX_KM / (grid_step * 111.2))) | 1)
    ok = np.isfinite(rain)
    r = np.where(ok, rain, 0.0).astype(np.float64)
    n = np.maximum(sky.window_sum(ok.astype(np.float64), k), 1)
    mean = sky.window_sum(r, k) / n
    var = np.maximum(sky.window_sum(r * r, k) / n - mean * mean, 0)
    return np.sqrt(var) / np.maximum(mean, RAIN_CV_FLOOR)

def showers_share(cprate, rain, overdev=None, cudepth=None, radar=None, cover=(None, None, None), grid_step=None):
    if cprate is not None:
        share = np.clip(np.nan_to_num(cprate, nan=0) / np.maximum(rain, 1e-06), 0, 1)
    elif radar is not None:
        conv = convective_rain(overdev, cudepth, radar, rain, None, *cover)
        if grid_step is not None:
            conv = conv | (rain_texture(rain, grid_step) >= RAIN_CV)
        share = conv.astype(np.float32)
    else:
        return None
    return np.where(rain > 0.1, share, np.nan)
STEADY_RAIN = (0.1, 0.5)
RAIN_WS = (1.0, 2.0)

def rain_damp(rain, convective, ws):
    wet = 1 - np.clip((rain - STEADY_RAIN[0]) / (STEADY_RAIN[1] - STEADY_RAIN[0]), 0, 1)
    damp = np.maximum(wet, np.clip((ws - RAIN_WS[0]) / (RAIN_WS[1] - RAIN_WS[0]), 0, 1))
    return np.where(convective | ~(rain > STEADY_RAIN[0]), 1.0, damp)
DRIFT_KM = (2.5, 5.0, 7.5, 10.0, 15.0)
DRIFT_THERMAL_S = 900.0
DRIFT_WIND_M = 500.0
DRIFT_MIN_WS = 0.25

def thermal_drift(zs, hth, zi, ws, u, v, t2, ps, lat2d, grid_step):
    U = np.hypot(np.nan_to_num(u, nan=0), np.nan_to_num(v, nan=0))
    windy = U > 0.5
    ex = np.where(windy, np.nan_to_num(u, nan=0) / np.maximum(U, 1e-06), 0)
    ey = np.where(windy, np.nan_to_num(v, nan=0) / np.maximum(U, 1e-06), 0)
    ny, nx = zi.shape
    km_x = grid_step * 111.32 * np.maximum(np.cos(np.radians(lat2d)), 0.05)
    km_y = grid_step * 111.32
    rows, cols = np.indices(zi.shape)
    ws_in = np.zeros_like(ws)
    top_in = np.zeros_like(zi)
    lam = np.maximum(U * DRIFT_THERMAL_S / 1000, 1.0)
    for d in DRIFT_KM:
        c = np.clip(np.rint(cols - ex * d / km_x), 0, nx - 1).astype(np.int64)
        r = np.clip(np.rint(rows + ey * d / km_y), 0, ny - 1).astype(np.int64)
        cand = np.nan_to_num(ws[r, c], nan=0) * np.exp(-d / lam)
        better = cand > ws_in
        ws_in = np.where(better, cand, ws_in)
        top_in = np.where(better, np.nan_to_num(zs[r, c] + zi[r, c], nan=0), top_in)
    drifted = windy & (ws_in >= DRIFT_MIN_WS) & (ws_in > np.nan_to_num(ws, nan=0)) & np.isfinite(zs)
    zi2 = np.where(drifted, np.fmax(zi, top_in - zs), zi)
    ws2 = np.where(drifted, ws_in, ws)
    return (zi2.astype(zi.dtype, copy=False), ws2.astype(ws.dtype, copy=False))

def analyse(s, lv, levels, grid_step, lat2d, lon2d=None, valid=None):
    zs = s['zs']
    prof = P.Profile(zs, s['ps'], s['t2'], s['td2'], s['u10'], s['v10'], lv['z'], levels, lv['t'], lv['td'], lv['u'], lv['v'], s.get('uh'), s.get('vh'), s.get('hub_m', 100.0))
    if s.get('shf') is not None:
        heat = np.maximum(np.nan_to_num(s['shf'], nan=0), 0)
    else:
        heat = heat_from_sunshine(s['sw'], s['t2'], s['td2'], s.get('lsm'))
    hth = heat * sun_gate(valid, lat2d, lon2d) if valid is not None and lon2d is not None else heat
    zi = s.get('hpbl')
    if zi is not None and s.get('hpbl_season') and (valid is not None) and (lon2d is not None):
        zi = zi * hpbl_season(valid, lat2d, lon2d)
    if zi is None or np.all(np.isnan(zi)):
        zi = P.parcel_ml_depth(prof, np.where(hth > 20, 1.0, 0.2))
        if s.get('blh') is not None:
            w = np.clip((hth - BLH_HEAT[0]) / (BLH_HEAT[1] - BLH_HEAT[0]), 0, 1)
            zi = np.where(np.isfinite(s['blh']), w * s['blh'] + (1 - w) * zi, zi)
    zi = np.maximum(np.nan_to_num(zi, nan=0), 0)
    ws = np.where(hth > 10, P.wstar(hth, zi, s['t2'], s['ps']), 0)
    ud, vd = prof.at(zs + DRIFT_WIND_M, 'u', 'v')
    zi, ws = thermal_drift(zs, hth, zi, ws, ud, vd, s['t2'], s['ps'], lat2d, grid_step)
    up15, vp15 = prof.at(zs + 1500, 'u', 'v')
    cool = P.night_cooling(s['shf']) if s.get('shf') is not None else None
    pump = P.mountain_plain(zs, hth, up15, vp15, grid_step, cool=cool, coslat=np.cos(np.radians(lat2d)))
    pump_sink, pump_lift = (pump['sink'], pump['lift'])
    held = np.where(hth > 0, np.minimum(pump_sink * P.pump_unresolved(grid_step) * P.PUMP_TOP_S, P.PUMP_TOP_CAP * zi), 0)
    dy = grid_step * 111320.0
    dx = dy * np.cos(np.radians(lat2d))

    def ddx(a):
        return np.gradient(a, axis=1) / dx

    def ddy(a):
        return -np.gradient(a, axis=0) / dy
    relief = P.local_relief(zs, max(1, round(0.1 / grid_step)))
    ur, vr = prof.at(zs + 300, 'u', 'v')
    upslope = ur * ddx(zs) + vr * ddy(zs)
    ramp = np.clip((relief - 100) / 300, 0, 1)
    lee_down = np.maximum(0, -upslope) * P.SINK_LEE * ramp * np.where(hth > 0, P.lee_gate(np.hypot(ur, vr), ws), 1)
    lee_held = np.where(hth > 0, P.lee_hold(-upslope, np.hypot(ur, vr), ws, zi, ramp), 0)
    held = np.minimum(held + lee_held, P.HOLD_CAP * zi)
    ws_free = ws
    ws = np.where((held > 0) & (zi > 0), ws * np.cbrt(np.maximum(zi - held, 1) / np.maximum(zi, 1)), ws)
    zi = np.maximum(zi - held, 0)
    lcl = np.maximum(0, 125 * (s['t2'] - np.minimum(s['td2'], s['t2'])))
    thm = np.zeros_like(zs)
    n = 8
    for i in range(n):
        t, p = prof.at(zs + 2 + zi * i / (n - 1), 't', 'lnp')
        thm += P.theta(t, p)
    t_ml = thm / n * (s['ps'] / 1000.0) ** P.KAPPA - P.K0
    tops = lv.get('cover_tops', P.COVER_TOPS_NCEP)
    ccl_rh = []
    ccl = cloud_levels(s, lv, levels, tops, rh_out=ccl_rh)
    p_base = prof.at(zs + 2 + lcl, 'lnp')
    cu_cover = cumulus_cover(p_base, s['ps'], tops, [s.get('lcc'), s.get('mcc'), s.get('hcc')], np.stack([ccl[f'cc{p}'] for p in P.CC_LEVELS]))
    has_cu = cumulus_forms(ws, zi, lcl, cu_cover)
    zi = np.where(~has_cu & (ws > 0.5) & (zi > lcl + BLUE_OVER_LCL), lcl + BLUE_OVER_LCL, zi)
    depth = np.where(has_cu, lcl, np.where(ws > 0, zi, 0))
    d = np.maximum(np.where(has_cu, lcl, zi), 100)
    ub, vb = P.layer_mean(prof, zs + 2, zs + 2 + d, ['u', 'v'])
    ut, vt = prof.at(zs + 2 + d, 'u', 'v')
    ul, vl = prof.at(zs + 2 + 0.1 * d, 'u', 'v')
    tilt = P.shear_tilt(ws, np.hypot(ut - ul, vt - vl))
    z1, z2 = (zs + 2 + 0.3 * zi, zs + 2 + 0.9 * zi)
    blapse = np.where(zi >= 300, (prof.at(z1, 't') - prof.at(z2, 't')) / np.maximum(z2 - z1, 1) * 1000, np.nan)
    twidth = np.where(ws > 0.5, thermal_width(zi, blapse, tilt, ~has_cu), np.nan)
    base, tbase = (zs + 2 + lcl, t_ml - P.DRY_LAPSE * lcl)
    ctop = P.cumulus_top(prof, base, tbase, has_cu, mix=P.cumulus_mix(twidth))
    cudepth = np.where(has_cu, np.maximum(ctop - base, zi - lcl), 0)
    cudepth = np.where(has_cu & (lcl >= zi), np.maximum(cudepth, CU_FLAT), cudepth)
    cape = np.nan_to_num(s.get('cape', np.zeros_like(zs)), nan=0)
    rain = np.nan_to_num(s.get('prate', np.zeros_like(zs)), nan=0)
    refc = s.get('refc')
    refc = None if refc is None else np.nan_to_num(refc, nan=0)
    says = storm_says(s, rain, refc)
    wet_unstable = (rain > 0.5) & (cape > np.where(says, WET_CAPE[0], WET_CAPE[1]))
    convecting = wet_unstable & says
    if refc is not None:
        convecting |= refc >= STORM_DBZ - 10
    if convecting.any():
        deep = P.cumulus_top(prof, base, tbase, has_cu & convecting)
        cudeep = np.where(has_cu & convecting, np.maximum(deep - base, zi - lcl), cudepth)
    else:
        cudeep = cudepth
    overdev = np.clip((cudeep - 2000) / 4000, 0, 1) * np.clip(0.4 + cape / 1500, 0, 1) * 100
    if s.get('cin') is not None:
        lo, hi = CIN_LID
        overdev = overdev * np.clip((hi - np.nan_to_num(s['cin'], nan=0)) / (hi - lo), 0, 1)
    overdev = np.where(wet_unstable, np.maximum(overdev, np.where(says, 60, 45)), overdev)
    echo = np.zeros_like(rain) if refc is None else refc
    if s.get('refmax') is not None:
        echo = np.maximum(echo, np.nan_to_num(s['refmax'], nan=0))
    t_top = prof.at(base + cudeep, 't')
    warm = has_cu & (t_top > GLACIATE_T) & (echo < STORM_DBZ - 10)
    overdev = np.where(warm, np.minimum(overdev, 45), overdev)
    if refc is not None:
        storm = (refc >= STORM_DBZ) | (refc >= STORM_DBZ - 10) & (cape >= 100)
        overdev = np.where(storm, np.maximum(overdev, 70), overdev)
    radar = refc if refc is None or s.get('refmax') is None else np.maximum(refc, np.nan_to_num(s['refmax'], nan=0))
    cover = (s.get('tcc'), s.get('lcc'), s.get('mcc'))
    showers = showers_share(s.get('cprate'), rain, overdev, cudepth, radar, cover, grid_step)
    sun_now = sunshine(s.get('sw'), valid, lat2d, lon2d)
    ws_out = ws * rain_damp(rain, convective_rain(overdev, cudepth, refc, rain, showers, *cover, sun_now), ws)
    bs = P.bs_ratio(ws_out, np.hypot(ub, vb))
    cudepth_out = np.where(has_cu & convecting & (overdev >= 50), np.maximum(cudepth, cudeep), cudepth)
    thermals_here = ws > 0
    zt = zs + 2 + zi
    thml = P.theta(*prof.at(zs + 2 + 0.5 * zi, 't', 'lnp'))
    thx = np.clip(P.theta(s['t2'], s['ps']) - thml, 0, 5)
    sounding = {'thml': thml, 'thx': thx}
    for dz in ML_TOP_DZ:
        t, p = prof.at(zt + dz, 't', 'lnp')
        td = prof.at(zt + dz, 'td')
        sounding[f'dth_m{dz}'] = P.theta(t, p) - thml
        sounding[f'q_m{dz}'] = P.mixing_ratio(np.minimum(td, t), p) * 1000
    sounding = {k: np.where(thermals_here, v, np.nan) for k, v in sounding.items()}
    out = {'wstar': ws_out, 'zi': zi, 'depth': depth, 'cubase': np.where(has_cu, lcl, np.nan), 'cudepth': cudepth_out, 'overdev': overdev, 'bs': bs, 'blapse': np.clip(blapse, 0, 15), 'twidth': twidth, 'cape': cape, 'rain': rain, 'showers': showers, 't2': s['t2'], 'td2': s['td2'], 'mslp': s.get('mslp'), 'sst': water_temp(s), 'heat': heat, 'sun': sun_now, 'densalt': P.density_altitude(s['ps'], s['t2']), 'gust': s.get('gust'), 'cloud': s.get('tcc'), 'cloudlow': s.get('lcc'), 'cloudmid': s.get('mcc'), 'cloudhigh': s.get('hcc'), 'pm25': s.get('smoke'), 'snowd': None if s.get('snod') is None else np.maximum(s['snod'], 0), 'soilm': P.soil_wetness(s.get('mstav'), s.get('soilw'), s.get('wilt'), s.get('fldcp'), s.get('soilm9')), 'u_sfc': s['u10'], 'v_sfc': s['v10'], 'u_blavg': ub, 'v_blavg': vb, 'u_bltop': ut, 'v_bltop': vt}
    out.update(ccl)
    tw_top = np.maximum(zi, np.where(has_cu, lcl + np.maximum(np.nan_to_num(cudepth_out, nan=0), 0), 0))
    for name, dz in (('twt', 0.0), ('twa', TW_ABOVE)):
        u_tw, v_tw = prof.at(zs + 2 + tw_top + dz, 'u', 'v')
        out[f'u_{name}'], out[f'v_{name}'] = (np.where(thermals_here, a, np.nan) for a in (u_tw, v_tw))
    cud_w = np.minimum(np.nan_to_num(cudepth_out, nan=0), CU_WIND_DEPTH)
    out['u_cuw'], out['v_cuw'] = (np.where(has_cu, a, np.nan) for a in prof.at(zs + 2 + lcl + cud_w / 2, 'u', 'v'))
    rh = [P.rh_from_td(*prof.at(zs + 2 + lcl + f * np.maximum(np.nan_to_num(cudepth_out, nan=0), CU_FLAT), 't', 'td')) for f in (0.2, 0.5, 0.8)]
    out['curh'] = np.where(has_cu, (rh[0] + rh[1] + rh[2]) / 300, np.nan)
    agl = bands_agl([s.get('lcc'), s.get('mcc'), s.get('hcc')], np.stack([ccl[f'cc{p}'] for p in P.CC_LEVELS]), P.level_heights(levels, lv['z']) - zs[None], s['ps'], tops, rh=ccl_rh[0] if ccl_rh else None)
    if agl is not None:
        out['cloudlow'], out['cloudmid'], out['cloudhigh'] = agl
    out['accas'] = accas_score(lv, levels, zs, s['t2'])
    out.update(castellanus(lv, levels, zs, zi, s, grid_step, lat2d))
    out.update(sounding)
    for m in WIND_AGL_M:
        out[f'u_g{m}'], out[f'v_g{m}'] = prof.at(zs + m, 'u', 'v')
    for key, z in (('sfc', 10.0), *((f'g{m}', float(m)) for m in WIND_AGL_M)):
        w = np.clip(1 - z / pump['depth'], 0, 1)
        out[f'u_{key}'] = out[f'u_{key}'] + w * pump['u']
        out[f'v_{key}'] = out[f'v_{key}'] + w * pump['v']
    for ft in WIND_ALTS_FT:
        z = np.full_like(zs, ft * P.FT)
        u, v = prof.at(z, 'u', 'v')
        below = z < zs + 30
        out[f'u_a{ft}'] = np.where(below, np.nan, u)
        out[f'v_a{ft}'] = np.where(below, np.nan, v)
    flat = np.clip((CONV_RELIEF[1] - relief) / (CONV_RELIEF[1] - CONV_RELIEF[0]), 0, 1)
    out['conv'] = -(ddx(ub) + ddy(vb)) * np.maximum(zi, 300) * flat + pump_lift
    out['bumpy'] = np.where(np.isfinite(s['t2']), P.bumpiness(prof, ws_out, zi, s.get('gust'), s['u10'], s['v10'], s.get('ustar'), relief), np.nan)
    hilly = relief >= 150
    out['ridge'] = np.where(hilly, np.maximum(0, upslope), np.nan)
    th0 = P.theta(*prof.at(zs + 10, 't', 'lnp'))
    th1 = P.theta(*prof.at(zs + 10 + BLOCK_LAYER, 't', 'lnp'))
    n2 = P.G / (0.5 * (th0 + th1)) * (th1 - th0) / BLOCK_LAYER
    out['nlow'] = np.where(hilly, np.sqrt(np.maximum(n2, 0)), np.nan)
    if lv.get('w') is not None:
        wz = lv['z']
        w = lv['w']
        order = np.argsort(-np.asarray(levels))
        wz, w = (wz[order], w[order])
        above = wz > zs[None] + 500
        wmask = np.where(above, w, -np.inf)
        k = np.argmax(wmask, axis=0)
        wmax = P.take_along_levels(wmask, k)
        out['wavemax'] = np.where(np.isfinite(wmax), wmax, np.nan)
        out['wavemaxz'] = np.where(np.isfinite(wmax) & (wmax > 0), P.take_along_levels(wz, k), np.nan)
        for ft in WAVE_ALTS_FT:
            zq = ft * P.FT
            hi = P.level_positions(np.clip(P.levels_below(wz, zq), 1, wz.shape[0] - 1))
            lo = hi - zs.size
            z0, z1 = (P.take_level(wz, lo, zs.shape), P.take_level(wz, hi, zs.shape))
            w0, w1 = (P.take_level(w, lo, zs.shape), P.take_level(w, hi, zs.shape))
            f = np.clip((zq - z0) / np.maximum(z1 - z0, 1), 0, 1)
            out[f'wv{ft}'] = np.where(zq > zs + 100, w0 + f * (w1 - w0), np.nan)
    for ft in WAVE_ALTS_FT:
        zq = np.full_like(zs, ft * P.FT)
        tq, tdq = prof.at(zq, 't', 'td')
        out[f'dpd{ft}'] = np.where(zq > zs + 100, np.maximum(0, tq - tdq), np.nan)
        out[f't{ft}'] = np.where(zq > zs + 100, tq, np.nan)
    wz_up = w_up = None
    if lv.get('w') is not None and (not lv.get('w_wave_only')):
        up = np.argsort(-np.asarray(levels))
        wz_up, w_up = (lv['z'][up], lv['w'][up])
    if w_up is not None and grid_step <= WAVE_BAND_MAX_STEP:
        under = wz_up < (zs + 10)[None]
        k = np.arange(wz_up.shape[0], dtype=np.float32)[:, None, None]
        zc = np.concatenate([zs[None], np.where(under, zs[None] + 0.01 * (k + 1), wz_up)])
        wc = np.concatenate([np.zeros_like(zs)[None], np.where(under, 0, np.nan_to_num(w_up, nan=0))])
        wl = sum((P.column_at(zc, wc, zs + h) for h in WAVE_BAND_M)) / len(WAVE_BAND_M)
        out['wlow'] = np.where(hilly & np.isfinite(s['t2']), wl, np.nan)
    lee = lee_down
    falling = P.precip_type(rain, s.get('crain'), s.get('csnow'), s.get('cfrzr'), s.get('cicep'), s.get('ptype'), s['t2'])
    snow = P.snow_share(falling, prof, depth)
    sink, between, why = P.sink_rate(prof, wz_up, w_up, ws_out, zi, depth, rain, s.get('refd1'), lee, parts=True, snow=snow)
    if s.get('snod') is not None and valid is not None and (lon2d is not None):
        gl = P.glacier_sink(s['snod'], s['t2'], np.hypot(ddx(zs), ddy(zs)), sun_gate(valid, lat2d, lon2d)) - between
        why = np.where((gl > sink) & (gl >= 0.05), np.float32(4), why)
        sink = np.maximum(sink, gl)
    why = np.where((pump_sink > sink) & (pump_sink >= 0.05), np.float32(5), why)
    sink = np.maximum(sink, pump_sink)
    inside = np.isfinite(s['t2'])
    out['sink'], out['sinkth'], out['sinkby'] = (np.where(inside, a, np.nan) for a in (sink, between, why))
    desc = np.maximum(np.where((np.rint(why) == 2) | (np.rint(why) == 5), sink, 0), pump_sink)
    ws0 = out['wstar']
    ws_air = P.air_motion_climb(ws0, P.vertical_motion(out['conv'], desc, np.full_like(desc, 2)))
    hold_f = np.where(ws_free > 0, ws / np.maximum(ws_free, 1e-06), 1)
    ws_air = np.maximum(ws_air, ws0 * np.minimum(1, (1 - P.SINK_W_CAP) / np.maximum(hold_f, 1e-06)))
    ws_air = np.where(has_cu & (ws0 >= 0.5), np.maximum(ws_air, np.minimum(ws0, 0.52)), ws_air)
    out['wstar'] = np.where(inside, ws_air, ws0)
    out['bs'] = np.where(inside, P.bs_ratio(out['wstar'], np.hypot(ub, vb)), out['bs'])
    score, wavelen = P.wave_environment(prof)
    lb = zs + 500
    ul, vl = P.layer_mean(prof, lb, np.maximum(lb + 800, np.minimum(zs + 3000, prof.ztop - 2500)), ['u', 'v'], 6)
    trapping = np.nan_to_num(score, nan=0) > 0
    out['wtrap'] = np.where(trapping, score * 100, np.nan)
    out['wlam'] = np.where(trapping, P.carry_wavelength(score, wavelen, ul, vl, grid_step, lat2d), np.nan)
    out['u_wl'], out['v_wl'] = (np.where(trapping | hilly, a, np.nan) for a in (ul, vl))
    col = trapping | hilly
    for m in COL_WIND_M:
        u, v, t, td = prof.at(zs + m, 'u', 'v', 't', 'td')
        fade = np.clip(1 - m / pump['depth'], 0, 1)
        out[f'u_g{m}'], out[f'v_g{m}'] = (np.where(col, a + fade * pump[c], np.nan) for a, c in ((u, 'u'), (v, 'v')))
        if m in COL_AGL_M:
            out[f'dpd_g{m}'] = np.where(col, np.maximum(0, t - td), np.nan)
    th = [P.theta(*prof.at(zs + m, 't', 'lnp')) for m in STAB_AGL_M]
    for name, a, b, tha, thb in zip(STAB_NAMES, STAB_AGL_M, STAB_AGL_M[1:], th, th[1:]):
        n2 = P.G / (0.5 * (tha + thb)) * (thb - tha) / (b - a)
        out[name] = np.where(col, np.sqrt(np.maximum(n2, 0)), np.nan)
    out.update(general_aviation(s, lv, levels, prof, rain, grid_step, ddx, ddy, [out[f'cc{p}'] for p in P.CC_LEVELS], zi=zi))
    ceil = out.get('ceil')
    nanf = lambda a: np.full(zs.shape, np.nan, np.float32) if a is None else a
    el = sun_elevation(valid, lat2d, lon2d) if valid is not None and lon2d is not None else np.full(zs.shape, 45.0, np.float32)
    out.update(decks.organisation(prof, zs, s['t2'], s['td2'], s['ps'], s['u10'], s['v10'], nanf(out.get('sst')), hth, out['zi'], np.stack([out[f'cc{p}'] for p in P.CC_LEVELS]), P.level_heights(levels, lv['z']), nanf(out.get('cloudlow')), nanf(out.get('cloudmid')), nanf(out.get('cloudhigh')), np.where(np.isfinite(nanf(ceil)) & (nanf(ceil) < P.CEIL_NONE), nanf(ceil), np.nan), nanf(out.get('lowbase')), el))
    out.update(spotted(levels, lv, zs, s, out, relief, grid_step, lat2d, lon2d, valid, el))
    out.update(sunsets(s, out, grid_step, lat2d, lon2d, valid))
    return out
SPOTTING = os.environ.get('SPOTTING', '1') != '0'

def spotted(levels, lv, zs, s, out, relief, grid_step, lat2d, lon2d, valid, el):
    if not SPOTTING:
        return {}
    try:
        flights = None
        if valid is not None and lon2d is not None:
            try:
                import contrails
                tr = contrails.traffic()
            except ImportError:
                tr = None
            if tr is not None:
                flights = contrails._bilinear(tr[0][valid.hour].sum(axis=(0, 1)), lat2d, lon2d) * float(tr[1][valid.weekday()])
        return spotting.spot(levels, lv, zs, s, out, relief, grid_step * 111.2, el if valid is not None and lon2d is not None else None, flights, s.get('contrail'), lat2d, lon2d, valid)
    except Exception as e:
        log(f'  spotting skipped: {e!r}')
        return {}
SUNSET = os.environ.get('SUNSET', '1') != '0'

def sunsets(s, out, grid_step, lat2d, lon2d, valid):
    if not SUNSET or valid is None or lon2d is None:
        return {}
    try:
        nanf = lambda a: np.full(lat2d.shape, np.nan, np.float32) if a is None else a
        return {'glow': sunset.glow(lat2d, lon2d, valid, nanf(out.get('cloudlow')), nanf(out.get('cloudmid')), nanf(out.get('cloudhigh')), out.get('rain'), s['t2'] - s['td2'], grid_step * 111.2)}
    except Exception as e:
        log(f'  sunset skipped: {e!r}')
        return {}

def cloud_rh(lv):
    if lv.get('rh_cloud') is not None:
        return lv['rh_cloud']
    if lv.get('rh') is not None:
        return lv['rh']
    return P.rh_cloud(lv['t'], lv['td'])
ACCAS_RH = 90.0
ACCAS_TOP = 500.0

def accas_score(lv, levels, zs, t2):
    order = np.argsort(-np.asarray(levels))
    p = np.asarray(levels, np.float32)[order]
    z, t, rh = (lv['z'][order], lv['t'][order], cloud_rh(lv)[order])
    score = np.zeros(zs.shape, np.float32)
    for k in range(len(p) - 1):
        if p[k + 1] < ACCAS_TOP:
            break
        dz = z[k + 1] - z[k]
        lapse = (t[k] - t[k + 1]) / np.maximum(dz, 1)
        moist = P.moist_lapse(0.5 * (t[k] + t[k + 1]), float(np.sqrt(p[k] * p[k + 1])))
        sc = np.clip((lapse - moist) / (P.DRY_LAPSE - moist), 0, 1) * 100
        ok = (z[k] - zs >= LOW_AGL) & (np.nan_to_num(rh[k], nan=0) >= ACCAS_RH) & (dz > 0)
        score = np.where(ok, np.maximum(score, sc), score)
    return np.where(np.isfinite(t2), score, np.nan).astype(np.float32)
EA_GRID_KM = 6.0
EA_W_SIGMA_KM = 15.0
EA_OWN_ONLY = os.environ.get('ACC_CLASS') == 'own'
EA_SCHEME = (1000.0, 500.0)

def castellanus(lv, levels, zs, zi, s, grid_step, lat2d):
    from scipy import ndimage
    if s.get('cvbase') is not None and s.get('cvtop') is not None:
        return castellanus_scheme(s, zs, zi, lat2d)
    own_w = lv.get('w') is not None and (not lv.get('w_wave_only'))
    if EA_OWN_ONLY and (not own_w):
        return {}
    km = grid_step * 111.2
    n = max(1, int(round(EA_GRID_KM / km)))
    g2, g3 = ((slice(None, None, n),) * 2, (slice(None),) + (slice(None, None, n),) * 2)
    src = (np.asarray(levels) <= P.EA_P[0]) & (np.asarray(levels) >= P.EA_P[1])
    if own_w:
        sig = EA_W_SIGMA_KM / (km * n)
        w = [None] * len(levels)
        for k in np.nonzero(src)[0]:
            wk = np.nan_to_num(lv['w'][k][g2])
            w[k] = ndimage.gaussian_filter(wk, sig, mode='nearest') if sig > 0.5 else wk
    else:
        w = P.kinematic_w(lv['u'][g3], lv['v'][g3], lv['t'][g3], levels, s['ps'][g2], grid_step * n, lat2d[g2], want=src)
    ea = P.elevated_parcels(lv['z'][g3], lv['t'][g3], lv['td'][g3], cloud_rh(lv)[g3], levels, zs[g2], zi[g2], s['t2'][g2], s['td2'][g2], s['ps'][g2], w=w, lat=lat2d[g2])
    out = {'acscore': 100 * ea['score'], 'acbase': ea['base'], 'accape': ea['cape'], 'accin': -ea['cin']}
    if n > 1:
        iy, ix = np.meshgrid(np.arange(zs.shape[0]) / n, np.arange(zs.shape[1]) / n, indexing='ij')
        up = lambda a, order: ndimage.map_coordinates(a, [iy, ix], order=order, mode='nearest').astype(np.float32)
        has = up((ea['score'] > 0).astype(np.float32), 0) > 0
        out = {k: up(np.nan_to_num(v), 1) if k == 'acscore' else np.where(has, up(np.nan_to_num(v), 0), np.nan) for k, v in out.items()}
    inside = np.isfinite(s['t2'])
    return {k: np.where(inside, v, np.nan).astype(np.float32) for k, v in out.items()}

def castellanus_scheme(s, zs, zi, lat2d):
    ramp = lambda x, a, b: np.clip((x - a) / (b - a), 0, 1)
    b, t = (np.asarray(s['cvbase'], np.float32), np.asarray(s['cvtop'], np.float32))
    on = np.isfinite(b) & np.isfinite(t) & (b > -400)
    base = np.where(on, b - zs, np.nan)
    top = np.where(np.abs(lat2d) < 23.5, P.EA_BASE[2], P.EA_BASE[1])
    f = np.nan_to_num(ramp(base, P.EA_BASE[0] - P.EA_BASE[3], P.EA_BASE[0]) * (1 - ramp(base, top, top + P.EA_BASE[3])))
    ok = on & (base >= np.nan_to_num(zi) + EA_SCHEME[0]) & (t - b >= EA_SCHEME[1])
    score = np.where(ok, 100 * f, 0)
    inside = np.isfinite(s['t2'])
    nan = np.full(score.shape, np.nan, np.float32)
    return {'acscore': np.where(inside, score, np.nan).astype(np.float32), 'acbase': np.where(inside & (score > 0), base, np.nan).astype(np.float32), 'accape': nan, 'accin': nan}

def cloud_levels(s, lv, levels, tops=P.COVER_TOPS_NCEP, rh_out=None):
    ps = np.where(np.isfinite(s['t2']), s['ps'], np.nan)
    if lv.get('cloud') is not None:
        levels, rh, clw, own = lv['cloud']
    else:
        rh = cloud_rh(lv)
        clw, own = (lv.get('clw'), lv.get('cc'))
    have = [float(p) for p in levels]
    if own is not None and all((p in have for p in P.CC_LEVELS)):
        idx = [have.index(p) for p in P.CC_LEVELS]
        own = np.clip(own[idx], 0, 100)
        with np.errstate(invalid='ignore'):
            cc = np.where(np.array(P.CC_LEVELS, np.float32)[:, None, None] <= ps[None], own, np.nan)
        rh_t = list(rh[idx]) if rh is not None else None
    else:
        rh_t = []
        cc = P.cloud_cover_levels(levels, rh, ps, P.rh_cloud(s['t2'], s['td2']), clw, rh_out=rh_t)
    if rh_t is not None and any((s.get(k) is not None for k in ('lcc', 'mcc', 'hcc'))):
        cc = P.calibrate_cover(cc, np.stack(rh_t), ps, [s.get('lcc'), s.get('mcc'), s.get('hcc')], tops)
    if rh_out is not None and rh_t:
        rh_out.append(np.stack(rh_t))
    return {f'cc{p}': a for p, a in zip(P.CC_LEVELS, cc)}
RADAR_BLUR_KM = 4.0

def blur_dbz(dbz, sigma):
    from scipy.ndimage import gaussian_filter
    z = np.power(10, np.asarray(dbz, np.float32) / 10)
    return (10 * np.log10(np.maximum(gaussian_filter(z, sigma, mode='nearest'), 1))).astype(np.float32)

def radar(s, rain=None, step=None):
    if s.get('radar') is not None:
        return np.maximum(np.nan_to_num(s['radar'], nan=0), 0)
    if s.get('refd1') is not None:
        d = np.maximum(np.nan_to_num(s['refd1'], nan=0), 0)
        sigma = RADAR_BLUR_KM / (step * 111.32) if step else 0
        return blur_dbz(d, sigma) if sigma >= 0.3 else d
    if s.get('refc') is not None:
        return np.maximum(np.nan_to_num(s['refc'], nan=0), 0)
    return None if rain is None else P.reflectivity_from_rain(rain)

def ceiling_own_cover(ceil, s, lv, levels, inside):
    below = np.asarray(levels, np.float32)[:, None, None] > s['ps'][None]
    own = lambda a: np.where(below, np.nan, a)
    rh2 = P.rh_from_td(s['t2'], s['td2'])
    cov = P.ceiling_from_cover(lv['z'], own(np.clip(lv['cc'], 0, 100)), s['zs'], rh2, s.get('tcc'), own(lv['rh']), cover=P.CEIL_COVER_OWN, tcc_min=P.CEIL_COVER_OWN)
    return np.where(inside & (cov < P.CEIL_NONE), cov, ceil)

def column_liquid_levels(tclw, z, t, rh, p, zs):
    w = np.clip((rh - 80) / 15, 0, 1) * (z > zs[None] + 30)
    rho = p[:, None, None] * 100 / (P.RD * (t + 273.15))
    dz = np.abs(np.gradient(z, axis=0))
    mass = (w * rho * dz).sum(axis=0)
    q = np.nan_to_num(tclw, nan=0) / np.where(mass > 0, mass, np.inf)
    return (w * q[None]).astype(np.float32)

def general_aviation(s, lv, levels, prof, rain, grid_step, ddx, ddy, cc=None, zi=None):
    zs = s['zs']
    inside = np.isfinite(s['t2'])
    tcc = s.get('tcc')
    zc = P.level_heights(levels, lv['z'])
    if s.get('ceil') is not None:
        ceil = P.ceiling_agl(s['ceil'], zs, inside)
        if lv.get('cc') is not None and lv.get('rh') is not None:
            ceil = ceiling_own_cover(ceil, s, lv, levels, inside)
    elif cc is not None:
        rh_cc = []
        P.cloud_cover_levels(levels, cloud_rh(lv), s['ps'], P.rh_cloud(s['t2'], s['td2']), rh_out=rh_cc)
        ceil = np.where(inside, P.ceiling_from_cover(zc, np.stack(cc), zs, P.rh_from_td(s['t2'], s['td2']), tcc, np.stack(rh_cc)), np.nan)
    else:
        ceil = np.where(inside, P.ceiling_from_humidity(prof, tcc), np.nan)
    ptype = P.precip_type(rain, s.get('crain'), s.get('csnow'), s.get('cfrzr'), s.get('cicep'), s.get('ptype'), s['t2'])
    vis = np.clip(s['vis'], 0, 16100) if s.get('vis') is not None else P.visibility_estimate(s['t2'], s['td2'], rain, ptype == 2)
    vis = np.where(inside, vis, np.nan)
    derived_frz = P.freezing_level(prof)
    frz = s.get('frz')
    frzl = derived_frz if frz is None else np.where(np.isfinite(frz), np.maximum(frz, zs), derived_frz)
    ctop = s['ctop'] if s.get('ctop') is not None else P.cloud_top_from_humidity(prof, tcc)
    if s.get('ctop_olr') is not None:
        ctop = np.where(np.isfinite(s['ctop_olr']), np.fmax(s['ctop_olr'], np.nan_to_num(ctop, nan=0)), ctop)
    ceiltop = P.ceiling_top(ceil, zs, zc, np.stack(cc)) if cc is not None else None
    cloudy = np.nan_to_num(s['lcc'] if s.get('lcc') is not None else tcc if tcc is not None else np.full_like(zs, 100.0), nan=0) >= P.LOWBASE_COVER
    lowbase = np.where(inside & cloudy, P.low_deck_base({m: np.maximum(0, np.subtract(*prof.at(zs + m, 't', 'td'))) for m in P.LOWBASE_LEVELS}, ceil), np.nan)
    refc = radar(s, rain, grid_step)
    order = np.argsort(-np.asarray(levels))
    z_l, t_l = (lv['z'][order], lv['t'][order])
    rh_l = lv['rh'][order] if lv.get('rh') is not None else P.rh_from_td(t_l, lv['td'][order])
    clw_l = lv['clw'][order] if lv.get('clw') is not None else None
    if clw_l is None and s.get('tclw') is not None:
        clw_l = column_liquid_levels(s['tclw'], z_l, t_l, rh_l, np.asarray(levels)[order], zs)
    icing, icebase, icetop = P.icing_potential(z_l, t_l, rh_l, clw_l, zs)
    from scipy.ndimage import uniform_filter
    k = max(1, round(0.25 / grid_step))
    smooth = (lambda a: uniform_filter(a, k, mode='nearest')) if k > 1 else None
    turb, turbz, turbbase, turbtop = P.ellrod_turbulence(prof, ddx, ddy, smooth, zi, band=True)
    u2k, v2k = prof.at(zs + 2 + 610, 'u', 'v')
    llws = np.hypot(u2k - s['u10'], v2k - s['v10'])
    nan = lambda a: np.where(inside, a, np.nan)
    share = s.get('snowshare')
    snowing = inside & (np.isin(ptype, (2, 3)) | (np.nan_to_num(share, nan=0) > 0.05 if share is not None else False))
    slr = np.full(zs.shape, np.nan, np.float32)
    if snowing.any():
        slr = np.where(snowing, P.kuchera_ratio(prof), np.nan)
    return {'ceil': ceil, 'vis': vis, 'fltcat': P.flight_category(ceil, vis), 'ptype': nan(ptype), 'slr': slr, 'refc': nan(refc), 'snowshare': None if share is None else np.where(snowing, share, np.nan), 'frzl': nan(frzl), 'ctop': nan(ctop), 'ceiltop': None if ceiltop is None else nan(ceiltop), 'lowbase': lowbase, 'icing': nan(icing), 'icebase': icebase, 'icetop': icetop, 'turb': nan(turb), 'turbz': np.where(turb >= 2, turbz, np.nan), 'turbbase': turbbase, 'turbtop': turbtop, 'llws': nan(llws)}

class Blend:
    __slots__ = ('m0', 'm1', 'a', 'fh0', 'fh1', 'nearest')

    def __init__(self, m0, m1, a, fh0, fh1, nearest=False):
        self.m0, self.m1, self.a, self.fh0, self.fh1, self.nearest = (m0, m1, a, fh0, fh1, nearest)

    def values(self):
        if self.nearest:
            return grib.decode(self.m0 if self.a < 0.5 else self.m1)[0]
        v = grib.decode(self.m0)[0]
        v *= 1 - self.a
        v += self.a * grib.decode(self.m1)[0]
        return v
BLEND_NEAREST = ('ptype',)

def _decode(m, region=None, boxes=None):
    if isinstance(m, Blend):
        return m.values()
    if hasattr(m, 'read'):
        return m.read(region, boxes)
    return grib.decode(m)[0]

def _fetch(src, run, fh, keys):
    raw = src.fetch(run, fh, src.inventory(run, fh, keys))
    return {k: grib.decode(m)[0] for k, m in raw.items()}

def prev_step(src, fh):
    p = fh - 1
    while p > 0 and (not src.fstep(p)):
        p -= 1
    return p

def next_step(src, fh):
    p = fh + 1
    while not src.fstep(p):
        p += 1
    return p

def _window_mean(src, run, fh, canon, now_vals, earlier):
    kind = src.windowed[canon]
    key = src.surface[canon]
    if fh <= 0:
        return (None, 0)
    if kind == 'avg6':
        start = (fh - 1) // 6 * 6
        prev = prev_step(src, fh)
        if prev <= start:
            return (now_vals, fh - start)
        a_prev = earlier(prev, key)
        if a_prev is None:
            return (now_vals, fh - start)
        return (((fh - start) * now_vals - (prev - start) * a_prev) / (fh - prev), fh - prev)
    if kind == 'acc1h':
        return (now_vals / 3600, 1)
    if kind in ('mean', 'sum'):
        step = fh - prev_step(src, fh)
        return (now_vals if kind == 'mean' else now_vals / (step * 3600), step)
    if kind in ('acc', 'accw'):
        prev = prev_step(src, fh)
        a_prev = earlier(prev, key)
        if a_prev is None:
            return (None, 0)
        step = fh - prev
        return ((now_vals - a_prev) / (step * 3600) * (1000 if kind == 'accw' else 1), step)
    raise ValueError(kind)
SUNLIT = {'sw', 'shf', 'rnsw'}

def analysis_from_prev(src, run, fh, native, fetch=None):
    prev = run - timedelta(hours=1)
    if fh != 0 or not src.analysis_prev or prev.hour not in src.runs:
        return native
    keys = {c: src.surface[c] for c in src.analysis_prev if c in src.surface}
    try:
        got = (fetch or _fetch)(src, prev, 1, list(keys.values()))
    except Exception as e:
        log(f"  f000: the hour's {', '.join(keys)} from {prev:%HZ} f001 unavailable ({e})")
        return native
    for c, k in keys.items():
        v = got.get(k)
        if v is None:
            continue
        native[c] = ('win', v / 3600, 1) if src.windowed.get(c) == 'acc1h' else v
    return native

def resolve_windows(src, run, fh, native, earlier):
    for canon in src.windowed:
        if isinstance(native.get(canon), tuple):
            continue
        if native.get(canon) is None:
            native[canon] = native.get(f'{canon}_now')
            continue
        mean, hours = _window_mean(src, run, fh, canon, native[canon], earlier)
        native[canon] = ('win', mean, hours) if mean is not None else native.get(f'{canon}_now')
    if src.gust_window and fh == 0:
        native['gust'] = None
    return native

def snd_hour(src, fh):
    after, every = SND_EVERY.get(src.id, (10 ** 6, 1))
    return bool(src.snd_levels) and bool(src.fstep(fh)) and (fh <= after or fh % every == 0)
STEPS_KEPT = 2
_STEPS: dict = {}
_STEPS_LOCK = threading.Lock()

def step_messages(src, run, fh, wanted):
    if not src.between_to:
        return src.fetch(run, fh, src.inventory(run, fh, wanted))
    key = (src.id, run, fh, wanted is None)
    with _STEPS_LOCK:
        fut, mine = (_STEPS.pop(key, None), False)
        if fut is None:
            fut, mine = (Future(), True)
        _STEPS[key] = fut
        while len(_STEPS) > STEPS_KEPT:
            _STEPS.pop(next(iter(_STEPS)))
    if mine:
        try:
            fut.set_result(src.fetch(run, fh, src.inventory(run, fh, wanted)))
        except Exception as e:
            fut.set_exception(e)
            with _STEPS_LOCK:
                if _STEPS.get(key) is fut:
                    del _STEPS[key]
    return fut.result()

def kept_step(src, run, fh):
    with _STEPS_LOCK:
        for (sid, r, f, _), fut in _STEPS.items():
            if sid == src.id and r == run and (f == fh) and fut.done() and (fut.exception() is None):
                return fut.result()
    return None

def fetch_hour(src, run, fh):
    if src.fstep(fh):
        return step_messages(src, run, fh, None if snd_hour(src, fh) else src.wanted(sounding=False))
    fh0, fh1 = (prev_step(src, fh), next_step(src, fh))
    r0, r1 = (step_messages(src, run, f, None if snd_hour(src, f) else src.wanted(sounding=False)) for f in (fh0, fh1))
    a = (fh - fh0) / (fh1 - fh0)
    keep = set(src.wanted(sounding=False))
    nearest = {src.surface[c] for c in BLEND_NEAREST if c in src.surface}
    return {k: Blend(m, r1[k], a, fh0, fh1, k in nearest) for k, m in r0.items() if k in r1 and k in keep}

def blend_window(src, canon, m, run):
    kind = src.windowed[canon]
    if kind not in ('acc', 'accw'):
        raise ValueError(f'{src.id} {canon}: {kind} has no window between steps')
    v0, v1 = (grib.decode(m.m0)[0], grib.decode(m.m1)[0])
    step = m.fh1 - m.fh0
    mean = (v1 - v0) / (step * 3600) * (1000 if kind == 'accw' else 1)
    return ('win', mean.astype(np.float32), step, run + timedelta(hours=m.fh1))

def omega_to_w(omega, t, p):
    return -omega * P.RD * (t + 273.15) / (p * 100 * P.G)

def total_of_bands(tcc, bands):
    clear = np.prod([1 - np.clip(np.nan_to_num(b, nan=0), 0, 100) / 100 for b in bands], axis=0)
    return np.where(np.isfinite(tcc), np.fmax(tcc, 100 * (1 - clear)), tcc).astype(np.float32)

def inside_of(s):
    return np.isfinite(s['t2'])

def open_meteo_extra(src, run, fh, between=False, idx=None):
    if not src.om_extra:
        return {}
    import om
    domain, names = src.om_extra
    names = {k: (v, None) for k, v in names.items()}
    if fh <= src.between_to:
        names.update(src.om_hourly)
        if between:
            names.update(src.om_between)
    valid = run + timedelta(hours=fh)
    url = f'{OpenMeteoBase}/data_spatial/{domain}/{run:%Y/%m/%d/%H%M}Z/{valid:%Y-%m-%dT%H%M}.om'
    try:
        take = None if idx is None else lambda a: [a[i] for i in idx]
        flat = om.read_flat(om.download(url, workers=8), {v for v, _ in names.values()}, take)
    except Exception as e:
        log(f"  f{fh:03d}: no Open-Meteo {domain} hour ({e}): without {', '.join(names)}")
        return {}
    return {canon: (flat[v], conv) for canon, (v, conv) in names.items() if v in flat}

def om_convert(v, conv, valid, lat2d, lon2d):
    if conv == 'K':
        return v - 273.15
    if conv == 'Pa':
        return v / 100
    if conv == 'sun':
        return v * instant_factor(valid, 1, lat2d, lon2d)
    return v
OpenMeteoBase = 'https://openmeteo.s3.amazonaws.com'
CIN_LID = (100.0, 250.0)
BLH_HEAT = (20.0, 80.0)
OLR_TOP_COVER = 70
OLR_TOP_COLDER = 15
STEFAN_BOLTZMANN = 5.67e-08

def cloud_top_from_olr(olr, t, z, t_hi, p_lv, p_hi):
    tb = (np.maximum(np.nan_to_num(olr, nan=0), 1.0) / STEFAN_BOLTZMANN) ** 0.25 - 273.15
    ts, zs_, p = (list(t), list(z), list(p_lv))
    for th, ph in zip(t_hi, p_hi):
        tv = (ts[-1] + th) / 2 + 273.15
        zs_.append(zs_[-1] + P.RD * tv / P.G * np.log(p[-1] / ph))
        ts.append(th)
        p.append(ph)
    T, Z = (np.stack(ts), np.stack(zs_))
    below = T <= tb[None]
    k = np.argmax(below, axis=0)
    found = below.any(axis=0)
    k0 = np.maximum(k - 1, 0)
    take = lambda a, i: np.take_along_axis(a, i[None], 0)[0]
    t1, t0, z1, z0 = (take(T, k), take(T, k0), take(Z, k), take(Z, k0))
    f = np.where(k > 0, np.clip((t0 - tb) / np.where(t0 - t1 == 0, 1, t0 - t1), 0, 1), 0)
    inside = z0 + f * (z1 - z0)
    above = Z[-1] + np.minimum(np.maximum(T[-1] - tb, 0) / 0.0065, 3000)
    return (np.where(found, inside, above).astype(np.float32), tb)
CONTRAILS = os.environ.get('CONTRAILS', '1') != '0'

def add_contrails(s, lv, hi, hi_ok, src, valid, lat2d, lon2d, levels=None):
    if not CONTRAILS or valid is None or lv.get('cloud') is None or (s.get('hcc') is None):
        return
    try:
        import contrails
        fix = contrails.ISSR_FIX.get(src.id)
        if fix is None:
            return
        have, rh_c = (list(lv['cloud'][0]), lv['cloud'][1])
        lev = list(src.levels if levels is None else levels)
        if 300 not in lev or 400 not in lev or lv.get('u') is None or (lv.get('z') is None):
            return
        i3, i4 = (lev.index(300), lev.index(400))
        hi_t = {src.cloud_levels[j]: hi['t'][j] - 273.15 for j in hi_ok} if 't' in hi else {}
        nanf = np.full(lat2d.shape, np.nan, np.float32)
        t = np.stack([lv['t'][i3] if p == 300 else hi_t.get(p, nanf) for p in contrails.LEVELS])
        rhi = np.stack([rh_c[have.index(float(p))] if float(p) in have else nanf for p in contrails.LEVELS])
        u = np.stack([lv['u'][i3]] * len(contrails.LEVELS))
        v = np.stack([lv['v'][i3]] * len(contrails.LEVELS))
        dz = np.maximum(lv['z'][i3] - lv['z'][i4], 500)
        shear = np.hypot(lv['u'][i3] - lv['u'][i4], lv['v'][i3] - lv['v'][i4]) / dz
        r = contrails.at_grid(lat2d, lon2d, (t, rhi, u, v, shear), valid, None)
        if r is None:
            return
        mu = np.sin(np.radians(np.maximum(sun_elevation(valid, lat2d, lon2d), 0)))
        contrails.fold_issr(s, contrails.issr_mask(t, rhi), r[0], r[1], mu, fix)
        s['contrail'] = r[0]
    except Exception as e:
        log(f'  contrails skipped: {e!r}')

def load_hour(src, run, fh, regrid_holder, region, valid, raw=None, fronts_out=None):
    raw = fetch_hour(src, run, fh) if raw is None else raw
    if regrid_holder.get('r') is None and hasattr(src, 'static_url'):
        regrid_holder['r'], regrid_holder['rot'] = (Regridder.identity(region), None)
    if regrid_holder.get('r') is None:
        one = lambda m: m.m0 if isinstance(m, Blend) else m
        any_msg = one(next(iter(raw.values())))
        lat, lon = grib.latlons(any_msg)
        if lat.ndim == 1 or src.grid == 'latlon':
            lat = lat.reshape(lat.shape[0], -1)
        regrid_holder['r'] = Regridder.regular(lat, lon, region) if src.regular else Regridder(lat, lon, region)
        regrid_holder['rot'] = grib.earth_rotation(one(raw.get(src.surface['u10'])) or any_msg)
    rg = regrid_holder['r']
    if 'parts' not in regrid_holder:
        windows = regrid_holder['windows'] = keep_windows(src.id, region)
        regrid_holder['boxes_ll'] = [b for b in KEEP.get(src.id, ()) if boxes_windows(region, [b])]
        regrid_holder['parts'] = [rg] if windows is None else [rg.part(*w) for w, _ in windows]
        regrid_holder['boxes'] = None if windows is None else [w for w, _ in windows]
    windows, parts = (regrid_holder['windows'], regrid_holder['parts'])
    boxes = regrid_holder['boxes']
    coords = [np.meshgrid(p.lats, p.lons, indexing='ij') for p in parts]
    grib_msg = lambda m: isinstance(m, (bytes, bytearray, Blend))
    between = any((isinstance(m, Blend) for m in raw.values()))

    def sfc(nk):
        name, k = nk
        m = raw[k]
        if isinstance(m, Blend) and name in src.windowed:
            return blend_window(src, name, m, run)
        return _decode(m, region, boxes)
    with ThreadPoolExecutor(DECODE_THREADS) as ex:
        sk = [(name, k) for name, k in src.surface.items() if raw.get(k) is not None]
        vals = ex.map(sfc, sk) if all((grib_msg(raw[k]) for _, k in sk)) else map(sfc, sk)
        native = {name: None for name in src.surface}
        native.update(zip([n for n, _ in sk], vals))
        decoded = {}
        lk = [k for k, m in raw.items() if k not in src.surface.values() and grib_msg(m)]
        fbox = FRONTS_BOXES.get(src.id) if fronts_out is not None and src.levels and (850 in src.levels) else None
        f850, want850 = ({}, {(src.level_vars[c], '850 mb') for c in ('t', 'rh', 'td', 'u', 'v') if c in src.level_vars} if fbox else set())

        def cut(k):
            f = _decode(raw[k], region, boxes)
            if k in want850:
                f850[k] = f
            return [p(f) for p in parts]
        decoded.update(zip(lk, ex.map(cut, lk)))
    cache = {}

    def earlier(pfh, key):
        if pfh not in cache:
            keys = [src.surface[c] for c in src.windowed if native.get(c) is not None]
            have = kept_step(src, run, pfh)
            try:
                cache[pfh] = {k: grib.decode(have[k])[0] for k in keys if k in have} if have is not None and all((k in have for k in keys)) else _fetch(src, run, pfh, keys)
            except Exception as e:
                log(f'  f{fh:03d}: earlier step f{pfh:03d} unavailable ({e})')
                cache[pfh] = {}
        return cache[pfh].get(key)
    analysis_from_prev(src, run, fh, native)
    resolve_windows(src, run, fh, native, earlier)
    if src.zs_step0_only:
        if regrid_holder.get('zs') is None:
            key = src.surface['zs']
            z0 = native.get('zs') if fh == 0 else _fetch(src, run, 0, [key]).get(key)
            regrid_holder['zs'] = [None if z0 is None else p(z0 / (P.G if src.zs_geopotential else 1)) for p in parts]
            regrid_holder['zs_whole'] = None if z0 is None else z0 / (P.G if src.zs_geopotential else 1)
        native['zs'] = ('done', regrid_holder['zs'])

    def surface(v, name, i):
        p, (lat2d, lon2d) = (parts[i], coords[i])
        if v is None:
            return None
        if isinstance(v, tuple) and v[0] == 'win':
            a = p(v[1])
            if name not in SUNLIT:
                return a
            f = instant_factor(valid, v[2], lat2d, lon2d, end=v[3] if len(v) > 3 else None)
            return np.where(a > 0, a * f, a) if name == 'shf' else a * f
        if isinstance(v, tuple) and v[0] == 'done':
            return v[1][i]
        return p(v)
    as_fraction = {}
    for k in ('tcc', 'lcc', 'mcc', 'hcc') if windows is not None else ():
        v = native.get(k)
        if v is not None:
            as_fraction[k] = np.nanmax(rg(v[1] if isinstance(v, tuple) else v)) <= 1.01

    def get(key):
        if key not in decoded:
            m = raw.get(key)
            decoded[key] = [None] * len(parts) if m is None else [p(f) for f in [_decode(m, region, boxes)] for p in parts]
        return decoded[key]
    lvp = {canon: [get((gname, f'{l} mb')) for l in src.levels] for canon, gname in src.level_vars.items()}
    keep = [k for k in range(len(src.levels)) if any((st[k][0] is not None for st in lvp.values()))]
    if len(keep) < len(src.levels):
        log(f'  f{fh:03d} levels left out (no messages): {[src.levels[k] for k in range(len(src.levels)) if k not in keep]}')
        lvp = {c: [st[k] for k in keep] for c, st in lvp.items()}
    use_levels = [src.levels[k] for k in keep]
    hip = {c: [get((src.level_vars[c], f'{l} mb')) for l in src.cloud_levels] for c in src.cloud_vars()}
    snd_extra = [l for l in src.snd_levels if l not in src.levels]
    sndp = {c: [get((src.level_vars[c], f'{l} mb')) for l in snd_extra] for c in src.snd_vars()} if snd_extra else {}
    rot = regrid_holder.get('rot')
    levels = np.array(use_levels, dtype=np.float32)
    if src.om_extra and 'om_idx' not in regrid_holder:
        import om
        rows = om.gaussian_rows()
        regrid_holder['om_idx'] = [om.gaussian_index(p.lats, p.lons, rows) for p in parts]
    extra = open_meteo_extra(src, run, fh, between, regrid_holder.get('om_idx'))
    out = []
    for i in range(len(parts)):
        lat2d, lon2d = coords[i]
        s = {name: surface(v, name, i) for name, v in native.items()}
        if isinstance(native.get('refd1'), np.ndarray):
            s['radar'] = parts[i].peak(native['refd1'])
        if src.zs_geopotential and (not src.zs_step0_only) and (s.get('zs') is not None):
            s['zs'] = s['zs'] / P.G
        lv = {}
        for canon, stack in lvp.items():
            stack = [x[i] for x in stack]
            lv[canon] = None if any((x is None for x in stack)) else np.stack(stack)
        for stack in lvp.values():
            for x in stack:
                x[i] = None
        hi = {c: [x[i] for x in v] for c, v in hip.items()}
        hi_ok = [j for j in range(len(src.cloud_levels)) if all((hi[c][j] is not None for c in hi if c != 'cc'))]
        for a, b in (('u10', 'v10'), ('uh', 'vh')):
            s[a], s[b] = grib.earth_winds(s.get(a), s.get(b), lon2d, rot)
        lv['u'], lv['v'] = grib.earth_winds(lv.get('u'), lv.get('v'), lon2d[None], rot)
        for k in ('t2', 'td2', 'tsfc'):
            if s.get(k) is not None:
                s[k] = s[k] - 273.15
        lv['t'] = lv['t'] - 273.15
        if src.moisture == 'td' and lv.get('td') is not None:
            lv['td'] = lv['td'] - 273.15
        s['ps'] = s['ps'] / 100
        if s.get('mslp') is not None:
            s['mslp'] = s['mslp'] / 100
        for k in ('prate', 'cprate', 'snowrate'):
            if s.get(k) is not None:
                s[k] = np.maximum(s[k], 0) * 3600
        if s.get('smoke') is not None:
            s['smoke'] = s['smoke'] * 1000000000.0
        for k in ('tcc', 'lcc', 'mcc', 'hcc'):
            if s.get(k) is not None and (np.nanmax(s[k]) <= 1.01 if windows is None else as_fraction[k]):
                s[k] = s[k] * 100
        bands = [s.get(k) for k in ('lcc', 'mcc', 'hcc')]
        if src.cover_calibrated and s.get('tcc') is not None and all((b is not None for b in bands)):
            s['tcc'], s['lcc'], s['mcc'] = P.icon_uncalibrate(s['tcc'], *bands, s.get('lsm'))
            bands = [s.get(k) for k in ('lcc', 'mcc', 'hcc')]
        if src.tcc_column_max and s.get('tcc') is not None and all((b is not None for b in bands)):
            s['tcc'] = total_of_bands(s['tcc'], bands)
        if src.moisture == 'rh':
            lv['td'] = P.dewpoint_from_rh(lv['t'], lv['rh'])
        rh_c = P.rh_cloud(lv['t'], lv['td']) if src.rh_over_water else cloud_rh(lv)
        lv['rh_cloud'] = rh_c
        clw_c, cc_c = (lv.get('clw'), lv.get('cc'))
        if hi_ok:
            pick = lambda c: np.stack([hi[c][j] for j in hi_ok])
            rh_hi = pick('rh') if 'rh' in hi else P.rh_cloud(pick('t') - 273.15, pick('td') - 273.15)
            if 'rh' in hi and src.rh_over_water:
                t_hi = pick('t') - 273.15
                rh_hi = P.rh_cloud(t_hi, P.dewpoint_from_rh(t_hi, rh_hi))
            rh_c = np.concatenate([rh_c, rh_hi])
            clw_c = None if clw_c is None else np.concatenate([clw_c, pick('clw')])
            cc_c = None if cc_c is None or any((hi['cc'][j] is None for j in hi_ok)) else np.concatenate([cc_c, pick('cc')])
        lv['cloud'] = (np.array(list(use_levels) + [src.cloud_levels[j] for j in hi_ok], np.float32), rh_c, clw_c, cc_c)
        lv['cover_tops'] = src.cover_tops
        if lv.get('w') is not None and src.omega:
            lv['w'] = omega_to_w(lv['w'], lv['t'], levels[:, None, None])
        lv['w_wave_only'] = src.w_wave_only
        if s.get('zs') is None:
            tv = s['t2'] + 273.15
            s['zs'] = np.maximum(0, P.RD * tv / P.G * np.log(s['mslp'] / s['ps']))
        if s.get('sw') is None:
            s['sw'] = solar_estimate(valid, lat2d, lon2d, s.get('tcc', np.zeros_like(s['t2'])))
        for k, (flat, conv) in extra.items():
            v = om_convert(flat[i], conv, valid, lat2d, lon2d)
            if k == 'sw' and s.get('rnsw') is not None and (s.get('sw') is not None):
                absorbed = np.clip(s['rnsw'] / np.maximum(s['sw'], 0.001), 0, 1)
                s['rnsw'] = np.where(np.isfinite(v) & (s['sw'] > 5), v * absorbed, s['rnsw']).astype(np.float32)
            old = s.get(k)
            s[k] = np.where(inside_of(s), v if old is None else np.where(np.isfinite(v), v, old), np.nan).astype(np.float32)
        snow = s.pop('snowrate', None)
        if snow is not None and s.get('prate') is not None:
            s['snowshare'] = np.where(s['prate'] >= 0.05, np.clip(snow / np.maximum(s['prate'], 1e-06), 0, 1), np.nan).astype(np.float32)
        olr = s.pop('olr', None)
        if olr is not None:
            t_hi = [hi['t'][j] - 273.15 for j in hi_ok] if 't' in hi else []
            z_olr, tb = cloud_top_from_olr(-olr, lv['t'], lv['z'], t_hi, list(use_levels), [src.cloud_levels[j] for j in hi_ok] if t_hi else [])
            cold = (np.nan_to_num(s.get('tcc', 0), nan=0) >= OLR_TOP_COVER) & (tb <= s['t2'] - OLR_TOP_COLDER)
            s['ctop_olr'] = np.where(cold, z_olr, np.nan).astype(np.float32)
        add_contrails(s, lv, hi, hi_ok, src, valid, lat2d, lon2d, use_levels)
        s['hub_m'] = src.hub_m
        s['hpbl_season'] = src.hpbl_season
        if src.derive:
            s = src.derive(s)
        rnsw, rnlw = (s.pop('rnsw', None), s.pop('rnlw', None))
        if s.get('shf') is None and rnsw is not None and (rnlw is not None):
            s['shf'] = heat_from_net_radiation(rnsw + rnlw, s['t2'], s['td2'], s.get('soilm9'), s.get('lsm'))
        if snd_hour(src, fh):
            s['_snd'] = sounding_column(src, s, lv, levels, {c: [x[i] for x in v] for c, v in sndp.items()}, snd_extra, lon2d, rot, origin=(0, 0) if windows is None else (windows[i][0][0], windows[i][0][2]))
        out.append((s, lv, levels))
    if fbox:
        try:
            fronts_out.extend(fronts_fields(src, rg, region, fbox, native, f850, regrid_holder))
        except Exception as e:
            log(f"  f{fh:03d} fronts' wide windows skipped: {e!r}")
    return out

def fronts_fields(src, rg, region, fboxes, native, f850, holder):
    if 'fparts' not in holder:
        holder['fparts'] = [rg.part(*w) for w, _ in boxes_windows(region, fboxes, halo=0)]
    need = ('t2', 'td2', 'u10', 'v10', 'ps', 'mslp')
    plain = lambda v: v if isinstance(v, np.ndarray) else None
    if any((plain(native.get(k)) is None for k in need)):
        return []
    lvk = {c: (src.level_vars[c], '850 mb') for c in ('t', 'rh', 'td', 'u', 'v') if c in src.level_vars}
    moist = 'rh' if src.moisture == 'rh' else 'td'
    if any((f850.get(lvk.get(c)) is None for c in ('t', moist, 'u', 'v'))):
        return []
    zs_whole = holder.get('zs_whole') if src.zs_step0_only else plain(native.get('zs'))
    if zs_whole is not None and src.zs_geopotential and (not src.zs_step0_only):
        zs_whole = zs_whole / P.G
    rot = holder.get('rot')
    out = []
    for p in holder['fparts']:
        lat2d, lon2d = np.meshgrid(p.lats, p.lons, indexing='ij')
        s = {k: p(native[k]) for k in need}
        s['u10'], s['v10'] = grib.earth_winds(s['u10'], s['v10'], lon2d, rot)
        t = p(f850[lvk['t']]) - 273.15
        u, v = grib.earth_winds(p(f850[lvk['u']]), p(f850[lvk['v']]), lon2d, rot)
        td = P.dewpoint_from_rh(t, p(f850[lvk['rh']])) if moist == 'rh' else p(f850[lvk['td']]) - 273.15
        s['t2'], s['td2'] = (s['t2'] - 273.15, s['td2'] - 273.15)
        s['ps'], s['mslp'] = (s['ps'] / 100, s['mslp'] / 100)
        if zs_whole is not None:
            s['zs'] = p(zs_whole)
        else:
            s['zs'] = np.maximum(0, P.RD * (s['t2'] + 273.15) / P.G * np.log(s['mslp'] / s['ps']))
        out.append((s, {'t': t[None], 'td': td[None], 'u': u[None], 'v': v[None]}, np.array([850.0], np.float32), p))
    return out
SND_VARS = ('t', 'dpd', 'u', 'v', 'z', 'w')
SND_SFC = {'ps': (400.0, 1100.0), 'sw': (0.0, 1270.0)}
SND_MAP = ('elevation', 't2', 'td2', 'mslp', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'rain', 'showers', 'cape', 'zi', 'heat', 'u_sfc', 'v_sfc', 'gust', 'vis', *(f'cc{p}' for p in P.CC_LEVELS))
SND_BLOCK = 8
SND_EVERY = {'gfs': (48, 3)}

def snd_enc(p):
    h = 44330.8 * (1 - (p / 1013.25) ** 0.190263)
    t = 15.0 - 0.0065 * min(h, 11000.0)
    return {'t': (round(t - 60, 2), round(t + 60, 2)), 'dpd': (0.0, 50.0), 'u': (-100.0, 100.0), 'v': (-100.0, 100.0), 'z': (round(h - 1200), round(h + 1200)), 'w': (-10.0, 10.0)}

def snd_quantize(a, lo, hi):
    a = np.asarray(a, np.float32)
    q = np.clip(np.rint((a - lo) / (hi - lo) * 254), 0, 254)
    return np.where(np.isfinite(a), q, 255).astype(np.uint8)

def sounding_column(src, s, lv, levels, extra, extra_levels, lon2d, rot, origin=(0, 0)):
    S = src.snd_step
    oy, ox = (-origin[0] % S, -origin[1] % S)
    thin = lambda a: None if a is None else np.ascontiguousarray(np.asarray(a, np.float32)[..., oy::S, ox::S])
    lev = [float(x) for x in levels]
    rows = {}
    for j, p in enumerate(extra_levels):
        g = lambda c: (extra.get(c) or [None] * len(extra_levels))[j]
        t = g('t')
        t = None if t is None else t - 273.15
        if 'td' in src.level_vars:
            td = g('td')
            td = None if td is None else td - 273.15
        else:
            rh = g('rh')
            td = None if t is None or rh is None else P.dewpoint_from_rh(t, rh)
        u, v = (g('u'), g('v'))
        if u is not None and v is not None:
            u, v = grib.earth_winds(u, v, lon2d, rot)
        w = g('w')
        if w is not None and src.omega and (t is not None):
            w = omega_to_w(w, t, p)
        rows[p] = {'t': t, 'td': td, 'u': u, 'v': v, 'z': g('z'), 'w': w}
    shape = thin(s['ps']).shape
    out = {c: [] for c in ('t', 'dpd', 'u', 'v', 'z', 'w')}
    for p in src.snd_levels:
        if p in rows:
            d = rows[p]
        else:
            k = lev.index(float(p)) if float(p) in lev else None
            d = {c: None if k is None or lv.get(c) is None else lv[c][k] for c in ('t', 'td', 'u', 'v', 'z', 'w')}
        d = {c: thin(x) for c, x in d.items()}
        d['dpd'] = None if d['t'] is None or d['td'] is None else np.maximum(d['t'] - d['td'], 0)
        for c in out:
            out[c].append(np.full(shape, np.nan, np.float32) if d[c] is None else d[c])
    col = {c: np.stack(a) for c, a in out.items()}
    col['ps'], col['sw'] = (thin(s['ps']), thin(s.get('sw')))
    return col

def paste_thin(windows, grids, shape, S):
    if windows is None or S == 1:
        return paste(windows, grids, shape)
    up = lambda a: -(-a // S)
    out = None
    for ((y0, y1, x0, x1), (iy0, iy1, ix0, ix1)), v in zip(windows, grids):
        if v is None:
            continue
        if out is None:
            out = np.full((up(shape[0]), up(shape[1])), np.nan, np.float32)
        gy0, gy1, gx0, gx1 = (up(y0 + iy0), up(y0 + iy1), up(x0 + ix0), up(x0 + ix1))
        out[gy0:gy1, gx0:gx1] = v[gy0 - up(y0):gy1 - up(y0), gx0 - up(x0):gx1 - up(x0)]
    return out

def sounding_grid(src, parts, windows, shape, q):
    S = src.snd_step
    cols = [s.get('_snd') for s, _, _ in parts]
    if any((c is None for c in cols)):
        return None
    out = []
    for k, p in enumerate(src.snd_levels):
        enc = snd_enc(p)
        for c in SND_VARS:
            out.append(snd_quantize(paste_thin(windows, [col[c][k] for col in cols], shape, S), *enc[c]))
    for c, (lo, hi) in SND_SFC.items():
        g = paste_thin(windows, [col[c] for col in cols], shape, S)
        out.append(np.full(out[0].shape, 255, np.uint8) if g is None else snd_quantize(g, lo, hi))
    for n in SND_MAP:
        out.append(np.ascontiguousarray(q[n][::S, ::S]) if n in q else np.full(out[0].shape, 255, np.uint8))
    return np.stack(out)
_CTX: dict = {}
HOURS_VERSION = 28

def _job_hour(job):
    run, fh = job if isinstance(job, tuple) else (_CTX['run'], job)
    return (run, fh, run + timedelta(hours=fh))

def _kept(run, valid):
    hours_dir = _CTX['hours']
    npz, stamp = (os.path.join(hours_dir, f'{valid:%Y%m%d%H}{ext}') for ext in ('.npz', '.run'))
    if _CTX.get('recompute') or not (os.path.exists(npz) and os.path.exists(stamp)):
        return False
    with open(stamp) as f:
        return f.read().strip() == f'{run:%Y%m%d%H}'

def _prefetch(job):
    run, fh, valid = _job_hour(job)
    if _kept(run, valid):
        return None
    try:
        return fetch_hour(_CTX['src'], run, fh)
    except Exception as e:
        return e

def _process_hour(job, raw=None):
    src, region, hours_dir, holder = (_CTX[k] for k in ('src', 'region', 'hours', 'holder'))
    run, fh, valid = _job_hour(job)
    t0 = time.time()
    npz, stamp = (os.path.join(hours_dir, f'{valid:%Y%m%d%H}{ext}') for ext in ('.npz', '.run'))
    if _kept(run, valid):
        log(f'  f{fh:03d} {valid:%a %HZ} kept (already computed from this run)')
        return valid
    try:
        if isinstance(raw, Exception):
            raise raw
        step = src is None or bool(src.fstep(fh))
        wide = []
        parts = load_hour(src, run, fh, holder, region, valid, raw=raw, fronts_out=wide if step else None)
    except Exception as e:
        log(f'  f{fh:03d} skipped: {e}')
        return None
    rg, windows = (holder['r'], holder['windows'])
    if 'll' not in holder:
        holder['ll'] = (np.repeat(rg.lats[:, None], len(rg.lons), axis=1), np.repeat(rg.lons[None, :], len(rg.lats), axis=0))
    lat2d, lon2d = holder['ll']
    q = analyse_codes(parts, part_coords(holder['parts']), region[4], valid, windows, lat2d.shape)
    q['elevation'] = paste_codes(windows, [quantize(s['zs'], *ENC['elevation']) for s, _, _ in parts], lat2d.shape)
    keep = published_mask(src.id, lat2d, lon2d, q)
    if keep is not None:
        for v in q.values():
            v[~keep] = 255
    if _CTX.pop('pick_physics', False):
        pick_physics(q, lambda: analyse_parts(parts, part_coords(holder['parts']), region[4], valid, windows, lat2d.shape), fh, keep)
    if not step:
        fr = None
    elif wide:
        cover = fronts_cover(holder['boxes_ll'] if 'boxes_ll' in holder else KEEP.get(src.id, ()), FRONTS_BOXES[src.id])
        rest = [i for i, c in enumerate(cover) if not c] if holder.get('windows') is not None else []
        fr = hour_fronts([(s, lv, lev) for s, lv, lev, _ in wide] + [parts[i] for i in rest], [r for *_, r in wide] + [holder['parts'][i] for i in rest], fh)
    else:
        fr = hour_fronts(parts, holder['parts'], fh)
    if fr is not None:
        q['fronts'], fp = fr
        if fp is not None:
            q['frontp'] = fp
    if snd_hour(src, fh):
        snd = sounding_grid(src, parts, windows, lat2d.shape, q)
        if snd is not None:
            q['snd'] = snd
    del parts, wide
    trim_malloc()
    q.update(sky.cloud_overview(q, ENC, region[4] <= sky.FINE_STEP))
    q.update(_run_fields(q, lat2d, lon2d, region, valid))
    part = os.path.join(hours_dir, f'{valid:%Y%m%d%H}.tmp.npz')
    tiles.savez(part, q)
    os.replace(part, npz)
    with open(stamp, 'w') as f:
        f.write(f'{run:%Y%m%d%H}')
    log(f'  f{fh:03d} {valid:%a %HZ} done in {time.time() - t0:.1f}s{_mem_note()}')
    return valid

def hour_fronts(parts, regrids, fh=0):
    try:
        res = fronts.merge([fronts.from_part(s, lv, levels, rg.lats, rg.lons) for (s, lv, levels), rg in zip(parts, regrids)])
        fp = fronts.pack_states(res['states']) if res['states'] else None
        return (np.frombuffer(json.dumps(fronts.encode(res), separators=(',', ':')).encode(), np.uint8), fp)
    except Exception as e:
        log(f'  f{fh:03d} fronts skipped: {e!r}')
        return None
NOT_TILES = {'elevation', 'snd', 'fronts', 'frontp', 'acscore', 'acbase', 'accape', 'accin'}
STEP_FRONTS_H = 2

def write_fronts(rdir, hours_dir, times, day_of=None, around=None):
    folders: dict[str, dict] = {}
    own = []
    for t in sorted(set(around or ()) | set(times)):
        with np.load(os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz')) as h:
            if 'fronts' in h.files:
                own.append((t, json.loads(h['fronts'].tobytes())))

    def states(t):
        try:
            with np.load(os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz')) as h:
                return fronts.unpack_states(h['frontp']) if 'frontp' in h.files else None
        except Exception:
            return None
    try:
        steady = fronts.day_fronts(own, states)
    except Exception as e:
        log(f'  fronts over the hours skipped: {e!r}')
        steady = dict(own)
    want = set(times)
    have = sorted(steady)
    for t in sorted(want - set(steady)):
        near = min(have, key=lambda u: (abs((u - t).total_seconds()), u > t), default=None)
        if near is not None and abs((near - t).total_seconds()) <= STEP_FRONTS_H * 3600:
            steady[t] = steady[near]
    for t, enc in steady.items():
        if t not in want:
            continue
        folders.setdefault((day_of or day_key)(t), {})[f'{t:%Y%m%d%H}'] = enc
    for folder, hours in folders.items():
        os.makedirs(os.path.join(rdir, folder), exist_ok=True)
        with open(os.path.join(rdir, folder, 'fronts.json'), 'w') as f:
            json.dump(hours, f, separators=(',', ':'))
    return bool(folders)
NATIVE_TOLERANCE = 0.0001

def pick_physics(q, compute, fh, keep=None):
    global P
    import physics_native
    if not physics_native.available():
        log(f'physics: physics.py (C++ not available: {physics_native.error})')
        return
    t0 = time.time()
    P = physics_native
    try:
        fields = compute()
        qc = {k: quantize(v, *ENC[k]) for k, v in fields.items() if v is not None and k in ENC}
        qc['elevation'] = q['elevation']
        if keep is not None:
            for v in qc.values():
                v[~keep] = 255
    except Exception as e:
        P = physics
        log(f'::warning::physics: the C++ failed on f{fh:03d} ({e!r}), physics.py for this run')
        return
    bad, near = ([], [])
    for k in sorted(set(q) | set(qc)):
        a, b = (q.get(k), qc.get(k))
        if a is None or b is None or a.shape != b.shape:
            bad.append(f'{k} missing')
            continue
        diff = a != b
        if not diff.any():
            continue
        share = float(diff.mean())
        worst = int(np.abs(a.astype(np.int16) - b.astype(np.int16))[diff].max())
        (near if share <= NATIVE_TOLERANCE and worst <= 1 else bad).append(f'{k} {diff.sum()} cells, {worst} code(s)')
    if bad:
        P = physics
        log(f"::warning::physics: the C++ differs from physics.py on f{fh:03d} ({'; '.join(bad[:8])}), physics.py for this run")
        return
    log(f'physics: C++ (matches physics.py on f{fh:03d}: {len(q) - len(near)} of {len(q)} fields identical' + (f"; {', '.join(near)}" if near else '') + f'; checked in {time.time() - t0:.1f}s)')
PREFETCH_AHEAD = 2

def lean_malloc():
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        libc.mallopt(-3, 1 << 20)
        return libc
    except Exception:
        return None

def trim_malloc():
    import gc
    gc.collect()
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass
_LIBC = None

def _mem_note():
    try:
        hwm = avail = None
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    hwm = int(line.split()[1]) / 1048576
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) / 1048576
        return f' (peak {hwm:.1f} GB, {avail:.1f} GB free)' if hwm is not None and avail is not None else ''
    except (OSError, ValueError):
        return ''
LANE_STAGGER_S = 20

def _lane(jobs, results, stagger=0.0):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(PREFETCH_AHEAD) as ex:
        ahead = deque((ex.submit(_prefetch, job) for _, job in jobs[:PREFETCH_AHEAD]))
        if stagger > 0:
            time.sleep(stagger)
        for n, (i, job) in enumerate(jobs):
            try:
                raw = ahead.popleft().result()
            except Exception as e:
                raw = e
            if n + PREFETCH_AHEAD < len(jobs):
                ahead.append(ex.submit(_prefetch, jobs[n + PREFETCH_AHEAD][1]))
            try:
                r = _process_hour(job, raw)
            except Exception as e:
                log(f'  {job}: failed ({e!r})')
                r = None
            del raw
            trim_malloc()
            results.put((i, r))
LANE_RUN = 12

def shard_hours(fhs, i, n, run=1):
    blocks = [fhs[k:k + run] for k in range(0, len(fhs), run)]
    return [fh for b in blocks[i::n] for fh in b]

def split_first(jobs):
    kept, rest, first = ([], [], None)
    for j in jobs:
        r, _, valid = _job_hour(j)
        if _kept(r, valid):
            kept.append(j)
        elif first is None:
            first = j
        else:
            rest.append(j)
    return (kept, first, rest)

def lane_jobs(jobs, workers, run=1):
    indexed = list(enumerate(jobs))
    blocks = [indexed[i:i + run] for i in range(0, len(indexed), run)]
    return [[j for b in blocks[w::workers] for j in b] for w in range(workers)]

def process_lanes(jobs, workers, progress=None, run=1):
    import multiprocessing as mp
    import queue as _queue
    ctx = mp.get_context('fork')
    results = ctx.Queue()
    shares = lane_jobs(jobs, workers, run)
    procs = [ctx.Process(target=_lane, args=(shares[w], results, w * LANE_STAGGER_S), daemon=True) for w in range(workers)]
    for p in procs:
        p.start()
    out, got = ([None] * len(jobs), 0)
    while got < len(jobs):
        try:
            i, r = results.get(timeout=30)
        except _queue.Empty:
            if not any((p.is_alive() for p in procs)):
                log(f'::warning::hour workers stopped with {len(jobs) - got} hours left')
                break
            continue
        out[i] = r
        got += 1
        if progress:
            progress.tick()
    for p in procs:
        p.join(timeout=10)
    return out
QUARTER_MAX_FH = 18
QUARTER_FIELDS = ('ceil', 'vis', 'refc', 'rain', 'ptype', 'u_sfc', 'v_sfc', 'gust', 't2', 'td2', 'ctop')
SUBH_VARS = {'zs': ('HGT', 'surface'), 't2': ('TMP', '2 m above ground'), 'td2': ('DPT', '2 m above ground'), 'u10': ('UGRD', '10 m above ground'), 'v10': ('VGRD', '10 m above ground'), 'gust': ('GUST', 'surface'), 'vis': ('VIS', 'surface'), 'refc': ('REFC', 'entire atmosphere'), 'refd1': ('REFD', '1000 m above ground'), 'prate': ('PRATE', 'surface'), 'ceil': ('HGT', 'cloud ceiling'), 'ctop': ('HGT', 'cloud top'), 'crain': ('CRAIN', 'surface'), 'csnow': ('CSNOW', 'surface'), 'cfrzr': ('CFRZR', 'surface'), 'cicep': ('CICEP', 'surface')}

def subh_url(run, fh):
    return f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrfsubhf{fh:02d}.grib2'

def quarter_fields(s, step=None):
    t2 = s['t2'] - 273.15
    inside = np.isfinite(t2)
    nan = lambda a: None if a is None else np.where(inside, a, np.nan)
    rain = np.maximum(np.nan_to_num(s['prate'], nan=0), 0) * 3600 if s.get('prate') is not None else None
    out = {'t2': t2, 'td2': None if s.get('td2') is None else s['td2'] - 273.15, 'u_sfc': s.get('u10'), 'v_sfc': s.get('v10'), 'gust': nan(s.get('gust')), 'vis': None if s.get('vis') is None else nan(np.clip(s['vis'], 0, 16100)), 'refc': nan(radar({'radar': s.get('refd1'), 'refc': s.get('refc')}, step=step)), 'rain': nan(rain), 'ctop': nan(s.get('ctop'))}
    if s.get('ceil') is not None and s.get('zs') is not None:
        out['ceil'] = P.ceiling_agl(s['ceil'], s['zs'], inside)
    if rain is not None:
        out['ptype'] = nan(P.precip_type(rain, s.get('crain'), s.get('csnow'), s.get('cfrzr'), s.get('cicep'), None, t2))
    return {k: v for k, v in out.items() if v is not None}

def _process_quarters(fh):
    run, region, qdir, holder = (_CTX[k] for k in ('run', 'region', 'quarters', 'holder'))
    t0 = time.time()
    url = subh_url(run, fh)
    done = []
    try:
        rows = grib.parse_noaa_idx(grib.http_get(url + '.idx', tries=4).decode())
    except Exception as e:
        log(f'  q{fh:03d} skipped: {e}')
        return done
    for m in (fh * 60 - 45, fh * 60 - 30, fh * 60 - 15):
        valid = run + timedelta(minutes=m)
        npz, stamp = (os.path.join(qdir, f'{valid:%Y%m%d%H%M}{ext}') for ext in ('.npz', '.run'))
        if not _CTX.get('recompute') and os.path.exists(npz) and os.path.exists(stamp):
            with open(stamp) as f:
                if f.read().strip() == f'{run:%Y%m%d%H}':
                    done.append(valid)
                    continue
        want = f'{m} min fcst'
        sel = {}
        for canon, (var, lev) in SUBH_VARS.items():
            r = next((r for r in rows if r['var'] == var and r['level'] == lev and (r['fcst'] == want)), None)
            if r:
                sel[canon] = r
        if 't2' not in sel:
            continue
        try:
            raw = grib.fetch_messages(url, sel, workers=8, gap=1 << 20, tries=4)
            if holder.get('r') is None:
                lat, lon = grib.latlons(next(iter(raw.values())))
                holder['r'] = Regridder(lat, lon, region)
            if 'rot' not in holder:
                holder['rot'] = grib.earth_rotation(raw.get('u10') or next(iter(raw.values())))
            rg = holder['r']
            s = {k: (rg.peak if k == 'refd1' else rg)(grib.decode(v)[0]) for k, v in raw.items()}
            s['u10'], s['v10'] = grib.earth_winds(s.get('u10'), s.get('v10'), rg.lons[None, :], holder['rot'])
            fields = quarter_fields(s, region[4])
        except Exception as e:
            log(f'  q{fh:03d} +{m % 60:02d} skipped: {e}')
            continue
        q = {k: quantize(v, *ENC[k]) for k, v in fields.items() if k in ENC}
        part = os.path.join(qdir, f'{valid:%Y%m%d%H%M}.tmp.npz')
        tiles.savez(part, q)
        os.replace(part, npz)
        with open(stamp, 'w') as f:
            f.write(f'{run:%Y%m%d%H}')
        done.append(valid)
    log(f"  q{fh:03d} quarter hours {', '.join((f'{t:%H%M}' for t in done)) or 'none'} in {time.time() - t0:.1f}s")
    return done

def _quarter_job(job):
    rdir, qdir, day, stamps, name, tile = job
    frames = [np.load(os.path.join(qdir, f'{s}.npz')) for s in stamps]
    shape = frames[0]['t2'].shape
    stack = np.stack([f[name] if name in f.files else np.full(shape, 255, np.uint8) for f in frames])
    for f in frames:
        f.close()
    return (f'q_{name}', _write_strips(stack, tile, os.path.join(rdir, day, f'q_{name}')))

def write_quarters(rdir, qdir, times, tile, only=None, workers=1):
    days: dict[str, list] = {}
    for t in sorted(times):
        days.setdefault(day_key(t), []).append(t)
    jobs = []
    for day, dts in days.items():
        stamps = [f'{t:%Y%m%d%H%M}' for t in dts]
        files = set()
        for st in stamps:
            with np.load(os.path.join(qdir, f'{st}.npz')) as f:
                files |= set(f.files)
        jobs += [(rdir, qdir, day, stamps, n, tile) for n in QUARTER_FIELDS if (only is None or n in only) and n in files]
    present: dict[str, set] = {}
    for name, keys in run_jobs(_quarter_job, jobs, workers):
        if keys:
            present.setdefault(name, set()).update(keys)
    return ({d: [t.strftime('%Y%m%d%H%M') for t in ts] for d, ts in days.items()}, present)

def quarter_manifest(qdays, qpresent):
    return {k: {'min': ENC[k[2:]][0], 'max': ENC[k[2:]][1], 'transform': ENC[k[2:]][2], 'tiles': sorted(v)} for k, v in sorted(qpresent.items())}

def stored_quarters(qdir, lo, hi, prune=False):
    out = []
    if not os.path.isdir(qdir):
        return out
    for f in sorted(os.listdir(qdir)):
        if not f.endswith('.npz') or f.endswith('.tmp.npz'):
            continue
        t = datetime.strptime(f[:-4], '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
        if lo <= t <= hi:
            out.append(t)
        elif prune:
            for ext in ('.npz', '.run'):
                if os.path.exists(os.path.join(qdir, f[:-4] + ext)):
                    os.remove(os.path.join(qdir, f[:-4] + ext))
    return out

def day_key(valid):
    return f'{valid - timedelta(hours=10):%Y%m%d}'
SQRT_MEAN = {'cudepth'}

def _sqrt_from_zero(name):
    e = ENC.get(name) or (ENC.get(name[2:]) if name.startswith('q_') else None)
    return bool(e) and e[2] == 'sqrt' and (e[0] == 0) and (name not in SQRT_MEAN)

def level_grid(grid, k, tile=LEVEL_TILE):
    step = grid['step']
    nx, ny = (-(-grid['nx'] // k), -(-grid['ny'] // k))
    return {'west': round(grid['west'] + (k - 1) / 2 * step, 9), 'north': round(grid['north'] - (k - 1) / 2 * step, 9), 'step': round(k * step, 9), 'nx': nx, 'ny': ny, 'tile': tile, 'ntx': -(-nx // tile), 'nty': -(-ny // tile)}

def overview_manifest(grid, levels, lpresent):
    return [{'k': k, 'path': f'o{k}', 'grid': level_grid(grid, k, tile), 'tiles': {n: sorted(v) for n, v in sorted(lpresent.get(k, {}).items())}} for k, tile in levels]
COMPRESS_LEVEL = 4
PNG_TOO_UNTIL = datetime(2026, 10, 11, 19, tzinfo=timezone.utc)

def png_too():
    return datetime.now(timezone.utc) < PNG_TOO_UNTIL
HOUR_SEGS = True

def _write_strips(stack, tile, folder):
    stack = stack if stack.ndim == 3 else stack[None]
    n, ny, nx = stack.shape
    ntx, nty = (-(-nx // tile), -(-ny // tile))
    keys, blobs = (set(), [b''] * (ntx * nty))
    hours = [None] * (ntx * nty) if n > 1 and HOUR_SEGS else None
    for ty in range(0, ny, tile):
        for tx in range(0, nx, tile):
            block = stack[:, ty:ty + tile, tx:tx + tile]
            if (block == 255).all():
                continue
            pad = np.full((n, tile, tile), 255, np.uint8)
            pad[:, :block.shape[1], :block.shape[2]] = block
            i = ty // tile * ntx + tx // tile
            if hours is not None:
                blobs[i], hours[i] = tiles.png_hours(pad)
            else:
                blobs[i] = tiles.png_encode(pad.reshape(-1, tile), COMPRESS_LEVEL)
            keys.add(f'{tx // tile}_{ty // tile}')
    if keys:
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        pack.write_pack(folder + '.pack', blobs, ntx, nty, hours, n)
        if png_too():
            os.makedirs(folder, exist_ok=True)
            for k in keys:
                tx, ty = (int(v) for v in k.split('_'))
                with open(os.path.join(folder, f'{k}.png'), 'wb') as f:
                    f.write(blobs[ty * ntx + tx])
    return keys

def run_jobs(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(min(workers, len(jobs))) as pool:
        return list(pool.imap_unordered(fn, jobs))

def _link_strips(src, dst):
    if os.path.exists(src + '.pack'):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.link(src + '.pack', dst + '.pack')
        names = sorted((f for f in os.listdir(src) if f.endswith('.png'))) if png_too() and os.path.isdir(src) else []
        if names:
            os.makedirs(dst, exist_ok=True)
        for f in names:
            os.link(os.path.join(src, f), os.path.join(dst, f))
        return pack.keys_in(dst + '.pack')
    return None if os.path.isdir(src) else set()

def _run_fields(q, lat2d, lon2d, region, valid):
    grid = {'west': float(lon2d[0, 0]), 'north': float(lat2d[0, 0]), 'step': region[4], 'nx': lat2d.shape[1], 'ny': lat2d.shape[0]}
    try:
        return runfields.publish(q, ENC, grid, valid)
    except Exception as e:
        log(f'  {valid:%a %HZ} ridge/wave running and shade not published: {e}')
        return {}

def _runfields_job(job):
    hours_dir, t, region = job
    path = os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz')
    with np.load(path) as h:
        if 'wtop' in h.files or 'wstar' not in h.files:
            return 0
        q = {k: h[k] for k in h.files}
    lons, lats = target_grid(region)
    lat2d, lon2d = (np.repeat(np.asarray(lats)[:, None], len(lons), axis=1), np.repeat(np.asarray(lons)[None, :], len(lats), axis=0))
    add = _run_fields(q, lat2d, lon2d, region, t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t)
    if not add:
        return 0
    q.update(add)
    part = path[:-4] + '.tmp.npz'
    tiles.savez(part, q)
    os.replace(part, path)
    return 1

def _sky_job(job):
    hours_dir, t, fine = job
    path = os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz')
    with np.load(path) as h:
        if 'cuclass' in h.files or 'wstar' not in h.files:
            return 0
        q = {k: h[k] for k in h.files}
    add = sky.cloud_overview(q, ENC, fine)
    if not add:
        return 0
    q.update(add)
    part = path[:-4] + '.tmp.npz'
    tiles.savez(part, q)
    os.replace(part, path)
    return 1
HOUR_CLASS = os.environ.get('HOUR_CLASS') == '1'
HOUR_CLASS_READS = tuple(dict.fromkeys(('cuclass',) + sky.INPUTS + sky.TIME_SMOOTHED))

def _stored(path, names):
    with np.load(path) as h:
        return {k: h[k] for k in names if k in h.files}

def _hour_class_job(job):
    hours_dir, t, before, after, fine = job
    path = os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz')
    side = lambda u: None if u is None else _stored(os.path.join(hours_dir, f'{u:%Y%m%d%H}.npz'), sky.TIME_SMOOTHED)
    add = sky.hour_class(_stored(path, HOUR_CLASS_READS), side(before), side(after), ENC, fine)
    if not add:
        return 0
    with np.load(path) as h:
        q = {k: h[k] for k in h.files}
    q.update(add)
    part = path[:-4] + '.tmp.npz'
    tiles.savez(part, q)
    os.replace(part, path)
    return 1

def hour_to_hour(hours_dir, times, fine, workers=1):
    times = sorted(times)
    have = set(times)
    near = lambda t, dh: t + timedelta(hours=dh) if t + timedelta(hours=dh) in have else None
    if HOUR_CLASS:
        run_jobs(_hour_class_job, [(hours_dir, t, near(t, -1), near(t, 1), fine) for t in times], workers)
    held, prev, writes = (0, None, [])
    lows = {}
    low_of = lambda u: None if u is None else lows.setdefault(u, _stored(os.path.join(hours_dir, f'{u:%Y%m%d%H}.npz'), ('cloudlow',)))
    for t in times:
        path = os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz')
        q = low_of(t)
        cls = _stored(path, ('cuclass',)).get('cuclass')
        if cls is None or 'cloudlow' not in q:
            prev = None
            continue
        b, a = (low_of(near(t, -1)), low_of(near(t, 1)))
        low = sky.decode(sky.time_smoothed(q, b, a, ENC)['cloudlow'], *ENC['cloudlow'])
        new = sky.hold_decks(prev if near(t, -1) else None, cls, low)
        changed = int((new != cls).sum())
        if changed:
            held += changed
            np.save(path[:-4] + '.hold.npy', new)
            writes.append((path,))
        prev = new
        lows.pop(near(t, -1), None)
    run_jobs(_hold_job, writes, workers)
    return held

def _hold_job(job):
    path, = job
    side = path[:-4] + '.hold.npy'
    with np.load(path) as h:
        q = {k: h[k] for k in h.files}
    q['cuclass'] = np.load(side)
    part = path[:-4] + '.tmp.npz'
    tiles.savez(part, q)
    os.replace(part, path)
    os.remove(side)
    return 1

def _tile_job(job):
    rdir, hours_dir, day, stamps, name, tile, levels, reuse = job
    if reuse is not None:
        keys = _link_strips(os.path.join(reuse, day, name), os.path.join(rdir, day, name))
        lkeys = {}
        for k, _ in levels if name not in LEVEL_SKIP else ():
            lk = _link_strips(os.path.join(reuse, f'o{k}', day, name), os.path.join(rdir, f'o{k}', day, name))
            if lk is None:
                keys = None
            elif lk:
                lkeys[k] = lk
        if keys is not None:
            return (name, keys, lkeys)
        for p in [os.path.join(rdir, day, name)] + [os.path.join(rdir, f'o{k}', day, name) for k, _ in levels]:
            if os.path.exists(p + '.pack'):
                os.remove(p + '.pack')
            shutil.rmtree(p, ignore_errors=True)
    hours = [np.load(os.path.join(hours_dir, f'{s}.npz')) for s in stamps]
    if day is None:
        grid = hours[0]['elevation']
    else:
        shape = hours[0]['wstar'].shape
        grid = np.stack([h[name] if name in h.files else np.full(shape, 255, np.uint8) for h in hours])
        by = AT_PEAK.get(name)
        if by:
            by = np.stack([h[by] if by in h.files else np.full(shape, 255, np.uint8) for h in hours])
    for h in hours:
        h.close()
    path = () if day is None else (day,)
    keys = _write_strips(grid, tile, os.path.join(rdir, *path, name))
    lkeys = {}
    for k, ltile in levels if name not in LEVEL_SKIP else ():
        lv = downsample_at_peak(grid, by, k) if day is not None and name in AT_PEAK else downsample_peak(grid, k) if name in PEAK_FIELDS else downsample(grid, k, nearest=name in CLASS_FIELDS or name in SAMPLED_FIELDS, none=LEVEL_NONE.get(name), rms=_sqrt_from_zero(name))
        lk = _write_strips(lv, ltile, os.path.join(rdir, f'o{k}', *path, name))
        if lk:
            lkeys[k] = lk
    return (name, keys, lkeys)

def write_tiles(rdir, hours_dir, times, tile, elev_time, only=None, levels=(), workers=1, reuse=None, day_of=None, elevation=True):
    days: dict[str, list] = {}
    for t in times:
        days.setdefault((day_of or day_key)(t), []).append(t)
    present: dict[str, set] = {}
    lpresent: dict[int, dict[str, set]] = {k: {} for k, _ in levels}
    jobs = []
    for day, dts in days.items():
        stamps = [f'{t:%Y%m%d%H}' for t in dts]
        files = set()
        for st in stamps:
            with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
                files |= set(h.files)
        names = sorted((n for n in files - NOT_TILES if n in ENC))
        if only is not None:
            names = [n for n in names if n in only]
        same = reuse is not None and reuse[1].get(day) == stamps and os.path.isdir(os.path.join(reuse[0], day))
        jobs += [(rdir, hours_dir, day, stamps, n, tile, levels, reuse[0] if same else None) for n in names]
    if days and elevation:
        jobs.append((rdir, hours_dir, None, [f'{elev_time:%Y%m%d%H}'], 'elevation', tile, levels, None))
    for name, keys, lkeys in run_jobs(_tile_job, jobs, workers):
        if keys:
            present.setdefault(name, set()).update(keys)
        for k, lk in lkeys.items():
            lpresent[k].setdefault(name, set()).update(lk)
    return ({d: [t.strftime('%Y%m%d%H') for t in ts] for d, ts in days.items()}, present, lpresent)
POINT_FIELDS = ('t2', 'td2', 'rain', 'ptype', 'slr', 'u_sfc', 'v_sfc', 'gust', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ceil', 'ctop', 'vis', 'mslp', 'cape', 'frzl', 'wstar', 'depth', 'zi', 'cubase', 'cudepth', 'overdev', 'bumpy', 'densalt', 'refc', 'showers', 'sun', 'accas', *(f'cc{p}' for p in P.CC_LEVELS), 'elevation', 'snowshare')
POINT_BLOCK = 16

def _pack_job(job):
    path, start, keys, folder = job
    mm = np.load(path, mmap_mode='r')
    for i, key in zip(range(start, start + len(keys)), keys):
        tiles.write_png(os.path.join(folder, f'{key}.png'), np.ascontiguousarray(mm[i]), COMPRESS_LEVEL)
    return len(keys)

def write_points(rdir, hours_dir, times, elev_time, only=None, workers=1, block=POINT_BLOCK, fields=POINT_FIELDS):
    if not times:
        return None
    stamps = [f'{t:%Y%m%d%H}' for t in sorted(times)]
    files = set()
    for st in stamps:
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            files |= set(h.files)
    names = [n for n in fields if n in files and (only is None or n in only or n == 'elevation')]
    with np.load(os.path.join(hours_dir, f'{elev_time:%Y%m%d%H}.npz')) as h:
        elev = h['elevation'] if 'elevation' in h.files else None
        ny, nx = h['wstar' if 'wstar' in h.files else h.files[0]].shape
    if elev is None and 'elevation' in names:
        names.remove('elevation')
    B, W, nf = (block, block + 1, len(names))
    nby, nbx = (max(1, -(-(ny - 1) // B)), max(1, -(-(nx - 1) // B)))
    NY, NX = (nby * B + 1, nbx * B + 1)

    def stack(h):
        out = np.full((nf, NY, NX), 255, np.uint8)
        for i, n in enumerate(names):
            if n == 'elevation':
                out[i, :ny, :nx] = elev
            elif n in h.files:
                out[i, :ny, :nx] = h[n]
        return out

    def blocks(a):
        return np.lib.stride_tricks.sliding_window_view(a, (W, W), axis=(1, 2))[:, ::B, ::B]
    data = np.zeros((nby, nbx), bool)
    fc = [i for i, n in enumerate(names) if n != 'elevation']
    for st in {stamps[0], stamps[-1]}:
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            data |= (blocks(stack(h))[fc] != 255).any(axis=(0, 3, 4))
    by, bx = np.nonzero(data)
    if not len(by):
        return None
    folder = os.path.join(rdir, 'pt')
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(rdir, 'pt.tmp.npy')
    mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(by), len(stamps) * W, nf * W))
    for k, st in enumerate(stamps):
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            sel = blocks(stack(h))[:, by, bx]
        mm[:, k * W:(k + 1) * W, :] = sel.transpose(1, 2, 0, 3).reshape(len(by), W, nf * W)
    mm.flush()
    del mm
    keys = [f'{x}_{y}' for y, x in zip(by, bx)]
    step = -(-len(keys) // max(1, workers * 4))
    run_jobs(_pack_job, [(tmp, i, keys[i:i + step], folder) for i in range(0, len(keys), step)], workers)
    os.remove(tmp)
    pack.pack_folder(folder, keys, keep=png_too())
    return {'block': B, 'fields': names, 'stamps': stamps, 'tiles': keys}

def write_soundings(rdir, hours_dir, times, src, workers=1, block=SND_BLOCK):
    if not src.snd_levels or not times:
        return None
    shapes = {}
    for t in sorted(times):
        with np.load(os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz')) as h:
            if 'snd' in h.files:
                shapes[t] = h['snd'].shape
    if not shapes:
        return None
    shape = shapes[max(shapes)]
    have = [t for t, s in shapes.items() if s == shape]
    old = [t for t, s in shapes.items() if s != shape]
    if old:
        log(f'  soundings: {len(old)} hour(s) from other levels left out ({old[0]:%d %HZ}…{old[-1]:%d %HZ})')
    nv, sy, sx = shape
    B, W = (block, block + 1)
    nby, nbx = (max(1, -(-(sy - 1) // B)), max(1, -(-(sx - 1) // B)))

    def grid(st):
        a = np.full((nv, nby * B + 1, nbx * B + 1), 255, np.uint8)
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            a[:, :sy, :sx] = h['snd']
        return np.lib.stride_tricks.sliding_window_view(a, (W, W), axis=(1, 2))[:, ::B, ::B]
    days = {}
    for t in have:
        days.setdefault(f'{t:%Y%m%d}', []).append(f'{t:%Y%m%d%H}')
    out = {}
    for day, stamps in sorted(days.items()):
        data = np.zeros((nby, nbx), bool)
        for st in {stamps[0], stamps[-1]}:
            data |= (grid(st) != 255).any(axis=(0, 3, 4))
        by, bx = np.nonzero(data)
        if not len(by):
            continue
        folder = os.path.join(rdir, 'snd', day)
        os.makedirs(folder, exist_ok=True)
        tmp = os.path.join(rdir, f'snd-{day}.tmp.npy')
        mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(by), len(stamps) * W, nv * W))
        for k, st in enumerate(stamps):
            sel = grid(st)[:, by, bx]
            mm[:, k * W:(k + 1) * W, :] = sel.transpose(1, 2, 0, 3).reshape(len(by), W, nv * W)
        mm.flush()
        del mm
        keys = [f'{x}_{y}' for y, x in zip(by, bx)]
        step = -(-len(keys) // max(1, workers * 4))
        run_jobs(_pack_job, [(tmp, i, keys[i:i + step], folder) for i in range(0, len(keys), step)], workers)
        os.remove(tmp)
        pack.pack_folder(folder, keys, keep=png_too())
        out[day] = {'stamps': stamps, 'tiles': keys}
    if not out:
        return None
    return {'step': src.snd_step, 'block': B, 'nx': sx, 'ny': sy, 'levels': list(src.snd_levels), 'vars': list(SND_VARS), 'enc': [snd_enc(p) for p in src.snd_levels], 'sfc': {k: list(v) for k, v in SND_SFC.items()}, 'map': list(SND_MAP), 'days': out}
FREE_MODELS = tuple(OPEN_AHEAD_H)
FREE_SLACK = timedelta(hours=OPEN_SLACK_H)

def free_ahead(model):
    return timedelta(hours=OPEN_AHEAD_H[model])

def hour_token(model, run_id, stamp, secret=None):
    import hashlib
    import hmac
    secret = os.environ.get('DATA_PATH_SECRET', '') if secret is None else secret
    if not secret:
        return stamp
    tok = hmac.new(secret.encode(), f'free-hour|{model}|{run_id}|{stamp}'.encode(), hashlib.sha256).hexdigest()[:32]
    if os.environ.get('GITHUB_ACTIONS') == 'true' and tok not in _MASKED:
        print(f'::add-mask::{tok}', flush=True)
        _MASKED.add(tok)
    return tok
_MASKED: set = set()
FREE_FIELDS = {'wstar', 'depth', 'zi', 'cubase', 'cudepth', 'overdev', 'rain', 'showers', 'sun', 'ptype', 'cape', 'bs', 'densalt', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'accas', *(f'cc{p}' for p in P.CC_LEVELS), 'u_sfc', 'v_sfc', 'u_blavg', 'v_blavg', 'bumpy', 'ceil', 'lowbase', 'vis', 'fltcat', *sky.FIELDS, 'rowstr', 'rowdir', 'rowgap', 'wdeck', 'dtop', 't2', 'td2', 'sst', 'mslp', 'gust', 'pm25', 'spot1', 'spot2', 'glow', *(f'{c}_a{ft}' for ft in WIND_ALTS_FT for c in ('u', 'v'))}
NOW_AHEAD = timedelta(hours=24) + FREE_SLACK
NOW_FIELDS = FREE_FIELDS | {'twidth', 'sinkth', 'heat', 'soilm', 'ctop', 'ceiltop', 'wtrap', 'wlam', 'u_wl', 'v_wl', 'u_cuw', 'v_cuw', 'curh', 'thml', 'thx', *(f'{k}_m{dz}' for dz in ML_TOP_DZ for k in ('dth', 'q')), *(f'wv{ft}' for ft in (8000, 10000, 12000, 14000, 18000, 24000)), *(f'dpd{ft}' for ft in (8000, 10000, 12000, 14000, 18000, 24000)), *(f't{ft}' for ft in (8000, 10000, 12000, 14000, 18000, 24000)), *(f'{c}_g{m}' for m in (100, 300, 600, 1000, 1500, 2000, 2500, 3000) for c in ('u', 'v')), *(f'dpd_g{m}' for m in (600, 1000, 1500, 2000, 3000)), *(f'n_g{m}' for m in (750, 1250, 1750, 2500, 3500, 5000)), *(f'{c}_a{ft}' for ft in (2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000) for c in ('u', 'v'))}

def write_free(out, model, now=None, workers=1):
    if model not in FREE_MODELS:
        return None
    base = os.path.join(out, model)
    try:
        with open(os.path.join(base, 'latest.json')) as f:
            full = json.load(f)
    except FileNotFoundError:
        log(f'{model}: no full product yet, no free window')
        return None
    hours_dir = os.path.join(base, '_hours')
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    ahead = free_ahead(model)
    cutoff = now + ahead + FREE_SLACK
    stamps = sorted((s for d in full['days'].values() for s in d))
    times = [datetime.strptime(s, '%Y%m%d%H').replace(tzinfo=timezone.utc) for s in stamps]
    times = [t for t in times if t <= cutoff and os.path.exists(os.path.join(hours_dir, f'{t:%Y%m%d%H}.npz'))]
    hourly = [t for t in times if t > now + ahead]
    manifest = _publish_open(out, os.path.join(out, 'free', model), model, full, times, FREE_FIELDS, cutoff, workers, points=True, hourly=hourly)
    if manifest:
        log(f"{model}: free window to {cutoff:%Y-%m-%d %HZ}: {len(times)} hours ({len(hourly)} by the hour), {len(manifest['fields'])} fields")
    now_times = [t for t in times if now - timedelta(hours=1) <= t <= now + NOW_AHEAD]
    nm = _publish_open(out, os.path.join(out, 'free', 'now', model), model, full, now_times, NOW_FIELDS, now + NOW_AHEAD, workers)
    if nm:
        log(f"{model}: 3D window: {len(now_times)} hours, {len(nm['fields'])} fields")
    return manifest

def _free_soundings(rdir, run_dir, snd, times):
    if not snd or not snd.get('days'):
        return None
    from archive import trim_rows
    want = {f'{t:%Y%m%d%H}' for t in times}
    days = {}
    for d, e in snd['days'].items():
        src = os.path.join(run_dir, 'snd', f'{d}.pack')
        if not isinstance(e, dict) or not e.get('stamps') or (not os.path.exists(src)):
            continue
        rows = [k for k, st in enumerate(e['stamps']) if st in want]
        if not rows:
            continue
        os.makedirs(os.path.join(rdir, 'snd'), exist_ok=True)
        try:
            trim_rows(src, os.path.join(rdir, 'snd', f'{d}.pack'), rows, snd['block'] + 1, workers=2)
        except Exception as ex:
            log(f'free soundings: {d} not cut ({ex})')
            continue
        days[d] = {**e, 'stamps': [e['stamps'][k] for k in rows]}
    return {**snd, 'days': days} if days else None

def _publish_open(out, fbase, model, full, times, fields, cutoff, workers, points=False, hourly=()):
    base = os.path.join(out, model)
    hours_dir = os.path.join(base, '_hours')
    os.makedirs(fbase, exist_ok=True)
    if not times:
        log(f'{model}: no hours for {os.path.relpath(fbase, out)}')
        return None
    free_id = f"{full['runId']}-{cutoff:%Y%m%d%H}"
    rdir = os.path.join(fbase, free_id)
    shutil.rmtree(rdir, ignore_errors=True)
    tile = full['grid']['tile']
    levels = [(lv['k'], lv['grid']['tile']) for lv in full.get('levels', [])]
    reuse = (os.path.join(base, full['runId']), full.get('days', {}))
    hourly = set(hourly)
    daily = [t for t in times if t not in hourly]
    days, present, lpresent = write_tiles(rdir, hours_dir, daily, tile, times[-1], only=fields, levels=levels, workers=workers, reuse=reuse)
    pts = write_points(rdir, hours_dir, daily, times[-1], only=fields, workers=workers) if points else None
    has_fronts = points and write_fronts(rdir, hours_dir, daily, around=times)
    hours, hpts = ([], {})
    if hourly:
        folder = lambda t: f"h/{hour_token(model, free_id, f'{t:%Y%m%d%H}')}"
        _, hp, hlp = write_tiles(rdir, hours_dir, sorted(hourly), tile, times[-1], only=fields, levels=levels, workers=workers, day_of=folder, elevation=False)
        for name, keys in hp.items():
            present.setdefault(name, set()).update(keys)
        for k, by_name in hlp.items():
            for name, keys in by_name.items():
                lpresent[k].setdefault(name, set()).update(keys)
        if points:
            has_fronts = write_fronts(rdir, hours_dir, sorted(hourly), day_of=folder, around=times) or has_fronts
        for t in sorted(hourly):
            st = f'{t:%Y%m%d%H}'
            hours.append(st)
            if points:
                p = write_points(os.path.join(rdir, folder(t)), hours_dir, [t], times[-1], only=fields, workers=workers)
                if p:
                    hpts[st] = {k: p[k] for k in ('fields', 'tiles')}
    last_open = max(daily) if daily else times[0] - timedelta(hours=1)
    qtimes = [q for q in stored_quarters(os.path.join(base, '_quarters'), times[0], cutoff) if q <= last_open] if model == 'hrrr' else []
    qdays, qpresent = write_quarters(rdir, os.path.join(base, '_quarters'), qtimes, tile, only=fields, workers=workers) if qtimes else ({}, {})
    manifest = {**full, 'runId': free_id, 'tier': 'free', 'freeUntil': cutoff.isoformat(), 'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'days': days, 'fields': {**{k: {**full['fields'][k], 'tiles': sorted(v)} for k, v in sorted(present.items()) if k in full['fields']}, **quarter_manifest(qdays, qpresent)}, 'quarters': qdays, 'levels': [{**lv, 'tiles': {k: sorted(v) for k, v in sorted(lpresent[lv['k']].items()) if k in full['fields']}} for lv in full.get('levels', [])], 'waveAltsFt': full.get('waveAltsFt', []) if any((k.startswith('wv') for k in present)) else [], 'points': pts, 'sounding': _free_soundings(rdir, os.path.join(base, full['runId']), full.get('sounding'), daily) if points else None, 'hours': hours, 'hourPoints': hpts, 'fronts': 1 if has_fronts else 0, 'pack': 1}
    with open(os.path.join(fbase, 'latest.json'), 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))
    for old in os.listdir(fbase):
        if old != free_id and old != 'latest.json':
            shutil.rmtree(os.path.join(fbase, old), ignore_errors=True)
    open(os.path.join(out, '.changed'), 'w').close()
    return manifest
STORE_AHEAD_H = {'gfs': 60, 'icon': 60, 'ecmwf': 48}
STORE_AFTER_FREE_H: dict[str, int] = {}

def prune_store(hours_dir, ahead_h, now=None):
    cut = (now or datetime.now(timezone.utc)) + timedelta(hours=ahead_h)
    n = 0
    for f in os.listdir(hours_dir):
        if not f.endswith('.npz') or f.endswith('.tmp.npz'):
            continue
        try:
            t = datetime.strptime(f[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if t > cut:
            for ext in ('.npz', '.run'):
                p = os.path.join(hours_dir, f[:10] + ext)
                if os.path.exists(p):
                    os.remove(p)
            n += 1
    return n

def store_fits(hours_dir, region, src_id=None):
    lons, lats = target_grid(region)
    keep = keep_mask(src_id, *np.meshgrid(lats, lons, indexing='ij')) if src_id else None
    for f in sorted(os.listdir(hours_dir)) if os.path.isdir(hours_dir) else []:
        if f.endswith('.npz') and (not f.endswith('.tmp.npz')):
            try:
                with np.load(os.path.join(hours_dir, f)) as h:
                    a = h[h.files[0]]
                    if a.shape != (len(lats), len(lons)):
                        return False
                    return keep is None or bool(np.all(a[~keep] == 255))
            except Exception:
                return False
    return True

def older_runs(tail, times):
    have = set(times)
    out = []
    for t, r in tail:
        if t in have and (not out or out[-1]['run'] != r.isoformat()):
            out.append({'from': t.isoformat(), 'run': r.isoformat()})
    return out

def backfill_jobs(src, run, needed, hours_dir):
    have = {f[:-4] for f in os.listdir(hours_dir) if f.endswith('.npz') and (not f.endswith('.tmp.npz'))}
    end = run + timedelta(hours=src.last_fh(run))
    jobs = []
    for t in needed:
        if t > end:
            runs = candidate_runs(src, run, back_hours=25)
        elif (t > run or (t == run and src.fstep(0))) or f'{t:%Y%m%d%H}' in have:
            continue
        else:
            runs = candidate_runs(src, t, back_hours=24)
        for r in runs:
            fh = int((t - r).total_seconds() // 3600)
            if r < run and 0 <= fh <= src.last_fh(r) and src.builds(fh):
                jobs.append((r, fh))
                break
    return jobs

def pick_run(src, needed, pin=None):
    if pin:
        run = datetime.strptime(pin, '%Y%m%d%H').replace(tzinfo=timezone.utc)
        fhs = [int((v - run).total_seconds() // 3600) for v in needed]
        fhs = [f for f in fhs if 0 <= f <= src.last_fh(run) and src.builds(f)]
        if not fhs and run < needed[0]:
            fhs = [f for f in range(0, 49) if f <= src.last_fh(run) and src.builds(f)]
        return (run, fhs)
    run, fhs = latest_complete_run(src, needed, log)
    if run is None:
        raise SystemExit(f'no complete {src.id} run found')
    return (run, fhs)

def main():
    global _LIBC
    _LIBC = lean_malloc()
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=SOURCES)
    ap.add_argument('--out', default='dist-data')
    ap.add_argument('--days', type=int)
    ap.add_argument('--bbox', help='west,south,east,north override (testing)')
    ap.add_argument('--step', type=float)
    ap.add_argument('--max-hours', type=int, help='limit valid times (testing)')
    ap.add_argument('--utc-hours', help='with --max-hours: the UTC hours taken (e.g. 12,13,14; 18,19,20 without)')
    ap.add_argument('--keep-runs', type=int, default=1)
    ap.add_argument('--tile', type=int)
    ap.add_argument('--force', action='store_true', help='publish again even if this run is already published')
    ap.add_argument('--recompute', action='store_true', help='also recompute hours already computed from this run')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '3')))
    ap.add_argument('--free-only', action='store_true', help='only refresh the free 24 h window from stored hours')
    ap.add_argument('--no-quarters', action='store_true', help="skip HRRR's 15-minute output")
    ap.add_argument('--python-physics', action='store_true', help='physics.py, leelattice.py and runwave.py only (never their C++)')
    ap.add_argument('--shard', help="i/n: compute only share i (0…n-1) of the run's hours into the store, nothing published (shard_hours)")
    ap.add_argument('--run', help="YYYYMMDDHH: this run, not the newest complete one (the hour shards')")
    args = ap.parse_args()
    if args.free_only:
        write_free(args.out, args.model, workers=args.workers)
        hours_dir = os.path.join(args.out, args.model, '_hours')
        if args.model in STORE_AFTER_FREE_H and os.path.isdir(hours_dir):
            ahead = STORE_AFTER_FREE_H[args.model]
            log(f'{args.model}: {prune_store(hours_dir, ahead)} stored hours past +{ahead} h dropped (free window written)')
        return 0
    src = SOURCES[args.model]
    region = list(REGIONS[src.id])
    if args.bbox:
        region[:4] = [float(x) for x in args.bbox.split(',')]
    if args.step:
        region[4] = args.step
    needed = needed_times(src, args.days)
    run, fhs = pick_run(src, needed, args.run)
    if args.max_hours:
        hrs = tuple((int(x) for x in args.utc_hours.split(','))) if args.utc_hours else (18, 19, 20)
        mid = [f for f in fhs if (run + timedelta(hours=f)).hour in hrs] or fhs
        fhs = sorted(set(mid[:args.max_hours]))
    log(f'{src.id}: run {run:%Y-%m-%d %HZ}, {len(fhs)} valid times')
    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        fhs = shard_hours(fhs, i, n, LANE_RUN if src.between_to else 1)
        shard = (i, n)
        log(f'{src.id}: shard {i + 1} of {n}: {len(fhs)} hours')
        if not fhs:
            return 0
    run_id = f'{run:%Y%m%d%H}'
    base = os.path.join(args.out, src.id)
    latest = os.path.join(base, 'latest.json')
    if os.path.exists(latest) and (not args.force) and (shard is None):
        with open(latest) as f:
            cur = json.load(f)
        if cur.get('runId') == run_id and cur.get('complete'):
            log(f'{src.id}: {run_id} already built, nothing to do')
            return 0
    rdir = os.path.join(base, run_id)
    build_dir = os.path.join(base, f'{run_id}.building')
    hours_dir = os.path.join(base, '_hours')
    for d in os.listdir(base) if os.path.isdir(base) else []:
        if d.endswith('.building'):
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)
    vfile = os.path.join(hours_dir, 'VERSION')
    kfile = os.path.join(hours_dir, 'KEEP')
    keep_now = repr(KEEP.get(src.id))
    if not os.path.exists(vfile) or open(vfile).read().strip() != str(HOURS_VERSION) or (not store_fits(hours_dir, region, src.id)) or (src.id in KEEP and (not os.path.exists(kfile) or open(kfile).read() != keep_now)):
        shutil.rmtree(hours_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(base, '_quarters'), ignore_errors=True)
    os.makedirs(hours_dir, exist_ok=True)
    with open(vfile, 'w') as f:
        f.write(str(HOURS_VERSION))
    if src.id in KEEP:
        with open(kfile, 'w') as f:
            f.write(keep_now)
    _CTX.update({'src': src, 'run': run, 'region': region, 'hours': hours_dir, 'holder': {}, 'recompute': args.recompute, 'pick_physics': not args.python_physics})
    if args.python_physics:
        runfields.NATIVE = False
    jobs = list(fhs)
    back = backfill_jobs(src, run, needed, hours_dir) if not args.max_hours and shard is None else []
    if back:
        log(f'{src.id}: backfilling {len(back)} hours from earlier runs ' + ', '.join(sorted({f'{r:%d %HZ}' for r, _ in back})))
    jobs += back
    end = run + timedelta(hours=src.last_fh(run))
    tail = sorted(((r + timedelta(hours=fh), r) for r, fh in back if r + timedelta(hours=fh) > end))
    kept, first, rest = split_first(jobs)
    results = {j: _process_hour(j) for j in kept}
    if first is not None:
        results[first] = _process_hour(first)
    trim_malloc()
    if rest:
        workers = max(1, min(args.workers, len(rest)))
        progress = Progress(src.id, len(rest))
        if workers == 1:
            for j in rest:
                results[j] = _process_hour(j)
                progress.tick()
        else:
            results.update(zip(rest, process_lanes(rest, workers, progress, run=LANE_RUN if src.between_to else 1)))
    done = [t for t in (results.get(j) for j in fhs) if t is not None]
    if not done:
        raise SystemExit('no hours processed')
    if shard is not None:
        log(f'{src.id}: shard {shard[0] + 1} of {shard[1]}: {len(done)} of {len(fhs)} hours computed')
        with open(os.path.join(hours_dir, 'SHARD_RUN'), 'w') as f:
            f.write(run_id)
        return 0
    qdir = os.path.join(base, '_quarters')
    if src.id == 'hrrr' and (not args.no_quarters):
        os.makedirs(qdir, exist_ok=True)
        _CTX['quarters'] = qdir
        lo, hi = (min(needed), max(needed))
        qjobs = [fh for fh in fhs if 1 <= fh <= QUARTER_MAX_FH and lo <= run + timedelta(hours=fh) - timedelta(minutes=45) <= hi]
        if qjobs:
            t0 = time.time()
            workers = max(1, min(args.workers, len(qjobs)))
            if workers == 1:
                qdone = [t for fh in qjobs for t in _process_quarters(fh)]
            else:
                import multiprocessing as mp
                with mp.get_context('fork').Pool(workers) as pool:
                    qdone = [t for part in pool.imap_unordered(_process_quarters, qjobs) for t in part]
            log(f'{src.id}: {len(qdone)} quarter hours from the 15-minute output in {_fmt_dur(time.time() - t0)}')
    window = {t.strftime('%Y%m%d%H') for t in needed}
    times = []
    for f in sorted(os.listdir(hours_dir)):
        stamp = f[:-4]
        if not f.endswith('.npz') or f.endswith('.tmp.npz'):
            continue
        if stamp in window:
            times.append(datetime.strptime(stamp, '%Y%m%d%H').replace(tzinfo=timezone.utc))
        else:
            os.remove(os.path.join(hours_dir, f))
            if os.path.exists(os.path.join(hours_dir, f'{stamp}.run')):
                os.remove(os.path.join(hours_dir, f'{stamp}.run'))
    lons, lats = target_grid(region)
    tile = args.tile or TILE[src.id]
    levels = [(k, LEVEL_TILE) for k in OVERVIEW.get(src.id, ())]
    os.makedirs(build_dir, exist_ok=True)
    t0 = time.time()
    n = sum(run_jobs(_sky_job, [(hours_dir, t, region[4] <= sky.FINE_STEP) for t in times], args.workers))
    if n:
        log(f'{src.id}: Cloud Layers added to {n} stored hours')
    n = sum(run_jobs(_runfields_job, [(hours_dir, t, tuple(region)) for t in times], args.workers))
    if n:
        log(f'{src.id}: ridge/wave running and shade added to {n} stored hours')
    t1 = time.time()
    n = hour_to_hour(hours_dir, times, region[4] <= sky.FINE_STEP, args.workers)
    log(f'{src.id}: Cloud Layers hour to hour over {len(times)} hours ({n} decks held) in {_fmt_dur(time.time() - t1)}')
    days, present, lpresent = write_tiles(build_dir, hours_dir, times, tile, done[0], levels=levels, workers=args.workers)
    qtimes = stored_quarters(qdir, min(needed), max(needed), prune=True) if src.id == 'hrrr' else []
    qdays, qpresent = write_quarters(build_dir, qdir, qtimes, tile, workers=args.workers) if qtimes else ({}, {})
    log(f'{src.id}: tiles for {len(times)} hours in {_fmt_dur(time.time() - t0)}')
    has_fronts = write_fronts(build_dir, hours_dir, times)
    t0 = time.time()
    points = write_points(build_dir, hours_dir, times, done[0], workers=args.workers)
    log(f"{src.id}: point packs ({(len(points['tiles']) if points else 0)}) in {_fmt_dur(time.time() - t0)}")
    t0 = time.time()
    soundings = write_soundings(build_dir, hours_dir, times, src, workers=args.workers)
    if soundings:
        log(f"{src.id}: sounding packs ({sum((len(d['tiles']) for d in soundings['days'].values()))} blocks, {len(soundings['days'])} days) in {_fmt_dur(time.time() - t0)}")
    ntx = -(-len(lons) // tile)
    nty = -(-len(lats) // tile)
    grid = {'west': float(lons[0]), 'north': float(lats[0]), 'step': region[4], 'nx': len(lons), 'ny': len(lats), 'tile': tile, 'ntx': ntx, 'nty': nty}
    manifest = {'model': src.id, 'label': src.label, 'run': run.isoformat(), 'runId': run_id, 'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'complete': not args.max_hours and len(done) == len(fhs), 'older': older_runs(tail, times), 'grid': grid, 'levels': overview_manifest(grid, levels, lpresent), 'days': days, 'quarters': qdays, 'fields': {**{k: {'min': ENC[k][0], 'max': ENC[k][1], 'transform': ENC[k][2], 'tiles': sorted(v)} for k, v in sorted(present.items())}, **quarter_manifest(qdays, qpresent)}, 'heights': 'agl', 'sunGate': True, 'rainGate': True, 'cuNear': True, 'waveBars': True, 'bandsAgl': True, 'highAgl': True, 'bsRasp': True, 'windAltsFt': WIND_ALTS_FT, 'waveAltsFt': WAVE_ALTS_FT if 'wv12000' in present else [], 'notes': src.notes, 'points': points, 'sounding': soundings, 'fronts': 1 if has_fronts else 0, 'pack': 1, 'hourSegs': 1 if HOUR_SEGS else 0}
    shutil.rmtree(rdir, ignore_errors=True)
    os.replace(build_dir, rdir)
    with open(latest + '.tmp', 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))
    os.replace(latest + '.tmp', latest)
    open(os.path.join(args.out, '.changed'), 'w').close()
    runs = sorted((d for d in os.listdir(base) if d.isdigit()))
    for old in runs[:-args.keep_runs]:
        shutil.rmtree(os.path.join(base, old), ignore_errors=True)
    if src.id in STORE_AHEAD_H and (not args.max_hours):
        log(f'{src.id}: {prune_store(hours_dir, STORE_AHEAD_H[src.id])} stored hours past +{STORE_AHEAD_H[src.id]} h dropped (published already)')
    log(f'{src.id}: run {run_id}: computed {len(done)} hours, published {len(times)} in {len(days)} days')
if __name__ == '__main__':
    sys.exit(main())
