from __future__ import annotations
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import numpy as np
import build as B
import grib
import physics as P
from sources import SOURCES
SRC = SOURCES['ens']
RANGE_WORKERS = 48
LAST_FH = SRC.max_fhour
MEMBERS = 50
LEVELS = (925, 850, 700, 600, 500)
SFC = ('2t', '2d', 'sp', 'ssr', 'str', 'tp', 'cp', '10u', '10v', 'mucape')
GUSTS = ('10fg3', '10fg6', '10fg')
SOIL = ('vsw', '1')
REGION = B.REGIONS['gfs']
KEEP_ID = 'gfs'
CHUNK = 10
NEED = 1.0
PCT = (10, 50, 90)
RAIN_MM_H = 0.1
STORM_MM_H, STORM_CP, STORM_CAPE = (0.5, 0.5, 300.0)
DAY_HOURS = (9.5, 18.0)
SCORE_W = (0.6, 1, 1.5, 2, 2.75, 3.5)
SCORE_TOP = (400, 700, 1000, 1400, 1900, 2500)
CORE, CU_CORE_TOP, BLUE_CORE_TOP = (1.3, 0.75, 0.2)
RD, G = (287.05, 9.80665)
ENC = {**{f't2_{k}': (-40, 50, None) for k in ('lo', 'mid', 'hi')}, **{f'wind_{k}': (0, 40, None) for k in ('lo', 'mid', 'hi')}, **{f'gust_{k}': (0, 50, None) for k in ('mid', 'hi')}, **{f'ws_{k}': (0, 6, None) for k in ('lo', 'mid', 'hi')}, **{f'top_{k}': (0, 6000, None) for k in ('lo', 'mid', 'hi')}, 'rainp': (0, 100, None), 'stormp': (0, 100, None), 'rain_hi': (0, 50, 'sqrt'), **{f'day{k}': (0, 100, None) for k in range(1, 7)}, 'daystorm': (0, 100, None), 'elevation': B.ENC['elevation']}
FIELDS = tuple(ENC)

def steps(last=LAST_FH):
    return [fh for fh in range(0, last + 1) if SRC.fstep(fh)]

def _mirrors(call):
    last = None
    for m in range(len(SRC.MIRRORS)):
        try:
            return call(m)
        except Exception as e:
            last = e
    raise last

def latest_run(now=None, last=LAST_FH):
    now = now or datetime.now(timezone.utc)
    t = now.replace(minute=0, second=0, microsecond=0)
    for h in range(48):
        run = t - timedelta(hours=h)
        if run.hour in SRC.runs and any((grib.exists(SRC.idx_url(run, last, m)) for m in range(len(SRC.MIRRORS)))):
            return run
    return None

def wanted(rows):
    sel = {}
    for r in rows:
        p, lt, lev = (r['param'], r.get('levtype'), r.get('levelist'))
        if r.get('type') != 'pf':
            continue
        if lt == 'sfc' and (p in SFC or p in GUSTS) or (lt == 'pl' and p == 't' and (int(lev) in LEVELS)) or (lt == 'sol' and (p, lev) == SOIL):
            sel[p, lev, int(r['number'])] = {'offset': r['_offset'], 'end': r['_offset'] + r['_length'] - 1}
    return sel
RANGE_TIMEOUT = 15
RANGE_GAP = 64 * 1024

def step_rows(run, fh, m):
    text = grib.http_get(SRC.idx_url(run, fh, m), tries=3, timeout=RANGE_TIMEOUT).decode()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return ([SRC.url(run, fh, k) for k in (m, *(k for k in range(len(SRC.MIRRORS)) if k != m))], wanted(rows))

def fetch_step(run, fh):

    def one(m):
        urls, sel = step_rows(run, fh, m)
        return grib.fetch_messages(urls, sel, workers=RANGE_WORKERS, gap=RANGE_GAP, tries=6, timeout=RANGE_TIMEOUT)
    return _mirrors(one)
AHEAD = 2
RANGE_AGAIN = (5, 15, 30, 60, 120)
STEP_AGAIN = (30, 120, 300)

class Downloads:

    def __init__(self, run, fhs):
        self.run, self.fhs, self.next = (run, list(fhs), 0)
        self.pool = ThreadPoolExecutor(RANGE_WORKERS)
        self.index = ThreadPoolExecutor(1)
        self.retry = ThreadPoolExecutor(16)
        self.jobs = {}

    def _put(self):
        while self.next < len(self.fhs) and len(self.jobs) < AHEAD + 1:
            fh = self.fhs[self.next]
            self.next += 1
            self.jobs[fh] = self.index.submit(self._submit, fh)

    def _submit(self, fh):
        urls, sel = step_rows(self.run, fh, 0)
        return grib.submit_messages(self.pool, urls, sel, RANGE_GAP, tries=6, timeout=RANGE_TIMEOUT)

    @staticmethod
    def _again(what, waits, call):
        err = None
        for w in waits:
            B.log(f'ens: {what}; again in {w} s')
            time.sleep(w)
            try:
                return call()
            except Exception as e:
                err = e
                what = f"{what.split(':')[0]}: {e}"
        raise err

    def take(self, fh):
        self._put()
        job = self.jobs.pop(fh)
        self._put()
        try:
            futs = job.result()
        except Exception as e:
            return self._again(f'+{fh} h: {e}', STEP_AGAIN, lambda: fetch_step(self.run, fh))
        out, failed, err = ({}, [], None)
        for f in futs:
            try:
                out.update(f.result())
            except Exception as e:
                failed.append(f)
                err = e
        for w in RANGE_AGAIN:
            if not failed:
                return out
            B.log(f'ens: +{fh} h: {len(failed)} of {len(futs)} requests failed ({err}); again in {w} s')
            time.sleep(w)
            again = [(f, self.retry.submit(f.again)) for f in failed]
            failed = []
            for f, g in again:
                try:
                    out.update(g.result())
                except Exception as e:
                    failed.append(f)
                    err = e
        if not failed:
            return out
        return self._again(f'+{fh} h: {err}', STEP_AGAIN, lambda: fetch_step(self.run, fh))

    def close(self):
        self.retry.shutdown(wait=False, cancel_futures=True)
        self.index.shutdown(wait=False, cancel_futures=True)
        self.pool.shutdown(wait=False, cancel_futures=True)

def fetch_static(run):

    def one(m):
        text = grib.http_get(SRC.idx_url(run, 0, m, 'oper'), tries=3).decode()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        sel = {r['param']: {'offset': r['_offset'], 'end': r['_offset'] + r['_length'] - 1} for r in rows if r.get('levtype') == 'sfc' and r['param'] in ('z', 'lsm')}
        return grib.fetch_messages(SRC.url(run, 0, m, 'oper'), sel, workers=2, gap=0, tries=4)
    return _mirrors(one)

class Cells:

    def __init__(self, msg, region=REGION, keep=KEEP_ID):
        lat, lon = grib.latlons(msg)
        r = B.Regridder.regular(lat, lon, region)
        self.lons, self.lats = (r.lons, r.lats)
        self.shape = r.shape
        glon, glat = np.meshgrid(r.lons, r.lats)
        mask = B.keep_mask(keep, glat, glon)
        mask = np.ones(r.shape, bool) if mask is None else mask
        self.flat = np.nonzero(mask.ravel())[0]
        self.src = r.index[self.flat]
        self.lat = glat.ravel()[self.flat].astype(np.float32)
        self.lon = glon.ravel()[self.flat].astype(np.float32)

    def take(self, msg):
        vals, _ = grib.decode(msg)
        return vals.ravel()[self.src]

    def grid(self, v):
        out = np.full(self.shape[0] * self.shape[1], 255, np.uint8)
        out[self.flat] = v
        return out.reshape(self.shape)

def decode_step(cells, raw, threads=B.DECODE_THREADS):
    keys = sorted(raw)
    with ThreadPoolExecutor(threads) as ex:
        vals = dict(zip(keys, ex.map(lambda k: cells.take(raw[k]), keys)))
    out = {}
    for (p, lev, n), v in vals.items():
        name = f't{lev}' if p == 't' else 'gust' if p in GUSTS else 'soil' if p == 'vsw' else p
        out.setdefault(name, np.full((MEMBERS, len(cells.flat)), np.nan, np.float32))[n - 1] = v
    for k in ('2t', '2d', *(f't{p}' for p in LEVELS)):
        if k in out:
            out[k] -= 273.15
    if 'sp' in out:
        out['sp'] /= 100.0
    return out

def heights(zs, ps, t2, t_lv):
    z = np.empty((len(LEVELS),) + ps.shape, np.float32)
    zp, tp, lp = (np.broadcast_to(zs, ps.shape).astype(np.float32), t2 + 273.15, np.log(ps))
    for i, p in enumerate(LEVELS):
        tk, lk = (t_lv[i] + 273.15, math.log(p))
        dz = RD / G * 0.5 * (tp + tk) * (lp - lk)
        z[i] = zp + dz
        up = lk < lp
        zp, tp, lp = (np.where(up, z[i], zp), np.where(up, tk, tp), np.where(up, lk, lp))
    return z

def usable_top(ws, zi, base):
    blue = ~(base > 0)
    reach = np.where(blue | (base <= zi), 1.0, 1 - np.minimum(1, (base - zi) / np.maximum(B.CU_OVERSHOOT * zi, B.CU_OVERSHOOT_MIN)))
    e = 1 - reach
    d = np.where(blue, zi, np.where(base <= zi, base, base - (base - zi) * e))
    hold = np.where(blue, BLUE_CORE_TOP, CU_CORE_TOP * (1 - e) + BLUE_CORE_TOP * e)
    r = NEED / np.maximum(CORE * ws, 1e-06)
    frac = np.where(r >= 1, 0.0, np.minimum(1, 0.6 + 0.4 * (1 - r) / (1 - hold)))
    return np.where((ws > 0) & (d > 0), d * frac, 0.0)

def thermals(f, sl, zs, land, valid, hours, lat, lon):
    t2, td2, ps = (f['2t'][sl], f['2d'][sl], f['sp'][sl])
    zs2 = np.broadcast_to(zs, t2.shape).astype(np.float32)
    rn = f['rnsw'][sl] * B.instant_factor(valid, hours, lat, lon) + f['rnlw'][sl] if hours else np.zeros_like(t2)
    heat = B.heat_from_net_radiation(rn, t2, td2, f['soil'][sl] if 'soil' in f else None, land) * B.sun_gate(valid, lat, lon)
    t_lv = np.stack([f[f't{p}'][sl] for p in LEVELS])
    z = heights(zs, ps, t2, t_lv)
    zero = np.zeros_like(t_lv)
    prof = P.Profile(zs2, ps, t2, td2, np.zeros_like(t2), np.zeros_like(t2), z, list(LEVELS), t_lv, t_lv - 20, zero, zero)
    zi = np.maximum(np.nan_to_num(P.parcel_ml_depth(prof, np.where(heat > 20, 1.0, 0.2)), nan=0), 0)
    ws = np.where(heat > 10, P.wstar(heat, zi, t2, ps), 0)
    lcl = np.maximum(0, 125 * (t2 - np.minimum(td2, t2)))
    cu = B.cumulus_forms(ws, zi, lcl)
    rain, conv = (f['rain'][sl], f['conv'][sl])
    ws = ws * B.rain_damp(rain, conv > STORM_CP * np.maximum(rain, 1e-06), ws)
    top = usable_top(ws, zi, np.where(cu, lcl, np.nan))
    return (ws.astype(np.float32), top.astype(np.float32))

def hour_score(ws, top, rain, gust):
    s = np.minimum(sum(((ws >= x).astype(np.int8) for x in SCORE_W)), sum(((top >= x).astype(np.int8) for x in SCORE_TOP)))
    s = np.where((rain >= 0.5) | ~(ws > 0) | ~(top > 0), 0, s)
    g = np.nan_to_num(gust, nan=0)
    s = np.where(g >= 15, np.minimum(s, 1), np.where(g >= 12, np.minimum(s, 3), s))
    return s.astype(np.int8)

def solar(valid, lon):
    return valid.timestamp() / 3600.0 + np.asarray(lon, np.float64) / 15.0

def q(name, v):
    lo, hi, tr = ENC[name]
    return B.quantize(v, lo, hi, tr)

class Days:

    def __init__(self, run, last_valid, lon):
        self.first = solar(run, lon)
        self.last = solar(last_valid, lon)
        self.best, self.storm, self.n = ({}, {}, {})

    def add(self, valid, lon, score, storm):
        s = solar(valid, lon)
        day = np.floor(s / 24).astype(np.int64)
        h = s - day * 24
        on = (h >= DAY_HOURS[0]) & (h < DAY_HOURS[1])
        for d in np.unique(day[on]):
            m = on & (day == d)
            if d not in self.best:
                self.best[d] = np.zeros(score.shape, np.int8)
                self.storm[d] = np.zeros(score.shape, bool)
                self.n[d] = np.zeros(score.shape[1], np.int16)
            self.best[d][:, m] = np.maximum(self.best[d][:, m], score[:, m])
            self.storm[d][:, m] |= storm[:, m]
            self.n[d][m] += 1

    def fields(self, valid, lon):
        s = solar(valid, lon)
        day = np.floor(s / 24).astype(np.int64)
        out = {k: np.full(len(lon), np.nan, np.float32) for k in (*(f'day{k}' for k in range(1, 7)), 'daystorm')}
        for d in np.unique(day):
            if d not in self.best:
                continue
            m = (day == d) & (self.n[d] > 0) & (self.first <= d * 24 + DAY_HOURS[0]) & (self.last >= d * 24 + DAY_HOURS[1] - 1)
            if not m.any():
                continue
            best = self.best[d][:, m]
            for k in range(1, 7):
                out[f'day{k}'][m] = (best >= k).mean(axis=0) * 100
            out['daystorm'][m] = self.storm[d][:, m].mean(axis=0) * 100
        return out

def step_stats(f, cells, zs, land, valid, hours):
    ws = np.zeros_like(f['2t'])
    top = np.zeros_like(f['2t'])
    for a in range(0, MEMBERS, CHUNK):
        sl = slice(a, a + CHUNK)
        ws[sl], top[sl] = thermals(f, sl, zs, land, valid, hours, cells.lat, cells.lon)
    wind = np.hypot(f['10u'], f['10v'])
    rain, conv, cape = (f['rain'], f['conv'], np.nan_to_num(f['mucape'], nan=0))
    out = {}
    for name, v in (('t2', f['2t']), ('wind', wind), ('ws', ws), ('top', top)):
        for k, p in zip(('lo', 'mid', 'hi'), np.percentile(v, PCT, axis=0)):
            out[f'{name}_{k}'] = p
    gust = f.get('gust')
    if gust is not None and hours:
        out['gust_mid'], out['gust_hi'] = np.percentile(gust, (50, 90), axis=0)
    storm = (rain >= STORM_MM_H) & (conv >= STORM_CP * rain) & (cape >= STORM_CAPE)
    if hours:
        out['rainp'] = (rain >= RAIN_MM_H).mean(axis=0) * 100
        out['stormp'] = storm.mean(axis=0) * 100
        out['rain_hi'] = np.percentile(rain, 90, axis=0)
    score = hour_score(ws, top, rain, gust if gust is not None else np.zeros_like(rain))
    return (out, score, storm)

def window(f, prev, hours):
    for acc, name, scale in (('ssr', 'rnsw', 1.0), ('str', 'rnlw', 1.0), ('tp', 'rain', 3600000.0), ('cp', 'conv', 3600000.0)):
        now = f.pop(acc, None)
        if hours and now is not None and (prev.get(acc) is not None):
            f[name] = np.maximum(now - prev[acc], 0 if acc in ('tp', 'cp') else -np.inf) / (hours * 3600) * scale
        else:
            f[name] = np.zeros_like(f['2t'])
        prev[acc] = now

def build(out, run=None, last=LAST_FH, region=REGION, keep=KEEP_ID, workers=1, force=False):
    run = run or latest_run(last=last)
    if run is None:
        raise SystemExit('no complete ensemble run found')
    run_id = f'{run:%Y%m%d%H}'
    base = os.path.join(out, 'ens')
    latest = os.path.join(base, 'latest.json')
    if os.path.exists(latest) and (not force):
        with open(latest) as fh:
            cur = json.load(fh)
        if cur.get('runId') == run_id and cur.get('complete'):
            B.log(f'ens: {run_id} already built, nothing to do')
            return 0
    for d in os.listdir(base) if os.path.isdir(base) else []:
        if d.endswith('.building'):
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)
    build_dir = os.path.join(base, f'{run_id}.building')
    hours_dir = os.path.join(base, f'{run_id}.hours')
    shutil.rmtree(hours_dir, ignore_errors=True)
    os.makedirs(hours_dir)
    os.makedirs(build_dir)
    fhs = steps(last)
    B.log(f'ens: run {run:%Y-%m-%d %HZ}, {len(fhs)} steps, {MEMBERS} members')
    t0 = time.time()
    static = fetch_static(run)
    cells = Cells(static['z'], region, keep)
    zs = cells.take(static['z']) / G
    land = cells.take(static['lsm'])
    B.log(f'ens: {len(cells.flat)} cells')
    downloads = Downloads(run, fhs)
    prev, stats, prev_fh = ({}, {}, None)
    days = Days(run, run + timedelta(hours=fhs[-1]), cells.lon)
    try:
        for fh in fhs:
            t1 = time.time()
            raw = downloads.take(fh)
            waited = time.time() - t1
            f = decode_step(cells, raw)
            del raw
            hours = 0 if prev_fh is None else fh - prev_fh
            window(f, prev, hours)
            valid = run + timedelta(hours=fh)
            st, score, storm = step_stats(f, cells, zs, land, valid, hours)
            days.add(valid, cells.lon, score, storm)
            stats[valid] = {k: q(k, v) for k, v in st.items()}
            prev_fh = fh
            B.log(f'ens: +{fh} h in {time.time() - t1:.0f} s ({waited:.0f} s waiting for its download)')
    finally:
        downloads.close()
    times = sorted(stats)
    for valid in times:
        d = {k: q(k, v) for k, v in days.fields(valid, cells.lon).items()}
        if valid == times[0]:
            d['elevation'] = q('elevation', zs)
        grids = {k: cells.grid(v) for k, v in {**stats[valid], **d}.items()}
        np.savez(os.path.join(hours_dir, f'{valid:%Y%m%d%H}.npz'), **grids)
    points = B.write_points(build_dir, hours_dir, times, times[0], workers=workers, fields=FIELDS)
    shutil.rmtree(hours_dir, ignore_errors=True)
    lons, lats = (cells.lons, cells.lats)
    present = set(points['fields']) if points else set()
    stamps = [f'{t:%Y%m%d%H}' for t in times]
    by_day = {}
    for s in stamps:
        by_day.setdefault(s[:8], []).append(s)
    manifest = {'model': 'ens', 'label': 'ECMWF ensemble', 'run': run.isoformat(), 'runId': run_id, 'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'complete': last == LAST_FH and len(times) == len(fhs), 'members': MEMBERS, 'percentiles': list(PCT), 'need': NEED, 'grid': {'west': float(lons[0]), 'north': float(lats[0]), 'step': region[4], 'nx': len(lons), 'ny': len(lats)}, 'days': by_day, 'fields': {k: {'min': ENC[k][0], 'max': ENC[k][1], 'transform': ENC[k][2], 'tiles': []} for k in FIELDS if k in present}, 'heights': 'agl', 'sunGate': True, 'points': points, 'pack': 1}
    rdir = os.path.join(base, run_id)
    shutil.rmtree(rdir, ignore_errors=True)
    os.replace(build_dir, rdir)
    with open(latest + '.tmp', 'w') as fh:
        json.dump(manifest, fh, separators=(',', ':'))
    os.replace(latest + '.tmp', latest)
    open(os.path.join(out, '.changed'), 'w').close()
    for old in sorted((d for d in os.listdir(base) if d.isdigit()))[:-1]:
        shutil.rmtree(os.path.join(base, old), ignore_errors=True)
    B.log(f'ens: run {run_id}: {len(times)} steps in {B._fmt_dur(time.time() - t0)}')
    return 0
