from __future__ import annotations
import argparse
import json
import math
import os
import time
import urllib.request
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
RUN_H = {'ecmwf': 0, 'icon': 0, 'icond2': 0, 'arome': 0}
GRID = {'ecmwf': (-170.0, -54.9, 0.1), 'icon': (-170.0, -54.875, 0.125), 'icond2': (-3.94, 43.18, 0.02), 'arome': (-12.0, 37.5, 0.025)}
DOMAIN = {'icond2': (-3.94, 43.18, 20.34, 58.08), 'arome': (-12.0, 37.5, 16.0, 55.4)}
ECMWF_025_FIRST = '2024-03-01'
OM_DOMAIN, OM_CHUNK = ('ecmwf_ifs', 504)
OM_RUN_BASE = 'https://openmeteo.s3.amazonaws.com/data_run/ecmwf_ifs'
OM_LATE = {'convective_inhibition': '2025-09-18', 'showers': '2025-09-18', 'visibility': '2025-09-18'}
MISSING = ('tcc', 'lcc', 'mcc', 'hcc', 'cprate', 'tclw')
BANDS = {'tcc': ('cloud_cover', None), 'lcc': ('cloud_cover_low', None), 'mcc': ('cloud_cover_mid', None), 'hcc': ('cloud_cover_high', None)}
MIRRORS = ('https://storage.googleapis.com/ecmwf-open-data', 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com')

def snap(bbox, model: str) -> list[float]:
    w0, s0, step = GRID[model]
    w, s, e, n = bbox
    f = lambda v, o, up: round(o + (math.ceil if up else math.floor)(round((v - o) / step, 6)) * step, 6)
    return [f(w, w0, False), f(s, s0, False), f(e, w0, True), f(n, s0, True)]

def covers(model: str, bbox) -> bool:
    d = DOMAIN.get(model)
    if not d:
        return True
    w, s, e, n = snap(bbox, model)
    return d[0] - 1e-06 <= w and d[1] - 1e-06 <= s and (e <= d[2] + 1e-06) and (n <= d[3] + 1e-06)

def utc_hours(day: str, tz: str, from_local: int=6, to_local: int=21) -> list[datetime]:
    z = ZoneInfo(tz)
    d = datetime.strptime(day, '%Y-%m-%d')
    return sorted({datetime(d.year, d.month, d.day, h, tzinfo=z).astimezone(timezone.utc) for h in range(from_local, to_local + 1)})

def forecast_hours(day: str, hours: list[datetime], model: str) -> tuple[datetime, list[int]]:
    d = datetime.strptime(day, '%Y-%m-%d')
    run = datetime(d.year, d.month, d.day, RUN_H[model], tzinfo=timezone.utc)
    return (run, [f for f in (int((t - run).total_seconds() // 3600) for t in hours) if f >= 1])

def run_meta(run: datetime) -> dict:
    for k in range(4):
        try:
            with urllib.request.urlopen(f'{OM_RUN_BASE}/{run:%Y/%m/%d/%H%M}Z/meta.json', timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == 3:
                raise
            time.sleep(5 * 2 ** k)

def om_names(src, day: str) -> set:
    _, extra = src.om_extra
    names = set(extra.values()) | {v for v, _ in src.om_hourly.values()} | {v for v, _ in src.om_between.values()} | {v for v, _ in BANDS.values()}
    return {n for n in names if day >= OM_LATE.get(n, '')}

def _ecmwf(a, hours: list[datetime]) -> int:
    import copy
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import build
    import om
    import replay
    import sources
    if a.day < ECMWF_025_FIRST:
        print(f'ecmwf: no 0.25° open data before {ECMWF_025_FIRST}', flush=True)
        return 2
    run, fhs = forecast_hours(a.day, hours, 'ecmwf')
    orig = sources.SOURCES['ecmwf']
    if a.lead:
        run -= timedelta(days=a.lead)
        fhs = [f for f in (int((t - run).total_seconds() // 3600) for t in hours) if f >= 1 and orig.builds(f) and (f <= orig.max_fhour)]
        hours = [run + timedelta(hours=f) for f in fhs]
    per_run = f'{run:%Y-%m-%d}' >= common.ECMWF_RUNS_FIRST
    if a.lead and (not per_run):
        print(f'ecmwf: run {run:%Y-%m-%d %HZ}: Open-Meteo keeps runs from {common.ECMWF_RUNS_FIRST} only', flush=True)
        return 2

    class IFS025(sources.IFS):
        MIRRORS = MIRRORS
        PARTS = ('sfc',)

        def url(self, run_, fh, part='sfc', mirror=None):
            m = self.MIRRORS[self.mirror if mirror is None else mirror]
            return f'{m}/{run_:%Y%m%d}/{run_:%H}z/ifs/0p25/oper/{run_:%Y%m%d%H}0000-{fh}h-oper-fc.grib2'

        def idx_url(self, run_, fh, part='sfc'):
            return self.url(run_, fh)[:-len('.grib2')] + '.index'

        def _each_mirror(self, call):
            import time
            for k in range(6):
                self.mirror = k % len(self.MIRRORS)
                try:
                    return call()
                except Exception:
                    if k == 5:
                        raise
                    time.sleep(5 * 2 ** k)
    src = copy.copy(orig)
    src.__class__ = IFS025
    src.surface = {k: v for k, v in orig.surface.items() if k not in MISSING}
    src.windowed = {k: v for k, v in orig.windowed.items() if k in src.surface}
    src.notes = [f"Replay of {a.day}: ECMWF's archived 0.25° open data with Open-Meteo's archive of the 9 km model"] + list(orig.notes)
    build.SOURCES['ecmwf'] = src
    w, s, e, n = snap([float(x) for x in a.bbox.split(',')], 'ecmwf')
    names = om_names(orig, a.day)
    rows = om.gaussian_rows()
    box = om.gaussian_index(np.arange(n + 0.2, s - 0.21, -0.05), np.arange(w - 0.2, e + 0.21, 0.05), rows)
    lo, hi = (int(box.min()), int(box.max()) + 1)
    if per_run:
        meta = run_meta(run)
        steps = {datetime.strptime(v, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc): i for i, v in enumerate(meta['valid_times'])}
        tidx = [steps.get(t) for t in hours]
        cache = os.path.join(a.cache, f'{OM_DOMAIN}-run{run:%Y%m%d%H}-{a.day}-{lo}-{hi}')
    else:
        idx = [int((t - replay.EPOCH).total_seconds() // 3600) for t in hours]
        cache = os.path.join(a.cache, f'{OM_DOMAIN}-{a.day}-{lo}-{hi}')
    os.makedirs(cache, exist_ok=True)
    print(f"ecmwf: run {run:%Y-%m-%d %HZ}{(f' (+{a.lead} d)' if a.lead else '')}, {len(fhs)} hours; caching {len(names)} fields of Open-Meteo's {('copy of the run' if per_run else 'archive')}, {hi - lo} cells", flush=True)

    def si(name, arr):
        if name.startswith(('temperature_', 'dew_point_', 'surface_temperature')):
            return arr + 273.15
        if name == 'pressure_msl':
            return arr * 100
        return arr

    def one(name):
        f = os.path.join(cache, f'{name}.npy')
        try:
            if not os.path.exists(f) and per_run:
                arr = np.full((len(hours), hi - lo), np.nan, np.float32)
                have = [(k, i) for k, i in enumerate(tidx) if i is not None]
                if have:
                    t0, t1 = (min((i for _, i in have)), max((i for _, i in have)) + 1)
                    arr_r = replay._read(f'{OM_RUN_BASE}/{run:%Y/%m/%d/%H%M}Z/{name}.om', slice(0, 1), slice(lo, hi), slice(t0, t1))[0]
                    for k, i in have:
                        arr[k] = arr_r[:, i - t0]
                np.save(f + '.tmp.npy', si(name, arr))
                os.replace(f + '.tmp.npy', f)
            elif not os.path.exists(f):
                arr = np.full((len(hours), hi - lo), np.nan, np.float32)
                for chunk in sorted({i // OM_CHUNK for i in idx}):
                    want = [(k, i % OM_CHUNK) for k, i in enumerate(idx) if i // OM_CHUNK == chunk]
                    t0, t1 = (min((t for _, t in want)), max((t for _, t in want)) + 1)
                    arr_c = replay._read(f'{replay.BASE}/{OM_DOMAIN}/{name}/chunk_{chunk}.om', slice(0, 1), slice(lo, hi), slice(t0, t1))[0]
                    for k, t in want:
                        arr[k] = arr_c[:, t - t0]
                np.save(f + '.tmp.npy', si(name, arr))
                os.replace(f + '.tmp.npy', f)
            return (name, np.load(f, mmap_mode='r'))
        except Exception as ex:
            print(f'  missing: {name}: {ex}', flush=True)
            return (name, None)
    with ThreadPoolExecutor(4) as ex:
        data = {k: v for k, v in ex.map(one, sorted(names)) if v is not None}
    domain, extra = orig.om_extra

    def om_extra(src_, run_, fh, between=False, idx_=None):
        k = fhs.index(fh) if fh in fhs else None
        if k is None or idx_ is None:
            return {}
        want = {c: (v, None) for c, v in extra.items()}
        if fh <= src_.between_to:
            want.update(src_.om_hourly)
            if between:
                want.update(src_.om_between)
        want.update({c: v for c, v in BANDS.items() if c not in want})
        return {c: ([np.asarray(data[v][k])[np.asarray(p) - lo] for p in idx_], conv) for c, (v, conv) in want.items() if v in data}
    build.open_meteo_extra = om_extra
    build.needed_times = lambda s_, days=None, now=None: hours
    build.pick_run = lambda s_, needed, pin=None: (run, fhs)
    build.backfill_jobs = lambda *x, **k: []
    sys.argv = ['build.py', '--model', 'ecmwf', '--out', a.out, '--force', '--workers', str(a.workers), f'--bbox={w},{s},{e},{n}']
    return build.main()

def _replay(a, hours: list[datetime]) -> int:
    bbox = [float(x) for x in a.bbox.split(',')]
    if not covers(a.model, bbox):
        print(f"{a.model}: the box is outside the model's domain", flush=True)
        return 2
    first = common.FIRST_DAY.get(a.model, common.ICON_FIRST)
    if a.day < first:
        print(f"{a.model}: the archive's layout before {first} isn't read (docs/replay.md, Older days)", flush=True)
        return 2
    import replay
    w, s, e, n = snap(bbox, a.model)
    utc_day = hours[0].strftime('%Y-%m-%d')
    if hours[-1].strftime('%Y-%m-%d') != utc_day:
        hours = [t for t in hours if t.strftime('%Y-%m-%d') == utc_day]
    sys.argv = ['replay.py', '--model', a.model, '--day', utc_day, '--from-h', str(hours[0].hour), '--to-h', str(hours[-1].hour), '--out', a.out, '--cache', a.cache, f'--bbox={w},{s},{e},{n}', '--workers', str(a.workers)]
    return replay.main()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--tz', required=True)
    ap.add_argument('--bbox', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='ecmwf', choices=sorted(RUN_H))
    ap.add_argument('--cache', default='../replay-cache')
    ap.add_argument('--from-local', type=int, default=6)
    ap.add_argument('--to-local', type=int, default=21)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lead', type=int, default=0, help='ECMWF: the run this many days before (common.LEADS)')
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    if a.lead and a.model != 'ecmwf':
        print(f'{a.model}: no lead runs read', flush=True)
        return 2
    hours = utc_hours(a.day, a.tz, a.from_local, a.to_local)
    return _ecmwf(a, hours) if a.model == 'ecmwf' else _replay(a, hours)
if __name__ == '__main__':
    sys.exit(main())
