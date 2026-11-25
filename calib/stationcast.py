from __future__ import annotations
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog
import common
UTC = timezone.utc
UNIT_V = 1
LEAD_MAX_H = 48
DAYLIGHT_H = {'US': tuple(range(14, 24)), 'SH': (*range(20, 24), *range(0, 8)), 'EU': tuple(range(6, 17))}
LEAD_STEP = {'hrrr': 1, 'gfs': 1, 'ecmwf': 3, 'icon': 1, 'icond2': 1, 'arome': 1}
JOB_ENV = {'ACC_CLASS': 'own', 'AIR_CORRECTIONS': '0'}
READY_AFTER_H = LEAD_MAX_H + 12
MONTHS = '2026-05:2026-11'
PLAN_AREAS = {'hrrr': ('US',), 'gfs': ('US', 'SH', 'EU'), 'ecmwf': ('US', 'SH', 'EU')}
AREA_BOX = {'US': common.US_BOX, 'EU': (5.5, 43.5, 16.0, 48.5), 'SH': (112.0, -49.0, 179.75, -18.5)}
AREA_MONTHS = {'SH': '2026-10:2026-11', 'EU': '2026-10:2026-11'}
AREA_MARGIN = 1.0
BUILD_TIMEOUT = 45 * 60

def _pipeline():
    if common.PIPELINE not in sys.path:
        sys.path.append(common.PIPELINE)

def _verify():
    _pipeline()
    import verify
    return verify

def enabled() -> bool:
    return (os.environ.get('CALIB_STCAST') or '').strip().lower() in ('on', '1', 'yes')

def months(text: str | None=None) -> list[tuple[int, int]]:
    t = os.environ.get('CALIB_STCAST_MONTHS', MONTHS) if text is None else text
    out = set()
    for part in (p.strip() for p in t.split(',')):
        if not part:
            continue
        a, _, b = part.partition(':')
        y0, m0 = (int(x) for x in a.split('-'))
        y1, m1 = (int(x) for x in (b or a).split('-'))
        if not (1 <= m0 <= 12 and 1 <= m1 <= 12):
            raise ValueError('stationcast: a month is 01-12')
        y, m = (y0, m0)
        while (y, m) <= (y1, m1):
            out.add((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return sorted(out)

def plan_areas(text: str | None=None) -> dict[str, tuple]:
    t = os.environ.get('CALIB_STCAST_AREAS') if text is None else text
    out = dict(PLAN_AREAS)
    for part in (t or '').split(','):
        m, _, a = part.partition(':')
        m = m.strip()
        if m in catalog.STCAST_AREAS:
            out[m] = tuple((x.strip() for x in a.split('+') if x.strip() in catalog.STCAST_AREAS[m]))
    return out

def area_region(model: str, area: str) -> list:
    _pipeline()
    import build
    rw, rs, re_, rn, step = build.REGIONS[model][:5]
    bw, bs, be, bn = (v + d * AREA_MARGIN for v, d in zip(AREA_BOX[area], (-1, -1, 1, 1)))
    nx = int(round((re_ - rw) / step))
    ny = int(round((rn - rs) / step))
    x0 = max(0, math.floor((bw - rw) / step + 1e-09))
    x1 = min(nx, math.ceil((be - rw) / step - 1e-09))
    y0 = max(0, math.floor((rn - bn) / step + 1e-09))
    y1 = min(ny, math.ceil((rn - bs) / step - 1e-09))
    return [round(rw + x0 * step, 6), round(max(rs, rn - y1 * step), 6), round(min(re_, rw + x1 * step), 6), round(rn - y0 * step, 6), step]

def station_cells(model: str, stations: list, region: list, area: str | None=None):
    import numpy as np
    verify = _verify()
    _pipeline()
    import build
    full = build.REGIONS[model]
    idx, iy, ix = verify.station_cells(model, stations)
    x0 = int(round((region[0] - full[0]) / full[4]))
    y0 = int(round((full[3] - region[3]) / full[4]))
    lons, lats = build.target_grid(region)
    iy, ix = (iy - y0, ix - x0)
    ok = (iy >= 0) & (iy < len(lats)) & (ix >= 0) & (ix < len(lons))
    if area is not None:
        w, s, e, n = AREA_BOX[area]
        lat = np.array([float(stations[i][1]) for i in idx], float)
        lon = np.array([float(stations[i][2]) for i in idx], float)
        ok &= (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    return (np.asarray(idx[ok], np.int32), np.asarray(iy[ok]), np.asarray(ix[ok]))

def day_hours(area: str) -> set[int]:
    return set(_verify().VALID_H) | set(DAYLIGHT_H.get(area, ()))

def want_leads(model: str, run_hour: int, area: str, lead_max: int=LEAD_MAX_H) -> list[int]:
    hours, step = (day_hours(area), LEAD_STEP[model])
    return [fh for fh in range(0, lead_max + 1, step) if (run_hour + fh) % 24 in hours]

def lead_hours(src, run: datetime, hours, lead_max: int=LEAD_MAX_H) -> list[int]:
    return [fh for fh in range(0, min(lead_max, src.last_fh(run)) + 1) if (run + timedelta(hours=fh)).hour in hours and src.fstep(fh) and src.builds(fh)]

def build_env(env: dict) -> dict:
    out = {**dict(env), **JOB_ENV}
    out.setdefault('CONTRAILS', '0')
    return out

def build_settings(env: dict) -> dict:
    e = build_env(env)
    return {'acc_class': e.get('ACC_CLASS', ''), 'aircorr': e.get('AIR_CORRECTIONS', '0') == '1', 'contrails': e.get('CONTRAILS', '1') != '0'}

def workers(model: str) -> int:
    return _verify().WORKERS.get(model, 3)

def units(name_of, month_list=None, now: datetime | None=None, areas: dict | None=None, counts: dict | None=None) -> list[dict]:
    now = now or datetime.now(UTC)
    month_list = months() if month_list is None else month_list
    areas = plan_areas() if areas is None else areas
    first = _verify().FIRST_RUN
    last = now - timedelta(hours=READY_AFTER_H)
    out = []
    for y, m in month_list:
        d = date(y, m, 1)
        while d.month == m:
            for h in catalog.STCAST_RUN_H:
                run = datetime(d.year, d.month, d.day, h, tzinfo=UTC)
                if run > last:
                    continue
                for model in catalog.STCAST_AREAS:
                    if model in first and run < first[model]:
                        continue
                    for area in areas.get(model, ()):
                        if area in AREA_MONTHS and (y, m) not in months(AREA_MONTHS[area]):
                            continue
                        try:
                            u = catalog.check_unit({'kind': catalog.STCAST, 'daykey': common.day_key(d.isoformat(), area), 'model': model, 'init': f'{run:%Y%m%d%H}', 'box': area})
                        except catalog.UnitError:
                            if counts is not None:
                                counts['bad_unit'] = counts.get('bad_unit', 0) + 1
                            continue
                        out.append({'unit': u, 'name': name_of(catalog.logical_id(u)), 'raw': [], 'kind': catalog.STCAST, 'day': u['daykey'], 'area': common.gv_area(model, area), 'model': model, 'run': model, 'lead_d': None, 'box': area, 'bbox': list(AREA_BOX[area]), 'tz': None, 'want_h': want_leads(model, h, area)})
            d += timedelta(days=1)
    return out

def bbox_text(region: list) -> str:
    return ','.join((f'{v:.6f}'.rstrip('0').rstrip('.') for v in region[:4]))

def build_command(model: str, run: str, region: list, out: str, n_workers: int, area: str) -> list[str]:
    return [sys.executable, os.path.abspath(__file__), '_build', model, run, out, str(n_workers), bbox_text(region), area]

def _build(model: str, run: str, out: str, n_workers: int, bbox: str, area: str) -> int:
    sys.path.insert(0, os.getcwd())
    import build
    r = datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=UTC)
    fhs = lead_hours(build.SOURCES[model], r, day_hours(area))
    if not fhs:
        print(f'{model}: none of the hours', flush=True)
        return 2
    hours = [r + timedelta(hours=f) for f in fhs]
    build.needed_times = lambda s, days=None, now=None: hours
    sys.argv = ['build.py', '--model', model, '--out', out, '--shard', '0/1', '--run', run, '--workers', str(n_workers), f'--bbox={bbox}']
    return build.main()

def build_of(cs: str) -> str:
    return f'stcast-{cs}'

def unit_arrays(hours_dir: str, u: dict, region: list, gv: dict, cs: str, stations: list, work: str) -> dict:
    _pipeline()
    import numpy as np
    import opslake
    import labbuild
    model, run = (u['model'], u['init'])
    out = tempfile.mkdtemp(prefix='stcast-', dir=work)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            done = opslake.write(hours_dir, run, out, model=model, build_id=build_of(cs), published=False, day_has_grids=True, region=list(region), stations=stations, blend_in=False, workers=1, complete=True, cells=station_cells(model, stations, region, u['box']))
        p = os.path.join(out, 'ops', model, run[:8], run[8:], f'b{build_of(cs)}', 'stations.npz')
        if not done or not os.path.exists(p):
            raise catalog.UnitError('nohours')
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    finally:
        shutil.rmtree(out, ignore_errors=True)
    stamps = [str(s) for s in arrays['stamps']]
    r0 = datetime.strptime(run, '%Y%m%d%H')
    leads = [int((datetime.strptime(s, '%Y%m%d%H') - r0).total_seconds() // 3600) for s in stamps]
    if set(want_leads(model, r0.hour, u['box'])) - set(leads):
        raise catalog.UnitError('hours')
    w, s, e, n, step = region[:5]
    meta = {'v': UNIT_V, 'id': u, 'model_id': catalog.model_id_of(u), 'area': common.gv_area(model, u['box']), 'macro': u['box'], 'bbox': [w, s, e, n], 'grid': {'west': w, 'north': n, 'step': step}, 'init': run, 'runs': [run] * len(stamps), 'stamps': stamps, 'lead_h': leads, 'src': catalog.STCAST, 'cs': cs, 'gv': {g: gv.get(g) for g in common.GROUPS}, 'score': None, 'sample_v': labbuild.sample_v(str(arrays['stations_v'])), 'stations': int(len(arrays['idx'])), 'grids': False, 'settings': build_settings(os.environ)}
    arrays['src'] = np.array(catalog.STCAST)
    arrays['meta'] = np.array(json.dumps(meta, sort_keys=True, separators=(',', ':')))
    return arrays

def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 7 and a[0] == '_build':
        return _build(a[1], a[2], a[3], int(a[4]), a[5], a[6])
    print(__doc__)
    return 2
if __name__ == '__main__':
    sys.exit(main())
