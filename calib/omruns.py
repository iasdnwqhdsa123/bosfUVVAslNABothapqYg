from __future__ import annotations
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog
import common
import stationcast
UTC = timezone.utc
URL = 'https://customer-single-runs-api.open-meteo.com/v1/forecast'
VIA = 'om-single-runs'
FIRST = datetime(2026, 4, 2, tzinfo=UTC)
VIA_RUN = 'om-data-run'
DATA_RUN = 'https://openmeteo.s3.amazonaws.com/data_run'
DATA_RUN_FIRST = {'icon': datetime(2026, 8, 24, tzinfo=UTC), 'icond2': datetime(2026, 8, 24, tzinfo=UTC), 'arome': datetime(2026, 8, 24, tzinfo=UTC)}
RUN_VARS = {'t2': 'temperature_2m', 'rh': 'relative_humidity_2m', 'u_sfc': 'wind_u_component_10m', 'v_sfc': 'wind_v_component_10m'}
MODELS = {'icon': {'om': 'icon_global', 'domain': 'dwd_icon', 'areas': ('US', 'SH'), 'runs': (0, 12)}, 'icond2': {'om': 'icon_d2', 'domain': 'dwd_icon_d2', 'areas': ('EU',), 'runs': (0, 12)}, 'arome': {'om': 'meteofrance_arome_france', 'domain': 'meteofrance_arome_france0025', 'areas': ('EU',), 'runs': (3, 15)}}
HOURLY = ('temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m')
BATCH = 50
GAP_S = 0.2
TRIES = 6
WAIT_MAX = 60.0

def key() -> str:
    return (os.environ.get('OPEN_METEO_API_KEY') or '').strip()

def enabled() -> bool:
    return (os.environ.get('CALIB_OMRUNS') or '1').strip() != '0'

def first_run(model: str) -> datetime:
    return FIRST if key() else DATA_RUN_FIRST[model]

def via_of(model: str, run: datetime) -> str:
    return VIA_RUN if run >= DATA_RUN_FIRST[model] else VIA

def run_hours(model: str) -> tuple:
    return MODELS[model]['runs']

def units(name_of, now: datetime | None=None, counts: dict | None=None) -> list[dict]:
    now = now or datetime.now(UTC)
    last = now - timedelta(hours=stationcast.READY_AFTER_H)
    out = []
    d = FIRST.date()
    while d <= last.date():
        for model, cfg in MODELS.items():
            for h in cfg['runs']:
                run = datetime(d.year, d.month, d.day, h, tzinfo=UTC)
                if run < first_run(model) or run > last:
                    continue
                for area in cfg['areas']:
                    try:
                        u = catalog.check_unit({'kind': catalog.STCAST, 'daykey': common.day_key(d.isoformat(), area), 'model': model, 'init': f'{run:%Y%m%d%H}', 'box': area})
                    except catalog.UnitError:
                        if counts is not None:
                            counts['bad_unit'] = counts.get('bad_unit', 0) + 1
                        continue
                    out.append({'unit': u, 'name': name_of(catalog.logical_id(u)), 'raw': [], 'kind': catalog.STCAST, 'day': u['daykey'], 'area': common.gv_area(model, area), 'model': model, 'run': model, 'lead_d': None, 'box': area, 'bbox': list(stationcast.AREA_BOX[area]), 'tz': None, 'want_h': stationcast.want_leads(model, h, area), 'via': via_of(model, run)})
        d += timedelta(days=1)
    return out

def is_om(u: dict) -> bool:
    return u.get('kind') == catalog.STCAST and u.get('model') in MODELS

def url_of(model: str, run: str, pts: list, lead_max: int) -> str:
    r = datetime.strptime(run, '%Y%m%d%H')
    q = {'latitude': ','.join((f'{p[0]:.4f}' for p in pts)), 'longitude': ','.join((f'{p[1]:.4f}' for p in pts)), 'hourly': ','.join(HOURLY), 'models': MODELS[model]['om'], 'run': f'{r:%Y-%m-%dT%H:%M}', 'forecast_hours': str(lead_max + 1), 'wind_speed_unit': 'ms', 'temperature_unit': 'celsius', 'timezone': 'GMT', 'timeformat': 'unixtime', 'cell_selection': 'nearest', 'elevation': ','.join(('nan' for _ in pts)), 'apikey': key()}
    return (os.environ.get('OPEN_METEO_SINGLE_RUNS_URL') or URL) + '?' + urllib.parse.urlencode(q, safe=',:')

def get_json(url: str, opener=None, sleep=time.sleep):
    op = opener or urllib.request.urlopen
    err = None
    for k in range(TRIES):
        wait = float(2 ** k)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'lab'})
            with op(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise
            err = e
            if e.code == 429 or e.code >= 500:
                ra = (e.headers.get('Retry-After') if e.headers else None) or ''
                wait = float(ra) if ra.strip().isdigit() else 2.0 ** (k + 2)
        except Exception as e:
            err = e
        if k < TRIES - 1:
            sleep(min(WAIT_MAX, wait))
    raise err

def values_of(reply: dict, r0: datetime) -> dict:
    h = reply.get('hourly') or {}
    out = {}
    for i, t in enumerate(h.get('time') or []):
        lead = int((datetime.fromtimestamp(int(t), UTC) - r0).total_seconds() // 3600)
        get = lambda n: (h.get(n) or [None] * (i + 1))[i]
        t2, rh, ws, wd = (get(n) for n in HOURLY)
        v = {}
        if t2 is not None:
            v['t2'] = float(t2)
            if rh is not None:
                v['td2'] = td2_of(float(t2), float(rh))
        if ws is not None and wd is not None:
            a = math.radians(float(wd))
            v['u_sfc'], v['v_sfc'] = (-float(ws) * math.sin(a), -float(ws) * math.cos(a))
        out[lead] = v
    return out

def td2_of(t2_c: float, rh: float) -> float:
    import numpy as np
    stationcast._pipeline()
    import om

    class G:

        def __call__(self, n):
            return np.array([t2_c + om.K0 if n == 'temperature_2m' else rh], np.float64)
    return float(om._td2(G())[0] - om.K0)

def run_meta(base: str, opener=None) -> dict:
    op = opener or urllib.request.urlopen
    err = None
    for k in range(TRIES):
        try:
            with op(urllib.request.Request(f'{base}/meta.json', headers={'User-Agent': 'lab'}), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            err = e
        except Exception as e:
            err = e
        if k < TRIES - 1:
            time.sleep(min(WAIT_MAX, 2.0 ** k))
    raise err

def _read_window(url: str, rows: slice, cols: slice, times: slice):
    stationcast._pipeline()
    import replay
    return replay._read(url, rows, cols, times)

def run_values(model: str, r0: datetime, lats, lons, leads: list, meta=None, read=None) -> dict:
    import numpy as np
    stationcast._pipeline()
    import sources
    lat0, lon0, step, ny, nx = sources.SOURCES[model].grid_def
    base = f"{DATA_RUN}/{MODELS[model]['domain']}/{r0:%Y/%m/%d/%H%M}Z"
    meta = meta if meta is not None else run_meta(base)
    vt = [datetime.strptime(t, '%Y-%m-%dT%H:%MZ').replace(tzinfo=UTC) for t in meta.get('valid_times') or []]
    ti = {ld: vt.index(r0 + timedelta(hours=ld)) for ld in leads if r0 + timedelta(hours=ld) in vt}
    n = len(lats)
    out = {f: np.full((n, len(leads)), np.nan) for f in ('t2', 'td2', 'u_sfc', 'v_sfc')}
    rows = np.rint((np.asarray(lats, np.float64) - lat0) / step).astype(int)
    cols = np.rint((np.asarray(lons, np.float64) - lon0) / step).astype(int)
    ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    if not ti or not ok.any():
        return out
    have = set(meta.get('variables') or RUN_VARS.values())
    r_lo, r_hi, c_lo, c_hi = (int(rows[ok].min()), int(rows[ok].max()) + 1, int(cols[ok].min()), int(cols[ok].max()) + 1)
    t_lo, t_hi = (min(ti.values()), max(ti.values()) + 1)
    read = read or _read_window
    got = {}
    for f, name in RUN_VARS.items():
        if name not in have:
            continue
        a = np.asarray(read(f'{base}/{name}.om', slice(r_lo, r_hi), slice(c_lo, c_hi), slice(t_lo, t_hi)), np.float64)
        got[f] = a
    k = [j for j, ld in enumerate(leads) if ld in ti]
    tk = [ti[leads[j]] - t_lo for j in k]
    pr, pc = (rows[ok] - r_lo, cols[ok] - c_lo)
    for f in ('t2', 'u_sfc', 'v_sfc'):
        if f in got:
            out[f][np.ix_(np.where(ok)[0], k)] = got[f][pr, pc][:, tk]
    if 't2' in got and 'rh' in got:
        import om
        t2, rh = (out['t2'], np.full_like(out['t2'], np.nan))
        rh[np.ix_(np.where(ok)[0], k)] = got['rh'][pr, pc][:, tk]

        class G:

            def __call__(self, name):
                return t2 + om.K0 if name == 'temperature_2m' else rh
        with np.errstate(invalid='ignore'):
            out['td2'] = om._td2(G()) - om.K0
    return out

def unit_arrays(u: dict, gv: dict, cs: str, stations: list, opener=None, sleep=time.sleep, clock=time.monotonic, hard_at: float | None=None, meta=None, read=None) -> dict:
    import numpy as np
    stationcast._pipeline()
    import build
    import opslake
    import labbuild
    model, run, area = (u['model'], u['init'], u['box'])
    region = stationcast.area_region(model, area)
    idx, iy, ix = stationcast.station_cells(model, stations, region, area)
    if not len(idx):
        raise catalog.UnitError('nostations')
    r0 = datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=UTC)
    want = stationcast.want_leads(model, r0.hour, area)
    fields = list(opslake.STATION_FIELDS)
    codes = np.full((len(want), len(idx), len(fields)), 255, np.uint8)
    col = {f: fields.index(f) for f in ('t2', 'td2', 'u_sfc', 'v_sfc')}
    pos = {ld: k for k, ld in enumerate(want)}
    pts = [(stations[i][1], stations[i][2]) for i in idx]
    via = via_of(model, r0)
    if via == VIA_RUN:
        w_, n_, step_ = (region[0], region[3], region[4])
        try:
            vals = run_values(model, r0, [n_ - y * step_ for y in iy], [w_ + x * step_ for x in ix], want, meta, read)
        except urllib.error.HTTPError as e:
            raise catalog.UnitError('norun' if e.code in (403, 404) else 'fetch') from None
        except Exception:
            raise catalog.UnitError('fetch') from None
        for f in ('t2', 'td2', 'u_sfc', 'v_sfc'):
            v = vals[f].T
            ok = np.isfinite(v)
            if ok.any():
                q = build.quantize(np.where(ok, v, 0).astype(np.float32), *build.ENC[f])
                codes[:, :, col[f]] = np.where(ok, q, 255)
    elif not key():
        raise catalog.UnitError('nokey')
    for b in range(0, len(pts) if via == VIA else 0, BATCH):
        if hard_at is not None and clock() > hard_at:
            raise catalog.UnitError('time')
        part = pts[b:b + BATCH]
        try:
            reply = get_json(url_of(model, run, part, max(want)), opener, sleep)
        except urllib.error.HTTPError as e:
            raise catalog.UnitError('norun' if e.code == 400 else 'fetch') from None
        except Exception:
            raise catalog.UnitError('fetch') from None
        reply = reply if isinstance(reply, list) else [reply]
        if len(reply) != len(part):
            raise catalog.UnitError('fetch')
        for j, rep in enumerate(reply):
            for ld, v in values_of(rep, r0).items():
                if ld not in pos:
                    continue
                for f, x in v.items():
                    codes[pos[ld], b + j, col[f]] = int(build.quantize(np.array([x], np.float32), *build.ENC[f])[0])
        sleep(GAP_S)
    has = (codes[:, :, col['t2']] != 255).any(axis=0)
    if not has.any():
        raise catalog.UnitError('nohours')
    got = [ld for ld in want if (codes[pos[ld], :, col['t2']] != 255).any()]
    if set(want) - set(got):
        raise catalog.UnitError('hours')
    import hashlib
    stations_v = hashlib.sha256(json.dumps(stations, separators=(',', ':')).encode()).hexdigest()[:12]
    enc = {k: list(build.ENC[k]) for k in fields if k in build.ENC}
    enc['overdev_nb'] = enc.get('overdev')
    stamps = [f'{r0 + timedelta(hours=ld):%Y%m%d%H}' for ld in want]
    arrays = {'stamps': np.array(stamps), 'idx': np.asarray(idx)[has], 'codes': codes[:, has, :], 'fields': np.array(fields), 'enc': np.array(json.dumps(enc)), 'model': np.array(model), 'run': np.array(run), 'stations_v': np.array(stations_v), 'build': np.array(stationcast.build_of(cs))}
    w, s, e, n, step = region[:5]
    meta = {'v': stationcast.UNIT_V, 'id': u, 'model_id': catalog.model_id_of(u), 'area': common.gv_area(model, area), 'macro': area, 'bbox': [w, s, e, n], 'grid': {'west': w, 'north': n, 'step': step}, 'init': run, 'runs': [run] * len(stamps), 'stamps': stamps, 'lead_h': list(want), 'src': catalog.STCAST, 'cs': cs, 'gv': {g: gv.get(g) for g in common.GROUPS}, 'score': None, 'sample_v': labbuild.sample_v(stations_v), 'stations': int(has.sum()), 'grids': False, 'via': via, 'fields_from': ['t2', 'td2', 'u_sfc', 'v_sfc']}
    arrays['src'] = np.array(catalog.STCAST)
    arrays['meta'] = np.array(json.dumps(meta, sort_keys=True, separators=(',', ':')))
    return arrays
