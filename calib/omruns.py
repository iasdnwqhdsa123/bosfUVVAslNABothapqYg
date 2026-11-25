from __future__ import annotations
import json
import math
import os
import sys
import time
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
MODELS = {'icon': {'om': 'icon_global', 'areas': ('US', 'SH'), 'runs': (0, 12)}, 'icond2': {'om': 'icon_d2', 'areas': ('EU',), 'runs': (0, 12)}, 'arome': {'om': 'meteofrance_arome_france', 'areas': ('EU',), 'runs': (3, 15)}}
HOURLY = ('temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m')
BATCH = 50
GAP_S = 0.2
TRIES = 6
WAIT_MAX = 60.0

def key() -> str:
    return (os.environ.get('OPEN_METEO_API_KEY') or '').strip()

def enabled() -> bool:
    return bool(key()) and (os.environ.get('CALIB_OMRUNS') or '1').strip() != '0'

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
                if run < FIRST or run > last:
                    continue
                for area in cfg['areas']:
                    try:
                        u = catalog.check_unit({'kind': catalog.STCAST, 'daykey': common.day_key(d.isoformat(), area), 'model': model, 'init': f'{run:%Y%m%d%H}', 'box': area})
                    except catalog.UnitError:
                        if counts is not None:
                            counts['bad_unit'] = counts.get('bad_unit', 0) + 1
                        continue
                    out.append({'unit': u, 'name': name_of(catalog.logical_id(u)), 'raw': [], 'kind': catalog.STCAST, 'day': u['daykey'], 'area': common.gv_area(model, area), 'model': model, 'run': model, 'lead_d': None, 'box': area, 'bbox': list(stationcast.AREA_BOX[area]), 'tz': None, 'want_h': stationcast.want_leads(model, h, area), 'via': VIA})
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

def unit_arrays(u: dict, gv: dict, cs: str, stations: list, opener=None, sleep=time.sleep, clock=time.monotonic, hard_at: float | None=None) -> dict:
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
    for b in range(0, len(pts), BATCH):
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
    meta = {'v': stationcast.UNIT_V, 'id': u, 'model_id': catalog.model_id_of(u), 'area': common.gv_area(model, area), 'macro': area, 'bbox': [w, s, e, n], 'grid': {'west': w, 'north': n, 'step': step}, 'init': run, 'runs': [run] * len(stamps), 'stamps': stamps, 'lead_h': list(want), 'src': catalog.STCAST, 'cs': cs, 'gv': {g: gv.get(g) for g in common.GROUPS}, 'score': None, 'sample_v': labbuild.sample_v(stations_v), 'stations': int(has.sum()), 'grids': False, 'via': VIA, 'fields_from': ['t2', 'td2', 'u_sfc', 'v_sfc']}
    arrays['src'] = np.array(catalog.STCAST)
    arrays['meta'] = np.array(json.dumps(meta, sort_keys=True, separators=(',', ':')))
    return arrays
