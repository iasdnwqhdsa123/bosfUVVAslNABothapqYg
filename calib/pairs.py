from __future__ import annotations
import hashlib
import io
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
UTC = timezone.utc
PV = 'p1'
KINDS = ('metar', 'raob', 'sat', 'flight_leg', 'flight_climb', 'flight_probe', 'ogn_climb')
SRCS = ('ops', 'archive', 'stcast', 'hind')
MODELS = ('hrrr', 'gfs', 'ecmwf', 'icon', 'icond2', 'arome')
LAG_D = {'metar': 1, 'raob': 10, 'sat': 2, 'ogn_climb': 3, 'flight_leg': None, 'flight_climb': None, 'flight_probe': None}
REVIEW_KINDS = ('ogn_climb',)
COLLECTOR_TRAIN_ENV = 'COLLECTOR_TRAIN'
FEEDS = ('ogn', 'adsb', 'skylines')
FLIGHT_KINDS = ('flight_leg', 'flight_climb', 'flight_probe', 'ogn_climb')
METAR_VARS = ('t2', 'td2', 'u_sfc', 'v_sfc', 'gust', 'mslp', 'vis', 'ceil', 'cubase', 'cuclass', 'cloud', 'cloudlow', 'rain', 'overdev_nb', 'elevation')
SAT_VARS = ('cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ctop', 'cuclass', 'cucover', 'cubase', 'elevation')
RAOB_VARS = ('zi', 'depth', 'blapse', 'cubase', 'cuclass', 'wstar', 't2', 'td2', 'u_sfc', 'v_sfc', 'u_blavg', 'v_blavg', 'u_bltop', 'v_bltop', 'elevation')
WIDE = ('mslp',)
RAOB_NEAR_KM = 25.0

def _rows_mod():
    import rows
    return rows

def flight_vars() -> tuple:
    return tuple(_rows_mod().MODEL_KEYS)

def kind_vars(kind: str) -> tuple:
    if kind == 'metar':
        return METAR_VARS
    if kind == 'sat':
        return SAT_VARS
    if kind == 'raob':
        return RAOB_VARS
    if kind in ('flight_leg', 'flight_climb', 'ogn_climb'):
        return flight_vars()
    if kind == 'flight_probe':
        import track
        return tuple(track.PROBE_KEYS)
    raise ValueError('pairs: kind')

def _pipeline():
    if common.PIPELINE not in sys.path:
        sys.path.append(common.PIPELINE)

def _verify():
    _pipeline()
    import verify
    return verify

def classes_of(codes):
    _pipeline()
    import sky
    return sky.classes_of(codes)

def collector_train(env=None) -> bool:
    e = os.environ if env is None else env
    return (e.get(COLLECTOR_TRAIN_ENV) or '').strip() == '1'

def train_rows(table: dict, env=None) -> dict:
    if collector_train(env):
        return table
    if 'review' not in table:
        if n_rows(table):
            raise ValueError('pairs: a table without its review column')
        return table
    if not len(table['review']):
        return table
    keep = ~np.asarray(table['review'], bool)
    return {k: np.asarray(v)[keep] for k, v in table.items()}

def _ts(s: str) -> int:
    return int(datetime.strptime(s, '%Y%m%d%H').replace(tzinfo=UTC).timestamp())

def _day_of_ts(t: int) -> str:
    return datetime.fromtimestamp(int(t), UTC).strftime('%Y-%m-%d')

def _iso_ts(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(str(s).replace('Z', '+00:00')).timestamp())
    except ValueError:
        return None
SLOTS = {'01:15': {'hrrr': 23, 'arome': 21}, '02:00': {'hrrr': 0, 'ecmwf': 18, 'icond2': 0}, '02:30': {'hrrr': 1}, '03:30': {'hrrr': 2}, '03:55': {'icond2': 3, 'hrrr': 3, 'icon': 0}, '05:30': {'gfs': 0, 'hrrr': 4, 'arome': 3}, '06:30': {'hrrr': 5}, '08:00': {'hrrr': 6, 'icond2': 6}, '08:30': {'hrrr': 7}, '09:40': {'ecmwf': 0, 'icon': 6, 'hrrr': 8}, '10:30': {'hrrr': 9, 'icond2': 9}, '11:30': {'gfs': 6, 'hrrr': 10}, '12:30': {'hrrr': 11}, '13:30': {'icond2': 12, 'arome': 9}, '14:00': {'hrrr': 12, 'ecmwf': 6}, '14:30': {'hrrr': 13}, '15:40': {'hrrr': 14, 'icon': 12}, '16:30': {'icond2': 15, 'hrrr': 15}, '17:30': {'gfs': 12, 'hrrr': 16}, '18:30': {'hrrr': 17, 'arome': 15}, '19:30': {'icond2': 18}, '20:00': {'hrrr': 18}, '20:30': {'hrrr': 19}, '21:40': {'icon': 18, 'ecmwf': 12, 'hrrr': 20}, '22:30': {'icond2': 21, 'hrrr': 21}, '23:20': {'gfs': 18, 'hrrr': 22}}
OVERDUE_S = 3600
BUILD_MIN = 45

def avail_after(model: str, hour: int) -> timedelta:
    hm = lambda s: timedelta(hours=int(s[:-3]), minutes=int(s[-2:]))
    if model == 'hrrr':
        t = '1:47' if hour % 6 == 0 else '1:52' if hour == 23 else '1:25'
    elif model == 'icond2':
        t = {12: '2:00', 15: '1:55', 18: '1:50', 21: '1:46'}.get(hour, '2:04')
    elif model == 'arome':
        t = {3: '2:50', 9: '4:16', 15: '3:50', 21: '4:19'}.get(hour, '5:10')
    elif model == 'icon':
        t = {0: '4:20', 6: '3:41', 12: '4:01'}.get(hour, '3:34')
    elif model == 'gfs':
        t = '5:18'
    else:
        t = '9:35' if hour % 12 == 0 else '8:28'
    return hm(t)

def slot_of(model: str, hour: int) -> str | None:
    return next((n for n, ms in SLOTS.items() if ms.get(model) == hour), None)

def build_start(model: str, run: str) -> int:
    r0 = _ts(run)
    land = r0 + int(avail_after(model, int(run[8:10])).total_seconds())
    name = slot_of(model, int(run[8:10]))
    if name is None:
        return land + OVERDUE_S
    d = datetime.fromtimestamp(r0, UTC)
    s = int(d.replace(hour=int(name[:2]), minute=int(name[3:]), second=0).timestamp())
    if s <= r0:
        s += 86400
    return max(s, land)

def rebuilt_avail(model: str, run: str, build_min: float | None=None) -> int:
    return build_start(model, run) + int(60 * (BUILD_MIN if build_min is None else build_min))

def build_minutes(records: list) -> dict:
    by = {}
    for r in records:
        if r.get('src') == 'ops' and r.get('avail_t') is not None and re.fullmatch('\\d{10}', r.get('run') or ''):
            dt = (int(r['avail_t']) - build_start(r['model'], r['run'])) / 60
            if 0 <= dt <= 6 * 60:
                by.setdefault(r['model'], []).append(dt)
    return {m: float(np.median(v)) for m, v in by.items()}

def _s(x) -> str:
    return str(np.asarray(x).item()) if np.ndim(x) == 0 else str(x)

def stations_v_of(stations: list) -> str:
    try:
        return hashlib.sha256(json.dumps(stations, separators=(',', ':')).encode()).hexdigest()[:12]
    except TypeError:
        raise ValueError('pairs: the station list is not as read from its JSON file (numpy values in it)') from None

def _same_list(stations: list, v) -> None:
    if v is None:
        return
    v = np.asarray(v)
    v = v.item() if v.ndim == 0 else ''
    if isinstance(v, bytes):
        v = v.decode('ascii', 'replace')
    v = str(v)
    if re.fullmatch('[0-9a-f]{12}', v) and stations_v_of(stations) != v:
        raise ValueError('pairs: the station list is not the one the record was sampled with (stations_v)')

def record_from_stations(z, stations: list, done: dict | None=None, src: str='ops', build_min=None) -> dict:
    model, run = (_s(z['model']), _s(z['run']))
    stamps = [str(s) for s in np.asarray(z['stamps'])]
    idx = np.asarray(z['idx'], np.int64)
    if src not in SRCS:
        raise ValueError('pairs: src')
    if done is None and src in ('stcast', 'archive'):
        avail = rebuilt_avail(model, run, build_min)
    else:
        avail = _iso_ts((done or {}).get('published_t')) if (done or {}).get('published') else None
    enc = z['enc']
    enc = json.loads(_s(enc)) if not isinstance(enc, dict) else enc
    files = list(z.files) if hasattr(z, 'files') else list(z.keys())
    _same_list(stations, z['stations_v'] if 'stations_v' in files else None)
    return {'model': model, 'run': run, 'build': _s(z['build']) if 'build' in files else (done or {}).get('build', ''), 'src': src, 'replay': False, 'missing_w': False, 'missing_blh_heat': False, 'review': False, 'avail_t': avail, 'stamps': stamps, 'runs': [run] * len(stamps), 'leads': [(_ts(s) - _ts(run)) // 3600 for s in stamps], 'obs_ids': [str(stations[i][0]) for i in idx], 'codes': np.asarray(z['codes'], np.uint8), 'fields': [str(f) for f in np.asarray(z['fields'])], 'enc': enc, 'stations_v': _s(z['stations_v']) if 'stations_v' in files else None, 'elev_st': np.array([float(stations[i][3]) for i in idx]), 'lat': np.array([float(stations[i][1]) for i in idx]), 'lon': np.array([float(stations[i][2]) for i in idx])}

def record_from_unit(u: dict, stations: list, build_min=None) -> dict:
    meta = u.get('meta') or {}
    model = str(np.asarray(u['st.model']).item())
    runs = [str(r) for r in np.asarray(u['st.runs'])]
    stamps = [str(s) for s in np.asarray(u['st.stamps'])]
    leads = [int(x) for x in np.asarray(u['st.lead_h'])]
    run = str(np.asarray(u['st.run']).item())
    idx = np.asarray(u['st.idx'], np.int64)
    replay = meta.get('src') == 'replay'
    enc = u['st.enc']
    enc = json.loads(_s(enc)) if not isinstance(enc, dict) else enc
    _same_list(stations, u.get('st.stations_v'))
    return {'model': model, 'run': run, 'build': str(meta.get('cs') or ''), 'src': 'hind', 'replay': replay, 'missing_w': replay, 'missing_blh_heat': replay, 'review': str(np.asarray(u.get('st.selected_by', 'flight')).item()) == 'collector', 'avail_t': rebuilt_avail(model, run, build_min) if re.fullmatch('\\d{10}', run) else None, 'avail_runs': {r: rebuilt_avail(model, r, build_min) for r in set(runs) if re.fullmatch('\\d{10}', r or '')}, 'stamps': stamps, 'runs': runs, 'leads': leads, 'obs_ids': [str(stations[i][0]) for i in idx], 'codes': np.asarray(u['st.codes'], np.uint8), 'fields': [str(f) for f in np.asarray(u['st.fields'])], 'enc': enc, 'stations_v': str(np.asarray(u.get('st.stations_v', '')).item()), 'elev_st': np.array([float(stations[i][3]) for i in idx]), 'lat': np.array([float(stations[i][1]) for i in idx]), 'lon': np.array([float(stations[i][2]) for i in idx])}

def decoded(rec: dict, var: str, codes: np.ndarray | None=None) -> np.ndarray:
    c = rec['codes'] if codes is None else codes
    if var not in rec['fields']:
        return np.full(c.shape[:-1], np.nan)
    a = c[..., rec['fields'].index(var)]
    if var == 'cuclass':
        return classes_of(a)
    if var not in rec['enc']:
        return np.full(a.shape, np.nan)
    lo, hi, tr = rec['enc'][var]
    return _verify().decode(a, lo, hi, tr)

def _ftype(var: str):
    return np.float32 if var in WIDE else np.float16
PAIR_COMMON = (('kind', 'str'), ('obs_id', 'str'), ('t_valid', 'i64'), ('model', 'str'), ('run', 'str'), ('build', 'str'), ('lead_h', 'i16'), ('avail_t', 'i64?'), ('src', 'str'), ('replay', 'bool'), ('missing_w', 'bool'), ('missing_blh_heat', 'bool'), ('review', 'bool'))
PAIR_EXTRA = {'raob': (('cu_call', 'i8'), ('station', 'str'), ('dist_km', 'f32'))}
KEY = ('obs_id', 't_valid', 'model', 'run', 'build', 'src')

def pairs_schema(kind: str) -> tuple:
    f = tuple(((f'f_{v}', 'f32' if v in WIDE else 'f16') for v in kind_vars(kind)))
    if kind in FLIGHT_KINDS:
        f += tuple(((f'f_{v}_c', 'f16') for v in kind_vars(kind)))
    return PAIR_COMMON + PAIR_EXTRA.get(kind, ()) + f
METAR_O = ('t2', 'td2', 'ws', 'wd', 'gust', 'vis', 'ceil', 'cu', 'cubase', 'low_lo', 'low_hi', 'precip', 'storm', 'fog', 'mslp')
METAR_Q = ('t_half', 'ws_half', 'base_half', 'p_half', 'auto')
SAT_O = ('cloud_frac', 'water_frac', 'low_water_frac', 'ice_frac', 'cth_mean', 'cth_max', 'cbh_mean', 'cu_cover')
SAT_Q = ('n_pix', 'n_good', 'n_phase', 'n_height', 'n_base', 'dt_mask', 'sat_zen', 'shift_km', 'mask_levels')
LEG_O = ('speed', 'vg', 'vair', 'ca', 'dist', 't', 'dh', 'therm', 'agl', 'alt', 'tcount', 'wind')
LEG_Q = ('ok', 'engine', 'last')
CLIMB_O = ('rate', 'gain', 'dur', 'agl1', 'z1', 'r100')
CLIMB_Q = ('engon',)
OGN_O = ('rate', 'gain', 'dur', 'agl0', 'agl1', 'z1', 'air')
OGN_Q = ('type',)
OGN_TYPE = {'glider': 1, 'hg': 2, 'pg': 3}

def raob_names() -> tuple:
    import obs_raob as R
    names = ['t2', 'td2', 'ps', 'u_sfc', 'v_sfc', 'zi', 'zi_blue', 'zi_cool', 'lcl', 'depth', 'blapse', 'thml', 'thx']
    names += [f'{k}_m{dz}' for dz in R.ML_TOP_DZ for k in ('dth', 'q')]
    names += ['u_blavg', 'v_blavg', 'u_bltop', 'v_bltop']
    names += [f'{k}{ft}' for ft in R.WAVE_ALTS_FT for k in ('t', 'dpd')]
    names += [f'{k}_a{ft}' for ft in R.WIND_LOW_FT + R.WIND_ALTS_FT for k in ('u', 'v')]
    return tuple(names)

def obs_schema(kind: str) -> tuple:
    base = (('kind', 'str'), ('obs_id', 'str'), ('t_valid', 'i64'), ('lat', 'f32'), ('lon', 'f32'), ('review', 'bool'))
    if kind == 'metar':
        return base + (('elev', 'f32'),) + tuple(((f'o_{n}', 'f32') for n in METAR_O)) + tuple(((f'q_{n}', 'f32') for n in METAR_Q))
    if kind == 'raob':
        names = raob_names()
        return base + (('elev', 'f32'), ('t_release', 'i64?'), ('n_lev', 'i32'), ('top_hpa', 'f32')) + tuple(((f'o_{n}', 'f32') for n in names)) + tuple(((f'q_{n}', 'i32') for n in names))
    if kind == 'sat':
        return base + (('status', 'str'), ('sensor', 'str'), ('parallax', 'str')) + tuple(((f'o_{n}', 'f32') for n in SAT_O)) + tuple(((f'q_{n}', 'f32') for n in SAT_Q))
    if kind == 'flight_leg':
        return base + (('macro', 'str'),) + tuple(((f'o_{n}', 'f32') for n in LEG_O)) + tuple(((f'q_{n}', 'i8') for n in LEG_Q))
    if kind == 'flight_climb':
        return base + (('macro', 'str'),) + tuple(((f'o_{n}', 'f32') for n in CLIMB_O)) + tuple(((f'q_{n}', 'i8') for n in CLIMB_Q))
    if kind == 'ogn_climb':
        return base + (('macro', 'str'),) + tuple(((f'o_{n}', 'f32') for n in OGN_O)) + tuple(((f'q_{n}', 'i8') for n in OGN_Q))
    if kind == 'flight_probe':
        return base + (('macro', 'str'),)
    raise ValueError('pairs: kind')
STATIC_SCHEMA = (('obs_id', 'str'), ('model', 'str'), ('elev_obs', 'f32'), ('elev_model', 'f32'), ('dz_terrain', 'f32'), ('relief5km', 'f32'))
_NP = {'str': object, 'i64': np.int64, 'i64?': object, 'i32': np.int32, 'i16': np.int16, 'i8': np.int8, 'bool': bool, 'f32': np.float32, 'f16': np.float16}

def empty(schema) -> dict:
    return {n: np.zeros(0, _NP[t]) for n, t in schema}

def table(schema, cols: dict, n: int | None=None) -> dict:
    extra = set(cols) - {k for k, _ in schema}
    if extra:
        raise ValueError('pairs: a column outside the schema')
    if n is None:
        n = len(next(iter(cols.values()))) if cols else 0
    out = {}
    for name, t in schema:
        v = cols.get(name)
        if v is None:
            fill = {'str': '', 'i64?': None, 'bool': False}.get(t, np.nan if t.startswith('f') else -1)
            a = np.full(n, fill, dtype=_NP[t]) if t not in ('str', 'i64?') else np.array([fill] * n, dtype=object)
        elif t == 'str':
            a = np.array([str(x) for x in (v if np.ndim(v) else [v] * n)], dtype=object)
        elif t == 'i64?':
            a = np.array([None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x) for x in (v if np.ndim(v) else [v] * n)], dtype=object)
        else:
            a = np.asarray(v if np.ndim(v) else np.full(n, v)).astype(_NP[t])
        if len(a) != n:
            raise ValueError('pairs: columns of unequal length')
        out[name] = a
    return out

def concat(schema, parts: list) -> dict:
    parts = [p for p in parts if p and n_rows(p)]
    if not parts:
        return empty(schema)
    return {n: np.concatenate([p[n] for p in parts]) for n, _ in schema if n in parts[0]}

def n_rows(t: dict) -> int:
    return len(next(iter(t.values()))) if t else 0

def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError('pairs: pyarrow is missing (calib/requirements.txt)') from None
    return (pa, pq)

def to_parquet(schema, t: dict) -> bytes:
    pa, pq = _pa()
    types = {'str': pa.string(), 'i64': pa.int64(), 'i64?': pa.int64(), 'i32': pa.int32(), 'i16': pa.int16(), 'i8': pa.int8(), 'bool': pa.bool_(), 'f32': pa.float32(), 'f16': pa.float16()}
    arrays = []
    for name, ty in schema:
        v = t[name]
        if ty in ('str', 'i64?'):
            arrays.append(pa.array(list(v), type=types[ty]))
        else:
            arrays.append(pa.array(np.asarray(v, _NP[ty]), type=types[ty]))
    tab = pa.Table.from_arrays(arrays, names=[n for n, _ in schema])
    buf = io.BytesIO()
    pq.write_table(tab, buf, compression='zstd', write_statistics=False, store_schema=False)
    return buf.getvalue()

def from_parquet(schema, blob: bytes, columns=None) -> dict:
    pa, pq = _pa()
    names = [n for n, _ in schema if columns is None or n in columns]
    tab = pq.read_table(io.BytesIO(blob), columns=names)
    out = {}
    ty = dict(schema)
    for n in names:
        col = tab.column(n)
        if ty[n] in ('str', 'i64?'):
            out[n] = np.array(col.to_pylist(), dtype=object)
        elif ty[n] == 'f16':
            out[n] = np.asarray(col.to_numpy(zero_copy_only=False), np.float16)
        else:
            out[n] = np.asarray(col.to_numpy(zero_copy_only=False), _NP[ty[n]])
    return out

def _avail_of(rec: dict, ks, at):
    by = rec.get('avail_runs')
    if not by:
        return rec['avail_t']
    return np.array([by.get(rec['runs'][ks[i]]) for i in at], dtype=object)

def station_pairs(rec: dict, kind: str, day: str) -> dict:
    vars_ = kind_vars(kind)
    ks = [k for k, s in enumerate(rec['stamps']) if f'{s[:4]}-{s[4:6]}-{s[6:8]}' == day and re.fullmatch('\\d{10}', rec['runs'][k] or '') and (rec['leads'][k] >= 0)]
    sch = pairs_schema(kind)
    if not ks or not len(rec['obs_ids']):
        return empty(sch)
    codes = rec['codes'][ks]
    have = np.zeros(codes.shape[:2], bool)
    for v in vars_:
        if v in rec['fields']:
            have |= codes[:, :, rec['fields'].index(v)] != 255
    hh, ss = np.nonzero(have)
    tv = np.array([_ts(rec['stamps'][k]) for k in ks], np.int64)
    cols = {'kind': kind, 'obs_id': np.array(rec['obs_ids'], dtype=object)[ss], 't_valid': tv[hh], 'model': rec['model'], 'run': np.array([rec['runs'][k] for k in ks], dtype=object)[hh], 'build': rec['build'], 'lead_h': np.array([rec['leads'][k] for k in ks], np.int16)[hh], 'avail_t': _avail_of(rec, ks, hh), 'src': rec['src'], 'replay': rec['replay'], 'missing_w': rec['missing_w'], 'missing_blh_heat': rec['missing_blh_heat'], 'review': rec['review']}
    sel = codes[hh, ss]
    for v in vars_:
        cols[f'f_{v}'] = decoded(rec, v, sel[:, None, :])[:, 0].astype(_ftype(v)) if v in rec['fields'] else None
    return table(sch, cols, len(hh))

def metar_obs(arr: np.ndarray, day: str, stations: list) -> dict:
    V = _verify()
    k = {n: i for i, n in enumerate(V.OBS)}
    have = np.isfinite(arr).any(axis=2)
    si, hi = np.nonzero(have)
    d0 = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())
    cols = {'kind': 'metar', 'obs_id': np.array([str(stations[i][0]) for i in si], dtype=object), 't_valid': (d0 + hi * 3600).astype(np.int64), 'lat': np.array([stations[i][1] for i in si], np.float32), 'lon': np.array([stations[i][2] for i in si], np.float32), 'elev': np.array([stations[i][3] for i in si], np.float32)}
    for n in METAR_O:
        cols[f'o_{n}'] = arr[si, hi, k[n]]
    for n in METAR_Q:
        cols[f'q_{n}'] = arr[si, hi, k[n]]
    return table(obs_schema('metar'), cols, len(si))

def routine_first(text: str) -> tuple[str, str]:
    import csv
    from collections import Counter
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return (text, '')
    head, body = (rows[0], rows[1:])
    try:
        si, vi = (head.index('station'), head.index('valid'))
    except ValueError:
        return (text, '')
    minute = {}
    for r in body:
        if len(r) > max(si, vi) and len(r[vi]) >= 16:
            minute.setdefault(r[si], Counter())[r[vi][14:16]] += 1
    mode = {k: c.most_common(1)[0][0] for k, c in minute.items()}
    routine, rest = (io.StringIO(), io.StringIO())
    wr, wx = (csv.writer(routine, lineterminator='\n'), csv.writer(rest, lineterminator='\n'))
    wr.writerow(head)
    wx.writerow(head)
    for r in body:
        ok = len(r) > max(si, vi) and r[vi][14:16] == mode.get(r[si])
        (wr if ok else wx).writerow(r)
    return (routine.getvalue(), rest.getvalue())

def metar_obs_from_csv(text: str, day: str, stations: list) -> dict:
    V = _verify()
    d = datetime.fromisoformat(day).replace(tzinfo=UTC)
    ids = {V.iem_id(s[0]): i for i, s in enumerate(stations)}
    elev = [s[3] for s in stations]
    arr = np.full((len(stations), 24, len(V.OBS)), np.nan, np.float32)
    routine, rest = routine_first(text)
    for part in (rest, routine):
        for i, hours in V.parse_reports(part, ids, elev).items():
            for hr, vals in hours.items():
                if hr.date() == d.date():
                    arr[i, hr.hour] = vals
    V.qc(arr)
    return metar_obs(arr, day, stations)

def sat_obs(truth_rows: list, ids: list) -> dict:
    if len(truth_rows) != len(ids):
        raise ValueError('pairs: one id a truth row')
    cols = {'kind': 'sat', 'obs_id': np.array([str(i) for i in ids], dtype=object), 't_valid': np.array([_iso_ts(r['t_valid']) for r in truth_rows], np.int64), 'lat': np.array([r['lat'] for r in truth_rows], np.float32), 'lon': np.array([r['lon'] for r in truth_rows], np.float32), 'status': [r.get('status') or '' for r in truth_rows], 'sensor': [r.get('sensor') or '' for r in truth_rows], 'parallax': [r.get('parallax') or '' for r in truth_rows]}
    num = lambda x: np.nan if x is None else float(x)
    for n in SAT_O:
        cols[f'o_{n}'] = np.array([num(r.get(n)) for r in truth_rows], np.float32)
    for n in SAT_Q:
        cols[f'q_{n}'] = np.array([num(r.get(n)) for r in truth_rows], np.float32)
    return table(obs_schema('sat'), cols, len(truth_rows))
SAT_SAMPLE = 0.5

def cell_id(lat: float, lon: float) -> str:
    return f'g{lat:+07.2f}{lon:+08.2f}'

def sat_cell_obs(cells: list, t_valid: int, sensor: str='', step: float=SAT_SAMPLE) -> dict:
    rows, seen = ([], set())
    for c in cells:
        la, lo = (float(c['lat']), float(c['lon']))
        nla = (math.floor(la / step) + 0.5) * step
        nlo = (math.floor(lo / step) + 0.5) * step
        if abs(la - nla) > 0.051 or abs(lo - nlo) > 0.051:
            continue
        cid = cell_id(nla, nlo)
        if cid in seen:
            continue
        seen.add(cid)
        rows.append((cid, nla, nlo, c))
    if not rows:
        return empty(obs_schema('sat'))
    num = lambda x: np.nan if x is None else float(x)
    cols = {'kind': 'sat', 'obs_id': [r[0] for r in rows], 't_valid': np.full(len(rows), int(t_valid), np.int64), 'lat': np.array([r[1] for r in rows], np.float32), 'lon': np.array([r[2] for r in rows], np.float32), 'status': 'ok', 'sensor': sensor, 'parallax': 'product'}
    for n in SAT_O:
        cols[f'o_{n}'] = np.array([num(r[3].get(n)) for r in rows], np.float32)
    cols['q_n_pix'] = np.array([num(r[3].get('n_pix')) for r in rows], np.float32)
    cols['q_sat_zen'] = np.array([num(r[3].get('sat_zen')) for r in rows], np.float32)
    when = [_iso_ts(r[3].get('t')) for r in rows]
    cols['q_dt_mask'] = np.array([np.nan if w is None else w - int(t_valid) for w in when], np.float32)
    return table(obs_schema('sat'), cols, len(rows))

def slim_at_points(z, done: dict | None, ids: list, lat, lon, src: str='ops') -> dict:
    g = json.loads(_s(z['grid']))
    lat, lon = (np.asarray(lat, np.float64), np.asarray(lon, np.float64))
    c = z['codes']
    x = np.round((lon - g['west']) / g['step']).astype(int)
    y = np.round((g['north'] - lat) / g['step']).astype(int)
    ok = (x >= 0) & (y >= 0) & (y < c.shape[2]) & (x < c.shape[3])
    codes = np.ascontiguousarray(np.asarray(c[:, :, y[ok], x[ok]], np.uint8).transpose(0, 2, 1))
    model, run = (_s(z['model']), _s(z['run']))
    stamps = [str(s) for s in np.asarray(z['stamps'])]
    enc = z['enc']
    enc = json.loads(_s(enc)) if not isinstance(enc, dict) else enc
    avail = _iso_ts((done or {}).get('published_t')) if (done or {}).get('published') else None
    return {'model': model, 'run': run, 'build': str((done or {}).get('build') or ''), 'src': src, 'replay': False, 'missing_w': False, 'missing_blh_heat': False, 'review': False, 'avail_t': avail, 'stamps': stamps, 'runs': [run] * len(stamps), 'leads': [(_ts(s) - _ts(run)) // 3600 for s in stamps], 'obs_ids': [str(i) for i, k in zip(ids, ok) if k], 'codes': codes, 'fields': [str(f) for f in np.asarray(z['fields'])], 'enc': enc, 'stations_v': None, 'elev_st': np.full(int(ok.sum()), np.nan), 'lat': lat[ok], 'lon': lon[ok]}

def slim_ground(z):
    g = json.loads(_s(z['grid']))
    enc = z['enc']
    enc = json.loads(_s(enc)) if not isinstance(enc, dict) else enc
    e = enc.get('elevation')
    codes = np.asarray(z['elev'], np.uint8)
    if not e:
        return lambda la, lo: np.full(np.shape(la), np.nan)
    lo_, hi_ = (float(e[0]), float(e[1]))
    vals = np.where(codes == 255, np.nan, lo_ + codes.astype(np.float64) / 254 * (hi_ - lo_))

    def at(la, lo):
        la, lo = (np.asarray(la, np.float64), np.asarray(lo, np.float64))
        x = np.round((lo - g['west']) / g['step']).astype(int)
        y = np.round((g['north'] - la) / g['step']).astype(int)
        ok = (x >= 0) & (y >= 0) & (y < vals.shape[0]) & (x < vals.shape[1])
        out = np.full(la.shape, np.nan)
        out[ok] = vals[y[ok], x[ok]]
        return out
    return at

def record_at_points(grid: dict, ids: list, lat, lon) -> dict:
    lat, lon = (np.asarray(lat, np.float64), np.asarray(lon, np.float64))
    c = np.asarray(grid['codes'], np.uint8)
    x = np.round((lon - grid['west']) / grid['step']).astype(int)
    y = np.round((grid['north'] - lat) / grid['step']).astype(int)
    ok = (x >= 0) & (y >= 0) & (y < c.shape[1]) & (x < c.shape[2])
    rec = {k: v for k, v in grid.items() if k not in ('codes', 'west', 'north', 'step')}
    rec.update(obs_ids=[str(i) for i, k in zip(ids, ok) if k], codes=c[:, y[ok], x[ok]], lat=lat[ok], lon=lon[ok], elev_st=np.full(int(ok.sum()), np.nan))
    for k, v in (('replay', False), ('missing_w', False), ('missing_blh_heat', False), ('review', False)):
        rec.setdefault(k, v)
    rec.setdefault('runs', [rec['run']] * len(rec['stamps']))
    return rec

def raob_obs(rows: list) -> dict:
    names = raob_names()
    cols = {'kind': 'raob', 'obs_id': [r['obs_id'] for r in rows], 't_valid': np.array([r['t_valid'] for r in rows], np.int64), 'lat': np.array([r['lat'] for r in rows], np.float32), 'lon': np.array([r['lon'] for r in rows], np.float32), 'elev': np.array([r.get('elev', np.nan) for r in rows], np.float32), 't_release': [r.get('t_release') for r in rows], 'n_lev': np.array([r.get('n_lev', -1) for r in rows], np.int32), 'top_hpa': np.array([r.get('top_hpa', np.nan) for r in rows], np.float32)}
    num = lambda x: np.nan if x is None else float(x)
    for n in names:
        cols[f'o_{n}'] = np.array([num(r.get(f'o_{n}')) for r in rows], np.float32)
        cols[f'q_{n}'] = np.array([int(r.get(f'q_{n}') or 0) for r in rows], np.int32)
    return table(obs_schema('raob'), cols, len(rows))

def km(lat1, lon1, lat2, lon2):
    p1, p2 = (np.radians(lat1), np.radians(lat2))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))

def raob_pairs(rec: dict, obs: dict) -> dict:
    sch = pairs_schema('raob')
    if not n_rows(obs) or not len(rec['obs_ids']):
        return empty(sch)
    hour_of = {_ts(s): k for k, s in enumerate(rec['stamps']) if re.fullmatch('\\d{10}', rec['runs'][k] or '') and rec['leads'][k] >= 0}
    out = {n: [] for n, _ in sch}
    rows_c = []
    for i in range(n_rows(obs)):
        k = hour_of.get(int(obs['t_valid'][i]))
        if k is None:
            continue
        d = km(float(obs['lat'][i]), float(obs['lon'][i]), rec['lat'], rec['lon'])
        j = int(np.argmin(d))
        if d[j] > RAOB_NEAR_KM:
            continue
        rows_c.append((i, k, j, float(d[j])))
    if not rows_c:
        return empty(sch)
    ii, kk, jj, dd = (np.array(x) for x in zip(*rows_c))
    sel = rec['codes'][kk, jj]
    cols = {'kind': 'raob', 'obs_id': obs['obs_id'][ii], 't_valid': obs['t_valid'][ii], 'model': rec['model'], 'run': np.array([rec['runs'][k] for k in kk], dtype=object), 'build': rec['build'], 'lead_h': np.array([rec['leads'][k] for k in kk], np.int16), 'avail_t': _avail_of(rec, list(kk), range(len(kk))), 'src': rec['src'], 'replay': rec['replay'], 'missing_w': rec['missing_w'], 'missing_blh_heat': rec['missing_blh_heat'], 'review': rec['review'], 'station': np.array(rec['obs_ids'], dtype=object)[jj], 'dist_km': dd.astype(np.float32)}
    for v in RAOB_VARS:
        cols[f'f_{v}'] = decoded(rec, v, sel[:, None, :])[:, 0].astype(_ftype(v)) if v in rec['fields'] else None
    cols['cu_call'] = cu_call(rec, sel)
    return table(sch, cols, len(ii))

def cu_call(rec: dict, sel: np.ndarray) -> np.ndarray:
    f = rec['fields']
    if 'cubase' not in f:
        return np.full(len(sel), -1, np.int8)
    cu = (sel[:, f.index('cubase')] != 255).astype(np.int8)
    hour = sel[:, f.index('wstar')] != 255 if 'wstar' in f else np.ones(len(sel), bool)
    return np.where(hour, cu, -1).astype(np.int8)

def raob_observed(obs_row: dict, cu_call: int) -> dict:
    import obs_raob as R
    g = lambda n: obs_row.get(f'o_{n}')
    q = int(obs_row.get('q_depth') or 0)
    if cu_call not in (0, 1):
        return {'zi': None, 'depth': None}
    if cu_call == 1:
        depth = g('lcl') if q & (R.Q_REACH | R.Q_NEARCU) else g('depth')
        return {'zi': g('zi'), 'depth': depth}
    depth = g('zi_blue') if q & R.Q_REACH else g('zi') if q & R.Q_NEARCU else g('depth')
    return {'zi': g('zi_blue'), 'depth': depth}

def _num(x):
    try:
        return np.nan if x is None else float(x)
    except (TypeError, ValueError):
        return np.nan

def _mid(t0: str | None, t1: str | None) -> int | None:
    a, b = (_iso_ts(t0), _iso_ts(t1))
    return None if a is None or b is None else (a + b) // 2

def _leg_events(items: list, name) -> list:
    out = []
    for f in items:
        if f.get('kind', 'flight') == 'collector':
            continue
        for i, L in enumerate(f.get('legs') or []):
            tv = _mid(L.get('tstart'), L.get('tend'))
            if tv is None:
                continue
            a, b = (L.get('a') or [None, None], L.get('b') or [None, None])
            lon = (_num(a[0]) + _num(b[0])) / 2
            lat = (_num(a[1]) + _num(b[1])) / 2
            out.append((str(f.get('id')), i, name(f"leg|{f.get('id')}|{i}"), tv, lat, lon, common.macro_of(f.get('region')), L))
    return out

def _climb_events(items: list, name, collector: bool) -> list:
    out = []
    for f in items:
        if (f.get('kind', 'flight') == 'collector') != collector:
            continue
        for i, c in enumerate(f.get('climbs') or []):
            t0 = _iso_ts(c.get('t0'))
            if t0 is None:
                continue
            tv = t0 + int(_num(c.get('dur')) / 2 if math.isfinite(_num(c.get('dur'))) else 0)
            tag = 'ogn' if collector else 'climb'
            out.append((str(f.get('id')), i, name(f"{tag}|{f.get('id')}|{i}"), tv, _num(c.get('lat')), _num(c.get('lon')), common.macro_of(f.get('region')), c, f))
    return out
NOT_FLOWN = ('collector', 'asos', 'raob')

def _probe_events(items: list, name, times=None) -> list:
    if times is None:
        import day
        times = day.probe_times
    out = []
    for f in items:
        if f.get('kind', 'flight') in NOT_FLOWN:
            continue
        for p in times(f) or []:
            t = int(p['t'])
            out.append((str(f.get('id')), str(t), name(f"probe|{f.get('id')}|{t}"), t, _num(p.get('lat')), _num(p.get('lon')), common.macro_of(f.get('region'))))
    return out

def flight_obs(items: list, kind: str, name, times=None) -> dict:
    sch = obs_schema(kind)
    if kind == 'flight_leg':
        ev = _leg_events(items, name)
        cols = {'o_' + n: np.array([_num(e[7].get(n)) for e in ev]) for n in LEG_O}
        cols.update({'q_ok': [1 if e[7].get('ok') else 0 for e in ev], 'q_engine': [1 if e[7].get('engine') else 0 for e in ev], 'q_last': [1 if e[1] == len(next((f for f in items if str(f.get('id')) == e[0]))['legs']) - 1 else 0 for e in ev]})
    elif kind in ('flight_climb', 'ogn_climb'):
        ev = _climb_events(items, name, kind == 'ogn_climb')
        names = CLIMB_O if kind == 'flight_climb' else OGN_O
        cols = {'o_' + n: np.array([_num(e[7].get('z1') if n == 'z1' else e[7].get(n)) for e in ev]) for n in names}
        if kind == 'flight_climb':
            cols['q_engon'] = [1 if e[7].get('engon') else 0 for e in ev]
        else:
            cols['q_type'] = [OGN_TYPE.get(e[8].get('y'), 0) for e in ev]
    elif kind == 'flight_probe':
        ev = _probe_events(items, name, times)
        cols = {}
    else:
        raise ValueError('pairs: kind')
    if not ev:
        return empty(sch)
    cols.update({'kind': kind, 'obs_id': [e[2] for e in ev], 't_valid': np.array([e[3] for e in ev], np.int64), 'lat': np.array([e[4] for e in ev], np.float32), 'lon': np.array([e[5] for e in ev], np.float32), 'macro': [e[6] for e in ev], 'review': kind in REVIEW_KINDS})
    return table(sch, cols, len(ev))

def _at(s, i):
    return s[i] if isinstance(s, list) and i < len(s) and isinstance(s[i], dict) else None

def flight_pairs(items: list, unit: dict, kind: str, name, times=None, build_min=None) -> dict:
    sch = pairs_schema(kind)
    meta, score = (unit.get('meta') or {}, unit.get('score') or {})
    run = str(meta.get('init') or '')
    if not re.fullmatch('\\d{10}', run):
        return empty(sch)
    model = str((meta.get('id') or {}).get('model') or common.base_model(score.get('model_id') or ''))
    by_id = {str(e.get('id')): e for e in score.get('items') or []}
    keys = kind_vars(kind)
    if kind == 'flight_leg':
        ev = _leg_events(items, name)
        pick = lambda e, sfx: (_at((by_id.get(e[0]) or {}).get('legs' + sfx), e[1]) or {}).get('own')
    elif kind == 'flight_probe':
        ev = _probe_events(items, name, times)
        pick = lambda e, sfx: ((by_id.get(e[0]) or {}).get('probes' + sfx) or {}).get(e[1])
    else:
        ev = _climb_events(items, name, kind == 'ogn_climb')
        pick = lambda e, sfx: _at((by_id.get(e[0]) or {}).get('climbs' + sfx), e[1])
    got = [(e, v, pick(e, '_c')) for e in ev for v in [pick(e, '')] if isinstance(v, dict) and v]
    if not got:
        return empty(sch)
    replay = meta.get('src') == 'replay'
    r0 = _ts(run)
    cols = {'kind': kind, 'obs_id': [e[2] for e, _, _ in got], 't_valid': np.array([e[3] for e, _, _ in got], np.int64), 'model': model, 'run': run, 'build': str(meta.get('cs') or ''), 'lead_h': np.array([int(round((e[3] - r0) / 3600)) for e, _, _ in got], np.int16), 'avail_t': rebuilt_avail(model, run, build_min), 'src': 'hind', 'replay': replay, 'missing_w': replay, 'missing_blh_heat': replay, 'review': kind in REVIEW_KINDS or (meta.get('id') or {}).get('kind') == 'collector'}
    for k in keys:
        cols[f'f_{k}'] = np.array([_num(v.get(k)) for _, v, _ in got]).astype(_ftype(k))
        cols[f'f_{k}_c'] = np.array([_num((c or {}).get(k)) for _, _, c in got]).astype(np.float16)
    return table(sch, cols, len(got))

def ogn_items(records: list, ground=None, drop=None) -> list:
    import collector
    keep = [r for r in records if str(r.get('s') or '').lower() in FEEDS]
    return collector.convert(keep, ground or (lambda lon, lat: None), drop)

def static_rows(rec: dict, elev_grid: np.ndarray | None=None, grid: dict | None=None) -> dict:
    k = next((i for i, r in enumerate(rec['runs']) if r), 0)
    em = decoded(rec, 'elevation', rec['codes'][k:k + 1])[0] if len(rec['stamps']) else np.zeros(0)
    rel = np.full(len(rec['obs_ids']), np.nan)
    if elev_grid is not None and grid is not None:
        _pipeline()
        import blend
        r = blend.relief(np.asarray(elev_grid, np.float64), grid['step'])
        x = np.round((rec['lon'] - grid['west']) / grid['step']).astype(int)
        y = np.round((grid['north'] - rec['lat']) / grid['step']).astype(int)
        ok = (x >= 0) & (y >= 0) & (y < r.shape[0]) & (x < r.shape[1])
        rel[ok] = r[y[ok], x[ok]]
    cols = {'obs_id': rec['obs_ids'], 'model': rec['model'], 'elev_obs': rec['elev_st'], 'elev_model': em, 'dz_terrain': rec['elev_st'] - em, 'relief5km': rel}
    return table(STATIC_SCHEMA, cols, len(rec['obs_ids']))
_TOKEN = re.compile('[a-z0-9]{1,16}')
_DAY = re.compile('\\d{4}-\\d{2}-\\d{2}')
_MONTH = re.compile('\\d{4}-\\d{2}')

def _check(kind, day=None, batch=None, month=None):
    if kind not in KINDS + ('snap',) or (day is not None and (not _DAY.fullmatch(day))) or (batch is not None and (not _TOKEN.fullmatch(batch))) or (month is not None and (not _MONTH.fullmatch(month))):
        raise ValueError("pairs: a key's part")
TABLES = {'pairs': ('derived/pairs', 'derived/units/pair'), 'obs': ('derived/obs', 'derived/units/obs'), 'snap': ('derived/snap', 'derived/units/snap')}

def part_key(what: str, kind: str, day: str, batch: str, n: int, pv: str=PV) -> str:
    _check(kind, day, batch)
    if what == 'snap':
        return f'derived/snap/{pv}/date={day}/part={batch}-{n}.parquet.lenc'
    return f'{TABLES[what][0]}/{pv}/kind={kind}/date={day}/part={batch}-{n}.parquet.lenc'

def marker_key(what: str, kind: str, day: str, batch: str, n: int=1, pv: str=PV) -> str:
    _check(kind, day, batch)
    return f'{TABLES[what][1]}/{pv}/{kind}.{day}.{batch}' + (f'.r{n}' if n > 1 else '') + '.json.lenc'

def compact_key(kind: str, month: str, n: int, pv: str=PV) -> str:
    _check(kind, month=month)
    return f'derived/pairs/{pv}/kind={kind}/month={month}/c{n}.parquet.lenc'

def compact_marker_key(kind: str, month: str, n: int, pv: str=PV) -> str:
    _check(kind, month=month)
    return f'derived/units/pair/{pv}/{kind}.{month}.c{n}.json.lenc'

def static_key(model: str, stations_v: str, pv: str=PV) -> str:
    if model not in MODELS or not re.fullmatch('[0-9a-f]{6,64}', stations_v or ''):
        raise ValueError("pairs: a key's part")
    return f'derived/static/{pv}/model={model}/{stations_v}.parquet.lenc'
_MARK = re.compile('(?P<kind>[a-z_]+)\\.(?P<day>\\d{4}-\\d{2}-\\d{2})\\.(?P<batch>[a-z0-9]+)(?:\\.r(?P<n>\\d+))?\\.json\\.lenc')
_CMARK = re.compile('(?P<kind>[a-z_]+)\\.(?P<month>\\d{4}-\\d{2})\\.c(?P<n>\\d+)\\.json\\.lenc')

def schema_of(what: str, kind: str) -> tuple:
    if what == 'snap':
        import snap
        return snap.SCHEMA
    return pairs_schema(kind) if what == 'pairs' else obs_schema(kind)

def union_key(what: str) -> tuple:
    return {'pairs': KEY, 'obs': ('obs_id', 't_valid'), 'snap': ('obs_id', 't_valid', 'slot')}[what]

def markers(lake, what: str, kind: str, pv: str=PV, day: str | None=None, batch: str | None=None) -> list:
    pre = f'{TABLES[what][1]}/{pv}/{kind}.' + (f'{day}.' if day else '')
    out = []
    for k in lake.list(pre):
        m = _MARK.fullmatch(k.rsplit('/', 1)[-1])
        if m and m['kind'] == kind and (batch is None or m['batch'] == batch):
            out.append((m['day'], m['batch'], int(m['n'] or 1), k))
    return sorted(out)

def _id_of(inp) -> str:
    return inp if isinstance(inp, str) else f'{inp[0]}@{inp[1]}'

def write_unit(lake, what: str, kind: str, day: str, batch: str, inputs: list, build, pv: str=PV) -> dict:
    ids = sorted({_id_of(x) for x in inputs})
    have = markers(lake, what, kind, pv, day, batch)
    seen = set()
    for *_, k in have:
        seen.update(json.loads(lake.get(k)).get('inputs') or [])
    new = [i for i in ids if i not in seen]
    if not new:
        return {'status': 'seen', 'part': None, 'marker': None, 'rows': 0, 'new': 0}
    t = build(new, ids)
    sch = schema_of(what, kind)
    n = max([x[2] for x in have], default=0) + 1
    nr = n_rows(t)
    part = sha = None
    if nr:
        blob = to_parquet(sch, t)
        for n in range(n, n + 99):
            part = part_key(what, kind, day, batch, n, pv)
            res = lake.put(part, blob, strict=False)
            if res.status != 'differs':
                sha = res.sha256
                break
        else:
            raise RuntimeError('pairs: no free slot for a part')
    mk = marker_key(what, kind, day, batch, n, pv)
    body = {'v': 1, 'what': what, 'kind': kind, 'day': day, 'batch': batch, 'pv': pv, 'slot': n, 'inputs': new, 'seen': sorted(seen & set(ids)), 'part': part, 'sha256': sha, 'rows': nr}
    lake.put(mk, json.dumps(body, sort_keys=True, separators=(',', ':')).encode())
    return {'status': 'written' if nr else 'empty', 'part': part, 'marker': mk, 'rows': nr, 'new': len(new)}

def _union(sch, tables: list, key: tuple) -> dict:
    tables = [t for t in tables if t and n_rows(t)]
    t = concat(sch, tables)
    if not n_rows(t):
        return t
    last = {}
    for i, x in enumerate(zip(*(t[c] for c in key))):
        last[x] = i
    if len(last) == n_rows(t):
        return t
    keep = np.array(sorted(last.values()), np.int64)
    return {c: v[keep] for c, v in t.items()}

def compactions(lake, kind: str, pv: str=PV) -> dict:
    out = {}
    for k in lake.list(f'derived/units/pair/{pv}/{kind}.'):
        m = _CMARK.fullmatch(k.rsplit('/', 1)[-1])
        if m and m['kind'] == kind:
            n = int(m['n'])
            if m['month'] not in out or n > out[m['month']][0]:
                out[m['month']] = (n, k, None)
    for month, (n, k, _) in list(out.items()):
        out[month] = (n, k, json.loads(lake.get(k)))
    return out
PART_DATE = ('part_date', 'str')

def _read(lake, what: str, kind: str, days: list, pv: str, need) -> dict:
    sch = schema_of(what, kind)
    sch2 = sch + (PART_DATE,)
    key = union_key(what)
    need2 = None if need is None else list(need) + [PART_DATE[0]]
    days = sorted(set(days))
    want = set(days)
    comp = compactions(lake, kind, pv) if what == 'pairs' else {}
    parts, covered = ([], set())
    for month in sorted({d[:7] for d in days}):
        if month in comp:
            body = comp[month][2]
            covered.update(body.get('markers') or [])
            if body.get('part'):
                t = from_parquet(sch2, lake.get(body['part']), need2)
                sel = np.array([x in want for x in t[PART_DATE[0]]], bool)
                parts.append({c: v[sel] for c, v in t.items()})
    for d in days:
        for _, _, _, mk in markers(lake, what, kind, pv, d):
            if mk in covered:
                continue
            body = json.loads(lake.get(mk))
            if body.get('part'):
                t = from_parquet(sch, lake.get(body['part']), need)
                t[PART_DATE[0]] = np.array([d] * n_rows(t), dtype=object)
                parts.append(t)
    return _union(sch2, parts, key)

def read_kind(lake, what: str, kind: str, days, pv: str=PV, purpose: str='train', columns=None, env=None) -> dict:
    if purpose not in ('train', 'review'):
        raise ValueError('pairs: purpose')
    need = None if columns is None else sorted(set(columns) | set(union_key(what)) | {'review'})
    t = _read(lake, what, kind, days, pv, need)
    t.pop(PART_DATE[0], None)
    if purpose == 'train':
        t = train_rows(t, env)
    if columns is not None:
        t = {c: t[c] for c in columns if c in t}
    return t

def day_complete(kind: str, day: str, today: date) -> bool:
    lag = LAG_D[kind]
    return lag is not None and date.fromisoformat(day) + timedelta(days=lag) <= today

def month_closed(kind: str, month: str, today: date) -> bool:
    y, m = (int(month[:4]), int(month[5:7]))
    last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).isoformat()
    return day_complete(kind, last, today)

def compact(lake, kind: str, month: str, today: date, pv: str=PV) -> dict:
    _check(kind, month=month)
    if not month_closed(kind, month, today):
        return {'status': 'open', 'rows': 0}
    sch2 = pairs_schema(kind) + (PART_DATE,)
    mks = [k for d, _, _, k in markers(lake, 'pairs', kind, pv) if d[:7] == month]
    comp = compactions(lake, kind, pv).get(month)
    if comp and set(mks) <= set(comp[2].get('markers') or []):
        return {'status': 'same', 'rows': comp[2].get('rows', 0)}
    days = sorted({d for d, _, _, _ in markers(lake, 'pairs', kind, pv) if d[:7] == month})
    t = _read(lake, 'pairs', kind, days, pv, None)
    blob = to_parquet(sch2, t)
    first = (comp[0] if comp else 0) + 1
    for n in range(first, first + 99):
        ck = compact_key(kind, month, n, pv)
        if lake.put(ck, blob, strict=False).status != 'differs':
            break
    else:
        raise RuntimeError('pairs: no free slot for a compaction')
    lake.put(compact_marker_key(kind, month, n, pv), json.dumps({'v': 1, 'kind': kind, 'month': month, 'markers': sorted(mks), 'part': ck, 'rows': n_rows(t)}, sort_keys=True, separators=(',', ':')).encode())
    return {'status': 'written', 'rows': n_rows(t)}

def _by_id(inputs: dict) -> dict:
    return {_id_of(k): v for k, v in inputs.items()}

def station_unit(lake, kind: str, day: str, batch: str, records: dict, pv: str=PV) -> dict:
    if kind not in ('metar', 'sat'):
        raise ValueError('pairs: kind')
    sch = pairs_schema(kind)
    by = _by_id(records)
    return write_unit(lake, 'pairs', kind, day, batch, list(records), lambda new, _all: concat(sch, [station_pairs(by[i], kind, day) for i in new]), pv)

def sat_unit(lake, day: str, batch: str, records: dict, observed: dict, pv: str=PV) -> dict:
    sch = pairs_schema('sat')
    recs, obs = (_by_id(records), _by_id(observed))

    def keys_of(t) -> set:
        if isinstance(t, (set, frozenset)):
            return set(t)
        if not n_rows(t):
            return set()
        ok = np.array([str(x) == 'ok' for x in t['status']], bool) if 'status' in t else np.ones(n_rows(t), bool)
        return {(str(a), int(b)) for a, b, k in zip(t['obs_id'], t['t_valid'], ok) if k}
    seen = {oi: keys_of(t) for oi, t in obs.items()}

    def build(new, _all):
        new = set(new)
        out = []
        for ri, rec in recs.items():
            want = set().union(*(k for oi, k in seen.items() if ri in new or oi in new)) if seen else set()
            if not want:
                continue
            t = station_pairs(rec, 'sat', day)
            if not n_rows(t):
                continue
            keep = np.array([(str(a), int(b)) in want for a, b in zip(t['obs_id'], t['t_valid'])], bool)
            if keep.any():
                out.append({c: v[keep] for c, v in t.items()})
        return concat(sch, out)
    return write_unit(lake, 'pairs', 'sat', day, batch, list(records) + list(observed), build, pv)

def latest_soundings(soundings: dict) -> dict:
    owner = {}
    for si, obs in soundings.items():
        if not n_rows(obs):
            continue
        for j, (a, b) in enumerate(zip(obs['obs_id'], obs['t_valid'])):
            owner[str(a), int(b)] = (si, j)
    out = {}
    for si, obs in soundings.items():
        if not n_rows(obs):
            continue
        keep = np.array([owner[str(a), int(b)] == (si, j) for j, (a, b) in enumerate(zip(obs['obs_id'], obs['t_valid']))], bool)
        if keep.any():
            out[si] = {c: v[keep] for c, v in obs.items()}
    return out

def raob_unit(lake, day: str, batch: str, records: dict, soundings: dict, pv: str=PV) -> dict:
    sch = pairs_schema('raob')
    recs, snds = (_by_id(records), _by_id(soundings))

    def build(new, _all):
        new = set(new)
        out = []
        for ri, rec in recs.items():
            for si, obs in latest_soundings(snds).items():
                if ri in new or si in new:
                    out.append(raob_pairs(rec, obs))
        return concat(sch, out)
    return write_unit(lake, 'pairs', 'raob', day, batch, list(records) + list(soundings), build, pv)

def summary(res: list) -> str:
    by = {}
    for r in res:
        by[r['status']] = by.get(r['status'], 0) + 1
    return 'pairs: ' + ', '.join((f'{k} {v}' for k, v in sorted(by.items()))) + f"; {sum((r['rows'] for r in res))} rows"

def name_of(keys):
    return lambda text: keys.name('obs|' + text)

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
