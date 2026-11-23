from __future__ import annotations
import math
import os
import sys
import warnings
from datetime import datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pairs as P
UTC = timezone.utc
SLOT_H = 6
HORIZON_H = 120
MEAN_VARS = ('blapse', 'cape', 'zi', 'depth', 'wstar')
COND_FIELDS = ('t2', 'td2', 'u_sfc', 'v_sfc', 'cubase', 'cucover', 'cuclass', 'wstar', 'zi', 'depth', 'elevation', 'blapse', 'cape', 'u_blavg', 'v_blavg')
KEY = ('obs_id', 't_valid', 'slot')

def _blend():
    P._pipeline()
    import blend
    return blend

def _build():
    P._pipeline()
    import build
    return build

def slots_for(t_valid: int) -> list:
    step = SLOT_H * 3600
    last = int(t_valid) // step * step
    first = int(t_valid) - HORIZON_H * 3600
    return [s for s in range(last, first - 1, -step) if s >= first][::-1]

def ops_published(cands: list) -> dict:
    out = {}
    for c in cands:
        if c.get('src') == 'ops' and c.get('avail_t') is not None:
            out[c['run']] = min(out.get(c['run'], int(c['avail_t'])), int(c['avail_t']))
    return out

def on_hand(c: dict, ops: dict):
    if c.get('avail_t') is None:
        return None
    if c.get('src') != 'ops' and c['run'] in ops:
        return ops[c['run']]
    return int(c['avail_t'])

def choose(cands: list, slot: int, ops: dict | None=None):
    ops = ops_published(cands) if ops is None else ops
    ok = [(c, t) for c in cands for t in [on_hand(c, ops)] if t is not None and t <= slot]
    if not ok:
        return None
    return max(ok, key=lambda x: (x[0]['run'], x[1], x[0].get('src') == 'ops', str(x[0].get('build') or '')))[0]

def schema(models=P.MODELS) -> tuple:
    s = [('obs_id', 'str'), ('t_valid', 'i64'), ('slot', 'i64'), ('lat', 'f32'), ('lon', 'f32'), ('lead_h', 'i16'), ('run_age_h', 'f32'), ('n_models', 'i8'), ('regime', 'i8'), ('cuclass', 'f32'), ('ws', 'f32'), ('wd', 'f32'), ('ws_bl', 'f32'), ('wd_bl', 'f32')]
    s += [(v, 'f32') for v in MEAN_VARS]
    s += [(f'spread_{f}', 'f32') for f in ('t2', 'td2', 'wind', 'cubase', 'cu')]
    s += [('solar_h', 'f32'), ('sun_elev', 'f32'), ('doy_sin', 'f32'), ('doy_cos', 'f32'), ('season', 'i8'), ('review', 'bool')]
    for m in models:
        s += [(f'run_{m}', 'str'), (f'build_{m}', 'str'), (f'src_{m}', 'str'), (f'lead_{m}', 'i16'), (f'age_{m}', 'f32'), (f'regime_{m}', 'i8')]
    return tuple(s)
SCHEMA = schema()

def enc_ok(rec: dict) -> bool:
    enc = _build().ENC
    return all((f not in rec['enc'] or list(rec['enc'][f]) == list(enc[f]) for f in COND_FIELDS if f in enc))

def _hour_index(rec: dict) -> dict:
    if '_hours' not in rec:
        rec['_hours'] = {P._ts(s): k for k, s in enumerate(rec['stamps']) if rec['runs'][k] == rec['run'] and rec['leads'][k] >= 0}
    return rec['_hours']

def _col_index(rec: dict, ids: dict) -> np.ndarray:
    return np.array([ids.get(o, -1) for o in rec['obs_ids']], np.int64)

def season_of(t: datetime, lat) -> tuple:
    doy = t.timetuple().tm_yday - 1 + t.hour / 24
    south = np.asarray(lat) < 0
    ph = 2 * math.pi * (doy + np.where(south, 365.25 / 2, 0.0)) / 365.25
    month = np.where(south, (t.month + 5) % 12 + 1, t.month)
    season = (month % 12 // 3).astype(np.int8)
    return (np.sin(ph), np.cos(ph), season)

def _wind(u, v):
    return (np.hypot(u, v), (np.degrees(np.arctan2(-u, -v)) + 360) % 360)

def _dt(t: int) -> datetime:
    return datetime.fromtimestamp(int(t), UTC)

def rows_at(records: dict, t_valid: int, slot: int, ids: list, lat, lon, counts: dict | None=None) -> dict:
    B = _blend()
    n = len(ids)
    pos = {o: i for i, o in enumerate(ids)}
    chosen = {}
    for m, recs in records.items():
        c = choose([r for r in recs if t_valid in _hour_index(r)], slot, ops_published(recs))
        if c is not None and 'wstar' in c['fields']:
            chosen[m] = c
    cols = {'obs_id': list(ids), 't_valid': np.full(n, t_valid, np.int64), 'slot': np.full(n, slot, np.int64), 'lat': lat, 'lon': lon, 'lead_h': np.full(n, (t_valid - slot) // 3600, np.int16)}
    codes, vals, ages, members = ({}, {}, [], {})
    review = np.zeros(n, bool)
    for m, rec in chosen.items():
        k = _hour_index(rec)[t_valid]
        ci = _col_index(rec, pos)
        ok = ci >= 0
        full = np.full((n, len(rec['fields'])), 255, np.uint8)
        full[ci[ok]] = rec['codes'][k][ok]
        codes[m] = {f: full[:, j] for j, f in enumerate(rec['fields'])}
        members[m] = {'run': _dt(P._ts(rec['run'])), 'wcodes': {f: a[None, :] for f, a in codes[m].items()}}
        vals[m] = {v: P.decoded(rec, v, full[:, None, :])[:, 0] for v in COND_FIELDS}
        r = np.asarray(B.regime_of(codes[m]))
        age = (slot - P._ts(rec['run'])) / 3600
        has = r >= 0
        ages.append(np.where(has, age, np.nan))
        if rec.get('review'):
            review |= has
        cols[f'run_{m}'] = np.where(has, rec['run'], '').astype(object)
        cols[f'build_{m}'] = np.where(has, str(rec.get('build') or ''), '').astype(object)
        cols[f'src_{m}'] = np.where(has, rec['src'], '').astype(object)
        cols[f'lead_{m}'] = np.where(has, (t_valid - P._ts(rec['run'])) // 3600, -1).astype(np.int16)
        cols[f'age_{m}'] = np.where(has, age, np.nan)
        cols[f'regime_{m}'] = r.astype(np.int8)
    if n:
        cond = B.conditions(members, _dt(t_valid), _dt(slot), (0.0, 0.0, float(n - 1), 0.0, 1.0), {'members': list(chosen), 'max_age_h': {}})
        cols['regime'] = np.asarray(cond['regime'])[0]
        cols['n_models'] = np.asarray(cond['n_models'])[0]
        for f in B.FIELDS:
            cols[f'spread_{f}'] = np.asarray(cond[f'spread_{f}'])[0]
    if chosen:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cols['run_age_h'] = np.nanmean(np.stack(ages), 0)
            pres = [np.where(cols[f'regime_{m}'] >= 0, 1.0, np.nan) for m in chosen]
            mean = lambda v: np.nanmean(np.stack([vals[m][v] * p for m, p in zip(chosen, pres)]), 0)
            for v in MEAN_VARS:
                cols[v] = mean(v)
            cols['ws'], cols['wd'] = _wind(mean('u_sfc'), mean('v_sfc'))
            cols['ws_bl'], cols['wd_bl'] = _wind(mean('u_blavg'), mean('v_blavg'))
            cls = np.stack([vals[m]['cuclass'] * p for m, p in zip(chosen, pres)])
            cols['cuclass'] = _mode(cls)
    valid = _dt(t_valid)
    cols['solar_h'] = ((valid.hour + np.asarray(lon, np.float64) / 15) % 24 + 24) % 24
    cols['sun_elev'] = _build().sun_elevation(valid.replace(tzinfo=None), np.asarray(lat, np.float64), np.asarray(lon, np.float64))
    cols['doy_sin'], cols['doy_cos'], cols['season'] = season_of(valid, lat)
    cols['review'] = review
    if counts is not None:
        counts['chosen'] = counts.get('chosen', 0) + len(chosen)
    return P.table(SCHEMA, cols, n)

def _mode(cls: np.ndarray) -> np.ndarray:
    out = np.full(cls.shape[1], np.nan)
    for j in range(cls.shape[1]):
        c = cls[:, j]
        c = c[np.isfinite(c)]
        if c.size:
            u, k = np.unique(c, return_counts=True)
            out[j] = u[np.argmax(k)]
    return out

def day_rows(records: dict, day: str, ids: list | None=None, counts: dict | None=None) -> dict:
    counts = counts if counts is not None else {}
    recs = {}
    for m, rs in records.items():
        keep = []
        for r in rs:
            if enc_ok(r):
                keep.append(r)
            else:
                counts['enc_refused'] = counts.get('enc_refused', 0) + 1
        if keep:
            recs[m] = keep
    where = {}
    for rs in recs.values():
        for r in rs:
            for o, la, lo in zip(r['obs_ids'], r['lat'], r['lon']):
                where.setdefault(o, (float(la), float(lo)))
    ids = sorted(where) if ids is None else [o for o in ids if o in where]
    lat = np.array([where[o][0] for o in ids], np.float64)
    lon = np.array([where[o][1] for o in ids], np.float64)
    d0 = P._ts(day.replace('-', '') + '00')
    hours = sorted({t for rs in recs.values() for r in rs for t in _hour_index(r) if d0 <= t < d0 + 86400})
    out = []
    for t in hours:
        for s in slots_for(t):
            out.append(rows_at(recs, t, s, ids, lat, lon, counts))
    return P.concat(SCHEMA, out)

def write_day(lake, day: str, batch: str, inputs: dict, ids: list | None=None, pv: str=P.PV) -> dict:
    ins = P._by_id(inputs)

    def build(_new, _all):
        by = {}
        for i in _all:
            m, r = ins[i]
            by.setdefault(m, []).append(r)
        return day_rows(by, day, ids)
    return P.write_unit(lake, 'snap', 'snap', day, batch, list(inputs), build, pv)
