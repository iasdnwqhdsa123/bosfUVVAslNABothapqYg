from __future__ import annotations
import glob
import math
import os
import statistics as st
from datetime import datetime
from zoneinfo import ZoneInfo
import common
import track
MODEL_KEYS = ('xc', 'gs', 'climb', 'ach', 'usable', 'ridge', 'ws', 'ws0', 'zi', 'top', 'band', 'shade', 'sun', 'sunElev', 'relief', 'elev', 'cb', 'cu', 'circ', 'wind', 'rain', 'od', 'cloud', 'cloudlow', 'blapse', 'twidth', 'sink', 'ridgev', 'densAlt', 'solar', 'localHour')
ROUTE_KEYS = ('len', 'autoMean', 'autoMax', 'straightMean', 'straightMax', 'sameShare', 'sameFlownKm', 'sameAutoKm', 'flownBend', 'autoBend', 'bendMatch', 'autoBendsAlone', 'bent', 'autoH', 'straightH', 'flownH', 'error')
FLIGHT_KEYS = ('date', 'region', 'state', 'cls', 'grp', 'key', 'key_exact', 'src', 'ac')

def load_days(data: str) -> list[dict]:
    out = []
    for fn in sorted(glob.glob(os.path.join(data, 'days', '*.json.gz'))):
        out.append(common.read_json(fn))
    return out

def load_pilots(data: str) -> dict:
    p = os.path.join(data, 'state.json')
    return common.read_json(p).get('pilots') or {} if os.path.exists(p) else {}

def model_of(d: dict, model: str | None) -> str:
    return model or d.get('model') or (d.get('models') or ['hrrr'])[0]

def mk(f: dict, model: str, corrected: bool=False) -> tuple[dict, dict]:
    if corrected:
        return (f.get(f'model_c_{model}') or {}, (f.get('probe_mc') or {}).get(model) or {})
    return (f.get(f'model_{model}') or {}, (f.get('probe_m') or {}).get(model) or {})

def mleg(x: dict, model: str, corrected: bool=False) -> dict:
    return (x.get('models_c' if corrected else 'models') or {}).get(model) or {}

def _base(f: dict, pilots: dict | None) -> dict:
    base = {k: f.get(k) for k in FLIGHT_KEYS}
    base['fid'] = f['id']
    base['macro'] = common.macro_of(f.get('region'))
    base['pilot'] = f.get('pilot') or (pilots or {}).get(str(f['id']))
    return base

def tables(days: list[dict], model: str | None=None, srcs: set | None=None, pilots: dict | None=None, corrected: bool=False) -> tuple[list, list, list]:
    legs, climbs, flights = ([], [], [])
    for d in days:
        m_ = model_of(d, model)
        for f in d['flights']:
            if srcs and f.get('src') not in srcs:
                continue
            base = {**_base(f, pilots), 'model': m_}
            T = mk(f, m_, corrected)[0].get('takeoffNode')
            flights.append({**base, 'free_km': f.get('free_km'), 'list_km': f.get('list_km'), 'dur_h': f.get('dur_h'), 'pfd': (T or {}).get('pfdCut'), 'paired': T is not None, 'nlegs': len(f['legs']), 'nclimbs': len(f['climbs'])})
            for L in f['legs']:
                m = mleg(L, m_, corrected)
                own, rt = (m.get('own') or {}, m.get('route') or {})
                row = {**base, 'leg': L.get('leg'), 'ok': L.get('ok'), 'cat': L.get('cat'), 'dist': L.get('dist'), 't': L.get('t'), 'ca': L.get('ca'), 'dh': L.get('dh'), 'therm': L.get('therm'), 'vg': L.get('vg'), 'vair': L.get('vair'), 'agl': L.get('agl'), 'alt': L.get('alt'), 'speed': L.get('speed'), 'wind_wg': L.get('wind'), 'tcount': L.get('tcount'), 'line_src': L.get('line_src'), 'engine': L.get('engine'), 'paired': bool(own), 'last': L is f['legs'][-1]}
                row.update({'m_' + k: own.get(k) for k in MODEL_KEYS})
                row.update({'r_' + k: rt.get(k) for k in ROUTE_KEYS})
                legs.append(row)
            for c in f['climbs']:
                own = mleg(c, m_, corrected)
                row = {**base, 'rate': c.get('rate'), 'gain': c.get('gain'), 'dur': c.get('dur'), 'agl1': c.get('agl1'), 'z1': c.get('z1'), 'r100': c.get('r100'), 'engon': c.get('engon'), 'leg': c.get('leg'), 'paired': bool(own)}
                row.update({'m_' + k: own.get(k) for k in MODEL_KEYS})
                climbs.append(row)
    return (legs, climbs, flights)

def flat(row: dict) -> bool:
    return (row.get('m_relief') or 0) < common.RELIEF_FLAT

def survival(days: list[dict], model: str | None=None, srcs: set | None=None, pilots: dict | None=None, corrected: bool=False) -> tuple[list, list]:
    hours, glides = ([], [])
    for d in days:
        old = (d.get('version') or 1) < 2
        m_ = model_of(d, model)
        for f in d['flights']:
            if srcs and f.get('src') not in srcs:
                continue
            base = {**_base(f, pilots), 'model': m_}
            pm = mk(f, m_, corrected)[1]
            try:
                hs, gs = (track.hour_samples(f), track.glides(f))
            except Exception:
                continue
            for kind, rs, out in (('hour', hs, hours), ('glide', gs, glides)):
                for x in rs:
                    m = pm.get(str(x['t'])) or {}
                    row = {**base, **{k: v for k, v in x.items() if k not in ('lon', 'lat', 't', 't0')}, 'old_day': old, 'paired': bool(m)}
                    row.update({'m_' + k: m.get(k) for k in track.PROBE_KEYS})
                    row['held'] = row['m_ws'] is not None and row['m_ws0'] is not None and (row['m_ws'] > row['m_ws0'] + 0.05) if m else None
                    out.append(row)
    return (hours, glides)

def strength(legs: list[dict], flights: list[dict]) -> dict:
    day_r = {}
    for r in legs:
        if r.get('pilot') and r.get('ok') and (r.get('cat') == 'thermal') and r.get('paired') and ((r.get('m_xc') or 0) > 0) and ((r.get('vair') or 0) > 0):
            day_r.setdefault(r['pilot'], {}).setdefault(r['date'], []).append(math.log(r['vair'] / r['m_xc']))
    med = {p: {d: st.median(v) for d, v in ds.items()} for p, ds in day_r.items()}
    val = {}
    for f in flights:
        p = f.get('pilot')
        others = [v for d, v in (med.get(p) or {}).items() if d != f['date']]
        if others:
            val[f['fid']] = st.median(others)
    if not val:
        return {}
    xs = sorted(val.values())
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    a, b = (q(1 / 3), q(2 / 3))
    return {k: 'low' if v < a else 'mid' if v < b else 'high' for k, v in val.items()}
DEAD_WS, DEAD_XC = (1.5, 50.0)
BEST_FROM, BEST_TO = (11, 17)
DEAD_QUIET, DEAD_NORMAL_UP = (0.5, 5)
DEAD_SHORT_UP, DEAD_SHORT_MIN, DEAD_MIN_AFT = (0.25, 2 / 3, 4)
DEAD_MIN_TRACKED, DEAD_MIN_NEVER = (2, 2)
DEAD_NO_LONG, DEAD_LONG_SHARE = (0.75, 0.25)
NORMAL_MIN_DAYS = 4

def half(day: str) -> str:
    return 'winter' if int(day[5:7]) in (11, 12, 1, 2) else 'season'

def best(T: dict | None, key: str, a: int=BEST_FROM, b: int=BEST_TO):
    if not T or not T.get(key):
        return None
    hours = T.get('hours') or list(range(6, 6 + len(T[key])))
    xs = [v for h, v in zip(hours, T[key]) if a <= h <= b and v is not None]
    return max(xs) if xs else None

def _at(pm: dict, t) -> dict:
    return (pm or {}).get(str(int(t))) or {} if t is not None else {}

def shorts(days: list[dict], model: str | None=None, pilots: dict | None=None, corrected: bool=False) -> list[dict]:
    out = []
    for d in days:
        m_ = model_of(d, model)
        for f in d.get('short') or []:
            rec, pm = mk(f, m_, corrected)
            T = rec.get('takeoffNode')
            a, _ = track.times_of(f)
            m = _at(pm, a)
            row = {k: f.get(k) for k in ('date', 'region', 'state', 'grp', 'cls', 'dur_h', 'list_km', 'afternoon', 'sled', 'solar0', 'solar1', 'tracked', 'climbed', 'best2', 'gain_max', 'rel_agl', 'home', 'engine')}
            row.update({'fid': f['id'], 'pilot': f.get('pilot') or (pilots or {}).get(str(f['id'])), 'day_type': common.day_type(f['date']), 'paired': bool(m), 'best_ws0': best(T, 'ws0'), 'best_ws': best(T, 'ws'), 'best_xc': best(T, 'xc')})
            row.update({'m_' + k: m.get(k) for k in track.PROBE_KEYS})
            out.append(row)
    return out

def regiondays(days: list[dict], daylists: dict, model: str | None=None, corrected: bool=False) -> list[dict]:
    norm = {}
    for d, x in daylists.items():
        for r, c in (x.get('regions') or {}).items():
            norm.setdefault((r, common.day_type(d), half(d)), []).append(c)
    built = {}
    for d in days:
        m_ = model_of(d, model)
        for f in (d.get('flights') or []) + (d.get('short') or []):
            T = mk(f, m_, corrected)[0].get('takeoffNode')
            b = built.setdefault((d['date'], f['region']), {'ws0': [], 'xc': [], 'tracked': 0, 'never': 0, 'long': 0})
            if f.get('kind') != 'short':
                b['long'] += 1
            for k in ('ws0', 'xc'):
                v = best(T, k)
                if v is not None:
                    b[k].append(v)
            if f.get('kind') == 'short' and f.get('tracked') and f.get('afternoon'):
                b['tracked'] += 1
                b['never'] += not f.get('climbed')
    out = []
    for d, r in sorted(set(built) | {(d, r) for d, x in daylists.items() for r in x.get('regions') or {}}):
        c = ((daylists.get(d) or {}).get('regions') or {}).get(r)
        b = built.get((d, r)) or {'ws0': [], 'xc': [], 'tracked': 0, 'never': 0, 'long': 0}
        dt = common.day_type(d)
        ns = norm.get((r, dt, half(d))) or []
        n_ok = len(ns) >= NORMAL_MIN_DAYS
        n_up = st.median((x['up'] for x in ns)) if n_ok else None
        n_short = sum((x['aft_short'] for x in ns)) / max(1, sum((x['aft'] for x in ns))) if n_ok else None
        n_long = sum((x['long'] > 0 for x in ns)) / len(ns) if n_ok else None
        ws = st.median(b['ws0']) if b['ws0'] else None
        xc = st.median(b['xc']) if b['xc'] else None
        row = {'date': d, 'region': r, 'day_type': dt, 'list': c is not None, 'list_full': bool((daylists.get(d) or {}).get('full')), 'up': c['up'] if c else None, 'long': c['long'] if c else None, 'aft': c['aft'] if c else None, 'aft_short': c['aft_short'] if c else None, 'aft_sled': c['aft_sled'] if c else None, 'aft_early': c['aft_early'] if c else None, 'normal_days': len(ns), 'normal_up': n_up, 'normal_aft_short': None if n_short is None else round(n_short, 2), 'normal_long_days': None if n_long is None else round(n_long, 2), 'best_ws0': ws, 'best_xc': xc, 'tracked': b['tracked'], 'never_climbed': b['never'], 'built': (d, r) in built}
        good = ws is not None and ws >= DEAD_WS or (xc is not None and xc >= DEAD_XC)
        few_long = c['long'] <= max(1, int(c['up'] * DEAD_LONG_SHARE)) if c else b['long'] <= 1
        busy = n_up is not None and n_up >= DEAD_NORMAL_UP
        why = []
        if good and few_long and (b['tracked'] >= DEAD_MIN_TRACKED) and (b['never'] >= DEAD_MIN_NEVER) and (b['never'] / b['tracked'] >= 0.5):
            why.append(f"{b['never']} of {b['tracked']} tracked afternoon short flights never climbed")
        if good and few_long and c:
            sh = c['aft_short'] / c['aft'] if c['aft'] else 0
            if n_short is not None and c['aft'] >= DEAD_MIN_AFT and (sh >= DEAD_SHORT_MIN) and (sh >= n_short + DEAD_SHORT_UP):
                why.append(f"short: {c['aft_short']} of {c['aft']} afternoon launches, normal {n_short:.0%}")
            if busy and (not c['long']) and (c['up'] <= DEAD_QUIET * n_up):
                why.append(f"quiet: {c['up']} uploads, no long flight, normal {n_up:g}")
            if busy and (not c['long']) and (n_long is not None) and (n_long >= DEAD_NO_LONG):
                why.append(f'no long flight, normal on {n_long:.0%} of such days')
        row['dead'] = '; '.join(why)
        out.append(row)
    return out

def observations(days: list[dict], model: str | None=None, corrected: bool=False) -> tuple[list, list]:
    sfc, snd = ([], [])
    for d in days:
        m_ = model_of(d, model)
        for o in d.get('obs') or []:
            pm = mk(o, m_, corrected)[1]
            ob = o.get('obs') or {}
            base = {'date': d['date'], 'region': o['region'], 'sid': o['sid'], 'paired': bool(pm)}
            if o['kind'] == 'asos':
                z = ZoneInfo(o['tz'])
                fc = {str(datetime.fromtimestamp(int(t), z).hour): m for t, m in pm.items() if m}
                hrs = ob.get('hours') or {}
                t2 = [m.get('t2') for m in fc.values() if m.get('t2') is not None]
                both = [h for h in hrs if h in fc]
                cu_o = [h for h in both if hrs[h].get('cu') is not None]
                cu_f = [h for h in both if fc[h].get('cu')]
                diffs = [hrs[h]['cu'] - fc[h]['cb'] for h in both if hrs[h].get('cu') is not None and fc[h].get('cb') is not None]
                sfc.append({**base, 'tmax_obs': ob.get('tmax'), 'tmax_fc': max(t2) if t2 else None, 'tmax_diff': None if ob.get('tmax') is None or not t2 else round(ob['tmax'] - max(t2), 1), 'hours': len(both), 'cu_obs': len(cu_o), 'cu_fc': len(cu_f), 'cu_both': len(set(cu_o) & set(cu_f)), 'base_diff': round(st.median(diffs)) if diffs else None})
            elif o['kind'] == 'raob':
                m = _at(pm, o.get('t'))
                snd.append({**base, 'mix_obs': ob.get('mix'), 'zi_fc': m.get('zi'), 'top_fc': m.get('top'), 'lapse_obs': ob.get('lapse2'), 'blapse_fc': m.get('blapse'), 't_sfc_obs': ob.get('t_sfc'), 't2_fc': m.get('t2'), 'mix_diff': None if ob.get('mix') is None or m.get('zi') is None else ob['mix'] - m['zi']})
    return (sfc, snd)
