from __future__ import annotations
import glob
import os
import common
MODEL_KEYS = ('xc', 'gs', 'climb', 'ach', 'usable', 'ridge', 'ws', 'ws0', 'zi', 'top', 'band', 'shade', 'sun', 'sunElev', 'relief', 'elev', 'cb', 'cu', 'circ', 'wind', 'rain', 'od', 'cloud', 'cloudlow', 'blapse', 'twidth', 'sink', 'ridgev', 'densAlt', 'solar', 'localHour')
ROUTE_KEYS = ('len', 'autoMean', 'autoMax', 'straightMean', 'straightMax', 'sameShare', 'sameFlownKm', 'sameAutoKm', 'flownBend', 'autoBend', 'bendMatch', 'autoBendsAlone', 'bent', 'autoH', 'straightH', 'flownH', 'error')
FLIGHT_KEYS = ('date', 'region', 'state', 'cls', 'grp', 'key', 'key_exact', 'src', 'ac')

def load_days(data: str) -> list[dict]:
    out = []
    for fn in sorted(glob.glob(os.path.join(data, 'days', '*.json.gz'))):
        out.append(common.read_json(fn))
    return out

def tables(days: list[dict], model: str='hrrr', srcs: set | None=None) -> tuple[list, list, list]:
    legs, climbs, flights = ([], [], [])
    for d in days:
        for f in d['flights']:
            if srcs and f.get('src') not in srcs:
                continue
            base = {k: f.get(k) for k in FLIGHT_KEYS}
            base['fid'] = f['id']
            T = (f.get('model_' + model) or {}).get('takeoffNode')
            flights.append({**base, 'free_km': f.get('free_km'), 'list_km': f.get('list_km'), 'dur_h': f.get('dur_h'), 'pfd': (T or {}).get('pfdCut'), 'paired': T is not None, 'nlegs': len(f['legs']), 'nclimbs': len(f['climbs'])})
            for L in f['legs']:
                m = (L.get('models') or {}).get(model) or {}
                own, rt = (m.get('own') or {}, m.get('route') or {})
                row = {**base, 'leg': L.get('leg'), 'ok': L.get('ok'), 'cat': L.get('cat'), 'dist': L.get('dist'), 't': L.get('t'), 'ca': L.get('ca'), 'dh': L.get('dh'), 'therm': L.get('therm'), 'vg': L.get('vg'), 'vair': L.get('vair'), 'agl': L.get('agl'), 'alt': L.get('alt'), 'speed': L.get('speed'), 'wind_wg': L.get('wind'), 'tcount': L.get('tcount'), 'line_src': L.get('line_src'), 'paired': bool(own), 'last': L is f['legs'][-1]}
                row.update({'m_' + k: own.get(k) for k in MODEL_KEYS})
                row.update({'r_' + k: rt.get(k) for k in ROUTE_KEYS})
                legs.append(row)
            for c in f['climbs']:
                own = (c.get('models') or {}).get(model) or {}
                row = {**base, 'rate': c.get('rate'), 'gain': c.get('gain'), 'dur': c.get('dur'), 'agl1': c.get('agl1'), 'z1': c.get('z1'), 'r100': c.get('r100'), 'engon': c.get('engon'), 'leg': c.get('leg'), 'paired': bool(own)}
                row.update({'m_' + k: own.get(k) for k in MODEL_KEYS})
                climbs.append(row)
    return (legs, climbs, flights)

def flat(row: dict) -> bool:
    return (row.get('m_relief') or 0) < common.RELIEF_FLAT
