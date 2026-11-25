from __future__ import annotations
import json
import math
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ACWIND_V = 1
KIND = 'wa'
Z_MIN = 50.0

def lines_of(text: bytes, counts: dict | None=None) -> list[dict]:
    counts = counts if counts is not None else {}
    out = []
    for raw in text.splitlines():
        if b'"wa"' not in raw:
            continue
        try:
            r = json.loads(raw)
        except (ValueError, RecursionError):
            counts['acwind_unreadable'] = counts.get('acwind_unreadable', 0) + 1
            continue
        if isinstance(r, dict) and r.get('k') == KIND and isinstance(r.get('h'), int) and isinstance(r.get('rows'), list):
            out.append(r)
    counts['acwind_hours'] = counts.get('acwind_hours', 0) + len(out)
    return out

def cell_stations(stations: list, g: float) -> dict:
    out = {}
    for s in stations:
        try:
            sid, la, lo, el = (str(s[0]), float(s[1]), float(s[2]), float(s[3]))
        except (TypeError, ValueError, IndexError):
            continue
        out.setdefault((math.floor(la / g), math.floor(lo / g)), []).append((sid, la, lo, el))
    return out

def rows_of(lines: list[dict], stations: list, counts: dict | None=None) -> dict:
    counts = counts if counts is not None else {}
    cols = {k: [] for k in ('obs_id', 't_valid', 'lat', 'lon', 'elev', 'z', 'zmsl', 'u', 'v', 'n', 'su', 'sv', 'share')}
    by_g = {}
    seen = set()
    for L in lines:
        g, band, h = (float(L.get('g') or 0), float(L.get('band') or 0), int(L['h']))
        if not (g > 0 and band > 0):
            continue
        cells = by_g.get(g) or by_g.setdefault(g, cell_stations(stations, g))
        for row in L['rows']:
            try:
                i, j, b, n, u, v, su, sv = (int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
            except (TypeError, ValueError, IndexError):
                counts['acwind_bad_rows'] = counts.get('acwind_bad_rows', 0) + 1
                continue
            sts = cells.get((i, j))
            if not sts:
                continue
            zmsl = (b + 0.5) * band
            for sid, la, lo, el in sts:
                z = zmsl - el
                key = (sid, h, b)
                if z < Z_MIN or key in seen:
                    continue
                seen.add(key)
                for c, x in (('obs_id', f'{sid}/{z:.0f}'), ('t_valid', h * 3600), ('lat', la), ('lon', lo), ('elev', el), ('z', z), ('zmsl', zmsl), ('u', u), ('v', v), ('n', n), ('su', su), ('sv', sv), ('share', len(sts))):
                    cols[c].append(x)
    counts['acwind_rows'] = counts.get('acwind_rows', 0) + len(cols['obs_id'])
    return cols

def table_of(cols: dict):
    import pairs as P
    n = len(cols['obs_id'])
    f = lambda k: np.array(cols[k], np.float32)
    return P.table(P.obs_schema('acwind'), {'kind': 'acwind', 'obs_id': np.array(cols['obs_id'], dtype=object), 't_valid': np.array(cols['t_valid'], np.int64), 'lat': f('lat'), 'lon': f('lon'), 'review': np.ones(n, bool), 'elev': f('elev'), 'o_z': f('z'), 'o_zmsl': f('zmsl'), 'o_u': f('u'), 'o_v': f('v'), 'q_n': f('n'), 'q_su': f('su'), 'q_sv': f('sv'), 'q_share': f('share')}, n)

def listing(lk) -> dict:
    from thermalml import objects as O
    return O.raw_days(lk)

def input_id(key: str) -> str:
    return f'acwind|v{ACWIND_V}|{key}'

def write_day(lk, day: str, stations: list, counts: dict | None=None, files: dict | None=None) -> dict:
    import pairs as P
    from thermalml import objects as O
    counts = counts if counts is not None else {}
    keys = list((files if files is not None else listing(lk)).get(day) or [])
    if not keys:
        counts['acwind_no_file'] = counts.get('acwind_no_file', 0) + 1
        return {'status': 'none', 'rows': 0}
    by_id = {input_id(k): k for k in keys}

    def build(new, _all):
        lines = []
        for i in new:
            lines += lines_of(O.read_raw(lk, by_id[i]), counts)
        return table_of(rows_of(lines, stations, counts))
    return P.write_unit(lk, 'obs', 'acwind', day, 'all', sorted(by_id), build)
