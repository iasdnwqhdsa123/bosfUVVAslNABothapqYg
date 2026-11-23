from __future__ import annotations
import hashlib
import json
import math
import re
from array import array
from collections import Counter
from datetime import date
from . import guard, objects
EXPOSURE_V = 'x1'
G = 0.005
BAND_M = 300
SLOT_S = 30
TYPES = ('glider', 'hg', 'pg')
COLS = {'h': '<i4', 'i': '<i4', 'j': '<i4', 'b': '<i2', 'y': '<u1', 's': '<u1', 'sec': '<i4', 'cov': '<i2', 'n': '<i4', 'zmin': '<i4', 'part': '<u1'}

def _int(x) -> int | None:
    return int(x) if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) and float(x).is_integer() else None

def hour_day(h: int, j: int) -> str:
    return objects.solar_day(h * 3600 + 1800, (j + 0.5) * G)
NAMES = ('h', 'i', 'j', 'b', 'y', 's', 'sec', 'cov', 'n', 'zmin')
BOUNDS = {'i': (-18000, 18000), 'j': (-36000, 36000), 'b': (-64, 191), 'sec': (0, 2 ** 31 - 1), 'cov': (0, 3600 // SLOT_S), 'n': (1, 65535), 'zmin': (-2 ** 31, 2 ** 31 - 1)}
_ROW_KEYS = ('i', 'j', 'b', 'sec', 'cov', 'n', 'zmin')
_HOUR_RE = re.compile('"h"\\s*:\\s*(\\d{1,10})\\b')
_KIND_RE = re.compile('"k"\\s*:\\s*"([^"]*)"')

def _hour() -> dict:
    return {'shed': 0, 'low': 0, 'full': {}, 'start': None, 'bad': 0, 'gap': 0}

def _merge(A: dict, H: dict, full=True) -> None:
    A['shed'], A['low'] = (max(A['shed'], H['shed']), max(A['low'], H['low']))
    if H['start'] is not None:
        A['start'] = H['start'] if A['start'] is None else min(A['start'], H['start'])
    A['bad'] += H['bad']
    A['gap'] = max(A['gap'], H['gap'])
    if full:
        for s, f in H['full'].items():
            A['full'][s] = max(A['full'].get(s, 0), f)

def parse(text: bytes, cnt: Counter | None=None) -> tuple:
    import numpy as np
    cnt = cnt if cnt is not None else Counter()
    rows, hours, runs, blind = (array('q'), {}, set(), 0)
    for line in text.decode('utf-8', 'replace').splitlines():
        if '"x"' not in line and '"k"' in line:
            continue
        try:
            r = json.loads(line)
        except (ValueError, RecursionError):
            k = _KIND_RE.search(line)
            if k and k.group(1) != 'x' or not line.strip():
                continue
            m = _HOUR_RE.search(line)
            h = int(m.group(1)) if m else None
            cnt['bad_unreadable'] += 1
            if h is not None and 0 < h < 2 ** 31:
                hours.setdefault(h, _hour())['bad'] += 1
            else:
                blind += 1
            continue
        if not isinstance(r, dict) or r.get('k') != 'x':
            continue
        h = _int(r.get('h'))
        if h is None or not 0 < h < 2 ** 31:
            cnt['bad_hour'] += 1
            blind += 1
            continue
        q = r.get('q') if isinstance(r.get('q'), dict) else {}
        H = hours.setdefault(h, _hour())
        lrows, s = (r.get('rows'), r.get('s'))
        one = _hour()
        for f in ('shed', 'low'):
            one[f] = max(0, min(_int(q.get(f)) or 0, 2 ** 31 - 1))
        st = _int(q.get('start'))
        if st is not None and 0 <= st < 2 ** 31:
            one['start'] = st
            runs.add(h * 3600 + st)
        one['full'] = {str(s): max(0, min(_int(q.get('full')) or 0, 2 ** 31 - 1))}
        _merge(H, one)
        if r.get('v') != 2 or not isinstance(lrows, list) or r.get('g') != G:
            H['bad'] += 1
            cnt['bad_line'] += 1
            continue
        if s is None:
            if lrows:
                H['bad'] += 1
                cnt['bad_line'] += 1
            else:
                H['gap'] = 1
                cnt['gap_lines'] += 1
            continue
        if s not in guard.FEEDS:
            cnt['feed_not_admitted'] += 1
            continue
        si = guard.FEEDS.index(s)
        cnt['lines'] += 1
        for row in lrows:
            if not isinstance(row, list) or len(row) != 8 or row[2] not in TYPES:
                cnt['bad_row'] += 1
                H['bad'] += 1
                continue
            v = [_int(row[k]) for k in (0, 1, 3, 4, 5, 6, 7)]
            if None in v or any((not BOUNDS[k][0] <= x <= BOUNDS[k][1] for k, x in zip(_ROW_KEYS, v))):
                cnt['bad_row'] += 1
                H['bad'] += 1
                continue
            i, j, b, sec, cov, n, zmin = v
            rows.extend((h, i, j, b, TYPES.index(row[2]), si, sec, cov, n, zmin))
            cnt['rows'] += 1
    if blind:
        cnt['bad_no_hour'] += blind
        for H in hours.values():
            H['bad'] += 1
    return (np.frombuffer(rows, dtype=np.int64).reshape(-1, len(NAMES)).copy(), hours, sorted(runs))

def status(H: dict) -> str:
    full = sum(H['full'].values())
    return 'partial' if H['shed'] or H['low'] or full or (H['start'] is not None) or H['bad'] or H['gap'] else 'whole'

def build(lk, sd: str, raw: dict, cache: dict | None=None) -> tuple[dict, dict]:
    import numpy as np
    cache = cache if cache is not None else {}
    cnt, rows, hours, runs, inputs = (Counter(), [], {}, set(), [])
    want = (date.fromisoformat(sd) - date(1970, 1, 1)).days
    _, missing = objects.ready(sd, raw)
    for d in objects.needs(sd):
        seen = np.zeros(0, dtype=np.int64)
        for rk in raw.get(d, []):
            if rk not in cache:
                text = objects.read_raw(lk, rk)
                c = Counter()
                cache[rk] = (*parse(text, c), c, hashlib.sha256(text).hexdigest())
            r, hs, rn, c, sha = cache[rk]
            inputs.append({'day': d, 'v': int(rk.split('/')[2][1:]), 'sha256': sha})
            if d == sd:
                cnt.update(c)
            hk = r[:, 0] * 8 + r[:, 5]
            if len(seen):
                again = np.isin(hk, seen)
                if again.any():
                    cnt['twice_rows'] += int(again.sum())
                    r, hk = (r[~again], hk[~again])
            seen = np.union1d(seen, hk)
            day = np.floor((r[:, 0] * 3600 + 1800 + (r[:, 2] + 0.5) * G * 240.0) / 86400.0).astype(np.int64)
            rows.append(r[day == want])
            for h, H in hs.items():
                if objects.solar_day(h * 3600 + 1800, -180.0) <= sd <= objects.solar_day(h * 3600 + 1800, 180.0):
                    _merge(hours.setdefault(h, _hour()), H)
            runs.update(rn)
    part = {h for h, H in hours.items() if status(H) == 'partial'}
    a = np.concatenate(rows) if rows else np.zeros((0, len(NAMES)), dtype=np.int64)
    a = a[np.lexsort([a[:, k] for k in (9, 8, 7, 6, 3, 4, 2, 1, 5, 0)])]
    cols = {k: a[:, x].astype(COLS[k]) for x, k in enumerate(NAMES)}
    cols['part'] = np.isin(a[:, 0], np.array(sorted(part), dtype=np.int64)).astype(COLS['part'])
    cnt['partial_rows'] = int(cols['part'].sum())
    cnt['sd_rows'] = int(len(a))
    meta = {'v': 1, 'rule': EXPOSURE_V, 'sd': sd, 'g': G, 'band': BAND_M, 'slot': SLOT_S, 'feeds': list(guard.FEEDS), 'types': list(TYPES), 'hours': [[h, status(H), H['shed'], H['low'], sum(H['full'].values()), H['start']] for h, H in sorted(hours.items())], 'runs': sorted(runs), 'inputs': inputs, 'missing': missing, 'counts': dict(sorted(cnt.items()))}
    return (meta, cols)

def run(lk, days=None, cache=None, n=None, most: int | None=None) -> dict:

    def make(sd, raw, missing):
        meta, cols = build(lk, sd, raw, cache)
        return (objects.pack_columns(meta, cols), {'rows': len(cols['h'])})
    return objects.run_days(lk, 'exposure', EXPOSURE_V, make, days, cache, n, most)
