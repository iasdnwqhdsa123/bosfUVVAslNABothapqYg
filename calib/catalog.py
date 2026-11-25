from __future__ import annotations
import gzip
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta
import common
UNITS = 'derived/u/'
ROWS = 'derived/rows/'
STARTS = 'derived/starts/'
PLANS = 'derived/plans/'
SNAPS = 'derived/catalog/'
PINS = 'derived/catalog/pins/'
RAW_WEGLIDE = 'raw/weglide/'
RAW_INDEX = 'derived/rawindex/'
STATION_LISTS = 'derived/stations/'
KINDS = ('flight', 'collector')
STCAST = 'stcast'
STCAST_RUN_H = (0, 12)
STCAST_AREAS = {'hrrr': ('US',), 'gfs': ('US', 'EU', 'SH'), 'ecmwf': ('US', 'EU', 'SH')}
STCAST_CONSUMERS = ('pairs',)

def stcast_consumers(area_gv: dict | None) -> tuple:
    if (area_gv or {}).get('stations'):
        return STCAST_CONSUMERS
    return STCAST_CONSUMERS + tuple((c for c in sorted(common.CONSUMERS) if c not in STCAST_CONSUMERS))
_RUN = re.compile('\\d{10}')
SWEEP = re.compile('\\d{8}T\\d{6}Z')
_PLAN_ID = re.compile('(\\d{8}T\\d{6}Z)-(0|[1-9]\\d{0,5})')
MAX_TRIES = 3
ATTEMPTS = 99
OK, OUTSIDE, SKIPPED = (('ok',), ('outside',), ('skipped',))
TRY = ('retry', 'failed', 'error')
STATUSES = OK + OUTSIDE + TRY + SKIPPED
ROW_FIELDS = ('unit', 'kind', 'day', 'area', 'model', 'init', 'lead_h', 'src', 'box', 'bbox', 'cs', 'gv', 'sha256', 'bytes', 'status', 'tries', 'sec', 'wave')
EXTRA_FIELDS = ('model_id', 'run', 'sv', 'key', 'put', 'why', 'try')
RAW_FIELDS = ('day', 'raw_sha256', 'flights', 'legs', 'climbs')
PLAN_FIELDS = ('v', 'id', 'sweep', 'wave', 'code_sha', 'cs', 'gv', 'chunks')
ENTRY_FIELDS = ('unit', 'name', 'raw', 'tries', 'lost', 'why', 'prev')
_HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.join(common.PIPELINE, 'verify.py')
STATIONS = os.path.join(common.PIPELINE, 'verify_stations.json')
OPSLAKE = os.path.join(common.PIPELINE, 'opslake.py')
RUNNER_FILES = ('day.py', 'common.py', 'track.py', 'collector.py', 'labbuild.py', 'catalog.py', 'stationcast.py')

class UnitError(ValueError):
    pass

def check_unit(u) -> dict:
    try:
        kind, key, model, init, box = (u[k] for k in ('kind', 'daykey', 'model', 'init', 'box'))
    except (KeyError, TypeError):
        raise UnitError('unit') from None
    if not isinstance(key, str) or not isinstance(init, str):
        raise UnitError('unit')
    d, macro = common.split_key(key)
    try:
        ok = len(d) == 10 and date.fromisoformat(d).isoformat() == d
        if init != 'own':
            ok = ok and bool(_RUN.fullmatch(init)) and bool(datetime.strptime(init, '%Y%m%d%H'))
    except ValueError:
        raise UnitError('unit') from None
    if kind == STCAST:
        if not ok or init == 'own' or macro not in common.MACRO or (key != common.day_key(d, macro)) or (init[:8] != d.replace('-', '')) or (int(init[8:]) not in STCAST_RUN_H) or (macro not in STCAST_AREAS.get(model, ())) or (box != macro):
            raise UnitError('unit')
        return {'kind': kind, 'daykey': key, 'model': model, 'init': init, 'box': box}
    if not ok or kind not in KINDS or macro not in common.MACRO or (key != common.day_key(d, macro)) or (model not in common.MODEL_MACRO) or common.is_lead(model) or (box not in common.REGION_MACRO) or (common.macro_of(box) != macro):
        raise UnitError('unit')
    return {'kind': kind, 'daykey': key, 'model': model, 'init': init, 'box': box}

def logical_id(u) -> str:
    u = check_unit(u)
    return 'unit|' + '|'.join((str(u[k]) for k in ('kind', 'daykey', 'model', 'init', 'box')))

def unit_of(daykey: str, model_id: str, box: str, kind: str='flight') -> dict:
    d, _ = common.split_key(daykey)
    base, n = (common.base_model(model_id), common.lead_of(model_id))
    if n is None:
        init = 'own'
    else:
        h = common.LEAD_RUN_H.get(common.macro_of(box), {}).get(base)
        if h is None:
            raise UnitError('lead')
        try:
            init = (datetime.fromisoformat(d) - timedelta(days=n)).replace(hour=h).strftime('%Y%m%d%H')
        except ValueError:
            raise UnitError('unit') from None
    return check_unit({'kind': kind, 'daykey': daykey, 'model': base, 'init': init, 'box': box})

def lead_days(u: dict) -> int | None:
    u = check_unit(u)
    if u['init'] == 'own' or u['kind'] == STCAST:
        return None
    r = datetime.strptime(u['init'], '%Y%m%d%H')
    n = (date.fromisoformat(common.day_of(u['daykey'])) - r.date()).days
    h = common.LEAD_RUN_H.get(common.macro_of(u['box']), {}).get(u['model'])
    if n < 0 or h is None or r.hour != h:
        raise UnitError('lead')
    return n

def model_id_of(u: dict) -> str:
    n = lead_days(u)
    return u['model'] if n is None else f"{u['model']}@{n}"

def source_files() -> list[str]:
    paths = set(common.PHYSICS_FILES) | set(common.PHYSICS_FILES_SH) | set(common.PHYSICS_FILES_LEAD) | set(common.SCORE_FILES)
    paths |= {os.path.join(_HERE, f) for f in RUNNER_FILES}
    paths |= {VERIFY, STATIONS, OPSLAKE}
    return sorted(paths)

def source_stamp() -> str:
    return common._version_of(source_files())

def stamp(t: float | None=None) -> str:
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(time.time() if t is None else t))

def unit_key(name: str, cs: str) -> str:
    return f'{UNITS}{cs}/{name}.lenc'

def plan_id(sweep: str, wave: int) -> str:
    return f'{sweep}-{int(wave)}'

def plan_key(sweep: str, wave: int) -> str:
    return f'{PLANS}{plan_id(sweep, wave)}.lenc'

def _attempt_key(base: str, attempt: int) -> str:
    return base + ('.lenc' if int(attempt) == 1 else f'.r{int(attempt)}.lenc')

def rows_key(sweep: str, wave: int, chunk: int, attempt: int=1) -> str:
    return _attempt_key(f'{ROWS}{sweep}/{int(wave)}-{int(chunk)}', attempt)

def start_key(sweep: str, wave: int, chunk: int, attempt: int=1) -> str:
    return _attempt_key(f'{STARTS}{sweep}/{int(wave)}-{int(chunk)}', attempt)

def encode(obj) -> bytes:
    return gzip.compress(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8'), mtime=0)

def decode(blob: bytes):
    if blob[:2] == b'\x1f\x8b':
        blob = gzip.decompress(blob)
    return json.loads(blob.decode('utf-8'))

def check_plan(plan, pid: str) -> dict:
    m = _PLAN_ID.fullmatch(pid if isinstance(pid, str) else '')
    if not m or not isinstance(plan, dict) or plan.get('id') not in (None, pid) or (str(plan.get('sweep')) != m.group(1)) or (str(plan.get('wave')) != m.group(2)) or (not isinstance(plan.get('chunks') or [], list)):
        raise UnitError('plan')
    return plan

def load_plan(lake, pid: str) -> dict:
    if not _PLAN_ID.fullmatch(pid if isinstance(pid, str) else ''):
        raise UnitError('plan id')
    return check_plan(decode(lake.get(f'{PLANS}{pid}.lenc')), pid)

def stations_version(stations: list) -> str:
    return hashlib.sha256(json.dumps(stations, separators=(',', ':')).encode()).hexdigest()[:12]

def stations_key(v: str) -> str:
    if not isinstance(v, str) or not re.fullmatch('[0-9a-f]{12}', v):
        raise UnitError('stations version')
    return f'{STATION_LISTS}{v}.lenc'

def keep_stations(lake, stations: list, v: str) -> str:
    if stations_version(stations) != v:
        raise UnitError('stations')
    key = stations_key(v)
    kept = vars(lake).setdefault('_kept_lists', set())
    if key not in kept:
        lake.put(key, json.dumps({'stations': stations}, separators=(',', ':')).encode())
        kept.add(key)
    return key

def stations_of(lake, v: str, fallback: str | None=STATIONS) -> list:
    import lake as lakem
    try:
        sts = json.loads(lake.get(stations_key(v)).decode('utf-8'))['stations']
    except lakem.LakeMissing:
        sts = None
        if fallback and os.path.exists(fallback):
            with open(fallback) as f:
                here = json.load(f)['stations']
            if stations_version(here) == v:
                sts = here
        if sts is None:
            raise
    if stations_version(sts) != v:
        raise UnitError('stations')
    return sts

def _once(lake, key_of, plain: bytes) -> str:
    for n in range(1, ATTEMPTS + 1):
        key = key_of(n)
        if lake.put(key, plain, strict=False).status != 'differs':
            return key
    raise UnitError('attempts')

def write_rows(lake, sweep: str, wave: int, chunk: int, body: dict) -> str:
    plain = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    return _once(lake, lambda n: rows_key(sweep, wave, chunk, n), plain)

def mark_start(lake, sweep: str, wave: int, chunk: int) -> str:
    plain = json.dumps({'v': 1, 'n': os.urandom(8).hex()}, separators=(',', ':')).encode()
    return _once(lake, lambda n: start_key(sweep, wave, chunk, n), plain)
ITEM_LISTS = ('flights', 'short', 'obs', 'collector')

def items_of(rec) -> list[dict]:
    if not isinstance(rec, dict):
        return []
    src = rec.get('items') if isinstance(rec.get('items'), dict) else rec
    return [x for lst in ITEM_LISTS for x in src.get(lst) or [] if isinstance(x, dict)]

def unite(recs) -> list[dict]:
    import day as dayb
    items, at = ([], {})
    for rec in recs:
        for f in items_of(rec):
            f = dayb.clean(f)
            ident = (f.get('kind') or 'flight', f.get('id'), f.get('sid'))
            if ident[1] is None and ident[2] is None:
                items.append(f)
            elif ident in at:
                items[at[ident]] = f
            else:
                at[ident] = len(items)
                items.append(f)
    return items

def read_items(lake, keys) -> list[dict]:
    return unite([decode(lake.get(k)) for k in keys or []])
_ATTEMPT_TAIL = re.compile('(\\d+)-(\\d+)(?:\\.r(\\d+))?\\.lenc')

def _order(key: str, prefix: str=ROWS) -> tuple | None:
    if not key.startswith(prefix):
        return None
    sweep, _, rest = key[len(prefix):].partition('/')
    m = _ATTEMPT_TAIL.fullmatch(rest)
    if not SWEEP.fullmatch(sweep) or not m:
        return None
    return (sweep, int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))

def is_rows_key(key: str) -> bool:
    return _order(key) is not None

def read_rows(lake, listing: list[str]) -> tuple[list[tuple[tuple, list[dict]]], int]:
    out, bad = ([], 0)
    for k in sorted((k for k in listing if is_rows_key(k)), key=_order):
        try:
            rows = decode(lake.get(k)).get('rows') or []
            if not isinstance(rows, list):
                raise ValueError
        except Exception:
            bad += 1
            continue
        out.append((_order(k), [r for r in rows if isinstance(r, dict)]))
    return (out, bad)

def attempts(listing: list[str], prefix: str) -> Counter:
    return Counter((o[:3] for o in (_order(k, prefix) for k in listing) if o))

def started(listing: list[str]) -> Counter:
    return attempts(listing, STARTS)

def read_plans(lake, keys) -> tuple[list[dict], int]:
    out, bad = ([], 0)
    for k in sorted(keys):
        try:
            out.append(decode(lake.get(k)))
        except Exception:
            bad += 1
    return (out, bad)

def day_boxes(items: list[dict]) -> dict[str, list[dict]]:
    import day as dayb
    out = {}
    for kind in KINDS:
        fs = [{'bbox': None, **dayb.clean(f)} for f in items if (f.get('kind') == 'collector') == (kind == 'collector') and f.get('region') in common.REGION_MACRO and f.get('tp') and f.get('tz')]
        bs = [{'box': b['region'], 'bbox': b['bbox'], 'tz': b['tz']} for b in dayb.boxes(fs)] if fs else []
        if bs:
            out[kind] = bs
    return out

def _counts(items: list[dict]) -> dict:
    fl = [f for f in items if f.get('kind') not in ('collector', 'obs', 'station')]
    return {'flights': len(fl), 'legs': sum((len(f.get('legs') or []) for f in fl)), 'climbs': sum((len(f.get('climbs') or []) for f in items))}

def _parts(listing: list[str]) -> dict[str, list[str]]:
    by = {}
    for k in listing:
        m = re.fullmatch(re.escape(RAW_WEGLIDE) + '([0-9a-f]{20})\\.r(\\d+)\\.lenc', k)
        if m:
            by.setdefault(m.group(1), []).append((int(m.group(2)), k))
    return {n: [k for _, k in sorted(v)] for n, v in by.items()}
RAW_V = 2

def raw_version() -> str:
    return f'{RAW_V}-' + common._version_of([os.path.join(_HERE, f) for f in ('catalog.py', 'day.py', 'common.py')])

def day_key_of(recs, name: str, lake) -> str | None:
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        cands = [rec.get('key')] if isinstance(rec.get('key'), str) and rec.get('key') else []
        d = rec.get('date')
        if isinstance(d, str) and d:
            macro = rec.get('macro') if isinstance(rec.get('macro'), str) else common.macro_of_items(items_of(rec))
            cands.append(common.day_key(d, macro))
        for k in cands:
            if lake.name(f'day|{k}') == name:
                return k
    return None

def raw_days(lake, listing: list[str] | None=None, t: float | None=None, write: bool=True) -> tuple[list[dict], dict]:
    listing = lake.list(RAW_WEGLIDE) if listing is None else listing
    parts = _parts(listing)
    idx_keys = sorted((k for k in lake.list(RAW_INDEX) if k.endswith('.lenc')))
    index, rv = ({}, raw_version())
    changed = False
    if idx_keys:
        try:
            got = decode(lake.get(idx_keys[-1]))
            index = got.get('names') or {} if got.get('rv') == rv else {}
            changed = got.get('rv') != rv
        except Exception:
            index, changed = ({}, True)
    n = {'raw_parts': sum((len(v) for v in parts.values())), 'raw_read': 0, 'raw_unreadable': 0, 'raw_other': 0}
    for name, keys in sorted(parts.items()):
        known = index.get(name)
        if known is not None and known.get('parts') == keys:
            continue
        changed = True
        recs, shas, bad = ([], [], False)
        for k in keys:
            try:
                blob = lake.store.get(k)
                recs.append(decode(lake.get(k)))
                shas.append(hashlib.sha256(blob).hexdigest())
                n['raw_read'] += 1
            except Exception:
                bad = True
        if bad:
            n['raw_unreadable'] += 1
            index.pop(name, None)
            changed = True
            continue
        key = day_key_of(recs, name, lake)
        if not key:
            index[name] = {'parts': keys, 'day': None}
            continue
        items = unite(recs)
        index[name] = {'parts': keys, 'day': key, 'raw_sha256': hashlib.sha256(''.join(shas).encode()).hexdigest(), **_counts(items), 'boxes': day_boxes(items)}
    for name in [nm for nm in index if nm not in parts]:
        index.pop(name)
        changed = True
    if changed and write:
        _put_timed(lake, RAW_INDEX, encode({'v': 1, 'rv': rv, 'names': index}), t)
    days = []
    for name in sorted(index):
        e = index[name]
        if not e.get('day'):
            n['raw_other'] += 1
            continue
        days.append({**{k: e.get(k) for k in RAW_FIELDS}, 'parts': e['parts'], 'boxes': e.get('boxes') or {}})
    days.sort(key=lambda d: d['day'])
    n['raw_days'] = len(days)
    return (days, n)

def _clean(r: dict) -> dict:
    return {k: r.get(k) for k in ROW_FIELDS + EXTRA_FIELDS}

def _counter(x) -> Counter:
    if x is None:
        return Counter()
    return x if isinstance(x, Counter) else Counter(x if isinstance(x, dict) else {k: 1 for k in x})

def _lost_ident(e: dict) -> dict:
    out = {'bbox': e.get('bbox')}
    try:
        u = check_unit(e.get('unit'))
        return {**out, 'kind': u['kind'], 'day': common.day_of(u['daykey']), 'area': common.gv_area(u['model'], u['box']), 'model': u['model'], 'model_id': model_id_of(u), 'init': u['init'], 'box': u['box']}
    except UnitError:
        u = e.get('unit') if isinstance(e.get('unit'), dict) else {}
        return {**out, **{k: e.get(k) for k in ('kind', 'area', 'model', 'box')}, 'day': common.day_of(str(e.get('day') or '')), 'model_id': e.get('run'), 'init': u.get('init')}

def fold(chunks: list[tuple[tuple, list[dict]]], plans=(), began=None, cs: str | None=None, wrote=None) -> tuple[dict[str, dict], dict]:
    wrote = _counter(wrote) if wrote is not None else Counter((o[:3] for o, _ in chunks))
    began = _counter(began)
    events = [(o, r) for o, rs in chunks for r in rs]
    for p in plans:
        sw, w = (p.get('sweep'), int(p.get('wave', 0)))
        for i, entries in enumerate(p.get('chunks') or []):
            dead = began.get((sw, w, i), 0) - wrote.get((sw, w, i), 0)
            for e in entries if dead > 0 else ():
                if not isinstance(e, dict) or not e.get('name'):
                    continue
                lost = {**_lost_ident(e), 'unit': e['name'], 'status': 'lost', 'cs': p.get('cs'), 'wave': w}
                events += [((sw, w, i, 0), lost)] * dead
    events.sort(key=lambda e: e[0])
    out: dict[str, dict] = {}
    n = {'no_unit': 0, 'unknown_status': 0}
    for (_, w, _, _), r in events:
        u, st = (r.get('unit'), r.get('status'))
        if not u:
            n['no_unit'] += 1
            continue
        cur = out.get(u)
        if st in OK or st in OUTSIDE:
            out[u] = {**_clean(r), 'state': 'done' if st in OK else 'nocover', 'tries_cs': {}, 'lost_cs': {}, 'wave': r.get('wave', w)}
        elif st in TRY and r.get('try') != 0 or st == 'lost':
            base = dict(cur) if cur else {**_clean(r), 'state': 'tried', 'sha256': None, 'bytes': None, 'gv': None, 'key': None, 'tries_cs': {}, 'lost_cs': {}}
            c = str(r.get('cs') or '')
            base['tries_cs'] = {**base.get('tries_cs', {}), c: base.get('tries_cs', {}).get(c, 0) + 1}
            if st == 'lost':
                base['lost_cs'] = {**base.get('lost_cs', {}), c: base.get('lost_cs', {}).get(c, 0) + 1}
            else:
                base['why'] = r.get('why')
            base['try_cs'] = c
            out[u] = base
        elif st in TRY or st in SKIPPED:
            pass
        else:
            n['unknown_status'] += 1
    for r in out.values():
        settle(r, cs)
    return (out, n)

def settle(r: dict, cs: str | None) -> dict:
    c = cs if cs is not None else r.get('try_cs') or r.get('cs')
    r['tries'] = int((r.get('tries_cs') or {}).get(c or '', 0))
    r['lost'] = int((r.get('lost_cs') or {}).get(c or '', 0))
    st = r.get('state') or r.get('status')
    if st == 'nocover' and (r.get('cs') == c or not c):
        r['status'] = 'nocover'
    elif r['tries'] >= MAX_TRIES:
        r['status'] = 'failed'
    elif st == 'missing':
        r['status'] = 'missing'
    elif r.get('sha256'):
        r['status'] = 'done'
    else:
        r['status'] = 'tried'
    return r

def object_of(r: dict, name: str) -> str:
    k = r.get('key')
    if isinstance(k, str) and k.startswith(UNITS) and k.endswith(f'/{name}.lenc'):
        return k
    return unit_key(name, r.get('cs') or '')

def check(rows: dict[str, dict], listing: list[str]) -> dict:
    listed = {k for k in listing if k.startswith(UNITS) and k.endswith('.lenc')}
    expect = set()
    missing = 0
    for name, r in rows.items():
        if not r.get('sha256'):
            continue
        k = object_of(r, name)
        expect.add(k)
        if k not in listed:
            missing += 1
            r['status'] = r['state'] = 'missing'
            for f in ('sha256', 'bytes', 'gv', 'key'):
                r[f] = None
    return {'missing': missing, 'orphans': sorted(listed - expect)}

def snapshot(chunks, raw: list[dict], unit_listing: list[str], plans=(), began=None, read_meta=None, cs=None, counts=None, wrote=None) -> dict:
    rows, n = fold(chunks, plans, began, cs, wrote)
    c = check(rows, unit_listing)
    adopted = orphans = superseded = 0
    for k in c['orphans']:
        name = k.rsplit('/', 1)[-1][:-len('.lenc')]
        if (rows.get(name) or {}).get('sha256'):
            superseded += 1
            continue
        got = read_meta(k) if read_meta else None
        if got and got[3] == name:
            meta, sha, size = (got[0], got[1], got[2])
            u = meta.get('id') or {}
            rows[name] = settle({'unit': name, 'kind': u.get('kind'), 'day': common.day_of(u.get('daykey') or ''), 'area': meta.get('area'), 'model': u.get('model'), 'model_id': meta.get('model_id'), 'init': u.get('init'), 'run': meta.get('init'), 'lead_h': meta.get('lead_h'), 'src': meta.get('src'), 'box': u.get('box'), 'bbox': meta.get('bbox'), 'cs': meta.get('cs'), 'gv': meta.get('gv'), 'sv': (meta.get('score') or {}).get('sv'), 'sha256': sha, 'bytes': size, 'state': 'done', 'sec': None, 'wave': None, 'key': k, 'put': 'adopted', 'why': None, 'try': None, 'tries_cs': {}, 'lost_cs': {}}, cs)
            adopted += 1
        else:
            orphans += 1
    by = {}
    for x in rows.values():
        by[x['status']] = by.get(x['status'], 0) + 1
    return {'v': 2, 'rows': rows, 'raw': {x['day']: x for x in raw}, 'counts': {'units': len(rows), 'by_status': by, 'missing': c['missing'], 'orphans': orphans, 'adopted': adopted, 'superseded': superseded, 'raw_days': len(raw), **n, **(counts or {})}}

def summary(snap: dict) -> str:
    c = snap.get('counts') or {}
    by = ', '.join((f'{k} {v}' for k, v in sorted((c.get('by_status') or {}).items())))
    return f"catalog: {c.get('units', 0)} units ({by or 'none'}), {c.get('missing', 0)} missing, {c.get('orphans', 0)} orphan(s), {c.get('adopted', 0)} adopted, {c.get('superseded', 0)} superseded, {c.get('rows_unreadable', 0)} rows object(s) unreadable, {c.get('unknown_status', 0)} row(s) of an unknown status; {c.get('raw_days', 0)} raw days of {c.get('raw_parts', 0)} part(s), {c.get('raw_read', 0)} read, {c.get('raw_unreadable', 0)} unreadable, {c.get('raw_other', 0)} not days"

def usable(snap: dict, current_gv: dict | None=None, consumers=None) -> list[dict]:
    out = []
    for r in (snap.get('rows') or {}).values():
        if not r.get('sha256') or r.get('status') == 'missing':
            continue
        if current_gv is not None and consumers:
            cur = current_gv.get(common.gv_area(r.get('model') or '', r.get('box')))
            if not cur or common.stale_groups(r.get('gv'), cur, consumers):
                continue
        out.append(r)
    return out

def _put_timed(lake, prefix: str, plain: bytes, t: float | None=None):
    t0 = time.time() if t is None else t
    for n in range(120):
        key = f'{prefix}{stamp(t0 + n)}.lenc'
        res = lake.put(key, plain, strict=False)
        if res.status != 'differs':
            return (key, res)
    raise ValueError('catalog: no free name for a timed object')

def write_snapshot(lake, snap: dict, t: float | None=None) -> tuple[str, str]:
    key, res = _put_timed(lake, SNAPS, encode(snap), t)
    return (key, res.sha256)

def pin(lake, snap_key: str, sha256: str, why: str, t: float | None=None, **info) -> str:
    return _put_timed(lake, PINS, encode({'v': 1, 'snapshot': snap_key, 'sha256': sha256, 'why': why, **info}), t)[0]

def rows_digest(snap: dict) -> str:
    return hashlib.sha256(encode({'rows': snap.get('rows'), 'raw': snap.get('raw')})).hexdigest()

def pin_if_new(lake, snap: dict, why: str, t: float | None=None, **info) -> str | None:
    d = rows_digest(snap)
    keys = sorted((k for k in lake.list(PINS) if k.endswith('.lenc')))
    if keys and decode(lake.get(keys[-1])).get('digest') == d:
        return None
    key, sha = write_snapshot(lake, snap, t)
    return pin(lake, key, sha, why, t, digest=d, **info)

def pinned(lake, listing: list[str] | None=None) -> dict | None:
    from lake import LakeError, _sha_hex
    keys = sorted((k for k in (listing if listing is not None else lake.list(PINS)) if k.startswith(PINS) and k.endswith('.lenc')))
    if not keys:
        return None
    p = decode(lake.get(keys[-1]))
    blob = lake.store.get(p['snapshot'])
    if blob is None or _sha_hex(blob) != p.get('sha256'):
        raise LakeError('catalog: the pinned snapshot is missing or not the one pinned')
    return decode(lake.get(p['snapshot']))

def unit_meta_reader(lake):
    import io
    from lake import _sha_hex

    def read(key: str):
        try:
            blob = lake.store.get(key)
            if blob is None:
                return None
            import numpy as np
            with np.load(io.BytesIO(lake.get(key)), allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
            return (meta, _sha_hex(blob), len(blob), lake.name(logical_id(meta.get('id'))))
        except Exception:
            return None
    return read

def build(lake, cs: str | None=None, read_meta: bool=True, t: float | None=None) -> dict:
    row_keys = lake.list(ROWS)
    chunks, bad = read_rows(lake, row_keys)
    wrote = attempts(row_keys, ROWS)
    began = started(lake.list(STARTS))
    lost_plans = {(sw, w) for (sw, w, c), k in began.items() if k > wrote.get((sw, w, c), 0)}
    plans, bad_plans = read_plans(lake, [plan_key(sw, w) for sw, w in lost_plans if lake.exists(plan_key(sw, w))])
    raw, rn = raw_days(lake, t=t)
    return snapshot(chunks, raw, lake.list(UNITS), plans=plans, began=began, read_meta=unit_meta_reader(lake) if read_meta else None, cs=cs, wrote=wrote, counts={'rows_unreadable': bad, 'plans_unreadable': bad_plans, **rn})
