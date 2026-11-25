from __future__ import annotations
import argparse
import json
import os
import re
import statistics
import sys
from datetime import date, datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_eu
import catalog
import common
import stationcast
import omruns
CHUNK_S = 40 * 60
SETUP_S = 4 * 60
MIN_SAMPLES = 5
MIN_S = 5.0
OM_PAR = 6
EARLY_OLD = 6
MAX_TRIES = catalog.MAX_TRIES
AREAS = ('US', 'EU', 'SH')
DEFAULT_S = {('own', 'hrrr'): 300, ('own', 'gfs'): 90, ('own', 'ecmwf'): 120, ('own', 'icon'): 150, ('own', 'icond2'): 330, ('own', 'arome'): 240, ('lead', 'hrrr'): 300, ('lead', 'gfs'): 90, ('lead', 'ecmwf'): 150, (catalog.STCAST, 'hrrr'): 1200, (catalog.STCAST, 'gfs'): 200, (catalog.STCAST, 'ecmwf'): 600, (catalog.STCAST, 'icon'): 60, (catalog.STCAST, 'icond2'): 20, (catalog.STCAST, 'arome'): 20}
FALLBACK_S = 200
OM_MODELS = ('icon', 'icond2', 'arome')
EXTRA_FIRST = {'ecmwf': common.EU_START, 'icon': common.ICON_FIRST}

def _extra(text: str | None=None) -> dict[str, tuple]:
    t = os.environ.get('CALIB_LAB_EXTRA', 'US:ecmwf+icon') if text is None else text
    out = {}
    for part in t.split(','):
        a, _, ms = part.partition(':')
        if a.strip() and ms.strip():
            out[a.strip()] = tuple((m for m in ms.split('+') if m.strip() in EXTRA_FIRST))
    return out
EXTRA = _extra()

def models_of(day: str, region: str, extra: dict | None=None) -> list[str]:
    ms = list(common.models_for([region], day))
    area = common.macro_of(region)
    for m in (EXTRA if extra is None else extra).get(area, ()):
        if m not in ms and day >= EXTRA_FIRST.get(m, ''):
            ms.append(m)
    return ms

def units_of_day(raw: dict, name_of, extra: dict | None=None, counts: dict | None=None) -> list[dict]:
    key = raw['day']
    d = common.day_of(key)
    out = []
    for kind, bs in sorted((raw.get('boxes') or {}).items()):
        for b in bs:
            region, bbox = (b['box'], b.get('bbox'))
            for m in models_of(d, region, extra):
                base = common.base_model(m)
                if bbox and (not build_eu.covers(base, bbox)):
                    continue
                try:
                    u = catalog.unit_of(key, m, region, kind)
                except ValueError as e:
                    if counts is not None:
                        k = 'no_lead_run' if str(e) == 'lead' else 'bad_unit'
                        counts[k] = counts.get(k, 0) + 1
                    continue
                out.append({'unit': u, 'name': name_of(catalog.logical_id(u)), 'raw': list(raw.get('parts') or []), 'kind': kind, 'day': key, 'area': common.gv_area(base, region), 'model': base, 'run': m, 'lead_d': catalog.lead_days(u), 'box': region, 'bbox': bbox, 'tz': b.get('tz')})
    return out

def to_do(units: list[dict], rows: dict[str, dict], pin: dict, consumers=None) -> tuple[list[dict], dict]:
    if consumers is None:
        consumers = tuple(common.CONSUMERS)
    elif not callable(consumers):
        consumers = tuple(consumers)
    gv, cs = (pin.get('gv') or {}, pin.get('cs'))
    out, n = ([], {'units': len(units), 'current': 0, 'nocover': 0, 'failed': 0, 'no_stamps': 0, 'missing': 0, 'stale': 0, 'tried': 0})
    for u in units:
        r = rows.get(u['name'])
        r = catalog.settle(dict(r), cs) if r else {}
        if r.get('status') == 'nocover':
            n['nocover'] += 1
            continue
        cur = gv.get(u['area'])
        if not cur:
            n['no_stamps'] += 1
            continue
        tries = int(r.get('tries') or 0)
        if tries >= MAX_TRIES:
            n['failed'] += 1
            continue
        if r.get('sha256'):
            cons = consumers(cur) if callable(consumers) else consumers
            hours_moved = u.get('want_h') is not None and list(r.get('lead_h') or []) != list(u['want_h'])
            if not hours_moved and (not common.stale_groups(r.get('gv'), cur, cons)):
                n['current'] += 1
                continue
            why = 'stale'
        else:
            why = 'tried' if tries else 'missing'
        n[why] += 1
        e = {**u, 'why': why, 'tries': tries, 'lost': int(r.get('lost') or 0)}
        if r.get('sha256'):
            e['prev'] = {'key': catalog.object_of(r, u['name']), 'sha256': r.get('sha256'), 'gv': r.get('gv'), 'cs': r.get('cs')}
        out.append(e)
    return (out, n)

def tier(u: dict) -> int:
    if u.get('kind') == catalog.STCAST:
        return 1
    return 0 if u.get('lead_d') is None or u['lead_d'] <= 1 else 1

def is_stcast(u: dict) -> bool:
    return u.get('kind') == catalog.STCAST

def om_sourced(u: dict) -> bool:
    return u['model'] in OM_MODELS or (u['model'] == 'ecmwf' and u.get('lead_d') is not None)
_AREAS = {'US': 0, 'EU': 1, 'SH': 2}

def _key(u: dict) -> tuple:
    d = date.fromisoformat(common.day_of(u['day'])).toordinal()
    return (tier(u), -d, _AREAS.get(common.macro_of(u['box']), 9), u['box'], u.get('kind') or '', common.MODEL_INDEX.get(u['model'], 99), u.get('lead_d') or 0, u['name'])

def _st_key(u: dict) -> tuple:
    return (-int(u['unit']['init']), _AREAS.get(u['box'], 9), common.MODEL_INDEX.get(u['model'], 99), u['name'])

def side_by_side(a: list[dict], b: list[dict], table: dict) -> list[dict]:
    out, i, j, ta, tb = ([], 0, 0, 0.0, 0.0)
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and ta <= tb):
            out.append(a[i])
            ta += est(a[i], table)
            i += 1
        else:
            out.append(b[j])
            tb += est(b[j], table)
            j += 1
    return out

def order(units: list[dict], early_old: int=0, table: dict | None=None) -> list[dict]:
    st = sorted((u for u in units if is_stcast(u)), key=_st_key)
    ranked = sorted((u for u in units if not is_stcast(u)), key=_key)
    om = sorted((u for u in ranked if om_sourced(u)), key=lambda u: (common.day_of(u['day']), _key(u)))[:max(0, early_old)]
    first = {u['name'] for u in om}
    rest = [u for u in ranked if u['name'] not in first]
    if st:
        rest = st + rest if stcast_first() else side_by_side(rest, st, table or DEFAULT_S)
    return om + rest

def stcast_first() -> bool:
    return (os.environ.get('CALIB_STCAST_ORDER') or 'first').strip().lower() != 'turns'

def timing(model: str | None, init: str | None, kind: str | None=None) -> tuple:
    if kind == catalog.STCAST:
        return (catalog.STCAST, common.base_model(model or ''))
    return ('own' if init in (None, 'own') else 'lead', common.base_model(model or ''))

def seconds(rows: dict[str, dict], min_samples: int=MIN_SAMPLES) -> dict[tuple, float]:
    got: dict[tuple, list] = {}
    for r in rows.values():
        if r.get('status') == 'done' and r.get('put') != 'kept' and isinstance(r.get('sec'), (int, float)) and (r['sec'] > 0):
            got.setdefault(timing(r.get('model'), r.get('init'), r.get('kind')), []).append(float(r['sec']))
    out = dict(DEFAULT_S)
    for k, v in got.items():
        if len(v) >= min_samples:
            out[k] = max(MIN_S, statistics.median(v))
    return out

def est(u: dict, table: dict) -> float:
    if is_stcast(u):
        return float(table.get((catalog.STCAST, u['model']), FALLBACK_S))
    return float(table.get(('own' if u.get('lead_d') is None else 'lead', u['model']), FALLBACK_S))

def chunks(units: list[dict], table: dict, budget: float=CHUNK_S - SETUP_S) -> list[list[dict]]:
    place = {u['name']: i for i, u in enumerate(units)}
    out = []
    for stream in ([u for u in units if om_sourced(u)], [u for u in units if is_stcast(u) and (not om_sourced(u))], [u for u in units if not om_sourced(u) and (not is_stcast(u))]):
        cur, t = ([], 0.0)
        for u in stream:
            s = est(u, table)
            if u.get('lost'):
                out.append([{**u, 'est_s': round(s)}])
                continue
            if cur and t + s > budget:
                out.append(cur)
                cur, t = ([], 0.0)
            cur.append({**u, 'est_s': round(s)})
            t += s
        if cur:
            out.append(cur)
    return sorted(out, key=lambda c: place[c[0]['name']])

def wave(cs: list[list[dict]], par: int, om_par: int=OM_PAR) -> list[list[dict]]:
    out, om = ([], 0)
    for c in cs:
        if len(out) >= max(0, par):
            break
        if om_sourced(c[0]):
            if om >= om_par:
                continue
            om += 1
        out.append(c)
    return out

def changed_groups(old: dict, new: dict, consumers=None) -> dict[str, set]:
    consumers = tuple(common.CONSUMERS) if consumers is None else tuple(consumers)
    groups = common.consumed(consumers)
    out = {}
    for a in sorted(set(old or {}) | set(new or {})):
        o, n = ((old or {}).get(a) or {}, (new or {}).get(a) or {})
        diff = {g for g in groups if (o.get(g) is not None or n.get(g) is not None) and o.get(g) != n.get(g)}
        if diff:
            out[a] = diff
    return out

def pin_rule(pin: dict | None, now: dict, on_change: str='finish', consumers=None) -> tuple[str, dict]:
    if on_change not in ('finish', 'restart'):
        raise ValueError('labplan: on_change is finish or restart')
    if pin is None:
        return ('start', dict(now)) if now.get('gv') else ('wait', {})
    if now.get('code_sha') == pin.get('code_sha') and now.get('cs') == pin.get('cs'):
        return ('same', pin)
    if not now.get('gv'):
        return ('wait', pin)
    if not changed_groups(pin.get('gv'), now['gv'], consumers):
        return ('forward', dict(now))
    return ('restart', dict(now)) if on_change == 'restart' else ('finish', pin)

def plan(raw: list[dict], rows: dict[str, dict], prev_pin: dict | None, now: dict, par: int, sweep: str, wave_no: int, name_of, on_change: str='finish', paused: bool=False, consumers=None, extra: dict | None=None, early_old: int=EARLY_OLD, om_par: int=OM_PAR, stcast: list[dict] | None=None) -> dict:
    decision, pin = pin_rule(prev_pin, now, on_change, consumers)
    bad = {'bad_unit': 0, 'no_lead_run': 0}
    units = [u for r in raw for u in units_of_day(r, name_of, extra, bad)]
    if decision == 'finish' and stcast and to_do(stcast, rows, now, catalog.stcast_consumers)[0]:
        decision, pin = ('restart', dict(now))
    hold_st = decision == 'wait'

    def what(p):
        todo, counts = to_do(units, rows, p, consumers)
        counts.update(bad)
        if stcast is not None:
            t2, c2 = to_do(stcast, rows, p, catalog.stcast_consumers)
            if not hold_st:
                todo += t2
            counts = {k: counts.get(k, 0) + c2.get(k, 0) for k in counts}
            counts['stcast'], counts['stcast_todo'] = (c2['units'], len(t2))
            if hold_st:
                counts['stcast_held'] = len(t2)
        return (todo, counts)
    todo, counts = what(pin or {})
    if decision == 'finish' and (not todo):
        decision, pin = ('finished', dict(now))
        todo, counts = what(pin)
    base = {'v': 1, 'id': catalog.plan_id(sweep, wave_no), 'sweep': sweep, 'wave': int(wave_no), 'code_sha': (pin or {}).get('code_sha'), 'cs': (pin or {}).get('cs'), 'gv': (pin or {}).get('gv'), 'decision': decision, 'on_change': on_change, 'par': int(par), 'paused': bool(paused), 'todo': len(todo)}
    if paused or par <= 0 or (not pin):
        return {**base, 'chunks': [], 'counts': counts, 'left': len(todo)}
    table = seconds(rows)
    cs = wave(chunks(order(todo, early_old if int(wave_no) == 0 else 0, table), table), par, om_par)
    n = sum((len(c) for c in cs))
    return {**base, 'chunks': cs, 'counts': {**counts, 'planned': n, 'chunks': len(cs)}, 'left': len(todo) - n}

def pin_of(p: dict | None) -> dict | None:
    if not p or not p.get('cs') or (not p.get('gv')):
        return None
    return {'code_sha': p.get('code_sha'), 'cs': p['cs'], 'gv': p['gv']}

def done_pinning(p: dict) -> bool:
    return p.get('decision') == 'finished' or (bool(p.get('cs')) and p.get('todo') == 0)

def matrix(p: dict) -> list[int]:
    return list(range(len(p.get('chunks') or [])))

def write_plan(lake, p: dict):
    return lake.put(catalog.plan_key(p['sweep'], p['wave']), catalog.encode(p))

def read_plan(lake, sweep: str, wave_no: int) -> dict:
    return catalog.load_plan(lake, catalog.plan_id(sweep, wave_no))

def prev_plan(lake, sweep: str, wave_no: int) -> dict | None:
    for w in range(int(wave_no) - 1, -1, -1):
        k = catalog.plan_key(sweep, w)
        if lake.exists(k):
            return catalog.decode(lake.get(k))
    return None
STAMPS = 'derived/stamps/'

def _stamps_of(blob: bytes) -> dict | None:
    try:
        t = catalog.decode(blob)
    except Exception:
        t = None
        try:
            import io
            import numpy as np
            with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                t = {'gv': json.loads(str(z['gv']))}
        except Exception:
            return None
    if not isinstance(t, dict):
        return None
    g = t.get('gv', t)
    return g if isinstance(g, dict) else None

def read_stamps(lake, cs: str) -> dict | None:
    out = {}
    for k in lake.list(STAMPS):
        parts = k[len(STAMPS):].split('/')
        if len(parts) != 2 or parts[1] != f'{cs}.lenc' or parts[0] not in AREAS:
            continue
        try:
            g = _stamps_of(lake.get(k))
        except Exception:
            g = None
        if g:
            out[parts[0]] = g
    return out if set(out) == set(AREAS) else None

def log_line(p: dict) -> str:
    c = p.get('counts') or {}
    why = 'paused' if p.get('paused') else 'no room (par 0)' if p.get('par', 0) <= 0 else 'no stamps' if not p.get('cs') else ''
    return f"plan: wave {p['wave']}, {p['decision']}, par {p['par']}: {c.get('planned', 0)} units in {c.get('chunks', 0)} chunks{(' (' + why + ')' if why else '')}; {p.get('left', 0)} left; of {c.get('units', 0)}: {c.get('current', 0)} current, {c.get('missing', 0)} missing, {c.get('stale', 0)} stale, {c.get('tried', 0)} tried, {c.get('failed', 0)} failed, {c.get('nocover', 0)} not covered, {c.get('no_stamps', 0)} without stamps; {c.get('bad_unit', 0)} refused by the unit check, {c.get('no_lead_run', 0)} leads without a run hour" + (f"; station hindcast: {c['stcast']} units, {c.get('stcast_todo', 0)} to do" if 'stcast' in c else '') + (" (held for the new code's stamps)" if c.get('stcast_held') else '')

def _output(**kv) -> None:
    path = os.environ.get('GITHUB_OUTPUT')
    lines = ''.join((f'{k}={v}\n' for k, v in kv.items()))
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(lines)
    else:
        sys.stdout.write(lines)

def _par(arg: int | None) -> tuple[int, str]:
    if arg is not None:
        return (max(0, arg), 'given')
    import room
    import verifyjob
    try:
        running, waiting = verifyjob.jobs_going([verifyjob.source(), os.environ['GITHUB_REPOSITORY']])
    except Exception:
        running = waiting = None
    return room.lab_par(datetime.now(timezone.utc), room.slot_table(), running, waiting)

def status(p: dict | None, made: str) -> dict:
    p = p or {}
    n = len(p.get('chunks') or [])
    if n:
        why = ''
    elif made == 'none':
        why = 'nolake'
    elif p.get('paused'):
        why = 'paused'
    elif int(p.get('par') or 0) <= 0:
        why = 'noroom'
    elif not p.get('cs'):
        why = 'nostamps'
    else:
        why = 'done'
    return {'v': 1, 'plan': p.get('id'), 'wave': p.get('wave'), 'made': made if n or made != 'now' else 'nothing', 'chunks': n, 'todo': int(p.get('todo') or 0), 'left': int(p.get('left') or 0), 'par': int(p.get('par') or 0), 'decision': p.get('decision'), 'code_sha': p.get('code_sha') if n else None, 'why': why}

def _status(path: str | None, p: dict | None, made: str) -> None:
    if path:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(status(p, made), f, sort_keys=True)
            f.write('\n')

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('plan')
    a.add_argument('--sweep', required=True)
    a.add_argument('--wave', type=int, required=True)
    a.add_argument('--on-change', default=None, choices=('finish', 'restart'))
    a.add_argument('--par', type=int, default=None)
    a.add_argument('--local', default=None, help='a folder as the store (tests)')
    a.add_argument('--pause-file', default='paused')
    a.add_argument('--status', default=None)
    a.add_argument('--code-sha', default=None, help="the commit checked out (GITHUB_SHA unless given: a job that checks out main's newest)")
    a = ap.parse_args(argv)
    if not catalog.SWEEP.fullmatch(a.sweep) or a.wave < 0:
        print('labplan: a sweep id is its start time (YYYYMMDDTHHMMSSZ), a wave 0 or more', file=sys.stderr)
        return 2
    if os.path.basename(a.pause_file) != a.pause_file or a.pause_file in ('', '.', '..'):
        print('labplan: the pause file is a name beside labplan.py', file=sys.stderr)
        return 2
    if a.code_sha is not None and (not re.fullmatch('[0-9a-f]{40}', a.code_sha)):
        print('labplan: a commit is 40 hex digits', file=sys.stderr)
        return 2
    pid = catalog.plan_id(a.sweep, a.wave)
    import lake as lk
    store = lk.from_env('LAKE', a.local)
    if store is None or 1 not in store.keys.ids():
        print('labplan: no lake (its secrets or CALIB_DATA_KEY unset): nothing planned')
        _output(plan=pid, chunks='[]', wave=a.wave, more='no')
        _status(a.status, {'id': pid, 'wave': a.wave}, 'none')
        return 0
    if store.exists(catalog.plan_key(a.sweep, a.wave)):
        p = read_plan(store, a.sweep, a.wave)
        print("labplan: this wave's plan was made already: its chunks again")
        print(log_line(p))
        _output(plan=pid, chunks=json.dumps(matrix(p)), wave=a.wave, more='yes' if p.get('chunks') else 'no')
        _status(a.status, p, 'before')
        return 0
    paused = os.path.exists(os.path.join(HERE, a.pause_file))
    par, why = (0, 'paused') if paused else _par(a.par)
    print(f'labplan: {why}')
    cs = catalog.source_stamp()
    now = {'code_sha': a.code_sha or os.environ.get('GITHUB_SHA') or 'local', 'cs': cs, 'gv': read_stamps(store, cs)}
    before = prev_plan(store, a.sweep, a.wave)
    on_change = a.on_change or (before or {}).get('on_change') or 'finish'
    _, pin = pin_rule(pin_of(before), now, on_change)
    snap = catalog.build(store, cs=(pin or {}).get('cs'))
    print(catalog.summary(snap))
    st, st_bad = (None, {})
    if stationcast.enabled():
        try:
            st = stationcast.units(store.name, counts=st_bad)
        except ValueError:
            print("labplan: CALIB_STCAST_MONTHS doesn't read (YYYY-MM:YYYY-MM): no station hindcast planned")
        if omruns.enabled():
            st = (st or []) + omruns.units(store.name, counts=st_bad)
    p = plan(list(snap['raw'].values()), snap['rows'], pin_of(before), now, par, a.sweep, a.wave, store.name, on_change, paused, stcast=st)
    if st_bad.get('bad_unit'):
        p['counts']['bad_unit'] = p['counts'].get('bad_unit', 0) + st_bad['bad_unit']
    if p['chunks']:
        write_plan(store, p)
    if done_pinning(p) and catalog.pin_if_new(store, snap, 'sweep done', sweep=a.sweep, wave=a.wave):
        print("labplan: a sweep's pin has nothing left: the catalog pinned")
    elif 'stcast' in p['counts'] and catalog.pin_hindcast(store, snap, p['counts'].get('stcast_todo') == 0, sweep=a.sweep, wave=a.wave):
        print(f"labplan: the station hindcast's units pinned for the pairs ({p['counts'].get('stcast_todo')} to do)")
    print(log_line(p))
    print(store.summary())
    _output(plan=pid, chunks=json.dumps(matrix(p)), wave=a.wave, more='yes' if p['chunks'] else 'no')
    _status(a.status, p, 'now')
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
