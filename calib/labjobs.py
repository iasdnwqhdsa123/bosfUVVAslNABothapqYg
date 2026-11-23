from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

class _OnUse:

    def __getattr__(self, name):
        import catalog as c
        return getattr(c, name)
catalog = _OnUse()
PAUSE_FILE = 'lab.paused'
SWEEP = re.compile('\\d{8}T\\d{6}Z')
STAMPS = 'derived/stamps/'
AREAS = ('US', 'EU', 'SH')
RECHECK_N = 30
RECHECK_START_S = 30 * 60
RECHECK_HARD_S = 55 * 60
WANTED = 'derived/lab/wanted/'
_PLAN = re.compile('(\\d{8}T\\d{6}Z)-(\\d+)\\.lenc')
_GV = re.compile('[0-9a-f]{12}')
_GROUP = re.compile('[a-z]+')

def _out(**kv) -> None:
    path = os.environ.get('GITHUB_OUTPUT')
    lines = ''.join((f'{k}={v}\n' for k, v in kv.items()))
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(lines)
    else:
        sys.stdout.write(lines)

def log(msg: str) -> None:
    print(f'lab: {msg}', flush=True)

def lake_of(local: str | None=None):
    import lake
    lk = lake.from_env('LAKE', local)
    return lk if lk is not None and 1 in lk.keys.ids() else None

def paused(here: str | None=None) -> bool:
    return os.path.exists(os.path.join(here or HERE, PAUSE_FILE))

def plans(lk) -> list[tuple[str, int]]:
    out = []
    for k in lk.list(catalog.PLANS):
        m = _PLAN.fullmatch(k[len(catalog.PLANS):])
        if m:
            out.append((m.group(1), int(m.group(2))))
    return sorted(out)

def begun(lk, sweep: str, wave: int) -> bool:
    return any((sw == sweep and w == wave for sw, w, _ in catalog.started(lk.list(f'{catalog.STARTS}{sweep}/'))))

def unrun(lk) -> tuple[str, int] | None:
    ps = plans(lk)
    if not ps:
        return None
    sw, w = ps[-1]
    return None if begun(lk, sw, w) else (sw, w)

def resolve(lk, what: str, sweep: str | None=None, wave: int | None=None) -> tuple[str, int] | None:
    if what == 'wave':
        if not sweep or not SWEEP.fullmatch(sweep) or wave is None or (wave < 0):
            raise ValueError('wave')
        return (sweep, int(wave))
    if what == 'start':
        if not sweep or not SWEEP.fullmatch(sweep):
            raise ValueError('sweep')
        if not any((sw == sweep for sw, _ in plans(lk))):
            return (sweep, 0)
    elif what != 'resume':
        raise ValueError('what')
    ps = [p for p in plans(lk) if what == 'resume' or p[0] == sweep]
    if not ps:
        return None
    sw, w = ps[-1]
    return (sw, w) if not begun(lk, sw, w) else (sw, w + 1)

def after(made: str, wave: int) -> int:
    if wave < 0:
        raise ValueError('wave')
    return wave + 1 if made in ('now', 'before') else wave

def parts(most: int, par: int) -> list[int]:
    return list(range(max(0, min(int(most), int(par)))))

def gv_in_force(lk) -> dict | None:
    for sw, w in reversed(plans(lk)):
        try:
            g = catalog.decode(lk.get(catalog.plan_key(sw, w))).get('gv')
        except Exception:
            continue
        if isinstance(g, dict) and g:
            return g
    return None

def read_request(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
    except (OSError, ValueError):
        raise ValueError('request') from None
    if not isinstance(r, dict):
        raise ValueError('request')
    what, sweep, oc = (r.get('what'), str(r.get('sweep') or ''), r.get('on_change') or '')
    if what not in ('start', 'resume', 'pause') or oc not in ('', 'finish', 'restart'):
        raise ValueError('request')
    if what == 'start' and (not SWEEP.fullmatch(sweep)):
        raise ValueError('request')
    at = str(r.get('at') or '')
    return {'what': what, 'sweep': sweep if what == 'start' else '', 'on_change': oc, 'at': at if re.fullmatch('\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z', at) else ''}
STANDING_S = 24 * 3600
_WANTED = re.compile('(\\d{8}T\\d{6}Z)\\.lenc')

def want(lk, sweep: str, on_change: str='', now: float | None=None) -> None:
    if not SWEEP.fullmatch(sweep or '') or on_change not in ('', 'finish', 'restart'):
        return
    t = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(time.time() if now is None else now))
    lk.put(f'{WANTED}{t}.lenc', json.dumps({'v': 1, 'sweep': sweep, 'on_change': on_change}).encode(), strict=False)

def wanted(lk, now: float | None=None) -> dict | None:
    import calendar
    for k in sorted(lk.list(WANTED), reverse=True):
        m = _WANTED.fullmatch(k[len(WANTED):])
        if not m:
            continue
        age = (time.time() if now is None else now) - calendar.timegm(time.strptime(m.group(1), '%Y%m%dT%H%M%SZ'))
        if not 0 <= age < STANDING_S:
            return None
        try:
            w = json.loads(lk.get(k))
        except Exception:
            continue
        if isinstance(w, dict) and SWEEP.fullmatch(str(w.get('sweep') or '')) and (w.get('on_change', '') in ('', 'finish', 'restart')):
            return {'sweep': w['sweep'], 'on_change': w.get('on_change', '')}
    return None

def _age_s(at: str, now: float | None=None) -> float | None:
    import calendar
    try:
        t = calendar.timegm(time.strptime(at, '%Y-%m-%dT%H:%M:%SZ'))
    except (TypeError, ValueError):
        return None
    return (time.time() if now is None else now) - t

def next_action(st: dict) -> str:
    if int(st.get('chunks') or 0) > 0:
        return 'dispatch'
    return {'noroom': 'wait', 'nostamps': 'fingerprint'}.get(st.get('why'), 'stop')

def stamps_key(area: str, cs: str) -> str:
    return f'{STAMPS}{area}/{cs}.lenc'

def refs_key(area: str, group: str, gv: str) -> str:
    if area not in AREAS or not _GROUP.fullmatch(group) or (not _GV.fullmatch(gv)):
        raise ValueError('refs')
    return f'{STAMPS}{area}/refs/{group}-{gv}.lenc'

def _npz(arrays: dict) -> bytes:
    import labbuild
    return labbuild.npz_bytes(arrays)

def _arrays(blob: bytes) -> dict:
    import io
    import numpy as np
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def need(lk, cs: str | None=None) -> bool:
    import labplan
    return labplan.read_stamps(lk, cs or catalog.source_stamp()) is None

def kept(lk, out: str) -> int:
    import fingerprint
    reps = {}
    for k in lk.list(STAMPS):
        parts = k[len(STAMPS):].split('/')
        if len(parts) != 3 or parts[0] not in AREAS or parts[1] != 'refs' or (not parts[2].endswith('.lenc')):
            continue
        g, _, v = parts[2][:-5].partition('-')
        if not _GROUP.fullmatch(g) or not _GV.fullmatch(v):
            continue
        reps[parts[0], g, v] = _arrays(lk.get(k))
    os.makedirs(out, exist_ok=True)
    fingerprint.save_group_reps(os.path.join(out, fingerprint.GROUP_REPS), reps)
    cur = gv_in_force(lk)
    if cur:
        with open(os.path.join(out, 'current.json'), 'w') as f:
            json.dump(cur, f, sort_keys=True)
    log(f'{len(reps)} kept group stamps; ' + ("the newest plan's in force" if cur else 'none in force yet'))
    return len(reps)

def write(lk, gv_dir: str, rechecks: str, parts: int, cs: str | None=None) -> bool:
    import fingerprint
    import labplan
    cs = cs or catalog.source_stamp()
    results = []
    for p in sorted(glob.glob(os.path.join(rechecks, '*.json'))):
        with open(p) as f:
            results.append(json.load(f))
    got = {r.get('part') for r in results}
    if got != set(range(parts)):
        log(f'the rechecks came back from {len(got)} of {parts} parts: no stamps written')
        return False
    tot = {k: sum((int(r.get(k) or 0) for r in results)) for k in ('picked', 'same', 'moved', 'missed', 'failed', 'nogrids', 'left')}
    groups = {}
    for r in results:
        for g, n in (r.get('groups') or {}).items():
            groups[g] = groups.get(g, 0) + int(n)
    log('recheck: ' + ', '.join((f'{k} {v}' for k, v in tot.items())))
    if tot['missed']:
        log(f"the reference boxes missed a change on {tot['missed']} stored unit(s) (" + ', '.join((f'{g} {n}' for g, n in sorted(groups.items()))) + '): no stamps written; look before a sweep builds on them')
        return False
    tried = tot['picked'] - tot['nogrids'] - tot['moved'] - tot['left']
    if tried > 0 and tot['failed'] * 2 > tried:
        log(f"too few units rebuilt to tell ({tot['failed']} of {tried} failed): no stamps written")
        return False
    try:
        with open(os.path.join(gv_dir, 'gv.json')) as f:
            gv = json.load(f)
    except (OSError, ValueError):
        log('no stamps were made (a reference box failed): none written')
        return False
    if set(gv) != set(AREAS) or not all((isinstance(gv[a], dict) and any(gv[a].values()) for a in AREAS)):
        log("the stamps don't cover every area: none written")
        return False
    reps = fingerprint.load_group_reps(os.path.join(gv_dir, fingerprint.GROUP_REPS))
    refs = {}
    for a in AREAS:
        for g, v in sorted(gv[a].items()):
            if not v:
                continue
            k = refs_key(a, g, v)
            refs.setdefault(a, {})[g] = k
            if lk.exists(k):
                continue
            if (a, g, v) not in reps:
                log("a stamp's reference fields are missing: none written")
                return False
            lk.put(k, _npz(reps[a, g, v]), strict=False)
    for a in AREAS:
        lk.put(stamps_key(a, cs), catalog.encode({'v': 1, 'cs': cs, 'gv': gv[a], 'refs': refs.get(a, {})}), strict=False)
    stood = labplan.read_stamps(lk, cs)
    if stood is None:
        log('the stamps did not read back: the planner takes none for this code')
        return False
    log('stamps written for this code in every area' + ('' if stood == gv else ' (others stood already: theirs kept)'))
    return True

def pick(lk, n: int=RECHECK_N, snap: dict | None=None) -> list[dict]:
    import labbuild
    import labplan
    import common
    snap = snap or catalog.build(lk)
    rows = snap.get('rows') or {}
    strata = {}
    for r in (snap.get('raw') or {}).values():
        for u in labplan.units_of_day(r, lk.name):
            row = rows.get(u['name'])
            if not row or row.get('status') != 'done' or (not row.get('sha256')):
                continue
            if common.is_lead(u['run']) and common.base_model(u['run']) not in labbuild.dayb.LAB_LEAD_GRIDS:
                continue
            strata.setdefault((u['area'], u['model'], labplan.tier(u)), []).append({**u, 'key': catalog.object_of(row, u['name'])})
    lists = [sorted(strata[k], key=lambda u: u['name']) for k in sorted(strata)]
    out, i = ([], 0)
    while len(out) < n and any((i < len(x) for x in lists)):
        for x in lists:
            if i < len(x) and len(out) < n:
                out.append(x[i])
        i += 1
    return out

def recheck_one(lk, u: dict, gv_now: dict, steps, work: str, cs: str, timeout: float | None=None) -> tuple[str, list[str]]:
    import shutil
    import common
    import fingerprint
    import labbuild
    stored = labbuild.read_unit(lk.get(u['key']))
    sgv = (stored.get('meta') or {}).get('gv') or {}
    now = gv_now.get(u['area']) or {}
    held = {g for g in common.GROUPS if any((k.startswith(f'g.{g}.') for k in stored))}
    if not held:
        return ('nogrids', [])
    kept_groups = sorted((g for g in held if sgv.get(g) and sgv.get(g) == now.get(g)))
    if not kept_groups:
        return ('moved', [])
    wd = tempfile.mkdtemp(prefix='rc-', dir=work)
    try:
        kind = u['unit']['kind']
        items = [f for f in labbuild.load_items(lk, u.get('raw') or []) if (f.get('kind') == 'collector') == (kind == 'collector')]
        B = next((b for b in labbuild.dayb.boxes(items) if b['region'] == u['box']), None)
        if B is None:
            return ('failed', [])
        model_id = labbuild.model_id_of(u['unit'])
        limit = labbuild.dayb.BUILD_TIMEOUT if timeout is None else min(labbuild.dayb.BUILD_TIMEOUT, timeout)
        base = steps.build(common.day_of(u['unit']['daykey']), B, model_id, wd, limit)
        if not base:
            return ('failed', [])
        arrays = labbuild.unit_arrays(base, u['unit'], model_id, B['bbox'], now, cs)
    except Exception:
        return ('failed', [])
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    missed = []
    for g in kept_groups:
        a = {f"{g}/{k.rsplit('.', 1)[1]}": v for k, v in stored.items() if k.startswith(f'g.{g}.')}
        b = {f"{g}/{k.rsplit('.', 1)[1]}": v for k, v in arrays.items() if k.startswith(f'g.{g}.')}
        if not fingerprint.group_close(a, b):
            missed.append(g)
    return ('missed' if missed else 'same', missed)

def recheck(lk, gv_path: str, part: int, parts: int, out: str, n: int=RECHECK_N, steps=None, work: str | None=None, clock=time.monotonic, elapsed: float=0.0, cs: str | None=None) -> dict:
    import shutil
    import labbuild
    with open(gv_path) as f:
        gv_now = json.load(f)
    cs = cs or catalog.source_stamp()
    steps = steps or labbuild.Steps()
    t0 = clock() - max(0.0, elapsed)
    mine = pick(lk, n)[part::parts]
    res = {'v': 1, 'part': part, 'picked': len(mine), 'same': 0, 'moved': 0, 'missed': 0, 'failed': 0, 'nogrids': 0, 'left': 0, 'groups': {}}
    own = work is None
    work = work or tempfile.mkdtemp(prefix='lab-rc-')
    os.makedirs(work, exist_ok=True)
    try:
        for i, u in enumerate(mine):
            if clock() - t0 >= RECHECK_START_S:
                res['left'] = len(mine) - i
                break
            try:
                st, gs = recheck_one(lk, u, gv_now, steps, work, cs, RECHECK_HARD_S - (clock() - t0))
            except Exception:
                st, gs = ('failed', [])
            res[st] += 1
            for g in gs:
                res['groups'][g] = res['groups'].get(g, 0) + 1
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)
    with open(out, 'w') as f:
        json.dump(res, f, sort_keys=True)
    log(f'recheck part {part}: ' + ', '.join((f'{k} {res[k]}' for k in ('picked', 'same', 'moved', 'missed', 'failed', 'nogrids', 'left'))))
    return res

def relay(lk, what: str='none', sweep: str | None=None, then: str='none', here: str | None=None, cs: str | None=None, standing: dict | None=None, now: float | None=None) -> tuple[tuple[str, int] | None, str]:
    import labplan
    if paused(here):
        return (None, 'paused (lab.paused)')
    if labplan.read_stamps(lk, cs or catalog.source_stamp()) is None and (not gv_in_force(lk)):
        return (None, 'no code has stamps yet')
    if what == 'pause':
        return (None, 'the request was a pause')
    if what in ('start', 'resume'):
        r = resolve(lk, what, sweep)
        return (r, f'the request ({what})' if r else 'no sweep to resume')
    if then == 'resume':
        r = resolve(lk, 'start', sweep) if sweep else resolve(lk, 'resume')
        return (r, 'lab-build asked for the stamps' if r else 'no sweep to resume')
    r = unrun(lk)
    if r:
        return (r, 'the newest wave was planned and never ran')
    s = standing or {}
    age = _age_s(s.get('at'), now)
    if age is not None and 0 <= age < STANDING_S:
        if s.get('what') == 'start' and (not any((sw == s['sweep'] for sw, _ in plans(lk)))):
            return ((s['sweep'], 0), "the starter's start, not begun yet")
        if s.get('what') == 'resume':
            r = resolve(lk, 'resume')
            if r:
                return (r, "the starter's resume, asked for under a day ago")
    w = wanted(lk, now)
    if w:
        return (resolve(lk, 'start', w['sweep']), 'the sweep a relay was asked for under a day ago')
    return (None, 'nothing waiting')

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('check', 'resolve', 'need', 'kept', 'recheck', 'write', 'relay'):
        p = sub.add_parser(name)
        p.add_argument('--local', help='a folder as the lake (tests, a dry run)')
        if name == 'resolve':
            p.add_argument('--what', required=True, choices=('start', 'resume', 'wave'))
            p.add_argument('--sweep')
            p.add_argument('--wave', type=int)
        elif name == 'kept':
            p.add_argument('--out', required=True)
        elif name == 'recheck':
            p.add_argument('--gv', required=True)
            p.add_argument('--part', required=True)
            p.add_argument('--out', required=True)
            p.add_argument('--n', type=int, default=RECHECK_N)
            p.add_argument('--work')
            p.add_argument('--started', type=float)
        elif name == 'write':
            p.add_argument('--gv', required=True)
            p.add_argument('--rechecks', required=True)
            p.add_argument('--parts', type=int, required=True)
        elif name == 'relay':
            p.add_argument('--what', default='none', choices=('start', 'resume', 'pause', 'none'))
            p.add_argument('--sweep')
            p.add_argument('--then', default='none', choices=('resume', 'none'))
            p.add_argument('--on-change', default='', choices=('', 'finish', 'restart'))
            p.add_argument('--request', help='the request file as it stands (calib/sweep.request)')
    nx = sub.add_parser('next')
    nx.add_argument('--status', required=True)
    af = sub.add_parser('after')
    af.add_argument('--made', required=True, choices=('now', 'before', 'nothing', 'none', ''))
    af.add_argument('--wave', type=int, required=True)
    pa = sub.add_parser('parts')
    pa.add_argument('--most', type=int, required=True)
    pa.add_argument('--par', type=int, help="the room given (tests): else room.lab_par's")
    rq = sub.add_parser('request')
    rq.add_argument('--file', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'request':
        try:
            r = read_request(a.file)
        except ValueError:
            log("the sweep's request doesn't read")
            return 1
        log(f"request: {r['what']}" + (f", sweep {r['sweep']}" if r['sweep'] else '') + (f", {r['on_change']}" if r['on_change'] else ''))
        _out(what=r['what'], sweep=r['sweep'], on_change=r['on_change'])
        return 0
    if a.cmd == 'after':
        try:
            w = after(a.made, a.wave)
        except ValueError:
            log('a wave is 0 or more')
            return 2
        log(f'the next job plans wave {w}')
        _out(wave=w)
        return 0
    if a.cmd == 'parts':
        if a.par is not None:
            par, why = (max(0, a.par), 'given')
        else:
            import labplan
            par, why = labplan._par(None)
        ps = parts(a.most, par)
        log(f'recheck: {len(ps)} part(s) ({why})')
        _out(parts=json.dumps(ps), n=len(ps))
        return 0
    if a.cmd == 'next':
        with open(a.status) as f:
            st = json.load(f)
        act = next_action(st)
        m = re.fullmatch('(\\d{8}T\\d{6}Z)-(\\d+)', str(st.get('plan') or ''))
        sw, w = (m.group(1), int(m.group(2))) if m else ('', -1)
        log(f"next: {act} (wave {w}: {st.get('chunks', 0)} chunks, {st.get('left', 0)} left after it, par {st.get('par', 0)}" + (f", {st['why']}" if st.get('why') else '') + ')')
        _out(action=act, sweep=sw, wave=w)
        return 0
    lk = lake_of(a.local)
    if a.cmd == 'check':
        log('the lake is set' if lk else 'no lake (LAKE_S3_* or CALIB_DATA_KEY unset): nothing to do')
        _out(ready='yes' if lk else 'no')
        return 0
    if lk is None:
        log('no lake (LAKE_S3_* or CALIB_DATA_KEY unset): nothing done')
        _out(**{'go': 'no'} if a.cmd == 'resolve' else {'need': 'no'} if a.cmd == 'need' else {'dispatch': 'no'} if a.cmd == 'relay' else {})
        return 0
    if a.cmd == 'resolve':
        try:
            r = resolve(lk, a.what, a.sweep, a.wave)
        except ValueError:
            log('a sweep is its start time (YYYYMMDDTHHMMSSZ), a wave 0 or more')
            _out(go='no')
            return 1
        log(f'sweep {r[0]}, wave {r[1]}' if r else 'no sweep to resume')
        _out(go='yes' if r else 'no', sweep=r[0] if r else '', wave=r[1] if r else '')
        return 0
    if a.cmd == 'need':
        n = need(lk)
        log('this code has no group stamps yet: the reference boxes are built' if n else "this code's group stamps stand")
        _out(need='yes' if n else 'no')
        return 0
    if a.cmd == 'kept':
        kept(lk, a.out)
        return 0
    if a.cmd == 'recheck':
        m = re.fullmatch('(\\d+)/(\\d+)', a.part)
        if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
            log('a part is i/n')
            return 2
        recheck(lk, a.gv, int(m.group(1)), int(m.group(2)), a.out, a.n, work=a.work, elapsed=time.time() - a.started if a.started else 0.0)
        return 0
    if a.cmd == 'write':
        return 0 if write(lk, a.gv, a.rechecks, a.parts) else 1
    if a.sweep is not None and (not SWEEP.fullmatch(a.sweep)):
        log('a sweep is its start time (YYYYMMDDTHHMMSSZ)')
        _out(dispatch='no')
        return 1
    if a.what == 'start' and (not a.sweep):
        log('a start names its sweep')
        _out(dispatch='no')
        return 1
    standing = None
    if a.request and os.path.exists(a.request):
        try:
            standing = read_request(a.request)
        except ValueError:
            standing = None
    r, why = relay(lk, a.what, a.sweep, a.then, standing=standing)
    oc = a.on_change
    if r and a.what == 'none' and (a.then == 'none'):
        if why.startswith("the starter's") and standing:
            oc = standing['on_change']
        elif why.startswith('the sweep a relay'):
            oc = (wanted(lk) or {}).get('on_change', '')
    if r and (a.what in ('start', 'resume') or a.then == 'resume'):
        want(lk, r[0], oc)
    log(f'relay: {why}' + (f': lab-build for sweep {r[0]}, wave {r[1]}' if r else ''))
    _out(dispatch='yes' if r else 'no', sweep=r[0] if r else '', wave=r[1] if r else '', on_change=oc if r else '')
    return 0
if __name__ == '__main__':
    sys.exit(main())
