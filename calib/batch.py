from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import time
from datetime import datetime, timezone
import common
import day as dayrun
import run as runmod
import state as statemod
import weglide
import workq
MAX_RUNS_A_DAY = 40
MIN_REQUESTS = 20
KEEP_LOGS = 80
RESCORE_COST = 0.3
GRIDS_PREFIX = 'calib-grids-'

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def _today():
    return datetime.now(timezone.utc).date()

def _output(key: str, value: str) -> None:
    p = os.environ.get('GITHUB_OUTPUT')
    if p:
        with open(p, 'a') as f:
            f.write(f'{key}={value}\n')

def used_today(state: dict, today) -> int:
    t = today.strftime('%Y%m%d')
    return sum((r.get('requests', 0) for r in state.get('runs') or [] if str(r.get('at', '')).startswith(t)))

def runs_today(state: dict, today) -> int:
    t = today.strftime('%Y%m%d')
    return sum((1 for r in state.get('runs') or [] if str(r.get('at', '')).startswith(t)))

def pending_days(data: str) -> set:
    return {os.path.basename(p)[:-8] for p in glob.glob(f'{data}/pending/*.json.gz')}

def doable(todo: list, pending: set, new_ok: bool) -> list:
    return [(d, s) for d, s in todo if s not in ('new', 'revisit') or d in pending or new_ok]

def listed_days(data: str) -> set:
    return {os.path.basename(p)[:-8] for p in glob.glob(f'{data}/lists/*.json.gz')}

def month_of(day: str) -> str:
    return day[:7]

def assign(costs: dict, n: int) -> dict:
    load = [0.0] * n
    out = {}
    for d in sorted(costs, key=lambda d: (-costs[d], d)):
        k = min(range(n), key=lambda i: (load[i], i))
        out[d] = k
        load[k] += costs[d]
    return out

def _seed_by(path: str) -> dict:
    seed_by = {}
    for f in common.read_json(path)['flights']:
        seed_by.setdefault(f['date'], []).append(f)
    return seed_by

def plan(a) -> int:
    t0 = time.time()
    data, out = (a.data, a.out)
    os.makedirs(f'{out}/in', exist_ok=True)
    os.makedirs(f'{out}/logs', exist_ok=True)
    stamp = _stamp()
    common.PRIVATE_LOG = f'{out}/logs/{stamp}-plan.log'
    seed_by = _seed_by(a.seed)
    sp = f'{data}/state.json'
    state = common.read_json(sp) if os.path.exists(sp) else workq.new_state(list(seed_by))
    for d in workq.spread([d for d in seed_by if d not in set(state['seed_days'])]):
        state['seed_days'].append(d)
    if not state.get('tries_reset_lost'):
        for x in state['done'].values():
            for k in ('redo_tries', 'lv_tries', 'rs_tries'):
                x.pop(k, None)
        state['tries_reset_lost'] = 1
    if not state.get('refused_empty_recheck'):
        back = [d for d, x in state['done'].items() if x.get('status') == 'empty' and x.get('src') == 'new' and (not (state.get('daylists') or {}).get(d))]
        for d in back:
            del state['done'][d]
        common.log(f'{len(back)} empty days to be chosen again (a refused list read as empty)')
        state['refused_empty_recheck'] = 1
    aircraft = common.load_aircraft()
    ac = common.aircraft_version(aircraft)
    today = _today()
    used = used_today(state, today)
    budget = 0 if a.no_new else max(0, min(a.requests, a.daily - used))
    new_ok = budget >= MIN_REQUESTS
    budget = budget if new_ok else 0
    client = weglide.Client(budget)
    pend = pending_days(data)
    corr = getattr(a, 'corrections', None)
    corr_v = dayrun.corr_version(corr)
    if corr_v:
        shutil.copyfile(corr, f'{out}/corrections.json')
    state['corr_v'] = corr_v
    if os.path.isdir(f'{data}/obs'):
        shutil.copytree(f'{data}/obs', f'{out}/obs', dirs_exist_ok=True)
    run = {'at': stamp, 'days': 0, 'ok': 0, 'empty': 0, 'failed': 0, 'skipped': 0, 'left': 0, 'lost': 0, 'planned': 0, 'revisited': 0, 'added': 0, 'rescored': 0}
    P = {'a': a, 'state': state, 'client': client, 'aircraft': aircraft, 'ac': ac, 'pend': pend, 'seed_by': seed_by, 'run': run, 't0': t0, 'costs': {}, 'srcs': {}, 'corr': corr, 'seen': set()}
    wg = workq.weglide_days(state, a.days, today, pend, ac, listed_days(data)) if new_ok else []
    common.log(f"{len(wg)} WeGlide days to plan ({sum((s == 'revisit' for _, s in wg))} chosen from again, {sum((common.macro_of_key(d) == 'EU' for d, _ in wg))} Europe's); WeGlide: {used} requests today, {budget} this run")
    for d, src in wg:
        _plan_day(P, d, src)
    free = [x for x in doable(workq.free_days(state, a.days, pend, corr_v), pend, new_ok) if x[0] not in P['seen']][:max(0, a.days - len(P['costs']))]
    common.log(f"{len(free)} days without a request to plan ({sum((s == 'seed' for _, s in free))} from the seed, {sum((s == 'redo' for _, s in free))} built again, {sum((s == 'rescore' for _, s in free))} scored again, {sum((s == 'lead' for _, s in free))} for their earlier runs, {sum((d in pend for d, _ in free))} waiting)")
    for d, src in free:
        _plan_day(P, d, src)
    costs, srcs = (P['costs'], P['srcs'])
    if client.stopped:
        common.log(f'WeGlide stopped for this run ({client.stopped})')
    if not a.no_new and budget:
        nb = runmod._backfill_pilots(client, state, seed_by, data, t0, a.minutes)
        if nb:
            common.log(f'pilots: {nb} day lists looked up')
    for d, items in (getattr(client, 'lists', None) or {}).items():
        common.write_json(f'{out}/lists/{d}.json.gz', items)
    n = min(a.shards, len(costs))
    sh = assign(costs, n) if n else {}
    days = [{'day': d, 'src': srcs[d], 'shard': sh[d]} for d in sorted(sh, key=lambda d: (sh[d], -costs[d], d))]
    run['planned'] = len(days)
    state['weglide']['requests'] = state['weglide'].get('requests', 0) + client.used
    for k, v in client.status.items():
        state['weglide']['status'][k] = state['weglide']['status'].get(k, 0) + v
    run.update({'requests': client.used, 'plan_minutes': round((time.time() - t0) / 60, 1), 'weglide_stopped': client.stopped, 'shards': n})
    state['runs'] = (state.get('runs') or [])[-199:] + [run]
    common.write_json(f'{out}/state.json', state)
    common.write_json(f'{out}/todo.json', {'at': stamp, 't0': t0, 'no_new': bool(a.no_new), 'corr_v': corr_v, 'days': days}, indent=1)
    common.log(f"planned {len(days)} days on {n} shards ({sum((s in ('new', 'revisit') for s in srcs.values()))} with WeGlide flights, {sum((s == 'rescore' for s in srcs.values()))} scored again), {run['empty']} empty, {run['skipped']} left for later, {run['revisited']} chosen from again ({run['added']} flights more); {client.used} WeGlide requests")
    _output('shards', json.dumps(list(range(n))))
    return 0

def _plan_day(P: dict, d: str, src: str) -> None:
    a, state, client, aircraft, ac, pend, seed_by, run, t0 = (P[k] for k in ('a', 'state', 'client', 'aircraft', 'ac', 'pend', 'seed_by', 'run', 't0'))
    P['seen'].add(d)
    data, out = (a.data, a.out)
    prev = state['done'].get(d, {})
    shorts, obs, extra = ([], [], {})
    if src == 'revisit':
        got = _plan_revisit(a, d, state, client, aircraft, ac, d in pend, run, t0)
        if got is None:
            return
        flights, shorts, obs, extra = got
        for f in flights:
            if f['id'] in set(extra['new_ids']):
                state['counts'][f['grp']] = state['counts'].get(f['grp'], 0) + 1
    elif src == 'rescore':
        fn = f'{data}/days/{d}.json.gz'
        if not os.path.exists(fn):
            prev['grids'] = False
            run['skipped'] += 1
            return
        rec = common.read_json(fn)
        common.write_json(f'{out}/in/{d}.json.gz', {'day': d, 'src': src, 'flights': rec.get('flights') or [], 'short': rec.get('short') or [], 'obs': rec.get('obs') or [], 'rec': rec})
        items = (rec.get('flights') or []) + (rec.get('short') or [])
        P['costs'][d] = RESCORE_COST * max(1, len({f.get('region') for f in items})) * len(rec.get('models') or runmod.models_of(items, d))
        P['srcs'][d] = src
        run['rescored'] += 1
        return
    elif src == 'lead':
        fn = f'{data}/days/{d}.json.gz'
        if not os.path.exists(fn):
            prev['lv_tries'] = workq.MAX_TRIES
            run['skipped'] += 1
            return
        rec = common.read_json(fn)
        common.write_json(f'{out}/in/{d}.json.gz', {'day': d, 'src': src, 'flights': rec.get('flights') or [], 'short': rec.get('short') or [], 'obs': rec.get('obs') or [], 'rec': rec})
        items = (rec.get('flights') or []) + (rec.get('short') or [])
        P['costs'][d] = max(1, len({f.get('region') for f in items})) * max(1, len(common.lead_models(common.macro_of_key(d), common.day_of(d))))
        P['srcs'][d] = src
        run['leads'] = run.get('leads', 0) + 1
        return
    elif src == 'redo':
        fn = f'{data}/days/{d}.json.gz'
        old = common.read_json(fn) if os.path.exists(fn) else {}
        shorts = [dayrun.clean(f) for f in old.get('short') or []]
        obs = [dayrun.clean(f) for f in old.get('obs') or []]
        if d in seed_by:
            flights = [dict(f) for f in seed_by[d]]
        else:
            flights = [dayrun.clean(f) for f in old.get('flights') or []]
        if not flights and (not shorts):
            prev['redo_tries'] = workq.MAX_TRIES
            run['skipped'] += 1
            return
    elif d in pend:
        pj = common.read_json(f'{data}/pending/{d}.json.gz')
        flights, shorts = (pj['flights'], pj.get('short') or [])
        extra = {'ac': pj.get('ac')}
    elif src == 'seed':
        flights = [dict(f) for f in seed_by.get(d, [])]
    else:
        if client.stopped or (time.time() - t0) / 60 > a.minutes:
            run['skipped'] += 1
            return
        try:
            flights, shorts = runmod._flights_from_weglide(client, d, state, aircraft)
        except weglide.Stop as e:
            common.note(f'{d}: WeGlide stopped: {e}')
            run['skipped'] += 1
            return
        extra = {'ac': ac}
        q = state.setdefault('quiet', {'days': 0, 'built': 0})
        if flights or shorts:
            q['built'] += 1
            q['days'] += any((c['region'] not in {f['region'] for f in flights} for c in shorts))
    if not flights and (not shorts):
        state['done'][d] = {'status': 'empty', 'src': src, 'tries': prev.get('tries', 0) + 1, **({'ac': extra['ac'], 'cap': weglide.MAX_PER_DAY} if extra.get('ac') else {})}
        run['empty'] += 1
        run['days'] += 1
        return
    if src == 'new':
        for f in flights:
            state['counts'][f['grp']] = state['counts'].get(f['grp'], 0) + 1
    common.write_json(f'{out}/in/{d}.json.gz', {'day': d, 'src': src, 'flights': flights, 'short': shorts, 'obs': obs, **extra})
    P['costs'][d] = max(1, len({f.get('region') for f in flights + shorts})) * len(runmod.models_of(flights + shorts, d))
    P['srcs'][d] = src

def _plan_revisit(a, d: str, state: dict, client, aircraft: dict, ac: str, from_pending: bool, run: dict, t0: float):
    data = a.data
    x = state['done'][d]
    if from_pending:
        pj = common.read_json(f'{data}/pending/{d}.json.gz')
        return (pj['flights'], pj.get('short') or [], pj.get('obs') or [], {'new_ids': pj.get('new_ids') or [], 'ac': pj.get('ac'), 'obs_for': pj.get('obs_for') or []})
    fn = f'{data}/days/{d}.json.gz'
    if x.get('status') != 'empty' and (not os.path.exists(fn)):
        x.update(ac=ac, cap=weglide.MAX_PER_DAY)
        run['revisited'] += 1
        run['days'] += 1
        return None
    old = common.read_json(fn) if os.path.exists(fn) else {}
    stored = [dayrun.clean(f) for f in old.get('flights') or []]
    if len(stored) >= weglide.MAX_PER_DAY:
        x.update(ac=ac, cap=weglide.MAX_PER_DAY)
        run['revisited'] += 1
        run['days'] += 1
        return None
    lp = f'{data}/lists/{d}.json.gz'
    items = common.read_json(lp) if os.path.exists(lp) else None
    if items is None and (client.stopped or (time.time() - t0) / 60 > a.minutes):
        run['skipped'] += 1
        return None
    try:
        new, stopped = runmod._revisit_from_weglide(client, d, state, aircraft, stored, items)
    except weglide.Stop as e:
        common.note(f'{d}: WeGlide stopped: {e}')
        run['skipped'] += 1
        return None
    run['revisited'] += 1
    if not new:
        if not stopped:
            x.update(ac=ac, cap=weglide.MAX_PER_DAY)
        run['days'] += 1
        return None
    run['added'] += len(new)
    shorts = [dayrun.clean(f) for f in old.get('short') or []]
    obs = [dayrun.clean(f) for f in old.get('obs') or []]
    obs_for = sorted({f['region'] for f in new} - {f.get('region') for f in stored}) if obs else []
    common.note(f'{d}: chosen from again: {len(new)} flights more ({len(stored)} stored)')
    return (stored + new, shorts, obs, {'new_ids': [f['id'] for f in new], 'ac': None if stopped else ac, 'obs_for': obs_for})

def _fetch_grids(month: str, dest: str) -> bool:
    try:
        return statemod.fetch(f'{GRIDS_PREFIX}{month}', dest)
    except Exception as e:
        common.note(f'grids {month}: not fetched: {e!r}')
        return False
LEAD_GRACE = 10

def build(a) -> int:
    t0 = time.time()
    dayrun.LEAD_DEADLINE = t0 + (a.minutes + LEAD_GRACE) * 60
    todo = common.read_json(f'{a.plan}/todo.json')
    mine = [j for j in todo['days'] if j['shard'] == a.shard]
    os.makedirs(f'{a.out}/days', exist_ok=True)
    os.makedirs(f'{a.out}/logs', exist_ok=True)
    common.PRIVATE_LOG = f"{a.out}/logs/{todo['at']}-build{a.shard}.log"
    status = {'shard': a.shard, 'built': [], 'errors': [], 'left': [], 'rebuilt': []}
    if os.path.isdir(f'{a.plan}/obs'):
        shutil.copytree(f'{a.plan}/obs', f'{a.out}/obs', dirs_exist_ok=True)
    obs_client = dayrun.Obs()
    sp = f'{a.out}/status.json'
    common.write_json(sp, status)
    corr = f'{a.plan}/corrections.json' if os.path.exists(f'{a.plan}/corrections.json') else None
    grids_out = f'{a.out}/grids'
    grids_in = getattr(a, 'grids', None) or os.path.join(a.work, 'grids')
    fetched = {}
    for i, j in enumerate(mine):
        d = j['day']
        if (time.time() - t0) / 60 > a.minutes:
            status['left'] = [x['day'] for x in mine[i:]]
            common.log(f"time's up after {i} days")
            break
        inp = common.read_json(f'{a.plan}/in/{d}.json.gz')
        work = os.path.join(a.work, f'd{i}')
        try:
            short = inp.get('short') or []
            if j['src'] == 'lead':
                res = dayrun.lead_day(inp['rec'], work, a.workers, grids_out, corr)
                common.write_json(f'{a.out}/days/{d}.json.gz', res)
                status['built'].append(d)
                lb = res.get('lead_boxes') or {}
                common.log(f"day {i + 1}/{len(mine)} (earlier runs): {res.get('lead_new', 0)} built, {lb.get('ok', 0)}/{lb.get('all', 0)} boxes" + ('' if res.get('lead_v') else '; the rest in a later run'))
            elif j['src'] == 'rescore':
                m = month_of(d)
                if m not in fetched:
                    fetched[m] = _fetch_grids(m, grids_in)
                res = dayrun.rescore_day(inp['rec'], grids_in, work, corr) if fetched[m] else None
                if res is None or any((str(b.get('error', '')).startswith('grids:') for b in res['boxes'])):
                    why = sorted({b['error'] for b in (res or {}).get('boxes', []) if b.get('error')}) if res else ['grids:none']
                    common.note(f"{d}: scored again failed ({', '.join(why)}): built again")
                    flights = [dayrun.clean(f) for f in inp['flights']]
                    short = [dayrun.clean(f) for f in short]
                    obs = [dayrun.clean(f) for f in inp.get('obs') or []] or runmod._observations(obs_client, d, flights + short, a.out)
                    res = dayrun.run_day(common.day_of(d), flights, work, models=runmod.models_of(flights + short, d), workers=a.workers, short=short, obs=obs, grids=grids_out, corrections=corr)
                    status['rebuilt'].append(d)
                    kind = 'built again'
                else:
                    kind = 'scored again'
                common.write_json(f'{a.out}/days/{d}.json.gz', res)
                status['built'].append(d)
                okb = sum((b.get('ok', False) for b in runmod.own(res)))
                common.log(f"day {i + 1}/{len(mine)} ({kind}): {len(inp['flights'])} flights, {okb}/{len(res['boxes'])} boxes")
            else:
                obs = inp.get('obs') or runmod._observations(obs_client, d, inp['flights'] + short, a.out)
                if inp.get('obs') and inp.get('obs_for'):
                    obs = obs + runmod._observations(obs_client, d, [f for f in inp['flights'] if f.get('region') in inp['obs_for']], a.out)
                res = dayrun.run_day(common.day_of(d), inp['flights'], work, models=runmod.models_of(inp['flights'] + short, d), workers=a.workers, short=short, obs=obs, grids=grids_out, corrections=corr)
                common.write_json(f'{a.out}/days/{d}.json.gz', res)
                status['built'].append(d)
                okb = sum((b.get('ok', False) for b in runmod.own(res)))
                common.log(f"day {i + 1}/{len(mine)} ({j['src']}): {len(inp['flights'])} flights, {len(short)} short, {len(obs)} stations, {okb}/{len(res['boxes'])} boxes built")
        except Exception as e:
            common.note(f'{d}: build failed: {e!r}')
            status['errors'].append(d)
            status.setdefault('why', {})[d] = f'{type(e).__name__}: {e}'[:300]
            common.log(f'day {i + 1}/{len(mine)}: failed')
        shutil.rmtree(work, ignore_errors=True)
        common.write_json(sp, status)
    status['obs_requests'] = obs_client.used
    common.write_json(sp, status)
    shutil.rmtree(grids_in, ignore_errors=True) if grids_in.startswith(a.work) else None
    return 0
WHY: dict = {}

def fail_reasons(d: str, how: str, fn: str | None) -> list[str]:
    if how == 'error':
        return [WHY.get(d) or 'shard error']
    if how == 'built' and fn:
        try:
            res = common.read_json(fn)
        except Exception:
            return ['unreadable day file']
        return [f"{b.get('model')} {b.get('error')}: {b.get('why') or '?'}" for b in res.get('boxes') or [] if not b.get('ok') and b.get('error')] or ['no box built']
    return [how]

def _shards(built: str) -> dict:
    got = {}
    for sp in sorted(glob.glob(f'{built}/*/status.json')):
        st = common.read_json(sp)
        base = os.path.dirname(sp)
        for d in st.get('built') or []:
            if os.path.exists(f'{base}/days/{d}.json.gz'):
                got[d] = ('built', f'{base}/days/{d}.json.gz')
        for d in st.get('errors') or []:
            got[d] = ('error', None)
            if (st.get('why') or {}).get(d):
                WHY[d] = st['why'][d]
        for d in st.get('left') or []:
            got[d] = ('left', None)
    return got

def merge(a) -> int:
    data = a.data
    todo = common.read_json(f'{a.plan}/todo.json')
    state = common.read_json(f'{a.plan}/state.json')
    run = state['runs'][-1]
    if run.get('at') != todo['at']:
        raise SystemExit("merge: the plan's state isn't this run's")
    os.makedirs(f'{data}/days', exist_ok=True)
    os.makedirs(f'{data}/logs', exist_ok=True)
    got = _shards(a.built)
    months = set()
    for j in todo['days']:
        d, src = (j['day'], j['src'])
        how, fn = got.get(d, ('lost', None))
        prev = state['done'].get(d, {})
        inp = common.read_json(f'{a.plan}/in/{d}.json.gz')
        stored = False
        if src == 'rescore':
            _merge_rescore(data, d, how, fn, state, run, months)
            continue
        if src == 'lead':
            _merge_lead(data, d, how, fn, state, run, months)
            continue
        if src == 'redo':
            if how == 'built':
                res = common.read_json(fn)
                okb = sum((b.get('ok', False) for b in runmod.own(res)))
                if okb:
                    shutil.copyfile(fn, f'{data}/days/{d}.json.gz')
                    prev.update({'status': 'ok' if okb == len(runmod.own(res)) else 'partial', 'boxes': len(runmod.own(res)), 'flights': len(res['flights']), **runmod.stamps(res)})
                    if res.get('grids'):
                        months.add(month_of(d))
                    run['ok'] += 1
                else:
                    prev['redo_tries'] = prev.get('redo_tries', 0) + 1
                    run['failed'] += 1
            elif how == 'left':
                run['left'] += 1
                continue
            else:
                if how != 'lost':
                    prev['redo_tries'] = prev.get('redo_tries', 0) + 1
                run['failed' if how == 'error' else 'lost'] += 1
            run['days'] += 1
            continue
        if src == 'revisit':
            if _merge_revisit(data, d, how, fn, inp, state, run):
                months.add(month_of(d))
            continue
        if how == 'built':
            res = common.read_json(fn)
            okb = sum((b.get('ok', False) for b in runmod.own(res)))
            status = 'ok' if okb == len(runmod.own(res)) else 'partial' if okb else 'failed'
            if status != 'failed':
                shutil.copyfile(fn, f'{data}/days/{d}.json.gz')
                stored = True
                if res.get('grids'):
                    months.add(month_of(d))
            state['done'][d] = {'status': status, 'src': src, 'tries': prev.get('tries', 0) + 1, 'boxes': len(runmod.own(res)), 'flights': len(inp['flights']), **runmod.stamps(res), **({'ac': inp['ac'], 'cap': weglide.MAX_PER_DAY} if inp.get('ac') else {})}
            run[status if status != 'partial' else 'ok'] += 1
            run['days'] += 1
        elif how == 'left':
            if src == 'new':
                state['done'][d] = {**prev, 'status': 'pending', 'src': src, 'tries': prev.get('tries', 0)}
            run['left'] += 1
        else:
            state['done'][d] = {**prev, 'status': 'failed', 'src': src, 'tries': prev.get('tries', 0) + 1}
            run['failed' if how == 'error' else 'lost'] += 1
            run['days'] += 1
        pf = f'{data}/pending/{d}.json.gz'
        if src == 'new':
            if not stored:
                for f in inp['flights']:
                    state['counts'][f['grp']] = max(0, state['counts'].get(f['grp'], 0) - 1)
            if stored or state['done'][d].get('tries', 0) >= workq.MAX_TRIES:
                if os.path.exists(pf):
                    os.remove(pf)
            else:
                common.write_json(pf, {'day': d, 'flights': inp['flights'], 'short': inp.get('short') or [], 'ac': inp.get('ac')})
    for p in glob.glob(f'{a.plan}/lists/*.json.gz'):
        os.makedirs(f'{data}/lists', exist_ok=True)
        shutil.copyfile(p, f'{data}/lists/{os.path.basename(p)}')
    for p in glob.glob(f'{a.built}/*/obs/*.json'):
        os.makedirs(f'{data}/obs', exist_ok=True)
        shutil.copyfile(p, f'{data}/obs/{os.path.basename(p)}')
    run['obs_requests'] = sum((common.read_json(p).get('obs_requests', 0) for p in glob.glob(f'{a.built}/*/status.json')))
    fails = []
    for j in todo['days']:
        d = j['day']
        how, fn = got.get(d, ('lost', None))
        x = state['done'].get(d, {})
        bad = how == 'error' or (how == 'built' and fn and (not any((b.get('ok') for b in runmod.own(common.read_json(fn))))))
        if bad:
            fails.append({'day': d, 'src': j['src'], 'at': run.get('at'), 'why': fail_reasons(d, how, fn)})
    if fails:
        fp = f'{data}/failures.json'
        old = common.read_json(fp) if os.path.exists(fp) else []
        common.write_json(fp, (old + fails)[-500:])
        kinds = {}
        for f in fails:
            for w in f['why']:
                k = dayrun.fail_kind(w.split(': ', 1)[1] if ': ' in w and ' ' in w.split(': ', 1)[0] else w)
                kinds[k] = kinds.get(k, 0) + 1
        common.log(f'failures by kind: {dict(sorted(kinds.items()))}')
    for p in glob.glob(f'{a.plan}/logs/*') + glob.glob(f'{a.built}/*/logs/*'):
        shutil.copyfile(p, f'{data}/logs/{os.path.basename(p)}')
    for old in sorted(glob.glob(f'{data}/logs/*'), key=os.path.getmtime)[:-KEEP_LOGS]:
        os.remove(old)
    run['minutes'] = round((time.time() - todo['t0']) / 60, 1)
    run['grid_months'] = sorted(months)
    common.write_json(f'{data}/state.json', state)
    pr = workq.progress(state, _today())
    common.log(f"merged: {run['days']} days ({run['ok']} built, {run.get('rescored_ok', 0)} scored again, {run['empty']} empty, {run['failed']} failed, {run['lost']} lost, {run['left']} waiting, {run['skipped']} left for later; {run.get('revisited', 0)} chosen from again, {run.get('stored_more', 0)} flights more stored), {run.get('requests', 0)} WeGlide requests; {pr['days_done']} days in all ({pr['grids']} with their grids saved), {pr['seed_left']} seed days left, {pr['redo_left']} to build again, {pr['rescore_left']} to score again, {pr['revisit_left']} to choose from again, {pr.get('lead_left', 0)} for their earlier runs ({run.get('leads_ok', 0)} done this run)")
    more, why = chain(state, data, a.daily, todo.get('no_new', False))
    common.log(f"next run: {('yes' if more else 'no')} ({why})")
    _output('more', 'yes' if more else 'no')
    keep = set(json.loads(getattr(a, 'keep_months', None) or '[]'))
    _output('months', json.dumps(sorted(months | keep)))
    return 0

def _merge_lead(data: str, d: str, how: str, fn, state: dict, run: dict, months: set) -> None:
    prev = state['done'][d]
    if how == 'left':
        run['left'] += 1
        return
    run['days'] += 1
    res = common.read_json(fn) if how == 'built' else None
    if res and (res.get('lead_v') or res.get('lead_new')):
        shutil.copyfile(fn, f'{data}/days/{d}.json.gz')
        prev['lv'] = res.get('lead_v')
        prev.pop('lv_tries', None)
        run['leads_ok' if res.get('lead_v') else 'leads_part'] = run.get('leads_ok' if res.get('lead_v') else 'leads_part', 0) + 1
        if any((b.get('saved') for b in res['boxes'] if common.is_lead(b.get('model')))):
            months.add(month_of(d))
        return
    if how != 'lost':
        prev['lv_tries'] = prev.get('lv_tries', 0) + 1
    run['failed' if how in ('built', 'error') else 'lost'] += 1

def _merge_rescore(data: str, d: str, how: str, fn, state: dict, run: dict, months: set) -> None:
    prev = state['done'][d]
    if how == 'left':
        run['left'] += 1
        return
    run['days'] += 1
    if how != 'built':
        if how != 'lost':
            prev['rs_tries'] = prev.get('rs_tries', 0) + 1
        run['failed' if how == 'error' else 'lost'] += 1
        return
    res = common.read_json(fn)
    okb = sum((b.get('ok', False) for b in runmod.own(res)))
    if runmod.own(res) and okb == len(runmod.own(res)):
        shutil.copyfile(fn, f'{data}/days/{d}.json.gz')
        prev.update({'status': 'ok', 'boxes': len(runmod.own(res)), 'flights': len(res['flights']), **runmod.stamps(res)})
        prev.pop('rs_tries', None)
        if res.get('rescored_at'):
            run['rescored_ok'] = run.get('rescored_ok', 0) + 1
        else:
            run['ok'] += 1
            months.add(month_of(d))
        return
    run['failed'] += 1
    if any((str(b.get('error', '')).startswith('grids:') for b in res['boxes'])):
        prev['pv'] = 'stale'
    else:
        prev['rs_tries'] = prev.get('rs_tries', 0) + 1

def _merge_revisit(data: str, d: str, how: str, fn, inp: dict, state: dict, run: dict) -> bool:
    prev = state['done'][d]
    new_ids = set(inp.get('new_ids') or [])
    new = [f for f in inp['flights'] if f['id'] in new_ids]
    pf = f'{data}/pending/{d}.json.gz'
    stored = False
    if how == 'built':
        res = common.read_json(fn)
        okb = sum((b.get('ok', False) for b in runmod.own(res)))
        if okb:
            shutil.copyfile(fn, f'{data}/days/{d}.json.gz')
            prev.update({'status': 'ok' if okb == len(runmod.own(res)) else 'partial', 'boxes': len(runmod.own(res)), 'flights': len(inp['flights']), **runmod.stamps(res)})
            if inp.get('ac'):
                prev.update(ac=inp['ac'], cap=weglide.MAX_PER_DAY)
            prev.pop('rv_tries', None)
            run['ok'] += 1
            run['stored_more'] = run.get('stored_more', 0) + len(new)
            stored = True
        else:
            prev['rv_tries'] = prev.get('rv_tries', 0) + 1
            run['failed'] += 1
    elif how == 'left':
        run['left'] += 1
    else:
        prev['rv_tries'] = prev.get('rv_tries', 0) + 1
        run['failed' if how == 'error' else 'lost'] += 1
    if how != 'left':
        run['days'] += 1
    if stored:
        if os.path.exists(pf):
            os.remove(pf)
        return bool(res.get('grids'))
    for f in new:
        state['counts'][f['grp']] = max(0, state['counts'].get(f['grp'], 0) - 1)
    if prev.get('rv_tries', 0) >= workq.MAX_TRIES:
        if os.path.exists(pf):
            os.remove(pf)
        prev.update(ac=inp.get('ac') or common.aircraft_version(), cap=weglide.MAX_PER_DAY)
        prev.pop('rv_tries', None)
    else:
        common.write_json(pf, {'day': d, 'revisit': True, 'flights': inp['flights'], 'short': inp.get('short') or [], 'obs': inp.get('obs') or [], 'new_ids': sorted(new_ids), 'ac': inp.get('ac'), 'obs_for': inp.get('obs_for') or []})
    return False

def chain(state: dict, data: str, daily: int, no_new: bool) -> tuple[bool, str]:
    today = _today()
    run = (state.get('runs') or [{}])[-1]
    if run.get('days', 0) == 0:
        return (False, 'this run did nothing')
    if runs_today(state, today) >= MAX_RUNS_A_DAY:
        return (False, "the day's runs are done")
    wg_ok = run.get('weglide_stopped') in (None, 'budget')
    left = daily - used_today(state, today)
    new_ok = not no_new and wg_ok and (left >= MIN_REQUESTS)
    pend = pending_days(data)
    work = doable(workq.next_days(state, 1, today, new_ok=new_ok, pending=pend, listed=listed_days(data), cv=state.get('corr_v')), pend, new_ok)
    if work:
        return (True, 'work left' if work[0][1] not in ('new', 'revisit') or work[0][0] in pend else f'new days, {left} WeGlide requests left today')
    return (False, 'nothing left without WeGlide; ' + ('WeGlide failing' if not wg_ok else "today's WeGlide budget spent" if left < MIN_REQUESTS else 'new days off' if no_new else 'no days left'))

def rescore(a) -> int:
    data = a.data
    sp = f'{data}/state.json'
    state = common.read_json(sp) if os.path.exists(sp) else workq.new_state([])
    corr = a.corrections
    days = a.days or sorted((d for d in os.listdir(a.grids) if os.path.isdir(os.path.join(a.grids, d)) and os.path.exists(f'{data}/days/{d}.json.gz')))
    n_ok = 0
    for i, d in enumerate(days):
        fn = f'{data}/days/{d}.json.gz'
        if not os.path.exists(fn):
            common.log(f'day {i + 1}: not in the data')
            continue
        res = dayrun.rescore_day(common.read_json(fn), a.grids, os.path.join(a.work, f'd{i}'), corr)
        shutil.rmtree(os.path.join(a.work, f'd{i}'), ignore_errors=True)
        okb = sum((b.get('ok', False) for b in runmod.own(res)))
        if runmod.own(res) and okb == len(runmod.own(res)):
            common.write_json(fn, res)
            x = state['done'].setdefault(d, {'status': 'ok', 'src': 'new'})
            x.update({'boxes': len(runmod.own(res)), **runmod.stamps(res)})
            n_ok += 1
            common.log(f'day {i + 1}/{len(days)}: scored again ({okb} boxes)')
        else:
            why = sorted({str(b.get('error')) for b in res['boxes'] if b.get('error')})
            common.log(f"day {i + 1}/{len(days)}: left as it was ({okb}/{len(res['boxes'])} boxes; {', '.join(why) or 'no box'})")
    state['corr_v'] = dayrun.corr_version(corr)
    common.write_json(sp, state)
    common.log(f"{n_ok} of {len(days)} days scored again (site code {common.SCORE_VERSION}, corrections {state['corr_v']})")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--data', required=True)
    p.add_argument('--seed', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--days', type=int, default=36)
    p.add_argument('--requests', type=int, default=300)
    p.add_argument('--daily', type=int, default=1500)
    p.add_argument('--minutes', type=float, default=20)
    p.add_argument('--shards', type=int, default=6)
    p.add_argument('--no-new', action='store_true', help='no WeGlide (seed days, days built again, days waiting)')
    p.add_argument('--corrections', help="a corrections.json (the training's layer 1) the days are scored with as well")
    b = sub.add_parser('build')
    b.add_argument('--plan', required=True)
    b.add_argument('--shard', type=int, required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--work', default='/tmp/calib-work')
    b.add_argument('--grids', help='where the saved grids are fetched to (default: under --work)')
    b.add_argument('--minutes', type=float, default=40)
    b.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '4')))
    m = sub.add_parser('merge')
    m.add_argument('--data', required=True)
    m.add_argument('--plan', required=True)
    m.add_argument('--built', required=True)
    m.add_argument('--daily', type=int, default=1500)
    m.add_argument('--keep-months', help="JSON list of grid bundles (YYYY-MM) to upload again (the gate's stale ones)")
    r = sub.add_parser('rescore')
    r.add_argument('--data', required=True)
    r.add_argument('--grids', required=True)
    r.add_argument('--corrections')
    r.add_argument('--days', nargs='*')
    r.add_argument('--work', default='/tmp/calib-work')
    a = ap.parse_args()
    return {'plan': plan, 'build': build, 'merge': merge, 'rescore': rescore}[a.cmd](a)
if __name__ == '__main__':
    raise SystemExit(main())
