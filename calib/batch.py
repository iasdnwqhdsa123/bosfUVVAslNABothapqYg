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
import weglide
import workq
MAX_RUNS_A_DAY = 40
MIN_REQUESTS = 20
KEEP_LOGS = 80

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
    return [(d, s) for d, s in todo if s != 'new' or d in pending or new_ok]

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
    aircraft = common.load_aircraft()
    today = _today()
    used = used_today(state, today)
    budget = 0 if a.no_new else max(0, min(a.requests, a.daily - used))
    new_ok = budget >= MIN_REQUESTS
    budget = budget if new_ok else 0
    client = weglide.Client(budget)
    pend = pending_days(data)
    if os.path.isdir(f'{data}/obs'):
        shutil.copytree(f'{data}/obs', f'{out}/obs', dirs_exist_ok=True)
    todo = doable(workq.next_days(state, a.days, today, new_ok=new_ok, pending=pend), pend, new_ok)
    common.log(f"{len(todo)} days to plan ({sum((s == 'seed' for _, s in todo))} from the seed, {sum((s == 'redo' for _, s in todo))} built again, {sum((d in pend for d, _ in todo))} waiting); WeGlide: {used} requests today, {budget} this run")
    run = {'at': stamp, 'days': 0, 'ok': 0, 'empty': 0, 'failed': 0, 'skipped': 0, 'left': 0, 'lost': 0, 'planned': 0}
    costs, srcs = ({}, {})
    for d, src in todo:
        prev = state['done'].get(d, {})
        shorts, obs = ([], [])
        if src == 'redo':
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
                continue
        elif d in pend:
            pj = common.read_json(f'{data}/pending/{d}.json.gz')
            flights, shorts = (pj['flights'], pj.get('short') or [])
        elif src == 'seed':
            flights = [dict(f) for f in seed_by.get(d, [])]
        else:
            if client.stopped or (time.time() - t0) / 60 > a.minutes:
                run['skipped'] += 1
                continue
            try:
                flights, shorts = runmod._flights_from_weglide(client, d, state, aircraft)
            except weglide.Stop as e:
                common.note(f'{d}: WeGlide stopped: {e}')
                run['skipped'] += 1
                continue
            q = state.setdefault('quiet', {'days': 0, 'built': 0})
            if flights or shorts:
                q['built'] += 1
                q['days'] += any((c['region'] not in {f['region'] for f in flights} for c in shorts))
        if not flights and (not shorts):
            state['done'][d] = {'status': 'empty', 'src': src, 'tries': prev.get('tries', 0) + 1}
            run['empty'] += 1
            run['days'] += 1
            continue
        if src == 'new':
            for f in flights:
                state['counts'][f['grp']] = state['counts'].get(f['grp'], 0) + 1
        common.write_json(f'{out}/in/{d}.json.gz', {'day': d, 'src': src, 'flights': flights, 'short': shorts, 'obs': obs})
        costs[d] = max(1, len({f.get('region') for f in flights + shorts})) * len(runmod.MODELS)
        srcs[d] = src
    if client.stopped:
        common.log(f'WeGlide stopped for this run ({client.stopped})')
    if not a.no_new and budget:
        nb = runmod._backfill_pilots(client, state, seed_by, data, t0, a.minutes)
        if nb:
            common.log(f'pilots: {nb} day lists looked up')
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
    common.write_json(f'{out}/todo.json', {'at': stamp, 't0': t0, 'no_new': bool(a.no_new), 'days': days}, indent=1)
    common.log(f"planned {len(days)} days on {n} shards, {run['empty']} empty, {run['skipped']} left for later; {client.used} WeGlide requests")
    _output('shards', json.dumps(list(range(n))))
    return 0

def build(a) -> int:
    t0 = time.time()
    todo = common.read_json(f'{a.plan}/todo.json')
    mine = [j for j in todo['days'] if j['shard'] == a.shard]
    os.makedirs(f'{a.out}/days', exist_ok=True)
    os.makedirs(f'{a.out}/logs', exist_ok=True)
    common.PRIVATE_LOG = f"{a.out}/logs/{todo['at']}-build{a.shard}.log"
    status = {'shard': a.shard, 'built': [], 'errors': [], 'left': []}
    if os.path.isdir(f'{a.plan}/obs'):
        shutil.copytree(f'{a.plan}/obs', f'{a.out}/obs', dirs_exist_ok=True)
    obs_client = dayrun.Obs()
    sp = f'{a.out}/status.json'
    common.write_json(sp, status)
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
            obs = inp.get('obs') or runmod._observations(obs_client, d, inp['flights'] + short, a.out)
            res = dayrun.run_day(d, inp['flights'], work, models=runmod.MODELS, workers=a.workers, short=short, obs=obs)
            common.write_json(f'{a.out}/days/{d}.json.gz', res)
            status['built'].append(d)
            okb = sum((b.get('ok', False) for b in res['boxes']))
            common.log(f"day {i + 1}/{len(mine)} ({j['src']}): {len(inp['flights'])} flights, {len(short)} short, {len(obs)} stations, {okb}/{len(res['boxes'])} boxes built")
        except Exception as e:
            common.note(f'{d}: build failed: {e!r}')
            status['errors'].append(d)
            common.log(f'day {i + 1}/{len(mine)}: failed')
        shutil.rmtree(work, ignore_errors=True)
        common.write_json(sp, status)
    status['obs_requests'] = obs_client.used
    common.write_json(sp, status)
    return 0

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
    for j in todo['days']:
        d, src = (j['day'], j['src'])
        how, fn = got.get(d, ('lost', None))
        prev = state['done'].get(d, {})
        inp = common.read_json(f'{a.plan}/in/{d}.json.gz')
        stored = False
        if src == 'redo':
            if how == 'built':
                res = common.read_json(fn)
                okb = sum((b.get('ok', False) for b in res['boxes']))
                if okb:
                    shutil.copyfile(fn, f'{data}/days/{d}.json.gz')
                    prev.update({'status': 'ok' if okb == len(res['boxes']) else 'partial', 'boxes': len(res['boxes']), 'flights': len(res['flights']), 'v': common.DAY_VERSION})
                    run['ok'] += 1
                else:
                    prev['redo_tries'] = prev.get('redo_tries', 0) + 1
                    run['failed'] += 1
            elif how == 'left':
                run['left'] += 1
                continue
            else:
                prev['redo_tries'] = prev.get('redo_tries', 0) + 1
                run['failed' if how == 'error' else 'lost'] += 1
            run['days'] += 1
            continue
        if how == 'built':
            res = common.read_json(fn)
            okb = sum((b.get('ok', False) for b in res['boxes']))
            status = 'ok' if okb == len(res['boxes']) else 'partial' if okb else 'failed'
            if status != 'failed':
                shutil.copyfile(fn, f'{data}/days/{d}.json.gz')
                stored = True
            state['done'][d] = {'status': status, 'src': src, 'tries': prev.get('tries', 0) + 1, 'boxes': len(res['boxes']), 'flights': len(inp['flights']), 'v': common.DAY_VERSION}
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
                common.write_json(pf, {'day': d, 'flights': inp['flights'], 'short': inp.get('short') or []})
    for p in glob.glob(f'{a.built}/*/obs/*.json'):
        os.makedirs(f'{data}/obs', exist_ok=True)
        shutil.copyfile(p, f'{data}/obs/{os.path.basename(p)}')
    run['obs_requests'] = sum((common.read_json(p).get('obs_requests', 0) for p in glob.glob(f'{a.built}/*/status.json')))
    for p in glob.glob(f'{a.plan}/logs/*') + glob.glob(f'{a.built}/*/logs/*'):
        shutil.copyfile(p, f'{data}/logs/{os.path.basename(p)}')
    for old in sorted(glob.glob(f'{data}/logs/*'), key=os.path.getmtime)[:-KEEP_LOGS]:
        os.remove(old)
    run['minutes'] = round((time.time() - todo['t0']) / 60, 1)
    common.write_json(f'{data}/state.json', state)
    pr = workq.progress(state, _today())
    common.log(f"merged: {run['days']} days ({run['ok']} built, {run['empty']} empty, {run['failed']} failed, {run['lost']} lost, {run['left']} waiting, {run['skipped']} left for later), {run.get('requests', 0)} WeGlide requests; {pr['days_done']} days in all, {pr['seed_left']} seed days left")
    more, why = chain(state, data, a.daily, todo.get('no_new', False))
    common.log(f"next run: {('yes' if more else 'no')} ({why})")
    _output('more', 'yes' if more else 'no')
    return 0

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
    work = doable(workq.next_days(state, 1, today, new_ok=new_ok, pending=pend), pend, new_ok)
    if work:
        return (True, 'work left' if work[0][1] != 'new' or work[0][0] in pend else f'new days, {left} WeGlide requests left today')
    return (False, 'nothing left without WeGlide; ' + ('WeGlide failing' if not wg_ok else "today's WeGlide budget spent" if left < MIN_REQUESTS else 'new days off' if no_new else 'no days left'))

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
    b = sub.add_parser('build')
    b.add_argument('--plan', required=True)
    b.add_argument('--shard', type=int, required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--work', default='/tmp/calib-work')
    b.add_argument('--minutes', type=float, default=40)
    b.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '4')))
    m = sub.add_parser('merge')
    m.add_argument('--data', required=True)
    m.add_argument('--plan', required=True)
    m.add_argument('--built', required=True)
    m.add_argument('--daily', type=int, default=1500)
    a = ap.parse_args()
    return {'plan': plan, 'build': build, 'merge': merge}[a.cmd](a)
if __name__ == '__main__':
    raise SystemExit(main())
