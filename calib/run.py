from __future__ import annotations
import argparse
import glob
import os
import time
from datetime import date, datetime, timezone
import common
import day as dayrun
import track
import weglide
import workq
KEEP_LOGS = 20
PILOT_BACKFILL_DAYS = 20

def models_of(items: list[dict], day: str | None=None) -> tuple:
    day = day or next((f.get('date') for f in items if f.get('date')), None)
    return common.models_for({f.get('region') for f in items if f.get('region')}, common.day_of(day) if day else None)
QUIET_SHARE = 0.2
QUIET_MIN_AFT = 2
QUIET_KM = 150
SHORT_PER_REGION = 15

def _inside(p, bbox) -> bool:
    return bbox[0] <= p[0] <= bbox[2] and bbox[1] <= p[1] <= bbox[3]

def short_flights(d: str, items: list, aircraft: dict, long: list[dict], summary: dict, state: dict, macro: str='US') -> list[dict]:
    taken = {f['id'] for f in long}
    cands = [c for c in (weglide.short_candidate(it, aircraft, macro) for it in items if it.get('id') not in taken) if c and c['date'] == d]
    boxes = {B['region']: B['bbox'] for B in dayrun.boxes(long)} if long else {}
    keep = [c for c in cands if c['region'] in boxes and _inside(c['tp'], boxes[c['region']])]
    q = state.setdefault('quiet', {'days': 0, 'built': 0})
    if common.day_type(d) == 'weekend' and q['days'] < QUIET_SHARE * (q['built'] + 1):
        regs = summary.get('regions') or {}
        quiet = sorted((r for r, x in regs.items() if not x['long'] and x['aft'] >= QUIET_MIN_AFT and (r not in boxes)), key=lambda r: (-regs[r]['aft'], r))
        if quiet:
            qc = [c for c in cands if c['region'] == quiet[0]]
            aft = [c for c in qc if c['afternoon']] or qc
            anchor = max(aft, key=lambda c: (sum((track.dist_km(c['tp'], o['tp']) <= QUIET_KM for o in aft)), -c['id']))['tp']
            keep += [c for c in qc if track.dist_km(c['tp'], anchor) <= QUIET_KM]
    by = {}
    for c in sorted(keep, key=lambda c: (not c['afternoon'], c['id'])):
        if len(by.setdefault(c['region'], [])) < SHORT_PER_REGION:
            by[c['region']].append(c)
    return [c for cs in by.values() for c in cs]

def _short_tracks(client, d: str, shorts: list[dict], long: list[dict]) -> None:
    import random
    rnd = random.Random(f'short:{d}')
    long_regs = {f['region'] for f in long}
    order = sorted(shorts, key=lambda c: (not c['afternoon'], c['region'] in long_regs, rnd.random()))
    for c in order[:weglide.SHORT_TRACKS_PER_DAY]:
        dat = client.data(c['id'])
        try:
            sm = track.short_summary(dat) if dat else None
        except Exception as e:
            common.note(f"{d}: short flight {c['id']} unreadable: {e!r}")
            sm = None
        if sm:
            c.update({k: v for k, v in sm.items() if k != 'tp'}, tp=sm['tp'], tp_src='track', tracked=True)

def _flights_from_weglide(client, d: str, state: dict, aircraft: dict) -> tuple[list[dict], list[dict]]:
    day, macro = common.split_key(d)
    items = client.day_list(day, macro)
    summary = weglide.day_summary(day, items, aircraft, macro, getattr(client, 'last_full', None))
    state.setdefault('daylists', {})[d] = summary
    seen = set(state['seen'])
    chosen = weglide.choose(day, items, aircraft, state['counts'], state['unmapped'], seen, macro=macro)
    common.note(f'{d}: list {len(items)} flights, chosen {len(chosen)}')
    out, stopped = _fetch_chosen(client, d, chosen, state)
    if stopped:
        raise weglide.Stop(client.stopped or 'stopped')
    shorts = short_flights(day, items, aircraft, out, summary, state, macro)
    if shorts:
        try:
            _short_tracks(client, d, shorts, out)
        except weglide.Stop:
            pass
        common.note(f"{d}: short flights kept {len(shorts)} ({sum((bool(c.get('tracked')) for c in shorts))} with the track)")
    return (out, shorts)

def _revisit_from_weglide(client, d: str, state: dict, aircraft: dict, stored: list[dict], items: list | None=None) -> tuple[list[dict], bool]:
    day, macro = common.split_key(d)
    if items is None:
        items = client.day_list(day, macro)
    state.setdefault('daylists', {})[d] = weglide.day_summary(day, items, aircraft, macro)
    seen = set(state['seen']) | {f['id'] for f in stored}
    taken = {(f.get('region'), f.get('grp')) for f in stored}
    chosen = weglide.choose(day, items, aircraft, state['counts'], {}, seen, taken=taken, room=weglide.MAX_PER_DAY - len(stored), macro=macro)
    common.note(f'{d}: chosen again from {len(items)} listed: {len(chosen)} more')
    return _fetch_chosen(client, d, chosen, state)

def _fetch_chosen(client, d: str, chosen: list[dict], state: dict) -> tuple[list[dict], bool]:
    out = []
    for c in chosen:
        try:
            det = client.detail(c['id'])
            if not det:
                continue
            dat = client.data(c['id'])
        except weglide.Stop:
            return (out, True)
        if not dat:
            continue
        state['seen'].append(c['id'])
        try:
            t = track.flight_from(None, det, dat)
        except Exception as e:
            common.note(f"{d}: flight {c['id']} unreadable: {e!r}")
            continue
        if not t['legs'] or not t['tp']:
            continue
        free = next((x for x in det.get('contest') or [] if x.get('name') == 'free'), {})
        ws = free.get('score') or {}
        rec = {**c, 'src': 'weglide', 'tp': t['tp'], 'zmax': t['zmax'], 'engine_max': t['engine_max'], 'engine_med': t['engine_med'], 'free_km': free.get('distance'), 'free_kmh': free.get('speed'), 'thermal_avg': ws.get('thermal_avg'), 'thermal_pct': ws.get('thermal_percentage'), 'glide_ratio': ws.get('glide_ratio'), 'agl_avg': ws.get('agl_avg'), 'wind_kmh': ws.get('wind_speed'), 'wind_dir': ws.get('wind_direction'), 'legs': t['legs'], 'climbs': t['climbs'], 'track': t['track']}
        out.append(rec)
    return (out, False)

def _observations(obs_client, d: str, items: list[dict], data: str) -> list[dict]:
    if not items or obs_client.used >= obs_client.budget:
        return []
    try:
        return dayrun.observations(obs_client, common.day_of(d), dayrun.boxes(items), os.path.join(data, 'obs'))
    except Exception as e:
        common.note(f'{d}: observations failed: {e!r}')
        return []

def collect(a) -> int:
    t0 = time.time()
    data = a.data
    os.makedirs(f'{data}/days', exist_ok=True)
    os.makedirs(f'{data}/logs', exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    common.PRIVATE_LOG = f'{data}/logs/{stamp}.log'
    for old in sorted(glob.glob(f'{data}/logs/*.log'), key=os.path.getmtime)[:-KEEP_LOGS]:
        os.remove(old)
    seed = common.read_json(a.seed)
    seed_by = {}
    for f in seed['flights']:
        seed_by.setdefault(f['date'], []).append(f)
    sp = f'{data}/state.json'
    state = common.read_json(sp) if os.path.exists(sp) else workq.new_state(list(seed_by))
    for d in workq.spread([d for d in seed_by if d not in set(state['seed_days'])]):
        state['seed_days'].append(d)
    aircraft = common.load_aircraft()
    client = weglide.Client(a.requests)
    obs_client = dayrun.Obs()
    os.makedirs(f'{data}/obs', exist_ok=True)
    todo = [x for x in workq.next_days(state, a.days, date.today(), new_ok=not a.no_new) if x[1] not in ('rescore', 'revisit')]
    common.log(f"{len(todo)} days to do ({sum((s == 'seed' for _, s in todo))} from the seed, {sum((s == 'redo' for _, s in todo))} built again)")
    run = {'at': stamp, 'days': 0, 'ok': 0, 'empty': 0, 'failed': 0, 'skipped': 0}
    for i, (d, src) in enumerate(todo):
        if (time.time() - t0) / 60 > a.minutes:
            common.log(f"time's up after {i} days")
            break
        if src == 'redo':
            _redo_day(a, d, i, seed_by, state, run, obs_client)
            common.write_json(sp, state)
            continue
        shorts = []
        if src == 'seed':
            flights = [dict(f) for f in seed_by.get(d, [])]
        else:
            if client.stopped:
                run['skipped'] += 1
                continue
            try:
                flights, shorts = _flights_from_weglide(client, d, state, aircraft)
            except weglide.Stop as e:
                common.log(f'day {i + 1}: WeGlide stopped for this run ({e})')
                common.note(f'{d}: WeGlide stopped: {e}')
                run['skipped'] += 1
                continue
        prev = state['done'].get(d, {})
        if not flights and (not shorts):
            state['done'][d] = {'status': 'empty', 'src': src, 'tries': prev.get('tries', 0) + 1}
            run['empty'] += 1
            common.log(f'day {i + 1} ({src}): no flights to use')
        else:
            obs = _observations(obs_client, d, flights + shorts, data)
            res = dayrun.run_day(common.day_of(d), flights, os.path.join(a.work, f'd{i}'), models=models_of(flights + shorts, d), workers=a.workers, short=shorts, obs=obs)
            okb = sum((b.get('ok', False) for b in res['boxes']))
            status = 'ok' if okb == len(res['boxes']) else 'partial' if okb else 'failed'
            if src == 'new' and status != 'failed':
                q = state.setdefault('quiet', {'days': 0, 'built': 0})
                q['built'] += 1
                q['days'] += any((c['region'] not in {f['region'] for f in flights} for c in shorts))
            if status != 'failed':
                common.write_json(f'{data}/days/{d}.json.gz', res)
                if src == 'new':
                    for f in flights:
                        state['counts'][f['grp']] = state['counts'].get(f['grp'], 0) + 1
            state['done'][d] = {'status': status, 'src': src, 'tries': prev.get('tries', 0) + 1, 'boxes': len(res['boxes']), 'flights': len(flights), **stamps(res), **({'cap': weglide.MAX_PER_DAY} if src == 'new' else {})}
            run[status if status != 'partial' else 'ok'] += 1
            common.log(f"day {i + 1} ({src}): {len(flights)} flights, {len(shorts)} short, {len(obs)} stations, {okb}/{len(res['boxes'])} boxes built")
        run['days'] += 1
        common.write_json(sp, state)
    if not a.no_new:
        nb = _backfill_pilots(client, state, seed_by, data, t0, a.minutes)
        if nb:
            common.log(f'pilots: {nb} day lists looked up')
    state['weglide']['requests'] = state['weglide'].get('requests', 0) + client.used
    for k, v in client.status.items():
        state['weglide']['status'][k] = state['weglide']['status'].get(k, 0) + v
    run.update({'requests': client.used, 'minutes': round((time.time() - t0) / 60, 1), 'weglide_stopped': client.stopped, 'obs_requests': obs_client.used})
    state['runs'] = (state.get('runs') or [])[-199:] + [run]
    common.write_json(sp, state)
    pr = workq.progress(state, date.today())
    common.log(f"done: {run['days']} days ({run['ok']} built, {run['empty']} empty, {run['failed']} failed, {run['skipped']} left for later), {client.used} WeGlide requests, {obs_client.used} for observations; {pr['days_done']} days in all, {pr['seed_left']} seed days left")
    return 0

def stamps(res: dict) -> dict:
    return {'v': res.get('version', common.DAY_VERSION), 'pv': res.get('physics_v'), 'sv': res.get('score_v'), 'cv': res.get('corr_v'), 'grids': bool(res.get('grids'))}

def _redo_day(a, d: str, i: int, seed_by: dict, state: dict, run: dict, obs_client) -> None:
    fn = f'{a.data}/days/{d}.json.gz'
    x = state['done'][d]
    old = common.read_json(fn) if os.path.exists(fn) else {}
    shorts = [dayrun.clean(f) for f in old.get('short') or []]
    obs = [dayrun.clean(f) for f in old.get('obs') or []]
    if d in seed_by:
        flights = [dict(f) for f in seed_by[d]]
    else:
        flights = [dayrun.clean(f) for f in old.get('flights') or []]
    if not flights and (not shorts):
        x['redo_tries'] = workq.MAX_TRIES
        run['skipped'] += 1
        return
    if not obs:
        obs = _observations(obs_client, d, flights + shorts, a.data)
    res = dayrun.run_day(common.day_of(d), flights, os.path.join(a.work, f'd{i}'), models=models_of(flights + shorts, d), workers=a.workers, short=shorts, obs=obs)
    okb = sum((b.get('ok', False) for b in res['boxes']))
    if okb:
        common.write_json(fn, res)
        x.update({'status': 'ok' if okb == len(res['boxes']) else 'partial', 'boxes': len(res['boxes']), 'flights': len(flights), **stamps(res)})
        run['ok'] += 1
    else:
        x['redo_tries'] = x.get('redo_tries', 0) + 1
        run['failed'] += 1
    run['days'] += 1
    common.log(f"day {i + 1} (built again): {len(flights)} flights, {okb}/{len(res['boxes'])} boxes built")

def _backfill_pilots(client, state: dict, seed_by: dict, data: str, t0: float, minutes: float) -> int:
    if not os.environ.get('CALIB_DATA_KEY') or client.stopped:
        return 0
    pil = state.setdefault('pilots', {})
    tried = state.setdefault('pilot_days', {})
    days = list(state['seed_days']) + sorted((d for d, x in state['done'].items() if x.get('src') == 'new' and common.macro_of_key(d) == 'US'))
    n, aircraft = (0, common.load_aircraft())
    for d in days:
        if n >= PILOT_BACKFILL_DAYS or (time.time() - t0) / 60 > minutes or tried.get(d, 0) >= 2:
            continue
        if d in seed_by:
            fids = {f['id'] for f in seed_by[d]}
        else:
            fn = f'{data}/days/{d}.json.gz'
            fids = {f['id'] for f in common.read_json(fn).get('flights') or [] if not f.get('pilot')} if os.path.exists(fn) else set()
        fids = {x for x in fids if str(x) not in pil}
        if not fids:
            tried[d] = 2
            continue
        try:
            items = client.day_list(d)
        except weglide.Stop:
            break
        n += 1
        tried[d] = tried.get(d, 0) + 1
        if items:
            state.setdefault('daylists', {}).setdefault(d, weglide.day_summary(d, items, aircraft))
        for it in items:
            if it.get('id') in fids:
                k = weglide.pilot_of(it)
                if k:
                    pil[str(it['id'])] = k
    return n

def report_cmd(a) -> int:
    import report
    os.makedirs(a.out, exist_ok=True)
    text = report.build(a.data, os.path.join(a.out, 'report-days.csv'))
    with open(os.path.join(a.out, 'report.md'), 'w') as f:
        f.write(text)
    common.log('report written')
    return 0

def export_cmd(a) -> int:
    import csv
    import gzip
    import io
    import json
    import rows
    days = rows.load_days(a.data)
    L, C, F = rows.tables(days)
    H, G = rows.survival(days)
    os.makedirs(a.out, exist_ok=True)
    for name, rs in (('legs', L), ('climbs', C), ('flights', F), ('hours', H), ('glides', G)):
        buf = io.StringIO()
        keys = [k for k in rs[0].keys() if k != 'pilot'] if rs else ['date']
        w = csv.DictWriter(buf, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rs)
        with open(os.path.join(a.out, f'{name}.csv.gz'), 'wb') as f:
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as g:
                g.write(buf.getvalue().encode())
    st = common.read_json(f'{a.data}/state.json')
    for k in ('seen', 'pilots', 'pilot_days', 'daylists'):
        st.pop(k, None)
    with open(os.path.join(a.out, 'state.json'), 'w') as f:
        json.dump(st, f, indent=1, sort_keys=True)
    common.log(f'exported {len(F)} flights, {len(L)} legs, {len(C)} climbs')
    return 0

def train_cmd(a) -> int:
    import train
    res = train.run(a.data, a.out, a.dataset)
    for t, r in res.items():
        common.log(f"{t}: {r['rows']} rows trained")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--data', required=True)
    c.add_argument('--seed', required=True)
    c.add_argument('--work', default='/tmp/calib-work')
    c.add_argument('--days', type=int, default=3)
    c.add_argument('--requests', type=int, default=150)
    c.add_argument('--minutes', type=float, default=35)
    c.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '4')))
    c.add_argument('--no-new', action='store_true', help='seed days only (no WeGlide)')
    r = sub.add_parser('report')
    r.add_argument('--data', required=True)
    r.add_argument('--out', required=True)
    e = sub.add_parser('export')
    e.add_argument('--data', required=True)
    e.add_argument('--out', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--dataset', choices=('all', 'seed'), default='all')
    a = ap.parse_args()
    return {'collect': collect, 'report': report_cmd, 'export': export_cmd, 'train': train_cmd}[a.cmd](a)
if __name__ == '__main__':
    raise SystemExit(main())
