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
MODELS = tuple((m for m in os.environ.get('CALIB_MODELS', 'hrrr').split(',') if m))

def _flights_from_weglide(client, d: str, state: dict, aircraft: dict) -> list[dict]:
    items = client.day_list(d)
    seen = set(state['seen'])
    chosen = weglide.choose(d, items, aircraft, state['counts'], state['unmapped'], seen)
    common.note(f'{d}: list {len(items)} flights, chosen {len(chosen)}')
    out = []
    for c in chosen:
        det = client.detail(c['id'])
        if not det:
            continue
        dat = client.data(c['id'])
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
    return out

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
    todo = workq.next_days(state, a.days, date.today(), new_ok=not a.no_new)
    common.log(f"{len(todo)} days to do ({sum((s == 'seed' for _, s in todo))} from the seed)")
    run = {'at': stamp, 'days': 0, 'ok': 0, 'empty': 0, 'failed': 0, 'skipped': 0}
    for i, (d, src) in enumerate(todo):
        if (time.time() - t0) / 60 > a.minutes:
            common.log(f"time's up after {i} days")
            break
        if src == 'seed':
            flights = [dict(f) for f in seed_by.get(d, [])]
        else:
            if client.stopped:
                run['skipped'] += 1
                continue
            try:
                flights = _flights_from_weglide(client, d, state, aircraft)
            except weglide.Stop as e:
                common.log(f'day {i + 1}: WeGlide stopped for this run ({e})')
                common.note(f'{d}: WeGlide stopped: {e}')
                run['skipped'] += 1
                continue
        prev = state['done'].get(d, {})
        if not flights:
            state['done'][d] = {'status': 'empty', 'src': src, 'tries': prev.get('tries', 0) + 1}
            run['empty'] += 1
            common.log(f'day {i + 1} ({src}): no flights to use')
        else:
            res = dayrun.run_day(d, flights, os.path.join(a.work, f'd{i}'), models=MODELS, workers=a.workers)
            okb = sum((b.get('ok', False) for b in res['boxes']))
            status = 'ok' if okb == len(res['boxes']) else 'partial' if okb else 'failed'
            if status != 'failed':
                common.write_json(f'{data}/days/{d}.json.gz', res)
                if src == 'new':
                    for f in flights:
                        state['counts'][f['grp']] = state['counts'].get(f['grp'], 0) + 1
            state['done'][d] = {'status': status, 'src': src, 'tries': prev.get('tries', 0) + 1, 'boxes': len(res['boxes']), 'flights': len(flights)}
            run[status if status != 'partial' else 'ok'] += 1
            common.log(f"day {i + 1} ({src}): {len(flights)} flights, {okb}/{len(res['boxes'])} boxes built")
        run['days'] += 1
        common.write_json(sp, state)
    state['weglide']['requests'] = state['weglide'].get('requests', 0) + client.used
    for k, v in client.status.items():
        state['weglide']['status'][k] = state['weglide']['status'].get(k, 0) + v
    run.update({'requests': client.used, 'minutes': round((time.time() - t0) / 60, 1), 'weglide_stopped': client.stopped})
    state['runs'] = (state.get('runs') or [])[-199:] + [run]
    common.write_json(sp, state)
    pr = workq.progress(state, date.today())
    common.log(f"done: {run['days']} days ({run['ok']} built, {run['empty']} empty, {run['failed']} failed, {run['skipped']} left for later), {client.used} WeGlide requests; {pr['days_done']} days in all, {pr['seed_left']} seed days left")
    return 0

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
    L, C, F = rows.tables(rows.load_days(a.data))
    os.makedirs(a.out, exist_ok=True)
    for name, rs in (('legs', L), ('climbs', C), ('flights', F)):
        buf = io.StringIO()
        keys = list(rs[0].keys()) if rs else ['date']
        w = csv.DictWriter(buf, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rs)
        with open(os.path.join(a.out, f'{name}.csv.gz'), 'wb') as f:
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as g:
                g.write(buf.getvalue().encode())
    st = common.read_json(f'{a.data}/state.json')
    st.pop('seen', None)
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
