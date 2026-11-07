from __future__ import annotations
import argparse
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from airspace import VERSION as AIRSPACE_VERSION
from aviation import VERSION as AVIATION_VERSION
from chart_tiles import VERSION as CHART_TILES_VERSION, cycle_of
from sources import OPEN_AHEAD_H, OPEN_SLACK_H, PAUSED, REQUIRED, SOURCES, latest_complete_run, needed_times
AIRSPACE_MAX_AGE = timedelta(days=7)
SLOTS = {'01:15': {'hrrr': 23}, '02:00': {'hrrr': 0, 'icond2': 0, 'ecmwf': 18, 'arome': 21}, '02:30': {'hrrr': 1}, '03:30': {'hrrr': 2}, '03:55': {'icon': 0, 'icond2': 3, 'hrrr': 3}, '05:30': {'gfs': 0, 'hrrr': 4}, '06:30': {'hrrr': 5, 'arome': 3}, '08:00': {'hrrr': 6, 'icond2': 6}, '08:30': {'hrrr': 7}, '09:40': {'ecmwf': 0, 'icon': 6, 'ens': 0, 'hrrr': 8}, '10:30': {'hrrr': 9}, '11:30': {'gfs': 6, 'icond2': 9, 'hrrr': 10}, '12:30': {'hrrr': 11}, '14:00': {'hrrr': 12, 'icond2': 12, 'ecmwf': 6, 'arome': 9}, '14:30': {'hrrr': 13}, '15:40': {'hrrr': 14}, '16:30': {'icon': 12, 'icond2': 15, 'hrrr': 15}, '17:30': {'gfs': 12, 'hrrr': 16}, '18:30': {'hrrr': 17}, '20:00': {'hrrr': 18, 'icond2': 18, 'arome': 15}, '20:30': {'hrrr': 19}, '21:40': {'icon': 18, 'ecmwf': 12, 'ens': 12, 'hrrr': 20}, '22:30': {'icond2': 21, 'hrrr': 21}, '23:20': {'gfs': 18, 'hrrr': 22}}

def lands(mid: str, hour: int) -> tuple[timedelta, timedelta]:
    hm = lambda s: timedelta(hours=int(s[:-3]), minutes=int(s[-2:]))
    if mid == 'hrrr':
        t = ('1:47', '2:30') if hour % 6 == 0 else ('2:09', '2:45') if hour == 23 else ('1:25', '2:05')
    elif mid == 'icond2':
        t = ('1:23', '2:00')
    elif mid == 'arome':
        t = {3: ('2:44', '3:15'), 9: ('4:15', '4:45'), 15: ('3:54', '4:25'), 21: ('4:13', '4:50')}.get(hour, ('5:10', '5:50'))
    elif mid == 'icon':
        t = ('3:35', '4:45') if hour == 0 else ('4:28', '4:50') if hour == 12 else ('3:20', '4:30')
    elif mid == 'gfs':
        t = ('5:18', '6:30')
    elif mid == 'ecmwf':
        t = ('9:35', '10:10') if hour % 12 == 0 else ('8:28', '9:00')
    else:
        t = ('7:46', '8:30')
    return (hm(t[0]), hm(t[1]))
OVERDUE = timedelta(minutes=60)
WAIT_STEP = timedelta(minutes=1)

def slot_at(now: datetime, name: str | None=None) -> tuple[str, datetime]:

    def last(n, early):
        t = now.replace(hour=int(n[:2]), minute=int(n[3:]), second=0, microsecond=0)
        return t if t <= now + early else t - timedelta(days=1)
    if name in SLOTS:
        return (name, last(name, timedelta(minutes=15)))
    return max(((n, last(n, timedelta(minutes=10))) for n in SLOTS), key=lambda x: x[1])

def expected_run(slot_time: datetime, hour: int) -> datetime:
    t = slot_time.replace(minute=0, second=0, microsecond=0)
    while t.hour != hour:
        t -= timedelta(hours=1)
    return t

def plan(mid: str, now: datetime, slot: tuple[str, datetime], newest: datetime | None, have: str | None, out_missing: bool=False, force: bool=False) -> tuple[bool, tuple[datetime, datetime] | None]:
    newer = newest is not None and (have is None or f'{newest:%Y%m%d%H}' > have)
    hour = SLOTS[slot[0]].get(mid)
    if hour is not None:
        run = expected_run(slot[1], hour)
        until = run + lands(mid, hour)[1]
        if (newest is None or newest < run) and now < until:
            return (True, (run, until))
        return (newer or out_missing or force, None)
    overdue = newer and now >= newest + lands(mid, newest.hour)[0] + OVERDUE
    return (overdue or out_missing or force, None)

def complete_run(src) -> datetime | None:
    return latest_complete_run(src, needed_times(src), log=lambda *a: None)[0]

def wait_for(src, run: datetime, until: datetime) -> bool:
    while True:
        newest = complete_run(src)
        if newest is not None and newest >= run:
            return True
        if datetime.now(timezone.utc) + WAIT_STEP > until:
            return False
        print(f'{src.id}: waiting for {run:%Y-%m-%d %HZ} (until {until:%H:%M} UTC)', flush=True)
        time.sleep(WAIT_STEP.total_seconds())

def published(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'wx-pipeline/1.0', 'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None

def map_data_due(ages: list, todo: list[str]) -> bool:
    if any((a is None for a in ages)):
        return True
    oldest = max(ages)
    if oldest > AIRSPACE_MAX_AGE + timedelta(days=1):
        return True
    return oldest > AIRSPACE_MAX_AGE and bool(todo)
FREE_EVERY = timedelta(hours=5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-url')
    ap.add_argument('--open-url', help='where the open data (free, airspace) is published, if not under --data-url')
    ap.add_argument('--slot', help='the slot this run is for (HH:MM UTC; else the latest one by the clock)')
    ap.add_argument('--outs', help='the models whose output (out-<model>) is on hand, comma-separated (and mapdata)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--wait-for', choices=list(SOURCES), help="in that model's job: wait for --run until --until (writes landed=true|false)")
    ap.add_argument('--run', help='with --wait-for: YYYYMMDDHH')
    ap.add_argument('--until', help='with --wait-for: ISO time')
    ap.add_argument('--missing-only', action='store_true', help="only build what isn't published (or on hand) at all")
    ap.add_argument('--all-models', action='store_true', help='build every model with a newer run (or no output on hand)')
    args = ap.parse_args()
    out = os.environ.get('GITHUB_OUTPUT')
    if args.wait_for:
        src = SOURCES[args.wait_for]
        run = datetime.strptime(args.run, '%Y%m%d%H').replace(tzinfo=timezone.utc)
        landed = wait_for(src, run, datetime.fromisoformat(args.until))
        print(f"{src.id}: {run:%Y-%m-%d %HZ} {('complete' if landed else 'still not complete: building the newest on hand')}")
        if out:
            with open(out, 'a') as f:
                f.write(f"landed={('true' if landed else 'false')}\n")
        return
    if not args.data_url:
        ap.error('--data-url is required')
    base = args.data_url.rstrip('/')
    open_base = (args.open_url or base).rstrip('/')
    now = datetime.now(timezone.utc)
    slot = slot_at(now, args.slot if args.slot and re.fullmatch('\\d\\d:\\d\\d', args.slot) else None)
    outs = None if args.outs is None else {m for m in args.outs.split(',') if m}
    print(f'slot {slot[0]} UTC ({slot[1]:%Y-%m-%d %H:%M}): ' + ', '.join((f'{m} {h:02d}Z' for m, h in SLOTS[slot[0]].items())))
    todo, waits = ([], {})
    for mid, src in SOURCES.items():
        if mid in PAUSED:
            continue
        cur = published(f'{base}/{mid}/latest.json')
        have = cur.get('runId') if cur and cur.get('complete') else None
        newest = complete_run(src)
        missing = outs is not None and mid not in outs
        if args.missing_only:
            build, wait = (have is None or missing, None)
        elif args.all_models:
            build, wait = (have is None or missing or args.force or (newest is not None and f'{newest:%Y%m%d%H}' > have), None)
        else:
            build, wait = plan(mid, now, slot, newest, have, missing, args.force)
        print(f"{mid}: published {have or '-'}, newest complete {(f'{newest:%Y%m%d%H}' if newest else '-')}" + (', its output not on hand' if missing else '') + (f' → waits for {wait[0]:%Y%m%d%H} until {wait[1]:%H:%M}' if wait else ' → build' if build else ''))
        if build:
            todo.append(mid)
        if wait:
            waits[mid] = [f'{wait[0]:%Y%m%d%H}', wait[1].isoformat()]
    free = []
    for mid, ahead in OPEN_AHEAD_H.items():
        if mid in PAUSED:
            continue
        fm = published(f'{open_base}/free/{mid}/latest.json')
        until = fm and fm.get('freeUntil')
        behind = not until or (not args.missing_only and now + timedelta(hours=ahead + OPEN_SLACK_H) - FREE_EVERY >= datetime.fromisoformat(until))
        print(f"free {mid}: until {until or '-'}{(' → refresh' if behind else '')}")
        if behind:
            free.append(mid)
    ages = []
    for name, url, version in (('airspace', f'{open_base}/airspace/meta.json', AIRSPACE_VERSION), ('aviation map data', f'{open_base}/airspace/aviation/meta.json', AVIATION_VERSION)):
        meta = published(url)
        updated = meta and meta.get('updated')
        age = now - datetime.fromisoformat(updated) if updated and meta.get('version') == version else None
        print(f"{name}: updated {updated or '-'}{('' if age is not None else ' (missing or new version)')}")
        ages.append(age)
    airspace = args.force or (any((a is None for a in ages)) if args.missing_only else map_data_due(ages, todo))
    cm = published(f'{open_base}/airspace/aviation/charts-low.json')
    charts = not (cm and cm.get('cycle') == cycle_of(now) and (cm.get('version') == CHART_TILES_VERSION))
    print(f"chart overviews: cycle {(cm or {}).get('cycle', '-')}{(f' → {cycle_of(now)}' if charts else '')}")
    mapdata = airspace or charts or (outs is not None and 'mapdata' not in outs)
    print(f"map data: {('refresh' if airspace else 'chart overviews' if charts else 'its output not on hand' if mapdata else 'keep')}")
    matrix = todo + [m for m in free if m not in todo]
    if out:
        with open(out, 'a') as f:
            f.write(f'slot={slot[0]}\n')
            f.write(f"models={' '.join(todo)}\n")
            f.write(f'matrix={json.dumps(matrix)}\n')
            f.write(f'waits={json.dumps(waits)}\n')
            f.write(f'active={json.dumps([m for m in SOURCES if m not in PAUSED])}\n')
            f.write(f'required={json.dumps([m for m in REQUIRED if m not in PAUSED])}\n')
            f.write(f"free={('true' if free else 'false')}\n")
            f.write(f'free_models={json.dumps(free)}\n')
            f.write(f"airspace={('true' if airspace else 'false')}\n")
            f.write(f"aviation={('true' if airspace else 'false')}\n")
            f.write(f"mapdata={('true' if mapdata else 'false')}\n")
            f.write(f"any={('true' if matrix or mapdata else 'false')}\n")
    print('build:', ' '.join(todo) or 'nothing', '| free window:', ' '.join(free) or '-', '| waits:', ' '.join((f'{m} {r}' for m, (r, _) in waits.items())) or '-')
if __name__ == '__main__':
    main()
