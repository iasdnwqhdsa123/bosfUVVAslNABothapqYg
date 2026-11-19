from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_eu
import common
MODELS = ('gfs', 'icon')
RUN_LEAD_H = 4

def sh_run(hours: list[datetime], lead_h: int=RUN_LEAD_H) -> datetime:
    t = hours[0] - timedelta(hours=lead_h)
    run = t.replace(minute=0, second=0, microsecond=0, hour=0 if t.hour < 12 else 12)
    return run

def forecast_hours(hours: list[datetime], run: datetime) -> list[int]:
    return [f for f in (int((t - run).total_seconds() // 3600) for t in hours) if f >= 1]

def replay_span(hours: list[datetime]) -> tuple[str, int, int]:
    d0 = hours[0].replace(hour=0, minute=0, second=0, microsecond=0)
    return (f'{d0:%Y-%m-%d}', hours[0].hour, int((hours[-1] - d0).total_seconds() // 3600))

def _gfs(a, hours: list[datetime]) -> int:
    import build
    run = sh_run(hours)
    fhs = forecast_hours(hours, run)
    print(f'gfs: run {run:%Y-%m-%d %HZ}, {len(fhs)} hours', flush=True)
    build.needed_times = lambda s, days=None, now=None: [run + timedelta(hours=f) for f in fhs]
    build.pick_run = lambda s, needed, pin=None: (run, fhs)
    build.backfill_jobs = lambda *x, **k: []
    sys.argv = ['build.py', '--model', 'gfs', '--out', a.out, '--force', '--no-quarters', '--workers', str(a.workers), f'--bbox={a.bbox}']
    return build.main()

def _icon(a, hours: list[datetime]) -> int:
    first = common.FIRST_DAY.get('icon', common.ICON_FIRST)
    if a.day < first:
        print(f"icon: the archive's layout before {first} isn't read (docs/replay.md, Older days)", flush=True)
        return 2
    import replay
    w, s, e, n = build_eu.snap([float(x) for x in a.bbox.split(',')], 'icon')
    day, h0, h1 = replay_span(hours)
    sys.argv = ['replay.py', '--model', 'icon', '--day', day, '--from-h', str(h0), '--to-h', str(h1), '--out', a.out, '--cache', a.cache, f'--bbox={w},{s},{e},{n}', '--workers', str(a.workers)]
    return replay.main()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--tz', required=True)
    ap.add_argument('--bbox', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='gfs', choices=MODELS)
    ap.add_argument('--cache', default='../replay-cache')
    ap.add_argument('--from-local', type=int, default=6)
    ap.add_argument('--to-local', type=int, default=21)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    hours = build_eu.utc_hours(a.day, a.tz, a.from_local, a.to_local)
    return _gfs(a, hours) if a.model == 'gfs' else _icon(a, hours)
if __name__ == '__main__':
    sys.exit(main())
