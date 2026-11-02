import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

def run_and_hours(day: str, tz: str, model: str, lead: int, area: str, src, from_local: int=6, to_local: int=21):
    z = ZoneInfo(tz)
    d = datetime.strptime(day, '%Y-%m-%d')
    hours = sorted({datetime(d.year, d.month, d.day, h, tzinfo=z).astimezone(timezone.utc) for h in range(from_local, to_local + 1)})
    run = datetime(d.year, d.month, d.day, common.LEAD_RUN_H[area][model], tzinfo=timezone.utc) - timedelta(days=lead)
    last = src.last_fh(run)
    fhs = [f for f in (int((t - run).total_seconds() // 3600) for t in hours) if 1 <= f <= last and src.builds(f)]
    return (run, fhs)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--tz', required=True)
    ap.add_argument('--bbox', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='gfs', choices=['hrrr', 'gfs'])
    ap.add_argument('--lead', type=int, required=True)
    ap.add_argument('--area', default='US', choices=sorted(common.LEAD_RUN_H))
    ap.add_argument('--from-local', type=int, default=6)
    ap.add_argument('--to-local', type=int, default=21)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    import build
    if a.model not in common.LEAD_RUN_H.get(a.area, {}):
        print(f'{a.model}: no lead runs in {a.area}', flush=True)
        return 2
    src = build.SOURCES[a.model]
    run, fhs = run_and_hours(a.day, a.tz, a.model, a.lead, a.area, src, a.from_local, a.to_local)
    if not fhs:
        print(f"{a.model} {run:%Y-%m-%d %HZ}: none of the day's hours", flush=True)
        return 2
    print(f'{a.model}: run {run:%Y-%m-%d %HZ} (+{a.lead} d), f{fhs[0]:03d}-f{fhs[-1]:03d}, {len(fhs)} hours', flush=True)
    build.needed_times = lambda s, days=None, now=None: [run + timedelta(hours=f) for f in fhs]
    build.pick_run = lambda s, needed, pin=None: (run, fhs)
    build.backfill_jobs = lambda *x, **k: []
    sys.argv = ['build.py', '--model', a.model, '--out', a.out, '--force', '--no-quarters', '--workers', str(a.workers), f'--bbox={a.bbox}']
    return build.main()
if __name__ == '__main__':
    sys.exit(main())
