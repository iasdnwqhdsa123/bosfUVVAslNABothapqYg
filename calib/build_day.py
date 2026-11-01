import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
sys.path.insert(0, os.getcwd())
import build
RUN_H = {'hrrr': 12, 'gfs': 12}
ap = argparse.ArgumentParser()
ap.add_argument('--day', required=True)
ap.add_argument('--tz', required=True)
ap.add_argument('--bbox', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--model', default='hrrr', choices=sorted(RUN_H))
ap.add_argument('--from-local', type=int, default=6)
ap.add_argument('--to-local', type=int, default=21)
ap.add_argument('--workers', type=int, default=2)
a = ap.parse_args()
tz = ZoneInfo(a.tz)
d = datetime.strptime(a.day, '%Y-%m-%d')
hours = sorted({datetime(d.year, d.month, d.day, h, tzinfo=tz).astimezone(timezone.utc) for h in range(a.from_local, a.to_local + 1)})
run = datetime(d.year, d.month, d.day, RUN_H[a.model], tzinfo=timezone.utc)
fhs = [f for f in (int((t - run).total_seconds() // 3600) for t in hours) if f >= 1]
build.needed_times = lambda s, days=None, now=None: [run + timedelta(hours=f) for f in fhs]
build.pick_run = lambda s, needed, pin=None: (run, fhs)
build.backfill_jobs = lambda *x, **k: []
sys.argv = ['build.py', '--model', a.model, '--out', a.out, '--force', '--no-quarters', '--workers', str(a.workers), f'--bbox={a.bbox}']
sys.exit(build.main())
