from __future__ import annotations
import glob
import json
import os
import random
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from . import learn as lab
BASELINE_V = 'b1'
OFFSET_DEG = 0.003
POINTS = ((0.0, 0.0), (OFFSET_DEG, OFFSET_DEG), (OFFSET_DEG, -OFFSET_DEG), (-OFFSET_DEG, OFFSET_DEG), (-OFFSET_DEG, -OFFSET_DEG))
MAX_ROWS = 60000
FLOOR = 1.0
TIMEOUT_S = 50 * 60

def script() -> str | None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return next(iter(sorted(glob.glob(os.path.join(here, 'trigger_eval.*')))), None)

def utc_ms(sd: str, solar_h: float, lon: float) -> int:
    t0 = datetime.strptime(sd, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
    return int(round((t0 + (solar_h - lon / 15.0) * 3600) * 1000))

def wind_uv(c: dict) -> tuple[float, float]:
    import math
    ws, wd = (c.get('wind'), c.get('wind_from'))
    if ws is None or wd is None:
        return (0.0, 0.0)
    return (-ws * math.sin(math.radians(wd)), -ws * math.cos(math.radians(wd)))

def rows_of(days: list, held: int, cnt: dict, seed: int=11) -> dict:
    from . import trigger
    rows = {}
    for x in trigger.table(days):
        if not x[6]:
            continue
        la, lo = lab.centre(x[1], x[2])
        rows.setdefault((la, lo, days[x[0]].sd, x[3]), x[6])
    if len(rows) <= MAX_ROWS:
        return rows
    test = {d.sd for d in days[-held:]} if held >= 1 else set()
    keep = {k: v for k, v in rows.items() if k[2] in test}
    train_days = sorted({k[2] for k in rows} - test)
    random.Random(seed).shuffle(train_days)
    by_day = {}
    for k, v in rows.items():
        by_day.setdefault(k[2], {})[k] = v
    for sd in train_days:
        if len(keep) + len(by_day[sd]) > MAX_ROWS:
            cnt['baseline_days_left_out'] = cnt.get('baseline_days_left_out', 0) + 1
            continue
        keep.update(by_day[sd])
    return keep

def points_of(rows: dict) -> tuple[list, list]:
    from . import trigger
    pts, keys = ([], [])
    half = trigger.TRIG['hour_bin'] / 2
    for (la, lo, sd, hb), c in sorted(rows.items()):
        u, v = wind_uv(c)
        cond = {'ws': c.get('wstar'), 'u': round(u, 3), 'v': round(v, 3), 'sd': c.get('snow'), 'sm': c.get('soil')}
        t = utc_ms(sd, hb + half, lo)
        for dla, dlo in POINTS:
            pts.append(['', round(la + dla, 5), round(lo + dlo, 5), t, cond])
            keys.append((la, lo, sd, hb))
    return (pts, keys)

def scores(rows: dict, work: str, run=subprocess.run, node: str='node', cnt: dict | None=None) -> dict:
    cnt = cnt if cnt is not None else {}
    sc = script()
    if not sc or not rows:
        return {}
    pts, keys = points_of(rows)
    os.makedirs(work, exist_ok=True)
    fin, fout = (os.path.join(work, 'baseline_in.json'), os.path.join(work, 'baseline_out.json'))
    with open(fin, 'w') as f:
        json.dump({'points': pts}, f)
    r = run([node, sc, fin, fout], capture_output=True, text=True, timeout=TIMEOUT_S)
    if r.returncode != 0 or not os.path.exists(fout):
        raise RuntimeError(f"trigger_eval: {(r.stderr or '')[-400:]}")
    with open(fout) as f:
        o = json.load(f)
    acc = {}
    for k, v in zip(keys, o['rows']):
        if v and v[0] is not None:
            acc.setdefault(k, []).append(float(v[0]))
    cnt.update(baseline_points=len(pts), baseline_tiles=o.get('tiles', 0), baseline_tiles_failed=o.get('failed', 0))
    for fn in (fin, fout):
        os.remove(fn)
    return {k: sum(v) / len(v) for k, v in acc.items()}

def make(days: list, held: int=7, work: str | None=None, run=subprocess.run, node: str='node', cnt: dict | None=None):
    from . import trigger
    cnt = cnt if cnt is not None else {}
    if not script():
        cnt['baseline'] = 'no script'
        return (None, cnt)
    if run is subprocess.run and (not shutil.which(node)):
        cnt['baseline'] = 'no node'
        return (None, cnt)
    rows = rows_of(days, held, cnt)
    if not rows:
        cnt['baseline'] = 'no row with conditions'
        return (None, cnt)
    work = work or os.path.join(os.environ.get('RUNNER_TEMP') or '/tmp', 'thermalml-baseline')
    try:
        got = scores(rows, work, run, node, cnt)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
        cnt['baseline'] = f'failed ({type(e).__name__})'
        return (None, cnt)
    if not got:
        cnt['baseline'] = 'no score'
        return (None, cnt)
    med = max(FLOOR, statistics.median(got.values()))
    cnt.update(baseline='site', baseline_rows=len(got), baseline_median=round(med, 2))

    def baseline(la, lo, sd, solar_h):
        k = (round(la, 4), round(lo, 4), sd, lab.hour_bin(solar_h, trigger.TRIG['hour_bin']))
        v = got.get(k)
        return max(FLOOR, v) if v is not None else med
    return (baseline, cnt)
