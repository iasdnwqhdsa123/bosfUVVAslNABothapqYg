from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
ARCHIVE_DAYS = 31
RUN_BY = {'hrrr': 12, 'gfs': 6, 'ecmwf': 0, 'icon': 6, 'icond2': 6, 'arome': 6}
LATE_H = 3
QUARTER_FIELDS = ('ceil', 'vis', 'refc', 'rain', 'ptype', 'u_sfc', 'v_sfc', 'gust', 't2', 'td2', 'ctop')
QUARTER = re.compile('^q_(%s)\\.pack$' % '|'.join(QUARTER_FIELDS))
DAY = re.compile('^\\d{8}$')
SND = re.compile('^(\\d{8})\\.pack$')

def run_time(run_id: str) -> datetime:
    return datetime.strptime(run_id, '%Y%m%d%H').replace(tzinfo=timezone.utc)

def day_for_run(model: str, run: datetime) -> str:
    t = run - timedelta(hours=RUN_BY.get(model, 6))
    d = t.date() if (t.hour, t.minute, t.second) == (0, 0, 0) else t.date() + timedelta(days=1)
    return f'{d:%Y%m%d}'

def late(model: str, run: datetime) -> bool:
    day = datetime.strptime(day_for_run(model, run), '%Y%m%d').replace(tzinfo=timezone.utc)
    return day + timedelta(hours=RUN_BY.get(model, 6)) - run <= timedelta(hours=LATE_H)

def trim_manifest(m: dict, day: str) -> dict:
    out = dict(m)
    stamps = list(m['days'][day])
    out['days'] = {day: stamps}
    out['quarters'] = {day: m['quarters'][day]} if day in (m.get('quarters') or {}) else {}
    snd = m.get('sounding')
    if snd and snd.get('days'):
        utc = {s[:8] for s in stamps}
        out['sounding'] = {**snd, 'days': {k: v for k, v in snd['days'].items() if k in utc}}
    out['quarters'] = {}
    out['older'] = []
    out.pop('hours', None)
    out['archive'] = {'day': day}
    return out

def wanted(rel: str, day: str, snd_days: set[str]) -> bool:
    parts = rel.split(os.sep)
    if parts[0] == 'h':
        return False
    if parts[0] == 'snd':
        m = SND.match(parts[-1]) if len(parts) == 2 else None
        if m:
            return m.group(1) in snd_days
        return len(parts) > 2 and parts[1] in snd_days
    if any((p.endswith('.building') or p.endswith('.tmp.npy') for p in parts)):
        return False
    if QUARTER.match(parts[-1]):
        return False
    return all((not DAY.match(p) or p == day for p in parts[:-1]))

def _png(img):
    try:
        import tiles
        if tiles.deflate is not None:
            return tiles.png_bytes(img)
    except ImportError:
        pass
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img, mode='L').save(buf, format='PNG', compress_level=9)
    return buf.getvalue()

def _trim_entry(job):
    blob, rows, W = job
    if not blob:
        return b''
    import io
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(io.BytesIO(blob)).convert('L'))
    return _png(np.ascontiguousarray(np.concatenate([img[k * W:(k + 1) * W] for k in rows])))

def trim_rows(src: str, dst: str, rows: list[int], W: int, workers: int | None=None):
    import struct
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with open(src, 'rb') as f:
        head = f.read(12)
        if head[:4] != b'APK1':
            raise ValueError(f'{src}: not a tile pack')
        ntx, nty = struct.unpack('<II', head[4:])
        n = ntx * nty
        ends = struct.unpack(f'<{n}I', f.read(4 * n))
        data = f.read(ends[-1] if n else 0)
    blobs = [data[ends[i - 1] if i else 0:ends[i]] for i in range(n)]
    jobs = [(b, rows, W) for b in blobs]
    workers = workers or os.cpu_count() or 1
    if workers > 1 and n > 64:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('fork')) as ex:
            out = list(ex.map(_trim_entry, jobs, chunksize=max(1, n // (workers * 8))))
    else:
        out = [_trim_entry(j) for j in jobs]
    total, offs = (0, [])
    for b in out:
        total += len(b)
        offs.append(total)
    tmp = dst + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(b'APK1' + struct.pack('<II', ntx, nty) + struct.pack(f'<{n}I', *offs))
        for b in out:
            f.write(b)
    os.replace(tmp, dst)

def _trim_packs(am: dict, rdir: str, dest_run: str, day: str):
    stamps = set(am['days'][day])
    pts = am.get('points')
    if pts and am.get('pack') and os.path.exists(os.path.join(rdir, 'pt.pack')):
        rows = [k for k, s in enumerate(pts['stamps']) if s in stamps]
        if len(rows) < len(pts['stamps']):
            trim_rows(os.path.join(rdir, 'pt.pack'), os.path.join(dest_run, 'pt.pack'), rows, pts['block'] + 1)
            am['points'] = {**pts, 'stamps': [pts['stamps'][k] for k in rows]}
    snd = am.get('sounding')
    for d, e in ((snd or {}).get('days') or {}).items():
        src = os.path.join(rdir, 'snd', f'{d}.pack')
        if not isinstance(e, dict) or not e.get('stamps') or (not os.path.exists(src)):
            continue
        rows = [k for k, s in enumerate(e['stamps']) if s in stamps]
        if len(rows) < len(e['stamps']):
            trim_rows(src, os.path.join(dest_run, 'snd', f'{d}.pack'), rows, snd['block'] + 1)
            snd['days'][d] = {**e, 'stamps': [e['stamps'][k] for k in rows]}

def stage(dist: str, model: str, out: str) -> tuple[str, str] | None:
    base = os.path.join(dist, model)
    try:
        with open(os.path.join(base, 'latest.json')) as f:
            m = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    run_id = m.get('runId') or ''
    if not re.fullmatch('\\d{10}', run_id) or not m.get('complete', True):
        return None
    day = day_for_run(model, run_time(run_id))
    if not (m.get('days') or {}).get(day):
        return None
    rdir = os.path.join(base, run_id)
    if not os.path.isdir(rdir):
        return None
    am = trim_manifest(m, day)
    snd_days = set(((am.get('sounding') or {}).get('days') or {}).keys())
    dest = os.path.join(out, day, model)
    shutil.rmtree(dest, ignore_errors=True)
    for root, dirs, files in os.walk(rdir):
        dirs[:] = [d for d in dirs if not d.endswith('.building')]
        for name in files:
            src = os.path.join(root, name)
            rel = os.path.relpath(src, rdir)
            if not wanted(rel, day, snd_days):
                continue
            to = os.path.join(dest, run_id, rel)
            os.makedirs(os.path.dirname(to), exist_ok=True)
            try:
                os.link(src, to)
            except OSError:
                shutil.copy2(src, to)
    _trim_packs(am, rdir, os.path.join(dest, run_id), day)
    with open(os.path.join(dest, 'latest.json'), 'w') as f:
        json.dump(am, f, separators=(',', ':'))
    return (day, run_id)

def _today(s: str | None) -> datetime:
    if s:
        return datetime.strptime(s, '%Y%m%d').replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

def _days_models(keys) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for k in keys:
        parts = k.strip().split('/')
        if len(parts) >= 3 and parts[-1] == 'latest.json' and DAY.match(parts[-3]) and (parts[-2] in RUN_BY):
            out.setdefault(parts[-3], set()).add(parts[-2])
    return out

def first_kept(today: datetime) -> str:
    return f'{today - timedelta(days=ARCHIVE_DAYS - 1):%Y%m%d}'

def build_index(keys, today: datetime) -> dict:
    lo, hi = (first_kept(today), f'{today + timedelta(days=1):%Y%m%d}')
    dm = _days_models(keys)
    days = [{'day': d, 'models': sorted(dm[d])} for d in sorted(dm, reverse=True) if lo <= d <= hi]
    return {'days': days, 'keep': ARCHIVE_DAYS, 'updated': datetime.now(timezone.utc).isoformat(timespec='seconds')}

def old_days(keys, today: datetime) -> list[str]:
    lo = first_kept(today)
    return sorted((d for d in _days_models(keys) if d < lo))

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage')
    s.add_argument('--dist', default='dist-data')
    s.add_argument('--out', required=True)
    s.add_argument('models', nargs='+')
    w = sub.add_parser('when')
    w.add_argument('--dist', default='dist-data')
    w.add_argument('models', nargs='+')
    for name in ('index', 'old'):
        p = sub.add_parser(name)
        p.add_argument('--keys', required=True)
        p.add_argument('--today')
    a = ap.parse_args(argv)
    if a.cmd == 'stage':
        for model in a.models:
            r = stage(a.dist, model, a.out)
            if r:
                print(f'{r[0]} {model} {r[1]}')
        return 0
    if a.cmd == 'when':
        for model in a.models:
            try:
                with open(os.path.join(a.dist, model, 'latest.json')) as f:
                    run_id = json.load(f).get('runId') or ''
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if re.fullmatch('\\d{10}', run_id):
                t = run_time(run_id)
                print(f"{day_for_run(model, t)} {model} {run_id} {('late' if late(model, t) else 'early')}")
        return 0
    with open(a.keys) as f:
        keys = f.read().split()
    today = _today(a.today)
    if a.cmd == 'index':
        json.dump(build_index(keys, today), sys.stdout, separators=(',', ':'))
        sys.stdout.write('\n')
    else:
        print('\n'.join(old_days(keys, today)))
    return 0
if __name__ == '__main__':
    sys.exit(main())
