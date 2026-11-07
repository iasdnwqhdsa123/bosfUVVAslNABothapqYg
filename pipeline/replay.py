from __future__ import annotations
import argparse
import copy
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import numpy as np
import build
import om
import sources
BASE = 'https://openmeteo.s3.amazonaws.com/data'
NT = {'dwd_icon_d2': 121, 'dwd_icon': 253}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
UNIT = {'pressure_msl': 'hPa'}

def unit_of(name):
    if name.startswith(('temperature_', 'soil_temperature_')):
        return '°C'
    return UNIT.get(name)

def variables(src):
    out = set()
    own = {**getattr(src, 'dwd_vars', {}), **getattr(src, 'mf_levels', {})}
    for key in src.wanted():
        if key[0] in own:
            continue
        name = om.om_name(key)
        out.update(om.RECIPES[name][0] if name in om.RECIPES else (name,))
    out.discard('HSURF')
    return sorted(out)

def _window(src, region):
    g = om.Grid(*src.grid_def)
    i_n, j_w, ny, nx = g.window(region)
    return (i_n - ny + 1, j_w, ny, nx)

class BlockFS:

    def __init__(self, block=8 << 20):
        self.block, self.blocks, self.sizes = (block, {}, {})

    def size(self, path):
        if path not in self.sizes:
            self.sizes[path] = om.head_size(path)
        return self.sizes[path]

    def _block(self, path, i):
        if (path, i) not in self.blocks:
            import grib
            a = i * self.block
            self.blocks[path, i] = grib.http_get(path, (a, min(a + self.block, self.size(path)) - 1))
        return self.blocks[path, i]

    def cat_file(self, path=None, start=None, end=None, url=None, **kw):
        path = path or url
        start, end = (start or 0, self.size(path) if end is None else end)
        out = bytearray()
        for i in range(start // self.block, (end - 1) // self.block + 1):
            b = self._block(path, i)
            out += b[max(0, start - i * self.block):end - i * self.block]
        return bytes(out)

    def cat(self, path, start=None, end=None, **kw):
        return self.cat_file(path, start, end)

def _read(path_or_url, rows, cols, times=None):
    import omfiles
    if path_or_url.startswith('http'):
        r = omfiles.OmFileReader.from_fsspec(BlockFS(), path_or_url)
    else:
        r = omfiles.OmFileReader.from_path(path_or_url)
    sel = (rows, cols) if times is None else (rows, cols, times)
    return np.asarray(r.read_array(sel), np.float32)

def _get(url, path):
    import shutil
    import urllib.request
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(path, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return path
        except Exception:
            if attempt == 4:
                raise
    return path

def cache_var(domain, name, hours, win, cache, whole):
    out = os.path.join(cache, f'{name}.npy')
    if os.path.exists(out):
        return out
    s0, w0, ny, nx = win
    nt = NT[domain]
    idx = [int((t - EPOCH).total_seconds() // 3600) for t in hours]
    arr = np.full((len(hours), ny, nx), np.nan, np.float32)
    for chunk in sorted({i // nt for i in idx}):
        want = [(k, i % nt) for k, i in enumerate(idx) if i // nt == chunk]
        lo, hi = (min((t for _, t in want)), max((t for _, t in want)) + 1)
        url = f'{BASE}/{domain}/{name}/chunk_{chunk}.om'
        if whole:
            path = _get(url, os.path.join(cache, f'.{name}.{chunk}.om'))
            try:
                a = _read(path, slice(s0, s0 + ny), slice(w0, w0 + nx), slice(lo, hi))
            finally:
                os.unlink(path)
        else:
            a = _read(url, slice(s0, s0 + ny), slice(w0, w0 + nx), slice(lo, hi))
        for k, t in want:
            arr[k] = a[:, :, t - lo]
    np.save(out + '.tmp.npy', arr)
    os.replace(out + '.tmp.npy', out)
    return out

def prefetch(src, region, hours, cache, workers=6):
    os.makedirs(cache, exist_ok=True)
    win = _window(src, region)
    g = src.grid_def
    whole = win[2] * win[3] > 0.3 * g[3] * g[4]
    names = variables(src)
    static = os.path.join(cache, 'HSURF.npy')
    if not os.path.exists(static):
        url = f'{BASE}/{src.domain}/static/HSURF.om'
        path = _get(url, os.path.join(cache, '.HSURF.om'))
        a = _read(path, slice(win[0], win[0] + win[2]), slice(win[1], win[1] + win[3]))
        os.unlink(path)
        np.save(static, a)
    missing = []

    def one(n):
        try:
            cache_var(src.domain, n, hours, win, cache, whole)
            return None
        except Exception as e:
            return f'{n}: {e}'
    with ThreadPoolExecutor(workers) as ex:
        for i, err in enumerate(ex.map(one, names)):
            if err:
                missing.append(err)
            print(f'  cached {i + 1}/{len(names)}', flush=True) if (i + 1) % 10 == 0 else None
    for m in missing:
        print('  missing:', m)
    return names

class ArchiveHour:

    def __init__(self, cache, k, grid_region):
        self.cache, self.k, self.region = (cache, k, grid_region)
        self._arr = {}

    def _get(self, name):
        if name not in self._arr:
            self._arr[name] = np.load(os.path.join(self.cache, f'{name}.npy'), mmap_mode='r')
        return self._arr[name]

    def has(self, name):
        return os.path.exists(os.path.join(self.cache, f'{name}.npy'))

    def unit(self, name):
        return unit_of(name)

    def read(self, name, region, boxes=None):
        a = self._get(name)
        a = a if name == 'HSURF' else a[self.k]
        if tuple(region[:5]) != tuple(self.region[:5]):
            raise ValueError(f'region {region} is not the cached {self.region}')
        full = np.array(a[::-1], np.float32)
        if not boxes:
            return full
        out = np.full(full.shape, np.nan, np.float32)
        for y0, y1, x0, x1 in boxes:
            out[y0:y1, x0:x1] = full[y0:y1, x0:x1]
        return out

class ArchiveSource(sources.OpenMeteo):

    def fetch(self, run, fh, sel):
        valid = run + timedelta(hours=fh)
        k = self.replay_hours.index(valid)
        hour = ArchiveHour(self.replay_cache, k, self.replay_region)
        out = {}
        for key in sel:
            f = om.Field(hour, om.om_name(key))
            if f.available():
                out[key] = f
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=('icon', 'icond2'))
    ap.add_argument('--day', required=True, help='UTC day, YYYY-MM-DD')
    ap.add_argument('--from-h', type=int, default=0, help='first UTC hour of the day')
    ap.add_argument('--to-h', type=int, default=23, help='last UTC hour of the day')
    ap.add_argument('--out', default='../replay-data')
    ap.add_argument('--cache', default='../replay-cache')
    ap.add_argument('--bbox', help="west,south,east,north (on the model's grid lines)")
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--recompute', action='store_true', help='recompute the hours already built (after a change to the physics)')
    a = ap.parse_args()
    day = datetime.strptime(a.day, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    hours = [day + timedelta(hours=h) for h in range(a.from_h, a.to_h + 1)]
    run = day - timedelta(hours=6)
    orig = sources.SOURCES[a.model]
    region = list(build.REGIONS[orig.id])
    if a.bbox:
        region[:4] = [float(x) for x in a.bbox.split(',')]
    src = copy.copy(orig)
    src.__class__ = ArchiveSource
    src.replay_hours, src.replay_region = (hours, tuple(region))
    src.replay_cache = os.path.join(a.cache, f"{orig.domain}-{a.day}-{'_'.join((f'{x:g}' for x in region[:4]))}")
    if not src.snd_levels:
        src.snd_levels, src.snd_step = (tuple(src.levels) + tuple(src.cloud_levels), 1)
    src.notes = [f"Replay of {a.day} from Open-Meteo's archive (the first hours of each run, joined)"] + list(orig.notes)
    print(f'{a.model}: caching {a.day} {a.from_h:02d}–{a.to_h:02d}Z from the archive', flush=True)
    prefetch(src, region, hours, src.replay_cache)
    build.SOURCES[a.model] = src
    fhs = [int((t - run).total_seconds() // 3600) for t in hours]
    build.needed_times = lambda s, days=None, now=None: hours
    build.pick_run = lambda s, needed, pin=None: (run, fhs)
    argv = ['build.py', '--model', a.model, '--out', a.out, '--force', '--workers', str(a.workers)]
    if a.bbox:
        argv += [f'--bbox={a.bbox}']
    if a.recompute:
        argv += ['--recompute']
    sys.argv = argv
    return build.main()
if __name__ == '__main__':
    sys.exit(main())
