from __future__ import annotations
import argparse
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import archive
import build
import tiles
import verify
UTC = timezone.utc
STATION_FIELDS = verify.FIELDS + ('blapse', 'u_blavg', 'v_blavg', 'u_bltop', 'v_bltop', 'u_g300', 'v_g300', 'sun', 'ctop', 'cloud5', 'cloud10', 'ridge') + verify.SMOOTHED
STATION_LEADS = frozenset([*range(0, 49), *range(51, 121, 3), *range(126, 241, 6)])
SLIM_FIELDS = ('wstar', 'zi', 'depth', 'cubase', 'cudepth', 'cucover', 'cuclass', 'cloudlow', 'cloudmid', 'cloudhigh', 'ctop', 'refc', 'u_blavg', 'v_blavg', 'snowd', 'soilm', 'heat')
SLIM_LEADS = range(0, 48)
SUN_MIN = 5.0
BLEND_FIELDS = ('t2', 'td2', 'u_sfc', 'v_sfc', 'cubase', 'cucover', 'cuclass', 'wstar', 'zi', 'depth')
BLEND_LEADS = frozenset([*range(0, 49), *range(51, 73, 3)])
BLEND_BASE = 'hrrr'
BLEND_REGIONS = {'conus': (BLEND_BASE, None), 'sh': ('ecmwf', (112.0, -49.0, 179.75, -18.5)), 'alps': ('icond2', (5.5, 43.5, 16.0, 48.5))}
BLEND_MEMBERS = {'sh': ('icon',), 'alps': ('arome', 'ecmwf')}
BLEND_V = 1
PHYS_GLOBS = ('*.py', 'native/*.cpp', 'native/*.h')

def log(*a):
    print(*a, flush=True)

def parse_run(run):
    return datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=UTC)

def own_hours(hours_dir, run):
    r0, out = (parse_run(run), [])
    for f in sorted(os.listdir(hours_dir)):
        if not f.endswith('.npz') or f.endswith('.tmp.npz') or len(f) != 14:
            continue
        stamp = f[:-4]
        if build.stamp_of(os.path.join(hours_dir, f'{stamp}.run'))[0] != run:
            continue
        v = parse_run(stamp)
        lead = int((v - r0).total_seconds() // 3600)
        if lead >= 0:
            out.append((v, lead))
    return out

def grids_due(model, run, day_has_grids):
    r = parse_run(run)
    return not day_has_grids and r.hour >= archive.RUN_BY.get(model, 6)

def conus_window(region, base=None):
    base = base or build.REGIONS[BLEND_BASE]
    lons, lats = build.target_grid(region)
    step = region[4]
    xs = np.nonzero((lons >= base[0] - step / 2) & (lons <= base[2] + step / 2))[0]
    ys = np.nonzero((lats >= base[1] - step / 2) & (lats <= base[3] + step / 2))[0]
    if not len(xs) or not len(ys):
        return None
    return (int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1)

def sha_of(paths, n=12):
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.relpath(p, HERE).encode() + b'\x00')
        try:
            with open(p, 'rb') as f:
                h.update(f.read())
        except OSError:
            h.update(b'-')
    return h.hexdigest()[:n]

def phys_files():
    out = set()
    for g in PHYS_GLOBS:
        out.update((p for p in glob.glob(os.path.join(HERE, g)) if not os.path.basename(p).startswith('test_')))
    return sorted(out)

def run_complete(hours_dir, run):
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(hours_dir)), 'latest.json')) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return True
    return m.get('runId') != run or bool(m.get('complete', True))

def npy_bytes(a):
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asanyarray(a), allow_pickle=False)
    return buf.getvalue()

class NpzStream:

    def __init__(self, path, level=1):
        self.zf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=level, allowZip64=True)
        self.big = None

    def add(self, name, a):
        if self.big is not None:
            self.big.close()
            self.big = None
        self.zf.writestr(f'{name}.npy', npy_bytes(a))

    def begin(self, name, shape, dtype=np.uint8):
        self.big = self.zf.open(f'{name}.npy', 'w', force_zip64=True)
        np.lib.format.write_array_header_1_0(self.big, {'descr': np.lib.format.dtype_to_descr(np.dtype(dtype)), 'fortran_order': False, 'shape': tuple(shape)})

    def part(self, a):
        self.big.write(np.ascontiguousarray(a).tobytes())

    def close(self):
        if self.big is not None:
            self.big.close()
        self.zf.close()
_CTX = {}

def _read_hour(v):
    c = _CTX
    with np.load(os.path.join(c['hours_dir'], f'{v:%Y%m%d%H}.npz')) as h:
        elev = h['elevation'] if 'elevation' in h.files else None
        st = verify.sample_hour(h, c['iy'], c['ix'], c['r_nb'], STATION_FIELDS, c['rs']) if v in c['st'] else None
        slim = None
        if v in c['slim']:
            keep, have = (c['keep'], h['t2'] != 255 if 't2' in h.files else None)
            mask = keep if have is None or have.all() else have if keep is None else keep & have
            off = build.sun_elevation(v, c['lats'][:, None], c['lons'][None, :]) < SUN_MIN
            if mask is not None:
                off |= ~mask
            slim = np.full((len(SLIM_FIELDS), len(c['lats']), len(c['lons'])), 255, np.uint8)
            for j, name in enumerate(SLIM_FIELDS):
                if name in h.files:
                    slim[j] = np.where(off, 255, h[name])
        bl = None
        if v in c['bl']:
            bl = {}
            for rname, (y0, y1, x0, x1) in c['wins'].items():
                a = np.full((len(BLEND_FIELDS), y1 - y0, x1 - x0), 255, np.uint8)
                for j, name in enumerate(BLEND_FIELDS):
                    if name in h.files:
                        a[j] = h[name][y0:y1, x0:x1]
                bl[rname] = a
    return (elev, st, slim, bl)

def write(hours_dir, run, out, model=None, build_id=None, published=False, day_has_grids=False, region=None, stations=None, blend_in=None, workers=None, published_t=None, complete=None, cells=None):
    t0 = time.time()
    model = model or os.path.basename(os.path.dirname(os.path.abspath(hours_dir)))
    region = region or build.REGIONS[model]
    if build_id is None:
        build_id = f"{os.environ.get('GITHUB_RUN_ID', 'local')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    if blend_in is None:
        blend_in = os.environ.get('BLEND_IN', '1') != '0'
    if stations is None:
        stations = verify.load_stations()
    if complete is None:
        complete = run_complete(hours_dir, run)
    stations_v = hashlib.sha256(json.dumps(stations, separators=(',', ':')).encode()).hexdigest()[:12]
    hours = own_hours(hours_dir, run)
    if not hours:
        log(f'ops: {model} {run}: none of its own hours in the store')
        return None
    r0 = parse_run(run)
    base = os.path.join(out, 'ops', model, f'{r0:%Y%m%d}', f'{r0:%H}', f'b{build_id}')
    os.makedirs(base, exist_ok=True)
    lons, lats = build.target_grid(region)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing='ij')
    keep = build.keep_mask(model, lat2d, lon2d)
    del lat2d, lon2d
    w, s, e, n, step = region[:5]
    grid = {'west': w, 'south': s, 'east': e, 'north': n, 'step': step, 'nx': len(lons), 'ny': len(lats)}
    files = {}
    st_hours = [(v, ld) for v, ld in hours if ld in STATION_LEADS]
    slim_hours = [(v, ld) for v, ld in hours if ld in SLIM_LEADS] if grids_due(model, run, day_has_grids) else []
    wins = blend_windows(model, region) if blend_in and complete else {}
    bl_hours = [(v, ld) for v, ld in hours if ld in BLEND_LEADS] if wins else []
    idx, iy, ix = cells if cells is not None else verify.station_cells(model, stations, region)
    r_nb = verify.near_cells(step)
    st_codes = np.full((len(st_hours), len(idx), len(STATION_FIELDS)), 255, np.uint8)
    slim = None
    if slim_hours:
        slim = NpzStream(os.path.join(base, 'slim.npz.tmp'))
        slim.begin('codes', (len(slim_hours), len(SLIM_FIELDS), len(lats), len(lons)))
    bl = None
    if bl_hours:
        bl = {rname: np.full((len(bl_hours), len(BLEND_FIELDS), y1 - y0, x1 - x0), 255, np.uint8) for rname, (y0, y1, x0, x1) in wins.items()}
    elev = None
    st_set, slim_set, bl_set = ({v: k for k, (v, _) in enumerate(x)} for x in (st_hours, slim_hours, bl_hours))
    todo = [v for v, _ in hours if v in st_set or v in slim_set or v in bl_set]
    _CTX.update(hours_dir=hours_dir, model=model, lats=lats, lons=lons, keep=keep, iy=iy, ix=ix, r_nb=r_nb, wins=wins, rs=verify.smooth_cells(step), st=set(st_set), slim=set(slim_set), bl=set(bl_set))
    workers = max(1, min(workers or os.cpu_count() or 1, len(todo)))
    pool = mp.get_context('fork').Pool(workers) if workers > 1 else None
    try:
        for i in range(0, len(todo), 2 * workers):
            part = todo[i:i + 2 * workers]
            for v, (e_h, st_h, slim_h, bl_h) in zip(part, pool.map(_read_hour, part) if pool else map(_read_hour, part)):
                if elev is None:
                    elev = e_h
                if st_h is not None:
                    st_codes[st_set[v]] = st_h
                if slim_h is not None:
                    slim.part(slim_h)
                if bl_h is not None:
                    for rname, a in bl_h.items():
                        bl[rname][bl_set[v]] = a
    finally:
        if pool:
            pool.close()
            pool.join()
        _CTX.clear()
    enc = {k: list(build.ENC[k]) for k in STATION_FIELDS if k in build.ENC}
    enc['overdev_nb'] = enc['overdev']
    for k in verify.SMOOTHED:
        if k[:-2] in enc:
            enc[k] = enc[k[:-2]]
    if st_hours:
        has = (st_codes[:, :, STATION_FIELDS.index('t2')] != 255).any(axis=0)
        p = os.path.join(base, 'stations.npz')
        np.savez_compressed(p, stamps=np.array([f'{v:%Y%m%d%H}' for v, _ in st_hours]), idx=idx[has], codes=st_codes[:, has, :], fields=np.array(STATION_FIELDS), enc=json.dumps(enc), model=model, run=run, stations_v=stations_v, build=build_id)
        files['stations.npz'] = {'hours': len(st_hours), 'stations': int(has.sum())}
    if slim is not None:
        e2 = np.full((len(lats), len(lons)), 255, np.uint8) if elev is None else elev
        if keep is not None:
            e2 = np.where(keep, e2, 255).astype(np.uint8)
        slim.add('stamps', np.array([f'{v:%Y%m%d%H}' for v, _ in slim_hours]))
        slim.add('fields', np.array(SLIM_FIELDS))
        slim.add('enc', np.array(json.dumps({k: list(build.ENC[k]) for k in SLIM_FIELDS + ('elevation',)})))
        slim.add('elev', e2)
        slim.add('grid', np.array(json.dumps(grid)))
        slim.add('model', np.array(model))
        slim.add('run', np.array(run))
        slim.add('sun_min', np.array(SUN_MIN))
        slim.close()
        os.replace(os.path.join(base, 'slim.npz.tmp'), os.path.join(base, 'slim.npz'))
        files['slim.npz'] = {'hours': len(slim_hours)}
        if complete:
            day = f'{r0:%Y%m%d}'
            marker = os.path.join(out, 'ops', model, 'grids', f'{day}.json')
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            with open(marker, 'w') as f:
                json.dump({'day': day, 'model': model, 'run': run, 'build': build_id, 'key': f'ops/{model}/{day}/{r0:%H}/b{build_id}/slim.npz'}, f)
    if bl is not None:
        for rname, win in wins.items():
            fk = 'blendin' if rname == 'conus' else f'blendin_{rname}'
            try:
                files[fk] = write_blendin(out, model, run, region, lons, lats, win, bl[rname], bl_hours, elev, rname)
            except Exception as e:
                log(f'ops: {model} {run}: blend inputs ({rname}) not written: {e!r}')
                files[fk] = {'error': repr(e)[:300]}
    elif blend_in and (not complete) and blend_windows(model, region):
        files['blendin'] = {'skipped': 'run built in part'}
    meta = os.path.join(out, 'ops', 'meta', f'stations-{stations_v}.json')
    os.makedirs(os.path.dirname(meta), exist_ok=True)
    with open(meta, 'w') as f:
        json.dump({'stations': stations}, f, separators=(',', ':'))
    size = sum((os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs))
    done = {'model': model, 'run': run, 'build': build_id, 'stations_v': stations_v, 'hours_v': build.HOURS_VERSION, 'phys': sha_of(phys_files()), 'complete': bool(complete), 'aircorr': os.environ.get('AIR_CORRECTIONS', '0') == '1', 'acc_class': os.environ.get('ACC_CLASS', ''), 'published': bool(published), 'published_t': published_t or datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ') if published else None, 'leads': [ld for _, ld in hours], 'files': files, 'bytes': size, 'sec': round(time.time() - t0, 1)}
    with open(os.path.join(base, 'done.json'), 'w') as f:
        json.dump(done, f, separators=(',', ':'))
    what = ', '.join((f"{k} {v['hours']} h" if 'hours' in v else f'{k} {next(iter(v.values()))}' for k, v in files.items()))
    log(f"ops: {model} {run}: {what}, {size / 1000000.0:.1f} MB in {done['sec']} s")
    return done

def blend_windows(model, region):
    out = {}
    for rname, (base, box) in BLEND_REGIONS.items():
        if model == base or (rname in BLEND_MEMBERS and model not in BLEND_MEMBERS[rname]):
            continue
        win = conus_window(region, box or build.REGIONS[base][:4])
        if win is not None:
            out[rname] = win
    return out

def write_blendin(out, model, run, region, lons, lats, win, bl, bl_hours, elev, rname='conus'):
    y0, y1, x0, x1 = win
    step = region[4]
    d = os.path.join(out, 'cache', 'blendin', f'v={BLEND_V}', model, run)
    os.makedirs(d, exist_ok=True)
    cgrid = {'west': round(region[0] + x0 * step, 6), 'north': round(region[3] - y0 * step, 6), 'step': step, 'nx': x1 - x0, 'ny': y1 - y0}
    ce = np.full((y1 - y0, x1 - x0), 255, np.uint8) if elev is None else elev[y0:y1, x0:x1]
    tiles.savez(os.path.join(d, f'{rname}.npz'), {'stamps': np.array([f'{v:%Y%m%d%H}' for v, _ in bl_hours]), 'runs': np.array([run] * len(bl_hours)), 'fields': np.array(BLEND_FIELDS), 'codes': bl, 'elev': ce, 'grid': json.dumps(cgrid), 'enc': json.dumps({k: list(build.ENC[k]) for k in BLEND_FIELDS + ('elevation',)}), 'model': model, 'run': run, 'aircorr': os.environ.get('AIR_CORRECTIONS', '0') == '1'})
    return {'hours': len(bl_hours), 'key': f'cache/blendin/v={BLEND_V}/{model}/{run}/{rname}.npz'}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('write')
    w.add_argument('hours')
    w.add_argument('run')
    w.add_argument('out')
    w.add_argument('--model')
    w.add_argument('--build')
    w.add_argument('--published', action='store_true')
    w.add_argument('--published-t', help='when the send finished (UTC, ISO)')
    w.add_argument('--incomplete', action='store_true', default=None, help="the run built in part (default: its manifest's)")
    w.add_argument('--day-has-grids', action='store_true')
    w.add_argument('--workers', type=int, help='processes reading the hours (default: one a core)')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    done = write(a.hours, a.run, a.out, a.model, a.build, a.published, a.day_has_grids, workers=a.workers, published_t=a.published_t or None, complete=False if a.incomplete else None)
    return 0 if done else 3
if __name__ == '__main__':
    sys.exit(main())
