from __future__ import annotations
import os
import time
import numpy as np
import leelattice
import runfields_native
import runwave
import shade as shade_mod
import sky
import wavebars
FIELDS = ('rrun', 'rrunf', 'wrun', 'wrunf', 'wrunz', 'wtop', 'crest', 'cshade', 'cthin', 'csun')
RUN_FIELDS = ('rrun', 'rrunf', 'wrun', 'wrunf', 'wrunz', 'wtop', 'crest')
FINE = {'rrunf': 'rrun', 'wrunf': 'wrun'}
SHADE_FIELDS = ('cshade', 'cthin', 'csun')
NATIVE = None
NATIVE_TOLERANCE = 0.0001
LOG = lambda *a: print(*a, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
_DIRS = {k: [os.path.join(HERE, 'geo', k), os.path.join(HERE, '..', 'public', 'geo', k)] for k in ('lee', 'ridge')}
_EXPOSURES: dict = {}

def exposure(kind):
    if kind not in _EXPOSURES:
        d = next((p for p in _DIRS[kind] if os.path.exists(os.path.join(p, 'index.json'))), None)
        _EXPOSURES[kind] = runwave.Exposure(d) if d else None
    return _EXPOSURES[kind]

def _steps(v, lo, hi, transform):
    x = np.clip(np.asarray(v, np.float64), lo, hi)
    if transform == 'sqrt':
        a, b = (np.sqrt(max(lo, 0)), np.sqrt(hi))
        return (np.sqrt(x) - a) / (b - a) * 254
    return (x - lo) / (hi - lo) * 254

def _encode(v, lo, hi, transform):
    s = _steps(v, lo, hi, transform)
    return np.where(np.isfinite(s), np.rint(np.nan_to_num(s)), 255).astype(np.uint8)

class _Decoded:

    def __init__(self, q, enc):
        self.q, self.enc, self.done = (q, enc, {})

    def __contains__(self, n):
        return n in self.enc and getattr(self.q.get(n), 'ndim', 0) == 2

    def get(self, n, default=None):
        if n not in self:
            return default
        if n not in self.done:
            self.done[n] = sky.decode(self.q[n], *self.enc[n]).astype(np.float32)
        return self.done[n]

    def __getitem__(self, n):
        if n not in self:
            raise KeyError(n)
        return self.get(n)
OVERVIEW_BLOCK = 64
OVERVIEW_REACH = 3

def _overview_again(q, enc, fine, changed):
    if any((n not in q for n in sky.FIELDS)):
        return sky.cloud_overview(q, enc, fine)
    ny, nx = changed.shape
    out = {n: np.array(q[n], copy=True) for n in sky.FIELDS}
    B, R, M = (OVERVIEW_BLOCK, OVERVIEW_REACH, OVERVIEW_REACH + 1)
    grids = {k: v for k, v in q.items() if getattr(v, 'shape', None) == (ny, nx)}
    for y0 in range(0, ny, B):
        for x0 in range(0, nx, B):
            y1, x1 = (min(ny, y0 + B), min(nx, x0 + B))
            if not changed[max(0, y0 - R):y1 + R, max(0, x0 - R):x1 + R].any():
                continue
            wy0, wy1, wx0, wx1 = (max(0, y0 - M), min(ny, y1 + M), max(0, x0 - M), min(nx, x1 + M))
            ov = sky.cloud_overview({k: v[wy0:wy1, wx0:wx1] for k, v in grids.items()}, enc, fine)
            for k, a in ov.items():
                out[k][y0:y1, x0:x1] = a[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0]
    return out

def _running(q, enc, grid, nodata, native):
    out = {}

    def put(n, v):
        out[n] = np.where(nodata, 255, _encode(v, *enc[n])).astype(np.uint8)
    lat_of, run_of = (runfields_native, runfields_native) if native else (leelattice, runwave)
    lee, ridge = (exposure('lee'), exposure('ridge'))
    bands = None
    if lee is not None and 'wtrap' in q:
        lat = lat_of.lattice(_Decoded(q, enc), grid, lee.edges)
        bands = lat_of.lattice_cells(lat, grid)
    dec = _Decoded(q, enc)
    bars = bands is not None and 'cuclass' in q
    r = run_of.run_fields(dec, grid, lee, ridge, bands, bars=bars)
    fine = float(grid['step']) <= sky.FINE_STEP
    if bars and 'bar_own' in r:
        low = wavebars.cloud(dec, r['bar_own'], grid)
        if low is not None:
            low0 = dec.get('cloudlow')
            gain = np.where(np.isfinite(low0), low - np.nan_to_num(low0, nan=0.0), 0.0) > 0
            q = dict(q)
            new = {'cloudlow': low}
            if dec.get('cloud') is not None:
                new['cloud'] = np.fmax(dec.get('cloud'), low)
            if dec.get('sun') is not None:
                new['sun'] = dec.get('sun') * (1 - wavebars.SUN_LOSS * np.clip((low - np.nan_to_num(low0, nan=0.0)) / 100, 0, 1))
            for n, v in new.items():
                q[n] = out[n] = np.where(gain, _encode(v, *enc[n]), q[n]).astype(np.uint8)
            ov = _overview_again(q, enc, fine, gain)
            q.update(ov)
            out.update(ov)
            dec = _Decoded(q, enc)
        wb = wavebars.apply(q['cuclass'], dec, r['bar_own'], grid, fine)
        if wb is not None:
            out['cuclass'] = wb
    put('rrun', r['ridge'])
    put('wrun', r['wave'])
    for f, n in FINE.items():
        s = _steps(r[{'rrun': 'ridge', 'wrun': 'wave'}[n]], *enc[n])
        put(f, s - np.rint(s))
    put('wrunz', np.where(r['wave'] > 0, r['wave_agl'], np.nan))
    put('wtop', r['wave_top'])
    put('crest', r['crest'])
    return (out, q, bands is not None)

def _differ(a, b):
    bad, near = ([], [])
    for k in sorted((set(a) | set(b)) - set(FINE)):
        x, y = (a.get(k), b.get(k))
        if x is None or y is None or x.shape != y.shape:
            bad.append(f'{k} missing')
            continue
        x, y = (x.astype(np.int32), y.astype(np.int32))
        f = next((f for f, n in FINE.items() if n == k), None)
        unit = ' code(s)'
        if f is not None and (f in a or f in b):
            if f not in a or f not in b or a[f].shape != x.shape or (b[f].shape != y.shape):
                bad.append(f'{f} missing')
                continue
            x, y, unit = (x * 254 + a[f], y * 254 + b[f], '/254 code(s)')
        diff = x != y
        if not diff.any():
            continue
        worst = int(np.abs(x - y)[diff].max())
        (near if float(diff.mean()) <= NATIVE_TOLERANCE and worst <= 1 else bad).append(f'{k} {diff.sum()} cells, {worst}{unit}')
    return (bad, near)

def _pick(q, enc, grid, nodata, when):
    global NATIVE
    if NATIVE is None and (not runfields_native.available()):
        NATIVE = False
        LOG(f'ridge/wave running: leelattice.py and runwave.py (C++ not available: {runfields_native.N.error})')
    if NATIVE:
        try:
            return _running(q, enc, grid, nodata, True)[:2]
        except Exception as e:
            NATIVE = False
            LOG(f'::warning::ridge/wave running: the C++ failed on {when:%a %HZ} ({e!r}), leelattice.py and runwave.py from here')
    out, q2, lattice = _running(q, enc, grid, nodata, False)
    if NATIVE is None and lattice:
        t0 = time.time()
        try:
            bad, near = _differ(out, _running(q, enc, grid, nodata, True)[0])
        except Exception as e:
            bad, near = ([f'it failed: {e!r}'], [])
        NATIVE = not bad
        if bad:
            LOG(f"::warning::ridge/wave running: the C++ differs from leelattice.py and runwave.py on {when:%a %HZ} ({'; '.join(bad[:8])}), the Python for this run")
        else:
            LOG(f'ridge/wave running: C++ (matches leelattice.py and runwave.py on {when:%a %HZ}: {len(out) - len(FINE) - len(near)} of {len(out) - len(FINE)} fields identical' + (f"; {', '.join(near)}" if near else '') + f'; checked in {time.time() - t0:.1f}s)')
    return (out, q2)

def publish(q, enc, grid, when, mid_by_ceil=False, agl_heights=True):
    if 'wstar' not in q or 'elevation' not in q:
        return {}
    nodata = ~np.isfinite(sky.decode(q['wstar'], *enc['wstar']))
    out, q = _pick(q, enc, grid, nodata, when)

    def put(n, v):
        out[n] = np.where(nodata, 255, _encode(v, *enc[n])).astype(np.uint8)
    s = shade_mod.shade(_Decoded(q, enc), grid, when, mid_by_ceil, agl_heights, rays=True)
    put('cshade', s['shade'])
    put('cthin', s['thin'])
    put('csun', s['sun'])
    return out
