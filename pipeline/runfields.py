from __future__ import annotations
import os
import numpy as np
import leelattice
import runwave
import shade as shade_mod
import sky
FIELDS = ('rrun', 'rrunf', 'wrun', 'wrunf', 'wrunz', 'wtop', 'crest', 'cshade', 'cthin', 'csun')
RUN_FIELDS = ('rrun', 'rrunf', 'wrun', 'wrunf', 'wrunz', 'wtop', 'crest')
FINE = {'rrunf': 'rrun', 'wrunf': 'wrun'}
SHADE_FIELDS = ('cshade', 'cthin', 'csun')
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

def publish(q, enc, grid, when, mid_by_ceil=False, agl_heights=True):
    if 'wstar' not in q or 'elevation' not in q:
        return {}
    nodata = ~np.isfinite(sky.decode(q['wstar'], *enc['wstar']))
    out = {}

    def put(n, v):
        out[n] = np.where(nodata, 255, _encode(v, *enc[n])).astype(np.uint8)
    lee, ridge = (exposure('lee'), exposure('ridge'))
    bands = None
    if lee is not None and 'wtrap' in q:
        lat = leelattice.lattice(_Decoded(q, enc), grid, lee.edges)
        bands = leelattice.lattice_cells(lat, grid)
    r = runwave.run_fields(_Decoded(q, enc), grid, lee, ridge, bands)
    put('rrun', r['ridge'])
    put('wrun', r['wave'])
    for f, n in FINE.items():
        s = _steps(r[{'rrun': 'ridge', 'wrun': 'wave'}[n]], *enc[n])
        put(f, s - np.rint(s))
    put('wrunz', np.where(r['wave'] > 0, r['wave_agl'], np.nan))
    put('wtop', r['wave_top'])
    put('crest', r['crest'])
    del r, bands
    s = shade_mod.shade(_Decoded(q, enc), grid, when, mid_by_ceil, agl_heights, rays=True)
    put('cshade', s['shade'])
    put('cthin', s['thin'])
    put('csun', s['sun'])
    return out
