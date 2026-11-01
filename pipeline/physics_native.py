from __future__ import annotations
import ctypes
import glob
import hashlib
import os
import subprocess
import sys
import numpy as np
import physics as _py
from physics import *
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'native')
FLAGS = ['-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off', '-fno-math-errno']
MAX_LEVELS = 62
_F = ctypes.POINTER(ctypes.c_float)
_I32 = ctypes.POINTER(ctypes.c_int32)

class _CProfile(ctypes.Structure):
    _fields_ = [('n', ctypes.c_int64), ('L', ctypes.c_int32), ('Lw', ctypes.c_int32), ('z', _F), ('lnp', _F), ('t', _F), ('td', _F), ('zw', _F), ('u', _F), ('v', _F), ('zs', _F)]
_lib = None
_tried = False
error = None

def _build():
    sources = sorted(glob.glob(os.path.join(SRC, '*.cpp')))
    headers = sorted(glob.glob(os.path.join(SRC, '*.h')))
    cxx = os.environ.get('CXX', 'g++')
    h = hashlib.sha256(' '.join([cxx, *FLAGS]).encode())
    for f in headers + sources:
        h.update(os.path.basename(f).encode())
        with open(f, 'rb') as fh:
            h.update(fh.read())
    out_dir = os.path.join(SRC, 'build')
    so = os.path.join(out_dir, f'libcxphys-{h.hexdigest()[:16]}.so')
    if not os.path.exists(so):
        os.makedirs(out_dir, exist_ok=True)
        tmp = f'{so}.{os.getpid()}.tmp'
        r = subprocess.run([cxx, *FLAGS, '-I', SRC, *sources, '-o', tmp], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f'{cxx} failed: {r.stderr.strip()[-2000:]}')
        os.replace(tmp, so)
    return ctypes.CDLL(so)

def lib():
    global _lib, _tried, error
    if _tried:
        return _lib
    _tried = True
    if os.environ.get('PIPELINE_PHYSICS', '').lower() == 'python':
        error = 'PIPELINE_PHYSICS=python'
        return None
    try:
        _lib = _build()
        _declare(_lib)
    except Exception as e:
        error = str(e)
        _lib = None
    return _lib

def available():
    return lib() is not None

def _declare(L):
    i64, i32, f32 = (ctypes.c_int64, ctypes.c_int32, ctypes.c_float)
    L.cx_profile_build.restype = None
    L.cx_profile_build.argtypes = [i64, i32, _I32, _F, *[_F] * 6, *[_F] * 5, _F, _F, f32, *[_F] * 7]
    L.cx_profile_at.restype = None
    L.cx_profile_at.argtypes = [ctypes.POINTER(_CProfile), _F, i32, ctypes.POINTER(_F), _I32, ctypes.POINTER(_F)]
    L.cx_column_at.restype = None
    L.cx_column_at.argtypes = [i64, i32, _F, _F, _F, _F]
    L.cx_block_at.restype = None
    L.cx_block_at.argtypes = [ctypes.POINTER(_CProfile), _F, _F, _F, _F, _F, _F]

def cfunc(name, argtypes, restype=None):
    L = lib()
    if L is None:
        return None
    f = getattr(L, name)
    f.argtypes, f.restype = (argtypes, restype)
    return f

def ptr(a):
    return None if a is None else a.ctypes.data_as(_F)

def f32(a):
    return a if isinstance(a, np.ndarray) and a.dtype == np.float32 and a.flags.c_contiguous else None

def grid_f32(a, shape):
    if not isinstance(a, np.ndarray) or a.dtype != np.float32:
        return None
    try:
        return np.ascontiguousarray(np.broadcast_to(a, shape))
    except ValueError:
        return None

def all_f32(arrays, shape):
    return all((a is None or (f32(a) is not None and a.shape == tuple(shape)) for a in arrays))

class Profile(_py.Profile):

    def __init__(self, zs, ps, t2, td2, u10, v10, z, p, t, td, u, v, uh=None, vh=None, hub=100.0):
        L = z.shape[0]
        shape = np.shape(zs)
        n = int(np.prod(shape))
        hub_ok = uh is not None and vh is not None
        if lib() is None or L + 2 > MAX_LEVELS or (not all_f32([zs, ps, t2, td2, u10, v10], shape)) or (not all_f32([z, t, td, u, v], (L,) + tuple(shape))) or (hub_ok and (not all_f32([uh, vh], shape))) or (not isinstance(hub, (int, float))):
            self.native = False
            super().__init__(zs, ps, t2, td2, u10, v10, z, p, t, td, u, v, uh, vh, hub)
            return
        self.native = True
        order = np.argsort(-np.asarray(p)).astype(np.int32)
        p_sorted = np.ascontiguousarray(np.asarray(p, dtype=np.float32)[order])
        Lw = L + 1 + (1 if hub_ok else 0)
        self.z, pl, self.t, self.td = (np.empty((L + 1,) + shape, np.float32) for _ in range(4))
        self.zw, self.u, self.v = (np.empty((Lw,) + shape, np.float32) for _ in range(3))
        lib().cx_profile_build(n, L, order.ctypes.data_as(_I32), ptr(p_sorted), ptr(zs), ptr(ps), ptr(t2), ptr(td2), ptr(u10), ptr(v10), ptr(z), ptr(t), ptr(td), ptr(u), ptr(v), ptr(uh) if hub_ok else None, ptr(vh) if hub_ok else None, float(hub), ptr(self.z), ptr(pl), ptr(self.t), ptr(self.td), ptr(self.zw), ptr(self.u), ptr(self.v))
        self.lnp = np.log(pl)
        self.zs = zs
        self.ztop = self.z[-1]

    def c(self):
        n = self.zs.size
        return _CProfile(n, self.z.shape[0], self.zw.shape[0], ptr(self.z), ptr(self.lnp), ptr(self.t), ptr(self.td), ptr(self.zw), ptr(self.u), ptr(self.v), ptr(self.zs))

    def at(self, zq, *names):
        if not getattr(self, 'native', False):
            return super().at(zq, *names)
        shape = self.zs.shape
        q = grid_f32(zq, shape)
        cols = [getattr(self, n, None) for n in names]
        if q is None or any((f32(a) is None for a in cols)):
            return super().at(zq, *names)
        outs = [np.empty(shape, np.float32) for _ in names]
        wind = np.array([n in ('u', 'v') for n in names], np.int32)
        cp = self.c()
        lib().cx_profile_at(ctypes.byref(cp), ptr(q), len(names), (_F * len(names))(*[ptr(a) for a in cols]), wind.ctypes.data_as(_I32), (_F * len(names))(*[ptr(o) for o in outs]))
        outs = [np.exp(o) if n == 'lnp' else o for n, o in zip(names, outs)]
        return outs if len(outs) > 1 else outs[0]

def column_at(z, a, zq):
    zq_ = grid_f32(zq, z.shape[1:])
    if lib() is None or zq_ is None or f32(z) is None or (f32(a) is None) or (a.shape != z.shape):
        return _py.column_at(z, a, zq)
    out = np.empty(z.shape[1:], np.float32)
    lib().cx_column_at(zq_.size, z.shape[0], ptr(z), ptr(a), ptr(zq_), ptr(out))
    return out

def is_native(prof):
    return isinstance(prof, Profile) and getattr(prof, 'native', False)
_exp_mode = []

def _np_exp_mode():
    if not _exp_mode:
        f = cfunc('cx_np_exp', [ctypes.c_int64, _F, ctypes.c_int32, _F])
        x = np.concatenate([np.linspace(-40, 12, 200001, dtype=np.float32), np.random.default_rng(0).uniform(-40, 12, 50000).astype(np.float32)])
        ref, out, mode = (np.exp(x), np.empty_like(x), None)
        for m in (1, 0):
            f(x.size, ptr(x), m, ptr(out))
            if np.array_equal(out, ref):
                mode = m
                break
        _exp_mode.append(mode)
    return _exp_mode[0]

def _py_number(x):
    return type(x) in (int, float) and np.isfinite(x) and (abs(x) < 1e+30)

def cumulus_top(prof, zbase, tbase, active, step=_py.CU_STEP, max_above=10000.0, mix=None):
    f = cfunc('cx_cumulus_top', [ctypes.POINTER(_CProfile), _F, _F, ctypes.c_void_p, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, _F])
    if f is None or not is_native(prof) or (not _py_number(step)) or (not _py_number(max_above)) or (step <= 0):
        return _py.cumulus_top(prof, zbase, tbase, active, step, max_above, mix)
    shape = prof.zs.shape
    steps = int(max_above / step)
    ok = zbase is not None and tbase is not None and all_f32([zbase, tbase], shape) and isinstance(active, np.ndarray) and (active.dtype == np.bool_) and (active.shape == shape) and active.flags.c_contiguous and (steps < 2 ** 31) and (mix is None or all_f32([mix], shape))
    mode = _np_exp_mode() if ok else None
    if mode is None:
        return _py.cumulus_top(prof, zbase, tbase, active, step, max_above, mix)
    out = np.empty(shape, np.float32)
    cp = prof.c()
    f(ctypes.byref(cp), ptr(zbase), ptr(tbase), active.ctypes.data, float(step), max(steps, 0), mode, None if mix is None else mix.ctypes.data, ptr(out))
    return out
_LAYER_COLS = {'t': 0, 'td': 1, 'u': 2, 'v': 3}

def layer_mean(prof, zb, zt, names, n=8):
    f = cfunc('cx_layer_mean', [ctypes.POINTER(_CProfile), _F, _F, ctypes.c_int32, ctypes.c_int32, _I32, ctypes.POINTER(_F)])
    which = [_LAYER_COLS.get(k) if isinstance(k, str) else None for k in names] if isinstance(names, (list, tuple)) else None
    shape = getattr(prof, 'zs', np.empty(0)).shape
    zb_, zt_ = (grid_f32(zb, shape), grid_f32(zt, shape))
    if f is None or not is_native(prof) or (not which) or (None in which) or (type(n) is not int) or (not 2 <= n < 2 ** 24) or (zb_ is None) or (zt_ is None):
        return _py.layer_mean(prof, zb, zt, names, n)
    outs = [np.empty(shape, np.float32) for _ in names]
    which = np.array(which, np.int32)
    cp = prof.c()
    f(ctypes.byref(cp), ptr(zb_), ptr(zt_), n, len(names), which.ctypes.data_as(_I32), (_F * len(names))(*[ptr(o) for o in outs]))
    return outs
_SINK_MAXN = 256
_SINK_MAXW = 64

def _sink_grid(a, shape, dtypes):
    ok = isinstance(a, np.ndarray) and a.dtype in dtypes and a.flags.c_contiguous and (a.shape == tuple(shape))
    return a if ok else None

def sink_rate(prof: Profile, wz, w, ws, zi, depth, rain, refd=None, lee=None, n=10, parts=False, snow=None):

    def fallback():
        return _py.sink_rate(prof, wz, w, ws, zi, depth, rain, refd, lee, n, parts, snow)
    if lib() is None or not is_native(prof) or type(n) is not int or (not 2 <= n <= _SINK_MAXN):
        return fallback()
    shape = prof.zs.shape
    both = (np.dtype(np.float32), np.dtype(np.float64))
    if any((a is None for a in (zi, depth, rain))) or not all_f32([zi, depth, rain, refd], shape) or _sink_grid(ws, shape, both) is None or (lee is not None and _sink_grid(lee, shape, both) is None):
        return fallback()
    K = 0
    if w is not None:
        if not isinstance(wz, np.ndarray) or wz.ndim != len(shape) + 1 or (not 1 <= wz.shape[0] < _SINK_MAXW) or (not all_f32([wz, w], (wz.shape[0],) + tuple(shape))):
            return fallback()
        K = wz.shape[0]
    parts = bool(parts)
    r = np.maximum(np.nan_to_num(rain, nan=0), 0)
    if refd is not None:
        r = np.maximum(r, np.where(np.nan_to_num(refd, nan=0) >= 15, np.minimum(_py.rain_from_dbz(refd), 50), 0))
    r = _py.snow_rate(r, snow)
    e = f32(np.exp(-r / 3))
    if e is None or e.shape != shape:
        return fallback()
    vp = ctypes.c_void_p
    ws64 = int(ws.dtype == np.float64)
    lee_kind = 0 if lee is None else 2 if lee.dtype == np.float64 else 1
    cells = prof.zs.size
    xs = np.empty(cells * n, np.float32)
    m = cfunc('cx_sink_x', [ctypes.c_int64, ctypes.c_int32, _F, _F, vp, ctypes.c_int32, _F], ctypes.c_int64)(cells, n, ptr(depth), ptr(zi), ws.ctypes.data, ws64, ptr(xs))
    if m < 0:
        return fallback()
    cb = np.cbrt(xs[:m * n])
    out = np.float64 if ws64 or lee_kind == 2 else np.float32
    extra = np.empty(shape, out)
    th = np.empty(shape, out) if parts else None
    why = np.empty(shape, np.float32) if parts else None
    cp = prof.c()
    rc = cfunc('cx_sink_rate', [ctypes.POINTER(_CProfile), ctypes.c_int32, _F, _F, _F, _F, vp, ctypes.c_int32, vp, ctypes.c_int32, _F, _F, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, vp, vp, _F], ctypes.c_int32)(ctypes.byref(cp), K, ptr(wz) if K else None, ptr(w) if K else None, ptr(depth), ptr(zi), ws.ctypes.data, ws64, None if lee is None else lee.ctypes.data, lee_kind, ptr(e), ptr(cb), m, n, int(parts), extra.ctypes.data, None if th is None else th.ctypes.data, ptr(why))
    if rc != 0:
        return fallback()
    return (extra, th, why) if parts else extra

def _exact_f32(x):
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x[()]
    if not isinstance(x, (int, float, np.integer, np.floating)):
        return None
    try:
        v = float(x)
    except (OverflowError, ValueError):
        return None
    with np.errstate(over='ignore'):
        return v if float(np.float32(v)) == v else None

def freezing_level(prof: Profile):
    fn = cfunc('cx_freezing_level', [ctypes.c_int64, ctypes.c_int32, _F, _F, _F, _F, _F])
    if fn is None or not is_native(prof):
        return _py.freezing_level(prof)
    t, z, zs, ztop = (prof.t, prof.z, prof.zs, prof.ztop)
    if f32(zs) is None or not all_f32([z, t], (t.shape[0],) + zs.shape) or t.shape[0] < 1 or (not all_f32([ztop], zs.shape)):
        return _py.freezing_level(prof)
    out = np.empty(zs.shape, np.float32)
    fn(zs.size, t.shape[0], ptr(z), ptr(t), ptr(zs), ptr(ztop), ptr(out))
    return out

def icing_potential(z, t, rh, clw, zs, zmax=7300.0):
    fn = cfunc('cx_icing_potential', [ctypes.c_int64, ctypes.c_int32, _F, _F, _F, _F, _F, ctypes.c_float, _F, _F, _F])
    zm = _exact_f32(zmax)
    if fn is None or zm is None or f32(zs) is None or (not isinstance(z, np.ndarray)) or (z.ndim != zs.ndim + 1) or (z.shape[0] < 1) or (not all_f32([z, t, rh, clw], (z.shape[0],) + zs.shape)):
        return _py.icing_potential(z, t, rh, clw, zs, zmax)
    pot, base, top = (np.empty(zs.shape, np.float32) for _ in range(3))
    fn(zs.size, z.shape[0], ptr(z), ptr(t), ptr(rh), ptr(clw), ptr(zs), zm, ptr(pot), ptr(base), ptr(top))
    return (pot, base, top)

def ellrod_turbulence(prof: Profile, ddx, ddy, smooth=None, zi=None, zmin_agl=CAT_FLOOR_AGL, zmax=10500.0, dz=500.0, band=False):
    fn = cfunc('cx_winds_at_heights', [ctypes.POINTER(_CProfile), ctypes.c_int32, _F, _F, _F])
    if fn is None or not is_native(prof) or f32(prof.zs) is None or (not all_f32([prof.zw, prof.u, prof.v], (prof.zw.shape[0],) + prof.zs.shape)):
        return _py.ellrod_turbulence(prof, ddx, ddy, smooth, zi, zmin_agl, zmax, dz, band)
    zs = prof.zs
    floor = zs + np.maximum(zmin_agl, 0 if zi is None else np.nan_to_num(zi, nan=0))
    hs = np.arange(dz, zmax + dz, dz, dtype=np.float32)
    uu, vv = (np.empty((len(hs),) + zs.shape, np.float32) for _ in range(2))
    cp = prof.c()
    fn(ctypes.byref(cp), len(hs), ptr(hs), ptr(uu), ptr(vv))
    winds = list(zip(uu, vv))
    return _py._ellrod_layers(prof, winds, hs, floor, ddx, ddy, smooth, dz, band)
_WAVE_CHUNK = 1 << 17

def wave_environment(prof: Profile):
    sample = cfunc('cx_wave_sample', [ctypes.POINTER(_CProfile), ctypes.c_int64, ctypes.c_int64, _F, _F, _F, ctypes.c_void_p, _F, _F, _F])
    finish = cfunc('cx_wave_finish', [ctypes.c_int64, _F, _F, _F, ctypes.c_void_p, _F, _F, ctypes.POINTER(ctypes.c_double), _F])
    if sample is None or not is_native(prof):
        return _py.wave_environment(prof)
    shape = prof.zs.shape
    n = prof.zs.size
    score = np.empty(n, np.float64)
    wavelen = np.empty(n, np.float32)
    cp = prof.c()
    for i0 in range(0, n, _WAVE_CHUNK):
        i1 = min(i0 + _WAVE_CHUNK, n)
        m = i1 - i0
        sl, su, cosang = (np.empty(m, np.float32) for _ in range(3))
        ok = np.empty(m, np.uint8)
        ts, lnps, us = (np.empty((30, m), np.float32) for _ in range(3))
        sample(ctypes.byref(cp), i0, i1, ptr(sl), ptr(su), ptr(cosang), ok.ctypes.data, ptr(ts), ptr(lnps), ptr(us))
        turn = np.ascontiguousarray(np.degrees(np.arccos(np.clip(cosang, -1, 1))), np.float32)
        th = np.ascontiguousarray(_py.theta(ts, np.exp(lnps)), np.float32)
        finish(m, ptr(sl), ptr(su), ptr(turn), ok.ctypes.data, ptr(th), ptr(us), score[i0:i1].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ptr(wavelen[i0:i1]))
    return (score.reshape(shape), wavelen.reshape(shape))
if __name__ == '__main__':
    print('C++ physics:', 'available' if available() else f'not available ({error})', file=sys.stderr)
