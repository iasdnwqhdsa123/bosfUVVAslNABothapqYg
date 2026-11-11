from __future__ import annotations
import ctypes
import math
import numpy as np
import leelattice as LL
import physics_native as N
import runwave as RW
NODES_A_PASS = 64
PAIRS_A_PASS = 8192
MAX_BANDS = 64
_P, _I32, _I64, _F64 = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_double)
_SIGS = {'cx_lee_sample': ([_I64, _P, _I32, _I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P], None), 'cx_lee_columns': ([_I64, _I32, _P, _P, _P, _P, _P, _P, _I32, _P, _P, _P, _I32, _P, _I32, _P, _P, _P, _P, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P, _P], None), 'cx_lee_bands': ([_I64, _I32, _P, _I32, _I32, _I32, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P], None), 'cx_lattice_at': ([_I64, _P, _P, _F64, _I64, _I64, _I64, _I64, _P, _P, _P, _P, _I32, _P, _P, _P, _P], None), 'cx_lee_cells': ([_I64, _I64, _P, _P, _I32, _I64, _I64, _I32, _I64, _P, _P, _P, _I32, _I32, _I32, _I32, _P, _F64, _P, _P, _P, _P, _P, _P, _P, _P, _I64, _P, _P, _P], _I64), 'cx_lee_best': ([_I64, _P, _P, _I32, _P, _P, _P, _P], None), 'cx_ridge_cells': ([_I64, _P, _P, _I32, _I64, _I64, _I32, _I64, _P, _P, _P, _I32, _I32, _F64, _F64, _P, _P, _P, _P, _P, _P], _I32)}
_fns: dict = {}

def _fn(name):
    if name not in _fns:
        _fns[name] = N.cfunc(name, *_SIGS[name])
    return _fns[name]

def available():
    return N.lib() is not None

def _c(a, dtype=np.float64):
    return np.ascontiguousarray(a, dtype)

def _p(a):
    return None if a is None else a.ctypes.data

class _Sampler(LL._Sampler):

    def __init__(self, grid, lat, lon):
        super().__init__(grid, lat, lon)
        self.shape = (int(grid['ny']), int(grid['nx']))
        self.c = [_c(a, np.int64) for a in self.idx] + [_c(a) for a in (self.tx, self.ty) + tuple(self.w)] + [_c(self.inb, np.bool_)]

    def __call__(self, a):
        f = _fn('cx_lee_sample')
        if f is None or not isinstance(a, np.ndarray) or a.dtype not in (np.float32, np.float64) or (a.shape != self.shape) or (not a.flags.c_contiguous) or (min(a.shape) < 2) or (self.tx.ndim != 1):
            return super().__call__(a)
        out = np.empty(self.tx.shape[0])
        f(out.shape[0], _p(a), int(a.dtype == np.float64), a.shape[1], *(_p(x) for x in self.c), _p(out))
        return out
_tables: dict = {}

def _solver_tables():
    if not _tables:
        h = LL.U_SMOOTH
        um_at = sorted({float(z) for m in LL.MIDS for z in ((m, m + h, m - h) if m - h >= 0 else (m,))})
        hs = sorted({max(0.0, z + o) for z in um_at for o in (-h / 2, 0.0, h / 2)})
        col = {z: i for i, z in enumerate(hs)}
        um = lambda z: (col[max(0.0, z - h / 2)], col[max(0.0, z)], col[max(0.0, z + h / 2)])
        idx = np.full((LL.MIDS.shape[0], 9), -1, np.int32)
        for j, z in enumerate(LL.MIDS):
            z = float(z)
            idx[j, :3] = um(z)
            if not z - h < 0:
                idx[j, 3:6], idx[j, 6:] = (um(z + h), um(z - h))
        _tables.update(hs=np.array(hs, np.float64), idx=idx, high=_c(LL.MIDS > LL.ALPHA_Z, np.uint8), consts=np.array([LL.MIN_U, LL.L2_CAP, LL.N2_MIN, LL.N2_MAX, h * h], np.float64))
    return _tables

def _solve(P, sel, ks, nband):
    columns, bands_of = (_fn('cx_lee_columns'), _fn('cx_lee_bands'))
    if columns is None or bands_of is None:
        return LL._solve(P, sel, ks, nband)
    M = sel.shape[0]
    na = len(LL.LEE_AGL)
    Z, U, V, n, zs = (P['Z'][sel], P['U'][sel], P['V'][sel], P['n'][sel], P['zs'][sel])
    ub0, vb0 = (x[:, 0] for x in LL._wind_at(Z, U, V, n, np.array([[LL.UB_LEVEL]])))
    Ub = LL.hypot(ub0, vb0)
    bands = np.zeros((M, nband * na))
    go = Ub > 0.5
    if not go.any():
        return (ub0, vb0, bands)
    gi = np.nonzero(go)[0]
    Z, U, V, n, zs, Ubg = (_c(Z[gi]), _c(U[gi]), _c(V[gi]), _c(n[gi], np.int64), zs[gi], Ub[gi])
    dx, dy = (_c(ub0[gi] / Ubg), _c(vb0[gi] / Ubg))
    pz, pn2, cnt = LL._stability(P, sel[gi])
    pz, pn2, cnt = (_c(pz), _c(pn2), _c(cnt, np.int64))
    agl = np.array(LL.LEE_AGL)
    at = _c(zs[:, None] + agl[None, :] - zs[:, None])
    grow = _c(np.exp(at / (2 * LL.H_SCALE)))
    f = _c(Ubg * LL.EFF / math.pi)
    ks = _c(ks)
    k2 = _c(ks * ks)
    ak = LL.ALPHA / np.maximum(ks, 2 * math.pi / LL.LAM_CAP)
    ui_low, ui_high = (_c(-ak), _c(-ak * LL.ALPHA_HIGH))
    T = _solver_tables()
    nl, nk, L, S = (LL.MIDS.shape[0], ks.shape[0], Z.shape[1], pz.shape[1])
    edges = _c(LL.EDGES)
    m = min(NODES_A_PASS, gi.size)
    mr, mi = (np.empty((m, nl, nk)) for _ in range(2))
    ar, ai, ca, sa, ch, sh = (np.empty((m, nl + na, nk)) for _ in range(6))
    layer = np.empty((m, na), np.int32)
    out = np.empty((gi.size, nband * na))
    for c0 in range(0, gi.size, NODES_A_PASS):
        c1 = min(gi.size, c0 + NODES_A_PASS)
        m = c1 - c0
        columns(m, L, _p(Z[c0:c1]), _p(U[c0:c1]), _p(V[c0:c1]), _p(n[c0:c1]), _p(dx[c0:c1]), _p(dy[c0:c1]), S, _p(pz[c0:c1]), _p(pn2[c0:c1]), _p(cnt[c0:c1]), T['hs'].shape[0], _p(T['hs']), nl, _p(T['idx']), _p(edges), _p(T['high']), _p(T['consts']), nk, _p(k2), _p(ui_low), _p(ui_high), na, _p(at[c0:c1]), _p(layer), _p(mr), _p(mi), _p(ar), _p(ai))
        np.cos(ar[:m], out=ca[:m])
        np.sin(ar[:m], out=sa[:m])
        np.cosh(ai[:m], out=ch[:m])
        np.sinh(ai[:m], out=sh[:m])
        bands_of(m, nl, _p(edges), nk, na, nband, LL.LEE_BAND_K, _p(at[c0:c1]), _p(layer), _p(mr), _p(mi), _p(ca), _p(sa), _p(ch), _p(sh), _p(grow[c0:c1]), _p(f[c0:c1]), _p(out[c0:c1]))
    bands[gi] = out
    return (ub0, vb0, bands)

def lattice(fields, grid, edges, chunk=LL.NODES_A_CHUNK):
    return LL.lattice(fields, grid, edges, chunk, solve=_solve, sampler=_Sampler)

def lattice_at(L, lat, lon):
    f = _fn('cx_lattice_at')
    if f is None:
        return LL.lattice_at(L, lat, lon)
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    shape = np.broadcast_shapes(lat.shape, lon.shape)
    lat, lon = (_c(np.broadcast_to(lat, shape).ravel()), _c(np.broadcast_to(lon, shape).ravel()))
    okL, uL, vL, B = (_c(L['ok'], np.bool_), _c(L['u']), _c(L['v']), _c(L['bands']))
    n, nb = (lat.shape[0], B.shape[-1])
    u, v, bands, ok = (np.empty(n), np.empty(n), np.empty((n, nb)), np.empty(n, np.uint8))
    f(n, _p(lat), _p(lon), LL.LEE_LAT, int(L['i0']), int(L['j0']), int(L['ni']), int(L['nj']), _p(okL), _p(uL), _p(vL), _p(B), nb, _p(u), _p(v), _p(bands), _p(ok))
    return (u.reshape(shape), v.reshape(shape), bands.reshape(shape + (nb,)), ok.view(np.bool_).reshape(shape))

def lattice_cells(L, grid):
    return LL.lattice_cells(L, grid, at=lattice_at)

def _need(a, cy, cx, r):
    T = a.T
    offs = sorted(set(range(-r, r + 1, T)) | {r})
    w = np.flatnonzero((cx + r >= 0) & (cx - r < a.nx) & (cy + r >= 0) & (cy - r < a.ny))
    cy, cx = (cy[w], cx[w])
    for oy in offs:
        ty = np.clip(cy + oy, 0, a.ny - 1) // T
        for ox in offs:
            tx = np.clip(cx + ox, 0, a.nx - 1) // T
            need = a.listed[ty, tx] & ~a.loaded[ty, tx]
            if need.any():
                for t in np.unique(ty[need] * a.ntx + tx[need]):
                    a._load(int(t % a.ntx), int(t // a.ntx))

def _area_args(a):
    return (a.ny, a.nx, a.T, a.ntx, _p(a.listed), _p(a.loaded), _p(a.mosaic), a.P)

def _not_loaded(a):
    return RuntimeError(f"exposure {a.name}: a cell in a tile that isn't loaded")

def _lee(lee, lat, lon, step, lu, lv, bands, lok, cols=False):
    cells, best = (_fn('cx_lee_cells'), _fn('cx_lee_best'))
    NA = RW.NA
    if cells is None or best is None or lee is None or (not 1 <= lee.nband <= MAX_BANDS) or (NA > MAX_BANDS) or (lee.dirs < 2):
        return RW._lee(lee, lat, lon, step, lu, lv, bands, lok, cols)
    n = lat.size
    top = np.full(n, np.nan)
    peak = np.zeros(n)
    kbest = np.zeros(n, np.int64)
    bcol = np.zeros((n, NA)) if cols else None
    aid, fx, fy = lee.locate(lat, lon)
    D, NB = (lee.dirs, lee.nband)
    dec = _c((np.arange(256) / 254.0) ** 2 * lee.scale)
    consts = np.array([RW.LEE_MAX_W, RW.LEE_EX_C, RW.LEE_BOOST], np.float64)
    for k in np.unique(aid[aid >= 0]):
        a = lee.areas[k]
        sel = np.flatnonzero(aid == k)
        r = 0 if step <= a.step + 1e-09 else max(1, RW._round1((step / a.step - 1) / 2))
        wv = lok[sel] if lok is not None else np.zeros(sel.size, bool)
        d0 = d1 = np.zeros(sel.size, np.int64)
        t = np.zeros(sel.size)
        B = None
        if wv.any():
            u, v = (np.where(wv, lu[sel], 0.0), np.where(wv, lv[sel], 0.0))
            U = np.hypot(u, v)
            d0, d1, t = RW._toward(u, v, D)
            d0, d1, t = (_c(np.where(U > 0, d0, 0), np.int64), _c(np.where(U > 0, d1, 1), np.int64), _c(np.where(U > 0, t, 0.0)))
            B = _c(bands)
        cy, cx = (RW.js_round(fy[sel]).astype(np.int64), RW.js_round(fx[sel]).astype(np.int64))
        _need(a, cy, cx, r)
        tsum = np.zeros(sel.size)
        tn = np.zeros(sel.size, np.int64)
        bp = np.zeros(sel.size)
        bk = np.zeros(sel.size, np.int64)
        bc = np.zeros((sel.size, NA)) if cols else None
        cap = max(PAIRS_A_PASS, (2 * r + 1) ** 2)
        pair = np.empty(cap, np.int64)
        S = np.empty((cap, NA))
        count = ctypes.c_int64(0)
        wv, area = (_c(wv, np.bool_), _area_args(a))
        i = 0
        while i < sel.size:
            i = cells(sel.size, i, _p(cy), _p(cx), r, *area, D, NB, NA, _p(dec), lee.top_step, _p(wv), _p(d0), _p(d1), _p(t), _p(B), _p(sel), _p(tsum), _p(tn), cap, _p(pair), _p(S), ctypes.addressof(count))
            if i < 0:
                raise _not_loaded(a)
            m = count.value
            if m:
                pw = _c(S[:m] ** (RW.LEE_EX_GAMMA / 2))
                best(m, _p(pair), _p(pw), NA, _p(consts), _p(bp), _p(bk), _p(bc))
        got = tn > 0
        top[sel[got]] = tsum[got] / tn[got]
        peak[sel], kbest[sel] = (bp, bk)
        if cols:
            bcol[sel] = bc
    return (top, peak, kbest, aid >= 0) + ((bcol,) if cols else ())

def _ridge_lift(ridge, lat, lon, u, v, nlow, step):
    cells = _fn('cx_ridge_cells')
    if cells is None or ridge is None or ridge.relief_step is None:
        return RW._ridge_lift(ridge, lat, lon, u, v, nlow, step)
    out = np.full(lat.size, np.nan)
    ok = ~np.isnan(u) & ~np.isnan(v)
    aid, fx, fy = ridge.locate(lat, lon)
    aid = np.where(ok, aid, -1)
    U = np.hypot(u, v)
    out[aid >= 0] = 0.0
    live = (aid >= 0) & (U > 0.5)
    D = ridge.dirs
    fk = ridge.scale / 254 * RW.RIDGE_XC_SCALE
    for k in np.unique(aid[live]):
        a = ridge.areas[k]
        sel = np.flatnonzero(live & (aid == k))
        r = 0 if step <= 2 * a.step + 1e-09 else max(1, RW._round1((step / a.step - 1) / 2))
        d0, d1, t = RW._toward(u[sel], v[sel], D)
        cy, cx = (RW.js_round(fy[sel]).astype(np.int64), RW.js_round(fx[sel]).astype(np.int64))
        _need(a, cy, cx, r)
        total = np.zeros(sel.size)
        Us, n, d0, d1, t = (_c(U[sel]), _c(nlow[sel]), _c(d0, np.int64), _c(d1, np.int64), _c(t))
        if cells(sel.size, _p(cy), _p(cx), r, *_area_args(a), D, fk, ridge.relief_step, _p(Us), _p(n), _p(d0), _p(d1), _p(t), _p(total)) != 0:
            raise _not_loaded(a)
        out[sel] = total / (2 * r + 1) ** 2
    return out

def run_fields(fields, grid, lee, ridge, lat_bands=None, chunk=RW.CELLS_A_CHUNK, bars=False):
    return RW.run_fields(fields, grid, lee, ridge, lat_bands, chunk, bars, kernels=(_lee, _ridge_lift))
