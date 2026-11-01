from __future__ import annotations
import math
import numpy as np
AGL_WIND_M = (100, 300, 600, 1000, 1500, 2000, 3000)
WIND_ALTS_FT = (2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 18000, 24000, 30000)
ML_DZ = (200, 500, 1000, 1800, 3000)
STAB_M = (750, 1250, 1750, 2500, 3500, 5000)
TEMP_FT = (8000, 10000, 12000, 14000, 18000, 24000)
LEE_LAT = 0.5
LEE_AGL = (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 7500.0, 9000.0)
LEE_BAND_K = 4
NO_THERMALS = 0.25
DZ_LOW, Z_MID, DZ_HIGH, Z_TOP = (100, 4000, 400, 14000)
ALPHA = 1 / 1000
LAM_CAP = 15000.0
ALPHA_Z, ALPHA_HIGH = (3000.0, 0.25)
H_SCALE = 8000.0
UB_LEVEL = 1000.0
N2_MIN, N2_MAX = (1e-06, 0.0012)
N2_FREE, N2_STRAT = (0.0001, 0.0004)
TROPO = 11000.0
L2_CAP = (2 * math.pi / 1500) ** 2
U_SMOOTH = 500.0
MIN_U = 0.5
EFF = 0.69
G, KAPPA = (9.81, 0.2857)
NODES_A_CHUNK = 2048

def hypot(a, b):
    aa, ab = (np.abs(a), np.abs(b))
    m = np.maximum(aa, ab)
    r = np.minimum(aa, ab, out=aa)
    with np.errstate(invalid='ignore', divide='ignore'):
        r /= m
    r *= r
    r += 1
    np.sqrt(r, out=r)
    r *= m
    odd = ~np.isfinite(r)
    if odd.any():
        r[odd & (m == 0)] = 0.0
        r[np.isinf(m)] = np.inf
    return r

class _Sampler:

    def __init__(self, grid, lat, lon):
        west, north, step = (float(grid['west']), float(grid['north']), float(grid['step']))
        nx, ny = (int(grid['nx']), int(grid['ny']))
        fx = (lon - west) / step
        fy = (north - lat) / step
        self.inb = ~((fx < -0.5) | (fy < -0.5) | (fx > nx - 0.5) | (fy > ny - 0.5))
        fxs = np.where(self.inb, fx, 0.0)
        fys = np.where(self.inb, fy, 0.0)
        x0 = np.maximum(0, np.minimum(nx - 2, np.floor(fxs))).astype(np.int64)
        y0 = np.maximum(0, np.minimum(ny - 2, np.floor(fys))).astype(np.int64)
        self.tx = tx = np.maximum(0.0, np.minimum(1.0, fxs - x0))
        self.ty = ty = np.maximum(0.0, np.minimum(1.0, fys - y0))
        self.idx = (y0, x0)
        self.w = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)

    def __call__(self, a):
        if a is None:
            return None
        y0, x0 = self.idx
        vals = (a[y0, x0], a[y0, x0 + 1], a[y0 + 1, x0], a[y0 + 1, x0 + 1])
        tx, ty = (self.tx, self.ty)
        v00, v10, v01, v11 = vals
        have = [np.isfinite(v) for v in vals]
        full = have[0] & have[1] & have[2] & have[3]
        with np.errstate(invalid='ignore'):
            bil = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty
            sw = np.zeros_like(tx)
            sv = np.zeros_like(tx)
            for w, v, h in zip(self.w, vals, have):
                sw = np.where(h, sw + w, sw)
                sv = np.where(h, sv + w * np.where(h, v, 0.0), sv)
            part = np.where(sw >= 0.5, sv / np.where(sw > 0, sw, 1.0), np.nan)
        return np.where(self.inb, np.where(full, bil, part), np.nan)

def _wind_at(Z, U, V, n, h):
    N, L = Z.shape
    h = np.broadcast_to(h, (N, h.shape[1]))
    rows = np.arange(N)[:, None]
    last = np.maximum(n - 1, 0)[:, None]
    zl = Z[rows, last]
    cnt = np.zeros(h.shape, np.int64)
    for j in range(1, L):
        cnt += Z[:, j:j + 1] < h
    k = np.minimum(np.maximum(1 + cnt, 1), L - 1)
    za, zb = (Z[rows, k - 1], Z[rows, k])
    with np.errstate(invalid='ignore', over='ignore'):
        f = (h - za) / np.maximum(zb - za, 0.001)
        out = []
        for A in (U, V):
            a0, ak = (A[rows, k - 1], A[rows, k])
            mid = a0 + (ak - a0) * f
            first = (n[:, None] == 1) | (h <= Z[:, :1])
            out.append(np.where(first, A[:, :1], np.where(h >= zl, A[rows, last], mid)))
    return out

def _compact(z, valid, *vals):
    order = np.argsort(~valid, axis=1, kind='stable')
    zc = np.take_along_axis(np.where(valid, z, np.inf), order, 1)
    return (zc,) + tuple((np.take_along_axis(np.where(valid, v, 0.0), order, 1) for v in vals))

def _profiles(fields, lat, lon, grid):
    s = _Sampler(grid, lat, lon)
    N = lat.shape[0]
    F = lambda k: fields.get(k)
    zs = s(F('elevation'))
    if zs is None:
        zs = np.full(N, np.nan)
    agl = [(10.0, 'sfc')] + [(float(m), f'g{m}') for m in AGL_WIND_M]
    msl = [(ft * 0.3048, f'a{ft}') for ft in WIND_ALTS_FT]
    zl, ul, vl, ok = ([], [], [], [])
    top = np.zeros(N)
    for z, k in agl:
        if F(f'u_{k}') is None or F(f'v_{k}') is None:
            continue
        u, v = (s(F(f'u_{k}')), s(F(f'v_{k}')))
        good = np.isfinite(u) & np.isfinite(v)
        top = np.where(good, z, top)
        (zl.append(np.full(N, z)), ul.append(u), vl.append(v), ok.append(good))
    for z, k in msl:
        if F(f'u_{k}') is None or F(f'v_{k}') is None:
            continue
        u, v = (s(F(f'u_{k}')), s(F(f'v_{k}')))
        za = z - zs
        with np.errstate(invalid='ignore'):
            good = np.isfinite(u) & np.isfinite(v) & (za > top)
        (zl.append(za), ul.append(u), vl.append(v), ok.append(good))
    have = np.isfinite(zs)
    if zl:
        valid = np.stack(ok, 1) & have[:, None]
        Z, U, V = _compact(np.stack(zl, 1), valid, np.stack(ul, 1), np.stack(vl, 1))
        Z = np.concatenate([Z, np.full((N, 1), np.inf)], 1)
        U = np.concatenate([U, np.zeros((N, 1))], 1)
        V = np.concatenate([V, np.zeros((N, 1))], 1)
        n = valid.sum(1)
    else:
        Z, U, V = (np.full((N, 2), np.inf), np.zeros((N, 2)), np.zeros((N, 2)))
        n = np.zeros(N, np.int64)
    nan = np.full(N, np.nan)
    g = lambda k: s(F(k)) if F(k) is not None else nan
    dth = [s(F(f'dth_m{dz}')) if F(f'dth_m{dz}') is not None else None for dz in ML_DZ]
    thml = g('thml')
    if any((d is None for d in dth)):
        dth = None
    else:
        dth = np.stack(dth, 1)
    nb = [(m, s(F(f'n_g{m}'))) for m in STAB_M if F(f'n_g{m}') is not None]
    t = [(ft, s(F(f't{ft}'))) for ft in TEMP_FT if F(f't{ft}') is not None]
    return dict(zs=zs, Z=Z, U=U, V=V, n=n, ws=g('wstar'), zi=g('zi'), thml=thml, nlow=g('nlow'), dth=dth, nb=nb, t=t, ok=have & (n > 0))

def _std_theta_factor(z):
    p = 1013.25 * math.pow(max(0.05, 1 - 2.25577e-05 * z), 5.25588)
    return math.pow(1000 / p, KAPPA)

def _stability(P, sel):
    zs, ws, zi, thml, nlow = (P[k][sel] for k in ('zs', 'ws', 'zi', 'thml', 'nlow'))
    M = zs.shape[0]
    pz, pn2, pv = ([], [], [])
    last = np.zeros(M)
    anyp = np.zeros(M, bool)

    def push(z, n2, valid):
        nonlocal last, anyp
        z = np.broadcast_to(np.asarray(z, np.float64), (M,))
        n2 = np.broadcast_to(np.asarray(n2, np.float64), (M,))
        valid = np.broadcast_to(valid, (M,))
        (pz.append(z), pn2.append(n2), pv.append(valid))
        last = np.where(valid, z, last)
        anyp = anyp | valid
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        day = (ws >= NO_THERMALS) & (zi > 0)
        night = ~day & np.isfinite(nlow)
        nn = nlow * nlow
        push(np.where(day, 0.0, 0.0), np.where(day, N2_MIN, nn), day | night)
        push(np.where(day, zi * 0.9, 500.0), np.where(day, N2_MIN, nn), day | night)
        if P['dth'] is not None:
            dth = P['dth'][sel]
            dok = day & np.all(np.isfinite(dth), 1) & np.isfinite(thml)
            z0, d0 = (zi, np.zeros(M))
            for i, dz in enumerate(ML_DZ):
                z1, d1 = (zi + dz, dth[:, i])
                push((z0 + z1) / 2, G / thml * ((d1 - d0) / (z1 - z0)), dok)
                z0, d0 = (z1, d1)
        top = np.where(anyp, last, -np.inf)
        for m, nv in P['nb']:
            b = nv[sel]
            push(float(m), b * b, (m > top + 200) & np.isfinite(b))
        lo = np.where(anyp, last, -np.inf)
        if P['t']:
            tz = np.array([ft * 0.3048 for ft, _ in P['t']])
            fac = np.array([_std_theta_factor(z) for z in tz])
            tv = np.stack([v[sel] for _, v in P['t']], 1)
            present = np.isfinite(tv)
            tzc, tvc, facc = _compact(np.broadcast_to(tz, tv.shape), present, tv, np.broadcast_to(fac, tv.shape))
            cnt = present.sum(1)
            for p in range(tv.shape[1] - 1):
                az, bz = (tzc[:, p], tzc[:, p + 1])
                tha = (tvc[:, p] + 273.15) * facc[:, p]
                thb = (tvc[:, p + 1] + 273.15) * facc[:, p + 1]
                zm = (az + bz) / 2 - zs
                ok = (p + 1 < cnt) & (zm > lo + 200)
                push(np.where(ok, zm, 0.0), np.where(ok, G / ((tha + thb) / 2) * ((thb - tha) / (bz - az)), 0.0), ok)
        need = ~anyp | (last < TROPO - zs - 2000)
        lastz = np.where(anyp, last, 0.0)
        push(np.maximum(np.maximum(lo + 1500, lastz + 1500), 3000.0), N2_FREE, need)
        push(TROPO - zs, N2_FREE, np.ones(M, bool))
        push(TROPO - zs + 500, N2_STRAT, np.ones(M, bool))
    z = np.stack(pz, 1)
    v = np.stack(pv, 1)
    key = np.where(v, z, np.inf)
    order = np.argsort(key, axis=1, kind='stable')
    return (np.take_along_axis(key, order, 1), np.take_along_axis(np.where(v, np.stack(pn2, 1), 0.0), order, 1), v.sum(1))

def _interp(pz, pn2, cnt, z):
    M, S = pz.shape
    rows = np.arange(M)
    c = np.zeros(M, np.int64)
    for j in range(1, S):
        c += pz[:, j] < z
    i = 1 + c
    inside = i < cnt
    ii = np.minimum(i, S - 1)
    az, bz = (pz[rows, ii - 1], pz[rows, ii])
    an, bn = (pn2[rows, ii - 1], pn2[rows, ii])
    with np.errstate(invalid='ignore', over='ignore'):
        t = (z - az) / np.maximum(1e-06, bz - az)
        mid = an + t * (bn - an)
    lastn = pn2[rows, cnt - 1]
    return np.where(z <= pz[:, 0], pn2[:, 0], np.where(inside, mid, lastn))

def _column_edges():
    e, z = ([], 0)
    while z < Z_TOP:
        e.append(z)
        z += DZ_LOW if z < Z_MID else DZ_HIGH
    e.append(Z_TOP)
    return np.array(e, np.float64)
EDGES = _column_edges()
MIDS = (EDGES[:-1] + EDGES[1:]) / 2

def band_wavenumbers(edges):
    ks = []
    for b in range(len(edges) - 1):
        hi = 2 * math.pi / edges[b]
        lo = 2 * math.pi / (edges[b + 1] if edges[b + 1] is not None else edges[b] * 6)
        for i in range(LEE_BAND_K):
            ks.append(lo * (hi / lo) ** ((i + 0.5) / LEE_BAND_K))
    return np.array(ks, np.float64)

def _step(mr, mi, D, sr, si, dr, di):
    ar, ai = (mr * D, mi * D)
    ch, sh, ca, sa = (np.cosh(ai), np.sinh(ai), np.cos(ar), np.sin(ar))
    cr, ci, snr, sni = (ca * ch, -sa * sh, sa * ch, ca * sh)
    mm = mr * mr + mi * mi
    small = mm * D * D < 1e-12
    with np.errstate(invalid='ignore', divide='ignore'):
        sdr = np.where(small, D, (snr * mr + sni * mi) / mm)
        sdi = np.where(small, 0.0, (sni * mr - snr * mi) / mm)
    msr, msi = (mr * snr - mi * sni, mr * sni + mi * snr)
    o1r = cr * sr - ci * si - (sdr * dr - sdi * di)
    o1i = cr * si + ci * sr - (sdr * di + sdi * dr)
    o2r = msr * sr - msi * si + (cr * dr - ci * di)
    o2i = msr * si + msi * sr + (cr * di + ci * dr)
    return (o1r, o1i, o2r, o2i)

def _transfer(n2, U, Upp, at, ks):
    M, nl = n2.shape
    nk, na = (ks.shape[0], at.shape[1])
    k2 = (ks * ks)[None, :]
    ak = (ALPHA / np.maximum(ks, 2 * math.pi / LAM_CAP))[None, :]
    Ul = np.where(np.abs(U) < MIN_U, np.where(U < 0, -MIN_U, MIN_U), U)

    def m_of(j):
        ur = Ul[:, j:j + 1]
        ui = -ak * ALPHA_HIGH if MIDS[j] > ALPHA_Z else -ak
        d2r, d2i = (ur * ur - ui * ui, 2 * ur * ui)
        dd, uu = (d2r * d2r + d2i * d2i, ur * ur + ui * ui)
        nj, pj = (n2[:, j:j + 1], Upp[:, j:j + 1])
        lr = nj * d2r / dd - pj * ur / uu
        li = -nj * d2i / dd + pj * ui / uu
        mag = hypot(lr, li)
        cap = mag > L2_CAP
        with np.errstate(invalid='ignore', divide='ignore'):
            f = np.where(cap, L2_CAP / mag, 1.0)
        lr, li = (np.where(cap, lr * f, lr), np.where(cap, li * f, li))
        re, im = (lr - k2, li)
        r = hypot(re, im)
        a = np.sqrt(np.maximum(0.0, (r + re) / 2))
        b = np.sqrt(np.maximum(0.0, (r - re) / 2))
        return (a, np.where(im < 0, -b, b))
    rec_ok = (at >= 0) & (at <= EDGES[nl])
    layer = np.minimum(np.searchsorted(EDGES, at, side='right') - 1, nl - 1)
    layer = np.where(rec_ok, layer, -1)
    recr = np.zeros((M, nk, na))
    reci = np.zeros((M, nk, na))
    done = np.zeros((M, na), bool)
    mr, mi = m_of(nl - 1)
    flip = (mi < 0) | (mi == 0) & (mr < 0)
    tr, ti = (np.where(flip, -mr, mr), np.where(flip, -mi, mi))
    wr, wi = (np.ones((M, nk)), np.zeros((M, nk)))
    pr, pi = (-ti, tr.copy())
    for j in range(nl - 1, -1, -1):
        if j != nl - 1:
            mr, mi = m_of(j)
        zt = EDGES[j + 1]
        for a in range(na):
            sel = np.nonzero(layer[:, a] == j)[0]
            if not sel.size:
                continue
            D = (zt - at[sel, a])[:, None]
            o1r, o1i, _, _ = _step(mr[sel], mi[sel], D, wr[sel], wi[sel], pr[sel], pi[sel])
            recr[sel, :, a], reci[sel, :, a] = (o1r, o1i)
            done[sel, a] = True
        wr, wi, pr, pi = _step(mr, mi, zt - EDGES[j], wr, wi, pr, pi)
        q = wr * wr + wi * wi
        near = ~((q >= 1e-99) & (q <= 1e+99))
        if near.any():
            s = np.where(near, hypot(wr, wi), 1.0)
            big = near & ((s > 1e+50) | (s < 1e-50) & (s > 0))
            if big.any():
                sc = np.where(big, s, 1.0)
                wr, wi, pr, pi = (wr / sc, wi / sc, pr / sc, pi / sc)
                scd = np.where(big[:, :, None] & done[:, None, :], sc[:, :, None], 1.0)
                recr, reci = (recr / scd, reci / scd)
    d = wr * wr + wi * wi
    d = np.where((d == 0) | np.isnan(d), 1e-300, d)[:, :, None]
    br, bi = (wr[:, :, None], wi[:, :, None])
    grow = np.exp(at / (2 * H_SCALE))[:, None, :]
    Tr = (recr * br + reci * bi) / d * grow
    Ti = (reci * br - recr * bi) / d * grow
    return np.where(done[:, None, :], hypot(Tr, Ti), 0.0)

def _solve(P, sel, ks, nband):
    M = sel.shape[0]
    na = len(LEE_AGL)
    Z, U, V, n, zs = (P['Z'][sel], P['U'][sel], P['V'][sel], P['n'][sel], P['zs'][sel])
    ub0, vb0 = (x[:, 0] for x in _wind_at(Z, U, V, n, np.array([[UB_LEVEL]])))
    Ub = hypot(ub0, vb0)
    bands = np.zeros((M, nband * na))
    go = Ub > 0.5
    if not go.any():
        return (ub0, vb0, bands)
    gi = np.nonzero(go)[0]
    Z, U, V, n, zs, Ubg = (Z[gi], U[gi], V[gi], n[gi], zs[gi], Ub[gi])
    dx, dy = ((ub0[gi] / Ubg)[:, None], (vb0[gi] / Ubg)[:, None])
    h = U_SMOOTH
    um_at = sorted({float(z) for m in MIDS for z in ((m, m + h, m - h) if m - h >= 0 else (m,))})
    hs = sorted({max(0.0, z + o) for z in um_at for o in (-h / 2, 0.0, h / 2)})
    col = {z: i for i, z in enumerate(hs)}
    u, v = _wind_at(Z, U, V, n, np.array([hs]))
    along = u * dx + v * dy

    def um(z):
        return (along[:, col[max(0.0, z - h / 2)]] + along[:, col[max(0.0, z)]] * 2 + along[:, col[max(0.0, z + h / 2)]]) / 4
    nl = MIDS.shape[0]
    Ucol = np.empty((gi.size, nl))
    Upp = np.zeros((gi.size, nl))
    for j, z in enumerate(MIDS):
        z = float(z)
        Ucol[:, j] = um(z) - 0
        if not z - h < 0:
            Upp[:, j] = (um(z + h) - 2 * um(z) + um(z - h)) / (h * h)
    sub = sel[gi]
    pz, pn2, cnt = _stability(P, sub)
    n2 = np.empty((gi.size, nl))
    for j, z in enumerate(MIDS):
        n2[:, j] = np.maximum(N2_MIN, np.minimum(N2_MAX, _interp(pz, pn2, cnt, float(z))))
    agl = np.array(LEE_AGL)
    at = zs[:, None] + agl[None, :] - zs[:, None]
    T = _transfer(n2, Ucol, Upp, at, ks)
    f = Ubg * EFF / math.pi
    out = np.zeros((gi.size, nband, na))
    for b in range(nband):
        sm = np.zeros((gi.size, na))
        for i in range(LEE_BAND_K):
            sm = sm + T[:, b * LEE_BAND_K + i, :]
        out[:, b, :] = sm / LEE_BAND_K * f[:, None]
    out = np.where((at > 0)[:, None, :], out, 0.0)
    bands[gi] = out.reshape(gi.size, nband * na)
    return (ub0, vb0, bands)

def node_range(grid):
    west, north, step = (float(grid['west']), float(grid['north']), float(grid['step']))
    east = west + (int(grid['nx']) - 1) * step
    south = north - (int(grid['ny']) - 1) * step
    i0, i1 = (math.floor(west / LEE_LAT), math.floor(east / LEE_LAT) + 1)
    j0, j1 = (math.floor(south / LEE_LAT), math.floor(north / LEE_LAT) + 1)
    return (i0, j0, i1 - i0 + 1, j1 - j0 + 1)

class _F64:

    def __init__(self, fields):
        self.fields, self.done = (fields, {})

    def get(self, k, default=None):
        if k not in self.done:
            a = self.fields.get(k)
            if a is not None:
                a = np.asarray(a)
                if a.dtype.kind != 'f':
                    a = a.astype(np.float64)
            self.done[k] = a
        a = self.done[k]
        return default if a is None else a

def lattice(fields, grid, edges, chunk=NODES_A_CHUNK):
    i0, j0, ni, nj = node_range(grid)
    jj, ii = np.meshgrid(np.arange(j0, j0 + nj), np.arange(i0, i0 + ni), indexing='ij')
    lat = (jj * LEE_LAT).astype(np.float64).ravel()
    lon = (ii * LEE_LAT).astype(np.float64).ravel()
    fields = _F64(fields)
    P = _profiles(fields, lat, lon, grid)
    ks = band_wavenumbers(list(edges))
    nband = len(edges) - 1
    N = lat.shape[0]
    u, v = (np.full(N, np.nan), np.full(N, np.nan))
    bands = np.zeros((N, nband * len(LEE_AGL)))
    idx = np.nonzero(P['ok'])[0]
    for c0 in range(0, idx.size, chunk):
        sel = idx[c0:c0 + chunk]
        u[sel], v[sel], bands[sel] = _solve(P, sel, ks, nband)
    return dict(i0=i0, j0=j0, ni=ni, nj=nj, u=u.reshape(nj, ni), v=v.reshape(nj, ni), bands=bands.reshape(nj, ni, -1), ok=P['ok'].reshape(nj, ni))

def lattice_at(L, lat, lon):
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    shape = np.broadcast_shapes(lat.shape, lon.shape)
    lat, lon = (np.broadcast_to(lat, shape).ravel(), np.broadcast_to(lon, shape).ravel())
    fi, fj = (lon / LEE_LAT, lat / LEE_LAT)
    i0, j0 = (np.floor(fi), np.floor(fj))
    tx, ty = (fi - i0, fj - j0)
    i0, j0 = (i0.astype(np.int64) - L['i0'], j0.astype(np.int64) - L['j0'])
    ni, nj = (L['ni'], L['nj'])
    nb = L['bands'].shape[-1]
    okL = np.append(L['ok'].ravel(), False)
    uL = np.append(np.where(L['ok'], L['u'], 0.0).ravel(), 0.0)
    vL = np.append(np.where(L['ok'], L['v'], 0.0).ravel(), 0.0)
    B = np.concatenate([np.where(L['ok'][..., None], L['bands'], 0.0).reshape(-1, nb), np.zeros((1, nb))])
    Bv = np.ascontiguousarray(B).view(np.dtype((np.void, 8 * nb))).ravel()
    none = okL.size - 1
    ws = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)
    rows = []
    W = np.zeros(lat.shape)
    for q, (di, dj) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        i, j = (i0 + di, j0 + dj)
        inside = (i >= 0) & (i < ni) & (j >= 0) & (j < nj)
        r = np.where(inside, j * ni + i, none)
        r = np.where(okL[r], r, none)
        rows.append(r)
        W = np.where(r != none, W + ws[q], W)
    ok = W > 0
    Ws = np.where(ok, W, 1.0)
    u, v = (np.zeros(lat.shape), np.zeros(lat.shape))
    wq = []
    for q, r in enumerate(rows):
        w = np.where(ok & (r != none), ws[q] / Ws, 0.0)
        u += uL[r] * w
        v += vL[r] * w
        wq.append(w)
    n = lat.shape[0]
    bands = np.empty((n, nb))
    blk = 4096
    acc = np.empty((blk, nb))
    tmpv = np.empty(blk, Bv.dtype)
    tmp = tmpv.view(np.float64).reshape(blk, nb)
    for b0 in range(0, n, blk):
        b1 = min(n, b0 + blk)
        m = b1 - b0
        a = acc[:m]
        a.fill(0.0)
        for r, w in zip(rows, wq):
            np.take(Bv, r[b0:b1], out=tmpv[:m])
            t = tmp[:m]
            t *= w[b0:b1, None]
            a += t
        bands[b0:b1] = a
    u, v = (np.where(ok, u, np.nan), np.where(ok, v, np.nan))
    return (u.reshape(shape), v.reshape(shape), bands.reshape(shape + (nb,)), ok.reshape(shape))

def lattice_cells(L, grid):
    west, north, step, nx = (float(grid['west']), float(grid['north']), float(grid['step']), int(grid['nx']))
    xs = np.arange(nx)

    def rows(y0, y1):
        iy = np.arange(y0, y1)
        lat = (north - iy * step)[:, None]
        lon = (west + xs * step)[None, :]
        return lattice_at(L, lat, lon)
    return rows
