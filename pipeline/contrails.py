from __future__ import annotations
import os
import numpy as np
TRAFFIC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contrail_traffic.npz')
LEVELS = (300, 250, 200)
ETA = np.array([0.3, 0.34, 0.37], np.float32)
W0 = np.array([120.0, 170.0, 260.0], np.float32)
DEPTH = np.array([200.0, 280.0, 380.0], np.float32)
R_EFF = np.array([1.1, 1.0, 0.9], np.float32)
EI_H2O = 1.25
Q_FUEL = 43200000.0
CP = 1004.0
EPS = 0.622
RV = 461.5
RHO_ICE = 917.0
RHI_PERSIST = (93.0, 100.0)
ISSR_SHARE = 0.5
S_FLOOR = 0.05
S_MAX = 0.4
LIFE_H = (1.0, 30.0, 7.0)
SHEAR_MAX = 0.015
DIFFUSE = 0.1
SPREAD_W = 5000.0
OPT_DEPTH = 500.0
R_YOUNG, R_OLD = (1e-05, 2e-05)
VIS_TAU = 0.05
G_ASYM = 0.77
COVER_SCALE = 0.8
SHORT_H = 5 / 60
ISSR_FIX = {'hrrr': (0.614, 0.82), 'gfs': (0.613, 1.92), 'ecmwf': (0.623, 1.58), 'icon': (0.595, 1.84), 'icond2': (0.595, 1.84), 'arome': (0.623, 1.58)}
ISSR_ETA = 0.34
STEP_DEG = 0.25
KIND_NONE, KIND_SHORT, KIND_NARROW, KIND_SPREAD = (0, 1, 2, 3)
_traffic = None

def traffic():
    global _traffic
    if _traffic is None:
        if not os.path.exists(TRAFFIC_FILE):
            _traffic = False
        else:
            z = np.load(TRAFFIC_FILE)
            q = z['q']
            lo, step = (float(z['lo']), float(z['step']))
            dens = np.where(q > 0, 10.0 ** (lo + (q.astype(np.float32) - 1) * step), 0).astype(np.float32)
            _traffic = (dens, z['weekday'].astype(np.float32))
    return _traffic or None

def es_water(t):
    return 611.2 * np.exp(17.67 * t / (t + 243.5))

def es_ice(t):
    return 612.11 * np.exp(22.587 * t / (t + 273.86))

def sac_saturated(p_hpa, eta):
    G = EI_H2O * CP * p_hpa * 100 / (EPS * Q_FUEL * (1 - eta))
    lg = np.log(np.maximum(G - 0.053, 0.001))
    return (-46.46 + 9.43 * lg + 0.72 * lg ** 2, G)

def sac_threshold(p_hpa, rhw, eta):
    t_lm, G = sac_saturated(p_hpa, eta)
    rhw = np.asarray(rhw, np.float64)
    lo, hi = (np.full(rhw.shape, t_lm - 25.0), np.full(rhw.shape, t_lm))
    e_lm = es_water(t_lm)
    for _ in range(24):
        mid = (lo + hi) / 2
        f = e_lm - G * (t_lm - mid) - rhw * es_water(mid)
        lo, hi = (np.where(f > 0, lo, mid), np.where(f > 0, mid, hi))
    return np.where(rhw >= 0.999, t_lm, (lo + hi) / 2)

def forms_at(T, rhw, p_hpa, eta):
    grid, tab = _sac_table(float(p_hpa), float(eta))
    out = T < tab[0]
    mid = np.nonzero((T >= tab[0]) & (T < tab[-1]))[0]
    if len(mid):
        out[mid] = T[mid] < np.interp(rhw[mid], grid, tab)
    return out
_SAC = {}

def _sac_table(p_hpa, eta):
    key = (p_hpa, eta)
    if key not in _SAC:
        grid = np.linspace(0, 1, 201)
        _SAC[key] = (grid, sac_threshold(p_hpa, grid, eta))
    return _SAC[key]

def reflectance(tau, mu):
    t = (1 - G_ASYM) * tau
    return t / (2 * np.maximum(mu, 0.05) + t)

def _bilinear(dens, lat, lon):
    y = np.clip(lat + 90 - 0.5, 0, 179)
    x = (lon + 180 - 0.5) % 360
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy, fx = (y - y0, x - x0)
    y1 = np.minimum(y0 + 1, 179)
    x1 = (x0 + 1) % 360
    return (1 - fy) * ((1 - fx) * dens[y0, x0] + fx * dens[y0, x1]) + fy * ((1 - fx) * dens[y1, x0] + fx * dens[y1, x1])

def contrail_cirrus(t, rhi, u, v, shear, lat, lon, valid):
    tr = traffic()
    if tr is None:
        return None
    dens, weekday = tr
    hour = valid.hour
    wd = float(weekday[valid.weekday()])
    shape = lat.shape
    lat, lon = (np.asarray(lat, np.float64).ravel(), np.asarray(lon, np.float64).ravel())
    n = lat.size
    clear = np.ones(n, np.float32)
    tau_sum = np.zeros(n, np.float32)
    cov_sum = np.zeros(n, np.float32)
    kinds = np.zeros((len(LEVELS), n), np.int8)
    S_all = np.clip(np.nan_to_num(np.asarray(shear, np.float32).ravel(), nan=0.0), 0, SHEAR_MAX)
    for li, p in enumerate(LEVELS):
        T = np.nan_to_num(np.asarray(t[li], np.float32).ravel(), nan=0.0)
        RHi = np.nan_to_num(np.asarray(rhi[li], np.float32).ravel(), nan=0.0)
        cold = np.nonzero(T < sac_saturated(p, float(ETA.max()))[0])[0]
        if not len(cold):
            continue
        T, RHi, la, lo, S = (T[cold], RHi[cold], lat[cold], lon[cold], S_all[cold])
        uu = np.nan_to_num(np.asarray(u[li], np.float32).ravel()[cold], nan=0.0)
        vv = np.nan_to_num(np.asarray(v[li], np.float32).ravel()[cold], nan=0.0)
        rhw = np.clip(RHi / 100 * es_ice(T) / es_water(T), 0, 1)
        persist = np.clip((RHi - RHI_PERSIST[0]) / (RHI_PERSIST[1] - RHI_PERSIST[0]), 0, 1)
        s = np.clip(np.maximum(RHi - 100, 0) / 100 + S_FLOOR, 0, S_MAX)
        life = np.minimum(LIFE_H[0] + LIFE_H[1] * s, LIFE_H[2])
        excess = s * es_ice(T) / (RV * (T + 273.15))
        km_lon = 111.32 * np.maximum(np.cos(np.radians(la)), 0.05)
        m = len(cold)
        lin = np.zeros(m, np.float32)
        tau_w = np.zeros(m, np.float32)
        formed = np.zeros(m, bool)
        wide = np.zeros(m, bool)
        for c in range(3):
            forms = forms_at(T, rhw, p, ETA[c])
            formed |= forms
            rate = DIFFUSE + S * DEPTH[c]
            wide |= forms & (persist > 0.5) & (W0[c] + rate * life * 3600 > SPREAD_W)
            sh = np.nonzero(forms & (persist < 1))[0]
            if len(sh):
                f0 = _bilinear(dens[hour, li, c], la[sh], lo[sh]) * wd
                lin[sh] += f0 * (W0[c] / 1000) * SHORT_H * (1 - persist[sh])
            k = np.nonzero(forms & (persist > 0))[0]
            if not len(k):
                continue
            for a in range(int(np.ceil(LIFE_H[2]))):
                age = a + 0.5
                alive = np.clip(life[k] - a, 0, 1)
                live = alive > 0
                if not np.any(live):
                    break
                k, alive = (k[live], alive[live])
                width = (W0[c] + rate[k] * age * 3600) / 1000
                r = (R_YOUNG + (R_OLD - R_YOUNG) * min(age / 2, 1)) * R_EFF[c]
                tau = 3 * excess[k] * min(DEPTH[c] + 250 * age, OPT_DEPTH) / (2 * RHO_ICE * r)
                seen = np.clip(tau / VIS_TAU, 0, 1)
                d_km = age * 3.6
                f = _bilinear(dens[(hour - a) % 24, li, c], la[k] - vv[k] * d_km / 111.32, lo[k] - uu[k] * d_km / km_lon[k]) * wd
                add = f * width * alive * seen * persist[k] * ISSR_SHARE
                lin[k] += add
                tau_w[k] += add * tau
        cov = 1 - np.exp(-COVER_SCALE * lin)
        tau_l = np.where(lin > 0, tau_w / np.maximum(lin, 1e-09), 0)
        clear[cold] *= 1 - cov
        cov_sum[cold] += cov
        tau_sum[cold] += cov * tau_l
        kinds[li, cold] = np.where(~formed, KIND_NONE, np.where(persist < 0.5, KIND_SHORT, np.where(wide, KIND_SPREAD, KIND_NARROW)))
    cover = 1 - clear
    tau = np.where(cov_sum > 0, tau_sum / np.maximum(cov_sum, 1e-09), 0)
    return (cover.reshape(shape).astype(np.float32), tau.reshape(shape).astype(np.float32), kinds.reshape((len(LEVELS),) + shape))

def issr_mask(t, rhi):
    m = np.zeros(np.shape(t)[1:], bool)
    for li, p in enumerate(LEVELS):
        with np.errstate(invalid='ignore'):
            m |= (np.asarray(t[li]) < sac_saturated(p, ISSR_ETA)[0]) & (np.asarray(rhi[li]) >= 100)
    return m

def fold_issr(s, issr, cover, tau, mu, fix):
    hcc = s.get('hcc')
    if hcc is None:
        return
    N, g = fix
    c = np.where(issr, 1 - np.exp(-g * np.clip(cover, 0, 1)), 0).astype(np.float32)
    s['contrail'] = (100 * (1 - N) * c).astype(np.float32)
    on = issr & np.isfinite(hcc)
    s['hcc'] = np.where(on, np.minimum(hcc, 100 * (N + (1 - N) * c)), hcc).astype(np.float32)
    if s.get('tcc') is not None:
        bands = [np.clip(np.nan_to_num(s.get(k), nan=0) if s.get(k) is not None else np.zeros_like(hcc), 0, 100) / 100 for k in ('lcc', 'mcc')]
        total = 100 * (1 - (1 - bands[0]) * (1 - bands[1]) * (1 - s['hcc'] / 100))
        s['tcc'] = np.where(on & np.isfinite(s['tcc']), total, s['tcc']).astype(np.float32)
    keep = (1 - (1 - N) * c * reflectance(tau, mu)).astype(np.float32)
    for k in ('sw', 'rnsw'):
        if s.get(k) is not None:
            s[k] = (s[k] * keep).astype(np.float32)
    if s.get('shf') is not None:
        s['shf'] = np.where(s['shf'] > 0, s['shf'] * keep, s['shf']).astype(np.float32)

def at_grid(lat2d, lon2d, fields, valid, mu=None):
    ny, nx = lat2d.shape
    step = max(float(np.nanmedian(np.abs(np.diff(lat2d[:, nx // 2])))) if ny > 1 else STEP_DEG, 0.001)
    k = max(1, int(round(STEP_DEG / step)))
    sub = (slice(k // 2, None, k), slice(k // 2, None, k))
    t, rhi, u, v, shear = fields
    r = contrail_cirrus(t[(slice(None),) + sub], rhi[(slice(None),) + sub], u[(slice(None),) + sub], v[(slice(None),) + sub], shear[sub], lat2d[sub], lon2d[sub], valid)
    if r is None:
        return None
    cover, tau, _ = r
    if k == 1:
        return (cover, tau)
    from scipy.ndimage import map_coordinates
    yy = (np.arange(ny, dtype=np.float32) - k // 2) / k
    xx = (np.arange(nx, dtype=np.float32) - k // 2) / k
    Y, X = np.meshgrid(yy, xx, indexing='ij')

    def full(a):
        return map_coordinates(a, [Y, X], order=1, mode='nearest').astype(np.float32)
    return (full(cover), full(tau))
