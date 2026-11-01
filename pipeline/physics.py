from __future__ import annotations
import math
import numpy as np
G = 9.80665
RD = 287.05
CP = 1004.0
LV = 2500000.0
EPS = 0.622
KAPPA = RD / CP
DRY_LAPSE = G / CP
K0 = 273.15
FT = 0.3048

def es(t):
    return 6.112 * np.exp(17.67 * t / (t + 243.5))

def mixing_ratio(td, p):
    e = es(td)
    return EPS * e / np.maximum(p - e, 1.0)

def dewpoint_from_w(w, p):
    e = w * p / (EPS + w)
    ln = np.log(np.maximum(e, 1e-06) / 6.112)
    return 243.5 * ln / (17.67 - ln)

def dewpoint_from_rh(t, rh):
    e = es(t) * np.clip(rh, 1, 100) / 100
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln)

def theta(t, p):
    return (t + K0) * (1000.0 / p) ** KAPPA
_CELLS: dict[int, np.ndarray] = {}

def levels_below(z, zq):
    n = np.zeros(z.shape[1:], np.uint8 if z.shape[0] < 256 else np.intp)
    for zk in z:
        n += zk < zq
    return n

def level_positions(idx):
    n = idx.size
    cells = _CELLS.get(n)
    if cells is None:
        cells = _CELLS[n] = np.arange(n)
    return idx.astype(np.intp).ravel() * n + cells

def take_level(a, pos, shape):
    return np.take(np.ascontiguousarray(a).reshape(-1), pos).reshape(shape)

def take_along_levels(a, idx):
    return take_level(a, level_positions(idx), np.shape(idx))

def sort_levels(a, order):
    n = order[0].size
    cells = _CELLS.get(n)
    if cells is None:
        cells = _CELLS[n] = np.arange(n)
    pos = order.astype(np.intp).reshape(order.shape[0], n) * n + cells
    return np.take(np.ascontiguousarray(a).reshape(-1), pos).reshape(order.shape)

def moist_lapse(t, p):
    tk = t + K0
    rs = mixing_ratio(t, p)
    return G * (1 + LV * rs / (RD * tk)) / (CP + LV * LV * rs * EPS / (RD * tk * tk))

class Profile:

    def __init__(self, zs, ps, t2, td2, u10, v10, z, p, t, td, u, v, uh=None, vh=None, hub=100.0):
        L = z.shape[0]
        order = np.argsort(-np.asarray(p))
        z, t, td, u, v = (z[order], t[order], td[order], u[order], v[order])
        p = np.asarray(p, dtype=np.float32)[order]
        under = z < (zs + 10)[None]
        k = np.arange(L, dtype=np.float32)[:, None, None]
        z = np.where(under, zs[None] + 2 + 0.01 * (k + 1), z)
        pl = np.broadcast_to(p[:, None, None], z.shape).astype(np.float32)
        pl = np.where(under, ps[None] * (1 - 1e-05 * (k + 1)), pl)
        fold = lambda arr, sfc: np.where(under, sfc[None], arr)
        self.z = np.concatenate([(zs + 2)[None], z]).astype(np.float32)
        self.lnp = np.log(np.concatenate([ps[None], pl])).astype(np.float32)
        self.t = np.concatenate([t2[None], fold(t, t2)]).astype(np.float32)
        self.td = np.concatenate([np.minimum(td2, t2)[None], np.minimum(fold(td, td2), fold(t, t2))]).astype(np.float32)
        self.zs = zs
        self.ztop = self.z[-1]
        zw = np.where(under, zs[None] + 10 + 0.01 * (k + 1), z)
        zw = np.concatenate([(zs + 10)[None], zw])
        uw = np.concatenate([u10[None], fold(u, u10)])
        vw = np.concatenate([v10[None], fold(v, v10)])
        if uh is not None and vh is not None:
            ok = np.isfinite(uh) & np.isfinite(vh)
            self.zw, self.u, self.v = (zw.astype(np.float32), uw.astype(np.float32), vw.astype(np.float32))
            zh = (zs + hub).astype(np.float32)
            u0, v0 = self.at(zh, 'u', 'v')
            zw = np.concatenate([zw, zh[None]])
            uw = np.concatenate([uw, np.where(ok, uh, u0)[None]])
            vw = np.concatenate([vw, np.where(ok, vh, v0)[None]])
            order = np.argsort(zw, axis=0, kind='stable')
            zw, uw, vw = (sort_levels(a, order) for a in (zw, uw, vw))
        self.zw = zw.astype(np.float32)
        self.u = uw.astype(np.float32)
        self.v = vw.astype(np.float32)

    def _weights(self, zq, z):
        zq = np.broadcast_to(zq, z.shape[1:])
        hi = level_positions(np.clip(levels_below(z, zq), 1, z.shape[0] - 1))
        lo = hi - zq.size
        z0, z1 = (take_level(z, lo, zq.shape), take_level(z, hi, zq.shape))
        w = np.clip((zq - z0) / np.maximum(z1 - z0, 0.001), 0, 1)
        return ((lo, hi, zq.shape), w)

    def at(self, zq, *names):
        cols = {}
        out = []
        for n in names:
            wind = n in ('u', 'v')
            if wind not in cols:
                cols[wind] = self._weights(zq, self.zw if wind else self.z)
            (lo, hi, shape), w = cols[wind]
            a = getattr(self, n)
            a0, a1 = (take_level(a, lo, shape), take_level(a, hi, shape))
            val = a0 + w * (a1 - a0)
            out.append(np.exp(val) if n == 'lnp' else val)
        return out if len(out) > 1 else out[0]

def parcel_ml_depth(prof: Profile, excess):
    ps = np.exp(prof.lnp[0])
    th_p = theta(prof.t[0], ps) + excess
    depth = np.full(prof.zs.shape, np.nan, np.float32)
    prev_z = prof.z[0]
    prev_d = np.broadcast_to(excess, prof.zs.shape).astype(np.float32)
    for k in range(1, prof.z.shape[0]):
        d = th_p - theta(prof.t[k], np.exp(prof.lnp[k]))
        hit = np.isnan(depth) & (d <= 0)
        f = prev_d / np.maximum(prev_d - d, 1e-06)
        depth = np.where(hit, prev_z + f * (prof.z[k] - prev_z) - prof.zs, depth)
        prev_z, prev_d = (prof.z[k], d)
    return np.where(np.isnan(depth), prof.ztop - prof.zs, np.maximum(depth, 0))

def layer_mean(prof: Profile, zb, zt, names, n=8):
    acc = [np.zeros_like(prof.zs) for _ in names]
    for i in range(n):
        zq = zb + (zt - zb) * (i / (n - 1))
        vals = prof.at(zq, *names)
        vals = vals if isinstance(vals, list) else [vals]
        for a, v in zip(acc, vals):
            a += v
    return [a / n for a in acc]

def bumpiness(prof, ws, zi, gust, u10, v10, ustar=None, relief=None):
    zs = prof.zs
    v10s = np.hypot(u10, v10)

    def layer(z0, z1):
        (u0, v0), (u1, v1) = (prof.at(zs + z0, 'u', 'v'), prof.at(zs + z1, 'u', 'v'))
        (t0, p0), (t1, p1) = (prof.at(zs + z0, 't', 'lnp'), prof.at(zs + z1, 't', 'lnp'))
        th0, th1 = (theta(t0, p0), theta(t1, p1))
        dz = z1 - z0
        shear = np.hypot(u1 - u0, v1 - v0) / dz
        ri = G / (0.5 * (th0 + th1)) * (th1 - th0) / dz / np.maximum(shear, 0.001) ** 2
        return (shear, ri)
    s_low, ri_low = layer(10, 300)
    s_up, ri_up = layer(300, 1000)
    unstable = lambda ri: np.clip((0.25 - ri) / 0.15, 0, 1)
    sh_low, sh_up = (unstable(ri_low) * s_low * 50, unstable(ri_up) * s_up * 50)
    if ustar is None or np.all(np.isnan(ustar)):
        ustar = 0.087 * v10s * np.clip(1 - ri_low / 0.5, 0.3, 1)
    rough = 1 + np.clip(np.nan_to_num(relief, nan=0) / 1000, 0, 0.5) if relief is not None else 1
    sig_mech = 1.25 * np.nan_to_num(ustar, nan=0) * rough
    sig_conv = 0.6 * np.nan_to_num(ws, nan=0)
    sig_gust = 0.5 * np.maximum(0, np.nan_to_num(gust, nan=0) - v10s) / 3 if gust is not None else 0
    depth = np.clip(np.nan_to_num(zi, nan=0) / 1000, 0.1, 1)
    sigma = np.sqrt((sig_conv ** 2 + sig_mech ** 2 + sig_gust ** 2) * depth + 0.29 * sh_low ** 2 + 0.7 * sh_up ** 2)
    return np.clip((sigma - 1.0) / 0.8, 0, 1)
SINK_BAND = (300.0, 1500.0, 3000.0)
SINK_K = 0.6
SINK_RAIN = (0.5, 3.0)
SINK_LAPSE = (6.0, 9.0)
SNOW_SINK = 2.0
SINK_LEE = 0.55
SINK_CAUSES = ('none', 'between thermals', 'air descending (model or lee slope)', 'precipitation', 'cold ground (snow or ice)', 'mountain–plain circulation (by day air drawn to the mountains sinks over the foreland; by night over the mountains as their cold air drains out)')
SINK_GLACIER = 0.8
SNOW_COVER_M = (0.02, 0.1)

def column_at(z, a, zq):
    hi = level_positions(np.clip(levels_below(z, zq), 1, z.shape[0] - 1))
    lo = hi - zq.size
    z0, z1 = (take_level(z, lo, zq.shape), take_level(z, hi, zq.shape))
    a0, a1 = (take_level(a, lo, zq.shape), take_level(a, hi, zq.shape))
    f = np.clip((zq - z0) / np.maximum(z1 - z0, 0.001), 0, 1)
    return a0 + f * (a1 - a0)

def sigma_w(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 1, np.sqrt(1.8) * np.cbrt(x) * np.clip(1 - 0.8 * x, 0, None), 0)

def snow_rate(r, snow):
    return r if snow is None else r * (1 + (SNOW_SINK - 1) * np.clip(np.nan_to_num(snow, nan=0), 0, 1))

def rain_from_dbz(dbz):
    return (10 ** (np.nan_to_num(dbz, nan=0) / 10) / 200) ** (1 / 1.6)

def snow_share(ptype, prof: Profile, depth):
    lo, top_min, top_max = SINK_BAND
    top = np.clip(np.nan_to_num(depth, nan=0), top_min, top_max)
    cold = prof.at(prof.zs + (lo + top) / 2, 't') <= 0
    p = np.round(np.nan_to_num(ptype, nan=0))
    return np.select([p == 2, (p == 3) | (p == 4), p == 0], [1.0, 0.5, np.where(cold, 1.0, 0.0)], 0.0).astype(np.float32)

def sink_rate(prof: Profile, wz, w, ws, zi, depth, rain, refd=None, lee=None, n=10, parts=False, snow=None):
    zs = prof.zs
    lo, top_min, top_max = SINK_BAND
    top = np.clip(np.nan_to_num(depth, nan=0), top_min, top_max)
    wmean = np.zeros_like(zs)
    drysum = np.zeros_like(zs)
    if w is not None:
        L = wz.shape[0]
        under = wz < (zs + 10)[None]
        k = np.arange(L, dtype=np.float32)[:, None, None]
        zc = np.concatenate([zs[None], np.where(under, zs[None] + 0.01 * (k + 1), wz)])
        wc = np.concatenate([np.zeros_like(zs)[None], np.where(under, 0, np.nan_to_num(w, nan=0))])
    for i in range(n):
        zq = zs + lo + (top - lo) * (i / (n - 1))
        if w is not None:
            wmean += column_at(zc, wc, zq) / n
        t, td = prof.at(zq, 't', 'td')
        drysum += np.maximum(t - td, 0) / n
    if lee is not None:
        wmean = np.where(wmean > 0.25, wmean, np.minimum(wmean, -np.maximum(np.nan_to_num(lee, nan=0), 0)))
    ws = np.nan_to_num(ws, nan=0)
    zi = np.maximum(np.nan_to_num(zi, nan=0), 1)
    d = np.nan_to_num(depth, nan=0)
    between = np.zeros_like(zs)
    for i in range(n):
        h = lo + (np.maximum(d, lo) - lo) * (i / (n - 1))
        between += sigma_w(h / zi) / n
    between = np.where((ws > 0) & (d > lo), SINK_K * ws * between, 0)
    r = np.maximum(np.nan_to_num(rain, nan=0), 0)
    if refd is not None:
        r = np.maximum(r, np.where(np.nan_to_num(refd, nan=0) >= 15, np.minimum(rain_from_dbz(refd), 50), 0))
    r = snow_rate(r, snow)
    lapse = (prof.at(zs + lo, 't') - prof.at(zs + top, 't')) / (top - lo) * 1000
    convective = np.clip((lapse - SINK_LAPSE[0]) / (SINK_LAPSE[1] - SINK_LAPSE[0]), 0, 1)
    peak = SINK_RAIN[0] + (SINK_RAIN[1] - SINK_RAIN[0]) * convective
    precip = peak * (1 - np.exp(-r / 3)) * np.clip(0.1 + drysum / 12, 0.1, 1)
    rain_dd = precip - np.maximum(wmean, 0)
    extra = np.maximum(0, np.maximum(-wmean, rain_dd - between))
    if not parts:
        return extra
    total = np.maximum(0, np.maximum(between - wmean, rain_dd))
    th = np.minimum(between, total)
    why = np.where(extra >= 0.05, np.where(rain_dd - between >= -wmean, 3, 2), np.where(th >= 0.05, 1, 0))
    return (extra, th, why.astype(np.float32))

def snow_cover(snowd):
    lo, hi = SNOW_COVER_M
    return np.clip((np.nan_to_num(snowd, nan=0) - lo) / (hi - lo), 0, 1)
SOIL_WET_ICON = (0.1, 0.3)

def soil_wetness(mstav=None, soilw=None, wilt=None, fldcp=None, soilm=None):
    if mstav is not None:
        return np.clip(np.asarray(mstav, np.float32) / 100.0, 0, 1)
    if soilw is not None:
        lo = np.asarray(wilt, np.float32) if wilt is not None else SOIL_WET_ICON[0]
        hi = np.asarray(fldcp, np.float32) if fldcp is not None else SOIL_WET_ICON[1]
        span = np.maximum(hi - lo, 0.02)
        return np.clip((np.asarray(soilw, np.float32) - lo) / span, 0, 1)
    if soilm is not None:
        lo, hi = SOIL_WET_ICON
        return np.clip((np.asarray(soilm, np.float32) - lo) / (hi - lo), 0, 1)
    return None

def glacier_sink(snowd, t2, slope, day):
    warm = np.clip(np.nan_to_num(t2, nan=0) / 8, 0, 1)
    steep = 0.3 + 0.7 * np.clip(np.nan_to_num(slope, nan=0) / 0.15, 0, 1)
    return (SINK_GLACIER * snow_cover(snowd) * warm * steep * (np.nan_to_num(day, nan=0) > 0)).astype(np.float32)
PUMP_SINK, PUMP_LIFT = (0.2, 0.3)
PUMP_REACH = (25.0, 100.0)
PUMP_HEAT = 150.0
PUMP_CALM = (6.0, 14.0)
PUMP_RESOLVED_KM = 25.0
PUMP_NIGHT = 0.3
PUMP_COOL_WM2 = 40.0
PUMP_WIND, PUMP_DEPTH = ((2.0, 1.5), (1000.0, 300.0))
PUMP_WIND_KM = (8.0, 25.0)
PUMP_TOP_S, PUMP_TOP_CAP = (7200.0, 0.3)
LEE_REL, LEE_RUN_M, LEE_TOP_CAP, HOLD_CAP = (0.3, 4000.0, 0.2, 0.35)
LEE_GATE = 2.0
SINK_W_CAP = 0.25
VM_CLIMB, VM_DOWN_CAP, VM_UP_CAP = (0.5, 0.2, 0.15)
_PUMP_GEO = {}

def lee_gate(speed, ws):
    ws = np.maximum(np.nan_to_num(np.asarray(ws, np.float32), nan=0), 0.3)
    return np.clip(np.maximum(np.asarray(speed, np.float32), 1.0) / (LEE_GATE * ws) - 1, 0, 1).astype(np.float32)

def lee_hold(down, speed, ws, zi, ramp=1.0):
    gate = lee_gate(speed, ws)
    speed = np.maximum(np.asarray(speed, np.float32), 1.0)
    held = LEE_REL * gate * ramp * np.maximum(np.nan_to_num(down, nan=0), 0) / speed * LEE_RUN_M
    return np.minimum(held, LEE_TOP_CAP * np.maximum(np.nan_to_num(zi, nan=0), 0)).astype(np.float32)

def vertical_motion(conv, sink, why):
    c = np.nan_to_num(np.asarray(conv, np.float32), nan=0)
    sk = np.nan_to_num(np.asarray(sink, np.float32), nan=0)
    k = np.rint(np.nan_to_num(why, nan=0))
    wd = np.where(((k == 2) | (k == 5)) & (sk > 0), -sk, 0).astype(np.float32)
    return np.where(wd == 0, c, np.where(c >= 0, c + wd, np.minimum(c, wd))).astype(np.float32)

def air_motion_climb(ws, vm):
    ws = np.nan_to_num(np.asarray(ws, np.float32), nan=0)
    d = np.clip(VM_CLIMB * np.nan_to_num(vm, nan=0), -VM_DOWN_CAP * ws, VM_UP_CAP * ws)
    return np.where(ws > 0, ws + d, ws).astype(np.float32)

def pump_unresolved(grid_step):
    return float(np.clip(grid_step * 111.0 / PUMP_RESOLVED_KM, 0.25, 1))

def pump_wind_share(grid_step):
    lo, hi = PUMP_WIND_KM
    return float(np.clip((grid_step * 111.0 - lo) / (hi - lo), 0, 1))

def night_cooling(shf):
    return np.clip(-np.nan_to_num(np.asarray(shf, np.float32), nan=0) / PUMP_COOL_WM2, 0, 1).astype(np.float32)

def mountain_plain(zs, heat, u, v, grid_step, cool=None, coslat=None):
    from scipy import ndimage
    zs = np.nan_to_num(np.asarray(zs, np.float32), nan=0)
    km = grid_step * 111.0
    cells = lambda r: max(1, int(round(r / km)))
    box = lambda a, r: ndimage.uniform_filter(a, size=2 * cells(r) + 1, mode='nearest')
    top = lambda a, r: ndimage.maximum_filter(a, size=2 * cells(r) + 1, mode='nearest')
    key = (zs.shape, grid_step, float(zs.sum(dtype=np.float64)), float(zs[::7, ::5].sum(dtype=np.float64)), None if coslat is None else float(np.mean(coslat)))
    geo = _PUMP_GEO.get(key)
    if geo is None:
        massif = box(zs, 10.0)
        near, far = (top(massif, PUMP_REACH[0]) - zs, top(massif, PUMP_REACH[1]) - zs)
        prox = 0.6 * np.clip((near - 300) / 1200, 0, 1) + 0.4 * np.clip((far - 300) / 1200, 0, 1)
        relief = top(zs, 5.0) + top(-zs, 5.0)
        above_low = zs + top(-zs, 5.0)
        above_plain = zs + top(-zs, PUMP_REACH[1])
        foreland = np.clip((500 - above_low) / 300, 0, 1) * np.clip((1000 - above_plain) / 500, 0, 1)
        ridge = np.clip((zs - massif - 50) / 250, 0, 1) * np.clip((relief - 300) / 500, 0, 1)
        mtn = np.clip((zs + top(-zs, 2 * PUMP_REACH[1]) - 500) / 1000, 0, 1)
        reg = box(zs, 40.0)
        gx = np.gradient(reg, axis=1) / (km * (1.0 if coslat is None else np.maximum(coslat, 0.1)))
        gy = -np.gradient(reg, axis=0) / km
        g = np.hypot(gx, gy)
        up = np.where(g > 0.001, 1 / np.maximum(g, 1e-09), 0)
        geo = _PUMP_GEO[key] = (prox * foreland, ridge, mtn, gx * up, gy * up)
        while len(_PUMP_GEO) > 8:
            _PUMP_GEO.pop(next(iter(_PUMP_GEO)))
    fore, ridge, mtn, ex, ey = geo
    drive = np.clip(box(np.nan_to_num(heat, nan=0), 40.0) / PUMP_HEAT, 0, 1)
    night = np.zeros_like(zs) if cool is None else np.clip(box(np.nan_to_num(cool, nan=0), 40.0), 0, 1)
    calm = np.clip((PUMP_CALM[1] - np.hypot(np.nan_to_num(u, nan=0), np.nan_to_num(v, nan=0))) / (PUMP_CALM[1] - PUMP_CALM[0]), 0, 1)
    sink = PUMP_SINK * calm * (fore * drive + PUMP_NIGHT * mtn * night)
    lift = PUMP_LIFT * calm * (ridge * drive + PUMP_NIGHT * fore * night) * pump_unresolved(grid_step)
    speed = fore * calm * (PUMP_WIND[0] * drive - PUMP_WIND[1] * night) * pump_wind_share(grid_step)
    f32 = lambda a: np.asarray(a, np.float32)
    return {'sink': f32(sink), 'lift': f32(lift), 'u': f32(speed * ex), 'v': f32(speed * ey), 'depth': f32(np.where(drive >= night, PUMP_DEPTH[0], PUMP_DEPTH[1]))}

def wstar(heat, zi, t, p):
    rho = p * 100 / (RD * (t + K0))
    x = G / (t + K0) * np.maximum(heat, 0) / (rho * CP) * np.maximum(zi, 0)
    return np.cbrt(x)
CD_BL = 0.003
BS_MAX = 20.0

def bs_ratio(ws, u_bl):
    return np.where(ws > 0, np.minimum(BS_MAX, ws / (math.sqrt(CD_BL) * np.maximum(u_bl, 0.5))), 0)

def shear_tilt(ws, shear):
    return np.minimum(99, np.where(ws > 0, ws ** 3 / (0.0128 * np.maximum(shear, 0.5) ** 3), 0))

def density_altitude(p, t):
    pa_ft = (1 - (p / 1013.25) ** 0.190284) * 145366.45
    return (pa_ft + 118.8 * (t - (15 - 1.98 * pa_ft / 1000))) * FT
LV_CP = LV / CP

def cloud_mix(tp, te, tde, p, mix):
    hp = tp + LV_CP * mixing_ratio(tp, p)
    he = te + LV_CP * mixing_ratio(tde, p)
    h = hp + mix * (he - hp)
    t = tp
    for _ in range(2):
        e = es(t)
        d = np.maximum(p - e, 1.0)
        rs = EPS * e / d
        u = t + 243.5
        t = t - (t + LV_CP * rs - h) / (1 + LV_CP * rs * p / d * 4302.645 / (u * u))
    return t
CU_STEP = 50.0

def cumulus_top(prof: Profile, zbase, tbase, active, step=CU_STEP, max_above=10000.0, mix=None):
    top = np.where(active, np.nan, zbase).astype(np.float32)
    tp = (tbase + 0.5).astype(np.float32)
    z = zbase.astype(np.float32).copy()
    for _ in range(int(max_above / step)):
        live = np.isnan(top)
        if not live.any():
            break
        if mix is None:
            te, p = prof.at(z, 't', 'lnp')
        else:
            te, p, tde = prof.at(z, 't', 'lnp', 'td')
        stop = live & ((tp < te) | (z >= prof.ztop))
        top = np.where(stop, z, top)
        tp = tp - moist_lapse(tp, p) * step
        if mix is not None:
            tp = cloud_mix(tp, te, tde, p, mix)
        z = z + step
    return np.where(np.isnan(top), z, top)
ENTRAIN_MIN, ENTRAIN_MAX = (0.0001, 0.001)

def cumulus_mix(twidth, step=CU_STEP):
    r = np.where(np.isfinite(twidth), np.asarray(twidth, np.float32), 1000.0)
    return (np.clip(0.2 / np.maximum(r, 1.0), ENTRAIN_MIN, ENTRAIN_MAX) * step).astype(np.float32)

def wave_environment(prof: Profile):
    zs = prof.zs
    dz = 500.0
    low_b = zs + 500
    low_t = np.minimum(zs + 3000, prof.ztop - 2500)
    up_t = np.minimum(low_t + 4500, prof.ztop - dz)
    ok = (low_t - low_b >= 800) & (up_t - low_t >= 1500)
    ul, vl = layer_mean(prof, low_b, low_t, ['u', 'v'], 6)
    uu, vu = layer_mean(prof, low_t, up_t, ['u', 'v'], 6)
    sl = np.hypot(ul, vl)
    su = np.hypot(uu, vu)
    ex, ey = (ul / np.maximum(sl, 0.001), vl / np.maximum(sl, 0.001))
    cosang = (ul * uu + vl * vu) / np.maximum(sl * su, 1e-06)
    turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    def sample(zq):
        t, p, u, v = prof.at(zq, 't', 'lnp', 'u', 'v')
        return (np.maximum(u * ex + v * ey, 1.5), theta(t, p))

    def scorer(zb, zt, n=5):
        l2 = np.zeros_like(zs)
        n2 = np.zeros_like(zs)
        for i in range(n):
            zc = zb + (zt - zb) * (i / (n - 1))
            ua, ta = sample(zc - dz)
            ub, tb = sample(zc)
            uc, tc = sample(zc + dz)
            N2 = G / tb * (tc - ta) / (2 * dz)
            uzz = (uc - 2 * ub + ua) / (dz * dz)
            l2 += N2 / (ub * ub) - uzz / ub
            n2 += N2
        return (l2 / n, n2 / n)
    l2l, n2l = scorer(low_b, low_t)
    l2u, _ = scorer(low_t, up_t)
    ratio = np.where(l2u > 1e-09, l2l / np.maximum(l2u, 1e-09), np.where(l2l > 0, 10, 0))
    nlow = np.sqrt(np.maximum(n2l, 0))
    c01 = lambda x: np.clip(x, 0, 1)
    sw = np.where(su > sl, (sl + su) / 2, sl)
    score = c01((sw - 5) / 7.5) * c01(1 - (turn - 20) / 40) * np.where(su >= sl, 1, c01(su / np.maximum(sl, 0.001))) * c01((nlow - 0.005) / 0.005) * (0.4 + 0.6 * c01((ratio - 1) / 3))
    kmean = np.sqrt(np.maximum((l2l + np.maximum(l2u, 0)) / 2, 0))
    wavelength = np.where((l2l > 0) & (kmean > 0), 2 * np.pi / np.maximum(kmean, 1e-09), np.nan)
    return (np.where(ok, score, np.nan), np.where(ok, np.minimum(wavelength, 60000), np.nan))
WAVE_CARRY_KM = (10.0, 20.0, 35.0, 50.0, 65.0, 80.0)
CARRY_SHARE = 0.8

def carry_wavelength(score, wavelen, u, v, step_deg, lat2d, reach_km=WAVE_CARRY_KM):
    sc = np.nan_to_num(score, nan=0.0)
    lam = np.array(wavelen, dtype=np.float32, copy=True)
    if lat2d is None or not step_deg:
        return lam
    ny, nx = sc.shape
    U = np.hypot(u, v)
    ok = np.isfinite(U) & (U > 0.5)
    ex, ey = (np.where(ok, u / np.maximum(U, 1e-06), 0.0), np.where(ok, v / np.maximum(U, 1e-06), 0.0))
    km_x = step_deg * 111.2 * np.cos(np.radians(lat2d)) + 1e-06
    km_y = step_deg * 111.2
    iy0, ix0 = np.indices((ny, nx))
    for d in reach_km:
        ix = np.clip(np.rint(ix0 - ex * d / km_x).astype(int), 0, nx - 1)
        iy = np.clip(np.rint(iy0 + ey * d / km_y).astype(int), 0, ny - 1)
        s_up, l_up = (sc[iy, ix], wavelen[iy, ix])
        take = ok & (s_up >= CARRY_SHARE * sc) & (s_up > 0) & np.isfinite(l_up) & ~(lam >= l_up)
        lam = np.where(take, l_up, lam)
    return np.where(sc > 0, lam, wavelen).astype(np.float32)

def local_relief(elev, radius=2):
    pad = np.pad(elev, radius, mode='edge')
    hi = np.full(elev.shape, -np.inf, np.float32)
    lo = np.full(elev.shape, np.inf, np.float32)
    ny, nx = elev.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            win = pad[dy:dy + ny, dx:dx + nx]
            hi = np.maximum(hi, win)
            lo = np.minimum(lo, win)
    return hi - lo
SM = 1609.344
CEIL_NONE = 6000.0

def rh_from_td(t, td):
    return np.clip(100 * es(np.minimum(td, t)) / es(t), 0, 100)

def ceiling_agl(ceil_msl, zs, inside):
    has = np.isfinite(ceil_msl) & (ceil_msl < 15000)
    agl = np.clip(np.nan_to_num(ceil_msl, nan=0) - zs, 0, CEIL_NONE)
    return np.where(inside, np.where(has, agl, CEIL_NONE), np.nan)
RH_CLOUD = 92.0

def ceiling_from_humidity(prof: Profile, tcc=None, zmax_agl=5000.0):
    rh = rh_from_td(prof.t, prof.td)
    agl = prof.z - prof.zs[None]
    base = np.where(rh[0] >= 97, 30.0, np.nan)
    for k in range(1, rh.shape[0]):
        hit = np.isnan(base) & (rh[k] >= RH_CLOUD) & (agl[k] <= zmax_agl)
        f = np.clip((RH_CLOUD - rh[k - 1]) / np.maximum(rh[k] - rh[k - 1], 0.001), 0, 1)
        base = np.where(hit, agl[k - 1] + f * (agl[k] - agl[k - 1]), base)
    broken = np.ones_like(prof.zs, dtype=bool) if tcc is None else np.nan_to_num(tcc, nan=100) >= 50
    return np.where(np.isfinite(base) & broken, np.maximum(base, 30), CEIL_NONE)

def cloud_top_from_humidity(prof: Profile, tcc=None):
    rh = rh_from_td(prof.t, prof.td)
    top = np.full(prof.zs.shape, np.nan, np.float32)
    for k in range(1, rh.shape[0]):
        top = np.where(rh[k] >= 90, prof.z[k], top)
    some = np.ones_like(prof.zs, dtype=bool) if tcc is None else np.nan_to_num(tcc, nan=0) >= 30
    return np.where(some, top, np.nan)
CC_LEVELS = (1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200)
T_ICE = -23.0
CC_RHC = (0.9, 0.8)
CC_SIGMA_TOP = 0.6
CC_CLW = (0.003, 0.3)

def es_ice(t):
    return 6.1121 * np.exp(22.587 * t / (t + 273.86))

def rh_cloud(t, td):
    a = np.clip((t - T_ICE) / -T_ICE, 0, 1) ** 2
    return 100 * es(np.minimum(td, t)) / (a * es(t) + (1 - a) * es_ice(t))

def cloud_fraction(rh, p, ps, clw=None):
    rhs, rht = CC_RHC
    sigma = np.asarray(p, np.float32) / ps
    rhc = rht + (rhs - rht) * np.clip((sigma - CC_SIGMA_TOP) / (1 - CC_SIGMA_TOP), 0, 1)
    r = np.clip(np.asarray(rh, np.float32) / 100, 0, 1)
    cf = 1 - np.sqrt(np.clip((1 - r) / (1 - rhc), 0, 1))
    if clw is not None:
        lo, hi = CC_CLW
        g = np.maximum(np.nan_to_num(clw, nan=0) * 1000, 1e-09)
        cf = np.maximum(cf, np.clip(np.log(g / lo) / math.log(hi / lo), 0, 1))
    with np.errstate(invalid='ignore'):
        return np.where(sigma <= 1, 100 * cf, np.nan).astype(np.float32)

def cloud_cover_levels(levels, rh, ps, rh2, clw=None, targets=CC_LEVELS, rh_out=None):
    levels = np.asarray(levels, np.float32)
    out = []
    for pt in targets:
        k = np.flatnonzero(levels == pt)
        if k.size:
            r, c = (rh[k[0]], None if clw is None else clw[k[0]])
        else:
            lower, upper = (np.flatnonzero(levels > pt), np.flatnonzero(levels < pt))
            if not lower.size or not upper.size:
                out.append(np.full(np.shape(ps), np.nan, np.float32))
                if rh_out is not None:
                    rh_out.append(np.full(np.shape(ps), np.nan, np.float32))
                continue
            i, j = (lower[np.argmin(levels[lower])], upper[np.argmax(levels[upper])])
            under = levels[i] > ps
            p0, r0 = (np.where(under, ps, levels[i]), np.where(under, rh2, rh[i]))
            with np.errstate(invalid='ignore', divide='ignore'):
                w = np.clip(np.log(pt / p0) / np.log(levels[j] / p0), 0, 1)
            r = r0 + w * (rh[j] - r0)
            if clw is not None:
                wn = math.log(pt / levels[i]) / math.log(levels[j] / levels[i])
                c = clw[i] + wn * (clw[j] - clw[i])
            else:
                c = None
        out.append(cloud_fraction(r, pt, ps, c))
        if rh_out is not None:
            rh_out.append(np.asarray(r, np.float32))
    return np.stack(out)
COVER_TOPS_NCEP = ('p', 642.0, 350.0)

def band_tops(tops, ps):
    kind, low_top, mid_top = tops
    ps = np.asarray(ps, np.float32)
    if kind == 'sigma':
        return (low_top * ps, mid_top * ps)
    if kind == 'icon':
        return (low_top * ps, mid_top * np.sqrt(1000.0 * np.maximum(ps, 0)))
    return (np.full(ps.shape, low_top, np.float32), np.full(ps.shape, mid_top, np.float32))

def calibrate_cover(cc, rh, ps, bulk, tops=COVER_TOPS_NCEP, targets=CC_LEVELS):
    p = np.asarray(targets, np.float32)[:, None, None]
    lo_t, mid_t = (a[None] for a in band_tops(tops, ps))
    out = np.array(cc, np.float32, copy=True)
    rh = np.asarray(rh, np.float32)
    for inband, target in zip((p >= lo_t, (p < lo_t) & (p >= mid_t), p < mid_t), bulk):
        if target is None:
            continue
        m = np.broadcast_to(inband, out.shape) & np.isfinite(out)
        t = np.clip(np.asarray(target, np.float32), 0, 100)
        ok = np.isfinite(t) & m.any(axis=0)
        peak = np.where(m, out, -np.inf).max(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            scaled = np.clip(out * (t / peak)[None], 0, 100)
        scale = ok & (peak >= 1)
        out = np.where(m & scale[None], scaled, out)
        put = ok & (peak < 1) & (t >= 1)
        k = np.where(m, np.nan_to_num(rh, nan=-1), -np.inf).argmax(axis=0)
        at = (np.arange(len(targets))[:, None, None] == k[None]) & put[None] & m
        out = np.where(at, t[None], np.where(m & put[None], 0, out))
    return out.astype(np.float32)

def level_heights(levels, z, targets=CC_LEVELS):
    levels = np.asarray(levels, np.float32)
    lp = np.log(levels)
    out = []
    for pt in targets:
        lower, upper = (np.flatnonzero(levels >= pt), np.flatnonzero(levels <= pt))
        if not lower.size or not upper.size:
            out.append(np.full(z.shape[1:], np.nan, np.float32))
            continue
        i, j = (lower[np.argmin(levels[lower])], upper[np.argmax(levels[upper])])
        w = 0.0 if i == j else (math.log(pt) - lp[i]) / (lp[j] - lp[i])
        out.append(z[i] + w * (z[j] - z[i]))
    return np.stack(out).astype(np.float32)
CEIL_COVER = 50.0
CEIL_COVER_OWN = 40.0
CEIL_LAYER_MIN = 12.5
FOG_RH = 97.0
CEIL_NEAR_GROUND, CEIL_NEAR_RH = (300.0, 97.0)

def ceiling_from_cover(zc, cc, zs, rh2, tcc=None, rh=None, cover=None, tcc_min=50.0):
    cover = CEIL_COVER if cover is None else cover
    base = np.full(zs.shape, np.nan, np.float32)
    prev_z, prev_t = (zs.astype(np.float32), np.zeros(zs.shape, np.float32))
    first = np.full(zs.shape, np.nan, np.float32)
    below = np.zeros(zs.shape, np.float32)
    layer = np.zeros(zs.shape, np.float32)
    for k in range(cc.shape[0]):
        c, z = (cc[k], zc[k])
        if rh is not None:
            c = np.where((z - zs < CEIL_NEAR_GROUND) & (np.nan_to_num(rh[k], nan=0) < CEIL_NEAR_RH), 0, c)
        ok = np.isfinite(c) & np.isfinite(z)
        first = np.where(np.isnan(first) & ok, c, first)
        cloudy = ok & (c >= 1)
        ends = ok & ~cloudy & (layer > 0)
        below = np.where(ends, 100 - (100 - below) * (100 - layer) / 100, below)
        layer = np.where(ends, 0, np.where(cloudy, np.maximum(layer, c), layer))
        total = 100 - (100 - below) * (100 - layer) / 100
        hit = ok & np.isnan(base) & (np.where(ok, c, 0) >= np.minimum(CEIL_LAYER_MIN, cover)) & (total >= cover)
        f = np.where(prev_t >= cover, 1, np.clip((cover - prev_t) / np.maximum(total - prev_t, 0.001), 0, 1))
        base = np.where(hit, prev_z + f * (z - prev_z) - zs, base)
        prev_z, prev_t = (np.where(ok, z, prev_z), np.where(ok, total, prev_t))
    fog = (np.nan_to_num(rh2, nan=0) >= FOG_RH) & (np.nan_to_num(first, nan=0) >= cover)
    base = np.where(fog, 30.0, base)
    broken = np.ones(zs.shape, bool) if tcc is None else np.nan_to_num(tcc, nan=100) >= tcc_min
    return np.where(np.isfinite(base) & broken & (base < CEIL_NONE), np.maximum(base, 30), CEIL_NONE).astype(np.float32)

def ceiling_top(ceil, zs, zc, cc):
    has = np.isfinite(ceil) & (ceil < CEIL_NONE)
    base = (zs + np.where(has, ceil, 0)).astype(np.float32)
    top = np.full(zs.shape, np.nan, np.float32)
    thr = np.full(zs.shape, np.nan, np.float32)
    prev_z, prev_c = (base, np.full(zs.shape, 100, np.float32))
    for k in range(cc.shape[0]):
        c, z = (cc[k], zc[k])
        at = has & np.isnan(thr) & np.isfinite(c) & np.isfinite(z) & (z >= base - 1)
        thr = np.where(at, np.clip(np.where(at, c, 0), CEIL_LAYER_MIN, CEIL_COVER), thr)
        ok = has & np.isfinite(c) & np.isfinite(z) & (z > base)
        end = ok & np.isnan(top) & (c < thr)
        f = np.clip((prev_c - thr) / np.maximum(prev_c - c, 0.001), 0, 1)
        top = np.where(end, prev_z + f * (z - prev_z), top)
        prev_z, prev_c = (np.where(ok, z, prev_z), np.where(ok, c, prev_c))
    return top
ICON_CALIB = {'low': (4.0, 1.5), 'mid': (3.0, 1.5)}
_EPS_CLC = 1e-07

def _icon_calib_layer(c, f):
    c = np.clip(c, 0, 1)
    return np.minimum(f * c, np.exp((1 + c) / 2 * np.log(np.maximum(_EPS_CLC, c))))

def icon_calibrate(tcc, lcc, mcc, hcc, lsm=None):
    land = np.ones(np.shape(tcc), bool) if lsm is None else np.nan_to_num(lsm, nan=1) > 0
    t, l, m, h = (np.clip(np.asarray(a, np.float32) / 100, 0, 1) for a in (tcc, lcc, mcc, hcc))
    lm = _icon_calib_layer(l, np.where(land, *ICON_CALIB['low']))
    mm = _icon_calib_layer(m, np.where(land, *ICON_CALIB['mid']))
    fac = (lm + mm + h) / np.maximum(_EPS_CLC, l + m + h)
    fac = np.minimum(fac, np.sqrt(1 / np.maximum(0.05, t)))
    return (100 * fac * t, 100 * lm, 100 * mm)

def icon_uncalibrate(tcc, lcc, mcc, hcc, lsm=None):
    land = np.ones(np.shape(tcc), bool) if lsm is None else np.nan_to_num(lsm, nan=1) > 0
    t, l, m, h = (np.clip(np.nan_to_num(np.asarray(a, np.float32), nan=0) / 100, 0, 1) for a in (tcc, lcc, mcc, hcc))

    def layer(y, f):
        x = np.maximum(y * y, 1e-12)
        ly = np.log(np.maximum(y, 1e-12))
        for _ in range(5):
            lx = np.log(x)
            x = np.clip(x - ((1 + x) / 2 * lx - ly) / (lx / 2 + (1 + x) / (2 * x)), 1e-12, 1)
        x = np.where(y <= 0, 0, np.where(y >= 1, 1, x))
        return np.minimum(np.maximum(y / f, x), y)
    lr = layer(l, np.where(land, *ICON_CALIB['low']))
    mr = layer(m, np.where(land, *ICON_CALIB['mid']))
    grow = np.where(l + m + h > _EPS_CLC, (l + m + h) / np.maximum(_EPS_CLC, lr + mr + h), 1)
    grow = np.where(lr + mr + h > _EPS_CLC, grow, 1)
    tr = np.maximum(t / np.maximum(grow, 1), np.where(t < math.sqrt(0.05), t / math.sqrt(20), t * t))
    tr = np.minimum(tr, t)
    keep = lambda a, r: np.where(np.isfinite(np.asarray(a, np.float32)), 100 * r, np.nan).astype(np.float32)
    return (keep(tcc, tr), keep(lcc, lr), keep(mcc, mr))
HAZE_SCALE = 3.0

def visibility_estimate(t2, td2, rain, snow=None):
    rh = rh_from_td(t2, td2)
    haze = HAZE_SCALE * 6000 * np.maximum(t2 - td2, 0) / np.maximum(rh, 1) ** 1.75
    r = np.maximum(np.nan_to_num(rain, nan=0), 0)
    haze = np.where(r >= 0.05, 10, haze)
    wet = 10 / (1 + 0.5 * r)
    if snow is not None:
        wet = np.where(snow, 3 / (1 + 1.5 * r), wet)
    precip = np.where(r >= 0.05, wet, 10)
    return np.clip(np.minimum(haze, precip), 0.1, 10) * SM

def flight_category(ceil_agl, vis):
    ft = np.nan_to_num(ceil_agl, nan=1000000000.0) / FT
    sm = np.nan_to_num(vis, nan=1000000000.0) / SM
    cat = np.ones(np.shape(ceil_agl), np.float32)
    cat = np.where((ft <= 3000) | (sm <= 5), 2, cat)
    cat = np.where((ft < 1000) | (sm < 3), 3, cat)
    cat = np.where((ft < 500) | (sm < 1), 4, cat)
    return np.where(np.isnan(ceil_agl) & np.isnan(vis), np.nan, cat)

def freezing_level(prof: Profile):
    t, z = (prof.t, prof.z)
    out = np.where(t[0] <= 0, prof.zs, np.nan)
    for k in range(1, t.shape[0]):
        cross = np.isnan(out) & (t[k - 1] > 0) & (t[k] <= 0)
        f = t[k - 1] / np.maximum(t[k - 1] - t[k], 1e-06)
        out = np.where(cross, z[k - 1] + f * (z[k] - z[k - 1]), out)
    return np.where(np.isnan(out), prof.ztop, out)
ICE_MIN = 20.0
ICE_BANDS = (ICE_MIN, 35.0, 60.0)

def icing_potential(z, t, rh, clw, zs, zmax=7300.0):
    tf = np.clip(-t / 3, 0, 1) * np.clip((t + 22) / 10, 0, 1)
    rhf = np.clip((rh - 75) / 20, 0, 1)
    if clw is not None:
        moist = np.maximum(np.clip(np.nan_to_num(clw, nan=0) * 1000 / 0.1, 0, 1), 0.4 * rhf)
    else:
        moist = 0.85 * rhf
    pot = np.where((z > zs[None] + 30) & (z <= zmax), 100 * tf * moist, 0)
    pot = np.nan_to_num(pot, nan=0)
    on = pot >= ICE_MIN
    base = np.where(on, z, np.inf).min(axis=0)
    top = np.where(on, z, -np.inf).max(axis=0)
    return (pot.max(axis=0), np.where(np.isfinite(base), base, np.nan), np.where(np.isfinite(top), top, np.nan))
CAT_FLOOR_AGL = 3000 * FT
TURB_BANDS = (6.0, 12.0, 18.0)

def ellrod_turbulence(prof: Profile, ddx, ddy, smooth=None, zi=None, zmin_agl=CAT_FLOOR_AGL, zmax=10500.0, dz=500.0, band=False):
    zs = prof.zs
    floor = zs + np.maximum(zmin_agl, 0 if zi is None else np.nan_to_num(zi, nan=0))
    hs = np.arange(dz, zmax + dz, dz, dtype=np.float32)
    winds = [prof.at(np.full(zs.shape, h, np.float32), 'u', 'v') for h in hs]
    return _ellrod_layers(prof, winds, hs, floor, ddx, ddy, smooth, dz, band)

def cat_layer_clear(bottom, floor, smooth=None):
    from scipy.ndimage import maximum_filter
    below = (bottom < floor).astype(np.float32)
    if smooth is not None:
        below = smooth(below)
    return maximum_filter(below, size=3, mode='nearest') < 1e-06

def _ellrod_layers(prof, winds, hs, floor, ddx, ddy, smooth, dz, band=False):
    zs = prof.zs
    best = np.zeros(zs.shape, np.float32)
    bestz = np.full(zs.shape, np.nan, np.float32)
    lo = np.full(zs.shape, np.inf, np.float32)
    hi = np.full(zs.shape, -np.inf, np.float32)
    for i in range(1, len(hs) - 1):
        (u0, v0), (um, vm), (u1, v1) = (winds[i - 1], winds[i], winds[i + 1])
        vws = np.hypot(u1 - u0, v1 - v0) / (2 * dz)
        um, vm = (np.nan_to_num(um, nan=0), np.nan_to_num(vm, nan=0))
        if smooth is not None:
            um, vm = (smooth(um), smooth(vm))
        deform = np.hypot(ddx(um) - ddy(vm), ddx(vm) + ddy(um))
        ti = np.nan_to_num(vws * deform * 10000000.0, nan=0)
        ti = np.where(cat_layer_clear(hs[i - 1], floor, smooth) & (hs[i + 1] <= prof.ztop), ti, 0)
        better = ti > best
        best = np.where(better, ti, best)
        bestz = np.where(better, hs[i], bestz)
        rough = ti >= TURB_BANDS[0]
        lo = np.where(rough, np.minimum(lo, hs[i - 1]), lo)
        hi = np.where(rough, np.maximum(hi, hs[i + 1]), hi)
    if band:
        return (best, bestz, np.where(np.isfinite(lo), lo, np.nan), np.where(np.isfinite(hi), hi, np.nan))
    return (best, bestz)

def precip_type(rain, crain=None, csnow=None, cfrzr=None, cicep=None, ecmwf=None, t2=None):
    wet = np.nan_to_num(rain, nan=0) >= 0.05
    f = lambda a: None if a is None else np.nan_to_num(a, nan=0) >= 0.5
    if ecmwf is not None:
        c = np.round(np.nan_to_num(ecmwf, nan=0))
        code = np.select([np.isin(c, [3, 12]), c == 8, c == 7, np.isin(c, [5, 6]), c == 1], [5, 4, 3, 2, 1], 1)
    elif crain is not None or csnow is not None:
        r, s, z, p = (f(a) if a is not None else np.zeros(np.shape(rain), bool) for a in (crain, csnow, cfrzr, cicep))
        code = np.select([z, p, r & s, s, r], [5, 4, 3, 2, 1], 1)
    elif t2 is not None:
        code = np.where(t2 <= 0.5, 2, 1)
    else:
        code = np.ones(np.shape(rain))
    return np.where(wet, code, 0).astype(np.float32)
SLR_RANGE = (3.0, 30.0)
KUCHERA_T = 271.16 - 273.15

def kuchera_ratio(prof: Profile):
    lim = np.log(500.0) - 1e-06
    tmax = prof.t[0].copy()
    for k in range(1, prof.t.shape[0]):
        low = prof.lnp[k] >= lim
        if low.all():
            np.maximum(tmax, prof.t[k], out=tmax)
        elif low.any():
            np.maximum(tmax, np.where(low, prof.t[k], -np.inf), out=tmax)
        else:
            break
    d = KUCHERA_T - tmax
    return np.clip(12 + np.where(d < 0, 2 * d, d), *SLR_RANGE).astype(np.float32)

def reflectivity_from_rain(rain):
    r = np.maximum(np.nan_to_num(rain, nan=0), 0)
    return np.where(r >= 0.05, 10 * np.log10(200 * np.maximum(r, 1e-06) ** 1.6), 0)
