from __future__ import annotations
import calendar
import math
from datetime import datetime, timedelta, timezone
import numpy as np
FINE_STEP = 0.09
COVERS = ('cloud', 'cloudlow', 'cloudmid', 'cloudhigh')
INPUTS = ('sun', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ceil', 'cudepth', 'overdev', 'zi', 'wstar', 'cubase', 'elevation')
SLANT_MIN_ELEV, SLANT_MAX_KM, SLANT_LOW, SLANT_MID, SLANT_HIGH = (10.0, 50.0, 1500.0, 4500.0, 9000.0)
LOW_CEIL, MID_CEIL = (3000.0, 5990.0)
THIN_SUN, SUN_CLEAR = (0.6, 0.95)
CU_COVER = (0.2, 0.25)
CU_OVERSHOOT, CU_OVERSHOOT_MIN = (0.5, 500.0)
W_MID, W_HIGH = (0.7, 0.4)
CU_CAP = 3000.0
PI = math.pi
RAD = PI / 180
CHUNK = 128
_NET9 = ((0, 3), (1, 7), (2, 5), (4, 8), (0, 7), (2, 4), (3, 8), (5, 6), (0, 2), (1, 3), (4, 5), (7, 8), (1, 4), (3, 6), (5, 7), (0, 1), (2, 4), (3, 5), (6, 8), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6))

def area_median(a):
    ny, nx = a.shape
    p = np.pad(np.where(np.isnan(a), np.inf, a), 1, constant_values=np.inf)
    out = np.empty((ny, nx))
    tmp = np.empty((32, nx))
    for y0 in range(0, ny, 32):
        y1 = min(ny, y0 + 32)
        s = [p[y0 + dy:y1 + dy, dx:dx + nx].copy() for dy in range(3) for dx in range(3)]
        t = tmp[:y1 - y0]
        for i, j in _NET9:
            np.minimum(s[i], s[j], out=t)
            np.maximum(s[i], s[j], out=s[j])
            s[i], t = (t, s[i])
        st = np.stack(s)
        n = np.isfinite(st).sum(0)
        lo = np.take_along_axis(st, np.maximum(n - 1, 0)[None] // 2, 0)[0]
        hi = np.take_along_axis(st, (n // 2)[None], 0)[0]
        out[y0:y1] = np.where(n % 2 == 1, lo, (lo + hi) / 2)
    return np.where(np.isnan(a), np.nan, out)

def js_round(x):
    r = np.floor(x)
    return r + (x - r >= 0.5)

def clamp01(x):
    return np.maximum(0.0, np.minimum(1.0, x))

def _ms(d):
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return (calendar.timegm(d.timetuple()) * 1000 + d.microsecond // 1000, d)

def sun_terms(grid, when_utc):
    ms, d = _ms(when_utc)
    ms0, _ = _ms(datetime(d.year, 1, 1) - timedelta(days=1))
    doy = (ms - ms0) / 86400000
    decl = 23.44 * RAD * math.sin(2 * PI * (284 + doy) / 365)
    hm = d.hour + d.minute / 60
    b = 2 * PI * (doy - 81) / 364
    eot = (9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)) / 60
    sd, cd = (math.sin(decl), math.cos(decl))
    north, west, step = (grid['north'], grid['west'], grid['step'])
    rows = []
    for iy in range(grid['ny']):
        la = (north - iy * step) * RAD
        sl, cl = (math.sin(la), math.cos(la))
        rows.append((sl * sd, cl * cd, cl * sd, sl * cd))
    cols = []
    for ix in range(grid['nx']):
        ha = 15 * (hm + (west + ix * step) / 15 + eot - 12) * RAD
        cols.append((math.cos(ha), -math.sin(ha) * cd))
    return (np.array(rows, dtype=np.float64).reshape(-1, 4), np.array(cols, dtype=np.float64).reshape(-1, 2))

def sun_at(rows, cols):
    r0, r1, r2, r3 = (rows[:, k:k + 1] for k in range(4))
    c0, c1 = (cols[None, :, 0], cols[None, :, 1])
    s = r0 + r1 * c0
    az = np.arctan2(c1, r2 - r3 * c0)
    return (np.arcsin(np.clip(s, -1.0, 1.0)) / RAD, np.fmod(az / RAD + 360, 360))

def _cu_reach(zi, cb):
    with np.errstate(invalid='ignore'):
        one = ~(cb > 0) | ~(zi > 0) | (cb <= zi)
        r = 1 - np.minimum(1.0, (cb - zi) / np.maximum(CU_OVERSHOOT * zi, CU_OVERSHOOT_MIN))
    return np.where(one, 1.0, r)

def _cu_cover(cud, od, reach):
    with np.errstate(invalid='ignore'):
        c = (CU_COVER[0] + CU_COVER[1] * clamp01(cud / 3000)) * clamp01(reach)
        c = np.where(od > 30, np.maximum(c, 0.4 + 0.3 * clamp01((od - 30) / 50)), c)
        return np.where(cud > 0, c, 0.0)

def _cu_at(cb, cud, od, zi):
    c = _cu_cover(cud, np.where(np.isnan(od), 0.0, od), _cu_reach(zi, cb))
    return np.where(np.isnan(cb), 0.0, c)

def _nz(v):
    return np.where(np.isnan(v), 0.0, v)

def _f(v, k):
    return 1 - k * clamp01(_nz(v) / 100)

def _cloud_shade(low, mid, high, total, cu, thin_mid):
    with np.errstate(invalid='ignore'):
        f_low = _f(np.maximum(_nz(low), cu * 100), 1)
        f_mid = _f(mid, np.where(thin_mid, W_HIGH, W_MID))
        layered = 1 - f_low * f_mid * _f(high, W_HIGH)
        by_total = np.where(np.isnan(total), np.where(cu > 0, cu, np.nan), np.maximum(0.8 * clamp01(total / 100), cu))
    has = ~np.isnan(low) | ~np.isnan(mid)
    thick = 1 - f_low * np.where(thin_mid, 1.0, _f(mid, W_MID)) * 1.0
    return (np.where(has, layered, by_total), thick, has)

def shade(fields, grid, when_utc, mid_by_ceil, agl_heights=True, rays=False):
    ny, nx, step = (int(grid['ny']), int(grid['nx']), float(grid['step']))
    north = float(grid['north'])
    nan = np.full((ny, nx), np.nan)

    def get(name):
        a = fields.get(name)
        return nan if a is None else np.asarray(a, dtype=np.float64).reshape(ny, nx)
    fine = step <= FINE_STEP
    F = {n: get(n) for n in ('sun', 'ceil', 'cudepth', 'overdev', 'zi', 'wstar')}
    for n in COVERS:
        F[n] = area_median(get(n)) if fine and fields.get(n) is not None else get(n)
    cub = get('cubase')
    if agl_heights:
        F['cb'] = cub
    elif fields.get('elevation') is None or fields.get('cubase') is None:
        F['cb'] = nan
    else:
        with np.errstate(invalid='ignore'):
            F['cb'] = np.where(np.isnan(cub) | np.isnan(get('elevation')), np.nan, np.maximum(0.0, cub - get('elevation')))
    flat = {k: v.ravel() for k, v in F.items()}
    rows, cols = sun_terms(grid, when_utc)
    cell = 111.2 * step
    out = {'shade': np.empty((ny, nx)), 'thin': np.empty((ny, nx))}
    if rays:
        for k in ('elev', 'az', 'sun'):
            out[k] = np.empty((ny, nx))
        for k in ('oL', 'oM', 'oH', 'oC'):
            out[k] = (np.empty((ny, nx)), np.empty((ny, nx)))
    IX = np.arange(nx)[None, :]
    for y0 in range(0, ny, CHUNK):
        y1 = min(ny, y0 + CHUNK)
        iy = np.arange(y0, y1)
        IY = iy[:, None]
        lat = (north - iy * step)[:, None]
        elev, az = sun_at(rows[y0:y1], cols)
        up = elev > 0
        tn = np.tan(np.maximum(elev, SLANT_MIN_ELEV) * PI / 180)
        a = az * PI / 180
        sa, ca = (np.sin(a), np.cos(a))
        den_x = cell * np.maximum(np.cos(lat * PI / 180), 0.1)
        node = IY * nx + IX
        sl = slice(y0, y1)

        def offset(h):
            with np.errstate(invalid='ignore'):
                ok = up & (h > 0)
                km = np.minimum(h / 1000 / tn, SLANT_MAX_KM)
                dx = np.where(ok, js_round(km * sa / den_x), 0.0).astype(np.int64)
                dy = np.where(ok, js_round(-km * ca / cell), 0.0).astype(np.int64)
            x, y = (IX + dx, IY + dy)
            inb = ok & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
            idx = np.where(inb, y * nx + x, node)
            return (ok, inb, idx, dx, dy)

        def at(name, o):
            v = flat[name][o[2]]
            return np.where(o[1], v, np.nan)

        def along(name, o):
            v = at(name, o)
            return np.where(o[0] & ~np.isnan(v), v, F[name][sl])
        ceil0, cb0, cud0 = (F['ceil'][sl], F['cb'][sl], F['cudepth'][sl])
        has_ceil = ~np.isnan(ceil0)
        with np.errstate(invalid='ignore'):
            h_low = np.where(has_ceil & (ceil0 < LOW_CEIL), ceil0, np.where(np.isnan(cb0), SLANT_LOW, cb0))
            h_mid = np.where(has_ceil & (ceil0 >= LOW_CEIL) & (ceil0 < MID_CEIL), ceil0, SLANT_MID)
            h_cu = cb0 + np.minimum(_nz(cud0), CU_CAP) / 2
        oL, oM, oH = (offset(h_low), offset(h_mid), offset(np.full(elev.shape, SLANT_HIGH)))
        has_cb = ~np.isnan(cb0)
        oc = offset(np.where(has_cb, h_cu, 0.0))
        oC = tuple((np.where(has_cb, p, q) for p, q in zip(oc, oL)))
        cu_node = _cu_at(cb0, cud0, F['overdev'][sl], F['zi'][sl])
        cu_ray = _cu_at(at('cb', oC), at('cudepth', oC), at('overdev', oC), at('zi', oC))
        cu = np.where(oC[0] & ~np.isnan(at('wstar', oC)), cu_ray, cu_node)
        c_low, c_mid, c_high = (along('cloudlow', oL), along('cloudmid', oM), along('cloudhigh', oH))
        c_all, c_ceil = (along('cloud', oM), along('ceil', oM))
        l0 = np.maximum(_nz(c_low), 100 * cu)
        l1, l2 = (W_MID * _nz(c_mid), W_HIGH * _nz(c_high))
        top = np.maximum(np.maximum(l0, l1), l2)
        s_low, s_mid = (l0 == top, (l0 != top) & (l1 == top))
        oS = tuple((np.where(s_low, p, np.where(s_mid, q, r)) for p, q, r in zip(oL, oM, oH)))
        sun = along('sun', oS)
        with np.errstate(invalid='ignore'):
            thin_mid = mid_by_ceil & ~np.isnan(c_ceil) & (c_ceil >= MID_CEIL)
        cs, thick, has = _cloud_shade(c_low, c_mid, c_high, c_all, cu, thin_mid)
        has_sun = ~np.isnan(sun)
        with np.errstate(invalid='ignore', divide='ignore'):
            out['shade'][sl] = np.where(has_sun, np.maximum(clamp01(1 - sun / SUN_CLEAR), cu), cs)
            share = np.where((cs > 0) & has, clamp01(1 - thick / cs), 0.0)
            out['thin'][sl] = np.where(has_sun & (sun < THIN_SUN), 0.0, share)
        if rays:
            out['elev'][sl], out['az'][sl], out['sun'][sl] = (elev, az, sun)
            for k, o in (('oL', oL), ('oM', oM), ('oH', oH), ('oC', oC)):
                out[k][0][sl] = np.where(o[0], o[3], np.nan)
                out[k][1][sl] = np.where(o[0], o[4], np.nan)
    return out
