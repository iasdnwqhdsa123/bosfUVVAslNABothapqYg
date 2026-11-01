from __future__ import annotations
import math
import numpy as np
from scipy import ndimage
R_EARTH = 6371000.0
STEP = 0.25
SMOOTH_KM = 90.0
WIND_SMOOTH_KM = 150.0
GRAD_MIN = 0.4
TFP_MIN = 0.3
MIN_LEN_KM = 400.0
MIN_PART_KM = 500.0
MOVE_MS = 1.5
ALONG_KM = 300.0
JOIN_KM = 200.0
JOIN_TURN = 50.0
CENTRE_KM = 800.0
CENTRE_HPA = 2.0
CENTRE_SMOOTH_KM = 100.0
CENTRE_RAW_KM = 200.0
LEVEL = 850
COORD = 0.1
SIMPLIFY_KM = 12.0
UNDERGROUND_HPA = 10.0
EDGE_KM = 150.0
TGRAD_MIN = 0.3
MIN_SPAN_DEG = 12.0
SUB_FROM, SUB_TO = (36.0, 30.0)
SUB_LEN = 2.0
SUB_RATIO = 0.6
LINE_GRAD = 0.6
WIND_SFC = 0.5
OCC_LOW_KM = 400.0
OCC_DEPTH = 8.0
OCC_PART_KM = 150.0

def theta_e(t_c, td_c, p_hpa):
    t = np.asarray(t_c, np.float64) + 273.15
    td = np.minimum(np.asarray(td_c, np.float64), np.asarray(t_c, np.float64)) + 273.15
    e = 6.112 * np.exp(17.67 * (td - 273.15) / (td - 29.65))
    r = 0.622 * e / np.maximum(p_hpa - e, 1.0)
    tl = 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0
    th = t * (1000.0 / p_hpa) ** (0.2854 * (1 - 0.28 * r))
    return th * np.exp((3.376 / tl - 0.00254) * r * 1000 * (1 + 0.81 * r))

def theta_w(t_c, td_c, p_hpa):
    te = theta_e(t_c, td_c, p_hpa)
    x = te / 273.15
    a = (7.101574, -20.68208, 16.11182, 2.574631, -5.205688)
    b = (-3.552497, 3.781782, -0.6899655, -0.592934)
    num = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3 + a[4] * x ** 4
    den = 1 + b[0] * x + b[1] * x ** 2 + b[2] * x ** 3 + b[3] * x ** 4
    return np.where(te >= 173.15, te - np.exp(num / den), te)

def block_mean(a, k):
    if k <= 1:
        return np.asarray(a, np.float64)
    ny, nx = (a.shape[0] // k * k, a.shape[1] // k * k)
    b = np.asarray(a[:ny, :nx], np.float64).reshape(ny // k, k, nx // k, k)
    ok = np.isfinite(b)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, b, 0).sum(axis=(1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

def smooth(a, sy, sx, fill=True):
    ok = np.isfinite(a)
    w = ndimage.gaussian_filter(ok.astype(np.float64), (sy, sx), mode='nearest')
    v = ndimage.gaussian_filter(np.where(ok, a, 0.0), (sy, sx), mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(w > 0.05, v / w, np.nan)
    return out if fill else np.where(ok, out, np.nan)

class Grid:

    def __init__(self, lats, lons):
        self.lats = np.asarray(lats, np.float64)
        self.lons = np.asarray(lons, np.float64)
        self.dlat = abs(self.lats[1] - self.lats[0])
        self.dlon = abs(self.lons[1] - self.lons[0])
        self.dy = R_EARTH * math.radians(self.dlat)
        self.dx = (R_EARTH * math.radians(self.dlon) * np.cos(np.radians(self.lats)))[:, None]

    def grad(self, a):
        gy, gx = np.gradient(a)
        return (gx / self.dx, -gy / self.dy)

    def sigma(self, km):
        sy = km * 1000 / self.dy
        c = max(0.2, math.cos(math.radians(float(np.mean(self.lats)))))
        return (sy, km * 1000 / (self.dy * self.dlon / self.dlat * c))

    def lonlat(self, fy, fx):
        return (self.lons[0] + fx * (self.lons[1] - self.lons[0]), self.lats[0] + fy * (self.lats[1] - self.lats[0]))

    def cell(self, lon, lat):
        return ((lat - self.lats[0]) / (self.lats[1] - self.lats[0]), (lon - self.lons[0]) / (self.lons[1] - self.lons[0]))

def sample(a, fy, fx):
    fy, fx = (np.asarray(fy, np.float64), np.asarray(fx, np.float64))
    ny, nx = a.shape
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    ty, tx = (fy - y0, fx - x0)
    v = a[y0, x0] * (1 - ty) * (1 - tx) + a[y0, x0 + 1] * (1 - ty) * tx + a[y0 + 1, x0] * ty * (1 - tx) + a[y0 + 1, x0 + 1] * ty * tx
    inside = (fy >= 0) & (fy <= ny - 1) & (fx >= 0) & (fx <= nx - 1)
    return np.where(inside, v, np.nan)

def km_between(lon0, lat0, lon1, lat1):
    p0, p1 = (np.radians(lat0), np.radians(lat1))
    dl = np.radians(np.asarray(lon1) - np.asarray(lon0))
    h = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH / 1000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def zero_lines(f, keep):
    ny, nx = f.shape
    fin = np.isfinite(f)
    s = np.where(fin, f > 0, False)

    def cross(a, b, ok):
        with np.errstate(invalid='ignore', divide='ignore'):
            t = a / (a - b)
        return (ok & ((a > 0) != (b > 0)), t)
    hc, ht = cross(f[:, :-1], f[:, 1:], fin[:, :-1] & fin[:, 1:])
    vc, vt = cross(f[:-1, :], f[1:, :], fin[:-1, :] & fin[1:, :])
    cell_ok = keep[:-1, :-1] & keep[:-1, 1:] & keep[1:, :-1] & keep[1:, 1:]
    del s
    H = ny * nx
    pts = {}

    def point(eid):
        if eid not in pts:
            if eid < H:
                y, x = divmod(eid, nx)
                pts[eid] = (float(y), x + float(ht[y, x]))
            else:
                y, x = divmod(eid - H, nx)
                pts[eid] = (y + float(vt[y, x]), float(x))
        return pts[eid]
    adj = {}
    ys, xs = np.nonzero(cell_ok & (hc[:-1, :] | hc[1:, :] | vc[:, :-1] | vc[:, 1:]))
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges = []
        if hc[y, x]:
            edges.append(y * nx + x)
        if vc[y, x + 1]:
            edges.append(H + y * nx + x + 1)
        if hc[y + 1, x]:
            edges.append((y + 1) * nx + x)
        if vc[y, x]:
            edges.append(H + y * nx + x)
        if len(edges) == 2:
            pairs = [edges]
        elif len(edges) == 4:
            c = f[y, x] + f[y, x + 1] + f[y + 1, x] + f[y + 1, x + 1]
            top_pos = f[y, x] > 0
            pairs = [[edges[0], edges[3]], [edges[1], edges[2]]] if (c > 0) != top_pos else [[edges[0], edges[1]], [edges[2], edges[3]]]
        else:
            continue
        for a, b in pairs:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    lines, seen = ([], set())
    starts = [e for e, n in adj.items() if len(n) == 1] + list(adj)
    for e0 in starts:
        if e0 in seen:
            continue
        line, prev, cur = ([e0], None, e0)
        seen.add(e0)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = (cur, nxt[0])
            seen.add(cur)
            line.append(cur)
        if len(line) >= 2:
            lines.append([point(e) for e in line])
    return lines

def simplify(xy, tol):
    n = len(xy)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = (xy[i], xy[j])
        ab = b - a
        L = math.hypot(*ab)
        seg = xy[i + 1:j] - a
        d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return list(np.nonzero(keep)[0])

def _runs(types, seglen, min_km):
    types = list(types)
    for _ in range(len(types)):
        runs, i = ([], 0)
        while i < len(types):
            j = i
            while j + 1 < len(types) and types[j + 1] == types[i]:
                j += 1
            runs.append((i, j, float(sum(seglen[i:j + 1]))))
            i = j + 1
        short = [r for r in runs if r[2] < min_km]
        if not short or len(runs) == 1:
            break
        i, j, _ = min(short, key=lambda r: r[2])
        k = runs.index((i, j, _))
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        nb = max([r for r in (left, right) if r], key=lambda r: r[2])
        for m in range(i, j + 1):
            types[m] = types[nb[0]]
    return types

def join_lines(lines, km=None, max_turn=None):
    km = JOIN_KM if km is None else km
    max_turn = JOIN_TURN if max_turn is None else max_turn
    lines = [ln for ln in lines if len(ln) >= 2]
    if len(lines) < 2:
        return lines

    def heading(ln, end):
        a, b = (ln[-min(4, len(ln))], ln[-1]) if end else (ln[min(3, len(ln) - 1)], ln[0])
        c = math.cos(math.radians(b[1]))
        v = np.array([(b[0] - a[0]) * c, b[1] - a[1]])
        n = math.hypot(*v)
        return v / n if n > 0 else v
    cos_max = math.cos(math.radians(max_turn))
    ends = np.array([p for ln in lines for p in (ln[0], ln[-1])])
    d = km_between(ends[:, None, 0], ends[:, None, 1], ends[None, :, 0], ends[None, :, 1])
    own = np.arange(len(ends)) // 2
    d[own[:, None] == own[None, :]] = np.inf
    ii, jj = np.nonzero(np.triu(d <= km))
    order = np.argsort(d[ii, jj], kind='stable')
    pi, pj = (ii[order].tolist(), jj[order].tolist())
    at = {}
    for k, (i, j) in enumerate(zip(pi, pj)):
        at.setdefault(i, []).append(k)
        at.setdefault(j, []).append(k)
    geom = {n: ln for n, ln in enumerate(lines)}
    first, last = ({n: 2 * n for n in geom}, {n: 2 * n + 1 for n in geom})
    chain = {e: e // 2 for e in range(len(ends))}
    used, k, nxt = (set(), 0, len(lines))
    while k < len(pi):
        i, j = (pi[k], pj[k])
        ci, cj = (chain[i], chain[j])
        if i in used or j in used or ci == cj:
            k += 1
            continue
        if float(np.dot(heading(geom[ci], last[ci] == i), -heading(geom[cj], last[cj] == j))) < cos_max:
            k += 1
            continue
        a, oa = (geom[ci], first[ci]) if last[ci] == i else (geom[ci][::-1], last[ci])
        b, ob = (geom[cj], last[cj]) if first[cj] == j else (geom[cj][::-1], first[cj])
        geom[nxt], first[nxt], last[nxt] = (np.vstack([a, b]), oa, ob)
        chain[oa] = chain[ob] = nxt
        for c in (ci, cj):
            del geom[c], first[c], last[c]
        used.update((i, j))
        nxt += 1
        k = min([x for e in (oa, ob) for x in at.get(e, ()) if x < k] + [k + 1])
    return list(geom.values())

def centres(grid, mslp):
    if mslp is None or not np.isfinite(mslp).any():
        return []
    raw = np.asarray(mslp, np.float64)
    p = smooth(mslp, *grid.sigma(CENTRE_SMOOTH_KM))
    sy, sx = grid.sigma(CENTRE_KM)
    size = (max(3, int(round(2 * sy)) | 1), max(3, int(round(2 * sx)) | 1))
    filled = np.where(np.isfinite(p), p, np.nanmean(p))
    mean = ndimage.uniform_filter(filled, size, mode='nearest')
    out = []
    ny, nx = p.shape
    for kind, ext, sign in (('L', ndimage.minimum_filter, -1), ('H', ndimage.maximum_filter, 1)):
        e = ext(np.where(np.isfinite(p), p, np.inf if sign < 0 else -np.inf), size=size, mode='nearest')
        ys, xs = np.nonzero((p == e) & np.isfinite(p) & (sign * (p - mean) >= CENTRE_HPA))
        for y, x in zip(ys.tolist(), xs.tolist()):
            if y < 2 or x < 2 or y > ny - 3 or (x > nx - 3):
                continue
            ry, rx = (max(1, int(round(sy * CENTRE_RAW_KM / CENTRE_KM))), max(1, int(round(sx * CENTRE_RAW_KM / CENTRE_KM))))
            box = raw[max(0, y - ry):y + ry + 1, max(0, x - rx):x + rx + 1]
            if np.isfinite(box).any():
                k = np.nanargmin(box) if sign < 0 else np.nanargmax(box)
                by, bx = np.unravel_index(k, box.shape)
                y, x = (max(0, y - ry) + by, max(0, x - rx) + bx)
                val = float(raw[y, x])
            else:
                val = float(p[y, x])
            lon, lat = grid.lonlat(y, x)
            out.append((kind, float(lon), float(lat), val))
    return out

def _ramp(lat):
    return float(np.clip((SUB_FROM - abs(float(lat))) / max(SUB_FROM - SUB_TO, 1e-06), 0, 1))

def _surround(g, pm):
    p = smooth(pm, *g.sigma(CENTRE_SMOOTH_KM))
    sy, sx = g.sigma(CENTRE_KM)
    size = (max(3, int(round(2 * sy)) | 1), max(3, int(round(2 * sx)) | 1))
    return ndimage.uniform_filter(np.where(np.isfinite(p), p, np.nanmean(p)), size, mode='nearest')

def _occluded(g, lon, lat, lows, pmean):
    ny, nx = pmean.shape
    out = np.zeros(len(lon), bool)
    for _, x, y, hpa in lows:
        fy, fx = g.cell(x, y)
        if float(pmean[int(np.clip(round(float(fy)), 0, ny - 1)), int(np.clip(round(float(fx)), 0, nx - 1))]) - hpa >= OCC_DEPTH:
            out |= km_between(lon, lat, x, y) <= OCC_LOW_KM
    return out

def detect(lats, lons, t_c, td_c, u, v, ps=None, mslp=None, p_hpa=LEVEL, u10=None, v10=None):
    lats, lons = (np.asarray(lats, np.float64), np.asarray(lons, np.float64))
    if abs(lats[-1] - lats[0]) < MIN_SPAN_DEG or abs(lons[-1] - lons[0]) < MIN_SPAN_DEG:
        return {'fronts': [], 'centres': []}
    step = abs(lats[1] - lats[0])
    k = max(1, int(round(STEP / step)))
    tw = theta_w(t_c, td_c, p_hpa)
    t_c = np.asarray(t_c, np.float64)
    under = np.zeros(tw.shape, bool)
    if ps is not None:
        under = ~(np.asarray(ps) > p_hpa + UNDERGROUND_HPA)
        tw, t_c = (np.where(under, np.nan, tw), np.where(under, np.nan, t_c))
        u = np.where(under, np.nan, u)
        v = np.where(under, np.nan, v)
    tw, t_c, u, v = (block_mean(np.asarray(a, np.float64), k) for a in (tw, t_c, u, v))
    if WIND_SFC > 0 and u10 is not None and (v10 is not None):
        u10, v10 = (block_mean(np.asarray(a, np.float64), k) for a in (u10, v10))
        u = np.where(np.isfinite(u), (1 - WIND_SFC) * u + WIND_SFC * u10, u10)
        v = np.where(np.isfinite(v), (1 - WIND_SFC) * v + WIND_SFC * v10, v10)
    g = Grid(lats[:tw.shape[0] * k:k] + (lats[1] - lats[0]) * (k - 1) / 2, lons[:tw.shape[1] * k:k] + (lons[1] - lons[0]) * (k - 1) / 2)
    under = block_mean(under.astype(np.float64), k) > 0.5
    ey, ex = g.sigma(EDGE_KM)
    near_under = ndimage.binary_dilation(under, np.ones((2 * int(round(ey)) + 1, 2 * int(round(ex)) + 1), bool)) if under.any() else under
    pm = block_mean(np.asarray(mslp, np.float64), k) if mslp is not None else None
    th = smooth(tw, *g.sigma(SMOOTH_KM))
    tt = smooth(t_c, *g.sigma(SMOOTH_KM))
    us, vs = (smooth(u, *g.sigma(WIND_SMOOTH_KM)), smooth(v, *g.sigma(WIND_SMOOTH_KM)))
    gx, gy = g.grad(th)
    G = np.hypot(gx, gy)
    with np.errstate(invalid='ignore', divide='ignore'):
        nx_, ny_ = (gx / G, gy / G)
    Gx, Gy = g.grad(G)
    tfp = -(Gx * nx_ + Gy * ny_)
    Tx, Ty = g.grad(tfp)
    loc = Tx * nx_ + Ty * ny_
    tgx, tgy = g.grad(tt)
    tn = tgx * nx_ + tgy * ny_
    hk = 100000.0
    keep = (tfp * hk * hk > TFP_MIN) & (G * hk > GRAD_MIN) & (tn * hk > TGRAD_MIN) & ~near_under
    cs = centres(g, pm) if pm is not None else []
    lows = [c for c in cs if c[0] == 'L']
    pmean = _surround(g, pm) if lows else None
    fronts = []
    lines = [np.array(g.lonlat(np.array([p[0] for p in ln]), np.array([p[1] for p in ln]))).T for ln in zero_lines(loc, keep)]
    for ll in join_lines(lines):
        lon, lat = (ll[:, 0], ll[:, 1])
        seg = km_between(lon[:-1], lat[:-1], lon[1:], lat[1:])
        if seg.sum() < MIN_LEN_KM * (SUB_LEN if np.median(np.abs(lat)) < SUB_TO else 1.0):
            continue
        fy0, fx0 = g.cell(lon, lat)
        line_g = float(np.nanmean(sample(G, fy0, fx0))) * hk
        if line_g < LINE_GRAD:
            continue
        ramp = _ramp(np.median(lat))
        if ramp > 0 and float(np.nanmean(sample(tn, fy0, fx0))) * hk < SUB_RATIO * ramp * line_g:
            continue
        if len(lon) >= 5:
            lon = np.r_[lon[0], np.convolve(lon, np.ones(3) / 3, 'valid'), lon[-1]]
            lat = np.r_[lat[0], np.convolve(lat, np.ones(3) / 3, 'valid'), lat[-1]]
        fy, fx = g.cell(lon, lat)
        n_x, n_y = (sample(nx_, fy, fx), sample(ny_, fy, fx))
        across = sample(us, fy, fx) * n_x + sample(vs, fy, fx) * n_y
        seg = km_between(lon[:-1], lat[:-1], lon[1:], lat[1:])
        dist = np.r_[0, np.cumsum(seg)]
        sm = np.array([np.nanmean(across[np.abs(dist - d) <= ALONG_KM / 2]) for d in dist])
        types = np.where(sm > MOVE_MS, 'c', np.where(sm < -MOVE_MS, 'w', 's'))
        seg_types = _runs(types[:-1], seg, MIN_PART_KM)
        if lows:
            occ = _occluded(g, lon, lat, lows, pmean)
            seg_types = _runs(np.where(occ[:-1] & occ[1:], 'o', np.array(seg_types)), seg, OCC_PART_KM)
        i = 0
        while i < len(seg_types):
            j = i
            while j + 1 < len(seg_types) and seg_types[j + 1] == seg_types[i]:
                j += 1
            plon, plat = (lon[i:j + 2], lat[i:j + 2])
            pnx, pny = (n_x[i:j + 2], n_y[i:j + 2])
            dxs = np.diff(plon) * np.cos(np.radians(plat[:-1]))
            dys = np.diff(plat)
            left = np.nansum(-dys * pnx[:-1] + dxs * pny[:-1])
            c = math.cos(math.radians(float(plat.mean())))
            keep_i = simplify(np.c_[plon * c * 111.2, plat * 111.2], SIMPLIFY_KM)
            plon, plat = (plon[keep_i], plat[keep_i])
            if left < 0:
                plon, plat = (plon[::-1], plat[::-1])
            fronts.append((str(seg_types[i]), list(zip(plon.tolist(), plat.tolist()))))
            i = j + 1
    return {'fronts': fronts, 'centres': cs}

def merge(results):
    out = {'fronts': [], 'centres': [], 'states': []}
    for r in results:
        if r:
            out['fronts'] += r['fronts']
            out['centres'] += r['centres']
            if r.get('state') is not None:
                out['states'].append(r['state'])
    return out

def encode(res):
    f = []
    for kind, pts in res['fronts']:
        q = [(round(lon / COORD), round(lat / COORD)) for lon, lat in pts]
        dedup = [q[0]] + [b for a, b in zip(q, q[1:]) if b != a]
        if len(dedup) < 2:
            continue
        flat = [kind, dedup[0][0], dedup[0][1]]
        for a, b in zip(dedup, dedup[1:]):
            flat += [b[0] - a[0], b[1] - a[1]]
        f.append(flat)
    p = [[kind, round(lon / COORD), round(lat / COORD), round(hpa)] for kind, lon, lat, hpa in res['centres']]
    return {'f': f, 'p': p}

def decode(enc):
    fronts = []
    for line in enc.get('f', []):
        kind, x, y, rest = (line[0], line[1], line[2], line[3:])
        pts = [(x * COORD, y * COORD)]
        for i in range(0, len(rest) - 1, 2):
            x, y = (x + rest[i], y + rest[i + 1])
            pts.append((x * COORD, y * COORD))
        fronts.append((kind, pts))
    return {'fronts': fronts, 'centres': [(k, x * COORD, y * COORD, h) for k, x, y, h in enc.get('p', [])]}
CNN_FILE = 'fronts_cnn.json'
ML_STEP = 0.25
ML_P0 = 0.25
ML_LINE_P = 0.25
ML_SMOOTH_KM = 35.0
ML_MIN_KM = 200.0
ML_JOIN_KM = 350.0
ML_PART_KM = 400.0
ML_LAT = (20.0, 72.0)
ML_TYPE_W = (1.0, 1.5, 1.0, 1.0)
DUP_KM = 250.0
DUP_TURN = 35.0
DUP_RATIO = 1.3
DUP_MIN_KM = 200.0
CROSS_TURN = 0.0
CROSS_NEAR_KM = 50.0
LINE_SMOOTH_KM = 150.0
HAIRPIN = 150.0
HAIRPIN_KM = 150.0
HIGH_HAIRPIN = 100.0
HIGH_M = 1500.0
HIGH_LINE_P = 0.35
HIGH_MIN_KM = 300.0
CURL = 0.4
CURL_KM = 800.0
SEA_P0 = 0.15
SEA_LINE_P = 0.18
SEA_M = 2.0
BLEND_H = 2
TYPE_H = 6
PREV_H = 3
KEEP_KM = 120.0
KEEP_P0 = 0.2
KEEP_LINE_P = 0.22
TYPE_HOLD = 1.3
_CNN = {}
ALAT = (20.0, 72.0)
FEATURES = ['G8', 'T8', 'Gs', 'Ts', 'Gt', 'Tt', 'Gd', 'G8b', 'T8b', 'Gsb', 'Tsb', 'vo10', 'dv10', 'vo8', 'dv8', 'lap', 'pnn', 'shear', 'alat', 'zs', 'under', 'x10', 'x8', 'adv10', 'adv8', 'spd10', 'G8c', 'T8c', 'Gsc', 'vo10b', 'dv10b', 'dp']

def _weights():
    if 'w' not in _CNN:
        import json
        import os
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), CNN_FILE)) as f:
                d = json.load(f)
            if d.get('names') != FEATURES:
                raise ValueError('features differ')
            _CNN['w'] = {k: np.asarray(v, np.float32) if k[0] in 'Cc' or k in ('mu', 'sd') else v for k, v in d.items()}
        except (OSError, ValueError):
            _CNN['w'] = None
    return _CNN['w']

def _zone(g, a, km):
    th = smooth(a, *g.sigma(km))
    gx, gy = g.grad(th)
    G = np.hypot(gx, gy)
    with np.errstate(invalid='ignore', divide='ignore'):
        nx, ny = (gx / G, gy / G)
    Gx, Gy = g.grad(G)
    return (G * 100000.0, -(Gx * nx + Gy * ny) * 10000000000.0, nx, ny, gx, gy)

def _kin(g, u, v, km):
    us, vs = (smooth(u, *g.sigma(km)), smooth(v, *g.sigma(km)))
    ux, uy = g.grad(us)
    vx, vy = g.grad(vs)
    return ((vx - uy) * 100000.0, (ux + vy) * 100000.0, us, vs)

def _quarter(lats, lons, arrays):
    step = abs(lats[1] - lats[0])
    k = max(1, int(round(ML_STEP / step)))
    lat = lats[:len(lats) // k * k:k] + (lats[1] - lats[0]) * (k - 1) / 2
    lon = lons[:len(lons) // k * k:k] + (lons[1] - lons[0]) * (k - 1) / 2
    out = [block_mean(np.asarray(a, np.float64), k) for a in arrays]
    if abs(abs(lat[1] - lat[0]) - ML_STEP) < 1e-06:
        return (lat, lon, out)
    nlat = lat[0] + np.sign(lat[1] - lat[0]) * ML_STEP * np.arange(int(abs(lat[-1] - lat[0]) / ML_STEP) + 1)
    nlon = lon[0] + ML_STEP * np.arange(int((lon[-1] - lon[0]) / ML_STEP) + 1)
    fy = (nlat - lat[0]) / (lat[1] - lat[0])
    fx = (nlon - lon[0]) / (lon[1] - lon[0])
    yy, xx = np.meshgrid(fy, fx, indexing='ij')
    return (nlat, nlon, [ndimage.map_coordinates(a, [yy, xx], order=1, mode='nearest') for a in out])

def features(lats, lons, t850, td850, u850, v850, ps, mslp, t2, td2, u10, v10, zs):
    under = ~(np.asarray(ps) > LEVEL + UNDERGROUND_HPA)
    tw8 = np.where(under, np.nan, theta_w(t850, td850, LEVEL))
    tws = theta_w(t2, td2, ps)
    ths = (np.asarray(t2, np.float64) + 273.15) * (1000.0 / np.asarray(ps, np.float64)) ** 0.2854
    u8, v8 = (np.where(under, np.nan, u850), np.where(under, np.nan, v850))
    lat, lon, (tw8, u8, v8, tws, ths, td2, mslp, u10, v10, zs, under) = _quarter(np.asarray(lats, np.float64), np.asarray(lons, np.float64), (tw8, u8, v8, tws, ths, td2, mslp, u10, v10, zs, under.astype(np.float64)))
    g = Grid(lat, lon)
    under = under > 0.5
    G8, T8, n8x, n8y, e8x, e8y = _zone(g, tw8, 90)
    Gs, Ts, nsx, nsy, esx, esy = _zone(g, tws, 90)
    Gt, Tt = _zone(g, ths, 90)[:2]
    Gd = _zone(g, td2, 90)[0]
    G8b, T8b = _zone(g, tw8, 180)[:2]
    Gsb, Tsb = _zone(g, tws, 180)[:2]
    vo10, dv10, u10s, v10s = _kin(g, u10, v10, 100)
    vo8, dv8, u8s, v8s = _kin(g, u8, v8, 100)
    pm = smooth(mslp, *g.sigma(100))
    px, py = g.grad(pm)
    pxx, pyy = (g.grad(px)[0], g.grad(py)[1])
    with np.errstate(invalid='ignore'):
        Px, Py = g.grad(px * nsx + py * nsy)
        pnn = (Px * nsx + Py * nsy) * 10000000000.0
        ax, ay = g.grad(-u10s * nsy + v10s * nsx)
        shear = (ax * nsx + ay * nsy) * 100000.0
        x10, x8 = (u10s * nsx + v10s * nsy, u8s * n8x + v8s * n8y)
        adv10, adv8 = (-(u10s * esx + v10s * esy) * 10800, -(u8s * e8x + v8s * e8y) * 10800)
    G8c, T8c = _zone(g, tw8, 300)[:2]
    Gsc = _zone(g, tws, 300)[0]
    vo10b, dv10b = _kin(g, u10s, v10s, 200)[:2]
    alat = np.repeat(np.clip(np.abs(g.lats), *ALAT)[:, None], len(g.lons), 1)
    F = [G8, T8, Gs, Ts, Gt, Tt, Gd, G8b, T8b, Gsb, Tsb, vo10, dv10, vo8, dv8, (pxx + pyy) * 10000000000.0, pnn, shear, alat, zs / 1000.0, under, x10, x8, adv10, adv8, np.hypot(u10s, v10s), G8c, T8c, Gsc, vo10b, dv10b, np.hypot(px, py) * 100000.0]
    X = np.stack([np.nan_to_num(np.asarray(a, np.float64), nan=0.0, posinf=0.0, neginf=0.0) for a in F]).astype(np.float32)
    return (X, g, (nsx, nsy))

def _conv(h, W, d):
    k = W.shape[0]
    c, ny, nx = h.shape
    if k == 1:
        return (W[0, 0].T @ h.reshape(c, -1)).reshape(-1, ny, nx)
    r = d * (k // 2)
    wp = nx + 2 * r
    hp = np.pad(h, ((0, 0), (r, r), (r, r))).reshape(c, -1)
    n = ny * wp - 2 * r
    out = np.zeros((W.shape[3], ny * wp), np.float32)
    for i in range(k):
        for j in range(k):
            off = i * d * wp + j * d
            out[:, :n] += W[i, j].T @ hp[:, off:off + n]
    return out.reshape(-1, ny, wp)[:, :, :nx]

def probabilities(W, X):
    h = (X - W['mu'][:, None, None]) / W['sd'][:, None, None]
    dils = [int(d) for d in W['dils']]
    n = len(dils) + 1
    h = np.maximum(_conv(h, W['C0'], 1) + W['c0'][:, None, None], 0)
    for i, d in enumerate(dils, 1):
        h = h + np.maximum(_conv(h, W[f'C{i}'], d) + W[f'c{i}'][:, None, None], 0)
    lg = _conv(h, W[f'C{n}'], 1) + W[f'c{n}'][:, None, None]
    e = np.exp(lg - lg.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)

def _prev_maps(g, prev):
    mask = np.zeros((len(g.lats), len(g.lons)), bool)
    tmap = np.zeros(mask.shape, np.int8)
    for kind, pts in prev:
        p = np.asarray(pts, np.float64)
        if len(p) < 2:
            continue
        seg = km_between(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1])
        n = np.maximum(1, (seg // 10).astype(int))
        lon = np.concatenate([np.linspace(a, b, k, endpoint=False) for a, b, k in zip(p[:-1, 0], p[1:, 0], n)] + [p[-1:, 0]])
        lat = np.concatenate([np.linspace(a, b, k, endpoint=False) for a, b, k in zip(p[:-1, 1], p[1:, 1], n)] + [p[-1:, 1]])
        fy, fx = g.cell(lon, lat)
        iy, ix = (np.rint(fy).astype(int), np.rint(fx).astype(int))
        ok = (iy >= 0) & (iy < mask.shape[0]) & (ix >= 0) & (ix < mask.shape[1])
        mask[iy[ok], ix[ok]] = True
        tmap[iy[ok], ix[ok]] = 'cwso'.index(kind)
    if not mask.any():
        return (None, None)
    dist, (ny_i, nx_i) = ndimage.distance_transform_edt(~mask, sampling=(g.dy / 1000, float(np.median(g.dx)) / 1000), return_indices=True)
    return (dist <= KEEP_KM, tmap[ny_i, nx_i])

def ml_lines(g, prob, nx_, ny_, prev=None, zs=None):
    near, ptype = _prev_maps(g, prev) if prev else (None, None)
    lat_ok = (np.abs(g.lats) >= ML_LAT[0]) & (np.abs(g.lats) <= ML_LAT[1])
    pa = smooth(np.where(lat_ok[:, None], 1 - prob[0], 0.0), *g.sigma(ML_SMOOTH_KM))
    px, py = g.grad(pa)
    pxx, pxy = g.grad(px)
    pyx, pyy = g.grad(py)
    pxy = (pxy + pyx) / 2
    lam = (pxx + pyy) / 2 - np.sqrt(((pxx - pyy) / 2) ** 2 + pxy ** 2)
    ex, ey = (pxy, lam - pxx)
    flat = np.abs(ex) + np.abs(ey) < 1e-30
    ex, ey = (np.where(flat, lam - pyy, ex), np.where(flat, pxy, ey))
    nrm = np.hypot(ex, ey) + 1e-30
    sgn = np.where(np.nan_to_num(ex * nx_ + ey * ny_) < 0, -1.0, 1.0) / nrm
    across = (px * ex + py * ey) * sgn
    p0 = ML_P0 if near is None else np.where(near, KEEP_P0, ML_P0)
    sea = None
    if zs is not None and SEA_P0 > 0:
        sea = np.nan_to_num(zs, nan=0.0) < SEA_M
        p0 = np.where(sea, np.minimum(p0, SEA_P0), p0)
    keep = (pa > p0) & (lam < 0)
    cls = [smooth(prob[i], *g.sigma(50.0)) * ML_TYPE_W[i - 1] for i in range(1, 5)]
    raw = [np.array(g.lonlat(np.array([p[0] for p in ln]), np.array([p[1] for p in ln]))).T for ln in zero_lines(across, keep)]
    out = []
    cands = []
    joined = join_lines(raw, km=ML_JOIN_KM)
    if HAIRPIN > 0:

        def high(ll):
            if zs is None:
                return None
            fy, fx = g.cell(ll[:, 0], ll[:, 1])
            return np.nan_to_num(sample(zs, fy, fx)) > HIGH_M
        joined = [q for ll in joined for q in _hairpins(ll, high(ll))]
    joined = [np.c_[_unloop(ll[:, 0], ll[:, 1])] for ll in joined]
    for ll in joined:
        lon, lat = (ll[:, 0], ll[:, 1])
        L = km_between(lon[:-1], lat[:-1], lon[1:], lat[1:]).sum()
        if L < ML_MIN_KM:
            continue
        fy, fx = g.cell(lon, lat)
        cont = near is not None and float(np.mean(sample(near.astype(np.float64), fy, fx) > 0.5)) >= 0.5
        pmean = float(np.nanmean(sample(pa, fy, fx)))
        need = KEEP_LINE_P if cont else ML_LINE_P
        if sea is not None and float(np.mean(sample(sea.astype(np.float64), fy, fx) > 0.5)) > 0.5:
            need = min(need, SEA_LINE_P)
        if pmean < need:
            continue
        if CURL > 0 and L < CURL_KM and (km_between(lon[0], lat[0], lon[-1], lat[-1]) < CURL * L):
            continue
        if zs is not None and (HIGH_LINE_P > 0 or HIGH_MIN_KM > 0) and (float(np.nanmean(sample(zs, fy, fx) > HIGH_M)) > 0.5):
            if pmean < HIGH_LINE_P or L < HIGH_MIN_KM:
                continue
        cands.append((pmean * L, lon, lat, pmean))
    if DUP_KM > 0:
        cands = _undouble(cands)
    for _, lon, lat, _pm in cands:
        if LINE_SMOOTH_KM > 0:
            lon, lat = _unloop(*_smooth_line(lon, lat, LINE_SMOOTH_KM))
        elif len(lon) >= 5:
            lon = np.r_[lon[0], np.convolve(lon, np.ones(3) / 3, 'valid'), lon[-1]]
            lat = np.r_[lat[0], np.convolve(lat, np.ones(3) / 3, 'valid'), lat[-1]]
        fy, fx = g.cell(lon, lat)
        seg = km_between(lon[:-1], lat[:-1], lon[1:], lat[1:])
        pc = np.stack([sample(c, fy, fx) for c in cls])
        pm = np.nan_to_num((pc[:, :-1] + pc[:, 1:]) / 2, nan=-1.0)
        ti = pm.argmax(0)
        if near is not None:
            my, mx = ((fy[:-1] + fy[1:]) / 2, (fx[:-1] + fx[1:]) / 2)
            iy = np.clip(np.rint(my).astype(int), 0, near.shape[0] - 1)
            ix = np.clip(np.rint(mx).astype(int), 0, near.shape[1] - 1)
            was = ptype[iy, ix].astype(int)
            k = np.arange(len(ti))
            hold = near[iy, ix] & (pm[ti, k] < TYPE_HOLD * pm[was, k])
            ti = np.where(hold, was, ti)
        types = np.array(list('cwso'))[ti]
        seg_types = _runs(types, seg, ML_PART_KM)
        n_x, n_y = (sample(nx_, fy, fx), sample(ny_, fy, fx))
        i = 0
        while i < len(seg_types):
            j = i
            while j + 1 < len(seg_types) and seg_types[j + 1] == seg_types[i]:
                j += 1
            plon, plat = (lon[i:j + 2], lat[i:j + 2])
            dxs = np.diff(plon) * np.cos(np.radians(plat[:-1]))
            left = np.nansum(-np.diff(plat) * n_x[i:j + 1] + dxs * n_y[i:j + 1])
            c = math.cos(math.radians(float(plat.mean())))
            k = simplify(np.c_[plon * c * 111.2, plat * 111.2], SIMPLIFY_KM)
            plon, plat = (plon[k], plat[k])
            if left < 0:
                plon, plat = (plon[::-1], plat[::-1])
            out.append((str(seg_types[i]), list(zip(plon.tolist(), plat.tolist()))))
            i = j + 1
    return out

def _dense(lon, lat, km=25.0):
    seg = km_between(lon[:-1], lat[:-1], lon[1:], lat[1:])
    P, H = ([], [])
    for i, s in enumerate(seg):
        n = max(1, int(s // km))
        h = math.atan2(lat[i + 1] - lat[i], (lon[i + 1] - lon[i]) * math.cos(math.radians(lat[i])))
        for j in range(n):
            f = j / n
            P.append((lon[i] + (lon[i + 1] - lon[i]) * f, lat[i] + (lat[i + 1] - lat[i]) * f))
            H.append(h)
    return (np.array(P), np.array(H))

def _undouble(cands):
    kept, keptP, keptH, keptM = ([], [], [], [])

    def runs(P, cut, end, min_km):
        Q = np.r_[P, [end]]
        out, i = ([], 0)
        while i < len(P):
            if cut[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(P) and (not cut[j + 1]):
                j += 1
            q = Q[i:j + 2]
            if len(q) >= 2 and km_between(q[:-1, 0], q[:-1, 1], q[1:, 0], q[1:, 1]).sum() >= min_km:
                out.append((q[:, 0], q[:, 1]))
            i = j + 1
        return out
    for c in sorted(cands, key=lambda c: -c[0]):
        P, H = _dense(c[1], c[2])
        pieces = [(c[1], c[2])]
        if keptP and len(P):
            KP, KH, KM = (np.concatenate(keptP), np.concatenate(keptH), np.concatenate(keptM))
            d = km_between(P[:, None, 0], P[:, None, 1], KP[None, :, 0], KP[None, :, 1])
            dd = np.abs((KH[None, :] - H[:, None] + np.pi / 2) % np.pi - np.pi / 2)
            dup = ((d < DUP_KM) & (dd < math.radians(DUP_TURN)) & (c[3] < DUP_RATIO * KM[None, :])).any(1)
            if dup.any():
                pieces = runs(P, dup, [c[1][-1], c[2][-1]], DUP_MIN_KM)
            out = []
            for lon, lat in pieces:
                Pp, cross = _crossing(lon, lat, KP, KH)
                out += [(lon, lat)] if not cross.any() else runs(Pp, cross, [lon[-1], lat[-1]], ML_MIN_KM)
            pieces = out
        for lon, lat in pieces:
            PP, HH = _dense(lon, lat)
            kept.append((c[0], lon, lat, c[3]))
            keptP.append(PP)
            keptH.append(HH)
            keptM.append(np.full(len(HH), c[3]))
    return kept

def _crossing(lon, lat, KP, KH):
    P, H = _dense(lon, lat)
    if not len(P):
        return (P, np.zeros(0, bool))
    d = km_between(P[:, None, 0], P[:, None, 1], KP[None, :, 0], KP[None, :, 1])
    dd = np.abs((KH[None, :] - H[:, None] + np.pi / 2) % np.pi - np.pi / 2)
    return (P, ((d < CROSS_NEAR_KM) & (dd >= math.radians(CROSS_TURN))).any(1))

def _smooth_line(lon, lat, km, step=20.0):
    d = np.r_[0, np.cumsum(km_between(lon[:-1], lat[:-1], lon[1:], lat[1:]))]
    if d[-1] < 2 * step:
        return (lon, lat)
    t = np.linspace(0, d[-1], max(3, int(d[-1] // step) + 1))
    x, y = (np.interp(t, d, lon), np.interp(t, d, lat))
    sig = min(km, d[-1] / 4) / (t[1] - t[0])
    n = int(math.ceil(3 * sig))

    def pad(a):
        m = min(n, len(a) - 1)
        head, tail = (2 * a[0] - a[1:m + 1][::-1], 2 * a[-1] - a[-m - 1:-1][::-1])
        return np.r_[np.full(n - m, head[0]), head, a, tail, np.full(n - m, tail[-1])]
    xs = ndimage.gaussian_filter1d(pad(x), sig, mode='nearest')[n:n + len(x)]
    ys = ndimage.gaussian_filter1d(pad(y), sig, mode='nearest')[n:n + len(y)]
    return (xs, ys)

def _unloop(lon, lat):
    while len(lon) >= 4:
        x, y = (lon * np.cos(np.radians(lat)), lat)
        ax, ay, dx, dy = (x[:-1], y[:-1], np.diff(x), np.diff(y))
        den = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
        ex, ey = (ax[None, :] - ax[:, None], ay[None, :] - ay[:, None])
        with np.errstate(invalid='ignore', divide='ignore'):
            t = (ex * dy[None, :] - ey * dx[None, :]) / den
            u = (ex * dy[:, None] - ey * dx[:, None]) / den
        hit = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        hit &= np.triu(np.ones_like(hit), 2)
        if not hit.any():
            break
        i, j = np.argwhere(hit)[0]
        f = t[i, j]
        lon = np.r_[lon[:i + 1], lon[i] + f * (lon[i + 1] - lon[i]), lon[j + 1:]]
        lat = np.r_[lat[:i + 1], lat[i] + f * (lat[i + 1] - lat[i]), lat[j + 1:]]
    return (lon, lat)

def _hairpins(ll, high=None):
    if len(ll) < 4:
        return [ll]
    lon, lat = (ll[:, 0], ll[:, 1])
    d = np.r_[0, np.cumsum(km_between(lon[:-1], lat[:-1], lon[1:], lat[1:]))]
    h = np.arctan2(np.diff(lat), np.diff(lon) * np.cos(np.radians(lat[:-1])))
    cuts, last = ([], 0)
    for i in range(1, len(ll) - 1):
        j = np.searchsorted(d, d[i] + HAIRPIN_KM)
        k = np.searchsorted(d, d[i] - HAIRPIN_KM)
        if j >= len(h) or k < 0 or i - last < 3:
            continue
        turn = abs((h[min(j, len(h) - 1)] - h[max(k, 0)] + np.pi) % (2 * np.pi) - np.pi)
        if turn > np.radians(HIGH_HAIRPIN if high is not None and high[i] else HAIRPIN):
            cuts.append(i)
            last = i
    out, a = ([], 0)
    for c in cuts:
        out.append(ll[a:c + 1])
        a = c
    out.append(ll[a:])
    return [q for q in out if len(q) >= 2]

def detect_ml(lats, lons, t850, td850, u850, v850, ps, mslp, t2, td2, u10, v10, zs):
    W = _weights()
    if W is None:
        return None
    lats, lons = (np.asarray(lats, np.float64), np.asarray(lons, np.float64))
    if abs(lats[-1] - lats[0]) < MIN_SPAN_DEG or abs(lons[-1] - lons[0]) < MIN_SPAN_DEG:
        return {'fronts': [], 'centres': []}
    step = abs(lats[1] - lats[0])
    k = max(1, int(round(STEP / step)))
    pm = block_mean(np.asarray(mslp, np.float64), k)
    g0 = Grid(lats[:pm.shape[0] * k:k] + (lats[1] - lats[0]) * (k - 1) / 2, lons[:pm.shape[1] * k:k] + (lons[1] - lons[0]) * (k - 1) / 2)
    cs = centres(g0, pm)
    south = float(np.mean(lats)) < 0
    fields = [t850, td850, u850, v850, ps, mslp, t2, td2, u10, v10, zs]
    if south:
        fields = [np.asarray(a)[::-1] for a in fields]
        fields[3], fields[9] = (-fields[3], -fields[9])
        lats = -lats[::-1]
    X, g, (nx_, ny_) = features(lats, lons, *fields)
    prob = probabilities(W, X)
    lines = _unmirror(ml_lines(g, prob, nx_, ny_, zs=X[FEATURES.index('zs')] * 1000.0), south)
    return {'fronts': lines, 'centres': cs, 'state': {'g': g, 'p': prob, 'n': (nx_, ny_), 'south': south, 'z': X[FEATURES.index('zs')] * 1000.0}}

def _unmirror(lines, south):
    return [(kind, [(x, -y) for x, y in pts[::-1]]) for kind, pts in lines] if south else lines

def pack_states(states):
    import io
    arrs = {}
    for k, st in enumerate(states):
        g = st['g']
        arrs[f'g{k}'] = np.array([g.lats[0], g.lats[1] - g.lats[0], len(g.lats), g.lons[0], g.lons[1] - g.lons[0], len(g.lons), float(st['south'])])
        arrs[f'p{k}'] = np.rint(np.clip(st['p'], 0, 1) * 250).astype(np.uint8)
        arrs[f'n{k}'] = np.rint(np.nan_to_num(np.stack(st['n'])) * 127).astype(np.int8)
        if st.get('z') is not None:
            arrs[f'z{k}'] = np.rint(np.clip(np.nan_to_num(st['z']), 0, 5100) / 20).astype(np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return np.frombuffer(buf.getvalue(), np.uint8)

def unpack_states(b):
    import io
    out = []
    with np.load(io.BytesIO(np.asarray(b, np.uint8).tobytes())) as z:
        for k in range(sum((1 for f in z.files if f.startswith('g')))):
            lat0, dlat, ny, lon0, dlon, nx, south = z[f'g{k}']
            g = Grid(lat0 + dlat * np.arange(int(ny)), lon0 + dlon * np.arange(int(nx)))
            n = z[f'n{k}'].astype(np.float64) / 127
            out.append({'g': g, 'p': z[f'p{k}'].astype(np.float32) / 250, 'n': (n[0], n[1]), 'south': bool(south), 'z': z[f'z{k}'].astype(np.float32) * 20 if f'z{k}' in z.files else None, 'key': (round(float(lat0), 4), round(float(dlat), 4), int(ny), round(float(lon0), 4), round(float(dlon), 4), int(nx))})
    return out

def day_fronts(hours, load):
    cache, prev, out = ({}, {}, {})

    def states(t):
        if t not in cache:
            cache[t] = load(t)
        return cache[t]
    span = max(BLEND_H, TYPE_H)
    times = [t for t, _ in hours]
    for i, (t, enc) in enumerate(hours):
        for old in [x for x in cache if abs((x - t).total_seconds()) > span * 3600]:
            del cache[old]
        mine = states(t)
        if not mine:
            out[t] = enc
            continue
        near_h = [(u, (u - t).total_seconds() / 3600) for u in times[max(0, i - 3 * span):i + 3 * span + 1]]
        lines = []
        for st in mine:
            acc, wsum = (np.zeros_like(st['p'][:1]), 0.0)
            typ, tsum = (np.zeros_like(st['p'][1:]), 0.0)
            for u, dh in near_h:
                if abs(dh) > span:
                    continue
                other = next((o for o in states(u) or [] if o['key'] == st['key']), None)
                if other is None:
                    continue
                if abs(dh) <= BLEND_H:
                    w = BLEND_H + 1 - abs(dh)
                    acc += w * other['p'][:1]
                    wsum += w
                if abs(dh) <= TYPE_H:
                    typ += other['p'][1:]
                    tsum += 1
            p = np.concatenate([acc / wsum, typ / tsum])
            before = prev.get(st['key'])
            pv = before[1] if before and 0 < (t - before[0]).total_seconds() <= PREV_H * 3600 else None
            ln = ml_lines(st['g'], p, *st['n'], prev=pv, zs=st.get('z'))
            prev[st['key']] = (t, ln)
            lines += _unmirror(ln, st['south'])
        out[t] = {'f': encode({'fronts': lines, 'centres': []})['f'], 'p': enc.get('p', [])}
    return out

def from_part(s, lv, levels, lats, lons):
    levels = list(levels)
    if LEVEL not in levels or lv.get('t') is None or lv.get('td') is None or (lv.get('u') is None):
        return None
    i = levels.index(LEVEL)
    need = ('ps', 'mslp', 't2', 'td2', 'u10', 'v10', 'zs')
    if all((s.get(k) is not None for k in need)):
        try:
            res = detect_ml(lats, lons, lv['t'][i], lv['td'][i], lv['u'][i], lv['v'][i], *(s[k] for k in need))
            if res is not None:
                return res
        except Exception:
            pass
    return detect(lats, lons, lv['t'][i], lv['td'][i], lv['u'][i], lv['v'][i], ps=s.get('ps'), mslp=s.get('mslp'), u10=s.get('u10'), v10=s.get('v10'))
