from __future__ import annotations
import json
import math
import os
import struct
import urllib.request
import zlib
from datetime import date, datetime, timezone
import numpy as np
import common
import report
import track
import train
T3 = {'spacing': 2, 'spacingMin': 1.6, 'spacingMax': 3.1, 'spacingW': 2, 'spacingWeak': 0.525, 'spacingStrong': 0.27, 'minSpacing': 1200, 'maxSpacing': 6000, 'peak': 1.3, 'windward': 0.35, 'flowPeak': 0.25, 'crest': 0.2, 'slideEdge': 0.15, 'hillM': 120, 'minW': 0.5, 'suck': 0.25, 'suckFrom': 500, 'suckFull': 2500, 'suckDepth': 250}
TRIGGER_RANGE = (-0.6, 1.6)
TRIGGER_W = {'hill': 0.15, 'sun': 0.7, 'ground': 0.1, 'cover': 0.1, 'flat': 2}
TRIGGER_FLAT = (30, 200)
LAG = {'entrain': 0.1, 'foot': 0.6}
LEAN_STEPS, RISE_MIN = (24, 0.5)
BARE = 0.5
RING_KM = 1.2
SLOPE_M = 200.0
NEG_KM = 3.0
LOW_AGL = 300
MIN_DEPTH = 300
TILE_Z = 11
DEM = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium'
FEATURES = ['trig', 'st', 'sunslope', 'up', 'flat', 'slope', 'tpi', 'relief', 'wstar', 'zi', 'cu', 'wind', 'solar', 'sunelev', 'elev', 'doy_s', 'doy_c']
PARAMS = dict(max_iter=100, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=30, l2_regularization=1.0, random_state=7)
MIN_ROWS, MIN_DAYS = (200, 10)

def allen_radius(zr: float, zi: float) -> float:
    x = min(max(zr, 0.0), 1.0)
    return max(10.0, 0.102 * x ** (1 / 3) * (1 - 0.25 * x) * zi)

def lift_shape(zr: float, cu: bool=False) -> float:
    if not (zr > 0 and zr <= 1):
        return 0.0
    return (1 - math.exp(-zr / 0.08)) * (1 - (0.25 if cu else 0.55) * zr ** 4)

def cloud_suck(depth, below) -> float:
    if not (depth is not None and depth > T3['suckFrom']) or not below >= 0:
        return 0.0
    return T3['suck'] * min(1.0, (depth - T3['suckFrom']) / (T3['suckFull'] - T3['suckFrom'])) * math.exp(-below / T3['suckDepth'])

def rise_at(t: dict, zr: float) -> float:
    q = max(0.001, min(1.0, zr))
    cu = t.get('cu') is not None
    suck = 1 + cloud_suck(t['cuTop'] - t['cu'], (1 - q) * t['zi']) if cu and t.get('cuTop') is not None else 1
    return T3['peak'] * t['W'] * t.get('m', 1.0) * max(RISE_MIN, lift_shape(q, cu) * suck)

def lean_walk(t: dict, zr: float) -> tuple[float, float]:
    zi = t['zi']
    n = math.ceil(min(1.0, zr) * LEAN_STEPS - 1e-09)
    x = y = 0.0
    q0, vx, vy = (0.0, math.nan, math.nan)
    for k in range(1, n + 1):
        q = min(zr, k / LEAN_STEPS)
        dz = (q - q0) * zi
        if dz > 0:
            ux, uy = (t['u'], t['v'])
            if k == 1:
                vx, vy = (LAG['foot'] * ux, LAG['foot'] * uy)
            L = allen_radius((q0 + q) / 2, zi) * t.get('rs', 1.0) / (2 * LAG['entrain'])
            e = math.exp(-dz / L)
            mean = L / dz * (1 - e)
            mx, my = (ux + (vx - ux) * mean, uy + (vy - uy) * mean)
            w = rise_at(t, (q0 + q) / 2)
            x += mx * dz / w
            y += my * dz / w
            vx, vy = (ux + (vx - ux) * e, uy + (vy - uy) * e)
        q0 = q
    return (x, y)

def spacing_factor(W: float, bubbly: bool=False) -> float:
    dW = W - T3['spacingW']
    return max(T3['spacingMin'], min(T3['spacingMax'], T3['spacing'] - (T3['spacingWeak'] if dW < 0 else T3['spacingStrong']) * dW + (0.3 if bubbly else 0)))

def placed_spacing_km(W: float, zi: float) -> float:
    return max(T3['minSpacing'], min(T3['maxSpacing'], spacing_factor(W) * zi)) / 1000

def sun_position(lat: float, lon: float, t: float) -> dict:
    d = datetime.fromtimestamp(t, timezone.utc)
    doy = (t - datetime(d.year, 1, 1, tzinfo=timezone.utc).timestamp()) / 86400 + 1
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    B = 2 * math.pi * (doy - 81) / 364
    eot = (9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)) / 60
    ha = math.radians(15 * (d.hour + d.minute / 60 + lon / 15 + eot - 12))
    la = math.radians(lat)
    se = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    el = math.asin(max(-1.0, min(1.0, se)))
    az = math.atan2(-math.cos(decl) * math.sin(ha), math.sin(decl) * math.cos(la) - math.cos(decl) * math.sin(la) * math.cos(ha))
    return {'elev': math.degrees(el), 'az': (math.degrees(az) + 360) % 360}

def sun_on_slope(hx: float, hy: float, sun: dict) -> float:
    e, a = (math.radians(max(sun['elev'], 3)), math.radians(sun['az']))
    s = (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))
    l = math.sqrt(hx * hx + hy * hy + 1)
    return max(0.0, (-hx * s[0] - hy * s[1] + s[2]) / l) / s[2]

def flatness(relief) -> float:
    if relief is None or not math.isfinite(relief):
        return 0.0
    return max(0.0, min(1.0, (TRIGGER_FLAT[1] - relief) / (TRIGGER_FLAT[1] - TRIGGER_FLAT[0])))

def trigger_score(st, sun_slope, src, up, peak, crest, wc=0.0, flat=0.0, heat=1.0, edge=0.0) -> float:
    W = TRIGGER_W
    gr = (1 + W['flat'] * flat) * (W['ground'] * src + W['cover'] * wc + T3['slideEdge'] * edge)
    t = W['hill'] * st + W['sun'] * sun_slope + (heat * gr if gr > 0 else gr) + T3['windward'] * up * (1 - flat) + T3['flowPeak'] * peak + T3['crest'] * crest
    return max(0.0, min(1.0, (t - TRIGGER_RANGE[0]) / (TRIGGER_RANGE[1] - TRIGGER_RANGE[0])))

def _unfilter(raw: bytes, w: int, h: int, bpp: int) -> bytes:
    stride = w * bpp
    out = bytearray(stride * h)
    prev = bytearray(stride)
    pos = 0
    for y in range(h):
        f = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if f == 1:
            for i in range(bpp, stride):
                line[i] = line[i] + line[i - bpp] & 255
        elif f == 2:
            for i in range(stride):
                line[i] = line[i] + prev[i] & 255
        elif f == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = line[i] + (a + prev[i] >> 1) & 255
        elif f == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = (abs(p - a), abs(p - b), abs(p - c))
                pr = a if pa <= pb and pa <= pc else b if pb <= pc else c
                line[i] = line[i] + pr & 255
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return bytes(out)

def decode_png(data: bytes) -> np.ndarray:
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG')
    pos, idat, w = (8, [], None)
    while pos < len(data):
        n, typ = struct.unpack('>I4s', data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if typ == b'IHDR':
            w, h, depth, ctype, _, _, inter = struct.unpack('>IIBBBBB', body)
            if depth != 8 or ctype not in (2, 6) or inter:
                raise ValueError('unsupported PNG')
            ch = 3 if ctype == 2 else 4
        elif typ == b'IDAT':
            idat.append(body)
        elif typ == b'IEND':
            break
    raw = zlib.decompress(b''.join(idat))
    return np.frombuffer(_unfilter(raw, w, h, ch), dtype=np.uint8).reshape(h, w, ch)

def terrarium(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    return a[:, :, 0] * 256 + a[:, :, 1] + a[:, :, 2] / 256 - 32768

class Terrain:

    def __init__(self, cache_dir: str | None=None, ground=None, z: int=TILE_Z, fetch=None):
        self.cache, self.ground, self.z, self.tiles, self.failed = (cache_dir, ground, z, {}, 0)
        self._fetch = fetch or self._http
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _http(url: str) -> bytes | None:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'terrain-check/1.0'}), timeout=30) as r:
                return r.read()
        except Exception:
            return None

    def tile(self, tx: int, ty: int):
        k = (tx, ty)
        if k in self.tiles:
            return self.tiles[k]
        n = 1 << self.z
        tx %= n
        arr = None
        fn = os.path.join(self.cache, f'{self.z}_{tx}_{ty}.png') if self.cache else None
        data = None
        if fn and os.path.exists(fn):
            with open(fn, 'rb') as f:
                data = f.read()
        else:
            data = self._fetch(f'{DEM}/{self.z}/{tx}/{ty}.png')
            if data and fn:
                with open(fn, 'wb') as f:
                    f.write(data)
        if data:
            try:
                arr = terrarium(decode_png(data))
            except (ValueError, zlib.error, struct.error):
                arr = None
        if arr is None:
            self.failed += 1
        self.tiles[k] = arr
        return arr

    def elev(self, lon: float, lat: float):
        if self.ground:
            return float(self.ground(lon, lat))
        n = 1 << self.z
        x = (lon + 180) / 360 * n
        la = math.radians(lat)
        y = (1 - math.log(math.tan(la) + 1 / math.cos(la)) / math.pi) / 2 * n
        tx, ty = (int(math.floor(x)), int(math.floor(y)))
        if not 0 <= ty < n:
            return None
        arr = self.tile(tx, ty)
        if arr is None:
            return None
        px, py = (min(255, int((x - tx) * 256)), min(255, int((y - ty) * 256)))
        return float(arr[py, px])

def terrain_features(T: Terrain, lon: float, lat: float, sun: dict, u: float, v: float) -> dict | None:
    kx = 111320 * math.cos(math.radians(lat))
    g = T.elev(lon, lat)
    if g is None:
        return None
    d = SLOPE_M
    e = [T.elev(lon + d / kx, lat), T.elev(lon - d / kx, lat), T.elev(lon, lat + d / 110540), T.elev(lon, lat - d / 110540)]
    if any((x is None for x in e)):
        return None
    hx, hy = ((e[0] - e[1]) / (2 * d), (e[2] - e[3]) / (2 * d))
    ring, hi, lo = (0.0, g, g)
    for q in range(8):
        r = T.elev(lon + RING_KM * 1000 * math.cos(q * math.pi / 4) / kx, lat + RING_KM * 1000 * math.sin(q * math.pi / 4) / 110540)
        if r is None:
            return None
        ring += r
        hi, lo = (max(hi, r), min(lo, r))
    tpi = g - ring / 8
    sun_f = sun_on_slope(hx, hy, sun) if sun['elev'] > 1 else 1.0
    st = max(-1.0, min(1.0, tpi / T3['hillM']))
    ss = max(-1.0, min(1.0, sun_f - 1))
    sun_slope = ss * (0.3 + 0.7 * BARE)
    flat = flatness(hi - lo)
    up = max(0.0, min(1.0, (u * hx + v * hy) / 1)) * (1 - flat)
    return {'g': g, 'slope': math.hypot(hx, hy), 'tpi': tpi, 'relief': hi - lo, 'flat': flat, 'st': st, 'sunslope': sun_slope, 'up': up, 'trig': trigger_score(st, sun_slope, 0.0, up, 0.0, 0.0, 0.0, flat, 1.0, 0.0)}

class Grids:

    def __init__(self, root: str | None):
        self.root, self.boxes = (root, {})

    def box(self, day: str, model: str, bi: int):
        k = (day, model, bi)
        if k in self.boxes:
            return self.boxes[k]
        B = None
        if self.root:
            base = os.path.join(self.root, day, f'{model}-box{bi}')
            if os.path.exists(os.path.join(base, 'latest.json')):
                import day as dayrun
                if not os.path.exists(os.path.join(base, 'fields.json')):
                    dayrun.dump(base)
                info = common.read_json(os.path.join(base, 'fields.json'))
                g = info['grid']
                nf = len(info['fields'])
                arr = np.fromfile(os.path.join(base, 'fields.bin'), dtype=np.float32).reshape(len(info['stamps']), nf, g['ny'], g['nx'])
                stamps = [int(datetime.strptime(s, '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp()) for s in info['stamps']]
                B = {'g': g, 'fi': {n: i for i, n in enumerate(info['fields'])}, 'arr': arr, 'stamps': stamps}
        self.boxes[k] = B
        return B

    def at(self, day: str, model: str, bi: int, name: str, lon: float, lat: float, t: float):
        B = self.box(day, model, bi)
        if not B or name not in B['fi']:
            return None
        g = B['g']
        ix, iy = (round((lon - g['west']) / g['step']), round((g['north'] - lat) / g['step']))
        if not (0 <= ix < g['nx'] and 0 <= iy < g['ny']):
            return None
        h = min(range(len(B['stamps'])), key=lambda k: abs(B['stamps'][k] - t))
        if abs(B['stamps'][h] - t) > 3600:
            return None
        v = float(B['arr'][h, B['fi'][name], iy, ix])
        return None if math.isnan(v) else v

def _env(own: dict, grids: Grids, day: str, model: str, bi, lon, lat, t) -> dict | None:
    e = {'u': own.get('u'), 'v': own.get('v'), 'ws': own.get('ws0') if own.get('ws0') is not None else own.get('ws'), 'zi': own.get('zi'), 'cb': own.get('cb'), 'solar': own.get('solar'), 'sunElev': own.get('sunElev'), 'elev': own.get('elev'), 'from': 'probe'}
    if grids and bi is not None:
        ws = grids.at(day, model, bi, 'wstar', lon, lat, t)
        if ws is not None:
            e.update({'ws': ws, 'zi': grids.at(day, model, bi, 'zi', lon, lat, t) or e['zi'], 'cb': grids.at(day, model, bi, 'cubase', lon, lat, t), 'u': grids.at(day, model, bi, 'u_blavg', lon, lat, t), 'v': grids.at(day, model, bi, 'v_blavg', lon, lat, t), 'elev': grids.at(day, model, bi, 'elevation', lon, lat, t) or e['elev'], 'from': 'grid'})
    if e['u'] is None or e['v'] is None or (not (e['ws'] or 0) >= T3['minW']) or (not (e['zi'] or 0) > 0):
        return None
    return e

def foot_of(lon: float, lat: float, agl: float, e: dict) -> tuple[float, float, float]:
    D = e['cb'] if e.get('cb') else e['zi']
    zr = max(0.02, min(1.0, agl / D))
    dx, dy = lean_walk({'zi': D, 'W': e['ws'], 'u': e['u'], 'v': e['v'], 'cu': 1 if e.get('cb') else None, 'cuTop': None}, zr)
    kx = 111320 * math.cos(math.radians(lat))
    return (lon - dx / kx, lat - dy / 110540, zr)

def points(days: list[dict], model: str, grids: Grids | None=None, bad: dict | None=None) -> list[dict]:
    out = []
    for d in days:
        if bad and d['date'] in bad:
            continue
        for f in d['flights']:
            tr = f.get('track') or []
            bi = (f.get('model_' + model) or {}).get('box')
            cs = [c for c in track._climbs_sorted(f) if not c.get('engon')]
            for i, c in enumerate(cs):
                own = (c.get('models') or {}).get(model) or {}
                p = track.pos_at(f, c['_t0']) if tr else {'lon': c.get('lon'), 'lat': c.get('lat'), 'agl': c.get('agl0')}
                if p['lon'] is None or own is None:
                    continue
                agl = p.get('agl') if p.get('agl') is not None else c.get('agl0')
                if agl is None:
                    continue
                e = _env(own, grids, d['date'], model, bi, p['lon'], p['lat'], c['_t0'])
                if not e or (e['cb'] or e['zi']) < MIN_DEPTH:
                    continue
                flon, flat, zr = foot_of(p['lon'], p['lat'], max(50.0, agl), e)
                out.append({'label': 1, 'lon': flon, 'lat': flat, 'zr': zr, 't': c['_t0'], 'date': d['date'], 'region': f['region'], 'fid': f['id'], 'pilot': f.get('pilot'), 'e': e, 'rate': c.get('rate'), 'agl': agl, 'met': (p['lon'], p['lat'])})
                if tr and i + 1 < len(cs):
                    t0, t1 = (c['_t1'], cs[i + 1]['_t0'])
                    last = None
                    for q in tr:
                        if not t0 < q[0] < t1 or q[4] is None:
                            continue
                        here = (q[1], q[2])
                        if last is not None and track.dist_km(last, here) < NEG_KM:
                            continue
                        last = here
                        agl_q = q[3] - q[4]
                        if agl_q < LOW_AGL:
                            continue
                        glon, glat, zr_q = foot_of(q[1], q[2], agl_q, e)
                        out.append({'label': 0, 'lon': glon, 'lat': glat, 'zr': zr_q, 't': q[0], 'date': d['date'], 'region': f['region'], 'fid': f['id'], 'pilot': f.get('pilot'), 'e': e, 'rate': None, 'agl': agl_q, 'met': here})
    return out

def featurize(pts: list[dict], T: Terrain) -> list[dict]:
    out = []
    for p in pts:
        sun = sun_position(p['lat'], p['lon'], p['t'])
        tf = terrain_features(T, p['lon'], p['lat'], sun, p['e']['u'], p['e']['v'])
        if not tf:
            continue
        e = p['e']
        doy = train.doy_of(p['date'])
        out.append({**p, **tf, 'wstar': e['ws'], 'zi': e['cb'] if e.get('cb') else e['zi'], 'cu': 1.0 if e.get('cb') else 0.0, 'wind': math.hypot(e['u'], e['v']), 'solar': e.get('solar'), 'sunelev': sun['elev'], 'elev': tf['g'], 'doy_s': math.sin(2 * math.pi * doy / 365.25), 'doy_c': math.cos(2 * math.pi * doy / 365.25)})
    return out

def top_decile(score, y) -> dict:
    score, y = (np.asarray(score, dtype=float), np.asarray(y, dtype=float))
    if len(y) < 10:
        return {'n': int(len(y)), 'share': None, 'base': None, 'lift': None}
    k = max(1, len(y) // 10)
    top = y[np.argsort(-score, kind='stable')[:k]]
    base = float(y.mean())
    return {'n': int(len(y)), 'share': float(top.mean()), 'base': base, 'lift': float(top.mean() / base) if base > 0 else None}

def _f(x, n: int) -> str:
    return '–' if x is None else f'{x:.{n}f}'

def score_table(F: list[dict], score_key: str, groups: dict) -> list[dict]:
    from layer1 import auc
    tab = []
    for name, sel in groups.items():
        rs = [r for r in F if sel(r)]
        y = [r['label'] for r in rs]
        if len(rs) < 30 or sum(y) < 5 or len(rs) - sum(y) < 5:
            continue
        s = [r[score_key] for r in rs]
        tab.append({'group': name, 'n': len(rs), 'climbs': int(sum(y)), 'auc': auc(s, y), 'top': top_decile(s, y)})
    return tab

def groups_of(F: list[dict]) -> dict:
    gs = {'all': lambda r: True, 'flat': lambda r: r['flat'] > 0.5, 'hills/mountains': lambda r: r['flat'] <= 0.5, 'cumulus': lambda r: r['cu'] > 0, 'blue': lambda r: r['cu'] == 0}
    for a, b in report.WIND_BANDS:
        gs[f"wind {a}–{(b if b < 99 else '+')}"] = (lambda a, b: lambda r: a <= r['wind'] < b)(a, b)
    for a, b in report.HOURS:
        gs[f'solar {a}–{b}'] = (lambda a, b: lambda r: r['solar'] is not None and a <= r['solar'] < b)(a, b)
    for reg in sorted({r['region'] for r in F}):
        gs[f'region {reg}'] = (lambda reg: lambda r: r['region'] == reg)(reg)
    return gs

def spacing_rows(pts: list[dict]) -> list[dict]:
    by = {}
    for p in pts:
        if p['label'] == 1:
            by.setdefault(p['fid'], []).append(p)
    out = []
    for fid, ps in by.items():
        ps.sort(key=lambda p: p['t'])
        for a, b in zip(ps, ps[1:]):
            d = track.dist_km((a['lon'], a['lat']), (b['lon'], b['lat']))
            if 0.3 < d < 60 and b['t'] - a['t'] < 3 * 3600:
                e = b['e']
                out.append({'flown_km': d, 'placed_km': placed_spacing_km(e['ws'], e['cb'] if e.get('cb') else e['zi']), 'region': b['region'], 'rate': b['rate'], 'ws': e['ws']})
    return out

def fit(F: list[dict]) -> dict | None:
    from layer1 import auc
    if len(F) < MIN_ROWS or len({r['date'] for r in F}) < MIN_DAYS:
        return None
    y = np.array([r['label'] for r in F], dtype=float)
    if y.sum() < 20 or (1 - y).sum() < 20:
        return None
    X = np.array([[np.nan if r.get(k) is None else float(r[k]) for k in FEATURES] for r in F], dtype=float)
    keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).mean() >= 0.02]
    names = [FEATURES[j] for j in keep]
    X = X[:, keep]
    g = np.array([r['date'] for r in F])
    p_ml, _ = train.crossval(X, y, g, g, None, PARAMS, train.classifier)
    m = train.classifier(PARAMS).fit(X, y)
    e = train.export(m, names)
    e['kind'] = 'logit'
    e['version'] = date.today().isoformat()
    chk = train.export_check(e, m, X)
    trig = np.array([r['trig'] for r in F])
    return {'export': e, 'rows': int(len(y)), 'days': int(len(set(g))), 'check': chk, 'auc_ml': auc(p_ml, y), 'auc_trig': auc(trig, y), 'top_ml': top_decile(p_ml, y), 'top_trig': top_decile(trig, y), 'train_dropped': train.crossval.dropped}

def run(D: dict, out: str, grids: str | None=None, terrain: Terrain | None=None) -> dict:
    md = [f"Each climb traced down the wind to its trigger point (the 3D thermals' lean at the height the glider met it); negatives along the glides between climbs (every {NEG_KM:g} km, over {LOW_AGL} m up), traced the same way: ground flown over without circling. The site's Thermal Triggers score (its terrain part: hill, sunlit slope, windward slope, flat land; no land cover or airflow here) scored against them: AUC (0.5 = chance) and the top decile's hit rate against the base rate (lift). Then a tree model P(climb | cell, hour) on the same terrain and the layer-1 air, cross-validated by day. Layer 3 sees the planner on that air, not where the thermals were: nothing here is counted again there.", '']
    res = {'scores': None, 'cv': None, 'points': 0, 'rows': 0}
    if D['days'] is None:
        md += ["No day files (exported rows only): the climbs' positions and the tracks aren't in them. Nothing scored.", '']
        return _finish(out, md, res)
    model = D['model']
    G = Grids(grids) if grids else None
    pts = points(D['days'], model, G, D['bad'])
    if not pts:
        md += ["No climbs with the forecast's wind and layer at them (the day files hold no sampled u, v, W*, zi per climb).", '']
        return _finish(out, md, res)
    T = terrain or Terrain(os.environ.get('CALIB_TERRAIN_CACHE') or os.path.join(out, 'terrain'))
    F = featurize(pts, T)
    res['points'] = res['rows'] = len(F)
    n_pos = sum((r['label'] for r in F))
    src = {'grid': sum((1 for p in pts if p['e']['from'] == 'grid')), 'probe': sum((1 for p in pts if p['e']['from'] == 'probe'))}
    md += [f"Points: {len(F)} with terrain ({n_pos} climbs, {len(F) - n_pos} glide points) of {len(pts)} traced; terrain tiles that couldn't be read: {T.failed}; the air from the saved grids at {src['grid']}, from the climb's own sampled values at {src['probe']}.", '']
    if not F:
        md += ["No terrain (the tiles couldn't be fetched and no cache): nothing scored.", '']
        return _finish(out, md, res)
    tab = score_table(F, 'trig', groups_of(F))
    res['scores'] = tab
    md += ["## The site's trigger score against the climbs flown", '', "| Group | points | climbs | AUC | top decile: climbs' share, base rate, lift |", '|---|---|---|---|---|']
    for r in tab:
        t = r['top']
        md.append(f"| {r['group']} | {r['n']} | {r['climbs']} | {_f(r['auc'], 3)} | {t['share']:.2f}, {t['base']:.2f}, {_f(t['lift'], 2)} |")
    sp = spacing_rows(pts)
    if sp:
        q = lambda xs, p: float(np.percentile(xs, p))
        fl, pl = ([r['flown_km'] for r in sp], [r['placed_km'] for r in sp])
        ratio = [math.log(r['flown_km'] / r['placed_km']) for r in sp]
        md += ['', "## The 3D placement's spacing and strength", '', f"Climbs in a row (same flight, under 3 h apart): {len(sp)}; distance flown between them (km): median {q(fl, 50):.1f} (IQR {q(fl, 25):.1f}–{q(fl, 75):.1f}); the placement's spacing there: median {q(pl, 50):.1f}; flown / placed: median {math.exp(q(ratio, 50)):.2f} (IQR {math.exp(q(ratio, 25)):.2f}–{math.exp(q(ratio, 75)):.2f}).", '']
        st_ = [math.log(r['rate'] / (T3['peak'] * r['ws'])) for r in sp if (r.get('rate') or 0) > 0 and (r.get('ws') or 0) > 0]
        if st_:
            md += [f"Climb flown / the placed core's peak (CORE {T3['peak']} × the model's W*): median {math.exp(q(st_, 50)):.2f} (IQR {math.exp(q(st_, 25)):.2f}–{math.exp(q(st_, 75)):.2f}); layer 1's W* correction is what moves this.", '']
        res['spacing'] = {'n': len(sp), 'flown_med_km': q(fl, 50), 'placed_med_km': q(pl, 50), 'ratio_med': math.exp(q(ratio, 50))}
    f = fit(F)
    md += ['## P(climb | cell, hour): the trees against the score', '']
    if not f:
        md += [f"Too few points ({len(F)} on {len({r['date'] for r in F})} days) for a model.", '']
    else:
        with open(os.path.join(out, 'triggers.json'), 'w') as fh:
            json.dump(f['export'], fh, separators=(',', ':'))
        md += [f"{f['rows']} points, {f['days']} days; each fold's training {f['train_dropped']:.0%} smaller for the test rows' days. AUC: the site's score {f['auc_trig']:.3f}, the trees {f['auc_ml']:.3f} (out of fold); top decile lift: {f['top_trig']['lift']:.2f} against {f['top_ml']['lift']:.2f}.", '', train.splits_md(f['export']), f"Export check (max |trees − sklearn|): {f['check']:.2e}. triggers.json: features {', '.join(f['export']['features'])}.", '']
        res['cv'] = {k: v for k, v in f.items() if k != 'export'}
    return _finish(out, md, res)

def _finish(out: str, md: list[str], res: dict) -> dict:
    with open(os.path.join(out, 'report-place.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    with open(os.path.join(out, 'layer2.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    return res
