from __future__ import annotations
import hashlib
import itertools
import json
import math
import re
import warnings
from datetime import datetime, timezone
import numpy as np
import aircorr
import build
import sky
SCHEMA = 'blend-bundle/1'
ID = re.compile('^bl1-\\d{8}-[0-9a-f]{8}$')
KINDS = ('add', 'log', 'logit')
FIELDS = ('t2', 'td2', 'wind', 'cubase', 'cu')
PRESENT_BY = {'t2': 't2', 'td2': 'td2', 'wind': 'u_sfc', 'cu': 'wstar', 'cubase': 'cubase'}
INPUTS = ('t2', 'td2', 'u_sfc', 'v_sfc', 'cubase', 'cucover', 'cuclass', 'wstar', 'zi', 'depth', 'elevation')
REGIMES = ('blue', 'cumulus', 'overdeveloping', 'stratiform', 'frontal', 'wind')
STRONG_WIND = 10.0
WAVE_CLASS = 14
_REGIME_OF_CLASS = {sky.NONE: 0, sky.BLUE: 0, 2: 1, 3: 1, 4: 1, sky.CB: 2, sky.SHOWERS: 2, sky.NS: 4, sky.FOG: 3, sky.ST: 3, sky.SC: 3, sky.AS: 4, sky.AC: 0, sky.CS: 0, WAVE_CLASS: 5, sky.ACC: 0}
BASE_ONLY_CLASSES = (WAVE_CLASS, sky.ACC)
NIGHT_ELEV = build.SUN_GATE[0]
CU_EPS = 0.02
CU_AT = 0.5
CB_MIN_AGL = 50.0
LAPSE = {'t2': 0.0065, 'td2': 0.0018}
SCALE_FIELDS = ('t2', 'td2', 'wind')
RELIEF_KM_DEG = 0.05
FEATURES = {'x', 'lead_h', 'run_age_h', 'solar_h', 'sun_elev', 'lat', 'relief5km', 'tpi', 'dz_terrain', 'regime', 'n_models', 'zi', 'depth', 'wstar', 'cuclass', 'ws', 'wd', 'src', 'replay'} | {f'spread_{f}' for f in FIELDS}
WENC = {'bw': (0, 1, None), 'blead': (0, 254, None)}
SPREAD_ENC = {'t2': (0, 20, None), 'td2': (0, 20, None), 'wind': (0, 20, None), 'cu': (0, 1, None), 'cubase': (0, 2000, None)}

class BundleError(ValueError):
    pass

class Refused(ValueError):

    def __init__(self, msg, model=None):
        super().__init__(msg)
        self.model = model

def check_sha(data: bytes, sha256: str) -> None:
    got = hashlib.sha256(data).hexdigest()
    if got != str(sha256).lower():
        raise BundleError(f"bundle sha256 {got[:12]} is not the pointer's {str(sha256)[:12]}")

def load(src, sha256: str | None=None) -> dict:
    data = src if isinstance(src, (bytes, bytearray)) else open(src, 'rb').read()
    if sha256 is not None:
        check_sha(bytes(data), sha256)
    try:
        b = json.loads(data)
    except ValueError as e:
        raise BundleError(f'bundle is not JSON: {e}') from None
    return prepare(validate(b))

def subset_key(names) -> str:
    return '+'.join(names)

def subsets(members: list) -> list:
    return [list(c) for r in range(1, len(members) + 1) for c in itertools.combinations(members, r)]

def _edges(e, what):
    if not isinstance(e, list) or len(e) < 2 or (not all((isinstance(x, (int, float)) for x in e))) or any((b <= a for a, b in zip(e, e[1:]))):
        raise BundleError(f'{what}: bin edges must rise')
    return e

def _trees(t, what):
    if not isinstance(t, dict) or not isinstance(t.get('trees'), list) or (not isinstance(t.get('features'), list)):
        raise BundleError(f'{what}: not exported trees')
    bad = set(t['features']) - FEATURES
    if bad:
        raise BundleError(f'{what}: unknown features {sorted(bad)}')
    nf = len(t['features'])
    for tree in t['trees']:
        n = len(tree)
        if not isinstance(tree, list) or not n:
            raise BundleError(f'{what}: an empty tree')
        for i, nd in enumerate(tree):
            if len(nd) == 1:
                if not math.isfinite(float(nd[0])):
                    raise BundleError(f'{what}: a leaf that is not a number')
                continue
            if len(nd) != 5 or not isinstance(nd[0], int) or (not 0 <= nd[0] < nf) or (not math.isfinite(float(nd[1]))) or (not all((isinstance(c, int) and i < c < n for c in (nd[3], nd[4])))):
                raise BundleError(f'{what}: a node out of range')
    if not math.isfinite(float(t.get('base', math.nan))):
        raise BundleError(f'{what}: no base')
    return t

def n_cells(bins: dict) -> int:
    return (len(bins['lead']) - 1) * len(bins['solar']) * len(bins['regime']) * n_relief(bins)

def n_relief(bins: dict) -> int:
    r = bins.get('relief')
    return len(r) - 1 if r else 1

def _clamp_ok(c, kind, what):
    if isinstance(c, bool) or c is None:
        raise BundleError(f'{what}: no clamp')
    if isinstance(c, (int, float)):
        if not (math.isfinite(c) and c > 0):
            raise BundleError(f'{what}: clamp {c!r}')
        return
    if kind == 'logit' or not isinstance(c, list) or len(c) != 2 or (not all((isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) for x in c))):
        raise BundleError(f'{what}: clamp {c!r}')
    mid = 1.0 if kind == 'log' else 0.0
    if not c[0] < mid < c[1] or (kind == 'log' and c[0] <= 0):
        raise BundleError(f'{what}: clamp {c!r} must bracket {mid:g}')

def validate(b: dict) -> dict:
    return _guard(_validate, b)

def _guard(fn, b):
    try:
        return fn(b)
    except BundleError:
        raise
    except (TypeError, KeyError, AttributeError, IndexError, ValueError, OverflowError) as e:
        raise BundleError(f'bundle malformed: {e!r}') from None

def _validate(b: dict) -> dict:
    if not isinstance(b, dict) or b.get('schema') != SCHEMA:
        raise BundleError(f'not a {SCHEMA}')
    if not ID.match(str(b.get('id', ''))):
        raise BundleError(f"bundle id {b.get('id')!r}")
    regions = b.get('regions')
    if not isinstance(regions, dict) or not regions:
        raise BundleError('no regions')
    for rn, r in regions.items():
        mem = r.get('members')
        if not isinstance(mem, list) or not mem or len(set(mem)) != len(mem) or (len(mem) > 8):
            raise BundleError(f'{rn}: members')
        if r.get('base') not in mem:
            raise BundleError(f'{rn}: the base is not a member')
        if not float(r.get('grid_step', 0)) > 0:
            raise BundleError(f'{rn}: grid_step')
        age = r.get('max_age_h', {})
        if not isinstance(age, dict) or any((m not in mem or isinstance(a, bool) or (not isinstance(a, (int, float))) or (not math.isfinite(a)) or (a < 0) for m, a in age.items())):
            raise BundleError(f'{rn}: max_age_h')
        groups = r.get('groups') or {}
        fields = r.get('fields')
        if not isinstance(fields, dict) or not fields:
            raise BundleError(f'{rn}: no fields')
        for gn, gf in groups.items():
            if not isinstance(gf, list) or not gf or any((x not in FIELDS for x in gf)):
                raise BundleError(f'{rn}: group {gn}')
        for fn, f in fields.items():
            w = f'{rn}.{fn}'
            if fn not in FIELDS:
                raise BundleError(f'{w}: not a field the blend knows')
            if f.get('kind') not in KINDS:
                raise BundleError(f"{w}: kind {f.get('kind')!r}")
            if f.get('move') not in (None, 'lapse') or (f.get('move') == 'lapse' and fn not in LAPSE):
                raise BundleError(f"{w}: move {f.get('move')!r}")
            if f.get('scale') not in (None, True, False) or (f.get('scale') and fn not in SCALE_FIELDS):
                raise BundleError(f"{w}: scale {f.get('scale')!r}")
            if f.get('gate') is not None:
                raise BundleError(f'{w}: a gate (version 1 weighs by its table only)')
            if not isinstance(f.get('pass'), bool):
                raise BundleError(f'{w}: pass')
            _clamp_ok(f.get('clamp'), f['kind'], w)
            for m, t in (f.get('debias') or {}).items():
                if m not in mem:
                    raise BundleError(f'{w}: debias for {m}, not a member')
                _trees(t, f'{w}.debias.{m}')
            if f.get('spread') is not None:
                _trees(f['spread'], f'{w}.spread')
            wt = f.get('weights')
            if not isinstance(wt, dict):
                raise BundleError(f'{w}: no weights')
            bins = wt.get('bins') or {}
            _edges(bins.get('lead'), f'{w}.lead')
            sol = bins.get('solar')
            if not isinstance(sol, list) or not sol or sol[0] != 'night' or (len(sol) < 2):
                raise BundleError(f'{w}: solar bins (night, then daylight bands)')
            _edges([x[0] for x in sol[1:]] + [sol[-1][1]], f'{w}.solar')
            if any((len(x) != 2 for x in sol[1:])) or any((a[1] != c[0] for a, c in zip(sol[1:], sol[2:]))):
                raise BundleError(f'{w}: solar bands must touch')
            if sorted(bins.get('regime') or []) != sorted(REGIMES):
                raise BundleError(f'{w}: regime bins')
            if bins.get('relief') is not None:
                _edges(bins['relief'], f'{w}.relief')
            nc = n_cells(bins)
            table = wt.get('table') or {}
            for s in subsets(mem):
                rows = table.get(subset_key(s))
                if not isinstance(rows, list) or len(rows) != nc:
                    raise BundleError(f'{w}: no row for {subset_key(s)}')
                a = np.asarray(rows, np.float64)
                if a.shape != (nc, len(s)) or not np.isfinite(a).all() or (a < 0).any() or (np.abs(a.sum(1) - 1) > 1e-06).any():
                    raise BundleError(f"{w}: {subset_key(s)}'s rows are not weights (>= 0, summing to 1)")
            extra = set(table) - {subset_key(s) for s in subsets(mem)}
            if extra:
                raise BundleError(f'{w}: rows for {sorted(extra)}')
            if not isinstance(wt.get('n_min'), int) or wt['n_min'] < 0:
                raise BundleError(f'{w}: n_min')
            for m in mem:
                n = (wt.get('n') or {}).get(m)
                if not isinstance(n, list) or len(n) != nc:
                    raise BundleError(f'{w}: no row counts for {m}')
                if not all((isinstance(x, int) and (not isinstance(x, bool)) and (x >= 0) for x in n)):
                    raise BundleError(f"{w}: {m}'s row counts are not counts")
    return b

def prepare(b: dict) -> dict:
    return _guard(_prepare, b)

def _prepare(b: dict) -> dict:
    for r in b['regions'].values():
        mem = r['members']
        K = len(mem)
        for f in r['fields'].values():
            for t in list((f.get('debias') or {}).values()) + ([f['spread']] if f.get('spread') else []):
                t['_flat'] = aircorr._flatten_all(t['trees'])
            wt = f['weights']
            nc = n_cells(wt['bins'])
            T = np.zeros((1 << K, nc, K), np.float64)
            for s in subsets(mem):
                rows = np.asarray(wt['table'][subset_key(s)], np.float64)
                rows = rows / rows.sum(1, keepdims=True)
                bits = sum((1 << mem.index(m) for m in s))
                for j, m in enumerate(s):
                    T[bits, :, mem.index(m)] = rows[:, j]
            wt['_T'] = T
            wt['_ok'] = np.array([[n >= wt['n_min'] for n in wt['n'][m]] for m in mem], bool)
    return b

def weight_region(base_region, step: float) -> tuple:
    w, s, e, n = base_region[:4]
    return (math.floor(w / step) * step, math.floor(s / step) * step, math.ceil(e / step) * step, math.ceil(n / step) * step, step)

def grid2d(region):
    lons, lats = build.target_grid(region)
    return (np.repeat(np.asarray(lats)[:, None], len(lons), 1), np.repeat(np.asarray(lons)[None, :], len(lats), 0))

def regrid_codes(codes: dict, rg) -> dict:
    out = {}
    near = rg.nearest if isinstance(rg, Bilinear) else rg
    for k, a in codes.items():
        a = np.asarray(a, np.float32)
        v = near(a)
        if isinstance(rg, Bilinear) and k in SMOOTH_CODES:
            b = rg(a)
            v = np.where(np.isfinite(v) & (v != 255) & np.isfinite(b), b, v)
        out[k] = np.where(np.isfinite(v), np.round(v), 255).astype(np.uint8)
    return out
SMOOTH_CODES = ('t2', 'td2', 'u_sfc', 'v_sfc', 'cubase', 'cucover', 'wstar', 'zi', 'depth', 'elevation')

class Bilinear:

    def __init__(self, lat, lon, region):
        lat, lon = (np.asarray(lat, np.float64), np.asarray(lon, np.float64))
        self.near = build.Regridder(np.repeat(lat[:, None], len(lon), 1), np.repeat(lon[None, :], len(lat), 0), region[:5])
        tlon, tlat = build.target_grid(region[:5])
        gy, gx = np.meshgrid(tlat, tlon, indexing='ij')
        self.shape = gy.shape
        fy = (gy - lat[0]) / (lat[1] - lat[0]) if len(lat) > 1 else np.zeros(gy.shape)
        fx = (gx - lon[0]) / (lon[1] - lon[0]) if len(lon) > 1 else np.zeros(gx.shape)
        ny, nx = (len(lat), len(lon))
        self.off = (fy < -0.5) | (fy > ny - 0.5) | (fx < -0.5) | (fx > nx - 0.5)
        y0 = np.clip(np.floor(fy), 0, max(ny - 2, 0)).astype(np.int64)
        x0 = np.clip(np.floor(fx), 0, max(nx - 2, 0)).astype(np.int64)
        ty, tx = (np.clip(fy - y0, 0, 1), np.clip(fx - x0, 0, 1))
        y1, x1 = (np.minimum(y0 + 1, ny - 1), np.minimum(x0 + 1, nx - 1))
        self.corners = [(y0, x0, (1 - ty) * (1 - tx)), (y0, x1, (1 - ty) * tx), (y1, x0, ty * (1 - tx)), (y1, x1, ty * tx)]

    def nearest(self, field):
        return self.near(field)

    def __call__(self, field):
        f = np.asarray(field, np.float32)
        f = np.where(f == 255, np.nan, f)
        num, den = (np.zeros(self.shape), np.zeros(self.shape))
        for y, x, w in self.corners:
            v = f[y, x]
            ok = np.isfinite(v)
            num += np.where(ok, w * v, 0.0)
            den += np.where(ok, w, 0.0)
        out = np.where(den > 1e-09, num / np.maximum(den, 1e-09), np.nan)
        return np.where(self.off, np.nan, out).astype(np.float32)

def regridder(lat, lon, region, bilinear: bool=False):
    lat, lon = (np.asarray(lat, np.float64), np.asarray(lon, np.float64))
    if bilinear and lat.ndim == 1 and (len(lat) > 1) and (len(lon) > 1):
        return Bilinear(lat, lon, region)
    if lat.ndim == 1:
        lat, lon = (np.repeat(lat[:, None], len(lon), 1), np.repeat(lon[None, :], len(lat), 0))
    return build.Regridder(lat, lon, region[:5])

def member(model: str, run: datetime, codes: dict, lat, lon, base_region, wregion, aircorr_on: bool) -> dict:
    keep = {k: v for k, v in codes.items() if k in INPUTS or k == 'cusky'}
    return {'model': model, 'run': run, 'aircorr': bool(aircorr_on), 'codes': regrid_codes(keep, regridder(lat, lon, base_region, bilinear=True)), 'wcodes': regrid_codes(keep, regridder(lat, lon, wregion))}

def height_moved(f: str, x, z_from, z_to):
    if f not in LAPSE:
        return x
    x = np.asarray(x, np.float64)
    dz = np.asarray(z_to, np.float64) - np.asarray(z_from, np.float64)
    return np.where(np.isfinite(dz), x - LAPSE[f] * dz, x)

def _dec(codes, k):
    if k not in codes:
        return None
    return sky.decode(codes[k], *build.ENC[k])

def _hours(a: datetime, b: datetime) -> float:
    return (_utc(a) - _utc(b)).total_seconds() / 3600

def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)

def regime_of(codes: dict) -> np.ndarray:
    ws = _dec(codes, 'wstar')
    shape = ws.shape
    out = np.full(shape, -1, np.int16)
    has = np.isfinite(ws)
    if 'cuclass' in codes:
        cls = sky.classes_of(codes['cuclass'])
        for c, r in _REGIME_OF_CLASS.items():
            out[has & (cls == c)] = r
        nocls = has & ~np.isfinite(cls)
    else:
        cls, nocls = (np.full(shape, np.nan), has)
    cb = _dec(codes, 'cubase')
    cu = np.isfinite(cb) if cb is not None else np.zeros(shape, bool)
    out[nocls] = np.where(cu[nocls], 1, 0)
    u, v = (_dec(codes, 'u_sfc'), _dec(codes, 'v_sfc'))
    windy = np.hypot(u, v) >= STRONG_WIND if u is not None and v is not None else np.zeros(shape, bool)
    out[has & (windy | (cls == WAVE_CLASS))] = REGIMES.index('wind')
    return out

def _values(codes: dict, field: str, elev=None) -> np.ndarray | None:
    if field == 'wind':
        u, v = (_dec(codes, 'u_sfc'), _dec(codes, 'v_sfc'))
        return None if u is None or v is None else np.hypot(u, v)
    if field == 'cu':
        ws = _dec(codes, 'wstar')
        if ws is None or 'cubase' not in codes:
            return None
        return np.where(np.isfinite(ws), (codes['cubase'] != 255).astype(np.float64), np.nan)
    if field == 'cubase':
        cb = _dec(codes, 'cubase')
        if cb is None:
            return None
        e = elev if elev is not None else _dec(codes, 'elevation')
        return cb + (np.nan_to_num(e, nan=0.0) if e is not None else 0.0)
    return _dec(codes, field)

def _bin(x, edges):
    return np.clip(np.searchsorted(np.asarray(edges, np.float64), x, side='right') - 1, 0, len(edges) - 2)

def sun_band(solar_h, sun_elev, bands) -> np.ndarray:
    starts = [b[0] for b in bands[1:]] + [bands[-1][1]]
    day = 1 + _bin(solar_h, starts)
    return np.where(sun_elev < NIGHT_ELEV, 0, day)

def available(members: dict, issued: datetime, region: dict) -> list:
    age = region.get('max_age_h') or {}
    out = []
    for m in region['members']:
        if m in members and (m not in age or _hours(issued, members[m]['run']) <= float(age[m])):
            out.append(m)
    return out

def conditions(members: dict, valid: datetime, issued: datetime, wregion, region: dict) -> dict:
    lat, lon = grid2d(wregion)
    valid = _utc(valid)
    utc_h = valid.hour + valid.minute / 60
    avail = available(members, issued, region)
    reg = np.stack([regime_of(members[m]['wcodes']) for m in avail]) if avail else np.full((1,) + lat.shape, -1)
    counts = np.stack([(reg == i).sum(0) for i in range(len(REGIMES))])
    n = (reg >= 0).sum(0)
    cond = {'lat': lat, 'lon': lon, 'lead_h': _hours(valid, issued), 'solar_h': ((utc_h + lon / 15) % 24 + 24) % 24, 'sun_elev': build.sun_elevation(valid.replace(tzinfo=None), lat, lon), 'n_models': n.astype(np.float64), 'regime': np.where(n > 0, counts.argmax(0), -1), 'members': avail}
    for f in FIELDS:
        vals = [_values(members[m]['wcodes'], f) for m in avail]
        vals = [v for v in vals if v is not None]
        st = np.stack(vals) if len(vals) >= 2 else None
        if st is None:
            cond[f'spread_{f}'] = np.full(lat.shape, np.nan)
            continue
        k = np.isfinite(st).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cond[f'spread_{f}'] = np.where(k >= 2, np.nanstd(st, 0), np.nan)
    return cond

def cell_index(cond: dict, bins: dict) -> np.ndarray:
    li = int(_bin(np.array([cond['lead_h']]), bins['lead'])[0])
    si = sun_band(cond['solar_h'], cond['sun_elev'], bins['solar'])
    order = [bins['regime'].index(r) for r in REGIMES]
    ri = np.where(cond['regime'] >= 0, np.take(order, np.maximum(cond['regime'], 0)), 0)
    c = (li * len(bins['solar']) + si) * len(bins['regime']) + ri
    nr = n_relief(bins)
    if nr > 1:
        rel = np.asarray(cond.get('relief5km', np.zeros(np.shape(ri))), np.float64)
        rb = np.where(np.isfinite(rel), _bin(np.nan_to_num(rel), bins['relief']), 0)
        c = c * nr + rb
    return np.where(cond['regime'] >= 0, c, -1)

def _node_weights(bits, cells, wt):
    ok = wt['_ok']
    K = ok.shape[0]
    c = np.maximum(cells, 0)
    okbits = sum((ok[i][c].astype(np.int64) << i for i in range(K)))
    eff = np.where(cells >= 0, bits & okbits, 0)
    return wt['_T'][eff, c]

def weights(members, cond: dict, bundle: dict, field: str, region: str='conus') -> dict:
    r = bundle['regions'][region]
    wt = r['fields'][field]['weights']
    mem = r['members']
    avail = set(cond['members'])
    cells = cell_index(cond, wt['bins'])
    bits = np.zeros(cells.shape, np.int64)
    for i, m in enumerate(mem):
        if m in avail and m in members:
            v = _values(members[m]['wcodes'], field) if isinstance(members[m], dict) and 'wcodes' in members[m] else None
            if v is not None:
                bits |= np.isfinite(v).astype(np.int64) << i
    W = _node_weights(bits, cells, wt)
    return {m: W[..., i] for i, m in enumerate(mem)}

class _Up:

    def __init__(self, wregion, base_lat, base_lon):
        w, s, e, n, step = wregion
        lons, lats = build.target_grid(wregion)
        ny, nx = (len(lats), len(lons))
        base_lat, base_lon = (np.asarray(base_lat, np.float64).ravel(), np.asarray(base_lon, np.float64).ravel())
        fy = np.clip((n - base_lat) / step, 0, ny - 1)
        fx = np.clip((base_lon - w) / step, 0, nx - 1)
        y0 = np.minimum(np.floor(fy).astype(np.int64), max(ny - 2, 0))
        x0 = np.minimum(np.floor(fx).astype(np.int64), max(nx - 2, 0))
        ty, tx = (fy - y0, fx - x0)
        y1, x1 = (np.minimum(y0 + 1, ny - 1), np.minimum(x0 + 1, nx - 1))
        self.idx = [y0 * nx + x0, y0 * nx + x1, y1 * nx + x0, y1 * nx + x1]
        self.share = [(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx]
        self.near = np.rint(fy).astype(np.int64) * nx + np.rint(fx).astype(np.int64)

    def bilinear(self, node_vals, sel):
        return sum((node_vals[i[sel]] * s[sel][:, None] for i, s in zip(self.idx, self.share)))

    def nearest(self, a):
        return np.asarray(a).ravel()[self.near]

def base_weights(present: dict, cond: dict, bundle: dict, field: str, up: _Up, region: str='conus') -> np.ndarray:
    r = bundle['regions'][region]
    wt = r['fields'][field]['weights']
    mem = r['members']
    K = len(mem)
    shape = next(iter(present.values())).shape if present else up.near.shape
    bits = np.zeros(shape, np.int64)
    for i, m in enumerate(mem):
        if m in present:
            bits |= present[m].astype(np.int64) << i
    cells = cell_index(cond, wt['bins']).ravel()
    out = np.zeros((K,) + shape, np.float64)
    flat = bits.ravel()
    o = out.reshape(K, -1)
    for b in np.unique(flat):
        if b == 0:
            continue
        sel = np.nonzero(flat == b)[0]
        W = _node_weights(np.full(cells.shape, b, np.int64), cells, wt)
        v = up.bilinear(W, sel)
        tot = v.sum(1, keepdims=True)
        v = np.where(tot > 0, v / np.where(tot > 0, tot, 1), 0)
        o[:, sel] = v.T
    return out

def _clamp(e, clamp, kind):
    if clamp is None:
        return e
    lo, hi = clamp if isinstance(clamp, (list, tuple)) else (None, None)
    if kind == 'log':
        if lo is None:
            lo, hi = (math.exp(-float(clamp)), math.exp(float(clamp)))
        return np.clip(np.exp(e), lo, hi)
    if lo is None:
        lo, hi = (-float(clamp), float(clamp))
    return np.clip(e, lo, hi)

def debias(kind: str, x: np.ndarray, e: np.ndarray, clamp=None) -> np.ndarray:
    if kind == 'add':
        return x + _clamp(e, clamp, kind)
    if kind == 'log':
        return x * (_clamp(e, clamp, kind) if clamp is not None else np.exp(e))
    p = np.clip(x, CU_EPS, 1 - CU_EPS)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + _clamp(e, clamp, kind))))

def relief(elev: np.ndarray, step: float) -> np.ndarray:
    return aircorr.relief_grid(elev, step, RELIEF_KM_DEG)

def at_nodes(a: np.ndarray, breg, lat, lon) -> np.ndarray:
    w, s_, e, n, step = breg[:5]
    a = np.asarray(a, np.float64)
    y = np.round((n - np.asarray(lat, np.float64)) / step).astype(np.int64)
    x = np.round((np.asarray(lon, np.float64) - w) / step).astype(np.int64)
    ok = (y >= 0) & (y < a.shape[0]) & (x >= 0) & (x < a.shape[1])
    out = np.full(np.shape(lat), np.nan)
    out[ok] = a[y[ok], x[ok]]
    return out

def ground_mean(elev: np.ndarray, step: float) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    e = np.asarray(elev, np.float64)
    r = int(round(RELIEF_KM_DEG / step))
    if r <= 0:
        return e.copy()
    known = np.isfinite(e)
    size = 2 * r + 1
    s = uniform_filter(np.where(known, e, 0.0), size=size, mode='constant')
    n = uniform_filter(known.astype(np.float64), size=size, mode='constant')
    return np.where(n > 1e-09, s / np.maximum(n, 1e-09), np.nan)

def tpi(elev: np.ndarray, step: float) -> np.ndarray:
    return np.asarray(elev, np.float64) - ground_mean(elev, step)

def feature_arrays(names, x, codes, ctx: dict, cells) -> np.ndarray:
    rows = []
    for k in names:
        if k == 'x':
            a = x
        elif k in ('lead_h', 'run_age_h'):
            a = np.full(x.shape, float(ctx[k]))
        elif k in ('src', 'replay'):
            a = np.zeros(x.shape)
        elif k == 'dz_terrain':
            me = _dec(codes, 'elevation')
            a = ctx['dem'] - me if me is not None else np.zeros(x.shape)
        elif k == 'ws' or k == 'wd':
            u, v = (_dec(codes, 'u_sfc'), _dec(codes, 'v_sfc'))
            if u is None or v is None:
                a = np.full(x.shape, np.nan)
            else:
                a = np.hypot(u, v) if k == 'ws' else (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        elif k == 'cuclass':
            a = sky.classes_of(codes['cuclass']) if 'cuclass' in codes else np.full(x.shape, np.nan)
        elif k in ctx:
            a = ctx[k]
        else:
            a = _dec(codes, k)
            if a is None:
                a = np.full(x.shape, np.nan)
        rows.append(np.asarray(a, np.float64).ravel()[cells])
    return np.vstack(rows) if rows else np.zeros((0, len(cells)))

def debiased(field: str, fdef: dict, m: str, x: np.ndarray, codes: dict, ctx: dict, native: bool=True) -> np.ndarray:
    t = (fdef.get('debias') or {}).get(m)
    out = x.copy()
    if t is None:
        return out
    cells = np.nonzero(np.isfinite(x).ravel())[0]
    if not len(cells):
        return out
    e = aircorr.predict(t, feature_arrays(t['features'], x, codes, ctx, cells), native=native)
    out.ravel()[cells] = debias(fdef['kind'], x.ravel()[cells], e, fdef.get('clamp'))
    return out

def _stamped(h) -> bool:
    return h.get('aircorr') is not False

def blend(base_hour: dict, members: dict, bundle: dict, region: str='conus', native: bool=True, issued: datetime | None=None) -> dict:
    r = bundle['regions'][region]
    base = r['base']
    if _stamped(base_hour):
        raise Refused(f'{base} is air-corrected or unstamped: the blend debiases uncorrected codes only', base)
    refused = [m for m, h in members.items() if m in r['members'] and m != base and _stamped(h)]
    breg = tuple(base_hour['region'][:5])
    wreg = weight_region(breg, float(r['grid_step']))
    q = {k: v.copy() for k, v in base_hour['codes'].items()}
    lat, lon = grid2d(breg)
    up = _Up(wreg, lat, lon)
    bm = {'model': base, 'run': base_hour['run'], 'codes': base_hour['codes'], 'wcodes': regrid_codes({k: v for k, v in base_hour['codes'].items() if k in INPUTS}, regridder(lat, lon, wreg))}
    allm = {**{m: h for m, h in members.items() if m in r['members'] and m != base and (m not in refused)}, base: bm}
    valid, brun = (_utc(base_hour['valid']), _utc(base_hour['run']))
    issued = _utc(issued) if issued is not None else brun
    cond = conditions(allm, valid, issued, wreg, r)
    avail = cond['members']
    elev = _dec(base_hour['codes'], 'elevation')
    dem = np.asarray(base_hour['dem'], np.float64) if base_hour.get('dem') is not None else elev if elev is not None else np.zeros(lat.shape)
    ground = elev if elev is not None else dem
    ctx = {'lat': lat, 'solar_h': up.nearest(cond['solar_h']).reshape(lat.shape), 'sun_elev': build.sun_elevation(valid.replace(tzinfo=None), lat, lon), 'relief5km': relief(dem, breg[4]), 'tpi': tpi(dem, breg[4]), 'dem': dem, 'regime': _nan_neg(up.nearest(cond['regime']).reshape(lat.shape)), 'n_models': up.nearest(cond['n_models']).reshape(lat.shape)}
    for f in FIELDS:
        ctx[f'spread_{f}'] = up.nearest(cond[f'spread_{f}']).reshape(lat.shape)
    cond['relief5km'] = at_nodes(ctx['relief5km'], breg, cond['lat'], cond['lon'])
    inside = base_hour['codes']['wstar'] != 255 if 'wstar' in base_hour['codes'] else np.ones(lat.shape, bool)
    from_base, vals = ({}, {})
    order = [f for f in ('t2', 'td2', 'wind', 'cu', 'cubase') if f in r['fields']]
    for f in order:
        fdef = r['fields'][f]
        if not fdef['pass']:
            from_base[f] = 'failed its test'
            continue
        if fdef.get('off'):
            from_base[f] = 'its source code moved since its training (the guard)'
            continue
        if f == 'cubase' and 'cu' not in vals:
            from_base[f] = 'no blended cumulus presence'
            continue
        try:
            out, W, xt, extra = _field(f, fdef, allm, avail, base, valid, issued, ctx, cond, bundle, region, up, ground, native)
        except Exception as e:
            from_base[f] = f'failed: {e}'
            continue
        if not (inside & (W.sum(0) > 0)).any():
            from_base[f] = 'no member had a weight anywhere'
            continue
        vals[f] = {'out': out, 'w': W, 'x': xt, 'mem': r['members'], 'base': base, **extra}
    _codes_out(q, vals, inside, base_hour, ground, from_base)
    if 'cu' in vals and 'cubase' in from_base:
        from_base['cubase'] += "; the base's height (its zi where it has none), only where the blended cumulus presence has cumulus"
    try:
        _classes(q, vals, allm, r['members'], breg, base_hour['codes'])
    except Exception as e:
        for k in ('cucover', 'cuwide', 'cueff', 'cuclass', 'cusky'):
            if k in q:
                q[k] = base_hour['codes'][k].copy()
        from_base['cuclass'] = f'failed: {e}'
    t2, td2 = (_dec(q, 't2'), _dec(q, 'td2'))
    if t2 is not None and td2 is not None and ('td2' in vals or 't2' in vals):
        over = np.isfinite(t2) & np.isfinite(td2) & (td2 > t2)
        c = build.quantize(t2, *build.ENC['td2'])
        c = np.where((c > 0) & (c < 255) & (sky.decode(c, *build.ENC['td2']) > t2), c - 1, c)
        q['td2'] = np.where(over, c, q['td2']).astype(np.uint8)
        if 'td2' in from_base and over.any():
            from_base['td2'] += '; held under the blended t2 where it was over it'
    wfields, wfailed = weight_fields(vals, cond, allm, bundle, region, up, ctx, native)
    ages = [{'model': m, 'run': f"{_utc(allm[m]['run']):%Y-%m-%dT%HZ}", 'age_h': round(_hours(issued, allm[m]['run']), 2)} for m in avail]
    return {'codes': q, 'fields_blended': [f for f in order if f in vals], 'fields_from_base': from_base, 'refused': refused, 'members': ages, 'weights': wfields, 'weights_failed': wfailed, 'weight_region': wreg, 'issued': issued, 'values': {f: v['out'] for f, v in vals.items()}, 'debiased': {f: v['x'] for f, v in vals.items()}, 'base_weights': {f: v['w'] for f, v in vals.items()}}

def _field(f, fdef, allm, avail, base, valid, issued, ctx, cond, bundle, region, up, ground, native):
    mem = bundle['regions'][region]['members']
    shape = ground.shape
    xs, present, uv, raw = ({}, {}, {}, {})
    for m in avail:
        codes = allm[m]['codes']
        x = _values(codes, f)
        if x is None:
            continue
        raw[m] = (x, _dec(codes, 'u_sfc'), _dec(codes, 'v_sfc')) if f == 'wind' else x
        if fdef.get('move') == 'lapse':
            x = height_moved(f, x, _dec(codes, 'elevation'), ctx['dem'])
        c = {**ctx, 'lead_h': _hours(valid, allm[m]['run']), 'run_age_h': _hours(issued, allm[m]['run'])}
        xt = debiased(f, fdef, m, x, codes, c, native)
        if f == 'cubase':
            xt = np.maximum(xt, ground + CB_MIN_AGL)
        if f == 'wind':
            xt = np.maximum(xt, 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                k = np.where(x > 1e-06, xt / np.where(x > 1e-06, x, 1), 0.0)
            uv[m] = (_dec(codes, 'u_sfc') * k, _dec(codes, 'v_sfc') * k)
        xs[m] = xt
        present[m] = np.isfinite(xt)
    if base not in xs:
        raise ValueError('the base has no values')
    W = base_weights(present, cond, bundle, f, up, region)
    num = np.zeros(shape)
    for i, m in enumerate(mem):
        if m in xs:
            num += W[i] * np.nan_to_num(xs[m])
    some = W.sum(0) > 0
    out = np.where(some, num, xs[base])
    if fdef.get('scale') and f in SCALE_FIELDS:
        out = scale_separated(out, W, raw, base, mem, ground.shape, _step_of(ctx))
    return (out, W, xs, {'uv': uv} if f == 'wind' else {})

def _step_of(ctx) -> float:
    lat = np.asarray(ctx['lat'])
    return float(abs(lat[0, 0] - lat[1, 0])) if lat.ndim == 2 and lat.shape[0] > 1 else 0.03

def box_mean(a: np.ndarray, rs: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    a = np.asarray(a, np.float64)
    if rs <= 0:
        return a.copy()
    ok = np.isfinite(a)
    size = 2 * rs + 1
    s_ = uniform_filter(np.where(ok, a, 0.0), size=size, mode='constant')
    n = uniform_filter(ok.astype(np.float64), size=size, mode='constant')
    return np.where(ok & (n > 1e-09), s_ / np.maximum(n, 1e-09), np.nan)

def scale_separated(out, W, raw: dict, base: str, mem: list, shape, step: float) -> np.ndarray:
    import verify
    rs = verify.smooth_cells(step)

    def departure(r):
        if isinstance(r, tuple):
            sp, u, v = r
            return sp - np.hypot(box_mean(u, rs), box_mean(v, rs))
        return r - box_mean(r, rs)
    anom = {m: departure(raw[m]) for m in raw}
    a_base = anom.get(base)
    if a_base is None:
        return out
    corr = np.zeros(shape)
    for i, m in enumerate(mem):
        if m in anom:
            corr += W[i] * np.nan_to_num(anom[m])
    ss = out + a_base - corr
    if isinstance(raw.get(base), tuple):
        ss = np.maximum(ss, 0.0)
    return np.where(np.isfinite(a_base), ss, out)

def _nan_neg(a):
    return np.where(a >= 0, a, np.nan).astype(np.float64)

def _codes_out(q, vals, inside, base_hour, ground, from_base):
    E = build.ENC
    put = lambda k, a: q.__setitem__(k, np.where(inside & np.isfinite(a), build.quantize(a, *E[k]), q[k]).astype(np.uint8))

    def step(f, keys, fn):
        if f not in vals:
            return
        try:
            fn()
        except Exception as e:
            for k in keys:
                if k in base_hour['codes']:
                    q[k] = base_hour['codes'][k].copy()
            vals.pop(f)
            from_base[f] = f'failed: {e}'
    for f in ('t2', 'td2'):
        step(f, (f,), lambda f=f: put(f, vals[f]['out']))

    def wind():
        d = vals['wind']
        W, xs, sp = (d['w'], d['x'], d['out'])
        uu, vv = (np.zeros(sp.shape), np.zeros(sp.shape))
        for i, m in enumerate(d['mem']):
            if m in xs:
                mu, mv = d['uv'][m]
                uu += W[i] * np.nan_to_num(mu)
                vv += W[i] * np.nan_to_num(mv)
        some = W.sum(0) > 0
        s = np.hypot(uu, vv)
        k = np.where(s > 1e-06, sp / np.where(s > 1e-06, s, 1), 0.0)
        bu, bv = d['uv'][d['base']]
        uu, vv = (np.where(some, uu * k, bu), np.where(some, vv * k, bv))
        d['u'], d['v'] = (uu, vv)
        put('u_sfc', uu)
        put('v_sfc', vv)
    step('wind', ('u_sfc', 'v_sfc'), wind)

    def cumulus():
        p = vals['cu']['out']
        has = p >= CU_AT
        cb_base = _dec(base_hour['codes'], 'cubase')
        if 'cubase' in vals:
            msl = np.where(vals['cubase']['w'].sum(0) > 0, vals['cubase']['out'], np.nan)
            agl = np.maximum(msl - ground, CB_MIN_AGL)
        else:
            agl = cb_base
        zi = _dec(base_hour['codes'], 'zi')
        add = has & ~np.isfinite(agl)
        agl = np.where(add & np.isfinite(zi) & (zi > 0), zi, agl)
        reach = zi + np.maximum(build.CU_OVERSHOOT * zi, build.CU_OVERSHOOT_MIN)
        over = has & np.isfinite(agl) & np.isfinite(zi) & (zi > 0) & (agl > reach)
        agl = np.where(over, reach, agl)
        vals['cu']['held'] = int(over.sum())
        agl = np.where(has, agl, np.nan)
        cb = np.where(inside, build.quantize(agl, *E['cubase']), q['cubase']).astype(np.uint8)
        dq = None
        if 'cudepth' in q:
            d = _dec(base_hour['codes'], 'cudepth')
            d = np.where(np.isfinite(agl), np.where(np.isfinite(d), d, aircorr.CU_DEPTH), np.nan)
            dq = np.where(inside, build.quantize(d, *E['cudepth']), q['cudepth']).astype(np.uint8)
        q['cubase'] = cb
        if dq is not None:
            q['cudepth'] = dq
        vals['cu']['has'], vals['cu']['cubase_agl'] = (has, agl)
    step('cu', ('cubase', 'cudepth'), cumulus)
    if 'cu' not in vals and 'cubase' in vals:
        vals.pop('cubase')
        from_base['cubase'] = 'no blended cumulus presence'

def _classes(q, vals, allm, mem, breg, base_codes):
    ov = sky.cloud_overview(q, build.ENC, float(breg[4]) <= sky.FINE_STEP)
    for k in ('cucover', 'cuwide', 'cueff', 'cuclass', 'cusky'):
        if k in ov and k in q:
            q[k] = ov[k]
    if 'cuclass' not in q:
        return
    if 'cu' in vals:
        has, W = (vals['cu']['has'], vals['cu']['w'])
        best = np.full(has.shape, -1)
        bw = np.zeros(has.shape)
        for i, m in enumerate(mem):
            if m not in allm or 'cuclass' not in allm[m]['codes']:
                continue
            c = allm[m]['codes']
            own = c['cubase'] != 255 if 'cubase' in c else np.zeros(has.shape, bool)
            ok = (c['cuclass'] != 255) & ~np.isin(sky.classes_of(c['cuclass']), BASE_ONLY_CLASSES)
            agree = (own == has) & ok & (W[i] > bw)
            best = np.where(agree, i, best)
            bw = np.where(agree, W[i], bw)
        for i, m in enumerate(mem):
            sel = best == i
            if not sel.any():
                continue
            c = allm[m]['codes']
            q['cuclass'] = np.where(sel, c['cuclass'], q['cuclass']).astype(np.uint8)
            if 'cusky' in c and 'cusky' in q:
                q['cusky'] = np.where(sel, c['cusky'], q['cusky']).astype(np.uint8)
    if 'cuclass' in base_codes:
        own = np.isin(sky.classes_of(base_codes['cuclass']), BASE_ONLY_CLASSES)
        q['cuclass'] = np.where(own, base_codes['cuclass'], q['cuclass']).astype(np.uint8)
        if 'cusky' in q and 'cusky' in base_codes:
            q['cusky'] = np.where(own, base_codes['cusky'], q['cusky']).astype(np.uint8)

def weight_fields(vals, cond, allm, bundle, region, up, ctx, native=True) -> tuple:
    r = bundle['regions'][region]
    mem = r['members']
    out, failed = ({}, {})
    for g, fields in (r.get('groups') or {}).items():
        fs = [f for f in fields if f in vals]
        if not fs:
            continue
        try:
            ws = [weights(allm, cond, bundle, f, region) for f in fs]
            k = sum(((sum((w[m] for m in mem)) > 0).astype(np.float64) for w in ws))
            bw = {m: np.where(k > 0, sum((w[m] for w in ws)) / np.maximum(k, 1), 0.0) for m in mem}
            st = np.stack([bw[m] for m in mem])
            tot = st.sum(0)
            lead = np.where(tot > 0, st.argmax(0), -1)
            sprd, sfield = (np.full(tot.shape, np.nan), None)
            for f in fs:
                t = r['fields'][f].get('spread')
                if t is None:
                    continue
                x = vals[f]['out']
                cells = np.nonzero(np.isfinite(x).ravel())[0]
                full = np.full(x.size, np.nan)
                c = {**ctx, 'lead_h': cond['lead_h'], 'run_age_h': 0.0}
                full[cells] = np.maximum(0, aircorr.predict(t, feature_arrays(t['features'], x, {}, c, cells), native=native))
                sprd, sfield = (_to_nodes(full.reshape(x.shape), up, tot.shape), f)
                break
        except Exception as e:
            failed[g] = f'failed: {e}'
            continue
        out[g] = {'bw': bw, 'blead': lead, 'bsprd': sprd, 'bsprd_field': sfield}
    return (out, failed)

def _to_nodes(a, up, shape):
    s = np.bincount(up.near, weights=np.nan_to_num(a).ravel(), minlength=shape[0] * shape[1])
    n = np.bincount(up.near, weights=np.isfinite(a).ravel().astype(np.float64), minlength=shape[0] * shape[1])
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / n, np.nan).reshape(shape)

def weight_codes(wf: dict) -> dict:
    out = {}
    for g, d in wf.items():
        for m, a in d['bw'].items():
            out[f'bw_{g}_{m}'] = build.quantize(a, *WENC['bw'])
        out[f'blead_{g}'] = np.where(d['blead'] >= 0, d['blead'], 255).astype(np.uint8)
        if d.get('bsprd_field'):
            out[f'bsprd_{g}'] = build.quantize(d['bsprd'], *SPREAD_ENC[d['bsprd_field']])
    return out
