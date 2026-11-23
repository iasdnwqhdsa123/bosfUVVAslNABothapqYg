from __future__ import annotations
import json
import math
import os
from datetime import datetime, timezone
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'air_corrections.json')
RELIEF_DEG = 0.15
CU_DEPTH = 300.0
CLAMPS = {'ws': [0.5, 2.0], 'top': [0.5, 2.0], 'zi': [0.5, 2.0], 'cb': 600.0, 'cu': [0.3, 0.7]}
CORRECTED = ('wstar', 'zi', 'depth', 'cubase', 'cudepth')

def on() -> bool:
    return os.environ.get('AIR_CORRECTIONS', '0') == '1'

def load(path: str=PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def _flat(tree: list) -> tuple:
    n = len(tree)
    feat, thr, ml, lt, rt, val, leaf = (np.zeros(n, t) for t in (np.int32, np.float64, bool, np.int32, np.int32, np.float64, bool))
    for i, nd in enumerate(tree):
        if len(nd) == 1:
            leaf[i], val[i] = (True, nd[0])
        else:
            feat[i], thr[i], ml[i], lt[i], rt[i] = (nd[0], nd[1], bool(nd[2]), nd[3], nd[4])
    return (feat, thr, ml, lt, rt, val, leaf)

def _flatten_all(trees: list) -> dict:
    parts = [_flat(t) for t in trees]
    roots, off = ([], 0)
    cols = [[] for _ in range(7)]
    for p in parts:
        roots.append(off)
        for k, a in enumerate(p):
            cols[k].append(a + off if k in (3, 4) else a)
        off += len(p[0])
    dt = (np.int32, np.float64, bool, np.int32, np.int32, np.float64, bool)
    feat, thr, ml, lt, rt, val, leaf = (np.ascontiguousarray(np.concatenate(c) if c else np.zeros(0, d)) for c, d in zip(cols, dt))
    return {'trees': parts, 'roots': np.array(roots, np.int32), 'feat': feat.astype(np.int32), 'thr': thr, 'ml': ml.astype(np.uint8), 'lt': lt.astype(np.int32), 'rt': rt.astype(np.int32), 'val': val, 'leaf': leaf.astype(np.uint8)}

def _native():
    try:
        import ctypes
        import physics_native as N
        P = lambda t: ctypes.POINTER(t)
        return N.cfunc('gbm_predict', [ctypes.c_int64, P(ctypes.c_double), ctypes.c_int32, P(ctypes.c_int32), P(ctypes.c_int32), P(ctypes.c_double), P(ctypes.c_uint8), P(ctypes.c_int32), P(ctypes.c_int32), P(ctypes.c_double), P(ctypes.c_uint8), ctypes.c_double, P(ctypes.c_double)])
    except Exception:
        return None

def predict(model: dict, X: np.ndarray, native: bool=True) -> np.ndarray:
    import ctypes
    fl = model.get('_flat') or _flatten_all(model['trees'])
    X = np.ascontiguousarray(X, np.float64)
    n = X.shape[1]
    f = _native() if native else None
    if f is not None:
        out = np.empty(n, np.float64)
        p = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))
        f(n, p(X, ctypes.c_double), len(fl['roots']), p(fl['roots'], ctypes.c_int32), p(fl['feat'], ctypes.c_int32), p(fl['thr'], ctypes.c_double), p(fl['ml'], ctypes.c_uint8), p(fl['lt'], ctypes.c_int32), p(fl['rt'], ctypes.c_int32), p(fl['val'], ctypes.c_double), p(fl['leaf'], ctypes.c_uint8), float(model['base']), p(out, ctypes.c_double))
        return out
    out = np.full(n, float(model['base']))
    cols = np.arange(n)
    miss = np.isnan(X)
    for feat, thr, ml, lt, rt, val, leaf in fl['trees']:
        i = np.zeros(n, np.int32)
        while True:
            go = ~leaf[i]
            if not go.any():
                break
            fi = feat[i]
            v, m = (X[fi, cols], miss[fi, cols])
            nxt = np.where(m, np.where(ml[i], lt[i], rt[i]), np.where(v <= thr[i], lt[i], rt[i]))
            i = np.where(go, nxt, i)
        out += val[i]
    return out

def prepare(corr: dict) -> dict:
    for m in corr['fields'].values():
        m['_flat'] = _flatten_all(m['trees'])
    return corr

def relief_grid(elev: np.ndarray, step: float, deg: float=RELIEF_DEG) -> np.ndarray:
    r = int(round(deg / step))
    size = 2 * r + 1
    known = np.isfinite(elev)
    hi = maximum_filter(np.where(known, elev, -np.inf), size=size, mode='constant', cval=-np.inf)
    lo = minimum_filter(np.where(known, elev, np.inf), size=size, mode='constant', cval=np.inf)
    return np.where(hi >= lo, hi - lo, np.nan).astype(np.float64)

def _doy(day: str) -> int:
    y, m, d = (int(x) for x in day.split('-'))
    return (datetime(y, m, d, tzinfo=timezone.utc) - datetime(y, 1, 1, tzinfo=timezone.utc)).days + 1

def apply(v: dict, lat: np.ndarray, lon: np.ndarray, utc_ms: float, day: str, model: str | int, step: float, corr: dict) -> dict:
    shape = lat.shape
    nan = np.full(shape, np.nan)
    get = lambda k: np.asarray(v[k], np.float64) if v.get(k) is not None else nan
    out = {k: get(k).copy() for k in CORRECTED if v.get(k) is not None}
    ws = get('wstar')
    cells = np.isfinite(ws)
    if not cells.any():
        return out
    doy = _doy(day)
    ds, dc = (math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25))
    utc_h = utc_ms / 3600000.0 % 24
    mid = model if isinstance(model, (int, float)) else corr.get('model_ids', {}).get(model, math.nan)
    relief = relief_grid(get('elevation'), step) if v.get('elevation') is not None else nan
    feats = {}
    for k in corr['features']:
        if k == 'wind':
            a = np.hypot(get('u_blavg'), get('v_blavg'))
        elif k == 'solar':
            a = ((utc_h + lon / 15) % 24 + 24) % 24
        elif k == 'doy_s':
            a = np.full(shape, ds)
        elif k == 'doy_c':
            a = np.full(shape, dc)
        elif k == 'relief':
            a = relief
        elif k == 'model':
            a = np.full(shape, float(mid))
        elif k == 'lat':
            a = lat
        else:
            a = get(k)
        feats[k] = np.asarray(a, np.float64)[cells]
    X = lambda m: np.vstack([feats[k] for k in m['features']])
    F, cl, cu_depth = (corr['fields'], {**CLAMPS, **(corr.get('clamps') or {})}, float(corr.get('cu_depth', CU_DEPTH)))
    cb = get('cubase')[cells]
    blue = ~np.isfinite(cb)

    def put(name, vals, where=None):
        if name not in out:
            return
        a = out[name][cells]
        a = vals if where is None else np.where(where, vals, a)
        out[name][cells] = a
    if 'ws' in F:
        put('wstar', ws[cells] * np.clip(np.exp(predict(F['ws'], X(F['ws']))), *cl['ws']))
    fac = None
    if 'zi' in F:
        fac = np.clip(np.exp(predict(F['zi'], X(F['zi']))), *cl['zi'])
    if 'top' in F:
        ft = np.clip(np.exp(predict(F['top'], X(F['top']))), *cl['top'])
        fac = np.where(blue, ft, fac) if fac is not None else np.where(blue, ft, np.nan)
    if fac is not None:
        for name in ('zi', 'depth'):
            if name in out:
                a = out[name][cells]
                ok = np.isfinite(a) & np.isfinite(fac)
                put(name, np.where(ok, a * fac, a))
    if 'cb' in F:
        s = np.clip(predict(F['cb'], X(F['cb'])), -cl['cb'], cl['cb'])
        put('cubase', np.maximum(50.0, cb + s), ~blue)
    if 'cu' in F and 'cubase' in out:
        p = 1 / (1 + np.exp(-predict(F['cu'], X(F['cu']))))
        drop = ~blue & (p < cl['cu'][0])
        put('cubase', np.nan, drop)
        put('cudepth', np.nan, drop)
        zi_c = out['zi'][cells] if 'zi' in out else np.full(cells.sum(), np.nan)
        add = blue & (p > cl['cu'][1]) & np.isfinite(zi_c) & (zi_c > 0)
        put('cubase', zi_c, add)
        put('cudepth', cu_depth, add)
    return out

def _decode(code: np.ndarray, lo: float, hi: float, transform) -> np.ndarray:
    c = code.astype(np.float64)
    if transform == 'sqrt':
        a, b = (math.sqrt(max(lo, 0)), math.sqrt(hi))
        x = (a + c / 254 * (b - a)) ** 2
    else:
        x = lo + c / 254 * (hi - lo)
    return np.where(code == 255, np.nan, x)

def correct_codes(q: dict, enc: dict, quantize, lat: np.ndarray, lon: np.ndarray, valid: datetime, model: str, step: float, corr: dict) -> dict:
    need = set(corr['features']) | set(CORRECTED) | {'u_blavg', 'v_blavg', 'elevation'}
    v = {k: _decode(q[k], *enc[k]) for k in need if k in q and k in enc}
    utc_ms = valid.replace(tzinfo=timezone.utc).timestamp() * 1000
    mid_lon = float(np.nanmean(lon))
    day = datetime.fromtimestamp(utc_ms / 1000 + mid_lon / 15 * 3600, timezone.utc).strftime('%Y-%m-%d')
    fixed = apply(v, lat, lon, utc_ms, day, model, step, corr)
    out = {}
    for k, a in fixed.items():
        c = quantize(a, *enc[k])
        out[k] = np.where(q[k] == 255, q[k], c).astype(q[k].dtype) if k not in ('cubase', 'cudepth') else c.astype(q[k].dtype)
    return out
