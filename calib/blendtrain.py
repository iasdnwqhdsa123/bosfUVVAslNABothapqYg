from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import split
if common.PIPELINE not in sys.path:
    sys.path.append(common.PIPELINE)
import aircorr
import blend
UTC = timezone.utc
REGION = 'conus'
BASE = 'hrrr'
REGIONS = {'conus': {'base': 'hrrr', 'members': ('hrrr', 'gfs', 'ecmwf', 'icon'), 'box': None}, 'sh': {'base': 'ecmwf', 'members': ('ecmwf', 'icon'), 'box': (112.0, -49.0, 179.75, -18.5)}, 'alps': {'base': 'icond2', 'members': ('icond2', 'arome', 'ecmwf'), 'box': (5.5, 43.5, 16.0, 48.5)}}
MEMBERS = ('hrrr', 'gfs', 'ecmwf', 'icon')
MAX_AGE_H = {'gfs': 12, 'ecmwf': 22, 'icon': 12, 'icond2': 6, 'arome': 9}
GRID_STEP = 0.125
GROUPS = {'surface': ['t2', 'td2', 'wind'], 'cumulus': ['cubase', 'cu']}
FIELDS = ('t2', 'td2', 'wind', 'cubase', 'cu')
KIND = {'t2': 'add', 'td2': 'add', 'wind': 'add', 'cubase': 'add', 'cu': 'logit'}
CLAMP = {'t2': 5.0, 'td2': 6.0, 'wind': 5.0, 'cubase': 600.0, 'cu': 10.0}
LEAD_BINS = [0, 6, 12, 24, 48, 72]
SOLAR_BINS = ['night', [0, 10], [10, 13], [13, 16], [16, 24]]
N_MIN = 500
SHRINK_N = 500
OPS_DAYS_MIN = {'icon': 60, 'arome': 60, 'icond2': 60}
ARCHIVED_RUNS = ('icon', 'icond2', 'arome')
MIN_FIT = 200
MIN_FEATURE = 0.02
SRC_CODE = {'ops': 0, 'archive': 1, 'stcast': 2, 'hind': 3}
AREA_DEG = 10.0
K_OUTER = 3
K_INNER = 2
MAX_ROWS = 3000000
SEED = split.SEED
TREE = dict(loss='absolute_error', max_iter=80, learning_rate=0.06, max_leaf_nodes=6, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False)
SPREAD_TREE = dict(TREE, loss='quantile', quantile=0.8, max_iter=40)
GATE_TREE = dict(max_iter=40, learning_rate=0.08, max_leaf_nodes=6, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False)
GATE_IN_BUNDLE = False
CU_FLOOR = blend.CU_EPS
BASE_FEATURES = ('x', 'lead_h', 'run_age_h', 'solar_h', 'sun_elev', 'lat', 'relief5km', 'dz_terrain', 'regime', 'n_models', 'cuclass', 'ws', 'wd', 'src', 'replay')
OWN_OPTIONAL = ('zi', 'depth', 'wstar')
SPREAD_FEATURES = ('x', 'lead_h', 'solar_h', 'sun_elev', 'lat', 'relief5km', 'regime', 'n_models')
GATE_FEATURES = ('lead_h', 'solar_h', 'sun_elev', 'lat', 'regime', 'n_models')

def _f(a):
    return np.asarray(a, np.float64)

def _ts_hours(runs) -> np.ndarray:
    a = np.asarray(runs, dtype=object)
    u, inv = np.unique(a.astype(str), return_inverse=True)
    h = np.array([int(datetime.strptime(x, '%Y%m%d%H').replace(tzinfo=UTC).timestamp()) // 3600 if x else -1 for x in u], np.int64)
    return h[inv.reshape(-1)]
KEY_RADIX = (1 << 17, 1 << 20, 1 << 10, 1 << 14)

class _Keys:

    def __init__(self):
        self.books = {}

    def code(self, name, values) -> np.ndarray:
        v = np.asarray(values, dtype=object).astype(str)
        book = self.books.setdefault(name, {})
        u, inv = np.unique(v, return_inverse=True)
        codes = np.array([book.setdefault(x, len(book)) for x in u], np.int64)
        return codes[inv.reshape(-1)] if len(v) else np.zeros(0, np.int64)

    @staticmethod
    def key(*cols) -> np.ndarray:
        key = np.zeros(len(cols[0]), np.int64)
        bad = np.zeros(len(cols[0]), bool)
        for a, r in zip(cols, KEY_RADIX[:len(cols)]):
            a = np.asarray(a, np.int64)
            bad |= (a < 0) | (a >= r)
            key = key * r + np.clip(a, 0, r - 1)
        return np.where(bad, -1, key)

def _lookup(keys_tab, keys_q) -> np.ndarray:
    if not len(keys_tab):
        return np.full(len(keys_q), -1, np.int64)
    order = np.argsort(keys_tab, kind='stable')
    k = keys_tab[order]
    pos = np.searchsorted(k, keys_q, side='right') - 1
    ok = (pos >= 0) & (k[np.clip(pos, 0, None)] == keys_q) & (keys_q >= 0)
    return np.where(ok, order[np.clip(pos, 0, None)], -1)

def _mode_regime(R: np.ndarray) -> np.ndarray:
    counts = np.stack([(R == i).sum(0) for i in range(len(blend.REGIMES))])
    return np.where((R >= 0).any(0), counts.argmax(0), -1)

def spread_of(st: np.ndarray) -> np.ndarray:
    k = np.isfinite(st).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.where(k >= 2, np.nanstd(st, 0), np.nan)

def assemble(snap: dict, pairs: dict, obs: dict, static: dict | None=None, members=MEMBERS, base: str=BASE, max_age_h: dict | None=None) -> dict:
    import pairs as P
    max_age_h = MAX_AGE_H if max_age_h is None else max_age_h
    for name, t in [('snapshots', snap), ('observations', obs)] + [(f'pairs of {m}', t) for m, t in pairs.items()]:
        if P.collector_train():
            break
        if 'review' in t and len(t['review']) and np.asarray(t['review'], bool).any():
            raise ValueError(f'blend train: review rows in the {name} (read them with purpose train)')
    K = _Keys()
    n = len(snap['obs_id'])
    sid = K.code('obs', snap['obs_id'])
    sh = np.asarray(snap['t_valid'], np.int64) // 3600
    F = {'obs_id': np.asarray(snap['obs_id'], dtype=object), 't_valid': np.asarray(snap['t_valid'], np.int64), 'slot': np.asarray(snap['slot'], np.int64), 'lat': _f(snap['lat']), 'lon': _f(snap['lon']), 'lead_h': _f(snap['lead_h']), 'solar_h': _f(snap['solar_h']), 'sun_elev': _f(snap['sun_elev'])}
    F['day'] = F['t_valid'] // 86400
    have = []
    for m in members:
        run = np.asarray(snap.get(f'run_{m}', np.full(n, '', dtype=object)), dtype=object)
        age = _f(snap.get(f'age_{m}', np.full(n, np.nan)))
        reg = np.asarray(snap.get(f'regime_{m}', np.full(n, -1)), np.int64)
        lim = max_age_h.get(m)
        pres = (run.astype(str) != '') & (reg >= 0) & np.isfinite(age) & (age <= lim if lim is not None else True)
        F[f'present_{m}'] = pres
        F[f'age_{m}'] = age
        F[f'lead_{m}'] = _f(snap.get(f'lead_{m}', np.full(n, -1)))
        F[f'regime_{m}'] = np.where(pres, reg, -1)
        su, sinv = np.unique(np.asarray(snap.get(f'src_{m}', [''] * n), dtype=object).astype(str), return_inverse=True)
        F[f'src_{m}'] = np.array([SRC_CODE.get(x, np.nan) for x in su], np.float64)[sinv.reshape(-1)] if n else np.zeros(0)
        src = np.asarray(snap.get(f'src_{m}', [''] * n), dtype=object).astype(str)
        F[f'opsday_{m}'] = pres & ((src == 'ops') | (src == 'stcast') & (m in ARCHIVED_RUNS))
        t = pairs.get(m)
        for v in ('t2', 'td2', 'wind', 'cubase', 'cu', 'ws', 'wd', 'cuclass', 'elev', 'replay', 'zi', 'depth', 'wstar'):
            F[f'{v}_{m}'] = np.full(n, np.nan)
        if t is None or not len(t.get('obs_id', [])):
            F[f'present_{m}'] = np.zeros(n, bool)
            continue
        bs = np.asarray(snap.get(f'build_{m}', [''] * n), dtype=object)
        ss = np.asarray(snap.get(f'src_{m}', [''] * n), dtype=object)
        th = np.asarray(t['t_valid'], np.int64) // 3600
        pk = K.key(K.code('obs', t['obs_id']), th, th - _ts_hours(t['run']), K.code('bs', [f'{b}|{s}' for b, s in zip(t['build'], t['src'])]))
        qk = K.key(sid, sh, sh - _ts_hours(run), K.code('bs', [f'{b}|{s}' for b, s in zip(bs, ss)]))
        j = _lookup(pk, qk)
        pres = pres & (j >= 0)
        F[f'present_{m}'] = pres
        jj = np.where(pres, j, 0)

        def col(name):
            return np.where(pres, _f(t[name])[jj], np.nan) if name in t else np.full(n, np.nan)
        u, v = (col('f_u_sfc'), col('f_v_sfc'))
        el = col('f_elevation')
        cb = col('f_cubase')
        F[f't2_{m}'], F[f'td2_{m}'] = (col('f_t2'), col('f_td2'))
        F[f'wind_{m}'] = np.hypot(u, v)
        F[f'ws_{m}'], F[f'wd_{m}'] = (np.hypot(u, v), (np.degrees(np.arctan2(-u, -v)) + 360) % 360)
        F[f'cubase_{m}'] = cb + np.nan_to_num(el, nan=0.0)
        F[f'cu_{m}'] = np.where(pres, np.isfinite(cb).astype(np.float64), np.nan)
        F[f'cuclass_{m}'] = col('f_cuclass')
        F[f'elev_{m}'] = el
        F[f'replay_{m}'] = np.where(pres, np.asarray(t['replay'], bool)[jj].astype(np.float64), np.nan) if 'replay' in t else np.where(pres, 0.0, np.nan)
        for v in OWN_OPTIONAL:
            F[f'{v}_{m}'] = col(f'f_{v}')
        have.append(m)
    static = static or {}
    for m in members:
        F[f'dz_{m}'] = np.full(n, np.nan)
        st = static.get(m)
        if st is not None and len(st.get('obs_id', [])):
            j = _lookup(K.code('obs', st['obs_id']), sid)
            F[f'dz_{m}'] = np.where(j >= 0, _f(st['dz_terrain'])[np.maximum(j, 0)], np.nan)
            if m == base:
                F['relief5km'] = np.where(j >= 0, _f(st['relief5km'])[np.maximum(j, 0)], np.nan)
    F.setdefault('relief5km', np.full(n, np.nan))
    j = _lookup(K.key(K.code('obs', obs['obs_id']), np.asarray(obs['t_valid'], np.int64) // 3600), K.key(sid, sh))
    hit = j >= 0
    jj = np.maximum(j, 0)
    ocol = lambda k: np.where(hit, _f(obs[k])[jj], np.nan) if k in obs else np.full(n, np.nan)
    F['o_t2'], F['o_td2'], F['o_wind'], F['o_cu'] = (ocol('o_t2'), ocol('o_td2'), ocol('o_ws'), ocol('o_cu'))
    F['o_cubase'] = np.where(F['o_cu'] == 1, ocol('o_cubase') + np.nan_to_num(ocol('elev'), nan=0.0), np.nan)
    R = np.stack([F[f'regime_{m}'] for m in members])
    F['regime'] = _mode_regime(R).astype(np.float64)
    F['n_models'] = (R >= 0).sum(0).astype(np.float64)
    for f in FIELDS:
        F[f'spread_{f}'] = spread_of(np.stack([F[f'{f}_{m}'] for m in members]))
    F['ground'] = F.get(f'elev_{base}', np.full(n, np.nan))
    keep = F[f'present_{base}']
    out = {k: v[keep] for k, v in F.items()}
    out['_members'] = [m for m in members if m in have and out[f'present_{m}'].any()]
    out['_base'] = base
    return out

def sample(snap: dict, max_rows: int=MAX_ROWS, seed: int=SEED) -> dict:
    n = len(snap['obs_id'])
    if n <= max_rows:
        return snap
    ids = np.asarray(snap['obs_id'], dtype=object).astype(str)
    u, inv = np.unique(ids, return_inverse=True)
    sid = np.array([int.from_bytes(hashlib.blake2b(x.encode(), digest_size=7).digest(), 'big') for x in u], np.int64)[inv.reshape(-1)]
    h = split._mix(seed, sid, np.asarray(snap['t_valid'], np.int64) // 3600, np.asarray(snap['slot'], np.int64) // 3600)
    keep = h < np.uint64(int(max_rows / n * float(2 ** 64 - 1)))
    return {k: np.asarray(v)[keep] for k, v in snap.items()}

def n_of(F) -> int:
    return len(F['t_valid'])

def _logit(p):
    p = np.clip(p, CU_FLOOR, 1 - CU_FLOOR)
    return np.log(p / (1 - p))

def feature_names(F, f: str, m: str, rows) -> list:
    names = list(BASE_FEATURES) + [f'spread_{f}'] + [v for v in OWN_OPTIONAL if np.isfinite(F[f'{v}_{m}']).any()]
    X = features(F, names, f, m, rows)
    return [k for k, col in zip(names, X) if k == 'x' or (np.isfinite(col).mean() if col.size else 0) >= MIN_FEATURE]

def features(F, names, f: str, m: str, rows) -> np.ndarray:
    src = {'x': f'{f}_{m}', 'lead_h': f'lead_{m}', 'run_age_h': f'age_{m}', 'dz_terrain': f'dz_{m}', 'src': f'src_{m}', 'replay': f'replay_{m}'}
    for k in ('cuclass', 'ws', 'wd') + OWN_OPTIONAL:
        src[k] = f'{k}_{m}'
    out = np.empty((len(names), len(rows)))
    for i, k in enumerate(names):
        out[i] = _f(F[src.get(k, k)])[rows]
    return out

def spread_features(F, names, x, rows) -> np.ndarray:
    out = np.empty((len(names), len(rows)))
    for i, k in enumerate(names):
        out[i] = x[rows] if k == 'x' else _f(F[k])[rows]
    return out

def field_rows(F, f: str, m: str) -> np.ndarray:
    x, o = (F[f'{f}_{m}'], F[f'o_{f}'])
    ok = F[f'present_{m}'] & np.isfinite(x) & np.isfinite(o)
    return np.flatnonzero(ok)

def _target(kind, x, o):
    return o - x if kind == 'add' else o

def _export(model, names, kind, X=None):
    import train
    e = train.export(model, names)
    if X is not None:
        e['_check'] = train.export_check(e, model, X)
    if kind == 'logit':
        lo = float(-_logit(np.array([0.0]))[0])
        e['trees'].append([[names.index('x'), 0.5, 0, 1, 2], [float(f'{lo:.7g}')], [float(f'{-lo:.7g}')]])
    e['kind'] = kind
    return e

def fit_trees(kind: str, X: np.ndarray, y: np.ndarray, names, seed: int, params=None) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    p = dict(params or TREE)
    p['random_state'] = int(seed)
    if kind == 'logit':
        p.pop('loss', None)
        if len(np.unique(y)) < 2:
            r = float(np.clip(np.mean(y), 0.01, 0.99))
            return _const_logit(names, math.log(r / (1 - r)))
        m = HistGradientBoostingClassifier(**{k: v for k, v in p.items() if k not in ('quantile',)}).fit(X.T, y.astype(int))
    else:
        m = HistGradientBoostingRegressor(**p).fit(X.T, y)
    return _export(m, names, kind, X.T)

def _const_logit(names, L):
    lo = float(-_logit(np.array([0.0]))[0])
    return {'features': list(names), 'base': float(f'{L:.7g}'), 'trees': [[[names.index('x'), 0.5, 0, 1, 2], [lo], [-lo]]], 'kind': 'logit', '_check': 0.0}

def apply_trees(e: dict, F, f: str, m: str, rows) -> np.ndarray:
    x = _f(F[f'{f}_{m}'])[rows]
    if e is None or not len(rows):
        return x.copy()
    d = aircorr.predict(e, features(F, e['features'], f, m, rows), native=True)
    out = blend.debias(e['kind'], x, d, CLAMP[f])
    if f == 'cubase':
        g = _f(F['ground'])[rows]
        out = np.where(np.isfinite(g), np.maximum(out, g + blend.CB_MIN_AGL), out)
    if f == 'wind':
        out = np.maximum(out, 0.0)
    return out

class Log:

    def __init__(self):
        self.fits = []

    def add(self, **kw):
        self.fits.append(kw)

def _fit(F, f, m, train_idx, seed, params, log, stage, scored=()):
    rows = np.intersect1d(field_rows(F, f, m), train_idx)
    if len(rows) < MIN_FIT:
        log.add(field=f, member=m, stage=stage, train=rows, scored=np.asarray(scored, np.int64), trees=False)
        return None
    names = feature_names(F, f, m, rows)
    X = features(F, names, f, m, rows)
    y = _target(KIND[f], _f(F[f'{f}_{m}'])[rows], _f(F[f'o_{f}'])[rows])
    e = fit_trees(KIND[f], X, y, names, seed, params)
    log.add(field=f, member=m, stage=stage, train=rows, scored=np.asarray(scored, np.int64), trees=True)
    return e

def simplex_min(S: np.ndarray) -> np.ndarray:
    K = S.shape[0]
    if K == 1:
        return np.ones(1)
    S = 0.5 * (S + S.T)
    ridge = 1e-09 * max(float(np.trace(S)) / K, 1e-12)
    best, bv = (None, math.inf)
    for r in range(1, K + 1):
        for sup in itertools.combinations(range(K), r):
            A = S[np.ix_(sup, sup)] + ridge * np.eye(r)
            try:
                z = np.linalg.solve(A, np.ones(r))
            except np.linalg.LinAlgError:
                continue
            tot = z.sum()
            if not np.isfinite(tot) or tot <= 0:
                continue
            w = z / tot
            if (w < -1e-12).any():
                continue
            full = np.zeros(K)
            full[list(sup)] = np.maximum(w, 0)
            full /= full.sum()
            v = float(full @ S @ full)
            if v < bv - 1e-15:
                best, bv = (full, v)
    return best if best is not None else np.full(K, 1.0 / K)

def _moments(E, cells, nc):
    K = E.shape[1]
    n = np.bincount(cells, minlength=nc).astype(np.float64)
    S1 = np.zeros((nc, K, K))
    S2 = np.zeros((nc, K, K))
    for i in range(K):
        for j in range(i, K):
            p = E[:, i] * E[:, j]
            S1[:, i, j] = S1[:, j, i] = np.bincount(cells, p, nc)
            S2[:, i, j] = S2[:, j, i] = np.bincount(cells, p * p, nc)
    return (n, S1, S2)

def ledoit_wolf(n, s1, s2) -> np.ndarray:
    if n < 2:
        return np.diag(np.diag(s1)) / max(n, 1)
    M = s1 / n
    var = np.maximum(s2 / n - M * M, 0) / n
    off = ~np.eye(len(M), dtype=bool)
    den = float((M[off] ** 2).sum())
    d = 1.0 if den <= 0 else float(np.clip(var[off].sum() / den, 0, 1))
    out = M * (1 - d)
    out[~off] = np.diag(M)
    return out

def solve_table(err: dict, cells: np.ndarray, members: list, nc: int, shrink_n: float=SHRINK_N) -> tuple:
    table, counts = ({}, {})
    cells = np.asarray(cells, np.int64)
    for s in blend.subsets(members):
        E = np.column_stack([_f(err[m]) for m in s])
        ok = np.isfinite(E).all(1) & (cells >= 0)
        E, c = (E[ok], cells[ok])
        K = len(s)
        n, S1, S2 = _moments(E, c, nc)
        npool = float(n.sum())
        if npool >= 2:
            pool = ledoit_wolf(npool, S1.sum(0), S2.sum(0))
        else:
            pool = np.diag([np.nanmean(_f(err[m]) ** 2) if np.isfinite(_f(err[m])).any() else 1.0 for m in s])
        rows = []
        wpool = simplex_min(pool)
        for ci in range(nc):
            if n[ci] < 2:
                rows.append(wpool)
                continue
            Sc = ledoit_wolf(n[ci], S1[ci], S2[ci])
            a = n[ci] / (n[ci] + shrink_n)
            rows.append(simplex_min(a * Sc + (1 - a) * pool))
        table[blend.subset_key(s)] = [[float(f'{x:.6g}') for x in _renorm(w)] for w in rows]
        counts[blend.subset_key(s)] = n.astype(int).tolist()
    return (table, counts)

def _renorm(w):
    w = np.maximum(np.asarray(w, np.float64), 0)
    w = np.round(w / w.sum(), 6)
    w[np.argmax(w)] += 1 - w.sum()
    return np.maximum(w, 0)

def bins() -> dict:
    return {'lead': list(LEAD_BINS), 'solar': [SOLAR_BINS[0]] + [list(b) for b in SOLAR_BINS[1:]], 'regime': list(blend.REGIMES)}

def cells_of(F, b: dict | None=None) -> np.ndarray:
    b = b or bins()
    out = np.full(n_of(F), -1, np.int64)
    lead = _f(F['lead_h'])
    for L in np.unique(lead):
        sel = lead == L
        cond = {'lead_h': float(L), 'solar_h': _f(F['solar_h'])[sel], 'sun_elev': _f(F['sun_elev'])[sel], 'regime': np.asarray(_f(F['regime'])[sel], np.int64)}
        out[sel] = blend.cell_index(cond, b)
    return out

def counts_n(F, f: str, members, cells, nc, rows=None) -> dict:
    out = {}
    for m in members:
        sel = np.zeros(n_of(F), bool)
        sel[field_rows(F, f, m)] = True
        if rows is not None:
            mask = np.zeros(n_of(F), bool)
            mask[rows] = True
            sel &= mask
        sel &= cells >= 0
        c = np.bincount(cells[sel], minlength=nc)[:nc]
        need = OPS_DAYS_MIN.get(m) if m != F.get('_base', BASE) else None
        if need is not None and len(np.unique(F['day'][F[f'opsday_{m}']])) < need:
            c = np.zeros(nc, np.int64)
        out[m] = [int(x) for x in c]
    return out

def _prepared(wt: dict, members: list) -> dict:
    b = {'regions': {REGION: {'members': list(members), 'fields': {'f': {'weights': wt}}}}}
    return blend.prepare(b)['regions'][REGION]['fields']['f']['weights']

def row_weights(wt: dict, members: list, present: dict, cells: np.ndarray) -> np.ndarray:
    p = _prepared(wt, members)
    bits = np.zeros(len(cells), np.int64)
    for i, m in enumerate(members):
        bits |= np.asarray(present[m], bool).astype(np.int64) << i
    return blend._node_weights(bits, cells, p)

def blended(W, X, members, base_x) -> np.ndarray:
    num = np.zeros(len(base_x))
    for i, m in enumerate(members):
        num += W[:, i] * np.nan_to_num(X[m])
    some = W.sum(1) > 0
    return np.where(some, num, base_x)

def _gate_X(F, rows, f):
    names = list(GATE_FEATURES) + [f'spread_{f}']
    return (names, np.column_stack([_f(F[k])[rows] for k in names]))

def fit_gate(F, f, rows, X: dict, members, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    K = len(members)
    o = _f(F[f'o_{f}'])[rows]
    A = np.column_stack([np.abs(o - X[m]) for m in members])
    if K < 2 or len(rows) < MIN_FIT:
        return None
    tau = float(np.nanmedian(A)) or 1.0
    with np.errstate(invalid='ignore', over='ignore'):
        Z = np.where(np.isfinite(A), -A / tau, -np.inf)
        Z = Z - np.nanmax(np.where(np.isfinite(Z), Z, np.nan), 1, keepdims=True)
        P = np.exp(Z)
    P = np.nan_to_num(P) / np.maximum(np.nansum(np.nan_to_num(P), 1, keepdims=True), 1e-12)
    _, G = _gate_X(F, rows, f)
    Xr = np.repeat(G, K, 0)
    yr = np.tile(np.arange(K), len(rows))
    wr = P.ravel()
    keep = wr > 1e-06
    if len(np.unique(yr[keep])) < K:
        return None
    m = HistGradientBoostingClassifier(**GATE_TREE, random_state=int(seed)).fit(Xr[keep], yr[keep], sample_weight=wr[keep])
    return m

def gate_weights(m, F, rows, f, present: dict, members) -> np.ndarray:
    _, G = _gate_X(F, rows, f)
    d = m.decision_function(G)
    if d.ndim == 1:
        d = np.column_stack([np.zeros(len(d)), d])
    pres = np.column_stack([np.asarray(present[k], bool) for k in members])
    d = np.where(pres, d, -np.inf)
    d = d - np.max(np.where(pres, d, -1e+300), 1, keepdims=True)
    e = np.where(pres, np.exp(d), 0)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-300)

def export_multi(m, names) -> dict:
    import train
    per = [[] for _ in range(len(m._predictors[0]))]
    for it in m._predictors:
        for k, pred in enumerate(it):
            one = type('One', (), {'_predictors': [[pred]], '_baseline_prediction': np.array([0.0])})()
            per[k].append(train.export(one, names)['trees'][0])
    base = np.ravel(m._baseline_prediction)
    return {'features': list(names), 'classes': [int(c) for c in m.classes_], 'it': [{'base': float(f'{(base[k] if k < len(base) else 0.0):.7g}'), 'trees': per[k]} for k in range(len(per))]}

def loss(f, o, p):
    return (p - o) ** 2 if f == 'cu' else np.abs(o - p)

def areas(F) -> np.ndarray:
    return np.floor(_f(F['lat']) / AREA_DEG).astype(np.int64) * 1000 + np.floor(_f(F['lon']) / AREA_DEG).astype(np.int64)

def _lead_bin(F):
    return np.clip(np.searchsorted(np.asarray(LEAD_BINS, np.float64), _f(F['lead_h']), side='right') - 1, 0, len(LEAD_BINS) - 2)

def _sun(F):
    return blend.sun_band(_f(F['solar_h']), _f(F['sun_elev']), bins()['solar'])

def serve_rows(F, f, members, wide: bool=False) -> np.ndarray:
    o = np.isfinite(_f(F[f'o_{f}']))
    if f == 'cubase' and wide:
        anym = np.zeros(n_of(F), bool)
        for m in members:
            anym |= F[f'present_{m}'] & np.isfinite(_f(F[f'cubase_{m}']))
        return np.flatnonzero(o & anym)
    base = F.get('_base', BASE)
    b = base if base in members else members[0]
    return np.flatnonzero(o & F[f'present_{b}'] & np.isfinite(_f(F[f'{f}_{b}'])))

def _present(F, f, m, rows):
    return F[f'present_{m}'][rows] & np.isfinite(_f(F[f'{f}_{m}'])[rows])

def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return float(f'{v:.6g}') if math.isfinite(v) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x

def train_field(F, f: str, members: list, seed: int=SEED, params=None, n_min: int=N_MIN, log: Log | None=None, k_outer: int=K_OUTER, k_inner: int=K_INNER) -> tuple:
    log = log if log is not None else Log()
    b = bins()
    nc = blend.n_cells(b)
    cells = cells_of(F, b)
    n = n_of(F)
    rows_f = serve_rows(F, f, members)
    rows_w = serve_rows(F, f, members, wide=True)
    allrows = np.arange(n)
    outer = list(split.folds(F['day'], F['lat'], F['lon'], k_time=k_outer, k_space=k_outer, pairs=[(i, i) for i in range(k_outer)], seed=seed))
    tfold = split.time_fold(F['day'], k=k_outer, seed=seed)
    xo = {m: np.full(n, np.nan) for m in members}
    for o in outer:
        scope = np.flatnonzero(tfold == o.t)
        for m in members:
            e = _fit(F, f, m, o.train, seed, params, log, ('outer', o.t), scope)
            xo[m][scope] = apply_trees(e, F, f, m, scope)
    report_cells = {}
    test_parts = []
    gate_cmp = []
    for o in outer:
        te = np.intersect1d(o.test, rows_f)
        tr = np.intersect1d(o.train, rows_f)
        trw = np.intersect1d(o.train, rows_w)
        if not len(te) or not len(tr):
            continue
        xi = {m: np.full(n, np.nan) for m in members}
        sub = o.train
        for g in split.folds(F['day'][sub], k_time=k_inner, seed=seed):
            sc = sub[g.test]
            for m in members:
                e = _fit(F, f, m, sub[g.train], seed, params, log, ('inner', o.t, g.t), sc)
                xi[m][sc] = apply_trees(e, F, f, m, sc)
        err = {m: np.where(_mask(n, trw) & _present_all(F, f, m), _f(F[f'o_{f}']) - xi[m], np.nan) for m in members}
        table, _ = solve_table(err, cells, members, nc)
        nmin_fold = max(1, int(round(n_min * len(o.train) / max(n, 1))))
        wt = {'bins': b, 'table': table, 'n_min': nmin_fold, 'n': counts_n(F, f, members, cells, nc, o.train)}
        pres = {m: _present(F, f, m, te) for m in members}
        W = row_weights(wt, members, pres, cells[te])
        Xte = {m: np.where(pres[m], xo[m][te], np.nan) for m in members}
        base_x = xo[F.get('_base', BASE)][te] if F.get('_base', BASE) in members else Xte[members[0]]
        p_tab = blended(W, Xte, members, base_x)
        if f == 'cu':
            p_tab = np.clip(p_tab, 0, 1)
        o_te = _f(F[f'o_{f}'])[te]
        gm = fit_gate(F, f, tr, {m: np.where(_present(F, f, m, tr), xi[m][tr], np.nan) for m in members}, members, seed)
        if gm is not None:
            Wg = gate_weights(gm, F, te, f, pres, members)
            p_gate = blended(Wg, Xte, members, base_x)
            gate_cmp.append((te, loss(f, o_te, p_gate), loss(f, o_te, p_tab)))
        both = np.concatenate([tr, te])
        fit = np.zeros(len(both), bool)
        fit[:len(tr)] = True
        bcell = areas(F)[both] * 10 + _lead_bin(F)[both]
        raw = {m: np.where(_present(F, f, m, both), loss(f, _f(F[f'o_{f}'])[both], _f(F[f'{f}_{m}'])[both]), np.nan) for m in members}
        deb_x = {m: np.concatenate([xi[m][tr], xo[m][te]]) for m in members}
        deb = {m: np.where(_present(F, f, m, both), loss(f, _f(F[f'o_{f}'])[both], deb_x[m]), np.nan) for m in members}
        base_loss, choice = split.pick_base(raw, bcell, fit)
        deb_loss, _ = split.pick_base(deb, bcell, fit)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            eq = np.nanmean(np.column_stack([Xte[m] for m in members]), 1)
        test_parts.append({'rows': te, 'blend': loss(f, o_te, p_tab), 'base': base_loss[len(tr):], 'deb': deb_loss[len(tr):], 'eq': loss(f, o_te, eq), 'choice': {str(k): v for k, v in choice.items()}, 'n_min': nmin_fold})
    final = {}
    for m in members:
        final[m] = _fit(F, f, m, allrows, seed, params, log, ('final',), ())
    err = {m: np.where(_mask(n, rows_w) & _present_all(F, f, m), _f(F[f'o_{f}']) - xo[m], np.nan) for m in members}
    table, tcounts = solve_table(err, cells, members, nc)
    wt = {'bins': b, 'table': table, 'n_min': n_min, 'n': counts_n(F, f, members, cells, nc)}
    pres = {m: _present(F, f, m, rows_w) for m in members}
    W = row_weights(wt, members, pres, cells[rows_w])
    Xr = {m: np.where(pres[m], xo[m][rows_w], np.nan) for m in members}
    pb = blended(W, Xr, members, xo[F.get('_base', BASE)][rows_w] if F.get('_base', BASE) in members else Xr[members[0]])
    rows_s = rows_w
    spread = None
    ok = np.isfinite(pb)
    if ok.sum() >= MIN_FIT:
        from sklearn.ensemble import HistGradientBoostingRegressor
        xb = _scatter(n, rows_s, pb)
        names = list(SPREAD_FEATURES)
        names = [k for k, c in zip(names, spread_features(F, names, xb, rows_s[ok])) if k == 'x' or np.isfinite(c).mean() >= MIN_FEATURE]
        Xs = spread_features(F, names, xb, rows_s[ok])
        y = np.abs(_f(F[f'o_{f}'])[rows_s[ok]] - pb[ok])
        sp = dict(SPREAD_TREE if params is None else dict(params, loss='quantile', quantile=0.8))
        sp['random_state'] = int(seed)
        import train
        spread = train.export(HistGradientBoostingRegressor(**sp).fit(Xs.T, y), names)
        log.add(field=f, member='blend', stage=('spread',), train=rows_s[ok], scored=np.zeros(0, np.int64), trees=True)
    verdict, cv = _verdict(F, f, test_parts, gate_cmp, cells, b, members, wt, seed)
    gate = None
    if cv['gate']['wins'] and GATE_IN_BUNDLE:
        gm = fit_gate(F, f, rows_w, Xr, members, seed)
        gate = export_multi(gm, list(GATE_FEATURES) + [f'spread_{f}']) if gm is not None else None
    feats = sorted({k for e in final.values() if e for k in e['features']})
    entry = {'kind': KIND[f], 'features': feats, 'clamp': CLAMP[f], 'debias': {m: _strip(e) for m, e in final.items() if e}, 'weights': wt, 'gate': gate, 'spread': spread, 'pass': bool(verdict['ship'])}
    if not entry['pass']:
        entry['from'] = 'base'
    cv['table_rows'] = tcounts.get(blend.subset_key(members), [])
    cv['export_check'] = max([float(e.get('_check', 0.0)) for e in final.values() if e] + [0.0])
    return (entry, cv)

def _strip(e):
    return {k: v for k, v in e.items() if k in ('features', 'base', 'trees', 'kind')}

def _scatter(n, idx, v):
    out = np.full(n, np.nan)
    out[idx] = v
    return out

def _mask(n, idx):
    m = np.zeros(n, bool)
    m[idx] = True
    return m

def _present_all(F, f, m):
    return F[f'present_{m}'] & np.isfinite(_f(F[f'{f}_{m}']))

def _verdict(F, f, parts, gate_cmp, cells, b, members, wt, seed):
    if not parts:
        v = {'ship': False, 'overall': {}, 'conditions': {'rows': {'ok': False, 'why': 'no test rows'}}}
        return (v, {'loss': 'brier' if f == 'cu' else 'mae', 'verdict': v, 'cells': {}, 'gate': {'wins': False, 'why': 'no test rows'}})
    rows = np.concatenate([p['rows'] for p in parts])
    L = {k: np.concatenate([p[k] for p in parts]) for k in ('blend', 'base', 'deb', 'eq')}
    week = split.week_of(F['day'][rows])
    ar = areas(F)[rows]
    blocks = split.blocks_of(week, ar)
    lb = _lead_bin(F)[rows]
    sun = _sun(F)[rows]
    region_cells = np.array([f'{a}|lead {LEAD_BINS[i]}-{LEAD_BINS[i + 1]}' for a, i in zip(ar, lb)], dtype=object)
    sun_names = ['night'] + [f'solar {x[0]}-{x[1]}' for x in b['solar'][1:]]
    tod = np.array([sun_names[s] for s in sun], dtype=object)
    v = split.ship_verdict(L['blend'], L['base'], blocks, {'region': region_cells, 'time of day': tod}, seed=seed)
    mean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else None
    worse = [list(map(str, c)) for c in v['conditions']['cells']['worse']]
    cv = {'loss': 'brier' if f == 'cu' else 'mae', 'rows': int(len(rows)), 'mae_blend': mean(L['blend']), 'mae_best_raw': mean(L['base']), 'mae_best_debiased': mean(L['deb']), 'mae_equal_debiased': mean(L['eq']), 'ci95': [v['overall'].get('lo'), v['overall'].get('hi')], 'diff': v['overall'].get('diff'), 'bins_failed': worse, 'pass': bool(v['ship']), 'base_choice': [p['choice'] for p in parts], 'n_min_folds': [p['n_min'] for p in parts], 'verdict': v}
    bm = F.get('_base', BASE)
    ops = _f(F[f'src_{bm}'])[rows] == SRC_CODE['ops'] if f'src_{bm}' in F else np.zeros(len(rows), bool)
    if ops.any():
        vo = split.ship_verdict(L['blend'][ops], L['base'][ops], blocks[ops], {'region': region_cells[ops], 'time of day': tod[ops]}, seed=seed)
        cv['ops'] = {'rows': int(ops.sum()), 'mae_blend': mean(L['blend'][ops]), 'mae_best_raw': mean(L['base'][ops]), 'mae_best_debiased': mean(L['deb'][ops]), 'mae_equal_debiased': mean(L['eq'][ops]), 'pass': bool(vo['ship']), 'verdict': vo}
    else:
        cv['ops'] = {'rows': 0}
    cv['oof_ops'] = {'day': F['day'][rows][ops], 'area': ar[ops], 'lead_bin': lb[ops], 'sun': sun[ops], 'blend': L['blend'][ops], 'base': L['base'][ops]}
    c = cells[rows]
    full = wt['table'][blend.subset_key(members)]
    names = cell_names(b)
    per = {}
    for ci in np.unique(c[c >= 0]):
        s = c == ci
        per[names[ci]] = {'n': int(s.sum()), 'blend': mean(L['blend'][s]), 'best_raw': mean(L['base'][s]), 'best_debiased': mean(L['deb'][s]), 'equal_debiased': mean(L['eq'][s]), 'weights': dict(zip(members, full[ci])), 'n_member': {m: wt['n'][m][ci] for m in members}}
    cv['cells'] = per
    if gate_cmp:
        gl = np.concatenate([g[1] for g in gate_cmp])
        tl = np.concatenate([g[2] for g in gate_cmp])
        gr = np.concatenate([g[0] for g in gate_cmp])
        bt = split.paired_bootstrap(gl, tl, split.blocks_of(split.week_of(F['day'][gr]), areas(F)[gr]), seed=seed)
        wins = bool(np.isfinite(bt['hi']) and bt['hi'] < 0)
        cv['gate'] = {'wins': wins, 'loss_gate': mean(gl), 'loss_table': mean(tl), 'diff': bt['diff'], 'ci95': [bt['lo'], bt['hi']], 'in_bundle': bool(wins and GATE_IN_BUNDLE)}
    else:
        cv['gate'] = {'wins': False, 'why': 'too few rows or members'}
    return (v, cv)

def cell_names(b) -> list:
    sun_names = ['night'] + [f'solar {x[0]}-{x[1]}' for x in b['solar'][1:]]
    out = []
    for li in range(len(b['lead']) - 1):
        for si in range(len(b['solar'])):
            for r in b['regime']:
                out.append(f"lead {b['lead'][li]}-{b['lead'][li + 1]}|{sun_names[si]}|{r}")
    return out

def code_stamp() -> str:
    h = hashlib.sha256()
    for p in (os.path.join(HERE, 'blendtrain.py'), os.path.join(HERE, 'split.py'), os.path.join(common.PIPELINE, 'blend.py')):
        with open(p, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]

def train_region(F, fields=FIELDS, seed: int=SEED, params=None, n_min: int=N_MIN, log: Log | None=None, k_outer: int=K_OUTER, k_inner: int=K_INNER) -> tuple:
    members = list(F['_members'])
    base = F.get('_base', BASE)
    if base not in members:
        raise ValueError('blend train: the base has no rows')
    log = log if log is not None else Log()
    fields_out, cv = ({}, {})
    for f in fields:
        entry, c = train_field(F, f, members, seed, params, n_min, log, k_outer, k_inner)
        fields_out[f] = entry
        cv[f] = c
    groups = {g: [x for x in fs if x in fields_out] for g, fs in GROUPS.items()}
    entry = {'base': base, 'members': members, 'grid_step': GRID_STEP, 'max_age_h': {m: a for m, a in MAX_AGE_H.items() if m in members}, 'groups': {g: fs for g, fs in groups.items() if fs}, 'fields': fields_out}
    return (entry, cv)

def _cv_summary(cv: dict) -> dict:
    return {f: {k: c.get(k) for k in ('loss', 'rows', 'mae_blend', 'mae_best_raw', 'mae_best_debiased', 'mae_equal_debiased', 'ci95', 'bins_failed', 'pass')} for f, c in cv.items()}

def bundle_of(parts: dict, seed: int=SEED, pv: str='p1') -> tuple:
    if not parts:
        raise ValueError('blend train: no region trained')
    main_r = 'conus' if 'conus' in parts else sorted(parts)[0]
    day = max((int(np.max(F['day'])) for F, _, _ in parts.values() if n_of(F)))
    ymd = datetime.fromtimestamp(day * 86400, UTC).strftime('%Y%m%d')
    b = {'schema': blend.SCHEMA, 'id': '', 'code': code_stamp(), 'pairs': pv, 'seed': int(seed), 'ships': any((e['pass'] for _, entry, _ in parts.values() for e in entry['fields'].values())), 'regions': {r: entry for r, (_, entry, _) in parts.items()}, 'cv': _cv_summary(parts[main_r][2]), 'cv_regions': {r: _cv_summary(cv) for r, (_, _, cv) in parts.items()}}
    b = _clean(b)
    body = json.dumps(b, sort_keys=True, separators=(',', ':')).encode()
    b['id'] = f'bl1-{ymd}-{hashlib.sha256(body).hexdigest()[:8]}'
    data = json.dumps(b, sort_keys=True, separators=(',', ':')).encode()
    return (b, data, hashlib.sha256(data).hexdigest(), {r: cv for r, (_, _, cv) in parts.items()})

def train_bundle(F, fields=FIELDS, seed: int=SEED, params=None, n_min: int=N_MIN, log: Log | None=None, k_outer: int=K_OUTER, k_inner: int=K_INNER, pv: str='p1') -> tuple:
    entry, cv = train_region(F, fields, seed, params, n_min, log, k_outer, k_inner)
    b, data, sha, _ = bundle_of({F.get('_region', REGION): (F, entry, cv)}, seed, pv)
    return (b, data, sha, cv)

def report_text(bundle: dict, cv: dict, sha: str, region: str | None=None) -> str:
    region = region or (REGION if REGION in bundle['regions'] else sorted(bundle['regions'])[0])
    r = bundle['regions'][region]
    lines = [f"blend bundle {bundle['id']} (sha256 {sha[:12]}), region {region}: ships {('yes' if bundle['ships'] else 'no')}; members {', '.join(r['members'])}; base {r['base']}", '']
    for f, c in cv.items():
        e = r['fields'][f]
        v = c['verdict']['conditions']
        lines.append(f"{f} ({c['loss']}, {c.get('rows', 0)} test rows): {('passes' if e['pass'] else 'from the base')}")
        fmt = lambda x: '-' if x is None or (isinstance(x, float) and (not math.isfinite(x))) else f'{x:.4g}'
        lines.append(f"  blend {fmt(c.get('mae_blend'))}, best raw {fmt(c.get('mae_best_raw'))}, best debiased {fmt(c.get('mae_best_debiased'))}, equal-weight debiased {fmt(c.get('mae_equal_debiased'))}")
        ci = c.get('ci95') or [None, None]
        lines.append(f"  difference {fmt(c.get('diff'))} (95 % {fmt(ci[0])} to {fmt(ci[1])}); " + '; '.join((f"{k} {('ok' if d.get('ok') else 'fails')}" for k, d in v.items())))
        op = c.get('ops') or {}
        if op.get('rows'):
            lines.append(f"  operational rows {op['rows']}: blend {fmt(op.get('mae_blend'))}, best raw {fmt(op.get('mae_best_raw'))}; {('passes' if op.get('pass') else 'does not pass')} on them alone")
        if c.get('bins_failed'):
            lines.append(f"  worse in: {', '.join(('/'.join(x) for x in c['bins_failed']))}")
        g = c.get('gate', {})
        lines.append(f"  gate: {('beats' if g.get('wins') else 'does not beat')} the table" + (f" ({fmt(g.get('loss_gate'))} against {fmt(g.get('loss_table'))})" if 'loss_gate' in g else ''))
        tab = e['weights']['table'][blend.subset_key(r['members'])]
        nb = len(e['weights']['bins']['solar']) * len(e['weights']['bins']['regime'])
        for li in range(len(e['weights']['bins']['lead']) - 1):
            w = np.mean(tab[li * nb:(li + 1) * nb], 0)
            lines.append(f'  lead {LEAD_BINS[li]}-{LEAD_BINS[li + 1]} h: ' + ', '.join((f'{m} {x:.2f}' for m, x in zip(r['members'], w))))
        lines.append('')
    return '\n'.join(lines)

def cv_json(bundle, cv, sha, by_region: dict | None=None) -> bytes:
    strip = lambda c_: {f: {k: v for k, v in c.items() if k != 'oof_ops'} for f, c in c_.items()}
    out = {'id': bundle['id'], 'sha256': sha, 'ships': bundle['ships'], 'fields': strip(cv)}
    if by_region:
        out['regions'] = {r: {'fields': strip(c)} for r, c in by_region.items()}
    return json.dumps(_clean(out), sort_keys=True, indent=1).encode()

def oof_json(bundle, cv, by_region: dict | None=None) -> bytes:
    rows = lambda c_: {f: {k: np.asarray(a).tolist() for k, a in c['oof_ops'].items()} for f, c in c_.items() if 'oof_ops' in c}
    out = {'id': bundle['id'], 'fields': rows(cv)}
    if by_region:
        out['regions'] = {r: {'fields': rows(c)} for r, c in by_region.items()}
    return json.dumps(_clean(out), sort_keys=True, separators=(',', ':')).encode()

def static_points(lk, members=MEMBERS, pv: str='p1') -> dict:
    import pairs as P
    out = {}
    for m in members:
        parts = [P.from_parquet(P.STATIC_SCHEMA, lk.get(k)) for k in sorted(lk.list(f'derived/static/{pv}/model={m}/'))]
        if parts:
            out[m] = P.concat(P.STATIC_SCHEMA, parts)
    return out
SNAP_COLS = ('obs_id', 't_valid', 'slot', 'lat', 'lon', 'lead_h', 'solar_h', 'sun_elev', 'review')
PAIR_COLS = ('obs_id', 't_valid', 'model', 'run', 'build', 'src', 'replay', 'review')
OBS_COLS = ('obs_id', 't_valid', 'elev', 'o_t2', 'o_td2', 'o_ws', 'o_cu', 'o_cubase', 'review')

def read_day(lk, day: str, members=MEMBERS, pv: str='p1', cap: int | None=None, seed: int=SEED, base: str=BASE, box=None) -> tuple:
    import pairs as P
    cols = list(SNAP_COLS) + [f'{k}_{m}' for m in members for k in ('run', 'build', 'src', 'lead', 'age', 'regime')]
    snap = P.read_kind(lk, 'snap', 'snap', [day], pv, purpose='train', columns=cols)
    if not P.n_rows(snap):
        return None
    keep = np.asarray(snap[f'run_{base}'], dtype=object).astype(str) != ''
    if box is not None:
        w, s_, e, n = box
        lat, lon = (_f(snap['lat']), _f(snap['lon']))
        keep &= (lat >= s_) & (lat <= n) & (lon >= w) & (lon <= e)
    snap = {k: np.asarray(v)[keep] for k, v in snap.items()}
    if cap is not None:
        snap = sample(snap, cap, seed)
    pc = list(PAIR_COLS) + [f'f_{v}' for v in P.METAR_VARS]
    allp = P.read_kind(lk, 'pairs', 'metar', [day], pv, purpose='train', columns=pc)
    pairs = {}
    if P.n_rows(allp):
        mod = np.asarray(allp['model'], dtype=object)
        for m in members:
            sel = mod == m
            if sel.any():
                pairs[m] = {k: np.asarray(v)[sel] for k, v in allp.items()}
    obs = P.read_kind(lk, 'obs', 'metar', [day], pv, purpose='train', columns=list(OBS_COLS))
    return (snap, pairs, obs)

def load_frame(lk, days=None, members=MEMBERS, pv: str='p1', max_rows: int=MAX_ROWS, seed: int=SEED, base: str=BASE, box=None) -> dict | None:
    import pairs as P
    sdays = sorted({d for d, *_ in P.markers(lk, 'snap', 'snap', pv)})
    if days is not None:
        sdays = [d for d in sdays if d in set(days)]
    if not sdays:
        return None
    static = static_points(lk, members, pv)
    cap = max(1, max_rows // len(sdays))
    frames = []
    for d in sdays:
        got = read_day(lk, d, members, pv, cap, seed, base, box)
        if got is None:
            continue
        F = assemble(got[0], got[1], got[2], static, members, base)
        if n_of(F):
            frames.append(F)
    return concat_frames(frames, members)

def concat_frames(frames: list, members=MEMBERS) -> dict | None:
    if not frames:
        return None
    keys = [k for k in frames[0] if not k.startswith('_')]
    out = {k: np.concatenate([f[k] for f in frames]) for k in keys}
    have = {m for f in frames for m in f['_members']}
    out['_members'] = [m for m in members if m in have]
    out['_base'] = frames[0]['_base']
    return out
OUT_FILES = ('cv.json', 'oof_ops.json', 'report.md', 'bundle.json')

def write_bundle(lk, bundle, data, cv_bytes, text, oof_bytes=b'{}') -> list:
    base = f"derived/models/blend/{bundle['id']}/"
    out = []
    blobs = {'cv.json': cv_bytes, 'oof_ops.json': oof_bytes, 'report.md': text.encode(), 'bundle.json': data}
    for name, blob in ((k + '.lenc', blobs[k]) for k in OUT_FILES):
        r = lk.put(base + name, blob)
        out.append(r.status)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The blend's training: its bundle, its ship test and its CV report")
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('run')
    a1.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    a1.add_argument('--out', help="also write the folder's files plain here (bundle.json, cv.json, oof_ops.json, report.md)")
    a1.add_argument('--seed', type=int, default=SEED)
    a1.add_argument('--from', dest='d0', help='first valid day, YYYY-MM-DD')
    a1.add_argument('--to', dest='d1', help='last valid day, YYYY-MM-DD')
    a = ap.parse_args(argv)
    import lake as lakemod
    lk = lakemod.from_env('LAKE', a.local)
    if lk is None:
        print('blend train: no lake (LAKE_S3_* not set): nothing trained', flush=True)
        return 0
    import pairs as P
    days = None
    if a.d0 or a.d1:
        days = [d for d, *_ in P.markers(lk, 'snap', 'snap') if (not a.d0 or d >= a.d0) and (not a.d1 or d <= a.d1)]
    log, parts = (Log(), {})
    for r, cfg in REGIONS.items():
        F = load_frame(lk, days, members=cfg['members'], seed=a.seed, base=cfg['base'], box=cfg['box'])
        if F is None or not n_of(F):
            print(f"blend train: {r}: no snapshot rows its base ({cfg['base']}) serves: left out", flush=True)
            continue
        F['_region'] = r
        try:
            entry, cv = train_region(F, seed=a.seed, log=log)
        except ValueError as e:
            print(f'blend train: {r}: {e}: left out', flush=True)
            continue
        parts[r] = (F, entry, cv)
        print(f"blend train: {r}: {n_of(F)} rows; passing {', '.join((f for f, e in entry['fields'].items() if e['pass'])) or 'none'}", flush=True)
    if not parts:
        print('blend train: no region has rows: nothing trained', flush=True)
        return 0
    bundle, data, sha, cvs = bundle_of(parts, a.seed)
    main_r = 'conus' if 'conus' in cvs else sorted(cvs)[0]
    text = '\n'.join((report_text(bundle, cvs[r], sha, r) for r in sorted(cvs, key=lambda r: r != main_r)))
    cvb = cv_json(bundle, cvs[main_r], sha, cvs)
    oof = oof_json(bundle, cvs[main_r], cvs)
    st = write_bundle(lk, bundle, data, cvb, text, oof)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        blobs = {'cv.json': cvb, 'oof_ops.json': oof, 'report.md': text.encode(), 'bundle.json': data}
        for name, blob in blobs.items():
            with open(os.path.join(a.out, name), 'wb') as fh:
                fh.write(blob)
    print(f"blend train: {sum((n_of(F) for F, _, _ in parts.values()))} rows, {len(log.fits)} fits; bundle {bundle['id']} sha256 {sha}; regions {', '.join(parts)}; ships {('yes' if bundle['ships'] else 'no')}; written {', '.join(st)}", flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
