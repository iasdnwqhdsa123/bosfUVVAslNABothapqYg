from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import airpairs
import common
import lake as lakem
import score
SV = 'air'
MODELS_PREFIX = 'derived/models/air'
SCHEMA = 'air-head/1'
MIN_PIECES = 3000
GAIN_MIN = 0.01
REGION_MIN, REGION_SLACK = (500, 0.01)
FIT_BANDS = (0, 1, 2)
AIR = ('ws', 'zi', 'depth', 'cb_agl', 'sun', 'cloud', 'nlow', 'blapse', 'dth200', 'dth500', 'ws_bl', 'ws_300', 'wd_300', 'ws_top', 'ws_at', 'agl', 'zr', 'band', 'step', 'solar_h', 'lat')
PHYS = ('ridge_g', 'sink', 'conv', 'line_km', 'wblmax', 'wv_at', 'wavemax')
COARSE = AIR + PHYS
FINE = ('ridge_site',)
MONO = {'ridge_site': 1, 'ridge_g': 1, 'sink': -1, 'conv': 1, 'line_km': -1, 'wblmax': 1, 'wv_at': 1}
FINE_BINS = {'ridge_site': [round(0.1 * i, 2) for i in range(41)]}
CLAMP = (-6.0, 6.0)
TREE = dict(loss='absolute_error', max_iter=150, learning_rate=0.08, max_leaf_nodes=8, min_samples_leaf=100, l2_regularization=1.0, random_state=0)

def frame(cols: dict) -> tuple[dict, np.ndarray]:
    m = np.isin(cols['band'], FIT_BANDS) & np.isfinite(cols['netto'])
    T = {k: v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v for k, v in cols.items()}
    T['line_km'] = np.where(np.isfinite(T['line_km']), T['line_km'], airpairs.FAR_KM)
    T['band'] = T['band'].astype(np.float64)
    T['day'] = np.array([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in T['t0']])
    return (T, m)

def X(T: dict, names) -> np.ndarray:
    n = len(T['netto']) if 'netto' in T else len(next(iter(T.values())))
    return np.column_stack([np.asarray(T[k], np.float64) if k in T else np.full(n, np.nan) for k in names])

def model(names, constrained: bool=True):
    from sklearn.ensemble import HistGradientBoostingRegressor
    mono = [MONO.get(n, 0) for n in names]
    cst = None
    fine = [n for n in names if n in FINE]
    if constrained and len(fine) > 1:
        coarse = [i for i, n in enumerate(names) if n not in FINE]
        cst = [set(coarse) | {names.index(f)} for f in fine]
    return HistGradientBoostingRegressor(monotonic_cst=mono, interaction_cst=cst, **TREE)

def layers(T: dict) -> np.ndarray:
    z = lambda k: np.nan_to_num(np.asarray(T[k], np.float64), nan=0.0) if k in T else np.zeros(len(T['netto']))
    return np.maximum(np.maximum(z('ridge_site'), z('ridge_g')), 0) - np.maximum(z('sink'), 0) + np.maximum(z('conv'), 0) + z('wv_at')

def folds(days, k: int=5, seed: int=0) -> list[set]:
    d = np.array(sorted(set(days)))
    np.random.default_rng(seed).shuffle(d)
    return [set(d[i::k]) for i in range(k)]
SETS = {'air': (AIR, False), 'full': (COARSE + FINE, False), 'head': (COARSE + FINE, True)}

def heldout(T: dict) -> dict:
    n = len(T['netto'])
    out = {k: np.full(n, np.nan) for k in SETS}
    for F in folds(T['day']):
        te = np.isin(T['day'], list(F))
        if not te.any() or te.all():
            continue
        for k, (names, con) in SETS.items():
            m = model(list(names), con).fit(common.no_empty(X(T, names)[~te]), T['netto'][~te])
            out[k][te] = m.predict(X(T, names)[te])
    out['layers'] = layers(T)
    return out

def boot_gain(T: dict, ea: np.ndarray, eb: np.ndarray, n: int=1000, seed: int=1) -> tuple[float, float]:
    days = np.array(sorted(set(T['day'])))
    idx = {d: np.where(T['day'] == d)[0] for d in days}
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n):
        pick = np.concatenate([idx[d] for d in rng.choice(days, len(days))])
        g.append(ea[pick].mean() - eb[pick].mean())
    return (float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)))

def terms(m, names, row: np.ndarray) -> dict:
    names = list(names)
    base = row.copy()
    for f in FINE:
        base[names.index(f)] = FINE_BINS[f][0]
    at0 = float(m.predict(base[None, :])[0])
    out = {}
    for f in FINE:
        B = np.repeat(base[None, :], len(FINE_BINS[f]), 0)
        B[:, names.index(f)] = FINE_BINS[f]
        out[f] = [round(float(v) - at0, 4) for v in m.predict(B)]
    return {'at0': at0, 'terms': out}

def response(m, names, T: dict) -> dict:
    out = {}
    Xa = X(T, names)
    for f in FINE:
        j = list(names).index(f)
        vals = []
        for b in FINE_BINS[f][::5]:
            Xb = Xa.copy()
            Xb[:, j] = b
            vals.append([b, float(m.predict(Xb).mean())])
        z = vals[0][1]
        out[f] = [[b, round(v - z, 3)] for b, v in vals]
    return out

def refit(cols: dict) -> dict:
    T, _ = frame(cols)
    n = len(T['netto'])
    out = {'pieces': n, 'days': len(set(T['day'])), 'models': sorted(set(map(str, T['model']))), 'bands': {}, 'regions': {}}
    if n < MIN_PIECES:
        out['pass'], out['why'] = (False, f'{n} pieces, under the {MIN_PIECES} the fit needs')
        return out
    P = heldout(T)
    E = {k: np.abs(P[k] - T['netto']) for k in P}
    ok = np.isfinite(E['head'])
    E = {k: v[ok] for k, v in E.items()}
    Tk = {k: v[ok] if isinstance(v, np.ndarray) and len(v) == len(ok) else v for k, v in T.items()}
    out['mae'] = {k: float(v.mean()) for k, v in E.items()}
    out['ci_layers'] = boot_gain(Tk, E['layers'], E['head'])
    out['ci_air'] = boot_gain(Tk, E['air'], E['head'])
    worse = []
    for name, col, groups in (('bands', 'band', sorted(set(Tk['band']))), ('regions', 'region', sorted(set(map(str, Tk['region']))))):
        for g in groups:
            s = np.asarray(Tk[col]).astype(str) == str(g)
            if s.sum() >= REGION_MIN:
                out[name][str(g)] = {k: float(E[k][s].mean()) for k in E} | {'n': int(s.sum())}
                if E['head'][s].mean() > E['layers'][s].mean() + REGION_SLACK:
                    worse.append(f'{name[:-1]} {g}')
    out['worse'] = worse
    g_l, g_a = (out['mae']['layers'] - out['mae']['head'], out['mae']['air'] - out['mae']['head'])
    if g_l < GAIN_MIN or out['ci_layers'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the head gains {g_l:.3f} m/s on the layers (95 % {out['ci_layers'][0]:.3f} to {out['ci_layers'][1]:.3f})")
    elif g_a < GAIN_MIN or out['ci_air'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the physics add {g_a:.3f} m/s to the air alone (95 % {out['ci_air'][0]:.3f} to {out['ci_air'][1]:.3f})")
    elif worse:
        out['pass'], out['why'] = (False, 'worse than the layers held out in ' + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f'held out the head gains {g_l:.3f} m/s on the layers and {g_a:.3f} on the air alone, no band or region worse')
    names = list(COARSE + FINE)
    m = model(names, True).fit(common.no_empty(X(T, names)), T['netto'])
    out['response'] = response(m, names, T)
    out['_model'] = (m, names)
    return out

def bundle(r: dict, month: str) -> dict | None:
    if '_model' not in r:
        return None
    import train
    m, names = r['_model']
    e = train.export(m, names)
    return {'schema': SCHEMA, 'month': month, 'pass': bool(r.get('pass')), 'coarse': list(COARSE), 'fine': list(FINE), 'fine_bins': FINE_BINS, 'mono': MONO, 'clamp': list(CLAMP), 'bands': list(FIT_BANDS), **e}

def report_md(month: str, r: dict) -> str:
    L = [f'# The air-motion head, {month}', '', f"{r['pieces']} straight pieces on {r['days']} days ({', '.join(r['models'])}).", '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'mae' in r:
        mae = r['mae']
        L += ['', 'Mean absolute error of the netto (m/s), held out by date:', '', '| | the layers as they are | the air alone | every input, no constraint | the head |', '| --- | --- | --- | --- | --- |', f"| all | {mae['layers']:.3f} | {mae['air']:.3f} | {mae['full']:.3f} | {mae['head']:.3f} |"]
        band = {'0.0': 'low (ridge band)', '1.0': 'in the thermal layer', '2.0': 'above the layer (wave band)'}
        for name, label in (('bands', band), ('regions', {})):
            for g, v in sorted(r[name].items()):
                L.append(f"| {label.get(g, g)} ({v['n']}) | {v['layers']:.3f} | {v['air']:.3f} | {v['full']:.3f} | {v['head']:.3f} |")
    for f, rows in (r.get('response') or {}).items():
        L += ['', f'What the head makes of {f} (m/s of netto over none, averaged over the pieces):', '', '| input | netto |', '| --- | --- |']
        L += [f'| {b} | {v:+.2f} |' for b, v in rows]
    return '\n'.join(L) + '\n'

def run(lk, today: date, again: bool=False, counts=None) -> dict | None:
    counts = counts if counts is not None else {}
    month = (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    cols = airpairs.load_all(lk, counts)
    if not cols:
        counts['no_pairs'] = 1
        return None
    r = refit(cols)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    b = bundle(r, month)
    r.pop('_model', None)
    pre = f'{score.PREFIX}/{SV}/{month}/{ts}'
    if b is not None:
        lk.put(f'{MODELS_PREFIX}/{month}/{ts}.json.lenc', json.dumps(b, separators=(',', ':')).encode(), strict=False)
        r['bundle'] = f'{MODELS_PREFIX}/{month}/{ts}.json.lenc'
    lk.put(pre + '.json.lenc', json.dumps(r, sort_keys=True, default=float).encode(), strict=False)
    lk.put(pre + '.md.lenc', report_md(month, r).encode(), strict=False)
    counts['written'] = 1
    r['month'] = month
    return r

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='The air-motion head, refitted monthly')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('run')
    a.add_argument('--again', action='store_true')
    a.add_argument('--day', help='today, YYYY-MM-DD (default: today, UTC)')
    a.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    f = sub.add_parser('fetch')
    f.add_argument('--out', required=True)
    f.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    x = ap.parse_args(argv)
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('airfit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('airfit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('airfit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('airfit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
