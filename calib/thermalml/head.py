from __future__ import annotations
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import airfit
import common
import score
SV = 'thermal'
MODELS_PREFIX = 'derived/models/thermal'
SCHEMA = 'thermal-head/1'
MIN_ROWS = 800
GAIN_MIN = 0.05
GROUP_MIN, GROUP_SLACK = (150, 0.05)
BAND_MID = 250.0
COARSE = ('wstar', 'zi', 'depth', 'cucover', 'wind', 'regime_n', 'agl', 'x', 'ground', 'relief', 'solar_h', 'season', 'lat', 'core')
MONO = {'core': 1, 'wstar': 1}
TREE = dict(airfit.TREE, max_iter=150, min_samples_leaf=30)
CLAMP = (0.0, 8.0)
TIMEOUT_S = 40 * 60
REGIME_N = {'cu': 1, 'blue': 0, 'weak': -1}

def script() -> str | None:
    return next(iter(sorted(glob.glob(os.path.join(HERE, 'core_eval.*')))), None)

def rows_of(body: dict) -> dict | None:
    cols = body.get('cols') or []
    R = [dict(zip(cols, r)) for r in body.get('rows') or []]
    R = [r for r in R if r.get('y') == 'air' and r.get('wstar') is not None and ((r.get('air_adj_q') if r.get('air_adj_q') is not None else r.get('air_q')) is not None)]
    if not R:
        return None
    num = lambda k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in R], np.float64)
    T = {k: num(k) for k in ('la', 'lo', 'hour_bin', 'band_m', 'climbs', 'wstar', 'zi', 'depth', 'cucover', 'wind')}
    T['y'] = np.array([float(r['air_adj_q'] if r.get('air_adj_q') is not None else r['air_q']) for r in R])
    T['regime_n'] = np.array([REGIME_N.get(str(r.get('regime')), np.nan) for r in R], np.float64)
    T['regime'] = np.array([str(r.get('regime') or '') for r in R])
    T['model'] = np.array([str(r.get('model') or '') for r in R])
    T['day'] = np.array([str(r['sd']) for r in R])
    T['solar_h'] = T['hour_bin'] + 1.0
    mon = np.array([int(d[5:7]) for d in T['day']], np.float64)
    T['season'] = np.cos(2 * np.pi * (mon - 7) / 12) * np.sign(T['la'])
    T['lat'] = T['la']
    T['z'] = T['band_m'] + BAND_MID
    T['region'] = np.array([str(common.place_region(o, a)[0] or '') for a, o in zip(T['la'], T['lo'])])
    return T

def site_values(T: dict, work: str, run=subprocess.run, node: str='node') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(T['y'])
    G, Rl, C = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    sc = script()
    if not sc or not n:
        return (G, Rl, C)
    f = lambda x: float(x) if math.isfinite(x) else None
    q = [[float(a), float(o), float(z), f(w), f(d), 1 if r == 'cu' else 0] for a, o, z, w, d, r in zip(T['la'], T['lo'], T['z'], T['wstar'], T['depth'], T['regime'])]
    os.makedirs(work, exist_ok=True)
    fin, fout = (os.path.join(work, 'core_in.json'), os.path.join(work, 'core_out.json'))
    with open(fin, 'w') as fh:
        json.dump({'cell': 0.1, 'q': q}, fh)
    r = run([node, sc, fin, fout], capture_output=True, text=True, timeout=TIMEOUT_S)
    if r.returncode != 0 or not os.path.exists(fout):
        raise RuntimeError(f"core_eval: {(r.stderr or '')[-400:]}")
    with open(fout) as fh:
        o = json.load(fh)
    for k, (g, rl, c) in enumerate(zip(o['ground'], o['relief'], o['core'])):
        G[k] = np.nan if g is None else g
        Rl[k] = np.nan if rl is None else rl
        C[k] = np.nan if c is None else c
    for fn in (fin, fout):
        os.remove(fn)
    return (G, Rl, C)

def frame(T: dict, G, Rl, C) -> dict:
    T = dict(T, ground=G, relief=Rl, core=C)
    T['agl'] = T['z'] - G
    T['x'] = np.divide(T['agl'], T['depth'], out=np.full(len(T['depth']), np.nan), where=T['depth'] > 50)
    ok = np.isfinite(C) & (T['agl'] > 0)
    return {k: v[ok] if isinstance(v, np.ndarray) and len(v) == len(ok) else v for k, v in T.items()}

def X(T: dict, names=COARSE) -> np.ndarray:
    return np.column_stack([np.asarray(T[k], np.float64) for k in names])

def model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(monotonic_cst=[MONO.get(k, 0) for k in COARSE], **TREE)

def refit(T: dict) -> dict:
    n = len(T['y'])
    out = {'rows': n, 'days': len(set(T['day'])), 'climbs': int(np.nansum(T['climbs'])) if n else 0, 'models': sorted(set(T['model'])), 'groups': {}}
    if n < MIN_ROWS:
        out['pass'], out['why'] = (False, f'{n} rows, under the {MIN_ROWS} the fit needs')
        return out
    w = np.sqrt(np.maximum(np.nan_to_num(T['climbs'], nan=1.0), 1.0))
    h = np.full(n, np.nan)
    for F in airfit.folds(T['day']):
        te = np.isin(T['day'], list(F))
        if not te.any() or te.all():
            continue
        m = model().fit(common.no_empty(X(T)[~te]), T['y'][~te], sample_weight=w[~te])
        h[te] = np.clip(m.predict(X(T)[te]), *CLAMP)
    ok = np.isfinite(h)
    Tk = {k: v[ok] if isinstance(v, np.ndarray) and len(v) == len(ok) else v for k, v in T.items()}
    E = {'core': np.abs(Tk['core'] - Tk['y']), 'head': np.abs(h[ok] - Tk['y'])}
    out['mae'] = {k: float(v.mean()) for k, v in E.items()}
    out['bias'] = {'core': float((Tk['core'] - Tk['y']).mean()), 'head': float((h[ok] - Tk['y']).mean())}
    out['ci'] = airfit.boot_gain(Tk, E['core'], E['head'])
    xb = np.where(Tk['x'] < 0.33, 'low third', np.where(Tk['x'] < 0.67, 'middle third', 'top third'))
    worse = []
    for name, col in (('regime', Tk['regime']), ('height', xb), ('region', Tk['region']), ('model', Tk['model'])):
        for g in sorted(set(col.tolist())):
            s = col == g
            if s.sum() < GROUP_MIN or not g:
                continue
            out['groups'][f'{name} {g}'] = {k: float(E[k][s].mean()) for k in E} | {'n': int(s.sum())}
            if E['head'][s].mean() > E['core'][s].mean() + GROUP_SLACK:
                worse.append(f'{name} {g}')
    out['worse'] = worse
    s = (Tk['wstar'] > 0.5) & (Tk['x'] > 0.2) & (Tk['x'] < 0.6)
    out['rise_over_wstar'] = float(np.median(Tk['y'][s] / Tk['wstar'][s])) if s.sum() >= 30 else None
    g = out['mae']['core'] - out['mae']['head']
    if g < GAIN_MIN or out['ci'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the head gains {g:.3f} m/s on the site's core (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f})")
    elif worse:
        out['pass'], out['why'] = (False, "worse than the site's core held out in " + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f"held out the head gains {g:.3f} m/s on the site's core, no regime, height, region or model worse")
    out['_model'] = model().fit(common.no_empty(X(T)), T['y'], sample_weight=w)
    return out

def bundle(r: dict, month: str) -> dict | None:
    if '_model' not in r:
        return None
    import train
    e = train.export(r['_model'], list(COARSE))
    return {'schema': SCHEMA, 'month': month, 'pass': bool(r.get('pass')), 'coarse': list(COARSE), 'fine': [], 'fine_bins': {}, 'mono': MONO, 'clamp': list(CLAMP), 'truth': 'air_adj_q', **e}

def report_md(month: str, r: dict) -> str:
    L = [f"# The thermals' strength, {month}", '', f"{r['rows']} rows of the air's rise ({r['climbs']} climbs) on {r['days']} days ({', '.join(r['models']) or 'no model'}), the thermal learning's.", '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'mae' in r:
        m, b = (r['mae'], r['bias'])
        L += ['', "Mean absolute error of the air's rise (m/s), held out by date (its bias beside):", '', "| | the site's core | the head |", '| --- | --- | --- |', f"| all | {m['core']:.3f} ({b['core']:+.3f}) | {m['head']:.3f} ({b['head']:+.3f}) |"]
        L += [f"| {g} ({v['n']}) | {v['core']:.3f} | {v['head']:.3f} |" for g, v in sorted(r['groups'].items())]
        if r.get('rise_over_wstar') is not None:
            L += ['', f"The air's rise over W* in the layer's middle (median): {r['rise_over_wstar']:.2f} (the site's CORE: 1.3)."]
    return '\n'.join(L) + '\n'

def month_fit(lk, strength_out: dict, today: date, n: dict | None=None, again: bool=False, values=site_values) -> dict | None:
    n = n if n is not None else {}
    month = (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        n['head_already'] = 1
        return None
    T = rows_of(strength_out or {})
    if T is None:
        n['head_no_rows'] = 1
        return None
    with tempfile.TemporaryDirectory() as work:
        G, Rl, C = values(T, work)
    r = refit(frame(T, G, Rl, C))
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    b = bundle(r, month)
    r.pop('_model', None)
    pre = f'{score.PREFIX}/{SV}/{month}/{ts}'
    if b is not None:
        lk.put(f'{MODELS_PREFIX}/{month}/{ts}.json.lenc', json.dumps(b, separators=(',', ':')).encode(), strict=False)
        r['bundle'] = f'{MODELS_PREFIX}/{month}/{ts}.json.lenc'
    lk.put(pre + '.json.lenc', json.dumps(r, sort_keys=True, default=float).encode(), strict=False)
    lk.put(pre + '.md.lenc', report_md(month, r).encode(), strict=False)
    n['head'] = f"{month}: {('pass' if r.get('pass') else 'no change')}"
    r['month'] = month
    return r

def fetch(lk, out: str, counts: dict | None=None) -> list:
    return score.fetch(lk, out, counts if counts is not None else {}, sv=SV)
