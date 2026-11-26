from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import airfit
import common
import lake as lakem
import score
import windpairs as WP
SV = 'wind'
MODELS_PREFIX = 'derived/models/wind'
SCHEMA = 'wind-head/1'
MIN_ROWS = 1500
GAIN_MIN = 0.2
GROUP_MIN, GROUP_SLACK = (200, 0.1)
AGL_MIN = 30.0
RES_MAX = 150.0
LEVELS = (10.0, 100.0, 300.0)
COARSE = ('ws10', 'ws100', 'ws300', 'turn', 'ws_bl', 'wstar', 'zi', 'nlow', 'blapse', 'dth200', 'sun', 'heat', 'cloudlow', 'agl', 'dz_ground', 'relief', 'solar_h', 'season', 'lat')
FINE: tuple = ()
TREE = dict(airfit.TREE, max_iter=120, min_samples_leaf=40)
CLAMP = (-25.0, 25.0)

def model_wind(c: dict) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(np.asarray(c['agl'], np.float64), LEVELS[0], LEVELS[-1])
    lz, L = (np.log(z), np.log(LEVELS))
    out = []
    for comp in ('u', 'v'):
        a, b, d = (np.asarray(c[k], np.float64) for k in (f'{comp}_sfc', f'{comp}_g100', f'{comp}_g300'))
        lo = a + (b - a) * (lz - L[0]) / (L[1] - L[0])
        hi = b + (d - b) * (lz - L[1]) / (L[2] - L[1])
        w = np.where(lz <= L[1], lo, hi)
        out.append(np.where(np.isfinite(w), w, b))
    return (out[0], out[1])

def frame(c: dict) -> dict:
    agl = np.asarray(c['agl'], np.float64)
    m = (agl >= AGL_MIN) & (agl <= WP.LOW_TOP) & (np.asarray(c['q_res'], np.float64) <= RES_MAX)
    m &= np.isfinite(c['u']) & np.isfinite(c['v']) & np.isfinite(c['u_g100']) & np.isfinite(c['v_g100'])
    T = {k: v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v for k, v in c.items()}
    mu, mv = model_wind(T)
    S = np.hypot(mu, mv)
    eu, ev = (np.where(S > 0.1, mu / np.maximum(S, 0.1), 1.0), np.where(S > 0.1, mv / np.maximum(S, 0.1), 0.0))
    T['mu'], T['mv'], T['eu'], T['ev'] = (mu, mv, eu, ev)
    T['along'] = T['u'] * eu + T['v'] * ev
    T['across'] = -T['u'] * ev + T['v'] * eu
    sp = lambda a, b: np.hypot(np.asarray(T[a], np.float64), np.asarray(T[b], np.float64))
    T['ws10'], T['ws100'], T['ws300'], T['ws_bl'] = (sp('u_sfc', 'v_sfc'), sp('u_g100', 'v_g100'), sp('u_g300', 'v_g300'), sp('u_blavg', 'v_blavg'))
    a10 = np.arctan2(np.asarray(T['v_sfc'], np.float64), np.asarray(T['u_sfc'], np.float64))
    a300 = np.arctan2(np.asarray(T['v_g300'], np.float64), np.asarray(T['u_g300'], np.float64))
    T['turn'] = np.degrees((a300 - a10 + np.pi) % (2 * np.pi) - np.pi)
    T['dz_ground'] = np.asarray(T['ground'], np.float64) - np.asarray(T['elevation'], np.float64)
    T['dth200'] = np.asarray(T['dth_m200'], np.float64)
    T['wstar'] = np.asarray(T['wstar'], np.float64)
    T['season'] = np.cos(2 * np.pi * (np.asarray(T['month'], np.float64) - 7) / 12)
    T['day'] = np.array([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in T['t']])
    return T

def X(T: dict, names) -> np.ndarray:
    n = len(T['u'])
    return np.column_stack([np.asarray(T[k], np.float64) if k in T else np.full(n, np.nan) for k in names])

def fit_pair(T: dict, rows: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xr = common.no_empty(X(T, COARSE)[rows])
    return tuple((HistGradientBoostingRegressor(**TREE).fit(Xr, T[k][rows]) for k in ('along', 'across')))

def predict(T: dict, pair, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xr = X(T, COARSE)[rows]
    a, c = (np.clip(m.predict(Xr), *CLAMP) for m in pair)
    eu, ev = (T['eu'][rows], T['ev'][rows])
    return (a * eu - c * ev, a * ev + c * eu)

def heldout(T: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(T['u'])
    hu, hv = (np.full(n, np.nan), np.full(n, np.nan))
    for F in airfit.folds(T['day']):
        te = np.isin(T['day'], list(F))
        if not te.any() or te.all():
            continue
        pair = fit_pair(T, ~te)
        hu[te], hv[te] = predict(T, pair, te)
    return (hu, hv)

def refit(cols: dict) -> dict:
    T = frame(cols)
    n = len(T['u'])
    out = {'rows': n, 'days': len(set(T['day'])), 'models': sorted(set(map(str, T['model']))), 'groups': {}}
    if n < MIN_ROWS:
        out['pass'], out['why'] = (False, f'{n} low drifts, under the {MIN_ROWS} the fit needs')
        return out
    hu, hv = heldout(T)
    ok = np.isfinite(hu)
    Tk = {k: v[ok] if isinstance(v, np.ndarray) and len(v) == len(ok) else v for k, v in T.items()}
    E = {'model': np.hypot(Tk['mu'] - Tk['u'], Tk['mv'] - Tk['v']), 'head': np.hypot(hu[ok] - Tk['u'], hv[ok] - Tk['v'])}
    out['mae'] = {k: float(v.mean()) for k, v in E.items()}
    obs_s, mod_s, head_s = (np.hypot(Tk['u'], Tk['v']), np.hypot(Tk['mu'], Tk['mv']), np.hypot(hu[ok], hv[ok]))
    out['speed_bias'] = {'model': float((mod_s - obs_s).mean()), 'head': float((head_s - obs_s).mean())}
    out['ci'] = airfit.boot_gain(Tk, E['model'], E['head'])
    worse = []
    for name, col in (('region', np.asarray(Tk['region']).astype(str)), ('model', np.asarray(Tk['model']).astype(str)), ('height', np.where(Tk['agl'] < 150, 'under 150 m', '150 m and over'))):
        for g in sorted(set(col.tolist())):
            s = col == g
            if s.sum() < GROUP_MIN or not g:
                continue
            out['groups'][f'{name} {g}'] = {k: float(E[k][s].mean()) for k in E} | {'n': int(s.sum())}
            if E['head'][s].mean() > E['model'][s].mean() + GROUP_SLACK:
                worse.append(f'{name} {g}')
    out['worse'] = worse
    g = out['mae']['model'] - out['mae']['head']
    if g < GAIN_MIN or out['ci'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the head gains {g:.2f} m/s on the model's wind (95 % {out['ci'][0]:.2f} to {out['ci'][1]:.2f})")
    elif worse:
        out['pass'], out['why'] = (False, "worse than the model's wind held out in " + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f"held out the head gains {g:.2f} m/s on the model's wind, no region, model or height worse")
    out['_model'] = fit_pair(T, np.ones(n, bool))
    return out

def bundle(r: dict, month: str) -> dict | None:
    if '_model' not in r:
        return None
    import train
    a, c = r['_model']
    return {'schema': SCHEMA, 'month': month, 'pass': bool(r.get('pass')), 'coarse': list(COARSE), 'fine': list(FINE), 'frame': "the model's wind at the height: along, across (+ left)", 'clamp': list(CLAMP), 'levels': list(LEVELS), 'along': train.export(a, list(COARSE)), 'across': train.export(c, list(COARSE))}

def report_md(month: str, r: dict) -> str:
    L = [f'# The wind head, {month}', '', f"{r['rows']} low drifts of circling gliders on {r['days']} days ({', '.join(r['models'])}).", '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'mae' in r:
        m, b = (r['mae'], r['speed_bias'])
        L += ['', "The vector error's mean (m/s), held out by date (the speed's bias beside):", '', "| | the model's wind at the height | the head |", '| --- | --- | --- |', f"| all | {m['model']:.2f} ({b['model']:+.2f}) | {m['head']:.2f} ({b['head']:+.2f}) |"]
        L += [f"| {g} ({v['n']}) | {v['model']:.2f} | {v['head']:.2f} |" for g, v in sorted(r['groups'].items())]
    return '\n'.join(L) + '\n'

def run(lk, today: date, again: bool=False, counts=None) -> dict | None:
    counts = counts if counts is not None else {}
    month = (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    cols = WP.load_all(lk, counts)
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
    ap = argparse.ArgumentParser(description='The wind head, refitted monthly')
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
        print('windfit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('windfit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('windfit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('windfit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
