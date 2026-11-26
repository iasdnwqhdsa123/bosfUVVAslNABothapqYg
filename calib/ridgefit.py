from __future__ import annotations
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import airpairs
import convfit
import lake as lakem
import score
SV = 'ridge'
MIN_PIECES = 2000
MAX_PIECES = 20000
GAIN_MIN = 0.005
REGION_MIN, REGION_SLACK = (300, 0.01)
WIND_MIN = 4.0
IN_USE = {'decay': 2.5, 'shape': 0.65, 'share': 0.5}
SETS = [{'decay': d, 'shape': s} for d in (2.0, 2.5, 3.0) for s in (0.45, 0.65, 0.85)]
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
TIMEOUT_S = 50 * 60

def script() -> str | None:
    return next(iter(sorted(glob.glob(os.path.join(HERE, 'ridge_eval.*')))), None)

def frame(cols: dict, seed: int=5) -> dict:
    n = len(cols['netto'])
    col = lambda k, fill=np.nan: np.asarray(cols[k], np.float64) if k in cols else np.full(n, fill)
    ws, wd = (col('ws_300'), col('wd_300'))
    m = (np.asarray(cols['band']) == 0) & np.isfinite(col('netto')) & np.isfinite(ws) & (ws >= WIND_MIN) & np.isfinite(wd)
    m &= np.isfinite(col('alt')) & ~(col('pre', 0.0) == 1)
    T = {k: np.asarray(v)[m] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in cols.items()}
    T['netto'] = np.asarray(T['netto'], np.float64) - np.nan_to_num(np.asarray(T.get('off', np.zeros(int(m.sum()))), np.float64), nan=0.0)
    T['u'] = -ws[m] * np.sin(np.radians(wd[m]))
    T['v'] = -ws[m] * np.cos(np.radians(wd[m]))
    T['day'] = np.array([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in T['t0']])
    if len(T['netto']) > MAX_PIECES:
        days = sorted(set(T['day'].tolist()))
        random.Random(seed).shuffle(days)
        keep, got = (set(), 0)
        for d in days:
            k = int((T['day'] == d).sum())
            if got + k > MAX_PIECES:
                continue
            keep.add(d)
            got += k
        s = np.isin(T['day'], list(keep))
        T = {k: v[s] if isinstance(v, np.ndarray) and len(v) == len(s) else v for k, v in T.items()}
    return T

def site_values(T: dict, work: str, run=subprocess.run, node: str='node') -> tuple[np.ndarray, np.ndarray]:
    n = len(T['netto'])
    R, L = (np.full((n, len(SETS)), np.nan), np.full((n, len(SETS)), np.nan))
    sc = script()
    if not sc or not n:
        return (R, L)
    nl = np.asarray(T.get('nlow', np.full(n, np.nan)), np.float64)
    q = [[float(a), float(o), float(z), round(float(u), 3), round(float(v), 3), float(x) if math.isfinite(x) else None] for a, o, z, u, v, x in zip(T['lat'], T['lon'], T['alt'], T['u'], T['v'], nl)]
    os.makedirs(work, exist_ok=True)
    fin, fout = (os.path.join(work, 'ridge_in.json'), os.path.join(work, 'ridge_out.json'))
    with open(fin, 'w') as f:
        json.dump({'sets': SETS, 'q': q}, f)
    r = run([node, sc, fin, fout], capture_output=True, text=True, timeout=TIMEOUT_S)
    if r.returncode != 0 or not os.path.exists(fout):
        raise RuntimeError(f"ridge_eval: {(r.stderr or '')[-400:]}")
    with open(fout) as f:
        o = json.load(f)
    for i, (a, b) in enumerate(zip(o['ridge'], o['lee'])):
        if a is not None and b is not None:
            R[i], L[i] = (a, b)
    for fn in (fin, fout):
        os.remove(fn)
    return (R, L)

def _mae_fit(y, R, L, k) -> tuple[float, float]:
    x = R - k * L
    b = float(np.median(y - x))
    return (b, float(np.mean(np.abs(y - (b + x)))))

def best(y, R, L) -> tuple[int, float, float]:
    top = None
    for s in range(len(SETS)):
        for k in SHARES:
            b, e = _mae_fit(y, R[:, s], L[:, s], k)
            if top is None or e < top[0] - 1e-12:
                top = (e, s, k, b)
    return (top[1], top[2], top[3])

def in_use_index() -> int:
    return next((i for i, c in enumerate(SETS) if c['decay'] == IN_USE['decay'] and c['shape'] == IN_USE['shape']))

def refit(T: dict, R: np.ndarray, L: np.ndarray) -> dict:
    ok = np.isfinite(R).all(axis=1) & np.isfinite(L).all(axis=1)
    y, R, L, day = (T['netto'][ok], R[ok], L[ok], T['day'][ok])
    region = np.asarray(T.get('region', np.full(len(ok), '')), dtype=str)[ok]
    n = len(y)
    out = {'pieces': int(n), 'days': len(set(day.tolist())), 'in_use': IN_USE, 'regions': {}}
    if n < MIN_PIECES:
        out['pass'], out['why'] = (False, f"{n} low pieces with the site's ridge lift, under the {MIN_PIECES} the fit needs")
        return out
    s0 = in_use_index()
    pn, pf, picks = (np.full(n, np.nan), np.full(n, np.nan), [])
    for F in convfit.folds(day):
        te = np.isin(day, list(F))
        if not te.any() or te.all():
            continue
        tr = ~te
        b0, _ = _mae_fit(y[tr], R[tr, s0], L[tr, s0], IN_USE['share'])
        pn[te] = b0 + R[te, s0] - IN_USE['share'] * L[te, s0]
        s, k, b = best(y[tr], R[tr], L[tr])
        picks.append({**SETS[s], 'share': k})
        pf[te] = b + R[te, s] - k * L[te, s]
    en, ef = (np.abs(pn - y), np.abs(pf - y))
    m = np.isfinite(en) & np.isfinite(ef)
    out['mae'] = [float(en[m].mean()), float(ef[m].mean())]
    out['ci'] = convfit.boot_gain({'day': day[m]}, en[m], ef[m])
    s, k, b = best(y, R, L)
    out['fit'] = {**SETS[s], 'share': k}
    out['b'] = round(b, 3)
    out['fit_folds'] = picks
    worse = []
    for r in sorted(set(region.tolist())):
        sel = (region == r) & m
        if sel.sum() >= REGION_MIN:
            out['regions'][r] = [float(en[sel].mean()), float(ef[sel].mean()), int(sel.sum())]
            if ef[sel].mean() > en[sel].mean() + REGION_SLACK:
                worse.append(r)
    out['worse'] = worse
    gain = out['mae'][0] - out['mae'][1]
    if gain < GAIN_MIN or out['ci'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the fitted constants gain {gain:.3f} m/s (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f})")
    elif worse:
        out['pass'], out['why'] = (False, 'worse held out in ' + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f"held out the fitted constants gain {gain:.3f} m/s (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f}), no region worse")
    return out

def report_md(month: str, r: dict) -> str:
    L = [f'# Ridge lift and lee sink, {month}', '', f"{r['pieces']} low pieces on {r['days']} days with the site's ridge lift.", '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'mae' in r:
        L += ['', f"Mean absolute error of the netto, held out by date: in use {r['mae'][0]:.3f} m/s, fitted {r['mae'][1]:.3f}.", '', "| constant (the browser's) | in use | fitted |", '| --- | --- | --- |', f"| RIDGE.decay | {r['in_use']['decay']} | {r['fit']['decay']} |", f"| RIDGE.shape | {r['in_use']['shape']} | {r['fit']['shape']} |", f"| LEE.share | {r['in_use']['share']} | {r['fit']['share']} |", '', '| region | in use | fitted | pieces |', '| --- | --- | --- | --- |']
        L += [f'| {k} | {a:.3f} | {b:.3f} | {n} |' for k, (a, b, n) in sorted(r['regions'].items())]
    return '\n'.join(L) + '\n'

def run(lk, today: date, again: bool=False, counts=None, values=site_values) -> dict | None:
    counts = counts if counts is not None else {}
    month = (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    cols = airpairs.load_all(lk, counts)
    if not cols:
        counts['no_pairs'] = 1
        return None
    T = frame(cols)
    try:
        R, L = values(T, os.path.join(os.environ.get('RUNNER_TEMP') or '/tmp', 'ridgefit'))
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        counts['site_failed'] = type(e).__name__
        R = L = np.full((len(T['netto']), len(SETS)), np.nan)
    r = refit(T, R, L)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    pre = f'{score.PREFIX}/{SV}/{month}/{ts}'
    lk.put(pre + '.json.lenc', json.dumps(r, sort_keys=True, default=float).encode(), strict=False)
    lk.put(pre + '.md.lenc', report_md(month, r).encode(), strict=False)
    counts['written'] = 1
    r['month'] = month
    return r

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Ridge lift and lee sink, refitted monthly')
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
        print('ridgefit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('ridgefit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('ridgefit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('ridgefit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
