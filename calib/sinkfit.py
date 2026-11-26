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
import convfit
import lake as lakem
import score
if common.PIPELINE not in sys.path:
    sys.path.append(common.PIPELINE)
import physics
SV = 'sink'
MIN_PIECES = 3000
GAIN_MIN = 0.005
REGION_MIN, REGION_SLACK = (500, 0.01)
K_RANGE = (0.1, 1.5)
K_STEP = 0.01
AVOID_MAX = 0.1
RIDGE_MAX = 0.1
LINE_FAR_KM = 10.0
BAND_N = 10

def shape(ws, zi, depth) -> np.ndarray:
    lo = physics.SINK_BAND[0]
    ws, zi, d = (np.nan_to_num(np.asarray(a, np.float64), nan=0.0) for a in (ws, zi, depth))
    zi = np.maximum(zi, 1.0)
    s = np.zeros_like(ws)
    for i in range(BAND_N):
        h = lo + (np.maximum(d, lo) - lo) * (i / (BAND_N - 1))
        s += physics.sigma_w(h / zi) / BAND_N
    return np.where((ws > 0) & (d > lo), ws * s, 0.0)

def frame(cols: dict) -> tuple[dict, np.ndarray]:
    n = len(cols['netto'])
    col = lambda k, fill=np.nan: np.asarray(cols[k], np.float64) if k in cols else np.full(n, fill)
    m = (np.asarray(cols['band']) == 1) & np.isfinite(col('netto')) & np.isfinite(col('ws')) & (col('ws') > 0)
    m &= np.isfinite(col('depth')) & (col('depth') > physics.SINK_BAND[0]) & np.isfinite(col('zi'))
    m &= ~(col('pre', 0.0) == 1)
    m &= ~(np.nan_to_num(col('sink'), nan=0.0) > AVOID_MAX)
    m &= ~(np.nan_to_num(col('ridge_site'), nan=0.0) > RIDGE_MAX) & ~(np.nan_to_num(col('ridge_g'), nan=0.0) > RIDGE_MAX)
    m &= ~(np.nan_to_num(col('line_km'), nan=airpairs.FAR_KM) < LINE_FAR_KM)
    T = {k: v[m] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in cols.items()}
    T['netto'] = np.asarray(T['netto'], np.float64) - np.nan_to_num(np.asarray(T.get('off', np.zeros(int(m.sum()))), np.float64), nan=0.0)
    T['S'] = shape(T['ws'], T['zi'], T['depth'])
    T['day'] = np.array([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in T['t0']])
    return (T, m)

def intercept(netto, S, K) -> float:
    return float(np.median(netto + K * S))

def fit_k(netto, S) -> tuple[float, float]:
    best = None
    for K in np.arange(K_RANGE[0], K_RANGE[1] + K_STEP / 2, K_STEP):
        b = intercept(netto, S, K)
        e = float(np.mean(np.abs(netto - (b - K * S))))
        if best is None or e < best[0] - 1e-12:
            best = (e, round(float(K), 4), b)
    return (best[1], best[2])

def heldout(T: dict, k_now: float) -> tuple[np.ndarray, np.ndarray, list]:
    pn, pf, ks = (np.full(len(T['netto']), np.nan), np.full(len(T['netto']), np.nan), [])
    for F in convfit.folds(T['day']):
        te = np.isin(T['day'], list(F))
        if not te.any() or te.all():
            continue
        y, S = (T['netto'][~te], T['S'][~te])
        pn[te] = intercept(y, S, k_now) - k_now * T['S'][te]
        K, b = fit_k(y, S)
        ks.append(K)
        pf[te] = b - K * T['S'][te]
    return (pn, pf, ks)

def refit(cols: dict, k_now: float | None=None) -> dict:
    k_now = physics.SINK_K if k_now is None else k_now
    T, _ = frame(cols)
    n = len(T['netto'])
    out = {'pieces': n, 'days': len(set(T['day'].tolist())), 'models': sorted(set(map(str, T.get('model', [])))), 'regions': {}, 'k_now': k_now}
    if n < MIN_PIECES:
        out['pass'], out['why'] = (False, f'{n} pieces, under the {MIN_PIECES} the fit needs')
        return out
    K, b = fit_k(T['netto'], T['S'])
    out['k'], out['b'] = (K, round(b, 3))
    pn, pf, ks = heldout(T, k_now)
    en, ef = (np.abs(pn - T['netto']), np.abs(pf - T['netto']))
    ok = np.isfinite(en) & np.isfinite(ef)
    out['k_folds'] = ks
    out['mae'] = [float(en[ok].mean()), float(ef[ok].mean())]
    out['ci'] = convfit.boot_gain({'day': T['day'][ok]}, en[ok], ef[ok])
    worse = []
    for r in sorted(set(map(str, T.get('region', [])))):
        s = (np.asarray(T['region']).astype(str) == r) & ok
        if s.sum() >= REGION_MIN:
            out['regions'][r] = [float(en[s].mean()), float(ef[s].mean()), int(s.sum())]
            if ef[s].mean() > en[s].mean() + REGION_SLACK:
                worse.append(r)
    out['worse'] = worse
    gain = out['mae'][0] - out['mae'][1]
    at_bound = min(abs(K - K_RANGE[0]), abs(K - K_RANGE[1])) < K_STEP / 2
    if at_bound:
        out['pass'], out['why'] = (False, f"the fitted K {K} is at its range's bound ({K_RANGE[0]} to {K_RANGE[1]}): the sink isn't this shape")
    elif gain < GAIN_MIN or out['ci'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out K {K} gains {gain:.3f} m/s on {k_now} (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f})")
    elif worse:
        out['pass'], out['why'] = (False, 'worse held out in ' + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f"held out K {K} gains {gain:.3f} m/s on {k_now} (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f}), no region worse")
    return out

def report_md(month: str, r: dict) -> str:
    L = [f'# The sink between thermals, {month}', '', f"{r['pieces']} thermal-layer glides on {r['days']} days ({', '.join(r['models'])}).", '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'mae' in r:
        L += ['', f"SINK_K: {r['k_now']} in use, {r['k']} fitted (intercept {r['b']:+.3f} m/s; by fold {', '.join((str(k) for k in r['k_folds']))}).", f"Mean absolute error of the netto, held out by date: in use {r['mae'][0]:.3f} m/s, fitted {r['mae'][1]:.3f}.", '', '| region | in use | fitted | pieces |', '| --- | --- | --- | --- |']
        L += [f'| {k} | {a:.3f} | {b:.3f} | {n} |' for k, (a, b, n) in sorted(r['regions'].items())]
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
    pre = f'{score.PREFIX}/{SV}/{month}/{ts}'
    lk.put(pre + '.json.lenc', json.dumps(r, sort_keys=True, default=float).encode(), strict=False)
    lk.put(pre + '.md.lenc', report_md(month, r).encode(), strict=False)
    counts['written'] = 1
    r['month'] = month
    return r

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='The sink between thermals, refitted monthly')
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
        print('sinkfit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('sinkfit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('sinkfit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('sinkfit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
