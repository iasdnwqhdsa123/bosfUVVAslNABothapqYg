from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import convpairs
import lake as lakem
import score
SV = 'lines'
MIN_PIECES = 3000
GAIN_MIN = 0.005
REGION_MIN, REGION_SLACK = (500, 0.01)
STEP_FINE = 0.05
LIFT_BINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
FAR_KM = 60.0
LINE = 0.15
BASE = ('ws', 'zi', 'agl', 'wind', 'step')
CONV = ('conv', 'best0.15', 'wblmax')
MONO = {'conv': 1, 'best0.15': -1, 'wblmax': 1}

def frame(cols: dict) -> tuple[dict, np.ndarray]:
    m = (cols['kind'] == 0) & np.isfinite(cols['netto']) & np.isfinite(cols['ws']) & np.isfinite(cols['zi'])
    T = {k: v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v for k, v in cols.items()}
    T['best0.15'] = np.where(np.isfinite(T['best0.15']), T['best0.15'], FAR_KM)
    T['day'] = np.array([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in T['t0']])
    return (T, m)

def X(T: dict, names) -> np.ndarray:
    return np.column_stack([T[n].astype(np.float64) for n in names])

def model(names):
    from sklearn.ensemble import HistGradientBoostingRegressor
    mono = [MONO.get(n, 0) for n in names]
    return HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=80, monotonic_cst=mono, random_state=0)

def folds(days, k: int=5, seed: int=0) -> list[set]:
    d = np.array(sorted(set(days)))
    np.random.default_rng(seed).shuffle(d)
    return [set(d[i::k]) for i in range(k)]

def heldout(T: dict) -> tuple[np.ndarray, np.ndarray]:
    pb, pc = (np.full(len(T['netto']), np.nan), np.full(len(T['netto']), np.nan))
    for F in folds(T['day']):
        te = np.isin(T['day'], list(F))
        for p, names in ((pb, BASE), (pc, BASE + CONV)):
            m = model(names).fit(common.no_empty(X(T, names)[~te]), T['netto'][~te])
            p[te] = m.predict(X(T, names)[te])
    return (pb, pc)

def boot_gain(T: dict, eb: np.ndarray, ec: np.ndarray, n: int=1000, seed: int=1) -> tuple[float, float]:
    days = np.array(sorted(set(T['day'])))
    idx = {d: np.where(T['day'] == d)[0] for d in days}
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n):
        pick = np.concatenate([idx[d] for d in rng.choice(days, len(days))])
        g.append(eb[pick].mean() - ec[pick].mean())
    return (float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)))

def lift_table(T: dict) -> dict:
    names = BASE + CONV
    m = model(names).fit(common.no_empty(X(T, names)), T['netto'])
    out = {}
    for cls, sel in (('fine', T['step'] <= STEP_FINE), ('coarse', T['step'] > STEP_FINE)):
        if sel.sum() < 200:
            continue
        S = {k: v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v for k, v in T.items()}
        none = dict(S, conv=np.zeros(sel.sum()), **{'best0.15': np.full(sel.sum(), FAR_KM)})
        p0 = m.predict(X(none, names)).mean()
        out[cls] = [[c, round(float(m.predict(X(dict(S, conv=np.full(sel.sum(), c), **{'best0.15': np.full(sel.sum(), 0.0 if c >= LINE else FAR_KM)}), names)).mean() - p0), 3)] for c in LIFT_BINS]
    return out

def refit(cols: dict) -> dict:
    T, _ = frame(cols)
    n = len(T['netto'])
    out = {'pieces': n, 'days': len(set(T['day'])), 'models': sorted(set(T['model'])), 'regions': {}}
    if n < MIN_PIECES:
        out['pass'], out['why'] = (False, f'{n} pieces, under the {MIN_PIECES} the fit needs')
        return out
    pb, pc = heldout(T)
    eb, ec = (np.abs(pb - T['netto']), np.abs(pc - T['netto']))
    out['mae'] = [float(eb.mean()), float(ec.mean())]
    out['ci'] = boot_gain(T, eb, ec)
    worse = []
    for r in sorted(set(T['region'])):
        s = T['region'] == r
        if s.sum() >= REGION_MIN:
            out['regions'][r] = [float(eb[s].mean()), float(ec[s].mean()), int(s.sum())]
            if ec[s].mean() > eb[s].mean() + REGION_SLACK:
                worse.append(r)
    out['worse'] = worse
    gain = out['mae'][0] - out['mae'][1]
    if gain < GAIN_MIN or out['ci'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the convergence gains {gain:.3f} m/s (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f})")
    elif worse:
        out['pass'], out['why'] = (False, 'worse held out in ' + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f"held out the convergence gains {gain:.3f} m/s (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f}), no region worse")
    out['lift'] = lift_table(T)
    return out

def report_md(month: str, r: dict) -> str:
    L = [f"# The convergence lines' lift, {month}", '', f"{r['pieces']} straight pieces on {r['days']} days ({', '.join(r['models'])}).", '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'mae' in r:
        L += ['', f"Mean absolute error of the netto, held out by date: thermals alone {r['mae'][0]:.3f} m/s, with the convergence {r['mae'][1]:.3f}.", '', '| region | thermals alone | with the convergence | pieces |', '| --- | --- | --- | --- |']
        L += [f'| {k} | {a:.3f} | {b:.3f} | {n} |' for k, (a, b, n) in sorted(r['regions'].items())]
    for cls, rows in (r.get('lift') or {}).items():
        L += ['', f'The lift along a line under the glider, {cls} grids (m/s of netto over none):', '', '| convergence | lift |', '| --- | --- |']
        L += [f'| {c} | {v:+.2f} |' for c, v in rows]
    return '\n'.join(L) + '\n'

def run(lk, today: date, again: bool=False, counts=None) -> dict | None:
    counts = counts if counts is not None else {}
    month = (today.replace(day=1) - __import__('datetime').timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    cols = convpairs.load_all(lk, counts)
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
    ap = argparse.ArgumentParser(description="The convergence lines' lift, refitted monthly")
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
        print('convfit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('convfit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('convfit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('convfit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
