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
import breakpairs as BP
import convfit
import lake as lakem
import score
SV = 'break'
IN_USE = {'b0': 7.0, 'from': 4.0, 'full': 8.0}
B0S = (5.0, 6.0, 7.0, 8.0)
FROMS = (2.0, 3.0, 4.0, 5.0, 6.0)
SPANS = (1.0, 2.0, 4.0, 6.0, 8.0)
CANDIDATES = [{'b0': b, 'from': f, 'full': f + s} for b in B0S for f in FROMS for s in SPANS]
WS_MIN = 0.5
MIN_ROWS = 5000
GAIN_MIN = 0.0005
REGION_MIN, REGION_SLACK = (1000, 0.002)
WS_BINS = (0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0, 3.5)
SOLAR_BINS = (11, 12, 13, 14, 15, 16, 17, 18)
BANDS = ((0.5, 1.25), (1.25, 1.75), (1.75, 2.25), (2.25, 2.75), (2.75, 99.0))
RIDGE = 0.001

def frame(c: dict) -> dict:
    ws = np.asarray(c['wstar'], np.float64)
    u, v = (np.asarray(c['u_blavg'], np.float64), np.asarray(c['v_blavg'], np.float64))
    m = np.isfinite(np.asarray(c['search'], np.float64)) & (ws > WS_MIN) & np.isfinite(u) & np.isfinite(v)
    T = {k: np.asarray(val)[m] if len(val) == len(m) else val for k, val in c.items()}
    T['wind'] = np.hypot(T['u_blavg'], T['v_blavg'])
    T['bs'] = BP.bs_rasp(T['wstar'], T['u_blavg'], T['v_blavg'])
    T['y'] = np.asarray(T['search'], np.float64)
    T['cu'] = np.isfinite(np.asarray(T['cubase'], np.float64)).astype(np.float64)
    T['day'] = np.array([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in T['t']])
    T['group'] = np.char.add(np.asarray(T['region']).astype(str), np.char.add('|', np.asarray(T['model']).astype(str)))
    return T

def controls(T: dict) -> np.ndarray:
    n = len(T['y'])
    cols = [np.ones(n)]
    for g in sorted(set(T['group'].tolist()))[1:]:
        cols.append((T['group'] == g).astype(np.float64))
    wb = np.digitize(T['wstar'], WS_BINS[1:])
    cols += [(wb == k).astype(np.float64) for k in range(1, len(WS_BINS))]
    sb = np.digitize(np.nan_to_num(np.asarray(T['solar_h'], np.float64), nan=14.0), SOLAR_BINS)
    cols += [(sb == k).astype(np.float64) for k in range(1, len(SOLAR_BINS) + 1)]
    cols.append(T['cu'])
    return np.column_stack(cols)

def ramp(wind: np.ndarray, p: dict) -> np.ndarray:
    return np.clip((wind - p['from']) / max(1e-06, p['full'] - p['from']), 0.0, 1.0)

def cost(T: dict, p: dict) -> np.ndarray:
    return np.maximum(0.0, np.log(p['b0'] / np.maximum(T['bs'], 0.001))) * ramp(T['wind'], p)

def wind_bs(bs, wind, p: dict=IN_USE) -> np.ndarray:
    bs = np.asarray(bs, np.float64)
    r = np.clip((np.asarray(wind, np.float64) - p['from']) / (p['full'] - p['from']), 0.0, 1.0)
    return np.where((bs > 0) & (bs < p['b0']), p['b0'] * (np.maximum(bs, 1e-09) / p['b0']) ** r, bs)

def logit(X: np.ndarray, y: np.ndarray, it: int=25):
    b = np.zeros(X.shape[1])
    R = RIDGE * np.eye(X.shape[1])
    R[0, 0] = 0.0
    H = R
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        H = X.T @ (X * (p * (1 - p) + 1e-09)[:, None]) + R
        step = np.linalg.solve(H, X.T @ (y - p) - R @ b)
        b += step
        if np.max(np.abs(step)) < 1e-07:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    return (b, float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))), H)

def loss(X: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    return -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

def best(C: np.ndarray, T: dict, rows: np.ndarray, cands=None) -> tuple[dict, np.ndarray, float]:
    out = None
    for p in cands or CANDIDATES:
        X = np.column_stack([C[rows], cost(T, p)[rows]])
        b, ll, _ = logit(X, T['y'][rows])
        if out is None or ll > out[2]:
            out = (p, b, ll)
    return out

def at_bound(p: dict) -> bool:
    return p['b0'] in (B0S[0], B0S[-1]) or p['from'] in (FROMS[0], FROMS[-1]) or p['full'] - p['from'] in (SPANS[0], SPANS[-1])

def refit(cols: dict) -> dict:
    T = frame(cols)
    n = len(T['y'])
    out = {'rows': n, 'days': len(set(T['day'].tolist())), 'models': sorted(set(map(str, T['model']))), 'in_use': dict(IN_USE), 'searches': float(T['y'].mean()) if n else None, 'groups': {}}
    if n < MIN_ROWS:
        out['pass'], out['why'] = (False, f'{n} glides, under the {MIN_ROWS} the fit needs')
        return out
    C = controls(T)
    l_use, l_fit = (np.full(n, np.nan), np.full(n, np.nan))
    picked = []
    for F in convfit.folds(T['day']):
        te = np.isin(T['day'], list(F))
        if not te.any() or te.all():
            continue
        p, b, _ = best(C, T, ~te)
        picked.append(p)
        l_fit[te] = loss(np.column_stack([C[te], cost(T, p)[te]]), T['y'][te], b)
        bu, _, _ = logit(np.column_stack([C[~te], cost(T, IN_USE)[~te]]), T['y'][~te])
        l_use[te] = loss(np.column_stack([C[te], cost(T, IN_USE)[te]]), T['y'][te], bu)
    ok = np.isfinite(l_use) & np.isfinite(l_fit)
    Tk = {'day': T['day'][ok]}
    out['logloss'] = {'in_use': float(l_use[ok].mean()), 'fitted': float(l_fit[ok].mean())}
    out['ci'] = airfit.boot_gain(Tk, l_use[ok], l_fit[ok])
    out['folds'] = picked
    p, b, _ = best(C, T, np.ones(n, bool))
    _, _, H = logit(np.column_stack([C, cost(T, p)]), T['y'])
    se = float(np.sqrt(np.linalg.inv(H)[-1, -1]))
    out['fitted'] = dict(p)
    out['c'], out['c_se'] = (float(b[-1]), se)
    bu, ll_use, Hu = logit(np.column_stack([C, cost(T, IN_USE)]), T['y'])
    out['c_in_use'], out['c_in_use_se'] = (float(bu[-1]), float(np.sqrt(np.linalg.inv(Hu)[-1, -1])))
    out['bands'] = {}
    for lo, hi in BANDS:
        s = (T['wstar'] > lo) & (T['wstar'] <= hi)
        h = cost(T, IN_USE)[s]
        if s.sum() < 500 or not (h > 0).any():
            continue
        Cs = C[s][:, C[s].std(0) > 0]
        bb, _, Hb = logit(np.column_stack([np.ones(s.sum()), Cs, h]), T['y'][s])
        out['bands'][f'{lo:g}-{hi:g}'] = {'n': int(s.sum()), 'c': float(bb[-1]), 'se': float(np.sqrt(np.linalg.inv(Hb)[-1, -1])), 'broken': float(np.mean(wind_bs(T['bs'][s], T['wind'][s]) < 5))}
    worse = []
    reg = np.asarray(T['region']).astype(str)[ok]
    for g in sorted(set(reg.tolist())):
        s = reg == g
        if s.sum() < REGION_MIN or not g:
            continue
        out['groups'][f'region {g}'] = {'in_use': float(l_use[ok][s].mean()), 'fitted': float(l_fit[ok][s].mean()), 'n': int(s.sum())}
        if l_fit[ok][s].mean() > l_use[ok][s].mean() + REGION_SLACK:
            worse.append(g)
    out['worse'] = worse
    gain = out['logloss']['in_use'] - out['logloss']['fitted']
    lo95 = out['c'] - 1.96 * se
    if not lo95 > 0:
        out['pass'], out['why'] = (False, f"the wind's cost not found: c {out['c']:.2f} (95 % from {lo95:.2f})")
    elif p == IN_USE:
        out['pass'], out['why'] = (False, 'the rule in use is the best fit')
    elif gain < GAIN_MIN or out['ci'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the fitted rule gains {gain:.5f} nats a glide (95 % {out['ci'][0]:.5f} to {out['ci'][1]:.5f})")
    elif worse:
        out['pass'], out['why'] = (False, 'worse held out in ' + ', '.join(worse))
    elif at_bound(p):
        out['pass'], out['why'] = (False, f'the fit at a bound of its grid: {p}')
    else:
        out['pass'], out['why'] = (True, f'held out the fitted rule gains {gain:.5f} nats a glide, no region worse')
    return out

def report_md(month: str, r: dict) -> str:
    L = [f"# The wind's cost on the thermals, {month}", '', f"{r['rows']} glides between climbs on {r['days']} days ({', '.join(r['models'])})" + (f", {r['searches']:.0%} of them searches." if r.get('searches') is not None else '.'), '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'logloss' in r:
        u, f = (r['in_use'], r['fitted'])
        L += ['', '| | b0 | the wind from (m/s) | fully from (m/s) | the cost c (se) | log loss a glide, held out |', '| --- | --- | --- | --- | --- | --- |', f"| in use | {u['b0']:g} | {u['from']:g} | {u['full']:g} | {r['c_in_use']:.2f} ({r['c_in_use_se']:.2f}) | {r['logloss']['in_use']:.5f} |", f"| fitted | {f['b0']:g} | {f['from']:g} | {f['full']:g} | {r['c']:.2f} ({r['c_se']:.2f}) | {r['logloss']['fitted']:.5f} |"]
        if r.get('bands'):
            L += ['', 'The cost by W* (m/s), fitted apart, the rule in use (the share it calls broken beside):', '', '| W* | glides | c (se) | broken |', '| --- | --- | --- | --- |']
            L += [f"| {k} | {v['n']} | {v['c']:.2f} ({v['se']:.2f}) | {v['broken']:.0%} |" for k, v in r['bands'].items()]
        if r.get('groups'):
            L += ['', '| region | glides | in use | fitted |', '| --- | --- | --- | --- |']
            L += [f"| {g} | {v['n']} | {v['in_use']:.5f} | {v['fitted']:.5f} |" for g, v in sorted(r['groups'].items())]
    return '\n'.join(L) + '\n'

def run(lk, today: date, again: bool=False, counts=None) -> dict | None:
    counts = counts if counts is not None else {}
    month = (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    cols = BP.load_all(lk, counts)
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
    ap = argparse.ArgumentParser(description="The wind's cost on the thermals, refitted monthly")
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
        print('breakfit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('breakfit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('breakfit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('breakfit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
