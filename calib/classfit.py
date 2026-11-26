from __future__ import annotations
import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import convfit
import lake as lakem
import pairs as P
import score
if common.PIPELINE not in sys.path:
    sys.path.append(common.PIPELINE)
import sky
SV = 'classes'
WINDOW_D = 90
MIN_ROWS = 5000
BA_GAIN = 0.01
RECALL_SLACK = 0.03
ROUNDS = 3
TRUTH = ('clear', 'cumulus', 'storm', 'deck', 'fog')
GROUPS = {'clear': (sky.NONE, sky.BLUE, sky.AC, sky.AS, sky.CS, sky.ACC), 'cumulus': (2, 3, 4), 'storm': (sky.CB,), 'deck': (sky.ST, sky.SC, sky.NS), 'fog': (sky.FOG,)}
PARAMS = {'CU_MIN_WS': np.round(np.arange(0.2, 1.21, 0.1), 2), 'FEW_CU': np.arange(10.0, 51.0, 5.0), 'SC_SPREAD': np.arange(50.0, 96.0, 5.0), 'SC_DEEP': np.arange(500.0, 2001.0, 250.0), 'LOW_TOP': np.arange(1500.0, 3001.0, 250.0)}
INPUTS = ('wstar', 'cudepth', 'overdev', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ceil', 'vis', 'rain', 'refc', 'cubase', 'zi', 'depth', 'sun')
OBS_COLS = ['obs_id', 't_valid', 'o_cu', 'o_low_lo', 'o_low_hi', 'o_storm', 'o_fog', 'o_vis', 'o_precip']

@contextmanager
def constants(over: dict):
    keep = {k: getattr(sky, k) for k in over}
    try:
        for k, v in over.items():
            setattr(sky, k, v)
        yield
    finally:
        for k, v in keep.items():
            setattr(sky, k, v)

def in_use() -> dict:
    return {k: float(getattr(sky, k)) for k in PARAMS}

def truth_of(o: dict) -> np.ndarray:
    f = lambda k: np.asarray(o[k], np.float64) if k in o else np.full(len(o['obs_id']), np.nan)
    cu, lo, hi, storm, fog, vis = (f('o_cu'), f('o_low_lo'), f('o_low_hi'), f('o_storm'), f('o_fog'), f('o_vis'))
    g = np.full(len(cu), -1, np.int8)
    g = np.where((cu == 0) & (hi <= 1 / 16 + 1e-09), 0, g)
    g = np.where(cu == 1, 1, g)
    g = np.where(lo >= 5 / 8 - 1e-09, 3, g)
    g = np.where(storm == 1, 2, g)
    g = np.where((fog == 1) | np.isfinite(vis) & (vis < 1000), 4, g)
    return g

def group_of(cls: np.ndarray) -> np.ndarray:
    out = np.full(len(cls), -1, np.int8)
    for i, name in enumerate(TRUTH):
        out = np.where(np.isin(cls, GROUPS[name]), i, out)
    return out

def classes(X: dict, over: dict | None=None) -> np.ndarray:
    n = len(X['wstar'])
    v = {k: np.asarray(X.get(k, np.full(n, np.nan)), np.float64) for k in INPUTS}
    v.update(showers=np.full(n, np.nan), accas=np.full(n, np.nan))
    with constants(over or {}):
        cls = sky.classify(v)[0]
    return group_of(np.nan_to_num(cls, nan=-1).astype(np.int64))

def balanced(truth: np.ndarray, pred: np.ndarray) -> tuple[float, dict]:
    rec = {}
    for i, name in enumerate(TRUTH):
        s = truth == i
        if s.sum():
            rec[name] = float((pred[s] == i).mean())
    return (float(np.mean(list(rec.values()))) if rec else float('nan'), rec)

def search(X: dict, truth: np.ndarray, start: dict | None=None) -> dict:
    best = dict(start or in_use())
    score_of = lambda c: balanced(truth, classes(X, c))[0]
    cur = score_of(best)
    for _ in range(ROUNDS):
        moved = False
        for k, grid in PARAMS.items():
            for val in grid:
                c = {**best, k: float(val)}
                s = score_of(c)
                if s > cur + 1e-09:
                    best, cur, moved = (c, s, True)
        if not moved:
            break
    return best

def frame(pairs_t: dict, obs_t: dict) -> tuple[dict, np.ndarray, np.ndarray] | None:
    if not P.n_rows(pairs_t) or not P.n_rows(obs_t) or 'f_wstar' not in pairs_t:
        return None
    oi = {(str(a), int(b)): i for i, (a, b) in enumerate(zip(obs_t['obs_id'], obs_t['t_valid']))}
    jp, jo = ([], [])
    for j, k in enumerate(zip((str(x) for x in pairs_t['obs_id']), (int(x) for x in pairs_t['t_valid']))):
        i = oi.get(k)
        if i is not None:
            (jp.append(j), jo.append(i))
    if not jp:
        return None
    jp, jo = (np.array(jp), np.array(jo))
    X = {k: np.asarray(pairs_t[f'f_{k}'], np.float64)[jp] if f'f_{k}' in pairs_t and pairs_t[f'f_{k}'] is not None else np.full(len(jp), np.nan) for k in INPUTS}
    t = truth_of({k: np.asarray(v)[jo] for k, v in obs_t.items() if k in OBS_COLS})
    keep = np.isfinite(X['wstar']) & (t >= 0)
    day = np.asarray(pairs_t['t_valid'], np.int64)[jp] // 86400
    return ({k: v[keep] for k, v in X.items()}, t[keep], day[keep])

def refit(X: dict, truth: np.ndarray, day: np.ndarray) -> dict:
    n = len(truth)
    out = {'rows': int(n), 'days': int(len(set(day.tolist()))), 'in_use': in_use(), 'truth': {name: int((truth == i).sum()) for i, name in enumerate(TRUTH)}}
    if n < MIN_ROWS:
        out['pass'], out['why'] = (False, f'{n} rows, under the {MIN_ROWS} the fit needs')
        return out
    days = np.array([str(d) for d in day])
    hit_now, hit_fit = (np.zeros(n, bool), np.zeros(n, bool))
    fits = []
    for F in convfit.folds(days):
        te = np.isin(days, list(F))
        if not te.any() or te.all():
            continue
        tr = ~te
        c = search({k: v[tr] for k, v in X.items()}, truth[tr])
        fits.append(c)
        Xte = {k: v[te] for k, v in X.items()}
        hit_now[te] = classes(Xte) == truth[te]
        hit_fit[te] = classes(Xte, c) == truth[te]
    ba = lambda hit: float(np.mean([hit[truth == i].mean() for i in range(len(TRUTH)) if (truth == i).any()]))
    out['ba'] = [ba(hit_now), ba(hit_fit)]
    out['recall'] = {name: [float(hit_now[truth == i].mean()), float(hit_fit[truth == i].mean())] for i, name in enumerate(TRUTH) if (truth == i).any()}
    ud = np.array(sorted(set(days.tolist())))
    idx = {d: np.where(days == d)[0] for d in ud}
    rng = np.random.default_rng(1)
    g = []
    for _ in range(400):
        pick = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        hn, hf, tt = (hit_now[pick], hit_fit[pick], truth[pick])
        gs = [hf[tt == i].mean() - hn[tt == i].mean() for i in range(len(TRUTH)) if (tt == i).any()]
        g.append(float(np.mean(gs)))
    out['ci'] = [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]
    out['fit'] = search(X, truth)
    out['fit_folds'] = fits
    gain = out['ba'][1] - out['ba'][0]
    worse = [k for k, (a, b) in out['recall'].items() if b < a - RECALL_SLACK]
    out['worse'] = worse
    if gain < BA_GAIN or out['ci'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the fitted set gains {gain:.3f} of balanced accuracy (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f})")
    elif worse:
        out['pass'], out['why'] = (False, "a group's recall down held out: " + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f"held out the fitted set gains {gain:.3f} of balanced accuracy (95 % {out['ci'][0]:.3f} to {out['ci'][1]:.3f}), no group worse")
    return out

def report_md(month: str, r: dict) -> str:
    L = [f"# The Cloud Layers' classes, {month}", '', f"{r['rows']} station hours on {r['days']} days against the reports ({', '.join((f'{k} {v}' for k, v in r['truth'].items()))}).", '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'ba' in r:
        L += ['', f"Balanced accuracy held out by date: in use {r['ba'][0]:.3f}, fitted {r['ba'][1]:.3f}.", '', '| group | recall in use | fitted |', '| --- | --- | --- |']
        L += [f'| {k} | {a:.3f} | {b:.3f} |' for k, (a, b) in r['recall'].items()]
        L += ['', "| constant (sky.py and the browser's twin) | in use | fitted |", '| --- | --- | --- |']
        L += [f"| {k} | {r['in_use'][k]:g} | {r['fit'][k]:g} |" for k in PARAMS]
        L += ['', "A pass changes nothing by itself: sky.py and the browser's cloudType change together."]
    return '\n'.join(L) + '\n'

def month_days(month: str) -> list[str]:
    end = date.fromisoformat(month + '-01')
    end = (end.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return [(end - timedelta(days=k)).isoformat() for k in range(WINDOW_D - 1, -1, -1)]

def run(lk, today: date, again: bool=False, counts=None, read=None) -> dict | None:
    counts = counts if counts is not None else {}
    month = (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    cols_p = list(P.KEY) + [f'f_{k}' for k in INPUTS if k in P.METAR_VARS]
    rd = read or (lambda what, d: P.read_kind(lk, what, 'metar', [d], purpose='review', columns=cols_p if what == 'pairs' else OBS_COLS))
    Xs, ts, ds = ([], [], [])
    for d in month_days(month):
        got = frame(rd('pairs', d), rd('obs', d))
        if got:
            (Xs.append(got[0]), ts.append(got[1]), ds.append(got[2]))
    if not Xs:
        counts['no_rows'] = 1
        return None
    X = {k: np.concatenate([x[k] for x in Xs]) for k in INPUTS}
    r = refit(X, np.concatenate(ts), np.concatenate(ds))
    ts_ = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    pre = f'{score.PREFIX}/{SV}/{month}/{ts_}'
    lk.put(pre + '.json.lenc', json.dumps(r, sort_keys=True, default=float).encode(), strict=False)
    lk.put(pre + '.md.lenc', report_md(month, r).encode(), strict=False)
    counts['written'] = 1
    r['month'] = month
    return r

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The Cloud Layers' classes, refitted monthly")
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
        print('classfit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('classfit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('classfit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('classfit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
