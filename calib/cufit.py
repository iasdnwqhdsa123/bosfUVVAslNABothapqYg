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
import pairs as P
import score
SV = 'cumulus'
MODELS_PREFIX = 'derived/models/cumulus'
SCHEMA = 'cu-head/1'
WINDOW_D = 90
MIN_ROWS = 3000
MAX_ROWS = 400000
GAIN_MIN = 0.01
GROUP_MIN, GROUP_SLACK = (300, 0.01)
ICE_MAX = 0.3
TOP_AGL = 3500.0
CU_CALL = 0.1
FRESH_S = 3600
SAT_IN = ('cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ctop', 'cuclass', 'cucover', 'cubase', 'elevation')
METAR_IN = ('wstar', 'cudepth', 'zi', 'depth', 'sun', 'overdev', 't2', 'td2', 'u_sfc', 'v_sfc', 'rain')
DERIVED = ('cb_agl', 'spread', 'ws_sfc', 'solar_h', 'lat')
COARSE = SAT_IN + METAR_IN + DERIVED
FINE: tuple = ()
MONO = {'cucover': 1}
CLAMP = (0.0, 0.95)
TREE = dict(airfit.TREE, max_iter=200)
VIS_COLS = ('obs_id', 't_valid', 'lat', 'lon', 'o_cloud', 'o_cu', 'o_broken', 'q_sun')
SAT_COLS = ('obs_id', 't_valid', 'status', 'o_ice_frac', 'o_cth_mean', 'o_cloud_frac')
SKIES = ('clear', 'cumulus', 'deck', 'mixed')

def station_rows(t: dict) -> np.ndarray:
    return np.array([not s[:2] in ('g+', 'g-') for s in map(str, t['obs_id'])], bool)

def station_batch(b: str) -> bool:
    import pairjob
    return not (b.startswith('grid') or any((b.startswith('g' + a) for a in pairjob.SAT_GRID)))

def _key(t: dict, i: int, run: bool=True) -> tuple:
    k = (str(t['obs_id'][i]), int(t['t_valid'][i]))
    return k + (str(t['model'][i]), str(t['run'][i])) if run else k

def truth(vis: dict, sat: dict, ground: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(at)
    si = {(str(a), int(b)): i for i, (a, b) in enumerate(zip(sat.get('obs_id', []), sat.get('t_valid', [])))}
    cu = np.asarray(vis['o_cu'], np.float64)[at]
    cl = np.asarray(vis['o_cloud'], np.float64)[at]
    br = np.asarray(vis['o_broken'], np.float64)[at]
    out = np.full(n, np.nan)
    for j, i in enumerate(at):
        s = si.get((str(vis['obs_id'][i]), int(vis['t_valid'][i])))
        if s is None or str(sat['status'][s]) != 'ok' or (not np.isfinite(cu[j])):
            continue
        ice, cf, top = (float(sat[c][s]) for c in ('o_ice_frac', 'o_cloud_frac', 'o_cth_mean'))
        if not np.isfinite(ice):
            if not (np.isfinite(cf) and cf < 0.05):
                continue
            ice = 0.0
        if ice > ICE_MAX or (np.isfinite(top) and np.isfinite(ground[j]) and (top - ground[j] > TOP_AGL)):
            continue
        out[j] = cu[j]
    sky = np.full(n, 3, np.int8)
    sky[cl < 0.1] = 0
    sky[(cu >= CU_CALL) & (np.nan_to_num(br) >= 0.5)] = 1
    sky[(cl >= 0.5) & (np.nan_to_num(br, nan=1.0) < 0.3)] = 2
    return (out, sky)

def freshest(p: dict) -> np.ndarray:
    best = {}
    tv = np.asarray(p['t_valid'], np.int64)
    lead = np.asarray(p['lead_h'], np.float64)
    av = p.get('avail_t')
    for i in range(len(tv)):
        a = None if av is None else av[i]
        a = None if a is None or (isinstance(a, float) and (not np.isfinite(a))) else int(a)
        if a is not None and a > tv[i] - FRESH_S or (a is None and lead[i] < 6):
            continue
        k = (str(p['obs_id'][i]), int(tv[i]), str(p['model'][i]))
        if k not in best or lead[i] < lead[best[k]]:
            best[k] = i
    return np.array(sorted(best.values()), np.int64)

def frame(vis: dict, sat: dict, sp: dict, mp: dict) -> dict | None:
    if not all((P.n_rows(t) for t in (vis, sp))):
        return None
    vi = {(str(a), int(b)): i for i, (a, b) in enumerate(zip(vis['obs_id'], vis['t_valid']))}
    mi = {_key(mp, i): i for i in range(P.n_rows(mp))} if P.n_rows(mp) else {}
    keep = freshest(sp)
    jp, jv, jm = ([], [], [])
    for i in keep:
        v = vi.get(_key(sp, i, run=False))
        if v is None:
            continue
        (jp.append(i), jv.append(v), jm.append(mi.get(_key(sp, i), -1)))
    if not jp:
        return None
    jp, jv, jm = (np.array(jp), np.array(jv), np.array(jm))
    n = len(jp)
    T = {}
    for k in SAT_IN:
        T[k] = np.asarray(sp[f'f_{k}'], np.float64)[jp] if f'f_{k}' in sp else np.full(n, np.nan)
    for k in METAR_IN:
        col = mp.get(f'f_{k}') if P.n_rows(mp) else None
        T[k] = np.where(jm >= 0, np.asarray(col, np.float64)[np.maximum(jm, 0)], np.nan) if col is not None else np.full(n, np.nan)
    lat = np.asarray(vis['lat'], np.float64)[jv]
    lon = np.asarray(vis['lon'], np.float64)[jv]
    tv = np.asarray(sp['t_valid'], np.int64)[jp]
    T['cb_agl'] = T['cubase'] - T['elevation']
    T['spread'] = T['t2'] - T['td2']
    T['ws_sfc'] = np.hypot(T['u_sfc'], T['v_sfc'])
    T['solar_h'] = (tv % 86400 / 3600.0 + lon / 15.0) % 24
    T['lat'] = lat
    y, sky = truth(vis, sat, T['elevation'], jv)
    ok = np.isfinite(y) & np.isfinite(T['cucover'])
    T = {k: v[ok] for k, v in T.items()}
    T['y'], T['sky'] = (y[ok], sky[ok])
    T['model'] = np.asarray(sp['model']).astype(str)[jp][ok]
    T['region'] = np.array([str(common.place_region(a, b)[0] or '') for a, b in zip(lon[ok], lat[ok])])
    T['day'] = np.array([datetime.fromtimestamp(int(t), timezone.utc).strftime('%Y-%m-%d') for t in tv[ok]])
    return T if len(T['y']) else None

def X(T: dict, names) -> np.ndarray:
    n = len(T['y'])
    return np.column_stack([np.asarray(T[k], np.float64) if k in T else np.full(n, np.nan) for k in names])

def model(names):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(monotonic_cst=[MONO.get(k, 0) for k in names], **TREE)
SETS = {'cover': ('cucover',), 'head': COARSE}

def heldout(T: dict) -> dict:
    n = len(T['y'])
    out = {k: np.full(n, np.nan) for k in SETS}
    for F in airfit.folds(T['day']):
        te = np.isin(T['day'], list(F))
        if not te.any() or te.all():
            continue
        for k, names in SETS.items():
            m = model(list(names)).fit(common.no_empty(X(T, names)[~te]), T['y'][~te])
            out[k][te] = np.clip(m.predict(X(T, names)[te]), *CLAMP)
    out['layer'] = np.clip(np.asarray(T['cucover'], np.float64), *CLAMP)
    return out

def calls(y: np.ndarray, p: np.ndarray) -> dict:
    seen, said = (y >= CU_CALL, p >= CU_CALL)
    hit = int((seen & said).sum())
    return {'seen': int(seen.sum()), 'called': int(said.sum()), 'hit': hit, 'pod': hit / max(int(seen.sum()), 1), 'far': 1 - hit / max(int(said.sum()), 1)}

def cap(T: dict, max_rows: int=MAX_ROWS, seed: int=0) -> dict:
    n = len(T['y'])
    if n <= max_rows:
        return T
    days = np.array(sorted(set(T['day'])))
    np.random.default_rng(seed).shuffle(days)
    per = n / len(days)
    pick = set(days[:max(1, int(max_rows / per))])
    s = np.isin(T['day'], list(pick))
    return {k: v[s] for k, v in T.items()}

def refit(T: dict) -> dict:
    T = cap(T)
    n = len(T['y'])
    out = {'rows': n, 'days': len(set(T['day'])), 'models': sorted(set(T['model'])), 'groups': {}}
    out['skies'] = {s: int((T['sky'] == i).sum()) for i, s in enumerate(SKIES)}
    if n < MIN_ROWS:
        out['pass'], out['why'] = (False, f'{n} rows, under the {MIN_ROWS} the fit needs')
        return out
    Pd = heldout(T)
    ok = np.isfinite(Pd['head']) & np.isfinite(Pd['cover'])
    E = {k: np.abs(v[ok] - T['y'][ok]) for k, v in Pd.items()}
    Tk = {k: v[ok] for k, v in T.items()}
    out['mae'] = {k: float(v.mean()) for k, v in E.items()}
    out['bias'] = {k: float((Pd[k][ok] - Tk['y']).mean()) for k in Pd}
    out['ci_layer'] = airfit.boot_gain(Tk, E['layer'], E['head'])
    out['ci_cover'] = airfit.boot_gain(Tk, E['cover'], E['head'])
    out['calls'] = {k: calls(Tk['y'], Pd[k][ok]) for k in ('layer', 'head')}
    hours = np.floor(Tk['solar_h'] / 3).astype(int) * 3
    worse = []
    for name, col in (('model', Tk['model']), ('region', Tk['region']), ('solar_h', hours.astype(str)), ('sky', np.array([SKIES[i] for i in Tk['sky']]))):
        for g in sorted(set(col.tolist())):
            s = col == g
            if s.sum() < GROUP_MIN or not g:
                continue
            out['groups'][f'{name} {g}'] = {k: float(E[k][s].mean()) for k in E} | {'n': int(s.sum())}
            if E['head'][s].mean() > E['layer'][s].mean() + GROUP_SLACK:
                worse.append(f'{name} {g}')
    out['worse'] = worse
    g_l, g_c = (out['mae']['layer'] - out['mae']['head'], out['mae']['cover'] - out['mae']['head'])
    if g_l < GAIN_MIN or out['ci_layer'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the head gains {g_l:.3f} of the sky on the layer (95 % {out['ci_layer'][0]:.3f} to {out['ci_layer'][1]:.3f})")
    elif g_c < GAIN_MIN or out['ci_cover'][0] <= 0:
        out['pass'], out['why'] = (False, f"held out the air adds {g_c:.3f} of the sky to the cover recalibrated alone (95 % {out['ci_cover'][0]:.3f} to {out['ci_cover'][1]:.3f})")
    elif worse:
        out['pass'], out['why'] = (False, 'worse than the layer held out in ' + ', '.join(worse))
    else:
        out['pass'], out['why'] = (True, f'held out the head gains {g_l:.3f} of the sky on the layer and {g_c:.3f} on the cover recalibrated alone, no model, region, hour or sky worse')
    names = list(COARSE)
    out['_model'] = (model(names).fit(common.no_empty(X(T, names)), T['y']), names)
    return out

def bundle(r: dict, month: str) -> dict | None:
    if '_model' not in r:
        return None
    import train
    m, names = r['_model']
    e = train.export(m, names)
    return {'schema': SCHEMA, 'month': month, 'pass': bool(r.get('pass')), 'coarse': list(COARSE), 'fine': list(FINE), 'fine_bins': {}, 'mono': MONO, 'clamp': list(CLAMP), 'truth': 'vis', **e}

def report_md(month: str, r: dict) -> str:
    L = [f'# The cumulus head, {month}', '', f"{r['rows']} station hours on {r['days']} days ({', '.join(r['models'])}) against GOES-East's band 2 (" + ', '.join((f'{k} {v}' for k, v in r['skies'].items())) + ').', '', f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}"]
    if 'mae' in r:
        m, b = (r['mae'], r['bias'])
        L += ['', "Mean absolute error of the cumulus' share of the sky, held out by date (its bias beside):", '', '| | the layer as it is | the cover recalibrated | the head |', '| --- | --- | --- | --- |', f"| all | {m['layer']:.3f} ({b['layer']:+.3f}) | {m['cover']:.3f} ({b['cover']:+.3f}) | {m['head']:.3f} ({b['head']:+.3f}) |"]
        L += [f"| {g} ({v['n']}) | {v['layer']:.3f} | {v['cover']:.3f} | {v['head']:.3f} |" for g, v in sorted(r['groups'].items())]
        L += ['', f'Cumulus skies (over {CU_CALL:.0%} of the sky):', '', '| | seen | called | hit | hit rate | false calls |', '| --- | --- | --- | --- | --- | --- |']
        L += [f"| {k} | {c['seen']} | {c['called']} | {c['hit']} | {c['pod']:.2f} | {c['far']:.2f} |" for k, c in r['calls'].items()]
    return '\n'.join(L) + '\n'

def window(month: str) -> list[str]:
    end = date.fromisoformat(month + '-01')
    end = (end.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return [(end - timedelta(days=k)).isoformat() for k in range(WINDOW_D - 1, -1, -1)]

def reader(lk):
    cols = {('obs', 'vis'): list(VIS_COLS), ('obs', 'sat'): list(SAT_COLS), ('pairs', 'sat'): list(P.KEY) + ['lead_h', 'avail_t'] + [f'f_{k}' for k in SAT_IN], ('pairs', 'metar'): list(P.KEY) + [f'f_{k}' for k in METAR_IN if k in P.METAR_VARS]}
    import pairjob
    hours = set(pairjob.SAT_HOURS)
    at_hours = lambda t: np.isin(np.asarray(t['t_valid'], np.int64) % 86400 // 3600, list(hours))
    return lambda what, kind, days: P.read_kind(lk, what, kind, days, columns=cols[what, kind], batch=station_batch, rows=(lambda t: station_rows(t) & at_hours(t)) if kind == 'metar' else station_rows)

def run(lk, today: date, again: bool=False, counts=None, read=None) -> dict | None:
    counts = counts if counts is not None else {}
    month = (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    rd = read or reader(lk)
    parts = []
    days = window(month)
    for mo in sorted({d[:7] for d in days}):
        ds = [d for d in days if d[:7] == mo]
        vis = rd('obs', 'vis', ds)
        if not P.n_rows(vis):
            continue
        sat, sp, mp = (rd('obs', 'sat', ds), rd('pairs', 'sat', ds), rd('pairs', 'metar', ds))
        got = frame(vis, sat, sp, mp)
        if got:
            parts.append(got)
    if not parts:
        counts['no_rows'] = 1
        return None
    T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    r = refit(T)
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
    ap = argparse.ArgumentParser(description='The cumulus head, refitted monthly')
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
        print('cufit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('cufit: ' + (f"the fit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('cufit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('cufit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
