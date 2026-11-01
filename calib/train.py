from __future__ import annotations
import argparse
import csv
import glob
import gzip
import json
import math
import os
import sys
from datetime import date
import numpy as np
import report
import rows
CLS = {'twoseat': 0, 'club': 1, 'standard': 2, '15m': 3, '18m': 4, 'open': 5}
GRP = {'vlow': 0, 'low': 1, 'high': 2}
STR = {'low': 0, 'mid': 1, 'high': 2}
FEATURES = ['m_xc', 'm_climb', 'm_ach', 'm_ws', 'm_zi', 'm_top', 'm_band', 'm_shade', 'm_sun', 'm_sunElev', 'm_relief', 'm_elev', 'm_cb', 'm_cu', 'm_wind', 'm_cloud', 'm_cloudlow', 'm_blapse', 'm_twidth', 'm_sink', 'm_ridgev', 'm_densAlt', 'm_solar', 'doy_s', 'doy_c', 'cls', 'grp', 'strength']
PARAMS = dict(loss='absolute_error', max_iter=120, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=25, l2_regularization=1.0, random_state=7)
FOLDS = 5
PILOT_MAX_SHARE = 0.05
MODEL_VERSION = 2

def doy_of(day: str) -> int:
    return date.fromisoformat(day).timetuple().tm_yday

def feats(r: dict) -> list[float]:
    d = doy_of(r['date'])
    v = []
    for k in FEATURES:
        if k == 'doy_s':
            x = math.sin(2 * math.pi * d / 365.25)
        elif k == 'doy_c':
            x = math.cos(2 * math.pi * d / 365.25)
        elif k == 'cls':
            x = CLS.get(r['cls'])
        elif k == 'grp':
            x = GRP.get(r['grp'])
        elif k == 'strength':
            x = STR.get(r.get('strength'))
        else:
            x = r.get(k)
            if isinstance(x, bool):
                x = float(x)
        v.append(np.nan if x is None else float(x))
    return v

def dataset(L, C, target, skip=()):
    if target == 'xc':
        rs = [r for r in L if report.usable_leg(r) and (r['m_xc'] or 0) > 0 and ((r['vair'] or 0) > 0)]
        y = [math.log(r['vair'] / r['m_xc']) for r in rs]
    elif target == 'climb':
        rs = [r for r in C if r['paired'] and (not r['engon']) and ((r['rate'] or 0) > 0) and ((r['m_ach'] or 0) > 0)]
        y = [math.log(r['rate'] / r['m_ach']) for r in rs]
    elif target == 'exit':
        rs = [r for r in C if r['paired'] and (not r['engon']) and ((r.get('agl1') or 0) > 100) and ((r['m_top'] or 0) > 200)]
        y = [math.log(r['agl1'] / r['m_top']) for r in rs]
    else:
        rs = [r for r in L if report.usable_leg(r) and (r['therm'] or 0) > 0 and ((r['m_circ'] or 0) > 0)]
        y = [math.log(r['therm'] / r['m_circ']) for r in rs]
    keep = [r['date'] not in skip for r in rs]
    dataset.dropped = len(rs) - sum(keep)
    rs, y = ([r for r, k in zip(rs, keep) if k], [v for v, k in zip(y, keep) if k])
    X = np.array([feats(r) for r in rs], dtype=float).reshape(-1, len(FEATURES))
    return (rs, X, np.array(y), np.array([r['date'] for r in rs]))

def model(params: dict | None=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**params or PARAMS)

def classifier(params: dict | None=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = {k: v for k, v in (params or PARAMS).items() if k != 'loss'}
    return HistGradientBoostingClassifier(**p)

def pilot_group(r) -> str:
    return f"p:{r['pilot']}" if r.get('pilot') else f"f:{r['fid']}"

def weather_group(r) -> str:
    return f"{r['date']}|{r['region']}"

def weights(rs, flights_of: dict | None=None) -> np.ndarray:
    if flights_of is None:
        fl = {}
        for r in rs:
            if r.get('pilot'):
                fl.setdefault(r['pilot'], set()).add(r['fid'])
        flights_of = {p: len(v) for p, v in fl.items()}
    w = np.array([1 / math.sqrt(flights_of.get(r.get('pilot'), 1)) if r.get('pilot') else 1.0 for r in rs], dtype=float)
    keys = np.array([r.get('pilot') or '' for r in rs])
    groups = len(set(keys) - {''}) + len({r['fid'] for r in rs if not r.get('pilot')})
    for _ in range(5 if groups * PILOT_MAX_SHARE >= 1 else 0):
        tot, over = (w.sum(), False)
        for p in set(keys) - {''}:
            idx = keys == p
            s = w[idx].sum()
            if tot > 0 and s > PILOT_MAX_SHARE * tot * 1.0001:
                rest = tot - s
                w[idx] *= PILOT_MAX_SHARE * rest / (1 - PILOT_MAX_SHARE) / s if rest > 0 else 1
                over = True
        if not over:
            break
    return w

def wmedian(y, w) -> float:
    o = np.argsort(y)
    c = np.cumsum(w[o])
    return float(y[o][np.searchsorted(c, c[-1] / 2)])

def folds(pg, wg, k):
    from sklearn.model_selection import GroupKFold
    k = min(k, len(set(pg)))
    for tr, te in GroupKFold(n_splits=k).split(np.zeros(len(pg)), groups=pg):
        bad = set(wg[te])
        yield (np.array([i for i in tr if wg[i] not in bad], dtype=int), te)

def crossval(X, y, pg, wg=None, w=None, params: dict | None=None, make=None):
    wg = np.asarray(pg if wg is None else wg)
    pg = np.asarray(pg)
    w = np.ones_like(y, dtype=float) if w is None else w
    p_ml, p_const = (np.zeros_like(y, dtype=float), np.zeros_like(y, dtype=float))
    dropped = []
    for tr, te in folds(pg, wg, FOLDS):
        dropped.append(1 - len(tr) / max(1, len(y) - len(te)))
        m = (make or model)(params).fit(X[tr], y[tr], sample_weight=w[tr])
        p_ml[te] = raw_predict(m, X[te])
        p_const[te] = wmedian(y[tr], w[tr])
    crossval.dropped = float(np.mean(dropped)) if dropped else 0.0
    return (p_ml, p_const)

def raw_predict(m, X):
    return m.decision_function(X) if hasattr(m, 'decision_function') else m.predict(X)

def export(m, names) -> dict:
    trees = []
    for it in m._predictors:
        nodes = it[0].nodes
        t = []
        for n in nodes:
            if n['is_leaf']:
                t.append([float(f"{n['value']:.5g}")])
            else:
                thr = float(n['num_threshold'])
                if not math.isfinite(thr):
                    thr = 1e+300 if thr > 0 else -1e+300
                t.append([int(n['feature_idx']), thr, int(n['missing_go_to_left']), int(n['left']), int(n['right'])])
        trees.append(t)
    base = float(np.ravel(m._baseline_prediction)[0])
    return {'features': list(names), 'base': float(f'{base:.7g}'), 'trees': trees}

def predict_export(e: dict, x) -> float:
    s = e['base']
    for t in e['trees']:
        i = 0
        while len(t[i]) > 1:
            f, thr, ml, lft, rgt = t[i]
            v = x[f]
            i = (lft if ml else rgt) if v is None or (isinstance(v, float) and math.isnan(v)) else lft if v <= thr else rgt
        s += t[i][0]
    return s

def export_check(e: dict, m, X, n: int=200) -> float:
    return float(max((abs(predict_export(e, list(X[i])) - raw_predict(m, X[i:i + 1])[0]) for i in range(min(n, len(X))))))

def metrics(y, p, rs, label) -> dict:
    e = y - p
    return {'label': label, 'n': int(len(y)), 'median_ratio': float(np.exp(np.median(e))), 'mae_log': float(np.mean(np.abs(e))), 'iqr_ratio': [float(np.exp(np.percentile(e, 25))), float(np.exp(np.percentile(e, 75)))]}

def groups_of(rs, ws_key='m_ws', solar_key='m_solar'):
    gs = {'all': lambda r: True, 'flat': rows.flat, 'hills/mountains': lambda r: not rows.flat(r)}
    for a, b in report.HOURS:
        gs[f'solar {a}–{b}'] = (lambda a, b: lambda r: r.get(solar_key) is not None and a <= r[solar_key] < b)(a, b)
    for a, b in report.WS_BANDS:
        gs[f"W* {a}–{(b if b < 99 else '+')}"] = (lambda a, b: lambda r: r.get(ws_key) is not None and a <= r[ws_key] < b)(a, b)
    return gs

def metric_table(rs, y, preds: dict, groups: dict, min_n: int=15) -> list[dict]:
    tab = []
    for name, sel in groups.items():
        idx = np.array([bool(sel(r)) for r in rs])
        if idx.sum() < min_n:
            continue
        row = {'group': name, 'n': int(idx.sum())}
        for lab, p in preds.items():
            row[lab] = metrics(y[idx], p[idx], None, lab)
        tab.append(row)
    return tab

def md_table(tab: list[dict], labels: list[str], head: str='Physics: median ratio, MAE | Constant | ML') -> list[str]:
    out = ['| Group | n | ' + head + ' |', '|---|---|' + '---|' * len(labels)]
    for row in tab:
        c = lambda k: f"{row[k]['median_ratio']:.2f}, {row[k]['mae_log']:.3f}"
        out.append(f"| {row['group']} | {row['n']} | " + ' | '.join((c(k) for k in labels)) + ' |')
    return out

def splits_md(e: dict) -> str:
    imp = {}
    for t in e['trees']:
        for n in t:
            if len(n) > 1:
                imp[e['features'][n[0]]] = imp.get(e['features'][n[0]], 0) + 1
    return 'Splits per feature: ' + ', '.join((f'{k} {v}' for k, v in sorted(imp.items(), key=lambda kv: -kv[1]))) + '.'

def _typed(v: str):
    if v == '':
        return None
    if v == 'True':
        return True
    if v == 'False':
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v

def csv_rows(path: str) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [{k: _typed(v) for k, v in r.items()} for r in csv.DictReader(f)]

def load(data: str, which: str='all', model: str | None=None) -> dict:
    days = rows.load_days(data)
    srcs = {'seed'} if which == 'seed' else None
    if days:
        pilots = rows.load_pilots(data)
        L, C, F = rows.tables(days, model=model, srcs=srcs, pilots=pilots)
        H, G = rows.survival(days, model=model, srcs=srcs, pilots=pilots)
        bad = report.bad_days(report.day_checks(days, *(rows.tables(days, model=model)[:2] if which == 'seed' else (L, C))))
        return {'days': days, 'L': L, 'C': C, 'F': F, 'H': H, 'G': G, 'bad': bad, 'csv': False, 'model': model}
    tabs = {}
    for name in ('legs', 'climbs', 'flights', 'hours', 'glides'):
        p = os.path.join(data, f'{name}.csv.gz')
        tabs[name] = csv_rows(p) if os.path.exists(p) else []
        if srcs:
            tabs[name] = [r for r in tabs[name] if r.get('src') in srcs]
    if not tabs['legs'] and (not tabs['climbs']):
        raise SystemExit(f'train: no days or exported rows in {data}')
    for t in tabs.values():
        for r in t:
            r.setdefault('pilot', None)
            r.setdefault('last', False)
    fake = [{'date': d, 'flights': [], 'boxes': []} for d in sorted({r['date'] for r in tabs['legs'] + tabs['climbs']})]
    for d in fake:
        d['flights'] = [{'region': r['region']} for r in tabs['flights'] if r['date'] == d['date']]
    bad = report.bad_days(report.day_checks(fake, tabs['legs'], tabs['climbs']))
    return {'days': None, 'L': tabs['legs'], 'C': tabs['climbs'], 'F': tabs['flights'], 'H': tabs['hours'], 'G': tabs['glides'], 'bad': bad, 'csv': True, 'model': model}
STAGES = ('air', 'place', 'pilot')

def run(data: str, out: str, which: str='all', stage: str='all', grids: str | None=None) -> dict:
    import layer1
    import layer2
    import layer3
    os.makedirs(out, exist_ok=True)
    D = load(data, which)
    res = {}
    if stage in ('air', 'all'):
        res['layer1'] = layer1.run(D, out)
    if stage in ('place', 'all'):
        res['layer2'] = layer2.run(D, out, grids=grids)
    if stage in ('pilot', 'all'):
        l3 = layer3.run(D, out, which)
        models, note = (l3.pop('models'), l3.pop('note'))
        res.update(l3)
        res['layer3'] = {'rows': sum((r['rows'] for r in l3.values())), 'note': note, 'models': models}
        assemble(out, which)
    return res

def _read(out: str, name: str):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f) if name.endswith('.json') else f.read()

def assemble(out: str, which: str) -> None:
    l1, l2, l3 = (_read(out, 'layer1.json'), _read(out, 'layer2.json'), _read(out, 'layer3.json'))
    corr, trig = (_read(out, 'corrections.json'), _read(out, 'triggers.json'))
    md = [f'# Calibration model ({which}, {date.today().isoformat()})', '', "Three layers trained in sequence, each on the residual after the layers before it (nothing counted twice): **1 the air** (the forecast fields corrected from the stations, the soundings and the flights read as observations of the air; the next layers see the planner rescored on that air), **2 the place** (where the thermals are, given that air), **3 the pilot** (how a pilot flies that air and those thermals; pilot features here only). Layers 1 and 2 are cross-validated by day, layer 3 by pilot with the test rows' days kept out per region. Ratio: observed / the forecast (1 = right on); MAE: mean |log(observed / forecast)| (0.10 ≈ 10 %). Physics: the site's model as it is; constant: one correction factor; ML: the trees.", '']
    for name, part in (('Layer 1: the air', 'report-air.md'), ('Layer 2: the place', 'report-place.md'), ('Layer 3: the pilot', 'report-pilot.md')):
        text = _read(out, part)
        md += [f'# {name}', ''] + ([text.rstrip(), ''] if text else ['Not run.', ''])
    blob = {'version': MODEL_VERSION, 'built': date.today().isoformat(), 'dataset': which, 'params': PARAMS, 'models': (l3 or {}).get('models') or {}, 'layer1': {'corrections': corr, 'tables': (l1 or {}).get('tables'), 'models_compared': (l1 or {}).get('models_compared')}, 'layer2': {'triggers': trig, 'scores': (l2 or {}).get('scores')}, 'layer3': {'tables': (l3 or {}).get('tables'), 'note': (l3 or {}).get('note')}}
    with open(os.path.join(out, 'model.json'), 'w') as f:
        json.dump(blob, f, separators=(',', ':'))
    size = os.path.getsize(os.path.join(out, 'model.json'))
    md += [f'model.json: {size / 1024:.0f} KB' + (f"; corrections.json: {os.path.getsize(os.path.join(out, 'corrections.json')) / 1024:.0f} KB" if corr else '') + (f"; triggers.json: {os.path.getsize(os.path.join(out, 'triggers.json')) / 1024:.0f} KB" if trig else '') + '.']
    with open(os.path.join(out, 'train-report.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    cv = {k: v for k, v in ((l3 or {}).get('cv') or {}).items()}
    cv['layer1'] = (l1 or {}).get('cv')
    cv['layer2'] = (l2 or {}).get('cv')
    with open(os.path.join(out, 'cv.json'), 'w') as f:
        json.dump(cv, f, indent=1)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data')
    ap.add_argument('out')
    ap.add_argument('dataset', nargs='?', default='all', choices=['all', 'seed'])
    ap.add_argument('--stage', default='all', choices=list(STAGES) + ['all'])
    ap.add_argument('--grids', default=None, help='the saved grids (calib-grids-<month> bundles unpacked), for layer 2')
    a = ap.parse_args(argv)
    run(a.data, a.out, a.dataset, a.stage, a.grids)
    return 0
if __name__ == '__main__':
    sys.exit(main())
