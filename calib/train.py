from __future__ import annotations
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
FEATURES = ['m_xc', 'm_climb', 'm_ach', 'm_ws', 'm_zi', 'm_top', 'm_band', 'm_shade', 'm_sun', 'm_sunElev', 'm_relief', 'm_elev', 'm_cb', 'm_cu', 'm_wind', 'm_cloud', 'm_cloudlow', 'm_blapse', 'm_twidth', 'm_sink', 'm_ridgev', 'm_densAlt', 'm_solar', 'doy_s', 'doy_c', 'cls', 'grp']
PARAMS = dict(loss='absolute_error', max_iter=120, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=25, l2_regularization=1.0, random_state=7)
FOLDS = 5
PILOT_MAX_SHARE = 0.05

def feats(r: dict) -> list[float]:
    d = date.fromisoformat(r['date']).timetuple().tm_yday
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
        else:
            x = r.get(k)
            if isinstance(x, bool):
                x = float(x)
        v.append(np.nan if x is None else float(x))
    return v

def dataset(L, C, target):
    if target == 'xc':
        rs = [r for r in L if report.usable_leg(r) and (r['m_xc'] or 0) > 0 and ((r['vair'] or 0) > 0)]
        y = [math.log(r['vair'] / r['m_xc']) for r in rs]
    elif target == 'climb':
        rs = [r for r in C if r['paired'] and (not r['engon']) and ((r['rate'] or 0) > 0) and ((r['m_ach'] or 0) > 0)]
        y = [math.log(r['rate'] / r['m_ach']) for r in rs]
    else:
        rs = [r for r in L if report.usable_leg(r) and (r['therm'] or 0) > 0 and ((r['m_circ'] or 0) > 0)]
        y = [math.log(r['therm'] / r['m_circ']) for r in rs]
    X = np.array([feats(r) for r in rs], dtype=float).reshape(-1, len(FEATURES))
    return (rs, X, np.array(y), np.array([r['date'] for r in rs]))

def model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**PARAMS)

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

def crossval(X, y, pg, wg=None, w=None):
    wg = np.asarray(pg if wg is None else wg)
    pg = np.asarray(pg)
    w = np.ones_like(y) if w is None else w
    p_ml, p_const = (np.zeros_like(y), np.zeros_like(y))
    dropped = []
    for tr, te in folds(pg, wg, FOLDS):
        dropped.append(1 - len(tr) / max(1, len(y) - len(te)))
        m = model().fit(X[tr], y[tr], sample_weight=w[tr])
        p_ml[te] = m.predict(X[te])
        p_const[te] = wmedian(y[tr], w[tr])
    crossval.dropped = float(np.mean(dropped)) if dropped else 0.0
    return (p_ml, p_const)

def export(m, names) -> dict:
    trees = []
    for it in m._predictors:
        nodes = it[0].nodes
        t = []
        for n in nodes:
            if n['is_leaf']:
                t.append([float(f"{n['value']:.5g}")])
            else:
                t.append([int(n['feature_idx']), float(n['num_threshold']), int(n['missing_go_to_left']), int(n['left']), int(n['right'])])
        trees.append(t)
    base = float(np.ravel(m._baseline_prediction)[0])
    return {'features': names, 'base': float(f'{base:.7g}'), 'trees': trees}

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

def metrics(y, p, rs, label) -> dict:
    e = y - p
    return {'label': label, 'n': int(len(y)), 'median_ratio': float(np.exp(np.median(e))), 'mae_log': float(np.mean(np.abs(e))), 'iqr_ratio': [float(np.exp(np.percentile(e, 25))), float(np.exp(np.percentile(e, 75)))]}

def groups_of(rs):
    gs = {'all': lambda r: True, 'flat': rows.flat, 'hills/mountains': lambda r: not rows.flat(r)}
    for a, b in report.HOURS:
        gs[f'solar {a}–{b}'] = (lambda a, b: lambda r: r['m_solar'] is not None and a <= r['m_solar'] < b)(a, b)
    for a, b in report.WS_BANDS:
        gs[f"W* {a}–{(b if b < 99 else '+')}"] = (lambda a, b: lambda r: r['m_ws'] is not None and a <= r['m_ws'] < b)(a, b)
    return gs

def run(data: str, out: str, which: str='all') -> dict:
    os.makedirs(out, exist_ok=True)
    days = rows.load_days(data)
    L, C, F = rows.tables(days, srcs={'seed'} if which == 'seed' else None, pilots=rows.load_pilots(data))
    per = {}
    for f in F:
        if f['pilot']:
            per[f['pilot']] = per.get(f['pilot'], 0) + 1
    n = sorted(per.values())
    res, models, md = ({}, {}, [f'# Calibration model ({which}, {date.today().isoformat()})', '', f"Cross-validated by pilot (5 folds; a pilot never in both training and test, a flight without a pilot key its own group) with the weather kept apart (each fold's training without the test rows' days in their regions). Rows weighted 1/√(the pilot's flights), no pilot over {PILOT_MAX_SHARE:.0%} of the weight. Ratio: flown / the model (1 = right on); MAE: mean |log(flown / model)| (0.10 ≈ 10 %). Physics: the site's model as it is; constant: one correction factor; ML: the trees.", '', f"Pilots: {len(per)} distinct, {sum(n)} of {len(F)} flights with a key; flights per pilot: max {(n[-1] if n else '–')}, median {(np.median(n) if n else '–')}.", ''])
    for target in ('xc', 'climb', 'circ'):
        rs, X, y, g = dataset(L, C, target)
        if len(y) < 60 or len(set(g)) < 10:
            md += [f'## {target}: too few rows ({len(y)} on {len(set(g))} days)', '']
            continue
        keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).mean() >= 0.02]
        names = [FEATURES[j] for j in keep]
        X = X[:, keep]
        pg = np.array([pilot_group(r) for r in rs])
        wg = np.array([weather_group(r) for r in rs])
        w = weights(rs, per)
        p_ml, p_c = crossval(X, y, pg, wg, w)
        dropped = crossval.dropped
        zero = np.zeros_like(y)
        tab = []
        for name, sel in groups_of(rs).items():
            idx = np.array([sel(r) for r in rs])
            if idx.sum() < 15:
                continue
            row = {'group': name, 'n': int(idx.sum())}
            for lab, p in (('physics', zero), ('constant', p_c), ('ml', p_ml)):
                row[lab] = metrics(y[idx], p[idx], None, lab)
            tab.append(row)
        m = model().fit(X, y, sample_weight=w)
        e = export(m, names)
        chk = max((abs(predict_export(e, list(X[i])) - m.predict(X[i:i + 1])[0]) for i in range(min(200, len(y)))))
        models[target] = e
        res[target] = {'rows': int(len(y)), 'days': int(len(set(g))), 'groups': int(len(set(pg))), 'train_dropped': dropped, 'table': tab, 'export_check': float(chk)}
        md += [f"## {target} ({len(y)} rows, {len(set(g))} days, {len(set(pg))} pilot groups; each fold's training {dropped:.0%} smaller for the test rows' weather)", '', '| Group | n | Physics: median ratio, MAE | Constant | ML |', '|---|---|---|---|---|']
        for row in tab:
            c = lambda k: f"{row[k]['median_ratio']:.2f}, {row[k]['mae_log']:.3f}"
            md.append(f"| {row['group']} | {row['n']} | {c('physics')} | {c('constant')} | {c('ml')} |")
        imp = {}
        for t in e['trees']:
            for n in t:
                if len(n) > 1:
                    imp[names[n[0]]] = imp.get(names[n[0]], 0) + 1
        md += ['', 'Splits per feature: ' + ', '.join((f'{k} {v}' for k, v in sorted(imp.items(), key=lambda kv: -kv[1]))) + '.', f'Export check (max |trees − sklearn|): {chk:.2e}.', '']
    blob = {'version': 1, 'built': date.today().isoformat(), 'dataset': which, 'params': PARAMS, 'models': models}
    with open(os.path.join(out, 'model.json'), 'w') as f:
        json.dump(blob, f, separators=(',', ':'))
    size = os.path.getsize(os.path.join(out, 'model.json'))
    md += [f'model.json: {size / 1024:.0f} KB.']
    with open(os.path.join(out, 'train-report.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(out, 'cv.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res
if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else 'all')
