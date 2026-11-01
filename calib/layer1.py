from __future__ import annotations
import hashlib
import json
import math
import os
import statistics as st
from datetime import date, datetime, timezone
import numpy as np
import common
import report
import train
FEATURES = ['wstar', 'zi', 'cubase', 'twidth', 'blapse', 'densalt', 'rain', 'overdev', 'sun', 'cloud', 'cloudlow', 'ridge', 'elevation', 'sink', 't2', 'wind', 'solar', 'doy_s', 'doy_c', 'relief', 'model', 'lat']
ROW_OF = {'wstar': 'm_ws0', 'zi': 'm_zi', 'cubase': 'm_cb', 'twidth': 'm_twidth', 'blapse': 'm_blapse', 'densalt': 'm_densAlt', 'rain': 'm_rain', 'overdev': 'm_od', 'sun': 'm_sun', 'cloud': 'm_cloud', 'cloudlow': 'm_cloudlow', 'ridge': 'm_ridgev', 'elevation': 'm_elev', 'sink': 'm_sink', 't2': 'm_t2', 'wind': 'm_wind', 'solar': 'm_solar', 'relief': 'm_relief'}
MODEL_ID = dict(common.MODEL_INDEX)
TARGETS = ('ws', 'top', 'zi', 'cb', 'cu')
KIND = {'ws': 'log', 'top': 'log', 'zi': 'log', 'cb': 'add', 'cu': 'logit'}
CLAMPS = {'ws': [0.5, 2.0], 'top': [0.5, 2.0], 'zi': [0.5, 2.0], 'cb': 600.0, 'cu': [0.3, 0.7]}
CU_DEPTH = 300.0
TOP_QUANTILE = 0.7
CELL_DEG = 0.1
NOLIFT_WS = 1.0
MIN_ROWS, MIN_DAYS = (60, 10)
PARAMS = dict(loss='absolute_error', max_iter=80, learning_rate=0.06, max_leaf_nodes=6, min_samples_leaf=20, l2_regularization=1.0, random_state=7)
WEAK_RATE = 0.5

def doy_feats(day: str) -> tuple[float, float]:
    d = train.doy_of(day)
    return (math.sin(2 * math.pi * d / 365.25), math.cos(2 * math.pi * d / 365.25))

def feature_vector(r: dict, day: str, lat, model: str) -> list[float]:
    s, c = doy_feats(day)
    out = []
    for k in FEATURES:
        if k == 'doy_s':
            x = s
        elif k == 'doy_c':
            x = c
        elif k == 'model':
            x = MODEL_ID.get(model)
        elif k == 'lat':
            x = lat
        elif k == 'wstar':
            x = r.get('m_ws0')
            x = r.get('m_ws') if x is None else x
        else:
            x = r.get(ROW_OF[k])
        if isinstance(x, bool):
            x = float(x)
        out.append(np.nan if x is None else float(x))
    return out

def sample_row(s: dict) -> dict:
    return {'m_' + k: v for k, v in (s or {}).items()}

def _med(xs):
    xs = [x for x in xs if x is not None and (not (isinstance(x, float) and math.isnan(x)))]
    return st.median(xs) if xs else None

def _row_lat(r: dict, pos) -> float | None:
    if pos is not None:
        return pos[1]
    c = common.centroid_of(r.get('state'))
    return c[1] if c else None

def climb_positions(days, srcs=None) -> dict:
    out = {}
    for d in days or []:
        for f in d['flights']:
            if srcs and f.get('src') not in srcs:
                continue
            for k, c in enumerate(f['climbs']):
                if c.get('lon') is not None and c.get('lat') is not None:
                    out[f['id'], k] = (c['lon'], c['lat'])
    return out

def _groups(C: list[dict], pos: dict) -> dict:
    g, k_of = ({}, {})
    for r in C:
        k = k_of.get(r['fid'], 0)
        k_of[r['fid']] = k + 1
        if not r['paired'] or r.get('engon') or (r.get('rate') or 0) <= 0:
            continue
        p = pos.get((r['fid'], k))
        h = r.get('m_localHour')
        if h is None:
            continue
        cell = (round(p[0] / CELL_DEG), round(p[1] / CELL_DEG)) if p else (r.get('state') or '', 0)
        g.setdefault((r['date'], r['region'], int(h), cell), []).append((r, p))
    return g

def observations(D: dict, model: str | None=None) -> dict:
    model = model or D['model']
    out = {t: [] for t in TARGETS}
    days = D['days']
    if days is not None and model != D['model']:
        import rows as rowsmod
        L, C, F = rowsmod.tables(days, model=model)
        H, G = rowsmod.survival(days, model=model)
    else:
        C, G = (D['C'], D['G'])
    bad = D['bad']
    pos = climb_positions(days)
    for (d, reg, h, cell), items in _groups(C, pos).items():
        if d in bad:
            continue
        by_pilot = {}
        for r, p in items:
            by_pilot.setdefault(train.pilot_group(r), []).append(r['rate'])
        med = st.median([st.median(v) for v in by_pilot.values()])
        ach = _med([r['m_ach'] for r, _ in items])
        ref = {k: _med([r.get(k) for r, _ in items]) if not isinstance(items[0][0].get(k), (str, bool)) else items[0][0].get(k) for k in items[0][0] if k.startswith('m_')}
        ref.update({'date': d, 'region': reg, 'state': items[0][0].get('state'), 'n': len(items), 'pilots': len(by_pilot), 'model': model or items[0][0].get('model')})
        lat = _med([_row_lat(r, p) for r, p in items])
        x = feature_vector(ref, d, lat, ref['model'])
        w = min(2.0, math.sqrt(len(by_pilot)))
        if ach and ach > 0:
            out['ws'].append({'x': x, 'y': math.log(med / ach), 'w': w, 'date': d, 'region': reg, 'kind': 'climbs', 'ref': ref})
        tops = [r['agl1'] for r, _ in items if (r.get('agl1') or 0) > 100]
        top = _med([r['m_top'] for r, _ in items])
        if tops and top and (top > 200):
            out['top'].append({'x': x, 'y': math.log(max(tops) / top), 'w': min(2.0, math.sqrt(len(tops))), 'date': d, 'region': reg, 'kind': 'exit tops', 'ref': ref})
    for r in G:
        if r.get('failed') and r.get('paired') and ((r.get('m_ach') or 0) > 0) and ((r.get('m_ws0') or 0) >= NOLIFT_WS) and (r['date'] not in bad):
            ref = {**r, 'n': 1, 'pilots': 1}
            out['ws'].append({'x': feature_vector(ref, r['date'], _row_lat(r, None), model or r.get('model')), 'y': math.log(WEAK_RATE / r['m_ach']), 'w': 0.5, 'date': r['date'], 'region': r['region'], 'kind': 'failed search', 'ref': ref})
    import rows as rowsmod
    for d in days or []:
        if d['date'] in bad:
            continue
        m_ = rowsmod.model_of(d, model)
        for o in d.get('obs') or []:
            pm = (o.get('probe_m') or {}).get(m_) or {}
            ob = o.get('obs') or {}
            lat = (o.get('tp') or [None, None])[1]
            if o['kind'] == 'raob':
                s = pm.get(str(int(o['t']))) if o.get('t') is not None else None
                if s and (s.get('zi') or 0) > 50 and ((ob.get('mix') or 0) > 50):
                    ref = {**sample_row(s), 'date': d['date'], 'region': o['region'], 'sid': o['sid']}
                    out['zi'].append({'x': feature_vector(ref, d['date'], lat, m_), 'y': math.log(ob['mix'] / s['zi']), 'w': 1.0, 'date': d['date'], 'region': o['region'], 'kind': 'sounding', 'ref': ref})
            elif o['kind'] == 'asos':
                from zoneinfo import ZoneInfo
                z = ZoneInfo(o['tz'])
                fc = {str(datetime.fromtimestamp(int(t), z).hour): s for t, s in pm.items() if s}
                for hh, rep in (ob.get('hours') or {}).items():
                    s = fc.get(hh)
                    if not s or s.get('ws0' if 'ws0' in s else 'ws') is None:
                        continue
                    ref = {**sample_row(s), 'date': d['date'], 'region': o['region'], 'sid': o['sid'], 'hour': int(hh)}
                    x = feature_vector(ref, d['date'], lat, m_)
                    cu = rep.get('cu')
                    out['cu'].append({'x': x, 'y': 1.0 if cu is not None else 0.0, 'w': 1.0, 'date': d['date'], 'region': o['region'], 'kind': 'station hour', 'ref': ref})
                    if cu is not None and s.get('cb') is not None:
                        out['cb'].append({'x': x, 'y': float(cu - s['cb']), 'w': 1.0, 'date': d['date'], 'region': o['region'], 'kind': 'station hour', 'ref': ref})
    return out

def _xy(obs: list[dict]):
    X = np.array([o['x'] for o in obs], dtype=float).reshape(-1, len(FEATURES))
    y = np.array([o['y'] for o in obs], dtype=float)
    w = np.array([o['w'] for o in obs], dtype=float)
    g = np.array([o['date'] for o in obs])
    return (X, y, w, g)

def _params(target: str) -> dict:
    if target == 'top':
        return {**PARAMS, 'loss': 'quantile', 'quantile': TOP_QUANTILE}
    if target == 'cb':
        return {**PARAMS}
    return PARAMS

def _make(target: str):
    return train.classifier if target == 'cu' else train.model

def _groups_for(obs: list[dict]) -> dict:
    refs = [o['ref'] for o in obs]
    gs = {'all': lambda r: True, 'flat': lambda r: (r.get('m_relief') or 0) < common.RELIEF_FLAT, 'hills/mountains': lambda r: (r.get('m_relief') or 0) >= common.RELIEF_FLAT}
    for a, b in report.HOURS:
        gs[f'solar {a}–{b}'] = (lambda a, b: lambda r: r.get('m_solar') is not None and a <= r['m_solar'] < b)(a, b)
    for a, b in report.WS_BANDS:
        gs[f"W* {a}–{(b if b < 99 else '+')}"] = (lambda a, b: lambda r: (r.get('m_ws0') if r.get('m_ws0') is not None else r.get('m_ws')) is not None and a <= (r.get('m_ws0') if r.get('m_ws0') is not None else r.get('m_ws')) < b)(a, b)
    gs['cumulus'] = lambda r: r.get('m_cb') is not None
    gs['blue'] = lambda r: r.get('m_cb') is None
    kinds = {o['kind'] for o in obs}
    if len(kinds) > 1:
        for k in sorted(kinds):
            gs[k] = (lambda k: lambda r: r.get('_kind') == k)(k)
    for r, o in zip(refs, obs):
        r['_kind'] = o['kind']
    return gs

def _metrics_add(y, p, label) -> dict:
    e = y - p
    return {'label': label, 'n': int(len(y)), 'median_m': float(np.median(e)), 'mae_m': float(np.mean(np.abs(e)))}

def _metrics_cu(y, p_logit, label, fc=None) -> dict:
    p = fc if fc is not None else 1 / (1 + np.exp(-p_logit))
    return {'label': label, 'n': int(len(y)), 'brier': float(np.mean((p - y) ** 2)), 'hit': float(np.mean((p >= 0.5) == (y >= 0.5))), 'auc': auc(p, y)}

def auc(score, y) -> float | None:
    score, y = (np.asarray(score, dtype=float), np.asarray(y, dtype=float))
    pos, neg = (score[y >= 0.5], score[y < 0.5])
    if not len(pos) or not len(neg):
        return None
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def fit_target(target: str, obs: list[dict]) -> dict | None:
    if len(obs) < MIN_ROWS or len({o['date'] for o in obs}) < MIN_DAYS:
        return None
    X, y, w, g = _xy(obs)
    if target == 'cu' and (y.sum() < 10 or (1 - y).sum() < 10):
        return None
    keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).mean() >= 0.02]
    names = [FEATURES[j] for j in keep]
    X = X[:, keep]
    p_ml, p_c = train.crossval(X, y, g, g, w, _params(target), _make(target))
    if target == 'cu':
        r = float(np.clip(np.average(y, weights=w), 0.01, 0.99))
        p_c = np.full_like(y, math.log(r / (1 - r)))
    zero = np.zeros_like(y)
    refs = [o['ref'] for o in obs]
    gs = _groups_for(obs)
    tab = []
    for name, sel in gs.items():
        idx = np.array([bool(sel(r)) for r in refs])
        if idx.sum() < 15:
            continue
        row = {'group': name, 'n': int(idx.sum())}
        if target == 'cu':
            fc = np.array([1.0 if r.get('m_cb') is not None else 0.0 for r in refs])
            row['physics'] = _metrics_cu(y[idx], None, 'physics', fc[idx])
            row['constant'] = _metrics_cu(y[idx], p_c[idx], 'constant')
            row['ml'] = _metrics_cu(y[idx], p_ml[idx], 'ml')
        elif target == 'cb':
            for lab, p in (('physics', zero), ('constant', p_c), ('ml', p_ml)):
                row[lab] = _metrics_add(y[idx], p[idx], lab)
        else:
            for lab, p in (('physics', zero), ('constant', p_c), ('ml', p_ml)):
                row[lab] = train.metrics(y[idx], p[idx], None, lab)
        tab.append(row)
    m = _make(target)(_params(target)).fit(X, y, sample_weight=w)
    e = train.export(m, names)
    e['kind'] = KIND[target]
    chk = train.export_check(e, m, X)
    return {'export': e, 'table': tab, 'rows': int(len(y)), 'days': int(len(set(g))), 'check': chk, 'train_dropped': train.crossval.dropped, 'kinds': {k: int(sum((1 for o in obs if o['kind'] == k))) for k in sorted({o['kind'] for o in obs})}}

def compare_models(D: dict, models: list[str]) -> dict:
    out = {}
    for m in models:
        obs = observations(D, m)
        for t in TARGETS:
            if t == 'cu':
                continue
            for o in obs[t]:
                r = o['ref']
                for grp in ('all', f"region {o['region']}", 'solar ' + report.hour_bin(r.get('m_solar'))):
                    out.setdefault(t, {}).setdefault(m, {}).setdefault(grp, []).append(o['y'])
    res = {}
    for t, by_m in out.items():
        for m, by_g in by_m.items():
            for grp, ys in by_g.items():
                if len(ys) >= 10:
                    a = np.array(ys)
                    res.setdefault(t, {}).setdefault(m, {})[grp] = {'n': int(len(a)), 'median': float(np.exp(np.median(a)) if KIND[t] == 'log' else np.median(a)), 'mae': float(np.mean(np.abs(a)))}
    return res

def compare_by_macro(D: dict, models: list[str]) -> dict:
    out = {}
    for macro in sorted({common.MODEL_MACRO.get(m, 'US') for m in models}):
        ms = [m for m in models if common.MODEL_MACRO.get(m, 'US') == macro]
        if len(ms) > 1:
            cmp_ = compare_models(D, ms)
            if cmp_:
                out[macro] = cmp_
    return out

def corrections_blob(fits: dict) -> dict:
    fields = {t: f['export'] for t, f in fits.items() if f}
    body = {'features': FEATURES, 'model_ids': MODEL_ID, 'clamps': CLAMPS, 'cu_depth': CU_DEPTH, 'fields': fields}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    return {'version': h, 'built': date.today().isoformat(), **body}

def run(D: dict, out: str) -> dict:
    models = sorted({m for d in D['days'] or [] for m in d.get('models') or [d.get('model') or 'hrrr']} or {D['model'] or 'hrrr'})
    obs = observations(D)
    fits = {t: fit_target(t, obs[t]) for t in TARGETS}
    corr = corrections_blob(fits)
    with open(os.path.join(out, 'corrections.json'), 'w') as f:
        json.dump(corr, f, separators=(',', ':'))
    by_macro = compare_by_macro(D, models) if len(models) > 1 else {}
    cmp_ = next(iter(by_macro.values())) if len(by_macro) == 1 else {m: t for c in by_macro.values() for m, t in c.items()} if by_macro else {}
    cmp_ = {t: {m: g for c in by_macro.values() for m, g in (c.get(t) or {}).items()} for t in TARGETS if any((t in c for c in by_macro.values()))}
    md = [f"Corrections to the forecast fields from every observation of the air: the stations' reports (cumulus and its base, every built day, flown or not), the evening soundings (the layer's depth) and the flights read as observations of the air, not of a pilot (a day-hour-cell's median climb across pilots against the model's achieved climb; the highest exit top as a lower bound on the usable top, fitted at the {TOP_QUANTILE:.0%} quantile; a failed search where the forecast's own W* was ≥ {NOLIFT_WS} m/s as a climb of at most {WEAK_RATE} m/s). Features: only what a grid cell has per hour ({', '.join(FEATURES)}). Cross-validated by day. What this layer explains is taken out before layers 2 and 3 learn: they see the planner rescored on the corrected air (models_c), not these residuals.", '']
    if D['csv']:
        md += ["Exported rows only (no day files): the flights' targets alone, no stations or soundings, cells by state.", '']
    if D['bad']:
        md += [f"Suspicious days left out: {len(D['bad'])}.", '']
    for t in TARGETS:
        f = fits[t]
        n = len(obs[t])
        if not f:
            md += [f"## {t}: too few rows ({n} on {len({o['date'] for o in obs[t]})} days)", '']
            continue
        md += [f"## {t} ({f['rows']} rows, {f['days']} days; " + ', '.join((f'{k} {v}' for k, v in f['kinds'].items())) + f"; each fold's training {f['train_dropped']:.0%} smaller for the test rows' days)", '']
        if t == 'cu':
            md += ['| Group | n | Forecast: Brier, hit rate, AUC | Constant | ML |', '|---|---|---|---|---|']
            for row in f['table']:
                c = lambda k: f"{row[k]['brier']:.3f}, {row[k]['hit']:.2f}, {(row[k]['auc'] if row[k]['auc'] is None else round(row[k]['auc'], 2))}"
                md.append(f"| {row['group']} | {row['n']} | {c('physics')} | {c('constant')} | {c('ml')} |")
        elif t == 'cb':
            md += ['| Group | n | Physics: median shift (m), MAE (m) | Constant | ML |', '|---|---|---|---|---|']
            for row in f['table']:
                c = lambda k: f"{row[k]['median_m']:.0f}, {row[k]['mae_m']:.0f}"
                md.append(f"| {row['group']} | {row['n']} | {c('physics')} | {c('constant')} | {c('ml')} |")
        else:
            md += train.md_table(f['table'], ['physics', 'constant', 'ml'])
        md += ['', train.splits_md(f['export']), f"Export check (max |trees − sklearn|): {f['check']:.2e}.", '']
    if by_macro:
        md += ['## Which model is nearer where and when', '', "The observations' median ratio (cb: shift in m) and MAE per model, the forecast as it is; the nearest (lowest MAE) in bold. Each area's models against each other (the US's HRRR and GFS, Europe's ECMWF and ICON, ICON-D2 where its domain covers the box, on the sampled days).", '']
        for macro, cmp_m in by_macro.items():
            if len(by_macro) > 1:
                md += [f'### {common.MACRO_LABEL.get(macro, macro)}', '']
            for t, by_m in cmp_m.items():
                ms = sorted(by_m)
                md += [f'### {t}' + (f' ({macro})' if len(by_macro) > 1 else ''), '', '| Group | ' + ' | '.join((f'{common.model_label(m)}: n, median, MAE' for m in ms)) + ' |', '|---|' + '---|' * len(ms)]
                groups = sorted({g for m in ms for g in by_m[m]}, key=lambda g: (g != 'all', g))
                for g in groups:
                    cells, best = ([], min(((by_m[m][g]['mae'], m) for m in ms if g in by_m[m]))[1])
                    for m in ms:
                        v = by_m[m].get(g)
                        s = f"{v['n']}, {v['median']:.2f}, {v['mae']:.3f}" if v else '–'
                        cells.append(f'**{s}**' if v and m == best else s)
                    md.append(f'| {g} | ' + ' | '.join(cells) + ' |')
                md.append('')
    else:
        md += [f'## Which model is nearer where and when', '', f"One model in the data ({', '.join(models)}): the comparison needs the data job to store a second one (models[<model>] per leg and climb).", '']
    md += [f"corrections.json: version {corr['version']}, fields {', '.join(corr['fields']) or 'none'}; clamps {json.dumps(CLAMPS)}.", '']
    with open(os.path.join(out, 'report-air.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    res = {'rows': sum((f['rows'] for f in fits.values() if f)), 'targets': {t: {k: v for k, v in f.items() if k != 'export'} if f else None for t, f in fits.items()}, 'tables': {t: f['table'] for t, f in fits.items() if f}, 'models_compared': cmp_, 'models_compared_by_area': by_macro, 'version': corr['version'], 'models': models, 'cv': {t: {'rows': f['rows'], 'days': f['days'], 'table': f['table']} for t, f in fits.items() if f}}
    with open(os.path.join(out, 'layer1.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res
RELIEF_DEG = 0.15

def _sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relief_grid(elev: np.ndarray, step: float) -> np.ndarray:
    r = int(round(RELIEF_DEG / step))
    ny, nx = elev.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            w = elev[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
            w = w[np.isfinite(w)]
            if w.size:
                out[iy, ix] = w.max() - w.min()
    return out

def apply_corrections(fields: dict, meta: dict, corr: dict) -> np.ndarray:
    names, nx, ny, nh = (fields['names'], fields['nx'], fields['ny'], fields['nh'])
    A = np.array(fields.get('data', fields.get('all')), dtype=np.float32).reshape(nh, len(names), ny, nx)
    out = A.copy()
    fi = {n: i for i, n in enumerate(names)}
    g = fields.get('grid') or meta['grid']
    ds, dc = doy_feats(meta['day'])
    mid = meta.get('modelIndex')
    mid = corr['model_ids'].get(meta.get('model'), math.nan) if mid is None else mid

    def hour_ms(h):
        st_ = (fields.get('stamps') or [None] * nh)[h]
        if st_ is None:
            v = meta['hours'][h]
            return v['utc'] if isinstance(v, dict) else v
        if meta.get('stampUtc') and st_ in meta['stampUtc']:
            return meta['stampUtc'][st_]
        return datetime.strptime(st_, '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp() * 1000
    relief = relief_grid(A[0, fi['elevation']], g['step']) if 'elevation' in fi else np.full((ny, nx), np.nan, np.float32)
    F = corr['fields']
    cl = corr['clamps']
    feat_idx = {k: i for i, k in enumerate(corr['features'])}

    def val(h, name, iy, ix):
        if name not in fi:
            return math.nan
        return float(A[h, fi[name], iy, ix])
    for h in range(nh):
        utc_h = hour_ms(h) / 3600000 % 24
        for iy in range(ny):
            lat = g['north'] - iy * g['step']
            for ix in range(nx):
                ws = val(h, 'wstar', iy, ix)
                if not math.isfinite(ws):
                    continue
                lon = g['west'] + ix * g['step']
                x = [math.nan] * len(corr['features'])
                for k in corr['features']:
                    if k == 'wind':
                        v = math.hypot(val(h, 'u_blavg', iy, ix), val(h, 'v_blavg', iy, ix))
                    elif k == 'solar':
                        v = ((utc_h + lon / 15) % 24 + 24) % 24
                    elif k == 'doy_s':
                        v = ds
                    elif k == 'doy_c':
                        v = dc
                    elif k == 'relief':
                        v = float(relief[iy, ix])
                    elif k == 'model':
                        v = mid
                    elif k == 'lat':
                        v = lat
                    else:
                        v = val(h, k, iy, ix)
                    x[feat_idx[k]] = v
                row = lambda e: [x[feat_idx[k]] for k in e['features']]
                cb = val(h, 'cubase', iy, ix)
                blue = not math.isfinite(cb)
                if 'ws' in F:
                    f = min(cl['ws'][1], max(cl['ws'][0], math.exp(train.predict_export(F['ws'], row(F['ws'])))))
                    out[h, fi['wstar'], iy, ix] = ws * f
                layer = F.get('top') if blue and 'top' in F else F.get('zi')
                if layer is not None:
                    key = 'top' if layer is F.get('top') else 'zi'
                    f = min(cl[key][1], max(cl[key][0], math.exp(train.predict_export(layer, row(layer)))))
                    for name in ('zi', 'depth'):
                        if name in fi and math.isfinite(val(h, name, iy, ix)):
                            out[h, fi[name], iy, ix] = val(h, name, iy, ix) * f
                if 'cb' in F and (not blue):
                    s = min(cl['cb'], max(-cl['cb'], train.predict_export(F['cb'], row(F['cb']))))
                    out[h, fi['cubase'], iy, ix] = max(50.0, cb + s)
                if 'cu' in F and 'cubase' in fi:
                    p = _sigmoid(train.predict_export(F['cu'], row(F['cu'])))
                    if not blue and p < cl['cu'][0]:
                        out[h, fi['cubase'], iy, ix] = math.nan
                        if 'cudepth' in fi:
                            out[h, fi['cudepth'], iy, ix] = math.nan
                    elif blue and p > cl['cu'][1]:
                        zi = float(out[h, fi['zi'], iy, ix]) if 'zi' in fi else math.nan
                        if math.isfinite(zi) and zi > 0:
                            out[h, fi['cubase'], iy, ix] = zi
                            if 'cudepth' in fi:
                                out[h, fi['cudepth'], iy, ix] = corr.get('cu_depth', CU_DEPTH)
    return out.reshape(-1)
