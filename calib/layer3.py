from __future__ import annotations
import copy
import json
import os
from datetime import date
import numpy as np
import report
import rows
import train
TARGETS = ('xc', 'climb', 'circ', 'exit')

def corrected_days(days: list[dict], model: str | None) -> tuple[list[dict], int, int]:
    out, done, total = ([], 0, 0)
    for d in days:
        m_ = rows.model_of(d, model)
        d2 = copy.copy(d)
        d2['flights'] = []
        for f in d['flights']:
            f2 = copy.copy(f)
            f2['legs'], f2['climbs'] = ([], [])
            for L in f.get('legs') or []:
                total += 1
                c = (L.get('models_c') or {}).get(m_)
                if c:
                    done += 1
                    L = {**L, 'models': {**(L.get('models') or {}), m_: c}}
                f2['legs'].append(L)
            for C in f.get('climbs') or []:
                total += 1
                c = (C.get('models_c') or {}).get(m_)
                if c:
                    done += 1
                    C = {**C, 'models': {**(C.get('models') or {}), m_: c}}
                f2['climbs'].append(C)
            d2['flights'].append(f2)
        out.append(d2)
    return (out, done, total)

def strength_tables(H: list[dict], S: dict) -> dict:
    out = {}
    for s in ('low', 'mid', 'high', 'unknown'):
        hs = [r for r in H if (S.get(r['fid']) or 'unknown') == s and (r.get('solar') or 0) >= report.LATE and r.get('paired')]
        held = [r for r in hs if r.get('held')]
        own = [r for r in hs if r.get('held') is False]
        ends = {}
        for r in H:
            if (S.get(r['fid']) or 'unknown') == s and r['out'] != 'flying':
                ends[r['fid']] = r['out']
        n = len(ends)
        out[s] = {'hours': len(hs), 'held': len(held), 'held_flying': sum((r['out'] == 'flying' for r in held)) / len(held) if held else None, 'own': len(own), 'own_flying': sum((r['out'] == 'flying' for r in own)) / len(own) if own else None, 'flights': n, 'landout': sum((v in ('away', 'engine') for v in ends.values())) / n if n else None}
    return out

def run(D: dict, out: str, which: str='all') -> dict:
    model = D['model']
    L, C, F, H = (D['L'], D['C'], D['F'], D['H'])
    note = "the raw forecast (no rescored rows: the data job's rescore hasn't run on these days)"
    if D['days'] is not None:
        days2, done, total = corrected_days(D['days'], model)
        if done:
            pilots = {r['fid']: r['pilot'] for r in F}
            srcs = {'seed'} if which == 'seed' else None
            L, C, F = rows.tables(days2, model=model, srcs=srcs, pilots=None)
            for r in L + C + F:
                r['pilot'] = r['pilot'] or pilots.get(r['fid'])
            note = f'the planner rescored on the layer-1-corrected air (models_c: {done} of {total} legs and climbs)'
    S = rows.strength(L, F)
    for r in L + C:
        r['strength'] = S.get(r['fid'])
    bad = D['bad']
    per = {}
    for f in F:
        if f.get('pilot'):
            per[f['pilot']] = per.get(f['pilot'], 0) + 1
    n = sorted(per.values())
    res, models, md = ({}, {}, [f"Against {note}. Cross-validated by pilot (5 folds; a pilot never in both training and test, a flight without a pilot key its own group) with the weather kept apart (each fold's training without the test rows' days in their regions). Rows weighted 1/√(the pilot's flights), no pilot over {train.PILOT_MAX_SHARE:.0%} of the weight. Pilot features here only (the glider's class and group, the pilot's strength: their XC speed against the site's on their other days, in terciles). What layer 1 corrected in the air is not learned again here: the targets are against the planner on that air.", '', f"Pilots: {len(per)} distinct, {sum(n)} of {len(F)} flights with a key; flights per pilot: max {(n[-1] if n else '–')}, median {(np.median(n) if n else '–')}; strength known for {len(S)} flights.", '', f"Suspicious days left out (the data report's, but for usable legs where the model had no lift alone): {len(bad)}" + (': ' + '; '.join((f'{d} ({w})' for d, w in sorted(bad.items()))) if bad else '') + '.', ''])
    for target in TARGETS:
        rs, X, y, g = train.dataset(L, C, target, bad)
        skipped = train.dataset.dropped
        if len(y) < 60 or len(set(g)) < 10:
            md += [f'## {target}: too few rows ({len(y)} on {len(set(g))} days)', '']
            continue
        keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).mean() >= 0.02]
        names = [train.FEATURES[j] for j in keep]
        X = X[:, keep]
        pg = np.array([train.pilot_group(r) for r in rs])
        wg = np.array([train.weather_group(r) for r in rs])
        w = train.weights(rs, per)
        p_ml, p_c = train.crossval(X, y, pg, wg, w)
        dropped = train.crossval.dropped
        p_nd = train.no_day_check(X, y, names, pg, wg, w)
        zero = np.zeros_like(y)
        preds = {'physics': zero, 'constant': p_c, 'ml': p_ml, **({'ml_noday': p_nd} if p_nd is not None else {})}
        tab = train.metric_table(rs, y, preds, train.groups_of(rs))
        if S:
            tab += train.metric_table(rs, y, preds, {f'pilot {s}': (lambda s: lambda r: r.get('strength') == s)(s) for s in ('low', 'mid', 'high')})
        m = train.model().fit(X, y, sample_weight=w)
        e = train.export(m, names)
        chk = train.export_check(e, m, X)
        models[target] = e
        res[target] = {'rows': int(len(y)), 'days': int(len(set(g))), 'groups': int(len(set(pg))), 'train_dropped': dropped, 'table': tab, 'export_check': float(chk), 'suspicious_dropped': skipped}
        md += [f"## {target} ({len(y)} rows, {len(set(g))} days, {len(set(pg))} pilot groups; {skipped} rows of suspicious days left out; each fold's training {dropped:.0%} smaller for the test rows' weather)", '']
        if p_nd is not None:
            md += train.md_table(tab, ['physics', 'constant', 'ml', 'ml_noday'], train.HEAD_NODAY)
        else:
            md += train.md_table(tab, ['physics', 'constant', 'ml'])
        md += ['', train.splits_md(e), f'Export check (max |trees − sklearn|): {chk:.2e}.', '']
    tables = strength_tables(H, S) if H else {}
    if tables:
        md += ['## Late in the day by pilot strength', '', f"Solar hour {report.LATE} on, hours with the forecast: where the evening hold put W* over the forecast's own (held) and where not (own), the share still flying an hour later; and how each pilot's flights ended: landed away or the engine started.", '', '| Strength | late hours | held: n, still flying | own: n, still flying | flights | landout or engine |', '|---|---|---|---|---|---|']
        pc = lambda x: '–' if x is None else f'{x:.0%}'
        for s, t in tables.items():
            md.append(f"| {s} | {t['hours']} | {t['held']}, {pc(t['held_flying'])} | {t['own']}, {pc(t['own_flying'])} | {t['flights']} | {pc(t['landout'])} |")
        md.append('')
    blob = {'models': models, 'cv': res, 'tables': tables, 'note': note, 'built': date.today().isoformat()}
    with open(os.path.join(out, 'layer3.json'), 'w') as f:
        json.dump(blob, f, separators=(',', ':'))
    with open(os.path.join(out, 'report-pilot.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    res['models'] = models
    res['note'] = note
    return res
