from __future__ import annotations
import math
from . import learn as lab
TRIGGER_V = 't3'
TRIG = {'hour_bin': 2, 'kappa': 1.0, 'ridge': 1.0, 'slot_cap': 120, 'boot': 400, 'seed': 1}
SLOT_H = 30 / 3600
SWEEP_MS = {'glider': 35.0, 'hg': 16.0, 'pg': 11.0}

def sweep_weights(sig: dict | None=None) -> list:
    by = (sig or {}).get('by_type') or {}
    s0 = (by.get('glider') or {}).get('m') or (sig or {}).get('m')
    out = []
    for t in lab.ex.TYPES:
        st = (by.get(t) or {}).get('m')
        ratio = st / s0 if st and s0 else 1.0
        out.append(round(min(1.0, SWEEP_MS.get(t, SWEEP_MS['glider']) / SWEEP_MS['glider'] * ratio), 4))
    return out
SPACING = {'near_m': 20000.0, 'edges_m': [0, 250, 500, 1000, 2000, 4000, 8000, 16000]}

def bins() -> list[int]:
    return list(range(0, 24, TRIG['hour_bin']))

def hour_bin_of(hr: int, j: int) -> int:
    lt = (hr * 3600 + 1800 + (j + 0.5) * lab.ex.G * 240.0) % 86400 / 3600.0
    return lab.hour_bin(lt, TRIG['hour_bin'])

def source_climbs(d: lab.Day) -> dict:
    out = {}
    for t, s in enumerate(d.source_of()):
        if s is not None:
            out.setdefault(s, []).extend(d.thermals[t]['climbs'])
    return {s: sorted(cl, key=lambda i: (d.climbs[i]['t0'], i)) for s, cl in out.items()}

def rows_of(d: lab.Day, cnt=None, w=None) -> dict:
    import numpy as np
    cnt = cnt if cnt is not None else {}
    out = {}
    c = d.cols
    if len(c.get('h', [])):
        keep = d.row_ok & (c['part'] == 0)
        h, i, j = (c[k][keep].astype(np.int64) for k in ('h', 'i', 'j'))
        wy = np.asarray(w if w is not None else sweep_weights(), dtype=float)
        y = c['y'][keep].astype(np.int64) if 'y' in c else np.zeros(len(h), dtype=np.int64)
        cov = c['cov'][keep].astype(float) * wy[np.clip(y, 0, len(wy) - 1)]
        if len(h):
            key = np.stack([h, i, j], axis=1)
            u, inv = np.unique(key, axis=0, return_inverse=True)
            slots = np.minimum(np.bincount(inv.reshape(-1), weights=cov, minlength=len(u)), TRIG['slot_cap'])
            for (hh, ii, jj), s in zip(u.tolist(), slots.tolist()):
                k = (ii // lab.CELL, jj // lab.CELL, hour_bin_of(hh, jj))
                out.setdefault(k, [0.0, 0])[0] += s * SLOT_H
    for t, th in enumerate(d.thermals):
        if d.thermal_ok[t]:
            continue
        hours = set()
        for i in sorted(th['climbs'], key=lambda i: (d.climbs[i]['t0'], i)):
            hr = d.climbs[i]['t0'] // 3600
            if hr not in hours:
                hours.add(hr)
                if d.climb_cell_ok[i]:
                    cnt['events_lost_to_k'] = cnt.get('events_lost_to_k', 0) + 1
    for src, cl in sorted(source_climbs(d).items()):
        hours = set()
        for i in cl:
            c0 = d.climbs[i]
            hr = c0['t0'] // 3600
            if hr in hours:
                continue
            hours.add(hr)
            if hr in d.partial or hr not in d.counted:
                cnt['events_hole'] = cnt.get('events_hole', 0) + 1
                continue
            I, J = lab.lcell(c0['la0'], c0['lo0'])
            k = (I, J, hour_bin_of(hr, lab.ecell(c0['la0'], c0['lo0'])[1]))
            if k not in out or out[k][0] <= 0:
                cnt['no_exposure'] = cnt.get('no_exposure', 0) + 1
                continue
            out[k][1] += 1
            cnt['events'] = cnt.get('events', 0) + 1
    return out

def key_hours(d: lab.Day) -> dict:
    import numpy as np
    best = {}
    c = d.cols
    if len(c.get('h', [])):
        keep = d.row_ok & (c['part'] == 0)
        h, i, j, cov = (c[k][keep].astype(np.int64) for k in ('h', 'i', 'j', 'cov'))
        acc = {}
        for hh, ii, jj, s in zip(h.tolist(), i.tolist(), j.tolist(), cov.tolist()):
            k = (ii // lab.CELL, jj // lab.CELL, hour_bin_of(hh, jj))
            a = acc.setdefault(k, {})
            a[hh] = a.get(hh, 0) + s
        for k, a in acc.items():
            best[k] = max(sorted(a), key=lambda x: a[x])
    return best

def table(days: list, cnt=None, w=None) -> list:
    out = []
    w = w if w is not None else sweep_weights(sigma(days))
    for di, d in enumerate(days):
        cond = getattr(d, 'cond', None) or {}
        for (I, J, hb), (E, N) in sorted(rows_of(d, cnt, w).items()):
            if E > 0:
                out.append((di, I, J, hb, E, N, cond.get((I, J, hb))))
    return out

def fit(X, E, N, prior=None, ridge=None):
    import numpy as np
    X, E, N = (np.asarray(X, float), np.asarray(E, float), np.asarray(N, float))
    if N.sum() <= 0 or E.sum() <= 0:
        return None
    lam = ridge if ridge is not None else TRIG['ridge']
    p = X.shape[1]
    m = np.zeros(p) if prior is None else np.asarray(prior, float)
    b = m.copy()
    for _ in range(100):
        mu = E * np.exp(np.clip(X @ b, -50, 50))
        g = X.T @ (N - mu) - lam * (b - m)
        H = (X * mu[:, None]).T @ X + lam * np.eye(p)
        step = np.linalg.solve(H, g)
        while np.abs(step).max() > 5:
            step = step / 2
        b = b + step
        if np.abs(step).max() < 1e-12:
            break
    return b
COND_TERMS = ('cu', 'weak', 'unknown', 'log_wstar', 'wind')

def cond_terms(c) -> list:
    if not c or c.get('regime') is None:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    w, ws = (c.get('wstar'), c.get('wind'))
    return [1.0 if c['regime'] == 'cu' else 0.0, 1.0 if c['regime'] == 'weak' else 0.0, 0.0, math.log(max(float(w), 0.3) / 1.5) if w is not None else 0.0, (float(ws) - 5.0) / 5.0 if ws is not None else 0.0]

def terms_of(rows) -> list:
    named = [f'bin {b}' for b in bins()]
    return named + list(COND_TERMS) if any((len(r) > 6 and r[6] for r in rows)) else named

def design(rows, terms=None) -> 'np.ndarray':
    import numpy as np
    B = bins()
    terms = terms_of(rows) if terms is None else terms
    X = np.zeros((len(rows), len(terms)))
    cond = len(terms) > len(B)
    for k, row in enumerate(rows):
        X[k, B.index(row[3])] = 1.0
        if cond:
            X[k, len(B):] = cond_terms(row[6] if len(row) > 6 else None)
    return X

def surface(rows) -> dict:
    import numpy as np
    if not rows:
        return None
    E = np.array([x[4] for x in rows])
    N = np.array([x[5] for x in rows])
    if N.sum() <= 0:
        return None
    mean = math.log(N.sum() / E.sum())
    terms = terms_of(rows)
    X = design(rows, terms)
    nb = len(bins())
    b = fit(X, E, N, prior=np.concatenate([np.full(nb, mean), np.zeros(X.shape[1] - nb)]))
    mu = E * np.exp(X @ b)
    th, kap = ({}, TRIG['kappa'])
    acc = {}
    for row, m in zip(rows, mu.tolist()):
        a = acc.setdefault((row[1], row[2]), [0, 0.0, 0.0])
        a[0] += row[5]
        a[1] += m
        a[2] += row[4]
    for c, (Nc, mc, Ec) in acc.items():
        th[c] = ((Nc + kap) / (mc + kap), Nc, mc, Ec)
    eh = {}
    for row in rows:
        a = eh.setdefault(row[3], [0.0, 0])
        a[0] += row[4]
        a[1] += row[5]
    return {'beta': b.tolist(), 'theta': th, 'mean': mean, 'bins': eh, 'terms': terms}

def sigma(days: list) -> dict:
    h2, who, by = ([], set(), {})
    for d in days:
        for g in d.join.get('glides', []):
            if g.get('same') or g.get('L') is None or g.get('d') is None:
                continue
            if not (d.climb_ok[g['climb']] and d.climb_ok[g['prev']]):
                continue
            L, D = (float(g['L']), float(g['d']))
            if D <= 0 or L < D:
                continue
            v = max(0.0, (L / 2) ** 2 - (D / 2) ** 2)
            c = d.climbs[g['climb']]
            h2.append(v)
            who.add(d.acday(c))
            b = by.setdefault(c.get('y'), ([], set()))
            b[0].append(v)
            b[1].add(d.acday(c))

    def one(xs, ws):
        ok = lab.acdays(ws) >= lab.K and xs
        return {'m': lab.r(math.sqrt(sum(xs) / len(xs)), 0) if ok else None, 'n': len(xs) if ok else 0, 'acdays': lab.acdays(ws) if ok else 0}
    return {**one(h2, who), 'by_type': {t: one(*by[t]) for t in sorted((k for k in by if isinstance(k, str)))}}

def dist_summary(xs: list, who: set) -> dict | None:
    import numpy as np
    if lab.acdays(who) < lab.K or not xs:
        return None
    q = np.quantile(np.array(xs, float), [0.1, 0.25, 0.5, 0.75, 0.9])
    e = SPACING['edges_m']
    hist = [[e[k], sum((1 for x in xs if e[k] <= x < (e[k + 1] if k + 1 < len(e) else float('inf'))))] for k in range(len(e))]
    return {'n': len(xs), 'acdays': lab.acdays(who), 'q': [lab.r(v, 0) for v in q.tolist()], 'hist': hist}

def spacing(days: list) -> dict:
    gl, nr = ({}, {})
    for d in days:
        for g in d.join.get('glides', []):
            if g.get('same') or g.get('d') is None or (not (d.climb_ok[g['climb']] and d.climb_ok[g['prev']])):
                continue
            c = d.climbs[g['climb']]
            for b in ('all', lab.region(c['la0'], c['lo0'])):
                a = gl.setdefault(b, ([], set()))
                a[0].append(float(g['d']))
                a[1].add(d.acday(c))
        firsts = {}
        for t, cl in sorted(source_climbs(d).items()):
            seen = set()
            for i in cl:
                c = d.climbs[i]
                hr = c['t0'] // 3600
                if hr not in seen:
                    seen.add(hr)
                    firsts.setdefault(hr, []).append((c['la0'], c['lo0'], d.acday(c), t))
        for hr, pts in sorted(firsts.items()):
            for la, lo, who, t in pts:
                best = min((lab.jn._dist((la, lo), (la2, lo2)) for la2, lo2, _, t2 in pts if t2 != t), default=None)
                if best is None or best > SPACING['near_m']:
                    continue
                for b in ('all', lab.region(la, lo)):
                    a = nr.setdefault(b, ([], set()))
                    a[0].append(best)
                    a[1].add(who)

    def name(b):
        return b if b == 'all' else f'{b[0] * lab.REGION_DEG:+.0f}{b[1] * lab.REGION_DEG:+.0f}'
    out = {'glide': {}, 'nearest': {}}
    for part, acc in (('glide', gl), ('nearest', nr)):
        for b, (xs, who) in sorted(acc.items(), key=lambda kv: name(kv[0])):
            s = dist_summary(xs, who)
            if s:
                out[part][name(b)] = s
    return out

def learn_surface(days: list) -> tuple[dict, dict]:
    cnt = {}
    rows = table(days, cnt)
    return (surface(rows), {**cnt, 'rows': len(rows)})

def learn(days: list) -> dict:
    s, cnt = learn_surface(days)
    out = {'params': dict(TRIG), 'cell_deg': round(lab.ex.G * lab.CELL, 4), 'counts': dict(sorted(cnt.items())), 'sigma': sigma(days), 'spacing': spacing(days), 'spacing_params': SPACING}
    if s is None:
        out.update(hours=[], cells=[], fitted=False)
        return out
    out['fitted'] = True
    eh = s['bins']
    out['hours'] = [[hb, lab.r(math.exp(b), 5), lab.r(eh[hb][0], 4), eh[hb][1]] for hb, b in zip(bins(), s['beta']) if eh.get(hb, (0,))[0] > 0]
    nb = len(bins())
    out['terms'] = {t: lab.r(math.exp(b), 5) for t, b in zip(s['terms'][nb:], s['beta'][nb:])}
    out['rows_with_conditions'] = int(sum((1 for x in table(days) if x[6]))) if out['terms'] else 0
    out['hour_cols'] = ['bin', 'rate_per_h', 'coverage_h', 'events']
    cells = []
    for (I, J), (t, Nc, mc, Ec) in sorted(s['theta'].items()):
        la, lo = lab.centre(I, J)
        q = max(-128, min(127, round(20 * math.log(t))))
        cells.append([la, lo, lab.r(Ec, 4), int(Nc), lab.r(t, 4), q, lab.r(t * mc / Ec, 5)])
    out['cells'] = cells
    out['cell_cols'] = ['la', 'lo', 'coverage_h', 'events', 'theta', 'q', 'rate_per_h']
    return out

def deviance(N, mu) -> list:
    out = []
    for n, m in zip(N, mu):
        m = max(m, 1e-12)
        out.append(2 * ((n * math.log(n / m) if n > 0 else 0.0) - (n - m)))
    return out

def skill(days: list, held: int=7, baseline=None) -> dict:
    import numpy as np
    if len(days) <= held or held < 1:
        return {'held': held, 'train_days': len(days), 'test_days': 0, 'note': 'too few days to hold a span out'}
    train, test = (days[:-held], days[-held:])
    tr, te = (table(train), table(test))
    s = surface(tr)
    out = {'held': held, 'train_days': len(train), 'test_days': len(test), 'test_rows': len(te), 'test_events': int(sum((x[5] for x in te)))}
    if s is None or not te:
        out['note'] = 'no events to fit or none held out'
        return out
    E = np.array([x[4] for x in te])
    N = [x[5] for x in te]
    preds = {'chance': E * math.exp(s['mean'])}
    bt = design(te, s['terms']) @ np.array(s['beta'])
    preds['beta'] = E * np.exp(bt)
    th = np.array([s['theta'].get((x[1], x[2]), (1.0,))[0] for x in te])
    preds['theta'] = preds['beta'] * th
    if baseline is not None:

        def score(rows, ds):
            v = []
            for x in rows:
                la, lo = lab.centre(x[1], x[2])
                sc = baseline(la, lo, ds[x[0]].sd, x[3] + TRIG['hour_bin'] / 2)
                v.append(math.log(max(float(sc), 1e-06)))
            return np.array(v)
        Xtr = np.stack([np.ones(len(tr)), score(tr, train)], axis=1)
        b = fit(Xtr, [x[4] for x in tr], [x[5] for x in tr], prior=[s['mean'], 0.0], ridge=1e-06)
        Xte = np.stack([np.ones(len(te)), score(te, test)], axis=1)
        preds['baseline'] = E * np.exp(Xte @ b)
        out['baseline_coef'] = [lab.r(b[0], 4), lab.r(b[1], 4)]
    dev = {k: np.array(deviance(N, v.tolist())) for k, v in preds.items()}
    out['deviance'] = {k: lab.r(v.sum(), 3) for k, v in sorted(dev.items())}
    day_of = np.array([x[0] for x in te])
    ud = sorted(set(day_of.tolist()))
    per = {k: np.array([v[day_of == u].sum() for u in ud]) for k, v in dev.items()}
    rng = np.random.default_rng(TRIG['seed'])
    draws = rng.integers(0, len(ud), size=(TRIG['boot'], len(ud)))
    gains = {}
    for ref, models in (('chance', ('beta', 'theta', 'baseline')), ('beta', ('theta',)), ('baseline', ('theta',))):
        if ref not in per:
            continue
        for m in models:
            if m not in per:
                continue
            g = 1 - per[m].sum() / per[ref].sum() if per[ref].sum() > 0 else None
            bs = [1 - per[m][ix].sum() / per[ref][ix].sum() for ix in draws if per[ref][ix].sum() > 0]
            lo, hi = (np.quantile(bs, 0.05), np.quantile(bs, 0.95)) if bs else (None, None)
            gains[f'{m}_vs_{ref}'] = [lab.r(g, 4), lab.r(lo, 4), lab.r(hi, 4)]
    out['gain'] = gains
    who = {}
    for d in test:
        for i, c in enumerate(d.climbs):
            if d.climb_ok[i]:
                who.setdefault(lab.region(c['la0'], c['lo0']), set()).add(d.acday(c))
    reg = {}
    for k, x in enumerate(te):
        rg = lab.region(*lab.centre(x[1], x[2]))
        a = reg.setdefault(rg, {'rows': 0, 'events': 0, **{m: 0.0 for m in dev}})
        a['rows'] += 1
        a['events'] += x[5]
        for m in dev:
            a[m] += float(dev[m][k])
    out['regions'] = {}
    for rg, a in sorted(reg.items()):
        if lab.acdays(who.get(rg, ())) < lab.K:
            continue
        g = {f'{m}_vs_{ref}': lab.r(1 - a[m] / a[ref], 4) if a[ref] > 0 else None for ref, m in (('chance', 'beta'), ('chance', 'theta'), ('beta', 'theta'), ('chance', 'baseline')) if m in a}
        out['regions'][f'{rg[0] * lab.REGION_DEG:+.0f}{rg[1] * lab.REGION_DEG:+.0f}'] = {'rows': a['rows'], 'events': a['events'], 'gain': g}
    if baseline is None:
        out['baseline'] = "not available: the lab computes the site's trigger score (the trigger evaluation, with its ground cover), but the collector's days have no forecast grids in the lab yet"
    return out
