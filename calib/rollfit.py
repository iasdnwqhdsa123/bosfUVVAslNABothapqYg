from __future__ import annotations
import copy
import math
import numpy as np
ROLLS_NOW = dict(zL=[0.5, 2, 30, 150], wind=[2, 7], turnDeg=[30, 60], shear=[0.4, 0.004], lo=[0.5, 0.5, 3], cap=[0.5, 4], lid=[0.5, 3, 5], tear=[0.4, 1.0], on=0.3, aspect=[2.5, 5.5], aspectZL=[2, 25])
K, Z0, SFC = (0.4, 0.1, 0.7)
SNOW_M = 0.01
GROUP_MIN = 300
GROUP_SLACK = 0.01
GAIN_MIN = 0.005
SPACING_KM = (2.0, 12.0)

def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)

def psi_m(zeta):
    out = np.zeros_like(zeta)
    m = zeta < 0
    x = np.sqrt(np.sqrt(1 - 16 * zeta[m]))
    out[m] = 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2) - 2 * np.arctan(x) + np.pi / 2
    return out

def dax(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180
    return np.minimum(d, 180 - d)

def prep(T: dict) -> dict:
    w, zi = (T['c_wstar'], T['c_zi'])
    u, v = (np.nan_to_num(T['c_u']), np.nan_to_num(T['c_v']))
    V = np.hypot(u, v)
    us_, vs_, ut, vt = (T['c_uSfc'], T['c_vSfc'], T['c_uTop'], T['c_vTop'])
    has_s = np.isfinite(us_) & np.isfinite(vs_)
    has_t = np.isfinite(ut) & np.isfinite(vt)
    U10 = np.where(has_s, np.hypot(np.nan_to_num(us_), np.nan_to_num(vs_)), SFC * V)
    lz = math.log(10 / Z0)
    us = K * np.maximum(0.3, U10) / lz
    with np.errstate(all='ignore'):
        for _ in range(4):
            zL = K * (w / us) ** 3
            L = -zi / zL
            us = K * np.maximum(0.3, U10) / np.maximum(0.5, lz - psi_m(np.nan_to_num(10 / L, nan=0.0)))
        zL = K * (w / us) ** 3
    both = has_s & has_t
    a = np.hypot(np.nan_to_num(us_), np.nan_to_num(vs_))
    b = np.hypot(np.nan_to_num(ut), np.nan_to_num(vt))
    with np.errstate(all='ignore'):
        cosang = np.clip((np.nan_to_num(us_) * np.nan_to_num(ut) + np.nan_to_num(vs_) * np.nan_to_num(vt)) / (a * b), -1, 1)
    turn = np.degrees(np.arccos(np.nan_to_num(cosang, nan=1.0)))
    S = np.hypot(np.nan_to_num(ut) - np.nan_to_num(us_), np.nan_to_num(vt) - np.nan_to_num(vs_))
    with np.errstate(all='ignore'):
        szi = np.where(both, S / zi, np.nan)
    return dict(lz=np.log(np.maximum(1e-06, np.nan_to_num(zL, nan=1e-06))), zL=np.nan_to_num(zL, nan=0.0), V=V, turn=turn, turnOK=both & (a > 0) & (b > 0), minab=np.minimum(a, b), both=both, Szi=szi, lo=T['c_shearLo'], cap=T['c_cap'], lid=T['c_lid'], tear=np.nan_to_num(T['c_tear']), ok=(w > 0) & (zi > 0), zi=zi)

def factors(D: dict, P: dict) -> dict:
    a0, a1, b0, b1 = np.log(P['zL'])
    f = {'in': smooth((D['lz'] - a0) / (a1 - a0)), 'out': 1 - smooth((D['lz'] - b0) / (b1 - b0)), 'wind': smooth((D['V'] - P['wind'][0]) / (P['wind'][1] - P['wind'][0]))}
    t0, t1 = P['turnDeg']
    f['turn'] = np.where(D['turnOK'], 1 - smooth((D['turn'] - t0) / (t1 - t0)) * smooth((D['minab'] - 1) / 2), 1.0)
    f['shear'] = np.where(D['both'], P['shear'][0] + (1 - P['shear'][0]) * smooth(np.nan_to_num(D['Szi']) / P['shear'][1]), 1.0)
    l0, l1, l2 = P['lo']
    f['lo'] = np.where(np.isfinite(D['lo']), l0 + (1 - l0) * smooth((np.nan_to_num(D['lo']) - l1) / (l2 - l1)), 1.0)
    f['cap'] = np.where(np.isfinite(D['cap']), P['cap'][0] + (1 - P['cap'][0]) * smooth(np.nan_to_num(D['cap']) / P['cap'][1]), 1.0)
    d0, d1, d2 = P['lid']
    f['lid'] = np.where(np.isfinite(D['lid']), d0 + (1 - d0) * smooth((np.nan_to_num(D['lid']) - d1) / (d2 - d1)), 1.0)
    f['tear'] = 1 - smooth((D['tear'] - P['tear'][0]) / (P['tear'][1] - P['tear'][0]))
    return f

def strength(D: dict, P: dict) -> np.ndarray:
    s = np.ones_like(D['V'])
    for v in factors(D, P).values():
        s = s * v
    return np.where(D['ok'], s, 0.0)

def spacing(D: dict, P: dict) -> np.ndarray:
    z0, z1 = P['aspectZL']
    with np.errstate(all='ignore'):
        x = np.clip(np.log(np.maximum(1e-06, D['zL']) / z0) / math.log(z1 / z0), 0, 1)
    a = P['aspect'][0] + (P['aspect'][1] - P['aspect'][0]) * x
    return a * D['zi'] / 1000

def columns(T: dict) -> dict:
    d = dax(T['axis'], T['wax'])
    T['dax'] = d
    T['y_off'] = (T['aniso'] >= 0.35) & (d <= 20)
    T['y_soft'] = (T['aniso'] >= 0.25) & (d <= 30)
    mvok = np.nan_to_num(T['mv_r'], nan=-1) >= 0.4
    stat = mvok & (np.nan_to_num(T['mv_shift'], nan=9000000000.0) < T['mv_exp'] / 3) & (np.nan_to_num(T['mv_r0'], nan=-1) >= T['mv_r'] - 0.05)
    high = np.nan_to_num(T['bt10'], nan=999) < T['cutopT'] - 15
    setup = (np.nan_to_num(T['relief']) >= 250) & (np.nan_to_num(T['Nabove']) >= 0.01) & (np.nan_to_num(T['w13']) >= 10)
    moving = mvok & ~stat
    snow = np.nan_to_num(T['snow'], nan=0.0) >= SNOW_M if 'snow' in T else np.zeros(len(d), bool)
    env = np.isfinite(T['c_strength'])
    for lab, thr in (('off', 0.35), ('soft', 0.25)):
        y = T[f'y_{lab}']
        wave = (T['aniso'] >= thr) & (d >= 70) & (d <= 110)
        drop = y & (stat | high | setup & ~moving)
        T[f'use_{lab}'] = env & ~wave & ~drop & ~snow
    T['D'] = prep(T)
    T['gate'] = T['c_gate'] == 1
    return T

def auc(s, y) -> float:
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def yesno(call, y) -> dict:
    tp = int((call & y).sum())
    fp = int((call & ~y).sum())
    fn = int((~call & y).sum())
    tn = int((~call & ~y).sum())
    hit = tp / max(1, tp + fn)
    return dict(hit=hit, fa=fp / max(1, tp + fp), ba=(hit + tn / max(1, tn + fp)) / 2, called=float(call.mean()) if len(call) else float('nan'), n=int(len(y)), pos=int(y.sum()))

def scores(T: dict, s, P: dict, mask, lab: str='off') -> dict:
    out = {}
    for km in (24, 48):
        m = mask & T[f'use_{lab}'] & (T['km'] == km)
        y = T[f'y_{lab}'][m]
        out[km] = dict(auc=auc(s[m], y), **yesno(T['gate'][m] & (s[m] >= P['on']), y))
    return out

def objective(T: dict, P: dict, mask, S=None) -> float:
    s = strength(T['D'], P) if S is None else S
    v = []
    for lab in ('off', 'soft'):
        sc = scores(T, s, P, mask, lab)
        for km in (24, 48):
            v += [sc[km]['auc'], sc[km]['ba']]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')
GRID = {'zL.0': [0.5, 1, 1.5, 2], 'zL.1': [2, 3, 4, 5, 6], 'zL.2': [4, 6, 8, 10, 15, 20, 30], 'zL.3': [30, 40, 60, 80, 100, 150, 200], 'wind.0': [1, 2, 3, 4, 5, 6], 'wind.1': [4, 5, 6, 7, 8, 9, 10], 'turnDeg.0': [20, 30, 40], 'turnDeg.1': [50, 60, 80, 90], 'shear.0': [0.4, 0.5, 0.6, 0.8, 1.0], 'shear.1': [0.001, 0.002, 0.003, 0.004, 0.006, 0.008], 'lo.0': [0.4, 0.5, 0.6, 0.7, 0.8, 1.0], 'lo.1': [0, 0.5, 1, 1.5, 2], 'lo.2': [2, 2.5, 3, 3.5, 4, 5], 'cap.0': [0.4, 0.5, 0.6, 0.8, 1.0], 'cap.1': [1, 2, 3, 4, 5, 6], 'lid.0': [0.4, 0.5, 0.6, 0.7, 0.8, 1.0], 'lid.1': [1, 2, 3, 4], 'lid.2': [4, 5, 6, 7, 8, 10], 'tear.0': [0.1, 0.2, 0.3, 0.4], 'tear.1': [0.6, 0.8, 1.0], 'on': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}

def ok(P: dict) -> bool:
    z = P['zL']
    return z[1] / z[0] >= 2 and z[3] / z[2] >= 2 and (z[1] < z[2]) and (P['wind'][1] - P['wind'][0] >= 2) and (P['turnDeg'][1] > P['turnDeg'][0]) and (P['lo'][2] - P['lo'][1] >= 1.5) and (P['lid'][2] - P['lid'][1] >= 2) and (P['tear'][1] - P['tear'][0] >= 0.2) and (min(P['shear'][0], P['lo'][0], P['cap'][0], P['lid'][0]) >= 0.4)

def setp(P: dict, key: str, v) -> dict:
    Q = copy.deepcopy(P)
    if '.' in key:
        g, i = key.split('.')
        Q[g][int(i)] = v
    else:
        Q[key] = v
    return Q

def search(T: dict, mask, P0: dict, keys=None, sweeps: int=3) -> tuple[dict, float]:
    P = copy.deepcopy(P0)
    best = objective(T, P, mask)
    for _ in range(sweeps):
        moved = False
        for k in keys or list(GRID):
            for v in GRID[k]:
                Q = setp(P, k, v)
                if not ok(Q):
                    continue
                o = objective(T, Q, mask)
                if o > best + 1e-05:
                    best, P, moved = (o, Q, True)
        if not moved:
            break
    return (P, best)

def folds(days, k: int, seed: int) -> list[set]:
    d = np.array(sorted(set(days)))
    rng = np.random.default_rng(seed)
    rng.shuffle(d)
    return [set(d[i::k]) for i in range(k)]

def season(days) -> np.ndarray:
    m = np.array([int(str(x)[5:7]) for x in days])
    return np.where((m >= 4) & (m <= 9), 'warm', 'cold')

def groups(T: dict) -> dict:
    se = season(T['day'])
    g = {f'season:{s}': se == s for s in ('warm', 'cold')}
    for b in sorted(set(T['box'])):
        g[f'box:{b}'] = T['box'] == b
        for s in ('warm', 'cold'):
            g[f'{s}:{b}'] = (T['box'] == b) & (se == s)
    return {k: m for k, m in g.items() if (m & T['use_off'] & (T['km'] == 24)).sum() >= GROUP_MIN}

def heldout(T: dict, P0: dict, k: int=5, seeds=(0, 1, 2), keys=None) -> list[tuple]:
    days = T['day']
    res = []
    for seed in seeds:
        s = np.zeros(len(days))
        on = np.zeros(len(days))
        Ps = []
        for F in folds(days, k, seed):
            te = np.isin(days, list(F))
            P, _ = search(T, ~te, P0, keys)
            s[te] = strength(T['D'], P)[te]
            on[te] = P['on']
            Ps.append(P)
        res.append((s, on, Ps))
    return res

def objective_held(T: dict, s, on, mask) -> float:
    v = []
    for lab in ('off', 'soft'):
        for km in (24, 48):
            m = mask & T[f'use_{lab}'] & (T['km'] == km)
            y = T[f'y_{lab}'][m]
            call = T['gate'][m] & (s[m] >= on[m])
            v += [auc(s[m], y), yesno(call, y)['ba']]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')

def gate(T: dict, P_old: dict, seeds=(0, 1, 2), k: int=5) -> dict:
    res = heldout(T, P_old, k, seeds)
    s_old = strength(T['D'], P_old)
    on_old = np.full(len(s_old), P_old['on'])
    every = np.ones(len(s_old), bool)
    old_all = objective_held(T, s_old, on_old, every)
    new_all = float(np.nanmean([objective_held(T, s, on, every) for s, on, _ in res]))
    out = {'all': (old_all, new_all), 'groups': {}, 'worse': []}
    for name, m in groups(T).items():
        o = objective_held(T, s_old, on_old, m)
        n = float(np.nanmean([objective_held(T, s, on, m) for s, on, _ in res]))
        out['groups'][name] = (o, n)
        if n < o - GROUP_SLACK:
            out['worse'].append(name)
    final, _ = search(T, every, P_old)
    out['final'] = final
    if not new_all >= old_all + GAIN_MIN:
        out['pass'], out['why'] = (False, f'held out {new_all:.3f} against {old_all:.3f}: under the {GAIN_MIN} gain')
    elif out['worse']:
        out['pass'], out['why'] = (False, 'worse held out in ' + ', '.join(out['worse']))
    else:
        out['pass'], out['why'] = (True, f'held out {new_all:.3f} against {old_all:.3f}, no group worse')
    return out

def spacing_fit(T: dict, P: dict) -> dict:
    m = T['use_off'] & T['y_off'] & (T['km'] == 24) & np.isfinite(T['spacing']) & (T['spacing'] >= SPACING_KM[0]) & (T['spacing'] <= SPACING_KM[1])
    m &= np.isfinite(T['D']['zi']) & (T['D']['zi'] > 0) & T['D']['ok']
    n = int(m.sum())
    if n < 30:
        return {'n': n}
    obs = T['spacing'][m]
    D = {k: v[m] if isinstance(v, np.ndarray) and v.shape == m.shape else v for k, v in T['D'].items()}

    def cost(Q):
        r = np.log(obs / np.maximum(0.1, spacing(D, Q)))
        return abs(float(np.median(r))) + 0.25 * float(np.median(np.abs(r - np.median(r))))
    best, bq = (cost(P), copy.deepcopy(P))
    for a0 in np.arange(1.5, 6.01, 0.25):
        for a1 in np.arange(a0, 8.01, 0.25):
            Q = dict(P, aspect=[float(a0), float(a1)])
            c = cost(Q)
            if c < best - 1e-06:
                best, bq = (c, Q)
    return {'n': n, 'ratio_now': float(np.median(obs / spacing(D, P))), 'aspect': bq['aspect'], 'ratio_new': float(np.median(obs / spacing(D, bq)))}
