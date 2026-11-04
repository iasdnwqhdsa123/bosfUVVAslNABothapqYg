from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))
import common
import track
G = 9.81
LEGS = 3
LEG_S = (10.0, 40.0)
MIN_VG = 15.0
AGL = (15.0, 600.0)
TURN_MAX = 50.0
EDGE_S = 300.0
CLIMB_PAD = 30.0
ENGINE_ON = 500.0
STEP_M = 100.0
MIN_RELIEF = 60.0
DOMAIN_DEG = 0.11
POLARS = {'club': ((80, 0.7), (120, 1.1), (160, 2.1)), 'standard': ((85, 0.65), (120, 0.9), (160, 1.65)), '15m': ((85, 0.62), (120, 0.85), (160, 1.5)), '18m': ((85, 0.55), (120, 0.78), (160, 1.3)), 'open': ((85, 0.5), (120, 0.7), (160, 1.15)), 'twoseat': ((85, 0.62), (120, 0.9), (160, 1.6))}
BINS = (-1000000000.0, -1.0, -0.25, 0.25, 1.0, 2.0, 1000000000.0)
MODELS = ('wn', 'layer')

def _poly(cls):
    pts = POLARS.get(cls) or POLARS['15m']
    return np.polyfit([p[0] / 3.6 for p in pts], [p[1] for p in pts], 2)

def sink(cls, v):
    pts = POLARS.get(cls) or POLARS['15m']
    return float(np.polyval(_poly(cls), max(v, pts[0][0] / 3.6)))

def classes() -> dict:
    with open(os.path.join(HERE, 'aircraft.json'), encoding='utf-8') as f:
        return {k: (v or {}).get('cls') for k, v in json.load(f)['types'].items()}

def _xy(o):
    k = math.cos(math.radians(o[2]))
    return lambda p: ((p[1] - o[1]) * 111320 * k, (p[2] - o[2]) * 110540)

def pieces(f: dict) -> list[dict]:
    tr = [p for p in f.get('track') or [] if p[4] is not None]
    if len(tr) < LEGS + 2:
        return []
    climbs = [(c['_t0'] - CLIMB_PAD, c['_t1'] + CLIMB_PAD) for c in track._climbs_sorted(f)]
    t_off, t_land = (tr[0][0] + EDGE_S, tr[-1][0] - EDGE_S)

    def leg_ok(a, b):
        dt = b[0] - a[0]
        if not LEG_S[0] <= dt <= LEG_S[1] or a[0] < t_off or b[0] > t_land:
            return False
        if any((lo < b[0] and a[0] < hi for lo, hi in climbs)):
            return False
        if any((len(p) > 5 and p[5] is not None and (p[5] > ENGINE_ON) for p in (a, b))):
            return False
        if not all((AGL[0] <= p[3] - p[4] <= AGL[1] for p in (a, b))):
            return False
        return track.dist_km(a[1:3], b[1:3]) * 1000 / dt >= MIN_VG
    out, i = ([], 0)
    while i + LEGS < len(tr):
        fx = tr[i:i + LEGS + 1]
        ok = all((leg_ok(fx[j], fx[j + 1]) for j in range(LEGS)))
        if ok:
            hd = [track.bearing(fx[j][1:3], fx[j + 1][1:3]) for j in range(LEGS)]
            ok = all((abs((hd[j + 1] - hd[j] + 180) % 360 - 180) < TURN_MAX for j in range(LEGS - 1)))
        if ok:
            mid = fx[LEGS // 2]
            out.append({'t': mid[0], 'lon': mid[1], 'lat': mid[2], 'fixes': [list(p[:4]) for p in fx]})
            i += LEGS
        else:
            i += 1
    return out

def netto(fixes, u, v, cls) -> tuple[float, float]:
    o = fixes[0]
    to = _xy(o)
    P = [to(p) for p in fixes]
    va = []
    for j in range(len(fixes) - 1):
        dt = fixes[j + 1][0] - fixes[j][0]
        gx, gy = ((P[j + 1][0] - P[j][0]) / dt, (P[j + 1][1] - P[j][1]) / dt)
        va.append(math.hypot(gx - u, gy - v))
    T = fixes[-1][0] - fixes[0][0]
    de = fixes[-1][3] - fixes[0][3] + (va[-1] ** 2 - va[0] ** 2) / (2 * G)
    vm = sum(va) / len(va)
    return (de / T + sink(cls, vm), vm)

def line(fixes, step=STEP_M):
    lo, la, al = ([], [], [])
    for a, b in zip(fixes, fixes[1:]):
        n = max(1, math.ceil(track.dist_km(a[1:3], b[1:3]) * 1000 / step))
        for k in range(n):
            t = k / n
            lo.append(a[1] + (b[1] - a[1]) * t)
            la.append(a[2] + (b[2] - a[2]) * t)
            al.append(a[3] + (b[3] - a[3]) * t)
    return (np.array(la), np.array(lo), np.array(al))

def cell_of(lat: float, lon: float) -> tuple[int, int]:
    r = round(lat / DOMAIN_DEG)
    return (r, round(lon * math.cos(math.radians(r * DOMAIN_DEG)) / DOMAIN_DEG))

def cell_centre(c) -> tuple[float, float]:
    lat = c[0] * DOMAIN_DEG
    return (lat, c[1] * DOMAIN_DEG / math.cos(math.radians(lat)))

def gather(days: list, grids, cls_of: dict, log=print) -> dict:
    import rows as rowsmod
    by = defaultdict(list)
    n_fl = 0
    for d in days:
        m = rowsmod.model_of(d, None)
        gkey = d.get('key') or d['date']
        for f in d.get('flights') or []:
            bi = (f.get('model_' + m) or {}).get('box')
            if bi is None or not f.get('track'):
                continue
            ps = pieces(f)
            if ps:
                n_fl += 1
            for p in ps:
                at = lambda name: grids.at(gkey, m, bi, name, p['lon'], p['lat'], p['t'])
                u, v = (at('u_g300'), at('v_g300'))
                if u is None or v is None:
                    u, v = (at('u_blavg'), at('v_blavg'))
                if u is None or v is None:
                    continue
                p.update(u=u, v=v, n=at('nlow'), ws=at('wstar'), day=gkey, model=m, fid=f.get('id'), cls=cls_of.get(f.get('ac')) or '15m')
                by[cell_of(p['lat'], p['lon'])].append(p)
        grids.boxes.clear()
    log(f'{sum((len(v) for v in by.values()))} pieces from {n_fl} flights in {len(by)} places')
    return by

def judge(D, p) -> dict | None:
    la, lo, al = line(p['fixes'])
    x, y = D.xy(la, lo)
    gw = D.ground_at(x, y)
    import terrainwind as tw
    ge = tw._bilinear(D.e, D.x0, D.y0, D.res, x, y)
    rel = tw._bilinear(D.relief, D.x0, D.y0, D.res, x, y)
    if not (np.all(np.isfinite(gw)) and np.all(np.isfinite(ge)) and D.in_core(x, y).all()):
        return None
    if np.nanmax(rel) < MIN_RELIEF or np.min(al - gw) < 5:
        return None
    _, _, w = D.wind(x, y, al - gw, p['u'], p['v'])
    lay = D.layer(x, y, al - ge, p['u'], p['v'], p.get('n'))
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(lay))):
        return None
    obs, va = netto(p['fixes'], p['u'], p['v'], p['cls'])
    fo = D.forest_at(x, y)
    return {'day': p['day'], 'model': p['model'], 'fid': p['fid'], 't': p['t'], 'lat': round(p['lat'], 4), 'lon': round(p['lon'], 4), 'obs': round(obs, 3), 'wn': round(float(w.mean()), 3), 'layer': round(float(lay.mean()), 3), 'va': round(va, 1), 'agl': round(float((al - gw).mean())), 'relief': round(float(np.nanmax(rel))), 'wind': round(math.hypot(p['u'], p['v']), 1), 'ws': p.get('ws'), 'n': p.get('n'), 'forest': None if np.all(np.isnan(fo)) else round(float(np.nanmean(fo)), 2)}

def run(args, log=print) -> None:
    import layer2
    import rows as rowsmod
    import terrainwind as tw
    os.makedirs(args.out, exist_ok=True)
    done_fn = os.path.join(args.out, 'done.json')
    done = set((tuple(c) for c in (common.read_json(done_fn) if os.path.exists(done_fn) else [])))
    by = gather(rowsmod.load_days(args.data), layer2.Grids(args.grids), classes(), log)
    order = sorted((c for c in by if c not in done), key=lambda c: -len(by[c]))
    start, n_ok, n_cells = (time.time(), 0, 0)
    with open(os.path.join(args.out, 'results.jsonl'), 'a') as out:
        for c in order:
            if (time.time() - start) / 60 > args.budget:
                log(f'out of time: {len(order) - n_cells} places left for the next run')
                break
            lat, lon = cell_centre(c)
            try:
                D = tw.Domain(lat, lon, os.path.join(args.out, '_work'), args.cache)
            except Exception as e:
                print(f'a place not solved: {e}', file=sys.stderr)
                continue
            for p in by[c]:
                r = judge(D, p)
                if r:
                    out.write(json.dumps(r) + '\n')
                    n_ok += 1
            done.add(c)
            n_cells += 1
            out.flush()
            common.write_json(done_fn, sorted(done))
    log(f'{n_cells} places solved, {n_ok} pieces judged')

def _stats(R: list, key: str) -> dict:
    o = np.array([r['obs'] for r in R])
    p = np.array([r[key] for r in R])
    if len(R) < 3 or p.std() == 0:
        return {'n': len(R)}
    rr = float(np.corrcoef(o, p)[0, 1])
    b, a = np.polyfit(p, o, 1)
    return {'n': len(R), 'r': rr, 'slope': float(b), 'icpt': float(a), 'rmse': float(np.sqrt(((o - p) ** 2).mean())), 'bias': float((p - o).mean())}

def _bins(R: list, key: str) -> list:
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        s = [r['obs'] for r in R if lo <= r[key] < hi]
        out.append((len(s), float(np.mean(s)) if s else None))
    return out

def summary(R: list) -> str:
    md = ['# Ridge check', '', f"{len(R)} pieces of low straight flight over hills (a minute each), {len({r['fid'] for r in R})} flights, {len({r['day'] for r in R})} days.", '', "netto: the air's vertical speed the glider met (m/s). wn: WindNinja's terrain wind (the forecast's wind 300 m up); layer: the Ridge Lift layer's ridge lift less lee sink, at the glider's height.", '']
    subsets = [('all', R), ('W* under 1 m/s (little thermal)', [r for r in R if (r.get('ws') or 0) < 1]), ('wind 8 m/s and more', [r for r in R if r['wind'] >= 8]), ('wind under 8 m/s', [r for r in R if r['wind'] < 8]), ('under 150 m up', [r for r in R if r['agl'] < 150]), ('150 m up and more', [r for r in R if r['agl'] >= 150]), ('relief 200 m and more', [r for r in R if r['relief'] >= 200]), ('over forest (half and more)', [r for r in R if (r.get('forest') or 0) >= 0.5]), ('over open ground', [r for r in R if r.get('forest') is not None and r['forest'] < 0.5])]
    md += ['| pieces | n | ' + ' | '.join((f'{m} r | {m} fit (netto = a + b x) | {m} rmse | {m} bias' for m in MODELS)) + ' |', '|---|---|' + '---|---|---|---|' * len(MODELS)]
    for name, S in subsets:
        row = [name, str(len(S))]
        for m in MODELS:
            s = _stats(S, m)
            if 'r' not in s:
                row += ['-'] * 4
            else:
                row += [f"{s['r']:.2f}", f"{s['icpt']:+.2f} + {s['slope']:.2f}x", f"{s['rmse']:.2f}", f"{s['bias']:+.2f}"]
        md.append('| ' + ' | '.join(row) + ' |')
    md += ['', "The mean netto in each model's bins (n, m/s): a model that's right has the netto near the bin's middle.", '', '| model | ' + ' | '.join((f'[{lo:g}, {hi:g})'.replace('-1e+09', '-inf').replace('1e+09', 'inf') for lo, hi in zip(BINS, BINS[1:]))) + ' |', '|---|' + '---|' * (len(BINS) - 1)]
    for m in MODELS:
        md.append(f'| {m} | ' + ' | '.join(('-' if mu is None else f'{n}: {mu:+.2f}' for n, mu in _bins(R, m))) + ' |')
    md += ['', 'Where they disagree (one 1 m/s or more, the other under 0.25): the mean netto.', '']
    for a, b in (('wn', 'layer'), ('layer', 'wn')):
        s = [r['obs'] for r in R if r[a] >= 1 and r[b] < 0.25]
        md.append(f'- {a} only: {len(s)} pieces' + (f', netto {np.mean(s):+.2f}' if s else ''))
    return '\n'.join(md) + '\n'

def report(args, log=print) -> None:
    fn = os.path.join(args.out, 'results.jsonl')
    R = [json.loads(x) for x in open(fn)] if os.path.exists(fn) else []
    sw = os.path.join(args.out, 'sweep.jsonl')
    S = [json.loads(x) for x in open(sw)] if os.path.exists(sw) else []
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write(summary(R) + sweep_summary(S))
    log(f'{len(R)} pieces in the report')
SWEEP = {'lee': (0.5, 1.0, 0.25, 0.0), 'decay': (2.5, 2.0, 3.0), 'shape': (0.65, 0.45, 0.9), 'lean': (0.0, 1 / math.sqrt(3), 0.3)}

def variants() -> list[dict]:
    import itertools
    ks = list(SWEEP)
    return [dict(zip(ks, vs)) for vs in itertools.product(*(SWEEP[k] for k in ks))]

def sweep(args, log=print) -> None:
    import layer2
    import rows as rowsmod
    import terrainwind as tw
    V = variants()
    done = set((tuple(c) for c in common.read_json(os.path.join(args.out, 'done.json'))))
    sdone_fn = os.path.join(args.out, 'sweep_done.json')
    sdone = set((tuple(c) for c in (common.read_json(sdone_fn) if os.path.exists(sdone_fn) else [])))
    by = gather(rowsmod.load_days(args.data), layer2.Grids(args.grids), classes(), log)
    order = sorted((c for c in by if c in done and c not in sdone), key=lambda c: -len(by[c]))
    start, n_ok, n_cells = (time.time(), 0, 0)
    with open(os.path.join(args.out, 'sweep.jsonl'), 'a') as out:
        for c in order:
            if (time.time() - start) / 60 > args.budget:
                log(f'out of time: {len(order) - n_cells} places left for the next sweep')
                break
            lat, lon = cell_centre(c)
            try:
                D = tw.Domain(lat, lon, os.path.join(args.out, '_work'), args.cache, solve=False, cover=False)
            except Exception as e:
                print(f'a place not solved: {e}', file=sys.stderr)
                continue
            for p in by[c]:
                la, lo, al = line(p['fixes'])
                x, y = D.xy(la, lo)
                ge = tw._bilinear(D.e, D.x0, D.y0, D.res, x, y)
                rel = tw._bilinear(D.relief, D.x0, D.y0, D.res, x, y)
                if not (np.all(np.isfinite(ge)) and D.in_core(x, y).all()) or np.nanmax(rel) < MIN_RELIEF or np.min(al - ge) < 5:
                    continue
                vals = [float(np.mean(D.layer(x, y, al - ge, p['u'], p['v'], p.get('n'), **v))) for v in V]
                if not all((math.isfinite(z) for z in vals)):
                    continue
                obs, _ = netto(p['fixes'], p['u'], p['v'], p['cls'])
                out.write(json.dumps({'obs': round(obs, 3), 'wind': round(math.hypot(p['u'], p['v']), 1), 'ws': p.get('ws'), 'v': [round(z, 3) for z in vals]}) + '\n')
                n_ok += 1
            sdone.add(c)
            n_cells += 1
            out.flush()
            common.write_json(sdone_fn, sorted(sdone))
    log(f'{n_cells} places swept, {n_ok} pieces')

def sweep_summary(S: list) -> str:
    if not S:
        return ''
    V = variants()
    o = np.array([s['obs'] for s in S])
    X = np.array([s['v'] for s in S])
    rs = []
    for k, v in enumerate(V):
        p = X[:, k]
        if p.std() == 0:
            continue
        b, a = np.polyfit(p, o, 1)
        pos, neg = (p >= 0.25, p <= -0.25)
        rs.append((float(np.corrcoef(o, p)[0, 1]), v, float(a), float(b), float(np.sqrt(((o - p) ** 2).mean())), float((p - o).mean()), float(o[neg].mean()) if neg.any() else None, float(p[neg].mean()) if neg.any() else None))
    rs.sort(key=lambda t: -t[0])
    now = next((t for t in rs if t[1] == V[0]))
    fmt = lambda v: ', '.join((f'{k} {x:.2f}' for k, x in v.items()))
    md = ['', "## The layer's settings swept", '', f"{len(S)} pieces; {len(V)} combinations of {', '.join((f'{k} {SWEEP[k]}' for k in SWEEP))} (lee: the share of the lee sink kept). The layer now: " + fmt(V[0]) + f', r {now[0]:.3f}.', '', '| r | settings | fit (netto = a + b x) | rmse | bias | where it says sink (≤ −0.25): netto / it |', '|---|---|---|---|---|---|']
    for t in rs[:12] + ([now] if now not in rs[:12] else []):
        md.append(f'| {t[0]:.3f} | {fmt(t[1])} | {t[2]:+.2f} + {t[3]:.2f}x | {t[4]:.2f} | {t[5]:+.2f} | ' + ('-' if t[6] is None else f'{t[6]:+.2f} / {t[7]:+.2f}') + ' |')
    md += ['', "Each setting's best r over the others:", '']
    for k in SWEEP:
        md.append(f'- {k}: ' + ', '.join((f'{x:g} → {max((t[0] for t in rs if t[1][k] == x)):.3f}' for x in SWEEP[k])))
    return '\n'.join(md) + '\n'

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--data', required=True)
    r.add_argument('--grids', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--budget', type=float, default=240)
    r.add_argument('--cache', default=None)
    w = sub.add_parser('sweep')
    w.add_argument('--data', required=True)
    w.add_argument('--grids', required=True)
    w.add_argument('--out', required=True)
    w.add_argument('--budget', type=float, default=240)
    w.add_argument('--cache', default=None)
    p = sub.add_parser('report')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    {'run': run, 'sweep': sweep, 'report': report}[a.cmd](a)
if __name__ == '__main__':
    main()
