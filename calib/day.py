from __future__ import annotations
import glob
import json
import math
import os
import shutil
import subprocess
import numpy as np
import common
FIELDS = ['wstar', 'depth', 'cubase', 'cudepth', 'twidth', 'blapse', 'zi', 'densalt', 'rain', 'overdev', 'sun', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ceil', 'u_blavg', 'v_blavg', 'ridge', 'u_g300', 'v_g300', 'nlow', 'elevation', 'sink']
PAD_LON, PAD_LAT = (1.5, 1.2)
BUILD_TIMEOUT, EVAL_TIMEOUT = (25 * 60, 15 * 60)

def boxes(flights: list[dict]) -> list[dict]:
    q = lambda v, up: (math.ceil if up else math.floor)(v / 0.03) * 0.03
    by = {}
    for f in flights:
        by.setdefault(f['region'], []).append(f)
    out = []
    for reg, fs in sorted(by.items()):
        bb = [f['bbox'] or [f['tp'][0], f['tp'][1], f['tp'][0], f['tp'][1]] for f in fs]
        w = min((min(b[0], f['tp'][0]) for b, f in zip(bb, fs))) - PAD_LON
        s = min((min(b[1], f['tp'][1]) for b, f in zip(bb, fs))) - PAD_LAT
        e = max((max(b[2], f['tp'][0]) for b, f in zip(bb, fs))) + PAD_LON
        n = max((max(b[3], f['tp'][1]) for b, f in zip(bb, fs))) + PAD_LAT
        tz = max({f['tz'] for f in fs}, key=lambda t: sum((f['tz'] == t for f in fs)))
        out.append({'region': reg, 'bbox': [round(q(w, 0), 2), round(q(s, 0), 2), round(q(e, 1), 2), round(q(n, 1), 2)], 'tz': tz, 'flights': fs})
    return out

def dump(base: str) -> dict:
    m = common.read_json(f'{base}/latest.json')

    def dec(name, q):
        f = m['fields'][name]
        lo, hi = (f['min'], f['max'])
        q = q.astype(np.float32)
        if f.get('transform') == 'sqrt':
            a, b = (math.sqrt(max(lo, 0)), math.sqrt(hi))
            v = (a + q / 254 * (b - a)) ** 2
        else:
            v = lo + q / 254 * (hi - lo)
        v[q == 255] = np.nan
        return v.astype(np.float32)
    stamps = []
    ny, nx = (m['grid']['ny'], m['grid']['nx'])
    with open(f'{base}/fields.bin', 'wb') as out:
        for fn in sorted(glob.glob(f'{base}/_hours/*.npz')):
            z = np.load(fn)
            stamps.append(os.path.basename(fn)[:-4])
            for k in FIELDS:
                ok = k in z.files and k in m['fields']
                out.write((dec(k, z[k]) if ok else np.full((ny, nx), np.nan, np.float32)).tobytes())
    info = {'grid': m['grid'], 'heights': m.get('heights'), 'highAgl': m.get('highAgl'), 'stamps': stamps, 'fields': FIELDS}
    with open(f'{base}/fields.json', 'w') as f:
        json.dump(info, f)
    return info

def _run(cmd, cwd, timeout, logf, env=None) -> bool:
    with open(logf, 'a') as lf:
        try:
            r = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            lf.write(f'timed out after {timeout} s\n')
            return False

def eval_input(f: dict, ref: int) -> dict:
    return {'ref': ref, 'date': f['date'], 'tz': f['tz'], 'key': f['key'], 'tp': f['tp'], 'legs': [{'n': L['leg'], 'a': L['a'], 'b': L['b'], 't0': L['tstart'], 't1': L['tend'], 'line': L.get('line')} for L in f['legs']], 'points': [{'t0': c['t0'], 'dur': c['dur'], 'lon': c['lon'], 'lat': c['lat']} for c in f['climbs']]}

def categorize(L: dict) -> str:
    m = L.get('model') or {}
    therm, agl = (L.get('therm') or 0, L.get('agl') or 0)
    if therm < 0.1:
        return 'straight'
    if agl < 600 and therm < 0.2:
        return 'ridge'
    top = max(m.get('zi') or 0, m.get('cb') or 0)
    if top > 0 and agl > 1.15 * top:
        return 'wave'
    return 'thermal'

def run_day(day: str, flights: list[dict], work: str, models=('hrrr',), workers: int=2) -> dict:
    res = {'date': day, 'models': list(models), 'boxes': [], 'flights': flights}
    os.makedirs(work, exist_ok=True)
    for model in models:
        _run_model(res, day, flights, work, model, workers)
    for f in flights:
        for L in f['legs']:
            if f.get('src') != 'seed' or 'cat' not in L:
                L['cat'] = categorize({**L, 'model': ((L.get('models') or {}).get(models[0]) or {}).get('own')})
    return res

def _run_model(res: dict, day: str, flights: list[dict], work: str, model: str, workers: int) -> None:
    logf = common.PRIVATE_LOG or os.path.join(work, 'day.log')
    for bi, B in enumerate(boxes(flights)):
        out = os.path.join(work, f'{model}-box{bi}')
        shutil.rmtree(out, ignore_errors=True)
        info = {'model': model, 'region': B['region'], 'bbox': B['bbox'], 'flights': len(B['flights']), 'ok': False}
        bbox = ','.join((f'{v:.2f}' for v in B['bbox']))
        common.note(f"{day} {model} box {B['region']} {bbox} flights {len(B['flights'])}")
        built = _run(['python3', os.path.join(common.HERE, 'build_day.py'), '--day', day, '--tz', B['tz'], f'--bbox={bbox}', '--out', out, '--model', model, '--workers', str(workers)], common.PIPELINE, BUILD_TIMEOUT, logf)
        base = os.path.join(out, model)
        if not built or not os.path.exists(f'{base}/latest.json'):
            info['error'] = 'build'
            res['boxes'].append(info)
            shutil.rmtree(out, ignore_errors=True)
            continue
        try:
            fi = dump(base)
            info['hours'] = len(fi['stamps'])
        except Exception as e:
            common.note(f'dump failed: {e!r}')
            info['error'] = 'dump'
            res['boxes'].append(info)
            shutil.rmtree(out, ignore_errors=True)
            continue
        for p in os.listdir(base):
            if not p.startswith('fields.'):
                q = os.path.join(base, p)
                shutil.rmtree(q) if os.path.isdir(q) else os.remove(q)
        refs = {id(f): k for k, f in enumerate(B['flights'])}
        fin, fout = (os.path.join(out, 'in.json'), os.path.join(out, 'out.json'))
        with open(fin, 'w') as fh:
            json.dump([eval_input(f, refs[id(f)]) for f in B['flights']], fh)
        env = {**os.environ, 'CALIB_PUB': common.PUB}
        ok = _run(['node', '--max-old-space-size=4000', os.path.join(common.HERE, 'evaluate.js'), base, fin, fout], common.HERE, EVAL_TIMEOUT, logf, env)
        if not ok or not os.path.exists(fout):
            info['error'] = 'evaluate'
            res['boxes'].append(info)
            shutil.rmtree(out, ignore_errors=True)
            continue
        ev = common.read_json(fout)
        for r in ev['flights']:
            f = B['flights'][r['ref']]
            f['model_' + model] = {'box': bi, 'takeoffNode': r['takeoffNode']}
            for L, e in zip(f['legs'], r['legs']):
                if e:
                    L.setdefault('models', {})[model] = {'own': e.get('own'), 'route': e.get('route')}
            for c, e in zip(f['climbs'], r['points']):
                c.setdefault('models', {})[model] = e.get('own')
        info['ok'] = True
        res['boxes'].append(info)
        shutil.rmtree(out, ignore_errors=True)
