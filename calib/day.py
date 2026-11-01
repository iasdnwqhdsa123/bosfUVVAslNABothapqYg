from __future__ import annotations
import csv
import glob
import io
import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import common
import track
FIELDS = ['wstar', 'depth', 'cubase', 'cudepth', 'twidth', 'blapse', 'zi', 'densalt', 'rain', 'overdev', 'sun', 'cloud', 'cloudlow', 'cloudmid', 'cloudhigh', 'ceil', 'u_blavg', 'v_blavg', 'ridge', 'u_g300', 'v_g300', 'nlow', 'elevation', 'sink', 't2']
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

def local_hours(day: str, tz: str, a: int | None=None, b: int | None=None) -> list[int]:
    a, b = (OBS_FROM if a is None else a, OBS_TO if b is None else b)
    z = ZoneInfo(tz)
    d = datetime.fromisoformat(day)
    return [int(datetime(d.year, d.month, d.day, h, tzinfo=z).timestamp()) for h in range(a, b + 1)]

def probe_times(f: dict) -> list[dict]:
    kind = f.get('kind')
    lon, lat = f['tp']
    if kind == 'short':
        a, b = track.times_of(f)
        if a is None:
            return []
        ts_ = [int(a)] + list(range(int(math.ceil(a / 3600) * 3600), int(b), 3600))
    elif kind == 'asos':
        ts_ = local_hours(f['date'], f['tz'])
    elif kind == 'raob':
        ts_ = [int(f['t'])]
    else:
        return track.probes(f)
    return [{'t': t, 'lon': lon, 'lat': lat} for t in sorted(set(ts_))]

def eval_input(f: dict, ref: int) -> dict:
    return {'ref': ref, 'date': f['date'], 'tz': f['tz'], 'key': f.get('key') or 'std', 'tp': f['tp'], 'legs': [{'n': L['leg'], 'a': L['a'], 'b': L['b'], 't0': L['tstart'], 't1': L['tend'], 'line': L.get('line')} for L in f.get('legs') or []], 'points': [{'t0': c['t0'], 'dur': c['dur'], 'lon': c['lon'], 'lat': c['lat']} for c in f.get('climbs') or []], 'probes': probe_times(f)}

def clean(f: dict) -> dict:
    g = {k: v for k, v in f.items() if not k.startswith('model_') and k != 'probe_m'}
    if 'legs' in f:
        g['legs'] = [{k: v for k, v in L.items() if k != 'models'} for L in f.get('legs') or []]
    if 'climbs' in f:
        g['climbs'] = [{k: v for k, v in c.items() if k != 'models'} for c in f.get('climbs') or []]
    return g
IEM = 'https://mesonet.agron.iastate.edu'
OBS_GAP, OBS_MAX = (2.0, 40)
OBS_FROM, OBS_TO = (9, 20)
ASOS_PER_BOX = 3
CU_MAX_M = 4000
RAOB_PAD = 0.5
MIX_DTHETA = 1.0

class Obs:

    def __init__(self, budget: int=OBS_MAX, opener=None, sleep=time.sleep, clock=time.monotonic):
        self.budget, self.used, self.fails = (budget, 0, 0)
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=60))
        self._sleep, self._clock, self._last = (sleep, clock, -1000000000.0)

    def get(self, path: str, params=None) -> str | None:
        if self.used >= self.budget or self.fails >= 3:
            return None
        url = f'{IEM}/{path}' + ('?' + urllib.parse.urlencode(params, doseq=True) if params else '')
        wait = OBS_GAP - (self._clock() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = self._clock()
        self.used += 1
        try:
            with self._open(urllib.request.Request(url, headers={'User-Agent': 'research-script/1.0'})) as r:
                self.fails = 0
                return r.read().decode('utf-8', 'replace')
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            self.fails += 1
            common.note(f'obs: {path} failed: {e!r}')
            return None

def obs_stations(client: Obs, cache: str, network: str) -> list[dict]:
    fn = os.path.join(cache, f'{network}.json')
    if os.path.exists(fn):
        return common.read_json(fn)
    txt = client.get(f'geojson/network/{network}.geojson')
    if not txt:
        return []
    out = []
    try:
        for ft in json.loads(txt).get('features') or []:
            p, g = (ft.get('properties') or {}, (ft.get('geometry') or {}).get('coordinates'))
            if g:
                out.append({'id': ft.get('id') or p.get('sid'), 'lon': g[0], 'lat': g[1], 'elev': p.get('elevation'), 'begin': p.get('archive_begin'), 'end': p.get('archive_end')})
    except ValueError:
        return []
    common.write_json(fn, out)
    return out

def _open_on(s: dict, day: str) -> bool:
    return (not s.get('begin') or s['begin'][:10] <= day) and (not s.get('end') or s['end'][:10] >= day)

def pick_stations(sts: list[dict], points: list, day: str, n: int, bbox=None, max_km=None) -> list[dict]:
    ok = [s for s in sts if _open_on(s, day) and (not bbox or (bbox[0] <= s['lon'] <= bbox[2] and bbox[1] <= s['lat'] <= bbox[3]))]
    out = []
    for p in points:
        if len(out) >= n:
            break
        cand = sorted(((track.dist_km(p, [s['lon'], s['lat']]), s['id'], s) for s in ok if s not in out))
        if cand and (max_km is None or cand[0][0] <= max_km):
            out.append(cand[0][2])
    return out

def parse_asos(text: str) -> dict:
    out = {}
    for r in csv.DictReader(io.StringIO('\n'.join((x for x in text.splitlines() if not x.startswith('#'))))):
        try:
            t = datetime.strptime(r['valid'], '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc).timestamp()
        except (KeyError, ValueError, TypeError):
            continue
        tf = r.get('tmpf')
        tc = round((float(tf) - 32) / 1.8, 1) if tf not in (None, '', 'M') else None
        layers = []
        for k in (1, 2, 3):
            c, b = ((r.get(f'skyc{k}') or '').strip(), r.get(f'skyl{k}'))
            if c and c != 'M':
                layers.append([c, round(float(b) * 0.3048) if b not in (None, '', 'M') else None])
        out.setdefault(r.get('station'), []).append([int(t), tc, layers])
    return out

def asos_summary(obs: list, day: str, tz: str) -> dict:
    z = ZoneInfo(tz)
    by = {}
    for t, tc, layers in obs:
        loc = datetime.fromtimestamp(t + 1800, z)
        if loc.date().isoformat() != day or not OBS_FROM <= loc.hour <= OBS_TO:
            continue
        cu = [b for c, b in layers if c in ('FEW', 'SCT') and b is not None and (b <= CU_MAX_M)]
        by.setdefault(loc.hour, {'t': tc, 'cu': min(cu) if cu else None, 'cover': layers[0][0] if layers else None})
    temps = [x['t'] for x in by.values() if x['t'] is not None]
    return {'tmax': max(temps) if temps else None, 'hours': {str(h): by[h] for h in sorted(by)}}

def raob_summary(text: str) -> dict | None:
    try:
        prof = (json.loads(text).get('profiles') or [{}])[0].get('profile') or []
    except (ValueError, IndexError, AttributeError):
        return None
    lv = sorted(([p['pres'], p['hght'], p['tmpc'], p.get('dwpc')] for p in prof if p.get('pres') and p.get('hght') is not None and (p.get('tmpc') is not None) and (p['pres'] >= 500)), key=lambda x: x[1])
    if len(lv) < 3:
        return None
    th = [(z, (t + 273.15) * (1000 / pr) ** 0.2857) for pr, z, t, _ in lv]
    z0, th0 = th[0]
    mix = None
    for (za, a), (zb, b) in zip(th, th[1:]):
        if b >= th0 + MIX_DTHETA:
            mix = za + (zb - za) * ((th0 + MIX_DTHETA - a) / (b - a) if b != a else 0) - z0
            break
    t2k = next((lv[i - 1][2] + (lv[i][2] - lv[i - 1][2]) * (z0 + 2000 - lv[i - 1][1]) / (lv[i][1] - lv[i - 1][1]) for i in range(1, len(lv)) if lv[i][1] >= z0 + 2000 > lv[i - 1][1]), None)
    return {'mix': None if mix is None else round(mix), 'lapse2': None if t2k is None else round((lv[0][2] - t2k) / 2, 2), 't_sfc': lv[0][2], 'levels': [[round(p, 1), round(z), round(t, 1), None if d is None else round(d, 1)] for p, z, t, d in lv]}

def observations(client: Obs, day: str, boxes_: list[dict], cache: str) -> list[dict]:
    out = []
    raobs = obs_stations(client, cache, 'RAOB')
    for B in boxes_:
        pts = [f['tp'] for f in B['flights'] if f.get('tp')]
        if not pts:
            continue
        sts = []
        for st_ in sorted({f['state'] for f in B['flights'] if f.get('state')}):
            sts += obs_stations(client, cache, f'{st_}_ASOS')
        chosen = pick_stations(sts, pts, day, ASOS_PER_BOX, B['bbox'])
        if chosen:
            d0 = datetime.fromisoformat(day)
            d1 = d0 + timedelta(days=2)
            txt = client.get('cgi-bin/request/asos.py', [('station', s['id']) for s in chosen] + [('data', k) for k in ('tmpf', 'skyc1', 'skyc2', 'skyc3', 'skyl1', 'skyl2', 'skyl3')] + [('year1', d0.year), ('month1', d0.month), ('day1', d0.day), ('year2', d1.year), ('month2', d1.month), ('day2', d1.day), ('tz', 'Etc/UTC'), ('format', 'onlycomma'), ('latlon', 'no'), ('missing', 'M'), ('report_type', '3')])
            got = parse_asos(txt) if txt else {}
            for s in chosen:
                if got.get(s['id']):
                    out.append({'kind': 'asos', 'sid': s['id'], 'date': day, 'region': B['region'], 'tz': B['tz'], 'tp': [s['lon'], s['lat']], 'bbox': None, 'elev': s.get('elev'), 'obs': asos_summary(got[s['id']], day, B['tz'])})
        bb = B['bbox']
        rs = pick_stations([r for r in raobs if r['id'] and r['id'].startswith('_')], pts, day, 1, [bb[0] - RAOB_PAD, bb[1] - RAOB_PAD, bb[2] + RAOB_PAD, bb[3] + RAOB_PAD])
        if rs:
            t = int((datetime.fromisoformat(day) + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp())
            ts_ = datetime.fromtimestamp(t, timezone.utc).strftime('%Y%m%d%H%M')
            txt = client.get('json/raob.py', {'ts': ts_, 'station': 'K' + rs[0]['id'][1:]})
            sm = raob_summary(txt) if txt else None
            if sm:
                tp = [min(max(rs[0]['lon'], bb[0]), bb[2]), min(max(rs[0]['lat'], bb[1]), bb[3])]
                out.append({'kind': 'raob', 'sid': rs[0]['id'], 'date': day, 'region': B['region'], 'tz': B['tz'], 'tp': tp, 'pos': [rs[0]['lon'], rs[0]['lat']], 'bbox': None, 'elev': rs[0].get('elev'), 't': t, 'obs': sm})
    return out

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

def run_day(day: str, flights: list[dict], work: str, models=('hrrr',), workers: int=2, short=None, obs=None) -> dict:
    res = {'version': common.DAY_VERSION, 'date': day, 'models': list(models), 'boxes': [], 'flights': flights, 'short': short or [], 'obs': obs or []}
    os.makedirs(work, exist_ok=True)
    items = flights + res['short'] + res['obs']
    for model in models:
        _run_model(res, day, items, work, model, workers)
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
        inputs = [eval_input(f, refs[id(f)]) for f in B['flights']]
        with open(fin, 'w') as fh:
            json.dump(inputs, fh)
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
            for L, e in zip(f.get('legs') or [], r['legs']):
                if e:
                    L.setdefault('models', {})[model] = {'own': e.get('own'), 'route': e.get('route')}
            for c, e in zip(f.get('climbs') or [], r['points']):
                c.setdefault('models', {})[model] = e.get('own')
            f.setdefault('probe_m', {})[model] = {str(p['t']): None if e is None else {k: e.get(k) for k in track.PROBE_KEYS} for p, e in zip(inputs[r['ref']]['probes'], r.get('probes') or [])}
        info['ok'] = True
        res['boxes'].append(info)
        shutil.rmtree(out, ignore_errors=True)
