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
GRID_FIELDS = os.environ.get('CALIB_GRID_FIELDS', 'all')
GRID_HOURS = os.environ.get('CALIB_GRID_HOURS', 'all')

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
    info = {'grid': m['grid'], 'heights': m.get('heights'), 'highAgl': m.get('highAgl'), 'stamps': stamps, 'fields': FIELDS, 'model': m.get('model') or os.path.basename(os.path.normpath(base))}
    with open(f'{base}/fields.json', 'w') as f:
        json.dump(info, f)
    return info

def _run(cmd, cwd, timeout, logf, env=None) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(logf)), exist_ok=True)
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
    g = {k: v for k, v in f.items() if not k.startswith(('model_', 'probe_m'))}
    if 'legs' in f:
        g['legs'] = [{k: v for k, v in L.items() if k not in ('models', 'models_c')} for L in f.get('legs') or []]
    if 'climbs' in f:
        g['climbs'] = [{k: v for k, v in c.items() if k not in ('models', 'models_c')} for c in f.get('climbs') or []]
    return g
IEM = 'https://mesonet.agron.iastate.edu'
OBS_GAP, OBS_MAX = (2.0, 40)
OBS_FROM, OBS_TO = (9, 20)
ASOS_PER_BOX = 3
CU_MAX_M = 4000
RAOB_PAD = 0.5
MIX_DTHETA = 1.0
_NO_RAOB_NOTED = False

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

def asos_network(code: str, macro: str='US') -> str:
    return f'{code}_ASOS' if macro == 'US' else f'{code}__ASOS'

def observations(client: Obs, day: str, boxes_: list[dict], cache: str) -> list[dict]:
    out = []
    raobs = obs_stations(client, cache, 'RAOB') if any((common.macro_of(B['region']) == 'US' for B in boxes_)) else []
    for B in boxes_:
        pts = [f['tp'] for f in B['flights'] if f.get('tp')]
        if not pts:
            continue
        macro = common.macro_of(B['region'])
        sts = []
        for st_ in sorted({f['state'] for f in B['flights'] if f.get('state')}):
            sts += obs_stations(client, cache, asos_network(st_, macro))
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
        if macro != 'US':
            global _NO_RAOB_NOTED
            if not _NO_RAOB_NOTED:
                common.note("obs: no soundings for Europe's boxes (IEM's RAOB archive has no European station)")
                _NO_RAOB_NOTED = True
            continue
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

def lead_day(rec: dict, work: str, workers: int=2, grids: str | None=None, corrections: str | None=None) -> dict:
    import copy
    day = rec['date']
    macro = rec.get('macro') or common.macro_of_items(rec.get('flights') or [])
    key = rec.get('key') or common.day_key(day, macro)
    leads = common.lead_models(macro, day)
    res = copy.deepcopy(rec)
    res['boxes'] = [b for b in res.get('boxes') or [] if not common.is_lead(b.get('model'))]
    res['models'] = [m for m in res.get('models') or [res.get('model') or 'hrrr'] if not common.is_lead(m)] + leads
    items = (res.get('flights') or []) + (res.get('short') or []) + (res.get('obs') or [])
    for f in items:
        for k in [k for k in f if k.startswith(('model_', 'model_c_')) and common.is_lead(k.split('_', 2)[-1] if k.startswith('model_c_') else k[6:])]:
            f.pop(k)
        for pk in ('probe_m', 'probe_mc'):
            for m in [m for m in f.get(pk) or {} if common.is_lead(m)]:
                f[pk].pop(m)
        for x in (f.get('legs') or []) + (f.get('climbs') or []):
            for mk in ('models', 'models_c'):
                for m in [m for m in x.get(mk) or {} if common.is_lead(m)]:
                    x[mk].pop(m)
    os.makedirs(work, exist_ok=True)
    for model in leads:
        _run_model(res, day, items, work, model, workers, grids, corrections)
    lb = [b for b in res['boxes'] if common.is_lead(b.get('model'))]
    res['lead_v'] = common.lead_version(key) if lb and any((b.get('ok') for b in lb)) else None
    res['lead_boxes'] = {'ok': sum((b.get('ok', False) for b in lb)), 'all': len(lb)}
    return res

def run_day(day: str, flights: list[dict], work: str, models=('hrrr',), workers: int=2, short=None, obs=None, grids: str | None=None, corrections: str | None=None) -> dict:
    models = tuple(models)
    macro = common.macro_of_items(flights + list(short or []))
    res = {'version': common.DAY_VERSION, 'date': day, 'macro': macro, 'key': common.day_key(day, macro), 'models': list(models), 'model': models[0], 'boxes': [], 'flights': flights, 'short': short or [], 'obs': obs or [], 'physics_v': common.physics_version(macro), 'score_v': common.SCORE_VERSION, 'corr_v': corr_version(corrections), 'grids': False}
    os.makedirs(work, exist_ok=True)
    items = flights + res['short'] + res['obs']
    for model in models:
        _run_model(res, day, items, work, model, workers, grids, corrections)
    res['grids'] = bool(grids) and any((b.get('saved') for b in res['boxes'] if not common.is_lead(b.get('model'))))
    lb = [b for b in res['boxes'] if common.is_lead(b.get('model'))]
    res['lead_v'] = common.lead_version(res['key']) if lb and any((b.get('ok') for b in lb)) else None
    _categorize(res)
    return res

def _categorize(res: dict) -> None:
    for f in res['flights']:
        for L in f['legs']:
            if f.get('src') != 'seed' or 'cat' not in L:
                L['cat'] = categorize({**L, 'model': ((L.get('models') or {}).get(res['model']) or {}).get('own')})

def corr_version(corrections: str | None) -> str | None:
    if not corrections or not os.path.exists(corrections):
        return None
    try:
        return str(common.read_json(corrections).get('version'))
    except (ValueError, AttributeError, OSError):
        return None

def local_hour_of(stamp: str, tz: str) -> int:
    t = datetime.strptime(stamp[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc)
    return t.astimezone(ZoneInfo(tz)).hour

def save_grids(base: str, dest: str, meta: dict, fields: str=GRID_FIELDS, hours: str=GRID_HOURS) -> dict:
    os.makedirs(os.path.join(dest, '_hours'), exist_ok=True)
    lo, hi = (int(x) for x in hours.split('-')) if hours != 'all' else (0, 23)
    keep_names = None if fields == 'all' else sorted(set(FIELDS) | {'elevation'})
    names = set()
    for fn in sorted(glob.glob(f'{base}/_hours/*.npz')):
        stamp = os.path.basename(fn)[:-4]
        if hours != 'all' and (not lo <= local_hour_of(stamp, meta['tz']) <= hi):
            continue
        if keep_names is None:
            shutil.copyfile(fn, os.path.join(dest, '_hours', f'{stamp}.npz'))
            with np.load(fn) as z:
                names.update(z.files)
        else:
            with np.load(fn) as z:
                arrays = {k: z[k] for k in keep_names if k in z.files}
            names.update(arrays)
            np.savez_compressed(os.path.join(dest, '_hours', f'{stamp}.npz'), **arrays)
    shutil.copyfile(f'{base}/latest.json', os.path.join(dest, 'latest.json'))
    out = {**meta, 'physics_v': common.model_physics(meta.get('model')), 'built_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'fields': sorted(names), 'hours': hours}
    common.write_json(os.path.join(dest, 'meta.json'), out, indent=1)
    return out

def grids_usable(gdir: str, bbox: list, model: str) -> str | None:
    mp = os.path.join(gdir, 'meta.json')
    if not os.path.exists(mp) or not os.path.exists(os.path.join(gdir, 'latest.json')) or (not glob.glob(os.path.join(gdir, '_hours', '*.npz'))):
        return 'missing'
    m = common.read_json(mp)
    if m.get('physics_v') != common.model_physics(model):
        return 'physics'
    if m.get('model') != model or [round(float(v), 2) for v in m.get('bbox') or []] != [round(float(v), 2) for v in bbox]:
        return 'box'
    if not set(FIELDS) <= set(m.get('fields') or []):
        return 'fields'
    return None

def _eval_box(base: str, B: dict, out: str, model: str, bi: int, info: dict, logf: str, corrections: str | None) -> bool:
    try:
        fi = dump(base)
        info['hours'] = len(fi['stamps'])
    except Exception as e:
        common.note(f'dump failed: {e!r}')
        info['error'] = 'dump'
        return False
    for p in os.listdir(base):
        if not p.startswith('fields.'):
            q = os.path.join(base, p)
            shutil.rmtree(q) if os.path.isdir(q) else os.remove(q)
    refs = {id(f): k for k, f in enumerate(B['flights'])}
    fin, fout = (os.path.join(out, 'in.json'), os.path.join(out, 'out.json'))
    inputs = [eval_input(f, refs[id(f)]) for f in B['flights']]
    with open(fin, 'w') as fh:
        json.dump(inputs, fh)
    env = {**os.environ, 'CALIB_PUB': common.PUB, 'CALIB_MODEL': common.base_model(model)}
    if corrections and os.path.exists(corrections):
        env['CALIB_CORRECTIONS'] = corrections
    else:
        env.pop('CALIB_CORRECTIONS', None)
    ok = _run(['node', '--max-old-space-size=4000', os.path.join(common.HERE, 'evaluate.js'), base, fin, fout], common.HERE, EVAL_TIMEOUT, logf, env)
    if not ok or not os.path.exists(fout):
        info['error'] = 'evaluate'
        return False
    ev = common.read_json(fout)
    _merge_eval(ev['flights'], B, inputs, model, bi, '')
    if ev.get('flights_c'):
        _merge_eval(ev['flights_c'], B, inputs, model, bi, '_c')
        info['corrected'] = True
    return True

def _merge_eval(results: list, B: dict, inputs: list, model: str, bi: int, suffix: str) -> None:
    for r in results:
        f = B['flights'][r['ref']]
        f[f'model{suffix}_{model}'] = {'box': bi, 'takeoffNode': r['takeoffNode']}
        for L, e in zip(f.get('legs') or [], r['legs']):
            if e:
                L.setdefault('models' + suffix, {})[model] = {'own': e.get('own'), 'route': e.get('route')}
        for c, e in zip(f.get('climbs') or [], r['points']):
            c.setdefault('models' + suffix, {})[model] = e.get('own')
        f.setdefault('probe_m' + ('c' if suffix else ''), {})[model] = {str(p['t']): None if e is None else {k: e.get(k) for k in track.PROBE_KEYS} for p, e in zip(inputs[r['ref']]['probes'], r.get('probes') or [])}
BUILDER = {'hrrr': 'build_day.py', 'gfs': 'build_day.py', 'ecmwf': 'build_eu.py', 'icon': 'build_eu.py', 'icond2': 'build_eu.py'}
LEAD_GRIDS = ('gfs', 'ecmwf')

def build_command(day: str, B: dict, out: str, model: str, workers: int, work: str) -> list[str] | None:
    bbox = ','.join((f'{v:.2f}' for v in B['bbox']))
    base, lead = (common.base_model(model), common.lead_of(model))
    script = BUILDER.get(base, 'build_day.py')
    if lead is not None and script == 'build_day.py':
        script = 'build_lead.py'
    cmd = ['python3', os.path.join(common.HERE, script), '--day', day, '--tz', B['tz'], f'--bbox={bbox}', '--out', out, '--model', base, '--workers', str(workers)]
    if lead is not None:
        cmd += ['--lead', str(lead)] + (['--area', common.macro_of(B['region'])] if script == 'build_lead.py' else [])
    if script == 'build_eu.py':
        import build_eu
        if not build_eu.covers(model, B['bbox']):
            return None
        cmd += ['--cache', os.path.join(work, 'eu-cache')]
    return cmd

def _run_model(res: dict, day: str, flights: list[dict], work: str, model: str, workers: int, grids: str | None=None, corrections: str | None=None) -> None:
    logf = common.PRIVATE_LOG or os.path.join(work, 'day.log')
    key = res.get('key') or day
    for bi, B in enumerate(boxes(flights)):
        out = os.path.join(work, f'{model}-box{bi}')
        shutil.rmtree(out, ignore_errors=True)
        info = {'model': model, 'box': bi, 'region': B['region'], 'bbox': B['bbox'], 'flights': len(B['flights']), 'ok': False}
        cmd = build_command(day, B, out, model, workers, work)
        if cmd is None:
            common.note(f"{day} {model} box {B['region']}: outside the model's domain, not built")
            continue
        common.note(f"{day} {model} box {B['region']} {cmd[6]} flights {len(B['flights'])}")
        built = _run(cmd, common.PIPELINE, BUILD_TIMEOUT, logf)
        base = os.path.join(out, common.base_model(model))
        if not built or not os.path.exists(f'{base}/latest.json'):
            info['error'] = 'build'
            res['boxes'].append(info)
            shutil.rmtree(out, ignore_errors=True)
            continue
        if grids and (not common.is_lead(model) or common.base_model(model) in LEAD_GRIDS):
            try:
                gdir = os.path.join(grids, key, f'{model}-box{bi}')
                shutil.rmtree(gdir, ignore_errors=True)
                save_grids(base, gdir, {'day': day, 'key': key, 'model': model, 'region': B['region'], 'bbox': B['bbox'], 'tz': B['tz'], 'box': bi})
                info['saved'] = True
            except Exception as e:
                common.note(f'{day} {model} box {bi}: grids not saved: {e!r}')
                shutil.rmtree(os.path.join(grids, key, f'{model}-box{bi}'), ignore_errors=True)
        if _eval_box(base, B, out, model, bi, info, logf, corrections):
            info['ok'] = True
        res['boxes'].append(info)
        shutil.rmtree(out, ignore_errors=True)

def _was_built(rec: dict, model: str, bi: int) -> bool:
    bs = [b for b in rec.get('boxes') or [] if b.get('model') == model]
    return any((b.get('box', bi) == bi for b in bs))

def rescore_day(rec: dict, grids: str, work: str, corrections: str | None=None) -> dict:
    day = rec['date']
    flights = [clean(f) for f in rec.get('flights') or []]
    short = [clean(f) for f in rec.get('short') or []]
    obs = [clean(f) for f in rec.get('obs') or []]
    macro = rec.get('macro') or common.macro_of_items(flights + short)
    key = rec.get('key') or common.day_key(day, macro)
    models = tuple(rec.get('models') or common.models_for({f.get('region') for f in flights + short}, day))
    res = {**rec, 'version': common.DAY_VERSION, 'macro': macro, 'key': key, 'models': list(models), 'model': models[0], 'boxes': [], 'flights': flights, 'short': short, 'obs': obs, 'physics_v': common.physics_version(macro), 'score_v': common.SCORE_VERSION, 'corr_v': corr_version(corrections), 'grids': True, 'rescored_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    os.makedirs(work, exist_ok=True)
    logf = common.PRIVATE_LOG or os.path.join(work, 'day.log')
    items = flights + short + obs
    for model in models:
        for bi, B in enumerate(boxes(items)):
            info = {'model': model, 'box': bi, 'region': B['region'], 'bbox': B['bbox'], 'flights': len(B['flights']), 'ok': False, 'saved': True}
            gdir = os.path.join(grids, key, f'{model}-box{bi}')
            why = grids_usable(gdir, B['bbox'], model)
            if why == 'missing' and model in common.SAMPLED and (not _was_built(rec, model, bi)):
                continue
            if why and common.is_lead(model):
                res['lead_v'] = None
                continue
            if why:
                info['error'] = f'grids:{why}'
                res['boxes'].append(info)
                continue
            out = os.path.join(work, f'{model}-box{bi}')
            shutil.rmtree(out, ignore_errors=True)
            base = os.path.join(out, model)
            shutil.copytree(gdir, base)
            if _eval_box(base, B, out, model, bi, info, logf, corrections):
                info['ok'] = True
            res['boxes'].append(info)
            shutil.rmtree(out, ignore_errors=True)
    _categorize(res)
    return res
