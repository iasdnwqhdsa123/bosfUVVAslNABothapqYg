from __future__ import annotations
import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import common
import track
TYPES = ('glider', 'hg', 'pg')
FEEDS_OUT = ('skylines',)
GAGGLE_S, GAGGLE_M = (15 * 60, 600.0)
RATE_MAX = 10.0
BELOW_M = -150.0
CONVERT_V = 1
FIRST_DAY = '2026-11-16'
FINAL_LAG = 2
MAX_DAYS = max(0, int(os.environ.get('CALIB_COLLECTOR_DAYS', '8') or 0))
MAX_CONVERT = 20
MAX_PULL = 40
MAX_TRIES = 3
MAX_3D = max(0, int(os.environ.get('CALIB_COLLECTOR_3D', '64') or 0))
REACH_KM = 2.0
NEAR_TIMEOUT = 15 * 60
PART_GAP, RETRIES = (1.5, 6)
UA = 'calibration-job/1.0'

class Overlap(SystemExit):
    pass

def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def write_lines(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with gzip.open(path + '.tmp', 'wt', encoding='utf-8') as f:
        f.write(''.join((x + '\n' for x in lines)))
    os.replace(path + '.tmp', path)

def new_state() -> dict:
    return {'fetched': {}, 'done': {}, 'items': {}, 'outside': {}}

def state_of(state: dict) -> dict:
    c = state.setdefault('collector', new_state())
    for k, v in new_state().items():
        c.setdefault(k, v)
    return c

class Puller:

    def __init__(self, url: str, secret: str, opener=None, sleep=time.sleep):
        self.url, self.secret = (url.split('?')[0], secret)
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=60))
        self._sleep = sleep
        self.used = 0

    def get(self, what: str, start: int | None=None) -> tuple[bytes, dict]:
        if what != 'list' and (not (len(what) == 10 and what[4] == '-' and (what[7] == '-'))):
            raise ValueError('collector: thermals=list or thermals=YYYY-MM-DD only')
        q = {'thermals': what, **({'from': int(start)} if start is not None else {})}
        req = urllib.request.Request(f'{self.url}?{urllib.parse.urlencode(q)}', headers={'x-feedback-secret': self.secret, 'User-Agent': UA})
        for k in range(RETRIES + 1):
            self.used += 1
            try:
                with self._open(req) as r:
                    return (r.read(), {h.lower(): v for h, v in (r.headers.items() if r.headers else [])})
            except urllib.error.HTTPError as e:
                if k < RETRIES and e.code in (429, 502, 503):
                    self._sleep(5 * 2 ** k)
                    continue
                raise RuntimeError(f'HTTP {e.code}') from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if k < RETRIES:
                    self._sleep(5 * 2 ** k)
                    continue
                raise RuntimeError(f'no answer ({type(e).__name__})') from None
        raise RuntimeError('no answer')

    def day(self, d: str, total: int) -> bytes:
        parts, at = ([], 0)
        while at < total:
            body, h = self.get(d, at)
            buf = gzip.decompress(body) if body[:2] == b'\x1f\x8b' else body
            if not buf:
                break
            total = int(h.get('x-total') or total)
            parts.append(buf)
            at += len(buf)
            if at < total:
                self._sleep(PART_GAP)
        return b''.join(parts)

def slim(text: bytes) -> tuple[list[str], Counter]:
    keep, cnt = ([], Counter())
    for line in text.decode('utf-8', 'replace').splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            cnt['unreadable'] += 1
            continue
        cnt[f"k_{r.get('k')}"] += 1
        if r.get('k') == 'c':
            keep.append(json.dumps(r, separators=(',', ':')))
    return (keep, cnt)

def pull(url: str | None, secret: str | None, out: str, fetched: dict, today: date | None=None, opener=None, sleep=time.sleep) -> dict:
    if not url or not secret:
        return {'skipped': 'THERMALS_PULL_URL / THERMALS_PULL_SECRET not set'}
    final = ((today or datetime.now(timezone.utc).date()) - timedelta(days=FINAL_LAG)).isoformat()
    p = Puller(url, secret, opener, sleep)
    body, _ = p.get('list')
    listing = json.loads(gzip.decompress(body) if body[:2] == b'\x1f\x8b' else body)
    days = [x for x in listing.get('days') or [] if str(x.get('day', '')) >= FIRST_DAY]
    want = [x for x in days if x['day'] <= final and (fetched.get(x['day']) or {}).get('bytes') != x.get('bytes')][:MAX_PULL]
    got = {}
    os.makedirs(os.path.join(out, 'collector', 'raw'), exist_ok=True)
    for x in want:
        d, n = (x['day'], int(x.get('bytes') or 0))
        text = p.day(d, n)
        keep, cnt = slim(text)
        write_lines(os.path.join(out, 'collector', 'raw', f'{d}.ndjson.gz'), keep)
        got[d] = {'bytes': len(text), 'records': sum((v for k, v in cnt.items() if k.startswith('k_'))), 'circled': len(keep), 'at': now_stamp()}
        common.note(f"collector: {d}: {got[d]['records']} records, {len(keep)} circled")
    common.write_json(os.path.join(out, 'collector', 'pulled.json'), {'days': got, 'listed': {x['day']: x.get('bytes') for x in days}, 'requests': p.used})
    return got

def item_id(r: dict) -> str:
    return hashlib.sha256(f"{r.get('a')}|{r.get('k')}|{r.get('t0')}".encode()).hexdigest()[:12]

def place_key(r: dict):
    try:
        t0, la0, lo0 = (float(r['t0']), float(r['la0']), float(r['lo0']))
    except (KeyError, TypeError, ValueError):
        return None
    region, tz, st_ = common.place_region(lo0, la0)
    if not region:
        return None
    return (common.day_key(datetime.fromtimestamp(t0, ZoneInfo(tz)).date().isoformat(), common.macro_of(region)), region, tz, st_)

def convert(records: list[dict], ground, drop=None, want=None) -> list[dict]:
    drop = drop if drop is not None else Counter()
    count = (lambda k, why: drop[k].__setitem__(why, drop[k][why] + 1)) if isinstance(drop, defaultdict) else lambda k, why: drop.__setitem__(why, drop[why] + 1)
    out = []
    for r in records:
        pk = place_key(r)
        if pk is None:
            count(None, "no place, or outside the job's areas")
            continue
        key, region, tz, st_ = pk
        if want is not None and (not want(key)):
            continue
        if r.get('k') != 'c':
            count(key, 'not circled')
            continue
        if r.get('y') not in TYPES:
            count(key, 'not a glider, hang glider or paraglider')
            continue
        if r.get('s') in FEEDS_OUT:
            count(key, 'SkyLines (no licence note yet)')
            continue
        try:
            t0, dur, z0, zt = (float(r['t0']), float(r['dur']), float(r['z0']), float(r['zt']))
            la0, lo0, la, lo = (float(r['la0']), float(r['lo0']), float(r['la']), float(r['lo']))
        except (KeyError, TypeError, ValueError):
            count(key, 'incomplete')
            continue
        gain = zt - z0
        if gain < track.CLIMB_GAIN or dur < track.CLIMB_DUR:
            count(key, f'under {track.CLIMB_GAIN:g} m or {track.CLIMB_DUR:g} s')
            continue
        rate = gain / dur
        if rate > RATE_MAX:
            count(key, f'implausible (over {RATE_MAX:g} m/s)')
            continue
        g0, g1 = (ground(lo0, la0), ground(lo, la))
        if g0 is not None and z0 - g0 < BELOW_M:
            count(key, 'under the ground (bad altitude)')
            continue
        day = common.day_of(key)
        air = r.get('air')
        c = {'t0': track.iso(t0), 'dur': round(dur), 'z0': round(z0), 'z1': round(zt), 'agl0': None if g0 is None else round(z0 - g0), 'agl1': None if g1 is None else round(zt - g1), 'g0': None if g0 is None else round(g0), 'gain': round(gain), 'rate': round(rate, 2), 'rate_c': r.get('rate'), 'air': None if air is None else float(air), 'lon': round(lo, 5), 'lat': round(la, 5), 'lon0': round(lo0, 5), 'lat0': round(la0, 5), 'rad': r.get('rad'), 'engon': 0, 'leg': None}
        out.append({'kind': 'collector', 'id': item_id(r), 'src': 'collector', 'date': day, 'region': region, 'tz': tz, 'state': st_, 'tp': [round((lo0 + lo) / 2, 5), round((la0 + la) / 2, 5)], 'bbox': None, 'key': r.get('pc') or ('hg' if r['y'] == 'hg' else 'pg' if r['y'] == 'pg' else 'club'), 'y': r['y'], 's': r.get('s'), 'a': r.get('a'), 'rv': r.get('rv'), 'climbs': [c]})
    return out

def _dist_m(a: tuple, b: tuple) -> float:
    k = math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot((a[0] - b[0]) * 111320.0, (a[1] - b[1]) * 111320.0 * k)

def gaggles(items: list[dict]) -> list[dict]:
    groups = []
    for f in sorted(items, key=lambda f: (track.ts(f['climbs'][0]['t0']), f['id'])):
        c = f['climbs'][0]
        t, p = (track.ts(c['t0']), (c['lat0'], c['lon0']))
        g = next((g for g in groups if abs(t - g['t1']) <= GAGGLE_S and _dist_m(g['at'], p) <= GAGGLE_M), None)
        if g:
            g['m'].append(f)
            g['t1'] = max(g['t1'], t)
            g['at'] = (sum((m['climbs'][0]['lat0'] for m in g['m'])) / len(g['m']), sum((m['climbs'][0]['lon0'] for m in g['m'])) / len(g['m']))
        else:
            groups.append({'t1': t, 'at': p, 'm': [f]})
    for i, g in enumerate(groups):
        for f in g['m']:
            f['g'], f['w'] = (i, round(1 / len(g['m']), 4))
    return items

def utc_days_of(day: str, tz: str) -> list[str]:
    z = ZoneInfo(tz)
    d = date.fromisoformat(day)
    a = datetime(d.year, d.month, d.day, 6, tzinfo=z).astimezone(timezone.utc).date()
    b = datetime(d.year, d.month, d.day, 21, tzinfo=z).astimezone(timezone.utc).date()
    return sorted({a.isoformat(), b.isoformat()})

def zones_of(macro: str) -> list[str]:
    if macro == 'SH':
        return sorted(set(common.SH_TZ.values()))
    if macro == 'EU':
        return sorted(set(common.EU_TZ.values()))
    return sorted(set(common.TZ.values()))

def ready(key: str, fetched: dict, listed: dict, today: date) -> bool:
    day, macro = common.split_key(key)
    for u in sorted({u for tz in zones_of(macro) for u in utc_days_of(day, tz)}):
        if u > (today - timedelta(days=FINAL_LAG)).isoformat():
            return False
        if u in listed and (fetched.get(u) or {}).get('bytes') != listed[u]:
            return False
    return True

def raw_days(*dirs: str) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, 'collector', 'raw', '*.ndjson.gz'))):
            out[os.path.basename(p)[:-len('.ndjson.gz')]] = p
    return out

def read_records(path: str) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def items_hash(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(sorted((f['id'] for f in items))).encode()).hexdigest()[:12]

def utc_days_for(key: str) -> list[str]:
    day, macro = common.split_key(key)
    return sorted({u for tz in zones_of(macro) for u in utc_days_of(day, tz)})

def src_of(key: str, raw: dict) -> dict:
    return {u: os.path.getsize(raw[u]) for u in utc_days_for(key) if u in raw}

def make_items(raw: dict, ground, only: set) -> dict:
    by, drops = (defaultdict(list), defaultdict(Counter))
    cache = {}

    def g(lon, lat):
        k = (round(lon, 4), round(lat, 4))
        if k not in cache:
            cache[k] = ground(lon, lat)
        return cache[k]
    days = sorted({u for k in only for u in utc_days_for(k) if u in raw})
    outside = {}
    for u in days:
        d = defaultdict(Counter)
        for f in convert(read_records(raw[u]), g, d, want=lambda k: k in only):
            by[common.day_key(f['date'], common.macro_of(f['region']))].append(f)
        outside[u] = sum(d[None].values())
        for k, c in d.items():
            if k is not None:
                drops[k].update(c)
    out = {k: {'items': gaggles(sorted(by.get(k, []), key=lambda f: f['id'])), 'dropped': dict(drops.get(k, {})), 'src': src_of(k, raw), 'v': CONVERT_V} for k in only}
    out[None] = {'outside': outside}
    return out

def keys_of(raw: dict) -> set:
    return {pk[0] for p in raw.values() for r in read_records(p) for pk in [place_key(r)] if pk}

def check_overlap(state: dict, keys, pending=()) -> None:
    done, seed, pend = (state.get('done') or {}, set(state.get('seed_days') or []), set(pending))
    clash = sorted((k for k in keys if k in done and (done[k].get('status') != 'empty' or done[k].get('flights')) or k in pend or k in seed))
    if clash:
        common.note(f"collector: these days are WeGlide's too: {', '.join(clash)}")
        raise Overlap(f'collector: {len(clash)} collector day(s) are WeGlide days too: the same flights could be counted twice. Stopped; which ones: the private log. Leave one source out for those days before the next run.')

def stale(key: str, x: dict | None, h: str, cv: str | None=None) -> str | None:
    if x is None or x.get('h') != h:
        return 'collector'
    if x.get('status') not in ('ok', 'partial'):
        return 'collector' if x.get('tries', 0) < MAX_TRIES else None
    if x.get('v', 1) < common.DAY_VERSION or not common.physics_matches(x.get('pv'), common.macro_of_key(key)):
        return 'collector' if x.get('redo_tries', 0) < MAX_TRIES else None
    if not x.get('grids'):
        return None
    return 'collector-rescore' if (x.get('sv') != common.SCORE_VERSION or x.get('cv') != cv) and x.get('rs_tries', 0) < MAX_TRIES else None

def plan_days(data: str, out: str, state: dict, today: date, cv: str | None, pending=(), ground=None, n: int=MAX_DAYS) -> list[tuple[str, str, float]]:
    cs = state_of(state)
    pj = os.path.join(out, 'collector', 'pulled.json')
    listed = {}
    if os.path.exists(pj):
        pulled = common.read_json(pj)
        cs['fetched'].update(pulled.get('days') or {})
        listed = pulled.get('listed') or {}
        cs['listed'] = listed
    listed = listed or cs.get('listed') or {}
    raw = raw_days(data, out)
    if not raw:
        return []
    keys = sorted((k for k in keys_of(raw) if ready(k, cs['fetched'], listed, today)))
    check_overlap(state, keys, pending)
    if n <= 0:
        return []
    redo = [k for k in keys if (cs['items'].get(k) or {}).get('v') != CONVERT_V or (cs['items'].get(k) or {}).get('src') != src_of(k, raw) or (not os.path.exists(items_path(data, out, k)))][:MAX_CONVERT]
    if redo:
        if ground is None:
            import layer2
            ground = layer2.Terrain(os.environ.get('CALIB_TERRAIN_CACHE') or os.path.join(out, '.terrain')).elev
        made = make_items(raw, ground, set(redo))
        cs['outside'].update(made.pop(None)['outside'])
        for k in redo:
            m = made[k]
            common.write_json(os.path.join(out, 'collector', 'items', f'{k}.json.gz'), {'key': k, **m})
            cs['items'][k] = {'v': CONVERT_V, 'src': m['src'], 'h': items_hash(m['items']), 'n': len(m['items']), 'thermals': len({f['g'] for f in m['items']}), 'dropped': m['dropped']}
        common.log(f"collector: {len(redo)} days' climbs made ({sum((cs['items'][k]['n'] for k in redo))} climbs)")
    work = []
    for k in keys:
        if len(work) >= n:
            break
        it = cs['items'].get(k)
        if not it or not it.get('n') or (not os.path.exists(items_path(data, out, k))):
            continue
        why = stale(k, cs['done'].get(k), it['h'], cv)
        if not why:
            continue
        inp = common.read_json(items_path(data, out, k))
        wid = f'collector/{k}'
        rec = None
        if why == 'collector-rescore':
            fn = os.path.join(data, 'collector', 'days', f'{k}.json.gz')
            if not os.path.exists(fn):
                why = 'collector'
            else:
                rec = common.read_json(fn)
        common.write_json(os.path.join(out, 'in', f'{wid}.json.gz'), {'day': wid, 'key': k, 'src': why, 'collector': inp['items'], 'rec': rec})
        regions = {f['region'] for f in inp['items']}
        cost = max(1, len(regions)) * len(models_of(k, inp['items'])) * (0.3 if why == 'collector-rescore' else 1)
        work.append((wid, why, cost))
    return work

def items_path(data: str, out: str, key: str) -> str:
    p = os.path.join(out, 'collector', 'items', f'{key}.json.gz')
    return p if os.path.exists(p) else os.path.join(data, 'collector', 'items', f'{key}.json.gz')

def models_of(key: str, items: list[dict]) -> tuple:
    return common.models_for({f.get('region') for f in items}, common.day_of(key), lead=False)

def build_one(j: dict, inp: dict, work: str, workers: int, grids_out: str, grids_in: str, fetch_grids, corr: str | None) -> dict:
    import day as dayrun
    key = inp['key']
    items = [dayrun.clean(f) for f in inp['collector']]
    if j['src'] == 'collector-rescore' and inp.get('rec'):
        if fetch_grids(key[:7], grids_in):
            res = dayrun.rescore_day(inp['rec'], grids_in, work, corr)
            if not any((str(b.get('error', '')).startswith('grids:') for b in res['boxes'])):
                return res
        common.note(f'{key}: collector day not scored again from its grids: built again')
    return dayrun.run_day(common.day_of(key), [], work, models=models_of(key, items), workers=workers, grids=grids_out, corrections=corr, collector=items)

def merge_one(data: str, wid: str, how: str, fn: str | None, state: dict, run: dict, months: set) -> None:
    import run as runmod
    cs = state_of(state)
    key = wid.split('/', 1)[1]
    prev = cs['done'].get(key, {})
    it = cs['items'].get(key) or {}
    if how == 'left':
        run['left'] += 1
        return
    run['days'] += 1
    run['collector'] = run.get('collector', 0) + 1
    res = common.read_json(fn) if how == 'built' else None
    own = runmod.own(res) if res else []
    okb = sum((b.get('ok', False) for b in own))
    if okb:
        os.makedirs(os.path.join(data, 'collector', 'days'), exist_ok=True)
        shutil.copyfile(fn, os.path.join(data, 'collector', 'days', f'{key}.json.gz'))
        cs['done'][key] = {'status': 'ok' if okb == len(own) else 'partial', 'tries': prev.get('tries', 0) + (0 if res.get('rescored_at') else 1), 'h': it.get('h'), 'n': len(res.get('collector') or []), 'boxes': len(own), **runmod.stamps(res)}
        if res.get('grids') and (not res.get('rescored_at')):
            months.add(key[:7])
        run['collector_ok'] = run.get('collector_ok', 0) + 1
        return
    x = dict(prev)
    if how != 'lost':
        if prev.get('status') in ('ok', 'partial') and prev.get('h') == it.get('h'):
            k = 'rs_tries' if res and res.get('rescored_at') else 'redo_tries'
            x[k] = prev.get(k, 0) + 1
        else:
            x = {'status': 'failed', 'h': it.get('h'), 'tries': (prev.get('tries', 0) if prev.get('h') == it.get('h') else 0) + 1}
    cs['done'][key] = x or {'status': 'failed', 'h': it.get('h'), 'tries': 0}
    run['failed' if how in ('error', 'built') else 'lost'] += 1

def store_pulled(plan: str, data: str) -> None:
    for sub in ('raw', 'items'):
        for p in glob.glob(os.path.join(plan, 'collector', sub, '*.json.gz')) + glob.glob(os.path.join(plan, 'collector', sub, '*.ndjson.gz')):
            os.makedirs(os.path.join(data, 'collector', sub), exist_ok=True)
            shutil.copyfile(p, os.path.join(data, 'collector', sub, os.path.basename(p)))

def load_days(data: str) -> list[dict]:
    return [common.read_json(p) for p in sorted(glob.glob(os.path.join(data, 'collector', 'days', '*.json.gz')))]

def place3d(base: str, B: dict, work: str, logf: str, info: dict) -> dict | None:
    import numpy as np
    import satcheck
    fs = [f for f in B['flights'] if f.get('kind') == 'collector']
    try:
        m = common.read_json(os.path.join(base, 'latest.json'))
        zs = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(base, '_hours', '*.npz'))}
        mids = [track.ts(f['climbs'][0]['t0']) + f['climbs'][0]['dur'] / 2 for f in fs]
        want = sorted({s for s in zs if any((abs(datetime.strptime(s[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp() - t) <= 5400 for t in mids))})
        if not want:
            info['near'] = {'error': 'no hours'}
            return None
        names, planes = ([], [])
        loaded = [np.load(zs[s]) for s in want]
        for name in satcheck.env_fields():
            vs = [satcheck.decode(m, z, name) for z in loaded]
            if all((v is None for v in vs)):
                continue
            names.append(name)
            planes.append(vs)
        ny, nx = (m['grid']['ny'], m['grid']['nx'])
        d = os.path.join(work, 'near')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'env.bin'), 'wb') as fh:
            for k in range(len(want)):
                for vs in planes:
                    v = vs[k]
                    fh.write((v if v is not None else np.full((ny, nx), np.nan, np.float32)).astype(np.float32).tobytes())
        with open(os.path.join(d, 'env.json'), 'w') as fh:
            json.dump({'grid': m['grid'], 'stamps': want, 'fields': names, 'tiles': [], 'px': satcheck.PIX_KM}, fh)
        pts = []
        for i, (f, t) in enumerate(zip(fs, mids)):
            c = f['climbs'][0]
            pts.append({'i': i, 'lat': (c['lat0'] + c['lat']) / 2, 'lon': (c['lon0'] + c['lon']) / 2, 't': t * 1000, 'z0': c['z0'], 'z1': c['z1'], 'g': c.get('g0')})
        with open(os.path.join(d, 'points.json'), 'w') as fh:
            json.dump({'points': pts, 'max': MAX_3D, 'reach': REACH_KM}, fh)
        env = {**os.environ}
        r = subprocess.run(['node', '--max-old-space-size=4000', satcheck.node_script(), 'near', d, os.path.join(d, 'points.json'), os.path.join(d, 'out.json')], capture_output=True, text=True, timeout=NEAR_TIMEOUT, env=env, cwd=common.HERE)
        if r.returncode or not os.path.exists(os.path.join(d, 'out.json')):
            with open(logf, 'a') as lf:
                lf.write(r.stderr[-2000:])
            info['near'] = {'error': 'node'}
            return None
        res = common.read_json(os.path.join(d, 'out.json'))
        info['near'] = {'placed': res.get('placed', 0), 'skipped': res.get('skipped', 0), 'fields': len(names)}
        return {fs[p['i']]['id']: {'prof': p.get('prof'), 't3d': p.get('t3d')} for p in res.get('points') or []}
    except Exception as e:
        common.note(f'collector: the profile and 3D pass failed: {e!r}')
        info['near'] = {'error': type(e).__name__}
        return None
    finally:
        shutil.rmtree(os.path.join(work, 'near'), ignore_errors=True)

def merge3d(B: dict, model: str, near: dict) -> None:
    for f in B['flights']:
        v = near.get(f.get('id')) if f.get('kind') == 'collector' else None
        if v:
            f['climbs'][0].setdefault('near', {})[model] = v

def rows_of(days: list[dict]) -> list[dict]:
    out = []
    for d in days:
        for f in d.get('collector') or []:
            c = (f.get('climbs') or [{}])[0]
            for m in d.get('models') or []:
                own = (c.get('models') or {}).get(m) or {}
                nr = (c.get('near') or {}).get(m) or {}
                prof, t3 = (nr.get('prof') or {}, nr.get('t3d') or {})
                out.append({'model': m, 'macro': d.get('macro') or common.macro_of(f.get('region')), 'region': f.get('region'), 'y': f.get('y'), 'key': f.get('key'), 'w': f.get('w', 1.0), 'g': (d.get('key'), f.get('g')), 'rate': c.get('rate'), 'air': c.get('air'), 'agl0': c.get('agl0'), 'agl1': c.get('agl1'), 'paired': bool(own), 'ws0': own.get('ws0'), 'ws': own.get('ws'), 'climb': own.get('climb'), 'ach': own.get('ach'), 'top': own.get('top'), 'solar': own.get('solar'), 'prof': prof.get('lift'), 'core': t3.get('core'), 'best': t3.get('best'), 'd3': t3.get('d'), 'near': bool(nr)})
    return out

def _wmedian(xs: list[tuple[float, float]]):
    xs = sorted(xs)
    tot = sum((w for _, w in xs))
    if not xs or tot <= 0:
        return None
    acc = 0.0
    for v, w in xs:
        acc += w
        if acc >= tot / 2:
            return v
    return xs[-1][0]

def _ranks(vs: list[float]) -> list[float]:
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    r = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r

def wspearman(ys, xs, ws):
    if len(ys) < 3:
        return None
    ry, rx = (_ranks(list(ys)), _ranks(list(xs)))
    tw = sum(ws)
    my, mx = (sum((w * a for w, a in zip(ws, ry))) / tw, sum((w * b for w, b in zip(ws, rx))) / tw)
    cov = sum((w * (a - my) * (b - mx) for w, a, b in zip(ws, ry, rx)))
    vy, vx = (sum((w * (a - my) ** 2 for w, a in zip(ws, ry))), sum((w * (b - mx) ** 2 for w, b in zip(ws, rx))))
    return None if vy <= 0 or vx <= 0 else cov / math.sqrt(vy * vx)

def compare(rows: list[dict], fy, fx, ratio_min: float=0.05) -> dict:
    ps = [(fy(r), fx(r), r['w'], r['g']) for r in rows]
    ps = [(y, x, w, g) for y, x, w, g in ps if y is not None and x is not None and math.isfinite(y) and math.isfinite(x)]
    if not ps:
        return {'n': 0, 't': 0, 'bias': None, 'ratio': None, 'rho': None}
    tw = sum((w for _, _, w, _ in ps))
    return {'n': len(ps), 't': len({g for *_, g in ps}), 'bias': sum((w * (y - x) for y, x, w, _ in ps)) / tw, 'ratio': _wmedian([(y / x, w) for y, x, w, _ in ps if x > ratio_min]), 'rho': wspearman([y for y, *_ in ps], [x for _, x, *_ in ps], [w for _, _, w, _ in ps])}

def cell(s: dict, unit: str='') -> str:
    if not s['n']:
        return '–'
    b = '–' if s['bias'] is None else f"{s['bias']:+.0f}" if unit == 'm' else f"{s['bias']:+.2f}"
    r = '–' if s['ratio'] is None else f"{s['ratio']:.2f}"
    p = '–' if s['rho'] is None else f"{s['rho']:.2f}"
    return f'{b} / {r} / {p}'
HEIGHT_BINS = [(0, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 0.95), (0.95, 9)]
COLS = [('air / W*', lambda r: r['air'], lambda r: r['ws0'], ''), ('climb / W*', lambda r: r['rate'], lambda r: r['ws0'], ''), ('air / profile', lambda r: r['air'], lambda r: r['prof'], ''), ('air / 3D nearest', lambda r: r['air'], lambda r: r['core'], ''), ('air / 3D best near', lambda r: r['air'], lambda r: r['best'], ''), ("climb / site's climb", lambda r: r['rate'], lambda r: r['climb'], ''), ('top AGL / usable top (m)', lambda r: r['agl1'], lambda r: r['top'], 'm')]

def section(days: list[dict], state: dict | None=None) -> list[str]:
    cs = state_of(state) if state is not None else new_state()
    rows = rows_of(days)
    climbs = [f for d in days for f in d.get('collector') or []]
    built = {d.get('key') for d in days}
    drop = Counter()
    for k, it in (cs.get('items') or {}).items():
        if k in built:
            drop.update(it.get('dropped') or {})
    fetched = cs.get('fetched') or {}
    n_rec, n_circ = (sum(((x or {}).get('records', 0) for x in fetched.values())), sum(((x or {}).get('circled', 0) for x in fetched.values())))
    out = ['## Collector climbs (review only)', '', f"The live-traffic collector's circled climbs (its own records: every glider, hang glider and paraglider OGN hears, all day; no WeGlide flight), paired with the forecast where and when each was climbed, as WeGlide's climbs are. **Review only**: nothing here changes the site and no training reads them (their days are kept apart, DATA/collector/). Kept: circled, {track.CLIMB_GAIN:g} m gained and {track.CLIMB_DUR:g} s at least (the job's own climb), not SkyLines' (no licence note yet), in an area the job builds (the southern hemisphere's regions with GFS and ICON, build_sh.py). The ground from AWS's terrain tiles (zoom {11}). A gaggle (climbs within {GAGGLE_S // 60} min and {GAGGLE_M:g} m) is one thermal: each climb weighs 1 / its thermal's climbs. Each cell: bias (flown − forecast, m/s; tops m) / ratio (flown / forecast, weighted median) / ρ (weighted rank correlation).", '', "- **W\\***: the Thermal Strength layer's value there (the forecast's W\\*, as the sounding's Thermal strength reads it); **air**: the air's rise the collector worked out (the climb plus the glider's circling sink); **climb**: the climb rate flown (gain / time).", "- **profile**: the atmospheric profile's lift through the climb's own heights (profile.js: 1.3 × W\\* × the core's profile up the thermal layer, over the real ground), the two hours round it blended.", f"- **3D**: the 3D map's placed thermals (cuplace.js placeChunk: placeThermals on the real terrain, at the hour nearest the climb; no ground cover from imagery and no learned hotspots, which the site's own 3D map has: the hotspots are learned from these same climbs), the nearest core at the climb's heights (its lean taken) and the strongest within {REACH_KM:g} km, their peaks' mean through the climb's heights; at most {MAX_3D} tile-hours a box (the most climbed first), a tile's neighbours not placed (near its edge the nearest may be the next tile's).", "- **site's climb**: the climb the site's model gives that polar there (XC Speed's); **usable top**: Usable Height's, over the model's ground.", '']
    out += [f"- Climbs: {len(climbs)} on {len(days)} days ({len({(d.get('key'), f.get('g')) for d in days for f in d.get('collector') or []})} thermals); by area: {dict(Counter((common.macro_of(f.get('region')) for f in climbs)))}; by kind: {dict(Counter((f.get('y') for f in climbs)))}.", f"- Records fetched: {n_rec} on {len(fetched)} UTC days; flown straight through lift (no climb), not kept: {n_rec - n_circ}; circled, with no place or outside the job's areas: {sum((cs.get('outside') or {}).values())}.", f"- Circled, dropped on the days built: {', '.join((f'{k} {v}' for k, v in drop.most_common())) or 'none'}.", '']
    if not rows:
        return out + ['No collector day built yet.', '']
    for m in sorted({r['model'] for r in rows}, key=lambda m: (common.model_macro(m), m)):
        rs = [r for r in rows if r['model'] == m]
        pr = [r for r in rs if r['paired']]
        out += [f'### {common.model_label(m)}', '', f"{len(rs)} climbs; with the forecast {len(pr)} (unpaired: out of the model's hours, or its box failed), with the profile {sum((r['prof'] is not None for r in rs))}, with a 3D thermal {sum((r['core'] is not None for r in rs))} (median distance {_fmt(_wmedian([(r['d3'], r['w']) for r in rs if r['d3'] is not None]), 2)} km).", '', '| Region | climbs (thermals) | ' + ' | '.join((h for h, *_ in COLS)) + ' |', '|---|---|' + '---|' * len(COLS)]
        groups = [('all', pr)] + sorted(((g, [r for r in pr if r['region'] == g]) for g in {r['region'] for r in pr}), key=lambda kv: kv[0])
        groups += sorted(((f'kind {g}', [r for r in pr if r['y'] == g]) for g in {r['y'] for r in pr} if len({r['y'] for r in pr}) > 1), key=lambda kv: kv[0])
        frac = lambda r: (r['agl0'] + r['agl1']) / 2 / r['top'] if r.get('agl0') is not None and r.get('agl1') is not None and ((r.get('top') or 0) > 100) else None
        groups += [(f'mid-climb at {a:g}–{b:g} of the usable top', [r for r in pr if frac(r) is not None and a <= frac(r) < b]) for a, b in HEIGHT_BINS]
        groups += [('topping out over the usable top', [r for r in pr if r.get('agl1') is not None and (r.get('top') or 0) > 100 and (r['agl1'] > r['top'])])]
        for g, grs in groups:
            if not grs:
                continue
            out.append(f"| {g} | {len(grs)} ({len({r['g'] for r in grs})}) | " + ' | '.join((cell(compare(grs, fy, fx), u) for _, fy, fx, u in COLS)) + ' |')
        out.append('')
    return out

def _fmt(v, p=2):
    return '–' if v is None else f'{v:.{p}f}'

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pull')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'pull':
        os.makedirs(os.path.join(a.out, 'logs'), exist_ok=True)
        common.PRIVATE_LOG = os.path.join(a.out, 'logs', f'{now_stamp()}-collector.log')
        sp = os.path.join(a.data, 'state.json')
        fetched = (state_of(common.read_json(sp)) if os.path.exists(sp) else new_state())['fetched']
        try:
            got = pull(os.environ.get('THERMALS_PULL_URL'), os.environ.get('THERMALS_PULL_SECRET'), a.out, fetched, datetime.now(timezone.utc).date())
        except Exception as e:
            msg = str(e).replace(os.environ.get('THERMALS_PULL_URL') or '\x00', '<url>')
            print(f'::warning::collector climbs: not fetched ({msg[:120]})')
            common.note(f'collector: pull failed: {e!r}'.replace(os.environ.get('THERMALS_PULL_URL') or '\x00', '<url>'))
            return 0
        if 'skipped' in got:
            print(f"::notice::collector climbs: none fetched ({got['skipped']})")
            return 0
        common.log(f"collector: {len(got)} days fetched ({sum((x['records'] for x in got.values()))} records, {sum((x['circled'] for x in got.values()))} circled)")
    return 0
if __name__ == '__main__':
    sys.exit(main())
