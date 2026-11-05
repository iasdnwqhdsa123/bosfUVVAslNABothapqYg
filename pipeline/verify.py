from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = os.path.join(HERE, 'verify_stations.json')
UTC = timezone.utc
VALID_H = (0, 9, 12, 15, 18, 21)
LEAD_MAX = {'hrrr': 48, 'icond2': 48, 'icon': 180, 'gfs': 240, 'ecmwf': 240}
BACK_DAYS = {'hrrr': 30, 'gfs': 30, 'ecmwf': 30, 'icon': 7, 'icond2': 7}
RUN_HOURS = (0, 12)
FIRST_RUN = {'ecmwf': datetime(2026, 10, 20, 12, tzinfo=timezone.utc)}
READY_H = {'hrrr': 3, 'icond2': 4, 'icon': 6, 'gfs': 6, 'ecmwf': 10}
HOUR_S = {'hrrr': 90, 'icond2': 60, 'icon': 60, 'gfs': 45, 'ecmwf': 150}
WORKERS = {'hrrr': 2, 'ecmwf': 3}
FIELDS = ('t2', 'td2', 'u_sfc', 'v_sfc', 'gust', 'cloud', 'cloudlow', 'ceil', 'vis', 'rain', 'cape', 'wstar', 'zi', 'depth', 'cubase', 'cudepth', 'cuclass', 'cucover', 'elevation', 'mslp', 'overdev', 'cloudmid', 'cloudhigh', 'cueff', 'refc', 'overdev_nb')
NEAR_KM = 10.0

def log(*a):
    print(*a, flush=True)

def load_stations():
    with open(STATIONS) as f:
        return json.load(f)['stations']

def make_stations(history, out):
    rows = list(csv.DictReader(open(history, encoding='utf-8', errors='replace')))
    best = {}
    for r in rows:
        icao = r['ICAO'].strip()
        if len(icao) != 4 or not icao.isalpha() or r['END'] < '20250801' or (r['BEGIN'] > '20200101'):
            continue
        try:
            lat, lon, elev = (float(r['LAT']), float(r['LON']), float(r['ELEV(M)']))
        except ValueError:
            continue
        if not -56 < lat < 72 or not -170 < lon < 179.5 or elev < -400 or (elev > 5000) or (lat == 0 and lon == 0):
            continue
        fine = 24 < lat < 50 and -125 < lon < -66 or (35 < lat < 60 and -10 < lon < 30)
        k = (int(math.floor(lat)), int(math.floor(lon))) if fine else (int(lat // 3), int(lon // 3), 'c')
        if k not in best or r['BEGIN'] < best[k][0]:
            best[k] = (r['BEGIN'], [icao, round(lat, 3), round(lon, 3), round(elev)])
    st = sorted((v[1] for v in best.values()))
    with open(out, 'w') as f:
        json.dump({'source': 'NOAA ISD station history: an airport a 1° box (US, Europe), 3° elsewhere', 'stations': st}, f, separators=(',', ':'))
    log(f'{len(st)} stations')

def iem_id(icao):
    return icao[1:] if icao.startswith('K') else icao

def runs_to_do(model, done, now):
    out = []
    t = now.replace(minute=0, second=0, microsecond=0)
    last = t - timedelta(hours=READY_H[model])
    first = max(t - timedelta(days=BACK_DAYS[model]), FIRST_RUN.get(model, datetime.min.replace(tzinfo=UTC)))
    r = last.replace(hour=0)
    while r >= first - timedelta(days=1):
        for h in sorted(RUN_HOURS, reverse=True):
            run = r.replace(hour=h)
            if first <= run <= last and f'{run:%Y%m%d%H}' not in done:
                out.append(run)
        r -= timedelta(days=1)
    return out

def valid_hours(src, run, model):
    out = []
    for fh in range(1, min(LEAD_MAX[model], src.last_fh(run)) + 1):
        v = run + timedelta(hours=fh)
        if v.hour in VALID_H and src.fstep(fh) and src.builds(fh):
            out.append(v)
    return out

def _build(model, run_id, out, workers, valid):
    sys.path.insert(0, HERE)
    import build
    hours = [datetime.strptime(v, '%Y%m%d%H').replace(tzinfo=UTC) for v in valid]
    build.needed_times = lambda s, days=None, now=None: hours
    sys.argv = ['build.py', '--model', model, '--out', out, '--shard', '0/1', '--run', run_id, '--workers', str(workers)]
    return build.main()

def station_cells(model, stations):
    import build
    w, s, e, n, step = build.REGIONS[model][:5]
    lons, lats = build.target_grid(build.REGIONS[model])
    idx, iy, ix = ([], [], [])
    for i, (_, lat, lon, _) in enumerate(stations):
        x, y = (round((lon - w) / step), round((n - lat) / step))
        if 0 <= x < len(lons) and 0 <= y < len(lats):
            (idx.append(i), iy.append(y), ix.append(x))
    return (np.array(idx, np.int32), np.array(iy), np.array(ix))

def sample_hours(model, hours_dir, stations):
    import build
    from scipy.ndimage import maximum_filter
    idx, iy, ix = station_cells(model, stations)
    r = int(round(NEAR_KM / (build.REGIONS[model][4] * 111.0)))
    stamps = sorted((f[:-4] for f in os.listdir(hours_dir) if f.endswith('.npz') and (not f.endswith('.tmp.npz'))))
    if not stamps:
        return None
    codes = np.full((len(stamps), len(idx), len(FIELDS)), 255, np.uint8)
    for k, st in enumerate(stamps):
        with np.load(os.path.join(hours_dir, f'{st}.npz')) as h:
            for j, name in enumerate(FIELDS):
                if name in h.files:
                    codes[k, :, j] = h[name][iy, ix]
            if 'overdev' in h.files:
                od = h['overdev']
                near = maximum_filter(np.where(od == 255, 0, od), size=2 * r + 1) if r else od
                codes[k, :, FIELDS.index('overdev_nb')] = np.where(od[iy, ix] == 255, 255, near[iy, ix])
    has = (codes[:, :, FIELDS.index('t2')] != 255).any(axis=0)
    return (stamps, idx[has], codes[:, has, :])

def cmd_sample(a):
    sys.path.insert(0, HERE)
    import build
    import sources
    model, state = (a.model, a.state)
    deadline = datetime.fromisoformat(a.deadline.replace('Z', '+00:00'))
    os.makedirs(state, exist_ok=True)
    book_path = os.path.join(state, 'done.json')
    book = json.load(open(book_path)) if os.path.exists(book_path) else {}
    done, failed = (book.setdefault('done', {}), book.setdefault('failed', {}))
    per_hour = book.get('hour_s', HOUR_S[model])
    stations = load_stations()
    src = sources.SOURCES[model]
    now = datetime.now(UTC)
    gave_up = {r for r, f in failed.items() if f.get('n', 0) >= 2}
    todo = runs_to_do(model, {**done, **dict.fromkeys(gave_up)}, now)
    if a.part:
        i, n = (int(x) for x in a.part.split('/'))
        todo = [r for r in todo if int(r.timestamp() // 43200) % n == i]
    log(f'{model}: {len(todo)} runs to do, {len(done)} done; until {deadline:%H:%M}Z')
    enc = {k: list(build.ENC[k]) for k in FIELDS if k in build.ENC}
    enc['overdev_nb'] = enc['overdev']
    scratch = os.path.join(a.scratch, model)
    for run in todo:
        rid = f'{run:%Y%m%d%H}'
        valid = valid_hours(src, run, model)
        if not valid:
            continue
        workers = WORKERS.get(model, 3)
        need = per_hour * len(valid) / workers * 1.2 + 120
        left = (deadline - datetime.now(UTC)).total_seconds()
        if left < need:
            log(f'{model}: {rid} would take ~{need / 60:.0f} min, {left / 60:.0f} left: stopping here')
            break
        shutil.rmtree(scratch, ignore_errors=True)
        t0 = time.time()
        log(f'{model}: {rid}: {len(valid)} hours')
        cmd = [sys.executable, os.path.abspath(__file__), '_build', model, rid, scratch, str(workers), *[f'{v:%Y%m%d%H}' for v in valid]]
        try:
            subprocess.run(cmd, check=True, timeout=max(300, left - 60), cwd=HERE)
            got = sample_hours(model, os.path.join(scratch, model, '_hours'), stations)
        except subprocess.TimeoutExpired:
            log(f'{model}: {rid} not done by the deadline: next time')
            break
        except Exception as e:
            log(f'{model}: {rid} failed: {e}')
            failed[rid] = {'n': failed.get(rid, {}).get('n', 0) + 1, 'e': str(e)[:200]}
            got = None
        if got:
            stamps, idx, codes = got
            np.savez_compressed(os.path.join(state, f'{rid}.npz'), stamps=np.array(stamps), idx=idx, codes=codes, fields=np.array(FIELDS), enc=json.dumps(enc), model=model, run=rid)
            took = time.time() - t0
            per_hour = round(0.5 * per_hour + 0.5 * took * workers / len(stamps), 1)
            done[rid] = {'hours': len(stamps), 'stations': len(idx), 's': round(took)}
            log(f'{model}: {rid}: {len(stamps)} hours at {len(idx)} stations in {took / 60:.1f} min')
            failed.pop(rid, None)
        elif rid not in failed:
            failed[rid] = {'n': 2, 'e': 'no hours'}
        book['hour_s'] = per_hour
        with open(book_path, 'w') as f:
            json.dump(book, f)
    shutil.rmtree(scratch, ignore_errors=True)
    oldest = f'{now - timedelta(days=BACK_DAYS[model] + 11):%Y%m%d%H}'
    for f in os.listdir(state):
        if f.endswith('.npz') and f[:10] < oldest:
            os.remove(os.path.join(state, f))
    return 0
OBS = ('t2', 'td2', 't_half', 'ws', 'ws_half', 'wd', 'gust', 'vis', 'ceil', 'base_half', 'cu', 'cubase', 'low_lo', 'low_hi', 'precip', 'storm', 'fog', 'mslp', 'p_half', 'auto')
CU_BASE_FT = (1000, 9000)
LOW_FT = 6500
OKTAS = {'FEW': (1 / 8, 2 / 8), 'SCT': (3 / 8, 4 / 8), 'BKN': (5 / 8, 7 / 8), 'OVC': (1.0, 1.0), 'VV': (1.0, 1.0)}
CALM_KT = 3
KT, FT = (0.514444, 0.3048)
PRECIP = ('DZ', 'RA', 'SN', 'SG', 'PL', 'GR', 'GS', 'UP', 'IC')
BODY_END = re.compile('\\s(?:RMK|TEMPO|BECMG|NOSIG|PROB\\d\\d|INTER)(?:\\s|$)')
CLOUD = re.compile('(?<!\\S)(FEW|SCT|BKN|OVC|VV)(\\d{3}|///)(CB|TCU|///)?(?!\\S)')
CLEAR = re.compile('(?<!\\S)(?:CLR|SKC|NSC|NCD|CAVOK)(?!\\S)')
TEMPS = re.compile('(?<!\\S)(M?\\d\\d)/(M?\\d\\d)?(?!\\S)')
TGROUP = re.compile('(?<!\\S)T([01])(\\d{3})([01])(\\d{3})(?!\\S)')

def base_half(ft):
    return 50 if ft <= 5000 else 250 if ft <= 10000 else 500

def parse_metar(r, elev):
    f = lambda k: float(r[k]) if r.get(k) not in (None, '', 'M') else math.nan
    text = (r.get('metar') or '').strip()
    end = BODY_END.search(text)
    body, rmk = (text[:end.start()], text[end.start():]) if end else (text, '')
    o = dict.fromkeys(OBS, math.nan)
    tg = TGROUP.search(rmk)
    tb = TEMPS.search(body)
    num = lambda s: -int(s[1:]) if s.startswith('M') else int(s)
    if tg:
        o['t2'] = (-1 if tg[1] == '1' else 1) * int(tg[2]) / 10
        o['td2'] = (-1 if tg[3] == '1' else 1) * int(tg[4]) / 10
        o['t_half'] = 0.05
    elif tb:
        o['t2'] = num(tb[1])
        o['td2'] = num(tb[2]) if tb[2] else math.nan
        o['t_half'] = 0.5
    else:
        o['t2'], o['td2'], o['t_half'] = ((f('tmpf') - 32) / 1.8, (f('dwpf') - 32) / 1.8, 0.5)
    kt = f('sknt')
    if math.isfinite(kt):
        o['ws'], o['ws_half'] = (CALM_KT / 2 * KT, CALM_KT / 2 * KT) if kt == 0 else (kt * KT, 0.5 * KT)
    o['wd'] = f('drct') if kt and kt >= 1 else math.nan
    o['gust'] = f('gust') * KT
    o['vis'] = f('vsby') * 1609.34
    if 9900 <= o['vis'] < 10000:
        o['vis'] = 10000.0
    layers = [(c, int(b) * 100, t or '') for c, b, t in CLOUD.findall(body) if b != '///']
    clear = bool(CLEAR.search(body))
    if layers or clear:
        ceil = [b for c, b, _ in layers if c in ('BKN', 'OVC', 'VV')]
        o['ceil'] = min(ceil) * FT if ceil else math.inf
        o['base_half'] = base_half(min(ceil)) * FT if ceil else 0.0
        low = [b for _, b, _ in layers if b < CU_BASE_FT[0]]
        cu = [b for c, b, t in layers if CU_BASE_FT[0] <= b <= CU_BASE_FT[1] and (c in ('FEW', 'SCT') or t in ('TCU', 'CB'))]
        o['cu'] = 1.0 if cu and (not low) else 0.0
        if cu and (not low):
            o['cubase'] = min(cu) * FT
            o['base_half'] = base_half(min(cu)) * FT if not ceil else o['base_half']
        below = [c for c, b, _ in layers if b < LOW_FT]
        o['low_lo'], o['low_hi'] = OKTAS[below[-1]] if below else (0.0, 1 / 16)
    wx = (r.get('wxcodes') or '').split()
    here = [w.lstrip('+-') for w in wx if not w.lstrip('+-').startswith(('VC', 'RE'))]
    o['precip'] = 1.0 if any((p in w for w in here for p in PRECIP)) else 0.0
    o['storm'] = 1.0 if any(('TS' in w for w in wx)) or any((t == 'CB' for *_, t in layers)) else 0.0
    fog = any((w in ('FG', 'FZFG') for w in here))
    o['fog'] = 1.0 if fog or (math.isfinite(o['vis']) and o['vis'] < 1000) else 0.0
    if math.isfinite(f('mslp')):
        o['mslp'], o['p_half'] = (f('mslp'), 0.05)
    elif math.isfinite(f('alti')) and elev < 300:
        o['mslp'], o['p_half'] = (f('alti') * 33.8639, 0.5)
    o['auto'] = 1.0 if ' AUTO ' in f' {body} ' or re.search('\\bAO[12]A?\\b', rmk) else 0.0
    return [o[k] for k in OBS]

def qc(arr):
    k = {n: i for i, n in enumerate(OBS)}
    t, td, ws, g, p = (arr[:, :, k[n]] for n in ('t2', 'td2', 'ws', 'gust', 'mslp'))
    bad = 0
    for x, m in ((t, (t < -60) | (t > 55)), (td, (td > t + 0.5) | (td < t - 50) | (td < -70)), (ws, ws > 60), (g, (g > 80) | (g < ws)), (p, (p < 900) | (p > 1080))):
        m &= np.isfinite(x)
        bad += int(m.sum())
        x[m] = np.nan
    for x, jump in ((t, 8.0), (td, 10.0)):
        mid, a, b = (x[:, 1:-1], x[:, :-2], x[:, 2:])
        with np.errstate(invalid='ignore'):
            m = (np.abs(mid - (a + b) / 2) > jump) & (np.abs(a - b) < jump / 2)
        bad += int(m.sum())
        mid[m] = np.nan
    return bad

def parse_reports(text, ids, elev):
    out = defaultdict(dict)
    best = {}
    for r in csv.DictReader(io.StringIO(text)):
        i = ids.get(r.get('station', ''))
        if i is None:
            continue
        try:
            t = datetime.strptime(r['valid'], '%Y-%m-%d %H:%M').replace(tzinfo=UTC)
        except (KeyError, ValueError):
            continue
        hr = (t + timedelta(minutes=30)).replace(minute=0)
        off = abs((t - hr).total_seconds())
        if off > 1200 or best.get((i, hr), 1000000000.0) <= off:
            continue
        best[i, hr] = off
        out[i][hr] = parse_metar(r, elev[i])
    return out

def fetch_day(day, stations, batch=150):
    ids = {iem_id(s[0]): i for i, s in enumerate(stations)}
    elev = [s[3] for s in stations]
    arr = np.full((len(stations), 24, len(OBS)), np.nan, np.float32)
    names = list(ids)
    data = ['tmpf', 'dwpf', 'sknt', 'drct', 'gust', 'vsby', 'mslp', 'alti', 'wxcodes', 'metar']
    for k in range(0, len(names), batch):
        q = [('station', s) for s in names[k:k + batch]] + [('data', d) for d in data] + [('tz', 'Etc/UTC'), ('format', 'onlycomma'), ('missing', 'empty'), ('trace', 'empty'), ('report_type', '3'), ('sts', f'{day - timedelta(minutes=20):%Y-%m-%dT%H:%MZ}'), ('ets', f'{day + timedelta(hours=23, minutes=40):%Y-%m-%dT%H:%MZ}')]
        url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urllib.parse.urlencode(q)
        for attempt in range(4):
            try:
                text = urllib.request.urlopen(url, timeout=180).read().decode('utf-8', 'replace')
                break
            except Exception as e:
                if attempt == 3:
                    raise
                log(f'reports {day:%Y-%m-%d}: {e}; again')
                time.sleep(10 * (attempt + 1))
        for i, hours in parse_reports(text, ids, elev).items():
            for hr, vals in hours.items():
                if hr.date() == day.date():
                    arr[i, hr.hour] = vals
        time.sleep(1)
    return (arr, qc(arr))

def load_day(path):
    try:
        with np.load(path) as z:
            fields = tuple((str(x) for x in z['fields']))
            return (z['obs'], fields) if fields == OBS else None
    except (OSError, KeyError, ValueError):
        return None

def cmd_obs(a):
    stations = load_stations()
    os.makedirs(a.state, exist_ok=True)
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    deadline = datetime.fromisoformat(a.deadline.replace('Z', '+00:00')) if a.deadline else None
    for f in os.listdir(a.state):
        if f.endswith('.npy'):
            os.remove(os.path.join(a.state, f))
    for d in range(a.days, -1, -1):
        day = today - timedelta(days=d)
        path = os.path.join(a.state, f'{day:%Y%m%d}.npz')
        if d > 1 and load_day(path) is not None:
            continue
        if deadline and datetime.now(UTC) > deadline:
            break
        arr, dropped = fetch_day(day, stations)
        np.savez_compressed(path, obs=arr, fields=np.array(OBS))
        log(f'reports {day:%Y-%m-%d}: {int(np.isfinite(arr[:, :, 0]).sum())} station-hours, {dropped} values failed QC')
    for f in os.listdir(a.state):
        if f.endswith('.npz') and f[:8] < f'{today - timedelta(days=a.days + 5):%Y%m%d}':
            os.remove(os.path.join(a.state, f))
    return 0
LEADS = ((1, 12), (13, 24), (25, 48), (49, 72), (73, 120), (121, 240))
DAY_H = {'us': (15, 18, 21), 'europe': (9, 12, 15)}
REGIONS = ('us', 'europe', 'elsewhere')
MAX_DZ = 300
LOW_P_M = 300
SUN_PARTS = {'afternoon': (12, 18), 'night': (0, 6)}
SUN_MEASURES = ('t2', 'td2', 'wind')
MIN_N = {'cont': 50, 'cat': 100}
STATION_OFF = 5.0
CONT = {'t2': ('2 m temperature', '°C'), 'td2': ('2 m dewpoint', '°C'), 'wind': ('10 m wind speed', 'm/s'), 'wdir': ('Wind direction, winds of 5 kt and more', '°'), 'mslp': ('Sea-level pressure, stations under 300 m', 'hPa'), 'cubase': ('Cloudbase where both have cumulus (daytime)', 'm'), 'lowcloud': ('Low cloud cover (under 6,500 ft)', '%')}
CAT = {'cumulus': 'Cumulus or not (daytime)', 'ceil3000': 'Ceiling under 3,000 ft', 'ceil1000': 'Ceiling under 1,000 ft', 'vis5k': 'Visibility under 5 km', 'fog': 'Fog (under 1 km)', 'mvfr': 'Flight rules MVFR or worse', 'ifr': 'Flight rules IFR or worse', 'precip': 'Precipitation falling', 'storm': 'Thunderstorm or Cb (forecast within 10 km)', 'gust20': 'Gusts of 20 kt or more'}

def region_of(lat, lon):
    if 24 < lat < 50 and -125 < lon < -66:
        return 'us'
    if 35 < lat < 60 and -10 < lon < 30:
        return 'europe'
    return 'elsewhere'

def lead_of(h):
    for lo, hi in LEADS:
        if lo <= h <= hi:
            return f'{lo}-{hi}h'
    return None

def decode(codes, lo, hi, transform):
    c = codes.astype(np.float64)
    if transform == 'sqrt':
        a, b = (math.sqrt(max(lo, 0)), math.sqrt(hi))
        v = (a + c / 254 * (b - a)) ** 2
    else:
        v = lo + c / 254 * (hi - lo)
    return np.where(codes == 255, np.nan, v)

def cu_class(code):
    return np.where(code == 255, -1, np.where(code < 225, np.round(code / 50), code - 225 + 5))

def beyond(err, half):
    return np.sign(err) * np.maximum(0.0, np.abs(err) - half)

def flight_rules(ceil, vis):
    ft, sm = (ceil / FT, vis / 1609.34)
    cat = np.where((ft <= 3000) | (sm <= 5), 2.0, 1.0)
    cat = np.where((ft < 1000) | (sm < 3), 3.0, cat)
    cat = np.where((ft < 500) | (sm < 1), 4.0, cat)
    return np.where(np.isnan(ceil) | np.isnan(vis), np.nan, cat)

def measures(fc, o, elev, day):
    nan = lambda: np.full(len(elev), np.nan)
    dz = fc['elevation'] - elev if 'elevation' in fc else np.zeros(len(elev))
    near = np.abs(dz) <= MAX_DZ
    cont, cat = ({}, {})

    def put(name, err, half, ok=True):
        err = np.where(ok & near, err, np.nan)
        cont[name] = (err, beyond(err, half))
    if 't2' in fc:
        put('t2', fc['t2'] + 0.0065 * dz - o['t2'], o['t_half'])
    if 'td2' in fc:
        put('td2', fc['td2'] + 0.0018 * dz - o['td2'], o['t_half'])
    if 'u_sfc' in fc:
        ws = np.hypot(fc['u_sfc'], fc['v_sfc'])
        put('wind', ws - o['ws'], o['ws_half'])
        wd = (np.degrees(np.arctan2(-fc['u_sfc'], -fc['v_sfc'])) + 360) % 360
        put('wdir', (wd - o['wd'] + 180) % 360 - 180, 5.0, (o['ws'] >= 5 * KT) & (ws >= 1))
    if 'mslp' in fc:
        put('mslp', fc['mslp'] - o['mslp'], o['p_half'], np.asarray(elev) < LOW_P_M)
    cls = fc.get('cuclass')
    lo_m, hi_m = (CU_BASE_FT[0] * FT, CU_BASE_FT[1] * FT)
    if cls is not None and 'cubase' in fc:
        f_cu = np.isin(cls, (2, 3, 4, 5)) & (fc['cubase'] >= lo_m) & (fc['cubase'] <= hi_m)
        both = day & f_cu & (o['cu'] == 1)
        put('cubase', fc['cubase'] - o['cubase'], o['base_half'], both)
        cat['cumulus'] = (f_cu, o['cu'] == 1, day & np.isfinite(o['cu']) & (cls >= 0))
    if 'cloudlow' in fc:
        lo, hi = (o['low_lo'] * 100, o['low_hi'] * 100)
        err = fc['cloudlow'] - (lo + hi) / 2
        cont['lowcloud'] = (np.where(near, err, np.nan), np.where(near, np.where(fc['cloudlow'] < lo, fc['cloudlow'] - lo, np.maximum(0, fc['cloudlow'] - hi)), np.nan))
    if 'ceil' in fc:
        from physics import CEIL_NONE
        fceil = np.where(fc['ceil'] >= CEIL_NONE - 1, np.inf, fc['ceil'])
        known = ~np.isnan(o['ceil']) & ~np.isnan(fceil)
        for name, lim in (('ceil3000', 3000 * FT), ('ceil1000', 1000 * FT)):
            cat[name] = (fceil < lim, o['ceil'] < lim, known)
        if 'vis' in fc:
            fr, orl = (flight_rules(fceil, fc['vis']), flight_rules(o['ceil'], o['vis']))
            ok = ~np.isnan(fr) & ~np.isnan(orl)
            cat['mvfr'] = (fr >= 2, orl >= 2, ok)
            cat['ifr'] = (fr >= 3, orl >= 3, ok)
    if 'vis' in fc:
        ok = ~np.isnan(fc['vis']) & ~np.isnan(o['vis'])
        cat['vis5k'] = (fc['vis'] < 5000, o['vis'] < 5000, ok)
        f_fog = (fc['vis'] < 1000) | (cls == 8 if cls is not None else False)
        cat['fog'] = (f_fog, o['fog'] == 1, ok & np.isfinite(o['fog']))
    if 'rain' in fc:
        cat['precip'] = (fc['rain'] >= 0.1, o['precip'] == 1, ~np.isnan(fc['rain']) & np.isfinite(o['precip']))
    if cls is not None:
        f_st = fc['overdev_nb'] >= 50 if 'overdev_nb' in fc else (cls == 5) | (fc['overdev'] >= 50 if 'overdev' in fc else False)
        cat['storm'] = (f_st, o['storm'] == 1, (cls >= 0) & np.isfinite(o['storm']))
    if 'gust' in fc:
        peak = np.fmax(o['gust'], o['ws'])
        cat['gust20'] = (fc['gust'] >= 20 * KT, peak >= 20 * KT, ~np.isnan(fc['gust']) & np.isfinite(o['ws']))
    return (cont, cat)

def pairs(samples_dirs, obs_dir, stations):
    obs = {}
    for f in os.listdir(obs_dir):
        if f.endswith('.npz'):
            got = load_day(os.path.join(obs_dir, f))
            if got is not None:
                obs[f[:8]] = got[0]
    for d in samples_dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith('.npz'):
                continue
            with np.load(os.path.join(d, f)) as z:
                model, run = (str(z['model']), str(z['run']))
                fields = [str(x) for x in z['fields']]
                enc = json.loads(str(z['enc']))
                idx, codes, stamps = (z['idx'], z['codes'], [str(s) for s in z['stamps']])
            r0 = datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=UTC)
            val = {k: decode(codes[:, :, j], *enc[k]) for j, k in enumerate(fields) if k in enc}
            if 'cuclass' in fields:
                val['cuclass'] = cu_class(codes[:, :, fields.index('cuclass')])
            for k, st in enumerate(stamps):
                v = datetime.strptime(st, '%Y%m%d%H').replace(tzinfo=UTC)
                ob = obs.get(st[:8])
                lead = int((v - r0).total_seconds() // 3600)
                if ob is None or lead_of(lead) is None:
                    continue
                o = ob[idx, v.hour]
                yield (model, lead, v, idx, {n: a[k] for n, a in val.items()}, {n: o[:, j].astype(np.float64) for j, n in enumerate(OBS)})

def gather(samples_dirs, obs_dir, stations):
    lat = np.array([s[1] for s in stations])
    lon = np.array([s[2] for s in stations])
    elev = np.array([s[3] for s in stations], float)
    reg = np.array([REGIONS.index(region_of(a, b)) for a, b in zip(lat, lon)], np.int8)
    bins = {f'{lo}-{hi}h': i for i, (lo, hi) in enumerate(LEADS)}
    parts = defaultdict(list)
    for model, lead, v, idx, fc, o in pairs(samples_dirs, obs_dir, stations):
        r = reg[idx]
        day = np.array([v.hour in DAY_H.get(REGIONS[x], ()) for x in r])
        cont, cat = measures(fc, o, elev[idx], day)
        vh = int(v.timestamp() // 3600)
        t = {'key': idx.astype(np.int64) * 10 ** 12 + vh * 1000 + lead, 'st': idx.astype(np.int32), 'reg': r, 'lead': np.full(len(idx), bins[lead_of(lead)], np.int8), 'h': np.full(len(idx), lead, np.int16), 'sol': ((v.hour + lon[idx] / 15) % 24).astype(np.float32)}
        for n in CONT:
            e, b = cont.get(n, (None, None))
            t[n] = np.full(len(idx), np.nan, np.float32) if e is None else e.astype(np.float32)
            t[n + '_b'] = np.full(len(idx), np.nan, np.float32) if b is None else b.astype(np.float32)
        for n in CAT:
            fy, oy, ok = cat.get(n, (None, None, None))
            t[n] = np.full(len(idx), -1, np.int8) if fy is None else np.where(ok, 2 * np.asarray(fy, int) + np.asarray(oy, int), -1).astype(np.int8)
        parts[model].append(t)
    return {m: {k: np.concatenate([p[k] for p in ps]) for k in ps[0]} for m, ps in parts.items()}

def bad_stations(tables):
    off = defaultdict(list)
    for t in tables.values():
        ok = np.isfinite(t['t2'])
        st, e = (t['st'][ok], t['t2'][ok])
        order = np.argsort(st, kind='stable')
        st, e = (st[order], e[order])
        cut = np.flatnonzero(np.diff(st)) + 1
        for s, chunk in zip(np.split(st, cut), np.split(e, cut)):
            if len(chunk) >= 20:
                off[int(s[0])].append(float(np.median(chunk)))
    return {s for s, m in off.items() if len(m) >= 2 and (all((x >= STATION_OFF for x in m)) or all((x <= -STATION_OFF for x in m)))}

def cont_score(e, b):
    return {'n': int(len(e)), 'bias': round(float(e.mean()), 2), 'mae': round(float(np.abs(b).mean()), 2), 'mae_raw': round(float(np.abs(e).mean()), 2), 'rmse': round(float(np.sqrt((b * b).mean())), 2)}

def cat_score(code):
    h, fa, mi, cn = ((code == 3).sum(), (code == 2).sum(), (code == 1).sum(), (code == 0).sum())
    n = h + fa + mi + cn
    exp = ((h + fa) * (h + mi) + (mi + cn) * (fa + cn)) / n
    return {'n': int(n), 'events': int(h + mi), 'pod': round(h / max(1, h + mi), 2), 'far': round(fa / max(1, h + fa), 2), 'csi': round(h / max(1, h + fa + mi), 3), 'bias': round((h + fa) / max(1, h + mi), 2), 'heidke': round(float((h + cn - exp) / (n - exp)), 3) if n != exp else None}

def cmd_score(a):
    stations = load_stations()
    tables = gather(a.samples, a.obs, stations)
    n_rows = sum((len(t['key']) for t in tables.values()))
    bad = bad_stations(tables)
    for t in tables.values():
        keep = ~np.isin(t['st'], list(bad))
        for k in t:
            t[k] = t[k][keep]
    names = list(CONT) + list(CAT)
    rows, side = ([], [])
    for ri, r in enumerate(REGIONS):
        for li, (lo, hi) in enumerate(LEADS):
            lead = f'{lo}-{hi}h'
            sel = {m: (t['reg'] == ri) & (t['lead'] == li) for m, t in tables.items()}
            for m, t in tables.items():
                row = {'model': m, 'region': r, 'lead': lead}
                for n in names:
                    s = sel[m] & (np.isfinite(t[n]) if n in CONT else t[n] >= 0)
                    if s.sum() >= MIN_N['cont' if n in CONT else 'cat']:
                        row[n] = cont_score(t[n][s], t[n + '_b'][s]) if n in CONT else cat_score(t[n][s])
                if len(row) > 3:
                    rows.append(row)
            for n in names:
                have = {}
                for m, t in tables.items():
                    s = sel[m] & (np.isfinite(t[n]) if n in CONT else t[n] >= 0)
                    if s.sum() >= MIN_N['cont']:
                        have[m] = s
                if len(have) < 2:
                    continue
                common = None
                for m, s in have.items():
                    k = tables[m]['key'][s]
                    common = k if common is None else np.intersect1d(common, k, assume_unique=True)
                if len(common) < MIN_N['cont' if n in CONT else 'cat']:
                    continue
                by = {}
                for m, s in have.items():
                    t = tables[m]
                    s = s & np.isin(t['key'], common)
                    by[m] = cont_score(t[n][s], t[n + '_b'][s]) if n in CONT else cat_score(t[n][s])
                side.append({'measure': n, 'region': r, 'lead': lead, 'n': int(len(common)), 'models': by})
    by_sun = []
    for ri, r in enumerate(REGIONS):
        for part, (a0, a1) in SUN_PARTS.items():
            for n in SUN_MEASURES:
                by = {}
                for m, t in tables.items():
                    s = (t['reg'] == ri) & (t['h'] <= 48) & np.isfinite(t[n]) & ((t['sol'] - a0) % 24 < (a1 - a0) % 24)
                    if s.sum() >= MIN_N['cont']:
                        by[m] = {'n': int(s.sum()), 'bias': round(float(t[n][s].mean()), 2)}
                if by:
                    by_sun.append({'measure': n, 'region': r, 'part': part, 'models': by})
    out = {'generated': f'{datetime.now(UTC):%Y-%m-%dT%H:%MZ}', 'pairs': n_rows, 'stations_left_out': sorted((stations[s][0] for s in bad)), 'scores': rows, 'side_by_side': side, 'by_sun': by_sun}
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    log(f"{n_rows} forecasts paired with reports; {len(bad)} stations left out (every model off by {STATION_OFF:.0f} °C or more the same way: {', '.join(sorted((stations[s][0] for s in bad))[:20])})")
    log("Errors count past the report's own precision; stations over 300 m from the model's ground left out of the surface.")
    log('Side by side: the same stations, hours and leads for every model.')
    for n in names:
        lines = [s for s in side if s['measure'] == n]
        if not lines:
            continue
        if n in CONT:
            title, units = CONT[n]
            log(f'\n{title} (mean error, {units}; bias):')
            for s in lines:
                log(f"  {s['region']:10} {s['lead']:9} n={s['n']:7}  " + '  '.join((f"{m} {v['mae']:.2f} ({v['bias']:+.2f})" if units not in ('m', '°', '%') else f"{m} {v['mae']:.0f} ({v['bias']:+.0f})" for m, v in s['models'].items())))
        else:
            log(f'\n{CAT[n]} (Heidke skill: 1 perfect, 0 none; POD/false alarms):')
            for s in lines:
                ev = max((v['events'] for v in s['models'].values()))
                log(f"  {s['region']:10} {s['lead']:9} n={s['n']:7} events {ev:6}  " + '  '.join((f"{m} {(v['heidke'] if v['heidke'] is not None else '-')} ({v['pod']:.2f}/{v['far']:.2f})" for m, v in s['models'].items())))
    log("\nBias by the sun's time, leads 1-48 h (each model's own pairs):")
    for s in by_sun:
        title, units = CONT[s['measure']]
        log(f"  {s['region']:10} {s['part']:9} {title.split(',')[0]:16} " + '  '.join((f"{m} {v['bias']:+.2f} {units}" for m, v in s['models'].items())))
    return 0

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_build':
        _, _, model, run_id, out, workers, *valid = sys.argv
        return _build(model, run_id, out, int(workers), valid)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('stations')
    p.add_argument('history')
    p.add_argument('out', nargs='?', default=STATIONS)
    p = sub.add_parser('sample')
    p.add_argument('--model', required=True, choices=sorted(LEAD_MAX))
    p.add_argument('--state', required=True)
    p.add_argument('--deadline', required=True, help='UTC, e.g. 2026-10-24T07:55Z: no run started that would end after it')
    p.add_argument('--part', help='i/n: only every n-th run (several runners a model)')
    p.add_argument('--scratch', default=os.path.join(HERE, '..', 'verify-scratch'))
    p = sub.add_parser('obs')
    p.add_argument('--state', required=True)
    p.add_argument('--days', type=int, default=32)
    p.add_argument('--deadline')
    p = sub.add_parser('score')
    p.add_argument('--samples', nargs='+', required=True)
    p.add_argument('--obs', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'stations':
        return make_stations(a.history, a.out)
    return {'sample': cmd_sample, 'obs': cmd_obs, 'score': cmd_score}[a.cmd](a)
if __name__ == '__main__':
    sys.exit(main())
