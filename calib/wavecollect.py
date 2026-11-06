from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.request
import track
import weglide
POINTS = {'S Alps FR': (44.3, 6.2), 'W Alps': (45.9, 7.0), 'Valais': (46.3, 8.0), 'Engadin': (46.6, 10.0), 'Tirol': (47.2, 11.2), 'Salzburg': (47.5, 13.2), 'Steiermark': (47.3, 15.0), 'Bavaria': (47.8, 12.0), 'Dolomites': (46.4, 11.8)}
OM_API = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
SINCE = '2023-12-01'
WAVE_MONTHS = (10, 11, 12, 1, 2, 3, 4)
GROUPS = (('AT', 'CH', 'SI', 'IT'), ('FR', 'ES', 'PT'), ('DE', 'NL', 'BE', 'LU', 'DK'))
ALPS = (4.5, 43.6, 17.0, 48.5)
MIN_H, MIN_KM = (1.5, 50.0)
KEEP_MIN = 30.0
N_SAMPLES = 3000

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def load(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default

def save(p, obj):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p + '.tmp', 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(p + '.tmp', p)

def _om(lat, lon, d0, d1):
    url = f'{OM_API}?latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}&hourly=wind_speed_700hPa,wind_direction_700hPa&models=icon_d2&timezone=UTC'
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())['hourly']
        except Exception:
            if k == 3:
                raise
            time.sleep(5 * (k + 1))

def season_end(s: dt.date) -> dt.date:
    return dt.date(s.year + (s.month > 6), 6 if s.month > 6 else 12, 30 if s.month > 6 else 31)

def day_max(hourly: dict, h0: int=8, h1: int=16) -> dict:
    out = {}
    for t, ws, wd in zip(hourly['time'], hourly['wind_speed_700hPa'], hourly['wind_direction_700hPa']):
        hh = int(t[11:13])
        if ws is None or hh < h0 or hh > h1:
            continue
        d = t[:10]
        v = ws / 3.6
        if d not in out or v > out[d]['max700']:
            out[d] = {'max700': round(v, 1), 'dir': None if wd is None else int(wd)}
    return out

def wave_days(fetch=_om, since: str=SINCE, today: dt.date | None=None) -> dict:
    today = today or dt.date.today()
    days: dict = {}
    for name, (lat, lon) in POINTS.items():
        s = dt.date.fromisoformat(since)
        while s < today:
            e = min(season_end(s), today - dt.timedelta(days=2))
            if e <= s:
                break
            for d, r in day_max(fetch(lat, lon, s.isoformat(), e.isoformat())).items():
                if d not in days or r['max700'] > days[d]['max700']:
                    days[d] = {**r, 'where': name}
            s = e + dt.timedelta(days=1)
    sel = {d: r for d, r in days.items() if dt.date.fromisoformat(d).month in WAVE_MONTHS}
    return {'criterion': '700 hPa wind at any point, 08-16 UTC, Oct-Apr, ICON-D2 (Open-Meteo)', 'points': POINTS, 'days': dict(sorted(sel.items()))}

def in_alps(item: dict) -> bool:
    b = item.get('bbox')
    if not b or len(b) < 4:
        return False
    return b[2] > ALPS[0] and b[0] < ALPS[2] and (b[3] > ALPS[1]) and (b[1] < ALPS[3])

def hours(item: dict) -> float:
    try:
        return (dt.datetime.fromisoformat(item['landing_time']) - dt.datetime.fromisoformat(item['takeoff_time'])).total_seconds() / 3600
    except Exception:
        return 0.0

def candidates(items: list) -> list:
    c = [it for it in items if isinstance(it, dict) and in_alps(it) and (hours(it) >= MIN_H) and (((it.get('contest') or {}).get('distance') or 0) >= MIN_KM)]
    c.sort(key=lambda it: -((it.get('contest') or {}).get('distance') or 0))
    return c

def samples(data: dict, n: int=N_SAMPLES) -> dict | None:
    fx = track.fixes(data)
    if len(fx) < 100:
        return None
    idx = [round(i * (len(fx) - 1) / (n - 1)) for i in range(n)]
    f = [fx[i] for i in idx]
    return {'t': [round(x['t']) for x in f], 'lat': [round(x['p'][1], 5) for x in f], 'lon': [round(x['p'][0], 5) for x in f], 'alt': [round(x['z']) for x in f], 'gnd': [None if x['g'] is None else round(x['g']) for x in f], 'engine': [1 if x['e'] is not None and x['e'] > track.ENGINE_ON else 0 for x in f]}

def bearing(lat0, lon0, lat1, lon1) -> float:
    k = math.cos(math.radians((lat0 + lat1) / 2))
    return math.degrees(math.atan2((lon1 - lon0) * k, lat1 - lat0)) % 360

def wave_minutes(s: dict) -> float:
    t, lat, lon, alt, gnd, eng = (s['t'], s['lat'], s['lon'], s['alt'], s['gnd'], s['engine'])
    n = len(t)
    wave_s = 0.0
    j = 0
    for i in range(n - 1):
        while j < n - 1 and t[j] - t[i] < 60:
            j += 1
        if t[j] - t[i] < 45 or t[j] - t[i] > 90:
            continue
        if any(eng[i:j + 1]):
            continue
        if alt[i] - (gnd[i] if gnd[i] is not None else 0) < 600:
            continue
        if (alt[j] - alt[i]) / (t[j] - t[i]) <= 0.5:
            continue
        path = 0.0
        for k in range(i, j):
            path += math.hypot((lon[k + 1] - lon[k]) * 111320 * math.cos(math.radians(lat[k])), (lat[k + 1] - lat[k]) * 110540)
        chord = math.hypot((lon[j] - lon[i]) * 111320 * math.cos(math.radians(lat[i])), (lat[j] - lat[i]) * 110540)
        if path <= 300 or chord < 0.7 * path:
            continue
        m = (i + j) // 2
        h0 = bearing(lat[i], lon[i], lat[m], lon[m])
        h1 = bearing(lat[m], lon[m], lat[j], lon[j])
        if abs((h1 - h0 + 540) % 360 - 180) >= 30:
            continue
        wave_s += t[i + 1] - t[i]
    return wave_s / 60

def flight_file(fid: int, day: str, detail: dict, s: dict, data: dict) -> dict:
    ac = detail.get('aircraft') or {}
    return {'id': fid, 'date': day, 'aircraft': ac.get('name'), 'aircraft_kind': ac.get('kind'), 'polar': {'encoded_coeffs': ac.get('encoded_coeffs'), 'reference_mass': ac.get('reference_mass'), 'wingarea': ac.get('wingarea')}, 'mass': ac.get('reference_mass'), 'takeoff': {'airport': (detail.get('takeoff_airport') or {}).get('name'), 'time': detail.get('takeoff_time')}, 'engine_sensor_present': any((e is not None for e in data.get('engine_sensor') or [])), **s}

def run(state: str, requests: int, wind: float, per_day: int, gap: float, client=None) -> dict:
    weglide.GAP = gap
    wd_path = os.path.join(state, 'wavedays.json')
    wd = load(wd_path, None)
    if not wd:
        wd = wave_days()
        save(wd_path, wd)
    days = sorted((d for d, r in wd['days'].items() if r['max700'] >= wind), key=lambda d: -wd['days'][d]['max700'])
    st = load(os.path.join(state, 'state.json'), {'done': {}})
    index = load(os.path.join(state, 'flights', 'index.json'), {'criteria': {'wave': 'straight (|net heading change| <30 deg over 60 s, chord >= 0.7 path, path > 300 m), climb >0.5 m/s (60 s), >600 m AGL, engine off', 'keep': f'wave_straight_climb_min >= {KEEP_MIN} (10 s samples)', 'source': 'WeGlide, Alps wave days (wavedays.json), the longest Alps flights per day'}, 'flights': []})
    have = {f['id'] for f in index['flights']}
    todo = [d for d in days if d not in st['done']]
    cl = client or weglide.Client(budget=requests)
    log(f"{len(days)} wave days at {wind} m/s or more, {len(st['done'])} done, {len(have)} flights kept, {len(todo)} to do, {requests} requests this run")
    for day in todo:
        kept = fetched = cands = 0
        stopped = None
        try:
            items = []
            for g in GROUPS:
                got = cl._list(day, ','.join(g))
                if isinstance(got, list):
                    items += got
            c = candidates(items)
            cands = len(c)
            for it in c[:per_day]:
                fid = int(it['id'])
                if fid in have:
                    continue
                det = cl.detail(fid)
                dat = cl.data(fid)
                fetched += 1
                if not det or not dat:
                    continue
                s = samples(dat)
                if not s:
                    continue
                wm = wave_minutes(s)
                ac = det.get('aircraft') or {}
                index['flights'].append({'id': fid, 'date': day, 'aircraft': ac.get('name'), 'kind': ac.get('kind'), 'max_alt': max(s['alt']), 'wave_straight_climb_min': round(wm, 1), 'wind700': wd['days'][day]['max700'], 'kept': wm >= KEEP_MIN})
                if wm >= KEEP_MIN:
                    save(os.path.join(state, 'flights', f'{fid}.json'), flight_file(fid, day, det, s, dat))
                    kept += 1
                    have.add(fid)
            st['done'][day] = {'lists': len(items), 'candidates': cands, 'fetched': fetched, 'kept': kept}
        except weglide.Stop as e:
            stopped = str(e)
            if fetched:
                pass
        finally:
            save(os.path.join(state, 'state.json'), st)
            save(os.path.join(state, 'flights', 'index.json'), index)
        log(f"day {len(st['done'])}: {cands} candidates, {fetched} fetched, {kept} kept (total {len(have)}); {cl.used} requests used")
        if stopped:
            log(f'WeGlide done for this run ({stopped})')
            break
    return {'days_done': len(st['done']), 'days_total': len(days), 'flights': len(have), 'requests': cl.used, 'stopped': cl.stopped}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run'])
    ap.add_argument('state')
    ap.add_argument('--requests', type=int, default=1000)
    ap.add_argument('--wind', type=float, default=15.0)
    ap.add_argument('--per-day', type=int, default=8)
    ap.add_argument('--gap', type=float, default=5.0)
    a = ap.parse_args(argv)
    r = run(a.state, a.requests, a.wind, a.per_day, a.gap)
    log(json.dumps(r))
    return 0
if __name__ == '__main__':
    sys.exit(main())
