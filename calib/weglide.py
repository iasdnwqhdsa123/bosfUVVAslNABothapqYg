from __future__ import annotations
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
import common
API = 'https://api.weglide.org/v1'
UA = 'Mozilla/5.0 (X11; Linux x86_64) research-script/1.0'
GAP = 2.5
MAX_FAILS = 3
MAX_PER_DAY = 10
MIN_KM = {'high': 100, 'low': 40, 'vlow': 15}
MIN_H = {'high': 1.5, 'low': 1.0, 'vlow': 0.75}
SHARE = {'low': 0.25, 'vlow': 0.065}

class Stop(Exception):
    pass

class Client:

    def __init__(self, budget: int, opener=None, sleep=time.sleep, clock=time.monotonic):
        self.budget, self.used, self.fails = (budget, 0, 0)
        self.stopped = None
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=40))
        self._sleep, self._clock, self._last = (sleep, clock, -1000000000.0)
        self.status = {}
        self.lists = {}

    def get(self, path: str, params: dict | None=None):
        if self.stopped:
            raise Stop(self.stopped)
        url = f'{API}/{path}' + ('?' + urllib.parse.urlencode(params) if params else '')
        for attempt in range(3):
            if self.used >= self.budget:
                self.stopped = 'budget'
                raise Stop('budget')
            wait = GAP - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()
            self.used += 1
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
            try:
                with self._open(req) as r:
                    body = r.read()
                self.fails = 0
                self.status['200'] = self.status.get('200', 0) + 1
                return json.loads(body)
            except urllib.error.HTTPError as e:
                self.status[str(e.code)] = self.status.get(str(e.code), 0) + 1
                if e.code == 404:
                    self.fails = 0
                    return None
                if e.code == 429 or e.code >= 500:
                    self.fails += 1
                    if self.fails >= MAX_FAILS:
                        self.stopped = f'http {e.code}'
                        raise Stop(self.stopped)
                    ra = e.headers.get('Retry-After') if e.headers else None
                    self._sleep(min(300, float(ra)) if ra and ra.isdigit() else 30 * (attempt + 1))
                    continue
                self.fails += 1
                if self.fails >= MAX_FAILS:
                    self.stopped = f'http {e.code}'
                    raise Stop(self.stopped)
                return None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                self.status['error'] = self.status.get('error', 0) + 1
                self.fails += 1
                if self.fails >= MAX_FAILS:
                    self.stopped = type(e).__name__
                    raise Stop(self.stopped)
                self._sleep(30 * (attempt + 1))
        return None

    def day_list(self, day: str) -> list:
        r = self.get('flight', {'country_id_in': 'US', 'scoring_date_start': day, 'scoring_date_end': day, 'order_by': '-distance', 'limit': 100})
        r = r if isinstance(r, list) else []
        self.lists[day] = [slim(it) for it in r if isinstance(it, dict)]
        return r

    def detail(self, fid: int):
        return self.get(f'flightdetail/{int(fid)}')

    def data(self, fid: int):
        return self.get(f'flightdata/{int(fid)}')

def _h(item) -> float | None:
    try:
        a = datetime.fromisoformat(item['takeoff_time']).timestamp()
        b = datetime.fromisoformat(item['landing_time']).timestamp()
        return (b - a) / 3600
    except Exception:
        return None

def _where(item: dict):
    ap = item.get('takeoff_airport') or {}
    reg = ap.get('region') or ''
    if not reg.startswith('US'):
        return None
    st = reg[3:] if reg.startswith('US-') else None
    bbox = item.get('bbox')
    from_ap = st is not None
    if st is None:
        if not bbox:
            return None
        st = common.nearest_state((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    if st in ('AK', 'HI') or st not in common.REGION:
        return None
    return (st, from_ap)

def candidate(item: dict, aircraft: dict, unmapped: dict) -> dict | None:
    w = _where(item)
    if not w:
        return None
    st, from_ap = w
    ap = item.get('takeoff_airport') or {}
    bbox = item.get('bbox')
    name = (item.get('aircraft') or {}).get('name') or ''
    if name not in aircraft:
        unmapped[name] = unmapped.get(name, 0) + 1
        return None
    ac = aircraft[name]
    if ac is None:
        return None
    km = (item.get('contest') or {}).get('distance') or 0
    h = _h(item)
    if h is None or km < MIN_KM[ac['grp']] or h < MIN_H[ac['grp']]:
        return None
    return {'id': item['id'], 'date': item.get('scoring_date'), 'state': st, 'state_from_airport': from_ap, 'region': common.REGION[st], 'tz': ap.get('timezone') or common.TZ.get(st, 'America/Chicago'), 'ac': name, 'key': ac['key'], 'key_exact': ac['exact'], 'cls': ac['cls'], 'grp': ac['grp'], 'takeoff': item.get('takeoff_time'), 'landing': item.get('landing_time'), 'dur_h': round(h, 2), 'bbox': bbox, 'dmst': item.get('dmst_index'), 'list_km': km, 'list_kmh': (item.get('contest') or {}).get('speed'), 'pilot': pilot_of(item)}

def pilot_of(item: dict) -> str | None:
    if 'pilot' in item:
        return item['pilot']
    return common.pilot_key((item.get('user') or {}).get('id'))

def slim(item: dict) -> dict:
    ap = item.get('takeoff_airport') or {}
    con = item.get('contest') or {}
    return {'id': item.get('id'), 'scoring_date': item.get('scoring_date'), 'takeoff_time': item.get('takeoff_time'), 'landing_time': item.get('landing_time'), 'bbox': item.get('bbox'), 'dmst_index': item.get('dmst_index'), 'takeoff_airport': {'region': ap.get('region'), 'timezone': ap.get('timezone')}, 'aircraft': {'name': (item.get('aircraft') or {}).get('name')}, 'contest': {'distance': con.get('distance'), 'speed': con.get('speed')}, 'pilot': pilot_of(item)}

def choose(day: str, items: list, aircraft: dict, counts: dict, unmapped: dict, seen: set, taken=(), room: int=MAX_PER_DAY) -> list:
    buckets = {}
    taken = set(taken)
    for it in items:
        if it.get('id') in seen:
            continue
        c = candidate(it, aircraft, unmapped)
        if c and c['date'] == day:
            buckets.setdefault((c['region'], c['grp']), []).append(c)
    rnd = random.Random(f'7:{day}')
    keys = sorted((k for k in buckets if k not in taken))
    rnd.shuffle(keys)
    total = max(1, sum(counts.values()))
    under = lambda g: g in SHARE and counts.get(g, 0) / total < SHARE[g]
    keys.sort(key=lambda k: 0 if under(k[1]) else 1)
    room = max(0, min(room, MAX_PER_DAY))
    out = []
    for k in keys[:room]:
        out.append(rnd.choice(buckets[k]))
    chosen = {c['id'] for c in out}
    rest = [c for k in sorted(buckets) for c in buckets[k] if c['id'] not in chosen]
    rnd2 = random.Random(f'10:{day}')
    rnd2.shuffle(rest)
    rest.sort(key=lambda c: 0 if under(c['grp']) else 1)
    return out + rest[:room - len(out)]
SLED_MIN = 25
AFTERNOON_SOLAR = 12
EARLY_SOLAR = 15
SHORT_TRACKS_PER_DAY = 4
LIST_LIMIT = 100

def solar_hour(t_iso: str | None, lon: float | None) -> float | None:
    try:
        t = datetime.fromisoformat(t_iso).timestamp()
    except Exception:
        return None
    return None if lon is None else round((t / 3600 % 24 + lon / 15) % 24, 2)

def short_candidate(item: dict, aircraft: dict) -> dict | None:
    w = _where(item)
    if not w:
        return None
    name = (item.get('aircraft') or {}).get('name') or ''
    ac = aircraft.get(name)
    h, bbox = (_h(item), item.get('bbox'))
    if not ac or h is None or (not bbox):
        return None
    km = (item.get('contest') or {}).get('distance') or 0
    if km >= MIN_KM[ac['grp']] and h >= MIN_H[ac['grp']]:
        return None
    st, from_ap = w
    tp = [round((bbox[0] + bbox[2]) / 2, 5), round((bbox[1] + bbox[3]) / 2, 5)]
    s0, s1 = (solar_hour(item.get('takeoff_time'), tp[0]), solar_hour(item.get('landing_time'), tp[0]))
    return {'id': item['id'], 'kind': 'short', 'date': item.get('scoring_date'), 'state': st, 'state_from_airport': from_ap, 'region': common.REGION[st], 'tz': (item.get('takeoff_airport') or {}).get('timezone') or common.TZ.get(st, 'America/Chicago'), 'ac': name, 'key': ac['key'], 'cls': ac['cls'], 'grp': ac['grp'], 'takeoff': item.get('takeoff_time'), 'landing': item.get('landing_time'), 'dur_h': round(h, 2), 'bbox': bbox, 'list_km': km, 'tp': tp, 'tp_src': 'bbox', 'solar0': s0, 'solar1': s1, 'afternoon': s0 is not None and s0 >= AFTERNOON_SOLAR, 'sled': h * 60 < SLED_MIN, 'pilot': pilot_of(item), 'src': 'weglide'}

def day_summary(day: str, items: list, aircraft: dict) -> dict:
    regs = {}
    for it in items:
        if it.get('scoring_date') != day:
            continue
        w = _where(it)
        name = (it.get('aircraft') or {}).get('name') or ''
        ac = aircraft.get(name)
        h, bbox = (_h(it), it.get('bbox'))
        if not w or not ac or h is None:
            continue
        km = (it.get('contest') or {}).get('distance') or 0
        lon = (bbox[0] + bbox[2]) / 2 if bbox else common.CENTROID[w[0]][0]
        s0, s1 = (solar_hour(it.get('takeoff_time'), lon), solar_hour(it.get('landing_time'), lon))
        r = regs.setdefault(common.REGION[w[0]], {'up': 0, 'long': 0, 'aft': 0, 'aft_short': 0, 'aft_sled': 0, 'aft_early': 0})
        r['up'] += 1
        long = km >= MIN_KM[ac['grp']] and h >= MIN_H[ac['grp']]
        r['long'] += long
        if s0 is not None and s0 >= AFTERNOON_SOLAR:
            r['aft'] += 1
            r['aft_short'] += not long
            r['aft_sled'] += h * 60 < SLED_MIN
            r['aft_early'] += s1 is not None and s1 < EARLY_SOLAR
    return {'n': len(items), 'full': len(items) >= LIST_LIMIT, 'regions': regs}
