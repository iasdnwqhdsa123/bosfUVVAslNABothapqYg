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
MAX_PER_DAY = 6
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
        return r if isinstance(r, list) else []

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

def candidate(item: dict, aircraft: dict, unmapped: dict) -> dict | None:
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
    return {'id': item['id'], 'date': item.get('scoring_date'), 'state': st, 'state_from_airport': from_ap, 'region': common.REGION[st], 'tz': ap.get('timezone') or common.TZ.get(st, 'America/Chicago'), 'ac': name, 'key': ac['key'], 'key_exact': ac['exact'], 'cls': ac['cls'], 'grp': ac['grp'], 'takeoff': item.get('takeoff_time'), 'landing': item.get('landing_time'), 'dur_h': round(h, 2), 'bbox': bbox, 'dmst': item.get('dmst_index'), 'list_km': km, 'list_kmh': (item.get('contest') or {}).get('speed')}

def choose(day: str, items: list, aircraft: dict, counts: dict, unmapped: dict, seen: set) -> list:
    buckets = {}
    for it in items:
        if it.get('id') in seen:
            continue
        c = candidate(it, aircraft, unmapped)
        if c and c['date'] == day:
            buckets.setdefault((c['region'], c['grp']), []).append(c)
    rnd = random.Random(f'7:{day}')
    keys = sorted(buckets)
    rnd.shuffle(keys)
    total = max(1, sum(counts.values()))
    under = lambda g: g in SHARE and counts.get(g, 0) / total < SHARE[g]
    keys.sort(key=lambda k: 0 if under(k[1]) else 1)
    out = []
    for k in keys[:MAX_PER_DAY]:
        out.append(rnd.choice(buckets[k]))
    return out
