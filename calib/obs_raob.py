from __future__ import annotations
import argparse
import gzip
import http.client
import io
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import lake as lakemod
IGRA = 'https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive'
Y2D = IGRA + '/access/data-y2d/{id}-data-beg{year}.txt.zip'
STATION_LIST = IGRA + '/doc/igra2-station-list.txt'
RAW = 'raw/obs/raob/igra2'
STATIONS_RAW = 'raw/obs/raob/igra2-stations'
MISSING, REMOVED = (-9999, -8888)
RAOB_LAG = 3
RAOB_SETTLE = 10
RAOB_REVISIT = 21
REV_MAX = 9
NEW_MAX = 60
RAOB_EVERY = 3
MAX_FETCH = 120
GAP_S = 2.0
TRIES = 4
WAIT_S = (1, 2, 4)
TIMEOUT_S = 120
ZIP_MAX = 100 * 2 ** 20
LAKE_STOP = 3
GAP_M, GAP_SHARE = (1000.0, 0.3)
GAP_SIG_M = 3500.0
DENSE_M = 150.0
HOLE_MIN_M = 300.0
USER_AGENT = 'obs-raob/1 (calibration lab)'
STATION_ID = re.compile('[A-Z0-9]{11}')
WIND_LOW_FT = [1000]
WIND_ALTS_FT = [2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 18000, 24000, 30000]
WAVE_ALTS_FT = [6000, 8000, 10000, 12000, 14000, 18000, 24000]
ML_TOP_DZ = (200, 500, 1000, 1800, 3000)
EXCESS_HEAT, EXCESS_COOL = (1.0, 0.2)
LCL_PER_C = 125.0
BLAPSE_MAX = 15.0
THX_MAX = 5.0
BLUE_OVER_LCL = 500.0
CU_OVERSHOOT, CU_OVERSHOOT_MIN = (0.5, 500.0)
Q_HYPSO, Q_NOSFC, Q_STNELEV, Q_REMOVED = (1, 2, 4, 8)
Q_GAP, Q_TOP, Q_NODATA = (16, 32, 64)
Q_REACH, Q_NEARCU = (128, 256)
Q_MISSING = Q_GAP | Q_TOP | Q_NODATA

@dataclass
class Level:
    lt1: int
    lt2: int
    etime: int | None
    p: float | None
    pflag: str
    z: float | None
    zflag: str
    t: float | None
    tflag: str
    rh: float | None
    dpd: float | None
    wdir: float | None
    wspd: float | None
    removed: frozenset = frozenset()

    @property
    def surface(self) -> bool:
        return self.lt2 == 1

@dataclass
class Sounding:
    id: str
    year: int
    month: int
    day: int
    hour: int
    reltime: int
    numlev: int
    p_src: str
    np_src: str
    lat: float
    lon: float
    levels: list = field(default_factory=list)
    lines: list = field(default_factory=list)

    @property
    def date(self) -> str:
        return f'{self.year:04d}-{self.month:02d}-{self.day:02d}'

    @property
    def t_valid(self) -> int | None:
        h = self.hour if 0 <= self.hour <= 23 else self.reltime // 100 if self.reltime != 9999 and self.reltime // 100 <= 23 else None
        if h is None:
            return None
        try:
            return int(datetime(self.year, self.month, self.day, h, tzinfo=timezone.utc).timestamp())
        except ValueError:
            return None

    @property
    def t_release(self) -> int | None:
        hh, mm = divmod(self.reltime, 100)
        tv = self.t_valid
        if tv is None or self.reltime == 9999 or mm > 59 or (hh > 23):
            return None
        t = int(datetime(self.year, self.month, self.day, hh, mm, tzinfo=timezone.utc).timestamp())
        return t - 86400 if t - tv > 12 * 3600 else t

def _int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None

def _val(s: str, scale: float, name: str, removed: set):
    v = _int(s)
    if v is None or v == MISSING:
        return None
    if v == REMOVED:
        removed.add(name)
        return None
    return v / scale

def parse_header(line: str) -> Sounding | None:
    s = line.rstrip('\r\n').ljust(71)
    if not s.startswith('#'):
        return None
    nums = [_int(s[a:b]) for a, b in ((13, 17), (18, 20), (21, 23), (24, 26), (27, 31), (32, 36), (55, 62), (63, 71))]
    if any((n is None for n in nums)):
        return None
    y, mo, d, h, rel, n, lat, lon = nums
    if not (1 <= mo <= 12 and 1 <= d <= 31 and (n >= 0)):
        return None
    return Sounding(s[1:12].strip(), y, mo, d, h, rel, n, s[37:45].strip(), s[46:54].strip(), lat / 10000.0, lon / 10000.0)

def parse_level(line: str) -> Level | None:
    s = line.rstrip('\r\n').ljust(51)
    lt1, lt2 = (_int(s[0]), _int(s[1]))
    if lt1 not in (0, 1, 2, 3) or lt2 not in (0, 1, 2):
        return None
    rem: set = set()
    et = _int(s[3:8])
    et_s = None if et is None or et in (MISSING, REMOVED) or et < 0 else et // 100 * 60 + et % 100
    lv = Level(lt1, lt2, et_s, _val(s[9:15], 100.0, 'p', rem), s[15].strip(), _val(s[16:21], 1.0, 'z', rem), s[21].strip(), _val(s[22:27], 10.0, 't', rem), s[27].strip(), _val(s[28:33], 10.0, 'rh', rem), _val(s[34:39], 10.0, 'dpd', rem), _val(s[40:45], 1.0, 'wdir', rem), _val(s[46:51], 10.0, 'wspd', rem))
    lv.removed = frozenset(rem)
    return lv

def parse(text: str) -> tuple[list[Sounding], int]:
    out, bad, cur = ([], 0, None)
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith('#'):
            cur = parse_header(line)
            if cur is None:
                bad += 1
            else:
                cur.lines.append(line)
                out.append(cur)
            continue
        if cur is None:
            bad += 1
            continue
        cur.lines.append(line)
        lv = parse_level(line)
        if lv is None:
            bad += 1
            continue
        cur.levels.append(lv)
    return (out, bad)

def _days(text: str) -> dict[str, tuple[str, set]]:
    sds, _ = parse(text)
    days: dict[str, tuple[list, set]] = {}
    for s in sds:
        ls, hs = days.setdefault(s.date, ([], set()))
        ls.extend(s.lines)
        tv = s.t_valid
        if tv is not None:
            hs.add(datetime.fromtimestamp(tv, timezone.utc).hour)
    return {d: ('\n'.join(ls) + '\n', hs) for d, (ls, hs) in days.items()}

def by_day(text: str) -> dict[str, str]:
    return {d: t for d, (t, _) in _days(text).items()}

def parse_stations(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        s = line.rstrip('\r\n').ljust(88)
        sid = s[0:11].strip()
        try:
            lat, lon, elev = (float(s[12:20]), float(s[21:30]), float(s[31:37]))
            first, last = (int(s[72:76]), int(s[77:81]))
        except ValueError:
            continue
        if not STATION_ID.fullmatch(sid) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        out.append({'id': sid, 'lat': lat, 'lon': lon, 'elev': None if elev <= -998 else elev, 'state': s[38:40].strip(), 'name': s[41:71].strip(), 'first': first, 'last': last})
    return out

def area_of(lon: float, lat: float) -> str | None:
    for macro, box in (('US', common.US_BOX), ('EU', common.EU_BOX)):
        if box[0] <= lon <= box[2] and box[1] <= lat <= box[3]:
            return macro
    for _, b, _ in common.SH_REGIONS:
        if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]:
            return 'SH'
    return None

def select(stations: list[dict], year: int) -> list[dict]:
    out = []
    for s in stations:
        a = area_of(s['lon'], s['lat'])
        if a and s['last'] >= year - 1:
            out.append({**s, 'area': a})
    return out

def due(stations: list[dict], day: date, every: int=RAOB_EVERY) -> list[dict]:
    every = max(1, int(every))
    return [s for s in stations if (zlib.crc32(s['id'].encode()) + day.toordinal()) % every == 0]

class Igra:

    def __init__(self, budget: int=MAX_FETCH, opener=None, sleep=time.sleep, clock=time.monotonic, gap: float=GAP_S):
        self.budget, self.used, self.fails, self.retries, self.requests = (budget, 0, 0, 0, 0)
        self.last = ''
        self._first_year = None
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=TIMEOUT_S))
        self._sleep, self._clock, self._last, self._gap = (sleep, clock, -1000000000.0, gap)

    def _wait(self) -> None:
        dt = self._gap - (self._clock() - self._last)
        if dt > 0:
            self._sleep(dt)
        self._last = self._clock()

    def get(self, url: str) -> tuple[int, bytes | None]:
        self.requests += 1
        for i in range(TRIES):
            self._wait()
            try:
                with self._open(urllib.request.Request(url, headers={'User-Agent': USER_AGENT})) as r:
                    return (200, r.read())
            except urllib.error.HTTPError as e:
                code = e.code
                if code == 404:
                    return (404, None)
                if code != 429 and code < 500:
                    break
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError, OSError, ValueError):
                pass
            if i < TRIES - 1:
                self.retries += 1
                self._sleep(WAIT_S[min(i, len(WAIT_S) - 1)])
        self.fails += 1
        return (0, None)

    def station_file(self, sid: str, today: date) -> str | None:
        if self.used >= self.budget:
            self.last = 'no budget'
            return None
        self.used += 1
        years = (today.year, today.year - 1)
        if self._first_year in years:
            years = (self._first_year,) + tuple((y for y in years if y != self._first_year))
        for year in years:
            code, body = self.get(Y2D.format(id=sid, year=year))
            if code == 404:
                continue
            if body is None:
                self.last = 'failed'
                return None
            self._first_year = year
            text = unzip(body, sid)
            self.last = '' if text is not None else 'failed'
            return text
        self.last = 'not found'
        return None

    def station_list(self) -> str | None:
        code, body = self.get(STATION_LIST)
        return body.decode('utf-8', 'replace') if body is not None else None

def unzip(body: bytes, sid: str) -> str | None:
    try:
        z = zipfile.ZipFile(io.BytesIO(body))
        names = [n for n in z.namelist() if not n.endswith('/')]
        name = f'{sid}-data.txt' if f'{sid}-data.txt' in names else names[0] if len(names) == 1 else None
        if name is None or z.getinfo(name).file_size > ZIP_MAX:
            return None
        with z.open(name) as f:
            data = f.read(ZIP_MAX + 1)
        if len(data) > ZIP_MAX:
            return None
        return data.decode('ascii', 'replace')
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, KeyError, EOFError, zlib.error, NotImplementedError, RuntimeError):
        return None

def day_key(sid: str, day: str, rev: int=1) -> str:
    return f'{RAW}/{sid}/{day}.txt.gz' if rev <= 1 else f'{RAW}/{sid}/{day}.r{rev}.txt.gz'

def _held(lk, sid: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in lk.list(f'{RAW}/{sid}/'):
        name = k.rsplit('/', 1)[-1]
        if not name.endswith('.txt.gz'):
            continue
        stem = name[:-len('.txt.gz')]
        d, _, r = stem.partition('.r')
        rev = int(r) if r.isdigit() else 1 if not r else None
        if rev is not None and len(d) == 10:
            out[d] = max(out.get(d, 0), rev)
    return out

def _gz(text: str) -> bytes:
    return gzip.compress(text.encode('ascii', 'replace'), compresslevel=9, mtime=0)

def _ungz(blob: bytes) -> str:
    return gzip.decompress(blob).decode('ascii', 'replace')

def put_text(lk, key: str, text: str, counts: dict) -> str:
    r = lk.put_plain(key, _gz(text), strict=False)
    st = r.status
    if st == 'differs':
        try:
            st = 'same' if _ungz(lk.get_plain(key)) == text else 'differs'
        except (OSError, EOFError, zlib.error, lakemod.LakeError):
            st = 'differs'
    counts[st] = counts.get(st, 0) + 1
    return st

def _n(counts: dict, k: str) -> None:
    counts[k] = counts.get(k, 0) + 1

def store_station(lk, sid: str, text: str, today: date, lag: int=RAOB_LAG, counts: dict | None=None, most: int=NEW_MAX) -> dict:
    counts = {} if counts is None else counts
    last = (today - timedelta(days=lag)).isoformat()
    settle = (today - timedelta(days=RAOB_SETTLE)).isoformat()
    revisit = (today - timedelta(days=RAOB_REVISIT)).isoformat()
    held = _held(lk, sid)
    days = _days(text)
    newest = max(days) if days else ''
    new = 0
    for d in sorted(days, reverse=True):
        t, hours = days[d]
        if d > last:
            _n(counts, 'recent')
        elif d in held:
            rev = held[d]
            if d < revisit:
                _n(counts, 'held')
                continue
            try:
                kept = _ungz(lk.get_plain(day_key(sid, d, rev)))
            except (OSError, EOFError, zlib.error):
                kept = None
            if kept is None or kept == t:
                _n(counts, 'held')
            elif rev >= REV_MAX:
                _n(counts, 'revisions full')
            else:
                put_text(lk, day_key(sid, d, rev + 1), t, counts)
                _n(counts, 'revised')
        elif d > settle and (d == newest or not {0, 12} <= hours):
            _n(counts, 'waiting')
        elif new >= most:
            _n(counts, 'deferred')
        else:
            new += 1
            put_text(lk, day_key(sid, d), t, counts)
    return counts

def store_stations(lk, text: str, today: date, counts: dict | None=None) -> None:
    put_text(lk, f'{STATIONS_RAW}/{today:%Y-%m}.txt.gz', text, {} if counts is None else counts)

def read_stations(lk, today: date) -> str | None:
    keys = sorted((k for k in lk.list(f'{STATIONS_RAW}/') if k.endswith('.txt.gz')))
    want = f'{STATIONS_RAW}/{today:%Y-%m}.txt.gz'
    key = want if want in keys else keys[-1] if keys else None
    return _ungz(lk.get_plain(key)) if key else None

def read_day(lk, sid: str, day: str) -> list[Sounding]:
    rev = _held(lk, sid).get(day, 1)
    return parse(_ungz(lk.get_plain(day_key(sid, day, rev))))[0]

def days_held(lk, sid: str) -> list[str]:
    return sorted(_held(lk, sid))

def ingest(lk, igra: Igra, today: date, every: int=RAOB_EVERY, most: int=MAX_FETCH, lag: int=RAOB_LAG, minutes: float | None=None, clock=time.monotonic) -> dict:
    t0 = clock()
    counts: dict = {}
    txt = igra.station_list()
    if txt is not None:
        try:
            store_stations(lk, txt, today, {})
        except lakemod.LakeError:
            _n(counts, 'lake failed')
    else:
        try:
            txt = read_stations(lk, today)
        except (lakemod.LakeError, OSError, EOFError, zlib.error):
            txt = None
    if not txt:
        counts['no station list'] = 1
        return counts
    sts = due(select(parse_stations(txt), today.year), today, every)[:max(0, most)]
    counts['due'] = len(sts)
    row = 0
    for i, s in enumerate(sts):
        if minutes is not None and clock() - t0 > minutes * 60:
            counts['out of time'] = len(sts) - i
            break
        text = igra.station_file(s['id'], today)
        if text is None:
            _n(counts, getattr(igra, 'last', '') or 'failed')
            continue
        _n(counts, 'fetched')
        try:
            store_station(lk, s['id'], text, today, lag, counts)
            row = 0
        except lakemod.LakeError:
            _n(counts, 'lake failed')
            row += 1
            if row >= LAKE_STOP:
                counts['lake stopped'] = 1
                break
    if igra.retries:
        counts['retried'] = igra.retries
    return counts

def _physics():
    for d in (os.path.join(HERE, os.pardir, 'pipeline'), os.path.join(os.getcwd(), 'pipeline'), os.getcwd()):
        if os.path.isfile(os.path.join(d, 'physics.py')):
            d = os.path.abspath(d)
            if d not in sys.path:
                sys.path.append(d)
            break
    import physics
    return physics

def _td(P, lv: Level) -> float | None:
    if lv.t is None:
        return None
    if lv.dpd is not None:
        return lv.t - max(lv.dpd, 0.0)
    if lv.rh is not None and lv.rh > 0:
        return float(P.dewpoint_from_rh(lv.t, lv.rh))
    return None

def column(sd: Sounding, elev: float | None=None) -> dict | None:
    P = _physics()
    flags = 0
    if any((lv.removed & {'p', 'z', 't', 'dpd', 'rh', 'wdir', 'wspd'} for lv in sd.levels)):
        flags |= Q_REMOVED
    sfc = next((lv for lv in sd.levels if lv.surface and lv.p is not None), None)
    if sfc is not None and sfc.t is None:
        sfc = None
    plv = [lv for lv in sd.levels if lv.p is not None and lv.p > 0 and (sfc is None or lv is sfc or (lv.p < sfc.p and (not lv.surface)))]
    plv.sort(key=lambda lv: -lv.p)
    seen, uniq = (set(), [])
    for lv in plv:
        if lv.p in seen:
            continue
        seen.add(lv.p)
        uniq.append(lv)
    plv = uniq
    if sfc is None:
        flags |= Q_NOSFC
        while plv and plv[0].t is None:
            plv.pop(0)
    if not plv or plv[0].t is None:
        return None
    lnp = [math.log(lv.p) for lv in plv]
    tv = []
    for lv in plv:
        if lv.t is None:
            tv.append(None)
            continue
        td = _td(P, lv)
        w = float(P.mixing_ratio(min(td, lv.t), lv.p)) if td is not None else 0.0
        tv.append((lv.t + P.K0) * (1 + 0.61 * w))
    known = [i for i, x in enumerate(tv) if x is not None]
    for i, x in enumerate(tv):
        if x is None:
            lo = max((k for k in known if k < i), default=None)
            hi = min((k for k in known if k > i), default=None)
            if lo is None or hi is None:
                tv[i] = tv[lo if hi is None else hi]
            else:
                f = (lnp[i] - lnp[lo]) / (lnp[hi] - lnp[lo])
                tv[i] = tv[lo] + f * (tv[hi] - tv[lo])
    H = [0.0]
    for i in range(1, len(plv)):
        H.append(H[-1] + P.RD / P.G * 0.5 * (tv[i - 1] + tv[i]) * (lnp[i - 1] - lnp[i]))
    zrep = [lv.z for lv in plv]
    if zrep[0] is None and elev is not None and (sfc is not None and plv[0] is sfc):
        zrep[0] = float(elev)
        flags |= Q_STNELEV
    anchors = [i for i, z in enumerate(zrep) if z is not None]
    if not anchors:
        return None
    z = [0.0] * len(plv)
    for i in range(len(plv)):
        if zrep[i] is not None:
            z[i] = zrep[i]
            continue
        flags |= Q_HYPSO
        lo = max((k for k in anchors if k < i), default=None)
        hi = min((k for k in anchors if k > i), default=None)
        if lo is not None and hi is not None and (H[hi] > H[lo]):
            z[i] = zrep[lo] + (H[i] - H[lo]) * (zrep[hi] - zrep[lo]) / (H[hi] - H[lo])
        else:
            a = lo if lo is not None else hi
            z[i] = zrep[a] + (H[i] - H[a])
    holes_t = [zz for lv, zz in zip(plv, z) if lv.t is None and (lv.lt1 == 1 or 't' in lv.removed)]
    thermo, zlast = ([], -1000000000.0)
    for lv, zz in zip(plv, z):
        if lv.t is None or zz <= zlast + 0.5:
            continue
        thermo.append([zz, lv.p, lv.t, _td(P, lv)])
        zlast = zz
    if len(thermo) < 2:
        return None
    zs = thermo[0][0]

    def z_of_p(p):
        ln = math.log(p)
        for i in range(1, len(plv)):
            if lnp[i - 1] >= ln >= lnp[i]:
                f = (lnp[i - 1] - ln) / (lnp[i - 1] - lnp[i]) if lnp[i - 1] != lnp[i] else 0
                return z[i - 1] + f * (z[i] - z[i - 1])
        return None
    wind, holes_w = ([], [])
    for lv in sd.levels:
        zz = lv.z if lv.z is not None else z_of_p(lv.p) if lv.p else None
        if lv.wdir is None or lv.wspd is None or (not 0 <= lv.wdir <= 360):
            if zz is not None and (not lv.surface) and (lv.lt1 == 1 or lv.removed & {'wdir', 'wspd'}):
                holes_w.append(zz)
            continue
        if lv.surface:
            zz = zs + 10.0 if sfc is not None else lv.z + 10.0 if lv.z is not None else None
        if zz is None or zz < zs + 10.0:
            continue
        a = math.radians(lv.wdir)
        wind.append([zz, -lv.wspd * math.sin(a), -lv.wspd * math.cos(a)])
    wind.sort(key=lambda r: r[0])
    wu, zl = ([], -1000000000.0)
    for r in wind:
        if r[0] > zl + 0.5:
            wu.append(r)
            zl = r[0]
    s = thermo[0]
    sw = next((r for r in wu if r[0] == zs + 10.0), None) if sfc is not None else None
    return {'thermo': thermo, 'wind': wu, 'zs': zs, 'ps': s[1], 't2': s[2], 'td2': s[3], 'u10': None if sw is None else sw[1], 'v10': None if sw is None else sw[2], 'flags': flags, 'holes_t': holes_t, 'holes_w': holes_w}

def _profile(P, col: dict):
    import numpy as np
    f32 = np.float32
    th = col['thermo']
    zs = col['zs']
    z = np.array([zs + 2.0] + [r[0] for r in th[1:]], f32)
    keep = np.concatenate([[True], z[1:] > zs + 2.0 + 0.5])
    rows = [r for r, k in zip(th, keep) if k]
    z = z[keep]
    prof = P.Profile.__new__(P.Profile)
    prof.z = z[:, None, None]
    prof.lnp = np.log(np.array([r[1] for r in rows], f32))[:, None, None]
    prof.t = np.array([r[2] for r in rows], f32)[:, None, None]
    prof.td = np.array([np.nan if r[3] is None else min(r[3], r[2]) for r in rows], f32)[:, None, None]
    prof.zs = np.full((1, 1), zs, f32)
    prof.ztop = prof.z[-1]
    w = col['wind']
    if len(w) >= 2:
        prof.zw = np.array([r[0] for r in w], f32)[:, None, None]
        prof.u = np.array([r[1] for r in w], f32)[:, None, None]
        prof.v = np.array([r[2] for r in w], f32)[:, None, None]
    else:
        prof.zw = prof.z.copy()
        prof.u = np.full(prof.z.shape, np.nan, f32)
        prof.v = np.full(prof.z.shape, np.nan, f32)
    prof.native = False
    return prof

def _dense(zlev, zs: float) -> bool:
    z = [x for x in zlev if x <= zs + 10000.0]
    steps = sorted((b - a for a, b in zip(z, z[1:])))
    return bool(steps) and steps[len(steps) // 2] < DENSE_M

def _span(zlev, zq: float, zs: float, dense: bool=True, holes=()) -> int:
    if not len(zlev) or zq > zlev[-1] + 0.01 or zq < min(zlev[0], zs) - 0.01:
        return Q_TOP
    h = max(zq - zs, 0.0)
    if zq <= zlev[0]:
        return Q_GAP if zlev[0] - zs > max(GAP_M, GAP_SHARE * h) else 0
    widest = max(GAP_M if dense else GAP_SIG_M, GAP_SHARE * h)
    for a, b in zip(zlev, zlev[1:]):
        if a <= zq <= b:
            if b - a > widest or (b - a > HOLE_MIN_M and any((a < x < b for x in holes))):
                return Q_GAP
            return 0
    return 0

def derive(sd: Sounding, elev: float | None=None) -> dict | None:
    import numpy as np
    tv = sd.t_valid
    col = column(sd, elev) if tv is not None else None
    if col is None:
        return None
    P = _physics()
    prof = _profile(P, col)
    zs = float(col['zs'])
    zt_ = prof.z[:, 0, 0].astype(float)
    zw_ = prof.zw[:, 0, 0].astype(float) if np.isfinite(prof.u).any() else np.array([])
    base = col['flags']
    dense_t, dense_w = (_dense(zt_, zs), _dense(zw_, zs))
    holes_t, holes_w = (col['holes_t'], col['holes_w'])
    row = {'kind': 'raob', 'obs_id': sd.id, 't_valid': tv, 't_release': sd.t_release, 'lat': sd.lat, 'lon': sd.lon, 'elev': round(zs, 1), 'n_lev': len(sd.levels), 'top_hpa': round(min((r[1] for r in col['thermo'])), 1)}

    def put(name, value, bits=0, nd=1):
        ok = value is not None and (not bits & Q_MISSING) and math.isfinite(value)
        row[f'o_{name}'] = round(float(value), nd) if ok else None
        row[f'q_{name}'] = base | bits | (0 if ok or bits & Q_MISSING else Q_NODATA)

    def at(zq, name):
        wind = name in ('u', 'v')
        bits = _span(zw_, zq, zs, dense_w, holes_w) if wind else _span(zt_, zq, zs, dense_t, holes_t)
        if bits:
            return (None, bits)
        v = float(prof.at(np.full((1, 1), zq, np.float32), 'lnp' if name == 'p' else name)[0, 0])
        return (v, 0) if math.isfinite(v) else (None, Q_NODATA)
    zsa = np.full((1, 1), zs, np.float32)
    t2, td2, ps = (col['t2'], col['td2'], col['ps'])
    put('t2', t2)
    put('td2', td2)
    put('ps', ps)
    put('u_sfc', col['u10'])
    put('v_sfc', col['v10'])

    def zi_of(excess):
        d = float(P.parcel_ml_depth(prof, np.float32(excess))[0, 0])
        if not math.isfinite(d):
            return (None, Q_NODATA)
        if d >= float(prof.ztop[0, 0]) - zs - 0.5:
            return (None, Q_TOP)
        return (d, _span(zt_, zs + d, zs, dense_t, holes_t))
    zi, zb = zi_of(EXCESS_HEAT)
    lcl = None if td2 is None else max(0.0, LCL_PER_C * (t2 - min(td2, t2)))
    cu = 0
    if zi is not None and (not zb) and (lcl is not None):
        if zi > lcl:
            cu = Q_REACH
        elif lcl < zi + max(CU_OVERSHOOT * zi, CU_OVERSHOOT_MIN):
            cu = Q_NEARCU
    put('zi', zi, zb, 0)
    put('zi_blue', None if zi is None or zb or lcl is None else min(zi, lcl + BLUE_OVER_LCL), zb or (0 if lcl is not None else Q_NODATA), 0)
    put('zi_cool', *zi_of(EXCESS_COOL), nd=0)
    put('lcl', lcl, 0, 0)
    depth = None if zi is None or zb or lcl is None else min(zi, lcl)
    put('depth', depth, zb or Q_NODATA if zi is None or zb else cu if lcl is not None else Q_NODATA, 0)
    if zi is None or zb:
        for name in ('blapse', 'thml', 'thx', *(f'{k}_m{dz}' for dz in ML_TOP_DZ for k in ('dth', 'q')), 'u_blavg', 'v_blavg', 'u_bltop', 'v_bltop'):
            put(name, None, zb or Q_NODATA)
    else:
        z1, z2 = (zs + 2 + 0.3 * zi, zs + 2 + 0.9 * zi)
        (ta, ba), (tb, bb) = (at(z1, 't'), at(z2, 't'))
        if zi < 300:
            put('blapse', None, Q_NODATA)
        elif ba or bb:
            put('blapse', None, ba | bb)
        else:
            put('blapse', min(max((ta - tb) / max(z2 - z1, 1) * 1000, 0.0), BLAPSE_MAX), 0, 2)
        (tm, bm), (pm, bp) = (at(zs + 2 + 0.5 * zi, 't'), at(zs + 2 + 0.5 * zi, 'p'))
        thml = None if bm or bp else float(P.theta(tm, pm))
        put('thml', thml, bm | bp, 2)
        put('thx', None if thml is None else min(max(float(P.theta(t2, ps)) - thml, 0.0), THX_MAX), bm | bp, 2)
        ztop = zs + 2 + zi
        for dz in ML_TOP_DZ:
            (t, b1), (p, b2), (td, b3) = (at(ztop + dz, 't'), at(ztop + dz, 'p'), at(ztop + dz, 'td'))
            b = b1 | b2 | bm | bp
            put(f'dth_m{dz}', None if b else float(P.theta(t, p)) - thml, b, 2)
            bq = b | b3
            put(f'q_m{dz}', None if bq else float(P.mixing_ratio(min(td, t), p)) * 1000, bq, 2)
        d = max(depth if depth is not None else zi, 100.0)
        hs = [zs + 2 + d * i / 7 for i in range(8)]
        wb = 0
        for h in hs:
            wb |= _span(zw_, h, zs, dense_w, holes_w)
        if wb:
            for name in ('u_blavg', 'v_blavg', 'u_bltop', 'v_bltop'):
                put(name, None, wb)
        else:
            ub, vb = P.layer_mean(prof, zsa + 2, zsa + 2 + np.float32(d), ['u', 'v'])
            put('u_blavg', float(ub[0, 0]))
            put('v_blavg', float(vb[0, 0]))
            (ut, b1), (vt, b2) = (at(zs + 2 + d, 'u'), at(zs + 2 + d, 'v'))
            put('u_bltop', ut, b1)
            put('v_bltop', vt, b2)
    FT = P.FT
    for ft in WAVE_ALTS_FT:
        zq = ft * FT
        if zq <= zs + 100:
            put(f't{ft}', None, Q_NODATA)
            put(f'dpd{ft}', None, Q_NODATA)
            continue
        (t, b1), (td, b2) = (at(zq, 't'), at(zq, 'td'))
        put(f't{ft}', t, b1)
        put(f'dpd{ft}', None if b1 | b2 else max(0.0, t - td), b1 | b2)
    for ft in WIND_LOW_FT + WIND_ALTS_FT:
        zq = ft * FT
        if zq < zs + 30:
            put(f'u_a{ft}', None, Q_NODATA)
            put(f'v_a{ft}', None, Q_NODATA)
            continue
        (u, b1), (v, b2) = (at(zq, 'u'), at(zq, 'v'))
        put(f'u_a{ft}', u, b1)
        put(f'v_a{ft}', v, b2)
    return row

def rows(soundings: list[Sounding], elev: float | None=None) -> list[dict]:
    return [r for r in (derive(s, elev) for s in soundings) if r is not None]

def rows_for_day(lk, sid: str, day: str, elev: float | None=None) -> list[dict]:
    return rows(read_day(lk, sid, day), elev)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IGRA2 soundings: keep them in the lake, or read a file's rows")
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('ingest')
    a1.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    a1.add_argument('--every', type=int, default=RAOB_EVERY)
    a1.add_argument('--max', type=int, default=MAX_FETCH)
    a1.add_argument('--day', help='today, YYYY-MM-DD (default: today, UTC)')
    a2 = sub.add_parser('derive')
    a2.add_argument('file')
    a2.add_argument('--elev', type=float)
    a = ap.parse_args(argv)
    if a.cmd == 'derive':
        with open(a.file, encoding='ascii', errors='replace') as f:
            for r in rows(parse(f.read())[0], a.elev):
                print(json.dumps(r))
        return 0
    lk = lakemod.from_env('LAKE', a.local)
    if lk is None:
        print('obs_raob: no lake (LAKE_S3_* not set): nothing fetched', flush=True)
        return 0
    today = date.fromisoformat(a.day) if a.day else datetime.now(timezone.utc).date()
    n = ingest(lk, Igra(budget=a.max), today, a.every, a.max)
    print('obs_raob: ' + ', '.join((f'{k} {v}' for k, v in sorted(n.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
