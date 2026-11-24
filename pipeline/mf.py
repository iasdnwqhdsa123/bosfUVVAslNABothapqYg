from __future__ import annotations
import fcntl
import http.client
import os
import random
import ssl
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import grib
BASE = 'https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net/pnt'
CONST = 'https://www.data.gouv.fr/fr/datasets/r/2ea15b4c-36e1-47ba-98d3-43c09c9aef65'
LAST_FH = 51
KEEP_S = 12 * 3600
PART = 8 << 20

def group(fh: int) -> tuple[int, int]:
    if fh <= 6:
        return (0, 6)
    lo = (fh - 7) // 6 * 6 + 7
    return (lo, min(lo + 5, LAST_FH))

def url(run, pkg: str, fh: int) -> str:
    lo, hi = group(fh)
    r = f'{run:%Y-%m-%dT%H:%M:%SZ}'
    return f'{BASE}/{r}/arome/0025/{pkg}/arome__0025__{pkg}__{lo:02d}H{hi:02d}H__{r}.grib2'

def folder() -> str:
    d = os.environ.get('MF_CACHE') or os.path.join(tempfile.gettempdir(), 'arome-mf')
    os.makedirs(d, exist_ok=True)
    return d

def _tidy(d: str, keep: str):
    now = time.time()
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if p != keep and (not f.endswith('.lock')) and (now - os.path.getmtime(p) > KEEP_S):
            for q in (p, p + '.lock'):
                try:
                    os.remove(q)
                except OSError:
                    pass

def local(u: str, workers: int=8) -> str:
    import om
    d = folder()
    path = os.path.join(d, u.rsplit('/', 1)[-1] if '/pnt/' in u else u.rsplit('/', 1)[-1] + '.grib2')
    if os.path.exists(path):
        return path
    with open(path + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not os.path.exists(path):
                if '/pnt/' in u:
                    tmp = om.download(u, workers=workers, part=PART, folder=d)
                else:
                    data = grib.http_get(u, timeout=120)
                    fd, tmp = tempfile.mkstemp(dir=d)
                    with os.fdopen(fd, 'wb') as f:
                        f.write(data)
                os.replace(tmp, path)
                _tidy(d, path)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return path
_INDEX: dict = {}
_INDEX_LOCK = threading.Lock()

def index(path: str) -> dict:
    with _INDEX_LOCK:
        if path in _INDEX:
            return _INDEX[path]
    import eccodes
    out = {}
    with open(path, 'rb') as fh:
        while True:
            g = eccodes.codes_grib_new_from_file(fh)
            if g is None:
                break
            try:
                key = (eccodes.codes_get(g, 'shortName'), str(eccodes.codes_get(g, 'stepRange')))
                out.setdefault(key, (int(eccodes.codes_get(g, 'offset')), int(eccodes.codes_get(g, 'totalLength'))))
            finally:
                eccodes.codes_release(g)
    with _INDEX_LOCK:
        _INDEX[path] = out
    return out

def message(path: str, key) -> bytes | None:
    at = index(path).get(key)
    if at is None:
        return None
    with open(path, 'rb') as f:
        f.seek(at[0])
        return f.read(at[1])

class Field:

    def __init__(self, grid, parts):
        self.grid, self.parts = (grid, parts)

    def read(self, region, boxes=None):
        import numpy as np
        i_n, j_w, ny, nx = self.grid.window(region)
        top = self.grid.ny - 1 - i_n
        v = None
        for c, m in self.parts:
            a = grib.decode(m)[0]
            if a.shape != (self.grid.ny, self.grid.nx):
                raise ValueError(f'AROME message on a {a.shape} grid, not {(self.grid.ny, self.grid.nx)}')
            v = c * a if v is None else v + c * a
        v = v[top:top + ny, j_w:j_w + nx]
        if not boxes:
            return v.astype(np.float32)
        out = np.full((ny, nx), np.nan, np.float32)
        for y0, y1, x0, x1 in boxes:
            out[y0:y1, x0:x1] = v[y0:y1, x0:x1]
        return out

def instant(run, pkg, name, fh):
    m = message(local(url(run, pkg, fh)), (name, str(fh)))
    return None if m is None else [(1.0, m)]

def hour_mean(run, pkg, name, fh, sign=1.0):
    now = message(local(url(run, pkg, fh)), (name, f'0-{fh}'))
    if now is None:
        return None
    parts = [(sign / 3600, now)]
    if fh > 1:
        prev = message(local(url(run, pkg, fh - 1)), (name, f'0-{fh - 1}'))
        if prev is None:
            return None
        parts.append((-sign / 3600, prev))
    return parts

def constant(name):
    m = message(local(CONST), (name, '0'))
    return None if m is None else [(1.0, m)]
IP3_FIELDS = ((0, 2, 0), (0, 2, 1), (0, 1, 0), (0, 0, 6), (0, 2, 8), (0, 2, 9), (0, 2, 14))
IP3_SHARE = (0.11, 0.127, 0.116, 0.114, 0.158, 0.204, 0.171)
IP3_LEVELS = (100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 1000)
WZ = (0, 2, 9)
HEAD = 256
PROBE = 5 << 18
HOPS = 3
PROBES = 8
_IP3: dict = {}
_IP3_LOCK = threading.Lock()

def head(b: bytes):
    if len(b) < 16 or b[:4] != b'GRIB' or b[7] != 2:
        return None
    total = int.from_bytes(b[8:16], 'big')
    if not 16 < total < 1 << 31:
        return None
    p, last = (16, 0)
    while p + 5 <= len(b):
        n, sec = (int.from_bytes(b[p:p + 4], 'big'), b[p + 4])
        if n < 5 or not last < sec <= 7:
            return None
        if sec == 4:
            s = b[p:p + 34]
            if len(s) < 28:
                return None
            if int.from_bytes(s[7:9], 'big') != 0 or s[17] != 1 or s[22] != 100:
                return (None, total)
            pa = int.from_bytes(s[24:28], 'big') / 10 ** s[23]
            return ((b[6], s[9], s[10], int.from_bytes(s[18:22], 'big'), round(pa / 100)), total)
        if sec > 4:
            return (None, total)
        p, last = (p + n, sec)
    return None

def ip3_index(key, lo: int, nh: int) -> int:
    d, c, n, h, p = key
    if not lo <= h < lo + nh:
        raise ValueError(f'hour {h} not in {lo}–{lo + nh - 1}')
    return (IP3_FIELDS.index((d, c, n)) * nh + h - lo) * len(IP3_LEVELS) + IP3_LEVELS.index(p)

def _share(i: int, nh: int) -> float:
    per = nh * len(IP3_LEVELS)
    f, r = divmod(i, per)
    if f >= len(IP3_FIELDS):
        return 1.0
    return (sum(IP3_SHARE[:f]) + IP3_SHARE[f] * r / per) / sum(IP3_SHARE)

class _Ranges:

    def __init__(self, u: str, tries: int=4, timeout: float=30):
        s = urllib.parse.urlsplit(u)
        self.url, self.host, self.path, self.tries, self.timeout = (u, s.hostname, s.path, tries, timeout)
        self.conn, self.size, self.requests, self.bytes = (None, None, 0, 0)

    def _connect(self):
        ctx = ssl.create_default_context()
        proxy = urllib.request.getproxies().get('https')
        if proxy and (not urllib.request.proxy_bypass(self.host)):
            p = urllib.parse.urlsplit(proxy if '//' in proxy else 'http://' + proxy)
            c = http.client.HTTPSConnection(p.hostname, p.port or 80, timeout=self.timeout, context=ctx)
            c.set_tunnel(self.host, 443)
            return c
        return http.client.HTTPSConnection(self.host, 443, timeout=self.timeout, context=ctx)

    def get(self, a: int, n: int) -> bytes:
        for attempt in range(self.tries):
            try:
                if self.conn is None:
                    self.conn = self._connect()
                self.conn.request('GET', self.path, headers={**grib.UA, 'Range': f'bytes={a}-{a + n - 1}'})
                r = self.conn.getresponse()
                body = r.read()
                if r.status in (403, 404):
                    raise FileNotFoundError(f'HTTP {r.status}: {self.url}')
                if r.status != 206:
                    raise OSError(f'HTTP {r.status} for bytes {a}–{a + n - 1}')
                total = r.getheader('Content-Range', '').rsplit('/', 1)[-1]
                if total.isdigit():
                    self.size = int(total)
                self.requests, self.bytes = (self.requests + 1, self.bytes + len(body))
                return body
            except FileNotFoundError:
                self.close()
                raise
            except Exception:
                self.close()
                if attempt == self.tries - 1:
                    raise
                time.sleep(0.5 * 2 ** attempt + random.random() / 2)

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

def _fits(known: dict, i: int, at: int, n: int) -> bool:
    below = max((j for j in known if j < i), default=None)
    above = min((j for j in known if j > i), default=None)
    if below is not None:
        end = sum(known[below])
        if end != at if below == i - 1 else end >= at:
            return False
    if above is not None:
        o = known[above][0]
        if at + n != o if above == i + 1 else at + n >= o:
            return False
    return True

def _take(known: dict, i: int, at: int, n: int) -> bool:
    if i in known:
        if known[i] != (at, n):
            raise OSError(f'IP3 not as expected: message {i} at {at}, not {known[i][0]}')
        return False
    if not _fits(known, i, at, n):
        raise OSError(f'IP3 not as expected: message {i} at {at}, out of order')
    known[i] = (at, n)
    return True

def _scan(buf: bytes, at: int, lo: int, nh: int, known: dict) -> int:
    k, new = (buf.find(b'GRIB'), 0)
    while k >= 0:
        r = head(buf[k:k + HEAD])
        if r is not None and r[0] is not None:
            try:
                i = ip3_index(r[0], lo, nh)
            except ValueError:
                raise OSError(f'IP3 not as expected: {r[0]} at {at + k}') from None
            new += _take(known, i, at + k, r[1])
            if k + r[1] < len(buf):
                k = buf.find(b'GRIB', k + r[1])
                continue
        k = buf.find(b'GRIB', k + 4)
    return new

def _find(rd: _Ranges, t: int, lo: int, nh: int, known: dict) -> tuple[int, int]:
    n_all = len(IP3_FIELDS) * nh * len(IP3_LEVELS)
    back = 0
    for _ in range(PROBES):
        if t in known:
            return known[t]
        a = max((i for i in known if i < t), default=None)
        if a is not None and t - a <= HOPS:
            for i in range(a + 1, t + 1):
                o = sum(known[i - 1])
                r = head(rd.get(o, HEAD))
                if r is None or r[0] is None or ip3_index(r[0], lo, nh) != i:
                    raise OSError(f'IP3 not as expected: {r and r[0]} at {o}, not message {i}')
                _take(known, i, o, r[1])
            return known[t]
        a_i, a_o = (a + 1, sum(known[a])) if a is not None else (0, 0)
        b = min((i for i in known if i > t), default=None)
        b_i, b_o = (b, known[b][0]) if b is not None else (n_all, rd.size)
        sa, sb = (_share(a_i, nh), _share(b_i, nh))
        est = a_o + (_share(t, nh) - sa) / max(sb - sa, 1e-12) * (b_o - a_o)
        mean = (b_o - a_o) / max(b_i - a_i, 1)
        at = int(min(max(a_o, est - 0.45 * mean - back), max(a_o, rd.size - HEAD)))
        back = 0 if _scan(rd.get(at, PROBE), at, lo, nh, known) else back + PROBE
    raise OSError(f'IP3 message {t} not found in {PROBES} reads')

def _download(u: str, at, key) -> bytes:
    m = grib.http_get(u, (at[0], at[0] + at[1] - 1), tries=4, timeout=60)
    r = head(m[:HEAD])
    if len(m) != at[1] or m[-4:] != b'7777' or r is None or (r[0] != key):
        raise OSError(f'IP3 {key} at {at[0]}: not that message')
    return m

def isobaric(run, fh: int, field, levels, workers: int=4) -> dict:
    u = url(run, 'IP3', fh)
    with _IP3_LOCK:
        cached = u in _IP3
    try:
        return _isobaric(u, fh, field, levels, workers)
    except OSError:
        if not cached:
            raise
        return _isobaric(u, fh, field, levels, workers)

def _isobaric(u: str, fh: int, field, levels, workers: int) -> dict:
    lo, hi = group(fh)
    nh = hi - lo + 1
    want = sorted(((ip3_index(tuple(field) + (fh, p), lo, nh), p) for p in levels))
    with _IP3_LOCK:
        size, known = _IP3.get(u, (None, {}))
        known = dict(known)
    rd = _Ranges(u)
    rd.size = size
    try:
        with ThreadPoolExecutor(workers) as pool:
            if size is None:
                r = head(rd.get(0, HEAD))
                if r is None or r[0] is None or ip3_index(r[0], lo, nh) != 0 or (rd.size is None):
                    raise OSError(f"IP3's first message not as expected: {r and r[0]}")
                _take(known, 0, 0, r[1])
            futs = {p: pool.submit(_download, u, _find(rd, i, lo, nh, known), tuple(field) + (fh, p)) for i, p in want}
            out = {p: f.result() for p, f in futs.items()}
    except Exception:
        with _IP3_LOCK:
            _IP3.pop(u, None)
        raise
    finally:
        rd.close()
    with _IP3_LOCK:
        _IP3[u] = (rd.size, {**_IP3.get(u, (None, {}))[1], **known})
    return out
