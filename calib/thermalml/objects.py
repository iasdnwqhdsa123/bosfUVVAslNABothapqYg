from __future__ import annotations
import gzip
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
RAW = 'raw/collector/'
DERIVED = 'derived/thermalml/'
FIRST_DAY = '2026-11-16'
GRACE_DAYS = 3
_RAW_RE = re.compile('^raw/collector/v(\\d{1,3})/(\\d{4}-\\d{2}-\\d{2})\\.lenc$')
DAY_RE = re.compile('^\\d{4}-\\d{2}-\\d{2}$')

def utc_day(t: float) -> str:
    return datetime.fromtimestamp(int(t // 1), timezone.utc).date().isoformat()

def solar_day(t: float, lon: float) -> str:
    return utc_day(t + lon * 240.0)

def shift(day: str, n: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()

def key(stage: str, version: str, sd: str, missing=()) -> str:
    if not DAY_RE.match(sd) or not re.match('^[a-z0-9-]{1,32}$', version) or stage not in ('events', 'exposure', 'join') or (not all((isinstance(d, str) and DAY_RE.match(d) for d in missing))):
        raise ValueError('thermalml: a malformed derived key')
    tail = '.without.' + '_'.join((d.replace('-', '') for d in sorted(missing))) if missing else ''
    return f'{DERIVED}{stage}/{version}/{sd}{tail}.lenc'
_DERIVED_RE = re.compile('/(\\d{4}-\\d{2}-\\d{2})(?:\\.without\\.(\\d{8}(?:_\\d{8})*))?\\.lenc$')

def find(lk, stage: str, version: str, sd: str) -> str | None:
    k = key(stage, version, sd)
    if lk.exists(k):
        return k
    best = None
    for x in lk.list(f'{DERIVED}{stage}/{version}/{sd}.without.'):
        m = _DERIVED_RE.search(x)
        if m and m.group(1) == sd and m.group(2):
            gone = m.group(2).split('_')
            rank = (len(gone), [-int(d) for d in gone])
            if best is None or rank < best[0]:
                best = (rank, x)
    return best[1] if best else None

def raw_days(lk) -> dict:
    out = {}
    for k in lk.list(RAW):
        m = _RAW_RE.match(k)
        if m:
            out.setdefault(m.group(2), []).append((int(m.group(1)), k))
    return {d: [k for _, k in sorted(v)] for d, v in sorted(out.items())}

def needs(sd: str) -> list[str]:
    return [shift(sd, -1), sd, shift(sd, 1)]

def ready(sd: str, have) -> tuple[bool, list[str]]:
    have = set(have)
    want = [d for d in needs(sd) if d >= FIRST_DAY]
    gone = [d for d in want if d not in have]
    if sd not in have:
        return (False, gone)
    if not gone:
        return (True, [])
    later = sum((1 for d in have if d > shift(sd, 1)))
    return (later >= GRACE_DAYS, gone)

def solar_days(have) -> list[str]:
    out = set()
    for d in have:
        out.update((x for x in (shift(d, -1), d, shift(d, 1)) if x >= FIRST_DAY))
    return sorted(out)

def read_raw(lk, raw_key: str) -> bytes:
    b = lk.get(raw_key)
    return gzip.decompress(b) if b[:2] == b'\x1f\x8b' else b

def evict(cache: dict, sd: str) -> None:
    for k in [k for k in cache if _RAW_RE.match(k) and _RAW_RE.match(k).group(2) < shift(sd, -1)]:
        del cache[k]

def dumps(obj) -> bytes:
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as f:
        f.write(text)
    return buf.getvalue()

def loads(blob: bytes):
    return json.loads(gzip.decompress(blob))

def pack_columns(meta: dict, cols: dict) -> bytes:
    import numpy as np
    names = sorted(cols)
    arrs = [np.ascontiguousarray(cols[n]) for n in names]
    head = {'meta': meta, 'cols': [[n, a.dtype.newbyteorder('<').str, int(a.size)] for n, a in zip(names, arrs)]}
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as f:
        f.write(json.dumps(head, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n')
        for a in arrs:
            f.write(a.astype(a.dtype.newbyteorder('<'), copy=False).tobytes())
    return buf.getvalue()

def unpack_columns(blob: bytes) -> tuple[dict, dict]:
    import numpy as np
    raw = gzip.decompress(blob)
    nl = raw.index(b'\n')
    head = json.loads(raw[:nl])
    at, cols = (nl + 1, {})
    for n, dt, size in head['cols']:
        d = np.dtype(dt)
        cols[n] = np.frombuffer(raw, dtype=d, count=size, offset=at).copy()
        at += d.itemsize * size
    if at != len(raw):
        raise ValueError("thermalml: a column object's length is not its header's")
    return (head['meta'], cols)

def put(lk, object_key: str, plain: bytes, n: dict) -> str:
    r = lk.put(object_key, plain, strict=False)
    n[r.status] = n.get(r.status, 0) + 1
    return r.status

def run_days(lk, stage: str, version: str, make, days=None, cache=None, n=None, most: int | None=None) -> dict:
    n = n if n is not None else {}
    raw = raw_days(lk)
    made = 0
    for sd in days or solar_days(raw):
        ok, missing = ready(sd, raw)
        if not ok:
            n['waiting'] = n.get('waiting', 0) + 1
            continue
        k = key(stage, version, sd, missing)
        if not days and lk.exists(k):
            n['there'] = n.get('there', 0) + 1
            continue
        if most is not None and made >= most:
            n['left'] = n.get('left', 0) + 1
            continue
        if not days and (not missing) and lk.list(f'{DERIVED}{stage}/{version}/{sd}.without.'):
            n['remade'] = n.get('remade', 0) + 1
        plain, extra = make(sd, raw, missing)
        put(lk, k, plain, n)
        for x, v in extra.items():
            n[x] = n.get(x, 0) + v
        made += 1
        if cache is not None:
            evict(cache, sd)
    return n
