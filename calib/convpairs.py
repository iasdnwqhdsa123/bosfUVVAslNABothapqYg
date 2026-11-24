from __future__ import annotations
import argparse
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog
import common
import lake as lakem
import linecheck as lc
PREFIX = 'derived/lines/p1/'
VERSION = 1
NUM = ('t0', 't1', 'lon', 'lat', 'netto', 'va', 'km', 'agl', 'hdg', 'conv', 'wind', 'wdir', 'ws', 'zi', 'ground', 't2', 'd0.15', 'best0.15', 'dh0.15', 'd0.3', 'best0.3', 'dh0.3', 'wblmax')
KINDS = {'straight': 0, 'line': 1, 'sink': 2}

def object_key(lk, r: dict) -> str:
    return PREFIX + lk.name(f"lines|{r['key']}|{r.get('sha256') or ''}")

def box_from_unit(z: dict) -> dict | None:
    meta = z.get('meta') or {}
    if not meta.get('grids'):
        return None
    enc = meta.get('enc') or {}
    stamps = [datetime.strptime(s[:10], '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp() for s in meta.get('stamps') or []]
    names = tuple(dict.fromkeys(lc.FIELDS + lc.ROLL_FIELDS))
    f = {}
    shape = (len(stamps), meta['grid']['ny'], meta['grid']['nx'])
    for n in names:
        key = f"g.{common.FIELD_GROUP.get(n, '')}.{n}"
        e = enc.get(n)
        if key in z and e:
            f[n] = lc.decode({'min': e[0], 'max': e[1], 'transform': e[2]}, z[key])
        else:
            f[n] = np.full(shape, np.nan, np.float32)
    has = np.isfinite(f['conv']).any()
    return {'g': meta['grid'], 'stamps': np.array(stamps), 'f': f, 'conv': bool(has), 'full': all((np.isfinite(f[n]).any() for n in lc.ROLL_FULL))}

def piece_wblmax(r: dict, B) -> float:
    h = lc.hour(B, (r['t0'] + r['t1']) / 2)
    if h is None:
        return float('nan')
    vals = []
    for lon, lat in r['pts'][::max(1, len(r['pts']) // 8)]:
        ij = lc.cell(B, lon, lat)
        if ij is not None:
            vals.append(float(B['f']['wblmax'][h][ij]))
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else float('nan')

def unit_pieces(z: dict, flights: list, model: str, cls_of: dict, log=print) -> list[dict]:
    B = box_from_unit(z)
    if B is None or not B['conv']:
        return []
    out = []
    for f in flights:
        if not f.get('track'):
            continue
        cls = cls_of.get(f.get('ac')) or '15m'
        try:
            rs = lc.runs(f, B, cls)
        except Exception:
            continue
        for r in rs:
            j = lc.judge(r, B, model)
            j['wblmax'] = piece_wblmax(r, B)
            j['region'] = f.get('region')
            out.append(j)
    return out

def columns(pieces: list[dict], model: str, step: float) -> dict:
    n = len(pieces)
    cols = {k: np.array([np.nan if p.get(k) is None else float(p[k]) for p in pieces], np.float64) for k in NUM}
    cols['kind'] = np.array([KINDS.get(p.get('kind'), -1) for p in pieces], np.int8)
    cols['region'] = np.array([str(p.get('region') or '') for p in pieces])
    cols['model'] = np.full(n, model)
    cols['step'] = np.full(n, float(step))
    return cols

def pack(cols: dict) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, meta=np.array(json.dumps({'v': VERSION})), **cols)
    return bio.getvalue()

def unpack(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != 'meta'}

def load_all(lk, counts=None) -> dict:
    cols = {}
    for k in lk.list(PREFIX):
        c = unpack(lk.get(k))
        if counts is not None:
            counts['units'] = counts.get('units', 0) + 1
        if not len(c.get('netto', [])):
            continue
        for name, v in c.items():
            cols.setdefault(name, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items() if len(v) == len(cols.get('netto', []))} if cols else {}

def flight_rows(snap: dict | None) -> list[dict]:
    if not snap:
        return []
    rs = [r for r in catalog.usable(snap) if r.get('kind') == 'flight' and r.get('key') and r.get('day')]
    return sorted(rs, key=lambda r: str(r['day']), reverse=True)

def run(lk, minutes: float=55, read_unit=None, read_items=None, boxes=None, clock=None, log=print) -> dict:
    import time
    clock = clock or time.monotonic
    if read_unit is None:
        import labbuild
        read_unit = labbuild.read_unit
    read_items = read_items or catalog.read_items
    if boxes is None:
        import day as dayb
        boxes = dayb.boxes
    import rows as rowsmod
    cls_of = lc.rc.classes()
    snap = catalog.pinned(lk)
    raw = (snap or {}).get('raw') or {}
    t_end = clock() + minutes * 60
    counts = {}
    for r in flight_rows(snap):
        if clock() > t_end:
            counts['left'] = counts.get('left', 0) + 1
            continue
        key = object_key(lk, r)
        if lk.exists(key):
            counts['done_before'] = counts.get('done_before', 0) + 1
            continue
        parts = (raw.get(r['day']) or {}).get('parts') or []
        try:
            z = read_unit(lk.get(r['key']))
            items = [f for f in read_items(lk, parts) if f.get('kind') != 'collector']
        except (lakem.LakeMissing, ValueError, KeyError):
            counts['unreadable'] = counts.get('unreadable', 0) + 1
            continue
        B = next((b for b in boxes(items) if b.get('region') == r.get('box')), None)
        flights = [f for f in (B or {}).get('flights') or [] if (f.get('kind') or 'flight') == 'flight']
        meta = z.get('meta') or {}
        pieces = unit_pieces(z, flights, str(r.get('model') or ''), cls_of, log)
        cols = columns(pieces, str(r.get('model') or ''), float((meta.get('grid') or {}).get('step') or np.nan))
        lk.put(key, pack(cols), strict=False)
        counts['units'] = counts.get('units', 0) + 1
        counts['pieces'] = counts.get('pieces', 0) + len(pieces)
    return counts

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The convergence pairs: straight flight against the forecast's convergence")
    ap.add_argument('--minutes', type=float, default=55)
    ap.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    x = ap.parse_args(argv)
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('lines: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    try:
        counts = run(lk, x.minutes)
    except lakem.LakeError:
        print('lines: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('lines: ' + (', '.join((f'{k} {v}' for k, v in sorted(counts.items()))) or 'nothing to do'), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
