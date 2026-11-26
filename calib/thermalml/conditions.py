from __future__ import annotations
import io
import json
import math
import os
import re
import sys
import zipfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
COND_V = 'c3'
MODELS = ('hrrr', 'icond2', 'icon', 'gfs')
FIELDS = ('wstar', 'zi', 'depth', 'cucover', 'u_blavg', 'v_blavg', 'snowd', 'soilm', 'heat')
CU_COVER = 0.05
BLUE_W = 1.0
REGIMES = ('cu', 'blue', 'weak')

def _pipeline():
    import common
    if common.PIPELINE not in sys.path:
        sys.path.append(common.PIPELINE)

def regions() -> dict:
    _pipeline()
    import build
    return {m: tuple(build.REGIONS[m][:4]) for m in MODELS if m in build.REGIONS}

def inside(reg, la: float, lo: float) -> bool:
    w, s, e, n = reg
    return s <= la <= n and w <= lo <= e

def regime_of(wstar, cucover) -> str | None:
    if wstar is None or not math.isfinite(wstar):
        return None
    if cucover is not None and math.isfinite(cucover) and (cucover >= CU_COVER):
        return 'cu'
    return 'blue' if wstar >= BLUE_W else 'weak'

def _npy_reader(zf: zipfile.ZipFile, name: str):
    import numpy as np
    f = zf.open(f'{name}.npy')
    version = np.lib.format.read_magic(f)
    read = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(tuple(version))
    if read is None:
        raise ValueError('conditions: an .npy version this does not read')
    shape, fortran, dtype = read(f)
    if fortran:
        raise ValueError('conditions: a Fortran-ordered array')
    return (f, shape, dtype)

def sample_slim(blob: bytes, wants: dict) -> dict:
    import numpy as np
    _pipeline()
    import sky
    out = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        small = {}
        for n in ('stamps', 'fields', 'enc', 'grid'):
            with zf.open(f'{n}.npy') as f:
                small[n] = np.load(f, allow_pickle=False)
        stamps = [str(s) for s in small['stamps']]
        fields = [str(x) for x in small['fields']]
        enc = json.loads(str(small['enc']))
        g = json.loads(str(small['grid']))
        if not any((s in wants for s in stamps)):
            return out
        f, shape, dtype = _npy_reader(zf, 'codes')
        nh, nf, ny, nx = shape
        per = nf * ny * nx * np.dtype(dtype).itemsize
        fi = {k: fields.index(k) for k in FIELDS if k in fields}
        for h in range(nh):
            raw = f.read(per)
            if len(raw) < per:
                break
            pts = wants.get(stamps[h])
            if not pts:
                continue
            a = np.frombuffer(raw, dtype).reshape(nf, ny, nx)
            for key, la, lo in pts:
                iy = int(round((g['north'] - la) / g['step']))
                ix = int(round((lo - g['west']) / g['step']))
                if not (0 <= iy < ny and 0 <= ix < nx):
                    continue
                vals = {}
                for k, j in fi.items():
                    c = int(a[j, iy, ix])
                    vals[k] = float('nan') if c == 255 else float(sky.decode(np.array([c], np.uint8), *_enc(enc[k]))[0])
                out[key] = vals
        f.close()
    return out

def _enc(e):
    e = list(e)
    return (e + [None])[:3] if len(e) < 3 else e[:3]
_DAY_KEY = re.compile('ops/[a-z0-9]+/\\d{8}/\\d{2}/b[A-Za-z0-9._-]+/slim\\.npz')

def day_grids(ops, model: str, ymd: str) -> str | None:
    raw = ops.get(f'ops/{model}/grids/{ymd}.json')
    if not raw:
        return None
    try:
        key = json.loads(raw).get('key') or ''
    except ValueError:
        return None
    return key if _DAY_KEY.fullmatch(key) else None

def describe(v: dict | None) -> dict | None:
    if not v:
        return None
    nan = float('nan')
    w = v.get('wstar', nan)
    u, vv = (v.get('u_blavg', nan), v.get('v_blavg', nan))
    ws = math.hypot(u, vv) if math.isfinite(u) and math.isfinite(vv) else nan
    wd = (math.degrees(math.atan2(-u, -vv)) + 360) % 360 if math.isfinite(ws) else nan
    reg = regime_of(w, v.get('cucover'))
    if reg is None:
        return None
    clean = lambda x: round(x, 3) if math.isfinite(x) else None
    return {'regime': reg, 'wstar': clean(w), 'zi': clean(v.get('zi', nan)), 'depth': clean(v.get('depth', nan)), 'cucover': clean(v.get('cucover', nan)), 'wind': clean(ws), 'wind_from': clean(wd), 'snow': clean(v.get('snowd', nan)), 'soil': clean(v.get('soilm', nan)), 'heat': clean(v.get('heat', nan))}

def at_points(ops, points: dict, counts: dict | None=None) -> dict:
    counts = counts if counts is not None else {}
    out = {k: None for k in points}
    if ops is None or not points:
        counts['cond_no_ops'] = counts.get('cond_no_ops', 0) + (1 if ops is None else 0)
        return out
    regs = regions()
    from datetime import datetime, timezone
    left = dict(points)
    for m in MODELS:
        if not left or m not in regs:
            continue
        by_day = {}
        for k, (la, lo, hr) in left.items():
            if inside(regs[m], la, lo):
                t = datetime.fromtimestamp(int(hr) * 3600, timezone.utc)
                by_day.setdefault(f'{t:%Y%m%d}', {}).setdefault(f'{t:%Y%m%d%H}', []).append((k, la, lo))
        for ymd, wants in sorted(by_day.items()):
            key = day_grids(ops, m, ymd)
            if not key:
                counts[f'cond_no_grids_{m}'] = counts.get(f'cond_no_grids_{m}', 0) + 1
                continue
            blob = ops.get(key)
            if not blob:
                counts['cond_missing'] = counts.get('cond_missing', 0) + 1
                continue
            try:
                got = sample_slim(blob, wants)
            except (ValueError, KeyError, OSError, zipfile.BadZipFile):
                counts['cond_unreadable'] = counts.get('cond_unreadable', 0) + 1
                continue
            for k, v in got.items():
                d = describe(v)
                if d is not None:
                    out[k] = {**d, 'model': m}
                    left.pop(k, None)
            counts[f'cond_days_{m}'] = counts.get(f'cond_days_{m}', 0) + 1
    counts['cond_points'] = counts.get('cond_points', 0) + sum((1 for v in out.values() if v))
    counts['cond_none'] = counts.get('cond_none', 0) + sum((1 for v in out.values() if not v))
    return out
