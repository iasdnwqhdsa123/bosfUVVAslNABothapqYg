from __future__ import annotations
import math
from datetime import datetime, timezone
CIRC_DEG, CIRC_WIN, CIRC_GAP = (220.0, 30.0, 25.0)
CLIMB_GAIN, CLIMB_DUR, CLIMB_TURN = (150.0, 60.0, 270.0)
ENGINE_ON = 500.0
ENGINE_SHARE = 0.2
SIMPLIFY_KM, MIN_PIECE_KM = (1.0, 5.0)
TRACK_S = 20

def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

def km_xy(o):
    k = math.cos(math.radians(o[1]))
    return lambda p: ((p[0] - o[0]) * 111.32 * k, (p[1] - o[1]) * 110.54)

def dist_km(a, b) -> float:
    k = math.cos(math.radians((a[1] + b[1]) / 2))
    return math.hypot((b[0] - a[0]) * 111.32 * k, (b[1] - a[1]) * 110.54)

def bearing(a, b) -> float:
    k = math.cos(math.radians((a[1] + b[1]) / 2))
    return math.degrees(math.atan2((b[0] - a[0]) * k, b[1] - a[1])) % 360

def fixes(data: dict) -> list[dict]:
    coords = (data.get('geom') or {}).get('coordinates') or []
    t, z, gz = (data.get('time') or [], data.get('alt') or [], data.get('ground_alt') or [])
    eng = data.get('engine_sensor')
    n, m = (len(coords), len(t))
    if n < 2 or m < 2 or len(z) != m:
        return []
    anchors = [round(i * (n - 1) / (m - 1)) for i in range(m)]
    out = []
    j = 0
    for k in range(n):
        while j < m - 2 and anchors[j + 1] <= k:
            j += 1
        k0, k1 = (anchors[j], anchors[j + 1])
        w = 0.0 if k1 == k0 else (k - k0) / (k1 - k0)
        w = min(1.0, max(0.0, w))
        lerp = lambda a: None if a is None or a[j] is None or a[j + 1] is None else a[j] + (a[j + 1] - a[j]) * w
        out.append({'t': lerp(t), 'z': lerp(z), 'g': lerp(gz) if len(gz) == m else None, 'e': lerp(eng) if eng and len(eng) == m else None, 'p': coords[k][:2]})
    return [f for f in out if f['t'] is not None and f['z'] is not None]

def climbs(fx: list[dict], legs: list[dict]) -> list[dict]:
    if len(fx) < 10:
        return []
    turn = [0.0] * len(fx)
    last_h, last_p = (None, fx[0]['p'])
    for i in range(1, len(fx)):
        p = fx[i]['p']
        if dist_km(last_p, p) < 0.015:
            continue
        h = bearing(last_p, p)
        if last_h is not None:
            d = (h - last_h + 540) % 360 - 180
            turn[i] = d
        last_h, last_p = (h, p)
    circ = [False] * len(fx)
    j, s = (0, 0.0)
    for i in range(len(fx)):
        s += turn[i]
        while fx[i]['t'] - fx[j]['t'] > CIRC_WIN:
            s -= turn[j]
            j += 1
        if abs(s) >= CIRC_DEG:
            for k in range(j, i + 1):
                circ[k] = True
    periods, i = ([], 0)
    while i < len(fx):
        if not circ[i]:
            i += 1
            continue
        a = i
        while i < len(fx) and circ[i]:
            i += 1
        b = i - 1
        if periods and fx[a]['t'] - fx[periods[-1][1]]['t'] < CIRC_GAP:
            periods[-1][1] = b
        else:
            periods.append([a, b])
    out = []
    for a, b in periods:
        lo = min(range(a, b + 1), key=lambda k: fx[k]['z'])
        hi = max(range(lo, b + 1), key=lambda k: fx[k]['z'])
        z0, z1 = (fx[lo]['z'], fx[hi]['z'])
        gain, dur = (z1 - z0, fx[hi]['t'] - fx[lo]['t'])
        turned = sum((abs(turn[k]) for k in range(lo + 1, hi + 1)))
        if gain < CLIMB_GAIN or dur < CLIMB_DUR or turned < CLIMB_TURN:
            continue
        k100 = next((k for k in range(hi, lo - 1, -1) if fx[k]['z'] <= z1 - 100))
        f100 = next((k for k in range(lo, hi + 1) if fx[k]['z'] >= z0 + 100))
        path = sum((dist_km(fx[k - 1]['p'], fx[k]['p']) for k in range(lo + 1, hi + 1))) * 1000
        es = [fx[k]['e'] for k in range(lo, hi + 1) if fx[k]['e'] is not None]
        top = fx[hi]
        g0, g1 = (fx[lo]['g'], top['g'])
        leg = next((L['n'] for L in legs if ts(L['t0']) <= top['t'] <= ts(L['t1'])), None)
        out.append({'t0': iso(fx[lo]['t']), 'dur': round(dur), 'z0': round(z0), 'z1': round(z1), 'agl0': None if g0 is None else round(z0 - g0), 'agl1': None if g1 is None else round(z1 - g1), 'gain': round(gain), 'rate': round(gain / dur, 2), 'r100': round(100 / max(1.0, top['t'] - fx[k100]['t']), 2), 'rf100': round(100 / max(1.0, fx[f100]['t'] - fx[lo]['t']), 2), 'turn': round(turned), 'radius': round(path / max(1e-06, math.radians(turned))), 'drift': round(dist_km(fx[lo]['p'], top['p']) * 1000), 'lon': round(top['p'][0], 5), 'lat': round(top['p'][1], 5), 'engon': 1 if es and sum((e > ENGINE_ON for e in es)) / len(es) >= ENGINE_SHARE else 0, 'leg': leg})
    return out

def along_kmh(wind: float, wdir: float, hdg: float) -> float:
    return wind * math.cos(math.radians(hdg - (wdir + 180)))

def leg_fields(L: dict, a, b, last: bool) -> dict:
    t0, t1 = (L.get('start_time'), L.get('end_time'))
    t = L.get('duration') or ts(t1) - ts(t0)
    dist = L.get('distance') or 0.0
    z0, z1 = (L.get('start_alt'), L.get('end_alt'))
    dh = z1 - z0 if z0 is not None and z1 is not None else 0
    ca = L.get('thermal_avg') or 0.0
    speed = L.get('speed') or (dist / t * 3600 if t else 0)
    tc = t - dh / ca if ca > 0 else t
    vg = dist / tc * 3600 if tc > 0 and ca > 0 else speed
    wind, wdir = (L.get('wind_speed') or 0.0, L.get('wind_direction') or 0.0)
    hdg = bearing(a, b)
    along = along_kmh(wind, wdir, hdg)
    tt = tc if tc > 0 else t
    gx, gy = (dist * math.sin(math.radians(hdg)), dist * math.cos(math.radians(hdg)))
    drift = wind * tt / 3600
    wx, wy = (drift * math.sin(math.radians(wdir + 180)), drift * math.cos(math.radians(wdir + 180)))
    vair = math.hypot(gx - wx, gy - wy) / tt * 3600 if tt > 0 else vg
    tcount = L.get('thermal_count') or 0
    corr = abs(t - tc) / t if t else 1
    ok = bool(t > 600 and ca > 0.2 and (dist >= 30) and (tcount >= 2) and (not (last and dh < -500)) and (corr < 0.3))
    r = lambda x, n=1: None if x is None else round(x, n)
    return {'kind': 'leg', 'leg': L.get('number'), 'ok': ok, 'dist': r(dist, 2), 't': round(t), 'ca': r(ca, 2), 'dh': dh, 'vg': r(vg), 'vair': r(vair), 'along': r(along), 'wind': r(wind), 'wdir': r(wdir, 0), 'alt': L.get('glide_alt_avg') or L.get('altitude_avg'), 'therm': r(L.get('thermal_percentage'), 3), 'detour': L.get('glide_detour'), 'gspd': L.get('glide_speed'), 'gld': L.get('glide_ratio'), 'glift': L.get('glide_percentage_lift'), 'att': L.get('attempt_percentage'), 'agl': L.get('agl_avg'), 'tagl': L.get('thermal_start_agl'), 'tgain': L.get('thermal_gain'), 'tcount': tcount, 'trad': L.get('thermal_radius'), 'tbank': L.get('thermal_bank'), 'speed': r(speed), 'z0': z0, 'z1': z1, 'tstart': iso(ts(t0)), 'tend': iso(ts(t1)), 'a': [round(a[0], 5), round(a[1], 5)], 'b': [round(b[0], 5), round(b[1], 5)]}

def free_legs(detail: dict) -> list[dict]:
    free = next((c for c in detail.get('contest') or [] if c.get('name') == 'free'), None)
    score = (free or {}).get('score') or {}
    legs = score.get('leg') or []
    geoms = {}
    for ft in (score.get('geom') or {}).get('features') or []:
        idx = (ft.get('properties') or {}).get('idx')
        c = (ft.get('geometry') or {}).get('coordinates') or []
        if idx is not None and len(c) >= 2:
            geoms[idx] = (c[0][:2], c[-1][:2])
    out = []
    for i, L in enumerate(legs):
        ab = geoms.get(L.get('number', i), geoms.get(i))
        if not ab or not L.get('start_time') or (not L.get('end_time')):
            continue
        out.append(leg_fields(L, ab[0], ab[1], i == len(legs) - 1))
    return out

def douglas_peucker(xy: list, tol: float) -> list[int]:
    keep = {0, len(xy) - 1}
    stack = [(0, len(xy) - 1)]
    while stack:
        a, b = stack.pop()
        (ax, ay), (bx, by) = (xy[a], xy[b])
        L = math.hypot(bx - ax, by - ay)
        best, k = (-1.0, None)
        for i in range(a + 1, b):
            px, py = xy[i]
            d = abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / L if L > 0 else math.hypot(px - ax, py - ay)
            if d > best:
                best, k = (d, i)
        if k is not None and best > tol:
            keep.add(k)
            stack += [(a, k), (k, b)]
    return sorted(keep)

def simplify(points: list, tol_km: float=SIMPLIFY_KM, min_km: float=MIN_PIECE_KM) -> list:
    if len(points) <= 2:
        return [list(p) for p in points]
    to = km_xy(points[0])
    xy = [to(p) for p in points]
    idx = douglas_peucker(xy, tol_km)
    while len(idx) > 2:
        short = [j for j in range(len(idx) - 1) if math.dist(xy[idx[j]], xy[idx[j + 1]]) < min_km]
        if not short:
            break
        cand = set()
        for j in short:
            if j > 0:
                cand.add(j)
            if j + 1 < len(idx) - 1:
                cand.add(j + 1)
        if not cand:
            break

        def area(j):
            (ax, ay), (bx, by), (cx, cy) = (xy[idx[j - 1]], xy[idx[j]], xy[idx[j + 1]])
            return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        idx.pop(min(cand, key=area))
    return [[round(points[i][0], 5), round(points[i][1], 5)] for i in idx]

def leg_line(fx: list[dict], leg: dict) -> list | None:
    t0, t1 = (ts(leg['tstart']), ts(leg['tend']))
    pts = [f['p'] for f in fx if t0 <= f['t'] <= t1]
    if len(pts) < 2:
        return None
    return simplify([leg['a']] + pts + [leg['b']])

def climbs_line(leg: dict, tops: list[dict]) -> list | None:
    t0, t1 = (ts(leg['tstart']), ts(leg['tend']))
    mid = sorted((c for c in tops if t0 <= ts(c['t0']) + c['dur'] <= t1 and (not c.get('engon'))), key=lambda c: ts(c['t0']))
    return simplify([leg['a']] + [[c['lon'], c['lat']] for c in mid] + [leg['b']])

def flight_from(item: dict, detail: dict, data: dict) -> dict:
    legs = free_legs(detail)
    fx = fixes(data)
    cl = climbs(fx, [{'n': L['leg'], 't0': L['tstart'], 't1': L['tend']} for L in legs])
    for L in legs:
        L['line'] = leg_line(fx, L)
        L['line_src'] = 'track'
    tp = (detail.get('takeoff_point') or {}).get('coordinates') or (fx[0]['p'] if fx else None)
    es = [f['e'] for f in fx if f['e'] is not None]
    thin, last = ([], -1000000000.0)
    for f in fx:
        if f['t'] - last >= TRACK_S:
            thin.append([round(f['t']), round(f['p'][0], 5), round(f['p'][1], 5), round(f['z']), None if f['g'] is None else round(f['g'])])
            last = f['t']
    return {'legs': legs, 'climbs': cl, 'tp': tp, 'track': thin, 'zmax': max((f['z'] for f in fx), default=None), 'engine_max': max(es) if es else None, 'engine_med': sorted(es)[len(es) // 2] if es else None}
