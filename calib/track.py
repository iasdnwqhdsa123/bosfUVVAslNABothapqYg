from __future__ import annotations
import bisect
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
CIRC_DEG, CIRC_WIN, CIRC_GAP = (220.0, 30.0, 25.0)
CLIMB_GAIN, CLIMB_DUR, CLIMB_TURN = (150.0, 60.0, 270.0)
ENGINE_ON = 500.0
ENGINE_SHARE = 0.2
MOTOR_KINDS = ('MG', 'FG')
GUESS_RATE, GUESS_WIN, GUESS_S, GUESS_VG, GUESS_SPREAD = (0.7, 10.0, 90.0, 20.0, 0.5)
GUESS_TURN, GUESS_TURN_S, GUESS_GAP, GUESS_PAD = (90.0, 60.0, 10.0, 30.0)
ENGINE_GUESS = 999.0
LAUNCH_S = 300.0
FES_SHARE, FES_MAX, FES_OVER, FES_RUN_S = (0.3, 300.0, 3.0, 30.0)
ENGINE_LEG_S = 60.0
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

def _turns(fx: list[dict]) -> list[float]:
    turn = [0.0] * len(fx)
    last_h, last_p = (None, fx[0]['p'] if fx else None)
    for i in range(1, len(fx)):
        p = fx[i]['p']
        if dist_km(last_p, p) < 0.015:
            continue
        h = bearing(last_p, p)
        if last_h is not None:
            turn[i] = (h - last_h + 540) % 360 - 180
        last_h, last_p = (h, p)
    return turn

def motor(kind) -> dict:
    return {'mg': None if kind is None else kind in MOTOR_KINDS, 'fes': None if kind is None else kind == 'FG'}

def engine_guess(fx: list[dict]) -> int:
    n = len(fx)
    if n < 10:
        return 0
    t = [f['t'] for f in fx]
    turn = _turns(fx)
    acc = [0.0]
    for x in turn:
        acc.append(acc[-1] + x)
    vz, vg, net = ([None] * n, [None] * n, [0.0] * n)
    a = b = a2 = b2 = 0
    for i in range(n):
        while t[a] < t[i] - GUESS_WIN / 2:
            a += 1
        while b + 1 < n and t[b + 1] <= t[i] + GUESS_WIN / 2:
            b += 1
        if t[b] > t[a]:
            vz[i] = (fx[b]['z'] - fx[a]['z']) / (t[b] - t[a])
            vg[i] = dist_km(fx[a]['p'], fx[b]['p']) * 1000 / (t[b] - t[a])
        while t[a2] < t[i] - GUESS_TURN_S / 2:
            a2 += 1
        while b2 + 1 < n and t[b2 + 1] <= t[i] + GUESS_TURN_S / 2:
            b2 += 1
        net[i] = abs(acc[b2 + 1] - acc[a2 + 1])
    up = [vz[i] is not None and vz[i] >= GUESS_RATE and (vg[i] >= GUESS_VG) and (net[i] < GUESS_TURN) for i in range(n)]
    runs, i = ([], 0)
    while i < n:
        if not up[i]:
            i += 1
            continue
        j = i
        while True:
            k = j + 1
            while k < n and (not up[k]) and (t[k] - t[j] < GUESS_GAP):
                k += 1
            if k < n and up[k] and (t[k] - t[j] < GUESS_GAP):
                j = k
            else:
                break
        rs = [vz[k] for k in range(i, j + 1) if vz[k] is not None]
        m = sum(rs) / len(rs)
        if t[j] - t[i] >= GUESS_S and math.sqrt(sum(((r - m) ** 2 for r in rs)) / len(rs)) < GUESS_SPREAD:
            runs.append((i, j))
        i = j + 1
    if not runs:
        return 0
    roll = next((t[i] for i in range(n) if vg[i] is not None and vg[i] >= GUESS_VG), t[0])
    on = [False] * n
    for i, j in runs:
        if t[i] - roll < LAUNCH_S:
            top = max(range(i, j + 1), key=lambda k: fx[k]['z'])
            for k in range(top):
                on[k] = True
        else:
            for k in range(n):
                if t[i] - GUESS_PAD <= t[k] <= t[j] + GUESS_PAD:
                    on[k] = True
    for f, o in zip(fx, on):
        f['e'] = ENGINE_GUESS if o else 0.0
    return len(runs)

def fes_on(fx: list[dict]) -> int:
    es = sorted((f['e'] for f in fx if f['e'] is not None))
    if not es or es[-1] < FES_MAX or es[-1] < FES_OVER * es[len(es) // 2]:
        return 0
    lim = FES_SHARE * es[-1]
    runs, i = (0, 0)
    while i < len(fx):
        if fx[i]['e'] is None or fx[i]['e'] <= lim:
            i += 1
            continue
        j = i
        while j + 1 < len(fx) and fx[j + 1]['e'] is not None and (fx[j + 1]['e'] > lim):
            j += 1
        if fx[j]['t'] - fx[i]['t'] >= FES_RUN_S:
            runs += 1
            for k in range(i, j + 1):
                fx[k]['e'] = max(fx[k]['e'], ENGINE_GUESS)
        i = j + 1
    return runs

def mark_engine(fx: list[dict], mg, fes) -> None:
    if not mg:
        return
    if not any((f['e'] is not None for f in fx)):
        engine_guess(fx)
    elif fes:
        fes_on(fx)

def engine_s(fx: list[dict], t0: float, t1: float) -> float:
    return sum((b['t'] - a['t'] for a, b in zip(fx, fx[1:]) if t0 <= a['t'] < t1 and a['e'] is not None and (a['e'] > ENGINE_ON)))

def climbs(fx: list[dict], legs: list[dict]) -> list[dict]:
    if len(fx) < 10:
        return []
    turn = _turns(fx)
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
    es = [f['e'] for f in fx if f['e'] is not None]
    mo = motor((detail.get('aircraft') or {}).get('kind'))
    mark_engine(fx, mo['mg'], mo['fes'])
    cl = climbs(fx, [{'n': L['leg'], 't0': L['tstart'], 't1': L['tend']} for L in legs])
    for L in legs:
        L['line'] = leg_line(fx, L)
        L['line_src'] = 'track'
        L['engine'] = engine_s(fx, ts(L['tstart']), ts(L['tend'])) >= ENGINE_LEG_S
        if L['engine']:
            L['ok'] = False
    tp = (detail.get('takeoff_point') or {}).get('coordinates') or (fx[0]['p'] if fx else None)
    marked = any((f['e'] is not None for f in fx))
    thin, last = ([], -1000000000.0)
    for f in fx:
        if f['t'] - last >= TRACK_S:
            p = [round(f['t']), round(f['p'][0], 5), round(f['p'][1], 5), round(f['z']), None if f['g'] is None else round(f['g'])]
            if marked:
                p.append(None if f['e'] is None else round(f['e']))
            thin.append(p)
            last = f['t']
    return {'legs': legs, 'climbs': cl, 'tp': tp, 'track': thin, 'zmax': max((f['z'] for f in fx), default=None), 'engine_max': max(es) if es else None, 'engine_med': sorted(es)[len(es) // 2] if es else None, **mo, 'sensor': bool(es)}
HOME_KM = 3.0
AFTER_TAKEOFF_S = 20 * 60
WEAK_RATE = 0.5
SEARCH_S, SEARCH_M = (15 * 60, 600)
LOW_AGL = 400
ENGINE_OFF_S = 300
PROBE_KEYS = ('ws', 'ws0', 'zi', 'top', 'band', 'climb', 'ach', 'usable', 'xc', 'cu', 'cb', 'shade', 'sun', 'sunElev', 'relief', 'elev', 'solar', 'localHour', 'wind', 't2', 'blapse')
FAILED = ('weak', 'away', 'engine')

def _t(x):
    try:
        return ts(x) if isinstance(x, str) else float(x) if x is not None else None
    except (TypeError, ValueError):
        return None

def _climbs_sorted(f: dict) -> list[dict]:
    out = []
    for c in f.get('climbs') or []:
        t0 = _t(c.get('t0'))
        if t0 is not None:
            out.append({**c, '_t0': t0, '_t1': t0 + (c.get('dur') or 0)})
    return sorted(out, key=lambda c: c['_t0'])

def times_of(f: dict):
    tr = f.get('track') or []
    a, b = (_t(f.get('takeoff')), _t(f.get('landing')))
    if a is None and tr:
        a = tr[0][0]
    if b is None and tr:
        b = tr[-1][0]
    return (a, b) if a is not None and b is not None and (b > a) else (None, None)

def pos_at(f: dict, t: float) -> dict:
    tr = f.get('track') or []
    if tr:
        i = bisect.bisect_left([p[0] for p in tr], t)
        i = min(range(max(0, i - 1), min(len(tr), i + 1)), key=lambda k: abs(tr[k][0] - t))
        p = tr[i]
        return {'lon': p[1], 'lat': p[2], 'z': p[3], 'agl': None if p[4] is None else p[3] - p[4], 'src': 'track'}
    t_off, _ = times_of(f)
    knots = ([(t_off, f['tp'][0], f['tp'][1])] if t_off is not None and f.get('tp') else []) + [(c['_t1'], c['lon'], c['lat']) for c in _climbs_sorted(f) if c.get('lon') is not None]
    knots.sort()
    if not knots:
        return {'lon': None, 'lat': None, 'z': None, 'agl': None, 'src': None}
    if t <= knots[0][0]:
        k = knots[0]
        return {'lon': k[1], 'lat': k[2], 'z': None, 'agl': None, 'src': 'climbs'}
    for (ta, xa, ya), (tb, xb, yb) in zip(knots, knots[1:]):
        if ta <= t <= tb:
            w = (t - ta) / (tb - ta) if tb > ta else 0.0
            return {'lon': xa + (xb - xa) * w, 'lat': ya + (yb - ya) * w, 'z': None, 'agl': None, 'src': 'climbs'}
    k = knots[-1]
    return {'lon': k[1], 'lat': k[2], 'z': None, 'agl': None, 'src': 'climbs'}

def engine_start(f: dict) -> float | None:
    tr, t_eng = (f.get('track') or [], None)
    off_since = None
    es = [(p[0], p[5]) for p in tr if len(p) > 5 and p[5] is not None]
    for (t, e), nxt in zip(es, es[1:] + [(None, None)]):
        if e <= ENGINE_ON:
            off_since = t if off_since is None else off_since
        elif nxt[1] is not None and nxt[1] > ENGINE_ON:
            if off_since is not None and t - off_since >= ENGINE_OFF_S:
                t_eng = t
                break
            off_since = None
    soared = False
    for c in _climbs_sorted(f):
        if not c.get('engon'):
            soared = True
        elif soared:
            t_eng = c['_t0'] if t_eng is None else min(t_eng, c['_t0'])
            break
    return t_eng

def ending(f: dict) -> dict | None:
    t_off, t_land = times_of(f)
    if t_off is None:
        return None
    t_eng = engine_start(f)
    if t_eng is not None and t_off < t_eng < t_land:
        p = pos_at(f, t_eng)
        return {'t': t_eng, 'how': 'engine', **p}
    p = pos_at(f, t_land)
    if p['src'] == 'track' and f.get('tp') and (p['lon'] is not None):
        how = 'home' if dist_km(f['tp'], [p['lon'], p['lat']]) <= HOME_KM else 'away'
    else:
        how = 'unknown'
    return {'t': t_land, 'how': how, **p}

def _when(f: dict, t: float, lon) -> dict:
    tz = ZoneInfo(f.get('tz') or 'America/Chicago')
    loc = datetime.fromtimestamp(t, tz)
    solar = None if lon is None else round((t / 3600 % 24 + lon / 15) % 24, 2)
    return {'lh': loc.hour + loc.minute / 60, 'solar': solar, 'month': loc.month}

def hour_samples(f: dict) -> list[dict]:
    end = ending(f)
    if not end:
        return []
    t_off, _ = times_of(f)
    T = math.ceil((t_off + AFTER_TAKEOFF_S) / 3600) * 3600
    out = []
    while T < end['t']:
        p = pos_at(f, T)
        out.append({'t': int(T), 'out': 'flying' if end['t'] >= T + 3600 else end['how'], 'lon': p['lon'], 'lat': p['lat'], 'agl': None if p['agl'] is None else round(p['agl']), 'pos': p['src'], **_when(f, T, p['lon'])})
        T += 3600
    return out

def glides(f: dict) -> list[dict]:
    end = ending(f)
    if not end:
        return []
    cs = [c for c in _climbs_sorted(f) if c['_t0'] < end['t'] and (not c.get('engon'))]
    out = []
    for i, c in enumerate(cs):
        t0, z0 = (c['_t1'], c.get('z1'))
        if i + 1 < len(cs):
            n = cs[i + 1]
            t1, z1, agl, lon, lat = (n['_t0'], n.get('z0'), n.get('agl0'), n.get('lon'), n.get('lat'))
            how = 'weak' if (n.get('rate') or 0) < WEAK_RATE else 'save' if agl is not None and agl < LOW_AGL else 'climb'
        else:
            t1, z1, agl, lon, lat, how = (end['t'], end['z'], end['agl'], end['lon'], end['lat'], end['how'])
        dur = t1 - t0
        if dur <= 0:
            continue
        loss = None if z0 is None or z1 is None else z0 - z1
        search = dur >= SEARCH_S or (loss is not None and loss >= SEARCH_M) or (agl is not None and agl < LOW_AGL)
        out.append({'t0': int(round(t0)), 't': int(round(t1)), 'dur': int(round(dur)), 'loss': None if loss is None else round(loss), 'agl': None if agl is None else round(agl), 'out': how, 'search': bool(search), 'failed': how in FAILED, 'lon': lon, 'lat': lat, **_when(f, t1, lon)})
    return out

def probes(f: dict) -> list[dict]:
    pts = {}
    for h in hour_samples(f):
        pts[h['t']] = (h['lon'], h['lat'])
    for g in glides(f):
        pts.setdefault(g['t'], (g['lon'], g['lat']))
    return [{'t': t, 'lon': round(lo, 5), 'lat': round(la, 5)} for t, (lo, la) in sorted(pts.items()) if lo is not None and la is not None]
SHORT_CLIMB_S = 120
RELEASE_UP, RELEASE_DROP, RELEASE_S = (100.0, 20.0, 60.0)

def short_summary(data: dict, mg=None, fes=None) -> dict | None:
    fx = fixes(data)
    if len(fx) < 10:
        return None
    mark_engine(fx, mg, fes)
    z0 = fx[0]['z']
    rel = None
    for i, f in enumerate(fx):
        if f['z'] - z0 < RELEASE_UP:
            continue
        for g in fx[i + 1:]:
            if g['t'] - f['t'] > RELEASE_S or g['z'] > f['z']:
                break
            if f['z'] - g['z'] >= RELEASE_DROP:
                rel = i
                break
        if rel is not None:
            break
    if rel is None:
        rel = max(range(len(fx)), key=lambda k: fx[k]['z'])
    after = fx[rel:]
    best, j = (None, 0)
    for i, f in enumerate(after):
        j = max(j, i)
        while j < len(after) and after[j]['t'] - f['t'] < SHORT_CLIMB_S:
            j += 1
        if j >= len(after):
            break
        if any(((after[k]['e'] or 0) > ENGINE_ON for k in range(i, j + 1))):
            continue
        r = (after[j]['z'] - f['z']) / (after[j]['t'] - f['t'])
        best = r if best is None else max(best, r)
    R = fx[rel]
    return {'rel_t': iso(R['t']), 'rel_agl': None if R['g'] is None else round(R['z'] - R['g']), 'gain_max': round(max((f['z'] for f in after)) - R['z']), 'best2': None if best is None else round(best, 2), 'climbed': best is not None and best >= WEAK_RATE, 'tp': [round(fx[0]['p'][0], 5), round(fx[0]['p'][1], 5)], 'home': dist_km(fx[0]['p'], fx[-1]['p']) <= HOME_KM, 'engine': any(((f['e'] or 0) > ENGINE_ON for f in after))}
