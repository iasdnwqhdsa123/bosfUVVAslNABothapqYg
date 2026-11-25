from __future__ import annotations
import hashlib
import math
from collections import Counter
import collector
import track
from . import guard, objects
EVENTS_V = 'e1'
VERSIONS = (1, 2)
KINDS = ('c', 's', 'f')
OTHER_KINDS = ('x', 'p', 'wa')
TYPES = tuple(collector.TYPES)
GLIDE_TOL_S = 90
ALT_RANGE = (-1000.0, 20000.0)
_HEX = set('0123456789abcdef')

class Bad(ValueError):
    pass

def _num(r: dict, k: str, lo: float | None=None, hi: float | None=None) -> float:
    x = r.get(k)
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (not math.isfinite(x)):
        raise Bad(f'field {k}')
    if lo is not None and x < lo or (hi is not None and x > hi):
        raise Bad(f'range {k}')
    return float(x)

def _opt(r: dict, k: str, nd: int | None=2):
    x = r.get(k)
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (not math.isfinite(x)):
        return None
    return round(float(x), nd) if nd is not None else float(x)

def _points(pts, v: int, latlon: bool):
    if pts is None:
        return None
    if not isinstance(pts, list):
        raise Bad('field points')
    out = []
    for p in pts:
        if not isinstance(p, list) or len(p) != (4 if v == 2 else 3):
            raise Bad('field points')
        q = [x if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) else None for x in p]
        if None in q[:3] or (latlon and (not (-90 <= q[0] <= 90 and -180 <= q[1] <= 180))):
            raise Bad('field points')
        out.append([round(q[0], 5 if latlon else 2), round(q[1], 5 if latlon else 2), round(q[2]), None if v == 1 or q[3] is None else int(q[3])])
    return out

def _vz(x):
    if not isinstance(x, dict) or not isinstance(x.get('r'), list):
        return None
    z0 = x.get('z0')
    if isinstance(z0, bool) or not isinstance(z0, (int, float)) or (not math.isfinite(z0)):
        return None
    return {'z0': round(z0), 'r': [round(v, 2) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) else None for v in x['r']]}
GL_FIELDS = ('t', 'd', 'L', 'dz', 'nm', 'n10', 'sl', 'ns', 'tas', 'hd', 'wu', 'wv')

def _gl(x):
    if not isinstance(x, dict):
        return None
    out = {k: _opt(x, k) for k in GL_FIELDS}
    return out if out['t'] is not None and out['d'] is not None else None

def check(r) -> tuple[str, dict, str, str]:
    if not isinstance(r, dict):
        raise Bad('not an object')
    v = r.get('v')
    if isinstance(v, bool) or v not in VERSIONS:
        raise Bad('version')
    k = r.get('k')
    if k not in KINDS:
        raise Bad('kind')
    if k == 'f' and v < 2:
        raise Bad('kind')
    a = r.get('a')
    if not isinstance(a, str) or not 4 <= len(a) <= 64:
        raise Bad('field a')
    y, s = (r.get('y'), r.get('s'))
    if not isinstance(y, str) or not (s is None or isinstance(s, str)):
        raise Bad('field y')
    t0 = _num(r, 't0', 1000000000.0, 4000000000.0)
    dur = _num(r, 'dur', 0, 6 * 3600)
    la0, lo0 = (_num(r, 'la0', -90, 90), _num(r, 'lo0', -180, 180))
    z0, zt = (_num(r, 'z0', *ALT_RANGE), _num(r, 'zt', *ALT_RANGE))
    sd = objects.solar_day(t0, lo0)
    if v == 2:
        if not isinstance(r.get('sd'), str) or not objects.DAY_RE.match(r['sd']):
            raise Bad('field sd')
        sd = r['sd']
    rv = r.get('rv') if isinstance(r.get('rv'), str) else None
    e = {'v': v, 'rv': rv, 'cv': r.get('cv') if v == 2 and isinstance(r.get('cv'), str) else None, 'y': y, 's': s, 'pc': r.get('pc') if isinstance(r.get('pc'), str) else None, 't0': int(round(t0)), 'dur': int(round(dur)), 'la0': round(la0, 5), 'lo0': round(lo0, 5), 'z0': round(z0), 'zt': round(zt), 'lt': round((t0 + lo0 * 240.0) % 86400 / 3600.0, 3)}
    if k in ('c', 's'):
        e.update(la=round(_num(r, 'la', -90, 90), 5), lo=round(_num(r, 'lo', -180, 180), 5), rate=_opt(r, 'rate'), air=_opt(r, 'air'), dt=_opt(r, 'dt', 1), rad=_opt(r, 'rad', 0))
    if k == 'c':
        x = r.get('x')
        e.update(x=x if x in ('top', 'early', 'lost') else None, re=_opt(r, 're'), tr=_points(r.get('tr'), v, True), w=_points(r.get('w'), v, False))
        if v == 2:
            e.update(vz=_vz(r.get('vz')), nc=_opt(r, 'nc', 0), per=_opt(r, 'per', 1), gl=_gl(r.get('gl')), ag=_opt(r, 'ag', 0))
        else:
            e.update(vz=None, nc=None, per=None, gl=None, ag=None)
    elif k == 's':
        e['ts'] = int(r['ts']) if v == 2 and _opt(r, 'ts') is not None else None
    else:
        e.update(gain=_opt(r, 'gain', 0), rate=_opt(r, 'rate'), turn=_opt(r, 'turn', 0), nc=_opt(r, 'nc', 0))
    return (k, e, f'{v}|{a}', sd)

def why_dropped(k: str, e: dict) -> str | None:
    if e['s'] not in guard.FEEDS:
        return 'feed not admitted'
    if e['y'] not in TYPES:
        return 'not a glider, hang glider or paraglider'
    if k == 'c':
        if e['zt'] - e['z0'] < track.CLIMB_GAIN or e['dur'] < track.CLIMB_DUR:
            return 'short climb'
        if (e['zt'] - e['z0']) / max(e['dur'], 1) > collector.RATE_MAX:
            return 'implausible rate'
    return None

def parse(text: bytes, cnt: Counter | None=None) -> list[tuple[str, dict, str, str]]:
    import json
    cnt = cnt if cnt is not None else Counter()
    out = []
    for line in text.decode('utf-8', 'replace').splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except (ValueError, RecursionError):
            cnt['bad_unreadable'] += 1
            continue
        k = r.get('k') if isinstance(r, dict) else None
        if k in OTHER_KINDS:
            cnt[f'other_{k}'] += 1
            continue
        try:
            got = check(r)
        except Bad as b:
            cnt['bad_' + str(b).replace(' ', '_')] += 1
            continue
        why = why_dropped(got[0], got[1])
        if why:
            cnt['dropped_' + why.replace(' ', '_').replace(',', '')] += 1
            continue
        cnt[f'read_{got[0]}'] += 1
        out.append(got)
    return out

def _order(k: str, e: dict, link: str):
    return (e['t0'], e['la0'], e['lo0'], e['z0'], KINDS.index(k), e['dur'], e['zt'], -e['v'], link)

def assemble(sd: str, parsed: list, cnt: Counter) -> dict:
    mine = [p for p in parsed if p[3] == sd]
    mine.sort(key=lambda p: _order(p[0], p[1], p[2]))
    seen_a, seen_p, kept = (set(), set(), [])
    for k, e, link, _ in mine:
        ka, kp = ((link, k, e['t0']), (k, e['t0'], e['la0'], e['lo0'], e['z0']))
        if ka in seen_a or kp in seen_p:
            cnt['twice'] += 1
            continue
        seen_a.add(ka)
        seen_p.add(kp)
        kept.append((k, e, link))
    ac, nth, last = ({}, Counter(), {})
    out = {'c': [], 's': [], 'f': []}
    for k, e, link in kept:
        e = dict(e, ac=ac.setdefault(link, len(ac)))
        if k == 'c':
            e['n'] = nth[link]
            nth[link] += 1
            e['prev'] = None
            p = last.get(link)
            if e['gl'] is not None and p is not None:
                prev = out['c'][p]
                if prev['t0'] + prev['dur'] <= e['t0'] and abs(e['t0'] - (prev['t0'] + prev['dur']) - e['gl']['t']) <= GLIDE_TOL_S:
                    e['prev'] = p
                else:
                    cnt['glide_unmatched'] += 1
            last[link] = len(out['c'])
        out[k].append(e)
    cnt['aircraft'] = len(ac)
    feeds = sorted({x['s'] for part in out.values() for x in part})
    return {'feeds': feeds, 'climbs': out['c'], 'straight': out['s'], 'failed': out['f']}

def build(lk, sd: str, raw: dict, cache: dict | None=None) -> dict:
    cache = cache if cache is not None else {}
    cnt, parsed, inputs = (Counter(), [], [])
    _, missing = objects.ready(sd, raw)
    for d in objects.needs(sd):
        for rk in raw.get(d, []):
            if rk not in cache:
                text = objects.read_raw(lk, rk)
                c = Counter()
                cache[rk] = (parse(text, c), c, hashlib.sha256(text).hexdigest())
            p, c, sha = cache[rk]
            parsed += p
            if d == sd:
                cnt.update(c)
            inputs.append({'day': d, 'v': int(rk.split('/')[2][1:]), 'sha256': sha})
    ev = assemble(sd, parsed, cnt)
    return {'v': 1, 'rule': EVENTS_V, 'sd': sd, 'inputs': inputs, 'missing': missing, 'counts': dict(sorted(cnt.items())), **ev}

def run(lk, days=None, cache=None, n=None, most: int | None=None) -> dict:

    def make(sd, raw, missing):
        obj = build(lk, sd, raw, cache)
        return (objects.dumps(obj), {'climbs': len(obj['climbs'])})
    return objects.run_days(lk, 'events', EVENTS_V, make, days, cache, n, most)
