from __future__ import annotations
import argparse
import io
import json
import os
import re
import sys
from datetime import date, datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import lake as lakem
import pairjob
import pairs as P
UTC = timezone.utc
SCHEMA = 'shadow-score/1'
SHADOW_DAYS = 14
NAMES = ('blend',)
REGION_MODEL = {'conus': 'blend', 'sh': 'blendsh'}
FIELDS = ('t2', 'td2', 'wind', 'cubase', 'cu')
SHADOW = 'shadow/'
MODELS_ROOT = 'derived/models/'
SCORES = 'derived/shadow/'

def shadow_key(name: str, bid: str) -> str:
    if name not in NAMES or not re.fullmatch('[A-Za-z0-9._-]{1,80}', bid or ''):
        raise ValueError('shadow: name or id')
    return f'{SHADOW}{name}/{bid}/'

def score_prefix(name: str, bid: str) -> str:
    shadow_key(name, bid)
    return f'{SCORES}{name}/{bid}/score/'

def shadow_ids(ops, name: str='blend') -> list[str]:
    out = set()
    for k in ops.list(f'{SHADOW}{name}/'):
        m = re.fullmatch(f'{SHADOW}{name}/([A-Za-z0-9._-]{{1,80}})/ops/.+', k)
        if m:
            out.add(m.group(1))
    return sorted(out)

def _get_any(lk, key: str) -> bytes | None:
    for k, get in ((key + '.lenc', lk.get), (key, lk.get_plain)):
        try:
            return get(k)
        except lakem.LakeMissing:
            continue
    return None

def bundle_of(lk, name: str, bid: str) -> dict:
    base = f'{MODELS_ROOT}{name}/{bid}/'
    raw = _get_any(lk, base + 'bundle.json')
    if raw is None:
        raise lakem.LakeMissing('shadow: no such bundle')
    b = json.loads(raw)
    oof = _get_any(lk, base + 'oof_ops.json')
    o = json.loads(oof) if oof else {}
    rows = o.get('fields', {})
    by_region = {r: (x or {}).get('fields', {}) for r, x in (o.get('regions') or {}).items()}
    return {'bundle': b, 'oof': rows, 'oof_regions': by_region}

def _region(b: dict, region: str | None=None) -> dict:
    regs = b.get('regions') or {}
    if region is not None:
        return regs.get(region) or {}
    return regs.get('conus') or (next(iter(regs.values())) if regs else {})

def _bins(b: dict, region: str | None=None) -> tuple[list, list]:
    import blendtrain as BT
    for fd in (_region(b, region).get('fields') or {}).values():
        bins = (fd.get('weights') or {}).get('bins') or {}
        if bins.get('lead') and bins.get('solar'):
            return (list(bins['lead']), list(bins['solar']))
    d = BT.bins()
    return (list(d['lead']), list(d['solar']))

def base_of(b: dict, region: str | None=None) -> str:
    return _region(b, region).get('base') or 'hrrr'

def _records(store, prefix: str, model: str, counts: dict) -> list:
    out, lists = ({}, {})
    keys = store.list(f'{prefix}ops/{model}/')
    have = set(keys)
    pat = re.compile(f'{re.escape(prefix)}ops/{model}/(\\d{{8}})/(\\d{{2}})/b([A-Za-z0-9._-]+)/stations\\.npz')
    for k in sorted(keys):
        g = pat.fullmatch(k)
        dk = k[:-len('stations.npz')] + 'done.json'
        if not g or dk not in have:
            continue
        try:
            done = json.loads(store.get(dk) or b'{}')
            with np.load(io.BytesIO(store.get(k)), allow_pickle=False) as z:
                arr = {x: z[x] for x in z.files}
            v = str(arr['stations_v'])
            if v not in lists:
                meta = store.get(f'{prefix}ops/meta/stations-{v}.json') or store.get(f'ops/meta/stations-{v}.json')
                lists[v] = json.loads(meta)['stations'] if meta else None
            if lists[v] is None:
                counts['no_list'] = counts.get('no_list', 0) + 1
                continue
            rec = P.record_from_stations(arr, lists[v], done, src='ops')
        except (ValueError, KeyError, OSError, TypeError):
            counts['unreadable'] = counts.get('unreadable', 0) + 1
            continue
        run = g[1] + g[2]
        prev = out.get(run)
        if prev is None or (bool(done.get('published')), k) > (prev[0], prev[1]):
            out[run] = (bool(done.get('published')), k, rec)
    return [(r, x[2]) for r, x in sorted(out.items())]

def values_of(t: dict) -> dict:
    f = lambda k: np.asarray(t[k], np.float64) if k in t else np.full(P.n_rows(t), np.nan)
    u, v, el, cb = (f('f_u_sfc'), f('f_v_sfc'), f('f_elevation'), f('f_cubase'))
    return {'t2': f('f_t2'), 'td2': f('f_td2'), 'wind': np.hypot(u, v), 'cubase': cb + np.nan_to_num(el, nan=0.0), 'cu': np.isfinite(cb).astype(np.float64)}

def obs_values(obs: dict) -> dict:
    f = lambda k: np.asarray(obs[k], np.float64) if k in obs else np.full(P.n_rows(obs), np.nan)
    cu = f('o_cu')
    return {'t2': f('o_t2'), 'td2': f('o_td2'), 'wind': f('o_ws'), 'cu': cu, 'cubase': np.where(cu == 1, f('o_cubase') + np.nan_to_num(f('elev'), nan=0.0), np.nan)}

def _index(t: dict) -> dict:
    return {(str(a), int(b)): i for i, (a, b) in enumerate(zip(t['obs_id'], t['t_valid']))}

def shadow_rows(blend_recs: list, base_recs: list, obs_by_day, lead_edges, solar_bins) -> dict:
    import blendtrain as BT
    import blend as B
    import build
    base_of = dict(base_recs)
    cols = {f: {k: [] for k in ('day', 'area', 'lead_bin', 'sun', 'blend', 'base')} for f in FIELDS}
    for run, rec in blend_recs:
        brec = base_of.get(run)
        if brec is None:
            continue
        days = sorted({f'{s[:4]}-{s[4:6]}-{s[6:8]}' for s in rec['stamps']})
        for day in days:
            obs = obs_by_day(day)
            a, b = (P.station_pairs(rec, 'metar', day), P.station_pairs(brec, 'metar', day))
            if not (P.n_rows(a) and P.n_rows(b) and P.n_rows(obs)):
                continue
            ib, io_ = (_index(b), _index(obs))
            ia, jb, jo = ([], [], [])
            for i, key in enumerate(zip((str(x) for x in a['obs_id']), (int(x) for x in a['t_valid']))):
                if key in ib and key in io_:
                    (ia.append(i), jb.append(ib[key]), jo.append(io_[key]))
            if not ia:
                continue
            ia, jb, jo = (np.array(ia), np.array(jb), np.array(jo))
            va, vb, vo = (values_of(a), values_of(b), obs_values(obs))
            where = {o: (la, lo) for o, la, lo in zip(rec['obs_ids'], rec['lat'], rec['lon'])}
            lat = np.array([where[str(o)][0] for o in a['obs_id'][ia]], np.float64)
            lon = np.array([where[str(o)][1] for o in a['obs_id'][ia]], np.float64)
            tv = np.asarray(a['t_valid'], np.int64)[ia]
            lead = np.asarray(a['lead_h'], np.float64)[ia]
            hour = tv // 3600 % 24
            solar_h = ((hour + lon / 15) % 24 + 24) % 24
            sun_elev = np.full(len(tv), np.nan)
            for t in np.unique(tv):
                s = tv == t
                sun_elev[s] = build.sun_elevation(datetime.fromtimestamp(int(t), UTC).replace(tzinfo=None), lat[s], lon[s])
            area = BT.areas({'lat': lat, 'lon': lon})
            lb = np.clip(np.searchsorted(np.asarray(lead_edges, np.float64), lead, side='right') - 1, 0, len(lead_edges) - 2)
            sun = B.sun_band(solar_h, sun_elev, solar_bins)
            for f in FIELDS:
                o = vo[f][jo]
                pa, pb = (va[f][ia], vb[f][jb])
                ok = np.isfinite(o) & np.isfinite(pb)
                if not ok.any():
                    continue
                c = cols[f]
                c['day'].append(tv[ok] // 86400)
                c['area'].append(area[ok])
                c['lead_bin'].append(lb[ok])
                c['sun'].append(np.asarray(sun)[ok])
                c['blend'].append(BT.loss(f, o[ok], pa[ok]))
                c['base'].append(BT.loss(f, o[ok], pb[ok]))
    out = {}
    for f, c in cols.items():
        out[f] = {k: np.concatenate(v) if v else np.zeros(0) for k, v in c.items()}
    return out

def _oof_arrays(oof: dict) -> dict:
    keys = ('day', 'area', 'lead_bin', 'sun', 'blend', 'base')
    return {k: np.asarray(oof.get(k) or [], np.float64 if k in ('blend', 'base') else np.int64) for k in keys}

def score(name: str, bid: str, b: dict, oof: dict, rows: dict, lead_edges, solar_bins, now: datetime | None=None, seed: int | None=None, reg: str | None=None) -> dict:
    import split
    seed = split.SEED if seed is None else seed
    days = sorted({int(d) for f in FIELDS for d in rows.get(f, {}).get('day', [])})
    fields = {}
    sun_names = ['night'] + [f'solar {x[0]}-{x[1]}' for x in solar_bins[1:]]
    for f in FIELDS:
        s, o = (rows.get(f) or {}, _oof_arrays(oof.get(f) or {}))
        n_s, n_o = (len(s.get('day', [])), len(o['day']))
        if not n_s and (not n_o):
            continue
        cat = lambda k, dt: np.concatenate([np.asarray(s.get(k, []), dt), np.asarray(o[k], dt)])
        day, area, lb, sun = (cat('day', np.int64), cat('area', np.int64), cat('lead_bin', np.int64), cat('sun', np.int64))
        lb_ = np.clip(lb, 0, len(lead_edges) - 2)
        new, base = (cat('blend', np.float64), cat('base', np.float64))
        blocks = split.blocks_of(split.week_of(day), area)
        region = np.array([f'{a}|lead {lead_edges[i]}-{lead_edges[i + 1]}' for a, i in zip(area, lb_)], dtype=object)
        tod = np.array([sun_names[int(x)] if 0 <= int(x) < len(sun_names) else 'night' for x in sun], dtype=object)
        v = split.ship_verdict(new, base, blocks, {'region': region, 'time of day': tod}, seed=seed)
        lo, hi = (v['overall'].get('lo'), v['overall'].get('hi'))
        mean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else None
        fields[f] = {'rows_shadow': int(n_s), 'rows_oof': int(n_o), 'mae_blend': mean(new), 'mae_base': mean(base), 'diff': v['overall'].get('diff'), 'ci95': [lo, hi], 'power': (float(hi) - float(lo)) / 2 if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) else None, 'pass': bool(v['ship']), 'verdict': _plain(v)}
    iso = lambda d: date.fromordinal(date(1970, 1, 1).toordinal() + d).isoformat()
    return {'schema': SCHEMA, 'name': name, 'id': bid, 'days': len(days), 'first': iso(days[0]) if days else None, 'last': iso(days[-1]) if days else None, 'base': base_of(b, reg), 'fields': fields, 'made_at': (now or datetime.now(UTC)).strftime('%Y-%m-%dT%H:%M:%SZ')}

def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and (not np.isfinite(x)):
        return None
    return x

def write_score(lk, sc: dict, t: datetime | None=None) -> str:
    t = t or datetime.now(UTC)
    body = json.dumps(sc, sort_keys=True, indent=1).encode()
    pre = score_prefix(sc['name'], sc['id'])
    for k in range(100):
        key = f'{pre}{t:%Y%m%dT%H%M%S}{k:02d}Z.json.lenc'
        if lk.put(key, body, strict=False).status != 'differs':
            return key
    raise lakem.LakeError('shadow: no free name for the score')

def newest_score(lk, name: str, bid: str) -> tuple[str, dict] | None:
    keys = sorted((k for k in lk.list(score_prefix(name, bid)) if k.endswith('.json.lenc')))
    if not keys:
        return None
    return (keys[-1], json.loads(lk.get(keys[-1])))

def run(lk, ops, name: str='blend', bid: str | None=None, now: datetime | None=None, counts: dict | None=None) -> list[str]:
    counts = counts if counts is not None else {}
    if ops is None:
        counts['no_ops'] = 1
        return []
    ids = [bid] if bid else shadow_ids(ops, name)
    written = []
    obs_cache = {}

    def obs_by_day(day):
        if day not in obs_cache:
            obs_cache[day] = P.read_kind(lk, 'obs', 'metar', [day], purpose='train', columns=['obs_id', 't_valid', 'elev', 'o_t2', 'o_td2', 'o_ws', 'o_cu', 'o_cubase'])
        return obs_cache[day]
    for i in ids:
        try:
            got = bundle_of(lk, name, i)
        except lakem.LakeMissing:
            counts['no_bundle'] = counts.get('no_bundle', 0) + 1
            continue
        b = got['bundle']
        by_region, runs = ({}, 0)
        for r in b.get('regions') or {}:
            model = REGION_MODEL.get(r)
            if model is None:
                continue
            lead_edges, solar_bins = _bins(b, r)
            shadow = _records(ops, shadow_key(name, i), model, counts)
            oof = got['oof'] if r == ('conus' if 'conus' in b['regions'] else sorted(b['regions'])[0]) else {}
            oof = got.get('oof_regions', {}).get(r, oof)
            if not shadow and (not oof):
                continue
            base_recs = _records(ops, '', base_of(b, r), counts) if shadow else []
            rows = shadow_rows(shadow, base_recs, obs_by_day, lead_edges, solar_bins) if shadow else {}
            by_region[r] = score(name, i, b, oof, rows, lead_edges, solar_bins, now, reg=r)
            runs += len(shadow)
        if not runs:
            counts['no_shadow_runs'] = counts.get('no_shadow_runs', 0) + 1
            continue
        main_r = 'conus' if 'conus' in by_region else sorted(by_region)[0]
        sc = dict(by_region[main_r])
        sc['region'] = main_r
        sc['regions'] = {r: {k: v[k] for k in ('days', 'first', 'last', 'base', 'fields')} for r, v in by_region.items()}
        written.append(write_score(lk, sc, now))
        counts['scored'] = counts.get('scored', 0) + 1
        counts['shadow_runs'] = counts.get('shadow_runs', 0) + runs
    return written

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A bundle's days in shadow, scored")
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('run')
    a.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    a.add_argument('--ops-local', help='a folder as OPS (else OPS_S3_ENDPOINT, OPS_S3_BUCKET, OPS_RO_*)')
    a.add_argument('--name', default='blend', choices=NAMES)
    a.add_argument('--id', help='one bundle (default: each with a shadow run)')
    x = ap.parse_args(argv)
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('shadow: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts: dict = {}
    try:
        keys = run(lk, pairjob.ops_store(local=x.ops_local), x.name, x.id, counts=counts)
    except lakem.LakeError:
        print('shadow: the lake failed', flush=True)
        return 1
    for k in keys:
        got = json.loads(lk.get(k))
        passes = sorted((f for f, v in got['fields'].items() if v['pass']))
        print(f"shadow: {got['id']}: {got['days']} days, fields passing {', '.join(passes) or 'none'}", flush=True)
    print('shadow: ' + (', '.join((f'{k} {v}' for k, v in sorted(counts.items()))) or 'nothing to score'), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
