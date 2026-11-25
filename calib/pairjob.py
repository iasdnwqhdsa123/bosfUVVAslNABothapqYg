from __future__ import annotations
import argparse
import gzip
import io
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog
import common
import lake as lakem
import obs_raob
import pairs as P
UTC = timezone.utc
LEAD_MAX_H = 48
KINDS = ('metar', 'raob', 'sat', 'snap', 'radar', 'acwind', 'window')
BACK = 10
MINUTES = 40.0
OPS_ROOT = 'ops'
IEM_ROOT = 'raw/obs/iem'
SAT_HOURS = (15, 18, 21)
SAT_AREA = 'US'
SAT_AFTER_D = 1
METAR_V = 2
SAT_V = 2
SAT_BACK = 2
SAT_GRID_V = 1
SAT_GRID = {'us': {'box': common.US_BOX, 'hours': SAT_HOURS, 'ground': 'hrrr'}, 'sh': {'box': (112.0, -49.0, 179.75, -18.5), 'hours': (1, 4, 6), 'ground': 'ecmwf'}}

def _n(counts: dict, k: str, v: int=1) -> None:
    counts[k] = counts.get(k, 0) + v

def _day_ts(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())

def run_reaches(run: str, day: str, lead_max: int=LEAD_MAX_H) -> bool:
    if not re.fullmatch('\\d{10}', run or ''):
        return False
    r = datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=UTC)
    d0 = datetime.fromisoformat(day).replace(tzinfo=UTC)
    return r < d0 + timedelta(days=1) and r + timedelta(hours=lead_max) >= d0

def run_dates(day: str, lead_max: int=LEAD_MAX_H) -> list[str]:
    d = date.fromisoformat(day)
    return [(d - timedelta(days=k)).strftime('%Y%m%d') for k in range(lead_max // 24, -1, -1)]

def stcast_rows(snap: dict | None, day: str) -> list[dict]:
    if not snap:
        return []
    return [r for r in catalog.usable(snap) if r.get('kind') == catalog.STCAST and r.get('key') and run_reaches(str(r.get('run') or r.get('init') or ''), day)]

def stcast_records(lk, snap: dict | None, day: str, counts: dict) -> dict:
    import labbuild
    out = {}
    for r in stcast_rows(snap, day):
        try:
            z = labbuild.read_unit(lk.get(r['key']))
            sts = catalog.stations_of(lk, str(z['stations_v']))
            rec = P.record_from_stations(z, sts, src='stcast')
        except lakem.LakeMissing:
            _n(counts, 'stcast_missing')
            continue
        except (catalog.UnitError, ValueError, KeyError):
            _n(counts, 'stcast_unreadable')
            continue
        out[r['key'], r['sha256']] = (rec['model'], rec)
        _n(counts, 'stcast_runs')
    return out

def ops_store(env=None, local: str | None=None):
    if local:
        return lakem.LocalStore(local)
    e = os.environ if env is None else env
    vals = [e.get(k) or '' for k in ('OPS_S3_ENDPOINT', 'OPS_S3_BUCKET', 'OPS_RO_KEY', 'OPS_RO_SECRET')]
    return lakem.S3Store(*vals) if all(vals) else None
_OPS_KEY = re.compile('ops/(?P<m>[a-z0-9]+)/(?P<d>\\d{8})/(?P<h>\\d{2})/b(?P<b>[A-Za-z0-9._-]+)/stations\\.npz')

def ops_records(ops, day: str, counts: dict, models=P.MODELS) -> dict:
    import numpy as np
    if ops is None:
        return {}
    out, lists = ({}, {})
    for m in models:
        for d in run_dates(day):
            keys = ops.list(f'{OPS_ROOT}/{m}/{d}/')
            have = set(keys)
            for k in sorted(keys):
                g = _OPS_KEY.fullmatch(k)
                if not g or not run_reaches(g['d'] + g['h'], day):
                    continue
                done_key = k[:-len('stations.npz')] + 'done.json'
                if done_key not in have:
                    _n(counts, 'ops_unfinished')
                    continue
                try:
                    blob, done = (ops.get(k), json.loads(ops.get(done_key) or b'{}'))
                    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                        arr = {x: z[x] for x in z.files}
                    v = str(arr['stations_v'])
                    if v not in lists:
                        meta = ops.get(f'{OPS_ROOT}/meta/stations-{v}.json')
                        lists[v] = json.loads(meta)['stations'] if meta else None
                    if lists[v] is None:
                        _n(counts, 'ops_no_list')
                        continue
                    rec = P.record_from_stations(arr, lists[v], done, src='ops')
                except (ValueError, KeyError, OSError, TypeError):
                    _n(counts, 'ops_unreadable')
                    continue
                out[k, P.sha256(blob)] = (rec['model'], rec)
                _n(counts, 'ops_runs')
    return out

def by_model(records: dict) -> dict:
    out = {}
    for i, (m, rec) in records.items():
        out.setdefault(m, {})[i] = rec
    return out

def _csv_parts(texts: list[str]) -> list[str]:
    groups = {}
    for t in texts:
        head, _, body = t.partition('\n')
        g = groups.setdefault(head, [head + '\n'])
        g.append(body if body.endswith('\n') or not body else body + '\n')
    return [''.join(g) for g in groups.values()]

def iem_inputs(lk, day: str) -> tuple[list, list[str]]:
    ids, texts = ([], [])
    for k in sorted(lk.list(f'{IEM_ROOT}/{day}/')):
        if not k.endswith('.csv.gz'):
            continue
        blob = lk.get_plain(k)
        ids.append((k, P.sha256(blob)))
        texts.append(gzip.decompress(blob).decode('utf-8', 'replace'))
    return (ids, texts)

def metar_table(texts: list[str], day: str, stations: list) -> dict:
    return P.concat(P.obs_schema('metar'), [P.metar_obs_from_csv(t, day, stations) for t in _csv_parts(texts)])
_RAOB_KEY = re.compile('(?P<sid>[A-Z0-9]{11})/(?P<day>\\d{4}-\\d{2}-\\d{2})(?:\\.r(?P<r>\\d+))?\\.txt\\.gz')

def raob_days(lk) -> dict:
    out = {}
    pre = obs_raob.RAW + '/'
    for k in lk.list(pre):
        g = _RAOB_KEY.fullmatch(k[len(pre):])
        if g:
            out.setdefault(g['day'], {}).setdefault(g['sid'], []).append((int(g['r'] or 1), k))
    return out

def raob_elevations(lk, today: date) -> dict:
    try:
        txt = obs_raob.read_stations(lk, today)
    except lakem.LakeError:
        txt = None
    return {s['id']: s.get('elev') for s in obs_raob.parse_stations(txt)} if txt else {}

def raob_inputs(lk, day: str, days: dict, elevs: dict, counts: dict) -> dict:
    out = {}
    for sid, revs in sorted((days.get(day) or {}).items()):
        for _, k in sorted(revs):
            blob = lk.get_plain(k)
            try:
                rows = obs_raob.rows(obs_raob.parse(gzip.decompress(blob).decode('ascii', 'replace'))[0], elevs.get(sid))
            except (ValueError, OSError, EOFError):
                _n(counts, 'raob_unreadable')
                continue
            out[k, P.sha256(blob)] = P.raob_obs(rows)
    return out

def sat_points(stations: list, day: str, hours=SAT_HOURS, area: str=SAT_AREA) -> tuple[list, list]:
    w, s, e, n = {'US': common.US_BOX, 'EU': common.EU_BOX}[area]
    d0 = datetime.fromisoformat(day).replace(tzinfo=UTC)
    pts, ids = ([], [])
    for h in hours:
        t = d0 + timedelta(hours=h)
        for st in stations:
            if s <= float(st[1]) <= n and w <= float(st[2]) <= e:
                pts.append((float(st[1]), float(st[2]), t))
                ids.append(str(st[0]))
    return (pts, ids)

def grid_nodes(box, step: float=P.SAT_SAMPLE) -> tuple[list, np.ndarray, np.ndarray]:
    import numpy as np
    w, s_, e, n = box
    la = np.arange((math.floor(s_ / step) + 0.5) * step, n, step)
    lo = np.arange((math.floor(w / step) + 0.5) * step, e, step)
    LA, LO = np.meshgrid(la, lo, indexing='ij')
    LA, LO = (LA.ravel(), LO.ravel())
    keep = (LA >= s_) & (LO >= w)
    LA, LO = (np.round(LA[keep], 4), np.round(LO[keep], 4))
    return ([P.cell_id(a, b) for a, b in zip(LA, LO)], LA, LO)

def slim_keys(ops, day: str, counts: dict, models=P.MODELS) -> list:
    out = []
    if ops is None:
        return out
    d0 = date.fromisoformat(day)
    for m in models:
        for d in (d0 - timedelta(days=1), d0):
            mk = ops.get(f'{OPS_ROOT}/{m}/grids/{d:%Y%m%d}.json')
            if not mk:
                continue
            try:
                out.append((json.loads(mk)['key'], m))
            except (ValueError, KeyError, TypeError):
                _n(counts, 'slim_unreadable')
    return out

def slim_load(ops, key: str, day: str, counts: dict):
    import numpy as np
    try:
        blob = ops.get(key)
        done = json.loads(ops.get(key[:-len('slim.npz')] + 'done.json') or b'{}')
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        if not run_reaches(P._s(z['run']), day):
            return None
    except (ValueError, KeyError, OSError, TypeError):
        _n(counts, 'slim_unreadable')
        return None
    return (P.sha256(blob), z, done)

def grid_sat(lk, ops, day: str, counts: dict, cells=None, now=None) -> list:
    import sattruth
    res = []
    keys = slim_keys(ops, day, counts)
    if not keys:
        return res
    cells = cells or sattruth.cells_for
    d0 = datetime.fromisoformat(day).replace(tzinfo=UTC)
    nodes, seen_by = ({}, {})
    for area, cfg in SAT_GRID.items():
        nodes[area] = grid_nodes(cfg['box'])
        gk = next((k for k, m in keys if m == cfg['ground']), None)
        got = slim_load(ops, gk, day, counts) if gk else None
        ground = P.slim_ground(got[1]) if got else None
        del got
        w, s_, e, n = cfg['box']
        hours = cfg['hours']

        def table(_new, _all, ground=ground, w=w, s_=s_, e=e, n=n, hours=hours):
            tabs = []
            for h in hours:
                t = d0 + timedelta(hours=h)
                src = next((x for x in sattruth.choose((s_ + n) / 2, ((w + e) / 2 + 180) % 360 - 180, t, now) if x['kind'] == 'geo'), None)
                if src is None:
                    _n(counts, 'grid_no_geo')
                    continue
                cs = cells(src, t, [w, s_, e, n], ground=ground)
                _n(counts, 'grid_cells', len(cs))
                tabs.append(P.sat_cell_obs(cs, int(t.timestamp()), src.get('sensor') or ''))
            return P.concat(P.obs_schema('sat'), tabs)
        batch = f'grid{area}'
        inp = f'satgrid|v{SAT_GRID_V}|{area}|{P.SAT_SAMPLE}|' + '-'.join((str(h) for h in hours))
        try:
            res.append(P.write_unit(lk, 'obs', 'sat', day, batch, [inp], table))
        except lakem.LakeError:
            raise
        except Exception:
            _n(counts, 'grid_failed')
            continue
        seen = {}
        for *_x, mk in P.markers(lk, 'obs', 'sat', day=day, batch=batch):
            body = json.loads(lk.get(mk))
            if body.get('part'):
                seen[mk] = P.from_parquet(P.obs_schema('sat'), lk.get(body['part']), columns=['obs_id', 't_valid', 'status'])
        if seen:
            seen_by[area] = seen
    for key, m in keys:
        if not seen_by:
            break
        got = slim_load(ops, key, day, counts)
        if got is None:
            continue
        _n(counts, 'slim_runs')
        sha, z, done = got
        for area, seen in seen_by.items():
            ids, lat, lon = nodes[area]
            rec = P.slim_at_points(z, done, ids, lat, lon)
            if rec['obs_ids']:
                res.append(P.sat_unit(lk, day, f'g{area}{m}'[:16], {(key, sha): rec}, seen))
        del z, got
    return res

def sat_input(stations: list, hours=SAT_HOURS, area: str=SAT_AREA) -> str:
    return f'sat|v{SAT_V}|{area}|{catalog.stations_version(stations)}|' + '-'.join((str(h) for h in hours))
STATIC_GROUNDS = ('hrrr', 'icond2', 'ecmwf')
GROUND_ROOT = f'derived/static/{P.PV}/ground'

def ground_grids(lk, ops, day: str, counts: dict) -> dict:
    import numpy as np
    out = {}
    for m in STATIC_GROUNDS:
        key = f'{GROUND_ROOT}/{m}.npz.lenc'
        blob = None
        try:
            blob = lk.get(key) if lk.exists(key) else None
        except lakem.LakeError:
            raise
        except Exception:
            blob = None
        if blob is None:
            k = next((k for k, mm in slim_keys(ops, day, counts, models=(m,)) if mm == m), None)
            got = slim_load(ops, k, day, counts) if k else None
            if got is None:
                continue
            z = got[1]
            g = json.loads(P._s(z['grid']))
            enc = z['enc']
            enc = json.loads(P._s(enc)) if not isinstance(enc, dict) else enc
            e = enc.get('elevation')
            if not e:
                continue
            codes = np.asarray(z['elev'], np.uint8)
            vals = np.where(codes == 255, np.nan, float(e[0]) + codes.astype(np.float64) / 254 * (float(e[1]) - float(e[0])))
            buf = io.BytesIO()
            np.savez_compressed(buf, elev=vals.astype(np.float32), west=g['west'], north=g['north'], step=g['step'])
            blob = buf.getvalue()
            lk.put(key, blob, strict=False)
            del got, z
        with np.load(io.BytesIO(blob)) as z:
            out[m] = (np.asarray(z['elev'], np.float64), {'west': float(z['west']), 'north': float(z['north']), 'step': float(z['step'])})
    return out

def write_static(lk, ops, recs: dict, day: str, counts: dict) -> int:
    import numpy as np
    have = set(lk.list(f'derived/static/{P.PV}/model='))
    grounds, n = (None, 0)
    for model, rec in recs.values():
        v = rec.get('stations_v')
        if not v or model not in P.MODELS:
            continue
        try:
            key = P.static_key(model, v)
        except ValueError:
            continue
        if key in have:
            continue
        if grounds is None:
            grounds = ground_grids(lk, ops, day, counts)
        if not grounds:
            _n(counts, 'static_no_ground')
            return n
        t = P.static_rows(rec)
        for g in STATIC_GROUNDS:
            if g not in grounds:
                continue
            tg = P.static_rows(rec, *grounds[g])
            for c in ('relief5km', 'tpi'):
                t[c] = np.where(np.isfinite(t[c]), t[c], tg[c]).astype(t[c].dtype)
        lk.put(key, P.to_parquet(P.STATIC_SCHEMA, t), strict=False)
        have.add(key)
        n += 1
    _n(counts, 'static_written', n)
    return n

def run_day(lk, ops, day: str, stations: list, today: date, kinds=KINDS, snap: dict | None=None, raob_listing=None, elevs=None, truth=None, counts: dict | None=None, sat_due=None, cells=None, raw_listing=None) -> list[dict]:
    counts = counts if counts is not None else {}
    res = []
    recs = {**stcast_records(lk, snap, day, counts), **ops_records(ops, day, counts)}
    models = by_model(recs)
    try:
        write_static(lk, ops, recs, day, counts)
    except lakem.LakeError:
        raise
    except Exception:
        _n(counts, 'static_failed')
    if 'metar' in kinds:
        ids, texts = iem_inputs(lk, day)
        if ids:
            text_of = dict(zip((P._id_of(i) for i in ids), texts))
            ver = f'metar|v{METAR_V}'
            res.append(P.write_unit(lk, 'obs', 'metar', day, 'all', ids + [ver], lambda new, _a: metar_table([text_of[i] for i in (text_of if ver in new else new) if i in text_of], day, stations)))
            for m, rs in sorted(models.items()):
                res.append(P.station_unit(lk, 'metar', day, m, rs))
        else:
            _n(counts, 'metar_no_parts')
    if 'radar' in kinds:
        import numpy as np
        import radar as R

        def radar_table(_new, _all):
            tabs = []
            for tv, r in R.day_rows(day, stations, common.US_BOX, counts=counts):
                n = len(r['obs_id'])
                tabs.append(P.table(P.obs_schema('radar'), {'kind': 'radar', 'obs_id': np.array(r['obs_id'], dtype=object), 't_valid': np.full(n, tv, np.int64), 'lat': r['lat'], 'lon': r['lon'], 'elev': r['elev'], 'review': np.zeros(n, bool), 'o_dbz_max': r['dbz_max'], 'o_storm': r['storm'], 'o_echo': r['echo'], 'q_cover': r['cover'], 'q_dt': r['dt']}, n))
            return P.concat(P.obs_schema('radar'), tabs)
        try:
            res.append(P.write_unit(lk, 'obs', 'radar', day, 'all', [R.input_id(day)], radar_table))
        except lakem.LakeError:
            raise
        except Exception:
            _n(counts, 'radar_failed')
    if 'acwind' in kinds:
        import acwind as A
        try:
            r = A.write_day(lk, day, stations, counts, raw_listing)
            if r.get('status') != 'none':
                res.append(r)
        except lakem.LakeError:
            raise
        except Exception:
            _n(counts, 'acwind_failed')
    if 'window' in kinds:
        try:
            wobs = P.read_kind(lk, 'obs', 'window', [day], purpose='review')
        except lakem.LakeError:
            raise
        except Exception:
            wobs = None
        if wobs is not None and P.n_rows(wobs):
            import hashlib
            cells = sorted({(str(a), int(b)) for a, b in zip(wobs['obs_id'], wobs['t_valid'])})
            oid = 'window|cells|' + hashlib.sha256(json.dumps(cells).encode()).hexdigest()[:16]
            for m, rs in sorted(models.items()):
                res.append(P.window_unit(lk, day, m, rs, {oid: wobs}, stations))
        else:
            _n(counts, 'window_no_obs')
    if 'raob' in kinds:
        snds = raob_inputs(lk, day, raob_listing if raob_listing is not None else raob_days(lk), elevs or {}, counts)
        if snds:
            res.append(P.write_unit(lk, 'obs', 'raob', day, 'all', list(snds), lambda new, _a: P.concat(P.obs_schema('raob'), [P._by_id(snds)[i] for i in new])))
            for m, rs in sorted(models.items()):
                res.append(P.raob_unit(lk, day, m, rs, snds))
    if 'sat' in kinds and (sat_due is None or day in sat_due) and (date.fromisoformat(day) <= today - timedelta(days=SAT_AFTER_D)):
        pts, oids = sat_points(stations, day)
        if pts:
            elev = {str(st[0]): float(st[3]) for st in stations if len(st) > 3 and st[3] is not None}
            grounds = [elev.get(i) for i in oids]

            def sat_table(_new, _all):
                rows = truth(pts) if truth else _truth(pts, grounds)
                _n(counts, 'sat_points', len(rows))
                return P.sat_obs(rows, oids)
            try:
                res.append(P.write_unit(lk, 'obs', 'sat', day, 'all', [sat_input(stations)], sat_table))
            except lakem.LakeError:
                raise
            except Exception:
                _n(counts, 'sat_failed')
            seen = {}
            for *_x, mk in P.markers(lk, 'obs', 'sat', day=day, batch='all'):
                body = json.loads(lk.get(mk))
                if body.get('part'):
                    seen[mk] = P.from_parquet(P.obs_schema('sat'), lk.get(body['part']), columns=['obs_id', 't_valid', 'status'])
            if seen:
                for m, rs in sorted(models.items()):
                    res.append(P.sat_unit(lk, day, m, rs, seen))
        if os.environ.get('SAT_GRID', '1') != '0':
            res += grid_sat(lk, ops, day, counts, cells)
    if 'snap' in kinds and recs:
        import snap as S
        res.append(S.write_day(lk, day, 'all', recs))
    return res

def _truth(points: list, grounds=None) -> list:
    import sattruth
    return sattruth.truth(points, grounds=grounds)

def hindcast_span(today: date, back: int=BACK) -> str:
    import labingest
    first = labingest.hindcast_span(today).split(':')[0]
    return f'{first}:{(today - timedelta(days=back + 1)).isoformat()}'

def days_due(today: date, back: int=BACK, span: str | None=None) -> list[str]:
    if span:
        a, _, b = span.partition(':')
        d0, d1 = (date.fromisoformat(a), date.fromisoformat(b or a))
        if d1 < d0:
            d0, d1 = (d1, d0)
        return [(d1 - timedelta(days=k)).isoformat() for k in range((d1 - d0).days + 1)]
    return [(today - timedelta(days=k)).isoformat() for k in range(1, back + 1)]

def run(lk, ops, days: list[str], stations: list, today: date, kinds=KINDS, minutes: float=MINUTES, clock=time.monotonic, truth=None, sat_all: bool=False, cells=None) -> dict:
    import labbuild
    t0 = clock()
    counts: dict = {}
    snap = catalog.pinned(lk)
    if snap is None:
        _n(counts, 'no_pin')
    listing = raob_days(lk) if 'raob' in kinds else {}
    elevs = raob_elevations(lk, today) if 'raob' in kinds else {}
    old_enough = [d for d in sorted(days, reverse=True) if date.fromisoformat(d) <= today - timedelta(days=SAT_AFTER_D)]
    sat_due = None if sat_all else set(old_enough[:SAT_BACK])
    raw_listing = None
    if 'acwind' in kinds:
        import acwind as A
        raw_listing = A.listing(lk)
    res = []
    for d in days:
        if clock() - t0 > minutes * 60:
            _n(counts, 'days_left', len(days) - days.index(d))
            break
        res += run_day(lk, ops, d, stations, today, kinds, snap, listing, elevs, truth, counts, sat_due, cells, raw_listing)
        _n(counts, 'days')
    return {'counts': counts, 'summary': P.summary(res) if res else 'pairs: nothing to do'}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The pair job: each day's pairs, observations and snapshot")
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('run')
    a.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    a.add_argument('--ops-local', help='a folder as OPS (else OPS_S3_ENDPOINT, OPS_S3_BUCKET, OPS_RO_*)')
    a.add_argument('--days', help='YYYY-MM-DD[:YYYY-MM-DD]: these days instead of the last BACK; "hindcast": the station hindcast\'s (labingest.hindcast_span)')
    a.add_argument('--status', help='a file to write {"more": true|false} to: days left for time (a backfill\'s relay)')
    a.add_argument('--back', type=int, default=BACK)
    a.add_argument('--kinds', default=','.join(KINDS))
    a.add_argument('--minutes', type=float, default=MINUTES)
    a.add_argument('--day', help='today, YYYY-MM-DD (default: today, UTC)')
    a.add_argument('--sat-all', action='store_true', help='the satellite for every day named, not the newest SAT_BACK')
    x = ap.parse_args(argv)
    kinds = tuple((k for k in x.kinds.split(',') if k))
    if set(kinds) - set(KINDS):
        print('pairjob: unknown kind', flush=True)
        return 2
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('pairjob: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    ops = ops_store(local=x.ops_local)
    if ops is None:
        print("pairjob: no OPS read secrets: the station hindcast's runs alone", flush=True)
    today = date.fromisoformat(x.day) if x.day else datetime.now(UTC).date()
    import labbuild
    span = x.days
    if span == 'hindcast':
        span = hindcast_span(today, x.back)
    try:
        out = run(lk, ops, days_due(today, x.back, span), labbuild.stations(), today, kinds, x.minutes, sat_all=x.sat_all)
    except lakem.LakeError:
        print('pairjob: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    if x.status:
        with open(x.status, 'w', encoding='utf-8') as f:
            json.dump({'more': bool(out['counts'].get('days_left'))}, f)
    print('pairjob: ' + ', '.join((f'{k} {v}' for k, v in sorted(out['counts'].items()))), flush=True)
    print(out['summary'], flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
