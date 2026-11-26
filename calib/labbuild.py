from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog
import common
import day as dayb
import lake as lakem
UnitError = catalog.UnitError
KINDS, MAX_TRIES = (catalog.KINDS, catalog.MAX_TRIES)
STATIONS, VERIFY = (catalog.STATIONS, catalog.VERIFY)
check_unit, logical_id, unit_of, model_id_of = (catalog.check_unit, catalog.logical_id, catalog.unit_of, catalog.model_id_of)
cs_now = catalog.source_stamp
load_items = catalog.read_items
UNIT_V = 1
START_LIMIT_S = 40 * 60
HARD_S = 85 * 60
_RUN = catalog._RUN
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

def _verify():
    if common.PIPELINE not in sys.path:
        sys.path.append(common.PIPELINE)
    import verify
    return verify
STATION_FIELDS = _verify().FIELDS + ('blapse', 'u_blavg', 'v_blavg', 'u_bltop', 'v_bltop', 'u_g300', 'v_g300', 'sun', 'ctop', 'cloud5', 'cloud10', 'ridge') + _verify().SMOOTHED

def src_of(model_id: str, region: str) -> str:
    base = common.base_model(model_id)
    if base in ('icon', 'icond2', 'arome') or (base == 'ecmwf' and common.is_lead(model_id)):
        return 'replay'
    return 'rerun'

def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            zi = zipfile.ZipInfo(f'{name}.npy', date_time=_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 384 << 16
            b = io.BytesIO()
            np.lib.format.write_array(b, np.asanyarray(arrays[name]), allow_pickle=False)
            zf.writestr(zi, b.getvalue(), compresslevel=6)
    return buf.getvalue()

def read_unit(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for k in ('meta', 'score', 'st.enc'):
        if k in out:
            out[k] = json.loads(str(out[k]))
    return out
_STATIONS = {}

def sample_v(stations_v: str) -> str:
    return hashlib.sha256(f'{common._version_of([VERIFY])}|{stations_v}'.encode()).hexdigest()[:12]

def stations() -> list:
    if 's' not in _STATIONS:
        with open(STATIONS) as f:
            _STATIONS['s'] = json.load(f)['stations']
    return _STATIONS['s']

def station_cells(grid: dict, sts: list):
    w, n, step, nx, ny = (grid['west'], grid['north'], grid['step'], grid['nx'], grid['ny'])
    idx, iy, ix = ([], [], [])
    for i, (_, lat, lon, _) in enumerate(sts):
        x, y = (round((lon - w) / step), round((n - lat) / step))
        if 0 <= x < nx and 0 <= y < ny:
            (idx.append(i), iy.append(y), ix.append(x))
    return (np.array(idx, np.int32), np.array(iy, np.int64), np.array(ix, np.int64))

def _store_enc(name: str):
    try:
        import build
    except ImportError:
        return None
    e = build.ENC.get(name) if name in build.NOT_TILES else None
    return [e[0], e[1], e[2]] if e else None

def unit_arrays(base: str, u: dict, model_id: str, bbox: list, gv: dict, cs: str, keep_grids: bool=True) -> dict:
    m = common.read_json(os.path.join(base, 'latest.json'))
    grid = {k: m['grid'][k] for k in ('west', 'north', 'step', 'nx', 'ny')}
    hours = sorted((f[:-4] for f in os.listdir(os.path.join(base, '_hours')) if f.endswith('.npz') and (not f.endswith('.tmp.npz'))))
    if not hours:
        raise UnitError('nohours')
    runs = []
    for s in hours:
        try:
            with open(os.path.join(base, '_hours', f'{s}.run')) as f:
                r = (f.read().split() or [''])[0]
        except OSError:
            r = ''
        runs.append(r if _RUN.fullmatch(r) else str(m.get('runId') or ''))
    lead_h = [int((datetime.strptime(s[:10], '%Y%m%d%H') - datetime.strptime(r, '%Y%m%d%H')).total_seconds() // 3600) if _RUN.fullmatch(r) else None for s, r in zip(hours, runs)]
    names = sorted(common.FIELD_GROUP)
    grids = {f: np.full((len(hours), grid['ny'], grid['nx']), 255, np.uint8) for f in names}
    have = set()
    verify = _verify()
    sts = stations()
    idx, iy, ix = station_cells(grid, sts)
    r_nb, rs = (verify.near_cells(grid['step']), verify.smooth_cells(grid['step']))
    codes = np.full((len(hours), len(idx), len(STATION_FIELDS)), 255, np.uint8)
    for k, s in enumerate(hours):
        with np.load(os.path.join(base, '_hours', f'{s}.npz')) as h:
            for f in names:
                if f in h.files:
                    a = np.asarray(h[f], np.uint8)
                    if a.shape != (grid['ny'], grid['nx']):
                        raise UnitError('shape')
                    grids[f][k] = a
                    have.add(f)
            if len(idx):
                codes[k] = verify.sample_hour(h, iy, ix, r_nb, STATION_FIELDS, rs)
    fields = m.get('fields') or {}
    enc = {f: [fields[f].get('min'), fields[f].get('max'), fields[f].get('transform')] for f in sorted(fields) if f in common.FIELD_GROUP or f in STATION_FIELDS}
    for f in sorted(have - set(enc)):
        e = _store_enc(f)
        if e:
            enc[f] = e
    st_enc = {f: enc[f] for f in STATION_FIELDS if f in enc}
    if 'overdev' in st_enc:
        st_enc['overdev_nb'] = st_enc['overdev']
    for f in verify.SMOOTHED:
        if f[:-2] in st_enc:
            st_enc[f] = st_enc[f[:-2]]
    has = (codes[:, :, STATION_FIELDS.index('t2')] != 255).any(axis=0) if len(idx) else np.zeros(0, bool)
    sv = hashlib.sha256(json.dumps(sts, separators=(',', ':')).encode()).hexdigest()[:12]
    area = common.gv_area(u['model'], u['box'])
    init = str(m.get('runId') or (max(set(runs), key=runs.count) if runs else ''))
    gv = {g: gv.get(g) for g in common.GROUPS}
    smp = sample_v(sv)
    score = {'sv': common.score_version(gv), 'score_v': common.SCORE_VERSION, 'corr_v': None}
    meta = {'v': UNIT_V, 'id': u, 'model_id': model_id, 'area': area, 'macro': common.macro_of(u['box']), 'bbox': bbox, 'grid': grid, 'init': init, 'runs': runs, 'stamps': hours, 'lead_h': lead_h, 'src': src_of(model_id, u['box']), 'cs': cs, 'gv': gv, 'physics_v': common.model_physics(model_id, u['box']), 'score': score, 'sample_v': smp, 'enc': enc, 'absent': [f for f in names if f not in have] if keep_grids else names, 'grids': keep_grids}
    out = {'meta': np.array(json.dumps(meta, sort_keys=True, separators=(',', ':')))}
    if keep_grids:
        for f in sorted(have):
            out[f'g.{common.FIELD_GROUP[f]}.{f}'] = grids[f]
    out.update({'st.stamps': np.array(hours), 'st.idx': idx[has], 'st.codes': codes[:, has, :], 'st.fields': np.array(STATION_FIELDS), 'st.enc': np.array(json.dumps(st_enc, sort_keys=True)), 'st.model': np.array(u['model']), 'st.run': np.array(init), 'st.runs': np.array(runs), 'st.lead_h': np.array([-1 if x is None else x for x in lead_h], np.int32), 'st.stations_v': np.array(sv), 'st.sample_v': np.array(smp), 'st.selected_by': np.array('collector' if u['kind'] == 'collector' else 'flight')})
    return out

def score_of(B: dict, model_id: str) -> dict:
    m = model_id
    items = []
    for k, f in enumerate(B['flights']):
        e = {'k': k, 'kind': f.get('kind') or 'flight', 'id': f.get('id'), 'sid': f.get('sid')}
        for sfx, pk in (('', 'probe_m'), ('_c', 'probe_mc')):
            if f'model{sfx}_{m}' not in f:
                continue
            e[f'takeoff{sfx}'] = f[f'model{sfx}_{m}'].get('takeoffNode')
            e[f'legs{sfx}'] = [(L.get('models' + sfx) or {}).get(m) for L in f.get('legs') or []]
            e[f'climbs{sfx}'] = [(c.get('models' + sfx) or {}).get(m) for c in f.get('climbs') or []]
            e[f'probes{sfx}'] = (f.get(pk) or {}).get(m)
        items.append(e)
    return {'model_id': m, 'items': items}

class Steps:
    workers = 2

    def build(self, day: str, B: dict, model_id: str, work: str, timeout: float) -> str | None:
        out = os.path.join(work, 'box')
        shutil.rmtree(out, ignore_errors=True)
        cmd = dayb.build_command(day, B, out, model_id, self.workers, os.path.dirname(os.path.abspath(work)))
        if cmd is None:
            raise UnitError('outside')
        logf = common.PRIVATE_LOG or os.path.join(work, 'day.log')
        env = dict(os.environ)
        env.setdefault('CONTRAILS', '0')
        if timeout <= 60 or not dayb._run(cmd, common.PIPELINE, timeout, logf, env):
            raise UnitError(dayb.fail_kind(dayb.LAST_FAIL['why']) if timeout > 60 else 'timeout')
        base = os.path.join(out, common.base_model(model_id))
        return base if os.path.exists(os.path.join(base, 'latest.json')) else None

    def score(self, base: str, B: dict, model_id: str, work: str, timeout: float) -> bool:
        info = {}
        logf = common.PRIVATE_LOG or os.path.join(work, 'day.log')
        was = dayb.EVAL_TIMEOUT
        dayb.EVAL_TIMEOUT = min(was, timeout)
        try:
            return dayb._eval_box(base, B, work, model_id, 0, info, logf, None)
        finally:
            dayb.EVAL_TIMEOUT = was

    def hindcast(self, model: str, run: str, region: list, work: str, timeout: float, area: str) -> str | None:
        import stationcast
        out = os.path.join(work, 'stcast')
        shutil.rmtree(out, ignore_errors=True)
        cmd = stationcast.build_command(model, run, region, out, stationcast.workers(model), area)
        logf = common.PRIVATE_LOG or os.path.join(work, 'day.log')
        env = stationcast.build_env(os.environ)
        if timeout <= 60 or not dayb._run(cmd, common.PIPELINE, timeout, logf, env):
            raise UnitError(dayb.fail_kind(dayb.LAST_FAIL['why']) if timeout > 60 else 'timeout')
        hours = os.path.join(out, model, '_hours')
        return hours if os.path.isdir(hours) else None

def consumers_for(gv: dict) -> list:
    return [c for c in sorted(common.CONSUMERS) if not common.unknown_groups(gv, [c])]

def _stored(lk, key: str, lid: str) -> tuple[str, int, dict] | None:
    blob = lk.store.get(key)
    if blob is None:
        return None
    try:
        meta = read_unit(lakem.unseal(blob, key, lk.keys)).get('meta') or {}
        if logical_id(meta.get('id') or {}) != lid:
            return None
    except Exception:
        return None
    return (hashlib.sha256(blob).hexdigest(), len(blob), meta)

def _gv6(gv) -> dict:
    gv = gv if isinstance(gv, dict) else {}
    return {g: gv.get(g) for g in common.GROUPS}

def _from_object(row: dict, key: str, have: tuple) -> None:
    sha, n, meta = have
    row.update(key=key, sha256=sha, bytes=n, put='kept', cs=meta.get('cs'), gv=_gv6(meta.get('gv')), run=meta.get('init'), lead_h=meta.get('lead_h'), bbox=meta.get('bbox'), src=meta.get('src'), sv=(meta.get('score') or {}).get('sv'))

def _hours_kept(u: dict, meta: dict) -> bool:
    import stationcast
    want = stationcast.want_leads(u['model'], int(u['init'][8:]), u['box'])
    return list(meta.get('lead_h') or []) == want

def _prev_keeps(prev, have, gv: dict, cons: list) -> bool:
    sha, _, meta = have
    if prev.get('sha256') and prev['sha256'] != sha:
        return False
    if prev.get('gv') is not None and _gv6(prev['gv']) != _gv6(meta.get('gv')):
        return False
    return isinstance(meta.get('gv'), dict) and (not common.unit_stale(meta['gv'], gv, cons))

def blank_row(cs: str, wave) -> dict:
    row = {k: None for k in catalog.ROW_FIELDS + catalog.EXTRA_FIELDS}
    row.update(cs=cs, tries=0, sec=0.0, wave=wave, **{'try': 0})
    return row

def _identify(row: dict, u: dict, model_id: str) -> str:
    area = common.gv_area(u['model'], u['box'])
    row.update(kind=u['kind'], day=common.day_of(u['daykey']), area=area, model=u['model'], model_id=model_id, init=u['init'], box=u['box'], src=catalog.STCAST if u['kind'] == catalog.STCAST else src_of(model_id, u['box']))
    return area

def cut_row(lk, entry, cs: str, wave, why: str='cancel') -> dict:
    row = blank_row(cs, wave)
    try:
        row['tries'] = max(0, int(entry.get('tries') or 0))
        u = check_unit(entry.get('unit'))
        _identify(row, u, model_id_of(u))
        row['unit'] = lk.name(logical_id(u))
    except Exception:
        pass
    row.update(status='retry', why=why, **{'try': 0})
    return row

class Cancelled(KeyboardInterrupt):
    pass

def _on_term(signum, frame):
    raise Cancelled()

def run_unit(lk, entry: dict, cs: str, gv_by_area: dict, steps: Steps, work: str, wave=None, hard_at: float | None=None, clock=time.monotonic) -> dict:
    t0 = clock()
    row = blank_row(cs, wave)
    tries0 = 0

    def done(status, why=None, tried=False):
        row['status'], row['why'] = (status, why)
        row['try'] = 1 if tried else 0
        if tried:
            row['tries'] = tries0 + 1
            if status != 'ok':
                row['status'] = 'failed' if row['tries'] >= MAX_TRIES else 'retry'
        row['sec'] = round(clock() - t0, 1)
        return row

    def near_stop():
        return hard_at is not None and clock() >= hard_at - 60
    try:
        if not isinstance(entry, dict):
            raise UnitError('unit')
        try:
            tries0 = int(entry.get('tries') or 0)
        except (TypeError, ValueError):
            raise UnitError('unit') from None
        if tries0 < 0:
            raise UnitError('unit')
        row['tries'] = tries0
        u = check_unit(entry.get('unit'))
        lid = logical_id(u)
        model_id = model_id_of(u)
    except UnitError as e:
        return done('skipped', str(e))
    area = _identify(row, u, model_id)
    try:
        name = lk.name(lid)
        if entry.get('name') not in (None, name):
            raise UnitError('name')
        key = catalog.unit_key(name, cs)
        row['unit'] = name
        have = _stored(lk, key, lid)
        if have:
            _from_object(row, key, have)
            return done('ok')
        gv = (gv_by_area or {}).get(area) or {}
        if not isinstance(gv, dict) or not any(gv.values()):
            return done('skipped', 'nostamp')
        prev = entry.get('prev')
        cons = consumers_for(gv)
        if u['kind'] == catalog.STCAST:
            cons = [c for c in cons if c in catalog.stcast_consumers(gv)]
        if isinstance(prev, dict) and prev.get('key') and cons:
            try:
                pkey = lakem.check_key(str(prev['key']))
            except lakem.LakeError:
                pkey = None
            have = _stored(lk, pkey, lid) if pkey and pkey != key else None
            if have and u['kind'] == catalog.STCAST and (not _hours_kept(u, have[2])):
                have = None
            if have and _prev_keeps(prev, have, gv, cons):
                _from_object(row, pkey, have)
                return done('ok')
        if tries0 >= MAX_TRIES:
            return done('failed', 'tries')
        row['gv'] = _gv6(gv)
        row['key'] = key
    except UnitError as e:
        return done('skipped', str(e))
    except lakem.LakeError:
        return done('error', 'lake', tried=True)
    except Exception as e:
        return done('error', type(e).__name__, tried=True)
    d = common.day_of(u['daykey'])
    wd = None

    def _put(arrays):
        meta = json.loads(str(arrays['meta']))
        row.update(lead_h=meta['lead_h'], run=meta['init'], sv=(meta.get('score') or {}).get('sv'))
        res = lk.put(key, npz_bytes(arrays), strict=False)
        row['sha256'], row['bytes'], row['put'] = (res.sha256, res.bytes, res.status)
        if res.status == 'differs':
            have = _stored(lk, key, lid)
            if have:
                _from_object(row, key, have)
                row['put'] = 'differs'
        return done('ok', tried=True)
    try:
        wd = tempfile.mkdtemp(prefix='u-', dir=work)
        if u['kind'] == catalog.STCAST:
            if u.get('model') in _om_models():
                import omruns
                arrays = omruns.unit_arrays(u, _gv6(gv), cs, stations(), hard_at=hard_at, clock=clock)
            else:
                arrays = _hindcast(u, gv, cs, steps, wd, hard_at, clock)
                if arrays is None:
                    return done('retry', 'hardstop')
            row['bbox'] = json.loads(str(arrays['meta']))['bbox']
            catalog.keep_stations(lk, stations(), str(arrays['stations_v']))
            return _put(arrays)
        items = [f for f in load_items(lk, entry.get('raw') or []) if (f.get('kind') == 'collector') == (u['kind'] == 'collector')]
        B = next((b for b in dayb.boxes(items) if b['region'] == u['box']), None)
        if B is None:
            return done('error', 'nobox', tried=True)
        B = copy.deepcopy(B)
        row['bbox'] = B['bbox']
        left = hard_at - clock() if hard_at is not None else dayb.BUILD_TIMEOUT
        if left < 120:
            return done('retry', 'hardstop')
        try:
            base = steps.build(d, B, model_id, wd, min(dayb.BUILD_TIMEOUT, left))
        except UnitError as e:
            if str(e) == 'outside':
                return done('outside')
            raise
        if not base:
            raise UnitError('build')
        keep = not common.is_lead(model_id) or common.base_model(model_id) in dayb.LAB_LEAD_GRIDS
        arrays = unit_arrays(base, u, model_id, B['bbox'], gv, cs, keep)
        left = hard_at - clock() if hard_at is not None else dayb.EVAL_TIMEOUT
        if not steps.score(base, B, model_id, wd, min(dayb.EVAL_TIMEOUT, left)):
            raise UnitError('evaluate')
        arrays['score'] = np.array(json.dumps(score_of(B, model_id), sort_keys=True, separators=(',', ':')))
        return _put(arrays)
    except UnitError as e:
        if near_stop() and str(e) in ('timeout', 'evaluate', 'build', 'time'):
            return done('retry', 'hardstop')
        return done('error', str(e), tried=True)
    except lakem.LakeError:
        return done('error', 'lake', tried=True)
    except Exception as e:
        return done('error', type(e).__name__, tried=True)
    finally:
        if wd:
            shutil.rmtree(wd, ignore_errors=True)

def _om_models() -> tuple:
    import omruns
    return tuple(omruns.MODELS)

def _hindcast(u: dict, gv: dict, cs: str, steps: Steps, wd: str, hard_at, clock) -> dict | None:
    import stationcast
    region = stationcast.area_region(u['model'], u['box'])
    left = hard_at - clock() if hard_at is not None else stationcast.BUILD_TIMEOUT
    if left < 120:
        return None
    hours = steps.hindcast(u['model'], u['init'], region, wd, min(stationcast.BUILD_TIMEOUT, left), u['box'])
    if not hours:
        raise UnitError('build')
    return stationcast.unit_arrays(hours, u, region, _gv6(gv), cs, stations(), wd)

def run_chunk(lk, plan_id: str, chunk: int, steps: Steps | None=None, work: str | None=None, cs: str | None=None, clock=time.monotonic, wall=None, elapsed: float=0.0) -> dict:
    steps = steps or Steps()
    wall = wall or (lambda: datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    t0 = clock() - max(0.0, float(elapsed or 0))
    plan = catalog.load_plan(lk, plan_id)
    sweep, wave = (str(plan['sweep']), int(plan['wave']))
    chunks = plan.get('chunks') or []
    if not 0 <= chunk < len(chunks):
        raise UnitError('chunk')
    cs = cs or catalog.source_stamp()
    body = {'v': 1, 'plan': plan_id, 'sweep': sweep, 'wave': wave, 'chunk': chunk, 'cs': cs, 'started': wall(), 'rows': [], 'left': 0, 'stopped': False}
    entries = chunks[chunk]
    try:
        old = signal.signal(signal.SIGTERM, _on_term)
    except ValueError:
        old = None
    cancel = None
    try:
        catalog.mark_start(lk, sweep, wave, chunk)
        if plan.get('cs') != cs:
            common.log('labbuild: the plan was made for other code: nothing built (the next wave plans again)')
            body['left'], body['refused'] = (len(entries), 'cs')
        else:
            own = work is None
            work = work or tempfile.mkdtemp(prefix='lab-')
            os.makedirs(work, exist_ok=True)
            running = None
            try:
                for i, entry in enumerate(entries):
                    if clock() - t0 >= START_LIMIT_S:
                        body['left'], body['stopped'] = (len(entries) - i, True)
                        break
                    running = i
                    try:
                        row = run_unit(lk, entry, cs, plan.get('gv') or {}, steps, work, wave, t0 + HARD_S, clock)
                    except Exception as e:
                        row = {**blank_row(cs, wave), 'status': 'error', 'why': type(e).__name__, 'try': 1}
                    body['rows'].append(row)
                    running = None
            except KeyboardInterrupt as e:
                if clock() - t0 >= HARD_S:
                    raise
                cancel = e
                if running is not None:
                    body['rows'].append(cut_row(lk, entries[running], cs, wave))
                body['left'], body['stopped'] = (len(entries) - len(body['rows']), 'cancelled')
            finally:
                if own:
                    shutil.rmtree(work, ignore_errors=True)
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
    if cancel is not None:
        was = [signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)]
        try:
            for sg in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sg, signal.SIG_IGN)
        except ValueError:
            was = None
        try:
            body['ended'] = wall()
            body['key'] = catalog.write_rows(lk, sweep, wave, chunk, body)
        finally:
            if was is not None:
                for sg, h in zip((signal.SIGINT, signal.SIGTERM), was):
                    signal.signal(sg, h)
        common.log(f"labbuild: chunk {chunk} cancelled: {len(body['rows'])} rows written, {body['left']} not started")
        raise cancel
    body['ended'] = wall()
    body['key'] = catalog.write_rows(lk, sweep, wave, chunk, body)
    n = {}
    for r in body['rows']:
        n[r['status']] = n.get(r['status'], 0) + 1
    kept = sum((r.get('put') == 'kept' for r in body['rows']))
    common.log(f"labbuild: chunk {chunk}: {len(body['rows'])} units (" + ', '.join((f'{k} {v}' for k, v in sorted(n.items()))) + f"; {kept} kept as stored), {body['left']} not started" + (' (the time limit)' if body['stopped'] else ''))
    common.log(lk.summary())
    return body

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('chunk')
    c.add_argument('--plan', required=True)
    c.add_argument('--chunk', type=int, required=True)
    c.add_argument('--work')
    c.add_argument('--local', help='a folder as the lake (tests, a dry run)')
    c.add_argument('--started', type=float, help="the job's start, epoch seconds (the limits count from it)")
    a = ap.parse_args(argv)
    lk = lakem.from_env('LAKE', a.local)
    if lk is None or 1 not in lk.keys.ids():
        common.log('labbuild: no lake secrets or no CALIB_DATA_KEY: nothing done')
        return 0
    try:
        run_chunk(lk, a.plan, a.chunk, work=a.work, elapsed=time.time() - a.started if a.started else 0.0)
    except (UnitError, lakem.LakeError, ValueError, KeyError) as e:
        common.log(f"labbuild: the chunk stopped ({type(e).__name__}{(': ' + str(e) if isinstance(e, UnitError) else '')})")
        return 1
    return 0
if __name__ == '__main__':
    sys.exit(main())
