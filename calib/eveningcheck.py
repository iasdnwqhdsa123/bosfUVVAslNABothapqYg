from __future__ import annotations
import argparse
import contextlib
import csv
import glob
import gzip
import json
import math
import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import common
import report
import rows
import track
SWEEP = [{'keep': k, 'cloudKeep': c, 'shift': s} for k in (0.8, 0.85, 0.9, 0.95) for c in (0.4, 0.7, 1.0) for s in (-2, 0, 2)] + [{'ziKeep': z, 'top': True} for z in (0.9, 0.95, 1.0)]
SWEEP_WS = [f'ws_s{i}' for i in range(len(SWEEP))]
SWEEP_TOP = [f'top_s{i}' for i, o in enumerate(SWEEP) if o.get('top')]
FROM = report.EVENING_FROM
CSV_KEYS = ('date', 'region', 'macro', 'model', 'rate', 'agl1', 'm_solar', 'm_sunElev', 'm_relief', 'm_sun', 'm_ws0', 'm_top') + tuple((f'm_{k}' for k in track.EVENING_KEYS)) + tuple((f'm_{k}' for k in SWEEP_WS + SWEEP_TOP))

def _key_of(rec: dict) -> str:
    return rec.get('key') or common.day_key(rec['date'], rec.get('macro') or 'US')

def _primary(rec: dict) -> str | None:
    ms = rec.get('models') or []
    return ms[0] if ms else rec.get('model')

@contextlib.contextmanager
def _sweep_rows():
    keep = rows.MODEL_KEYS
    rows.MODEL_KEYS = tuple(keep) + tuple(SWEEP_WS + SWEEP_TOP)
    try:
        yield
    finally:
        rows.MODEL_KEYS = keep

def cut(res: dict) -> dict:
    model = _primary(res)
    with _sweep_rows():
        _, C, _ = rows.tables([res], model)
    H, _ = rows.survival([res], model)
    keep = {'paired', 'engon', 'rate', 'agl1', 'date', 'region'} | {f'm_{k}' for k in rows.MODEL_KEYS} | {f'm_{k}' for k in SWEEP_WS + SWEEP_TOP}
    C = [{k: v for k, v in r.items() if k in keep} for r in C]
    hk = {'paired', 'out', 'solar', 'month', 'lh', 'region'} | {f'm_{k}' for k in track.PROBE_KEYS}
    H = [{k: v for k, v in r.items() if k in hk} for r in H]
    return {'day': res.get('key') or res['date'], 'model': model, 'macro': res.get('macro'), 'climbs': C, 'hours': H, 'boxes': [{k: b.get(k) for k in ('model', 'box', 'ok', 'error')} for b in res.get('boxes') or []]}

def _one(fn: str, grids: str, work: str) -> tuple[str, str, dict | None]:
    import day as dayrun
    rec = common.read_json(fn)
    d = _key_of(rec)
    model = _primary(rec)
    if not model:
        return (d, 'no model', None)
    os.environ['CALIB_EVENING_SWEEP'] = json.dumps(SWEEP)
    try:
        res = dayrun.rescore_day({**rec, 'models': [model]}, grids, os.path.join(work, d), any_physics=True)
    except Exception as e:
        return (d, f'error: {type(e).__name__}', None)
    ok = [b for b in res['boxes'] if b.get('ok')]
    if not ok:
        return (d, 'no box: ' + ','.join(sorted({str(b.get('error')) for b in res['boxes']})), None)
    return (d, 'ok', cut(res))

def run(a) -> int:
    os.makedirs(os.path.join(a.out, 'rows'), exist_ok=True)
    done_p = os.path.join(a.out, 'done.json')
    done = common.read_json(done_p) if os.path.exists(done_p) else {}
    todo = []
    for fn in sorted(glob.glob(os.path.join(a.data, 'days', '*.json.gz'))):
        rec = common.read_json(fn)
        d, model = (_key_of(rec), _primary(rec))
        if d in done or not model or (not glob.glob(os.path.join(a.grids, d, f'{model}-box*'))):
            continue
        todo.append(fn)
    common.log(f'evening check: {len(todo)} days to score ({len(done)} done before)')
    t0, n = (time.time(), 0)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        pending, it = (set(), iter(todo))

        def submit():
            fn = next(it, None)
            if fn is not None and time.time() - t0 < a.minutes * 60:
                pending.add(ex.submit(_one, fn, a.grids, a.work))
        for _ in range(a.workers):
            submit()
        while pending:
            f = next(as_completed(pending))
            pending.discard(f)
            d, status, cutd = f.result()
            done[d] = status
            if cutd:
                common.write_json(os.path.join(a.out, 'rows', f'{d}.json.gz'), cutd)
            n += 1
            if n % 25 == 0:
                common.write_json(done_p, done)
                common.log(f"evening check: {n} scored ({sum((v == 'ok' for v in done.values()))} ok in all), {(time.time() - t0) / 60:.0f} min")
            submit()
    common.write_json(done_p, done)
    left = len(todo) - n
    common.log(f'evening check: {n} scored this run, {left} left')
    with open(os.path.join(a.out, 'left'), 'w') as f:
        f.write(str(left))
    return 0

def _sweep_section(C: list[dict]) -> list[str]:
    cl = report._evening_climbs(C)
    mid = [r['rate'] / r['m_ws0'] for r in cl if report.EVENING_MID[0] <= r['m_solar'] < report.EVENING_MID[1] and r['m_ws0'] > 0.3]
    ev = [r for r in cl if r['m_solar'] >= FROM and r.get('m_ws_s0') is not None and any((r.get(f'm_{k}') is not None and abs(r[f'm_{k}'] - r['m_ws0']) > 0.05 for k in SWEEP_WS))]
    out = ["## The evening state's constants (the sweep)", '']
    if len(mid) < 50 or len(ev) < 30:
        return out + [f'- Too few climbs to rank them yet (midday {len(mid)}, afternoon {len(ev)}).', '']
    k = st.median(mid)

    def score(rs, col):
        xs = [math.log(r['rate'] / (k * max(r[col], 0.1))) for r in rs if r.get(col) is not None]
        return (st.median([abs(x) for x in xs]), math.exp(st.median(xs)), len(xs)) if xs else (math.inf, math.nan, 0)
    site = [i for i, key in enumerate(SWEEP_WS) if all((r.get(f'm_{key}') == r.get('m_ws_e2') for r in ev))]
    ranked = sorted(range(len(SWEEP_WS)), key=lambda i: score(ev, f'm_{SWEEP_WS[i]}')[0])
    label = lambda i: ', '.join((f'{p} {v}' for p, v in SWEEP[i].items() if p != 'top'))
    out += [f"Each entry's W* against the {len(ev)} afternoon climbs (solar {FROM} h on, where any entry differs from the forecast's own): |ln(flown ÷ (k × W*))| (median; less is better) and flown ÷ (k × W*) (median; 1: right). The site's: " + (', '.join((f's{i} ({label(i)})' for i in site)) or 'none of them (its constants moved)') + '.', '', '| rank | entry | |ln| | flown ÷ (k × W*) | n |', '|---|---|---|---|---|']
    for rank, i in enumerate(ranked, 1):
        if rank <= 10 or i in site:
            e, ratio, n = score(ev, f'm_{SWEEP_WS[i]}')
            out.append(f"| {rank} | s{i}: {label(i)}{(' (the site’s)' if i in site else '')} | {e:.3f} | {ratio:.2f} | {n} |")
    out.append('')
    month = lambda r: int((r.get('date') or '0000-00')[5:7] or 0)
    groups = [(g, rs) for g, rs in report.by(ev, lambda r: report.SEASON.get(month(r), '–')) if len(rs) >= 30]
    if groups:
        out += ['The best entry by season:', '', "| season | climbs | best | |ln| | the site's |ln| |", '|---|---|---|---|---|']
        for g, rs in groups:
            b = min(range(len(SWEEP_WS)), key=lambda i: score(rs, f'm_{SWEEP_WS[i]}')[0])
            s_ = f"{score(rs, f'm_{SWEEP_WS[site[0]]}')[0]:.3f}" if site else '–'
            out.append(f"| {g} | {len(rs)} | s{b}: {label(b)} | {score(rs, f'm_{SWEEP_WS[b]}')[0]:.3f} | {s_} |")
        out.append('')
    tops = [(i, f'm_top_s{i}') for i, o in enumerate(SWEEP) if o.get('top')]
    if tops:
        out += ["The depth's keep an hour (the rest the site's): a climb's top over the ground − the usable top (m; median).", '', '| entry | top − usable top |', '|---|---|']
        for i, col in tops:
            out.append(f"| s{i}: {label(i)} | {report.med([r['agl1'] - r[col] for r in ev if r.get('agl1') is not None and r.get(col) is not None])} |")
        out.append('')
    return out

def build_report(out: str) -> str:
    cuts = [common.read_json(fn) for fn in sorted(glob.glob(os.path.join(out, 'rows', '*.json.gz')))]
    done_p = os.path.join(out, 'done.json')
    done = common.read_json(done_p) if os.path.exists(done_p) else {}
    left = '?'
    if os.path.exists(os.path.join(out, 'left')):
        with open(os.path.join(out, 'left')) as f:
            left = f.read().strip()
    text = ['# The evening check', '', "The stored flight days scored again from their saved grids with the site's code as it is now (each day's own model), every evening method and the evening state's constants over a grid at every climb (calib/eveningcheck.py; docs/verification/evening-hold-2026-11).", '', f'- Days scored: {len(cuts)} (of {len(done)} tried; not scored: ' + (', '.join((f'{k} {v}' for k, v in sorted(_count((s for s in done.values() if s != 'ok')).items()))) or 'none') + f'); left for the next run: {left}.', '']
    by_model = {}
    for c in cuts:
        by_model.setdefault((c.get('macro') or 'US', c['model']), []).append(c)
    for (macro, model), cs in sorted(by_model.items()):
        C = [r for c in cs for r in c['climbs']]
        H = [r for c in cs for r in c['hours']]
        text += [f'# {macro} ({model}): {len(cs)} days, {len(C)} climbs', '']
        text += report.evening_section(C, H)
        text += _sweep_section(C)
    return '\n'.join(text) + '\n'

def _count(xs):
    out = {}
    for x in xs:
        x = x.split(':')[0]
        out[x] = out.get(x, 0) + 1
    return out

def write_csv(out: str) -> int:
    n = 0
    with gzip.open(os.path.join(out, 'climbs.csv.gz'), 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_KEYS)
        for fn in sorted(glob.glob(os.path.join(out, 'rows', '*.json.gz'))):
            c = common.read_json(fn)
            for r in c['climbs']:
                if not r.get('paired') or r.get('m_solar') is None or r['m_solar'] < report.EVENING_MID[0]:
                    continue
                row = {**r, 'macro': c.get('macro'), 'model': c['model']}
                w.writerow([row.get(k) for k in CSV_KEYS])
                n += 1
    return n

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='The evening methods on the stored flight days')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--data', required=True)
    r.add_argument('--grids', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--work', default='/tmp/evening-work')
    r.add_argument('--minutes', type=float, default=280)
    r.add_argument('--workers', type=int, default=4)
    p = sub.add_parser('report')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        return run(a)
    with open(os.path.join(a.out, 'report.md'), 'w') as f:
        f.write(build_report(a.out))
    common.log(f'evening check: report written, {write_csv(a.out)} climbs in the CSV')
    return 0
if __name__ == '__main__':
    sys.exit(main())
