from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lake as lakem
import rollfit
import score
import streetpairs
SV = 'streets'
MIN_BLOCKS = 2000
KEYS = ('zL', 'wind', 'turnDeg', 'shear', 'lo', 'cap', 'lid', 'tear', 'on', 'aspect', 'aspectZL')

def constants_now() -> dict:
    try:
        r = subprocess.run(['node', score_node(), 'constants'], capture_output=True, text=True, timeout=120)
        R = json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        R = None
    if not R:
        return dict(rollfit.ROLLS_NOW)
    return {k: R[k] for k in KEYS if k in R}

def score_node() -> str:
    import satcheck
    return satcheck.node_script()

def due_month(today: date) -> str:
    return (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')

def refit(T: dict, P_now: dict) -> dict:
    m24 = T['use_off'] & (T['km'] == 24)
    out = {'blocks24': int(m24.sum()), 'box_days': len(set(zip(T['box'], T['day']))), 'seasons': {s: int((rollfit.season(T['day'])[m24] == s).sum()) for s in ('warm', 'cold')}, 'boxes': {b: int((T['box'][m24] == b).sum()) for b in sorted(set(T['box']))}, 'now': P_now}
    if out['blocks24'] < MIN_BLOCKS:
        out['pass'], out['why'] = (False, f"{out['blocks24']} blocks, under the {MIN_BLOCKS} the refit needs")
        return out
    g = rollfit.gate(T, P_now)
    out.update({k: g[k] for k in ('pass', 'why', 'all', 'groups', 'worse')})
    out['refit'] = g['final']
    out['spacing'] = rollfit.spacing_fit(T, g['final'] if g['pass'] else P_now)
    return out

def report_md(month: str, r: dict) -> str:
    L = [f"# The rolls' refit, {month}", '', f"{r['box_days']} box-days, {r['blocks24']} 24 km blocks with cumulus to read (warm {r['seasons']['warm']}, cold {r['seasons']['cold']}).", '']
    L.append(f"**Verdict: {('pass' if r.get('pass') else 'no change')}.** {r.get('why', '')}")
    if 'all' in r:
        L += ['', 'Held out by date (the mean of AUC and balanced accuracy, 24 and 48 km, both labellings):', '', '| group | now | refit |', '| --- | --- | --- |', f"| all | {r['all'][0]:.3f} | {r['all'][1]:.3f} |"]
        L += [f'| {k} | {o:.3f} | {n:.3f} |' for k, (o, n) in sorted(r['groups'].items())]
    if 'refit' in r:
        L += ['', '| constant | now | refit |', '| --- | --- | --- |']
        L += [f"| {k} | {r['now'].get(k)} | {r['refit'].get(k)} |" for k in KEYS if r['now'].get(k) != r['refit'].get(k)]
    sp = r.get('spacing') or {}
    if sp.get('ratio_now') is not None:
        L += ['', f"Spacing: {sp['n']} blocks with lines along the wind; measured / forecast {sp['ratio_now']:.2f} at the median now, {sp['ratio_new']:.2f} with aspect {sp['aspect']}."]
    return '\n'.join(L) + '\n'

def run(lk, today: date, again: bool=False, counts=None, P_now=None) -> dict | None:
    counts = counts if counts is not None else {}
    month = due_month(today)
    if score.reports(lk, month, sv=SV) and (not again):
        counts['already'] = 1
        return None
    cols = streetpairs.load_all(lk, counts)
    if not cols:
        counts['no_pairs'] = 1
        return None
    T = rollfit.columns(cols)
    r = refit(T, P_now or constants_now())
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    pre = f'{score.PREFIX}/{SV}/{month}/{ts}'
    lk.put(pre + '.json.lenc', json.dumps(r, sort_keys=True, default=_num).encode(), strict=False)
    lk.put(pre + '.md.lenc', report_md(month, r).encode(), strict=False)
    counts['written'] = 1
    r['month'] = month
    return r

def _num(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o).__name__)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The rolls' monthly refit against the satellite")
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('run')
    a.add_argument('--again', action='store_true')
    a.add_argument('--day', help='today, YYYY-MM-DD (default: today, UTC)')
    a.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    f = sub.add_parser('fetch')
    f.add_argument('--out', required=True)
    f.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    x = ap.parse_args(argv)
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('streetfit: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            score.fetch(lk, x.out, counts, sv=SV)
        else:
            today = date.fromisoformat(x.day) if x.day else datetime.now(timezone.utc).date()
            r = run(lk, today, x.again, counts)
            print('streetfit: ' + (f"the refit of {r['month']} written: {('pass' if r.get('pass') else 'no change')}" if r else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('streetfit: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('streetfit: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
