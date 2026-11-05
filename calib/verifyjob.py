from __future__ import annotations
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
import common
import slots
import state
UTC = timezone.utc
KEYS = ('obs', 'hrrr', 'gfs', 'ecmwf-0', 'ecmwf-1', 'icon', 'icond2')
SAMPLE_KEYS = KEYS[1:]
WORKFLOW = 'Forecast data'
MIN_LEFT = 15
MIN_SINCE = 10
DEADLINE_BEFORE = 10
OLD_CIPHER = ['openssl', 'enc', '-d', '-aes-256-ctr', '-pbkdf2', '-pass', 'env:VERIFY_CARRY_KEY']

def _verify():
    if common.PIPELINE not in sys.path:
        sys.path.insert(0, common.PIPELINE)
    import numpy
    import verify
    return (verify, numpy)

def _read_json(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def _tokens() -> list[str | None]:
    t = [x for x in (os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'), os.environ.get('VERIFY_READ_TOKEN')) if x]
    return t or [None]

def _api(path: str, token: str | None=None) -> dict | list:
    return json.loads(state._get(f'{state.API}/{path}', token))

def source(owner: str | None=None, me: str | None=None, api=_api) -> str:
    me = me or os.environ['GITHUB_REPOSITORY']
    owner = owner or me.split('/')[0]
    tok = _tokens()[0]
    found = []
    for r in api(f'users/{owner}/repos?per_page=100&type=owner', tok):
        full = r.get('full_name') or ''
        if full.lower() == me.lower() or r.get('private') or r.get('archived'):
            continue
        try:
            wfs = api(f'repos/{full}/actions/workflows?per_page=100', tok).get('workflows', [])
        except urllib.error.HTTPError:
            continue
        if any(((w.get('name') or '').startswith(WORKFLOW) for w in wfs)):
            found.append(full)
    if len(found) != 1:
        raise RuntimeError(f'expected one forecast repo, found {len(found)}')
    return found[0]

def busy(repo: str, api=_api) -> int:
    runs = api(f'repos/{repo}/actions/runs?per_page=40', _tokens()[0]).get('workflow_runs', [])
    return sum((1 for r in runs if (r.get('name') or '').startswith(WORKFLOW) and r.get('status') != 'completed'))

def window(slot_times: list[str], now: datetime) -> tuple[datetime, float, float]:
    times = [(now + timedelta(days=d)).replace(hour=int(s[:2]), minute=int(s[3:]), second=0, microsecond=0) for d in (-1, 0, 1) for s in slot_times]
    nxt, prev = (min((t for t in times if t > now)), max((t for t in times if t <= now)))
    deadline = nxt - timedelta(minutes=DEADLINE_BEFORE)
    return (deadline, (deadline - now).total_seconds() / 60, (now - prev).total_seconds() / 60)

def gate(wait_min: float, slot_times=None, busy_fn=None, now_fn=None, sleep_fn=time.sleep, poll_s: float=60) -> tuple[bool, datetime]:
    slot_times = slot_times or slots.forecast_slots()
    now_fn = now_fn or (lambda: datetime.now(UTC))
    t0 = now_fn()
    while True:
        n = busy_fn()
        now = now_fn()
        deadline, left, since = window(slot_times, now)
        log(f'deadline {deadline:%H:%M}Z, {left:.0f} min left, {since:.0f} min since the last slot, forecast runs going: {n}')
        if n == 0 and left >= MIN_LEFT and (since >= MIN_SINCE):
            return (True, deadline)
        if (now - t0).total_seconds() / 60 + poll_s / 60 > wait_min:
            return (False, deadline)
        sleep_fn(poll_s)

def _safe_extract(tar_path: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            name = m.name[2:] if m.name.startswith('./') else m.name
            if m.isdir() and name in ('', '.'):
                continue
            if not m.isfile() or not name or '/' in name or name.startswith('.'):
                raise RuntimeError('unexpected entry in the old state')
            with tf.extractfile(m) as src, open(os.path.join(dest, name), 'wb') as out:
                shutil.copyfileobj(src, out)

def old(repo: str, name: str, dest: str) -> bool:
    last = None
    for tok in _tokens():
        try:
            a = state.newest(name, repo, tok)
            if not a:
                return False
            blob = state._get(a['archive_download_url'], tok)
            break
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (401, 403, 404):
                raise
    else:
        raise RuntimeError(f'artifact {name}: refused ({last}): set VERIFY_READ_TOKEN')
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z, tempfile.TemporaryDirectory() as tmp:
        names = z.namelist()
        if not name.startswith('verify-'):
            for n in names:
                if '/' in n or n.startswith('.'):
                    raise RuntimeError('unexpected entry in the artifact')
                with open(os.path.join(dest, n), 'wb') as f:
                    f.write(z.read(n))
            return True
        tars = [n for n in names if n.startswith(name + '.tar')]
        if len(tars) != 1:
            raise RuntimeError(f'{name}: no state file in the artifact')
        raw = os.path.join(tmp, 'in')
        with open(raw, 'wb') as f:
            f.write(z.read(tars[0]))
        t = raw
        if tars[0].endswith('.enc'):
            if not os.environ.get('VERIFY_CARRY_KEY'):
                raise RuntimeError(f"{name} is encrypted: VERIFY_CARRY_KEY isn't set")
            t = os.path.join(tmp, 'x.tar')
            r = subprocess.run(OLD_CIPHER + ['-in', raw, '-out', t], capture_output=True)
            if r.returncode or not tarfile.is_tarfile(t):
                raise RuntimeError(f"{name} didn't decrypt (VERIFY_CARRY_KEY wrong?)")
        _safe_extract(t, dest)
    return True

def _empty(key: str) -> dict:
    return {'key': key, 'files': 0, 'runs': 0, 'days': 0, 'rows': 0, 'bad': []}

def check(d: str, key: str) -> dict:
    out = _empty(key)
    if not os.path.isdir(d):
        return out
    verify, np = _verify()
    model = key.split('-')[0]
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        out['files'] += 1
        if f == 'done.json' and key != 'obs':
            try:
                book = _read_json(p)
                out['done'] = len(book.get('done', {}))
                out['failed'] = len(book.get('failed', {}))
            except ValueError:
                out['bad'].append(f)
            continue
        if not f.endswith('.npz'):
            out['bad'].append(f)
            continue
        try:
            if key == 'obs':
                got = verify.load_day(p)
                if got is None or len(f) != 12:
                    raise ValueError('not a day of reports')
                arr = got[0]
                out['days'] += 1
                out['rows'] += int(np.isfinite(arr[:, :, 0]).sum())
            else:
                with np.load(p) as z:
                    if str(z['model']) != model or str(z['run']) != f[:-4]:
                        raise ValueError("another model's or run's")
                    codes, idx, stamps = (z['codes'], z['idx'], z['stamps'])
                    json.loads(str(z['enc']))
                    if codes.shape != (len(stamps), len(idx), len(z['fields'])):
                        raise ValueError('its shape')
                out['runs'] += 1
                out['rows'] += int(codes.shape[0] * codes.shape[1])
        except Exception:
            out['bad'].append(f)
    return out

def merge(src: str, dst: str, key: str) -> int:
    os.makedirs(dst, exist_ok=True)
    n = 0
    if not os.path.isdir(src):
        return 0
    for f in sorted(os.listdir(src)):
        if f.endswith('.npz') and (not os.path.exists(os.path.join(dst, f))):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
            n += 1
    sb, db = (os.path.join(src, 'done.json'), os.path.join(dst, 'done.json'))
    if key != 'obs' and os.path.exists(sb):
        a = _read_json(sb)
        b = _read_json(db) if os.path.exists(db) else {}
        done = {**a.get('done', {}), **b.get('done', {})}
        failed = {k: v for k, v in {**a.get('failed', {}), **b.get('failed', {})}.items() if k not in done}
        book = {**a, **b, 'done': done, 'failed': failed}
        with open(db, 'w') as f:
            json.dump(book, f)
    return n

def carry(repo: str, key: str, work: str, out: str, counts: str) -> int:
    o, n = (os.path.join(work, 'old', key), os.path.join(work, 'new', key))
    shutil.rmtree(o, ignore_errors=True)
    shutil.rmtree(n, ignore_errors=True)
    res = {'key': key}
    try:
        res['found_old'] = old(repo, f'verify-{key}', o)
        res['found_here'] = state.fetch(f'calib-verify-{key}', n)
    except Exception as e:
        res['error'] = f"didn't download: {type(e).__name__}: {e}"
    before_old, before_new = (check(o, key), check(n, key))
    res |= {'old': before_old, 'here_before': before_new}
    for side, c in (('old', before_old), ('here', before_new)):
        if c['bad']:
            res.setdefault('error', f"{len(c['bad'])} files {side} don't read: nothing carried")
    if 'error' not in res:
        res['copied'] = merge(o, n, key)
        after = check(n, key)
        res['after'] = after
        unit = 'days' if key == 'obs' else 'runs'
        if after['bad'] or after[unit] < max(before_old[unit], before_new[unit]) or after['rows'] < max(before_old['rows'], before_new['rows']):
            res['error'] = 'the merged state has fewer rows than a side: not packed'
        elif res['copied']:
            os.makedirs(out, exist_ok=True)
            state.pack(n, os.path.join(out, f'calib-verify-{key}.enc'))
            res['packed'] = True
    os.makedirs(os.path.dirname(os.path.abspath(counts)), exist_ok=True)
    with open(counts, 'w') as f:
        json.dump(res, f, indent=1)
    log(f"{key}: old {before_old['rows']} rows, here {before_new['rows']} before, " + (f"{res['after']['rows']} after" if 'after' in res else 'not merged') + (f"; {res['error']}" if 'error' in res else ''))
    return 1 if 'error' in res else 0

def _row(label: str, c: dict | None, unit: str) -> str:
    if not c:
        return f'| {label} | - | - |'
    return f"| {label} | {c.get(unit, 0)} | {c.get('rows', 0):,} |"

def carry_report(counts_dir: str, out_dir: str) -> None:
    lines = [f'# Verify carry-over, {datetime.now(UTC):%Y-%m-%d %H:%MZ}', '', "The forecast repo's newest Verify state merged into this repo's (calib-verify-<key>). Rows: station-hours sampled (a model's runs) or reported (obs).", '']
    for key in KEYS:
        p = os.path.join(counts_dir, f'{key}.json')
        unit = 'days' if key == 'obs' else 'runs'
        lines += [f'## {key}', '']
        if not os.path.exists(p):
            lines += ["No result (the job didn't finish).", '']
            continue
        c = _read_json(p)
        lines += [f'| | {unit} | rows |', '|---|---:|---:|', _row('forecast repo', c.get('old'), unit), _row('here before', c.get('here_before'), unit), _row('here after', c.get('after'), unit), '']
        if not c.get('found_old'):
            lines.append('Nothing in the forecast repo (no unexpired artifact).')
        if c.get('copied') is not None:
            lines.append(f"{c['copied']} files carried over" + (', packed and uploaded.' if c.get('packed') else '; nothing to upload.'))
        if c.get('error'):
            lines.append(f"**Not carried:** {c['error']}.")
        lines.append('')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(lines))

def _runs(d: str) -> set[str]:
    return {f[:-4] for f in os.listdir(d) if f.endswith('.npz')} if os.path.isdir(d) else set()

def same_runs(old_dir: str, new_dir: str) -> dict:
    _, np = _verify()
    common_runs = sorted(_runs(old_dir) & _runs(new_dir))
    out = {'old_runs': len(_runs(old_dir)), 'new_runs': len(_runs(new_dir)), 'both': len(common_runs), 'cells': 0, 'same': 0, 'stations_old_only': 0, 'stations_new_only': 0, 'fields': {}}
    diff_sum: dict[str, float] = {}
    diff_n: dict[str, int] = {}
    for rid in common_runs:
        with np.load(os.path.join(old_dir, f'{rid}.npz')) as a, np.load(os.path.join(new_dir, f'{rid}.npz')) as b:
            fa, fb = ([str(x) for x in a['fields']], [str(x) for x in b['fields']])
            sa, sb = ([str(x) for x in a['stamps']], [str(x) for x in b['stamps']])
            ia, ib = (a['idx'], b['idx'])
            st = sorted(set(sa) & set(sb))
            stn, pa, pb = np.intersect1d(ia, ib, return_indices=True)
            out['stations_old_only'] += int(len(ia) - len(stn))
            out['stations_new_only'] += int(len(ib) - len(stn))
            if not st or not len(stn):
                continue
            ka, kb = ([sa.index(s) for s in st], [sb.index(s) for s in st])
            ca, cb = (a['codes'][ka][:, pa, :].astype(np.int16), b['codes'][kb][:, pb, :].astype(np.int16))
            for name in (f for f in fa if f in fb):
                x, y = (ca[:, :, fa.index(name)], cb[:, :, fb.index(name)])
                both = (x != 255) | (y != 255)
                d = np.abs(x - y)[both]
                out['cells'] += int(both.sum())
                out['same'] += int((d == 0).sum())
                if (d > 0).any():
                    diff_sum[name] = diff_sum.get(name, 0) + float(d[d > 0].sum())
                    diff_n[name] = diff_n.get(name, 0) + int((d > 0).sum())
    out['fields'] = {k: {'cells': diff_n[k], 'mean_code_diff': round(diff_sum[k] / diff_n[k], 2)} for k in sorted(diff_n)}
    out['same_share'] = round(out['same'] / out['cells'], 6) if out['cells'] else None
    return out

def score_diff(a: dict, b: dict) -> dict:

    def table(s):
        out = {}
        for r in s.get('scores', []):
            for n, v in r.items():
                if isinstance(v, dict):
                    out[r['model'], n, r['region'], r['lead']] = v
        return out
    ta, tb = (table(a), table(b))
    rows = []
    for k in sorted(set(ta) & set(tb)):
        va, vb = (ta[k], tb[k])
        what = 'mae' if 'mae' in va else 'heidke'
        if va.get(what) is None or vb.get(what) is None:
            continue
        rows.append({'model': k[0], 'measure': k[1], 'region': k[2], 'lead': k[3], 'what': what, 'n_old': va['n'], 'n_new': vb['n'], 'old': va[what], 'new': vb[what], 'diff': round(vb[what] - va[what], 3)})
    rows.sort(key=lambda r: -abs(r['diff']))
    return {'pairs_old': a.get('pairs'), 'pairs_new': b.get('pairs'), 'only_old': len(set(ta) - set(tb)), 'only_new': len(set(tb) - set(ta)), 'both': len(rows), 'max_abs_diff': max((abs(r['diff']) for r in rows), default=None), 'rows': rows}

def rescore(old_root: str, new_root: str, work: str) -> tuple[dict, dict] | None:
    obs = os.path.join(new_root, 'obs')
    if not os.path.isdir(obs):
        return None
    sides = {}
    for side, root in (('old', old_root), ('new', new_root)):
        dirs = []
        for key in SAMPLE_KEYS:
            both = _runs(os.path.join(old_root, key)) & _runs(os.path.join(new_root, key))
            if not both:
                continue
            d = os.path.join(work, side, key)
            os.makedirs(d, exist_ok=True)
            for rid in both:
                shutil.copyfile(os.path.join(root, key, f'{rid}.npz'), os.path.join(d, f'{rid}.npz'))
            dirs.append(d)
        if not dirs:
            return None
        out = os.path.join(work, f'scores-{side}.json')
        subprocess.run([sys.executable, os.path.join(common.PIPELINE, 'verify.py'), 'score', '--samples', *dirs, '--obs', obs, '--out', out], check=True, stdout=subprocess.DEVNULL)
        sides[side] = _read_json(out)
    return (sides['old'], sides['new'])

def _diff_lines(sd: dict, top: int=25) -> list[str]:
    lines = [f"Pairs: forecast repo {sd['pairs_old']}, here {sd['pairs_new']}. Scores both have: {sd['both']} (only there {sd['only_old']}, only here {sd['only_new']}); the largest difference {sd['max_abs_diff']}.", '']
    if sd['rows']:
        lines += ['| model | measure | region | lead | score | n there | n here | there | here | diff |', '|---|---|---|---|---|---:|---:|---:|---:|---:|']
        lines += [f"| {r['model']} | {r['measure']} | {r['region']} | {r['lead']} | {r['what']} | {r['n_old']} | {r['n_new']} | {r['old']} | {r['new']} | {r['diff']:+} |" for r in sd['rows'][:top]]
        lines.append('')
    return lines

def compare(old_root: str, new_root: str, old_scores: str | None, new_scores: str | None, out_dir: str, rescore_fn=rescore) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    res = {'generated': f'{datetime.now(UTC):%Y-%m-%dT%H:%MZ}', 'runs': {}, 'obs': {}}
    lines = [f"# Verify: the forecast repo's against this repo's, {res['generated']}", '', "The same model runs rebuilt and sampled by each (cell by cell), then scored on the same reports (this repo's). A match: every cell identical, or differences only where a run's input files differed.", '']
    have_old = os.path.isdir(old_root) and any((os.listdir(os.path.join(old_root, k)) for k in KEYS if os.path.isdir(os.path.join(old_root, k))))
    if have_old:
        lines += ['## The same runs, cell by cell', '', '| key | runs there | runs here | both | cells | identical | stations only there / here | fields differing |', '|---|---:|---:|---:|---:|---:|---|---|']
        for key in SAMPLE_KEYS:
            s = same_runs(os.path.join(old_root, key), os.path.join(new_root, key))
            res['runs'][key] = s
            fd = ', '.join((f"{k} ({v['cells']}, {v['mean_code_diff']})" for k, v in s['fields'].items())) or 'none'
            share = '-' if s['same_share'] is None else f"{100 * s['same_share']:.3f} %"
            lines.append(f"| {key} | {s['old_runs']} | {s['new_runs']} | {s['both']} | {s['cells']:,} | {share} | {s['stations_old_only']} / {s['stations_new_only']} | {fd} |")
        co, cn = (check(os.path.join(old_root, 'obs'), 'obs'), check(os.path.join(new_root, 'obs'), 'obs'))
        res['obs'] = {'old': co, 'new': cn}
        lines += ['', f"Reports: there {co['days']} days ({co['rows']:,} station-hours), here {cn['days']} days ({cn['rows']:,}).", '']
        got = rescore_fn(old_root, new_root, os.path.join(out_dir, 'work'))
        shutil.rmtree(os.path.join(out_dir, 'work'), ignore_errors=True)
        if got:
            sd = score_diff(*got)
            res['rescored'] = {k: v for k, v in sd.items() if k != 'rows'} | {'rows': sd['rows'][:200]}
            lines += ['## The runs both did, scored on the same reports', ''] + _diff_lines(sd)
            for side, s in zip(('old', 'new'), got):
                with open(os.path.join(out_dir, f'scores-same-runs-{side}.json'), 'w') as f:
                    json.dump(s, f, indent=1)
        else:
            lines += ['No run both have sampled yet (or no reports here): nothing scored side by side.', '']
    else:
        lines += ["The forecast repo's state wasn't there (none unexpired, or VERIFY_CARRY_KEY not set): only the two scores files compared.", '']
    if old_scores and new_scores and os.path.exists(old_scores) and os.path.exists(new_scores):
        sd = score_diff(_read_json(old_scores), _read_json(new_scores))
        res['scores'] = {k: v for k, v in sd.items() if k != 'rows'} | {'rows': sd['rows'][:200]}
        lines += ["## Each one's own scores (its latest scores, over the runs each has done)", ''] + _diff_lines(sd)
        for side, p in (('old', old_scores), ('new', new_scores)):
            shutil.copyfile(p, os.path.join(out_dir, f'scores-{side}.json'))
    else:
        lines += [f"Scores: {('there' if old_scores and os.path.exists(old_scores) else 'none there')}, {('here' if new_scores and os.path.exists(new_scores) else 'none here')}.", '']
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(lines))
    with open(os.path.join(out_dir, 'compare.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res

def prune(name: str, keep: int=3) -> int:
    repo, tok = (os.environ['GITHUB_REPOSITORY'], os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'))
    arts = _api(f'repos/{repo}/actions/artifacts?name={name}&per_page=100', tok).get('artifacts', [])
    arts = sorted((a for a in arts if a.get('name') == name), key=lambda a: a.get('created_at') or '', reverse=True)
    n = 0
    for a in arts[keep:]:
        req = urllib.request.Request(f"{state.API}/repos/{repo}/actions/artifacts/{a['id']}", method='DELETE', headers={'Authorization': f'Bearer {tok}', 'Accept': 'application/vnd.github+json', 'User-Agent': 'state-fetch'})
        try:
            urllib.request.urlopen(req, timeout=60).close()
            n += 1
        except urllib.error.HTTPError as e:
            log(f'{name}: one not deleted ({e.code})')
    return n

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('source')
    p = sub.add_parser('gate')
    p.add_argument('--wait', type=float, default=0)
    p = sub.add_parser('old')
    (p.add_argument('repo'), p.add_argument('name'), p.add_argument('dir'))
    p = sub.add_parser('check')
    (p.add_argument('dir'), p.add_argument('key', choices=KEYS))
    p = sub.add_parser('carry')
    (p.add_argument('--repo', required=True), p.add_argument('--key', required=True, choices=KEYS))
    (p.add_argument('--work', required=True), p.add_argument('--out', required=True), p.add_argument('--counts', required=True))
    p = sub.add_parser('carry-report')
    (p.add_argument('counts'), p.add_argument('out'))
    p = sub.add_parser('compare')
    (p.add_argument('--old', required=True), p.add_argument('--new', required=True))
    (p.add_argument('--old-scores'), p.add_argument('--new-scores'), p.add_argument('--out', required=True))
    p = sub.add_parser('prune')
    (p.add_argument('name'), p.add_argument('--keep', type=int, default=3))
    a = ap.parse_args(argv)
    if a.cmd == 'source':
        print(source())
    elif a.cmd == 'gate':
        try:
            repo = source()
        except Exception as e:
            log(f"::warning::the forecast repo wasn't found ({e}): standing down")
            print('go=false')
            return 0

        def busy_or_wait():
            try:
                return busy(repo)
            except Exception as e:
                log(f"the forecast repo's runs didn't list ({e})")
                return 1
        go, deadline = gate(a.wait, busy_fn=busy_or_wait)
        print(f'deadline={deadline:%Y-%m-%dT%H:%MZ}')
        print(f"go={('true' if go else 'false')}")
    elif a.cmd == 'old':
        return 0 if old(a.repo, a.name, a.dir) else 3
    elif a.cmd == 'check':
        c = check(a.dir, a.key)
        print(json.dumps(c))
        return 1 if c['bad'] else 0
    elif a.cmd == 'carry':
        return carry(a.repo, a.key, a.work, a.out, a.counts)
    elif a.cmd == 'carry-report':
        carry_report(a.counts, a.out)
    elif a.cmd == 'compare':
        compare(a.old, a.new, a.old_scores, a.new_scores, a.out)
    elif a.cmd == 'prune':
        log(f'{a.name}: {prune(a.name, a.keep)} older deleted')
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
