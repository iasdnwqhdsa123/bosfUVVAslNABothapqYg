from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
REFS = {'US': [('hrrr', '2025-07-15', 'America/Denver', '-105.6,39.4,-104.6,40.2', 13, 14), ('hrrr', '2025-04-10', 'America/New_York', '-78.5,40.3,-77.5,41.1', 11, 12), ('hrrr', '2025-02-10', 'America/Denver', '-111.9,40.4,-111.1,41.0', 12, 13), ('gfs', '2025-07-15', 'America/Denver', '-105.6,39.4,-104.6,40.2', 13, 14)], 'EU': [('ecmwf', '2025-07-02', 'Europe/Berlin', '7.6,46.4,8.6,47.0', 13, 14)]}
BUILDER = {'hrrr': 'build_day.py', 'gfs': 'build_day.py', 'ecmwf': 'build_eu.py'}
TIMEOUT = 15 * 60
TABLE = os.path.join(HERE, 'fingerprints.json')
REPS = 'fpref.npz'
TOL_STEPS, TOL_SHARE = (2, 0.05)

def fields() -> list[str]:
    import day
    return sorted(set(day.FIELDS) | set(day.SAVE_ALSO))

def _hash_out(h, out: str, names: list[str]) -> int:
    packs = {}
    for dp, _, fs in os.walk(out):
        for f in fs:
            if f.endswith('.pack') and f[:-5] in names:
                packs.setdefault(f[:-5], []).append(os.path.join(dp, f))
    n = 0
    for name in names:
        for p in sorted(packs.get(name, [])):
            h.update(name.encode() + b'\x00' + os.path.relpath(p, out).encode() + b'\x00')
            with open(p, 'rb') as fh:
                h.update(fh.read())
            n += 1
    return n

def _fields_of(out: str, box: int, names: list[str], into: dict) -> None:
    for dp, _, fs in os.walk(out):
        if os.path.basename(dp) != '_hours':
            continue
        for f in sorted(fs):
            if f.endswith('.npz'):
                z = np.load(os.path.join(dp, f))
                for k in names:
                    if k in z.files:
                        into[f'{box}/{f[:-4]}/{k}'] = np.asarray(z[k], np.uint8)

def close(a: dict, b: dict) -> bool:
    if not a or set(a) != set(b):
        return False
    n, c = ({}, {})
    for k, x in a.items():
        y = b[k]
        if x.shape != y.shape or not np.array_equal(x == 255, y == 255):
            return False
        d = np.abs(x.astype(np.int16) - y.astype(np.int16))
        if d.max(initial=0) > TOL_STEPS:
            return False
        f = k.rsplit('/', 1)[1]
        n[f] = n.get(f, 0) + d.size
        c[f] = c.get(f, 0) + int((d > 0).sum())
    return all((c[f] <= TOL_SHARE * n[f] for f in n))

def load_reps(*paths: str) -> dict:
    reps = {}
    for p in paths:
        try:
            z = np.load(p)
        except (OSError, ValueError):
            continue
        for k in z.files:
            m, fp, rest = k.split(':', 2)
            reps.setdefault((m, fp), {})[rest] = z[k]
    return reps

def save_reps(path: str, reps: dict) -> None:
    np.savez_compressed(path, **{f'{m}:{fp}:{k}': v for (m, fp), e in sorted(reps.items()) for k, v in e.items()})

def fingerprint(macro: str, root: str=common.ROOT, work: str | None=None, refs: dict | None=None) -> str | None:
    names = fields()
    env = dict(os.environ)
    env.setdefault('CONTRAILS', '0')
    h = hashlib.sha256(('\x00'.join(names) + '\x00').encode())
    own = work is None
    work = work or tempfile.mkdtemp(prefix='fp-')
    os.makedirs(work, exist_ok=True)
    try:
        for i, (model, day, tz, bbox, a, b) in enumerate(REFS[macro]):
            out = os.path.join(work, f'{macro}-{i}')
            shutil.rmtree(out, ignore_errors=True)
            cmd = [sys.executable, os.path.join(root, 'calib', BUILDER[model]), '--day', day, '--tz', tz, f'--bbox={bbox}', '--out', out, '--model', model, '--from-local', str(a), '--to-local', str(b)]
            cmd += ['--cache', os.path.join(work, 'eu-cache')] if BUILDER[model] == 'build_eu.py' else ['--workers', '2']
            with open(os.path.join(work, f'{macro}-{i}.log'), 'w') as lf:
                try:
                    r = subprocess.run(cmd, cwd=os.path.join(root, 'pipeline'), stdout=lf, stderr=subprocess.STDOUT, timeout=TIMEOUT, env=env)
                except subprocess.TimeoutExpired:
                    return None
            if r.returncode != 0:
                return None
            h.update(f'{model} {day} {bbox} {a}-{b}\x00'.encode())
            if not _hash_out(h, out, names):
                return None
            if refs is not None:
                _fields_of(out, i, names, refs)
        return h.hexdigest()[:12]
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)

def load(path: str) -> dict:
    try:
        with open(path) as f:
            t = json.load(f)
        return {m: dict(t.get(m) or {}) for m in REFS}
    except (OSError, ValueError):
        return {m: {} for m in REFS}

def save(path: str, table: dict) -> None:
    with open(path, 'w') as f:
        json.dump({m: dict(sorted(table.get(m, {}).items())) for m in REFS}, f, indent=1)
        f.write('\n')

def ensure(table_path: str, out: str, work: str | None=None) -> dict:
    have = load(table_path)
    for m, t in load(TABLE).items():
        have[m].update(t)
    new = have
    reps = load_reps(os.path.join(HERE, REPS), os.path.join(os.path.dirname(os.path.abspath(table_path)), REPS))
    mine = {}
    for m in REFS:
        stamps = {common.physics_version(m), common._WHOLE.get(m)} - {None}
        if all((s in have[m] for s in stamps)):
            common.log(f'fingerprint {m}: {common.physics_version(m)} known ({have[m][common.physics_version(m)]})')
            continue
        refs = {}
        fp = fingerprint(m, work=work, refs=refs)
        near = None
        if fp and fp not in have[m].values() and ((m, fp) not in reps):
            near = next((k[1] for k, r in sorted(reps.items()) if k[0] == m and close(refs, r)), None)
            if near is None and refs:
                mine[m, fp] = refs
        common.log(f"fingerprint {m}: {common.physics_version(m)} -> {fp or 'none (a reference box failed)'}" + (f', within the tolerance of {near}' if near else ''))
        fp = near or fp
        if fp:
            for s in stamps:
                new[m][s] = fp
    save(out, new)
    if mine:
        save_reps(os.path.join(os.path.dirname(os.path.abspath(out)), REPS), mine)
    return new

def merge(src: str, dest: str) -> None:
    t = load(dest)
    for m, e in load(src).items():
        t[m].update(e)
    save(dest, t)
    rs, rd = (os.path.join(os.path.dirname(os.path.abspath(p)), REPS) for p in (src, dest))
    if os.path.exists(rs) and rs != rd:
        save_reps(rd, {**load_reps(rd), **load_reps(rs)})

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('ensure')
    e.add_argument('--table', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--work')
    t = sub.add_parser('at')
    t.add_argument('--macro', required=True, choices=sorted(REFS))
    t.add_argument('--root', default=common.ROOT)
    t.add_argument('--work')
    g = sub.add_parser('merge')
    g.add_argument('src')
    g.add_argument('dest')
    a = ap.parse_args()
    if a.cmd == 'ensure':
        ensure(a.table, a.out, a.work)
    elif a.cmd == 'at':
        print(fingerprint(a.macro, os.path.abspath(a.root), a.work) or 'none')
    else:
        merge(a.src, a.dest)
if __name__ == '__main__':
    main()
