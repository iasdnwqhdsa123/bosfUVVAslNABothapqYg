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
GROUP_EXTRA = {'EU': [('build_eu.py', 'icon', '2025-06-18', 'Europe/Berlin', '10.0,51.8,11.0,52.4', 13, 14), ('build_eu.py', 'icond2', '2025-07-02', 'Europe/Berlin', '7.6,46.4,8.6,47.0', 13, 14), ('build_eu.py', 'arome', '2025-07-02', 'Europe/Berlin', '7.6,46.4,8.6,47.0', 13, 14)], 'SH': [('build_sh.py', 'gfs', '2025-12-10', 'Australia/Sydney', '147.7,-32.6,148.7,-31.8', 13, 14), ('build_sh.py', 'icon', '2025-12-10', 'Australia/Sydney', '147.7,-32.6,148.7,-31.8', 13, 14)]}
GROUP_REFS = {area: [(BUILDER[m], m, d, tz, bb, a, b) for m, d, tz, bb, a, b in REFS.get(area, [])] + GROUP_EXTRA.get(area, []) for area in ('US', 'EU', 'SH')}
GROUP_REPS = 'gvref.npz'

def group_of(key: str) -> str | None:
    if key.startswith('st/'):
        return 'stations'
    return common.FIELD_GROUP.get(key.rsplit('/', 1)[-1])

def split_groups(refs: dict) -> dict:
    out = {g: {} for g in common.GROUPS}
    for k, v in refs.items():
        g = group_of(k)
        if g is not None:
            out[g][k] = v
    return out

def group_hash(group: str, sub: dict, boxes=()) -> str | None:
    if not sub:
        return None
    h = hashlib.sha256(('\x00'.join([group, *common.GROUPS[group]]) + '\x00').encode())
    for box in boxes:
        h.update(('box ' + ' '.join((str(x) for x in box)) + '\x00').encode())
    for k in sorted(sub):
        x = np.ascontiguousarray(sub[k])
        h.update(f'{k}\x00{x.dtype.str}\x00{x.shape}\x00'.encode())
        h.update(x.tobytes())
    return h.hexdigest()[:12]

def group_close(a: dict, b: dict) -> bool:
    if all((np.asarray(v).dtype == np.uint8 for v in (*a.values(), *b.values()))):
        return close(a, b)
    return bool(a) and set(a) == set(b) and all((np.array_equal(a[k], b[k], equal_nan=True) for k in a))

def group_stamps(refs: dict, kept: dict, boxes=(), prefer: dict | None=None) -> tuple[dict, dict]:
    prefer = prefer or {}
    gv, new = ({}, {})
    for g, sub in split_groups(refs).items():
        h = group_hash(g, sub, boxes)
        if h is None or (g, h) in kept:
            gv[g] = h
            continue
        order = sorted((v for kg, v in kept if kg == g), key=lambda v: (v != prefer.get(g), v))
        near = next((v for v in order if group_close(sub, kept[g, v])), None)
        gv[g] = near or h
        if near is None:
            new[g, h] = sub
    return (gv, new)

def load_group_reps(*paths: str) -> dict:
    reps = {}
    for p in paths:
        try:
            z = np.load(p)
        except (OSError, ValueError):
            continue
        for k in z.files:
            area, g, v, rest = k.split(':', 3)
            reps.setdefault((area, g, v), {})[rest] = z[k]
    return reps

def save_group_reps(path: str, reps: dict) -> None:
    np.savez_compressed(path, **{f'{a}:{g}:{v}:{k}': x for (a, g, v), e in sorted(reps.items()) for k, x in e.items()})

def merge_group_reps(src: str, dest: str) -> None:
    save_group_reps(dest, {**load_group_reps(dest), **load_group_reps(src)})

def area_reps(reps: dict, area: str) -> dict:
    return {(g, v): e for (a, g, v), e in reps.items() if a == area}
BOX_LOG_TAIL = 40

def _box_failed(i: int, model: str, why: str, logpath: str) -> None:
    try:
        with open(logpath, errors='replace') as f:
            tail = f.readlines()[-BOX_LOG_TAIL:]
    except OSError:
        tail = []
    print(f'reference box r{i} ({model}): {why}', file=sys.stderr, flush=True)
    sys.stderr.writelines(tail)
    sys.stderr.flush()
    return None

def build_group_refs(boxes: list, root: str=common.ROOT, work: str | None=None) -> dict | None:
    names = sorted(common.FIELD_GROUP)
    env = dict(os.environ)
    env.setdefault('CONTRAILS', '0')
    own = work is None
    work = work or tempfile.mkdtemp(prefix='gv-')
    os.makedirs(work, exist_ok=True)
    refs = {}
    try:
        for i, (builder, model, day, tz, bbox, a, b) in enumerate(boxes):
            out = os.path.join(work, f'r{i}')
            shutil.rmtree(out, ignore_errors=True)
            cmd = [sys.executable, os.path.join(root, 'calib', builder), '--day', day, '--tz', tz, f'--bbox={bbox}', '--out', out, '--model', model, '--from-local', str(a), '--to-local', str(b)]
            cmd += ['--workers', '2'] if builder == 'build_day.py' else ['--cache', os.path.join(work, 'cache')]
            logpath = os.path.join(work, f'r{i}.log')
            with open(logpath, 'w') as lf:
                try:
                    r = subprocess.run(cmd, cwd=os.path.join(root, 'pipeline'), stdout=lf, stderr=subprocess.STDOUT, timeout=TIMEOUT, env=env)
                except subprocess.TimeoutExpired:
                    return _box_failed(i, model, f'timed out after {TIMEOUT} s', logpath)
            if r.returncode != 0:
                return _box_failed(i, model, f'exit {r.returncode}', logpath)
            n = len(refs)
            _fields_of(out, f'r{i}', names, refs)
            if len(refs) == n:
                return _box_failed(i, model, 'built no reference fields', logpath)
        return refs
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)

def groups(kept_paths: list[str], out: str, root: str=common.ROOT, work: str | None=None, current: str | None=None) -> dict | None:
    kept = load_group_reps(*kept_paths)
    now = {}
    if current:
        try:
            with open(current) as f:
                now = json.load(f)
        except (OSError, ValueError):
            pass
    gv, new = ({}, {})
    for area, boxes in GROUP_REFS.items():
        refs = build_group_refs(boxes, root, os.path.join(work, area) if work else None)
        if refs is None:
            common.log(f'group stamps: a reference box of {area} failed, none made')
            return None
        gv[area], made = group_stamps(refs, area_reps(kept, area), boxes, now.get(area))
        new.update({(area, g, v): e for (g, v), e in made.items()})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'gv.json'), 'w') as f:
        json.dump(gv, f, indent=1, sort_keys=True)
        f.write('\n')
    save_group_reps(os.path.join(out, GROUP_REPS), {**kept, **new})
    for area in sorted(gv):
        common.log(f'group stamps {area}: ' + ', '.join((f"{g} {v or '-'}" + (' (new)' if (area, g, v) in new else '') for g, v in sorted(gv[area].items()))))
    return gv

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
    s = sub.add_parser('groups')
    s.add_argument('--kept', action='append', default=[])
    s.add_argument('--out', required=True)
    s.add_argument('--root', default=common.ROOT)
    s.add_argument('--work')
    s.add_argument('--current')
    gm = sub.add_parser('merge-groups')
    gm.add_argument('src')
    gm.add_argument('dest')
    a = ap.parse_args()
    if a.cmd == 'ensure':
        ensure(a.table, a.out, a.work)
    elif a.cmd == 'at':
        print(fingerprint(a.macro, os.path.abspath(a.root), a.work) or 'none')
    elif a.cmd == 'groups':
        groups(a.kept, a.out, os.path.abspath(a.root), a.work, a.current)
    elif a.cmd == 'merge-groups':
        merge_group_reps(a.src, a.dest)
    else:
        merge(a.src, a.dest)
if __name__ == '__main__':
    main()
