from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lake

def restore(lk: lake.Lake, what: str, out: str) -> dict:
    n = {'restored': 0, 'plain': 0, 'kept': 0, 'failed': 0}
    if what in ('', '/'):
        keys = lk.list('')
    elif what.endswith('/'):
        keys = lk.list(what)
    else:
        keys = [lake.check_key(what)]
    n['matched'] = len(keys)
    done = set()
    for k in keys:
        try:
            lake.check_key(k)
        except lake.LakeError:
            n['failed'] += 1
            continue
        named_enc = k.endswith('.lenc')
        rel = k[:-len('.lenc')] if named_enc else k
        dest = os.path.join(out, *rel.split('/'))
        if dest in done:
            n['failed'] += 1
            continue
        if os.path.exists(dest):
            n['kept'] += 1
            continue
        try:
            blob = lk.store.get(k)
            if blob is None:
                n['failed'] += 1
                continue
            enc = lake.is_lenc(blob)
            if named_enc and (not enc):
                n['failed'] += 1
                continue
            data = lake.unseal(blob, k, lk.keys) if enc else blob
        except lake.LakeError:
            n['failed'] += 1
            continue
        tmp = dest + '.part'
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, dest)
        except OSError:
            n['failed'] += 1
            try:
                os.unlink(tmp)
            except OSError:
                pass
            continue
        done.add(dest)
        n['restored' if enc else 'plain'] += 1
    return n

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Restore objects from the lake, decrypted.')
    ap.add_argument('what', help="an object key, a prefix ending in '/', or 'name'")
    ap.add_argument('out', help="the folder to write into (with 'name': the logical id)")
    ap.add_argument('--local', help='a folder holding the objects instead of the bucket')
    ap.add_argument('--store', default='LAKE', choices=['LAKE', 'OPS', 'VAULT'])
    a = ap.parse_args(argv)
    try:
        if a.what == 'name':
            print(lake.Keys.from_env().name(a.out))
            return 0
        if a.local and (not os.path.isdir(a.local)):
            print('lakerestore: the --local folder does not exist', file=sys.stderr)
            return 2
        lk = lake.from_env(a.store, a.local)
        if lk is None:
            print(f'lakerestore: no {a.store}_S3_* secrets in the environment and no --local folder', file=sys.stderr)
            return 2
        n = restore(lk, a.what, a.out)
    except lake.LakeError as e:
        print(f'lakerestore: {e}', file=sys.stderr)
        return 1
    matched = n.pop('matched')
    if not matched:
        print('lakerestore: nothing matched; ' + lk.summary(), file=sys.stderr)
        return 1
    print('lakerestore: ' + ', '.join((f'{v} {k}' for k, v in n.items())) + '; ' + lk.summary())
    return 1 if n['failed'] else 0
if __name__ == '__main__':
    sys.exit(main())
