from __future__ import annotations
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import common
API = 'https://api.github.com'

class _NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, *a, **k):
        return None

def _get(url: str, token: str | None, follow: bool=True) -> bytes:
    h = {'Accept': 'application/vnd.github+json', 'User-Agent': 'state-fetch', 'X-GitHub-Api-Version': '2022-11-28'}
    if token:
        h['Authorization'] = f'Bearer {token}'
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(url, headers=h), timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308) and follow:
            loc = e.headers['Location']
            with urllib.request.urlopen(urllib.request.Request(loc, headers={'User-Agent': 'state-fetch'}), timeout=300) as r:
                return r.read()
        raise

def newest(name: str, repo: str, token: str | None) -> dict | None:
    js = json.loads(_get(f'{API}/repos/{repo}/actions/artifacts?name={name}&per_page=30', token))
    arts = [a for a in js.get('artifacts', []) if a.get('name') == name and (not a.get('expired'))]
    arts.sort(key=lambda a: a.get('created_at') or '', reverse=True)
    return arts[0] if arts else None

def fetch(name: str, dest: str, repo: str | None=None, id_file: str | None=None) -> bool:
    repo = repo or os.environ['GITHUB_REPOSITORY']
    tokens = [t for t in (os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'), os.environ.get('CALIB_ALT_TOKEN')) if t]
    last = None
    for tok in tokens or [None]:
        try:
            a = newest(name, repo, tok)
            if id_file:
                with open(id_file, 'w') as f:
                    f.write(f"{a['id']}\n" if a else 'none\n')
            if not a:
                return False
            _unpack_artifact(a, dest, tok)
            return True
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (401, 403, 404):
                raise
    raise RuntimeError(f'artifact {name}: refused ({last})')

def _unpack_artifact(a: dict, dest: str, tok: str | None) -> None:
    blob = _get(a['archive_download_url'], tok)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        enc = [n for n in z.namelist() if n.endswith('.enc')]
        if len(enc) != 1:
            raise RuntimeError('artifact without one .enc file')
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'x.enc')
            with open(p, 'wb') as f:
                f.write(z.read(enc[0]))
            unpack(p, dest)

def age_days(name: str, repo: str | None=None) -> float | None:
    from datetime import datetime, timezone
    repo = repo or os.environ['GITHUB_REPOSITORY']
    a = newest(name, repo, os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'))
    if not a:
        return None
    made = datetime.fromisoformat(a['created_at'].replace('Z', '+00:00'))
    return (datetime.now(timezone.utc) - made).total_seconds() / 86400

def listing(prefix: str, repo: str | None=None, older: float | None=None) -> dict:
    from datetime import datetime, timezone
    repo = repo or os.environ['GITHUB_REPOSITORY']
    tok = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    out, page = ({}, 1)
    while True:
        js = json.loads(_get(f'{API}/repos/{repo}/actions/artifacts?per_page=100&page={page}', tok))
        arts = js.get('artifacts') or []
        for a in arts:
            n = a.get('name') or ''
            if n.startswith(prefix) and (not a.get('expired')) and a.get('created_at'):
                made = datetime.fromisoformat(a['created_at'].replace('Z', '+00:00'))
                age = (datetime.now(timezone.utc) - made).total_seconds() / 86400
                out[n] = min(out.get(n, age), age)
        if len(arts) < 100:
            break
        if page >= 100:
            print('::warning::state.py list: over 10,000 artifacts, the oldest not read', file=sys.stderr)
            break
        page += 1
    if older is not None:
        out = {n: d for n, d in out.items() if d >= older}
    return {n: round(d, 1) for n, d in sorted(out.items())}

def raw(name: str, dest: str, repo: str | None=None) -> bool:
    repo = repo or os.environ['GITHUB_REPOSITORY']
    tokens = [t for t in (os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'), os.environ.get('CALIB_ALT_TOKEN')) if t]
    a = blob = None
    for tok in tokens or [None]:
        try:
            a = newest(name, repo, tok)
            if not a:
                return False
            blob = _get(a['archive_download_url'], tok)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403, 404) or tok is (tokens or [None])[-1]:
                raise
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        enc = [n for n in z.namelist() if n.endswith('.enc')]
        if len(enc) != 1:
            raise RuntimeError('artifact without one .enc file')
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, os.path.basename(enc[0])), 'wb') as f:
            f.write(z.read(enc[0]))
    return True

def _api_json(url: str, tok: str | None):
    return json.loads(_get(url, tok))

def copies(name: str, repo: str | None=None, tok: str | None=None, api=None) -> list[dict]:
    repo = repo or os.environ['GITHUB_REPOSITORY']
    api = api or _api_json
    out, page = ([], 1)
    while True:
        js = api(f'{API}/repos/{repo}/actions/artifacts?name={name}&per_page=100&page={page}', tok)
        arts = js.get('artifacts') or []
        out += [a for a in arts if a.get('name') == name and (not a.get('expired'))]
        if len(arts) < 100 or page >= 100:
            break
        page += 1
    return sorted(out, key=lambda a: a.get('created_at') or '', reverse=True)

def units(root: str, depth: int=2, tops: tuple[str, ...] | None=None) -> set[str]:
    out = set()
    for d, _, files in os.walk(root):
        for fn in files:
            parts = os.path.relpath(os.path.join(d, fn), root).split(os.sep)
            if tops is not None and parts[0] not in tops:
                continue
            out.add('/'.join(parts[:depth]))
    return out

def _add_unit(src_root: str, dest_root: str, unit: str) -> None:
    src, dst = (os.path.join(src_root, *unit.split('/')), os.path.join(dest_root, *unit.split('/')))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f'{dst}.part'
    shutil.rmtree(tmp, ignore_errors=True)
    (shutil.copytree if os.path.isdir(src) else shutil.copyfile)(src, tmp)
    os.replace(tmp, dst)

def _write(out: str | None, res: dict) -> None:
    if out:
        with open(out + '.tmp', 'w') as f:
            json.dump(res, f)
        os.replace(out + '.tmp', out)

def _read_id(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ''
    with open(path) as f:
        return f.read().strip()

def data_missing(old: str, new: str) -> int:
    have = units(new, 2, ('days', 'pending'))
    stored = {u.split('/', 1)[1] for u in have if u.startswith('days/')}
    n = 0
    for u in units(old, 2, ('days', 'pending')):
        if u in have or (u.startswith('pending/') and u.split('/', 1)[1] in stored):
            continue
        n += 1
    return n

def fold(name: str, dest: str, base_id: str='', budget_s: float=900, out: str | None=None, repo: str | None=None, check_only: bool=False, keep: int=0, oldest_first: bool=False, listed=None, unpack_fn=None, now=time.monotonic) -> dict:
    tok = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    arts = listed if listed is not None else copies(name, repo, tok)
    arts = [a for a in arts[keep:] if str(a['id']) != str(base_id)]
    if oldest_first:
        arts = arts[::-1]
    unpack_fn = unpack_fn or (lambda a, d: _unpack_artifact(a, d, tok))
    res = {'covered': [], 'added': 0, 'missing': 0, 'checked': 0, 'left': len(arts)}
    t0 = now()
    for a in arts:
        if now() - t0 > budget_s:
            break
        with tempfile.TemporaryDirectory() as tmp:
            try:
                unpack_fn(a, tmp)
            except Exception as e:
                print(f"a copy didn't read ({type(e).__name__}): kept", file=sys.stderr)
                res['checked'] += 1
                res['left'] -= 1
                continue
            if check_only:
                miss = data_missing(tmp, dest)
            else:
                have = units(dest)
                for u in sorted(units(tmp) - have):
                    _add_unit(tmp, dest, u)
                    res['added'] += 1
                miss = len(units(tmp) - units(dest))
            res['missing'] += miss
            if miss == 0:
                res['covered'].append(a['id'])
        res['checked'] += 1
        res['left'] -= 1
        _write(out, res)
    _write(out, res)
    return res

def _delete(repo: str, art_id, tok: str | None) -> None:
    req = urllib.request.Request(f'{API}/repos/{repo}/actions/artifacts/{art_id}', method='DELETE', headers={'Authorization': f'Bearer {tok}', 'Accept': 'application/vnd.github+json', 'User-Agent': 'state-fetch'})
    urllib.request.urlopen(req, timeout=60).close()

def tidy(name: str, covered: list, keep: int=3, after: str | None=None, repo: str | None=None, listed=None, delete_fn=None) -> int:
    repo = repo or os.environ.get('GITHUB_REPOSITORY', '')
    tok = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    arts = listed if listed is not None else copies(name, repo, tok)
    if not arts or (after and (arts[0].get('created_at') or '') < after):
        print("the new copy isn't listed yet: nothing deleted", file=sys.stderr)
        return 0
    delete_fn = delete_fn or (lambda i: _delete(repo, i, tok))
    newest = {str(a['id']) for a in arts[:max(1, keep)]}
    cov = {str(i) for i in covered}
    n = 0
    for a in arts:
        if str(a['id']) in cov and str(a['id']) not in newest:
            try:
                delete_fn(a['id'])
                n += 1
            except urllib.error.HTTPError as e:
                print(f'one not deleted ({e.code})', file=sys.stderr)
    return n

def crowded(prefix: str, over: int=3, most: int=3, repo: str | None=None, listed=None) -> list[str]:
    repo = repo or os.environ.get('GITHUB_REPOSITORY', '')
    tok = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if listed is None:
        listed, page = ([], 1)
        while True:
            arts = _api_json(f'{API}/repos/{repo}/actions/artifacts?per_page=100&page={page}', tok).get('artifacts') or []
            listed += arts
            if len(arts) < 100 or page >= 100:
                break
            page += 1
    n = {}
    for a in listed:
        if (a.get('name') or '').startswith(prefix) and (not a.get('expired')):
            n[a['name']] = n.get(a['name'], 0) + 1
    return [k for k, v in sorted(n.items(), key=lambda kv: (-kv[1], kv[0])) if v > over][:most]

def pack(src: str, out: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = os.path.join(tmp, 'x.tar.gz')
        with tarfile.open(t, 'w:gz') as tf:
            for root, _, files in os.walk(src):
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    tf.add(p, arcname=os.path.relpath(p, src))
        common.encrypt(t, out + '.tmp')
    os.replace(out + '.tmp', out)

def unpack(enc: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = os.path.join(tmp, 'x.tar.gz')
        common.decrypt(enc, t)
        with tarfile.open(t, 'r:gz') as tf:
            for m in tf.getmembers():
                if not m.isfile() or m.name.startswith(('/', '..')) or '/../' in m.name:
                    raise RuntimeError('unexpected entry in the state archive')
            tf.extractall(dest, filter='data') if sys.version_info >= (3, 12) else tf.extractall(dest)
if __name__ == '__main__':
    a = sys.argv[1:]
    idf = None
    if a[:1] == ['fetch'] and '--id' in a[:-1]:
        i = a.index('--id')
        idf = a[i + 1]
        del a[i:i + 2]
    if a[:1] == ['fetch'] and len(a) in (3, 4):
        sys.exit(0 if fetch(a[1], a[2], a[3] if len(a) == 4 else None, idf) else 3)
    elif a[:1] == ['age'] and len(a) == 2:
        d = age_days(a[1])
        print('none' if d is None else f'{d:.1f}')
    elif a[:1] == ['list'] and len(a) in (2, 4):
        older = float(a[3]) if len(a) == 4 and a[2] == '--older' else None
        print(json.dumps(listing(a[1], older=older)))
    elif a[:1] == ['raw'] and len(a) in (3, 4):
        sys.exit(0 if raw(a[1], a[2], a[3] if len(a) == 4 else None) else 3)
    elif a[:1] in (['fold'], ['covered']) and len(a) >= 3:
        opts = dict(zip(a[3::2], a[4::2]))
        res = fold(a[1], a[2], base_id=_read_id(opts.get('--base-id')), budget_s=float(opts.get('--budget', 900)), out=opts.get('--out'), check_only=a[0] == 'covered', keep=int(opts.get('--keep', 0)), oldest_first=a[0] == 'covered')
        print(f"older copies: {res['checked']} read, {len(res['covered'])} covered, {res['added']} units added back, {res['missing']} not in the newest, {res['left']} left for next time")
        if a[0] == 'covered' and res['missing']:
            print(f"::warning::{res['missing']} stored or waiting days in older copies of {a[1]} aren't in the newest: those copies kept")
    elif a[:1] == ['tidy'] and len(a) >= 2:
        opts = dict(zip(a[2::2], a[3::2]))
        with open(opts['--covered']) as f:
            cov = json.load(f).get('covered', [])
        print(f"older copies deleted: {tidy(a[1], cov, keep=int(opts.get('--keep', 3)), after=opts.get('--after'))}")
    elif a[:1] == ['crowded'] and len(a) >= 2:
        opts = dict(zip(a[2::2], a[3::2]))
        print(json.dumps(crowded(a[1], over=int(opts.get('--over', 3)), most=int(opts.get('--max', 3)))))
    elif a[:1] == ['pack'] and len(a) == 3:
        pack(a[1], a[2])
    elif a[:1] == ['unpack'] and len(a) == 3:
        unpack(a[1], a[2])
    elif a[:1] == ['decrypt'] and len(a) == 3:
        common.decrypt(a[1], a[2])
    elif a[:1] == ['encrypt'] and len(a) == 3:
        common.encrypt(a[1], a[2])
    else:
        sys.exit(__doc__)
