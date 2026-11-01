from __future__ import annotations
import io
import json
import os
import sys
import tarfile
import tempfile
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

def fetch(name: str, dest: str, repo: str | None=None) -> bool:
    repo = repo or os.environ['GITHUB_REPOSITORY']
    tokens = [t for t in (os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'), os.environ.get('CALIB_ALT_TOKEN')) if t]
    last = None
    for tok in tokens or [None]:
        try:
            a = newest(name, repo, tok)
            if not a:
                return False
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
            return True
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (401, 403, 404):
                raise
    raise RuntimeError(f'artifact {name}: refused ({last})')

def age_days(name: str, repo: str | None=None) -> float | None:
    from datetime import datetime, timezone
    repo = repo or os.environ['GITHUB_REPOSITORY']
    a = newest(name, repo, os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'))
    if not a:
        return None
    made = datetime.fromisoformat(a['created_at'].replace('Z', '+00:00'))
    return (datetime.now(timezone.utc) - made).total_seconds() / 86400

def raw(name: str, dest: str, repo: str | None=None) -> bool:
    repo = repo or os.environ['GITHUB_REPOSITORY']
    tok = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    a = newest(name, repo, tok)
    if not a:
        return False
    with zipfile.ZipFile(io.BytesIO(_get(a['archive_download_url'], tok))) as z:
        enc = [n for n in z.namelist() if n.endswith('.enc')]
        if len(enc) != 1:
            raise RuntimeError('artifact without one .enc file')
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, os.path.basename(enc[0])), 'wb') as f:
            f.write(z.read(enc[0]))
    return True

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
    if a[:1] == ['fetch'] and len(a) in (3, 4):
        sys.exit(0 if fetch(a[1], a[2], a[3] if len(a) == 4 else None) else 3)
    elif a[:1] == ['age'] and len(a) == 2:
        d = age_days(a[1])
        print('none' if d is None else f'{d:.1f}')
    elif a[:1] == ['raw'] and len(a) == 3:
        sys.exit(0 if raw(a[1], a[2]) else 3)
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
