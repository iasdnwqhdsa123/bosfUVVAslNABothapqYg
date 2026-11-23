from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lake
API = 'https://api.github.com'
CHUNK = 1 << 20
MINUTES = 50
WHOLE = {'calib-data': 'data', 'calib-wave': 'wave', 'calib-model': 'model', 'calib-sat': 'sat', 'calib-ridge': 'ridge', 'calib-lines': 'lines'}
PREFIXED = (('calib-grids-', 'grids'), ('calib-verify-', 'verify'))
RELEASE_ASSETS = {'calib-data.enc': 'data', 'calib-wave.enc': 'wave'}
ITEM_LISTS = ('flights', 'short', 'obs', 'collector')
ITEM_SCORED = ('model_', 'probe_m')
LEG_SCORED = ('models', 'models_c', 'cat')
CLIMB_SCORED = ('models', 'models_c', 'near')
FLIGHT_LISTS = ('flights', 'short')
MIGRATION = 'migration/'
CHECKS = 'migration/checks/'
RELEASES = 'migration/releases/'
RELEASE_CHECKS = 'migration/releases/checks/'
REVS = 99
RAW_TOPS = ('collector', 'obs')
UNITED = ('data', 'seed', 'verify', 'wave', 'grids')
MAX_PAGES = 1000
MAX_BYTES = 12000000000
READ_MAX = 2000000000
FIELD = re.compile('^[A-Za-z_][A-Za-z0-9_]{0,40}$')

class Failed(Exception):
    pass

def log(msg: str) -> None:
    print(f'migrate: {msg}', flush=True)

def stamp(t: float | None=None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def ts_of(iso: str | None) -> str:
    try:
        return datetime.fromisoformat((iso or '').replace('Z', '+00:00')).astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    except ValueError:
        return 'unknown'

def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def kind_of(name: str) -> str | None:
    if name in WHOLE:
        return WHOLE[name]
    for p, k in PREFIXED:
        if name.startswith(p) and len(name) > len(p):
            return k
    return None

def safe_part(s) -> str:
    return re.sub('[^0-9A-Za-z_-]', '_', str(s if s not in (None, '') else 'none'))[:64]

def json_of(name: str, b: bytes):
    return json.loads(gzip.decompress(b) if name.endswith('.gz') else b)

class _NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, *a, **k):
        return None

def _tokens() -> list[str | None]:
    return [t for t in (os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'), os.environ.get('CALIB_ALT_TOKEN')) if t] or [None]

class GitHub:

    def __init__(self, repo: str):
        self.repo = repo
        self.tokens = _tokens()
        self.tok = self.tokens[0]

    def _open(self, url: str, accept: str='application/vnd.github+json'):
        last = None
        for tok in [self.tok] + [t for t in self.tokens if t is not self.tok]:
            h = {'Accept': accept, 'User-Agent': 'calib-migrate', 'X-GitHub-Api-Version': '2022-11-28'}
            if tok:
                h['Authorization'] = f'Bearer {tok}'
            try:
                return urllib.request.build_opener(_NoRedirect).open(urllib.request.Request(url, headers=h), timeout=120)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    loc = e.headers.get('Location')
                    e.close()
                    self.tok = tok
                    return urllib.request.urlopen(urllib.request.Request(loc, headers={'User-Agent': 'calib-migrate'}), timeout=600)
                last = e.code
                e.close()
                if e.code not in (401, 403, 404):
                    break
        raise Failed(f'the old store refused a request (HTTP {last})')

    def json(self, path: str):
        with self._open(f'{API}/repos/{self.repo}/{path}') as r:
            return json.loads(r.read())

    def stream(self, url: str, accept: str='application/vnd.github+json'):
        with self._open(url, accept) as r:
            while True:
                b = r.read(CHUNK)
                if not b:
                    return
                yield b

class Artifacts:
    verbatim = 'zip'

    def __init__(self, repo: str, api: GitHub | None=None):
        self.api = api or GitHub(repo)

    def list(self) -> list[dict]:
        out, page = ([], 1)
        while True:
            r = self.api.json(f'actions/artifacts?per_page=100&page={page}') or {}
            arts = r.get('artifacts') or []
            for a in arts:
                dig = a.get('digest') or ''
                out.append({'id': str(a['id']), 'name': a.get('name') or '', 'bytes': int(a.get('size_in_bytes') or 0), 'created': a.get('created_at'), 'expires': a.get('expires_at'), 'expired': bool(a.get('expired')), 'digest': dig.split(':', 1)[1] if dig.startswith('sha256:') else None, 'url': a.get('archive_download_url')})
            if len(arts) < 100 or len(out) >= int(r.get('total_count') or 0):
                return out
            if page >= MAX_PAGES:
                raise Failed('the listing has more pages than it reads')
            page += 1

    def stream(self, a: dict):
        return self.api.stream(a['url'])

class Releases:
    verbatim = 'enc'

    def __init__(self, repo: str, api: GitHub | None=None):
        self.api = api or GitHub(repo)

    def list(self) -> list[dict]:
        out, page = ([], 1)
        while True:
            rels = self.api.json(f'releases?per_page=100&page={page}') or []
            for r in rels:
                for a in r.get('assets') or []:
                    if a.get('name') in RELEASE_ASSETS:
                        dig = a.get('digest') or ''
                        out.append({'id': str(a['id']), 'name': a['name'], 'bytes': int(a.get('size') or 0), 'created': a.get('created_at'), 'expires': None, 'expired': False, 'digest': dig.split(':', 1)[1] if dig.startswith('sha256:') else None, 'url': a.get('url')})
            if len(rels) < 100:
                return out
            if page >= MAX_PAGES:
                raise Failed('the listing has more pages than it reads')
            page += 1

    def stream(self, a: dict):
        return self.api.stream(a['url'], 'application/octet-stream')

def release_kind(a: dict) -> str | None:
    return RELEASE_ASSETS.get(a.get('name'))

def download(src, a: dict, path: str | None) -> tuple[str, int]:
    h, n = (hashlib.sha256(), 0)
    f = open(path, 'wb') if path else None
    try:
        for b in src.stream(a):
            h.update(b)
            n += len(b)
            if f:
                f.write(b)
    finally:
        if f:
            f.close()
    return (h.hexdigest(), n)

def enc_of_zip(blob: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            enc = [n for n in z.namelist() if n.endswith('.enc')]
            if len(enc) != 1 or z.testzip() is not None:
                raise Failed('an artifact is not one whole .enc')
            return z.read(enc[0])
    except zipfile.BadZipFile:
        raise Failed('an artifact is not a zip') from None

def check_enc(blob: bytes) -> None:
    if blob[:8] != b'Salted__':
        raise Failed("a file is not in the old store's format")

def enc_digest(path: str, zipped: bool) -> str:
    h = hashlib.sha256()
    try:
        if not zipped:
            with open(path, 'rb') as f:
                _hash_enc(f, h)
            return h.hexdigest()
        with zipfile.ZipFile(path) as z:
            enc = [n for n in z.namelist() if n.endswith('.enc')]
            if len(enc) != 1 or z.testzip() is not None:
                raise Failed('an artifact is not one whole .enc')
            with z.open(enc[0]) as f:
                _hash_enc(f, h)
        return h.hexdigest()
    except zipfile.BadZipFile:
        raise Failed('an artifact is not a zip') from None

def _hash_enc(f, h) -> None:
    head = f.read(8)
    check_enc(head)
    h.update(head)
    for b in iter(lambda: f.read(CHUNK), b''):
        h.update(b)

def decrypt_old(enc: bytes) -> bytes:
    import subprocess
    import common
    check_enc(enc)
    with tempfile.TemporaryDirectory() as tmp:
        a, b = (os.path.join(tmp, 'x.enc'), os.path.join(tmp, 'x'))
        with open(a, 'wb') as f:
            f.write(enc)
        try:
            common.decrypt(a, b)
        except subprocess.CalledProcessError:
            raise Failed('a file of the old store did not decrypt') from None
        with open(b, 'rb') as f:
            return f.read()

def untar(tgz: bytes) -> dict[str, bytes]:
    out = {}
    with tarfile.open(fileobj=io.BytesIO(tgz), mode='r:*') as tf:
        for m in tf.getmembers():
            if m.isdir():
                continue
            name = m.name[2:] if m.name.startswith('./') else m.name
            if not m.isfile() or name.startswith('/') or '..' in name.split('/'):
                raise Failed('an archive holds an unexpected entry')
            out[name] = tf.extractfile(m).read()
    return out

def tree_of(enc: bytes) -> dict[str, bytes]:
    return untar(decrypt_old(enc))

def tar_files(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tf:
        for name in sorted(files):
            ti = tarfile.TarInfo(name)
            ti.size = len(files[name])
            ti.mtime = 0
            ti.mode = 420
            tf.addfile(ti, io.BytesIO(files[name]))
    return buf.getvalue()

def route(rel: str) -> tuple[str, str]:
    parts = rel.split('/')
    if rel == 'state.json':
        return ('state', 'state')
    if len(parts) == 2 and parts[0] in ('days', 'pending') and parts[1].endswith('.json.gz'):
        kind = 'day' if parts[0] == 'days' else 'pending'
        return (f"{kind}|{parts[1][:-len('.json.gz')]}", kind)
    if len(parts) == 2 and parts[0] == 'lists':
        return (f'lists|{parts[1]}', 'lists')
    if len(parts) > 1 and parts[0] in RAW_TOPS:
        return (f'raw|{rel}', 'rawfile')
    return (f'file|{rel}', 'file')

def _split_item(it):
    if not isinstance(it, dict):
        return (it, None)
    raw = {k: v for k, v in it.items() if not k.startswith(ITEM_SCORED)}
    sc = {'own': {k: v for k, v in it.items() if k.startswith(ITEM_SCORED)}}
    for lst, keys in (('legs', LEG_SCORED), ('climbs', CLIMB_SCORED)):
        if isinstance(it.get(lst), list):
            raw[lst] = [{k: v for k, v in x.items() if k not in keys} if isinstance(x, dict) else x for x in it[lst]]
            sc[lst] = [{k: v for k, v in x.items() if k in keys} if isinstance(x, dict) else None for x in it[lst]]
    return (raw, sc)

def split_day(rec: dict) -> tuple[dict, dict]:
    raw = {'date': rec.get('date'), 'key': rec.get('key'), 'macro': rec.get('macro'), 'items': {}}
    score = {'top': {k: v for k, v in rec.items() if k not in ITEM_LISTS}, 'items': {}}
    for lst in ITEM_LISTS:
        if lst in rec:
            if not isinstance(rec[lst], list):
                score['top'][lst] = rec[lst]
                continue
            pairs = [_split_item(it) for it in rec[lst]]
            raw['items'][lst] = [p[0] for p in pairs]
            score['items'][lst] = [p[1] for p in pairs]
    return (raw, score)

def join_day(raw: dict, score: dict) -> dict:
    rec = dict(score.get('top') or {})
    for lst, items in (raw.get('items') or {}).items():
        scs = (score.get('items') or {}).get(lst) or [None] * len(items)
        if len(scs) != len(items):
            raise Failed("a day's raw part and its scores do not pair")
        out = []
        for it, sc in zip(items, scs):
            if sc is None or not isinstance(it, dict):
                out.append(it)
                continue
            g = {**it, **(sc.get('own') or {})}
            for x in ('legs', 'climbs'):
                if x in sc:
                    if len(sc[x]) != len(it.get(x) or []):
                        raise Failed("a day's raw part and its scores do not pair")
                    g[x] = [{**a, **b} if isinstance(a, dict) and b else a for a, b in zip(it[x], sc[x])]
            out.append(g)
        rec[lst] = out
    return rec

def flights_in(rec) -> list[dict]:
    if not isinstance(rec, dict):
        return []
    return [f for lst in FLIGHT_LISTS for f in rec.get(lst) or [] if isinstance(f, dict) and f.get('id') is not None]

def raw_sig(f: dict) -> str:
    return sha(canon(_split_item(f)[0]))

def counts_of(rec: dict) -> dict:
    fl = flights_in(rec)
    return {'flights': len([f for f in rec.get('flights') or [] if isinstance(f, dict)]), 'short': len(rec.get('short') or []), 'legs': sum((len(f.get('legs') or []) for f in fl)), 'climbs': sum((len(f.get('climbs') or []) for f in fl))}

def _field(k) -> str:
    return k if isinstance(k, str) and FIELD.match(k) else 'other'

def diff_fields(a: dict, b: dict) -> list[str]:
    out = set()
    for k in set(a) | set(b):
        va, vb = (a.get(k), b.get(k))
        if va == vb:
            continue
        if k in ('legs', 'climbs') and isinstance(va, list) and isinstance(vb, list) and (len(va) == len(vb)):
            for x, y in zip(va, vb):
                if isinstance(x, dict) and isinstance(y, dict):
                    out |= {f'{k}[].{_field(j)}' for j in set(x) | set(y) if x.get(j) != y.get(j)}
                elif x != y:
                    out.add(f'{k}[]')
        else:
            out.add(_field(k))
    return sorted(out)

def decodes(name: str, b: bytes) -> bool:
    try:
        if name.endswith('.npz'):
            import numpy as np
            with np.load(io.BytesIO(b), allow_pickle=False) as z:
                for k in z.files:
                    z[k].shape
            return True
        if name.endswith('.json'):
            json.loads(b)
            return True
        if name.endswith('.gz'):
            gzip.decompress(b)
            return True
        return True
    except Exception:
        return False

def grid_units(files: dict[str, bytes]) -> dict[str, dict[str, bytes]]:
    out = {}
    for rel, b in files.items():
        parts = rel.split('/')
        if len(parts) < 3:
            out.setdefault('', {})[rel] = b
            continue
        out.setdefault('/'.join(parts[:2]), {})['/'.join(parts[2:])] = b
    return out

def shard_of(name: str, n: int) -> int:
    return int(sha(name.encode())[:8], 16) % max(1, n)

def put_json(lk: lake.Lake, key: str, obj) -> None:
    lk.put(key, canon(obj))

def get_json(lk: lake.Lake, key: str):
    return json.loads(lk.get(key))

def put_rev(lk: lake.Lake, base: str, plain: bytes) -> str:
    for n in range(1, REVS + 1):
        key = f'{base}.r{n}.lenc'
        if lk.put(key, plain, strict=False).status != 'differs':
            return key
    raise Failed('an object has too many versions')

def put_addressed(lk: lake.Lake, key: str, plain: bytes) -> str:
    if not lk.exists(key):
        lk.put(key, plain)
    return key

def checks_of(lk: lake.Lake, prefix: str=CHECKS) -> dict[str, dict]:
    out = {}
    for k in lk.list(prefix):
        if k.endswith('.lenc') and '/' not in k[len(prefix):]:
            c = get_json(lk, k)
            out[str(c['id'])] = c
    return out

def newest(lk: lake.Lake, prefix: str, want=None):
    for k in sorted(lk.list(prefix), reverse=True):
        if not k.endswith('.lenc') or '/' in k[len(MIGRATION):]:
            continue
        obj = get_json(lk, k)
        if want is None or want(obj):
            return (k, obj)
    return (None, None)

def verbatim_key(c: dict) -> str:
    if c.get('obj'):
        return c['obj']
    return f"{MIGRATION}artifacts/{c['id']}.zip" if c.get('kind') != 'seed' else f"{MIGRATION}seed/{c['id'][5:]}.enc"

def read_verbatim(lk: lake.Lake, c: dict) -> bytes:
    key = verbatim_key(c)
    if (c.get('bytes') or 0) > READ_MAX:
        raise Failed('a verbatim copy is too large to read in memory')
    blob = lk.get_plain(key)
    if sha(blob) != (c.get('obj_sha256') or c['sha256']):
        raise Failed('a verbatim copy no longer matches its check')
    enc = enc_of_zip(blob) if key.endswith('.zip') else blob
    if c.get('enc_sha256') and sha(enc) != c['enc_sha256']:
        raise Failed('a verbatim copy no longer matches its check')
    return enc

def distinct(cs: list[dict]) -> tuple[list[dict], list[dict]]:
    seen, keep, dup = (set(), [], [])
    for c in cs:
        k = (c.get('name'), c.get('enc_sha256')) if c.get('enc_sha256') else None
        if k and k in seen:
            dup.append(c)
            continue
        if k:
            seen.add(k)
        keep.append(c)
    return (keep, dup)

def by_kind(checks: dict[str, dict]) -> dict[str, list[dict]]:
    by = {}
    for c in checks.values():
        by.setdefault(c['kind'], []).append(c)
    for v in by.values():
        v.sort(key=lambda c: (c.get('created') or '', c['id']), reverse=True)
    return by

def _days_left(iso: str | None, now: float) -> float | None:
    if not iso:
        return None
    try:
        return (datetime.fromisoformat(iso.replace('Z', '+00:00')).timestamp() - now) / 86400
    except ValueError:
        return None

def inventory(lk: lake.Lake, src, seed: str | None=None, releases: bool=False, now: float | None=None) -> dict:
    now = time.time() if now is None else now
    checked = checks_of(lk, RELEASE_CHECKS if releases else CHECKS)
    rows = []
    for a in src.list():
        k = release_kind(a) if releases else kind_of(a['name'])
        if not k or a.get('expired'):
            continue
        rows.append({'id': a['id'], 'name': a['name'], 'kind': k, 'bytes': a['bytes'], 'created': a.get('created'), 'expires': a.get('expires'), 'days_left': _days_left(a.get('expires'), now), 'digest': a.get('digest'), 'checked': a['id'] in checked})
    inv = {'version': 1, 'at': stamp(now), 'source': 'releases' if releases else 'artifacts', 'artifacts': rows}
    if seed and os.path.exists(seed):
        with open(seed, 'rb') as f:
            b = f.read()
        inv['seed'] = {'sha256': sha(b), 'bytes': len(b), 'checked': f'seed-{sha(b)[:16]}' in checked}
    put_json(lk, f"{MIGRATION}inventory-{('releases-' if releases else '')}{stamp(now)}.lenc", inv)
    by = {}
    for r in rows:
        x = by.setdefault(r['kind'], [0, 0, 0])
        x[0] += 1
        x[1] += r['bytes']
        x[2] += r['checked']
    for k, (n, b, c) in sorted(by.items()):
        log(f'{k}: {n} copies, {b / 1000000.0:.1f} MB, {c} copied and checked')
    soon = [r['days_left'] for r in rows if r['days_left'] is not None]
    if soon:
        log(f'the soonest to expire: in {min(soon):.1f} days')
    log(f"{len(rows)} in all, {sum((r['bytes'] for r in rows)) / 1000000000.0:.2f} GB, {sum((r['checked'] for r in rows))} checked" + (f"; the seed {('checked' if inv['seed']['checked'] else 'not copied yet')}" if 'seed' in inv else ''))
    return inv

def copy_one(lk: lake.Lake, src, a: dict, kind: str, prefix: str, check_prefix: str, tmp: str, zipped: bool, held: dict) -> str:
    key = f"{prefix}{a['id']}.{('zip' if zipped else 'enc')}"
    path = os.path.join(tmp, 'a')
    if (a.get('bytes') or 0) > MAX_BYTES:
        raise Failed("an artifact is larger than the runner's disk takes")
    try:
        s1, n1 = download(src, a, path)
        if a.get('digest') and a['digest'] != s1:
            raise Failed('a download differs from the digest listed')
        if n1 > MAX_BYTES:
            raise Failed("an artifact is larger than the runner's disk takes")
        s2, n2 = download(src, a, None)
        if s1 != s2 or n1 != n2:
            raise Failed('the second download differs')
        if lake.file_digest(path) != (s1, n1):
            raise Failed('the file on disk differs from its download')
        es = enc_digest(path, zipped)
        check = {'id': a['id'], 'name': a['name'], 'kind': kind, 'sha256': s1, 'bytes': n1, 'enc_sha256': es, 'created': a.get('created'), 'expires': a.get('expires'), 'at': stamp()}
        same = held.get((a['name'], es))
        if same is not None:
            k2 = verbatim_key(same)
            if lk.digest_plain(k2)[0] != (same.get('obj_sha256') or same['sha256']):
                raise Failed('a verbatim copy no longer matches its check')
            check.update({'obj': k2, 'obj_sha256': same.get('obj_sha256') or same['sha256'], 'same_as': same['id']})
            put_json(lk, f"{check_prefix}{a['id']}.lenc", check)
            held[a['name'], es] = check
            return 'same'
        r = lk.put_plain_file(key, path, strict=False)
        if r.status == 'differs':
            raise Failed('a verbatim copy is stored with other bytes')
    finally:
        if os.path.exists(path):
            os.remove(path)
    if not lk.digest_plain(key)[0] == s1 == r.sha256:
        raise Failed('the stored copy differs')
    check['obj'] = key
    put_json(lk, f"{check_prefix}{a['id']}.lenc", check)
    held[a['name'], es] = check
    return 'copied'

def copy_seed(lk: lake.Lake, seed: str) -> str:
    with open(seed, 'rb') as f:
        b = f.read()
    check_enc(b)
    s1 = sha(b)
    cid = f'seed-{s1[:16]}'
    if lk.exists(f'{CHECKS}{cid}.lenc'):
        return 'skipped'
    key = f'{MIGRATION}seed/{s1[:16]}.enc'
    with open(seed, 'rb') as f:
        s2 = sha(f.read())
    if s1 != s2:
        raise Failed("the seed's second read differs")
    r = lk.put_plain(key, b)
    if not sha(lk.get_plain(key)) == s1 == r.sha256:
        raise Failed("the seed's stored copy differs")
    put_json(lk, f'{CHECKS}{cid}.lenc', {'id': cid, 'name': 'seed', 'kind': 'seed', 'sha256': s1, 'bytes': len(b), 'enc_sha256': s1, 'obj': key, 'created': None, 'expires': None, 'at': stamp()})
    return 'copied'

def copy(lk: lake.Lake, src, seed: str | None=None, minutes: float=MINUTES, releases: bool=False, clock=time.monotonic) -> dict:
    t0 = clock()
    prefix, cprefix = (RELEASES, RELEASE_CHECKS) if releases else (f'{MIGRATION}artifacts/', CHECKS)
    checks = checks_of(lk, cprefix)
    done = set(checks)
    held = {(c['name'], c['enc_sha256']): c for c in checks.values() if c.get('enc_sha256') and c.get('kind') != 'seed'}
    arts = [a for a in src.list() if not a.get('expired') and (release_kind(a) if releases else kind_of(a['name']))]
    arts.sort(key=lambda a: (a.get('expires') or '9999', a.get('created') or ''))
    n = {'copied': 0, 'same': 0, 'skipped': 0, 'failed': 0, 'left': 0}
    why = {}
    with tempfile.TemporaryDirectory() as tmp:
        for a in arts:
            if a['id'] in done:
                n['skipped'] += 1
                continue
            if clock() - t0 > minutes * 60:
                n['left'] += 1
                continue
            try:
                n[copy_one(lk, src, a, release_kind(a) if releases else kind_of(a['name']), prefix, cprefix, tmp, zipped=not releases, held=held)] += 1
            except Failed as e:
                n['failed'] += 1
                why[str(e)] = why.get(str(e), 0) + 1
            except lake.LakeError as e:
                n['failed'] += 1
                why[str(e)] = why.get(str(e), 0) + 1
            except Exception as e:
                n['failed'] += 1
                w = f'{type(e).__name__}'
                why[w] = why.get(w, 0) + 1
    if seed and (not releases):
        try:
            n[copy_seed(lk, seed)] += 1
        except (Failed, lake.LakeError) as e:
            n['failed'] += 1
            why[str(e)] = why.get(str(e), 0) + 1
    log(f"copy: {n['copied']} copied, {n['same']} the same bytes as one copied before (checked, not stored twice), {n['skipped']} checked before, {n['failed']} failed, {n['left']} left for the next run")
    for w, c in sorted(why.items()):
        log(f'  failed: {w} ({c})')
    n['why'] = why
    return n

class Union:

    def __init__(self, lk: lake.Lake):
        self.lk = lk
        self.entries: dict[str, dict] = {}
        self.sig = {'day': {}, 'pending': {}}
        self.raws = {'day': {}, 'pending': {}}
        self.where: dict[str, str] = {}
        self.orphans: dict[tuple, dict] = {}
        self.conflicts: list[dict] = []
        self.seed_ids: set[str] = set()
        self.n = {'days': 0, 'pending': 0, 'lists': 0, 'states': 0, 'files': 0, 'rawfiles': 0, 'grids': 0, 'verify': 0, 'wave': 0, 'seed': 0, 'orphans': 0, 'conflicts': 0, 'flights': 0, 'legs': 0, 'climbs': 0}

    def name(self, fid) -> str:
        return self.lk.name(f'flight|{fid}')

    def _flight(self, f: dict, kind: str, holder: str, day_key: str, src: str, first: bool) -> None:
        h = self.name(f['id'])
        raw = _split_item(f)[0]
        s = sha(canon(raw))
        known = self.sig[kind].get(h)
        if known is not None and known != s:
            self.conflicts.append({'flight': h, 'kind': kind, 'holder': self.where.get(h), 'other': src, 'fields': diff_fields(self.raws[kind][h], raw)})
            return
        if known is None:
            self.sig[kind][h] = s
            self.raws[kind][h] = raw
        if h in self.where:
            return
        if first:
            self.where[h] = holder
        else:
            self.orphans.setdefault((kind, day_key), {})[h] = raw
            self.where[h] = f'orphans|{kind}|{day_key}'

    def data_copy(self, c: dict, files: dict[str, bytes]) -> None:
        src = c['id']
        for rel in sorted(files):
            b = files[rel]
            lid, kind = route(rel)
            if kind == 'state':
                key = put_rev(self.lk, f"raw/weglide/meta/state-{ts_of(c.get('created'))}", b)
                self.entries[f'state|{src}'] = {'kind': 'state', 'src': src, 'rel': rel, 'obj': key, 'sha256': sha(b)}
                self.n['states'] += 1
            elif kind in ('day', 'pending'):
                dk = lid.split('|', 1)[1]
                rec = json_of(rel, b)
                first = lid not in self.entries
                if first:
                    if kind == 'day':
                        self._day(lid, dk, rec, src, rel)
                    else:
                        key = put_rev(self.lk, f'raw/weglide/{self.lk.name(lid)}', b)
                        self.entries[lid] = {'kind': kind, 'src': src, 'rel': rel, 'obj': key, 'sha256': sha(b)}
                        self.n['pending'] += 1
                for f in flights_in(rec):
                    self._flight(f, kind, lid, dk, src, first)
            elif lid in self.entries:
                continue
            elif kind == 'lists':
                key = put_rev(self.lk, f'raw/weglide/{self.lk.name(lid)}', b)
                self.entries[lid] = {'kind': 'lists', 'src': src, 'rel': rel, 'obj': key, 'sha256': sha(b)}
                self.n['lists'] += 1
            elif kind == 'rawfile':
                key = put_rev(self.lk, f"raw/legacy/{rel.split('/')[0]}/{self.lk.name(lid)}", b)
                self.entries[lid] = {'kind': 'rawfile', 'src': src, 'rel': rel, 'obj': key, 'sha256': sha(b)}
                self.n['rawfiles'] += 1
            else:
                key = put_addressed(self.lk, f"derived/legacy/files/{self.lk.name(f'{lid}|{sha(b)}')}.lenc", b)
                self.entries[lid] = {'kind': 'file', 'src': src, 'rel': rel, 'obj': key, 'sha256': sha(b)}
                self.n['files'] += 1

    def _day(self, lid: str, dk: str, rec: dict, src: str, rel: str) -> None:
        raw, score = split_day(rec)
        rk = put_rev(self.lk, f'raw/weglide/{self.lk.name(lid)}', canon(raw))
        sb = canon(score)
        sv = safe_part(rec.get('score_v'))
        sk = put_addressed(self.lk, f"derived/legacy/scores/{sv}/{self.lk.name(f'score|{dk}|{sha(sb)}')}.lenc", sb)
        cn = counts_of(rec)
        self.entries[lid] = {'kind': 'day', 'src': src, 'rel': rel, 'obj': rk, 'score': sk, 'sv': rec.get('score_v'), 'sha256': sha(canon(rec)), 'counts': cn}
        self.n['days'] += 1
        for k in ('flights', 'legs', 'climbs'):
            self.n[k] += cn[k]

    def seed(self, c: dict, plain: bytes) -> None:
        lid = f"seed|{c['id']}"
        key = put_rev(self.lk, f'raw/weglide/{self.lk.name(lid)}', plain)
        self.entries[lid] = {'kind': 'seed', 'src': c['id'], 'obj': key, 'sha256': sha(plain)}
        self.seed_ids |= {self.name(f['id']) for f in flights_in(json_of('.gz', plain))}
        self.n['seed'] += 1

    def verify(self, c: dict, plain: bytes) -> None:
        sub = safe_part(c['name'][len('calib-verify-'):])
        key = f"raw/legacy/verify/{sub}/{c['id']}.lenc"
        self.lk.put(key, plain)
        self.entries[f"verify|{c['id']}"] = {'kind': 'verify', 'src': c['id'], 'obj': key, 'sha256': sha(plain)}
        self.n['verify'] += 1

    def wave(self, c: dict, plain: bytes) -> None:
        lid = f"wave|{c['id']}"
        key = put_rev(self.lk, f'raw/weglide/{self.lk.name(lid)}', plain)
        self.entries[lid] = {'kind': 'wave', 'src': c['id'], 'obj': key, 'sha256': sha(plain)}
        self.n['wave'] += 1

    def grids(self, c: dict, files: dict[str, bytes]) -> None:
        for unit, fs in sorted(grid_units(files).items()):
            lid = f"grid|{c['name']}|{unit}"
            if lid in self.entries:
                continue
            tb = tar_files(fs)
            key = put_addressed(self.lk, f"derived/legacy/grids/{self.lk.name(f'{lid}|{sha(tb)}')}.lenc", tb)
            self.entries[lid] = {'kind': 'grid', 'src': c['id'], 'name': c['name'], 'unit': unit, 'obj': key, 'sha256': sha(tb), 'files': {p: sha(b) for p, b in fs.items()}}
            self.n['grids'] += 1

    def finish(self) -> None:
        for (kind, dk), fs in sorted(self.orphans.items()):
            lid = f'orphans|{kind}|{dk}'
            plain = canon({'key': dk, 'kind': kind, 'flights': [fs[h] for h in sorted(fs)]})
            key = put_rev(self.lk, f'raw/weglide/{self.lk.name(lid)}', plain)
            self.entries[lid] = {'kind': 'orphans', 'obj': key, 'sha256': sha(plain), 'flights': sorted(fs)}
            self.n['orphans'] += len(fs)
        self.n['conflicts'] = len(self.conflicts)

def union(lk: lake.Lake, minutes: float=MINUTES, clock=time.monotonic) -> dict:
    t0 = clock()
    by = by_kind(checks_of(lk))
    _, prev = newest(lk, f'{MIGRATION}union-')
    U = Union(lk)
    errors = {}

    def err(e: Exception) -> None:
        w = str(e) if isinstance(e, (Failed, lake.LakeError)) else type(e).__name__
        errors[w] = errors.get(w, 0) + 1
    used = []
    for kind, read in (('data', lambda c: U.data_copy(c, tree_of(read_verbatim(lk, c)))), ('seed', lambda c: U.seed(c, decrypt_old(read_verbatim(lk, c)))), ('verify', lambda c: U.verify(c, decrypt_old(read_verbatim(lk, c)))), ('wave', lambda c: U.wave(c, decrypt_old(read_verbatim(lk, c))))):
        keep, dup = distinct(by.get(kind, []))
        for c in keep:
            try:
                read(c)
                used.append(c['id'])
            except Exception as e:
                err(e)
        used += [c['id'] for c in dup]
    months = {}
    for c in by.get('grids', []):
        months.setdefault(c['name'], []).append(c)
    done_before = (prev or {}).get('months') or {}
    months_done, complete = ({}, True)
    for name in sorted(months):
        ids = sorted((c['id'] for c in months[name]))
        if done_before.get(name) == ids:
            for lid, e in (prev.get('entries') or {}).items():
                if e.get('kind') == 'grid' and e.get('name') == name:
                    U.entries[lid] = e
                    U.n['grids'] += 1
            months_done[name] = ids
            used += ids
            continue
        if clock() - t0 > minutes * 60:
            complete = False
            continue
        ok = True
        keep, dup = distinct(months[name])
        for c in keep:
            try:
                U.grids(c, tree_of(read_verbatim(lk, c)))
                used.append(c['id'])
            except Exception as e:
                ok = False
                err(e)
        if ok:
            used += [c['id'] for c in dup]
            months_done[name] = ids
    U.finish()
    verbatim_only = sorted((c['id'] for k in ('model', 'sat', 'ridge', 'lines') for c in by.get(k, [])))
    idx = {'version': 2, 'at': stamp(), 'complete': complete and (not errors), 'sources': sorted(set(used)), 'verbatim_only': verbatim_only, 'months': months_done, 'entries': U.entries, 'flights': {'day': U.sig['day'], 'pending': U.sig['pending']}, 'where': U.where, 'seed': sorted(U.seed_ids), 'conflicts': U.conflicts, 'counts': U.n, 'errors': errors}
    put_json(lk, f'{MIGRATION}union-{stamp()}.lenc', idx)
    n = U.n
    seed_only = len(U.seed_ids - set(U.where))
    log(f"union: {n['days']} days ({n['flights']} flights, {n['legs']} legs, {n['climbs']} climbs), {n['pending']} waiting, {n['lists']} day lists, {n['states']} states, {n['rawfiles']} collector and station files, {n['files']} other files, {n['grids']} grid boxes, {n['verify']} Verify copies, {n['wave']} wave copies, {n['seed']} seeds ({seed_only} seed flights in no day)")
    log(f"union: {n['orphans']} flights held only by older copies (kept), {n['conflicts']} conflicts, {sum(errors.values())} errors, {len(verbatim_only)} kept verbatim only" + ('' if idx['complete'] else '; not complete'))
    fields = {}
    for c in U.conflicts:
        for f in c['fields']:
            fields[f] = fields.get(f, 0) + 1
    for f, c in sorted(fields.items()):
        log(f"  a conflict's field: {f} ({c})")
    for w, c in sorted(errors.items()):
        log(f'  error: {w} ({c})')
    return idx

class Gate:

    def __init__(self):
        self.ok: dict[str, int] = {}
        self.bad: dict[str, int] = {}

    def check(self, what: str, good: bool) -> bool:
        (self.ok if good else self.bad)[what] = (self.ok if good else self.bad).get(what, 0) + 1
        return good

    @property
    def passed(self) -> bool:
        return not self.bad

def rebuilt_day(lk: lake.Lake, e: dict) -> dict:
    return join_day(get_json(lk, e['obj']), get_json(lk, e['score']))

def _rescore_default(rec: dict, grids: str, work: str) -> dict:
    import day as dayrun
    return dayrun.rescore_day(rec, grids, work)

def lake_flights(lk: lake.Lake, entries: dict, g: Gate) -> dict[str, dict[str, str]]:
    sig = {'day': {}, 'pending': {}}
    for lid, e in sorted(entries.items()):
        k = e.get('kind')
        if k not in ('day', 'pending', 'orphans'):
            continue
        try:
            if k == 'day':
                fl, into = (flights_in(get_json(lk, e['obj']).get('items') or {}), 'day')
            elif k == 'pending':
                fl, into = (flights_in(json_of('.gz', lk.get(e['obj']))), 'pending')
            else:
                o = get_json(lk, e['obj'])
                fl, into = ([f for f in o.get('flights') or [] if isinstance(f, dict) and f.get('id') is not None], o.get('kind'))
            for f in fl:
                sig.setdefault(into, {})[lk.name(f"flight|{f['id']}")] = raw_sig(f)
        except Exception:
            g.check("a flight's holder read from the lake", False)
    return sig

def _flights_held(g: Gate, orig: dict[str, set], held: dict[str, dict[str, str]], what: str='a flight') -> None:
    for h, seen in orig.items():
        g.check(f'{what} held', any((h in v for v in held.values())))
        for kind, s in sorted(seen):
            if h in held.get(kind, {}):
                g.check(f"{what} the same as the lake's", held[kind][h] == s)

def _err(g: Gate, what: str) -> None:
    g.check(what, False)

def _coverage(lk, g: Gate, idx: dict, checks: dict, store, seed: str | None, frozen: bool | None) -> None:
    arts = None
    if store is not None:
        try:
            arts = store.list()
        except Exception:
            arts = None
    if g.check('the old store listed again', arts is not None):
        for a in arts:
            if a.get('expired') or not kind_of(a.get('name') or ''):
                continue
            g.check('an artifact of the old store copied and checked', a['id'] in checks)
    if g.check('the seed found', bool(seed) and os.path.exists(seed)):
        with open(seed, 'rb') as f:
            g.check('the seed copied and checked', f'seed-{sha(f.read())[:16]}' in checks)
    g.check('the collection paused (the freeze)', bool(frozen))
    used = set(idx.get('sources') or [])
    for c in checks.values():
        if c['kind'] in UNITED:
            g.check('a checked copy taken into the union', c['id'] in used)

def _gate_data(lk, g: Gate, entries: dict, by: dict, totals: dict, visited: set) -> dict:
    days = {}
    orig: dict[str, set] = {}
    keep, _ = distinct(by.get('data', []))
    first = set()
    for i, c in enumerate(keep):
        try:
            files = tree_of(read_verbatim(lk, c))
        except Exception:
            _err(g, 'a copy of the data read again')
            continue
        g.check('a copy of the data read again', True)
        sl = f"state|{c['id']}"
        for rel in sorted(files):
            b = files[rel]
            lid, kind = route(rel)
            if kind in ('day', 'pending'):
                try:
                    rec = json_of(rel, b)
                except Exception:
                    _err(g, 'an original file read')
                    continue
                for f in flights_in(rec):
                    orig.setdefault(lk.name(f"flight|{f['id']}"), set()).add((kind, raw_sig(f)))
            if kind == 'state':
                lid = sl
            elif lid in first:
                continue
            first.add(lid)
            visited.add(lid)
            e = entries.get(lid)
            if not g.check(f'a {kind} of the originals in the union', e is not None and e.get('kind') == kind):
                continue
            try:
                if kind == 'day':
                    o = json_of(rel, b)
                    rec = rebuilt_day(lk, e)
                    g.check('a day equal after canonical JSON', canon(rec) == canon(o))
                    cn = counts_of(o)
                    g.check("a day's counts equal", counts_of(rec) == cn)
                    days[lid.split('|', 1)[1]] = rec
                    totals['days'] += 1
                    totals['with_grids'] += bool(o.get('grids'))
                    totals['station_items'] += len(o.get('obs') or [])
                    for k in ('flights', 'legs', 'climbs'):
                        totals[k] += cn[k]
                else:
                    g.check(f'a {kind} file as it was', lk.get(e['obj']) == b)
            except Exception:
                _err(g, f'a {kind} read from the lake')
        del files
        log(f'gate: data copy {i + 1} of {len(keep)} read again')
    totals['flight_ids'] = len(orig)
    for kind in ('seed', 'wave', 'verify'):
        for c in distinct(by.get(kind, []))[0]:
            lid = f"{kind}|{c['id']}"
            visited.add(lid)
            e = entries.get(lid)
            if not g.check(f'a {kind} copy in the union', e is not None):
                continue
            try:
                g.check(f'the {kind} as it was', lk.get(e['obj']) == decrypt_old(read_verbatim(lk, c)))
            except Exception:
                _err(g, f'a {kind} copy read again')
    visited |= {lid for lid, e in entries.items() if e.get('kind') == 'orphans'}
    _flights_held(g, orig, lake_flights(lk, entries, g))
    return days

def _gate_grids(lk, g: Gate, entries: dict, cs: list[dict], totals: dict, visited: set, shard: tuple[int, int]) -> None:
    months = {}
    for c in cs:
        if shard_of(c['name'], shard[1]) == shard[0]:
            months.setdefault(c['name'], []).append(c)
    for i, name in enumerate(sorted(months)):
        seen = set()
        for c in distinct(months[name])[0]:
            try:
                files = tree_of(read_verbatim(lk, c))
            except Exception:
                _err(g, 'a copy of the grids read again')
                continue
            for unit, fs in sorted(grid_units(files).items()):
                lid = f'grid|{name}|{unit}'
                if lid in seen:
                    continue
                seen.add(lid)
                visited.add(lid)
                e = entries.get(lid)
                if not g.check('a grid box of the originals in the union', e is not None):
                    continue
                try:
                    got = untar(lk.get(e['obj']))
                except Exception:
                    _err(g, 'a grid box readable')
                    continue
                g.check("a grid box's files all there", set(got) == set(fs) == set(e.get('files') or {}))
                for p, b in sorted(fs.items()):
                    totals['grid_files'] += 1
                    gb = got.get(p)
                    g.check('a grid file decodes', gb is not None and decodes(p, gb))
                    g.check("a grid file's SHA-256 the original's", gb is not None and sha(gb) == sha(b) == (e.get('files') or {}).get(p))
            del files
        log(f'gate: grids month {i + 1} of {len(months)} read again')

def reassemble(lk: lake.Lake, work: str, rescore: int=20, rescore_fn=None, store=None, seed: str | None=None, frozen: bool | None=None, part: str='all', shard: tuple[int, int]=(0, 1)) -> dict:
    g = Gate()
    rep = {'version': 2, 'at': stamp(), 'part': part if part != 'grids' else f'grids {shard[0]}/{shard[1]}'}
    _, idx = newest(lk, f'{MIGRATION}union-')
    if not idx or not idx.get('complete'):
        g.check('a complete union', False)
        return _report(lk, g, rep, {})
    g.check('no conflicts', not idx.get('conflicts'))
    checks = checks_of(lk)
    entries = idx.get('entries') or {}
    by = by_kind(checks)
    totals = {'days': 0, 'flights': 0, 'legs': 0, 'climbs': 0, 'with_grids': 0, 'station_items': 0, 'flight_ids': 0, 'grid_files': 0}
    visited: set = set()
    if part in ('all', 'data'):
        _coverage(lk, g, idx, checks, store, seed, frozen)
        days = _gate_data(lk, g, entries, by, totals, visited)
        counts = idx.get('counts') or {}
        g.check("the originals' counts equal the union's", all((totals[k] == counts.get(k) for k in ('days', 'flights', 'legs', 'climbs'))))
        rep['rescored'] = _rescore(lk, g, entries, days, work, rescore, rescore_fn or _rescore_default)
    if part in ('all', 'grids'):
        _gate_grids(lk, g, entries, by.get('grids', []), totals, visited, (0, 1) if part == 'all' else shard)
    for lid, e in entries.items():
        mine = e.get('kind') == 'grid' and part != 'data' and (part == 'all' or shard_of(e.get('name') or '', shard[1]) == shard[0]) or (e.get('kind') != 'grid' and part != 'grids')
        if mine:
            g.check('a union entry matched by an original', lid in visited)
    rep['totals'] = totals
    return _report(lk, g, rep, totals)

def _only_models(rec: dict, keep: set) -> dict:
    rec = json.loads(canon(rec))
    for lst in ITEM_LISTS:
        for it in rec.get(lst) or []:
            if not isinstance(it, dict):
                continue
            for k in list(it):
                if k.startswith('model_c_'):
                    m = k[len('model_c_'):]
                elif k.startswith('model_'):
                    m = k[len('model_'):]
                else:
                    continue
                if m not in keep:
                    it.pop(k)
            for pk in ('probe_m', 'probe_mc'):
                if isinstance(it.get(pk), dict):
                    it[pk] = {m: v for m, v in it[pk].items() if m in keep}
            for x in (it.get('legs') or []) + (it.get('climbs') or []):
                if isinstance(x, dict):
                    for mk in ('models', 'models_c', 'near'):
                        if isinstance(x.get(mk), dict):
                            x[mk] = {m: v for m, v in x[mk].items() if m in keep}
    return rec

def _boxes_ok(rec: dict, model: str) -> set:
    return {b.get('box') for b in rec.get('boxes') or [] if b.get('model') == model and b.get('ok') and (not b.get('error'))}

def _rescore_compare(rec: dict, res: dict) -> tuple[bool, bool]:
    import common
    models = set(rec.get('models') or []) | {b.get('model') for b in rec.get('boxes') or [] if b.get('model')}
    scored = {m for m in models if _boxes_ok(rec, m) and _boxes_ok(rec, m) <= _boxes_ok(res, m)}
    own = {m for m in models if not common.is_lead(m) and _boxes_ok(rec, m)}
    lists = [k for k in ITEM_LISTS if isinstance(rec.get(k), list)]
    return (own <= scored, _scored(_only_models(res, scored), lists) == _scored(_only_models(rec, scored), lists))

def _scored(rec: dict, lists) -> bytes:
    items = split_day(rec)[1]['items']
    return canon({k: items.get(k) for k in lists})

def _rescore(lk, g: Gate, entries: dict, days: dict, work: str, want: int, fn) -> dict:
    import common
    grids_by = {}
    for e in entries.values():
        if e['kind'] == 'grid':
            grids_by.setdefault(e['unit'].split('/')[0], []).append(e)
    cands = []
    for dk, rec in days.items():
        if rec.get('score_v') != common.SCORE_VERSION or rec.get('corr_v') or (not rec.get('grids')) or (dk not in grids_by):
            continue
        if not common.physics_matches(rec.get('physics_v'), rec.get('macro') or 'US'):
            continue
        cands.append((lk.name(f'rescore|{dk}'), dk))
    picked = [dk for _, dk in sorted(cands)[:want]]
    out = {'asked': want, 'reproducible': len(cands), 'checked': 0}
    if want > 0:
        g.check('days scored again (some the code reproduces)', bool(picked))
    for dk in picked:
        root = os.path.join(work, 'rescore')
        shutil.rmtree(root, ignore_errors=True)
        rec = days[dk]
        try:
            for e in grids_by[dk]:
                for p, b in untar(lk.get(e['obj'])).items():
                    dest = os.path.join(root, *e['unit'].split('/'), *p.split('/'))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, 'wb') as f:
                        f.write(b)
            res = fn(json.loads(canon(rec)), root, os.path.join(work, 'rescore-work'))
            own, same = _rescore_compare(rec, res)
        except Exception:
            own, same = (False, False)
        g.check('a day scored again: its own models all scored', own)
        g.check('a day scored again equals its stored scores', same)
        out['checked'] += 1
        shutil.rmtree(root, ignore_errors=True)
    return out
EXPECTED = '907 days, 4,070 flights, 24,420 legs, 50,187 climbs, 812 with grids'

def _report(lk, g: Gate, rep: dict, totals: dict) -> dict:
    rep.update({'passed': g.passed, 'ok': g.ok, 'failed': g.bad})
    try:
        put_json(lk, f"{MIGRATION}verify-{stamp()}{('-' + safe_part(rep['part']) if rep.get('part') not in (None, 'all') else '')}.lenc", rep)
    except lake.LakeError:
        log('the report could not be written (a read-only key): the log has it')
    if totals and totals.get('days'):
        log(f"gate: the originals hold {totals['days']} days, {totals['flights']} flights, {totals['legs']} legs, {totals['climbs']} climbs, {totals['with_grids']} with grids, {totals['station_items']} station items; {totals['flight_ids']} flight ids in all their copies (expected before the freeze: {EXPECTED})")
    if totals and totals.get('grid_files'):
        log(f"gate: {totals['grid_files']} grid files")
    rs = rep.get('rescored')
    if rs:
        log(f"gate: {rs['checked']} days scored again of {rs['asked']} asked ({rs['reproducible']} the code reproduces)")
    for w, c in sorted(g.bad.items()):
        log(f'  FAILED: {w} ({c})')
    log(f"gate: {('passed' if g.passed else 'FAILED')} ({sum(g.ok.values())} checks passed, {sum(g.bad.values())} failed)")
    return rep

def reassemble_releases(vault: lake.Lake, lk: lake.Lake, store=None) -> dict:
    g = Gate()
    rep = {'version': 2, 'at': stamp(), 'source': 'releases'}
    _, idx = newest(lk, f'{MIGRATION}union-', lambda o: o.get('complete'))
    if not idx:
        g.check('a complete union', False)
        return _report(vault, g, rep, {})
    entries = idx.get('entries') or {}
    checks = checks_of(vault, RELEASE_CHECKS)
    arts = None
    if store is not None:
        try:
            arts = store.list()
        except Exception:
            arts = None
    if g.check('the release assets listed again', arts is not None):
        for a in arts:
            if release_kind(a):
                g.check('a release asset copied and checked', a['id'] in checks)
    orig: dict[str, set] = {}
    waves: list[dict[str, bytes]] = []
    n = 0
    for cid, c in sorted(checks.items()):
        try:
            blob = vault.get_plain(verbatim_key(c) if c.get('obj') else f'{RELEASES}{cid}.enc')
            if not g.check("a release asset's copy against its check", sha(blob) == (c.get('obj_sha256') or c['sha256'])):
                continue
            files = tree_of(blob)
        except Exception:
            _err(g, "a release asset's copy read")
            continue
        if c['kind'] == 'wave':
            waves.append({rel: b for rel, b in files.items() if rel.startswith('flights/')})
            continue
        n += 1
        for rel, b in files.items():
            lid, kind = route(rel)
            if kind in ('day', 'pending'):
                for f in flights_in(json_of(rel, b)):
                    orig.setdefault(lk.name(f"flight|{f['id']}"), set()).add((kind, raw_sig(f)))
    g.check('a copy of the data among the release assets', n > 0)
    _flights_held(g, orig, lake_flights(lk, entries, g), "a release's flight")
    if waves:
        have: dict[str, set] = {}
        for e in entries.values():
            if e.get('kind') == 'wave':
                try:
                    for rel, b in untar(lk.get(e['obj'])).items():
                        have.setdefault(rel, set()).add(sha(b))
                except Exception:
                    _err(g, 'a wave copy read from the lake')
        for w in waves:
            for rel, b in w.items():
                g.check("a release's wave flight held by the lake", sha(b) in have.get(rel, set()))
    log(f'releases: {n} copies of the data read, {len(orig)} flight ids, {len(waves)} copies of the wave flights')
    return _report(vault, g, rep, {})

def _store(prefix: str, local: str | None):
    if local:
        return lake.LocalStore(local)
    return lake.S3Store.from_env(prefix)

def needs(what: str, source: str) -> list[str]:
    if source == 'releases':
        return ['VAULT', 'LAKE'] if what in ('verify', 'reassemble') else ['VAULT']
    return ['LAKE']

def missing(what: str, source: str, local: str | None=None, vault_local: str | None=None) -> list[str]:
    out = []
    for p in needs(what, source):
        if p == 'LAKE' and local or (p == 'VAULT' and vault_local):
            continue
        if not all((os.environ.get(f'{p}_S3_{n}') for n in ('ENDPOINT', 'BUCKET', 'KEY', 'SECRET'))):
            out.append(p)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='migrate.py')
    ap.add_argument('cmd', choices=('check', 'inventory', 'copy', 'union', 'reassemble', 'verify'))
    ap.add_argument('what', nargs='?', default='')
    ap.add_argument('--source', choices=('artifacts', 'releases'), default='artifacts')
    ap.add_argument('--repo', default=os.environ.get('GITHUB_REPOSITORY', ''))
    ap.add_argument('--seed')
    ap.add_argument('--minutes', type=float, default=MINUTES)
    ap.add_argument('--rescore', type=int, default=20)
    ap.add_argument('--work', default=os.path.join(tempfile.gettempdir(), 'calib-migrate'))
    ap.add_argument('--part', choices=('all', 'data', 'grids'), default='all')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--paused', default=os.path.join(HERE, 'paused'))
    ap.add_argument('--local')
    ap.add_argument('--vault-local')
    a = ap.parse_args(argv)
    try:
        shard = tuple((int(x) for x in a.shard.split('/')))
        if len(shard) != 2 or not 0 <= shard[0] < shard[1]:
            raise ValueError
    except ValueError:
        log('--shard is i/n, 0 <= i < n')
        return 1
    cmd = 'reassemble' if a.cmd == 'verify' else a.cmd
    what = a.what if cmd == 'check' else cmd
    what = 'reassemble' if what == 'verify' else what
    gone = missing(what, a.source, a.local, a.vault_local)
    if cmd == 'check':
        if what not in ('inventory', 'copy', 'union', 'reassemble'):
            log('check: nothing asked')
            ready = False
        elif a.source == 'releases' and what == 'union':
            log('the union runs in the calibration job alone: nothing to do here')
            ready = False
        elif gone:
            log(f"the {' and '.join(gone)} secrets ({', '.join((p + '_S3_*' for p in gone))}) are not set: nothing done")
            ready = False
        elif not os.environ.get('CALIB_DATA_KEY'):
            log('CALIB_DATA_KEY is not set: nothing done')
            ready = False
        else:
            ready = True
        out = os.environ.get('GITHUB_OUTPUT')
        if out:
            with open(out, 'a') as f:
                f.write(f"ready={('yes' if ready else 'no')}\n")
        return 0
    if gone:
        log(f"the {' and '.join(gone)} secrets are not set: nothing done")
        return 0
    if not os.environ.get('CALIB_DATA_KEY'):
        log('CALIB_DATA_KEY is not set: nothing done')
        return 1
    try:
        keys = lake.Keys.from_env()
        main_lake = lake.Lake(_store('LAKE', a.local), keys) if 'LAKE' in needs(what, a.source) else None
        vault = lake.Lake(_store('VAULT', a.vault_local), keys) if a.source == 'releases' else None
        if a.source == 'releases':
            if cmd == 'union':
                log('the union runs in the calibration job alone: nothing to do here')
                return 0
            src = Releases(a.repo) if cmd in ('inventory', 'copy') else None
            if cmd == 'inventory':
                inventory(vault, src, releases=True)
                return 0
            if cmd == 'copy':
                return 1 if copy(vault, src, minutes=a.minutes, releases=True)['failed'] else 0
            return 0 if reassemble_releases(vault, main_lake, Releases(a.repo))['passed'] else 1
        if cmd == 'inventory':
            inventory(main_lake, Artifacts(a.repo), a.seed)
            return 0
        if cmd == 'copy':
            return 1 if copy(main_lake, Artifacts(a.repo), a.seed, a.minutes)['failed'] else 0
        if cmd == 'union':
            idx = union(main_lake, a.minutes)
            return 0 if idx['complete'] and (not idx['conflicts']) else 1
        os.makedirs(a.work, exist_ok=True)
        store = Artifacts(a.repo) if a.part != 'grids' else None
        return 0 if reassemble(main_lake, a.work, a.rescore, store=store, seed=a.seed, frozen=os.path.exists(a.paused), part=a.part, shard=shard)['passed'] else 1
    except (Failed, lake.LakeError) as e:
        log(f'{cmd} failed: {e}')
        return 1
    except Exception as e:
        log(f'{cmd} failed ({type(e).__name__})')
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
