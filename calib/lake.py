from __future__ import annotations
import base64
import binascii
import datetime as _dt
import errno
import functools
import hashlib
import hmac
import http.client
import os
import re
import secrets
import shutil
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
MAGIC = b'LK1\x00'
NONCE = 12
TAG = 16
HEAD = len(MAGIC) + 1 + NONCE
SALT = b'lake/v1'
ROUNDS = 200000
NAME_HEX = 20
NAME_LABEL = b'lake names v1'
TRIES = 4
WAIT_S = (1, 2, 4)
TIMEOUT_S = 120
PART_BYTES = 64 << 20
MULTIPART_AT = PART_BYTES
MAX_PARTS = 10000
CHUNK = 1 << 20

class LakeError(Exception):
    pass

class LakeMissing(LakeError):
    pass

@functools.lru_cache(maxsize=4)
def _name_key(passphrase: str) -> bytes:
    return hmac.new(_derive(passphrase), NAME_LABEL, hashlib.sha256).digest()

@functools.lru_cache(maxsize=4)
def _derive(passphrase: str) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', passphrase.encode('utf-8'), SALT, ROUNDS, 32)

def _raw_key(text: str) -> bytes:
    t = text.strip()
    try:
        if len(t) == 64:
            return bytes.fromhex(t)
    except ValueError:
        pass
    try:
        k = base64.b64decode(t + '=' * (-len(t) % 4), altchars=b'-_' if '-' in t or '_' in t else None, validate=True)
    except (binascii.Error, ValueError):
        raise LakeError('lake: key id 2 is neither 64 hex digits nor base64') from None
    if len(k) != 32:
        raise LakeError('lake: key id 2 is not 32 bytes')
    return k

class Keys:

    def __init__(self, passphrase: str | None=None, raw2: bytes | str | None=None):
        self._pass = passphrase or None
        if isinstance(raw2, str) and (not raw2.strip()):
            raw2 = None
        self._raw2 = raw2

    @classmethod
    def from_env(cls, env=None) -> 'Keys':
        e = os.environ if env is None else env
        return cls(e.get('CALIB_DATA_KEY') or None, e.get('NOWCAST_KEY') or None)

    def ids(self) -> list[int]:
        return [i for i, ok in ((1, self._pass), (2, self._raw2)) if ok]

    def key(self, kid: int) -> bytes:
        if kid == 1 and self._pass:
            return _derive(self._pass)
        if kid == 2 and self._raw2:
            k = _raw_key(self._raw2) if isinstance(self._raw2, str) else bytes(self._raw2)
            if len(k) != 32:
                raise LakeError('lake: key id 2 is not 32 bytes')
            return k
        raise LakeError(f'lake: no key for key id {kid}')

    def name(self, logical_id: str) -> str:
        if not self._pass:
            raise LakeError('lake: names need CALIB_DATA_KEY')
        return hmac.new(_name_key(self._pass), ('key|' + logical_id).encode('utf-8'), hashlib.sha256).hexdigest()[:NAME_HEX]

def _aead(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise LakeError('lake: the cryptography package is missing (calib/requirements.txt)') from None
    return AESGCM(key)

def seal(plain: bytes, object_key: str, keys: Keys, kid: int=1) -> bytes:
    if not 0 < kid < 256:
        raise LakeError('lake: a key id is one byte, not 0')
    nonce = secrets.token_bytes(NONCE)
    return MAGIC + bytes([kid]) + nonce + _aead(keys.key(kid)).encrypt(nonce, bytes(plain), object_key.encode('utf-8'))

def is_lenc(blob: bytes) -> bool:
    return blob[:len(MAGIC)] == MAGIC

def key_id(blob: bytes) -> int:
    if not is_lenc(blob) or len(blob) < HEAD + TAG:
        raise LakeError('lake: not a lenc object')
    return blob[len(MAGIC)]

def unseal(blob: bytes, object_key: str, keys: Keys) -> bytes:
    kid = key_id(blob)
    nonce = blob[len(MAGIC) + 1:HEAD]
    try:
        from cryptography.exceptions import InvalidTag
    except ImportError:
        raise LakeError('lake: the cryptography package is missing (calib/requirements.txt)') from None
    try:
        return _aead(keys.key(kid)).decrypt(nonce, blob[HEAD:], object_key.encode('utf-8'))
    except InvalidTag:
        raise LakeError('lake: an object failed to decrypt (wrong key, wrong name or altered)') from None

def check_key(object_key: str) -> str:
    parts = object_key.split('/')
    if not object_key or len(object_key) > 1024 or any((not p or p.startswith('.') for p in parts)) or any((not 32 < ord(c) < 127 or c == '\\' for c in object_key)):
        raise LakeError('lake: an object key is malformed')
    return object_key

def check_prefix(prefix: str) -> str:
    if prefix:
        check_key(prefix.rstrip('/') or 'x')
    return prefix

class LocalStore:

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def _path(self, object_key: str) -> str:
        return os.path.join(self.root, *check_key(object_key).split('/'))

    def put_if_absent(self, object_key: str, data: bytes) -> bool:
        return self._link_in(object_key, lambda f: f.write(data))

    def put_file_if_absent(self, object_key: str, path: str) -> bool:

        def fill(f):
            with open(path, 'rb') as src:
                shutil.copyfileobj(src, f, CHUNK)
        return self._link_in(object_key, fill)

    def _link_in(self, object_key: str, fill) -> bool:
        dest = self._path(object_key)
        tmp = None
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), prefix='.put-')
            with os.fdopen(fd, 'wb') as f:
                fill(f)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, dest)
            except FileExistsError:
                return False
            return True
        except OSError as e:
            raise LakeError(f'lake: a local write failed ({_errno(e)})') from None
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def get(self, object_key: str) -> bytes | None:
        path = self._path(object_key)
        try:
            with open(path, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise LakeError(f'lake: a local read failed ({_errno(e)})') from None

    def exists(self, object_key: str) -> bool:
        return os.path.isfile(self._path(object_key))

    def digest(self, object_key: str) -> tuple[str, int] | None:
        path = self._path(object_key)
        try:
            return file_digest(path)
        except FileNotFoundError:
            return None
        except OSError as e:
            raise LakeError(f'lake: a local read failed ({_errno(e)})') from None

    def get_file(self, object_key: str, path: str) -> tuple[str, int] | None:
        src = self._path(object_key)
        try:
            f = open(src, 'rb')
        except FileNotFoundError:
            return None
        except OSError as e:
            raise LakeError(f'lake: a local read failed ({_errno(e)})') from None
        sink = _Sink(path)
        try:
            with f:
                sink.start()
                for b in iter(lambda: f.read(CHUNK), b''):
                    sink.write(b)
        except OSError as e:
            raise LakeError(f'lake: a local copy failed ({_errno(e)})') from None
        finally:
            sink.close()
        return sink.result()

    def list(self, prefix: str='') -> list[str]:
        check_prefix(prefix)
        out = []
        for d, dirs, files in os.walk(self.root):
            dirs[:] = sorted((x for x in dirs if not x.startswith('.')))
            for f in sorted(files):
                if f.startswith('.'):
                    continue
                k = os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, '/')
                if k.startswith(prefix):
                    out.append(k)
        return sorted(out)

    def sizes(self, prefix: str='') -> dict[str, int]:
        out = {}
        for k in self.list(prefix):
            try:
                out[k] = os.path.getsize(self._path(k))
            except OSError:
                continue
        return out

def _errno(e: OSError) -> str:
    return errno.errorcode.get(e.errno or 0, type(e).__name__)

def _sha_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def file_digest(path: str) -> tuple[str, int]:
    h, n = (hashlib.sha256(), 0)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
            n += len(b)
    return (h.hexdigest(), n)

class _Sink:

    def __init__(self, path: str | None=None):
        self.path, self.f, self.h, self.n = (path, None, hashlib.sha256(), 0)

    def start(self) -> None:
        self.h, self.n = (hashlib.sha256(), 0)
        if self.path:
            if self.f:
                self.f.close()
            self.f = open(self.path, 'wb')

    def write(self, b: bytes) -> None:
        self.h.update(b)
        self.n += len(b)
        if self.f:
            self.f.write(b)

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None

    def result(self) -> tuple[str, int]:
        return (self.h.hexdigest(), self.n)

def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def _quote(s: str, safe: str='') -> str:
    return urllib.parse.quote(s, safe='-_.~' + safe)

def sign_v4(method: str, url: str, headers: dict, payload_sha: str, key_id: str, secret: str, region: str, now: _dt.datetime, service: str='s3') -> dict:
    u = urllib.parse.urlsplit(url)
    amz = now.strftime('%Y%m%dT%H%M%SZ')
    day = amz[:8]
    h = {k.lower(): ' '.join(str(v).split()) for k, v in headers.items()}
    h['host'] = u.netloc
    h['x-amz-date'] = amz
    h['x-amz-content-sha256'] = payload_sha
    names = sorted(h)
    q = sorted(urllib.parse.parse_qsl(u.query, keep_blank_values=True))
    query = '&'.join((f'{_quote(k)}={_quote(v)}' for k, v in q))
    canon = '\n'.join([method, u.path or '/', query, ''.join((f'{n}:{h[n]}\n' for n in names)), ';'.join(names), payload_sha])
    scope = f'{day}/{region}/{service}/aws4_request'
    to_sign = '\n'.join(['AWS4-HMAC-SHA256', amz, scope, _sha_hex(canon.encode('utf-8'))])
    k = _hmac(('AWS4' + secret).encode('utf-8'), day)
    for part in (region, service, 'aws4_request'):
        k = hmac.new(k, part.encode('utf-8'), hashlib.sha256).digest()
    sig = hmac.new(k, to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    out = {n: h[n] for n in names}
    out['authorization'] = f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, SignedHeaders={';'.join(names)}, Signature={sig}"
    return out
_S3NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'
_TRANSIENT = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError, http.client.HTTPException)

class S3Store:

    def __init__(self, endpoint: str, bucket: str, key_id: str, secret: str, region: str='auto', sleep=time.sleep, clock=None, opener=None):
        if not endpoint.lower().startswith(('https://', 'http://')):
            raise LakeError("lake: the store's endpoint is not an http(s) address")
        self._endpoint = endpoint.rstrip('/')
        self._bucket = bucket
        self._key_id = key_id
        self._secret = secret
        self._region = region
        self._sleep = sleep
        self._clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))
        self._open = opener or urllib.request.build_opener(_NoRedirect).open
        self.calls = 0
        self.retries = 0

    @classmethod
    def from_env(cls, prefix: str='LAKE', env=None, **kw) -> 'S3Store | None':
        e = os.environ if env is None else env
        vals = [e.get(f'{prefix}_S3_{n}') or '' for n in ('ENDPOINT', 'BUCKET', 'KEY', 'SECRET')]
        return cls(*vals, **kw) if all(vals) else None

    def __repr__(self) -> str:
        return 'S3Store(<from secrets>)'

    def _url(self, object_key: str | None=None, query: dict | None=None) -> str:
        path = '/' + _quote(self._bucket)
        if object_key is not None:
            path += '/' + _quote(check_key(object_key), safe='/')
        q = '&'.join((f'{_quote(k)}={_quote(v)}' for k, v in sorted((query or {}).items())))
        return self._endpoint + path + ('?' + q if q else '')

    def _call(self, method: str, url: str, body: bytes=b'', headers: dict | None=None, ok=(200,), final=()) -> tuple[int, bytes]:
        status, data, _ = self._request(method, url, body, headers, ok, final)
        return (status, data)

    def _request(self, method: str, url: str, body: bytes=b'', headers: dict | None=None, ok=(200,), final=()) -> tuple[int, bytes, dict]:
        sha = _sha_hex(body)
        last = 'no answer'
        for i in range(TRIES):
            if i:
                self.retries += 1
                self._sleep(WAIT_S[min(i - 1, len(WAIT_S) - 1)])
            self.calls += 1
            hs = sign_v4(method, url, dict(headers or {}), sha, self._key_id, self._secret, self._region, self._clock())
            if body:
                hs['content-length'] = str(len(body))
            try:
                req = urllib.request.Request(url, data=body if method in ('PUT', 'POST') else None, method=method, headers=hs)
                with self._open(req, timeout=TIMEOUT_S) as r:
                    status, data, rh = (r.status, r.read(), _headers(r))
            except urllib.error.HTTPError as e:
                status, rh = (e.code, _headers(e))
                try:
                    data = e.read()
                except Exception:
                    data = b''
                finally:
                    e.close()
            except _TRANSIENT as e:
                last = type(e).__name__
                continue
            except Exception as e:
                raise LakeError(f'lake: {method} failed ({type(e).__name__})') from None
            if status in ok or status in final:
                return (status, data, rh)
            if status == 429 or status >= 500:
                last = f'HTTP {status}'
                continue
            raise LakeError(f'lake: {method} refused (HTTP {status})')
        raise LakeError(f'lake: {method} failed after {TRIES} tries ({last})')

    def put_if_absent(self, object_key: str, data: bytes) -> bool:
        if len(data) > MULTIPART_AT:
            view = memoryview(data)
            return self._put_parts(object_key, len(data), lambda off, n: bytes(view[off:off + n]))
        for i in range(TRIES):
            if i:
                self.retries += 1
                self._sleep(WAIT_S[min(i - 1, len(WAIT_S) - 1)])
            status, _ = self._call('PUT', self._url(object_key), data, {'If-None-Match': '*', 'Content-Type': 'application/octet-stream'}, ok=(200, 201, 204), final=(409, 412))
            if status not in (409, 412):
                return True
            if status == 412 or self.exists(object_key):
                return False
        raise LakeError(f'lake: PUT met a conflict that did not clear after {TRIES} tries (HTTP 409)')

    def put_file_if_absent(self, object_key: str, path: str) -> bool:
        try:
            size = os.path.getsize(path)
            if size <= MULTIPART_AT:
                with open(path, 'rb') as f:
                    data = f.read()
                return self.put_if_absent(object_key, data)
            with open(path, 'rb') as f:

                def read(off: int, n: int) -> bytes:
                    f.seek(off)
                    return f.read(n)
                return self._put_parts(object_key, size, read)
        except OSError as e:
            raise LakeError(f'lake: a local read failed ({_errno(e)})') from None

    def _put_parts(self, object_key: str, size: int, read) -> bool:
        n = -(-size // PART_BYTES)
        if n > MAX_PARTS:
            raise LakeError('lake: an object is too large for one upload in parts')
        if self.exists(object_key):
            return False
        _, data = self._call('POST', self._url(object_key, {'uploads': ''}), b'', {'Content-Type': 'application/octet-stream'})
        upload = _xml_find(data, 'UploadId')
        if not upload:
            raise LakeError('lake: an upload in parts did not start (no upload id)')
        tags = []
        for i in range(n):
            off = i * PART_BYTES
            want = min(PART_BYTES, size - off)
            body = read(off, want)
            if len(body) != want:
                raise LakeError('lake: a file changed while it was sent')
            _, _, h = self._request('PUT', self._url(object_key, {'partNumber': str(i + 1), 'uploadId': upload}), body, {'Content-Type': 'application/octet-stream'})
            del body
            if not h.get('etag'):
                raise LakeError('lake: a part was taken without an ETag')
            tags.append(h['etag'])
        done = ('<CompleteMultipartUpload>' + ''.join((f'<Part><PartNumber>{i + 1}</PartNumber><ETag>{_xml_escape(t)}</ETag></Part>' for i, t in enumerate(tags))) + '</CompleteMultipartUpload>').encode()
        url = self._url(object_key, {'uploadId': upload})
        for i in range(TRIES):
            if i:
                self.retries += 1
                self._sleep(WAIT_S[min(i - 1, len(WAIT_S) - 1)])
            status, data = self._call('POST', url, done, {'If-None-Match': '*', 'Content-Type': 'application/xml'}, final=(404, 409, 412))
            if status == 200:
                err = _xml_error(data)
                if err is None:
                    return True
                if err in ('InternalError', 'SlowDown', 'ServiceUnavailable'):
                    continue
                raise LakeError(f'lake: an upload in parts was refused ({err})')
            if status == 412:
                return False
            if status == 404:
                if self.exists(object_key):
                    return False
                raise LakeError('lake: an upload in parts was lost (HTTP 404)')
            if self.exists(object_key):
                return False
        raise LakeError(f'lake: completing an upload in parts did not succeed after {TRIES} tries')

    def get(self, object_key: str) -> bytes | None:
        status, data = self._call('GET', self._url(object_key), final=(404,))
        return None if status == 404 else data

    def digest(self, object_key: str) -> tuple[str, int] | None:
        sink = _Sink()
        return sink.result() if self._stream(object_key, sink) else None

    def get_file(self, object_key: str, path: str) -> tuple[str, int] | None:
        sink = _Sink(path)
        try:
            found = self._stream(object_key, sink)
        finally:
            sink.close()
        return sink.result() if found else None

    def _stream(self, object_key: str, sink: '_Sink') -> bool:
        url = self._url(object_key)
        last = 'no answer'
        for i in range(TRIES):
            if i:
                self.retries += 1
                self._sleep(WAIT_S[min(i - 1, len(WAIT_S) - 1)])
            self.calls += 1
            hs = sign_v4('GET', url, {}, _sha_hex(b''), self._key_id, self._secret, self._region, self._clock())
            try:
                sink.start()
            except OSError as e:
                raise LakeError(f'lake: a local write failed ({_errno(e)})') from None
            try:
                req = urllib.request.Request(url, method='GET', headers=hs)
                with self._open(req, timeout=TIMEOUT_S) as r:
                    want = _headers(r).get('content-length', '')
                    for b in iter(lambda: r.read(CHUNK), b''):
                        sink.write(b)
                    if want.isdigit() and sink.n != int(want):
                        raise http.client.IncompleteRead(b'', int(want) - sink.n)
                return True
            except urllib.error.HTTPError as e:
                status = e.code
                e.close()
                if status == 404:
                    return False
                if status == 429 or status >= 500:
                    last = f'HTTP {status}'
                    continue
                raise LakeError(f'lake: GET refused (HTTP {status})') from None
            except _TRANSIENT as e:
                last = type(e).__name__
                continue
            except Exception as e:
                raise LakeError(f'lake: GET failed ({type(e).__name__})') from None
        raise LakeError(f'lake: GET failed after {TRIES} tries ({last})')

    def get_tagged(self, object_key: str) -> tuple[bytes, str] | None:
        status, data, h = self._request('GET', self._url(object_key), final=(404,))
        return None if status == 404 else (data, h.get('etag', ''))

    def put_over(self, object_key: str, data: bytes, etag: str | None) -> bool:
        cond = {'If-Match': etag} if etag else {'If-None-Match': '*'}
        status, _ = self._call('PUT', self._url(object_key), bytes(data), {**cond, 'Content-Type': 'application/json'}, ok=(200, 201, 204), final=(409, 412))
        return status not in (409, 412)

    def exists(self, object_key: str) -> bool:
        status, _ = self._call('HEAD', self._url(object_key), final=(404,))
        return status != 404

    def list(self, prefix: str='') -> list[str]:
        return sorted(self.sizes(prefix))

    def sizes(self, prefix: str='') -> dict[str, int]:
        check_prefix(prefix)
        out, token = ({}, None)
        while True:
            q = {'list-type': '2', 'prefix': prefix}
            if token:
                q['continuation-token'] = token
            _, data = self._call('GET', self._url(None, q))
            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                raise LakeError('lake: a listing could not be read') from None
            ns = _S3NS if root.tag.startswith(_S3NS) else ''
            for c in root.findall(f'{ns}Contents'):
                k = c.findtext(f'{ns}Key')
                if k:
                    try:
                        out[k] = int(c.findtext(f'{ns}Size') or -1)
                    except ValueError:
                        out[k] = -1
            if (root.findtext(f'{ns}IsTruncated') or '').lower() != 'true':
                return out
            token = root.findtext(f'{ns}NextContinuationToken')
            if not token:
                raise LakeError('lake: a truncated listing without a continuation token')

def _headers(r) -> dict:
    h = getattr(r, 'headers', None)
    try:
        return {str(k).lower(): str(v) for k, v in h.items()} if h else {}
    except AttributeError:
        return {}

def _xml_root(data: bytes):
    try:
        root = ET.fromstring(data.lstrip())
    except ET.ParseError:
        return (None, '')
    return (root, root.tag[:root.tag.index('}') + 1] if root.tag.startswith('{') else '')

def _xml_find(data: bytes, name: str) -> str | None:
    root, ns = _xml_root(data)
    return None if root is None else root.findtext(f'{ns}{name}')

def _xml_error(data: bytes) -> str | None:
    root, ns = _xml_root(data)
    if root is None:
        return 'an unreadable answer'
    if root.tag != f'{ns}Error':
        return None
    code = root.findtext(f'{ns}Code') or ''
    return code if re.fullmatch('[A-Za-z]{1,40}', code) else 'an error'

def _xml_escape(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

class _NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, *a, **k):
        return None

@dataclass
class PutResult:
    status: str
    sha256: str
    bytes: int

@dataclass
class Lake:
    store: object
    keys: Keys = field(default_factory=Keys.from_env)
    counts: dict = field(default_factory=dict)

    def _n(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def name(self, logical_id: str) -> str:
        return self.keys.name(logical_id)

    def put(self, object_key: str, plain: bytes, kid: int=1, strict: bool=True) -> PutResult:
        blob = seal(plain, check_key(object_key), self.keys, kid)
        if self.store.put_if_absent(object_key, blob):
            self._n('written')
            return PutResult('written', _sha_hex(blob), len(blob))
        stored = self.store.get(object_key)
        if stored is None:
            raise LakeError('lake: an object reported as existing could not be read')
        same = hmac.compare_digest(unseal(stored, object_key, self.keys), bytes(plain))
        return self._exists(same, _sha_hex(stored), len(stored), strict)

    def put_plain(self, object_key: str, data: bytes, strict: bool=True) -> PutResult:
        if self.store.put_if_absent(object_key, bytes(data)):
            self._n('written')
            return PutResult('written', _sha_hex(data), len(data))
        stored = self.store.get(object_key)
        if stored is None:
            raise LakeError('lake: an object reported as existing could not be read')
        return self._exists(hmac.compare_digest(stored, bytes(data)), _sha_hex(stored), len(stored), strict)

    def put_plain_file(self, object_key: str, path: str, strict: bool=True) -> PutResult:
        check_key(object_key)
        try:
            want, n = file_digest(path)
        except OSError as e:
            raise LakeError(f'lake: a local read failed ({_errno(e)})') from None
        if self.store.put_file_if_absent(object_key, path):
            self._n('written')
            return PutResult('written', want, n)
        got = self.store.digest(object_key)
        if got is None:
            raise LakeError('lake: an object reported as existing could not be read')
        return self._exists(hmac.compare_digest(got[0], want), got[0], got[1], strict)

    def _exists(self, same: bool, sha256: str, n: int, strict: bool) -> PutResult:
        self._n('same' if same else 'differs')
        if not same and strict:
            raise LakeError('lake: an object exists with other contents')
        return PutResult('same' if same else 'differs', sha256, n)

    def get(self, object_key: str) -> bytes:
        blob = self.store.get(object_key)
        if blob is None:
            raise LakeMissing('lake: an object is missing')
        self._n('read')
        return unseal(blob, object_key, self.keys)

    def get_plain(self, object_key: str) -> bytes:
        blob = self.store.get(object_key)
        if blob is None:
            raise LakeMissing('lake: an object is missing')
        self._n('read')
        return blob

    def digest_plain(self, object_key: str) -> tuple[str, int]:
        got = self.store.digest(check_key(object_key))
        if got is None:
            raise LakeMissing('lake: an object is missing')
        self._n('read')
        return got

    def get_plain_file(self, object_key: str, path: str) -> tuple[str, int]:
        got = self.store.get_file(check_key(object_key), path)
        if got is None:
            raise LakeMissing('lake: an object is missing')
        self._n('read')
        return got

    def exists(self, object_key: str) -> bool:
        return self.store.exists(object_key)

    def list(self, prefix: str='') -> list[str]:
        return self.store.list(prefix)

    def summary(self) -> str:
        parts = [f'{k} {v}' for k, v in sorted(self.counts.items())]
        r = getattr(self.store, 'retries', 0)
        return 'lake: ' + (', '.join(parts) or 'nothing done') + (f', {r} retried' if r else '')

def from_env(prefix: str='LAKE', local: str | None=None, env=None) -> Lake | None:
    store = LocalStore(local) if local else S3Store.from_env(prefix, env)
    return Lake(store, Keys.from_env(env)) if store else None
