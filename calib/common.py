from __future__ import annotations
import ast
import glob
import gzip
import hashlib
import hmac
import json
import math
import os
import subprocess
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPELINE = os.path.join(ROOT, 'pipeline')
PUB = os.environ.get('CALIB_PUB') or next((p for p in (os.path.join(ROOT, d) for d in ('public', 'pipeline')) if os.path.isdir(os.path.join(p, 'geo', 'ridge'))), ROOT)
REGION = {**{s: 'PNW' for s in ('WA', 'OR', 'ID')}, **{s: 'CA-NV' for s in ('CA', 'NV')}, **{s: 'Rockies' for s in ('CO', 'UT', 'WY', 'MT')}, **{s: 'Southwest' for s in ('AZ', 'NM')}, **{s: 'TX-Plains' for s in ('TX', 'OK', 'KS', 'NE')}, **{s: 'Midwest' for s in ('OH', 'IL', 'MN', 'IN', 'IA', 'WI', 'MO', 'MI', 'ND', 'SD')}, **{s: 'Southeast' for s in ('FL', 'NC', 'SC', 'GA', 'TN', 'AL', 'MS', 'LA', 'AR', 'KY')}, **{s: 'MidAtlantic' for s in ('VA', 'PA', 'WV', 'NJ', 'DE', 'MD', 'DC')}, **{s: 'Northeast' for s in ('NY', 'VT', 'MA', 'NH', 'ME', 'CT', 'RI')}}
FLAT_REGIONS = {'Midwest', 'TX-Plains', 'Southeast', 'Northeast', 'MidAtlantic'}
MACRO = {'US': {'models': tuple((m for m in os.environ.get('CALIB_MODELS', 'hrrr,gfs').split(',') if m)) or ('hrrr',), 'tz': 'America/Chicago'}}
REGION_MACRO = {r: 'US' for r in set(REGION.values())}
MODEL_INDEX = {'hrrr': 0, 'gfs': 1, 'ecmwf': 2, 'icon': 3}

def macro_of(region: str | None) -> str:
    return REGION_MACRO.get(region, 'US')

def models_for(regions) -> tuple:
    out = []
    for a in sorted({macro_of(r) for r in regions} or {'US'}):
        out += [m for m in MACRO[a]['models'] if m not in out]
    return tuple(out)

def primary_model(regions=()) -> str:
    return models_for(regions)[0]
RELIEF_FLAT = 300
TZ = {**{s: 'America/New_York' for s in 'CT DE FL GA MA MD ME NC NH NJ NY OH PA RI SC VA VT WV DC MI IN KY'.split()}, **{s: 'America/Chicago' for s in 'AL AR IA IL KS LA MN MO MS ND NE OK SD TN TX WI'.split()}, **{s: 'America/Denver' for s in 'CO MT NM UT WY ID'.split()}, 'AZ': 'America/Phoenix', **{s: 'America/Los_Angeles' for s in 'CA NV OR WA'.split()}}
CENTROID = {'AL': (-86.8, 32.8), 'AZ': (-111.7, 34.3), 'AR': (-92.4, 34.9), 'CA': (-119.4, 37.2), 'CO': (-105.5, 39.0), 'CT': (-72.7, 41.6), 'DE': (-75.5, 39.0), 'FL': (-81.7, 28.6), 'GA': (-83.4, 32.7), 'ID': (-114.6, 44.4), 'IL': (-89.2, 40.0), 'IN': (-86.3, 39.9), 'IA': (-93.5, 42.1), 'KS': (-98.4, 38.5), 'KY': (-85.3, 37.5), 'LA': (-92.0, 31.1), 'ME': (-69.2, 45.4), 'MD': (-76.8, 39.0), 'MA': (-71.8, 42.3), 'MI': (-85.4, 44.3), 'MN': (-94.3, 46.3), 'MS': (-89.7, 32.7), 'MO': (-92.5, 38.4), 'MT': (-109.6, 47.0), 'NE': (-99.8, 41.5), 'NV': (-116.6, 39.3), 'NH': (-71.6, 43.7), 'NJ': (-74.7, 40.2), 'NM': (-106.1, 34.4), 'NY': (-75.5, 42.9), 'NC': (-79.4, 35.6), 'ND': (-100.5, 47.5), 'OH': (-82.8, 40.3), 'OK': (-97.5, 35.6), 'OR': (-120.6, 43.9), 'PA': (-77.8, 40.9), 'RI': (-71.5, 41.7), 'SC': (-80.9, 33.9), 'SD': (-100.2, 44.4), 'TN': (-86.4, 35.9), 'TX': (-99.3, 31.5), 'UT': (-111.7, 39.3), 'VT': (-72.7, 44.1), 'VA': (-78.8, 37.5), 'WA': (-120.5, 47.4), 'WV': (-80.6, 38.6), 'WI': (-89.9, 44.6), 'WY': (-107.6, 43.0)}

def nearest_state(lon: float, lat: float) -> str:
    k = math.cos(math.radians(lat))
    return min(CENTROID, key=lambda s: ((CENTROID[s][0] - lon) * k) ** 2 + (CENTROID[s][1] - lat) ** 2)

def load_aircraft() -> dict:
    with open(os.path.join(HERE, 'aircraft.json'), encoding='utf-8') as f:
        return json.load(f)['types']

def aircraft_version(types: dict | None=None) -> str:
    return hashlib.sha256(json.dumps(types if types is not None else load_aircraft(), sort_keys=True).encode()).hexdigest()[:12]
DAY_VERSION = 3

def _stripped(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if path.endswith('.py'):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if isinstance(body, list) and body and all((isinstance(x, ast.stmt) for x in body)):
                body[:] = [x for x in body if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str))]
                if not body and (not isinstance(node, ast.Module)):
                    body.append(ast.Pass())
        return ast.unparse(tree) + '\n'
    if path.endswith(('.cpp', '.h')):
        out, i, n = ([], 0, len(text))
        while i < n:
            c = text[i]
            if c in '"\'':
                j = i + 1
                while j < n and text[j] != c:
                    j += 2 if text[j] == '\\' else 1
                out.append(text[i:j + 1])
                i = j + 1
            elif text.startswith('//', i):
                j = text.find('\n', i)
                i = n if j < 0 else j
            elif text.startswith('/*', i):
                j = text.find('*/', i + 2)
                i = n if j < 0 else j + 2
                out.append(' ')
            else:
                out.append(c)
                i += 1
        return '\n'.join((line.rstrip() for line in ''.join(out).splitlines() if line.strip())) + '\n'
    return text

def _version_of(paths: list[str]) -> str:
    h = hashlib.sha256()
    found = False
    for p in sorted(set(paths)):
        if os.path.isfile(p):
            found = True
            h.update(os.path.basename(p).encode() + b'\x00' + _stripped(p).encode() + b'\x00')
    return h.hexdigest()[:12] if found else 'none'
PHYSICS_FILES = [os.path.join(PIPELINE, f'{m}.py') for m in ('build', 'physics', 'physics_native', 'sky', 'runfields', 'leelattice', 'runwave', 'shade', 'sources', 'grib', 'fronts', 'pack')] + glob.glob(os.path.join(PIPELINE, 'native', '*.cpp')) + glob.glob(os.path.join(PIPELINE, 'native', '*.h')) + [os.path.join(HERE, 'build_day.py')]
PHYSICS_VERSION = _version_of(PHYSICS_FILES)
SCORE_FILES = [os.path.join(HERE, f) for f in ('evaluate.js', 'lines.js', 'site.js', 'gbm.js', 'corrections.mjs')] + glob.glob(os.path.join(ROOT, 'public', 'js', 'shared', '*')) + glob.glob(os.path.join(ROOT, 'public', 'js', 'ridgex.js')) + glob.glob(os.path.join(ROOT, 'public', 'js', 'png*.js'))
SCORE_VERSION = _version_of(SCORE_FILES)

def pilot_key(user_id) -> str | None:
    pw = os.environ.get('CALIB_DATA_KEY')
    if user_id is None or not pw:
        return None
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    k = hmac.new(pw.encode(), b'calib pilot key v1', hashlib.sha256).digest()
    return hmac.new(k, f'weglide-user:{uid}'.encode(), hashlib.sha256).hexdigest()[:16]

def holidays(year: int) -> set:
    from datetime import date, timedelta

    def nth(month, wd, n):
        if n > 0:
            d = date(year, month, 1)
            return d + timedelta(days=(wd - d.weekday()) % 7 + 7 * (n - 1))
        d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        return d - timedelta(days=(d.weekday() - wd) % 7)
    fixed = [date(year, 1, 1), date(year, 6, 19), date(year, 7, 4), date(year, 11, 11), date(year, 12, 25)]
    out = {nth(1, 0, 3), nth(2, 0, 3), nth(5, 0, -1), nth(9, 0, 1), nth(10, 0, 2), nth(11, 3, 4), nth(11, 3, 4) + timedelta(days=1)}
    for d in fixed:
        out.add(d)
        out.add(d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d)
    return out

def day_type(day: str) -> str:
    from datetime import date
    d = date.fromisoformat(day)
    return 'weekend' if d.weekday() >= 5 or d in holidays(d.year) else 'weekday'

def read_json(path: str):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def write_json(path: str, obj, indent=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    op = gzip.open if path.endswith('.gz') else open
    with op(tmp, 'wt', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, separators=None if indent else (',', ':'))
    os.replace(tmp, path)
OPENSSL = ['openssl', 'enc', '-aes-256-cbc', '-pbkdf2', '-iter', '200000', '-md', 'sha256', '-pass', 'env:CALIB_DATA_KEY']

def _need_key() -> None:
    if not os.environ.get('CALIB_DATA_KEY'):
        sys.exit('calib: CALIB_DATA_KEY is not set')

def encrypt(src: str, dst: str) -> None:
    _need_key()
    subprocess.run(OPENSSL + ['-salt', '-in', src, '-out', dst], check=True)

def decrypt(src: str, dst: str) -> None:
    _need_key()
    subprocess.run(OPENSSL + ['-d', '-in', src, '-out', dst], check=True)
_T0 = time.time()
PRIVATE_LOG = None

def log(msg: str) -> None:
    print(f'[{time.time() - _T0:6.0f}s] {msg}', flush=True)

def note(msg: str) -> None:
    if PRIVATE_LOG:
        os.makedirs(os.path.dirname(os.path.abspath(PRIVATE_LOG)), exist_ok=True)
        with open(PRIVATE_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
