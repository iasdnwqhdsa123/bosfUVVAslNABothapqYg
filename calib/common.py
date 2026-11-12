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
EU_REGION = {**{c: 'Germany-Benelux' for c in ('DE', 'NL', 'BE', 'LU')}, **{c: 'Alps' for c in ('AT', 'CH', 'LI')}, 'FR': 'France', 'MC': 'France', **{c: 'Iberia' for c in ('ES', 'PT', 'AD')}, **{c: 'UK-Ireland' for c in ('GB', 'IE', 'IM', 'JE', 'GG')}, **{c: 'Poland-Czech' for c in ('PL', 'CZ', 'SK', 'HU', 'LT', 'LV', 'EE')}, **{c: 'Scandinavia' for c in ('SE', 'NO', 'FI', 'DK', 'IS')}, **{c: 'Italy-Apennines' for c in ('IT', 'SM', 'MT')}, **{c: 'Balkans' for c in ('SI', 'HR', 'BA', 'RS', 'ME', 'MK', 'AL', 'GR', 'BG', 'RO', 'XK', 'MD')}}
EU_FLAT_REGIONS = {'Germany-Benelux', 'Poland-Czech'}
FLAT_REGIONS |= EU_FLAT_REGIONS
ALPS_BOXES = ((5.0, 45.3, 16.8, 48.1), (5.0, 43.7, 7.8, 45.3), (9.5, 44.9, 13.5, 45.3))
EU_BOX = (-25.0, 34.0, 44.75, 71.5)
MACRO = {'US': {'models': tuple((m for m in os.environ.get('CALIB_MODELS', 'hrrr,gfs').split(',') if m)) or ('hrrr',), 'tz': 'America/Chicago'}, 'EU': {'models': tuple((m for m in os.environ.get('CALIB_EU_MODELS', 'ecmwf,icon,icond2,arome').split(',') if m)) or ('ecmwf',), 'tz': 'Europe/Berlin'}}
REGION_MACRO = {**{r: 'US' for r in set(REGION.values())}, **{r: 'EU' for r in set(EU_REGION.values())}}
MODEL_INDEX = {'hrrr': 0, 'gfs': 1, 'ecmwf': 2, 'icon': 3, 'icond2': 4, 'arome': 5}
MODEL_MACRO = {'hrrr': 'US', 'gfs': 'US', 'ecmwf': 'EU', 'icon': 'EU', 'icond2': 'EU', 'arome': 'EU'}
MODEL_LABEL = {'hrrr': 'HRRR', 'gfs': 'GFS', 'ecmwf': 'ECMWF', 'icon': 'ICON', 'icond2': 'ICON-D2', 'arome': 'AROME'}
MACRO_LABEL = {'US': 'United States', 'EU': 'Europe'}
EU_SHARE = min(1.0, max(0.0, float(os.environ.get('CALIB_EU_SHARE', '0.4') or 0.4)))
EU_FIRST = max(0, int(os.environ.get('CALIB_EU_FIRST', '120') or 0))
EU_ICON_SHARE = min(1.0, max(0.0, float(os.environ.get('CALIB_EU_ICON_SHARE', '1') or 0)))
SAMPLED = {'icon', 'icond2'}
ICON_FIRST = '2025-01-17'
AROME_FIRST = '2025-01-14'
FIRST_DAY = {'icon': ICON_FIRST, 'icond2': ICON_FIRST, 'arome': AROME_FIRST}
PART_DOMAIN = {'icond2', 'arome'}
ADDED = ('icon', 'icond2', 'arome')
EU_START = '2024-03-01'
LEADS = {m: tuple((int(x) for x in ns.split('+') if x.strip().isdigit())) for m, _, ns in (p.partition(':') for p in os.environ.get('CALIB_LEADS', 'hrrr:1,gfs:1+3+5+7,ecmwf:1+3+5+7').split(',') if ':' in p)}
LEAD_SHARE = min(1.0, max(0.0, float(os.environ.get('CALIB_LEAD_SHARE', '1') or 0)))
LEAD_EXTRA = {'EU': ('gfs',)}
RUNS_FIRST = {'ecmwf': '2026-07-31'}
ECMWF_RUNS_FIRST = RUNS_FIRST['ecmwf']
LEAD_RUN_H = {'US': {'hrrr': 12, 'gfs': 12}, 'EU': {'gfs': 0, 'ecmwf': 0}}

def base_model(model: str) -> str:
    return (model or '').split('@')[0]

def lead_of(model: str) -> int | None:
    return int(model.split('@')[1]) if model and '@' in model else None

def model_macro(model: str) -> str:
    return MODEL_MACRO.get(base_model(model), 'US')

def lead_sampled(day: str) -> bool:
    if not day or LEAD_SHARE <= 0:
        return False
    return int(hashlib.sha256(f'lead-sample:{day}'.encode()).hexdigest()[:8], 16) / 2 ** 32 < LEAD_SHARE

def leads_text(model: str | None=None) -> str:
    ns = sorted(set(LEADS.get(model, ())) if model else {n for v in LEADS.values() for n in v})
    return ', '.join(map(str, ns[:-1])) + (' and ' if len(ns) > 1 else '') + (str(ns[-1]) if ns else '')

def lead_models(macro: str, day: str | None) -> list[str]:
    if not day or not lead_sampled(day):
        return []
    from datetime import date, timedelta
    out, d = ([], date.fromisoformat(day))
    for m in tuple(MACRO[macro]['models']) + LEAD_EXTRA.get(macro, ()):
        if m not in LEAD_RUN_H.get(macro, {}):
            continue
        ns = ((0,) if m in LEAD_EXTRA.get(macro, ()) else ()) + LEADS.get(m, ())
        for n in ns:
            if m in RUNS_FIRST and (d - timedelta(days=n)).isoformat() < RUNS_FIRST[m]:
                continue
            out.append(f'{m}@{n}')
    return out

def macro_of(region: str | None) -> str:
    return REGION_MACRO.get(region, 'US')

def model_label(model: str) -> str:
    return MODEL_LABEL.get(model, (model or '').upper())

def icon_sampled(day: str) -> bool:
    if not day or day < ICON_FIRST or EU_ICON_SHARE <= 0:
        return False
    h = int(hashlib.sha256(f'icon-sample:{day}'.encode()).hexdigest()[:8], 16) / 2 ** 32
    return h < EU_ICON_SHARE

def models_for(regions, day: str | None=None, lead: bool=True) -> tuple:
    out = []
    for a in sorted({macro_of(r) for r in regions} or {'US'}):
        for m in MACRO[a]['models']:
            if m in out or (m in SAMPLED and (not (day and icon_sampled(day)))) or (m in FIRST_DAY and (not (day and day >= FIRST_DAY[m]))):
                continue
            out.append(m)
    if day and lead:
        for a in sorted({macro_of(r) for r in regions} or {'US'}):
            out += [m for m in lead_models(a, day) if m not in out]
    return tuple(out)

def primary_model(regions=()) -> str:
    return models_for(regions)[0]

def is_lead(model: str) -> bool:
    return '@' in (model or '')

def day_key(day: str, macro: str='US') -> str:
    return day if macro == 'US' else f'{day}_{macro}'

def split_key(key: str) -> tuple[str, str]:
    if '_' in key:
        d, m = key.split('_', 1)
        return (d, m)
    return (key, 'US')

def day_of(key: str) -> str:
    return split_key(key)[0]

def macro_of_key(key: str) -> str:
    return split_key(key)[1]

def macro_of_items(items) -> str:
    ms = {macro_of(f.get('region')) for f in items if f.get('region')}
    return sorted(ms)[0] if len(ms) == 1 else 'US' if not ms else sorted(ms, key=lambda m: m != 'US')[0]
RELIEF_FLAT = 300
TZ = {**{s: 'America/New_York' for s in 'CT DE FL GA MA MD ME NC NH NJ NY OH PA RI SC VA VT WV DC MI IN KY'.split()}, **{s: 'America/Chicago' for s in 'AL AR IA IL KS LA MN MO MS ND NE OK SD TN TX WI'.split()}, **{s: 'America/Denver' for s in 'CO MT NM UT WY ID'.split()}, 'AZ': 'America/Phoenix', **{s: 'America/Los_Angeles' for s in 'CA NV OR WA'.split()}}
EU_TZ = {**{c: 'Europe/Berlin' for c in 'DE NL BE LU AT CH LI DK SE NO PL CZ SK HU SI HR BA RS ME MK AL IT SM MT ES AD MC'.split()}, 'FR': 'Europe/Paris', 'PT': 'Europe/Lisbon', 'GB': 'Europe/London', 'IE': 'Europe/Dublin', 'IM': 'Europe/London', 'JE': 'Europe/London', 'GG': 'Europe/London', 'IS': 'Atlantic/Reykjavik', 'FI': 'Europe/Helsinki', 'LT': 'Europe/Vilnius', 'LV': 'Europe/Riga', 'EE': 'Europe/Tallinn', 'GR': 'Europe/Athens', 'BG': 'Europe/Sofia', 'RO': 'Europe/Bucharest', 'MD': 'Europe/Chisinau', 'XK': 'Europe/Belgrade'}
CENTROID = {'AL': (-86.8, 32.8), 'AZ': (-111.7, 34.3), 'AR': (-92.4, 34.9), 'CA': (-119.4, 37.2), 'CO': (-105.5, 39.0), 'CT': (-72.7, 41.6), 'DE': (-75.5, 39.0), 'FL': (-81.7, 28.6), 'GA': (-83.4, 32.7), 'ID': (-114.6, 44.4), 'IL': (-89.2, 40.0), 'IN': (-86.3, 39.9), 'IA': (-93.5, 42.1), 'KS': (-98.4, 38.5), 'KY': (-85.3, 37.5), 'LA': (-92.0, 31.1), 'ME': (-69.2, 45.4), 'MD': (-76.8, 39.0), 'MA': (-71.8, 42.3), 'MI': (-85.4, 44.3), 'MN': (-94.3, 46.3), 'MS': (-89.7, 32.7), 'MO': (-92.5, 38.4), 'MT': (-109.6, 47.0), 'NE': (-99.8, 41.5), 'NV': (-116.6, 39.3), 'NH': (-71.6, 43.7), 'NJ': (-74.7, 40.2), 'NM': (-106.1, 34.4), 'NY': (-75.5, 42.9), 'NC': (-79.4, 35.6), 'ND': (-100.5, 47.5), 'OH': (-82.8, 40.3), 'OK': (-97.5, 35.6), 'OR': (-120.6, 43.9), 'PA': (-77.8, 40.9), 'RI': (-71.5, 41.7), 'SC': (-80.9, 33.9), 'SD': (-100.2, 44.4), 'TN': (-86.4, 35.9), 'TX': (-99.3, 31.5), 'UT': (-111.7, 39.3), 'VT': (-72.7, 44.1), 'VA': (-78.8, 37.5), 'WA': (-120.5, 47.4), 'WV': (-80.6, 38.6), 'WI': (-89.9, 44.6), 'WY': (-107.6, 43.0)}
EU_CENTROID = {'DE': (10.4, 51.1), 'NL': (5.3, 52.2), 'BE': (4.6, 50.6), 'LU': (6.1, 49.8), 'AT': (14.1, 47.6), 'CH': (8.2, 46.8), 'LI': (9.5, 47.2), 'FR': (2.5, 46.6), 'MC': (7.4, 43.7), 'ES': (-3.7, 40.3), 'PT': (-8.0, 39.6), 'AD': (1.6, 42.5), 'GB': (-2.0, 53.0), 'IE': (-8.0, 53.2), 'IM': (-4.5, 54.2), 'JE': (-2.1, 49.2), 'GG': (-2.6, 49.5), 'PL': (19.4, 52.1), 'CZ': (15.5, 49.8), 'SK': (19.7, 48.7), 'HU': (19.4, 47.2), 'LT': (23.9, 55.3), 'LV': (24.6, 56.9), 'EE': (25.0, 58.7), 'SE': (16.0, 62.0), 'NO': (9.5, 61.5), 'FI': (26.0, 64.0), 'DK': (9.9, 56.0), 'IS': (-18.6, 64.9), 'IT': (12.5, 42.8), 'SM': (12.5, 43.9), 'MT': (14.4, 35.9), 'SI': (14.8, 46.1), 'HR': (16.5, 45.3), 'BA': (17.8, 44.2), 'RS': (20.9, 44.2), 'ME': (19.3, 42.7), 'MK': (21.7, 41.6), 'AL': (20.0, 41.1), 'GR': (22.0, 39.1), 'BG': (25.2, 42.7), 'RO': (25.0, 45.9), 'XK': (20.9, 42.6), 'MD': (28.5, 47.2)}

def centroid_of(code: str | None):
    return CENTROID.get(code or '') or EU_CENTROID.get(code or '')

def nearest_state(lon: float, lat: float) -> str:
    k = math.cos(math.radians(lat))
    return min(CENTROID, key=lambda s: ((CENTROID[s][0] - lon) * k) ** 2 + (CENTROID[s][1] - lat) ** 2)
_SMALL = {'LI', 'MC', 'AD', 'SM', 'MT', 'IM', 'JE', 'GG', 'XK', 'LU', 'ME'}

def nearest_country(lon: float, lat: float) -> str | None:
    if not (EU_BOX[0] <= lon <= EU_BOX[2] and EU_BOX[1] <= lat <= EU_BOX[3]):
        return None
    k = math.cos(math.radians(lat))
    return min((c for c in EU_CENTROID if c not in _SMALL), key=lambda c: ((EU_CENTROID[c][0] - lon) * k) ** 2 + (EU_CENTROID[c][1] - lat) ** 2)

def in_alps(lon: float | None, lat: float | None) -> bool:
    return lon is not None and lat is not None and any((w <= lon <= e and s <= lat <= n for w, s, e, n in ALPS_BOXES))

def eu_region(country: str | None, lon: float | None=None, lat: float | None=None) -> str | None:
    if in_alps(lon, lat):
        return 'Alps'
    return EU_REGION.get(country or '')

def load_aircraft() -> dict:
    with open(os.path.join(HERE, 'aircraft.json'), encoding='utf-8') as f:
        return json.load(f)['types']

def aircraft_version(types: dict | None=None) -> str:
    return hashlib.sha256(json.dumps(types if types is not None else load_aircraft(), sort_keys=True).encode()).hexdigest()[:12]
DAY_VERSION = 3
DAY_BUILD_SKIP = {'build.py': {'write_free', '_free_soundings', '_publish_open', 'free_ahead', 'hour_token'}}

def _stripped(path: str, skip: bool=True) -> str:
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if path.endswith('.py'):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        drop = DAY_BUILD_SKIP.get(os.path.basename(path), set()) if skip else set()
        if drop:
            tree.body = [x for x in tree.body if not (isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name in drop)]
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

def _version_of(paths: list[str], skip: bool=True) -> str:
    h = hashlib.sha256()
    found = False
    for p in sorted(set(paths)):
        if os.path.isfile(p):
            found = True
            h.update(os.path.basename(p).encode() + b'\x00' + _stripped(p, skip).encode() + b'\x00')
    return h.hexdigest()[:12] if found else 'none'
_PIPELINE_FILES = [os.path.join(PIPELINE, f'{m}.py') for m in ('build', 'physics', 'physics_native', 'sky', 'runfields', 'runfields_native', 'leelattice', 'runwave', 'shade', 'sources', 'grib', 'fronts', 'pack')] + glob.glob(os.path.join(PIPELINE, 'native', '*.cpp')) + glob.glob(os.path.join(PIPELINE, 'native', '*.h'))
PHYSICS_FILES = _PIPELINE_FILES + [os.path.join(HERE, 'build_day.py')]
PHYSICS_FILES_EU = _PIPELINE_FILES + [os.path.join(PIPELINE, 'om.py'), os.path.join(PIPELINE, 'replay.py'), os.path.join(HERE, 'build_eu.py')]
PHYSICS_VERSION = _version_of(PHYSICS_FILES)
PHYSICS_VERSION_EU = _version_of(PHYSICS_FILES_EU)
_WHOLE = {'US': _version_of(PHYSICS_FILES, skip=False), 'EU': _version_of(PHYSICS_FILES_EU, skip=False)}
PHYSICS_FILES_LEAD = PHYSICS_FILES_EU + [os.path.join(HERE, 'build_lead.py')]
LEAD_VERSION = _version_of(PHYSICS_FILES_LEAD)
_WHOLE_LEAD = _version_of(PHYSICS_FILES_LEAD, skip=False)

def model_physics(model: str) -> str:
    return globals()['LEAD_VERSION'] if is_lead(model) else physics_version(model_macro(model))

def added_models(macro: str, day: str | None) -> list[str]:
    return [m for m in ADDED if m in MACRO[macro]['models'] and day and (day >= FIRST_DAY.get(m, '')) and (m not in SAMPLED or icon_sampled(day))]

def lead_version(key: str) -> str | None:
    macro, day = (macro_of_key(key), day_of(key))
    ms, add = (lead_models(macro, day), added_models(macro, day))
    if not ms and (not add):
        return None
    return f"{globals()['LEAD_VERSION']}:{','.join(ms)}" + (f"+{','.join(add)}" if add else '')

def lead_matches(stamp: str | None) -> bool:
    return stamp is not None and stamp in (globals()['LEAD_VERSION'], _WHOLE_LEAD)

def lead_version_matches(lv: str | None, key: str) -> bool:
    want = lead_version(key)
    if want is None or lv is None:
        return lv == want
    return lv == want or lv == f"{_WHOLE_LEAD}:{want.split(':', 1)[1]}"
FINGERPRINTS: dict = {'US': {}, 'EU': {}}

def load_fingerprints(*paths: str) -> None:
    for p in paths:
        try:
            with open(p, encoding='utf-8') as f:
                t = json.load(f)
        except (OSError, ValueError):
            continue
        for m, e in (t or {}).items():
            FINGERPRINTS.setdefault(m, {}).update(e or {})
load_fingerprints(*[p for p in os.environ.get('CALIB_FINGERPRINTS', '').split(':') if p], os.path.join(HERE, 'fingerprints.json'))

def physics_matches(stamp: str | None, macro: str='US') -> bool:
    if stamp is None:
        return False
    now = (physics_version(macro), _WHOLE.get(macro))
    if stamp in now:
        return True
    fp = FINGERPRINTS.get(macro) or {}
    mine = fp.get(physics_version(macro))
    return mine is not None and fp.get(stamp) == mine

def model_physics_matches(stamp: str | None, model: str) -> bool:
    if is_lead(model):
        return stamp is not None and stamp in (globals()['LEAD_VERSION'], _WHOLE_LEAD)
    return physics_matches(stamp, model_macro(model))

def physics_version(macro: str='US') -> str:
    return globals()['PHYSICS_VERSION_EU'] if macro == 'EU' else globals()['PHYSICS_VERSION']
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
