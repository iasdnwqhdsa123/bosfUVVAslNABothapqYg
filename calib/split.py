from __future__ import annotations
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from statistics import NormalDist
from typing import Iterator
import numpy as np
K_TIME = 5
K_SPACE = 5
BLOCK_WEEKS = 4
BUFFER_DAYS = 3
BLOCK_DEG = 2.0
BUFFER_DEG = 0.5
SEED = 7
GAIN = 0.03
LEVEL = 0.95
RESAMPLES = 1000
MARGIN = {'region': 0.02, 'time of day': 0.0}
MIN_ROWS = 500
MIN_BLOCKS = 8
_M64 = np.uint64(18446744073709551615)

def _mix(*parts) -> np.ndarray:
    with np.errstate(over='ignore'):
        h = np.uint64(11400714819323198485)
        for p in parts:
            x = np.asarray(p, dtype=np.int64).astype(np.uint64) ^ h
            x = x + np.uint64(11400714819323198485) & _M64
            x = (x ^ x >> np.uint64(30)) * np.uint64(13787848793156543929) & _M64
            x = (x ^ x >> np.uint64(27)) * np.uint64(10723151780598845931) & _M64
            h = x ^ x >> np.uint64(31)
    return h
_EPOCH = date(1970, 1, 1)

def _utc_day(x) -> int:
    if isinstance(x, datetime):
        dt = x
    elif isinstance(x, date):
        return (x - _EPOCH).days
    else:
        t = str(x).strip()
        if len(t) <= 10:
            return (date.fromisoformat(t) - _EPOCH).days
        dt = datetime.fromisoformat(t[:-1] + '+00:00' if t.endswith(('Z', 'z')) else t)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return (dt.date() - _EPOCH).days

def day_number(dates) -> np.ndarray:
    a = np.asarray(dates)
    if a.dtype.kind in 'iu':
        return a.astype(np.int64)
    if a.dtype.kind == 'f':
        if not (np.isfinite(a).all() and (a == np.round(a)).all()):
            raise ValueError('day numbers must be whole days')
        return a.astype(np.int64)
    if a.dtype.kind == 'M':
        return a.astype('datetime64[D]').astype(np.int64)
    flat = a.ravel()
    if a.dtype.kind in 'US':
        u, inv = np.unique(flat, return_inverse=True)
        return np.array([_utc_day(x) for x in u], dtype=np.int64)[inv.reshape(-1)].reshape(a.shape)
    return np.array([_utc_day(x) for x in flat], dtype=np.int64).reshape(a.shape)

def week_of(days) -> np.ndarray:
    return (np.asarray(days, dtype=np.int64) + 3) // 7

def _time_fold_of_days(days: np.ndarray, k: int, block_weeks: int, seed: int) -> np.ndarray:
    order = np.argsort(_mix(seed, np.arange(k)), kind='stable').astype(np.int64)
    return order[week_of(days) // block_weeks % k]

def time_fold(days, k: int=K_TIME, block_weeks: int=BLOCK_WEEKS, seed: int=SEED) -> np.ndarray:
    d = day_number(days)
    u, inv = np.unique(d, return_inverse=True)
    return _time_fold_of_days(u, k, block_weeks, seed)[inv.reshape(d.shape)]

def _lon(lon) -> np.ndarray:
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0

def space_block(lat, lon, block_deg: float=BLOCK_DEG) -> tuple[np.ndarray, np.ndarray]:
    return (np.floor(np.asarray(lat, dtype=float) / block_deg).astype(np.int64), np.floor(_lon(lon) / block_deg).astype(np.int64))

def _space_fold_of_blocks(i, j, k: int, seed: int) -> np.ndarray:
    return (_mix(seed, i, j) % np.uint64(k)).astype(np.int64)

def space_fold(lat, lon, k: int=K_SPACE, block_deg: float=BLOCK_DEG, seed: int=SEED) -> np.ndarray:
    i, j = space_block(lat, lon, block_deg)
    return _space_fold_of_blocks(i, j, k, seed)

def _near_time(u: np.ndarray, f: int, k: int, block_weeks: int, buffer_days: int, seed: int) -> np.ndarray:
    near = np.zeros(len(u), dtype=bool)
    for o in range(-buffer_days, buffer_days + 1):
        near |= _time_fold_of_days(u + o, k, block_weeks, seed) == f
    return near

def _near_space(lat: np.ndarray, lon: np.ndarray, f: int, k: int, block_deg: float, buffer_deg: float, seed: int) -> np.ndarray:
    near = np.zeros(len(lat), dtype=bool)
    for dy in (-buffer_deg, 0.0, buffer_deg):
        for dx in (-buffer_deg, 0.0, buffer_deg):
            i, j = space_block(lat + dy, lon + dx, block_deg)
            near |= _space_fold_of_blocks(i, j, k, seed) == f
    return near

@dataclass(frozen=True)
class Fold:
    t: int | None
    s: int | None
    train: np.ndarray
    test: np.ndarray

def folds(days=None, lat=None, lon=None, *, k_time: int=K_TIME, k_space: int=K_SPACE, block_weeks: int=BLOCK_WEEKS, buffer_days: int=BUFFER_DAYS, block_deg: float=BLOCK_DEG, buffer_deg: float=BUFFER_DEG, seed: int=SEED, pairs=None) -> Iterator[Fold]:
    if days is None and lat is None:
        raise ValueError("folds needs the days, the stations' positions, or both")
    n = len(days) if days is not None else len(lat)
    if (lat is None) != (lon is None):
        raise ValueError('lat and lon go together')
    if buffer_deg >= block_deg:
        raise ValueError('buffer_deg must be below block_deg')
    tf = ud = tinv = None
    if days is not None:
        d = day_number(days)
        ud, tinv = np.unique(d, return_inverse=True)
        tf = _time_fold_of_days(ud, k_time, block_weeks, seed)[tinv]
    sf = pos = sinv = None
    if lat is not None:
        p = np.column_stack([np.asarray(lat, dtype=float), _lon(lon)])
        if len(p) != n:
            raise ValueError('days, lat and lon differ in length')
        if not np.isfinite(p).all():
            raise ValueError("a row without its station's position")
        pos, sinv = np.unique(p, axis=0, return_inverse=True)
        sinv = sinv.reshape(-1)
        i, j = space_block(pos[:, 0], pos[:, 1], block_deg)
        sf = _space_fold_of_blocks(i, j, k_space, seed)[sinv]
    ts = range(k_time) if tf is not None else [None]
    ss = range(k_space) if sf is not None else [None]
    if pairs is None:
        pairs = [(t, s) for t in ts for s in ss]
    else:
        pairs = [tuple(p) for p in pairs]
        for t, s in pairs:
            for f, have, k, side in ((t, tf is not None, k_time, 'time'), (s, sf is not None, k_space, 'station')):
                if f is None and have:
                    raise ValueError(f'a pair without its {side} fold, though the {side} side was given')
                if f is not None and (not have or not isinstance(f, (int, np.integer)) or (not 0 <= f < k)):
                    raise ValueError(f"a pair's {side} fold is not one of 0..{k - 1} of a side that was given")
    near_t, near_s = ({}, {})
    for t, s in pairs:
        test = np.ones(n, dtype=bool)
        train = np.ones(n, dtype=bool)
        if t is not None:
            if t not in near_t:
                near_t[t] = _near_time(ud, t, k_time, block_weeks, buffer_days, seed)[tinv]
            test &= tf == t
            train &= ~near_t[t]
        if s is not None:
            if s not in near_s:
                near_s[s] = _near_space(pos[:, 0], pos[:, 1], s, k_space, block_deg, buffer_deg, seed)[sinv]
            test &= sf == s
            train &= ~near_s[s]
        yield Fold(t, s, np.flatnonzero(train), np.flatnonzero(test))

def key_fold(keys, k: int=K_TIME, seed: int=SEED) -> np.ndarray:
    a = np.asarray(keys).astype(str)
    u, inv = np.unique(a, return_inverse=True)
    f = np.array([int.from_bytes(hashlib.blake2b(f'{seed}|{x}'.encode(), digest_size=8).digest(), 'big') % k for x in u], dtype=np.int64)
    return f[inv.reshape(a.shape)]

def drop_shared(train: np.ndarray, test: np.ndarray, groups) -> np.ndarray:
    g = np.asarray(groups)
    return train[~np.isin(g[train], np.unique(g[test]))]

def blocks_of(*labels) -> np.ndarray:
    if not labels:
        raise ValueError('blocks_of needs a label')
    code = np.zeros(len(labels[0]), dtype=np.int64)
    for lab in labels:
        _, inv = np.unique(np.asarray(lab), return_inverse=True)
        inv = inv.reshape(-1)
        code = code * (int(inv.max()) + 1 if len(inv) else 1) + inv
    _, out = np.unique(code, return_inverse=True)
    return out.reshape(-1)

def _paired(new, base, blocks, w):
    a = np.asarray(new, dtype=float)
    b = np.asarray(base, dtype=float)
    g = np.asarray(blocks)
    if not len(a) == len(b) == len(g):
        raise ValueError('new, base and blocks differ in length')
    ww = np.ones(len(a)) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(ww)
    return (a[ok], b[ok], g[ok], ww[ok], int((~ok).sum()))

def t_quantile(p: float, df: float) -> float:
    z = NormalDist().inv_cdf(p)
    v = float(df)
    return float(z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v ** 2) + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3) + (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / (92160 * v ** 4))

def _small_sample_widen(nb: int, level: float) -> float:
    if nb < 3:
        return float('inf')
    q = 1 - (1 - level) / 2
    return float(np.sqrt(nb / (nb - 1)) * t_quantile(q, nb - 1) / NormalDist().inv_cdf(q))
CHUNK = 1 << 21

def paired_bootstrap(new, base, blocks, *, w=None, relative: bool=True, n: int=RESAMPLES, level: float=LEVEL, seed: int=SEED) -> dict:
    a, b, g, ww, dropped = _paired(new, base, blocks, w)
    out = {'diff': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'rows': int(len(a)), 'blocks': 0, 'dropped': dropped}
    if not len(a):
        return out
    _, inv = np.unique(g, return_inverse=True)
    inv = inv.reshape(-1)
    nb = int(inv.max()) + 1
    sa = np.bincount(inv, ww * a, nb)
    sb = np.bincount(inv, ww * b, nb)
    sw = np.bincount(inv, ww, nb)
    out['blocks'] = nb
    if relative and (not sb.sum() > 0) or not sw.sum() > 0:
        return out

    def stat(A, B, W):
        with np.errstate(divide='ignore', invalid='ignore'):
            return A / B - 1.0 if relative else (A - B) / W
    d = float(stat(sa.sum(), sb.sum(), sw.sum()))
    out['diff'] = d
    rng = np.random.default_rng(seed)
    step = max(1, CHUNK // nb)
    rs = []
    for s0 in range(0, n, step):
        m = min(step, n - s0)
        draw = rng.integers(0, nb, size=(m, nb)) + (np.arange(m) * nb)[:, None]
        c = np.bincount(draw.ravel(), minlength=m * nb).reshape(m, nb).astype(float)
        rs.append(stat(c @ sa, c @ sb, c @ sw))
    r = np.concatenate(rs) if rs else np.empty(0)
    r = r[np.isfinite(r)]
    if len(r):
        q = (1 - level) / 2
        lo, hi = (float(x) for x in np.quantile(r, [q, 1 - q]))
        k = _small_sample_widen(nb, level)
        if np.isfinite(k):
            out['lo'], out['hi'] = (d - (d - lo) * k, d + (hi - d) * k)
    return out
REQUIRED_CELLS = ('region', 'time of day')
COVER_TOL = 0.0

def ship_verdict(new, base, blocks, cells: dict | None=None, *, w=None, gain: float=GAIN, level: float=LEVEL, n: int=RESAMPLES, seed: int=SEED, margin: dict | None=None, min_rows: int=MIN_ROWS, min_blocks: int=MIN_BLOCKS, required=REQUIRED_CELLS, cover_tol: float=COVER_TOL, units=None, min_units: int=0) -> dict:
    margin = {**MARGIN, **(margin or {})}
    fa = np.isfinite(np.asarray(new, dtype=float))
    fb = np.isfinite(np.asarray(base, dtype=float))
    if len(fa) != len(fb):
        raise ValueError('new, base and blocks differ in length')
    keep = fa & fb
    if w is not None:
        keep &= np.isfinite(np.asarray(w, dtype=float))
    only_base, only_new = (int((fb & ~fa).sum()), int((fa & ~fb).sum()))
    both = int(keep.sum())
    tol = cover_tol * max(both + only_base + only_new, 1)
    cond_cov = {'ok': bool(only_base <= tol and only_new <= tol), 'new_missing': only_base, 'base_missing': only_new, 'rows': both, 'tol': cover_tol}
    if not cond_cov['ok']:
        cond_cov['why'] = "rows with one forecast's loss and not the other's"
    overall = paired_bootstrap(new, base, blocks, w=w, n=n, level=level, seed=seed)
    un = np.asarray(units, dtype=object) if units is not None else None
    if un is not None and len(un) != len(keep):
        raise ValueError('the units differ in length from the rows')
    n_units = lambda sel: len(set(un[keep][sel].tolist())) if un is not None else None
    if un is not None:
        overall['units'] = n_units(np.ones(int(keep.sum()), bool))
    enough = overall['rows'] >= min_rows and overall['blocks'] >= min_blocks and (un is None or overall['units'] >= min_units)
    d, hi = (overall['diff'], overall['hi'])
    cond = {'gain': {'ok': bool(np.isfinite(d) and d <= -gain), 'diff': d, 'need': -gain}, 'interval': {'ok': bool(enough and np.isfinite(hi) and (hi < 0)), 'lo': overall['lo'], 'hi': hi, 'level': level, **({} if enough else {'why': 'too few rows or blocks'})}}
    a, b, g, ww, _ = _paired(new, base, blocks, w)
    cells = dict(cells or {})
    groups, worse, why = ({}, [], [])
    for name in required:
        if name not in cells:
            why.append(f'no {name} grouping given')
    for name, lab in cells.items():
        lab = np.asarray(lab)
        if len(lab) != len(keep):
            raise ValueError(f'the cells of {name!r} differ in length from the rows')
        lab = lab[keep]
        m = margin.get(name, 0.02)
        rows = []
        for c in np.unique(lab):
            sel = lab == c
            label = c.item() if hasattr(c, 'item') else c
            cs = int.from_bytes(hashlib.blake2b(f'{seed}|{name}|{label!r}'.encode(), digest_size=4).digest(), 'big')
            r = paired_bootstrap(a[sel], b[sel], g[sel], w=ww[sel], n=n, level=level, seed=cs)
            nu = n_units(sel)
            thin = r['rows'] < min_rows or r['blocks'] < min_blocks or (nu is not None and nu < min_units)
            bad = bool(not thin and np.isfinite(r['lo']) and (r['lo'] > m))
            rows.append({'cell': label, 'diff': r['diff'], 'lo': r['lo'], 'hi': r['hi'], 'rows': r['rows'], 'blocks': r['blocks'], **({'units': nu} if nu is not None else {}), 'thin': thin, 'worse': bad})
            if bad:
                worse.append((name, label))
        groups[name] = {'margin': m, 'cells': rows}
        if name in required and (not any((not x['thin'] for x in rows))):
            why.append(f'no {name} cell with enough rows to test')
    cond['cells'] = {'ok': not worse and (not why), 'worse': worse, 'groups': groups, **({'why': why} if why else {})}
    cond['coverage'] = cond_cov
    return {'ship': all((c['ok'] for c in cond.values())), 'overall': overall, 'conditions': cond}
LAUNCH_WON = 0.9
LAUNCH_WEEKS = 8
LAUNCH_WEEKS_WON = 7
LAUNCH_LOST = 0.02
SOARABLE_W = 0.2

def launch_verdict(comparisons: list, *, level: float=LEVEL, n: int=RESAMPLES, seed: int=SEED) -> dict:
    rows, counted = ([], 0)
    for c in comparisons:
        name = c.get('name')
        if not c.get('changed', True):
            rows.append({'name': name, 'counted': False, 'incomplete': True, 'why': "the blend takes it from its base: not the blend's yet, so no launch"})
            continue
        counted += 1
        new, base = (np.asarray(c['new'], float), np.asarray(c['base'], float))
        w = None
        if c.get('soarable') is not None:
            w = np.where(np.asarray(c['soarable'], bool), 1.0, SOARABLE_W)
        if c.get('day') is not None:
            dy = np.asarray(c['day'])
            ok = np.isfinite(new) & np.isfinite(base)
            _, inv, cnt = np.unique(dy, return_inverse=True, return_counts=True)
            per_day = np.bincount(inv.reshape(-1), weights=ok.astype(float), minlength=len(cnt))
            wd = 1.0 / np.maximum(per_day[inv.reshape(-1)], 1.0)
            w = wd if w is None else w * wd
        cs = int.from_bytes(hashlib.blake2b(f'{seed}|launch|{name!r}'.encode(), digest_size=4).digest(), 'big')
        pooled = paired_bootstrap(new, base, c['blocks'], w=w, n=n, level=level, seed=cs)
        wk = np.asarray(c['week'])
        weeks = sorted(set(wk[np.isfinite(new) & np.isfinite(base)].tolist()))[-LAUNCH_WEEKS:]
        per = []
        for k in weeks:
            sel = (wk == k) & np.isfinite(new) & np.isfinite(base)
            ww = w[sel] if w is not None else np.ones(int(sel.sum()))
            per.append(bool(np.sum(ww * new[sel]) < np.sum(ww * base[sel])))
        won = bool(np.isfinite(pooled['hi']) and pooled['hi'] < 0)
        lost = bool(np.isfinite(pooled['lo']) and pooled['lo'] > LAUNCH_LOST)
        consistent = len(per) >= LAUNCH_WEEKS and sum(per) >= LAUNCH_WEEKS_WON
        rows.append({'name': name, 'counted': True, 'diff': pooled['diff'], 'lo': pooled['lo'], 'hi': pooled['hi'], 'won': won, 'lost': lost, 'weeks': len(per), 'weeks_won': int(sum(per)), 'consistent': consistent, **({} if len(per) >= LAUNCH_WEEKS else {'why': f'{len(per)} weeks of rows, {LAUNCH_WEEKS} asked'})})
    good = sum((1 for r in rows if r.get('counted') and r['won'] and r['consistent']))
    lost = [r['name'] for r in rows if r.get('counted') and r['lost']]
    share = good / counted if counted else 0.0
    incomplete = [r['name'] for r in rows if r.get('incomplete')]
    return {'launch': bool(counted and share >= LAUNCH_WON and (not lost) and (not incomplete)), 'counted': counted, 'won': good, 'share': share, 'lost': lost, 'incomplete': incomplete, 'chance_lost': counted * (1 - level) / 2, 'rows': rows}
MIN_COVER = 0.95

def pick_base(losses: dict, cells, fit, *, min_cover: float=MIN_COVER) -> tuple[np.ndarray, dict]:
    names = list(losses)
    L = np.column_stack([np.asarray(losses[k], dtype=float) for k in names])
    lab = np.asarray(cells)
    nrow = len(lab)
    fit = np.asarray(fit)
    if fit.dtype == bool:
        if len(fit) != nrow:
            raise ValueError('the fit mask differs in length from the rows')
    elif fit.dtype.kind in 'iu' or fit.size == 0:
        fi = fit.astype(np.int64).ravel()
        if len(fi) == nrow and nrow > 0 and np.isin(fi, [0, 1]).all():
            raise ValueError("fit looks like a 0/1 mask: pass a boolean mask, or the rows' indices")
        if len(fi) and (fi.min() < 0 or fi.max() >= nrow):
            raise ValueError('a fit index outside the rows')
        fit = np.isin(np.arange(nrow), fi)
    else:
        raise ValueError('fit is a boolean mask or integer row indices')

    def ranking(sel, relative=False):
        have = np.isfinite(L[sel])
        if not len(have):
            return None
        cover = have.mean(axis=0)
        cand = np.flatnonzero(cover >= min_cover * (cover.max() if relative else 1.0))
        if not len(cand) or not cover[cand].all():
            return None
        common = have[:, cand].all(axis=1)
        if not common.any():
            return None
        m = L[sel][common][:, cand].mean(axis=0)
        return [int(cand[i]) for i in np.argsort(m, kind='stable')]
    every = ranking(fit) or ranking(fit, relative=True)
    choice, out = ({}, np.full(nrow, np.nan))
    for c in np.unique(lab):
        sel = lab == c
        order = ranking(sel & fit) or every
        if order is None:
            continue
        choice[c.item() if hasattr(c, 'item') else c] = names[order[0]]
        idx = np.flatnonzero(sel)
        rest = [j for j in range(len(names)) if j not in order]
        with np.errstate(invalid='ignore'):
            rm = [np.nanmean(L[sel & fit, j]) if np.isfinite(L[sel & fit, j]).any() else np.inf for j in rest]
        order = order + [rest[i] for i in np.argsort(rm, kind='stable')]
        for j in reversed(order):
            v = L[idx, j]
            ok = np.isfinite(v)
            out[idx[ok]] = v[ok]
    return (out, choice)
