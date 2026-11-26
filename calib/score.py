from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lake as lakem
import pairs as P
UTC = timezone.utc
SV = 's1'
PREFIX = 'derived/score'
OPS_ROOT = 'ops'
REGIMES = ('cumulus seen', 'no cumulus', 'night')
GAP_MEASURES = ('t2', 'td2', 'wind')
MISSING_AFTER_S = 86400
_KEY = re.compile('derived/score/(?P<sv>[a-z0-9]+)/(?P<m>\\d{4}-\\d{2})/(?P<ts>\\d{8}T\\d{6}Z)\\.(?P<x>json|md)\\.lenc')
_DONE = re.compile('ops/(?P<m>[a-z0-9]+)/(?P<d>\\d{8})/(?P<h>\\d{2})/b[A-Za-z0-9._-]+/done\\.json')

def _verify():
    return P._verify()

def _n(counts: dict, k: str, by: int=1) -> None:
    counts[k] = counts.get(k, 0) + by

def month_days(month: str) -> list[str]:
    y, m = (int(month[:4]), int(month[5:7]))
    d, out = (date(y, m, 1), [])
    while d.month == m:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out

def due_month(today: date) -> str:
    first = today.replace(day=1)
    prev = (first - timedelta(days=1)).strftime('%Y-%m')
    if P.month_closed('metar', prev, today):
        return prev
    return ((first - timedelta(days=1)).replace(day=1) - timedelta(days=1)).strftime('%Y-%m')

def report_prefix(month: str, sv: str=SV) -> str:
    if not re.fullmatch('\\d{4}-\\d{2}', month or ''):
        raise ValueError('score: a month is YYYY-MM')
    return f'{PREFIX}/{sv}/{month}/'

def reports(lk, month: str | None=None, sv: str=SV) -> dict:
    pre = report_prefix(month, sv) if month else f'{PREFIX}/{sv}/'
    best = {}
    for k in lk.list(pre):
        g = _KEY.fullmatch(k)
        if not g or g['sv'] != sv:
            continue
        best.setdefault(g['m'], {}).setdefault(g['ts'], {})[g['x']] = k
    out = {}
    for m, by_ts in best.items():
        whole = [ts for ts, xs in by_ts.items() if 'json' in xs and 'md' in xs]
        if whole:
            ts = max(whole)
            out[m] = (by_ts[ts]['json'], by_ts[ts]['md'])
    return out

def _build_num(b) -> tuple:
    return tuple((int(x) if x.isdigit() else 0 for x in re.split('[^0-9]+', str(b)) if x != '')) or (0,)

def newest_builds(t: dict) -> np.ndarray:
    n = P.n_rows(t)
    if not n:
        return np.zeros(0, bool)
    groups = {}
    for i in range(n):
        k = (t['obs_id'][i], int(t['t_valid'][i]), t['model'][i], t['run'][i], t['src'][i])
        b = _build_num(t['build'][i])
        j = groups.get(k)
        if j is None or b > j[0]:
            groups[k] = (b, i)
    keep = np.zeros(n, bool)
    keep[[i for _, i in groups.values()]] = True
    return keep

def verify_view(t: dict) -> np.ndarray:
    V = _verify()
    n = P.n_rows(t)
    if not n:
        return np.zeros(0, bool)
    hour = np.asarray(t['t_valid'], np.int64) // 3600 % 24
    run_h = np.array([int(str(r)[8:10]) if len(str(r)) >= 10 else -1 for r in t['run']])
    lead = np.asarray(t['lead_h'], np.int64)
    lmax = np.array([V.LEAD_MAX.get(str(m), 0) for m in t['model']])
    return np.isin(hour, V.VALID_H) & np.isin(run_h, V.RUN_HOURS) & (lead >= 1) & (lead <= lmax)

def station_index(obs_t: dict, stations: dict) -> np.ndarray:
    out = np.empty(P.n_rows(obs_t), np.int64)
    for i, sid in enumerate(obs_t['obs_id']):
        s = stations.get(sid)
        if s is None:
            s = stations[sid] = [len(stations), float(obs_t['lat'][i]), float(obs_t['lon'][i]), float(obs_t['elev'][i])]
        out[i] = s[0]
    return out

def stations_lat_lon(stations: dict) -> tuple:
    lat, lon = (np.zeros(len(stations)), np.zeros(len(stations)))
    for i, la, lo, _ in stations.values():
        lat[i], lon[i] = (la, lo)
    return (lat, lon)

def region_codes(latlon: tuple, st: np.ndarray) -> np.ndarray:
    V = _verify()
    u, inv = np.unique(st, return_inverse=True)
    lat, lon = latlon
    per = np.array([V.REGIONS.index(V.region_of(float(lat[i]), float(lon[i]))) for i in u], np.int8)
    return per[inv] if len(u) else np.zeros(0, np.int8)

def lead_bins(lead: np.ndarray) -> np.ndarray:
    V = _verify()
    out = np.full(len(lead), -1, np.int8)
    for i, (lo, hi) in enumerate(V.LEADS):
        out[(lead >= lo) & (lead <= hi)] = i
    return out

def regime_of(cu_obs: np.ndarray, day: np.ndarray) -> np.ndarray:
    r = np.full(len(day), -1, np.int8)
    r[day & (cu_obs == 1)] = 0
    r[day & (cu_obs == 0)] = 1
    r[~day] = 2
    return r

def tables_of(pairs_t: dict, obs_t: dict, stations: dict, src: str='ops') -> dict:
    V = _verify()
    if not P.n_rows(pairs_t) or not P.n_rows(obs_t):
        return {}
    sel = (np.asarray(pairs_t['src']) == src) & verify_view(pairs_t)
    t = {k: np.asarray(v)[sel] for k, v in pairs_t.items()}
    keep = newest_builds(t)
    t = {k: v[keep] for k, v in t.items()}
    if not P.n_rows(t):
        return {}
    st_of_obs = station_index(obs_t, stations)
    at = {(sid, int(tv)): i for i, (sid, tv) in enumerate(zip(obs_t['obs_id'], obs_t['t_valid']))}
    oi = np.array([at.get((sid, int(tv)), -1) for sid, tv in zip(t['obs_id'], t['t_valid'])], np.int64)
    ok = oi >= 0
    t = {k: v[ok] for k, v in t.items()}
    oi = oi[ok]
    out = {}
    for m in sorted(set(t['model'])):
        s = t['model'] == m
        idx = oi[s]
        n = len(idx)
        lat, lon = (obs_t['lat'][idx].astype(np.float64), obs_t['lon'][idx].astype(np.float64))
        elev = obs_t['elev'][idx].astype(np.float64)
        tv = t['t_valid'][s].astype(np.int64)
        vh = tv // 3600
        hour = vh % 24
        st = st_of_obs[idx]
        reg = region_codes(stations_lat_lon(stations), st)
        day = np.zeros(n, bool)
        for ri, r in enumerate(V.REGIONS):
            day |= (reg == ri) & np.isin(hour, V.DAY_H.get(r, ()))
        fc = {v: t[f'f_{v}'][s].astype(np.float64) for v in P.METAR_VARS if f'f_{v}' in t}
        o = {n_: (obs_t[f'o_{n_}'] if f'o_{n_}' in obs_t else obs_t[f'q_{n_}'])[idx].astype(np.float64) for n_ in V.OBS}
        cont, cat = V.measures(fc, o, elev, day)
        lead = t['lead_h'][s].astype(np.int64)
        tab = {'key': st * 10 ** 12 + vh * 1000 + lead, 'st': st.astype(np.int32), 'reg': reg, 'lead': lead_bins(lead), 'h': lead.astype(np.int16), 'sol': ((hour + lon / 15) % 24).astype(np.float32), 'rg': regime_of(o['cu'], day), 'run': t['run'][s].astype(object)}
        for nm in V.CONT:
            e, b = cont.get(nm, (None, None))
            tab[nm] = np.full(n, np.nan, np.float32) if e is None else e.astype(np.float32)
            tab[nm + '_b'] = np.full(n, np.nan, np.float32) if b is None else b.astype(np.float32)
        for nm in V.CAT:
            fy, oy, okc = cat.get(nm, (None, None, None))
            tab[nm] = np.full(n, -1, np.int8) if fy is None else np.where(okc, 2 * np.asarray(fy, int) + np.asarray(oy, int), -1).astype(np.int8)
        out[m] = tab
    return out

def merge(into: dict, more: dict) -> dict:
    for m, t in more.items():
        if m in into:
            into[m] = {k: np.concatenate([into[m][k], t[k]]) for k in t}
        else:
            into[m] = t
    return into

def station_list(stations: dict) -> list:
    out = [None] * len(stations)
    for sid, (i, la, lo, el) in stations.items():
        out[i] = [sid, la, lo, el]
    return out

def _subset(tables: dict, pick) -> dict:
    return {m: {k: v[pick(t)] for k, v in t.items()} for m, t in tables.items()}
VERIFY_COLS = ('key', 'st', 'reg', 'lead', 'h', 'sol')

def _verify_tables(tables: dict) -> dict:
    V = _verify()
    cols = VERIFY_COLS + tuple((x for n in V.CONT for x in (n, n + '_b'))) + tuple(V.CAT)
    return {m: {k: t[k] for k in cols} for m, t in tables.items()}

def gap(ops_t: dict, re_t: dict) -> list:
    V = _verify()
    out = []
    for m in sorted(set(ops_t) & set(re_t)):
        a, b = (ops_t[m], re_t[m])
        ka = np.array([f'{k}|{r}' for k, r in zip(a['key'], a['run'])], dtype=object)
        kb = np.array([f'{k}|{r}' for k, r in zip(b['key'], b['run'])], dtype=object)
        common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        if not len(common):
            continue
        for ri, r in enumerate(V.REGIONS):
            row = {'model': m, 'region': r}
            for nm in GAP_MEASURES:
                ea, eb = (a[nm + '_b'][ia], b[nm + '_b'][ib])
                ok = (a['reg'][ia] == ri) & np.isfinite(ea) & np.isfinite(eb)
                if ok.sum() >= V.MIN_N['cont']:
                    row[nm] = {'n': int(ok.sum()), 'mae_ops': round(float(np.abs(ea[ok]).mean()), 2), 'mae_rebuilt': round(float(np.abs(eb[ok]).mean()), 2), 'bias_ops': round(float(a[nm][ia][ok].mean()), 2), 'bias_rebuilt': round(float(b[nm][ib][ok].mean()), 2)}
            if len(row) > 2:
                out.append(row)
    return out

def missing_runs(ops, month: str, now: datetime, counts: dict | None=None) -> list:
    if ops is None:
        return None
    counts = counts if counts is not None else {}
    have = {}
    for d in month_days(month):
        ymd = d.replace('-', '')
        for m in P.MODELS:
            for k in ops.list(f'{OPS_ROOT}/{m}/{ymd}/'):
                g = _DONE.fullmatch(k)
                if g:
                    have.setdefault(m, set()).add(g['d'] + g['h'])
    out = []
    for m in sorted(have):
        hours = [h for h in range(24) if P.slot_of(m, h) is not None]
        for d in month_days(month):
            for h in hours:
                run = d.replace('-', '') + f'{h:02d}'
                land = datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=UTC) + P.avail_after(m, h)
                if run not in have[m] and (now - land).total_seconds() >= MISSING_AFTER_S:
                    out.append({'model': m, 'run': run})
    _n(counts, 'runs_missing', len(out))
    return out

def kind_counts(lk, month: str) -> dict:
    out = {}
    days = set(month_days(month))
    for kind in P.KINDS:
        n = 0
        comp = P.compactions(lk, kind).get(month)
        covered = set()
        if comp:
            body = comp[2]
            covered.update(body.get('markers') or [])
            n += int(body.get('rows') or 0)
        for d in sorted(days):
            for _, _, _, mk in P.markers(lk, 'pairs', kind, day=d):
                if mk not in covered:
                    n += int(json.loads(lk.get(mk)).get('rows') or 0)
        if n:
            out[kind] = n
    return out

def score_month(lk, ops, month: str, now: datetime, counts: dict | None=None, read=None) -> dict:
    V = _verify()
    counts = counts if counts is not None else {}
    cols_p = list(P.KEY) + ['lead_h', 'model'] + [f'f_{v}' for v in P.METAR_VARS]
    rd = read or (lambda what, kind, d: P.read_kind(lk, what, kind, [d], purpose='review', columns=cols_p if what == 'pairs' else None))
    stations, ops_t, re_t = ({}, {}, {})
    for d in month_days(month):
        pt, ot = (rd('pairs', 'metar', d), rd('obs', 'metar', d))
        if not P.n_rows(pt) or not P.n_rows(ot):
            _n(counts, 'days_empty')
            continue
        _n(counts, 'days')
        _n(counts, 'pairs_read', P.n_rows(pt))
        merge(ops_t, tables_of(pt, ot, stations, 'ops'))
        merge(re_t, tables_of(pt, ot, stations, 'stcast'))
    sts = station_list(stations)
    main = V.summarise(_verify_tables(ops_t), sts) if ops_t else {'pairs': 0, 'stations_left_out': [], 'scores': [], 'side_by_side': [], 'by_sun': []}
    bad_ids = set(main['stations_left_out'])
    bad = {stations[s][0] for s in bad_ids if s in stations}
    regimes = []
    for k, name in enumerate(REGIMES):
        sub = _verify_tables(_subset(ops_t, lambda t, k=k: t['rg'] == k))
        if any((len(t['key']) for t in sub.values())):
            sc = V.summarise(sub, sts, bad)
            regimes.append({'regime': name, 'pairs': sc['pairs'], 'side_by_side': sc['side_by_side']})
    keep = lambda t: ~np.isin(t['st'], list(bad))
    out = {'schema': f'score/{SV}', 'month': month, 'generated': f'{now:%Y-%m-%dT%H:%MZ}', **main, 'regimes': regimes, 'gap': gap(_subset(ops_t, keep), _subset(re_t, keep)) if re_t else [], 'missing_runs': missing_runs(ops, month, now, counts), 'counts': kind_counts(lk, month) if lk is not None else {}, 'window': window_scores((rd := (read or (lambda what, kind, d: P.read_kind(lk, what, kind, [d], purpose='review')))), month), 'window_call': window_scores(rd, month, by_call=True), 'scoreboard': _scoreboard(lk)}
    _n(counts, 'pairs_scored', main['pairs'])
    return out
CALL_BINS = ((0.0, 1.3, 'weak'), (1.3, 2.0, 'moderate'), (2.0, float('inf'), 'strong'))

def call_of(wsmax: float) -> str | None:
    if not math.isfinite(wsmax):
        return None
    return next((name for lo, hi, name in CALL_BINS if lo <= wsmax < hi))

def _scoreboard(lk) -> dict | None:
    if lk is None:
        return None
    import shadowscore
    return shadowscore.latest_scoreboard(lk)

def window_scores(read, month: str, by_call: bool=False) -> list:
    by = {}
    for d in month_days(month):
        pt, ot = (read('pairs', 'window', d), read('obs', 'window', d))
        if not P.n_rows(pt) or not P.n_rows(ot) or 'f_first' not in pt or ('o_first' not in ot):
            continue
        o = {(str(a), int(b)): i for i, (a, b) in enumerate(zip(ot['obs_id'], ot['t_valid']))}
        for j, (a, b) in enumerate(zip(pt['obs_id'], pt['t_valid'])):
            i = o.get((str(a), int(b)))
            if i is None:
                continue
            lead = '0-24 h' if int(pt['lead_h'][j]) < 24 else '24-48 h' if int(pt['lead_h'][j]) < 48 else None
            if lead is None:
                continue
            call = (call_of(float(pt['f_wsmax'][j])) if 'f_wsmax' in pt else None) or 'unknown'
            s = by.setdefault((str(pt['model'][j]), lead) + ((call,) if by_call else ()), {'on': [], 'off': [], 'missed': 0, 'false': 0, 'n': 0})
            s['n'] += 1
            ff, fl = (float(pt['f_first'][j]), float(pt['f_last'][j]))
            of, ol, on = (float(ot['o_first'][i]), float(ot['o_last'][i]), float(ot['o_n'][i]))
            if math.isfinite(ff) and math.isfinite(of) and (float(ot['q_lead'][i]) >= P.WINDOW_QLEAD_H):
                s['on'].append(ff - of)
            if math.isfinite(fl) and math.isfinite(ol) and (float(ot['q_tail'][i]) >= P.WINDOW_QLEAD_H):
                s['off'].append(fl - ol)
            if on >= 1 and (not math.isfinite(ff)):
                s['missed'] += 1
            if on == 0 and math.isfinite(ff):
                s['false'] += 1
    out = []
    for key, s in sorted(by.items()):
        m, lead = key[:2]
        row = {'model': m, 'lead': lead, 'n': s['n'], 'missed': s['missed'], 'false': s['false'], **({'call': key[2]} if by_call else {})}
        for k in ('on', 'off'):
            e = np.array(s[k], np.float64)
            row[k] = {'n': int(len(e)), 'mae': float(np.mean(np.abs(e))) if len(e) else None, 'bias': float(np.mean(e)) if len(e) else None}
        out.append(row)
    return out

def _fmt_cont(units: str, v: dict) -> str:
    whole = units in ('m', '°', '%')
    return f"{v['mae']:.0f} ({v['bias']:+.0f})" if whole else f"{v['mae']:.2f} ({v['bias']:+.2f})"

def _fmt_cat(v: dict) -> str:
    h = '-' if v['heidke'] is None else f"{v['heidke']:.2f}"
    return f"{h} ({v['pod']:.2f}/{v['far']:.2f})"

def _side_tables(lines: list, side: list, models: list) -> None:
    V = _verify()
    for n in list(V.CONT) + list(V.CAT):
        rows = [s for s in side if s['measure'] == n]
        if not rows:
            continue
        if n in V.CONT:
            title, units = V.CONT[n]
            lines += ['', f"**{title}**: mean error past the report's precision ({units}), bias in brackets", '']
        else:
            lines += ['', f'**{V.CAT[n]}**: Heidke skill (POD / false alarms)', '']
        lines += ['| Region | Lead | n | ' + ' | '.join(models) + ' |', '|---|---|---|' + '---|' * len(models)]
        for s in rows:
            cells = []
            for m in models:
                v = s['models'].get(m)
                cells.append('' if v is None else _fmt_cont(V.CONT[n][1], v) if n in V.CONT else _fmt_cat(v))
            lines.append(f"| {s['region']} | {s['lead']} | {s['n']:,} | " + ' | '.join(cells) + ' |')

def report_md(sc: dict) -> str:
    V = _verify()
    models = sorted({r['model'] for r in sc['scores']})
    when = datetime.strptime(sc['month'], '%Y-%m').strftime('%B %Y')
    lines = [f"# The models' scores at weather stations, {when}", '', f"{sc['pairs']:,} forecasts paired with METARs (the forecast job's own 00Z and 12Z runs at Verify's valid hours), scored as Verify scored them. Generated {sc['generated']} (UTC).", '']
    if sc['stations_left_out']:
        lines.append(f'Stations left out (every model off by {V.STATION_OFF:.0f} °C or more the same way): ' + ', '.join(sc['stations_left_out']) + '.')
    lines += ['', '## Side by side', '', 'The same stations, hours and leads for every model.']
    _side_tables(lines, sc['side_by_side'], models)
    for r in sc['regimes']:
        lines += ['', f"## Side by side: {r['regime']} (as the reports saw the sky)", '', f"{r['pairs']:,} pairs."]
        _side_tables(lines, r['side_by_side'], models)
    if sc['by_sun']:
        lines += ['', "## Bias by the sun's time", '', "Leads 1-48 h, each model's own pairs.", '', '| Region | Part | Measure | ' + ' | '.join(models) + ' |', '|---|---|---|' + '---|' * len(models)]
        for s in sc['by_sun']:
            title, units = V.CONT[s['measure']]
            cells = [f"{s['models'][m]['bias']:+.2f} {units}" if m in s['models'] else '' for m in models]
            lines.append(f"| {s['region']} | {s['part']} | {title} | " + ' | '.join(cells) + ' |')
    lines += ['', "## The forecast job's runs against the same runs rebuilt", '']
    if sc['gap']:
        lines += ["Mean error past precision, the job's run then the rebuilt one (bias in brackets): a measurement, not a gate.", '', "| Model | Region | Measure | n | Job's run | Rebuilt |", '|---|---|---|---|---|---|']
        for g in sc['gap']:
            for nm in GAP_MEASURES:
                if nm in g:
                    v = g[nm]
                    lines.append(f"| {g['model']} | {g['region']} | {V.CONT[nm][0]} | {v['n']:,} | {v['mae_ops']:.2f} ({v['bias_ops']:+.2f}) | {v['mae_rebuilt']:.2f} ({v['bias_rebuilt']:+.2f}) |")
    else:
        lines.append('No station-hours scored from both yet.')
    if sc.get('window'):
        lines += ['', '## The soaring window', '', "When the usable thermals came on and went off (the air rising 1.7 m/s through the gliders' climbs; the forecast's W* reaching 1.3 m/s), by 1 degree cell and solar day, in solar hours: the error the forecast less the gliders' (bias in brackets), only where gliders were there half an hour before the first climb or after the last; missed, the cell-days with climbs and no forecast window; false, a forecast window where gliders flew and found none.", '', '| Model | Lead | Cell-days | Onset n | Onset h | End n | End h | Missed | False |', '|---|---|---|---|---|---|---|---|---|']
        f = lambda v: '' if v['mae'] is None else f"{v['mae']:.2f} ({v['bias']:+.2f})"
        for r in sc['window']:
            lines.append(f"| {r['model']} | {r['lead']} | {r['n']:,} | {r['on']['n']:,} | {f(r['on'])} | {r['off']['n']:,} | {f(r['off'])} | {r['missed']:,} | {r['false']:,} |")
    if sc.get('window_call'):
        lines += ['', "## The soaring window by the forecast's own call", '', "The same, split by the day's strongest forecast W* in the cell (weak under 1.3 m/s, moderate to 2, strong from 2): people fly the days the forecast promised, so the cell-days flown lean to its good calls, and a weak call's few cell-days are the days the gliders went anyway.", '', '| Model | Lead | Call | Cell-days | Onset h | End h | Missed | False |', '|---|---|---|---|---|---|---|---|']
        f = lambda v: '' if v['mae'] is None else f"{v['mae']:.2f} ({v['bias']:+.2f})"
        for r in sc['window_call']:
            lines.append(f"| {r['model']} | {r['lead']} | {r['call']} | {r['n']:,} | {f(r['on'])} | {f(r['off'])} | {r['missed']:,} | {r['false']:,} |")
    if sc.get('scoreboard'):
        b = sc['scoreboard']
        lines += ['', '## The launch scoreboard', '', f"The blend in shadow (bundle {b['id']}, scored {b.get('made_at')}) against each model, one at a time (ARCHITECTURE.md decision 8): won, the pooled interval under 0; consistent, 7 of the last 8 weeks better; lost, the interval wholly over +2 %. A field the blend takes from its base is listed, not counted."]
        for r, v in sorted(b['regions'].items()):
            lines += ['', f"**{r}: {('launch' if v.get('launch') else 'not yet')}**, {v.get('won', 0)} of {v.get('counted', 0)} counted won (lost: {', '.join((n.replace('|', ' · ') for n in v.get('lost') or [])) or 'none'}; by chance alone ~{v.get('chance_lost') or 0:.1f}).", '', '| Field, truth, model | Counted | Diff | 95 % | Weeks won | Won | Lost |', '|---|---|---|---|---|---|---|']
            for x in v.get('rows') or []:
                if not x.get('counted'):
                    lines.append(f"| {x['name'].replace('|', ' · ')} | no | | | | | |")
                    continue
                d = lambda y: '' if y is None else f'{y:+.3f}'
                lines.append(f"| {x['name'].replace('|', ' · ')} | yes | {d(x.get('diff'))} | {d(x.get('lo'))} to {d(x.get('hi'))} | {x.get('weeks_won', 0)} of {x.get('weeks', 0)} | {('yes' if x.get('won') else 'no')} | {('yes' if x.get('lost') else 'no')} |")
    lines += ['', '## Runs missing', '']
    miss = sc['missing_runs']
    if miss is None:
        lines.append('Not checked: no OPS read secrets.')
    elif not miss:
        lines.append('None: every run the forecast job builds of each model it recorded is in OPS.')
    else:
        by = {}
        for r in miss:
            by.setdefault(r['model'], []).append(r['run'])
        for m, runs in sorted(by.items()):
            lines.append(f"- {m}: {len(runs)} ({', '.join(runs[:12])}{(', ...' if len(runs) > 12 else '')})")
    if sc['counts']:
        lines += ['', '## Pairs kept this month', '', "Counts only: the flights' and climbs' values stay sealed.", '', '| Kind | Rows |', '|---|---|'] + [f'| {k} | {v:,} |' for k, v in sorted(sc['counts'].items())]
    return '\n'.join(lines) + '\n'

def write(lk, sc: dict, now: datetime) -> tuple[str, str]:
    pre = report_prefix(sc['month'])
    ts = f'{now:%Y%m%dT%H%M%SZ}'
    kj, km = (f'{pre}{ts}.json.lenc', f'{pre}{ts}.md.lenc')
    lk.put(kj, json.dumps(sc, sort_keys=True, separators=(',', ':')).encode(), strict=False)
    lk.put(km, report_md(sc).encode(), strict=False)
    return (kj, km)

def run(lk, ops, month: str | None=None, today: date | None=None, again: bool=False, now: datetime | None=None, counts: dict | None=None) -> dict | None:
    now = now or datetime.now(UTC)
    today = today or now.date()
    counts = counts if counts is not None else {}
    month = month or due_month(today)
    if not again and reports(lk, month):
        _n(counts, 'already')
        return None
    if not P.month_closed('metar', month, today):
        _n(counts, 'month_open')
    sc = score_month(lk, ops, month, now, counts)
    kj, km = write(lk, sc, now)
    _n(counts, 'written')
    return {'json': kj, 'md': km, 'month': month}

def fetch(lk, out_dir: str, counts: dict | None=None, sv: str=SV) -> list:
    counts = counts if counts is not None else {}
    os.makedirs(out_dir, exist_ok=True)
    wrote = []
    for m, (kj, km) in sorted(reports(lk, sv=sv).items()):
        for key, path in ((km, os.path.join(out_dir, f'{m}.md')), (kj, os.path.join(out_dir, f'{m}.json'))):
            body = lk.get(key)
            if key == kj:
                body = (json.dumps(json.loads(body), indent=1, sort_keys=True) + '\n').encode()
            old = None
            if os.path.exists(path):
                with open(path, 'rb') as fh:
                    old = fh.read()
            if old != body:
                with open(path, 'wb') as f:
                    f.write(body)
                wrote.append(path)
        _n(counts, 'months')
    _n(counts, 'files_written', len(wrote))
    return wrote

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='The monthly score report (what replaces Verify)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('run')
    a.add_argument('--month', help='YYYY-MM (default: the newest closed month)')
    a.add_argument('--again', action='store_true', help='write another report though the month has one')
    a.add_argument('--day', help='today, YYYY-MM-DD (default: today, UTC)')
    a.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    a.add_argument('--ops-local', help='a folder as OPS (else OPS_S3_ENDPOINT, OPS_S3_BUCKET, OPS_RO_*)')
    f = sub.add_parser('fetch')
    f.add_argument('--out', required=True)
    f.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    x = ap.parse_args(argv)
    if x.cmd == 'run' and x.month and (not re.fullmatch('\\d{4}-\\d{2}', x.month)):
        print('score: --month is YYYY-MM', flush=True)
        return 2
    lk = lakem.from_env('LAKE', x.local)
    if lk is None:
        print('score: no lake (LAKE_S3_* not set): nothing done', flush=True)
        return 0
    counts = {}
    try:
        if x.cmd == 'fetch':
            fetch(lk, x.out, counts)
        else:
            import pairjob
            ops = pairjob.ops_store(local=x.ops_local)
            today = date.fromisoformat(x.day) if x.day else None
            res = run(lk, ops, x.month, today, x.again, counts=counts)
            print('score: ' + (f"the report of {res['month']} written" if res else 'nothing to write'), flush=True)
    except lakem.LakeError:
        print('score: the lake failed', flush=True)
        print(lk.summary(), flush=True)
        return 1
    print('score: ' + ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))), flush=True)
    print(lk.summary(), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
