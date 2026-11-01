from __future__ import annotations
import csv
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import date
import common
import rows
import track
import weglide
import workq
HOURS = [(9, 13), (13, 15), (15, 16), (16, 17), (17, 22)]
WS_BANDS = [(0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3.5), (3.5, 99)]
WIND_BANDS = [(0, 3), (3, 6), (6, 9), (9, 99)]

def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    a, b = (math.floor(k), math.ceil(k))
    return xs[a] + (xs[b] - xs[a]) * (k - a)

def dist(xs) -> str:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return '–'
    return f'{q(xs, 0.5):.2f} ({q(xs, 0.25):.2f}–{q(xs, 0.75):.2f}; 5–95 %: {q(xs, 0.05):.2f}–{q(xs, 0.95):.2f}) n={len(xs)}'

def med(xs) -> str:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return f'{st.median(xs):.2f} n={len(xs)}' if xs else '–'

def ratio(a, b):
    return a / b if a is not None and b is not None and (a > 0) and (b > 0) else None

def leg_ratio(r):
    return ratio(r['vair'], r['m_xc'])

def climb_ratio(r):
    return ratio(r['rate'], r['m_ach'])

def circ_ratio(r):
    return ratio(r['therm'], r['m_circ'])

def usable_leg(r):
    return r['ok'] and r['cat'] == 'thermal' and r['paired']

def fails(r) -> list[str]:
    out = []
    t, ca, d = (r.get('t') or 0, r.get('ca') or 0, r.get('dist') or 0)
    if t <= 600:
        out.append('t ≤ 600 s')
    if ca <= 0.2:
        out.append('mean climb ≤ 0.2 m/s')
    if d < 30:
        out.append('< 30 km')
    if (r.get('tcount') or 0) < 2:
        out.append('< 2 thermals')
    if r.get('last') and (r.get('dh') or 0) < -500:
        out.append('final glide losing > 500 m')
    if ca > 0 and t and (abs((r.get('dh') or 0) / ca) / t >= 0.3):
        out.append('≥ 30 % height correction')
    return out

def table(title, groups, cols) -> list[str]:
    out = [f'### {title}', '', '| | ' + ' | '.join((h for h, _ in cols)) + ' |', '|---|' + '---|' * len(cols)]
    for g, rs in groups:
        out.append(f'| {g} | ' + ' | '.join((fn(rs) for _, fn in cols)) + ' |')
    return out + ['']

def by(rs, key):
    d = defaultdict(list)
    for r in rs:
        d[key(r)].append(r)

    def order(k):
        m = re.match('-?\\d+(\\.\\d+)?', str(k))
        return (0, float(m.group(0)), str(k)) if m else (1, 0.0, str(k))
    return sorted(d.items(), key=lambda kv: order(kv[0]))

def band(x, bands):
    if x is None:
        return '–'
    for a, b in bands:
        if a <= x < b:
            return f"{a}–{(b if b < 99 else '+')}"
    return '–'

def hour_bin(x):
    if x is None:
        return '–'
    for a, b in HOURS:
        if a <= x < b:
            return f'{a}–{b}'
    return 'other'
SEASON = {**{m: 'spring (Mar–May)' for m in (3, 4, 5)}, **{m: 'summer (Jun–Aug)' for m in (6, 7, 8)}, **{m: 'autumn (Sep–Nov)' for m in (9, 10, 11)}, **{m: 'winter (Dec–Feb)' for m in (12, 1, 2)}}
LATE = 15

def share(rs, test) -> str:
    return f'{sum((1 for r in rs if test(r))) / len(rs):.0%}' if rs else '–'

def mmed(xs) -> str:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return f'{st.median(xs):.2f}' if xs else '–'
SURV_COLS = [('airborne', lambda rs: str(len(rs))), ('still flying 1 h later', lambda rs: share(rs, lambda r: r['out'] == 'flying')), ('landed home', lambda rs: share(rs, lambda r: r['out'] == 'home')), ('landed away or engine', lambda rs: share(rs, lambda r: r['out'] in ('away', 'engine'))), ('landed, place unknown (seed)', lambda rs: share(rs, lambda r: r['out'] == 'unknown')), ('with the forecast', lambda rs: str(sum((r['paired'] for r in rs)))), ('W* own (median)', lambda rs: mmed([r['m_ws0'] for r in rs])), ('W* shown (held)', lambda rs: mmed([r['m_ws'] for r in rs])), ('held over own', lambda rs: share([r for r in rs if r['held'] is not None], lambda r: r['held']))]
FOUND = ('climb', 'save')

def _failed_share(rs) -> str:
    n = [r for r in rs if r['out'] in FOUND or r['failed']]
    return share(n, lambda r: r['failed'])
GLIDE_COLS = [('glides', lambda rs: str(len(rs))), ('searches', lambda rs: str(sum((r['search'] for r in rs)))), ('climb found', lambda rs: str(sum((r['out'] == 'climb' for r in rs)))), ('low save', lambda rs: str(sum((r['out'] == 'save' for r in rs)))), (f'weak (< {track.WEAK_RATE} m/s)', lambda rs: str(sum((r['out'] == 'weak' for r in rs)))), ('landed away or engine', lambda rs: str(sum((r['out'] in ('away', 'engine') for r in rs)))), ('landed home', lambda rs: str(sum((r['out'] == 'home' for r in rs)))), ('landed, unknown', lambda rs: str(sum((r['out'] == 'unknown' for r in rs)))), ('failed / (found + failed)', _failed_share), ('W* own: found / failed', lambda rs: f"{mmed([r['m_ws0'] for r in rs if r['out'] in FOUND])} / {mmed([r['m_ws0'] for r in rs if r['failed']])}"), ('W* shown: found / failed', lambda rs: f"{mmed([r['m_ws'] for r in rs if r['out'] in FOUND])} / {mmed([r['m_ws'] for r in rs if r['failed']])}")]

def late(r) -> bool:
    return r.get('solar') is not None and r['solar'] >= LATE

def survival_section(H, G) -> list[str]:
    out = ['## Late landings (survival)', '', f"The climbs found don't show the searches that failed: a pilot who found nothing late and landed leaves no late climb. Here each whole hour a glider was soaring (from {track.AFTER_TAKEOFF_S // 60} min after takeoff to its landing or engine start): still flying an hour later, or how it ended within it: home (within {track.HOME_KM:g} km of the takeoff: late in the day a pilot's choice, not a sign of no lift), away (a landout or another field) or engine (started in the air); seed flights kept no track, so where they landed is unknown. The forecast is the site's where the glider was then: W* own (the forecast's) and shown (with the evening hold). Days stored before the forecast was sampled there have none until they're built again ({sum((r['old_day'] for r in H))} of {len(H)} hours).", '']
    out += table('By local hour', by(H, lambda r: int(r['lh'])), SURV_COLS)
    out += table('By local solar hour', by(H, lambda r: hour_bin(r['solar'])), SURV_COLS)
    lt = [r for r in H if late(r)]
    out += table(f'Solar hour {LATE} on, by season', by(lt, lambda r: SEASON.get(r['month'], '–')), SURV_COLS)
    au = [r for r in H if r['month'] in (9, 10, 11)]
    out += table('Autumn (Sep–Nov) by month and solar hour', by(au, lambda r: f"{r['month']:02d} {hour_bin(r['solar'])}"), SURV_COLS)
    out += table(f"Solar hour {LATE} on, by the forecast's own W* (m/s)", by(lt, lambda r: band(r['m_ws0'], WS_BANDS)), SURV_COLS)
    out += table(f"Solar hour {LATE} on, held or not (the evening hold put W* over the forecast's own)", by([r for r in lt if r['held'] is not None], lambda r: 'held' if r['held'] else 'own'), SURV_COLS)
    out += ['## Searches: climbs found and failed', '', f"Each glide from a climb's top to the next climb or the end of soaring, by when it ended. A search: lasting ≥ {track.SEARCH_S // 60} min, losing ≥ {track.SEARCH_M} m or ending under {track.LOW_AGL} m above the ground. Found: a climb (a low save: from under {track.LOW_AGL} m); failed: the next climb under {track.WEAK_RATE} m/s (the site's least usable), a landing away or the engine. A landing home isn't counted as failed. W*: at the glide's end.", '']
    out += table('All glides by solar hour', by(G, lambda r: hour_bin(r['solar'])), GLIDE_COLS)
    sg = [r for r in G if r['search']]
    out += table('Searches by solar hour', by(sg, lambda r: hour_bin(r['solar'])), GLIDE_COLS)
    out += table(f'Searches from solar hour {LATE}, by season', by([r for r in sg if late(r)], lambda r: SEASON.get(r['month'], '–')), GLIDE_COLS)
    out += table(f"Searches from solar hour {LATE}, by the forecast's own W* (m/s)", by([r for r in sg if late(r)], lambda r: band(r['m_ws0'], WS_BANDS)), GLIDE_COLS)
    return out

def pilot_section(L, C, F, H, G) -> list[str]:
    keyed = [f for f in F if f['pilot']]
    per = Counter((f['pilot'] for f in keyed))
    days = defaultdict(set)
    for f in keyed:
        days[f['pilot']].add(f['date'])
    n = sorted(per.values())
    out = ['## Pilots', '', f"- Flights with a pilot key: {len(keyed)} of {len(F)} (the rest: the seed's and the first days' until their day lists are looked up, a few a run).", f"- Distinct pilots: {len(per)}; flights per pilot: max {(n[-1] if n else '–')}, median {(st.median(n) if n else '–')}; pilots on 2 days or more: {sum((len(v) >= 2 for v in days.values()))}.", '']
    sid = rows.strength(L, F)
    lab = lambda r: sid.get(r['fid'], 'unknown')
    order = {'low': 0, 'mid': 1, 'high': 2, 'unknown': 3}
    groups = lambda rs: sorted(by(rs, lab), key=lambda kv: order.get(kv[0], 9))
    cl = [r for r in C if r['paired'] and (not r['engon']) and ((r['rate'] or 0) > 0) and late({'solar': r['m_solar']})]
    lg = [r for r in L if usable_leg(r) and late({'solar': r['m_solar']})]
    lh = [r for r in H if late(r)]
    ls = [r for r in G if r['search'] and late(r)]
    ids = lambda rs: {r['fid'] for r in rs}
    out += [f'### By pilot strength (solar hour {LATE} on)', '', "Strength: the pilot's flown XC speed against the site's on their other days (leave the day out), in terciles; unknown: no key, or no other day yet. For scaling the evening hold by the pilot setting later.", '', '| Strength | flights | climb flown / model (late) | XC flown / model (late) | still flying 1 h later | landed away or engine | failed searches |', '|---|---|---|---|---|---|---|']
    keys = sorted({lab(f) for f in F}, key=lambda k: order.get(k, 9))
    for k in keys:
        sel = lambda rs: [r for r in rs if lab(r) == k]
        out.append(f"| {k} | {len(ids(sel(F)))} | {dist([climb_ratio(r) for r in sel(cl)])} | {dist([leg_ratio(r) for r in sel(lg)])} | {share(sel(lh), lambda r: r['out'] == 'flying')} of {len(sel(lh))} | {share(sel(lh), lambda r: r['out'] in ('away', 'engine'))} | {_failed_share(sel(ls))} of {len(sel(ls))} |")
    return out + ['']

def dead_section(S, R) -> list[str]:
    out = ['## Dead days', '', f"A day the forecast promised and that didn't give has few or no long flights, and the speed rows drop the short ones. So the day lists' short flights (under the minimum distance or time) are kept as a class of their own, never in the speed rows: afternoon launches from solar hour {weglide.AFTERNOON_SOLAR}, a sled ride down within {weglide.SLED_MIN} min, and up to {weglide.SHORT_TRACKS_PER_DAY} a day tracked (did it ever climb: {track.WEAK_RATE} m/s for {track.SHORT_CLIMB_S // 60} min after the release). Pattern, training and tow flights are short on good days too, so a day is judged by its region's share, not by single flights.", '']
    aft = [r for r in S if r['afternoon']]
    tr = [r for r in aft if r['tracked']]
    out += [f"- Short flights kept: {len(S)} ({len(aft)} afternoon launches, {len(tr)} of them tracked; never climbed: {sum((not r['climbed'] for r in tr))}).", '']
    cols = [('short flights', lambda rs: str(len(rs))), ('sled rides', lambda rs: share(rs, lambda r: r['sled'])), ('tracked', lambda rs: str(sum((bool(r['tracked']) for r in rs)))), ('never climbed (tracked)', lambda rs: share([r for r in rs if r['tracked']], lambda r: not r['climbed'])), ('forecast W* own at takeoff', lambda rs: mmed([r['m_ws0'] for r in rs])), ("day's best W* own there", lambda rs: mmed([r['best_ws0'] for r in rs]))]
    out += table("Afternoon short flights by the day's best W* own there (m/s)", by(aft, lambda r: band(r['best_ws0'], WS_BANDS)), cols)
    listed = [r for r in R if r['list']]
    dead = [r for r in R if r['dead']]
    out += [f"### Region-days ({len(R)}: {len(listed)} with a day list, {sum((r['built'] for r in R))} with the forecast built)", '', f'Possibly dead: the forecast good there (best afternoon W* own, median over the takeoffs, ≥ {rows.DEAD_WS} m/s, or XC Speed ≥ {rows.DEAD_XC:g} km/h), few went far (long flights at most {rows.DEAD_LONG_SHARE:.0%} of the uploads, 1 at least) and: {rows.DEAD_MIN_NEVER} or more tracked afternoon short flights never climbed (half of those tracked); or {rows.DEAD_MIN_AFT}+ afternoon launches, {rows.DEAD_SHORT_MIN:.0%}+ of them short, {rows.DEAD_SHORT_UP:.0%} over the normal; or, where the normal is {rows.DEAD_NORMAL_UP}+ uploads, no long flight with uploads at most {rows.DEAD_QUIET:.0%} of it, or where {rows.DEAD_NO_LONG:.0%} of such days have one (normal: same region, weekend or weekday, half of the year). A full day list (100 flights) may miss its shortest flights.', '', f'{len(dead)} flagged.', '']
    if dead:
        out += ['| Day | Region | Type | Uploads (normal) | Long | Afternoon: short / sled / landed early | Best W* own | Best XC | Why |', '|---|---|---|---|---|---|---|---|---|']
        f = lambda x, n=1: '–' if x is None else f'{x:.{n}f}'
        for r in dead[:150]:
            out.append(f"| {r['date']} | {r['region']} | {r['day_type']} | {r['up']} ({f(r['normal_up'], 0)}) | {r['long']} | {r['aft_short']} / {r['aft_sled']} / {r['aft_early']} of {r['aft']} | {f(r['best_ws0'], 2)} | {f(r['best_xc'], 0)} | {r['dead']} |")
        out.append('')
    gb = [r for r in R if r['built'] and r['list']]
    rcols = [('region-days', lambda rs: str(len(rs))), ('uploads (median)', lambda rs: mmed([r['up'] for r in rs])), ('with a long flight', lambda rs: share(rs, lambda r: r['long'])), ('afternoon short share', lambda rs: (lambda a, b: f'{a / b:.0%}' if b else '–')(sum((r['aft_short'] for r in rs)), sum((r['aft'] for r in rs)))), ('possibly dead', lambda rs: str(sum((bool(r['dead']) for r in rs))))]
    out += table('Built region-days with a list, by the best W* own (m/s) and day type', by(gb, lambda r: f"{band(r['best_ws0'], WS_BANDS)} {r['day_type']}"), rcols)
    return out

def obs_section(O, N) -> list[str]:
    out = ['## Observations against the forecast', '', "Hourly surface reports (ASOS) at up to 3 airports a box, and the evening sounding (00Z, about 8 PM EDT / 5 PM PDT) in the box, from Iowa Environmental Mesonet's archive: every day built, flown or not. Cumulus: FEW / SCT layers under 4,000 m; heights above the ground.", '']
    cols = [('station-days', lambda rs: str(len(rs))), ('highest temperature: reported − forecast (°C)', lambda rs: dist([r['tmax_diff'] for r in rs])), ('hours with cumulus: reported / forecast / both', lambda rs: f"{sum((r['cu_obs'] for r in rs))} / {sum((r['cu_fc'] for r in rs))} / {sum((r['cu_both'] for r in rs))}"), ('cloudbase reported − forecast (m)', lambda rs: dist([r['base_diff'] for r in rs]))]
    out += table('Surface reports by region', [('all', O)] + by(O, lambda r: r['region']), cols)
    scols = [('soundings', lambda rs: str(len(rs))), ('mixing height reported (m)', lambda rs: dist([r['mix_obs'] for r in rs])), ('reported − forecast zi (m)', lambda rs: dist([r['mix_diff'] for r in rs])), ('lapse 0–2 km reported (°C/km)', lambda rs: dist([r['lapse_obs'] for r in rs])), ('forecast blapse', lambda rs: dist([r['blapse_fc'] for r in rs]))]
    out += table('Evening soundings by region', [('all', N)] + by(N, lambda r: r['region']), scols)
    return out
NOLIFT = 'usable legs where the model had no lift'

def day_checks(days, L, C) -> list[dict]:
    per = []
    for d in days:
        dl = [r for r in L if r['date'] == d['date']]
        dc = [r for r in C if r['date'] == d['date']]
        ur = [leg_ratio(r) for r in dl if usable_leg(r)]
        cr = [climb_ratio(r) for r in dc if r['paired'] and (not r['engon'])]
        paired = (sum((r['paired'] for r in dl)) + sum((r['paired'] for r in dc))) / max(1, len(dl) + len(dc))
        why = []
        if any((not b.get('ok') for b in d.get('boxes', []))):
            why.append('a box failed')
        if paired < 0.5:
            why.append(f'paired {paired:.0%}')
        ur2, cr2 = ([x for x in ur if x], [x for x in cr if x])
        if ur2 and (not 0.4 <= st.median(ur2) <= 2.5):
            why.append(f'XC ratio {st.median(ur2):.2f}')
        if cr2 and (not 0.3 <= st.median(cr2) <= 3):
            why.append(f'climb ratio {st.median(cr2):.2f}')
        if dc and sum((bool(r['engon']) for r in dc)) / len(dc) > 0.3:
            why.append('engine on in > 30 % of climbs')
        nolift = sum((1 for r in dl if usable_leg(r) and (not (r['m_xc'] or 0) > 0)))
        if nolift >= 2:
            why.append(f'{nolift} {NOLIFT}')
        regions = sorted({f['region'] for f in d['flights']})
        per.append({'date': d['date'], 'regions': ' '.join(regions), 'flights': len(d['flights']), 'legs': len(dl), 'usable_legs': len(ur), 'climbs': len(dc), 'paired': round(paired, 2), 'xc_ratio': round(st.median(ur2), 2) if ur2 else '', 'climb_ratio': round(st.median(cr2), 2) if cr2 else '', 'flags': '; '.join(why)})
    return per

def bad_days(per) -> dict:
    out = {}
    for p in per:
        why = [w for w in p['flags'].split('; ') if w and (not w.endswith(NOLIFT))]
        if why:
            out[p['date']] = '; '.join(why)
    return out

def build(data: str, days_csv: str | None=None, model: str='hrrr') -> str:
    state = common.read_json(f'{data}/state.json') if __import__('os').path.exists(f'{data}/state.json') else workq.new_state([])
    days = rows.load_days(data)
    pilots = state.get('pilots') or {}
    L, C, F = rows.tables(days, model, pilots=pilots)
    H, G = rows.survival(days, model, pilots=pilots)
    out = [f'# Calibration data report ({model.upper()})', '', f'Built {date.today().isoformat()}. Review this before training (train runs only when started by hand).', '']
    pr = workq.progress(state, date.today())
    out += ['## Progress', '', f"- Days done: {pr['days_done']} ({', '.join((f'{k} {v}' for k, v in sorted(pr['by_status'].items(), key=str)))}); seed days left: {pr['seed_left']} of {pr['seed_days']}; stored with an older file, to build again: {pr['redo_left']}.", f"- WeGlide requests so far: {state['weglide'].get('requests', 0)}; answers: {dict(sorted(state['weglide'].get('status', {}).items()))}.", f"- Runs: {len(state.get('runs', []))}; last: {(state['runs'][-1] if state.get('runs') else '–')}.", f"- Flights {len(F)} (seed {sum((f['src'] == 'seed' for f in F))}), legs {len(L)} (usable thermal legs: {sum((usable_leg(r) for r in L))}), circling climbs {len(C)}.", '']
    cnt = lambda key: ', '.join((f'{k} {v}' for k, v in sorted(Counter((key(f) for f in F)).items(), key=lambda kv: -kv[1])))
    out += ['## Rows', '', f"- By region: {cnt(lambda f: f['region'])}.", f"- By glider class: {cnt(lambda f: f['cls'])}.", f"- By group: {cnt(lambda f: f['grp'])}.", f"- By month: {cnt(lambda f: (f['date'] or '')[5:7])}.", f"- By year: {cnt(lambda f: (f['date'] or '')[:4])}.", f"- Polars: {cnt(lambda f: f['key'] + ('' if f['key_exact'] else '~'))} (~: the nearest similar polar).", '']
    um = state.get('unmapped') or {}
    out += ['### Aircraft types not in aircraft.json (skipped; add them there)', '', (', '.join((f"{k or '(none)'} {v}" for k, v in sorted(um.items(), key=lambda kv: -kv[1]))) or 'none') + '.', '']
    boxes = [b for d in days for b in d.get('boxes', [])]
    out += ['## Pairing with the model', '', f"- Boxes built: {sum((b.get('ok', False) for b in boxes))} of {len(boxes)} (failed: {dict(Counter((b.get('error') for b in boxes if not b.get('ok'))))}).", f"- Legs with the model's values: {sum((r['paired'] for r in L))} of {len(L)}; climbs: {sum((r['paired'] for r in C))} of {len(C)}; flights with PFD at the takeoff: {sum((f['paired'] for f in F))} of {len(F)}.", "- Unpaired legs and climbs are outside the model's hours (6 AM–9 PM local) or in a box that failed.", '']
    out += ['## Field coverage (share present among paired rows)', '', '| Field | Legs | Climbs |', '|---|---|---|']
    pl, pc = ([r for r in L if r['paired']], [r for r in C if r['paired']])
    for k in rows.MODEL_KEYS:
        f = lambda rs: f"{sum((r['m_' + k] is not None for r in rs)) / len(rs):.0%}" if rs else '–'
        out.append(f'| {k} | {f(pl)} | {f(pc)} |')
    out.append('')
    fl = Counter((x for r in L for x in fails(r)))
    out += ['## Outliers and filters', '', f"- Climbs over 6 m/s: {sum(((r['rate'] or 0) > 6 for r in C))}; engine on: {sum((bool(r['engon']) for r in C))}; climbs under 0.2 m/s: {sum(((r['rate'] or 0) < 0.2 for r in C))}.", f"- Legs over 200 km/h (height-corrected ground speed): {sum(((r['vg'] or 0) > 200 for r in L))}; in the air: {sum(((r['vair'] or 0) > 200 for r in L))}.", f"- Legs failing the filters (a leg can fail several): {', '.join((f'{k} {v}' for k, v in fl.most_common())) or 'none'}.", f"- Leg categories: {dict(Counter((r['cat'] for r in L)))}.", '']
    ul = [r for r in L if usable_leg(r)]
    uc = [r for r in C if r['paired'] and (not r['engon']) and ((r['rate'] or 0) > 0)]
    cols = [('XC: flown in the air / model (still air)', lambda rs: dist([leg_ratio(r) for r in rs])), ('Ground speed flown / model along the leg', lambda rs: dist([ratio(r['vg'], r['m_gs']) for r in rs])), ('Circling share flown / model', lambda rs: dist([circ_ratio(r) for r in rs]))]
    ccols = [("Climb flown / model's achieved climb", lambda rs: dist([climb_ratio(r) for r in rs])), ("Flown top AGL − model's usable top (m)", lambda rs: dist([(r['agl1'] - r['m_top']) / 1 if r['agl1'] is not None and r['m_top'] is not None else None for r in rs]))]
    out += ["## Flown / the site's model", '', 'Usable thermal legs (filters passed, cat thermal) and circling climbs (engine off). Median (IQR; 5–95 %). Model 0 (no usable lift there then) leaves a ratio out: counted below.', '']
    out += [f"- Usable legs where the model had no XC speed: {sum((not (r['m_xc'] or 0) > 0 for r in ul))} of {len(ul)}; climbs where it had no climb: {sum((not (r['m_ach'] or 0) > 0 for r in uc))} of {len(uc)} (false 'no lift').", '']
    terr = lambda r: 'flat' if rows.flat(r) else 'hills/mountains'
    out += table('Legs, all and by terrain', [('all', ul)] + by(ul, terr), cols)
    out += table('Legs by local solar hour', by(ul, lambda r: hour_bin(r['m_solar'])), cols)
    out += table("Legs by the model's W* (m/s)", by(ul, lambda r: band(r['m_ws'], WS_BANDS)), cols)
    out += table('Legs by region', by(ul, lambda r: r['region']), cols)
    out += table('Legs by glider class', by(ul, lambda r: r['cls']), cols)
    out += table('Legs by source (seed: legs from the first collection)', by(ul, lambda r: r['src']), cols)
    out += table('Climbs, all and by terrain', [('all', uc)] + by(uc, terr), ccols)
    out += table('Climbs by local solar hour', by(uc, lambda r: hour_bin(r['m_solar'])), ccols)
    out += table("Climbs by the model's W* (m/s)", by(uc, lambda r: band(r['m_ws'], WS_BANDS)), ccols)
    out += table('Climbs by region', by(uc, lambda r: r['region']), ccols)
    out += table("Climbs by source (seed: detected by the first collection's script; new: calib/track.py)", by(uc, lambda r: r['src']), ccols)
    out += survival_section(H, G)
    out += pilot_section(L, C, F, H, G)
    out += dead_section(rows.shorts(days, model, pilots), rows.regiondays(days, state.get('daylists') or {}, model))
    out += obs_section(*rows.observations(days, model))
    rl = [r for r in L if r['r_autoMean'] is not None and (r['dist'] or 0) >= 20]
    rcols = [('legs', lambda rs: str(len(rs))), ('mean distance flown→Auto route (km)', lambda rs: med([r['r_autoMean'] for r in rs])), ('…→straight line (km)', lambda rs: med([r['r_straightMean'] for r in rs])), ('Auto route nearer / farther / same', lambda rs: f"{sum((r['r_autoMean'] < r['r_straightMean'] - 0.05 for r in rs))} / {sum((r['r_autoMean'] > r['r_straightMean'] + 0.05 for r in rs))} / {sum((abs(r['r_autoMean'] - r['r_straightMean']) <= 0.05 for r in rs))}"), ('max distance (km)', lambda rs: med([r['r_autoMax'] for r in rs])), ('share off to the same side', lambda rs: med([r['r_sameShare'] for r in rs])), ('bends where the pilot did (> 2 km)', lambda rs: (lambda xs: f'{sum(xs) / len(xs):.0%} of {len(xs)}' if xs else '–')([r['r_bendMatch'] for r in rs if r['r_bendMatch'] is not None])), ("bends where the pilot didn't", lambda rs: (lambda xs: f'{sum(xs) / len(xs):.0%} of {len(xs)}' if xs else '–')([r['r_autoBendsAlone'] for r in rs if r['r_autoBendsAlone'] is not None])), ("Auto route's time / flown", lambda rs: med([ratio(r['r_autoH'], r['r_flownH']) for r in rs]))]
    out += ['## Auto route against the line flown', '', "Per free-distance leg of 20 km or more, its own turnpoints, from the hour flown: the line flown in straight pieces of ≥ 5 km (new flights: the track; seed: through the leg's climbs, the track wasn't kept) against xctask autoPaths' way through the site's model (without what only the app has: Thermal Triggers' weighting, the 3D thermals, the wave bars). Medians.", '']
    out += table('All legs, by how the line was made', [('all', rl)] + by(rl, lambda r: r['line_src']), rcols)
    out += table("Where the pilot's line bent (> 2 km off the course) or didn't", by(rl, lambda r: 'bent' if abs(r['r_flownBend'] or 0) > 2 else 'straight'), rcols)
    out += table('By region', by(rl, lambda r: r['region']), rcols)
    out += table("Cumulus or blue (the model's)", by(rl, lambda r: 'cu' if r['m_cu'] else 'blue'), rcols)
    out += table("By the model's wind (m/s)", by(rl, lambda r: band(r['m_wind'], WIND_BANDS)), rcols)
    out += table('Thermal legs that passed the filters', by(rl, lambda r: 'usable' if usable_leg(r) else 'other'), rcols)
    out += [f"- Auto route errors: {sum((r['r_error'] is not None for r in L))}.", '']
    pf = [ratio(f['pfd'], f['free_km']) for f in F if f['paired']]
    out += ['## Potential Flight Distance (Std, at the takeoff, storm cut 2 %) / km flown', '', dist(pf) + ' (typical flights: PFD reads above them by design).', '']
    per = day_checks(days, L, C)
    sus = [p for p in per if p['flags']]
    out += ['## Suspicious days', '', f'{len(sus)} of {len(per)} days flagged (all days: the CSV beside this report).', '']
    if sus:
        out += ['| Day | Regions | Flights | Usable legs | Climbs | Paired | XC ratio | Climb ratio | Why |', '|---|---|---|---|---|---|---|---|---|']
        out += [f"| {p['date']} | {p['regions']} | {p['flights']} | {p['usable_legs']} | {p['climbs']} | {p['paired']} | {p['xc_ratio']} | {p['climb_ratio']} | {p['flags']} |" for p in sus[:200]]
        out.append('')
    if days_csv:
        with open(days_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(per[0].keys()) if per else ['date'])
            w.writeheader()
            w.writerows(per)
    return '\n'.join(out) + '\n'
if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    text = build(sys.argv[1], sys.argv[3] if len(sys.argv) == 4 else None)
    with open(sys.argv[2], 'w') as f:
        f.write(text)
