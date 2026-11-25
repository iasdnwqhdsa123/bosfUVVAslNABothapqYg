from __future__ import annotations
from . import cycle, movement, strength
from . import learn as lab
from . import trigger as tg
REPORT_V = 'r1'

def volume(days: list) -> dict:
    climbs = thermals = failed = glides = 0
    who = set()
    for d in days:
        climbs += d.climb_ok.count(True)
        thermals += d.thermal_ok.count(True)
        failed += d.failed_ok.count(True)
        glides += sum((1 for g in d.join.get('glides', []) if d.climb_ok[g['climb']]))
        who.update((d.acday(c) for i, c in enumerate(d.climbs) if d.climb_ok[i]))
    return {'days': len(days), 'climbs': climbs, 'thermals': thermals, 'multi_climb_thermals': sum((1 for d in days for t, th in enumerate(d.thermals) if d.thermal_ok[t] and th['n'] > 1)), 'failed': failed, 'glides': glides, 'acdays': lab.acdays(who)}

def where(days: list) -> list:
    acc = {}
    for d in days:
        for (I, J, hb), (E, N) in tg.rows_of(d).items():
            la, lo = lab.centre(I, J)
            a = acc.setdefault(lab.region(la, lo), [0, 0.0, set()])
            a[0] += N
            a[1] += E
        for i, c in enumerate(d.climbs):
            if d.climb_ok[i]:
                acc.setdefault(lab.region(c['la0'], c['lo0']), [0, 0.0, set()])[2].add(d.acday(c))
    out = []
    for (a, b), (N, E, who) in sorted(acc.items()):
        if lab.acdays(who) >= lab.K:
            out.append([f'{a * lab.REGION_DEG:+.0f}{b * lab.REGION_DEG:+.0f}', N, lab.r(E, 1)])
    return out

def area_covered(t: dict) -> list:
    deg = float(t.get('cell_deg') or 0)
    if not deg or not t.get('cells'):
        return []
    acc = {}
    for c in t['cells']:
        k = lab.region(c[0], c[1])
        acc[k] = acc.get(k, 0) + 1
    per = (lab.REGION_DEG / deg) ** 2
    out = [[f'{a * lab.REGION_DEG:+.0f}{b * lab.REGION_DEG:+.0f}', n, lab.r(n / per, 8)] for (a, b), n in acc.items()]
    return sorted(out, key=lambda x: (-x[1], x[0]))

def text(rep: dict) -> str:
    v, sk = (rep['volume'], rep['skill'])
    lines = ['thermal learners ' + ', '.join((f'{k} {x}' for k, x in rep['versions'].items())), 'data: ' + ', '.join((f'{k} {v[k]}' for k in ('days', 'acdays', 'climbs', 'thermals', 'multi_climb_thermals', 'failed', 'glides'))), f"k rule: {rep['k']['cells_kept']} cells kept, {rep['k']['cells_dropped']} dropped", f"surface: {rep['trigger']['events']} first finders over {rep['trigger']['coverage_h']} coverage hours in {rep['trigger']['cells']} cells, {len(rep['where'])} regions; {rep['trigger']['events_lost_to_k']} lost to the k rule, {rep['trigger']['no_exposure']} without exposure"]
    if rep['trigger'].get('area'):
        lines.append("surface's area: " + ', '.join((f'{r} {n} cells ({sh * 100:.2f} %)' for r, n, sh in rep['trigger']['area'][:8])))
    if sk.get('deviance'):
        lines.append(f"held out {sk['test_days']} days ({sk['test_events']} events): deviance " + ', '.join((f'{k} {x}' for k, x in sk['deviance'].items())))
        lines += [f'  gain {k}: {g[0]} (90 % {g[1]} to {g[2]})' for k, g in sk['gain'].items()]
        lines += [f"  region {k}: {x['events']} events, " + ', '.join((f'{m} {g}' for m, g in x['gain'].items())) for k, x in sk.get('regions', {}).items()]
    else:
        lines.append(f"held out: {sk.get('note', 'none')}")
    if 'baseline' in sk:
        lines.append('  baseline: not available')
    c = rep['cycle'].get('all')
    if c:
        iv = c['intervals']
        lines.append(f"cycle: {rep['cycle_sources']} sources, {rep['cycle_multi']} used more than once; re-use n {iv['n']}" + (f", median {iv['q50']} s" if iv.get('q50') is not None else '') + f", {iv['censored']} censored" + f"; lives n {c['life']['n']}, {c['life']['ended']} seen to end")
    for part in ('glide', 'nearest'):
        s = rep['spacing'].get(part, {}).get('all')
        if s:
            lines.append(f"spacing {part}: n {s['n']}, median {s['q'][2]} m")
    m = rep['movement'].get('all', {})
    if m:
        lines.append('movement: ' + ', '.join((f'{k} {x}' for k, x in m.items())))
    mx = rep['mixture'].get('all')
    if mx:
        lines.append(f"  pairs {mx['pairs']} from {mx['sources']} sources: anchored {mx['anchored']}, drifting {mx['drifting']}, background {mx['background']}")
    lines.append(f"strength: {rep['strength_rows']} rows ({rep['strength_below_k']} under k)")
    return '\n'.join(lines) + '\n'

def learn(days: list, outs: dict, kept: dict, held: int=7, baseline=None) -> dict:
    t = outs['trigger']
    rep = {'versions': {'trigger': lab.version(tg.TRIGGER_V), 'cycle': lab.version(cycle.CYCLE_V), 'movement': lab.version(movement.MOVEMENT_V), 'strength': lab.version(strength.STRENGTH_V)}, 'volume': volume(days), 'k': {'cells_kept': kept.get('cells_kept', 0), 'cells_dropped': kept.get('cells_dropped', 0)}, 'where': where(days), 'trigger': {'events': int(sum((c[3] for c in t.get('cells', [])))), 'coverage_h': lab.r(sum((c[2] for c in t.get('cells', []))), 1), 'cells': len(t.get('cells', [])), 'hours': t.get('hours', []), 'sigma': t.get('sigma'), 'events_lost_to_k': t.get('counts', {}).get('events_lost_to_k', 0), 'no_exposure': t.get('counts', {}).get('no_exposure', 0), 'area': area_covered(t)}, 'skill': tg.skill(days, held=held, baseline=baseline), 'cycle': outs['cycle']['bins'], 'cycle_sources': outs['cycle']['sources'], 'cycle_multi': outs['cycle']['multi_use'], 'spacing': t.get('spacing', {}), 'movement': outs['movement']['thermals'], 'mixture': outs['movement']['mixture'], 'strength_rows': len(outs['strength']['rows']), 'strength_below_k': outs['strength']['below_k']}
    rep['text'] = text(rep)
    return rep
