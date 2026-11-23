from __future__ import annotations
import argparse
import os
import sys
import lake
from . import events, exposure, guard, join
STAGES = {'events': events.run, 'exposure': exposure.run, 'join': join.run}

def main(argv=None, env=None) -> int:
    ap = argparse.ArgumentParser(prog='thermalml')
    ap.add_argument('stage', choices=[*STAGES, 'all', 'learn'], nargs='?', default='all')
    ap.add_argument('--local', help='a folder as the lake (tests, a restored copy)')
    ap.add_argument('--days', help='these solar days only (made again: a check that the rule still gives the same)')
    ap.add_argument('--max', type=int, default=None, help='solar days made a stage at most')
    ap.add_argument('--held', type=int, default=7, help="learn: the last days held out for the surface's skill")
    a = ap.parse_args(argv)
    e = os.environ if env is None else env
    if not e.get('CALIB_DATA_KEY') or (not a.local and (not all((e.get(f'LAKE_S3_{k}') for k in ('ENDPOINT', 'BUCKET', 'KEY', 'SECRET'))))):
        print('thermalml: no lake (LAKE_S3_* / CALIB_DATA_KEY not set): nothing done')
        return 0
    lk = lake.from_env('LAKE', a.local, e)
    if lk is None:
        print('thermalml: no lake: nothing done')
        return 0
    days = [d for d in (a.days or '').split(',') if d] or None
    if a.stage == 'learn':
        from . import learn
        if not guard.allowed(e):
            print(f'thermalml learn: {guard.SWITCH} is not 1: nothing learned')
            return 0
        try:
            n = learn.run(lk, days=days, env=e, held=a.held)
        except guard.Refused:
            print('thermalml learn: refused: an object holds a feed that is not admitted')
            return 1
        print('thermalml learn: ' + (', '.join((f'{k} {v}' for k, v in sorted(n.items()))) or 'nothing to do'))
        print(lk.summary())
        return 1 if n.get('differs') else 0
    bad = 0
    for name in list(STAGES) if a.stage == 'all' else [a.stage]:
        n = STAGES[name](lk, days=days, cache={}, most=a.max)
        print(f'thermalml {name}: ' + (', '.join((f'{k} {v}' for k, v in sorted(n.items()))) or 'nothing to do'))
        bad += n.get('differs', 0)
    print(lk.summary())
    return 1 if bad else 0
if __name__ == '__main__':
    sys.exit(main())
