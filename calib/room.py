from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
import slots
import verifyjob
ACCOUNT_JOBS = verifyjob.ACCOUNT_JOBS
HEADROOM = 2
WINDOW_MIN = 65
SLOT_EXTRA = 4
ECMWF_HOURS = 2
VERIFY_JOBS = verifyjob.VERIFY_JOBS
OWN_AFTER = 1

def slot_table() -> dict[str, dict]:
    import ast
    with open(os.path.join(slots.HERE, '..', 'pipeline', 'check.py'), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any((getattr(t, 'id', None) == 'SLOTS' for t in node.targets)):
            return ast.literal_eval(node.value)
    raise RuntimeError('no SLOTS in check.py')

def _due(hhmm: str, now: datetime, window: float) -> bool:
    t = now.replace(hour=int(hhmm[:2]), minute=int(hhmm[3:]), second=0, microsecond=0)
    return any((now <= x <= now + timedelta(minutes=window) for x in (t, t + timedelta(days=1))))

def coming(now: datetime, table: dict[str, dict], window: float=WINDOW_MIN) -> int:
    n = 0
    for hhmm, models in table.items():
        if _due(hhmm, now, window):
            n += len(models) + SLOT_EXTRA + (ECMWF_HOURS if 'ecmwf' in models else 0)
    n += sum((VERIFY_JOBS for hhmm in slots.OTHER if _due(hhmm, now, window)))
    return n

def shards(base: int, most: int, running: int, waiting: int, upcoming: int) -> tuple[int, str]:
    if waiting:
        return (base, f'{waiting} job(s) waiting for a runner')
    free = ACCOUNT_JOBS - HEADROOM - running - upcoming - OWN_AFTER
    n = max(base, min(most, free))
    return (n, f'{running} running, up to {upcoming} starting in the next {WINDOW_MIN} min: room for {free}')
LAB_JOBS = ('ingest', 'fingerprint', 'build', 'score', 'train', 'thermal')
LAB_BACK = 15

def lab_par(now: datetime, table: dict[str, dict], running: int | None, waiting: int | None, kinds: dict[str, int] | None=None, total: int=ACCOUNT_JOBS, headroom: int=HEADROOM, window: float=WINDOW_MIN, back: float=LAB_BACK) -> tuple[int, str]:
    if now.tzinfo is None:
        raise ValueError('lab_par: now without a time zone')
    now = now.astimezone(timezone.utc)
    if running is None or waiting is None:
        return (0, "the account's jobs not listed: par 0")
    if int(waiting) > 0:
        return (0, f'{int(waiting)} job(s) waiting for a runner: par 0')
    going = max(0, int(running))
    up = coming(now - timedelta(minutes=back), table, window + back)
    par = max(0, total - headroom - up - going)
    kinds = {k: max(0, int(n)) for k, n in (kinds or {}).items()}
    lab, verify = (sum((n for k, n in kinds.items() if k != 'verify')), kinds.get('verify', 0))
    return (par, f'{going} running ({lab} lab, {verify} Verify), up to {up} starting from {back:g} min ago to {window:g} min on: par {par}')

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=int, required=True)
    ap.add_argument('--most', type=int, required=True)
    a = ap.parse_args(argv)
    try:
        repo = verifyjob.source()
        running, waiting = verifyjob.jobs_going([repo, os.environ['GITHUB_REPOSITORY']])
        n, why = shards(a.base, a.most, running, waiting, coming(datetime.now(timezone.utc), slot_table()))
    except Exception as e:
        n, why = (a.base, f"the runners' jobs didn't list ({e})")
    print(f'{n} shards: {why}', file=sys.stderr)
    print(f'shards={n}')
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
