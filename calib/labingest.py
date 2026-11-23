from __future__ import annotations
import argparse
import gzip
import os
import sys
import time
import zlib
from datetime import date, datetime, timedelta, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import lake as lakemod
import obs_raob
if common.PIPELINE not in sys.path:
    sys.path.insert(0, common.PIPELINE)
import verify
IEM_ROOT = 'raw/obs/iem'
IEM_LAG = 2
IEM_BACK = 10
IEM_TYPES = ('3', '4')
IEM_MINUTES = 15.0
FAIL_STOP = 3
LAKE_STOP = 3
CSV_HEAD = b'station,'

def _n(counts: dict, k: str, v: int=1) -> None:
    counts[k] = counts.get(k, 0) + v

def iem_key(day: date, k: int) -> str:
    return f'{IEM_ROOT}/{day.isoformat()}/part-{k}.csv.gz'

def iem_days(today: date, lag: int=IEM_LAG, back: int=IEM_BACK) -> list[date]:
    return [today - timedelta(days=d) for d in range(lag, back + 1)]

def iem_parts(stations: list, batch: int=verify.IEM_BATCH) -> list[list[str]]:
    names = list(dict.fromkeys((verify.iem_id(s[0]) for s in stations)))
    return [names[k:k + batch] for k in range(0, len(names), batch)]

def held_parts(lk, day: date) -> set[int]:
    out = set()
    for key in lk.list(f'{IEM_ROOT}/{day.isoformat()}/'):
        name = key.rsplit('/', 1)[-1]
        if name.startswith('part-') and name.endswith('.csv.gz') and name[5:-7].isdigit():
            out.add(int(name[5:-7]))
    return out

def reply_check(reply: bytes, names) -> str:
    if not reply.startswith(CSV_HEAD):
        return 'not a reply'
    first = reply.split(b'\n', 2)[1:2]
    if not first or not first[0]:
        return 'empty'
    if not reply.endswith(b'\n'):
        return 'cut'
    if first[0].split(b',', 1)[0].decode('utf-8', 'replace') not in set(names):
        return 'not a reply'
    return ''

def _gz(reply: bytes) -> bytes:
    return gzip.compress(reply, compresslevel=9, mtime=0)

def store_part(lk, key: str, reply: bytes, counts: dict) -> str:
    st = lk.put_plain(key, _gz(reply), strict=False).status
    if st == 'differs':
        try:
            st = 'same' if gzip.decompress(lk.get_plain(key)) == reply else 'differs'
        except (OSError, EOFError, zlib.error, lakemod.LakeError):
            st = 'differs'
    _n(counts, st)
    return st

def ingest_iem(lk, today: date, stations: list | None=None, opener=None, sleep=time.sleep, clock=time.monotonic, minutes: float=IEM_MINUTES, lag: int=IEM_LAG, back: int=IEM_BACK, types=IEM_TYPES) -> dict:
    counts: dict = {}
    parts = iem_parts(stations if stations is not None else verify.load_stations())
    t0, fails, lake_row, requests = (clock(), 0, 0, 0)
    for day in iem_days(today, lag, back):
        try:
            held = held_parts(lk, day)
        except lakemod.LakeError:
            _n(counts, 'lake failed')
            lake_row += 1
            if lake_row >= LAKE_STOP:
                counts['lake stopped'] = 1
                break
            continue
        lake_row = 0
        todo = [k for k in range(len(parts)) if k not in held]
        if not todo:
            _n(counts, 'days held')
            continue
        stop = ''
        for k in todo:
            if clock() - t0 > minutes * 60:
                stop = 'out of time'
                break
            if requests:
                sleep(verify.IEM_GAP_S)
            requests += 1
            when = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            try:
                reply = verify.iem_get(verify.iem_url(when, parts[k], types), when, opener=opener, sleep=sleep)
            except Exception:
                why = 'failed'
            else:
                why = reply_check(reply, parts[k])
            if why:
                _n(counts, why)
                fails += 1
                if fails >= FAIL_STOP:
                    stop = 'stopped'
                    break
                continue
            fails = 0
            _n(counts, 'fetched')
            _n(counts, 'bytes', len(reply))
            try:
                store_part(lk, iem_key(day, k), reply, counts)
                lake_row = 0
            except lakemod.LakeError:
                _n(counts, 'lake failed')
                lake_row += 1
                if lake_row >= LAKE_STOP:
                    stop = 'lake stopped'
                    break
        if stop:
            counts[stop] = 1
            break
    counts['requests'] = requests
    return counts

def ingest_raob(lk, today: date, igra=None, every: int=obs_raob.RAOB_EVERY, most: int=obs_raob.MAX_FETCH) -> dict:
    return obs_raob.ingest(lk, igra if igra is not None else obs_raob.Igra(budget=most), today, every, most)

def weglide_note(env=None) -> str:
    on = (os.environ if env is None else env).get('WEGLIDE_ON', '0')
    return 'weglide: parked (WEGLIDE_ON=0)' if on == '0' else 'weglide: parked (no ingest built for it yet): nothing fetched'

def run(lk, today: date, what: str='all', minutes: float=IEM_MINUTES, opener=None, sleep=time.sleep) -> dict:
    out = {}
    if what in ('all', 'iem'):
        out['iem'] = ingest_iem(lk, today, opener=opener, sleep=sleep, minutes=minutes)
    if what in ('all', 'raob'):
        out['raob'] = ingest_raob(lk, today)
    return out

def _line(counts: dict) -> str:
    return ', '.join((f'{k} {v}' for k, v in sorted(counts.items()))) or 'nothing to do'

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The lab's daily ingest: METARs and soundings into the lake")
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('run')
    a1.add_argument('--what', choices=('all', 'iem', 'raob'), default='all')
    a1.add_argument('--local', help='a folder as the lake (else the LAKE_S3_* secrets)')
    a1.add_argument('--day', help='today, YYYY-MM-DD (default: today, UTC)')
    a1.add_argument('--minutes', type=float, default=IEM_MINUTES, help='no METAR request started after this')
    a = ap.parse_args(argv)
    lk = lakemod.from_env('LAKE', a.local)
    if lk is None:
        print('lab ingest: no lake (LAKE_S3_* not set): nothing fetched', flush=True)
        return 0
    today = date.fromisoformat(a.day) if a.day else datetime.now(timezone.utc).date()
    out = run(lk, today, a.what, a.minutes)
    for k, counts in out.items():
        print(f'{k}: {_line(counts)}', flush=True)
    print(weglide_note(), flush=True)
    print(lk.summary(), flush=True)
    return 1 if any(('lake failed' in c or 'lake stopped' in c for c in out.values())) else 0
if __name__ == '__main__':
    sys.exit(main())
