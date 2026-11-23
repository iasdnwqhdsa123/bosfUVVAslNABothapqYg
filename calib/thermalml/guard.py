from __future__ import annotations
import os
from . import objects
SWITCH = 'COLLECTOR_TRAIN'
FEEDS = ('ogn', 'adsb')
STAGES = ('events', 'exposure', 'join')

class Refused(PermissionError):
    pass

def allowed(env=None) -> bool:
    return (os.environ if env is None else env).get(SWITCH) == '1'

def read_for_learning(lk, stage: str, version: str, sd: str, env=None):
    if not allowed(env):
        raise Refused(f"thermalml: the collector's records are not for training ({SWITCH} is not 1)")
    if stage not in STAGES:
        raise ValueError('thermalml: no such stage')
    k = objects.find(lk, stage, version, sd)
    if k is None:
        raise LookupError('thermalml: no such object')
    blob = lk.get(k)
    if stage == 'exposure':
        meta, cols = objects.unpack_columns(blob)
        feeds = set(meta.get('feeds') or [])
        if not feeds <= set(FEEDS) or (len(cols.get('s', [])) and int(cols['s'].max()) >= len(meta.get('feeds') or [])):
            raise Refused('thermalml: an exposure object holds a feed that is not admitted')
        return (meta, cols)
    obj = objects.loads(blob)
    seen = set(obj.get('feeds') or [])
    if stage == 'events':
        seen |= {e.get('s') for part in ('climbs', 'straight', 'failed') for e in obj.get(part) or []}
    if not seen <= set(FEEDS):
        raise Refused('thermalml: an object holds a feed that is not admitted')
    return obj
