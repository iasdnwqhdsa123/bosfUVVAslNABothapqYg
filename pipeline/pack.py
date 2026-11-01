import os
import shutil
import struct
import numpy as np
MAGIC = b'APK1'
HEAD = 12
HOURS_MAGIC = b'HRS1'
FOOT = 12

def hour_table(hours, n, nh):
    seg = np.zeros((n, nh + 1), np.int64)
    for i, h in enumerate(hours):
        if h is not None:
            if len(h) != nh + 1:
                raise ValueError(f'entry {i}: {len(h)} hour offsets for {nh} hours')
            seg[i, 0] = h[0]
            seg[i, 1:] = np.diff(h)
    return None if seg.max(initial=0) > 65535 else seg.astype('<u2')

def write_pack(path, blobs, ntx, nty, hours=None, nh=0):
    if len(blobs) != ntx * nty:
        raise ValueError(f'{len(blobs)} entries for a {ntx}×{nty} pack')
    ends = np.cumsum([len(b) for b in blobs], dtype=np.uint64) if blobs else np.zeros(0, np.uint64)
    if len(ends) and ends[-1] >= 2 ** 32:
        raise ValueError('pack over 4 GB')
    table = hour_table(hours, ntx * nty, nh) if hours is not None else None
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(MAGIC + struct.pack('<II', ntx, nty))
        f.write(ends.astype('<u4').tobytes())
        for b in blobs:
            f.write(b)
        if table is not None:
            f.write(table.tobytes())
            f.write(HOURS_MAGIC + struct.pack('<II', nh, ntx * nty))
    os.replace(tmp, path)
    return table is not None

def read_hours(path):
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size < HEAD + FOOT:
            return None
        f.seek(size - FOOT)
        foot = f.read(FOOT)
        if foot[:4] != HOURS_MAGIC:
            return None
        nh, n = struct.unpack('<II', foot[4:])
        f.seek(size - FOOT - 2 * n * (nh + 1))
        seg = np.frombuffer(f.read(2 * n * (nh + 1)), '<u2').reshape(n, nh + 1).astype(np.int64)
    return np.cumsum(seg, axis=1)

def read_hour(path, i, k):
    at = read_hours(path)
    _, _, ends, start = read_index(path)
    a = int(ends[i - 1]) if i else 0
    if at is None or at[i, k + 1] <= at[i, k]:
        return b''
    with open(path, 'rb') as f:
        f.seek(start + a + int(at[i, k]))
        return f.read(int(at[i, k + 1] - at[i, k]))

def read_index(path):
    with open(path, 'rb') as f:
        head = f.read(HEAD)
        if head[:4] != MAGIC:
            raise ValueError(f'{path}: not a tile pack')
        ntx, nty = struct.unpack('<II', head[4:])
        ends = np.frombuffer(f.read(4 * ntx * nty), '<u4').astype(np.int64)
    return (ntx, nty, ends, HEAD + 4 * ntx * nty)

def keys_in(path):
    ntx, _, ends, _ = read_index(path)
    sizes = np.diff(ends, prepend=0)
    return {f'{i % ntx}_{i // ntx}' for i in np.nonzero(sizes)[0]}

def read_entry(path, i):
    _, _, ends, start = read_index(path)
    a = int(ends[i - 1]) if i else 0
    with open(path, 'rb') as f:
        f.seek(start + a)
        return f.read(int(ends[i]) - a)

def read_tile(path, key):
    ntx, nty, _, _ = read_index(path)
    tx, ty = (int(v) for v in key.split('_'))
    return read_entry(path, ty * ntx + tx) if tx < ntx and ty < nty else b''

def pack_folder(folder, keys, keep=False):
    blobs = []
    for k in keys:
        p = os.path.join(folder, f'{k}.png')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                blobs.append(f.read())
        else:
            blobs.append(b'')
    write_pack(folder + '.pack', blobs, len(keys), 1)
    if not keep:
        shutil.rmtree(folder, ignore_errors=True)
