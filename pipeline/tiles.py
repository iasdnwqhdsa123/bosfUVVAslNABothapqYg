from __future__ import annotations
import ctypes
import io
import os
import struct
import zlib
import numpy as np
try:
    import deflate
except ImportError:
    deflate = None
PNG_LEVEL = 6
NPZ_LEVEL = 6
_U8 = ctypes.POINTER(ctypes.c_uint8)
_fns: dict = {}

def _cfn(name, argtypes):
    if name not in _fns:
        try:
            import physics_native
            _fns[name] = physics_native.cfunc(name, argtypes)
        except Exception:
            _fns[name] = None
    return _fns[name]

def _p(a):
    return a.ctypes.data_as(_U8)

def downsample_np(codes, k, none=None, rms=False):
    codes = np.asarray(codes, np.uint8)
    ny, nx = codes.shape[-2:]
    my, mx = (-(-ny // k), -(-nx // k))
    pad = np.full(codes.shape[:-2] + (my * k, mx * k), 255, np.uint8)
    pad[..., :ny, :nx] = codes
    total = np.zeros(codes.shape[:-2] + (my, mx), np.uint32)
    n = np.zeros(total.shape, np.uint16)
    m = np.zeros_like(n)
    for dy in range(k):
        for dx in range(k):
            c = pad[..., dy::k, dx::k]
            ok = c != 255
            n += ok
            if none is not None:
                ok &= c != none
            cc = c.astype(np.uint32)
            total += np.where(ok, cc * cc if rms else cc, 0)
            m += ok
    cells = np.outer(np.minimum(k, ny - np.arange(my) * k), np.minimum(k, nx - np.arange(mx) * k))
    if rms:
        mean = np.rint(np.sqrt(total / np.maximum(m, 1))).astype(np.uint16)
    else:
        mean = (2 * total + m) // np.maximum(2 * m, 1)
    if none is not None:
        mean = np.where(2 * (n - m) > n, none, mean)
    return np.where((n > 0) & (2 * n >= cells), mean, 255).astype(np.uint8)

def downsample_peak(codes, k):
    codes = np.asarray(codes, np.uint8)
    ny, nx = codes.shape[-2:]
    my, mx = (-(-ny // k), -(-nx // k))
    pad = np.full(codes.shape[:-2] + (my * k, mx * k), 255, np.uint8)
    pad[..., :ny, :nx] = codes
    top = np.zeros(codes.shape[:-2] + (my, mx), np.uint8)
    n = np.zeros(top.shape, np.uint16)
    for dy in range(k):
        for dx in range(k):
            c = pad[..., dy::k, dx::k]
            ok = c != 255
            n += ok
            top = np.maximum(top, np.where(ok, c, 0))
    cells = np.outer(np.minimum(k, ny - np.arange(my) * k), np.minimum(k, nx - np.arange(mx) * k))
    return np.where((n > 0) & (2 * n >= cells), top, 255).astype(np.uint8)

def downsample_at_peak(codes, by, k):
    codes, by = (np.asarray(codes, np.uint8), np.asarray(by, np.uint8))
    ny, nx = codes.shape[-2:]
    my, mx = (-(-ny // k), -(-nx // k))
    pc = np.full(codes.shape[:-2] + (my * k, mx * k), 255, np.uint8)
    pb = np.full(pc.shape, 255, np.uint8)
    pc[..., :ny, :nx] = codes
    pb[..., :ny, :nx] = by
    best = np.full(codes.shape[:-2] + (my, mx), -1, np.int16)
    out = np.full(best.shape, 255, np.uint8)
    for dy in range(k):
        for dx in range(k):
            b = pb[..., dy::k, dx::k].astype(np.int16)
            b[b == 255] = -1
            up = b > best
            best = np.where(up, b, best)
            out = np.where(up, pc[..., dy::k, dx::k], out)
    return np.where(downsample_peak(by, k) == 255, 255, out).astype(np.uint8)

def downsample(codes, k, nearest=False, none=None, rms=False):
    codes = np.asarray(codes, np.uint8)
    ny, nx = codes.shape[-2:]
    my, mx = (-(-ny // k), -(-nx // k))
    if nearest:
        iy = np.minimum(np.arange(my) * k + (k - 1) // 2, ny - 1)
        ix = np.minimum(np.arange(mx) * k + (k - 1) // 2, nx - 1)
        return codes[..., iy[:, None], ix[None, :]]
    f = _cfn('cx_downsample', [_U8, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, _U8])
    if f is None:
        return downsample_np(codes, k, none, rms)
    src = np.ascontiguousarray(codes)
    out = np.empty(codes.shape[:-2] + (my, mx), np.uint8)
    f(_p(src), src.size // (ny * nx) if ny * nx else 0, ny, nx, k, -1 if none is None else int(none), int(bool(rms)), _p(out))
    return out
_SIG = b'\x89PNG\r\n\x1a\n'

def png_filter_np(img):
    img = np.asarray(img, np.uint8)
    h, w = img.shape
    x = img.astype(np.int16)
    a = np.zeros_like(x)
    a[:, 1:] = x[:, :-1]
    b = np.zeros_like(x)
    b[1:] = x[:-1]
    c = np.zeros_like(x)
    c[1:, 1:] = x[:-1, :-1]
    p = a + b - c
    pa, pb, pc = (np.abs(p - a), np.abs(p - b), np.abs(p - c))
    pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    rows = np.stack([x, x - a, x - b, x - (a + b) // 2, x - pred]).astype(np.uint8)
    cost = np.abs(rows.view(np.int8).astype(np.int32)).sum(axis=2)
    best = np.argmin(cost, axis=0)
    out = np.empty((h, w + 1), np.uint8)
    out[:, 0] = best
    out[:, 1:] = rows[best, np.arange(h)]
    return out.ravel()

def png_filter(img):
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape
    f = _cfn('cx_png_filter', [_U8, ctypes.c_int64, ctypes.c_int64, _U8, _U8])
    if f is None:
        return png_filter_np(img)
    out = np.empty(h * (w + 1), np.uint8)
    f(_p(img), h, w, _p(out), _p(np.empty(5 * w, np.uint8)))
    return out

def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

def png_bytes(img, level=None):
    h, w = img.shape
    idat = deflate.zlib_compress(png_filter(img).tobytes(), PNG_LEVEL if level is None else level)
    return _SIG + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) + _chunk(b'IDAT', idat) + _chunk(b'IEND', b'')
HOUR_LEVEL = 6

def png_hours(stack, level=None):
    stack = np.ascontiguousarray(stack, np.uint8)
    n, T, W = stack.shape
    img = stack.reshape(-1, W)
    rows = png_filter(img).reshape(n * T, W + 1).copy()
    x = img[::T].astype(np.int16)
    a = np.zeros_like(x)
    a[:, 1:] = x[:, :-1]
    none, sub = (x.astype(np.uint8), (x - a).astype(np.uint8))
    cost = lambda r: np.abs(r.view(np.int8).astype(np.int32)).sum(axis=1)
    use_sub = cost(sub) < cost(none)
    first = rows[::T]
    first[:, 0] = use_sub
    first[:, 1:] = np.where(use_sub[:, None], sub, none)
    rows[::T] = first
    c = zlib.compressobj(HOUR_LEVEL if level is None else level, zlib.DEFLATED, 15, 9)
    head = c.compress(b'')
    segs = [c.compress(rows[k * T:(k + 1) * T].tobytes()) + c.flush(zlib.Z_FULL_FLUSH) for k in range(n)]
    stream = head + b''.join(segs) + c.flush(zlib.Z_FINISH)
    at = 8 + 25 + 8
    offs = [at + 2]
    s = at + len(head)
    for g in segs:
        s += len(g)
        offs.append(s)
    png = _SIG + _chunk(b'IHDR', struct.pack('>IIBBBBB', W, n * T, 8, 0, 0, 0, 0)) + _chunk(b'IDAT', stream) + _chunk(b'IEND', b'')
    return (png, offs)

def png_encode(img, fallback_level=4):
    if deflate is None:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img, mode='L').save(buf, format='PNG', compress_level=fallback_level)
        return buf.getvalue()
    return png_bytes(img)

def write_png(path, img, fallback_level=4):
    if deflate is None:
        from PIL import Image
        Image.fromarray(img, mode='L').save(path, compress_level=fallback_level)
        return
    with open(path, 'wb') as f:
        f.write(png_bytes(img))

def savez(path, arrays, level=None):
    if deflate is None:
        np.savez_compressed(path, **arrays)
        return
    out, central = (io.BytesIO(), io.BytesIO())
    for name, a in arrays.items():
        buf = io.BytesIO()
        np.lib.format.write_array(buf, np.asanyarray(a), allow_pickle=False)
        raw = buf.getvalue()
        data, crc, fname = (deflate.deflate_compress(raw, NPZ_LEVEL if level is None else level), zlib.crc32(raw), f'{name}.npy'.encode())
        head = struct.pack('<HHHHHIIIHH', 20, 0, 8, 0, 33, crc, len(data), len(raw), len(fname), 0)
        central.write(b'PK\x01\x02' + struct.pack('<H', 20) + head + struct.pack('<HHHII', 0, 0, 0, 0, out.tell()) + fname)
        out.write(b'PK\x03\x04' + head + fname + data)
    n, cd = (len(arrays), central.getvalue())
    end = b'PK\x05\x06' + struct.pack('<HHHHIIH', 0, 0, n, n, len(cd), out.tell(), 0)
    with open(path, 'wb') as f:
        f.write(out.getvalue() + cd + end)
