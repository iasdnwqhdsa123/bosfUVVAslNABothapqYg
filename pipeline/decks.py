from __future__ import annotations
import math
import numpy as np
import physics as P
ROLLS = {'kappa': 0.4, 'z0_land': 0.1, 'z0_water': 0.0002, 'zL': (1.0, 3.0, 6.0, 60.0), 'shear': (0.4, 0.004), 'turnDeg': (30.0, 60.0), 'aspect': (2.5, 5.5), 'aspectZL': (2.0, 25.0), 'turn': 15.0, 'turnShear': 3.0}
DECK_LOW = 25.0
DECK_CC = 20.0
DECK_THIN = 200.0
DECK_TOP_MAX = 3000.0
INV_DZ, INV_MIN = (200.0, 3.0)
DECK_WIND = (1.5, 4.0)
W_MIN = (0.15, 0.4)
CELL_ASPECT = 12.0
TOP_SHARE = (0.25, 0.5)
DECK_SHOWN = (60.0, 75.0)
TOP = {'lw': 70.0, 'mid': 0.8, 'high': 0.4, 'sw': 60.0, 'eff': 0.4}
SEA = {'ch': 0.0012, 'ce': 0.0012, 'u_min': 2.0}

def smooth(x):
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def sea_flux(sst, t2, td2, u10, ps):
    tk = t2 + P.K0
    rho = ps * 100.0 / (P.RD * tk)
    u = np.maximum(np.nan_to_num(u10, nan=0.0), SEA['u_min'])
    h = rho * P.CP * SEA['ch'] * u * (sst - t2)
    qs = 0.98 * P.mixing_ratio(sst, ps)
    qa = P.mixing_ratio(np.minimum(td2, t2), ps)
    le = rho * P.LV * SEA['ce'] * u * (qs - qa)
    return np.maximum(0.0, h + 0.61 * tk * P.CP / P.LV * le)

def deck_layer(zs, cc, zc, low, ceil, lowbase, t2, td2):
    lcl = 125.0 * np.maximum(0.0, t2 - td2)
    base = np.where(np.isfinite(ceil) & (ceil < DECK_TOP_MAX), ceil, np.where(np.isfinite(lowbase), lowbase, lcl))
    deck = np.nan_to_num(low, nan=0.0) >= DECK_LOW
    first = np.full(zs.shape, np.nan, np.float32)
    top = np.full(zs.shape, np.nan, np.float32)
    prev_z = np.full(zs.shape, np.nan, np.float32)
    prev_c = np.full(zs.shape, np.nan, np.float32)
    for k in range(cc.shape[0]):
        c, z = (cc[k], zc[k] - zs)
        ok = np.isfinite(c) & np.isfinite(z) & (z > 0)
        cloudy = ok & (c >= DECK_CC)
        start = cloudy & np.isnan(first) & (z <= DECK_TOP_MAX)
        first = np.where(start, z, first)
        end = ok & ~cloudy & np.isfinite(first) & np.isnan(top)
        f = np.clip((prev_c - DECK_CC) / np.maximum(prev_c - c, 0.001), 0, 1)
        top = np.where(end, prev_z + f * (z - prev_z), top)
        prev_z, prev_c = (np.where(ok, z, prev_z), np.where(ok, c, prev_c))
    has = np.isfinite(first)
    base = np.where(has, np.minimum(base, first), base)
    top = np.where(has, np.where(np.isfinite(top), top, first), base + DECK_THIN)
    top = np.clip(np.maximum(top, base + 50.0), 0, DECK_TOP_MAX)
    return (base, top)

def inversion_top(prof, zs, above, zmax=DECK_TOP_MAX, dz=100.0):
    hs = np.arange(dz, zmax + dz, dz, dtype=np.float32)
    th = np.stack([P.theta(*prof.at(zs + h, 't', 'lnp')) for h in hs])
    k = max(1, int(round(INV_DZ / dz)))
    grad = (th[k:] - th[:-k]) / (INV_DZ / 1000.0)
    mid = (hs[k:] + hs[:-k]) / 2
    ok = mid[:, None, None] >= above[None] + 100.0
    g = np.where(ok, grad, -np.inf)
    j = np.argmax(g, axis=0)
    best = np.take_along_axis(g, j[None], 0)[0]
    return np.where(best >= INV_MIN, mid[j], np.nan).astype(np.float32)

def deck_bounds(prof, zs, cc, zc, low, ceil, lowbase, t2, td2):
    base, top = deck_layer(zs, cc, zc, low, ceil, lowbase, t2, td2)
    deck = np.nan_to_num(low, nan=0.0) >= DECK_LOW
    inv = inversion_top(prof, zs, np.nan_to_num(base, nan=0.0))
    top = np.where(np.isfinite(inv), inv, top)
    top = np.clip(np.maximum(top, base + DECK_THIN), 0, DECK_TOP_MAX)
    return (np.where(deck, base, np.nan).astype(np.float32), np.where(deck, top, np.nan).astype(np.float32))

def psi_m(zeta):
    x = np.sqrt(np.sqrt(np.maximum(1 - 16 * np.minimum(zeta, 0.0), 1.0)))
    out = 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2) - 2 * np.arctan(x) + math.pi / 2
    return np.where(zeta < 0, out, 0.0)

def instability(wstar, u10, zi, z0):
    k = ROLLS['kappa']
    lz = np.log(10.0 / np.maximum(1e-05, z0))
    u10 = np.maximum(0.3, u10)
    us = k * u10 / lz
    for _ in range(4):
        zl = k * (wstar / us) ** 3
        L = -np.maximum(zi, 1.0) / np.maximum(zl, 1e-09)
        us = k * u10 / np.maximum(0.5, lz - psi_m(10.0 / L))
    return (k * (wstar / us) ** 3, us)

def organisation(prof, zs, t2, td2, ps, u10, v10, sst, heat_th, zi, cc, zc, low, mid, high, ceil, lowbase, sun_el):
    dbase, dtop = deck_bounds(prof, zs, cc, zc, low, ceil, lowbase, t2, td2)
    deck = np.isfinite(dtop)
    water = np.isfinite(sst)
    tk = t2 + P.K0
    rho_cp = ps * 100.0 / (P.RD * tk) * P.CP
    gt = P.G / tk
    D = np.where(deck, dtop, np.maximum(np.nan_to_num(zi, nan=0.0), 300.0))
    hv = np.where(water, sea_flux(np.nan_to_num(sst, nan=0.0), t2, td2, np.hypot(u10, v10), ps), np.maximum(np.nan_to_num(heat_th, nan=0.0), 0.0))
    w_sfc3 = gt * hv / rho_cp * D
    m, h = (np.clip(np.nan_to_num(a, nan=0.0) / 100.0, 0, 1) for a in (mid, high))
    shield = 1 - (1 - TOP['mid'] * m) * (1 - TOP['high'] * h)
    sw = TOP['sw'] * np.maximum(0.0, np.sin(np.radians(sun_el)))
    f_top = np.maximum(0.0, TOP['lw'] * (1 - shield) - sw)
    w_top3 = np.where(deck, TOP['eff'] * gt * f_top / rho_cp * D, 0.0)
    w = np.cbrt(w_sfc3 + w_top3)
    ub, vb = P.layer_mean(prof, zs + 10.0, zs + np.maximum(D, 60.0), ['u', 'v'], n=6)
    ut, vt = prof.at(zs + D, 'u', 'v')
    V = np.hypot(ub, vb)
    U10 = np.hypot(u10, v10)
    z0 = np.where(water, ROLLS['z0_water'], ROLLS['z0_land'])
    zl, us = instability(w, U10, D, z0)
    us = np.maximum(us, math.sqrt(P.CD_BL) * V)
    zl = ROLLS['kappa'] * (w / np.maximum(us, 0.001)) ** 3
    lz = np.log(np.maximum(zl, 1e-06))
    a0, a1, b0, b1 = (math.log(x) for x in ROLLS['zL'])
    f_in = smooth((lz - a0) / (a1 - a0))
    past = smooth((lz - b0) / (b1 - b0))
    f_wind = smooth((V - DECK_WIND[0]) / (DECK_WIND[1] - DECK_WIND[0]))
    sx, sy = (ut - u10, vt - v10)
    f_shear = ROLLS['shear'][0] + (1 - ROLLS['shear'][0]) * smooth(np.hypot(sx, sy) / np.maximum(D, 1.0) / ROLLS['shear'][1])
    a, b = (U10, np.hypot(ut, vt))
    with np.errstate(invalid='ignore', divide='ignore'):
        turn = np.degrees(np.arccos(np.clip((u10 * ut + v10 * vt) / np.maximum(a * b, 1e-06), -1, 1)))
    f_turn = 1 - smooth((turn - ROLLS['turnDeg'][0]) / (ROLLS['turnDeg'][1] - ROLLS['turnDeg'][0])) * smooth((np.minimum(a, b) - 1) / 2)
    f_w = smooth((w - W_MIN[0]) / (W_MIN[1] - W_MIN[0]))
    shown = 1 - smooth((np.nan_to_num(low, nan=0.0) - DECK_SHOWN[0]) / (DECK_SHOWN[1] - DECK_SHOWN[0]))
    share = w_top3 / np.maximum(w_sfc3 + w_top3, 1e-09)
    shown = shown * smooth((share - TOP_SHARE[0]) / (TOP_SHARE[1] - TOP_SHARE[0]))
    rows = f_in * (1 - past) * f_wind * f_shear * f_turn * f_w * shown
    cells = f_in * past * f_w
    rowstr = np.where(water, np.nan, np.where(rows >= cells, rows, 0.0))
    ang = np.arctan2(ub, vb)
    sang = np.arctan2(sx, sy)
    d = (sang - ang + math.pi) % (2 * math.pi) - math.pi
    d = np.where(d > math.pi / 2, d - math.pi, np.where(d < -math.pi / 2, d + math.pi, d))
    turn_by = np.clip(d, -math.radians(ROLLS['turn']), math.radians(ROLLS['turn'])) * np.clip(np.hypot(sx, sy) / ROLLS['turnShear'], 0, 1)
    rowdir = np.degrees(ang + turn_by) % 180
    t = np.clip((lz - math.log(ROLLS['aspectZL'][0])) / (math.log(ROLLS['aspectZL'][1]) - math.log(ROLLS['aspectZL'][0])), 0, 1)
    aspect = ROLLS['aspect'][0] + (ROLLS['aspect'][1] - ROLLS['aspect'][0]) * t
    rowgap = np.where(rows >= cells, aspect, CELL_ASPECT) * D / 1000.0
    nan = lambda x: np.where(deck, x, np.nan).astype(np.float32)
    return {'rowstr': nan(rowstr), 'rowdir': nan(rowdir), 'rowgap': nan(rowgap), 'wdeck': nan(w), 'dtop': nan(dtop)}
