#include "cxphys.h"
using namespace cx;
extern "C" {
void cx_freezing_level(int64_t n, int32_t L, const float* z, const float* t, const float* zs, const float* ztop,
                       float* out) {
    float o[BLOCK];
    for (int64_t i0 = 0; i0 < n; i0 += BLOCK) {
        const int64_t m = i0 + BLOCK < n ? BLOCK : n - i0;
        for (int64_t j = 0; j < m; ++j) o[j] = t[i0 + j] <= 0.0f ? zs[i0 + j] : NAN;
        for (int32_t k = 1; k < L; ++k) {
            const float *t0 = t + (k - 1) * n + i0, *t1 = t + k * n + i0;
            const float *z0 = z + (k - 1) * n + i0, *z1 = z + k * n + i0;
            for (int64_t j = 0; j < m; ++j) {
                if (std::isnan(o[j]) && t0[j] > 0.0f && t1[j] <= 0.0f) {
                    const float f = t0[j] / npmax(t0[j] - t1[j], 1e-6f);
                    o[j] = z0[j] + f * (z1[j] - z0[j]);
                }
            }
        }
        for (int64_t j = 0; j < m; ++j) out[i0 + j] = std::isnan(o[j]) ? ztop[i0 + j] : o[j];
    }
}
}
template <bool CLW>
static inline void icing_level(int64_t o, int64_t m, const float* z, const float* t, const float* rh,
                               const float* clw, const float* zs, float zmax, float* pm, float* lo, float* hi) {
    const float ice_min = 20.0f;
    for (int64_t j = 0; j < m; ++j) {
        const float tk = t[o + j];
        const float tf = npclip(-tk / 3.0f, 0.0f, 1.0f) * npclip((tk + 22.0f) / 10.0f, 0.0f, 1.0f);
        const float rhf = npclip((rh[o + j] - 75.0f) / 20.0f, 0.0f, 1.0f);
        const float moist = CLW ? npmax(npclip(nan_to_num(clw[o + j], 0.0f) * 1000.0f / 0.1f, 0.0f, 1.0f), 0.4f * rhf)
                                : 0.85f * rhf;
        const float zk = z[o + j];
        const bool in = (zk > zs[j] + 30.0f) && (zk <= zmax);
        const float pot = nan_to_num(in ? 100.0f * tf * moist : 0.0f, 0.0f);
        const bool on = pot >= ice_min;
        const float b = on ? zk : INFINITY, tp = on ? zk : -INFINITY;
        pm[j] = pot > pm[j] ? pot : pm[j];
        lo[j] = b < lo[j] ? b : lo[j];
        hi[j] = tp > hi[j] ? tp : hi[j];
    }
}
extern "C" {
void cx_icing_potential(int64_t n, int32_t L, const float* z, const float* t, const float* rh, const float* clw,
                        const float* zs, float zmax, float* pot_max, float* base, float* top) {
    float pm[BLOCK], lo[BLOCK], hi[BLOCK];
    for (int64_t i0 = 0; i0 < n; i0 += BLOCK) {
        const int64_t m = i0 + BLOCK < n ? BLOCK : n - i0;
        for (int64_t j = 0; j < m; ++j) {
            pm[j] = -INFINITY;
            lo[j] = INFINITY;
            hi[j] = -INFINITY;
        }
        for (int32_t k = 0; k < L; ++k) {
            if (clw)
                icing_level<true>(k * n + i0, m, z, t, rh, clw, zs + i0, zmax, pm, lo, hi);
            else
                icing_level<false>(k * n + i0, m, z, t, rh, clw, zs + i0, zmax, pm, lo, hi);
        }
        for (int64_t j = 0; j < m; ++j) {
            pot_max[i0 + j] = pm[j];
            base[i0 + j] = std::isfinite(lo[j]) ? lo[j] : NAN;
            top[i0 + j] = std::isfinite(hi[j]) ? hi[j] : NAN;
        }
    }
}
void cx_winds_at_heights(const Profile* p, int32_t H, const float* hs, float* u, float* v) {
    const int64_t n = p->n;
    for_each_cell(*p, COL_WIND, [&](const Column& c, int64_t i) {
        for (int32_t k = 0; k < H; ++k) c.uv_at(hs[k], u[k * n + i], v[k * n + i]);
    });
}
}
