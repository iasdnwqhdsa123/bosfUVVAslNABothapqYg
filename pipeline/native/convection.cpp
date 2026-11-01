#include "cxphys.h"
#include <cstring>
using namespace cx;
namespace {
constexpr float f(double x) { return static_cast<float>(x); }
inline float numpy_expf(float x) {
    if (std::isnan(x)) return NAN;
    if (x >= 88.72283935546875f) return HUGE_VALF;
    if (x <= -103.97208404541015625f) return 0.0f;
    const float magic = 0x1.8p+23f;
    float k = x * 1.442695040888963407359924681001892137f;
    k = k + magic;
    k = k - magic;
    float y = std::fma(k, -6.93145752e-1f, x);
    y = std::fma(k, -1.42860677e-6f, y);
    y = std::fma(k, 0.0f, y);
    float num = std::fma(5.082762527590693718096e-04f, y, 6.757896990527504603057e-03f);
    num = std::fma(num, y, 5.114512081637298353406e-02f);
    num = std::fma(num, y, 2.473615434895520810817e-01f);
    num = std::fma(num, y, 7.257664613233124478488e-01f);
    num = std::fma(num, y, 9.999999999980870924916e-01f);
    float den = std::fma(2.159509375685829852307e-02f, y, -2.742335390411667452936e-01f);
    den = std::fma(den, y, 1.0f);
    const float r = num / den;
    const int32_t e = static_cast<int32_t>(k);
    if (e < -125 || e > 127) return std::ldexp(r, e);
    const uint32_t bits = static_cast<uint32_t>(e + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof scale);
    return r * scale;
}
inline float np_expf(float x, int mode) { return mode == 0 ? std::exp(x) : numpy_expf(x); }
inline float moist_lapse(float t, float p, int mode) {
    const float tk = t + f(273.15);
    const float e = f(6.112) * np_expf(f(17.67) * t / (t + f(243.5)), mode);
    const float rs = f(0.622) * e / npmax(p - e, 1.0f);
    const float rt = f(287.05) * tk;
    return f(9.80665) * (1.0f + f(2.5e6) * rs / rt) / (f(1004.0) + f(2.5e6 * 2.5e6) * rs * f(0.622) / (rt * tk));
}
inline float cloud_mix(float tp, float te, float tde, float p, float m, int mode) {
    const float k = f(2.5e6 / 1004.0);
    auto rsat = [&](float t) {
        const float e = f(6.112) * np_expf(f(17.67) * t / (t + f(243.5)), mode);
        return f(0.622) * e / npmax(p - e, 1.0f);
    };
    const float hp = tp + k * rsat(tp);
    const float he = te + k * rsat(tde);
    const float h = hp + m * (he - hp);
    float t = tp;
    for (int it = 0; it < 2; ++it) {
        const float e = f(6.112) * np_expf(f(17.67) * t / (t + f(243.5)), mode);
        const float d = npmax(p - e, 1.0f);
        const float rs = f(0.622) * e / d;
        const float u = t + f(243.5);
        t = t - (t + k * rs - h) / (1.0f + k * rs * p / d * f(4302.645) / (u * u));
    }
    return t;
}
inline bool ascending(const float* z, int32_t L, int64_t n, int64_t i) {
    bool up = true;
    for (int32_t k = 1; k < L; ++k) up &= z[(k - 1) * n + i] <= z[k * n + i];
    return up;
}
}
extern "C" {
#define CX_FMA_CLONES __attribute__((target_clones("fma", "default")))
CX_FMA_CLONES void cx_np_exp(int64_t n, const float* x, int32_t mode, float* out) {
    for (int64_t i = 0; i < n; ++i) out[i] = np_expf(x[i], mode);
}
CX_FMA_CLONES void cx_cumulus_top(const Profile* p, const float* zbase, const float* tbase, const uint8_t* active,
                                  float step, int32_t steps, int32_t exp_mode, const float* mix, float* out) {
    const int64_t n = p->n;
    const int32_t L = p->L;
    const float* ztop = p->z + static_cast<int64_t>(L - 1) * n;
    constexpr int G = 8;
    int64_t cell[G];
    float lz[G], ltp[G];
    int32_t lk[G], lc[G];
    bool up[G], on[G];
    int64_t next = 0;
    auto start = [&](int g) {
        while (next < n) {
            const int64_t i = next++;
            out[i] = zbase[i];
            if (!active[i] || std::isnan(zbase[i])) continue;
            cell[g] = i;
            lz[g] = zbase[i];
            ltp[g] = tbase[i] + 0.5f;
            lk[g] = 0;
            lc[g] = 0;
            up[g] = ascending(p->z, L, n, i);
            return true;
        }
        return false;
    };
    int live = 0;
    for (int g = 0; g < G; ++g) live += (on[g] = start(g));
    while (live) {
        for (int g = 0; g < G; ++g) {
            if (!on[g]) continue;
            const int64_t i = cell[g];
            const float z = lz[g], tp = ltp[g];
            bool stop = lk[g] >= steps;
            if (!stop) {
                Bracket b;
                if (up[g]) {
                    int32_t c = lc[g];
                    while (c < L && p->z[c * n + i] < z) ++c;
                    lc[g] = c;
                    b = bracket_from_count(p->z, L, n, i, z, c);
                } else {
                    b = bracket(p->z, L, n, i, z);
                }
                const float te = interp(p->t, b);
                stop = tp < te || z >= ztop[i];
                if (!stop) {
                    const float pz = np_expf(interp(p->lnp, b), exp_mode);
                    float t2 = tp - moist_lapse(tp, pz, exp_mode) * step;
                    if (mix) t2 = cloud_mix(t2, te, interp(p->td, b), pz, mix[i], exp_mode);
                    ltp[g] = t2;
                    lz[g] = z + step;
                    ++lk[g];
                }
            }
            if (stop) {
                out[i] = z;
                if (!(on[g] = start(g))) --live;
            }
        }
    }
}
__attribute__((target_clones("avx2", "default"))) void cx_layer_mean(const Profile* p, const float* zb, const float* zt, int32_t ns, int32_t count,
                   const int32_t* which, float* const* outs) {
    const float* cols[4] = {p->t, p->td, p->u, p->v};
    bool need_t = false, need_w = false;
    for (int32_t a = 0; a < count; ++a) (which[a] >= 2 ? need_w : need_t) = true;
    const int64_t n = p->n;
    const float nf = static_cast<float>(ns);
    uint8_t ct[BLOCK], cw[BLOCK];
    float zq[BLOCK];
    for (int64_t i0 = 0; i0 < n; i0 += BLOCK) {
        const int64_t m = i0 + BLOCK < n ? BLOCK : n - i0;
        for (int32_t a = 0; a < count; ++a)
            for (int64_t j = 0; j < m; ++j) outs[a][i0 + j] = 0.0f;
        for (int32_t s = 0; s < ns; ++s) {
            const float frac = static_cast<float>(static_cast<double>(s) / (ns - 1));
            for (int64_t j = 0; j < m; ++j) zq[j] = zb[i0 + j] + (zt[i0 + j] - zb[i0 + j]) * frac;
            if (need_t) count_below(p->z, p->L, n, i0, m, zq, ct);
            if (need_w) count_below(p->zw, p->Lw, n, i0, m, zq, cw);
            for (int64_t j = 0; j < m; ++j) {
                const int64_t i = i0 + j;
                Bracket bt{0, 0, 0.0f}, bw{0, 0, 0.0f};
                if (need_t) bt = bracket_from_count(p->z, p->L, n, i, zq[j], ct[j]);
                if (need_w) bw = bracket_from_count(p->zw, p->Lw, n, i, zq[j], cw[j]);
                for (int32_t a = 0; a < count; ++a) outs[a][i] += interp(cols[which[a]], which[a] >= 2 ? bw : bt);
            }
        }
        for (int32_t a = 0; a < count; ++a)
            for (int64_t j = 0; j < m; ++j) outs[a][i0 + j] = outs[a][i0 + j] / nf;
    }
}
}
