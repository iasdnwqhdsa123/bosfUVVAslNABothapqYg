#include "cxphys.h"
#include <cstring>
using namespace cx;
namespace {
constexpr float f(double x) { return static_cast<float>(x); }
#define CX_INLINE __attribute__((always_inline)) inline
CX_INLINE float numpy_expf(float x) {
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
CX_INLINE float pow2i(int32_t e) {
    const uint32_t bits = static_cast<uint32_t>(e + 127) << 23;
    float s;
    std::memcpy(&s, &bits, sizeof s);
    return s;
}
CX_INLINE float numpy_expf_lanes(float x) {
    const bool nan = x != x, hi = x >= 88.72283935546875f, lo = x <= -103.97208404541015625f;
    x = (nan | hi | lo) ? 0.0f : x;
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
    const int32_t e = static_cast<int32_t>(k), e1 = e >> 1;
    const float v = r * pow2i(e1) * pow2i(e - e1);
    return nan ? NAN : hi ? HUGE_VALF : lo ? 0.0f : v;
}
CX_INLINE float np_expf(float x, int mode) {
    return mode == 0 ? std::exp(x) : mode == 1 ? numpy_expf(x) : numpy_expf_lanes(x);
}
CX_INLINE float moist_lapse(float t, float p, int mode) {
    const float tk = t + f(273.15);
    const float e = f(6.112) * np_expf(f(17.67) * t / (t + f(243.5)), mode);
    const float rs = f(0.622) * e / npmax(p - e, 1.0f);
    const float rt = f(287.05) * tk;
    return f(9.80665) * (1.0f + f(2.5e6) * rs / rt) / (f(1004.0) + f(2.5e6 * 2.5e6) * rs * f(0.622) / (rt * tk));
}
CX_INLINE float mixing_ratio(float td, float p, int mode) {
    const float e = f(6.112) * np_expf(f(17.67) * td / (td + f(243.5)), mode);
    return f(0.622) * e / npmax(p - e, 1.0f);
}
CX_INLINE float cloud_mix_re(float tp, float te, float re, float p, float m, int mode) {
    const float k = f(2.5e6 / 1004.0);
    const float hp = tp + k * mixing_ratio(tp, p, mode);
    const float he = te + k * re;
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
CX_INLINE float cloud_mix(float tp, float te, float tde, float p, float m, int mode) {
    return cloud_mix_re(tp, te, mixing_ratio(tde, p, mode), p, m, mode);
}
inline bool ascending(const float* z, int32_t L, int64_t n, int64_t i) {
    bool up = true;
    for (int32_t k = 1; k < L; ++k) up &= z[(k - 1) * n + i] <= z[k * n + i];
    return up;
}
constexpr int32_t CORE_MAX_AT = 8;
constexpr int CORE_LANES = 64;
template <int MODE>
__attribute__((always_inline)) inline void core_run(const Profile* p, const uint8_t* active, const float* dz,
                                                    const float* half, const float* th0, int32_t k0, int32_t steps,
                                                    double dx, double ref_lo, double ref_hi, int32_t nat,
                                                    const double* at, const float* kc, float* const* outs, float* top) {
    constexpr int B = CORE_LANES;
    const int64_t n = p->n;
    const int32_t L = p->L;
    const float kappa = kc[0], ln1000 = kc[1], excess = kc[2], ceps = kc[3], drag2 = kc[4], cmax = kc[5];
    const float K0 = f(273.15), G = f(9.80665), TV = f(0.61), MIXMAX = f(0.9);
    int64_t cell[B];
    int32_t below[B], up[B];
    alignas(64) int32_t moves[B];
    alignas(64) float zlast[B], znext[B];
    alignas(64) float z0b[B], z1b[B], t0b[B], t1b[B], l0b[B], l1b[B], d0b[B], d1b[B];
    alignas(64) float zg[B], dzb[B], reach[B];
    alignas(64) float th[B], rp[B], tp[B], w2[B], ref[B], stop[B], wv[B];
    alignas(64) int32_t sat[B], dead[B];
    alignas(64) float zq[B], te[B], lzb[B], pz[B], mixb[B], epsb[B], reb[B], tdry[B], rvb[B], tmb[B];
    alignas(64) float acc[CORE_MAX_AT][B], cnt[CORE_MAX_AT][B];
    auto find = [&](int j, float z) {
        const int64_t i = cell[j];
        int32_t c = below[j];
        if (up[j] && z >= zlast[j]) {
            while (c < L && p->z[c * n + i] < z) ++c;
        } else {
            c = 0;
            for (int32_t k = 0; k < L; ++k) c += p->z[k * n + i] < z;
        }
        below[j] = c;
        zlast[j] = z;
        znext[j] = !up[j] ? -INFINITY : c < L ? p->z[c * n + i] : INFINITY;
        const int32_t h = c < 1 ? 1 : (c > L - 1 ? L - 1 : c);
        const int64_t hi = h * n + i, lo = hi - n;
        z0b[j] = p->z[lo], z1b[j] = p->z[hi], t0b[j] = p->t[lo], t1b[j] = p->t[hi];
        l0b[j] = p->lnp[lo], l1b[j] = p->lnp[hi], d0b[j] = p->td[lo], d1b[j] = p->td[hi];
    };
    int64_t next = 0;
    for (;;) {
        int m = 0;
        while (m < B && next < n) {
            const int64_t i = next++;
            if (active[i]) {
                cell[m++] = i;
            } else {
                for (int32_t a = 0; a < nat; ++a) outs[a][i] = NAN;
                if (top) top[i] = NAN;
            }
        }
        if (m == 0) break;
        for (int j = m; j < B; ++j) cell[j] = cell[m - 1];
        for (int j = 0; j < B; ++j) {
            const int64_t i = cell[j];
            up[j] = ascending(p->z, L, n, i);
            below[j] = 0, zlast[j] = -INFINITY, znext[j] = -INFINITY;
            zg[j] = p->z[i], dzb[j] = dz[i], reach[j] = half[i] + f(1e-6);
            float lp = p->lnp[i];
            rp[j] = mixing_ratio(p->td[i], np_expf(lp, MODE), MODE);
            if (th0) {
                th[j] = th0[i];
                const float z = zg[j] + dzb[j] * static_cast<float>(k0);
                find(j, z);
                const float w = npclip((z - z0b[j]) / npmax(z1b[j] - z0b[j], 1e-3f), 0.0f, 1.0f);
                lp = l0b[j] + w * (l1b[j] - l0b[j]);
            } else {
                th[j] = (p->t[i] + K0) / np_expf(kappa * (lp - ln1000), MODE) + excess;
            }
            tp[j] = th[j] * np_expf(kappa * (lp - ln1000), MODE) - K0;
            sat[j] = 0, dead[j] = 0;
            w2[j] = 0.0f, ref[j] = 0.0f, stop[j] = NAN, wv[j] = 0.0f;
            for (int32_t a = 0; a < nat; ++a) acc[a][j] = 0.0f, cnt[a][j] = 0.0f;
        }
        bool live = true;
        for (int32_t k = k0 + 1; k <= steps; ++k) {
            const double x = static_cast<double>(k) * dx;
            const float kf = static_cast<float>(k), xf = static_cast<float>(x);
            if (live) {
                for (int j = 0; j < B; ++j) {
                    const float z = zg[j] + dzb[j] * kf;
                    zq[j] = z;
                    moves[j] = (dead[j] == 0) & (!(z >= zlast[j]) | (znext[j] < z));
                }
                uint64_t mask = 0;
                for (int j = 0; j < B; ++j) mask |= static_cast<uint64_t>(moves[j]) << j;
                for (; mask; mask &= mask - 1) {
                    const int j = __builtin_ctzll(mask);
                    find(j, zq[j]);
                }
                int32_t dry = 0;
                for (int j = 0; j < B; ++j) {
                    const float dzj = dzb[j];
                    const float eps = ceps / (dzj * kf);
                    const float w = npclip((zq[j] - z0b[j]) / npmax(z1b[j] - z0b[j], 1e-3f), 0.0f, 1.0f);
                    const float t_de = d0b[j] + w * (d1b[j] - d0b[j]);
                    const float lz = l0b[j] + w * (l1b[j] - l0b[j]);
                    const float pr = np_expf(lz, MODE);
                    te[j] = t0b[j] + w * (t1b[j] - t0b[j]), lzb[j] = lz, pz[j] = pr;
                    reb[j] = mixing_ratio(t_de, pr, MODE), epsb[j] = eps, mixb[j] = npmin(eps * dzj, MIXMAX);
                    dry |= !sat[j] & !dead[j];
                }
                int32_t wet = !dry;
                if (dry) {
                    for (int j = 0; j < B; ++j) {
                        const float ex = np_expf(kappa * (lzb[j] - ln1000), MODE), mix = mixb[j];
                        const float th1 = th[j] + mix * ((te[j] + K0) / ex - th[j]);
                        const float rpm = rp[j] + mix * (reb[j] - rp[j]);
                        const float rp1 = sat[j] ? rp[j] : rpm;
                        const float td1 = th1 * ex - K0;
                        const int32_t s1 = sat[j] | (rp1 >= mixing_ratio(td1, pz[j], MODE));
                        th[j] = th1, rp[j] = rp1, sat[j] = s1, tdry[j] = td1;
                        wet |= s1 & !dead[j];
                    }
                }
                if (wet) {
                    for (int j = 0; j < B; ++j) {
                        const float t0 = tp[j];
                        tmb[j] = cloud_mix_re(t0 - moist_lapse(t0, pz[j], MODE) * dzb[j], te[j], reb[j], pz[j], mixb[j],
                                              MODE);
                    }
                    for (int j = 0; j < B; ++j) tp[j] = sat[j] ? tmb[j] : tdry[j];
                    for (int j = 0; j < B; ++j) tmb[j] = mixing_ratio(tp[j], pz[j], MODE);
                    for (int j = 0; j < B; ++j) rvb[j] = sat[j] ? tmb[j] : rp[j];
                } else {
                    for (int j = 0; j < B; ++j) tp[j] = tdry[j], rvb[j] = rp[j];
                }
                int32_t rising = 0;
                for (int j = 0; j < B; ++j) {
                    const float tve = (te[j] + K0) * (1.0f + TV * reb[j]);
                    const float buoy = G * ((tp[j] + K0) * (1.0f + TV * rvb[j]) - tve) / tve;
                    const float w2a = w2[j] + dzb[j] * (2.0f * buoy - drag2 * epsb[j] * w2[j]);
                    const int32_t out = w2a <= 0.0f;
                    stop[j] = (out & !dead[j]) ? xf : stop[j];
                    dead[j] |= out;
                    w2[j] = dead[j] ? 0.0f : w2a;
                    wv[j] = std::sqrt(w2[j]);
                    rising |= !dead[j];
                }
                live = rising;
            }
            if (ref_lo <= x && x <= ref_hi)
                for (int j = 0; j < B; ++j) ref[j] = npmax(ref[j], wv[j]);
            for (int32_t a = 0; a < nat; ++a) {
                const double d = std::fabs(x - at[a]);
                float dc = static_cast<float>(d);
                if (static_cast<double>(dc) < d) dc = std::nextafter(dc, INFINITY);
                for (int j = 0; j < B; ++j) {
                    const bool near = reach[j] >= dc;
                    acc[a][j] = acc[a][j] + (near ? wv[j] : 0.0f);
                    cnt[a][j] = cnt[a][j] + (near ? 1.0f : 0.0f);
                }
            }
        }
        for (int j = 0; j < m; ++j) {
            const int64_t i = cell[j];
            const bool ok = ref[j] > f(0.05);
            for (int32_t a = 0; a < nat; ++a)
                outs[a][i] = ok ? npclip(acc[a][j] / npmax(cnt[a][j], 1.0f) / npmax(ref[j], f(1e-6)), 0.0f, cmax) : NAN;
            if (top) top[i] = ok ? stop[j] : NAN;
        }
    }
}
}
extern "C" {
#define CX_FMA_CLONES __attribute__((target_clones("fma", "default")))
#define CX_CORE_CLONES __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
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
CX_CORE_CLONES void cx_core_shape(const Profile* p, const uint8_t* active, const float* dz, const float* half,
                                  const float* th0, int32_t k0, int32_t steps, double dx, double ref_lo, double ref_hi,
                                  int32_t nat, const double* at, const float* kc, int32_t exp_mode, float* const* outs,
                                  float* top) {
    if (exp_mode == 0)
        core_run<0>(p, active, dz, half, th0, k0, steps, dx, ref_lo, ref_hi, nat, at, kc, outs, top);
    else
        core_run<2>(p, active, dz, half, th0, k0, steps, dx, ref_lo, ref_hi, nat, at, kc, outs, top);
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
