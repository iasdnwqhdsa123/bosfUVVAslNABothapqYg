#include <limits>
#include <new>
#include <type_traits>
#include "cxphys.h"
using namespace cx;
namespace {
constexpr float f(double x) { return static_cast<float>(x); }
constexpr float LO = 300.0f, TOP_MIN = 1500.0f, TOP_MAX = 3000.0f;
constexpr int32_t MAXN = 256;
constexpr int32_t MAXW = 64;
constexpr int64_t B = 512;
template <class T>
inline T vmax(T a, T b) { return (a > b || std::isnan(a)) ? a : b; }
template <class T>
inline T vmin(T a, T b) { return (a < b || std::isnan(a)) ? a : b; }
template <class T>
inline T vclip(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }
template <class T>
inline T nan0(T x) {
    if (std::isnan(x)) return T(0);
    if (std::isinf(x)) return x > 0 ? std::numeric_limits<T>::max() : -std::numeric_limits<T>::max();
    return x;
}
inline void steps(int32_t n, float* fi) {
    for (int32_t i = 0; i < n; ++i) fi[i] = f(static_cast<double>(i) / static_cast<double>(n - 1));
}
template <class TB>
inline bool thermals(TB ws, float d) { return nan0(ws) > TB(0) && d > LO; }
inline float band_x(float d, float zi, float fi) {
    const float h = LO + (vmax(d, LO) - LO) * fi;
    return vmax(h / zi, 0.0f);
}
struct Scratch {
    float zc[MAXW * B], wc[MAXW * B];
    float base[B], span[B], top[B], zq[B], wsum[B], dry[B], tlo[B];
    uint8_t cnt[B];
};
inline Scratch* scratch() {
    static thread_local Scratch* s = nullptr;
    if (!s) s = new (std::nothrow) Scratch();
    return s;
}
inline void load_w(Scratch& s, const float* wz, const float* w, int32_t K, int64_t n, int64_t i0, int64_t m,
                   const float* zs) {
    for (int64_t j = 0; j < m; ++j) {
        s.zc[j] = zs[i0 + j];
        s.wc[j] = 0.0f;
    }
    for (int32_t k = 0; k < K; ++k) {
        const float kf = static_cast<float>(k) + 1.0f;
        const float* zr = wz + k * n + i0;
        const float* wr = w + k * n + i0;
        float* zo = s.zc + (k + 1) * B;
        float* wo = s.wc + (k + 1) * B;
        for (int64_t j = 0; j < m; ++j) {
            const float g = zs[i0 + j], zz = zr[j];
            const bool under = zz < g + 10.0f;
            zo[j] = under ? g + f(0.01) * kf : zz;
            wo[j] = under ? 0.0f : nan_to_num(wr[j], 0.0f);
        }
    }
}
inline void t_at(const Profile& p, int64_t i0, int64_t m, const float* zq, uint8_t* cnt, float* out) {
    count_below(p.z, p.L, p.n, i0, m, zq, cnt);
    for (int64_t j = 0; j < m; ++j) out[j] = interp(p.t, bracket_from_count(p.z, p.L, p.n, i0 + j, zq[j], cnt[j]));
}
template <class TW, class TB>
int32_t sink(const Profile& p, int32_t K, const float* wz, const float* w, const float* depth, const float* zi,
             const TB* ws, const void* lee, int32_t lee_kind, const float* e, const float* cb, int64_t m,
             int32_t n, bool parts, void* extra_out, void* th_out, float* why) {
    using E = std::common_type_t<TW, TB>;
    E* extra = static_cast<E*>(extra_out);
    E* th = static_cast<E*>(th_out);
    Scratch* sp = scratch();
    if (!sp) return -1;
    Scratch& s = *sp;
    float fi[MAXN];
    steps(n, fi);
    const float nf = static_cast<float>(n);
    const double nd = static_cast<double>(n);
    const double sqrt18 = std::sqrt(1.8);
    const int32_t Lc = K + 1;
    int64_t a = 0;
    for (int64_t i0 = 0; i0 < p.n; i0 += B) {
        const int64_t bm = i0 + B < p.n ? B : p.n - i0;
        for (int64_t j = 0; j < bm; ++j) {
            s.top[j] = vclip(nan_to_num(depth[i0 + j], 0.0f), TOP_MIN, TOP_MAX);
            s.base[j] = p.zs[i0 + j] + LO;
            s.span[j] = s.top[j] - LO;
            s.wsum[j] = 0.0f;
            s.dry[j] = 0.0f;
        }
        if (w) load_w(s, wz, w, K, p.n, i0, bm, p.zs);
        for (int32_t i = 0; i < n; ++i) {
            const float fs = fi[i];
            for (int64_t j = 0; j < bm; ++j) s.zq[j] = s.base[j] + s.span[j] * fs;
            if (w) {
                count_below(s.zc, Lc, B, 0, bm, s.zq, s.cnt);
                for (int64_t j = 0; j < bm; ++j)
                    s.wsum[j] += interp(s.wc, bracket_from_count(s.zc, Lc, B, j, s.zq[j], s.cnt[j])) / nf;
            }
            count_below(p.z, p.L, p.n, i0, bm, s.zq, s.cnt);
            for (int64_t j = 0; j < bm; ++j) {
                const Bracket b = bracket_from_count(p.z, p.L, p.n, i0 + j, s.zq[j], s.cnt[j]);
                s.dry[j] += vmax(interp(p.t, b) - interp(p.td, b), 0.0f) / nf;
            }
        }
        t_at(p, i0, bm, s.base, s.cnt, s.tlo);
        for (int64_t j = 0; j < bm; ++j) s.zq[j] = p.zs[i0 + j] + s.top[j];
        t_at(p, i0, bm, s.zq, s.cnt, s.zq);
        for (int64_t j = 0; j < bm; ++j) {
            const int64_t c = i0 + j;
            const float wmean = s.wsum[j], top = s.top[j];
            TW wm = wmean;
            if (lee_kind != 0 && !(wmean > 0.25f)) {
                if constexpr (std::is_same_v<TW, double>) {
                    wm = vmin<double>(wmean, -vmax<double>(nan0(static_cast<const double*>(lee)[c]), 0.0));
                } else {
                    wm = vmin(wmean, -vmax(nan_to_num(static_cast<const float*>(lee)[c], 0.0f), 0.0f));
                }
            }
            TB between = TB(0);
            const float d = nan_to_num(depth[c], 0.0f);
            if (thermals(ws[c], d)) {
                if (a >= m) return -2;
                const float z = vmax(nan_to_num(zi[c], 0.0f), 1.0f);
                const float* y = cb + a * n;
                ++a;
                float acc = 0.0f;
                for (int32_t i = 0; i < n; ++i) {
                    const float x = band_x(d, z, fi[i]);
                    const double sw = x <= 1.0f ? sqrt18 * static_cast<double>(y[i]) * static_cast<double>(vmax(1.0f - f(0.8) * x, 0.0f)) : 0.0;
                    acc = static_cast<float>(static_cast<double>(acc) + sw / nd);
                }
                between = TB(0.6) * nan0(ws[c]) * TB(acc);
            }
            const float lapse = (s.tlo[j] - s.zq[j]) / (top - LO) * 1000.0f;
            const float convective = vclip((lapse - 6.0f) / 3.0f, 0.0f, 1.0f);
            const float peak = 0.5f + 2.5f * convective;
            const float precip = peak * (1.0f - e[c]) * vclip(f(0.1) + s.dry[j] / 12.0f, f(0.1), 1.0f);
            const TW rain_dd = TW(precip) - vmax<TW>(wm, TW(0));
            const E ex = vmax<E>(E(0), vmax<E>(-E(wm), E(rain_dd) - E(between)));
            extra[c] = ex;
            if (!parts) continue;
            const E total = vmax<E>(E(0), vmax<E>(E(between) - E(wm), E(rain_dd)));
            const E t = vmin<E>(E(between), total);
            th[c] = t;
            why[c] = ex >= E(0.05) ? (E(rain_dd) - E(between) >= -E(wm) ? 3.0f : 2.0f) : (t >= E(0.05) ? 1.0f : 0.0f);
        }
    }
    return a == m ? 0 : -2;
}
}
extern "C" {
int64_t cx_sink_x(int64_t cells, int32_t n, const float* depth, const float* zi, const void* ws, int32_t ws64,
                  float* x) {
    if (n < 2 || n > MAXN) return -1;
    float fi[MAXN];
    steps(n, fi);
    int64_t m = 0;
    for (int64_t c = 0; c < cells; ++c) {
        const float d = nan_to_num(depth[c], 0.0f);
        const bool on = ws64 ? thermals(static_cast<const double*>(ws)[c], d) : thermals(static_cast<const float*>(ws)[c], d);
        if (!on) continue;
        const float z = vmax(nan_to_num(zi[c], 0.0f), 1.0f);
        for (int32_t i = 0; i < n; ++i) x[m * n + i] = band_x(d, z, fi[i]);
        ++m;
    }
    return m;
}
int32_t cx_sink_rate(const Profile* p, int32_t K, const float* wz, const float* w, const float* depth,
                     const float* zi, const void* ws, int32_t ws64, const void* lee, int32_t lee_kind,
                     const float* e, const float* cb, int64_t m, int32_t n, int32_t parts,
                     void* extra, void* th, float* why) {
    if (n < 2 || n > MAXN || (w && (K < 1 || K + 1 > MAXW || !wz)) || p->L < 2 || p->L > 255 || lee_kind < 0 ||
        lee_kind > 2 || (lee_kind != 0 && !lee) || (parts && (!th || !why)))
        return -1;
    if (!w) K = 0;
    const bool pt = parts != 0;
    if (lee_kind == 2) {
        if (ws64) return sink<double, double>(*p, K, wz, w, depth, zi, static_cast<const double*>(ws), lee, lee_kind, e, cb, m, n, pt, extra, th, why);
        return sink<double, float>(*p, K, wz, w, depth, zi, static_cast<const float*>(ws), lee, lee_kind, e, cb, m, n, pt, extra, th, why);
    }
    if (ws64) return sink<float, double>(*p, K, wz, w, depth, zi, static_cast<const double*>(ws), lee, lee_kind, e, cb, m, n, pt, extra, th, why);
    return sink<float, float>(*p, K, wz, w, depth, zi, static_cast<const float*>(ws), lee, lee_kind, e, cb, m, n, pt, extra, th, why);
}
}
