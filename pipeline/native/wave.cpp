#include "cxphys.h"
using namespace cx;
namespace {
constexpr float F(double x) { return static_cast<float>(x); }
constexpr int NI = 5;
constexpr int NS = 2 * NI * 3;
const float DZ = F(500.0);
inline double clipd(double x, double lo, double hi) {
    double y = std::isnan(x) ? x : (x > lo ? x : lo);
    return std::isnan(y) ? y : (y < hi ? y : hi);
}
inline void layer_mean_uv(const Column& c, float zb, float zt, float& um, float& vm) {
    float au = 0.0f, av = 0.0f;
    for (int i = 0; i < 6; ++i) {
        const float zq = zb + (zt - zb) * F(static_cast<double>(i) / 5.0);
        float u, v;
        c.uv_at(zq, u, v);
        au += u;
        av += v;
    }
    um = au / F(6.0);
    vm = av / F(6.0);
}
}
extern "C" {
void cx_wave_sample(const Profile* p, int64_t i0, int64_t i1, float* sl, float* su, float* cosang, uint8_t* ok,
                    float* ts, float* lnps, float* us) {
    static thread_local Block* blk = nullptr;
    if (!blk) blk = new Block();
    const int64_t m = i1 - i0;
    for (int64_t b0 = i0; b0 < i1; b0 += BLOCK_CELLS) {
        const int64_t bm = b0 + BLOCK_CELLS < i1 ? BLOCK_CELLS : i1 - b0;
        blk->load(*p, b0, bm, COL_LNP | COL_T | COL_WIND);
        for (int64_t jb = 0; jb < bm; ++jb) {
            const Column c = blk->col(jb);
            const int64_t j = b0 + jb - i0;
            const float zs = c.zs;
            const float low_b = zs + F(500.0);
            const float low_t = npmin(zs + F(3000.0), c.ztop - F(2500.0));
            const float up_t = npmin(low_t + F(4500.0), c.ztop - DZ);
            ok[j] = (low_t - low_b >= F(800.0)) && (up_t - low_t >= F(1500.0));
            float ul, vl, uu, vu;
            layer_mean_uv(c, low_b, low_t, ul, vl);
            layer_mean_uv(c, low_t, up_t, uu, vu);
            const float s_l = std::hypot(ul, vl), s_u = std::hypot(uu, vu);
            sl[j] = s_l;
            su[j] = s_u;
            const float ex = ul / npmax(s_l, F(1e-3)), ey = vl / npmax(s_l, F(1e-3));
            cosang[j] = (ul * uu + vl * vu) / npmax(s_l * s_u, F(1e-6));
            for (int layer = 0; layer < 2; ++layer) {
                const float zb = layer ? low_t : low_b, zt = layer ? up_t : low_t;
                for (int i = 0; i < NI; ++i) {
                    const float zc = zb + (zt - zb) * F(static_cast<double>(i) / (NI - 1));
                    const float zq3[3] = {zc - DZ, zc, zc + DZ};
                    for (int o = 0; o < 3; ++o) {
                        const int64_t at = static_cast<int64_t>(layer * 15 + i * 3 + o) * m + j;
                        const LocalBracket b = c.tb(zq3[o]);
                        ts[at] = local_interp(c.t, b);
                        lnps[at] = local_interp(c.lnp, b);
                        float u, v;
                        c.uv_at(zq3[o], u, v);
                        us[at] = npmax(u * ex + v * ey, F(1.5));
                    }
                }
            }
        }
    }
}
void cx_wave_finish(int64_t m, const float* sl, const float* su, const float* turn, const uint8_t* ok,
                    const float* th, const float* us, double* score, float* wavelen) {
    const float G = F(9.80665), twodz = F(2 * 500.0), dz2 = F(500.0 * 500.0), n = F(NI);
    const float nan = std::nanf("");
    for (int64_t j = 0; j < m; ++j) {
        if (!ok[j]) {
            score[j] = std::nan("");
            wavelen[j] = nan;
            continue;
        }
        float l2s[2], n2s[2];
        for (int layer = 0; layer < 2; ++layer) {
            float l2 = 0.0f, n2 = 0.0f;
            for (int i = 0; i < NI; ++i) {
                const int64_t k = static_cast<int64_t>(layer * 15 + i * 3) * m + j;
                const float ta = th[k], tb = th[k + m], tc = th[k + 2 * m];
                const float ua = us[k], ub = us[k + m], uc = us[k + 2 * m];
                const float N2 = G / tb * (tc - ta) / twodz;
                const float uzz = (uc - F(2.0) * ub + ua) / dz2;
                l2 += N2 / (ub * ub) - uzz / ub;
                n2 += N2;
            }
            l2s[layer] = l2 / n;
            n2s[layer] = n2 / n;
        }
        const float l2l = l2s[0], n2l = n2s[0], l2u = l2s[1];
        const double ratio = l2u > F(1e-9) ? static_cast<double>(l2l / npmax(l2u, F(1e-9))) : (l2l > 0.0f ? 10.0 : 0.0);
        const float nlow = std::sqrt(npmax(n2l, 0.0f));
        const float s_l = sl[j], s_u = su[j];
        const float s_w = s_u > s_l ? (s_l + s_u) / F(2.0) : s_l;
        const float f1 = npclip((s_w - F(5.0)) / F(7.5), 0.0f, 1.0f);
        const float f2 = npclip(1.0f - (turn[j] - F(20.0)) / F(40.0), 0.0f, 1.0f);
        const float f3 = s_u >= s_l ? 1.0f : npclip(s_u / npmax(s_l, F(1e-3)), 0.0f, 1.0f);
        const float f4 = npclip((nlow - F(0.005)) / F(0.005), 0.0f, 1.0f);
        const double f5 = 0.4 + 0.6 * clipd((ratio - 1.0) / 3.0, 0.0, 1.0);
        score[j] = static_cast<double>(f1 * f2 * f3 * f4) * f5;
        const float kmean = std::sqrt(npmax((l2l + npmax(l2u, 0.0f)) / F(2.0), 0.0f));
        const float wl = (l2l > 0.0f && kmean > 0.0f) ? F(2 * 3.141592653589793) / npmax(kmean, F(1e-9)) : nan;
        wavelen[j] = npmin(wl, F(60000.0));
    }
}
}
