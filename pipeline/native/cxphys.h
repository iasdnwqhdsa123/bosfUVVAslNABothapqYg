#pragma once
#include <cfloat>
#include <cmath>
#include <cstdint>
namespace cx {
inline float npmax(float a, float b) { return (a >= b || std::isnan(a)) ? a : b; }
inline float npmin(float a, float b) { return (a <= b || std::isnan(a)) ? a : b; }
inline float npclip(float x, float lo, float hi) {
    float y = std::isnan(x) ? x : (x > lo ? x : lo);
    return std::isnan(y) ? y : (y < hi ? y : hi);
}
inline float nan_to_num(float x, float v = 0.0f) {
    if (std::isnan(x)) return v;
    if (std::isinf(x)) return x > 0 ? FLT_MAX : -FLT_MAX;
    return x;
}
struct Profile {
    int64_t n;
    int32_t L, Lw;
    const float *z, *lnp, *t, *td;
    const float *zw, *u, *v;
    const float *zs;
};
struct Bracket {
    int64_t lo, hi;
    float w;
};
inline Bracket bracket_from_count(const float* z, int32_t L, int64_t n, int64_t i, float zq, int32_t c) {
    if (c < 1) c = 1;
    if (c > L - 1) c = L - 1;
    int64_t hi = static_cast<int64_t>(c) * n + i, lo = hi - n;
    float z0 = z[lo], z1 = z[hi];
    return {lo, hi, npclip((zq - z0) / npmax(z1 - z0, 1e-3f), 0.0f, 1.0f)};
}
inline Bracket bracket(const float* z, int32_t L, int64_t n, int64_t i, float zq) {
    int32_t c = 0;
    for (int32_t k = 0; k < L; ++k) c += z[k * n + i] < zq;
    return bracket_from_count(z, L, n, i, zq, c);
}
constexpr int64_t BLOCK = 1024;
inline void count_below(const float* z, int32_t L, int64_t n, int64_t i0, int64_t m, const float* zq, uint8_t* c) {
    for (int64_t j = 0; j < m; ++j) c[j] = 0;
    for (int32_t k = 0; k < L; ++k) {
        const float* zk = z + k * n + i0;
        for (int64_t j = 0; j < m; ++j) c[j] += zk[j] < zq[j];
    }
}
inline float interp(const float* a, const Bracket& b) {
    float a0 = a[b.lo], a1 = a[b.hi];
    return a0 + b.w * (a1 - a0);
}
inline Bracket thermo_at(const Profile& p, int64_t i, float zq) { return bracket(p.z, p.L, p.n, i, zq); }
inline Bracket wind_at(const Profile& p, int64_t i, float zq) { return bracket(p.zw, p.Lw, p.n, i, zq); }
struct LocalBracket {
    int32_t lo, hi;
    float w;
};
inline LocalBracket local_bracket(const float* z, int32_t L, float zq) {
    int32_t c = 0;
    for (int32_t k = 0; k < L; ++k) c += z[k] < zq;
    if (c < 1) c = 1;
    if (c > L - 1) c = L - 1;
    float z0 = z[c - 1], z1 = z[c];
    return {c - 1, c, npclip((zq - z0) / npmax(z1 - z0, 1e-3f), 0.0f, 1.0f)};
}
inline float local_interp(const float* a, const LocalBracket& b) {
    float a0 = a[b.lo], a1 = a[b.hi];
    return a0 + b.w * (a1 - a0);
}
struct Column {
    int32_t L, Lw;
    const float *z, *lnp, *t, *td, *zw, *u, *v;
    float zs, ztop;
    LocalBracket tb(float zq) const { return local_bracket(z, L, zq); }
    LocalBracket wb(float zq) const { return local_bracket(zw, Lw, zq); }
    float t_at(float zq) const { return local_interp(t, tb(zq)); }
    float td_at(float zq) const { return local_interp(td, tb(zq)); }
    float lnp_at(float zq) const { return local_interp(lnp, tb(zq)); }
    float p_at(float zq) const { return std::exp(lnp_at(zq)); }
    void uv_at(float zq, float& uu, float& vv) const {
        LocalBracket b = wb(zq);
        uu = local_interp(u, b);
        vv = local_interp(v, b);
    }
};
constexpr int64_t BLOCK_CELLS = 512;
enum : unsigned { COL_Z = 1, COL_LNP = 2, COL_T = 4, COL_TD = 8, COL_WIND = 16, COL_ALL = 31 };
class Block {
  public:
    void load(const Profile& p, int64_t i0, int64_t m, unsigned which = COL_ALL) {
        i0_ = i0;
        m_ = m;
        L_ = p.L;
        Lw_ = p.Lw;
        zs_ = p.zs + i0;
        copy(z_, p.z, p.n, L_);
        if (which & COL_LNP) copy(lnp_, p.lnp, p.n, L_);
        if (which & COL_T) copy(t_, p.t, p.n, L_);
        if (which & COL_TD) copy(td_, p.td, p.n, L_);
        if (which & COL_WIND) {
            copy(zw_, p.zw, p.n, Lw_);
            copy(u_, p.u, p.n, Lw_);
            copy(v_, p.v, p.n, Lw_);
        }
        which_ = which;
    }
    Column col(int64_t j) const {
        const bool w = which_ & COL_WIND;
        return {L_, Lw_,
                z_ + j * L_,
                (which_ & COL_LNP) ? lnp_ + j * L_ : nullptr,
                (which_ & COL_T) ? t_ + j * L_ : nullptr,
                (which_ & COL_TD) ? td_ + j * L_ : nullptr,
                w ? zw_ + j * Lw_ : nullptr,
                w ? u_ + j * Lw_ : nullptr,
                w ? v_ + j * Lw_ : nullptr,
                zs_[j], z_[j * L_ + L_ - 1]};
    }
    int64_t size() const { return m_; }
  private:
    void copy(float* dst, const float* src, int64_t n, int32_t levels) {
        for (int32_t k = 0; k < levels; ++k) {
            const float* row = src + k * n + i0_;
            for (int64_t j = 0; j < m_; ++j) dst[j * levels + k] = row[j];
        }
    }
    static constexpr int32_t MAXL = 64;
    int64_t i0_ = 0, m_ = 0;
    int32_t L_ = 0, Lw_ = 0;
    unsigned which_ = 0;
    const float* zs_ = nullptr;
    float z_[BLOCK_CELLS * MAXL], lnp_[BLOCK_CELLS * MAXL], t_[BLOCK_CELLS * MAXL], td_[BLOCK_CELLS * MAXL];
    float zw_[BLOCK_CELLS * MAXL], u_[BLOCK_CELLS * MAXL], v_[BLOCK_CELLS * MAXL];
};
template <class Fn>
inline void for_each_cell(const Profile& p, unsigned which, Fn fn) {
    static thread_local Block* blk = nullptr;
    if (!blk) blk = new Block();
    for (int64_t i0 = 0; i0 < p.n; i0 += BLOCK_CELLS) {
        const int64_t m = i0 + BLOCK_CELLS < p.n ? BLOCK_CELLS : p.n - i0;
        blk->load(p, i0, m, which);
        for (int64_t j = 0; j < m; ++j) fn(blk->col(j), i0 + j);
    }
}
}
