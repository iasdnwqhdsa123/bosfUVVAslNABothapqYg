#include "cxphys.h"
using namespace cx;
extern "C" {
void cx_profile_build(int64_t n, int32_t L, const int32_t* order, const float* p_sorted,
                      const float* zs, const float* ps, const float* t2, const float* td2,
                      const float* u10, const float* v10,
                      const float* z, const float* t, const float* td, const float* u, const float* v,
                      const float* uh, const float* vh, float hub,
                      float* oz, float* opl, float* ot, float* otd,
                      float* ozw, float* ou, float* ov) {
    const bool with_hub = uh != nullptr && vh != nullptr;
    const int32_t Lw = L + 1 + (with_hub ? 1 : 0);
    for (int64_t i0 = 0; i0 < n; i0 += BLOCK) {
        const int64_t i1 = i0 + BLOCK < n ? i0 + BLOCK : n;
        for (int64_t i = i0; i < i1; ++i) {
            oz[i] = zs[i] + 2.0f;
            opl[i] = ps[i];
            ot[i] = t2[i];
            otd[i] = npmin(td2[i], t2[i]);
            ozw[i] = zs[i] + 10.0f;
            ou[i] = u10[i];
            ov[i] = v10[i];
        }
        for (int32_t j = 0; j < L; ++j) {
            const int64_t src = static_cast<int64_t>(order[j]) * n;
            const int64_t dst = static_cast<int64_t>(j + 1) * n;
            const float kf = static_cast<float>(j) + 1.0f;
            const float pj = p_sorted[j];
            for (int64_t i = i0; i < i1; ++i) {
                const float s = zs[i], zz = z[src + i];
                const bool under = zz < s + 10.0f;
                oz[dst + i] = under ? (s + 2.0f) + 0.01f * kf : zz;
                opl[dst + i] = under ? ps[i] * (1.0f - 1e-5f * kf) : pj;
                const float tt = under ? t2[i] : t[src + i];
                ot[dst + i] = tt;
                otd[dst + i] = npmin(under ? td2[i] : td[src + i], tt);
                ozw[dst + i] = under ? (s + 10.0f) + 0.01f * kf : zz;
                ou[dst + i] = under ? u10[i] : u[src + i];
                ov[dst + i] = under ? v10[i] : v[src + i];
            }
        }
        if (!with_hub) continue;
        for (int64_t i = i0; i < i1; ++i) {
            const float zh = zs[i] + hub;
            const Bracket b = bracket(ozw, L + 1, n, i, zh);
            const bool ok = std::isfinite(uh[i]) && std::isfinite(vh[i]);
            float cz[64], cu[64], cv[64];
            for (int32_t k = 0; k <= L; ++k) {
                cz[k] = ozw[k * n + i];
                cu[k] = ou[k * n + i];
                cv[k] = ov[k * n + i];
            }
            cz[L + 1] = zh;
            cu[L + 1] = ok ? uh[i] : interp(ou, b);
            cv[L + 1] = ok ? vh[i] : interp(ov, b);
            for (int32_t m = 1; m < Lw; ++m) {
                const float kz = cz[m], ku = cu[m], kv = cv[m];
                int32_t j = m - 1;
                while (j >= 0 && !std::isnan(kz) && (std::isnan(cz[j]) || kz < cz[j])) {
                    cz[j + 1] = cz[j];
                    cu[j + 1] = cu[j];
                    cv[j + 1] = cv[j];
                    --j;
                }
                cz[j + 1] = kz;
                cu[j + 1] = ku;
                cv[j + 1] = kv;
            }
            for (int32_t k = 0; k < Lw; ++k) {
                ozw[k * n + i] = cz[k];
                ou[k * n + i] = cu[k];
                ov[k * n + i] = cv[k];
            }
        }
    }
}
void cx_profile_at(const Profile* p, const float* zq, int32_t count, const float* const* arrays,
                   const int32_t* wind, float* const* outs) {
    bool need_t = false, need_w = false;
    for (int32_t j = 0; j < count; ++j) (wind[j] ? need_w : need_t) = true;
    const int64_t n = p->n;
    uint8_t ct[BLOCK], cw[BLOCK];
    for (int64_t i0 = 0; i0 < n; i0 += BLOCK) {
        const int64_t m = i0 + BLOCK < n ? BLOCK : n - i0;
        if (need_t) count_below(p->z, p->L, n, i0, m, zq + i0, ct);
        if (need_w) count_below(p->zw, p->Lw, n, i0, m, zq + i0, cw);
        for (int64_t j = 0; j < m; ++j) {
            const int64_t i = i0 + j;
            Bracket bt{0, 0, 0.0f}, bw{0, 0, 0.0f};
            if (need_t) bt = bracket_from_count(p->z, p->L, n, i, zq[i], ct[j]);
            if (need_w) bw = bracket_from_count(p->zw, p->Lw, n, i, zq[i], cw[j]);
            for (int32_t a = 0; a < count; ++a) outs[a][i] = interp(arrays[a], wind[a] ? bw : bt);
        }
    }
}
void cx_column_at(int64_t n, int32_t L, const float* z, const float* a, const float* zq, float* out) {
    uint8_t c[BLOCK];
    for (int64_t i0 = 0; i0 < n; i0 += BLOCK) {
        const int64_t m = i0 + BLOCK < n ? BLOCK : n - i0;
        count_below(z, L, n, i0, m, zq + i0, c);
        for (int64_t j = 0; j < m; ++j) out[i0 + j] = interp(a, bracket_from_count(z, L, n, i0 + j, zq[i0 + j], c[j]));
    }
}
}
extern "C" {
void cx_block_at(const Profile* p, const float* zq, float* t, float* td, float* lnp, float* u, float* v) {
    for_each_cell(*p, COL_ALL, [&](const Column& c, int64_t i) {
        const LocalBracket b = c.tb(zq[i]);
        t[i] = local_interp(c.t, b);
        td[i] = local_interp(c.td, b);
        lnp[i] = local_interp(c.lnp, b);
        c.uv_at(zq[i], u[i], v[i]);
    });
}
}
