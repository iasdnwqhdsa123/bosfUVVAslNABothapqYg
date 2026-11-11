#pragma GCC optimize("no-trapping-math")
#include <cmath>
#include <cstdint>
#include <vector>
namespace {
inline double dmax(double a, double b) { return ((a >= b) | (a != a)) ? a : b; }
inline double dmin(double a, double b) { return ((a <= b) | (a != a)) ? a : b; }
inline double hyp(double a, double b) {
    const double aa = std::fabs(a), ab = std::fabs(b);
    const double m = dmax(aa, ab);
    double r = dmin(aa, ab);
    r = r / m;
    r = r * r;
    r = r + 1.0;
    r = std::sqrt(r);
    r = r * m;
    r = m == 0.0 ? 0.0 : r;
    return m == INFINITY ? INFINITY : r;
}
inline void wind_at(const double* Z, const double* U, const double* V, int32_t L, int64_t n, double h, double& u, double& v) {
    int32_t cnt = 0;
    for (int32_t j = 1; j < L; ++j) cnt += Z[j] < h;
    int32_t k = 1 + cnt;
    if (k > L - 1) k = L - 1;
    const int64_t last = n - 1 > 0 ? n - 1 : 0;
    if (n == 1 || h <= Z[0]) {
        u = U[0];
        v = V[0];
    } else if (h >= Z[last]) {
        u = U[last];
        v = V[last];
    } else {
        const double f = (h - Z[k - 1]) / dmax(Z[k] - Z[k - 1], 1e-3);
        u = U[k - 1] + (U[k] - U[k - 1]) * f;
        v = V[k - 1] + (V[k] - V[k - 1]) * f;
    }
}
inline double n2_at(const double* pz, const double* pn2, int32_t S, int64_t cnt, double z) {
    int32_t c = 0;
    for (int32_t j = 1; j < S; ++j) c += pz[j] < z;
    const int32_t i = 1 + c;
    if (z <= pz[0]) return pn2[0];
    if (!(i < cnt)) return pn2[cnt - 1];
    const int32_t ii = i < S - 1 ? i : S - 1;
    const double t = (z - pz[ii - 1]) / dmax(1e-6, pz[ii] - pz[ii - 1]);
    return pn2[ii - 1] + t * (pn2[ii] - pn2[ii - 1]);
}
#define CX_WIDE __attribute__((target_clones("avx2", "default")))
CX_WIDE void m_of(double ur, double nj, double pj, double l2cap, int32_t nk, const double* __restrict ui_,
                  const double* __restrict k2, double* __restrict mr, double* __restrict mi) {
    for (int32_t k = 0; k < nk; ++k) {
        const double ui = ui_[k];
        const double d2r = ur * ur - ui * ui, d2i = 2 * ur * ui;
        const double dd = d2r * d2r + d2i * d2i, uu = ur * ur + ui * ui;
        double lr = (nj * d2r) / dd - (pj * ur) / uu;
        double li = (-nj * d2i) / dd + (pj * ui) / uu;
        const double mag = hyp(lr, li);
        const bool cap = mag > l2cap;
        const double fc = l2cap / mag;
        const double f = cap ? fc : 1.0;
        lr = cap ? lr * f : lr;
        li = cap ? li * f : li;
        const double re = lr - k2[k];
        const double r = hyp(re, li);
        const double a = std::sqrt(dmax(0.0, (r + re) / 2));
        const double b = std::sqrt(dmax(0.0, (r - re) / 2));
        mr[k] = a;
        mi[k] = li < 0 ? -b : b;
    }
}
#define CX_STEP_HEAD                                                                                \
    const double mr = mr_[k], mi = mi_[k], sr = sr_[k], si = si_[k], dr = dr_[k], di = di_[k];      \
    const double cr = ca[k] * ch[k], ci = -sa[k] * sh[k], snr = sa[k] * ch[k], sni = ca[k] * sh[k]; \
    const double mm = mr * mr + mi * mi;                                                            \
    const bool small = mm * D * D < 1e-12;                                                          \
    const double qr = (snr * mr + sni * mi) / mm, qi = (sni * mr - snr * mi) / mm;                  \
    const double sdr = small ? D : qr, sdi = small ? 0.0 : qi;                                      \
    o1r[k] = cr * sr - ci * si - (sdr * dr - sdi * di);                                             \
    o1i[k] = cr * si + ci * sr - (sdr * di + sdi * dr);
CX_WIDE void step(int32_t nk, const double* __restrict mr_, const double* __restrict mi_, double D,
                  const double* __restrict ch, const double* __restrict sh, const double* __restrict ca,
                  const double* __restrict sa, const double* __restrict sr_, const double* __restrict si_,
                  const double* __restrict dr_, const double* __restrict di_, double* __restrict o1r,
                  double* __restrict o1i, double* __restrict o2r, double* __restrict o2i) {
    for (int32_t k = 0; k < nk; ++k) {
        CX_STEP_HEAD
        const double msr = mr * snr - mi * sni, msi = mr * sni + mi * snr;
        o2r[k] = msr * sr - msi * si + (cr * dr - ci * di);
        o2i[k] = msr * si + msi * sr + (cr * di + ci * dr);
    }
}
CX_WIDE void step_w(int32_t nk, const double* __restrict mr_, const double* __restrict mi_, double D,
                    const double* __restrict ch, const double* __restrict sh, const double* __restrict ca,
                    const double* __restrict sa, const double* __restrict sr_, const double* __restrict si_,
                    const double* __restrict dr_, const double* __restrict di_, double* __restrict o1r,
                    double* __restrict o1i) {
    for (int32_t k = 0; k < nk; ++k) {
        CX_STEP_HEAD
    }
}
template <class T>
void sample(int64_t n, const T* a, int64_t nx, const int64_t* y0, const int64_t* x0, const double* tx_, const double* ty_,
            const double* w0, const double* w1, const double* w2, const double* w3, const uint8_t* inb, double* out) {
    const double nan = std::nan("");
    for (int64_t i = 0; i < n; ++i) {
        if (!inb[i]) {
            out[i] = nan;
            continue;
        }
        const T* p = a + y0[i] * nx + x0[i];
        const double v[4] = {p[0], p[1], p[nx], p[nx + 1]};
        const bool h[4] = {std::isfinite(v[0]), std::isfinite(v[1]), std::isfinite(v[2]), std::isfinite(v[3])};
        if (h[0] & h[1] & h[2] & h[3]) {
            const double tx = tx_[i], ty = ty_[i];
            out[i] = (v[0] * (1 - tx) + v[1] * tx) * (1 - ty) + (v[2] * (1 - tx) + v[3] * tx) * ty;
            continue;
        }
        const double w[4] = {w0[i], w1[i], w2[i], w3[i]};
        double sw = 0.0, sv = 0.0;
        for (int q = 0; q < 4; ++q) {
            if (!h[q]) continue;
            sw = sw + w[q];
            sv = sv + w[q] * v[q];
        }
        out[i] = sw >= 0.5 ? sv / (sw > 0 ? sw : 1.0) : nan;
    }
}
}
extern "C" {
void cx_lee_sample(int64_t n, const void* a, int32_t f64, int64_t nx, const int64_t* y0, const int64_t* x0,
                   const double* tx, const double* ty, const double* w0, const double* w1, const double* w2,
                   const double* w3, const uint8_t* inb, double* out) {
    if (f64)
        sample(n, static_cast<const double*>(a), nx, y0, x0, tx, ty, w0, w1, w2, w3, inb, out);
    else
        sample(n, static_cast<const float*>(a), nx, y0, x0, tx, ty, w0, w1, w2, w3, inb, out);
}
void cx_lee_columns(int64_t M, int32_t L, const double* Z, const double* U, const double* V, const int64_t* n,
                    const double* dx, const double* dy, int32_t S, const double* pz, const double* pn2,
                    const int64_t* cnt, int32_t nh, const double* hs, int32_t nl, const int32_t* um_idx,
                    const double* edges, const uint8_t* high, const double* consts, int32_t nk, const double* k2,
                    const double* ui_low, const double* ui_high, int32_t na, const double* at, int32_t* layer,
                    double* mr, double* mi, double* ar, double* ai) {
    const double MIN_U = consts[0], L2_CAP = consts[1], N2_MIN = consts[2], N2_MAX = consts[3], hh = consts[4];
    std::vector<double> along(nh);
    const int64_t ns = nl + na;
    for (int64_t i = 0; i < M; ++i) {
        const double *Zi = Z + i * L, *Ui = U + i * L, *Vi = V + i * L;
        for (int32_t c = 0; c < nh; ++c) {
            double u, v;
            wind_at(Zi, Ui, Vi, L, n[i], hs[c], u, v);
            along[c] = u * dx[i] + v * dy[i];
        }
        auto um = [&](const int32_t* c) { return (along[c[0]] + along[c[1]] * 2 + along[c[2]]) / 4; };
        double *mri = mr + i * nl * nk, *mii = mi + i * nl * nk;
        double *ari = ar + i * ns * nk, *aii = ai + i * ns * nk;
        for (int32_t j = 0; j < nl; ++j) {
            const int32_t* c = um_idx + j * 9;
            const double z = (edges[j] + edges[j + 1]) / 2;
            const double u0 = um(c);
            const double upp = c[3] < 0 ? 0.0 : (um(c + 3) - 2 * u0 + um(c + 6)) / hh;
            const double n2 = dmax(N2_MIN, dmin(N2_MAX, n2_at(pz + i * S, pn2 + i * S, S, cnt[i], z)));
            const double ul = std::fabs(u0) < MIN_U ? (u0 < 0 ? -MIN_U : MIN_U) : u0;
            m_of(ul, n2, upp, L2_CAP, nk, high[j] ? ui_high : ui_low, k2, mri + j * nk, mii + j * nk);
            const double D = edges[j + 1] - edges[j];
            for (int32_t k = 0; k < nk; ++k) {
                ari[j * nk + k] = mri[j * nk + k] * D;
                aii[j * nk + k] = mii[j * nk + k] * D;
            }
        }
        for (int32_t a = 0; a < na; ++a) {
            const double h = at[i * na + a];
            int32_t j = -1;
            if (h >= 0 && h <= edges[nl]) {
                int32_t c = 0;
                for (int32_t e = 0; e <= nl; ++e) c += edges[e] <= h;
                j = c - 1 < nl - 1 ? c - 1 : nl - 1;
            }
            layer[i * na + a] = j;
            double *ra = ari + (nl + a) * nk, *ia = aii + (nl + a) * nk;
            if (j < 0) {
                for (int32_t k = 0; k < nk; ++k) ra[k] = ia[k] = 0.0;
                continue;
            }
            const double D = edges[j + 1] - h;
            for (int32_t k = 0; k < nk; ++k) {
                ra[k] = mri[j * nk + k] * D;
                ia[k] = mii[j * nk + k] * D;
            }
        }
    }
}
void cx_lee_bands(int64_t M, int32_t nl, const double* edges, int32_t nk, int32_t na, int32_t nband, int32_t bandk,
                  const double* at, const int32_t* layer, const double* mr, const double* mi, const double* ca,
                  const double* sa, const double* ch, const double* sh, const double* grow, const double* f,
                  double* bands) {
    const int64_t ns = nl + na;
    std::vector<double> buf(static_cast<size_t>(nk) * (8 + 2 * na));
    double *wr = buf.data(), *wi = wr + nk, *pr = wi + nk, *pi = pr + nk;
    double *xr = pi + nk, *xi = xr + nk, *yr = xi + nk, *yi = yr + nk;
    double *recr = yi + nk, *reci = recr + static_cast<size_t>(na) * nk;
    std::vector<uint8_t> done(na);
    for (int64_t i = 0; i < M; ++i) {
        const double *mri = mr + i * nl * nk, *mii = mi + i * nl * nk;
        const int64_t s0 = i * ns * nk;
        const int32_t* lay = layer + i * na;
        const double *mt = mri + static_cast<int64_t>(nl - 1) * nk, *nt = mii + static_cast<int64_t>(nl - 1) * nk;
        for (int32_t k = 0; k < nk; ++k) {
            const bool flip = (nt[k] < 0) || (nt[k] == 0 && mt[k] < 0);
            const double tr = flip ? -mt[k] : mt[k], ti = flip ? -nt[k] : nt[k];
            wr[k] = 1.0;
            wi[k] = 0.0;
            pr[k] = -ti;
            pi[k] = tr;
        }
        for (int32_t a = 0; a < na; ++a) {
            done[a] = 0;
            for (int32_t k = 0; k < nk; ++k) recr[a * nk + k] = reci[a * nk + k] = 0.0;
        }
        for (int32_t j = nl - 1; j >= 0; --j) {
            const double *mj = mri + static_cast<int64_t>(j) * nk, *nj = mii + static_cast<int64_t>(j) * nk;
            const double zt = edges[j + 1];
            for (int32_t a = 0; a < na; ++a) {
                if (lay[a] != j) continue;
                const int64_t o = s0 + static_cast<int64_t>(nl + a) * nk;
                step_w(nk, mj, nj, zt - at[i * na + a], ch + o, sh + o, ca + o, sa + o, wr, wi, pr, pi, recr + a * nk,
                       reci + a * nk);
                done[a] = 1;
            }
            const int64_t o = s0 + static_cast<int64_t>(j) * nk;
            step(nk, mj, nj, zt - edges[j], ch + o, sh + o, ca + o, sa + o, wr, wi, pr, pi, xr, xi, yr, yi);
            bool near = false;
            for (int32_t k = 0; k < nk; ++k) {
                wr[k] = xr[k];
                wi[k] = xi[k];
                pr[k] = yr[k];
                pi[k] = yi[k];
                const double q = xr[k] * xr[k] + xi[k] * xi[k];
                near |= !(q >= 1e-99 && q <= 1e99);
            }
            if (!near) continue;
            for (int32_t k = 0; k < nk; ++k) {
                const double q = wr[k] * wr[k] + wi[k] * wi[k];
                if (q >= 1e-99 && q <= 1e99) continue;
                const double s = hyp(wr[k], wi[k]);
                if (!(s > 1e50 || (s < 1e-50 && s > 0))) continue;
                wr[k] = wr[k] / s;
                wi[k] = wi[k] / s;
                pr[k] = pr[k] / s;
                pi[k] = pi[k] / s;
                for (int32_t a = 0; a < na; ++a) {
                    if (!done[a]) continue;
                    recr[a * nk + k] = recr[a * nk + k] / s;
                    reci[a * nk + k] = reci[a * nk + k] / s;
                }
            }
        }
        double* out = bands + i * nband * na;
        for (int32_t k = 0; k < nk; ++k) {
            const double d = wr[k] * wr[k] + wi[k] * wi[k];
            xr[k] = (d == 0 || d != d) ? 1e-300 : d;
        }
        for (int32_t a = 0; a < na; ++a) {
            const double g = grow[i * na + a];
            const double *rr = recr + a * nk, *ri = reci + a * nk;
            for (int32_t k = 0; k < nk; ++k) {
                const double Tr = (rr[k] * wr[k] + ri[k] * wi[k]) / xr[k] * g;
                const double Ti = (ri[k] * wr[k] - rr[k] * wi[k]) / xr[k] * g;
                yr[k] = done[a] ? hyp(Tr, Ti) : 0.0;
            }
            const bool over = at[i * na + a] > 0;
            for (int32_t b = 0; b < nband; ++b) {
                double sm = 0.0;
                for (int32_t q = 0; q < bandk; ++q) sm = sm + yr[b * bandk + q];
                out[b * na + a] = over ? (sm / bandk) * f[i] : 0.0;
            }
        }
    }
}
void cx_lattice_at(int64_t n, const double* lat, const double* lon, double step, int64_t i0, int64_t j0, int64_t ni,
                   int64_t nj, const uint8_t* okL, const double* uL, const double* vL, const double* bandsL,
                   int32_t nb, double* u, double* v, double* bands, uint8_t* ok) {
    const double nan = std::nan("");
    for (int64_t p = 0; p < n; ++p) {
        const double fi = lon[p] / step, fj = lat[p] / step;
        const double fi0 = std::floor(fi), fj0 = std::floor(fj);
        const double tx = fi - fi0, ty = fj - fj0;
        const int64_t ci = static_cast<int64_t>(fi0) - i0, cj = static_cast<int64_t>(fj0) - j0;
        const double ws[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
        int64_t rows[4];
        double W = 0.0;
        for (int q = 0; q < 4; ++q) {
            const int64_t i = ci + (q & 1), j = cj + (q >> 1);
            int64_t r = (i >= 0 && i < ni && j >= 0 && j < nj) ? j * ni + i : -1;
            if (r >= 0 && !okL[r]) r = -1;
            rows[q] = r;
            if (r >= 0) W = W + ws[q];
        }
        double* b = bands + p * nb;
        for (int32_t c = 0; c < nb; ++c) b[c] = 0.0;
        const bool good = W > 0;
        ok[p] = good;
        double su = 0.0, sv = 0.0;
        for (int q = 0; q < 4; ++q) {
            const int64_t r = rows[q];
            if (r < 0) continue;
            const double w = good ? ws[q] / W : 0.0;
            su += uL[r] * w;
            sv += vL[r] * w;
            const double* src = bandsL + r * nb;
            for (int32_t c = 0; c < nb; ++c) b[c] += src[c] * w;
        }
        u[p] = good ? su : nan;
        v[p] = good ? sv : nan;
    }
}
}
