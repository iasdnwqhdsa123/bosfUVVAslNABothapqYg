#include <cmath>
#include <cstdint>
namespace {
inline double dmax(double a, double b) { return (a >= b || a != a) ? a : b; }
inline double dmin(double a, double b) { return (a <= b || a != a) ? a : b; }
}
extern "C" {
int64_t cx_lee_cells(int64_t nsel, int64_t i0, const int64_t* cy, const int64_t* cx, int32_t r, int64_t ny, int64_t nx,
                     int32_t T, int64_t ntx, const uint8_t* listed, const uint8_t* loaded, const uint8_t* mosaic,
                     int32_t P, int32_t D, int32_t NB, int32_t NA, const double* dec, double top_step, const uint8_t* wv,
                     const int64_t* d0, const int64_t* d1, const double* t, const double* B, const int64_t* sel,
                     double* tsum, int64_t* tn, int64_t cap, int64_t* pair_cell, double* S, int64_t* npairs) {
    const int64_t side = 2 * static_cast<int64_t>(r) + 1;
    int64_t np_ = 0, i = i0;
    double m[64];
    for (; i < nsel; ++i) {
        if (np_ + side * side > cap) break;
        const double t1 = t[i];
        const double* Bi = wv[i] ? B + sel[i] * NB * NA : nullptr;
        for (int32_t dy = -r; dy <= r; ++dy) {
            const int64_t iy = cy[i] + dy;
            if (iy < 0 || iy >= ny) continue;
            for (int32_t dx = -r; dx <= r; ++dx) {
                const int64_t ix = cx[i] + dx;
                if (ix < 0 || ix >= nx) continue;
                const int64_t tile = (iy / T) * ntx + ix / T;
                if (!listed[tile]) continue;
                if (!loaded[tile]) return -1;
                const uint8_t* c = mosaic + (iy * nx + ix) * P;
                tsum[i] += c[NB * D] * top_step;
                tn[i] += 1;
                if (!wv[i]) continue;
                bool some = false;
                for (int32_t b = 0; b < NB; ++b) {
                    m[b] = dec[c[b * D + d0[i]]] * (1 - t1) + dec[c[b * D + d1[i]]] * t1;
                    some |= m[b] != 0;
                }
                if (!some) continue;
                double* s = S + np_ * NA;
                for (int32_t a = 0; a < NA; ++a) {
                    double v = Bi[a] * m[0];
                    v *= v;
                    s[a] = v;
                }
                for (int32_t b = 1; b < NB; ++b) {
                    const double* Bb = Bi + b * NA;
                    for (int32_t a = 0; a < NA; ++a) {
                        double y = Bb[a] * m[b];
                        y *= y;
                        s[a] += y;
                    }
                }
                pair_cell[np_++] = i;
            }
        }
    }
    *npairs = np_;
    return i;
}
void cx_lee_best(int64_t npairs, const int64_t* pair_cell, const double* pw, int32_t NA, const double* consts,
                 double* bp, int64_t* bk, double* bc) {
    const double MAX_W = consts[0], EX_C = consts[1], BOOST = consts[2];
    double col[64];
    for (int64_t p = 0; p < npairs; ++p) {
        const double* s = pw + p * NA;
        for (int32_t a = 0; a < NA; ++a) {
            const double w = EX_C * s[a];
            const double tt = dmin(1.0, dmax(0.0, (w - 1.0) / 2.0));
            col[a] = dmin(MAX_W, w * (1.0 + BOOST * tt * tt * (3.0 - 2.0 * tt)));
        }
        int32_t kk = 0;
        for (int32_t a = 1; a < NA; ++a)
            if (col[kk] == col[kk] && (col[a] > col[kk] || col[a] != col[a])) kk = a;
        const int64_t i = pair_cell[p];
        if (col[kk] > bp[i]) {
            bp[i] = col[kk];
            bk[i] = kk;
            if (bc)
                for (int32_t a = 0; a < NA; ++a) bc[i * NA + a] = col[a];
        }
    }
}
int32_t cx_ridge_cells(int64_t nsel, const int64_t* cy, const int64_t* cx, int32_t r, int64_t ny, int64_t nx,
                       int32_t T, int64_t ntx, const uint8_t* listed, const uint8_t* loaded, const uint8_t* mosaic,
                       int32_t P, int32_t D, double fk, double rs, const double* Us, const double* nlow,
                       const int64_t* d0, const int64_t* d1, const double* t, double* total) {
    for (int64_t i = 0; i < nsel; ++i) {
        const double tw = t[i], U = Us[i], nn = nlow[i];
        const bool nok = nn > 0;
        double sum = 0.0;
        for (int32_t dy = -r; dy <= r; ++dy) {
            const int64_t iy = cy[i] + dy;
            if (iy < 0 || iy >= ny) continue;
            for (int32_t dx = -r; dx <= r; ++dx) {
                const int64_t ix = cx[i] + dx;
                if (ix < 0 || ix >= nx) continue;
                const int64_t tile = (iy / T) * ntx + ix / T;
                if (!listed[tile]) continue;
                if (!loaded[tile]) return -1;
                const uint8_t* c = mosaic + (iy * nx + ix) * P;
                const double f = (c[d0[i]] * (1 - tw) + c[d1[i]] * tw) * fk;
                if (!(f != 0)) continue;
                const double H = c[D] * rs;
                const double fr = (nok && H > 0) ? dmax(0.05, dmin(1.0, U / (nn * H))) : 1.0;
                sum += U * f * fr;
            }
        }
        total[i] = sum;
    }
    return 0;
}
}
