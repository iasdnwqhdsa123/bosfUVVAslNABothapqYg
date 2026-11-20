#include <cmath>
#include <cstdint>
extern "C" void gbm_predict(int64_t n, const double* X, int32_t ntrees, const int32_t* roots, const int32_t* feat,
                            const double* thr, const uint8_t* ml, const int32_t* lt, const int32_t* rt,
                            const double* val, const uint8_t* leaf, double base, double* out) {
  for (int64_t j = 0; j < n; j++) {
    double s = base;
    for (int32_t t = 0; t < ntrees; t++) {
      int32_t i = roots[t];
      while (!leaf[i]) {
        const double v = X[(int64_t)feat[i] * n + j];
        i = std::isnan(v) ? (ml[i] ? lt[i] : rt[i]) : (v <= thr[i] ? lt[i] : rt[i]);
      }
      s += val[i];
    }
    out[j] = s;
  }
}
