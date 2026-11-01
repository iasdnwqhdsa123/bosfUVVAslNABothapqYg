#include <cmath>
#include <cstdint>
#include <cstdlib>
extern "C" {
void cx_downsample(const uint8_t* codes, int64_t n, int64_t ny, int64_t nx, int32_t k,
                   int32_t none, int32_t rms, uint8_t* out) {
    const int64_t my = (ny + k - 1) / k, mx = (nx + k - 1) / k;
    for (int64_t h = 0; h < n; h++) {
        const uint8_t* g = codes + h * ny * nx;
        uint8_t* o = out + h * my * mx;
        for (int64_t by = 0; by < my; by++) {
            const int64_t y0 = by * k, y1 = y0 + k < ny ? y0 + k : ny;
            for (int64_t bx = 0; bx < mx; bx++) {
                const int64_t x0 = bx * k, x1 = x0 + k < nx ? x0 + k : nx;
                uint32_t total = 0, valid = 0, m = 0;
                for (int64_t y = y0; y < y1; y++) {
                    const uint8_t* row = g + y * nx;
                    for (int64_t x = x0; x < x1; x++) {
                        const uint32_t c = row[x];
                        if (c == 255) continue;
                        valid++;
                        if (none >= 0 && c == (uint32_t)none) continue;
                        total += rms ? c * c : c;
                        m++;
                    }
                }
                const int64_t cells = (y1 - y0) * (x1 - x0);
                uint32_t mean;
                if (rms)
                    mean = (uint32_t)std::nearbyint(std::sqrt((double)total / (double)(m > 0 ? m : 1)));
                else
                    mean = (2 * total + m) / (2 * m > 0 ? 2 * m : 1);
                if (none >= 0 && 2 * (valid - m) > valid) mean = (uint32_t)none;
                o[by * mx + bx] = (valid > 0 && 2 * (int64_t)valid >= cells) ? (uint8_t)mean : 255;
            }
        }
    }
}
static inline uint32_t row_cost(const uint8_t* r, int64_t w) {
    uint32_t s = 0;
    for (int64_t x = 0; x < w; x++) {
        const int v = (int8_t)r[x];
        s += (uint32_t)(v < 0 ? -v : v);
    }
    return s;
}
void cx_png_filter(const uint8_t* img, int64_t h, int64_t w, uint8_t* out, uint8_t* scratch) {
    if (w <= 0) return;
    for (int64_t y = 0; y < h; y++) {
        const uint8_t* cur = img + y * w;
        const uint8_t* up = y > 0 ? img + (y - 1) * w : nullptr;
        uint8_t *r0 = scratch, *r1 = r0 + w, *r2 = r1 + w, *r3 = r2 + w, *r4 = r3 + w;
        for (int64_t x = 0; x < w; x++) r0[x] = cur[x];
        r1[0] = cur[0];
        for (int64_t x = 1; x < w; x++) r1[x] = (uint8_t)(cur[x] - cur[x - 1]);
        if (up) {
            for (int64_t x = 0; x < w; x++) r2[x] = (uint8_t)(cur[x] - up[x]);
            r3[0] = (uint8_t)(cur[0] - (up[0] >> 1));
            for (int64_t x = 1; x < w; x++) r3[x] = (uint8_t)(cur[x] - ((cur[x - 1] + up[x]) >> 1));
            r4[0] = (uint8_t)(cur[0] - up[0]);
            for (int64_t x = 1; x < w; x++) {
                const int a = cur[x - 1], b = up[x], c = up[x - 1];
                const int pa = b > c ? b - c : c - b, pb = a > c ? a - c : c - a;
                const int pc = (a + b - 2 * c) < 0 ? -(a + b - 2 * c) : (a + b - 2 * c);
                const int pred = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                r4[x] = (uint8_t)(cur[x] - pred);
            }
        } else {
            for (int64_t x = 0; x < w; x++) r2[x] = cur[x];
            r3[0] = cur[0];
            for (int64_t x = 1; x < w; x++) r3[x] = (uint8_t)(cur[x] - (cur[x - 1] >> 1));
            for (int64_t x = 0; x < w; x++) r4[x] = r1[x];
        }
        const uint8_t* rows[5] = {r0, r1, r2, r3, r4};
        uint32_t best = row_cost(r0, w);
        int bf = 0;
        for (int f = 1; f < 5; f++) {
            const uint32_t c = row_cost(rows[f], w);
            if (c < best) best = c, bf = f;
        }
        uint8_t* o = out + y * (w + 1);
        o[0] = (uint8_t)bf;
        for (int64_t x = 0; x < w; x++) o[1 + x] = rows[bf][x];
    }
}
}
