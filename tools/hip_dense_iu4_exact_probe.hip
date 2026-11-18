// Probe (measurement only, not linked into any engine): the EXACT IU4 dense core (src/core/dense_linear_iu4.hpp) over
// prefill windows. For each shape and window size it runs
//   sweep16  the incumbent: run() in 16-row sweeps (tail in powers of two, as core::dense_row_sweeps), one weight
//            stream per 8 rows (kernel<16, MaxG, 8>: two wave_rows<8> passes)
//   grid     run() over the whole window: one launch of 8-row blocks, row chunks of a column block adjacent
// on the same FP16 rows (prepare() once) and compares every stored F32 bitwise (memcmp of the window). Arms alternate
// (A B A B ...) over 4 weight copies (> 128 MB per shape: the weights do not stay in the 32 MB MALL between calls).
// Usage: hip-dense-iu4-exact-probe [reps=5]   prints one line per (shape, rows) and a final verdict line.
#include "core/dense_linear_iu4.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        const hipError_t e_ = (x);                                                               \
        if (e_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

namespace iu = strix::core::dense_iu4;

namespace {
struct Shape { unsigned n, k; const char* name; };
constexpr unsigned kCopies = 4U;

void sweep16(const std::uint8_t* w, unsigned k, unsigned n, const _Float16* xh, float* out, unsigned rows, hipStream_t s) {
    unsigned at = 0;
    for (unsigned tile = 16U; tile; tile /= 2U)
        while (rows - at >= tile) {
            iu::run(w, k, n, xh + std::size_t(at) * k, out + std::size_t(at) * n, tile, s);
            at += tile;
        }
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
    const Shape shapes[] = {{17408U, 5120U, "gate/up"}, {5120U, 17408U, "down"}, {5120U, 6144U, "out"}};
    const unsigned windows[] = {16U, 64U, 100U, 256U, 512U};
    constexpr unsigned kMaxWin = 512U;
    hipStream_t s;
    CK(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    std::mt19937 rng(1234U);
    bool all_equal = true;
    for (const Shape& sh : shapes) {
        if (!iu::supported(sh.n, sh.k)) { std::printf("EXACTPROBE %s unsupported\n", sh.name); continue; }
        const std::size_t wbytes = iu::tensor_bytes(sh.n, sh.k), packed = iu::packed_bytes(sh.n, sh.k);
        std::vector<std::uint8_t> hw(wbytes);
        for (std::size_t i = 0; i < packed; ++i) hw[i] = std::uint8_t(rng());
        std::uniform_real_distribution<float> sc(0.002F, 0.03F);
        for (std::size_t i = packed; i + 1U < wbytes; i += 2U) {
            const _Float16 v = static_cast<_Float16>(sc(rng));
            std::memcpy(&hw[i], &v, 2U);
        }
        std::uint8_t* dw = nullptr;
        CK(hipMalloc(&dw, wbytes * kCopies));
        for (unsigned c = 0; c < kCopies; ++c) CK(hipMemcpy(dw + c * wbytes, hw.data(), wbytes, hipMemcpyHostToDevice));
        std::vector<float> hx(std::size_t(kMaxWin) * sh.k);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        for (auto& v : hx) v = nd(rng);
        float *dx = nullptr, *da = nullptr, *db = nullptr;
        _Float16* dxh = nullptr;
        CK(hipMalloc(&dx, hx.size() * 4U));
        CK(hipMalloc(&dxh, iu::wide_xh_rows(kMaxWin) * sh.k * 2U));
        CK(hipMalloc(&da, std::size_t(kMaxWin) * sh.n * 4U));
        CK(hipMalloc(&db, std::size_t(kMaxWin) * sh.n * 4U));
        CK(hipMemcpy(dx, hx.data(), hx.size() * 4U, hipMemcpyHostToDevice));
        std::vector<float> ha(std::size_t(kMaxWin) * sh.n), hb(ha.size());
        for (unsigned rows : windows) {
            iu::prepare(dx, dxh, sh.k, rows, s);
            CK(hipMemsetAsync(da, 0xFF, std::size_t(rows) * sh.n * 4U, s));
            CK(hipMemsetAsync(db, 0xFF, std::size_t(rows) * sh.n * 4U, s));
            sweep16(dw, sh.k, sh.n, dxh, da, rows, s);
            iu::run(dw, sh.k, sh.n, dxh, db, rows, s);
            CK(hipStreamSynchronize(s));
            CK(hipMemcpy(ha.data(), da, std::size_t(rows) * sh.n * 4U, hipMemcpyDeviceToHost));
            CK(hipMemcpy(hb.data(), db, std::size_t(rows) * sh.n * 4U, hipMemcpyDeviceToHost));
            unsigned diff_rows = 0;
            for (unsigned r = 0; r < rows; ++r)
                if (std::memcmp(&ha[std::size_t(r) * sh.n], &hb[std::size_t(r) * sh.n], sh.n * 4U) != 0) ++diff_rows;
            all_equal = all_equal && diff_rows == 0U;
            double ta = 0.0, tb = 0.0;
            for (int i = 0; i < reps; ++i) {
                for (int arm = 0; arm < 2; ++arm) {
                    const std::uint8_t* w = dw + std::size_t((2 * i + arm) % kCopies) * wbytes;
                    CK(hipEventRecord(e0, s));
                    if (arm == 0) sweep16(w, sh.k, sh.n, dxh, da, rows, s);
                    else iu::run(w, sh.k, sh.n, dxh, db, rows, s);
                    CK(hipEventRecord(e1, s));
                    CK(hipEventSynchronize(e1));
                    float ms = 0.0F;
                    CK(hipEventElapsedTime(&ms, e0, e1));
                    if (i > 0 || reps == 1) (arm == 0 ? ta : tb) += ms;  // first pair = warm-up
                }
            }
            const double nrep = reps > 1 ? reps - 1 : 1, ua = 1000.0 * ta / nrep, ub = 1000.0 * tb / nrep;
            const double gmac = double(rows) * sh.n * sh.k * 1e-9;
            std::printf("EXACTPROBE %-7s n %5u k %5u rows %3u sweep16 %9.1f us (%7.2f us/row, %5.2f TMAC/s) grid %9.1f us (%7.2f us/row, %5.2f TMAC/s) "
                        "speed %.2fx bits %s (%u/%u rows differ)\n",
                        sh.name, sh.n, sh.k, rows, ua, ua / rows, gmac / ua * 1e3, ub, ub / rows, gmac / ub * 1e3, ua / ub,
                        diff_rows ? "DIFF" : "EQUAL", diff_rows, rows);
        }
        CK(hipFree(dw)); CK(hipFree(dx)); CK(hipFree(dxh)); CK(hipFree(da)); CK(hipFree(db));
    }
    std::printf("EXACTPROBE verdict: grid vs 16-row sweeps %s (scalar_x %d, row_reuse %d)\n", all_equal ? "BITWISE all shapes/windows" : "MISMATCH",
                int(iu::scalar_x()), int(iu::row_reuse()));
    return all_equal ? 0 : 1;
}
