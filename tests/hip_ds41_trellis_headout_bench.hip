// Trellis LM head adapter bench (STRIX_DS41_TRELLIS_HEADOUT; not a ctest, run explicitly on a GPU). One rank's 64,640-row slice of
// the pack's head through the engine adapter, as DeviceForward's Op::head + Op::argmax use it, per row count R = 1..8:
//   A  today: enqueue_trellis_linear into logits [8][129280] at the slice (R > 1: library -> workspace -> 2D copy);
//   B  enqueue_trellis_linear_rows into the same logits (R1 in place, R > 1 as A);
//   C  enqueue_trellis_linear_rows into a per-slice rows buffer [8][64640] (in place at every R);
//   D  C with R5..7 submitted as 8 rows in place (rows R..7 of x / of the rows buffer are scratch).
// Checks (bitwise, every row and value): B, C, D == A; D with NaN in x rows R..7 == A. Timing: median of hip events per
// call (head alone, and head + argmax over the written rows), cold (256 MB write between reps) and warm.
// usage: strix-hip-ds41-trellis-headout-bench PACK_DIR [rank=0] [reps=20]
#include "strix/ds41_trellis.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::ds41;

namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint32_t lcg(std::uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(s >> 33U);
}
std::vector<float> make_x(std::size_t rows, std::size_t width, std::uint64_t seed) {
    std::vector<float> x(rows * width);
    for (auto& v : x) {
        float a = 0;
        for (int k = 0; k < 4; ++k) a += static_cast<float>(lcg(seed) % 20001U) / 10000.0F - 1.0F;
        if (lcg(seed) % 997U == 0U) a *= 16.0F;
        v = a;
    }
    return x;
}
double median_us(int reps, hipStream_t s, void* flush, std::size_t flush_bytes, const std::function<void()>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    std::vector<double> t;
    for (int i = 0; i < reps + 2; ++i) {
        if (flush) check(hipMemsetAsync(flush, i & 0xff, flush_bytes, s), "flush");
        check(hipEventRecord(a, s), "record");
        launch();
        check(hipEventRecord(b, s), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        if (i >= 2) t.push_back(ms * 1000.0);
    }
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s PACK_DIR [rank=0] [reps=20]\n", argv[0]);
        return 2;
    }
    try {
        const std::uint32_t rank = argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 0U;
        const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
        constexpr std::uint32_t V = 129280U, H = 5120U, W = 8U, pad_from = 5U;
        const auto sl = vocab_slice(V, 2U, rank);
        TrellisPack pack(argv[1]);
        TrellisContext ctx;
        hipStream_t st;
        check(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
        const auto& site = ctx.prepare(pack, "head", {TrellisSlice::Kind::rows, sl.first, sl.count});
        ctx.reserve(st, W);
        std::printf("head rank %u slice [%u, +%u): N %u x I %u, K %u, %.1f MB; headout env %d\n", rank, sl.first, sl.count, site.rows,
                    site.columns, site.bits, static_cast<double>(site.device_bytes) / 1e6, trellis_headout_enabled() ? 1 : 0);
        const auto x = make_x(W, H, 0xbeefULL + rank);
        float *dx = nullptr, *dlog = nullptr, *drows = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&dx), std::size_t{W} * H * 4U), "x");
        check(hipMalloc(reinterpret_cast<void**>(&dlog), std::size_t{W} * V * 4U), "logits");
        check(hipMalloc(reinterpret_cast<void**>(&drows), std::size_t{W} * sl.count * 4U), "rows");
        ArgmaxCandidate *partial = nullptr, *cand = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&partial), std::size_t{W} * argmax_partial_blocks(sl.count) * sizeof(ArgmaxCandidate)), "partial");
        check(hipMalloc(reinterpret_cast<void**>(&cand), std::size_t{W} * sizeof(ArgmaxCandidate)), "cand");
        const std::size_t flush_bytes = std::size_t{256} << 20U;
        void* flush = nullptr;
        check(hipMalloc(&flush, flush_bytes), "flush");
        const auto upload_x = [&](std::uint32_t R, bool nan_tail) {
            std::vector<float> h(x);
            if (nan_tail)
                for (std::size_t i = std::size_t{R} * H; i < h.size(); ++i) h[i] = std::numeric_limits<float>::quiet_NaN();
            check(hipMemcpy(dx, h.data(), h.size() * 4U, hipMemcpyHostToDevice), "x upload");
        };
        const auto fetch = [&](const float* p, std::size_t stride, std::uint32_t R) {
            std::vector<float> h(std::size_t{R} * sl.count);
            check(hipStreamSynchronize(st), "sync");
            check(hipMemcpy2D(h.data(), std::size_t{sl.count} * 4U, p, stride * 4U, std::size_t{sl.count} * 4U, R, hipMemcpyDeviceToHost),
                  "fetch");
            return h;
        };
        const auto run = [&](char v, std::uint32_t R) {
            switch (v) {
            case 'A': enqueue_trellis_linear(site, dx, dlog + sl.first, R, H, V, false, st); break;
            case 'B': enqueue_trellis_linear_rows(site, dx, dlog + sl.first, R, H, V, W, 0U, st); break;
            case 'C': enqueue_trellis_linear_rows(site, dx, drows, R, H, sl.count, W, 0U, st); break;
            default: enqueue_trellis_linear_rows(site, dx, drows, R, H, sl.count, W, pad_from, st); break;
            }
        };
        const auto out = [&](char v) -> std::pair<const float*, std::size_t> {
            return v == 'A' || v == 'B' ? std::pair<const float*, std::size_t>{dlog + sl.first, V} : std::pair<const float*, std::size_t>{drows, sl.count};
        };
        std::size_t compared = 0, differ = 0;
        for (std::uint32_t R = 1; R <= W; ++R) {
            upload_x(R, false);
            check(hipMemset(dlog, 0xff, std::size_t{W} * V * 4U), "poison");
            run('A', R);
            const auto ref = fetch(dlog + sl.first, V, R);
            for (const char v : {'B', 'C', 'D', 'N'}) {
                upload_x(R, v == 'N');
                check(hipMemset(dlog, 0xff, std::size_t{W} * V * 4U), "poison");
                check(hipMemset(drows, 0xff, std::size_t{W} * sl.count * 4U), "poison");
                run(v == 'N' ? 'D' : v, R);
                const auto [p, s] = out(v == 'N' ? 'D' : v);
                const auto got = fetch(p, s, R);
                std::size_t bad = 0;
                for (std::size_t i = 0; i < got.size(); ++i) bad += std::memcmp(&got[i], &ref[i], 4) != 0 ? 1U : 0U;
                ++compared;
                differ += bad != 0U ? 1U : 0U;
                if (bad != 0U) std::printf("DIFFER R%u variant %c: %zu values\n", R, v, bad);
            }
        }
        std::printf("EXACTNESS %s: %zu comparisons (R1..8 x B,C,D,D+NaN tail vs A), %zu differ\n", differ == 0U ? "PASS" : "FAIL", compared, differ);
        upload_x(W, false);
        std::printf("%-3s %-8s %10s %10s %10s %10s   %10s %10s %10s %10s\n", "R", "mode", "A us", "B us", "C us", "D us", "A+am", "B+am",
                    "C+am", "D+am");
        for (const bool cold : {true, false}) {
            for (std::uint32_t R = 1; R <= W; ++R) {
                double t[4], ta[4];
                int k = 0;
                for (const char v : {'A', 'B', 'C', 'D'}) {
                    t[k] = median_us(reps, st, cold ? flush : nullptr, flush_bytes, [&] { run(v, R); });
                    const auto [p, s] = out(v);
                    ta[k] = median_us(reps, st, cold ? flush : nullptr, flush_bytes, [&] {
                        run(v, R);
                        enqueue_argmax(p, R, sl.count, static_cast<std::uint32_t>(s), sl.first, partial, cand, st);
                    });
                    ++k;
                }
                std::printf("%-3u %-8s %10.1f %10.1f %10.1f %10.1f   %10.1f %10.1f %10.1f %10.1f\n", R, cold ? "cold" : "warm", t[0], t[1], t[2],
                            t[3], ta[0], ta[1], ta[2], ta[3]);
            }
        }
        std::fflush(stdout);
        return differ == 0U ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
