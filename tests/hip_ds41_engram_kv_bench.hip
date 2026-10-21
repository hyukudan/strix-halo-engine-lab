// Engram kv projection micro-benchmark (not a ctest): the engine's `engram` op at the decode shapes (F8 rows of
// engram_kv 6144 -> 25600, 162 MB, hipMalloc'd like the engine's Dev buffers, cold: rotating copies) at 1..8 rows.
// Checks the fast F8 kernel (STRIX_DS41_ENGRAM_FAST) bit for bit against enqueue_f16_rows on F8 rows and the F16 rows,
// then times: the current kernel, the fast variants (16-byte loads in flight x dequant), and the whole op
// (row decode + bf16 rounding + projection + injection) with either kernel.
// Round 2: the compact 1-row kernel (codes in LDS, linear stream per wave) and the RN kernels (exact 2..8 rows) are
// checked bit for bit against the site kernel, then swept (cold).
// usage: strix-hip-ds41-engram-kv-bench [iters] [all|r1|rn|old]
#include "strix/hip_ds41_f16_rows.hpp"
#include "strix/hip_ds41_f8_rows_fast.hpp"
#include "strix/hip_ds41_layer.hpp"

#include "strix/ds41_formats.hpp"
#include "strix/ds41_packed.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace strix::ds41;
int failures = 0;
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc"); return p; }
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 40;
    const std::string mode = argc > 2 ? argv[2] : "all";
    const bool do_old = mode == "all" || mode == "old", do_r1 = mode == "all" || mode == "r1", do_rn = mode == "all" || mode == "rn";
    constexpr std::uint32_t width = 6144, count = 25600, hidden = 5120, cols = 24, copies = 3;
    const std::size_t per = std::size_t{width} * count;
    std::mt19937 rng(41016);
    std::vector<std::uint16_t> w(per);
    std::uniform_int_distribution<int> code(0, 255), kd(-14, -8);
    for (std::size_t b = 0; b < per / 32U; ++b) {
        const int k = kd(rng);
        for (std::size_t j = 0; j < 32U; ++j) {
            auto c = static_cast<std::uint8_t>(code(rng));
            if ((c & 0x7fU) == 0x7fU) c = 0x80U;
            w[b * 32U + j] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(std::ldexp(e4m3_to_float(c), k)));
        }
    }
    const auto f8 = pack_f16_as_f8(w, width);
    if (!f8) { std::fprintf(stderr, "F8 packing refused\n"); return 2; }
    const std::size_t fb = f8->size();
    std::uint8_t* d8 = dev<std::uint8_t>(fb * copies);
    for (std::uint32_t c = 0; c < copies; ++c) check(hipMemcpy(d8 + c * fb, f8->data(), fb, hipMemcpyHostToDevice), "upload f8");
    std::uint16_t* dw = dev<std::uint16_t>(per);
    check(hipMemcpy(dw, w.data(), per * 2, hipMemcpyHostToDevice), "upload f16");
    std::vector<std::uint16_t>().swap(w);

    const std::uint32_t stride = width;  // the engine: x = m.emb, x_stride = width
    std::vector<float> x(std::size_t{8} * stride);
    std::uniform_real_distribution<float> ud(-2.0F, 2.0F);
    for (auto& e : x) e = ud(rng);
    float* dx = dev<float>(x.size());
    check(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "upload x");
    float* d1 = dev<float>(std::size_t{8} * count);
    float* d2 = dev<float>(std::size_t{8} * count);
    float* d3 = dev<float>(std::size_t{8} * count);

    // ---- exactness: fast (dequant 0 and 1, rows per wave 2/4/8) == enqueue_f16_rows (F8) == F16 rows lanes, every row count
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        F16RowsView v;
        v.weight_f8 = d8; v.width = width; v.count = count; v.x = dx; v.rows = rows; v.x_stride = stride; v.out_stride = count; v.out = d1;
        enqueue_f16_rows(v, nullptr);
        F16RowsView h = v;
        h.weight_f8 = nullptr; h.weight = dw; h.out = d3;
        enqueue_f16_rows_lanes(h, nullptr);
        check(hipDeviceSynchronize(), "ref");
        std::vector<float> g1(std::size_t{rows} * count), g2(g1.size()), g3(g1.size());
        check(hipMemcpy(g1.data(), d1, g1.size() * 4, hipMemcpyDeviceToHost), "down");
        check(hipMemcpy(g3.data(), d3, g3.size() * 4, hipMemcpyDeviceToHost), "down");
        if (std::memcmp(g1.data(), g3.data(), g1.size() * 4) != 0) { ++failures; std::fprintf(stderr, "FAIL rows %u: F8 site != F16 lanes\n", rows); }
        for (const unsigned rpw : {2U, 4U, 8U})
            for (const unsigned deq : {0U, 1U}) {
                F16RowsView q = v;
                q.out = d2;
                check(hipMemset(d2, 0xff, g2.size() * 4), "memset");
                enqueue_f8_rows_fast_variant(q, rpw, deq, nullptr);
                check(hipDeviceSynchronize(), "fast");
                check(hipMemcpy(g2.data(), d2, g2.size() * 4, hipMemcpyDeviceToHost), "down");
                if (std::memcmp(g1.data(), g2.data(), g1.size() * 4) != 0) {
                    ++failures;
                    std::fprintf(stderr, "FAIL rows %u rpw %u dequant %u: fast != site\n", rows, rpw, deq);
                }
            }
    }
    std::printf("exactness (fast == site F8 == F16 lanes, rows 1..8, rows per wave 2/4/8, dequant 0/1): %s\n", failures ? "FAIL" : "PASS");

    // ---- round 2 exactness: every compact (1 row) and RN (2..8 rows) variant == the site kernel, bit for bit
    struct Rn { unsigned cl, k, w, a; };
    std::vector<Rn> rn_all;
    for (const unsigned cl : {8U, 16U})
        for (const unsigned k : {1U, 2U})
            for (const unsigned nw : {4U, 8U})
                for (const unsigned a : {4U, 8U}) rn_all.push_back({cl, k, nw, a});
    const int f0 = failures;
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        F16RowsView v;
        v.weight_f8 = d8; v.width = width; v.count = count; v.x = dx; v.rows = rows; v.x_stride = stride; v.out_stride = count; v.out = d1;
        enqueue_f16_rows(v, nullptr);
        check(hipDeviceSynchronize(), "ref");
        std::vector<float> g1(std::size_t{rows} * count), g2(g1.size());
        check(hipMemcpy(g1.data(), d1, g1.size() * 4, hipMemcpyDeviceToHost), "down");
        const auto cmp = [&](const char* what) {
            check(hipDeviceSynchronize(), what);
            check(hipMemcpy(g2.data(), d2, g2.size() * 4, hipMemcpyDeviceToHost), "down");
            if (std::memcmp(g1.data(), g2.data(), g1.size() * 4) != 0) { ++failures; std::fprintf(stderr, "FAIL rows %u: %s != site\n", rows, what); }
        };
        F16RowsView q = v;
        q.out = d2;
        if (rows == 1) {
            for (const unsigned nw : {4U, 8U})
                for (const unsigned rpw : {4U, 8U, 16U})
                    for (const unsigned d : {2U, 4U, 8U}) {
                        check(hipMemset(d2, 0xff, g2.size() * 4), "memset");
                        enqueue_f8_rows_compact_variant(q, nw, rpw, d, nullptr);
                        char what[64];
                        std::snprintf(what, sizeof what, "compact w%u rpw%u d%u", nw, rpw, d);
                        cmp(what);
                    }
        } else {
            for (const Rn& c : rn_all) {
                check(hipMemset(d2, 0xff, g2.size() * 4), "memset");
                if (!enqueue_f8_rows_rn_variant(q, c.cl, c.k, c.w, c.a, nullptr)) continue;
                char what[64];
                std::snprintf(what, sizeof what, "rn cl%u k%u w%u a%u", c.cl, c.k, c.w, c.a);
                cmp(what);
            }
        }
    }
    std::printf("exactness round 2 (compact rows 1: waves 4/8 x rows per wave 4/8/16 x depth 2/4/8; rn rows 2..8: every shape): %s\n",
                failures == f0 ? "PASS" : "FAIL");
    std::fflush(stdout);

    hipEvent_t e0, e1;
    check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
    const auto timed = [&](const std::function<void(int)>& launch) {
        for (int i = 0; i < 6; ++i) launch(i);
        check(hipEventRecord(e0), "rec");
        for (int i = 0; i < iters; ++i) launch(i);
        check(hipEventRecord(e1), "rec"); check(hipEventSynchronize(e1), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return static_cast<double>(ms) / iters;
    };
    const double gb = static_cast<double>(fb) / 1e9;
    const auto view = [&](std::uint32_t rows, int i) {
        F16RowsView v;
        v.weight_f8 = d8 + (static_cast<std::size_t>(i) % copies) * fb; v.width = width; v.count = count; v.x = dx; v.rows = rows;
        v.x_stride = stride; v.out_stride = count; v.out = d1;
        return v;
    };
    // ---- kernel alone: current vs fast variants (cold weights: 3 rotating 162 MB copies)
    if (do_r1) {
        const double base = timed([&](int i) { enqueue_f16_rows(view(1, i), nullptr); });
        const double r1 = timed([&](int i) { enqueue_f8_rows_fast_variant(view(1, i), 4U, 1U, nullptr); });
        std::printf("rows 1  site(F8)            %8.1f us  %6.1f GB/s\n", base * 1e3, gb / (base * 1e-3));
        std::printf("rows 1  fast r1 (rpw4 deq1) %8.1f us  %6.1f GB/s  (%.2fx)\n", r1 * 1e3, gb / (r1 * 1e-3), base / r1);
        for (const unsigned nw : {4U, 8U})
            for (const unsigned rpw : {4U, 8U, 16U})
                for (const unsigned d : {2U, 4U, 8U}) {
                    const double t = timed([&](int i) { enqueue_f8_rows_compact_variant(view(1, i), nw, rpw, d, nullptr); });
                    std::printf("rows 1  compact w%u rpw%2u d%u   %8.1f us  %6.1f GB/s  (%.2fx site, %.3fx r1)\n", nw, rpw, d, t * 1e3,
                                gb / (t * 1e-3), base / t, r1 / t);
                }
        std::fflush(stdout);
    }
    if (do_rn)
        for (const std::uint32_t rows : {2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
            const double base = timed([&](int i) { enqueue_f16_rows(view(rows, i), nullptr); });
            std::printf("rows %u  site(F8)            %8.1f us  %6.1f GB/s\n", rows, base * 1e3, gb / (base * 1e-3));
            for (const Rn& c : rn_all) {
                if (!enqueue_f8_rows_rn_variant(view(rows, 0), c.cl, c.k, c.w, c.a, nullptr)) continue;
                const double t = timed([&](int i) { enqueue_f8_rows_rn_variant(view(rows, i), c.cl, c.k, c.w, c.a, nullptr); });
                std::printf("rows %u  rn cl%2u k%u w%u a%u    %8.1f us  %6.1f GB/s  (%.2fx)\n", rows, c.cl, c.k, c.w, c.a, t * 1e3,
                            gb / (t * 1e-3), base / t);
            }
            std::fflush(stdout);
        }
    if (do_old)
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 6U, 8U}) {
        const double base = timed([&](int i) { enqueue_f16_rows(view(rows, i), nullptr); });
        std::printf("rows %u  site(F8)            %8.1f us  %6.1f GB/s\n", rows, base * 1e3, gb / (base * 1e-3));
        for (const unsigned deq : {1U, 0U, 2U})
            for (const unsigned rpw : {2U, 4U, 8U}) {
                const double t = timed([&](int i) { enqueue_f8_rows_fast_variant(view(rows, i), rpw, deq, nullptr); });
                std::printf("rows %u  fast rpw %u deq %u%s %8.1f us  %6.1f GB/s  (%.2fx)\n", rows, rpw, deq, deq == 2 ? " roof" : "     ",
                            t * 1e3, gb / (t * 1e-3), base / t);
            }
        std::fflush(stdout);
    }
    // ---- the whole engine op (decode rows + bf16 rounding, projection, injection) at 1 and 6 rows
    std::uint8_t* erows = dev<std::uint8_t>(std::size_t{8} * cols * 264U);
    check(hipMemset(erows, 0, std::size_t{8} * cols * 264U), "memset");
    float* emb = dev<float>(std::size_t{8} * cols * 256U);
    float* hc = dev<float>(std::size_t{8} * 4U * hidden);
    check(hipMemset(hc, 0, std::size_t{8} * 4U * hidden * 4U), "memset");
    std::vector<float> ones(hidden, 1.0F);
    float* qn = dev<float>(hidden);
    float* kn = dev<float>(hidden);
    check(hipMemcpy(qn, ones.data(), hidden * 4, hipMemcpyHostToDevice), "up");
    check(hipMemcpy(kn, ones.data(), hidden * 4, hipMemcpyHostToDevice), "up");
    for (const std::uint32_t rows : {1U, 6U}) {
        for (const bool fast : {false, true}) {
            const double t = timed([&](int i) {
                enqueue_engram_rows_bf16(erows, rows * cols, emb, nullptr);
                F16RowsView v = view(rows, i);
                v.x = emb;
                if (fast) enqueue_f8_rows_fast(v, nullptr);
                else enqueue_f16_rows(v, nullptr);
                enqueue_engram_inject(hc, d1, qn, kn, rows, hidden, 1e-20F, nullptr);
            });
            const double k = timed([&](int i) {
                F16RowsView v = view(rows, i);
                v.x = emb;
                if (fast) enqueue_f8_rows_fast(v, nullptr);
                else enqueue_f16_rows(v, nullptr);
            });
            std::printf("op rows %u %s: whole op %8.1f us (%6.1f GB/s), projection alone %8.1f us, decode+bf16+inject %6.1f us\n", rows,
                        fast ? "fast" : "site", t * 1e3, gb / (t * 1e-3), k * 1e3, (t - k) * 1e3);
        }
    }
    std::fflush(stdout);
    std::printf("%s\n", failures == 0 ? "engram kv bench: exact" : "engram kv bench: FAILED");
    return failures == 0 ? 0 : 1;
}
