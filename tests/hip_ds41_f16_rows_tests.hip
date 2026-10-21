// F16 rows with 16-byte loads (enqueue_f16_rows_lanes) against the host f16_rows and the site kernel enqueue_f16_rows,
// bit for bit, on the decode F16 shapes at 1..8 rows; STRIX_DS41_F16_BENCH=1 adds cold-weight timings of both kernels.
#include "strix/hip_ds41_f16_rows.hpp"

#include "strix/ds41_formats.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_packed.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct Shape { const char* name; std::uint32_t width, count; };

void run(std::mt19937& rng, const Shape& s, bool bench) {
    const std::size_t per = static_cast<std::size_t>(s.width) * s.count;
    // enough copies that the timed weights do not stay in the 32 MB last-level cache
    const std::size_t copies = bench ? std::max<std::size_t>(1, (256ULL << 20) / (per * 2) + 1) : 1;
    std::vector<std::uint16_t> w(per);
    std::normal_distribution<float> nd(0.0F, 0.05F);
    for (auto& h : w) {
        const _Float16 f = static_cast<_Float16>(nd(rng));
        h = __builtin_bit_cast(std::uint16_t, f);
    }
    for (std::size_t i = 0; i < per; i += 977) w[i] = static_cast<std::uint16_t>(0x0001U + (i % 7));  // subnormals
    std::uint16_t* dw = dev<std::uint16_t>(per * copies);
    for (std::size_t c = 0; c < copies; ++c) check(hipMemcpy(dw + c * per, w.data(), per * 2, hipMemcpyHostToDevice), "upload w");
    const std::uint32_t stride = s.width + 64;
    std::vector<float> x(static_cast<std::size_t>(8) * stride);
    std::uniform_real_distribution<float> ud(-2.0F, 2.0F);
    for (auto& v : x) v = ud(rng);
    float* dx = dev<float>(x.size());
    check(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "upload x");
    float* d1 = dev<float>(static_cast<std::size_t>(8) * s.count);
    float* d2 = dev<float>(static_cast<std::size_t>(8) * s.count);
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 8U}) {
        F16RowsView v;
        v.weight = dw; v.width = s.width; v.count = s.count; v.x = dx; v.rows = rows; v.x_stride = stride; v.out_stride = s.count;
        v.out = d1;
        enqueue_f16_rows_lanes(v, nullptr);
        F16RowsView p = v;
        p.out = d2;
        enqueue_f16_rows(p, nullptr);
        check(hipDeviceSynchronize(), "run");
        std::vector<float> g1(static_cast<std::size_t>(rows) * s.count), g2(g1.size()), h(s.count);
        check(hipMemcpy(g1.data(), d1, g1.size() * 4, hipMemcpyDeviceToHost), "down");
        check(hipMemcpy(g2.data(), d2, g2.size() * 4, hipMemcpyDeviceToHost), "down");
        bool same_site = std::memcmp(g1.data(), g2.data(), g1.size() * 4) == 0, same_host = true;
        for (std::uint32_t r = 0; r < rows && same_host; r += (rows > 2 ? rows - 1 : 1)) {   // host on the first and last row
            f16_rows(w.data(), s.width, 0, s.count, &x[static_cast<std::size_t>(r) * stride], h.data());
            same_host = std::memcmp(h.data(), &g1[static_cast<std::size_t>(r) * s.count], s.count * 4) == 0;
        }
        if (!same_site || !same_host) { ++failures; std::fprintf(stderr, "FAIL %s rows %u (site %d host %d)\n", s.name, rows, same_site, same_host); }
        if (!bench) continue;
        hipEvent_t e0, e1;
        check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
        const auto timed = [&](auto launch) {
            for (int i = 0; i < 20; ++i) launch(i);
            check(hipEventRecord(e0), "rec");
            const int iters = 60;
            for (int i = 0; i < iters; ++i) launch(i);
            check(hipEventRecord(e1), "rec"); check(hipEventSynchronize(e1), "sync");
            float ms = 0; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            return ms / iters;
        };
        const double ms_l = timed([&](int i) { F16RowsView t = v; t.weight = dw + (static_cast<std::size_t>(i) % copies) * per; enqueue_f16_rows_lanes(t, nullptr); });
        const double ms_p = timed([&](int i) { F16RowsView t = p; t.weight = dw + (static_cast<std::size_t>(i) % copies) * per; enqueue_f16_rows(t, nullptr); });
        const double gb = static_cast<double>(per) * 2 / 1e9;
        std::printf("  %-28s rows %u: lanes %8.1f us (%5.1f GB/s)  site %8.1f us (%5.1f GB/s)\n", s.name, rows, ms_l * 1e3, gb / (ms_l * 1e-3),
                    ms_p * 1e3, gb / (ms_p * 1e-3));
        check(hipEventDestroy(e0), "event"); check(hipEventDestroy(e1), "event");
    }
    std::printf("%s: lanes == site kernel == host f16_rows\n", s.name);
    (void)hipFree(dw); (void)hipFree(dx); (void)hipFree(d1); (void)hipFree(d2);
}
// F8 rows (MX-FP8-valued F16 weights packed by ds41_packed): the lanes kernel reading them equals the F16 rows bit for
// bit at every row count and form; with bench, cold timings of both (the F8 rows are 0.52x the bytes).
void run_f8(std::mt19937& rng, const Shape& s, bool bench) {
    const std::size_t per = static_cast<std::size_t>(s.width) * s.count;
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
    const auto f8 = pack_f16_as_f8(w, s.width);
    if (!f8) { ++failures; std::fprintf(stderr, "FAIL %s: F8 packing refused\n", s.name); return; }
    const std::size_t copies = bench ? std::max<std::size_t>(1, (256ULL << 20) / (per * 2) + 1) : 1;
    std::uint16_t* dw = dev<std::uint16_t>(per * copies);
    std::uint8_t* d8 = dev<std::uint8_t>(f8->size() * copies);
    for (std::size_t c = 0; c < copies; ++c) {
        check(hipMemcpy(dw + c * per, w.data(), per * 2, hipMemcpyHostToDevice), "upload w");
        check(hipMemcpy(d8 + c * f8->size(), f8->data(), f8->size(), hipMemcpyHostToDevice), "upload f8");
    }
    const std::uint32_t stride = s.width + 64;
    std::vector<float> x(static_cast<std::size_t>(8) * stride);
    std::uniform_real_distribution<float> ud(-2.0F, 2.0F);
    for (auto& v : x) v = ud(rng);
    float* dx = dev<float>(x.size());
    check(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "upload x");
    float* d1 = dev<float>(static_cast<std::size_t>(8) * s.count);
    float* d2 = dev<float>(static_cast<std::size_t>(8) * s.count);
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 8U}) {
        F16RowsView v;
        v.weight = dw; v.width = s.width; v.count = s.count; v.x = dx; v.rows = rows; v.x_stride = stride; v.out_stride = s.count;
        v.out = d1;
        enqueue_f16_rows_lanes(v, nullptr);
        F16RowsView q = v;
        q.weight = nullptr;
        q.weight_f8 = d8;
        q.out = d2;
        enqueue_f16_rows_lanes(q, nullptr);
        check(hipDeviceSynchronize(), "run");
        std::vector<float> g1(static_cast<std::size_t>(rows) * s.count), g2(g1.size());
        check(hipMemcpy(g1.data(), d1, g1.size() * 4, hipMemcpyDeviceToHost), "down");
        check(hipMemcpy(g2.data(), d2, g2.size() * 4, hipMemcpyDeviceToHost), "down");
        if (std::memcmp(g1.data(), g2.data(), g1.size() * 4) != 0) { ++failures; std::fprintf(stderr, "FAIL %s F8 rows %u\n", s.name, rows); }
        if (!bench) continue;
        hipEvent_t e0, e1;
        check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
        const auto timed = [&](auto launch) {
            for (int i = 0; i < 20; ++i) launch(i);
            check(hipEventRecord(e0), "rec");
            const int iters = 60;
            for (int i = 0; i < iters; ++i) launch(i);
            check(hipEventRecord(e1), "rec"); check(hipEventSynchronize(e1), "sync");
            float ms = 0; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            return ms / iters;
        };
        const double ms_h = timed([&](int i) { F16RowsView t = v; t.weight = dw + (static_cast<std::size_t>(i) % copies) * per; enqueue_f16_rows_lanes(t, nullptr); });
        const double ms_8 = timed([&](int i) { F16RowsView t = q; t.weight_f8 = d8 + (static_cast<std::size_t>(i) % copies) * f8->size(); enqueue_f16_rows_lanes(t, nullptr); });
        std::printf("  %-28s rows %u: F16 %8.1f us (%5.1f GB/s)  F8 %8.1f us (%5.1f GB/s)\n", s.name, rows, ms_h * 1e3,
                    static_cast<double>(per) * 2 / 1e6 / ms_h, ms_8 * 1e3, static_cast<double>(f8->size()) / 1e6 / ms_8);
        check(hipEventDestroy(e0), "event"); check(hipEventDestroy(e1), "event");
    }
    std::printf("%s: F8 rows == F16 rows (lanes)\n", s.name);
    (void)hipFree(dw); (void)hipFree(d8); (void)hipFree(dx); (void)hipFree(d1); (void)hipFree(d2);
}
}  // namespace

int main() {
    setenv("STRIX_DS41_F16_LANES", "0", 1);   // enqueue_f16_rows = the site kernel, for the comparison
    const bool bench = std::getenv("STRIX_DS41_F16_BENCH") != nullptr;
    std::mt19937 rng(41016);
    const Shape shapes[] = {{"engram kv 6144->25600", 6144, 25600}, {"compressor 5120->512", 5120, 512},
                            {"indexer q_b 1536->4096", 1536, 4096}, {"indexer proj 5120->32", 5120, 32},
                            {"odd count 1024->545", 1024, 545}};
    for (const auto& s : shapes) run(rng, s, bench);
    for (const auto& s : {Shape{"engram kv F8 6144->25600", 6144, 25600}, Shape{"indexer q_b F8 1280->4096", 1280, 4096}}) run_f8(rng, s, bench);
    std::printf("%s\n", failures == 0 ? "F16 rows (lanes) tests pass" : "FAILED");
    return failures == 0 ? 0 : 1;
}
