// Shared exact decode attention (attn_core/dense_decode.hpp): every attend form (main's schedule attend_main_kernel,
// attend_kernel<Dim, P, U>, the device plan) must give the bits of the frozen main kernel AS THE ENGINE BUILDS IT: v0
// and the inputs live in tests/hip_dense_decode_rows_v0.hip, always compiled with STRIX_HIP_EXACT_FP_OPTIONS. CMake
// builds this file twice: with those options (strix-hip-dense-decode-rows-tests) and with the HIP default flags
// (strix-hip-dense-decode-rows-default-tests); both must print PASS and the same v0 digest.
// Synthetic BF16 K/V (hash fill), Dim 256. Cases (pass = every arm BITWISE, memcmp of ctx):
//   grid      24 q / 4 kv, gated: rows 1/3/4/5/7/12/13/16 x len 1K / 32K / 33000 / 48K / 131K
//   tail      1 row at len 32768+1 .. 32768+15 (every residue of the two-positions-per-step walk)
//   short     1 row at len 1, 2, 7, 8, 9, 15, 16, 17, 24; 16 rows at len 16 (pos0 = 0)
//   rows      rows 2/6/8/9/10/11/14/15 x len 1K / 33000 / 48K
//   wide      rows 17 and 64 (past the table: the main schedule grid) x len 1K / 33000
//   shape     8 q / 2 kv and 10 q / 2 kv (P = 2, 3 straddle rows: per-pair causal skip, partial last block) x rows
//             1/3/5 x len 1K / 33000
//   ungated   q_stride = Dim, gated = false: rows 1/5/16 x len 1K / 33000
// --bench [LENS] [ROWS] [BUDGET_S]: no-model timing, kernel-only events per call, the 64 MiB cache-flush write between
// calls outside the events: ms per call of v0 (main's kernel, engine options) and every (P, U) form plus the device
// plan, for each context in LENS (comma list, default 8192,32768,131072) and row count in ROWS (default 1,2,4,8,12,16);
// cells past BUDGET_S seconds (default 80) are skipped. x16 = one Qwen 27B dense step.
#include <hip/hip_runtime.h>

#include "attn_core/dense_decode.hpp"
#include "hip_dense_decode_rows_v0.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

namespace dd = strix::attn_core::dense_decode;

namespace {
constexpr unsigned D = 256U, H = 24U, KV = 4U, G = H / KV, MAXR = 64U, MAXL = 131072U;

struct Bufs {
    float *qn, *qin, *ctx;
    std::uint16_t *k, *v;
    unsigned char* flush;
};
constexpr std::size_t flush_bytes = std::size_t(64) << 20U;

dd::Params params(unsigned rows, unsigned len, unsigned heads, unsigned kv, bool gated) {
    return dd::Params{rows, heads, kv, gated ? 2U * D : D, 64U, len - rows, 1e-6F, 0.0625F, 1e7, gated};
}
// arm: 0 = v0, 11 = the production main schedule (attend_main_kernel), other 10 * U + P = attend_kernel<D, P, U> (the
// forms the launch instantiates: P 1..3 at U 1, P 1..2 at U 2), 99 = the device plan (dd::attend)
template <unsigned P, unsigned U>
void one(const dd::Params& p, const Bufs& b, hipStream_t s) {
    const dim3 grid(strix::attn_row_share::blocks_for(p.rows, p.heads / p.kv_heads, P), p.kv_heads);
    dd::attend_kernel<D, P, U><<<grid, dim3(32U * dd::waves), 0, s>>>(p, b.qn, b.qin, b.k, b.v, b.ctx);
}
void run(int arm, const dd::Params& p, const Bufs& b, hipStream_t s) {
    switch (arm) {
    case 0: rows_v0::attend(rows_v0::Params{p.rows, p.heads, p.kv_heads, p.q_stride, p.rot_dim, p.pos0, p.eps, p.scale, p.theta, p.gated}, b.qn, b.qin, b.k, b.v, b.ctx, s); break;
    case 11: dd::attend_main_kernel<D><<<dim3(p.rows, p.heads), dim3(32U * dd::waves), 0, s>>>(p, b.qn, b.qin, b.k, b.v, b.ctx); break;
    case 12: one<2U, 1U>(p, b, s); break;
    case 13: one<3U, 1U>(p, b, s); break;
    case 21: one<1U, 2U>(p, b, s); break;
    case 22: one<2U, 2U>(p, b, s); break;
    default: dd::attend<D>(p, b.qn, b.qin, b.k, b.v, b.ctx, s); break;
    }
}
constexpr int arms[] = {0, 11, 12, 13, 21, 22, 99};
static_assert(sizeof(rows_v0::Params) == sizeof(dd::Params), "v0 Params mirrors dense_decode::Params");
strix::attn_row_share::Plan launch_plan(unsigned rows, unsigned group, unsigned len, unsigned kv = KV) {
    return strix::attn_row_share::plan_device(rows, group, len, dd::device_table(group, kv, D));
}
std::vector<unsigned> list(int argc, char** argv, int at, const char* fallback) {
    std::vector<unsigned> out;
    const std::string text = argc > at ? argv[at] : fallback;
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t j = std::min(text.find(',', i), text.size());
        out.push_back(unsigned(std::strtoul(text.substr(i, j - i).c_str(), nullptr, 10)));
        i = j + 1U;
    }
    return out;
}
struct Case {
    const char* set;
    unsigned rows, len, heads, kv;
    bool gated;
};
std::vector<Case> cases() {
    std::vector<Case> c;
    for (const unsigned len : {1024U, 32768U, 33000U, 49152U, MAXL})
        for (const unsigned rows : {1U, 3U, 4U, 5U, 7U, 12U, 13U, 16U})
            c.push_back({"grid", rows, len, H, KV, true});
    for (unsigned t = 1U; t <= 15U; ++t)
        c.push_back({"tail", 1U, 32768U + t, H, KV, true});
    for (const unsigned len : {1U, 2U, 7U, 8U, 9U, 15U, 16U, 17U, 24U})
        c.push_back({"short", 1U, len, H, KV, true});
    c.push_back({"short", 16U, 16U, H, KV, true});
    for (const unsigned len : {1024U, 33000U, 49152U})
        for (const unsigned rows : {2U, 6U, 8U, 9U, 10U, 11U, 14U, 15U})
            c.push_back({"rows", rows, len, H, KV, true});
    for (const unsigned len : {1024U, 33000U})
        for (const unsigned rows : {17U, 64U})
            c.push_back({"wide", rows, len, H, KV, true});
    for (const unsigned heads : {8U, 10U})
        for (const unsigned len : {1024U, 33000U})
            for (const unsigned rows : {1U, 3U, 5U})
                c.push_back({"shape", rows, len, heads, 2U, true});
    for (const unsigned len : {1024U, 33000U})
        for (const unsigned rows : {1U, 5U, 16U})
            c.push_back({"ungated", rows, len, H, KV, false});
    return c;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    Bufs b{};
    const std::size_t kv_elems = std::size_t(MAXL) * KV * D;
    CHECK(hipMalloc(&b.qn, MAXR * H * D * 4)); CHECK(hipMalloc(&b.qin, MAXR * H * 2 * D * 4)); CHECK(hipMalloc(&b.ctx, MAXR * H * D * 4));
    CHECK(hipMalloc(&b.k, kv_elems * 2)); CHECK(hipMalloc(&b.v, kv_elems * 2)); CHECK(hipMalloc(&b.flush, flush_bytes));
    rows_v0::fill(b.qn, MAXR * H * D, b.qin, MAXR * H * 2 * D, b.k, b.v, kv_elems);
    CHECK(hipDeviceSynchronize());
    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    {
        hipDeviceProp_t prop{};
        int dev = 0;
        CHECK(hipGetDevice(&dev));
        CHECK(hipGetDeviceProperties(&prop, dev));
        std::printf("device %s multiprocessors %d: row-share table %s\n", prop.gcnArchName, prop.multiProcessorCount,
                    dd::device_table(G, KV, D) != nullptr ? "found" : "NONE (every plan is the main schedule)");
    }
    const std::size_t out = std::size_t(MAXR) * H * D;
    std::vector<float> ref(out), got(out);
    int failures = 0;
    if (!bench) {
        std::uint64_t digest = 1469598103934665603ULL;  // FNV-1a over every case's v0 ctx bytes
        const auto all = cases();
        for (const Case& c : all) {
            const auto p = params(c.rows, c.len, c.heads, c.kv, c.gated);
            const std::size_t n = std::size_t(c.rows) * c.heads * D;
            CHECK(hipMemset(b.ctx, 0xff, out * 4));
            run(0, p, b, s);
            CHECK(hipStreamSynchronize(s));
            CHECK(hipMemcpy(ref.data(), b.ctx, n * 4, hipMemcpyDeviceToHost));
            const auto* bytes = reinterpret_cast<const unsigned char*>(ref.data());
            for (std::size_t i = 0; i < n * 4; ++i)
                digest = (digest ^ bytes[i]) * 1099511628211ULL;
            const auto plan = launch_plan(c.rows, c.heads / c.kv, c.len, c.kv);
            std::printf("%-7s %2u/%u len %6u rows %2u plan P%u%u:", c.set, c.heads, c.kv, c.len, c.rows, plan.walk, plan.pairs);
            for (const int arm : arms) {
                if (arm == 0)
                    continue;
                CHECK(hipMemset(b.ctx, 0xff, out * 4));
                run(arm, p, b, s);
                CHECK(hipStreamSynchronize(s));
                CHECK(hipMemcpy(got.data(), b.ctx, n * 4, hipMemcpyDeviceToHost));
                const bool same = std::memcmp(ref.data(), got.data(), n * 4) == 0;
                failures += same ? 0 : 1;
                std::printf(" %s%d=%s", arm == 99 ? "plan" : "P", arm == 99 ? int(plan.walk * 10U + plan.pairs) : arm, same ? "BITWISE" : "DIFF");
            }
            std::printf("\n");
        }
        std::printf("v0 digest %016llx over %zu cases\n", static_cast<unsigned long long>(digest), all.size());
        std::printf("%s: %d case(s) differ from v0\n", failures ? "FAIL" : "PASS", failures);
        return failures ? 1 : 0;
    }
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    std::printf("bench: ms per call (kernel-only, cache flushed between calls); step16 = 16 attention layers (Qwen 27B dense)\n");
    const auto lens = list(argc, argv, 2, "8192,32768,131072"), rowss = list(argc, argv, 3, "1,2,4,8,12,16");
    const double budget = argc > 4 ? std::atof(argv[4]) : 80.0;
    for (const unsigned len : lens)
        for (const unsigned rows : rowss) {
            if (len < rows || len > MAXL || rows < 1U || rows > 16U || elapsed() > budget) {
                std::printf("len %u rows %u SKIPPED (%s)\n", len, rows, elapsed() > budget ? "budget" : "range");
                continue;
            }
            const auto p = params(rows, len, H, KV, true);
            const auto plan = launch_plan(rows, G, len);
            std::printf("len %6u rows %2u:", len, rows);
            double v0ms = 0.0, planms = 0.0;
            for (const int arm : arms) {
                run(arm, p, b, s);  // warm-up (code object, first touch)
                float total = 0.0F;
                unsigned reps = 0;
                while (reps < 3U || (total < 200.0F && reps < 12U)) {
                    CHECK(hipMemsetAsync(b.flush, int(reps & 0xffU), flush_bytes, s));
                    CHECK(hipEventRecord(e0, s));
                    run(arm, p, b, s);
                    CHECK(hipEventRecord(e1, s));
                    CHECK(hipEventSynchronize(e1));
                    float ms = 0.0F;
                    CHECK(hipEventElapsedTime(&ms, e0, e1));
                    total += ms;
                    ++reps;
                }
                const double ms = double(total) / reps;
                if (arm == 0) v0ms = ms;
                if (arm == 99) planms = ms;
                if (arm == 0)
                    std::printf(" v0 %.3f", ms);
                else
                    std::printf(" %s%d %.3f", arm == 99 ? "plan" : "P", arm == 99 ? int(plan.walk * 10U + plan.pairs) : arm, ms);
            }
            std::printf(" | step16 v0 %.1f plan %.1f ms (x%.2f)\n", 16.0 * v0ms, 16.0 * planms, v0ms / planms);
            std::fflush(stdout);
        }
    return 0;
}
