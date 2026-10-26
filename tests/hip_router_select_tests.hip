// Prefill router selection: STRIX_PREFILL_ROUTER_SELECT=1 form (one wave per row) vs the block-per-row form, bitwise
// (indices and weight bits of every row) over generated logits; then a bench at T 1024 / 8192 (E 512, top-10).
// usage: strix-hip-router-select-tests [bench_reps]
#include "strix/hip_moe_route_launcher.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const unsigned reps = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 50U;
    unsigned failed = 0, cases = 0;
    double rows_total = 0;
    for (unsigned flavor = 0; flavor <= 5U; ++flavor)
        for (const unsigned experts : {512U, 505U, 64U})
            for (const unsigned top_k : {10U, 1U, 32U, 8U})
                for (const bool normalize : {true, false}) {
                    if (top_k > experts) continue;
                    const auto r = strix::router_prefill_select_selftest(flavor, 64U, experts, top_k, normalize, 7U + flavor, 0U);
                    ++cases; rows_total += r[0];
                    if (r[1] != 0.0) {
                        ++failed;
                        std::printf("FAIL flavor %u E%u k%u norm %d: %.0f of %.0f rows differ\n", flavor, experts, top_k, normalize ? 1 : 0, r[1], r[0]);
                    }
                }
    std::printf("router select bitwise: %u cases, %.0f rows, %u failed\n", cases, rows_total, failed);
    for (const unsigned tiles : {1U, 128U, 1024U}) {   // T8 ~ one decode window launch
        const auto r = strix::router_prefill_select_selftest(0U, tiles, 512U, 10U, true, 99U, reps);
        std::printf("bench T%u E512 k10: block-per-row %.1f us, wave-per-row %.1f us (%.2fx), rows differing %.0f\n",
            tiles * 8U, r[2], r[3], r[3] > 0 ? r[2] / r[3] : 0.0, r[1]);
        if (r[1] != 0.0) ++failed;
    }
    for (const unsigned rows : {1U, 2U, 4U, 8U})
        for (const bool f32 : {false, true}) {
            if (f32 && rows > 2U) continue;
            for (unsigned flavor = 0; flavor < 6U; ++flavor) {   // bit 2: the chunk-256 form (rows <= 4)
                if (flavor == 2U || flavor == 3U || ((flavor & 4U) != 0U && rows > 4U)) continue;
                const auto r = strix::router_staged_logits_selftest(rows, f32, flavor, 11U + rows, (flavor & 3U) == 0U ? reps : 0U);
                if (r[1] != 0.0) ++failed;
                std::printf("staged logits R%u %s flavor %u%s: %.0f of %.0f logits differ%s\n", rows, f32 ? "f32" : "bf16", flavor, (flavor & 4U) ? " (chunk 256)" : "", r[1], r[0],
                    r[1] != 0.0 ? " FAIL" : "");
                if ((flavor & 3U) == 0U)
                    std::printf("  bench R%u %s%s: incumbent %.2f us, staged activations %.2f us (%.1f / %.1f %% of 246.7 GB/s for 5.24 MB)\n",
                        rows, f32 ? "f32" : "bf16", (flavor & 4U) ? " chunk256" : "", r[2], r[3], 5242880.0 / (r[2] * 1e-6) / 246.7e9 * 100.0,
                        5242880.0 / (r[3] * 1e-6) / 246.7e9 * 100.0);
            }
        }
    for (const bool f32 : {false, true})
        for (unsigned flavor : {8U, 9U}) {   // 16 loader waves (rows 1)
            const auto r = strix::router_staged_logits_selftest(1U, f32, flavor, 21U, flavor == 8U ? reps : 0U);
            if (r[1] != 0.0) ++failed;
            std::printf("staged logits R1 %s loaders16 flavor %u: %.0f of %.0f logits differ%s\n", f32 ? "f32" : "bf16", flavor, r[1], r[0], r[1] != 0.0 ? " FAIL" : "");
            if (flavor == 8U) std::printf("  bench R1 %s loaders16: incumbent %.2f us, loaders16 %.2f us\n", f32 ? "f32" : "bf16", r[2], r[3]);
        }
    std::printf("router select: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
