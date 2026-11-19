// CPU test: launch shapes of the exact v2 MTP kernels (include/strix/cuda/qwen_mtp_v2_shape.hpp).
// Invariants: (1) every (output, row) of the fused projection and every (feature, row) of the hyper output is owned by exactly one
// warp / CTA slot for every row count (the kernels keep each pair's FP32 chain, so ownership is what the shape may change);
// (2) decode rows (<= 8) keep the shapes they had before the prefill shapes; (3) the hyper-output CTA fits the 48 KB static smem.
#include "strix/cuda/qwen_mtp_v2_shape.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace pm = strix::cuda::qwen_plemtp;

static int g_fail = 0;
static void check(bool ok, const char* what, std::uint32_t rows) {
    if (!ok) { ++g_fail; std::printf("FAIL %s rows %u\n", what, rows); }
}

static void projection_coverage(std::uint32_t rows) {
    const auto sh = pm::mtp_v2_projection_shape(rows);
    check(sh.grid_x * pm::kV2ProjWarpsPerCta * sh.outs_per_warp == pm::kV2Hidden, "projection outputs tile exactly", rows);
    std::vector<std::uint8_t> owned(static_cast<std::size_t>(pm::kV2Hidden) * rows, 0);
    for (std::uint32_t bx = 0; bx < sh.grid_x; ++bx)
        for (std::uint32_t by = 0; by < sh.grid_y; ++by)
            for (std::uint32_t w = 0; w < pm::kV2ProjWarpsPerCta; ++w) {
                const std::uint32_t o0 = (bx * pm::kV2ProjWarpsPerCta + w) * sh.outs_per_warp;   // kernel: o0 >= kHidden returns
                if (o0 >= pm::kV2Hidden) continue;
                for (std::uint32_t n = 0; n < sh.outs_per_warp; ++n)
                    for (std::uint32_t r = 0; r < sh.rows_per_cta; ++r) {
                        const std::uint32_t row = by * sh.rows_per_cta + r;                     // kernel: row < rows predicate
                        if (row < rows) ++owned[static_cast<std::size_t>(row) * pm::kV2Hidden + o0 + n];
                    }
            }
    bool once = true;
    for (auto c : owned) once = once && c == 1U;
    check(once, "projection (output,row) owned exactly once", rows);
}

static void hyper_coverage(std::uint32_t rows) {
    const auto sh = pm::mtp_v2_hyper_shape(rows);
    check(sh.grid_x * 32U == pm::kV2Hidden, "hyper features tile exactly", rows);
    std::vector<std::uint8_t> owned(rows, 0);
    for (std::uint32_t by = 0; by < sh.grid_y; ++by) {
        const std::uint32_t row0 = by * sh.rows_per_cta;
        check(row0 < rows, "hyper CTA has at least one row (nr = min(RB, rows - row0) >= 1)", rows);
        const std::uint32_t nr = sh.rows_per_cta < rows - row0 ? sh.rows_per_cta : rows - row0;
        for (std::uint32_t r = 0; r < nr; ++r) ++owned[row0 + r];
    }
    bool once = true;
    for (auto c : owned) once = once && c == 1U;
    check(once, "hyper row owned exactly once", rows);
    // static smem of mtp_hyper_output_v2_kernel<RB>: 4 streams x 32 x 340 B staged rows + RB x 128 sigmoids
    check(4U * 32U * 340U + sh.rows_per_cta * 128U * 4U <= 48U * 1024U, "hyper static smem <= 48 KB", rows);
}

static void decode_shapes_unchanged(std::uint32_t rows) {
    // Shapes of the decode forms before the prefill shapes existed: projection <1> for 1 row, <4> otherwise, grid (2560 / 8, ceil);
    // hyper output grid (2560 / 32, rows), one row per CTA.
    const auto p = pm::mtp_v2_projection_shape(rows);
    const std::uint32_t r = rows == 1U ? 1U : 4U;
    check(p.rows_per_cta == r && p.outs_per_warp == 1U && p.grid_x == 320U && p.grid_y == (rows + r - 1U) / r, "decode projection shape", rows);
    const auto h = pm::mtp_v2_hyper_shape(rows);
    check(h.rows_per_cta == 1U && h.grid_x == 80U && h.grid_y == rows, "decode hyper shape", rows);
}

int main() {
    for (std::uint32_t rows = 1; rows <= 300U; ++rows) { projection_coverage(rows); hyper_coverage(rows); }
    for (std::uint32_t rows : {2019U, 4096U, 8074U, 8076U, 8191U}) { projection_coverage(rows); hyper_coverage(rows); }
    for (std::uint32_t rows = 1; rows <= pm::kV2DecodeRowsMax; ++rows) decode_shapes_unchanged(rows);
    const auto big = pm::mtp_v2_projection_shape(8076U);
    check(big.outs_per_warp == 4U && big.grid_x == 80U && big.grid_y == 2019U, "prefill projection shape 4 x 4", 8076U);
    const auto hb = pm::mtp_v2_hyper_shape(8074U);
    check(hb.rows_per_cta == 8U && hb.grid_y == 1010U, "prefill hyper shape RB 8", 8074U);
    std::printf("qwen_mtp_v2_shape_tests: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
