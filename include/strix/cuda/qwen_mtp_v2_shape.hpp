#pragma once
// Launch shapes of the exact v2 MTP kernels (qwen_plemtp.cu: mtp_fused_projection_v2, mtp_hyper_output_v2), host-visible so the
// launcher and the CPU coverage test use one definition. Every (output, row) of the fused projection and every (feature, row) of the
// hyper output is owned by exactly one warp / CTA; the per-pair FP32 chain does not depend on the shape (see the kernels).
#include <cstdint>

namespace strix::cuda::qwen_plemtp {

constexpr std::uint32_t kV2Hidden = 2560U;          // = kHidden (qwen_plemtp.hpp)
constexpr std::uint32_t kV2ProjWarpsPerCta = 8U;    // 256 threads
constexpr std::uint32_t kV2DecodeRowsMax = 8U;      // decode-side shapes up to here (unchanged by the prefill shapes)

struct MtpV2ProjectionShape {
    std::uint32_t rows_per_cta, outs_per_warp, grid_x, grid_y;
};
struct MtpV2HyperShape {
    std::uint32_t rows_per_cta, grid_x, grid_y;
};

// rows == 1 -> 1 row x 1 output per warp; rows <= 8 -> 4 rows x 1 output; prefill rows -> 4 rows x 4 outputs per warp.
constexpr MtpV2ProjectionShape mtp_v2_projection_shape(std::uint32_t rows) {
    const std::uint32_t r = rows == 1U ? 1U : 4U, no = rows <= kV2DecodeRowsMax ? 1U : 4U;
    return {r, no, kV2Hidden / (kV2ProjWarpsPerCta * no), (rows + r - 1U) / r};
}
// rows <= 8 -> one row per CTA; prefill rows -> 8 rows per CTA reusing the staged weight rows. 32 features x 4 streams per CTA.
constexpr MtpV2HyperShape mtp_v2_hyper_shape(std::uint32_t rows) {
    const std::uint32_t rb = rows <= kV2DecodeRowsMax ? 1U : 8U;
    return {rb, kV2Hidden / 32U, (rows + rb - 1U) / rb};
}

}  // namespace strix::cuda::qwen_plemtp
