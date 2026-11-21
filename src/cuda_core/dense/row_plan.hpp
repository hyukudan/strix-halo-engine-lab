#pragma once

namespace strix::cuda_core {

inline constexpr unsigned kFp32GridRows = 8;
inline constexpr unsigned kFp32GridMaxRows = 65535U * kFp32GridRows;
inline constexpr unsigned kCompactBf16Outputs = 32;

enum class Bf16RowPath { short_rows, fp32_grid, big, big_compact };

// An MMA3 shape rejection already means FP32 SIMT for the unrounded projection.
// Extending that path to a row grid keeps its reduction order.
[[nodiscard]] constexpr Bf16RowPath plan_bf16_row_projection(
    unsigned in, unsigned out, unsigned rows, unsigned short_limit, bool tall_mma3) {
    if (!tall_mma3 || rows <= short_limit || in == 0 || out == 0)
        return Bf16RowPath::short_rows;
    if (in % 32U == 0 && out % 16U == 0)
        return out <= kCompactBf16Outputs ? Bf16RowPath::big_compact : Bf16RowPath::big;
    return rows <= kFp32GridMaxRows ? Bf16RowPath::fp32_grid : Bf16RowPath::short_rows;
}

}  // namespace strix::cuda_core
