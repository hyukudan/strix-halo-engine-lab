#pragma once

namespace strix::cuda_core {

inline constexpr unsigned kFp32GridRows = 8;
inline constexpr unsigned kFp32GridMaxRows = 65535U * kFp32GridRows;
inline constexpr unsigned kCompactBf16Outputs = 32;

enum class Bf16RowPath { short_rows, fp32_grid, big, big_compact };

// An MMA shape rejection already means FP32 SIMT for the unrounded projection.
// Extending that path to a row grid keeps its reduction order. `terms` is the
// activation plane count of the tall split (3 exact mma3, 2 / 1 FAST mma2 /
// mma1). The compact tile is three-term only, so fewer terms take plain big.
[[nodiscard]] constexpr Bf16RowPath plan_bf16_row_projection(
    unsigned in, unsigned out, unsigned rows, unsigned short_limit, bool tall_mma, int terms = 3) {
    if (!tall_mma || terms < 1 || terms > 3 || rows <= short_limit || in == 0 || out == 0)
        return Bf16RowPath::short_rows;
    if (in % 32U == 0 && out % 16U == 0)
        return out <= kCompactBf16Outputs && terms == 3 ? Bf16RowPath::big_compact : Bf16RowPath::big;
    return rows <= kFp32GridMaxRows ? Bf16RowPath::fp32_grid : Bf16RowPath::short_rows;
}

}  // namespace strix::cuda_core
