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

// The activation planes a dispatch reports for one projection that took `path`: the term count of the tall split when
// the path is a tall MMA one (big, big_compact), 0 when it splits no activation (FP32 grid, short rows). A dispatch
// reports the largest over its projections, so a mixed one reports the split it ran.
[[nodiscard]] constexpr int bf16_split_planes(Bf16RowPath path, int terms) {
    return path == Bf16RowPath::big || path == Bf16RowPath::big_compact ? terms : 0;
}

}  // namespace strix::cuda_core
