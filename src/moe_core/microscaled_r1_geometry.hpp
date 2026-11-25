#pragma once

// Geometry helpers run on the host (launch grids) and in the kernels (column and row maps).
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_R1_GEOMETRY_HD __host__ __device__
#else
#define STRIX_R1_GEOMETRY_HD
#endif

namespace strix::moe_core::microscaled_r1 {

// Launch geometry. A wave owns `columns` consecutive outputs and streams their rows together (gate_up: the gate and
// up rows of each column; down_combine: one row per column per routed expert). Each row keeps its own accumulator,
// lane->K mapping and butterfly, so the column count changes no output bit. It sets how many rows share one wave's
// dependent load chain (ids -> matrices -> codes -> scales): on gfx1151 one row per wave left that chain carrying
// 1.3 KB (gate) / 0.65 KB (down) and the kernels at 61 % / 53 % of the byte roof.
// sm_120 (no-model bench cuda_core_moe_r1_geometry_bench, synthetic top-8 2304 / 1152 experts, rows 1): one column
// per warp left gate_up / down_combine at 880 / 655-700 GB/s (NVFP4); two / four columns are 10 % / 14-19 % faster
// per launch, with the same bits.
struct Geometry {
    unsigned gate_columns, down_columns;
};
inline constexpr unsigned kWavesPerBlock = 8;
inline constexpr Geometry kHipGeometry{2U, 4U};
inline constexpr Geometry kCudaGeometry{2U, 4U};

STRIX_R1_GEOMETRY_HD constexpr unsigned gate_blocks(unsigned intermediate, unsigned columns) {
    return (intermediate - 1U) / (kWavesPerBlock * columns) + 1U;
}
STRIX_R1_GEOMETRY_HD constexpr unsigned gate_first_column(unsigned block, unsigned wave, unsigned columns) {
    return (block * kWavesPerBlock + wave) * columns;
}
STRIX_R1_GEOMETRY_HD constexpr unsigned down_blocks(unsigned hidden, unsigned columns) { return (hidden - 1U) / columns + 1U; }
// Grid of the rows launch (x = column blocks, y = CTAs over rows): gate_up has one CTA per (row, route slot),
// down_combine one per row. The decode rows launch and the verify step launch both use it, so a verify row is
// the decode row's CTA.
struct RowsGrid {
    unsigned x, y;
};
STRIX_R1_GEOMETRY_HD constexpr RowsGrid gate_up_grid(unsigned intermediate, unsigned rows, unsigned top_k, unsigned columns) {
    return {gate_blocks(intermediate, columns), rows * top_k};
}
STRIX_R1_GEOMETRY_HD constexpr RowsGrid down_combine_grid(unsigned hidden, unsigned rows, unsigned columns) {
    return {down_blocks(hidden, columns), rows};
}
// Weight row read by output slot i of a wave whose first column is col; `live` columns remain in the matrix.
// Slots past the live columns re-read the first valid row (their results are never stored).
STRIX_R1_GEOMETRY_HD constexpr unsigned slot_row(unsigned col, unsigned i, unsigned columns, unsigned stride, unsigned live) {
    const unsigned offset = i % columns < live ? i % columns : 0U;
    return col + offset + (i / columns) * stride;
}

}  // namespace strix::moe_core::microscaled_r1


#undef STRIX_R1_GEOMETRY_HD
