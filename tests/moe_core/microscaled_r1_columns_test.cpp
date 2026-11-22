// Host check of the microscaled R1 decode geometry: for every backend geometry, the gate_up and down_combine grids
// store each output column exactly once, from its own weight row(s), and never read past the matrix. The per-row
// arithmetic does not depend on the column count (dot_rows keeps one accumulator per row), so this coverage is the
// whole invariant that a change of rows per wave can break.
#include "moe_core/microscaled_r1.hpp"

#include <cstdio>
#include <vector>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;

bool check_gate(unsigned intermediate, unsigned columns) {
    std::vector<unsigned> stored(intermediate, 0U);
    const unsigned blocks = r1::gate_blocks(intermediate, columns);
    for (unsigned block = 0; block < blocks; ++block) {
        for (unsigned wave = 0; wave < r1::kWavesPerBlock; ++wave) {
            const unsigned col = r1::gate_first_column(block, wave, columns);
            if (col >= intermediate) continue;
            const unsigned live = intermediate - col;
            for (unsigned i = 0; i < 2U * columns; ++i) {
                if (r1::slot_row(col, i, columns, intermediate, live) >= 2U * intermediate) return false;
            }
            for (unsigned i = 0; i < columns && i < live; ++i) {
                if (r1::slot_row(col, i, columns, intermediate, live) != col + i) return false;
                if (r1::slot_row(col, columns + i, columns, intermediate, live) != intermediate + col + i) return false;
                ++stored[col + i];
            }
        }
    }
    for (unsigned count : stored)
        if (count != 1U) return false;
    return true;
}

bool check_down(unsigned hidden, unsigned columns) {
    std::vector<unsigned> stored(hidden, 0U);
    const unsigned blocks = r1::down_blocks(hidden, columns);
    for (unsigned block = 0; block < blocks; ++block) {
        const unsigned col = block * columns;
        if (col >= hidden) return false;  // the kernel has no early exit: every CTA must own a live column
        const unsigned live = hidden - col;
        for (unsigned i = 0; i < columns; ++i) {
            const unsigned row = r1::slot_row(col, i, columns, 0U, live);
            if (row >= hidden) return false;
            if (i < live) {
                if (row != col + i) return false;
                ++stored[col + i];
            }
        }
    }
    for (unsigned count : stored)
        if (count != 1U) return false;
    return true;
}

}  // namespace

int main() {
    static_assert(r1::kGeometry.gate_columns == r1::kCudaGeometry.gate_columns &&
                      r1::kGeometry.down_columns == r1::kCudaGeometry.down_columns,
                  "host / CUDA builds keep one row per wave");
    // HIP: gate_up streams 2 x gate_columns rows, down_combine down_columns rows per wave: same register shape.
    static_assert(2U * r1::kHipGeometry.gate_columns == r1::kHipGeometry.down_columns);
    // Inkling 2 Mini decode grids (hidden 2304, expert width 1152).
    static_assert(r1::gate_blocks(1152U, r1::kHipGeometry.gate_columns) == 72U);
    static_assert(r1::down_blocks(2304U, r1::kHipGeometry.down_columns) == 576U);
    static_assert(r1::gate_blocks(1152U, 1U) == 144U && r1::down_blocks(2304U, 1U) == 2304U);

    const r1::Geometry geometries[] = {r1::kHipGeometry, r1::kCudaGeometry, {4U, 8U}, {3U, 3U}};
    // Model widths are multiples of the 16-column scale block (fits()); 48 is the fixture's short width.
    const unsigned widths[] = {16U, 48U, 1152U, 1536U, 2304U, 4096U, 7168U};
    unsigned failures = 0;
    for (const auto& geometry : geometries) {
        for (unsigned width : widths) {
            if (!check_gate(width, geometry.gate_columns)) {
                std::printf("FAIL gate width %u columns %u\n", width, geometry.gate_columns);
                ++failures;
            }
            if (!check_down(width, geometry.down_columns)) {
                std::printf("FAIL down width %u columns %u\n", width, geometry.down_columns);
                ++failures;
            }
        }
    }
    if (failures) return 1;
    std::printf("microscaled R1 geometry: every column stored once from its own rows (4 geometries x 7 widths)\n");
    return 0;
}
