// Host check of the microscaled R1 decode geometry: for every backend geometry, the gate_up and down_combine grids
// store each output column exactly once, from its own weight row(s), and never read past the matrix. The per-row
// arithmetic does not depend on the column count (dot_rows keeps one accumulator per row), so this coverage is the
// whole invariant that a change of rows per wave can break.
#include "moe_core/microscaled_r1.hpp"

#include <cstddef>
#include <cstdint>
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

// Rows launch layout (kMaxRows): the activation row gate_up writes for assignment a = row * TopK + slot is the one
// down_combine of that row reads for the slot; ids / weights / input / output rows are disjoint and in range; one row
// keeps every offset at zero (the one-row launch is unchanged).
bool check_rows(unsigned rows, unsigned hidden, unsigned intermediate) {
    constexpr unsigned top_k = 8U;
    std::vector<unsigned> written(std::size_t{rows} * top_k, 0U), routed(std::size_t{rows} * top_k, 0U);
    std::vector<unsigned> inputs(rows, 0U), outputs(rows, 0U);
    for (unsigned a = 0; a < rows * top_k; ++a) {
        const auto at = r1::row_offsets<top_k>(a / top_k, hidden, intermediate);
        const std::size_t write = at.activation + std::size_t{a % top_k} * intermediate;
        if (write % intermediate != 0U || write / intermediate != a) return false;
        ++written[write / intermediate];
        if (at.route + a % top_k != a) return false;  // gate_up reads ids[a]
    }
    for (unsigned row = 0; row < rows; ++row) {
        const auto at = r1::row_offsets<top_k>(row, hidden, intermediate);
        if (row == 0U && (at.input || at.route || at.activation || at.output)) return false;
        if (at.input % hidden || at.input / hidden >= rows || at.output % hidden || at.output / hidden >= rows) return false;
        ++inputs[at.input / hidden];
        ++outputs[at.output / hidden];
        for (unsigned slot = 0; slot < top_k; ++slot) {
            const std::size_t read = at.activation + std::size_t{slot} * intermediate;
            if (read / intermediate != std::size_t{row} * top_k + slot || written[read / intermediate] != 1U) return false;
            ++routed[at.route + slot];
        }
    }
    for (unsigned count : routed)
        if (count != 1U) return false;
    for (unsigned row = 0; row < rows; ++row)
        if (inputs[row] != 1U || outputs[row] != 1U) return false;
    return true;
}

bool check_row_admission() {
    namespace packed = strix::weight::microscaled;
    const strix::moe_core::device_grouped::ExpertWeights matrix{};
    const unsigned ids[1]{};
    const float weights[1]{}, input[1]{};
    std::uint16_t activation[1]{};
    float output[1]{};
    const r1::Args args{&matrix, ids, weights, nullptr, input, activation, output, 128U, 2304U, 1152U, 4.0F, 25.0F};
    for (const auto& limits : {r1::kHipGridLimits, r1::kCudaGridLimits}) {
        if (r1::fits_rows<packed::MatrixFormat::nvfp4, 8U>(args, 0U, limits) ||
            r1::fits_rows<packed::MatrixFormat::nvfp4, 8U>(args, r1::kMaxRows + 1U, limits)) return false;
        for (unsigned rows = 1; rows <= r1::kMaxRows; ++rows)
            if (!r1::fits_rows<packed::MatrixFormat::nvfp4, 8U>(args, rows, limits) ||
                !r1::fits_rows<packed::MatrixFormat::mxfp8, 8U>(args, rows, limits)) return false;
    }
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
    // Rows launch grid shared by decode rows and verify step rows: one gate_up CTA per (row, route slot), one
    // down_combine CTA per row, the same column blocks as one row.
    for (unsigned rows = 1U; rows <= 16U; ++rows) {
        const auto gate = r1::gate_up_grid(1152U, rows, 8U, r1::kCudaGeometry.gate_columns);
        const auto down = r1::down_combine_grid(2304U, rows, r1::kCudaGeometry.down_columns);
        if (gate.x != 72U || gate.y != rows * 8U || down.x != 576U || down.y != rows) {
            std::printf("FAIL rows grid %u\n", rows);
            return 1;
        }
    }

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
    for (unsigned rows = 1; rows <= r1::kMaxRows; ++rows) {
        if (!check_rows(rows, 2304U, 1152U) || !check_rows(rows, 48U, 16U)) {
            std::printf("FAIL rows layout %u\n", rows);
            ++failures;
        }
    }
    if (!check_row_admission()) {
        std::printf("FAIL rows admission\n");
        ++failures;
    }
    if (failures) return 1;
    std::printf("microscaled R1 geometry: every column stored once from its own rows (4 geometries x 7 widths); "
                "rows 1..%u layout and admission\n", r1::kMaxRows);
    return 0;
}
