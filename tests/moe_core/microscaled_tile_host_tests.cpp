#include "cuda/core/mma_tile.cuh"
#include "inkling2/inkling2_mini_config.hpp"
#include "moe_core/grouped_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

namespace mma = strix::cuda::core::mma;
namespace tile = mma::microscaled;
namespace native = strix::cuda::core::native_mma;
namespace moe = strix::moe_core;
namespace model = strix::inkling2;
namespace packed = strix::weight::microscaled;
using Small = tile::TileShape<native::kDecodeMBlocks, tile::kDecodeOutputs>;
using Wide = tile::TileShape<native::kTileMBlocks, tile::kTileOutputs>;

// Model a shared K16 stage and the routing of each ldmatrix register using the
// non-transposed m8n8 map: address lane = 8 * matrix + row, word = lane % 4.
template<class Shape, unsigned OutputTile>
bool operands(unsigned rows, unsigned inputs) {
    constexpr unsigned padding = ~0U;
    for (unsigned slice = 0; slice < tile::row_slices(rows, Shape::kRows); ++slice) {
        const unsigned base = Shape::first_row(slice);
        const unsigned first_output = Shape::first_output(1U);
        unsigned next_k = 0U;
        for (unsigned step = 0; step < inputs / tile::kKStep; ++step) {
            const unsigned k = Shape::k_start(step);
            if (k != next_k) return false;
            next_k += 16U;
            std::array<std::array<unsigned, tile::kKStep>, Shape::kRows> a{};
            std::array<std::array<unsigned, tile::kKStep>, OutputTile> b{};
            for (auto& staged_row : a) staged_row.fill(padding);
            for (auto& staged_row : b) staged_row.fill(padding);
            for (unsigned thread = 0; thread < Shape::kThreads; ++thread) {
                for (unsigned load = 0; load < Shape::kActivationLoads; ++load) {
                    const unsigned row = Shape::activation_row(thread, load);
                    const unsigned slot = 2U * Shape::activation_pair(thread);
                    if (row >= a.size() || slot + 1U >= a[row].size()) return false;
                    for (unsigned half = 0; half < 2U; ++half) {
                        if (base + row < rows) a[row][slot + half] = (base + row) * inputs + k + slot + half;
                    }
                }
                const unsigned column = Shape::weight_row(thread);
                if (column >= OutputTile) continue;
                const unsigned first_slot = 2U * Shape::weight_first_pair(thread);
                for (unsigned slot = first_slot; slot < first_slot + 2U * Shape::kWeightPairs; ++slot) {
                    if (slot >= b[column].size()) return false;
                    b[column][slot] = (first_output + column) * inputs + k + slot;
                }
            }
            for (unsigned warp = 0; warp < Shape::kThreads / 32U; ++warp) {
                for (unsigned lane = 0; lane < 32U; ++lane) {
                    for (unsigned m = 0; m < Shape::kWarpMBlocks; ++m) {
                        for (unsigned reg = 0; reg < 4U; ++reg) {
                            const auto addr = mma::detail::ldsm_a_address(8U * reg + lane / 4U);
                            const unsigned staged_row = Shape::fragment_row(warp, m) + addr.row;
                            const unsigned out_row = Shape::output_row(warp, m, lane, 2U * (reg & 1U));
                            if (staged_row >= a.size()) return false;
                            for (unsigned half = 0; half < 2U; ++half) {
                                const unsigned slot = addr.segment * 8U + 2U * (lane % 4U) + half;
                                const unsigned expected = base + out_row < rows
                                    ? (base + out_row) * inputs + k + native::a_slot(lane, reg, half) : padding;
                                if (slot >= a[staged_row].size() || a[staged_row][slot] != expected) return false;
                            }
                        }
                    }
                    for (unsigned n = 0; n < Shape::kWarpNBlocks; ++n) {
                        constexpr unsigned per_load = Shape::kNBlocksPerLoad;
                        const unsigned first_n = n - n % per_load;
                        for (unsigned reg = 0; reg < 2U; ++reg) {
                            const auto addr = mma::detail::ldsm_b_address<2U * per_load>(
                                16U * (n % per_load) + 8U * reg + lane / 4U);
                            const unsigned staged_col = Shape::fragment_column(warp, first_n) + addr.row;
                            // B uses the lane group as the n8 column, independently of C's lane layout.
                            const unsigned n8_col = native::b_col(lane);
                            const unsigned column = Shape::output_column(warp, n, n8_col / 2U, n8_col % 2U);
                            if (staged_col >= b.size()) return false;
                            for (unsigned half = 0; half < 2U; ++half) {
                                const unsigned slot = addr.segment * 8U + 2U * (lane % 4U) + half;
                                const unsigned expected = (first_output + column) * inputs + k +
                                                          native::b_slot(lane, reg, half);
                                if (slot >= b[staged_col].size() || b[staged_col][slot] != expected) return false;
                            }
                        }
                    }
                }
            }
        }
        if (next_k != inputs) return false;
    }
    return true;
}

template<class Shape, unsigned OutputTile>
bool stores(unsigned rows, unsigned outputs) {
    std::vector<unsigned> writes(std::size_t{rows} * outputs);
    for (unsigned slice = 0; slice < tile::row_slices(rows, Shape::kRows); ++slice) {
        for (unsigned block = 0; block < outputs / OutputTile; ++block) {
            for (unsigned warp = 0; warp < Shape::kThreads / 32U; ++warp) {
                for (unsigned m = 0; m < Shape::kWarpMBlocks; ++m) {
                    for (unsigned n = 0; n < Shape::kWarpNBlocks; ++n) {
                        for (unsigned lane = 0; lane < 32U; ++lane) {
                            for (unsigned i = 0; i < 4U; ++i) {
                                const unsigned row = Shape::first_row(slice) + Shape::output_row(warp, m, lane, i);
                                const unsigned col = Shape::first_output(block) + Shape::output_column(warp, n, lane, i);
                                if (col >= outputs) return false;
                                if (row < rows && ++writes[std::size_t{row} * outputs + col] != 1U) return false;
                            }
                        }
                    }
                }
            }
        }
    }
    return std::ranges::all_of(writes, [](unsigned count) { return count == 1U; });
}

bool planner(unsigned rows, bool concentrated, std::mt19937& random) {
    constexpr unsigned k = model::kExpertsPerToken;
    constexpr unsigned experts = model::kExperts;
    constexpr unsigned capacity = moe::grouped_dispatch::CudaBackend::rows_per_tile;
    std::vector<std::uint32_t> ids(std::size_t{rows} * k);
    const std::vector<float> weights(ids.size(), 1.0F / k);
    std::array<std::uint32_t, experts> pool{};
    std::iota(pool.begin(), pool.end(), 0U);
    for (unsigned row = 0; row < rows; ++row) {
        if (!concentrated) std::shuffle(pool.begin(), pool.end(), random);
        std::copy_n(pool.data(), k, ids.data() + std::size_t{row} * k);
    }
    const auto plan = moe::build_expert_major_plan({rows, k, experts, 0, ids}, weights, capacity);
    const unsigned bound = moe::device_grouped::tile_bound(rows, k, experts, capacity);
    unsigned covered = 0U;
    for (const auto& descriptor : plan.tiles) {
        if (descriptor.first_assignment != covered || descriptor.rows == 0U ||
            descriptor.rows > std::min(rows, capacity)) return false;
        covered += descriptor.rows;
    }
    return capacity == 128U && covered == ids.size() && plan.tiles.size() <= bound &&
           (!concentrated || rows < capacity ||
            std::ranges::any_of(plan.tiles, [](const auto& descriptor) { return descriptor.rows == capacity; }));
}

bool admission() {
    constexpr unsigned inputs = 32U;
    constexpr unsigned outputs = 160U;
    alignas(16) std::uint8_t codes[outputs * inputs / 2U]{}, scales[outputs * inputs / 16U]{};
    alignas(16) std::uint16_t input[64U * inputs]{};
    float output[64U * outputs]{};
    moe::grouped_microscaled::TileRows descriptor{{codes, scales, outputs, inputs}, input, output, inputs, outputs, 64U};
    moe::grouped_microscaled::Args args{&descriptor, 1U, inputs, outputs, nullptr, nullptr, 64U};
    if (!tile::fits(args) || !tile::fits_tile<packed::MatrixFormat::nvfp4>(descriptor, inputs, outputs) ||
        tile::fits_tile<packed::MatrixFormat::nvfp4>(descriptor, inputs, outputs + 1U)) return false;
    ++args.outputs;
    if (tile::fits(args)) return false;

    args.inputs = moe::grouped_microscaled::gemm::kBK;
    args.outputs = moe::grouped_microscaled::gemm::kBN;
    const bool hip_fits = moe::grouped_dispatch::HipBackend::fits(args);
    args.max_tile_rows = 0U;
    return hip_fits && moe::grouped_dispatch::HipBackend::fits(args);
}

}  // namespace

int main() {
    constexpr unsigned outputs = model::kHiddenWidth;
    static_assert(tile::kRowsPerTile == 128U && tile::kDecodeRows == 16U);
    static_assert(tile::output_tile(outputs, 0U) == tile::kTileOutputs);
    static_assert(tile::output_tile(outputs, 16U) == tile::kDecodeOutputs);
    static_assert(tile::output_tile(outputs, 17U) == tile::kTileOutputs);
    static_assert(tile::output_tile(outputs + 32U, 17U) == tile::kDecodeOutputs);
    static_assert(tile::row_slices(0U, Small::kRows) == 8U);
    static_assert(tile::row_slices(0U, Wide::kRows) == 2U && tile::row_slices(64U, Wide::kRows) == 1U);
    static_assert(tile::row_slices(16U, Small::kRows) == 1U);
    static_assert(tile::row_slices(17U, Small::kRows) == 2U);
    std::mt19937 random(0x6432U);
    std::array<unsigned, tile::kRowsPerTile> counts{};
    std::iota(counts.begin(), counts.end(), 1U);
    std::shuffle(counts.begin(), counts.end(), random);
    for (const unsigned rows : counts) {
        if (!stores<Small, tile::kDecodeOutputs>(rows, outputs) ||
            !stores<Wide, tile::kTileOutputs>(rows, outputs) ||
            !stores<Small, tile::kDecodeOutputs>(rows, outputs + 32U) ||
            !planner(rows, true, random) || !planner(rows, false, random)) {
            std::cerr << "microscaled tile stores or partition failed for rows=" << rows << '\n';
            return 1;
        }
    }
    for (const unsigned inputs : {model::kHiddenWidth, model::kExpertWidth}) {
        for (const unsigned rows : counts) {
            if (!operands<Small, tile::kDecodeOutputs>(rows, inputs) ||
                !operands<Wide, tile::kTileOutputs>(rows, inputs)) {
                std::cerr << "microscaled K16 operands failed for rows=" << rows << " K=" << inputs << '\n';
                return 1;
            }
        }
    }
    if (!admission() || !planner(200U, true, random) || !planner(4096U, false, random)) {
        std::cerr << "microscaled admission or large partition failed\n";
        return 1;
    }
    std::cout << "microscaled tile host maps: PASS rows=1..128 K=1152,2304 N=2304 and planner bounds\n";
    return 0;
}
