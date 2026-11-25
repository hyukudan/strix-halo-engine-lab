#include "cuda/core/mma_tile.cuh"
#include "core/microscaled_f16_staging.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "moe_core/grouped_dispatch.hpp"
#include "strix/weight/host_lowbit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
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

bool small_mxfp8_weight_codes() {
    using strix::core::microscaled_f16::e4m3_f16_pair;
    namespace sf = strix::weight::scalar_float;
    const auto matches = [](std::uint32_t code, std::uint16_t bits) {
        const float expected = sf::e4m3(static_cast<std::uint8_t>(code)) / 256.0F;
        const float actual = strix::weight::lowbit::f16_bits_to_f32(bits);
        return std::isnan(expected) ? std::isnan(actual) && std::signbit(actual) == std::signbit(expected)
                                    : sf::f2u(actual) == sf::f2u(expected);
    };
    for (std::uint32_t lo = 0; lo <= 255U; ++lo) {
        for (std::uint32_t hi = 0; hi <= 255U; ++hi) {
            const auto pair = e4m3_f16_pair((lo << 8U) | (hi << 24U));
            if (!matches(lo, static_cast<std::uint16_t>(pair)) ||
                !matches(hi, static_cast<std::uint16_t>(pair >> 16U))) {
                std::cerr << "MXFP8 weight codes " << lo << ", " << hi << ": wrong FP16 operands\n";
                return false;
            }
        }
    }
    return true;
}

bool small_scale_range() {
    using strix::core::microscaled_f16::mxfp8_group_scale;
    using Operand = packed::PackedOperand<packed::MatrixFormat::mxfp8>;
    for (unsigned byte = 0; byte <= 255U; ++byte) {
        const auto code = static_cast<std::uint8_t>(byte);
        const auto scale = mxfp8_group_scale(code);
        for (float dot : {0.0F, 1.0F, -1.0F}) {
            const float actual = std::fma((dot / 256.0F) * scale.partial_multiplier, scale.multiplier, 0.0F);
            const float expected = dot * Operand::decode_scale(code);
            if (byte == 255U ? !std::isnan(actual) : !std::isfinite(actual) || actual != expected) {
                std::cerr << "MXFP8 scale byte " << byte << ", dot " << dot << ": " << actual << " vs " << expected << '\n';
                return false;
            }
        }
    }
    return true;
}

bool small_activation_range() {
    namespace gm = moe::grouped_microscaled;
    for (unsigned i = 0; i <= 0xFFFFU; ++i) {
        const float value = strix::weight::scalar_float::u2f(i << 16U);
        const bool fits_fp16 = std::fabs(value) <= 65504.0F;
        if (gm::small_f16_needs_bf16(static_cast<std::uint16_t>(i)) == fits_fp16) {
            std::cerr << "Wrong FP16 admission for BF16 bits " << i << '\n';
            return false;
        }
    }
    return true;
}

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

// A whole grouped visit (grouped_dispatch::enqueue) on a recording backend: every submission counts, the gather
// writes the workspace, and the projection refuses what the HIP launcher refuses (fits, and fits_small for the small
// form). Hidden 256 / intermediate 64: gate-up (K 256, N 128) fits the small form, down (K 64, N 256) does not.
int visit_submissions = 0;

template<unsigned Hidden, unsigned Intermediate>
struct VisitConfig {
    enum class Projection { gate_up, down };
    static constexpr unsigned hidden = Hidden, intermediate = Intermediate, experts = 4U, top_k = 2U, layers = 1U,
                              max_rows = 4U;
    static constexpr packed::MatrixFormat format(unsigned) { return packed::MatrixFormat::nvfp4; }
    static constexpr packed::RowOrder row_order(Projection) { return packed::RowOrder::contiguous; }
    struct Activation {
        static constexpr float gate_cap = 4.0F, up_cap = 25.0F;
        static int launch(const moe::expert_activation::Args&, int) { return ++visit_submissions, 0; }
    };
    struct Combine {
        static int launch(const moe::grouped_rows::CombineArgs&, int) { return ++visit_submissions, 0; }
    };
};

struct VisitBackend {
    using Status = int;
    using Stream = int;
    static constexpr Status success = 0, invalid_value = 1;
    static constexpr unsigned rows_per_tile = moe::grouped_microscaled::kRowsPerTile;
    static constexpr auto fits = moe::grouped_microscaled::fits;
    static constexpr bool small_form = true;
    static Status gather(const moe::grouped_rows::GatherArgs& args, Stream) {
        args.grouped[0] = 0U;
        return ++visit_submissions, success;
    }
    template<packed::MatrixFormat Format>
    static Status launch_projection(const moe::grouped_microscaled::Args& args, Stream) {
        if (!moe::grouped_microscaled::fits(args) ||
            (args.form == moe::grouped_microscaled::Form::small_f16 && !moe::grouped_microscaled::fits_small(args)))
            return invalid_value;
        return ++visit_submissions, success;
    }
};

template<class Config>
bool visit(moe::grouped_microscaled::Form form, int expect_status, int expect_submissions) {
    namespace dispatch = moe::grouped_dispatch;
    constexpr unsigned rows = 1U, assignments = rows * Config::top_k;
    const auto need = dispatch::workspace_requirement<Config>(rows);
    std::vector<std::uint16_t> bf16(need.bf16, std::uint16_t{0x1234U});
    std::vector<float> fp32(need.fp32, 3.0F), input(Config::hidden, 1.0F), output(Config::hidden, 5.0F);
    const std::array<std::uint32_t, assignments> assignment_rows{0U, 0U}, combine_rows{0U, 1U};
    const std::array<float, assignments> weights{0.5F, 0.5F};
    const std::array<moe::grouped_microscaled::TileRows, 2> gate_up{}, down{};
    unsigned failure = 0U;
    dispatch::DevicePlan plan{0U, rows, assignment_rows, combine_rows, weights, gate_up, down, nullptr, &failure};
    plan.form = form;
    visit_submissions = 0;
    const int status = dispatch::enqueue<Config, VisitBackend>(plan, input, output, {bf16, fp32}, 0);
    const bool untouched = std::ranges::all_of(bf16, [](std::uint16_t x) { return x == 0x1234U; }) &&
                           std::ranges::all_of(fp32, [](float x) { return x == 3.0F; }) &&
                           std::ranges::all_of(output, [](float x) { return x == 5.0F; });
    if (status != expect_status || visit_submissions != expect_submissions ||
        (expect_submissions == 0 && !untouched)) {
        std::cout << "visit hidden " << Config::hidden << " intermediate " << Config::intermediate << " form "
                  << static_cast<unsigned>(form) << ": status " << status << " submissions " << visit_submissions
                  << " untouched " << untouched << "\n";
        return false;
    }
    return true;
}

bool whole_visit_admission() {
    using Fits = VisitConfig<256U, 128U>;
    using DownTooNarrow = VisitConfig<256U, 64U>;
    using Form = moe::grouped_microscaled::Form;
    // Both legs fit: gather, gate-up, activation, down, combine. Down refused by the small form: nothing submitted,
    // workspace and output untouched (before: gather, gate-up and the activation were queued, then down returned
    // invalid_value). The BF16 tile takes both legs of that shape.
    return visit<Fits>(Form::small_f16, 0, 5) && visit<DownTooNarrow>(Form::small_f16, 1, 0) &&
           visit<DownTooNarrow>(Form::bf16_tile, 0, 5);
}

bool small_panel_admission() {
    namespace gm = moe::grouped_microscaled;
    using dispatch = moe::grouped_dispatch::HipBackend;
    const gm::TileRows tile_rows{};
    const std::uint32_t map = 0U;
    alignas(16) const std::array<std::uint16_t, 128> input{};
    gm::Args args{&tile_rows, 1U, 128U, 64U, nullptr, nullptr, 1U};
    args.form = gm::Form::small_f16;
    const bool direct = dispatch::fits(args);
    const bool indirect = gm::fits_indirect({args, &map, input.data()});
    const bool bad_indirect = gm::fits_indirect({args, nullptr, input.data()}) ||
                              gm::fits_indirect({args, &map, input.data() + 1});
    args.form = gm::Form::bf16_tile;
    const bool incumbent = dispatch::fits(args) || gm::fits_indirect({args, &map, input.data()});
    if (!direct || !indirect || bad_indirect || incumbent) {
        std::cerr << "K=128 N=64 admission: direct " << direct << ", indirect " << indirect
                  << ", invalid gather " << bad_indirect << ", BF16 " << incumbent << '\n';
        return false;
    }
    return true;
}

struct ResidentVisitBackend : VisitBackend {
    template<unsigned Experts, unsigned TopK>
    struct Planner {
        static constexpr std::size_t scratch_bytes(unsigned) { return sizeof(unsigned); }
    };
    template<unsigned Experts, unsigned TopK>
    static Status materialize(const moe::device_grouped::Args& args, std::span<unsigned>, Stream) {
        args.csr.offsets[0] = 0U;
        return ++visit_submissions, success;
    }
    static constexpr auto token_rows = gather;
    template<packed::MatrixFormat Format>
    static Status launch_indirect(const moe::grouped_microscaled::IndirectArgs& args, Stream stream) {
        if (!moe::grouped_microscaled::fits_indirect(args)) return invalid_value;
        return launch_projection<Format>(args, stream);
    }
};

struct ResidentVisitConfig : VisitConfig<128U, 128U> {
    static constexpr unsigned experts = 32U, max_rows = 33U;
};

bool resident_visit_admission() {
    namespace dispatch = moe::grouped_dispatch;
    namespace gm = moe::grouped_microscaled;
    using Config = ResidentVisitConfig;
    using Act = moe::grouped_w4a4::Activations;
    constexpr unsigned rows = 33U, assignments = rows * Config::top_k, max_tiles = gm::kMaxTiles;
    constexpr unsigned sentinel = 12345U;
    alignas(16) const std::array<std::uint8_t, 16> plane{};
    const packed::PackedMatrix gate_up{plane.data(), plane.data(), 2U * Config::intermediate, Config::hidden};
    const packed::PackedMatrix down{plane.data(), plane.data(), Config::hidden, Config::intermediate};
    std::array<dispatch::ExpertWeights, Config::experts> matrices;
    matrices.fill({gate_up, down});
    const auto need = dispatch::workspace_requirement<Config>(rows);
    alignas(16) std::array<std::uint16_t, assignments * Config::hidden + 1U> bf16{};
    std::vector<float> fp32(need.fp32), input(rows * Config::hidden), output(input.size());
    std::array<float, Config::experts> bias{};
    std::array<float, rows * Config::experts> logits{};
    std::array<float, assignments> weights{}, assignment_weights{};
    std::array<unsigned, assignments> ids{}, combine{}, order{}, slots{};
    std::array<unsigned, rows> status{};
    std::array<unsigned, 1> scratch{};
    std::array<unsigned, Config::experts + 1U> offsets{};
    std::array<unsigned, Config::experts> touched{};
    std::vector<unsigned> experts(max_tiles), begins(max_tiles + 1U);
    std::vector<gm::TileRows> gate_tiles(max_tiles), down_tiles(max_tiles);
    unsigned tile_count = 0U, failure = 0U;
    const moe::device_grouped::CsrTables csr{offsets.data(), touched.data(), experts.data(), begins.data(),
                                            order.data(), slots.data()};
    const dispatch::RouteBuffers route{ids, status, scratch, weights};
    const auto select = [](auto, auto, dispatch::RouteBuffers r, unsigned, int) {
        r.ids[0] = 0U;
        return ++visit_submissions, ResidentVisitBackend::success;
    };
    const auto run = [&](const char* name, gm::Form form, unsigned capacity, Act activations, unsigned misalign,
                         int expected_submissions) {
        ids[0] = offsets[0] = sentinel;
        bf16.fill(0x1234U);
        visit_submissions = 0;
        const moe::device_grouped::Output plan_out{assignment_weights.data(), combine.data(), gate_tiles.data(),
                                                   down_tiles.data(), &tile_count, &failure, capacity};
        const dispatch::Workspace workspace{{bf16.data() + misalign, need.bf16}, fp32};
        moe::device_grouped::Args args{};
        if (!dispatch::prepare_resident<Config, ResidentVisitBackend>(0U, rows, route, matrices, csr, plan_out,
                                                                       workspace, args)) {
            std::cerr << "resident fixture does not pass planner admission: " << name << '\n';
            return false;
        }
        const auto result = dispatch::enqueue_resident<Config, ResidentVisitBackend>(
            0U, rows, input, logits, bias, output, route, matrices, csr, plan_out, workspace, 0,
            dispatch::LogitExtent::exact, select, gm::Decode::reference, activations, gm::Schedule::automatic, {}, form);
        const bool untouched = ids[0] == sentinel && offsets[0] == sentinel && workspace.bf16[0] == 0x1234U;
        const int expected = expected_submissions == 0 ? ResidentVisitBackend::invalid_value : ResidentVisitBackend::success;
        if (result != expected || visit_submissions != expected_submissions || untouched != (expected_submissions == 0)) {
            std::cerr << "resident " << name << ": status " << result << ", submissions " << visit_submissions
                      << ", untouched " << untouched << '\n';
            return false;
        }
        return true;
    };
    bool ok = run("small grid overflow", gm::Form::small_f16, max_tiles, Act::bf16, 0U, 0);
    ok = run("small incompatible operand", gm::Form::small_f16, max_tiles / 2U, Act::nvfp4, 0U, 0) && ok;
    ok = run("misaligned in-place workspace", gm::Form::small_f16, max_tiles / 2U, Act::bf16, 1U, 0) && ok;
    ok = run("small grid boundary", gm::Form::small_f16, max_tiles / 2U, Act::bf16, 0U, 7) && ok;
    ok = run("BF16 incumbent", gm::Form::bf16_tile, max_tiles, Act::bf16, 0U, 7) && ok;
    return ok;
}

// Small-form admission of the expert banks: a scale plane that starts one byte past a 4-byte boundary (legal for the
// BF16 tile) is refused for the small form, in either bank, by admit_small and by prepare_form(..., small_f16).
bool small_scale_admission() {
    namespace dispatch = moe::grouped_dispatch;
    using Config = VisitConfig<256U, 128U>;
    using Form = moe::grouped_microscaled::Form;
    std::vector<std::uint8_t> storage(std::size_t{1} << 16U);
    auto* base = storage.data() + (256U - reinterpret_cast<std::uintptr_t>(storage.data()) % 256U) % 256U;
    const auto* codes = base;
    const auto* scales = base + 32768U;  // 4-byte aligned scale plane
    const packed::PackedMatrix gate_up{codes, scales, 2U * Config::intermediate, Config::hidden};
    const packed::PackedMatrix down{codes, scales, Config::hidden, Config::intermediate};
    std::vector<moe::grouped_dispatch::ExpertWeights> matrices(Config::experts, moe::grouped_dispatch::ExpertWeights{gate_up, down});
    const auto need = dispatch::workspace_requirement<Config>(1U);
    std::vector<std::uint16_t> bf16(need.bf16);
    std::vector<float> fp32(need.fp32);
    const std::array<std::uint32_t, 2> ids{0U, 1U};
    const std::array<float, 2> weights{0.5F, 0.5F};
    const auto prepares = [&](Form form) {
        try {
            (void)dispatch::prepare_form<Config, dispatch::HipBackend>(0U, 1U, ids, weights, matrices, {bf16, fp32},
                                                                       form);
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        }
    };
    if (!dispatch::admit_small<Config>(0U, matrices) || !prepares(Form::small_f16) || !prepares(Form::bf16_tile))
        return false;
    for (const bool in_down : {false, true}) {
        auto odd = matrices;
        (in_down ? odd[1].down : odd[1].gate_up).block_scales = scales + 1;
        matrices.swap(odd);
        const bool refused = !dispatch::admit_small<Config>(0U, matrices) && !prepares(Form::small_f16);
        const bool incumbent = prepares(Form::bf16_tile);
        matrices.swap(odd);
        if (!refused || !incumbent) {
            std::cout << "small-form scale admission (" << (in_down ? "down" : "gate-up") << "): refused " << refused
                      << " incumbent admits " << incumbent << "\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const bool weight_codes = small_mxfp8_weight_codes();
    const bool scale_range = small_scale_range(), activation_range = small_activation_range();
    if (!weight_codes || !scale_range || !activation_range) return 1;
    std::cout << "small-form MXFP8 weight codes: PASS\n";
    std::cout << "small-form MXFP8 scale range and BF16 activation range: PASS\n";
    const bool panel = small_panel_admission(), resident = resident_visit_admission();
    if (!panel || !resident) return 1;
    std::cout << "small-form 64-column and resident preflight admission: PASS\n";
    // Small form row plan: one form for any tile height (no row cap); 16-row blocks up to 16 rows, then 32-row blocks.
    {
        namespace gm = moe::grouped_microscaled;
        constexpr auto plan = [](unsigned rows) { return gm::small_rows(rows); };
        static_assert(plan(0U).row_tiles == 0U && plan(0U).row_blocks == 0U);
        static_assert(plan(1U).row_tiles == 1U && plan(1U).row_blocks == 1U);
        static_assert(plan(16U).row_tiles == 1U && plan(16U).row_blocks == 1U);
        static_assert(plan(17U).row_tiles == 2U && plan(17U).row_blocks == 1U);
        static_assert(plan(32U).row_tiles == 2U && plan(32U).row_blocks == 1U);
        static_assert(plan(33U).row_tiles == 2U && plan(33U).row_blocks == 2U);
        static_assert(plan(48U).row_blocks == 2U && plan(49U).row_blocks == 2U && plan(64U).row_blocks == 2U);
        static_assert(plan(65U).row_blocks == 3U && plan(256U).row_blocks == 8U);
        static_assert(plan(0xFFFFFFFFU).row_tiles == 2U && plan(0xFFFFFFFFU).row_blocks == 0x8000000U);
        for (unsigned rows = 1U; rows <= 300U; ++rows) {
            const auto p = plan(rows);
            const unsigned block = 16U * p.row_tiles;
            if (p.row_blocks * block < rows || (p.row_blocks - 1U) * block >= rows) {
                std::cout << "small_rows: rows " << rows << " not covered exactly once\n";
                return 1;
            }
        }
    }
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
    if (!whole_visit_admission() || !small_scale_admission()) {
        std::cerr << "small-form visit or scale admission failed\n";
        return 1;
    }
    if (!admission() || !planner(200U, true, random) || !planner(4096U, false, random)) {
        std::cerr << "microscaled admission or large partition failed\n";
        return 1;
    }
    std::cout << "microscaled tile host maps: PASS rows=1..128 K=1152,2304 N=2304 and planner bounds, small-form visit and scale admission\n";
    return 0;
}
