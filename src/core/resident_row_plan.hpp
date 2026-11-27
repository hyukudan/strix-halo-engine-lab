#pragma once

#include "core/byte_range.hpp"
#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/gemv_rows.hpp"
#include "strix/device/lowbit_rows.hpp"
#include "strix/weight/lowbit_folded_scale_profile.hpp"
#include "strix/weight/lowbit_rows_select_profile.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/resident_slices.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace strix::core {

using ResidentRowWeight = std::variant<const std::uint16_t*, weight::ResidentMatrixView>;

struct ResidentRowProjection {
    ResidentRowWeight weights;
    unsigned outputs;
    float* output;
};

// Byte ranges of resident launches: [a, a + a_bytes) and [b, b + b_bytes) share no byte.
inline bool disjoint_byte_ranges(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const ByteRange first{a, a_bytes}, second{b, b_bytes};
    return byte_range_valid(first) && byte_range_valid(second) && !byte_ranges_overlap(first, second);
}
// The same for float counts the caller has bounded (count * sizeof(float) fits in std::size_t).
inline bool disjoint_float_ranges(const float* a, std::size_t a_floats, const float* b, std::size_t b_floats) {
    return disjoint_byte_ranges(a, a_floats * sizeof(float), b, b_floats * sizeof(float));
}

inline bool valid_resident_q8_rows(const weight::ResidentMatrixView& matrix, unsigned outputs,
                                   const float* input, unsigned rows, const float* output) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_extent = std::numeric_limits<unsigned>::max() - 31U;
    return matrix.storage == weight::kQ8WeightStorage && input && output &&
           reinterpret_cast<std::uintptr_t>(input) % 16U == 0U &&
           matrix.bytes.data() && reinterpret_cast<std::uintptr_t>(matrix.bytes.data()) % 16U == 0U &&
           rows != 0U && outputs != 0U && outputs <= matrix.rows && outputs <= max_extent &&
           outputs <= max_floats / rows && matrix.columns != 0U && matrix.columns <= max_floats / rows &&
           matrix.columns <= max_extent && matrix.columns % weight::Q8Storage::elements == 0U &&
           matrix.row_bytes == weight::Q8Storage::row_bytes(matrix.columns) &&
           matrix.rows <= matrix.bytes.size() / matrix.row_bytes &&
           disjoint_float_ranges(input, std::size_t{rows} * matrix.columns, output, std::size_t{rows} * outputs) &&
           disjoint_byte_ranges(output, std::size_t{rows} * outputs * sizeof(float), matrix.bytes.data(),
                                matrix.bytes.size());
}

// Pending FAST/TURBO candidates use either folded tile operands or a measured standalone sweep.
enum class LowBitArithmetic { fp32_sweep, f16_group_scale, f16_folded_scale, f16_rows_select };

// Both product entry points and qualification arms bind the same resolved value to the launch policy.
inline LowBitArithmetic resident_lowbit_arithmetic(const profile::Resolution& tier) {
    const auto enabled = [&](const profile::Lever& lever) {
        const auto* entry = tier.find(lever.name);
        if (!entry || entry->value == lever.builtin || entry->value == "off") return false;
        if (entry->value != lever.target)
            throw std::invalid_argument(std::string(lever.env) + " requires " + std::string(lever.target) + " or off");
        if (tier.profile != profile::Name::fast && tier.profile != profile::Name::turbo)
            throw std::invalid_argument(std::string(lever.env) + " requires FAST or TURBO");
        return true;
    };
    const bool folded = enabled(weight::kLowBitFoldedScaleLever);
    const bool rows_select = enabled(weight::kLowBitRowsSelectLever);
    if (folded && rows_select)
        throw std::invalid_argument("folded scale and rows-select are separate low-bit candidate identities");
    if (folded) return LowBitArithmetic::f16_folded_scale;
    return rows_select ? LowBitArithmetic::f16_rows_select : LowBitArithmetic::f16_group_scale;
}

namespace detail {
template <class Storage, class Read>
std::string_view folded_scale_decline(const weight::ResidentMatrixView& matrix, const Read& read) {
    if (matrix.columns == 0U || matrix.columns % Storage::elements != 0U || matrix.rows == 0U ||
        matrix.row_bytes != Storage::row_bytes(matrix.columns) || matrix.rows > matrix.bytes.size() / matrix.row_bytes)
        return "invalid compact matrix geometry";
    constexpr std::size_t chunk_bytes = (64U * 1024U / Storage::bytes) * Storage::bytes;
    const auto size = static_cast<std::size_t>(matrix.rows * matrix.row_bytes);
    std::vector<std::uint8_t> scratch(std::min(size, chunk_bytes));
    for (std::size_t offset = 0U; offset < size;) {
        const auto chunk = std::span(scratch).first(std::min(scratch.size(), size - offset));
        read(matrix.bytes.subspan(offset, chunk.size()), chunk);
        for (std::size_t pos = 0U; pos < chunk.size(); pos += Storage::bytes) {
            const auto* block = chunk.data() + pos;
            for (unsigned g = 0U; g < Storage::groups; ++g) {
                const auto scale = weight::lowbit::f16_bits_to_f32(Storage::scale_bits(block, g));
                if (!std::isfinite(scale)) return "nonfinite FP16 group scale";
                for (unsigned j = g * Storage::group; j < (g + 1U) * Storage::group; ++j) {
                    const auto product = scale * static_cast<float>(Storage::code(block, j));
                    if (std::abs(product) > 65504.0F) return "scale x code exceeds finite FP16 range";
                    if (weight::lowbit::f16_bits_to_f32(weight::lowbit::f32_to_f16_bits(product)) != product)
                        return "scale x code is not exact in FP16";
                }
            }
        }
        offset += chunk.size();
    }
    return {};
}
}  // namespace detail

// Certify the immutable staged compact image using a bounded host reader. Dispatch never dereferences device bytes.
template <class Read>
std::string_view certify_folded_scale(weight::ResidentMatrixView& matrix, const Read& read) {
    matrix.folded_scale_admitted = false;
    std::string_view decline;
    if (matrix.storage == weight::kQ4G64WeightStorage)
        decline = detail::folded_scale_decline<weight::Q4G64Storage>(matrix, read);
    else if (matrix.storage == weight::kQ6G64WeightStorage)
        decline = detail::folded_scale_decline<weight::Q6G64Storage>(matrix, read);
    else
        return "storage has no folded-scale staging";
    matrix.folded_scale_admitted = decline.empty();
    return decline;
}

constexpr LowBitArithmetic admitted_lowbit_arithmetic(LowBitArithmetic arithmetic,
                                                       const weight::ResidentMatrixView& matrix) {
    return arithmetic == LowBitArithmetic::f16_folded_scale && !matrix.folded_scale_admitted
        ? LowBitArithmetic::f16_group_scale : arithmetic;
}

constexpr bool lowbit_f16_form(LowBitArithmetic arithmetic) { return arithmetic != LowBitArithmetic::fp32_sweep; }
constexpr unsigned lowbit_max_rows(LowBitArithmetic arithmetic) {
    return lowbit_f16_form(arithmetic) ? 65535U * f16_tile_gemm::kBM : gemv_rows::max_rows;
}

// One measured standalone cell of "lowbit.rows-select". Each entry lists only the exact row counts where
// the FP32 sweep beat the incumbent. Adding or removing a measured count requires an identity revision.
struct LowBitSweepCell {
    std::string_view arch;
    unsigned bits, outputs, width;
    std::span<const unsigned> rows;
};
inline constexpr weight::ArithmeticId kLowBitRowsSelectIdentity{"lowbit.rows-select.fp32-sweep-cells", 2};
inline constexpr unsigned kRowsOneTwo[]{1U, 2U};
inline constexpr unsigned kRowsOneTwoFour[]{1U, 2U, 4U};
inline constexpr std::array<LowBitSweepCell, 5> kLowBitSweepCells{{
    {"gfx1151", 4U, 1024U, 5120U, kRowsOneTwoFour},
    {"gfx1151", 4U, 2304U, 5120U, kRowsOneTwo},
    {"gfx1151", 4U, 17408U, 5120U, kRowsOneTwoFour},
    {"gfx1151", 6U, 1024U, 5120U, kRowsOneTwoFour},
    {"gfx1151", 6U, 17408U, 5120U, kRowsOneTwoFour},
}};
constexpr bool lowbit_sweep_cell(std::string_view arch, unsigned bits, unsigned outputs, unsigned width, unsigned rows,
                                 std::span<const LowBitSweepCell> cells = kLowBitSweepCells) {
    if (arch.empty()) return false;
    for (const auto& cell : cells)
        if (cell.arch == arch && cell.bits == bits && cell.outputs == outputs && cell.width == width)
            return std::ranges::find(cell.rows, rows) != cell.rows.end();
    return false;
}
// True when a measured cell exists for this device and one of the model's projection input widths: the rows-select
// lever can change a launch only then (every other launch keeps the group-scale arithmetic).
constexpr bool lowbit_sweep_reachable(std::string_view arch, std::span<const unsigned> widths,
                                      std::span<const LowBitSweepCell> cells = kLowBitSweepCells) {
    if (arch.empty()) return false;
    for (const auto& cell : cells)
        if (cell.arch == arch && std::ranges::find(widths, cell.width) != widths.end()) return true;
    return false;
}
static_assert([] {
    for (const auto& cell : kLowBitSweepCells) {
        if ((cell.bits != 4U && cell.bits != 6U) || cell.rows.empty() || cell.width % 128U != 0U)
            return false;
        for (const auto rows : cell.rows)
            if (rows == 0U || rows > gemv_rows::max_rows) return false;
    }
    return true;
}());

// The receipt table covers standalone projections. Every grouped storage run uses its incumbent arithmetic.
constexpr LowBitArithmetic grouped_lowbit_arithmetic(LowBitArithmetic arithmetic) {
    return arithmetic == LowBitArithmetic::f16_rows_select ? LowBitArithmetic::f16_group_scale : arithmetic;
}

inline bool valid_resident_lowbit_rows(const weight::ResidentMatrixView& matrix, unsigned outputs,
                                       const float* input, unsigned rows, const float* output,
                                       LowBitArithmetic arithmetic = LowBitArithmetic::fp32_sweep) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_extent = std::numeric_limits<unsigned>::max() - 127U;
    const bool q4 = matrix.storage == weight::kQ4G64WeightStorage;
    const bool sp4 = matrix.storage == weight::kSP4TWeightStorage;
    if (!q4 && !sp4 && !(matrix.storage == weight::kQ6G64WeightStorage)) return false;
    // SP4 has the FP16 form only, on the small-M schedule (16-row blocks, grid rows <= 65535); the tensor base is an
    // outer power of two within the FP32 exponent range.
    if (sp4 && (arithmetic != LowBitArithmetic::f16_group_scale || rows > 65535U * 16U ||
                matrix.outer_exponent < weight::sp::kMinBase || matrix.outer_exponent > weight::sp::kMaxBase))
        return false;
    if (!sp4 && matrix.outer_exponent != 0) return false;
    const auto row_bytes = q4    ? weight::Q4G64Storage::row_bytes(matrix.columns)
                           : sp4 ? weight::sp::SP4Storage::row_bytes(matrix.columns)
                                 : weight::Q6G64Storage::row_bytes(matrix.columns);
    return input && output && reinterpret_cast<std::uintptr_t>(input) % 16U == 0U &&
           matrix.bytes.data() && reinterpret_cast<std::uintptr_t>(matrix.bytes.data()) % 16U == 0U &&
           rows != 0U && rows <= lowbit_max_rows(arithmetic) && outputs != 0U && outputs <= matrix.rows &&
           outputs <= max_extent && outputs <= max_floats / rows && matrix.columns != 0U &&
           matrix.columns <= max_floats / rows && matrix.columns <= max_extent &&
           matrix.columns % weight::Q4G64Storage::elements == 0U && matrix.row_bytes == row_bytes &&
           matrix.rows <= matrix.bytes.size() / matrix.row_bytes &&
           disjoint_float_ranges(input, std::size_t{rows} * matrix.columns, output, std::size_t{rows} * outputs) &&
           disjoint_byte_ranges(output, std::size_t{rows} * outputs * sizeof(float), matrix.bytes.data(),
                                matrix.bytes.size());
}

// With valid tight Q8_0 rows, the tile's K and load alignment already hold.
// Keep the tight sweep for decode/verify and when the tile row grid is too tall.
inline constexpr bool use_resident_q8_tile(unsigned rows) {
    return rows > gemv_rows::max_rows && rows <= 65535U * f16_tile_gemm::kBM;
}

// Versioned arithmetic of the resident Q8_0 routes above, per backend: the implementation identity a load-time
// weight decision is scoped to (weight::implementation_identity). Bump a revision when that route's accumulation
// order changes; that retires every decision row measured on the old arithmetic.
inline constexpr std::array<weight::ArithmeticId, 2> kResidentQ8ArithmeticHip{{
    {"hip.q8_0.dense-linear-tight-rows", 1}, {"hip.q8_0.f16-tile-staging", 1}}};
inline constexpr std::array<weight::ArithmeticId, 2> kResidentQ8ArithmeticCuda{{
    {"cuda.q8_0.dense-rows-mma3", 1}, {"cuda.q8_0.dense-big-split3", 1}}};
static_assert(kResidentQ8ArithmeticHip[0].name != kResidentQ8ArithmeticHip[1].name &&
              kResidentQ8ArithmeticCuda[0].name != kResidentQ8ArithmeticCuda[1].name &&
              kResidentQ8ArithmeticHip[0].revision && kResidentQ8ArithmeticHip[1].revision &&
              kResidentQ8ArithmeticCuda[0].revision && kResidentQ8ArithmeticCuda[1].revision);

// Route of a low-bit resident launch, backend-neutral. Both routes compute ONE arithmetic (integer codes as the FP16
// operand, one WMMA chain per K64 group from +0, acc = fma(group sum, d, acc) over the groups in ascending order, no K
// split: core/f16_tile_gemm_small.hpp), so a row's bits
// depend on its activation row, the weights and K only; never on the batch width, the grouping or the route. The plan
// is a speed choice alone, and it is device data (LowBitRowPlan below):
//   small_tile rows <= plan.small_rows: the small-M schedule (16 / 32-row blocks of one 64-column panel);
//   tile       every other launch: the 128-row projection tile with the low-bit staging trait.
enum class LowBitRoute : std::uint8_t { small_tile, tile };
inline constexpr unsigned kLowBitPanelColumns = f16_tile_gemm::kBN / 2U;  // one small-M CTA per 64-column panel

// The switch points of one device. The empty plan is one route everywhere: the tile at every row count, 16-row
// blocks if a caller forces the small schedule.
struct LowBitRowPlan {
    unsigned small_rows = 0U;   // launches of at most this many rows run the small-M schedule (0: none)
    unsigned wide_panels = 0U;  // small-M launches of >= this many panels run 17..32 rows as one 32-row block (0: never)
};
// The plan of a device is data of that device, never a constant of this header: the optional `lowbit_rows` key of its
// entry or selected catalogue profile (docs/devices.md; measured by tests/hip_dense_lowbit_bench.hip), bound with
// the device by the device gate (strix/device/lowbit_rows.hpp). With no bound switch points, use the empty plan.
constexpr LowBitRoute plan_lowbit_rows(unsigned rows, const LowBitRowPlan& plan) {
    return rows <= plan.small_rows ? LowBitRoute::small_tile : LowBitRoute::tile;
}

// 32-row blocks of the small-M schedule for this launch (speed only: both blocks give the same bits per row).
constexpr bool lowbit_wide_block(unsigned rows, unsigned panels, const LowBitRowPlan& plan) {
    return plan.wide_panels != 0U && rows > 16U && rows <= 32U && panels >= plan.wide_panels;
}

// The plan bound to a runtime device ordinal (the backend's own device index), or the empty plan.
inline LowBitRowPlan bound_lowbit_row_plan(int device, bool allow_profile = true) noexcept {
    const auto rows = device::bound_lowbit_rows(device, allow_profile);
    return {rows.small_rows, rows.wide_panels};
}

inline constexpr std::array<weight::ArithmeticId, 2> kResidentLowBitArithmeticHip{{
    {"hip.q4_g64.dense-linear-tight-rows", 1}, {"hip.q6_g64.dense-linear-tight-rows", 1}}};
inline constexpr std::array<weight::ArithmeticId, 2> kResidentLowBitTileArithmeticHip{{
    {"hip.q4_g64.f16-group-scale-chain", 1}, {"hip.q6_g64.f16-group-scale-chain", 1}}};
// Strix pack SP4 (fixed table) on the small-M schedule: table values as exact FP16 operands, one WMMA chain per K32
// group from +0, acc = fma(group sum, (1 + m / 8) 2^e, acc) in ascending groups, then x 2^base at the store.
inline constexpr std::array<weight::ArithmeticId, 1> kResidentSp4ArithmeticHip{{{"hip.sp4t.f16-group32-scale-chain", 1}}};
inline constexpr std::array<weight::ArithmeticId, 2> kResidentLowBitFoldedArithmeticHip{{
    {"hip.q4_g64.f16-folded-scale-tile", 1}, {"hip.q6_g64.f16-folded-scale-tile", 1}}};

// The weight bytes one projection of a list reads: a BF16 projection outputs tight rows of width elements, a compact
// one its whole view. Empty (nullptr, 0) when the descriptor has no weights or the extent does not fit.
inline std::pair<const void*, std::size_t> resident_weight_bytes(const ResidentRowProjection& projection, unsigned width) {
    if (const auto* bits = std::get_if<const std::uint16_t*>(&projection.weights)) {
        constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
        if (!*bits || width == 0U || projection.outputs > max_elements / width) return {nullptr, 0U};
        return {*bits, std::size_t{projection.outputs} * width * sizeof(std::uint16_t)};
    }
    const auto& matrix = std::get<weight::ResidentMatrixView>(projection.weights);
    return {matrix.bytes.data(), matrix.bytes.size()};
}

// A list of projections sharing one input (rows x width floats) writes each output once and never anything it reads:
// every output range (rows x outputs floats) is disjoint from the input, from every other output of the list and
// from every weight range the list reads (resident_weight_bytes), whatever the storage of either (a list mixes BF16
// and compact runs launched in order, so an output may not alias a later run's weights either). Checked for the
// whole list before any launch.
inline bool resident_list_ranges_disjoint(const float* input, unsigned width, unsigned rows,
                                          std::span<const ResidentRowProjection> projections) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!input || rows == 0U || width > max_floats / rows) return false;
    for (const auto& projection : projections)
        if (!projection.output || projection.outputs == 0U || projection.outputs > max_floats / rows ||
            !resident_weight_bytes(projection, width).first ||
            !disjoint_float_ranges(input, std::size_t{rows} * width, projection.output,
                                   std::size_t{rows} * projection.outputs))
            return false;
    for (const auto& written : projections)
        for (const auto& read : projections) {
            const auto [weights, bytes] = resident_weight_bytes(read, width);
            if (!disjoint_byte_ranges(written.output, std::size_t{rows} * written.outputs * sizeof(float), weights, bytes))
                return false;
        }
    for (std::size_t i = 0; i < projections.size(); ++i)
        for (std::size_t j = i + 1U; j < projections.size(); ++j)
            if (!disjoint_float_ranges(projections[i].output, std::size_t{rows} * projections[i].outputs,
                                       projections[j].output, std::size_t{rows} * projections[j].outputs))
                return false;
    return true;
}

// Validate every run with the backend's descriptor checks before any run can write output.
template <std::size_t Limit, class Status, class Bf16, class Compact, class Bf16Admitted, class CompactAdmitted>
Status dispatch_resident_rows(const float* input, unsigned width, unsigned rows,
                              std::span<const ResidentRowProjection> projections, Status success, Status invalid,
                              const Bf16& bf16, const Compact& compact, const Bf16Admitted& bf16_admitted,
                              const CompactAdmitted& compact_admitted) {
    if (projections.empty() || projections.size() > Limit ||
        !resident_list_ranges_disjoint(input, width, rows, projections))
        return invalid;

    const auto for_each_run = [&](const auto& visit_bf16, const auto& visit_compact) {
        std::array<Bf16RowProjection, Limit> run{};
        for (std::size_t i = 0; i < projections.size();) {
            const auto first = i;
            if (std::holds_alternative<const std::uint16_t*>(projections[i].weights)) {
                while (i < projections.size()) {
                    const auto& projection = projections[i];
                    const auto* bits = std::get_if<const std::uint16_t*>(&projection.weights);
                    if (!bits) break;
                    run[i - first] = {*bits, projection.outputs, projection.output};
                    ++i;
                }
                if (const auto status = visit_bf16(std::span<const Bf16RowProjection>(run).first(i - first));
                    status != success) return status;
            } else {
                while (i < projections.size() &&
                       std::holds_alternative<weight::ResidentMatrixView>(projections[i].weights)) ++i;
                if (const auto status = visit_compact(projections.subspan(first, i - first)); status != success)
                    return status;
            }
        }
        return success;
    };
    const auto admitted = for_each_run(
        [&](auto run) { return bf16_admitted(run) ? success : invalid; },
        [&](auto run) { return compact_admitted(run) ? success : invalid; });
    if (admitted != success) return admitted;
    return for_each_run(bf16, compact);
}

}  // namespace strix::core
