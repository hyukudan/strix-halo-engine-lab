#pragma once

#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/gemv_rows.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/resident_slices.hpp"

#include <array>
#include <limits>
#include <string_view>
#include <variant>

namespace strix::core {

using ResidentRowWeight = std::variant<const std::uint16_t*, weight::ResidentMatrixView>;

struct ResidentRowProjection {
    ResidentRowWeight weights;
    unsigned outputs;
    float* output;
};

// Byte ranges of resident launches. [a, a + a_floats) and [b, b + b_floats) share no byte; the counts are float
// counts the caller has bounded (count * sizeof(float) fits in std::size_t).
inline bool disjoint_float_ranges(const float* a, std::size_t a_floats, const float* b, std::size_t b_floats) {
    const auto first = reinterpret_cast<std::uintptr_t>(a), second = reinterpret_cast<std::uintptr_t>(b);
    return first >= second ? first - second >= b_floats * sizeof(float) : second - first >= a_floats * sizeof(float);
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
           disjoint_float_ranges(input, std::size_t{rows} * matrix.columns, output, std::size_t{rows} * outputs);
}

// Low-bit resident rows (strix/weight/host_lowbit.hpp) on HIP: every row count up to the tile grid limit; the route
// (plan_lowbit_rows below) picks the small-M WMMA schedule or the 128-row projection tile (same bits per row).
inline bool valid_resident_lowbit_rows(const weight::ResidentMatrixView& matrix, unsigned outputs,
                                       const float* input, unsigned rows, const float* output) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_extent = std::numeric_limits<unsigned>::max() - 127U;
    const bool q4 = matrix.storage == weight::kQ4G64WeightStorage;
    if (!q4 && !(matrix.storage == weight::kQ6G64WeightStorage)) return false;
    const auto row_bytes = q4 ? weight::Q4G64Storage::row_bytes(matrix.columns) : weight::Q6G64Storage::row_bytes(matrix.columns);
    return input && output && reinterpret_cast<std::uintptr_t>(input) % 16U == 0U &&
           matrix.bytes.data() && reinterpret_cast<std::uintptr_t>(matrix.bytes.data()) % 16U == 0U &&
           rows != 0U && rows <= 65535U * f16_tile_gemm::kBM && outputs != 0U && outputs <= matrix.rows &&
           outputs <= max_extent && outputs <= max_floats / rows && matrix.columns != 0U &&
           matrix.columns <= max_floats / rows && matrix.columns <= max_extent &&
           matrix.columns % weight::Q4G64Storage::elements == 0U && matrix.row_bytes == row_bytes &&
           matrix.rows <= matrix.bytes.size() / matrix.row_bytes &&
           disjoint_float_ranges(input, std::size_t{rows} * matrix.columns, output, std::size_t{rows} * outputs);
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
// The plan of a device is device data, never a constant of this header: the measured switch points of a device model
// live in the shared device catalogue (strix/device/profiles.hpp lowbit_route_points, data/devices/profiles.toml
// `lowbit_rows`, measured on the no-model fixture tests/hip_dense_lowbit_bench.hip) and the backend reads them for the
// device it runs on; a device without a row runs the empty plan.
constexpr LowBitRoute plan_lowbit_rows(unsigned rows, const LowBitRowPlan& plan) {
    return rows <= plan.small_rows ? LowBitRoute::small_tile : LowBitRoute::tile;
}

// 32-row blocks of the small-M schedule for this launch (speed only: both blocks give the same bits per row).
constexpr bool lowbit_wide_block(unsigned rows, unsigned panels, const LowBitRowPlan& plan) {
    return plan.wide_panels != 0U && rows > 16U && rows <= 32U && panels >= plan.wide_panels;
}

// The arithmetic every low-bit resident launch of a storage runs on HIP (both routes, see above).
inline constexpr std::array<weight::ArithmeticId, 2> kResidentLowBitArithmeticHip{{
    {"hip.q4_g64.f16-group-scale-chain", 1}, {"hip.q6_g64.f16-group-scale-chain", 1}}};

// A list of projections sharing one input (rows x width floats) writes each output once and never its input: every
// output range (rows x outputs floats) is disjoint from the input and from every other output of the list, whatever
// the storage of either (a list mixes BF16 and compact runs). Checked for the whole list before any launch.
inline bool resident_list_ranges_disjoint(const float* input, unsigned width, unsigned rows,
                                          std::span<const ResidentRowProjection> projections) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!input || rows == 0U || width > max_floats / rows) return false;
    for (const auto& projection : projections)
        if (!projection.output || projection.outputs > max_floats / rows ||
            !disjoint_float_ranges(input, std::size_t{rows} * width, projection.output,
                                   std::size_t{rows} * projection.outputs))
            return false;
    for (std::size_t i = 0; i < projections.size(); ++i)
        for (std::size_t j = i + 1U; j < projections.size(); ++j)
            if (!disjoint_float_ranges(projections[i].output, std::size_t{rows} * projections[i].outputs,
                                       projections[j].output, std::size_t{rows} * projections[j].outputs))
                return false;
    return true;
}

// Dispatch consecutive projections of each storage class together, preserving
// order across BF16 and compact runs. The compact binding validates its format;
// the whole list's byte ranges are checked first (resident_list_ranges_disjoint), so a refused list writes nothing.
template <std::size_t Limit, class Status, class Bf16, class Compact>
Status dispatch_resident_rows(const float* input, unsigned width, unsigned rows,
                              std::span<const ResidentRowProjection> projections, Status success, Status invalid,
                              const Bf16& bf16, const Compact& compact) {
    if (projections.empty() || projections.size() > Limit ||
        !resident_list_ranges_disjoint(input, width, rows, projections))
        return invalid;
    std::array<Bf16RowProjection, Limit> run{};
    std::size_t count = 0;
    for (std::size_t i = 0; i < projections.size();) {
        const auto& projection = projections[i];
        if (const auto* bits = std::get_if<const std::uint16_t*>(&projection.weights)) {
            run[count++] = {*bits, projection.outputs, projection.output};
            ++i;
            continue;
        }
        if (count != 0) {
            if (auto status = bf16(std::span(run).first(count)); status != success) return status;
            count = 0;
        }
        const auto first = i++;
        while (i < projections.size() && std::holds_alternative<weight::ResidentMatrixView>(projections[i].weights)) ++i;
        if (auto status = compact(projections.subspan(first, i - first)); status != success) return status;
    }
    return count != 0 ? bf16(std::span(run).first(count)) : success;
}

}  // namespace strix::core
