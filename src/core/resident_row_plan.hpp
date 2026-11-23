#pragma once

#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/gemv_rows.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/resident_slices.hpp"

#include <array>
#include <limits>
#include <variant>

namespace strix::core {

using ResidentRowWeight = std::variant<const std::uint16_t*, weight::ResidentMatrixView>;

struct ResidentRowProjection {
    ResidentRowWeight weights;
    unsigned outputs;
    float* output;
};

inline bool valid_resident_q8_rows(const weight::ResidentMatrixView& matrix, unsigned outputs,
                                   const float* input, unsigned rows, const float* output) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_extent = std::numeric_limits<unsigned>::max() - 31U;
    return matrix.storage == weight::kQ8WeightStorage && input && output && input != output &&
           reinterpret_cast<std::uintptr_t>(input) % 16U == 0U &&
           matrix.bytes.data() && reinterpret_cast<std::uintptr_t>(matrix.bytes.data()) % 16U == 0U &&
           rows != 0U && outputs != 0U && outputs <= matrix.rows && outputs <= max_extent &&
           outputs <= max_floats / rows && matrix.columns != 0U && matrix.columns <= max_floats / rows &&
           matrix.columns <= max_extent && matrix.columns % weight::Q8Storage::elements == 0U &&
           matrix.row_bytes == weight::Q8Storage::row_bytes(matrix.columns) &&
           matrix.rows <= matrix.bytes.size() / matrix.row_bytes;
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
    return input && output && input != output && reinterpret_cast<std::uintptr_t>(input) % 16U == 0U &&
           matrix.bytes.data() && reinterpret_cast<std::uintptr_t>(matrix.bytes.data()) % 16U == 0U &&
           rows != 0U && rows <= 65535U * f16_tile_gemm::kBM && outputs != 0U && outputs <= matrix.rows &&
           outputs <= max_extent && outputs <= max_floats / rows && matrix.columns != 0U &&
           matrix.columns <= max_floats / rows && matrix.columns <= max_extent &&
           matrix.columns % weight::Q4G64Storage::elements == 0U && matrix.row_bytes == row_bytes &&
           matrix.rows <= matrix.bytes.size() / matrix.row_bytes;
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
// is a speed choice alone:
//   small_tile rows <= kLowBitSmallRows: the small-M schedule (16 / 32-row blocks of one 64-column panel);
//   tile       wider windows (prefill): the 128-row projection tile with the low-bit staging trait.
// The switch point is the row count where the tile overtakes the small schedule on the no-model fixture
// (tests/hip_dense_lowbit_bench.hip).
enum class LowBitRoute : std::uint8_t { small_tile, tile };
inline constexpr unsigned kLowBitSmallRows = 64U;
inline constexpr unsigned kLowBitPanelColumns = f16_tile_gemm::kBN / 2U;  // one small-M CTA per 64-column panel

constexpr LowBitRoute plan_lowbit_rows(unsigned rows) {
    return rows <= kLowBitSmallRows ? LowBitRoute::small_tile : LowBitRoute::tile;
}

// The arithmetic every low-bit resident launch of a storage runs on HIP (both routes, see above).
inline constexpr std::array<weight::ArithmeticId, 2> kResidentLowBitArithmeticHip{{
    {"hip.q4_g64.f16-group-scale-chain", 1}, {"hip.q6_g64.f16-group-scale-chain", 1}}};

// Dispatch consecutive projections of each storage class together, preserving
// order across BF16 and compact runs. The compact binding validates its format.
template <std::size_t Limit, class Status, class Bf16, class Compact>
Status dispatch_resident_rows(std::span<const ResidentRowProjection> projections, Status success, Status invalid,
                              const Bf16& bf16, const Compact& compact) {
    if (projections.empty() || projections.size() > Limit) return invalid;
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
