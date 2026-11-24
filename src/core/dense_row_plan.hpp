#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "core/weight_certificate.hpp"

namespace strix::core {

// Resident BF16 weights [output][input] and tight FP32 output [row][output].
struct Bf16RowProjection {
    const std::uint16_t* weights{};
    unsigned outputs{};
    float* output{};
    bool round_bf16{};  // Round the completed FP32 sum and store it widened to FP32.
    std::size_t row_bytes{};  // Zero uses input_width * sizeof(std::uint16_t).

    constexpr bool contiguous(unsigned input_width) const {
        return row_bytes == 0U ||
            (row_bytes % sizeof(std::uint16_t) == 0U && row_bytes / sizeof(std::uint16_t) == input_width);
    }
};

// Value class of a dense activation operand, declared by the caller that produced it.
// bf16_exact: every element is a finite FP32 value exactly representable in BF16 (the
// producer stored RN_bf16 of its result, or widened a BF16 tensor).
enum class Bf16Operand : unsigned char { fp32, bf16_exact };

// Activation planes the exact BF16 split needs. The split writes hi = RN_bf16(x),
// mid = RN_bf16(x - hi), lo = RN_bf16(x - hi - mid), and each K16 step of the tall
// chain issues lo, mid, hi into one accumulator that starts at +0. For a BF16-exact
// finite x, mid and lo are +0, so the two leading MMAs add only zero products (finite
// weights: 0 x inf would be NaN) to an accumulator that is never -0, and with sums that
// stay finite one plane (hi) gives the same bits. This is an EXACT
// schedule of the three-term arithmetic, not a FAST operand class.
constexpr int exact_bf16_planes(Bf16Operand operand) {
    return operand == Bf16Operand::bf16_exact ? 1 : 3;
}

// A shared caller whose plan rounds every element of a dense input to BF16 (`rounded`) passes that operand class
// to a backend dense(input, width, rows, projections, Bf16Operand); a backend without that overload keeps its
// FP32 operand. Both give the same bits (exact_bf16_planes); the declared form only skips the zero planes.
template <class Operations, class Projections>
void dense_declared(Operations& operations, const float* input, unsigned width, unsigned rows,
                    const Projections& projections, bool rounded) {
    if constexpr (requires { operations.dense(input, width, rows, projections, Bf16Operand::fp32); })
        operations.dense(input, width, rows, projections, rounded ? Bf16Operand::bf16_exact : Bf16Operand::fp32);
    else
        operations.dense(input, width, rows, projections);
}

// Normalised rows the down projection of a low-rank controller reads as ONE operand: the raw row and one message
// readout, concatenated. This is the whole operand whatever the number of readout banks.
constexpr unsigned kLowRankDownRows = 2;

// Admission of a low-rank controller's declared operands to the one-plane class (core/weight_certificate.hpp: down
// reads kLowRankDownRows RMS-normalised BF16 rows of `width`, the gates read `rank` latent values). Precondition: the
// residual rows are finite, so each normalised row has |row|_2 <= sqrt(width) up to rounding and the down operand has
// |x|_1 <= 1.01 kLowRankDownRows width; with the weight maxima (finite_magnitude_max*) the latent is at most that
// times down_max and each gate sum at most rank times latent times gate_max. With finite weights and both bounds below
// 2^126, no partial sum, product or BF16 rounding of either chain leaves the finite range, so the one-plane bits are
// the three-plane bits. False keeps the FP32 operand class. Non-finite residual rows are non-finite under either class.
inline bool low_rank_operands_bounded(unsigned width, unsigned rank, float down_max, float gate_max) {
    const float down_l1 = 1.01F * static_cast<float>(kLowRankDownRows) * static_cast<float>(width);
    if (!one_plane_sum_bounded(down_l1, down_max)) return false;
    const float latent = down_l1 * down_max;
    return one_plane_sum_bounded(static_cast<float>(rank) * latent, gate_max);
}

// Verify step rows of the BF16 -> FP32 dense family: one warp reads its weight row once for a tile of up to
// kDenseStepRowTile descriptor rows (each (output, row) keeps the one-row accumulation chain), so a width-W window
// reads every weight W / tile times instead of W times. Grid y = width / tile.
inline constexpr unsigned kDenseStepRowTile = 8U;
constexpr unsigned dense_step_row_tile(unsigned width) {
    return width < kDenseStepRowTile ? width : kDenseStepRowTile;
}

template <auto Limit, class Size, class Status, class Launch>
Status for_each_dense_chunk(Size total, Status success, const Launch& launch) {
    static_assert(Limit > 0);
    constexpr Size limit = Limit;
    for (Size first = 0; first < total;) {
        const Size count = std::min(total - first, limit);
        if (const auto status = launch(first, count); status != success) return status;
        first += count;
    }
    return success;
}

template <std::size_t Limit, class Status, class Launch>
Status for_each_bf16_projection_group(std::span<const Bf16RowProjection> projections, Status success,
                                      const Launch& launch) {
    return for_each_dense_chunk<Limit>(projections.size(), success, [&](std::size_t first, std::size_t count) {
        return launch(projections.subspan(first, count));
    });
}

}  // namespace strix::core
