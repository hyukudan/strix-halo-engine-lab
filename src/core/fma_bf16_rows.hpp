#pragma once

#include "core/controller_rms_rows.hpp"
#include "core/dense_linear.hpp"
#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/fma_rows.hpp"
#include "strix/hip/legacy_launch.hpp"

#include <algorithm>
#include <limits>

// HIP launch boundary: the row-owner body remains in the shared core header. The launch re-checks the admitted shape
// (rank selects the kernel's K16 unit count; a refused shape never reaches a specialization with other strides).
namespace strix::core::controller_rms_rows {
inline hipError_t enqueue(const Args& a, hipStream_t stream) {
    if (!admits(a, 0U)) return hipErrorInvalidValue;
    ::strix::hip::arch::require_current_device(::strix::hip::arch::LaunchRequirement::Rdna3Layout);
    const dim3 grid(grid_blocks(a.rows));
    switch (a.rank / 16U) {
    case 1U: detail::controller_rms_kernel<1U><<<grid, kThreads, 0, stream>>>(a); break;
    case 2U: detail::controller_rms_kernel<2U><<<grid, kThreads, 0, stream>>>(a); break;
    case 3U: detail::controller_rms_kernel<3U><<<grid, kThreads, 0, stream>>>(a); break;
    case 4U: detail::controller_rms_kernel<4U><<<grid, kThreads, 0, stream>>>(a); break;
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}
}  // namespace strix::core::controller_rms_rows

namespace strix::core::fma_rows {

// Small NK projections use the decode sweep. Wide projections stage RNE BF16
// activations and resident BF16 weights in the shared WMMA tile.
// A round_bf16 projection group of at most gemv_rows::max_rows rows runs the dense_linear tight-row body, whose
// store_tight_rows_bf16 epilogue rounds each completed FP32 sum: the value a later Bf16Round pass over the stored sum
// writes (a NaN stays NaN; only its low payload bits may differ). Larger groups keep their separate round.
inline constexpr bool rounds_in_store(unsigned rows) noexcept { return rows <= gemv_rows::max_rows; }

inline hipError_t enqueue_bf16_rows(const float* input, unsigned input_width, unsigned rows,
                                    std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                    hipStream_t stream) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!input || rows > std::numeric_limits<unsigned>::max() - 128U || !admits(input_width, rows, 16U) ||
        input_width > max_elements / rows || reinterpret_cast<std::uintptr_t>(input) % 16U != 0 ||
        projections.empty() || projections.size() > kGroupMax) return hipErrorInvalidValue;

    const bool round_bf16 = projections.front().round_bf16;
    const auto natural_row_bytes = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto effective_pitch = [natural_row_bytes](const Bf16RowProjection& projection) {
        return projection.row_bytes != 0U ? projection.row_bytes : natural_row_bytes;
    };
    Group group{};
    group.count = static_cast<unsigned>(projections.size());
    for (unsigned p = 0; p < group.count; ++p) {
        const auto& projection = projections[p];
        const auto pitch = effective_pitch(projection);
        if (projection.round_bf16 != round_bf16 || !projection.weights || !projection.output || projection.outputs == 0 || projection.outputs > max_outputs ||
            projection.outputs > max_elements / rows ||
            projection.outputs > std::numeric_limits<unsigned>::max() - 128U ||
            pitch < natural_row_bytes || pitch % 16U != 0U ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / pitch)
            return hipErrorInvalidValue;
    }

    if (rows <= gemv_rows::max_rows) {
        // Same-input projections share one launch; each output keeps its single-launch instructions (bitwise).
        static_assert(kGroupMax <= dense_linear::kTightGroupMax);
        dense_linear::View views[dense_linear::kTightGroupMax]{};
        float* outputs[dense_linear::kTightGroupMax]{};
        for (unsigned p = 0; p < group.count; ++p) {
            views[p] = {reinterpret_cast<const std::uint8_t*>(projections[p].weights), input_width,
                        projections[p].outputs, effective_pitch(projections[p])};
            outputs[p] = projections[p].output;
        }
        if (group.count == 1U)
            return round_bf16 ? dense_linear::enqueue_bf16_tight_rows<true>(views[0], input, outputs[0], rows, stream)
                              : dense_linear::enqueue_bf16_tight_rows(views[0], input, outputs[0], rows, stream);
        return round_bf16
            ? dense_linear::enqueue_bf16_tight_rows_group<true>(views, outputs, group.count, input, rows, stream)
            : dense_linear::enqueue_bf16_tight_rows_group(views, outputs, group.count, input, rows, stream);
    }

    const auto aligned_weights = [](const Bf16RowProjection& projection) {
        return reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U;
    };
    if (rows <= 65535U * f16_tile_gemm::kBM &&
        input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK &&
        std::ranges::all_of(projections, aligned_weights)) {
        f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> wmma_group{};
        const auto pitch = effective_pitch(projections.front());
        for (const auto& projection : projections) {
            if (effective_pitch(projection) != pitch) return hipErrorInvalidValue;
            wmma_group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
        }
        if (pitch == natural_row_bytes)
            f16_tile_gemm::gemm_bf16_projections(input, rows, input_width, wmma_group, stream);
        else
            f16_tile_gemm::gemm_bf16_projections(input, rows, input_width, wmma_group, stream,
                                                 pitch / sizeof(std::uint16_t));
        return hipGetLastError();
    }

    // Preserve existing admission for callers whose weights cannot use vector
    // loads or whose row grid exceeds the WMMA launch limit.
    for (unsigned p = 0; p < group.count; ++p) {
        if (!projections[p].contiguous(input_width)) return hipErrorInvalidValue;
        group.w[p] = projections[p].weights;
        group.y[p] = projections[p].output;
        group.n[p] = projections[p].outputs;
    }
    if (round_bf16) {
        if (admits(input_width, rows, 32U)) launch_auto<true>(input, Weight::BF16, input_width, rows, group, stream);
        else launch<128, 64, 1, 16, true, true>(input, Weight::BF16, input_width, rows, group, stream);
    } else if (admits(input_width, rows, 32U)) {
        launch_auto(input, Weight::BF16, input_width, rows, group, stream);
    } else {
        launch<128, 64, 1, 16, true>(input, Weight::BF16, input_width, rows, group, stream);
    }
    return hipGetLastError();
}

// Rows whose activations arrive as a BF16 operand image (u16 [rows][input_width], element = the RNE BF16 encoding of
// the FP32 value): the shared WMMA tile reads the image as is, the operand bits its FP32 staging makes of those values,
// so each output keeps the bits enqueue_bf16_rows stores for the FP32 rows; half the activation bytes. The tile's
// prefill shapes only (decode rows keep their FP32 sweep): callers check admits_bf16_image_projections before they
// store the image.
inline constexpr bool admits_bf16_image_rows(unsigned input_width, unsigned rows) noexcept {
    return rows > gemv_rows::max_rows && rows <= 65535U * f16_tile_gemm::kBM && input_width >= f16_tile_gemm::kBK &&
           input_width % f16_tile_gemm::kBK == 0U &&
           input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK;
}

// The image route for these projections, asked before the image is stored: the shape (admits_bf16_image_rows) and
// every weight the image launch reads (contiguous rows, 16-byte aligned, outputs within max_outputs, one rounding
// class). A projection outside it keeps its FP32 rows, where enqueue_bf16_rows serves pitched and unaligned weights.
inline bool admits_bf16_image_projections(unsigned input_width, unsigned rows,
                                          std::span<const Bf16RowProjection> projections,
                                          unsigned max_outputs) noexcept {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!admits_bf16_image_rows(input_width, rows) || input_width > max_elements / rows || projections.empty() ||
        projections.size() > kGroupMax)
        return false;
    const bool round_bf16 = projections.front().round_bf16;
    return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
        return projection.weights && projection.output && projection.outputs != 0U &&
               projection.outputs <= max_outputs && projection.outputs <= max_elements / rows &&
               projection.contiguous(input_width) &&
               reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U && projection.round_bf16 == round_bf16;
    });
}

inline hipError_t enqueue_bf16_image_rows(const std::uint16_t* input, unsigned input_width, unsigned rows,
                                          std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                          hipStream_t stream) {
    if (!input || reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
        !admits_bf16_image_projections(input_width, rows, projections, max_outputs))
        return hipErrorInvalidValue;
    f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> group{};
    for (const auto& projection : projections)
        group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
    f16_tile_gemm::gemm_bf16_projections_image(input, rows, input_width, group, stream);
    return hipGetLastError();
}

// The [forget, gain] pair of a low-rank controller with the controller as the tile epilogue:
// output[r][n] = controller_mix(current[r][n], message[r][n], forget sum, gain sum), the same bits as
// enqueue_bf16_rows (FP32 sums) followed by the pointwise controller. hipErrorNotSupported where the shared WMMA tile
// does not take the pair (decode rows, unequal widths, rounded sums, unaligned operands): the caller runs the two-step
// form instead.
inline hipError_t enqueue_bf16_controller_rows(const float* input, unsigned input_width, unsigned rows,
                                               std::span<const Bf16RowProjection> gates, const float* current,
                                               const float* message, float* output, hipStream_t stream) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    for (const auto& gate : gates)
        if (!gate.contiguous(input_width)) return hipErrorInvalidValue;
    if (gates.size() != 2U || rows <= gemv_rows::max_rows) return hipErrorNotSupported;
    const auto& forget = gates[0];
    const auto& gain = gates[1];
    const auto aligned = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U; };
    if (forget.outputs != gain.outputs || forget.round_bf16 || gain.round_bf16 || forget.outputs == 0U ||
        rows > 65535U * f16_tile_gemm::kBM || input_width == 0U || input_width % 16U != 0U ||
        input_width > std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK || !aligned(input) ||
        !aligned(forget.weights) || !aligned(gain.weights) || forget.outputs > max_elements / rows)
        return hipErrorNotSupported;
    if (!current || !message || !output || !forget.weights || !gain.weights) return hipErrorInvalidValue;
    f16_tile_gemm::Bf16ProjectionGroup<2U> group{};
    group.append({forget.weights, forget.outputs, forget.output, forget.round_bf16});
    group.append({gain.weights, gain.outputs, gain.output, gain.round_bf16});
    f16_tile_gemm::gemm_bf16_controller(input, rows, input_width, group, {current, message, output, forget.outputs}, stream);
    return hipGetLastError();
}

// The [forget, gain] pair with the controller epilogue followed by the norm of the conditioned rows
// (ops::norm::enqueue_rms_bf16_rows, parallel form, BF16 scale) as one row-owner launch: output = the norm of
// controller_mix(current, message, forget sum, gain sum), the bits of enqueue_bf16_controller_rows into a separate
// buffer followed by that norm into output. hipErrorNotSupported where the row-owner form does not take the shape
// (controller_rms_rows::admits): the caller runs the two launches.
inline hipError_t enqueue_bf16_controller_rms_rows(const float* input, unsigned input_width, unsigned rows,
                                                   std::span<const Bf16RowProjection> gates, const float* current,
                                                   const float* message, const std::uint16_t* scale, float epsilon,
                                                   float* output, hipStream_t stream) {
    if (gates.size() != 2U) return hipErrorNotSupported;
    const auto& forget = gates[0];
    const auto& gain = gates[1];
    if (!forget.contiguous(input_width) || !gain.contiguous(input_width) || forget.round_bf16 || gain.round_bf16 ||
        forget.outputs != gain.outputs)
        return hipErrorNotSupported;
    const controller_rms_rows::Args args{input,   forget.weights, gain.weights, current, message, scale,
                                         output,  rows,           forget.outputs, input_width, epsilon};
    if (!controller_rms_rows::admits(args, gemv_rows::max_rows)) return hipErrorNotSupported;
    return controller_rms_rows::enqueue(args, stream);
}

}  // namespace strix::core::fma_rows
