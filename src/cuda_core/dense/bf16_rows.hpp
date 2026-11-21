#pragma once

#include "core/dense_row_plan.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/norms.hpp"

#include <array>
#include <limits>

namespace strix::cuda_core {

inline constexpr std::size_t kMaxBf16RowProjections = 4;

// Tall MMA3 projections share a split activation. Remaining projections
// stay grouped and use consecutive tight short windows in the requested mode.
inline cudaError_t enqueue_bf16_dense_rows(const float* input, unsigned input_width, unsigned rows,
                                           std::span<const core::Bf16RowProjection> projections,
                                           cudaStream_t stream, DenseArith arithmetic, BigCtx* context) {
    const auto aligned = [](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0;
    };
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!aligned(input) || input_width == 0 || input_width % 8U != 0 || rows == 0 ||
        input_width > max_floats / rows || projections.empty() || projections.size() > kMaxBf16RowProjections)
        return cudaErrorInvalidValue;

    const bool round_bf16 = projections.front().round_bf16;
    if (round_bf16 && arithmetic != DenseArith::mma3) return cudaErrorInvalidValue;
    std::array<DenseWeight, kMaxBf16RowProjections> weights;
    std::uint64_t total_outputs = 0;
    for (std::size_t i = 0; i < projections.size(); ++i) {
        const auto& projection = projections[i];
        if (projection.round_bf16 != round_bf16 ||
            (round_bf16 && (input_width % 32U != 0U || projection.outputs % 16U != 0U)) ||
            projection.outputs == 0 || projection.outputs > max_floats / rows ||
            !aligned(projection.weights) || !aligned(projection.output) || projection.output == input ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / input_width)
            return cudaErrorInvalidValue;
        total_outputs += projection.outputs;
        weights[i] = {projection.weights, WeightFormat::bf16, projection.outputs, input_width};
    }
    if (total_outputs > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidValue;

    std::array<DenseWeight, kMaxBf16RowProjections> short_weights{};
    std::array<float*, kMaxBf16RowProjections> short_outputs{};
    unsigned short_count = 0;
    BigAct activation{};
    const bool eligible = context && rows > kMaxRows && arithmetic == DenseArith::mma3;
    for (std::size_t i = 0; i < projections.size(); ++i) {
        if (eligible && dense_big_supported(weights[i])) {
            if (activation.planes == nullptr) {
                const auto status = dense_big_split(input, rows, input_width, WeightFormat::bf16, 3,
                                                    &activation, stream, context);
                if (status != cudaSuccess) return status;
            }
            const auto status = dense_big_gemm(weights[i], activation, projections[i].output,
                                               nullptr, round_bf16 ? BigEpi::round_bf16 : BigEpi::none, stream, context);
            if (status != cudaSuccess) return status;
        } else {
            short_weights[short_count] = weights[i];
            short_outputs[short_count++] = projections[i].output;
        }
    }
    if (short_count == 0) return cudaSuccess;
    return core::for_each_dense_chunk<kMaxRows>(rows, cudaSuccess, [&](unsigned first, unsigned count) {
        std::array<float*, kMaxBf16RowProjections> outputs;
        for (unsigned i = 0; i < short_count; ++i)
            outputs[i] = short_outputs[i] + std::size_t{first} * short_weights[i].out;
        if (round_bf16)
            return dense_bf16_mma3_rows_round(short_weights.data(), outputs.data(), static_cast<int>(short_count),
                                              input + std::size_t{first} * input_width, count, stream);
        return dense_rows_multi(short_weights.data(), outputs.data(), static_cast<int>(short_count),
                                input + std::size_t{first} * input_width, count, nullptr, stream, arithmetic);
    });
}

// Device token IDs must already be validated. The gather widens BF16 and
// the in-place RMSNorm applies BF16 gain before its single BF16 output round.
inline cudaError_t enqueue_bf16_embeddings_rmsnorm(const std::uint16_t* table, unsigned vocabulary,
                                                   unsigned width, const std::uint16_t* gain, float epsilon,
                                                   const std::uint32_t* ids, float* output, unsigned rows,
                                                   cudaStream_t stream) {
    if (!table || !gain || !ids || !output || rows == 0 || rows > 65535U || vocabulary == 0 ||
        width == 0 || width > 8192U || !(epsilon > 0.0F)) return cudaErrorInvalidValue;
    const DenseWeight embed{table, WeightFormat::bf16, vocabulary, width};
    if (const auto status = embedding_rows(embed, ids, output, rows, stream); status != cudaSuccess) return status;
    const GroupedView layout{rows, 1U, width};
    return rmsnorm_bf16(output, layout, gain, output, layout, epsilon, stream);
}

}  // namespace strix::cuda_core
