#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/norms.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace strix::inkling2 {

struct MiniCudaDenseProjection {
    const std::uint16_t* weights;  // BF16 [output][input]
    unsigned outputs;
    float* output;                // FP32 [row][output]
};

inline bool mini_cuda_dense_aligned(const void* pointer) {
    return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0;
}

inline bool valid_mini_cuda_projection(const MiniCudaDenseProjection& projection, unsigned rows) {
    return projection.outputs != 0 && rows != 0 &&
           projection.outputs <= std::numeric_limits<std::size_t>::max() / sizeof(float) / rows &&
           mini_cuda_dense_aligned(projection.weights) && mini_cuda_dense_aligned(projection.output);
}

// FP32 outputs with the core's chosen reduction order. A Mini stage may then
// round explicitly to widened BF16. Tightly packed device buffers are disjoint.
inline cudaError_t enqueue_mini_cuda_dense_rows(
    const float* input, unsigned input_width, unsigned rows, std::span<const MiniCudaDenseProjection> projections,
    cudaStream_t stream, cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3,
    cuda_core::BigCtx* context = nullptr) {
    constexpr std::size_t max_projections = 4;
    if (!mini_cuda_dense_aligned(input) || input_width == 0 || input_width % 8U != 0 || rows == 0 ||
        input_width > std::numeric_limits<std::size_t>::max() / sizeof(float) / rows || projections.empty() ||
        projections.size() > max_projections) return cudaErrorInvalidValue;

    std::array<cuda_core::DenseWeight, max_projections> weights;
    std::uint64_t total_outputs = 0;
    for (std::size_t i = 0; i < projections.size(); ++i) {
        const auto& projection = projections[i];
        if (!valid_mini_cuda_projection(projection, rows) || projection.output == input ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / input_width) {
            return cudaErrorInvalidValue;
        }
        total_outputs += projection.outputs;
        weights[i] = {projection.weights, cuda_core::WeightFormat::bf16, projection.outputs, input_width};
    }
    if (total_outputs > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidValue;

    std::array<cuda_core::DenseWeight, max_projections> short_weights{};
    std::array<float*, max_projections> short_outputs{};
    unsigned short_count = 0;
    cuda_core::BigAct activation{};
    const bool eligible = context && rows > cuda_core::kMaxRows && arithmetic == cuda_core::DenseArith::mma3;
    for (std::size_t i = 0; i < projections.size(); ++i) {
        if (eligible && cuda_core::dense_big_supported(weights[i])) {
            if (activation.planes == nullptr) {
                const auto status = cuda_core::dense_big_split(
                    input, rows, input_width, cuda_core::WeightFormat::bf16, 3, &activation, stream, context);
                if (status != cudaSuccess) return status;
            }
            const auto status = cuda_core::dense_big_gemm(weights[i], activation, projections[i].output,
                nullptr, cuda_core::BigEpi::none, stream, context);
            if (status != cudaSuccess) return status;
        } else {
            short_weights[short_count] = weights[i];
            short_outputs[short_count++] = projections[i].output;
        }
    }
    for (unsigned first = 0; short_count != 0U && first < rows;) {
        const unsigned count = std::min(rows - first, cuda_core::kMaxRows);
        std::array<float*, max_projections> outputs;
        for (unsigned i = 0; i < short_count; ++i)
            outputs[i] = short_outputs[i] + std::size_t{first} * short_weights[i].out;
        const auto status = cuda_core::dense_rows_multi(short_weights.data(), outputs.data(), static_cast<int>(short_count),
            input + std::size_t{first} * input_width, count, nullptr, stream, arithmetic);
        if (status != cudaSuccess) return status;
        first += count;
    }
    return cudaSuccess;
}

inline cudaError_t enqueue_mini_cuda_rmsnorm(const float* input, const std::uint16_t* gain, float* output,
                                             unsigned rows, cudaStream_t stream) {
    const cuda_core::GroupedView layout{rows, 1U, kHiddenWidth};
    return cuda_core::rmsnorm_bf16(input, layout, gain, output, layout, 1.0e-6F, stream);
}

// The caller validates token IDs against kVocabulary before upload.
inline cudaError_t enqueue_mini_cuda_embeddings(const std::uint16_t* table, const std::uint16_t* gain,
                                                const std::uint32_t* ids, float* output, unsigned rows,
                                                cudaStream_t stream) {
    if (!table || !gain || !ids || !output || rows == 0 || rows > 65535U) return cudaErrorInvalidValue;
    const cuda_core::DenseWeight embed{table, cuda_core::WeightFormat::bf16, kVocabulary, kHiddenWidth};
    const auto status = cuda_core::embedding_rows(embed, ids, output, rows, stream);
    if (status != cudaSuccess) return status;
    return enqueue_mini_cuda_rmsnorm(output, gain, output, rows, stream);
}

}  // namespace strix::inkling2
