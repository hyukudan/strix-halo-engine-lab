#pragma once

#include "inkling2_mini_config.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#if defined(__HIPCC__)
#include "core/fma_rows.hpp"
#include "core/prepared_rows_device.hpp"
#include "ops/norm/norm_presets.hpp"
#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <stdexcept>
#endif

namespace strix::inkling2 {

// Resident original BF16 matrix [outputs][input_width] and tight FP32 result
// [rows][outputs]. Each member of a group uses the same activation rows.
struct MiniHipDenseProjection {
    const std::uint16_t* weights{};
    unsigned outputs{};
    float* output{};
};

// With null scale, normalize alone. Result is the BF16 store widened to F32.
// Buffers may coincide when both row strides match, or must be disjoint.
struct MiniHipRmsRows {
    const float* input{};
    const std::uint16_t* scale{};
    float* output{};
    unsigned rows{};
    unsigned width{};
    std::size_t input_row{};
    std::size_t output_row{};
};

enum class MiniHipRmsReduction { parallel, serial_reference };

#if defined(__HIPCC__)

// The result is the shared ascending-K FMA chain, with model-specific rounding
// left to the consumer. Caller keeps all buffers valid and nonoverlapping.
inline hipError_t enqueue_mini_hip_dense_rows(const float* input, unsigned input_width, unsigned rows,
                                               std::span<const MiniHipDenseProjection> projections,
                                               hipStream_t stream) {
    namespace fma = core::fma_rows;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!input || rows == 0 || rows > std::numeric_limits<unsigned>::max() - 128U ||
        input_width == 0 || input_width % 16U != 0 || input_width > max_elements / rows ||
        reinterpret_cast<std::uintptr_t>(input) % 16U != 0 ||
        projections.empty() || projections.size() > fma::kGroupMax) return hipErrorInvalidValue;

    fma::Group group{};
    group.count = static_cast<unsigned>(projections.size());
    for (unsigned p = 0; p < group.count; ++p) {
        const auto& projection = projections[p];
        if (!projection.weights || !projection.output || projection.outputs == 0 || projection.outputs > kVocabulary ||
            projection.outputs > max_elements / rows)
            return hipErrorInvalidValue;
        group.w[p] = projection.weights;
        group.y[p] = projection.output;
        group.n[p] = projection.outputs;
    }

    if (input_width % 32U == 0) {
        fma::launch_auto(input, fma::Weight::BF16, input_width, rows, group, stream);
    } else {
        // In particular, the relative-logit projection reads 16 channels per
        // head. Its matrix must be transposed once on admission to this layout.
        fma::launch<128, 64, 1, 16, true>(input, fma::Weight::BF16, input_width, rows, group, stream);
    }
    return hipGetLastError();
}

namespace dense_hip_detail {

struct RmsTraits : ops::norm::DeclaredF32Traits<ops::norm::OutputForm::bf16_once> {
    using Scale = std::uint16_t;
    __device__ static float inv_sqrt(float value) { return __fdiv_rn(1.0F, __fsqrt_rn(value)); }
    __device__ static float load(const Scale* scale, ScaleFormat, unsigned i) {
        return weight::scalar_float::u2f(std::uint32_t{scale[i]} << 16U);
    }
};

}  // namespace dense_hip_detail

// Parallel is the shared lane-strided reduction, with a different FP32 sum
// order from the CPU. Serial reference retains the CPU's ascending sum for
// device qualification. Both round the scaled output to BF16 once.
inline hipError_t enqueue_mini_hip_rms(const MiniHipRmsRows& args, hipStream_t stream,
                                       MiniHipRmsReduction reduction = MiniHipRmsReduction::parallel) {
    constexpr unsigned lanes = 256;
    constexpr float epsilon = 1.0e-6F;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.output || args.rows == 0 || args.rows > kMaxContext || args.width == 0 ||
        args.width > std::numeric_limits<unsigned>::max() - lanes ||
        args.input_row < args.width || args.output_row < args.width ||
        args.rows > max_elements / args.input_row || args.rows > max_elements / args.output_row ||
        (args.input == args.output && args.input_row != args.output_row)) return hipErrorInvalidValue;
    using Traits = dense_hip_detail::RmsTraits;
    if (reduction == MiniHipRmsReduction::serial_reference) {
        ops::norm::rmsnorm_strided_kernel<Traits, 1><<<dim3(args.rows), 1, 0, stream>>>(
            args.input, args.input_row, args.scale, 0, args.width, epsilon, args.output, args.output_row);
    } else if (reduction == MiniHipRmsReduction::parallel) {
        ops::norm::rmsnorm_strided_kernel<Traits, lanes><<<dim3(args.rows), lanes, 0, stream>>>(
            args.input, args.input_row, args.scale, 0, args.width, epsilon, args.output, args.output_row);
    } else {
        return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

// Host token IDs are validated by the shared gather before enqueue. The
// output can feed the embedding RMS with scale from embed_norm.weight.
inline void enqueue_mini_hip_embeddings(std::span<const std::uint16_t> table,
                                        std::span<const std::uint32_t> ids,
                                        std::span<float> output, hipStream_t stream) {
    if (table.size() != std::size_t{kVocabulary} * kHiddenWidth ||
        ids.size() > kMaxContext || output.size() < ids.size() * kHiddenWidth)
        throw std::invalid_argument("invalid Inkling 2 Mini embedding buffers");
    constexpr unsigned ggml_bf16 = 30U;
    launch_hip_embedding_rows({table.data(), ggml_bf16, kHiddenWidth, kVocabulary},
                              HipActivationStorage::fp32, ids, output.data(), kHiddenWidth, stream);
}

// Original unembed weights contain padded vocabulary rows. Project only the
// valid output vocabulary. The softcapped logits are ready for scoring or
// selection afterward on the same stream.
inline hipError_t enqueue_mini_hip_head(const float* input, unsigned rows,
                                       std::span<const std::uint16_t> weights,
                                       std::span<float> logits, hipStream_t stream) {
    const std::size_t elements = std::size_t{rows} * kOutputVocabulary;
    if (weights.size() != std::size_t{kVocabulary} * kHiddenWidth || logits.size() < elements)
        return hipErrorInvalidValue;
    const MiniHipDenseProjection projection{weights.data(), kOutputVocabulary, logits.data()};
    const auto status = enqueue_mini_hip_dense_rows(input, kHiddenWidth, rows, {&projection, 1U}, stream);
    if (status != hipSuccess) return status;
    namespace prep = core::prepared_rows_device;
    return prep::elements(elements, prep::Bf16Softcap{logits.data(), kLogitSoftcap}, stream);
}

inline void enqueue_mini_hip_greedy(std::span<const ExecutionBufferAddress> logits,
                                   ExecutionBufferAddress partials, HipGreedyRowsReply* replies,
                                   hipStream_t stream) {
    enqueue_hip_greedy_rows(logits, kOutputVocabulary, partials, replies, stream);
}

#endif

}  // namespace strix::inkling2
