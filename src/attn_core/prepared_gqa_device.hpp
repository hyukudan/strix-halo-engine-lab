#pragma once

#include "attn_core/prepared_gqa.hpp"
#include "attn_core/prepared_gqa_attention.hpp"
#include "core/dense_row_plan.hpp"

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/prepared_rows_device.hpp"
#endif
#if defined(__HIPCC__)
#include "core/hip_short_convolution.hpp"
#elif defined(__CUDACC__)
#include "cuda/core/short_convolution.cuh"
#endif

namespace strix::attn_core {

struct PreparedGqaInputs {
    const float* query;
    const float* key;
    const float* value;
};

struct PreparedGqaWeights {
    using Role = PreparedGqaProjection;
    static constexpr unsigned role_count = static_cast<unsigned>(Role::output) + 1U;
    std::array<const std::uint16_t*, role_count> matrix{};
    const std::uint16_t* query_gain{};
    const std::uint16_t* relative_gain{};
    const float* key_convolution{};
    const float* value_convolution{};

    const std::uint16_t* projection(Role role) const { return matrix[static_cast<unsigned>(role)]; }
};

struct PreparedGqaWorkspace {
    float* raw_keys;                      // [row][KvWidth]
    float* raw_values;                    // [row][KvWidth]
    float* dynamic_keys;                  // [row][KvHeads][ConvTaps], full visits only
    float* dynamic_values;                // [row][KvHeads][ConvTaps], full visits only
    float* relative_features;             // [row][QueryHeads][RelativeWidth]
    float* key_history;                   // [ConvTaps - 1][KvWidth], per visit
    float* value_history;                 // [ConvTaps - 1][KvWidth], per visit
};

struct PreparedGqaRows {
    float* query;            // [row][QueryWidth]
    float* key;              // [row][KvWidth]
    float* value;            // [row][KvWidth]
    float* relative_logits;  // [row][QueryHeads][RelativeExtent], natural units
    float* sinks;            // [row][QueryHeads], natural units
    float* gate_logits;      // [row][QueryWidth], rounded by the output gate
};

struct PreparedGqaOutputRows {
    float* attended;         // [row][QueryWidth]
    float* gated;            // [row][QueryWidth]
    float* output;           // [row][HiddenWidth]
};

using PreparedGqaLinear = core::Bf16RowProjection;

inline bool prepared_gqa_aligned(const void* pointer) {
    return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0;
}

namespace detail {

template <class Shape, class Backend>
typename Backend::Status prepared_gqa_kv_convolution(
    const Backend& backend, bool full, unsigned rows, const float* convolution, float* activation,
    float* dynamic, float* history, float* output) {
    if (auto status = backend.round(activation, std::size_t{rows} * Shape::kKvWidth); status != Backend::success)
        return status;
    if (full) {
        if (auto status = backend.round(dynamic, std::size_t{rows} * Shape::kKvHeads * Shape::kConvTaps); status != Backend::success)
            return status;
    }
    const core::ShortConvolution plan{Shape::kKvWidth, Shape::kConvTaps, full ? Shape::kKvHeads : 0U, 1.0F,
                                      full ? Shape::kFullConvDivisor : Shape::kLocalConvDivisor};
    return backend.convolve(plan, rows, activation, convolution, full ? dynamic : nullptr, history, output);
}

}  // namespace detail

// Normalized inputs may alias on a local-attention visit. Scratch is disjoint.
template <class Shape, class Backend>
typename Backend::Status enqueue_prepared_gqa(
    const Backend& backend, bool full, unsigned first_position, unsigned rows,
    const PreparedGqaInputs& inputs, const PreparedGqaWeights& weights,
    const PreparedGqaWorkspace& work, const PreparedGqaRows& output, unsigned* failure = nullptr) {
    using Role = PreparedGqaProjection;
    if (rows == 0 || rows > 65535U || first_position >= Shape::kMaxContext ||
        rows > Shape::kMaxContext - first_position || !weights.query_gain || !weights.relative_gain ||
        !weights.key_convolution || !weights.value_convolution) return Backend::invalid;
    const std::array<const float*, 14> required{inputs.query, inputs.key, inputs.value, output.query,
        output.key, output.value, output.relative_logits, output.sinks, output.gate_logits, work.raw_keys, work.raw_values,
        work.relative_features, work.key_history, work.value_history};
    if (!std::ranges::all_of(required, prepared_gqa_aligned) ||
        (full && (!prepared_gqa_aligned(work.dynamic_keys) || !prepared_gqa_aligned(work.dynamic_values))))
        return Backend::invalid;
    for (unsigned i = 0; i < weights.matrix.size(); ++i) {
        const auto role = static_cast<Role>(i);
        if (role == Role::output || (!full && (role == Role::key_taps || role == Role::value_taps))) continue;
        if (!prepared_gqa_aligned(weights.projection(role))) return Backend::invalid;
    }

    const auto project = [&](Role role, float* to) {
        return PreparedGqaLinear{weights.projection(role), Shape::projection_width(role), to};
    };
    const auto linear = [&](const float* from, std::span<const PreparedGqaLinear> projections) {
        return backend.linear(from, Shape::kHiddenWidth, rows, projections);
    };
    const std::array q_projections{project(Role::query, output.query), project(Role::relative, work.relative_features),
                                  project(Role::sink, output.sinks), project(Role::output_gate, output.gate_logits)};
    const std::array k_projections{project(Role::key, work.raw_keys), project(Role::key_taps, work.dynamic_keys)};
    const std::array v_projections{project(Role::value, work.raw_values), project(Role::value_taps, work.dynamic_values)};
    if (!full && inputs.query == inputs.key && inputs.query == inputs.value) {
        const std::array all{q_projections[0], q_projections[1], q_projections[2], q_projections[3],
                             k_projections[0], v_projections[0]};
        if (auto status = linear(inputs.query, all); status != Backend::success) return status;
    } else {
        if (auto status = linear(inputs.query, q_projections); status != Backend::success) return status;
        if (!full && inputs.key == inputs.value) {
            const std::array kv{ k_projections[0], v_projections[0] };
            if (auto status = linear(inputs.key, kv); status != Backend::success) return status;
        } else {
            const std::size_t count = full ? 2U : 1U;
            if (auto status = linear(inputs.key, std::span(k_projections).first(count)); status != Backend::success)
                return status;
            if (auto status = linear(inputs.value, std::span(v_projections).first(count)); status != Backend::success)
                return status;
        }
    }

    if (auto status = detail::prepared_gqa_kv_convolution<Shape>(backend, full, rows, weights.key_convolution,
            work.raw_keys, work.dynamic_keys, work.key_history, output.key);
        status != Backend::success) return status;
    if (auto status = detail::prepared_gqa_kv_convolution<Shape>(backend, full, rows, weights.value_convolution,
            work.raw_values, work.dynamic_values, work.value_history, output.value);
        status != Backend::success) return status;

    if (auto status = backend.clip(output.query, weights.query_gain, rows, Shape::kQueryHeads,
            Shape::kHeadWidth, Shape::kQkClip, full, first_position); status != Backend::success) return status;
    if (auto status = backend.clip(output.key, nullptr, rows, Shape::kKvHeads, Shape::kHeadWidth,
            Shape::kQkClip, false, 0); status != Backend::success) return status;
    if (auto status = backend.round(output.value, std::size_t{rows} * Shape::kKvWidth); status != Backend::success)
        return status;
    if (auto status = backend.rms_values(output.value, rows); status != Backend::success) return status;
    if (auto status = backend.clip(work.relative_features, weights.relative_gain, rows, Shape::kQueryHeads,
            Shape::kRelativeWidth, Shape::kRelativeClip, full, first_position); status != Backend::success) return status;
    if (auto status = backend.relative_logits(work.relative_features, weights.projection(Role::relative_bias),
            output.relative_logits, rows, failure); status != Backend::success) return status;
    return backend.round(output.sinks, std::size_t{rows} * Shape::kQueryHeads, failure);
}

template <class Shape, class Backend>
typename Backend::Status enqueue_prepared_gqa_output(
    const Backend& backend, unsigned rows, const std::uint16_t* output_weight, const float* attended,
    const float* gate_logits, float* gated, float* output) {
    if (rows == 0 || rows > Shape::kMaxContext || !attended || !gate_logits || !prepared_gqa_aligned(gated) ||
        !prepared_gqa_aligned(output_weight) || !prepared_gqa_aligned(output) || gated == output)
        return Backend::invalid;
    if (auto status = backend.gate(attended, gate_logits, gated, std::size_t{rows} * Shape::kQueryWidth);
        status != Backend::success) return status;
    const std::array projection{PreparedGqaLinear{output_weight, Shape::kHiddenWidth, output}};
    if (auto status = backend.linear(gated, Shape::kQueryWidth, rows, projection); status != Backend::success) return status;
    return backend.round(output, std::size_t{rows} * Shape::kHiddenWidth);
}

template <class Shape, class Backend>
typename Backend::Status enqueue_prepared_gqa_post_convolution(
    const Backend& backend, unsigned rows, const float* weights, const float* input,
    float* history, float* output) {
    if (rows == 0 || rows > Shape::kMaxContext || !weights || !input || !history || !output || input == output)
        return Backend::invalid;
    constexpr core::ShortConvolution plan{Shape::kHiddenWidth, Shape::kConvTaps, 0U, 1.0F, Shape::kLocalConvDivisor};
    return backend.template convolve<true>(plan, rows, input, weights, nullptr, history, output);
}

// The backend admits the cache and workspace before any preparation writes.
template <class Shape, class Backend, class Append>
typename Backend::Status enqueue_prepared_gqa_forward(
    const Backend& backend, bool full, unsigned first_position, unsigned rows, const PreparedGqaInputs& inputs,
    const PreparedGqaWeights& weights, const PreparedGqaWorkspace& work, const PreparedGqaRows& prepared,
    const PreparedGqaOutputRows& output, unsigned* operand_failure, Append&& append) {
    const auto* output_weight = weights.projection(PreparedGqaProjection::output);
    if (!prepared_gqa_aligned(output_weight) || !prepared_gqa_aligned(output.gated) ||
        !prepared_gqa_aligned(output.output) || output.gated == output.output) return Backend::invalid;
    if (auto status = enqueue_prepared_gqa<Shape>(backend, full, first_position, rows, inputs, weights, work,
                                                  prepared, operand_failure); status != Backend::success) return status;
    if (auto status = append(); status != Backend::success) return status;
    return enqueue_prepared_gqa_output<Shape>(
        backend, rows, output_weight, output.attended, prepared.gate_logits, output.gated, output.output);
}

#if defined(__HIPCC__) || defined(__CUDACC__)

// Bind static convolution weights once before any row uses the tables.
inline auto enqueue_bind_convolution_weights(std::span<const std::uint16_t> source, float* lag_weights,
                                             unsigned width, unsigned taps, bool channel_taps,
                                             core::prepared_rows_device::Stream stream) {
    namespace prep = core::prepared_rows_device;
    if (!lag_weights || width == 0U || taps == 0U || source.size() != std::size_t{width} * taps)
        return prep::invalid;
    return prep::elements(source.size(), prep::WidenConvolutionBf16{source.data(), lag_weights, width, taps, channel_taps},
                          stream);
}

// One HIP/CUDA implementation for all prepared elementwise stages. Backends
// supply their dense and RMS launch adapters.
template<class Shape, class Stream>
struct PreparedGqaRowsDevice {
    Stream stream;

    template <bool RoundOutput = false>
    auto convolve(const core::ShortConvolution& plan, unsigned rows, const float* input, const float* lag_weights,
                  const float* dynamic, float* history, float* output) const {
#if defined(__HIPCC__)
        namespace conv = core::hip_short_conv;
#else
        namespace conv = cuda_core::short_conv;
#endif
        return conv::launch_with_store<RoundOutput>(plan, {rows, input, lag_weights, dynamic, history, output}, stream);
    }

    auto round(float* row, std::size_t elements, unsigned* failure = nullptr) const {
        namespace prep = core::prepared_rows_device;
        if (failure) return prep::elements(elements, prep::CheckedBf16Round{row, failure}, stream);
        return prep::elements(elements, prep::Bf16Round{row}, stream);
    }
    auto clip(float* row, const std::uint16_t* gain, unsigned rows, unsigned heads, unsigned width,
              float threshold, bool position_scale, unsigned first_position) const {
        namespace prep = core::prepared_rows_device;
        const auto scale = position_scale
            ? prep::LogPositionScale{first_position, Shape::kPositionFloor, Shape::kPositionAlpha}
            : prep::LogPositionScale{};
        return prep::l2_clip_bf16({row, gain, row, rows, heads, width, Shape::kAttentionEpsilon, threshold, scale}, stream);
    }
    auto relative_logits(const float* input, const std::uint16_t* weights, float* output, unsigned rows,
                         unsigned* failure) const {
        namespace prep = core::prepared_rows_device;
        const auto size = std::size_t{rows} * Shape::kQueryHeads * Shape::kRelativeExtent;
        if (failure) return prep::elements(size,
            prep::ShortBf16Projection<Shape::kRelativeWidth, true>{input, weights, output, Shape::kRelativeExtent, failure},
            stream);
        return prep::elements(size,
            prep::ShortBf16Projection<Shape::kRelativeWidth>{input, weights, output, Shape::kRelativeExtent}, stream);
    }
    auto gate(const float* input, const float* logits, float* output, std::size_t elements) const {
        namespace prep = core::prepared_rows_device;
        return prep::elements(elements, prep::Bf16SigmoidProduct{input, logits, output}, stream);
    }
};

#endif

}  // namespace strix::attn_core
