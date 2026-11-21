#pragma once

#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_state.hpp"

#include <string>

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/prepared_rows_device.hpp"
#endif

namespace strix::inkling2 {

struct MiniProjectionInputs {
    const float* query;
    const float* key;
    const float* value;
};

struct MiniProjectionWeights {
    using Role = MiniAttentionProjection;
    static constexpr unsigned role_count = static_cast<unsigned>(Role::output) + 1U;
    std::array<const std::uint16_t*, role_count> matrix{};
    const std::uint16_t* query_gain{};
    const std::uint16_t* relative_gain{};
    const std::uint16_t* key_convolution{};
    const std::uint16_t* value_convolution{};

    const std::uint16_t* projection(Role role) const { return matrix[static_cast<unsigned>(role)]; }
};

template <class Resident>
[[nodiscard]] MiniProjectionWeights mini_device_projection_weights(const Resident& resident, LayerVisit visit) {
    if (!valid_mini_visit(visit)) throw std::invalid_argument("invalid Mini device projection visit");
    MiniProjectionWeights result;
    const auto prefix = "layers." + std::to_string(visit.physical_layer) + ".";
    const auto read = [&](std::string_view suffix, std::size_t expected) {
        const auto view = resident.bf16(prefix + std::string(suffix));
        if (view.size() != expected) throw std::invalid_argument("invalid Mini projection weight: " + std::string(suffix));
        return view.data();
    };
    const bool full = is_full_attention(visit.physical_layer);
    using Role = MiniAttentionProjection;
    for (unsigned i = 0; i < result.matrix.size(); ++i) {
        const auto role = static_cast<Role>(i);
        if (!full && (role == Role::key_taps || role == Role::value_taps)) continue;
        const auto spec = mini_attention_projection_spec(role);
        result.matrix[i] = read(spec.tensor_suffix, std::size_t{spec.input_width} * spec.output_width);
    }
    result.query_gain = read("attn.q_l2_clip.scale", kQueryWidth);
    result.relative_gain = read("attn.rel_l2_clip.scale", kQueryHeads * kRelativeWidth);
    result.key_convolution = read(full ? "attn.k_sconv.static_w" : "attn.k_sconv.weight", kKvWidth * kConvTaps);
    result.value_convolution = read(full ? "attn.v_sconv.static_w" : "attn.v_sconv.weight", kKvWidth * kConvTaps);
    return result;
}

struct MiniProjectionWorkspace {
    float* raw_keys;                      // [row][kKvWidth]
    float* raw_values;                    // [row][kKvWidth]
    float* dynamic_keys;                  // [row][kKvHeads][kConvTaps], full visits only
    float* dynamic_values;                // [row][kKvHeads][kConvTaps], full visits only
    float* relative_features;             // [row][kQueryHeads][kRelativeWidth]
    float* lag_weights;                   // [kConvTaps][kKvWidth], shared between ordered K and V kernels
    float* key_history;                   // [kConvTaps - 1][kKvWidth], per visit
    float* value_history;                 // [kConvTaps - 1][kKvWidth], per visit
};

struct MiniPreparedAttention {
    float* query;            // [row][kQueryWidth]
    float* key;              // [row][kKvWidth]
    float* value;            // [row][kKvWidth]
    float* relative_logits;  // [row][kQueryHeads][kRelativeExtent], natural units
    float* sinks;            // [row][kQueryHeads], natural units
    float* gate_logits;      // [row][kQueryWidth], rounded by the output gate
};

struct MiniAttentionOutputRows {
    float* attended;         // [row][kQueryWidth]
    float* gated;            // [row][kQueryWidth]
    float* output;           // [row][kHiddenWidth]
};

struct MiniAttentionLinear {
    const std::uint16_t* weights;
    unsigned outputs;
    float* output;
};

inline bool mini_attention_device_aligned(const void* pointer) {
    return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0;
}

namespace detail {

inline constexpr float kMiniAttentionEpsilon = 1e-6F;

template <class Backend>
typename Backend::Status mini_device_kv_convolution(
    const Backend& backend, bool full, unsigned rows, const std::uint16_t* convolution, float* activation,
    float* dynamic, float* lag_weights, float* history, float* output) {
    if (auto status = backend.round(activation, std::size_t{rows} * kKvWidth); status != Backend::success)
        return status;
    if (full) {
        if (auto status = backend.round(dynamic, std::size_t{rows} * kKvHeads * kConvTaps); status != Backend::success)
            return status;
    }
    if (auto status = backend.widen_convolution(convolution, lag_weights, kKvWidth, kConvTaps, !full);
        status != Backend::success) return status;
    const core::ShortConvolution plan{kKvWidth, kConvTaps, full ? kKvHeads : 0U, 1.0F,
                                      full ? 3.0F : 2.2360679774997898F};
    return backend.convolve(plan, rows, activation, lag_weights, full ? dynamic : nullptr, history, output);
}

}  // namespace detail

// Normalized inputs may alias on a local-attention visit. Scratch is disjoint
// except for the lag-weight staging reused in order on the same stream.
template <class Backend>
typename Backend::Status enqueue_mini_prepared_attention(
    const Backend& backend, LayerVisit visit, unsigned first_position, unsigned rows,
    const MiniProjectionInputs& inputs, const MiniProjectionWeights& weights,
    const MiniProjectionWorkspace& work, const MiniPreparedAttention& output) {
    using Role = MiniAttentionProjection;
    if (!valid_mini_visit(visit) || rows == 0 || rows > 65535U || first_position >= kMaxContext ||
        rows > kMaxContext - first_position || !weights.query_gain || !weights.relative_gain ||
        !weights.key_convolution || !weights.value_convolution) return Backend::invalid;
    const bool full = is_full_attention(visit.physical_layer);
    const std::array<const float*, 15> required{inputs.query, inputs.key, inputs.value, output.query,
        output.key, output.value, output.relative_logits, output.sinks, output.gate_logits, work.raw_keys, work.raw_values,
        work.relative_features, work.lag_weights, work.key_history, work.value_history};
    if (!std::ranges::all_of(required, mini_attention_device_aligned) ||
        (full && (!mini_attention_device_aligned(work.dynamic_keys) || !mini_attention_device_aligned(work.dynamic_values))))
        return Backend::invalid;
    for (unsigned i = 0; i < weights.matrix.size(); ++i) {
        const auto role = static_cast<Role>(i);
        if (role == Role::output || (!full && (role == Role::key_taps || role == Role::value_taps))) continue;
        if (!mini_attention_device_aligned(weights.projection(role))) return Backend::invalid;
    }

    const auto project = [&](Role role, float* to) {
        return MiniAttentionLinear{weights.projection(role), mini_attention_projection_spec(role).output_width, to};
    };
    const auto linear = [&](const float* from, std::span<const MiniAttentionLinear> projections) {
        return backend.linear(from, kHiddenWidth, rows, projections);
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

    if (auto status = detail::mini_device_kv_convolution(backend, full, rows, weights.key_convolution,
            work.raw_keys, work.dynamic_keys, work.lag_weights, work.key_history, output.key);
        status != Backend::success) return status;
    if (auto status = detail::mini_device_kv_convolution(backend, full, rows, weights.value_convolution,
            work.raw_values, work.dynamic_values, work.lag_weights, work.value_history, output.value);
        status != Backend::success) return status;

    if (auto status = backend.clip(output.query, weights.query_gain, rows, kQueryHeads,
            kHeadWidth, detail::kQkClip, full, first_position); status != Backend::success) return status;
    if (auto status = backend.clip(output.key, nullptr, rows, kKvHeads, kHeadWidth,
            detail::kQkClip, false, 0); status != Backend::success) return status;
    if (auto status = backend.round(output.value, std::size_t{rows} * kKvWidth); status != Backend::success)
        return status;
    if (auto status = backend.rms_values(output.value, rows); status != Backend::success) return status;
    if (auto status = backend.clip(work.relative_features, weights.relative_gain, rows, kQueryHeads,
            kRelativeWidth, detail::kRelativeClip, full, first_position); status != Backend::success) return status;
    if (auto status = backend.relative_logits(work.relative_features, weights.projection(Role::relative_bias),
            output.relative_logits, rows); status != Backend::success) return status;
    return backend.round(output.sinks, std::size_t{rows} * kQueryHeads);
}

template <class Backend>
typename Backend::Status enqueue_mini_prepared_attention_output(
    const Backend& backend, unsigned rows, const std::uint16_t* output_weight, const float* attended,
    const float* gate_logits, float* gated, float* output) {
    if (rows == 0 || rows > kMaxContext || !attended || !gate_logits || !mini_attention_device_aligned(gated) ||
        !mini_attention_device_aligned(output_weight) || !mini_attention_device_aligned(output) || gated == output)
        return Backend::invalid;
    if (auto status = backend.gate(attended, gate_logits, gated, std::size_t{rows} * kQueryWidth);
        status != Backend::success) return status;
    const std::array projection{MiniAttentionLinear{output_weight, kHiddenWidth, output}};
    if (auto status = backend.linear(gated, kQueryWidth, rows, projection); status != Backend::success) return status;
    return backend.round(output, std::size_t{rows} * kHiddenWidth);
}

template<class Backend>
typename Backend::Status enqueue_mini_prepared_post_convolution(
    const Backend& backend, unsigned rows, const std::uint16_t* weights, const float* input, float* lag_weights,
    float* history, float* output) {
    if (rows == 0 || rows > kMaxContext || !weights || !input || !lag_weights || !history || !output || input == output)
        return Backend::invalid;
    if (auto status = backend.widen_convolution(weights, lag_weights, kHiddenWidth, kConvTaps, true);
        status != Backend::success) return status;
    constexpr core::ShortConvolution plan{kHiddenWidth, kConvTaps, 0U, 1.0F, 2.2360679774997898F};
    if (auto status = backend.convolve(plan, rows, input, lag_weights, nullptr, history, output);
        status != Backend::success) return status;
    return backend.round(output, std::size_t{rows} * kHiddenWidth);
}

#if defined(__HIPCC__) || defined(__CUDACC__)

// One HIP/CUDA implementation for all prepared elementwise stages. Backends
// supply only their dense, convolution and RMS launch adapters.
template<class Stream>
struct MiniPreparedRowsDevice {
    Stream stream;

    auto round(float* row, std::size_t elements) const {
        namespace prep = core::prepared_rows_device;
        return prep::elements(elements, prep::Bf16Round{row}, stream);
    }
    auto widen_convolution(const std::uint16_t* from, float* to, unsigned width, unsigned taps, bool channel_taps) const {
        namespace prep = core::prepared_rows_device;
        return prep::elements(std::size_t{width} * taps, prep::WidenConvolutionBf16{from, to, width, taps, channel_taps}, stream);
    }
    auto clip(float* row, const std::uint16_t* gain, unsigned rows, unsigned heads, unsigned width,
              float threshold, bool position_scale, unsigned first_position) const {
        namespace prep = core::prepared_rows_device;
        const auto scale = position_scale ? prep::LogPositionScale{first_position, 8192U, 0.1F}
                                          : prep::LogPositionScale{};
        return prep::l2_clip_bf16({row, gain, row, rows, heads, width, detail::kMiniAttentionEpsilon, threshold, scale}, stream);
    }
    auto relative_logits(const float* input, const std::uint16_t* weights, float* output, unsigned rows) const {
        namespace prep = core::prepared_rows_device;
        return prep::elements(std::size_t{rows} * kQueryHeads * kRelativeExtent,
            prep::ShortBf16Projection<kRelativeWidth>{input, weights, output, kRelativeExtent}, stream);
    }
    auto gate(const float* input, const float* logits, float* output, std::size_t elements) const {
        namespace prep = core::prepared_rows_device;
        return prep::elements(elements, prep::Bf16SigmoidProduct{input, logits, output}, stream);
    }
};

#endif

}  // namespace strix::inkling2
