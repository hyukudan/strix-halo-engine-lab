#pragma once

#include "core/dense_row_plan.hpp"
#include "core/resident_row_plan.hpp"
#include "core/short_convolution.hpp"
#include "strix/attn/group_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/row_transforms.hpp"
#endif

namespace strix::attn_core {

enum class PreparedGqaProjection {
    query, key, value, key_taps, value_taps, relative, relative_bias, sink, output_gate, output,
};

struct PreparedGqaHostWeights {
    std::span<const float> query_gain;      // [query_head][head_width]
    std::span<const float> relative_gain;   // [query_head][relative_width]
    std::span<const float> key_lag_weights;   // [lag][KV channel], current first
    std::span<const float> value_lag_weights; // [lag][KV channel], current first
};

template <class Shape>
struct PreparedGqaRow {
    std::array<float, Shape::kQueryWidth> query;
    std::array<float, Shape::kKvWidth> key;
    std::array<float, Shape::kKvWidth> value;
    std::array<float, Shape::kQueryHeads * Shape::kRelativeExtent> relative_logits;
    std::array<float, Shape::kQueryHeads> sinks;
    std::array<float, Shape::kQueryWidth> output_gate;
};

// Convert channel-major taps, earliest first, to lag-major taps, current first.
template <class T>
[[nodiscard]] std::vector<T> short_conv_lag_weights(
    std::span<const T> channel_taps, std::size_t channels, std::size_t taps) {
    if (channels == 0 || taps == 0 || channel_taps.size() % taps != 0 ||
        channel_taps.size() / taps != channels)
        throw std::invalid_argument("invalid static convolution weights");
    std::vector<T> weights(channel_taps.size());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t lag = 0; lag < taps; ++lag) {
            weights[lag * channels + channel] = channel_taps[channel * taps + taps - 1U - lag];
        }
    }
    return weights;
}

namespace detail {

inline void round_prepared_gqa_row(std::span<float> row) {
    std::ranges::transform(row, row.begin(), attn::group_indexer::bf16_round);
}

inline void clip_prepared_gqa_head(std::span<float> head, std::span<const float> gain, float threshold,
                                   float epsilon) {
    float squared_norm = 0.0F;
    for (std::size_t i = 0; i < head.size(); ++i) {
        const float scaled = gain.empty() ? head[i] : head[i] * gain[i];
        head[i] = scaled;
        squared_norm = std::fma(scaled, scaled, squared_norm);
    }
    const float scale = std::min(1.0F, threshold / (std::sqrt(squared_norm) + epsilon));
    for (float& value : head) value = attn::group_indexer::bf16_round(value * scale);
}

inline void scale_prepared_gqa_heads(std::span<float> rows, std::span<const float> gains, unsigned width,
                                       float threshold, float position_scale, float epsilon) {
    for (std::size_t base = 0; base < rows.size(); base += width) {
        const auto head = rows.subspan(base, width);
        clip_prepared_gqa_head(head, gains.subspan(base, width), threshold, epsilon);
        if (position_scale != 1.0F) {
            for (float& value : head) value = attn::group_indexer::bf16_round(value * position_scale);
        }
    }
}

template <class Shape, class Project>
void prepare_gqa_kv(PreparedGqaProjection projection, PreparedGqaProjection taps, bool full,
                     std::span<const float> input, std::span<const float> lag_weights,
                     std::span<float> history, std::span<float> out, Project& project) {
    std::array<float, Shape::kKvWidth> raw;
    project(projection, input, raw);
    round_prepared_gqa_row(raw);

    std::array<float, Shape::kKvHeads * Shape::kConvTaps> dynamic_weights;
    std::span<const float> dynamic;
    if (full) {
        project(taps, input, dynamic_weights);
        round_prepared_gqa_row(dynamic_weights);
        dynamic = dynamic_weights;
    }
    const core::ShortConvolution plan{
        Shape::kKvWidth, Shape::kConvTaps, full ? Shape::kKvHeads : 0U, 1.0F,
        full ? Shape::kFullConvDivisor : Shape::kLocalConvDivisor,
    };
    core::causal_short_convolution(plan, raw, lag_weights, dynamic, history, out);
    round_prepared_gqa_row(out);
}

} // namespace detail

// Inputs are the normalized, BF16-rounded Q/K/V streams from the layer adapter.
// Each visit and sequence owns its own K and V histories of (taps - 1) rows.
// A projection supplies FP32 accumulations; this bridge rounds each layer-local
// activation at the published BF16 boundary and keeps gate sigmoid FP32.
template <class Shape, class Project>
[[nodiscard]] PreparedGqaRow<Shape> prepare_gqa_host(
    bool full, unsigned position, std::span<const float> q_input, std::span<const float> k_input,
    std::span<const float> v_input, const PreparedGqaHostWeights& weights, std::span<float> key_history,
    std::span<float> value_history, Project& project) {
    if (position >= Shape::kMaxContext || q_input.size() != Shape::kHiddenWidth ||
        k_input.size() != Shape::kHiddenWidth || v_input.size() != Shape::kHiddenWidth || weights.query_gain.size() != Shape::kQueryWidth ||
        weights.relative_gain.size() != Shape::kQueryHeads * Shape::kRelativeWidth ||
        weights.key_lag_weights.size() != Shape::kConvTaps * Shape::kKvWidth ||
        weights.value_lag_weights.size() != Shape::kConvTaps * Shape::kKvWidth ||
        key_history.size() != (Shape::kConvTaps - 1U) * Shape::kKvWidth || value_history.size() != key_history.size()) {
        throw std::invalid_argument("invalid prepared GQA attention projection shape");
    }
    using Projection = PreparedGqaProjection;
    PreparedGqaRow<Shape> row;
    const float position_scale = full && position >= Shape::kPositionFloor
        ? 1.0F + Shape::kPositionAlpha * std::log(static_cast<float>(position + 1U) /
                                                  static_cast<float>(Shape::kPositionFloor))
        : 1.0F;

    project(Projection::query, q_input, row.query);
    detail::round_prepared_gqa_row(row.query);
    detail::scale_prepared_gqa_heads(row.query, weights.query_gain, Shape::kHeadWidth, Shape::kQkClip,
                                     position_scale, Shape::kAttentionEpsilon);

    detail::prepare_gqa_kv<Shape>(Projection::key, Projection::key_taps, full, k_input, weights.key_lag_weights,
                            key_history, row.key, project);
    detail::prepare_gqa_kv<Shape>(Projection::value, Projection::value_taps, full, v_input, weights.value_lag_weights,
                            value_history, row.value, project);
    for (std::size_t head = 0; head < Shape::kKvHeads; ++head) {
        const auto key_head = std::span<float>(row.key).subspan(head * Shape::kHeadWidth, Shape::kHeadWidth);
        detail::clip_prepared_gqa_head(key_head, {}, Shape::kQkClip, Shape::kAttentionEpsilon);
        const auto value_head = std::span<float>(row.value).subspan(head * Shape::kHeadWidth, Shape::kHeadWidth);
        const float scale = attn::group_pool::rms_inv(value_head.data(), Shape::kHeadWidth, Shape::kAttentionEpsilon);
        for (float& value : value_head) value = attn::group_indexer::bf16_round(value * scale);
    }

    std::array<float, Shape::kQueryHeads * Shape::kRelativeWidth> relative;
    project(Projection::relative, q_input, relative);
    detail::round_prepared_gqa_row(relative);
    detail::scale_prepared_gqa_heads(relative, weights.relative_gain, Shape::kRelativeWidth,
                                       Shape::kRelativeClip, position_scale, Shape::kAttentionEpsilon);
    for (std::size_t head = 0; head < Shape::kQueryHeads; ++head) {
        const auto input = std::span<const float>(relative).subspan(head * Shape::kRelativeWidth, Shape::kRelativeWidth);
        const auto logits = std::span<float>(row.relative_logits).subspan(head * Shape::kRelativeExtent, Shape::kRelativeExtent);
        project(Projection::relative_bias, input, logits);
        detail::round_prepared_gqa_row(logits);
    }

    project(Projection::sink, q_input, row.sinks);
    detail::round_prepared_gqa_row(row.sinks);
    project(Projection::output_gate, q_input, row.output_gate);
    detail::round_prepared_gqa_row(row.output_gate);
    for (float& value : row.output_gate) value = 1.0F / (1.0F + std::exp(-value));
    return row;
}

// Apply the per-channel gate before the output projection. The attended row is
// stored at BF16, as is the gated input to O and the projected hidden row.
template <class Shape, class Project>
void project_gqa_host_output(std::span<const float> attended, std::span<const float> gate,
                                   Project& project, std::span<float> output) {
    if (attended.size() != Shape::kQueryWidth || gate.size() != Shape::kQueryWidth || output.size() != Shape::kHiddenWidth) {
        throw std::invalid_argument("invalid prepared GQA attention output shape");
    }
    std::array<float, Shape::kQueryWidth> gated;
    for (std::size_t i = 0; i < gated.size(); ++i) {
        gated[i] = attn::group_indexer::bf16_round(attn::group_indexer::bf16_round(attended[i]) * gate[i]);
    }
    project(PreparedGqaProjection::output, gated, output);
    detail::round_prepared_gqa_row(output);
}

struct PreparedGqaInputs {
    const float* query;
    const float* key;
    const float* value;
};

template <class Matrix>
struct PreparedGqaWeightsWith {
    using Role = PreparedGqaProjection;
    static constexpr unsigned role_count = static_cast<unsigned>(Role::output) + 1U;
    std::array<Matrix, role_count> matrix{};
    const std::uint16_t* query_gain{};
    const std::uint16_t* relative_gain{};
    const float* key_convolution{};
    const float* value_convolution{};
    std::size_t output_row_bytes{};  // Zero uses the natural layout.

    const Matrix& projection(Role role) const { return matrix[static_cast<unsigned>(role)]; }
};

using PreparedGqaWeights = PreparedGqaWeightsWith<const std::uint16_t*>;
using ResidentPreparedGqaWeights = PreparedGqaWeightsWith<core::ResidentRowWeight>;

struct PreparedGqaWorkspace {
    float* raw_keys;                      // [row][KvWidth]
    float* raw_values;                    // [row][KvWidth]
    float* dynamic_keys;                  // [row][KvHeads][ConvTaps], full visits only
    float* dynamic_values;                // [row][KvHeads][ConvTaps], full visits only
    float* relative_features;             // [row][QueryHeads][RelativeWidth]
    float* key_history;                   // [ConvTaps - 1][KvWidth], per visit
    float* value_history;                 // [ConvTaps - 1][KvWidth], per visit
};

// Row intervals follow packed input order. Each interval has its own stream
// position and convolution history. Dense projections cover the packed rows.
struct PreparedGqaGroup {
    unsigned first_position, rows;
    float* key_history;
    float* value_history;
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

inline bool projection_aligned(const std::uint16_t* matrix) { return prepared_gqa_aligned(matrix); }

inline bool projection_aligned(const core::ResidentRowWeight& matrix) {
    if (const auto* bits = std::get_if<const std::uint16_t*>(&matrix)) return projection_aligned(*bits);
    return prepared_gqa_aligned(std::get<weight::ResidentMatrixView>(matrix).bytes.data());
}

inline const std::uint16_t* bf16_projection(const std::uint16_t* matrix) { return matrix; }

inline const std::uint16_t* bf16_projection(const core::ResidentRowWeight& matrix) {
    const auto* bits = std::get_if<const std::uint16_t*>(&matrix);
    return bits ? *bits : nullptr;
}

inline PreparedGqaLinear bind_projection(const std::uint16_t* matrix, unsigned outputs, float* output) {
    return {matrix, outputs, output};
}

inline core::ResidentRowProjection bind_projection(const core::ResidentRowWeight& matrix,
                                                    unsigned outputs, float* output) {
    return {matrix, outputs, output};
}

// Opt-in (static constexpr bool kFoldRounds = true, runtime fold_rounds() = its off switch): the backend folds the stage's plain BF16 rounds into their
// producers or consumers with the same bits. Its convolve<RoundOutput, RoundInput> rounds the input rows, the dynamic
// taps and the stored output with round()'s bf16 round, and store_round(rows) is true only when linear() rounds each
// completed FP32 sum in the store (Bf16RowProjection::round_bf16) exactly as round() would round the stored sum.
// Failure-checking rounds are never folded. Each fold removes one elementwise launch.
template <class Backend>
concept PreparedGqaFoldsRounds = requires(const Backend& backend) {
    requires Backend::kFoldRounds;
    { backend.fold_rounds() } -> std::convertible_to<bool>;
};

template <class Backend>
bool folds_rounds(const Backend& backend) {
    if constexpr (PreparedGqaFoldsRounds<Backend>) return backend.fold_rounds();
    return false;
}

// round_output (folding backends only): the convolution stores its output rounded, replacing a later plain round.
template <class Shape, class Backend>
typename Backend::Status prepared_gqa_kv_convolution(
    const Backend& backend, bool full, unsigned rows, const float* convolution, float* activation,
    float* dynamic, float* history, float* output, bool round_output = false) {
    if constexpr (PreparedGqaFoldsRounds<Backend>) {
        if (folds_rounds(backend)) {
            const core::ShortConvolution plan{Shape::kKvWidth, Shape::kConvTaps, full ? Shape::kKvHeads : 0U, 1.0F,
                                              full ? Shape::kFullConvDivisor : Shape::kLocalConvDivisor};
            const float* taps = full ? dynamic : nullptr;
            return round_output
                ? backend.template convolve<true, true>(plan, rows, activation, convolution, taps, history, output)
                : backend.template convolve<false, true>(plan, rows, activation, convolution, taps, history, output);
        }
    }
    if (round_output) return Backend::invalid;
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
template <class Shape, class Backend, class Matrix>
typename Backend::Status enqueue_prepared_gqa(
    const Backend& backend, bool full, unsigned first_position, unsigned rows,
    const PreparedGqaInputs& inputs, const PreparedGqaWeightsWith<Matrix>& weights,
    const PreparedGqaWorkspace& work, const PreparedGqaRows& output, unsigned* failure = nullptr,
    const std::uint32_t* device_position = nullptr, std::span<const PreparedGqaGroup> groups = {}) {
    using Role = PreparedGqaProjection;
    const PreparedGqaGroup single{first_position, rows, work.key_history, work.value_history};
    if (groups.empty()) groups = {&single, 1U};
    if (rows == 0 || rows > 65535U || !weights.query_gain || !weights.relative_gain ||
        !weights.key_convolution || !weights.value_convolution) return Backend::invalid;
    unsigned remaining = rows;
    for (const auto& group : groups) {
        if (!group.rows || group.rows > remaining || group.first_position >= Shape::kMaxContext ||
            group.rows > Shape::kMaxContext - group.first_position ||
            !prepared_gqa_aligned(group.key_history) || !prepared_gqa_aligned(group.value_history))
            return Backend::invalid;
        remaining -= group.rows;
    }
    if (remaining != 0U) return Backend::invalid;
    const std::array<const float*, 12> required{inputs.query, inputs.key, inputs.value, output.query,
        output.key, output.value, output.relative_logits, output.sinks, output.gate_logits, work.raw_keys, work.raw_values,
        work.relative_features};
    if (!std::ranges::all_of(required, prepared_gqa_aligned) ||
        (full && (!prepared_gqa_aligned(work.dynamic_keys) || !prepared_gqa_aligned(work.dynamic_values))))
        return Backend::invalid;
    for (unsigned i = 0; i < weights.matrix.size(); ++i) {
        const auto role = static_cast<Role>(i);
        if (role == Role::output || (!full && (role == Role::key_taps || role == Role::value_taps))) continue;
        if (!detail::projection_aligned(weights.projection(role))) return Backend::invalid;
    }
    const auto* relative_bias = detail::bf16_projection(weights.projection(Role::relative_bias));
    if (!relative_bias) return Backend::invalid;

    const auto project = [&](Role role, float* to) {
        return detail::bind_projection(weights.projection(role), Shape::projection_width(role), to);
    };
    const auto linear = [&](const float* from, const auto& projections) {
        return backend.linear(from, Shape::kHiddenWidth, rows, std::span(projections));
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

    // Publish every projection before the first KV consumer in any request group.
    if constexpr (requires { backend.finish_projections(); }) {
        if (auto status = backend.finish_projections(); status != Backend::success) return status;
    }

    // A device position (graph replay) applies to the one single-lane group only.
    const auto clip_query = [&](float* values, const std::uint16_t* gain, unsigned group_rows, unsigned width,
                                float threshold, unsigned group_first) {
        if constexpr (requires { backend.clip(values, gain, group_rows, Shape::kQueryHeads, width, threshold,
                                               full, group_first, device_position); }) {
            return backend.clip(values, gain, group_rows, Shape::kQueryHeads, width, threshold, full, group_first,
                                full && groups.size() == 1U ? device_position : nullptr);
        } else {
            return backend.clip(values, gain, group_rows, Shape::kQueryHeads, width, threshold, full, group_first);
        }
    };

    std::size_t base = 0;
    for (const auto& group : groups) {
        const auto kv = base * Shape::kKvWidth;
        const auto taps = base * Shape::kKvHeads * Shape::kConvTaps;
        if (auto status = detail::prepared_gqa_kv_convolution<Shape>(backend, full, group.rows, weights.key_convolution,
                work.raw_keys + kv, full ? work.dynamic_keys + taps : nullptr, group.key_history, output.key + kv);
            status != Backend::success) return status;
        if (auto status = detail::prepared_gqa_kv_convolution<Shape>(backend, full, group.rows, weights.value_convolution,
                work.raw_values + kv, full ? work.dynamic_values + taps : nullptr, group.value_history, output.value + kv,
                detail::folds_rounds(backend));
            status != Backend::success) return status;

        if (auto status = clip_query(output.query + base * Shape::kQueryWidth, weights.query_gain,
                group.rows, Shape::kHeadWidth, Shape::kQkClip, group.first_position);
            status != Backend::success) return status;
        base += group.rows;
    }
    if (auto status = backend.clip(output.key, nullptr, rows, Shape::kKvHeads, Shape::kHeadWidth,
            Shape::kQkClip, false, 0); status != Backend::success) return status;
    if (!detail::folds_rounds(backend)) {  // folded: each value convolution stored it rounded
        if (auto status = backend.round(output.value, std::size_t{rows} * Shape::kKvWidth); status != Backend::success)
            return status;
    }
    if (auto status = backend.rms_values(output.value, rows); status != Backend::success) return status;
    base = 0;
    for (const auto& group : groups) {
        if (auto status = clip_query(work.relative_features + base * Shape::kQueryHeads * Shape::kRelativeWidth,
                weights.relative_gain, group.rows, Shape::kRelativeWidth,
                Shape::kRelativeClip, group.first_position);
            status != Backend::success) return status;
        base += group.rows;
    }
    // A backend whose attention projects the relative bias from the clipped features (dense::DenseProjectedBias)
    // skips the [row][head][distance] image; its flash computes the same logits and flags them as this stage does.
    bool image = true;
    if constexpr (requires { { backend.relative_bias_in_attention(rows) } -> std::convertible_to<bool>; })
        image = !backend.relative_bias_in_attention(rows);
    if (image) {
        if (auto status = backend.relative_logits(work.relative_features, relative_bias,
                output.relative_logits, rows, failure); status != Backend::success) return status;
    }
    return backend.round(output.sinks, std::size_t{rows} * Shape::kQueryHeads, failure);
}

template <class Shape, class Backend, class Matrix>
typename Backend::Status enqueue_prepared_gqa_output(
    const Backend& backend, unsigned rows, const Matrix& output_weight, const float* attended,
    const float* gate_logits, float* gated, float* output, std::size_t output_row_bytes = 0U) {
    if (rows == 0 || rows > Shape::kMaxContext || !attended || !gate_logits || !prepared_gqa_aligned(gated) ||
        !detail::projection_aligned(output_weight) || !prepared_gqa_aligned(output) || gated == output)
        return Backend::invalid;
    auto linear = detail::bind_projection(output_weight, Shape::kHiddenWidth, output);
    if constexpr (requires { linear.row_bytes; }) linear.row_bytes = output_row_bytes;
    else if (output_row_bytes != 0U) return Backend::invalid;
    bool rounded = false;  // folded: the projection stores each completed sum rounded
    if constexpr (detail::PreparedGqaFoldsRounds<Backend> && requires { linear.round_bf16; }) {
        rounded = detail::folds_rounds(backend) && backend.store_round(rows);
        linear.round_bf16 = rounded;
    }
    if (auto status = backend.gate(attended, gate_logits, gated, std::size_t{rows} * Shape::kQueryWidth);
        status != Backend::success) return status;
    const std::array projection{linear};
    if (auto status = backend.linear(gated, Shape::kQueryWidth, rows, projection); status != Backend::success) return status;
    if (rounded) return Backend::success;
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
template <class Shape, class Backend, class Matrix, class Append>
typename Backend::Status enqueue_prepared_gqa_forward(
    const Backend& backend, bool full, unsigned first_position, unsigned rows, const PreparedGqaInputs& inputs,
    const PreparedGqaWeightsWith<Matrix>& weights, const PreparedGqaWorkspace& work, const PreparedGqaRows& prepared,
    const PreparedGqaOutputRows& output, unsigned* operand_failure, Append&& append,
    const std::uint32_t* device_position = nullptr) {
    const auto& output_weight = weights.projection(PreparedGqaProjection::output);
    if (!detail::projection_aligned(output_weight) || !prepared_gqa_aligned(output.gated) ||
        !prepared_gqa_aligned(output.output) || output.gated == output.output) return Backend::invalid;
    if (auto status = enqueue_prepared_gqa<Shape>(backend, full, first_position, rows, inputs, weights, work,
                                                  prepared, operand_failure, device_position); status != Backend::success) return status;
    if (auto status = append(); status != Backend::success) return status;
    return enqueue_prepared_gqa_output<Shape>(
        backend, rows, output_weight, output.attended, prepared.gate_logits, output.gated, output.output,
        weights.output_row_bytes);
}

#if defined(__HIPCC__) || defined(__CUDACC__)

// Bind static convolution weights once before any row uses the tables.
inline auto enqueue_bind_convolution_weights(std::span<const std::uint16_t> source, float* lag_weights,
                                             unsigned width, unsigned taps, bool channel_taps,
                                             core::row_transforms::Stream stream) {
    namespace prep = core::row_transforms;
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

    template <bool RoundOutput = false, bool RoundInput = false>
    auto convolve(const core::ShortConvolution& plan, unsigned rows, const float* input, const float* lag_weights,
                  const float* dynamic, float* history, float* output) const {
        return core::causal_conv::launch_with_store<RoundOutput, RoundInput>(
            plan, {rows, input, lag_weights, dynamic, history, output}, stream);
    }

    auto round(float* row, std::size_t elements, unsigned* failure = nullptr) const {
        namespace prep = core::row_transforms;
        if (failure) return prep::elements(elements, prep::CheckedBf16Round{row, failure}, stream);
        return prep::elements(elements, prep::Bf16Round{row}, stream);
    }
    auto clip(float* row, const std::uint16_t* gain, unsigned rows, unsigned heads, unsigned width,
              float threshold, bool position_scale, unsigned first_position,
              const std::uint32_t* device_position = nullptr) const {
        namespace prep = core::row_transforms;
        const auto scale = position_scale
            ? prep::LogPositionScale{first_position, Shape::kPositionFloor, Shape::kPositionAlpha}
            : prep::LogPositionScale{};
        return prep::l2_clip_bf16({row, gain, row, rows, heads, width, Shape::kAttentionEpsilon, threshold, scale},
                                 stream, device_position);
    }
    template <bool ReverseOrder = false>
    auto relative_logits(const float* input, const std::uint16_t* weights, float* output, unsigned rows,
                         unsigned* failure) const {
        namespace prep = core::row_transforms;
        const auto lines = std::size_t{rows} * Shape::kQueryHeads;
        const auto size = lines * Shape::kRelativeExtent;
        // Four adjacent outputs per element where the layout admits it: the same per-output chain (same bits).
        using Wide = prep::ShortBf16Projection<Shape::kRelativeWidth, true, 4U, ReverseOrder>;
        if (Wide::fits(input, weights, output, Shape::kRelativeExtent)) {
            if (failure) return prep::elements(Wide::launch_size(lines, Shape::kRelativeExtent),
                Wide{input, weights, output, Shape::kRelativeExtent, failure}, stream);
            return prep::elements(Wide::launch_size(lines, Shape::kRelativeExtent),
                prep::ShortBf16Projection<Shape::kRelativeWidth, false, 4U, ReverseOrder>{input, weights, output, Shape::kRelativeExtent},
                stream);
        }
        if (failure) return prep::elements(size,
            prep::ShortBf16Projection<Shape::kRelativeWidth, true, 1U, ReverseOrder>{input, weights, output, Shape::kRelativeExtent, failure},
            stream);
        return prep::elements(size,
            prep::ShortBf16Projection<Shape::kRelativeWidth, false, 1U, ReverseOrder>{input, weights, output, Shape::kRelativeExtent}, stream);
    }
    auto gate(const float* input, const float* logits, float* output, std::size_t elements) const {
        namespace prep = core::row_transforms;
        return prep::elements(elements, prep::Bf16SigmoidProduct{input, logits, output}, stream);
    }
};

#endif

}  // namespace strix::attn_core
