#pragma once

#include "attn_core/causal_gqa_reference.hpp"
#include "core/linear_reference.hpp"
#include "strix/attn/group_pool.hpp"

namespace strix::attn_core {

struct SelectedLatentShape {
    unsigned heads, latent, nope, rotary, value;
    [[nodiscard]] unsigned qk() const { return nope + rotary; }
};

// Head-major projection. Within each head, plain key rows precede value rows.
// Every projected channel is a row of `latent` weights.
struct SelectedLatentProjection {
    SelectedLatentShape shape;
    std::span<const float> weights;

    [[nodiscard]] std::span<const float> head(std::size_t h) const {
        const auto stride = (std::size_t{shape.nope} + shape.value) * shape.latent;
        return weights.subspan(h * stride, stride);
    }
    [[nodiscard]] std::span<const float> key(std::size_t h) const {
        return head(h).first(std::size_t{shape.nope} * shape.latent);
    }
    [[nodiscard]] std::span<const float> value(std::size_t h) const {
        return head(h).last(std::size_t{shape.value} * shape.latent);
    }
};

struct SelectedLatentRow {
    std::span<const float> latent, rotary_key, inverse_rms;
};

inline SelectedLatentProjection split_latent_projection(SelectedLatentShape s, std::span<const float> weights) {
    const auto key_count = std::size_t{s.heads} * s.nope * s.latent;
    const auto value_count = std::size_t{s.heads} * s.value * s.latent;
    if (s.heads == 0 || s.latent == 0 || s.nope == 0 || s.value == 0 || weights.size() != key_count + value_count)
        throw std::invalid_argument("latent attention: invalid KV projection");
    return {s, weights};
}

inline void latent_key_inverse_rms_reference(SelectedLatentShape s, SelectedLatentProjection kv,
    std::span<const float> latent, std::span<const float> rotary, float epsilon, std::span<float> inv) {
    if (inv.size() != s.heads || latent.size() != s.latent || rotary.size() != s.rotary)
        throw std::invalid_argument("latent attention: invalid key geometry");
    std::vector<float> head(s.qk());
    std::ranges::copy(rotary, head.begin() + s.nope);
    for (std::size_t h = 0; h < s.heads; ++h) {
        core::linear_reference(kv.key(h), latent, std::span(head).first(s.nope));
        inv[h] = attn::group_pool::rms_inv(head.data(), s.qk(), epsilon);
    }
}

// Absorb the plain-key projection into Q and the value projection into the
// softmax output. The per-head key RMS factor cannot be absorbed into Q since
// it varies with each cached row. The rotary key remains a separate operand.
template<class RowAt>
void selected_latent_attention_reference(SelectedLatentShape s, SelectedLatentProjection kv,
    std::span<const float> query, std::span<const float> sinks, std::span<const std::uint32_t> selected,
    RowAt row_at, std::span<float> output) {
    if (s.qk() == 0 || s.latent == 0 || s.heads == 0 || query.size() != std::size_t{s.heads} * s.qk() ||
        output.size() != std::size_t{s.heads} * s.value || sinks.size() != s.heads || selected.empty())
        throw std::invalid_argument("latent attention: invalid selected attention geometry");
    const float scale = 1.0F / std::sqrt(static_cast<float>(s.qk()));
    std::vector<float> q_latent(s.latent), o_latent(s.latent);
    for (std::size_t h = 0; h < s.heads; ++h) {
        const auto q = query.subspan(h * s.qk(), s.qk());
        const auto w = kv.key(h);
        for (std::size_t d = 0; d < s.latent; ++d) {
            float sum = 0.0F;
            for (std::size_t i = 0; i < s.nope; ++i) sum = std::fma(q[i], w[i * s.latent + d], sum);
            q_latent[d] = sum;
        }
        const auto value_at = [&](std::size_t i) { return row_at(selected[i]).latent; };
        const auto score_at = [&](std::size_t i) {
            const auto row = row_at(selected[i]);
            const float plain = core::dot_f32(q_latent, row.latent);
            const float rotated = core::dot_f32(q.last(s.rotary), row.rotary_key);
            return (plain + rotated) * row.inverse_rms[h] * scale;
        };
        std::ranges::fill(o_latent, 0.0F);
        detail::causal_fp32_head(o_latent, selected.size(), sinks[h], score_at, value_at);
        core::linear_reference(kv.value(h), o_latent, output.subspan(h * s.value, s.value));
    }
}

} // namespace strix::attn_core
