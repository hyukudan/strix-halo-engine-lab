#pragma once

#include "attn_core/causal_gqa_reference.hpp"
#include "strix/attn/group_pool.hpp"

#include <array>

namespace strix::attn_core {

// A complete non-overlapping group, with the last projected row supplied
// separately so callers can commit carry only after attention succeeds.
template<unsigned Ratio>
void projected_ape_pool_reference(std::span<const float> values, std::span<const float> logits,
    std::span<const float> last_value, std::span<const float> last_logit, std::span<const float> ape,
    std::span<float> pooled) {
    const auto dim = pooled.size();
    if (dim == 0 || values.size() != (Ratio - 1U) * dim || logits.size() != values.size() ||
        last_value.size() != dim || last_logit.size() != dim || ape.size() != Ratio * dim)
        throw std::invalid_argument("gated APE: invalid projected group");
    const auto exp = [](float value) { return std::exp(value); };
    for (std::size_t i = 0; i < dim; ++i) {
        float channel_logits[Ratio], channel_values[Ratio];
        for (std::size_t row = 0; row < Ratio - 1U; ++row) {
            const auto at = row * dim + i;
            channel_logits[row] = logits[at] + ape[at];
            channel_values[row] = values[at];
        }
        channel_logits[Ratio - 1U] = last_logit[i] + ape[(Ratio - 1U) * dim + i];
        channel_values[Ratio - 1U] = last_value[i];
        pooled[i] = attn::group_indexer::gated_pool_channel<Ratio>(channel_logits, channel_values, exp);
    }
}

struct ReferenceHybridQueries {
    unsigned dim;
    std::span<const float> window_heads, compressed_heads, sink;
};

// A common softmax over window rows followed by complete compressed groups.
// Keys share their own domain's rotated query. The compressed value accessor
// supplies values already expressed in the window value domain, permitting a
// single inverse rotation after the common reduction.
template<class WindowAt, class PooledKeyAt, class PooledValueAt>
void hybrid_mqa_reference(const ReferenceHybridQueries& q, std::size_t window_count, std::size_t pooled_count,
    WindowAt window_at, PooledKeyAt pooled_key, PooledValueAt pooled_value, std::span<float> output) {
    if (q.dim == 0 || window_count == 0 || q.window_heads.size() != q.sink.size() * q.dim ||
        q.compressed_heads.size() != q.window_heads.size() || output.size() != q.window_heads.size())
        throw std::invalid_argument("hybrid attention: invalid head geometry");
    const float scale = 1.0F / std::sqrt(static_cast<float>(q.dim));
    std::ranges::fill(output, 0.0F);
    for (std::size_t h = 0; h < q.sink.size(); ++h) {
        const auto at = h * q.dim;
        const auto score_at = [&](std::size_t row) {
            const bool local = row < window_count;
            const auto head = (local ? q.window_heads : q.compressed_heads).subspan(at, q.dim);
            const auto key = local ? window_at(row) : pooled_key(row - window_count);
            if (key.size() != q.dim) throw std::invalid_argument("hybrid attention: invalid key width");
            float score = 0.0F;
            for (std::size_t d = 0; d < head.size(); ++d) score += head[d] * key[d];
            return score * scale;
        };
        const auto value_at = [&](std::size_t row) {
            return row < window_count ? window_at(row) : pooled_value(row - window_count);
        };
        detail::causal_fp32_head(output.subspan(at, q.dim), window_count + pooled_count,
                                 q.sink[h], score_at, value_at);
    }
}

} // namespace strix::attn_core
