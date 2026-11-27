#pragma once

#include "core/bf16_reference.hpp"

#include <functional>

#include <cstdint>
#include <limits>

namespace strix::core {

// The block model supplies fixed base logits. Markov readout then walks the
// proposals once, conditioning each bias on the last selected token.
template <class Postprocess>
void draft_markov_greedy_reference(std::span<const float> base_logits, std::size_t vocab,
    std::span<const float> embedding_by_rank, std::span<const float> projection,
    std::uint32_t anchor, std::span<std::uint32_t> proposals, const Postprocess& postprocess) {
    if (vocab == 0 || vocab > std::numeric_limits<std::uint32_t>::max() || anchor >= vocab ||
        embedding_by_rank.empty() || embedding_by_rank.size() % vocab != 0 ||
        projection.size() != embedding_by_rank.size() ||
        base_logits.size() % vocab != 0 || base_logits.size() / vocab < proposals.size())
        throw std::invalid_argument("draft Markov readout: incompatible geometry");

    std::vector<float> state(embedding_by_rank.size() / vocab), row(vocab);
    auto previous = anchor;
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        for (std::size_t r = 0; r < state.size(); ++r) state[r] = embedding_by_rank[r * vocab + previous];
        linear_reference(projection, state, row);
        round_bf16_reference(row);
        std::ranges::transform(row, base_logits.subspan(i * vocab, vocab), row.begin(), std::plus<>{});
        postprocess(std::span(row));
        const auto best = std::ranges::max_element(row);
        if (!std::isfinite(*best)) throw std::invalid_argument("draft Markov readout: no finite winning logit");
        previous = static_cast<std::uint32_t>(best - row.begin());
        proposals[i] = previous;
    }
}

} // namespace strix::core
