#include "strix/spec/draft_reference.hpp"
#include "strix/spec/row_rule.hpp"
#include "core/bf16_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::spec {

void draft_stream_mean_bf16_reference(std::span<const float> streams, std::span<float> output) {
    if (output.empty() || streams.empty() || streams.size() % output.size() != 0)
        throw std::invalid_argument("draft tap: incompatible stream shape");
    const auto count = streams.size() / output.size();
    for (std::size_t i = 0; i < output.size(); ++i) {
        float sum = 0.0F;
        for (std::size_t stream = 0; stream < count; ++stream) sum += streams[stream * output.size() + i];
        output[i] = attn::group_indexer::bf16_round(sum / static_cast<float>(count));
    }
}

void draft_project_norm_bf16_reference(std::span<const float> matrix, std::span<const float> input,
    std::span<const float> gain, float epsilon, std::span<float> output) {
    core::linear_reference(matrix, input, output);
    core::round_bf16_reference(output);
    core::rms_norm_bf16_reference(output, gain, epsilon);
}

void mtp_join_bf16_reference(std::span<const float> previous, std::span<const float> embedding,
    std::span<const float> hidden_gain, std::span<const float> embedding_gain,
    std::span<const float> hidden_matrix, std::span<const float> embedding_matrix, float epsilon,
    std::span<float> output) {
    const auto width = embedding.size();
    if (width == 0 || previous.empty() || previous.size() % width != 0 || previous.size() != output.size())
        throw std::invalid_argument("MTP join: incompatible stream shape");

    std::vector<float> row(embedding.begin(), embedding.end()), projected_embedding(width);
    core::rms_norm_bf16_reference(row, embedding_gain, epsilon);
    core::linear_reference(embedding_matrix, row, projected_embedding);
    core::round_bf16_reference(projected_embedding);
    for (std::size_t offset = 0; offset < previous.size(); offset += width) {
        std::ranges::copy(previous.subspan(offset, width), row.begin());
        core::rms_norm_bf16_reference(row, hidden_gain, epsilon);
        auto stream = output.subspan(offset, width);
        core::linear_reference(hidden_matrix, row, stream);
        core::round_bf16_reference(stream);
        std::ranges::transform(stream, projected_embedding, stream.begin(), std::plus<>{});
        core::round_bf16_reference(stream);
    }
}

void draft_markov_greedy_reference(std::span<const float> base_logits, std::size_t vocab,
    std::span<const float> embedding_by_rank, std::span<const float> projection,
    std::uint32_t anchor, std::span<std::uint32_t> proposals, const DraftScoreTransform& transform) {
    if (vocab == 0 || vocab > std::numeric_limits<std::uint32_t>::max() || anchor >= vocab ||
        embedding_by_rank.empty() || embedding_by_rank.size() % vocab != 0 ||
        projection.size() != embedding_by_rank.size() ||
        base_logits.size() % vocab != 0 || base_logits.size() / vocab < proposals.size())
        throw std::invalid_argument("draft Markov readout: incompatible geometry");

    constexpr float mask = -std::numeric_limits<float>::infinity();
    const RowRule greedy;
    std::vector<float> embedding(embedding_by_rank.size() / vocab), row(vocab);
    auto previous = anchor;
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        for (std::size_t r = 0; r < embedding.size(); ++r) embedding[r] = embedding_by_rank[r * vocab + previous];
        core::linear_reference(projection, embedding, row);
        core::round_bf16_reference(row);
        const auto base = base_logits.subspan(i * vocab, vocab);
        std::ranges::transform(row, base, row.begin(), std::plus<>{});
        if (transform) transform(row);
        for (std::size_t id = 0; id < vocab; ++id)
            if (base[id] == mask || std::isnan(base[id])) row[id] = base[id];
        previous = greedy.row_token(row, i);
        if (!std::isfinite(row[previous])) throw std::invalid_argument("draft Markov readout: no finite winning logit");
        proposals[i] = previous;
    }
}

} // namespace strix::spec
