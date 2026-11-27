#pragma once

#include "core/bf16_reference.hpp"

#include <functional>

namespace strix::core {

// A target tap takes the mean of the complete residual streams for one token.
inline void draft_stream_mean_bf16_reference(std::span<const float> streams, std::span<float> output) {
    if (output.empty() || streams.empty() || streams.size() % output.size() != 0)
        throw std::invalid_argument("draft tap: incompatible stream shape");
    const auto count = streams.size() / output.size();
    for (std::size_t i = 0; i < output.size(); ++i) {
        float sum = 0.0F;
        for (std::size_t stream = 0; stream < count; ++stream) sum += streams[stream * output.size() + i];
        output[i] = attn::group_indexer::bf16_round(sum / static_cast<float>(count));
    }
}

inline void draft_project_norm_bf16_reference(std::span<const float> matrix, std::span<const float> input,
    std::span<const float> gain, float epsilon, std::span<float> output) {
    linear_reference(matrix, input, output);
    round_bf16_reference(output);
    rms_norm_bf16_reference(output, gain, epsilon);
}

// Normalize both sources separately, then add one projected embedding to
// each projected residual stream. The returned streams feed the draft block.
inline void mtp_join_bf16_reference(std::span<const float> previous, std::span<const float> embedding,
    std::span<const float> hidden_gain, std::span<const float> embedding_gain,
    std::span<const float> hidden_matrix, std::span<const float> embedding_matrix, float epsilon,
    std::span<float> output) {
    const auto width = embedding.size();
    if (width == 0 || previous.empty() || previous.size() % width != 0 || previous.size() != output.size())
        throw std::invalid_argument("MTP join: incompatible stream shape");

    std::vector<float> row(embedding.begin(), embedding.end()), projected_embedding(width);
    rms_norm_bf16_reference(row, embedding_gain, epsilon);
    linear_reference(embedding_matrix, row, projected_embedding);
    round_bf16_reference(projected_embedding);
    for (std::size_t offset = 0; offset < previous.size(); offset += width) {
        std::ranges::copy(previous.subspan(offset, width), row.begin());
        rms_norm_bf16_reference(row, hidden_gain, epsilon);
        auto stream = output.subspan(offset, width);
        linear_reference(hidden_matrix, row, stream);
        round_bf16_reference(stream);
        std::ranges::transform(stream, projected_embedding, stream.begin(), std::plus<>{});
        round_bf16_reference(stream);
    }
}

} // namespace strix::core
