#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace strix::spec {

// One target tap: [stream][hidden] -> [hidden], with a BF16 store.
void draft_stream_mean_bf16_reference(std::span<const float> streams, std::span<float> output);

// The projected target taps and each drafter KV projection use this form.
void draft_project_norm_bf16_reference(std::span<const float> matrix, std::span<const float> input,
    std::span<const float> gain, float epsilon, std::span<float> output);

// The MTP input combines independently normalized and projected hidden
// streams and embedding. The corresponding chain scheduler is MtpChain.
void mtp_join_bf16_reference(std::span<const float> previous, std::span<const float> embedding,
    std::span<const float> hidden_gain, std::span<const float> embedding_gain,
    std::span<const float> hidden_matrix, std::span<const float> embedding_matrix, float epsilon,
    std::span<float> output);

using DraftScoreTransform = std::function<void(std::span<float>)>;

// The block forward supplies fixed base logits for the proposal rows. A
// rank-major embedding and vocabulary-major projection give the correction
// for each row, using the anchor and then the ID selected on the prior row.
// Apply the optional transform once per proposal, after bias and before the
// shared greedy row rule. The base row's masked IDs remain excluded.
void draft_markov_greedy_reference(std::span<const float> base_logits, std::size_t vocab,
    std::span<const float> embedding_by_rank, std::span<const float> projection,
    std::uint32_t anchor, std::span<std::uint32_t> proposals, const DraftScoreTransform& transform = {});

} // namespace strix::spec
