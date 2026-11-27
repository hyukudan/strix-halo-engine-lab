#pragma once

#include "strix/ernie/attention.hpp"

#include <array>

namespace strix::ernie {

using LayerObserver = std::function<void(std::size_t position, unsigned layer, const LayerResult&)>;
using ReferenceRows = std::vector<std::vector<float>>;
using LayerBoundaryObserver = std::function<void(unsigned layer, const ReferenceRows&)>;

// Candidate arithmetic for checkpoint exploration. The logits are the raw head
// output. Keep this identity separate from a validated checkpoint reference.
enum class CandidateActivation { swiglu, soft_clip, hard_clip };
[[nodiscard]] ReferenceActivation candidate_activation(CandidateActivation activation);

class CpuReference {
public:
    CpuReference(ReferenceShape shape, ReferenceRead read, ReferenceActivation activation,
                 ReferenceProject project = {}, ExpertObserver expert_observer = {},
                 AttentionHeadExecutor execute_heads = {}, std::size_t max_rows = data::dims::max_context);

    // Append a suffix using one layer at a time. Only the last row uses the
    // vocabulary head. A failed forward invalidates this sequence.
    [[nodiscard]] std::vector<float> append(std::span<const std::uint32_t> tokens,
                                           const LayerObserver& observe = {});
    // Diagnostic over an independent position-zero sequence. Boundaries are
    // indexed by the next layer to execute, including kLayerCount at the head.
    [[nodiscard]] ReferenceRows embed_rows(std::span<const std::uint32_t> tokens) const;
    [[nodiscard]] ReferenceRows prefill(ReferenceRows rows, unsigned first_layer, unsigned end_layer,
                                        const LayerBoundaryObserver& boundary = {},
                                        const LayerObserver& observe = {}) const;
    [[nodiscard]] std::vector<float> head_input(std::span<const float> row) const;
private:
    void run_layer(ReferenceRows& rows, unsigned layer, AttentionHistory& history,
                   std::size_t position, const LayerObserver& observe, bool index_prefill) const;
    ReferenceShape shape_;
    ReferenceRead read_;
    ReferenceProject project_;
    AttentionHeadExecutor execute_heads_;
    FirstRowReference block_;
    std::array<AttentionHistory, kLayerCount> history_;
    std::size_t max_rows_;
    std::size_t position_ = 0;
    bool failed_ = false;
};

} // namespace strix::ernie
