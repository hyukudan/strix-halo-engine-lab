#pragma once

#include "strix/ernie/attention.hpp"

#include <array>

namespace strix::ernie {

using LayerObserver = std::function<void(std::size_t position, unsigned layer, const LayerResult&)>;

// Candidate arithmetic for checkpoint exploration. The logits are the raw head
// output. Keep this identity separate from a validated checkpoint reference.
enum class CandidateActivation { swiglu, soft_clip, hard_clip };
[[nodiscard]] ReferenceActivation candidate_activation(CandidateActivation activation);

class CpuReference {
public:
    CpuReference(ReferenceShape shape, ReferenceRead read, ReferenceActivation activation,
                 ReferenceProject project = {}, ExpertObserver expert_observer = {});

    // Append a suffix using one layer at a time. Only the last row uses the
    // vocabulary head. A failed forward invalidates this sequence.
    [[nodiscard]] std::vector<float> append(std::span<const std::uint32_t> tokens,
                                           const LayerObserver& observe = {});
private:
    ReferenceShape shape_;
    ReferenceRead read_;
    ReferenceProject project_;
    FirstRowReference block_;
    std::array<AttentionHistory, kLayerCount> history_;
    std::size_t position_ = 0;
    bool failed_ = false;
};

} // namespace strix::ernie
