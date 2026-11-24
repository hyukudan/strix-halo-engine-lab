#pragma once

#include "strix/spec/block_parallel.hpp"
#include "strix/spec/block_draft_reference.hpp"

namespace strix::spec {
// The model contributes feature capture and shared target embedding/head hooks.
// Policy, reference backbone, candidate selection and cycle remain shared.
BlockPolicy candidate_block_policy(const weight::BlockDraftConfig&);
struct DraftFeatureWindow {
    reference::Matrix rows;
    std::uint64_t first_position{};
};
struct CandidateTargetHooks {
    // Live width of the target output head: the number of real token ids. Distinct from the padded
    // storage count in the checkpoint config. raw_logits must return exactly this many columns.
    std::uint32_t live_vocab{};
    std::function<DraftFeatureWindow()> features;
    // Rows exactly as the target's first layer receives them (a normalizing target returns normalized rows).
    std::function<reference::Matrix(std::span<const std::uint32_t>)> embed;
    std::function<reference::Matrix(const reference::Matrix&)> raw_logits;
    std::function<void(std::span<const std::uint32_t>)> accepted;
};
// Greedy host oracle adapter. Uses the EXISTING ParallelBlockDrafter ->
// BlockDrafter -> BlockSpeculator controller. No second acceptance/rollback loop.
class CandidateBlockForward final : public BlockForward {
public:
    CandidateBlockForward(weight::BlockDraftConfig config, reference::Weights weights, CandidateTargetHooks hooks);
    void forward(const BlockRequest&, std::vector<std::uint32_t>&, std::vector<float>&) override;
    void accept(std::span<const std::uint32_t>) override;
private:
    weight::BlockDraftConfig config_;
    reference::Weights weights_;
    CandidateTargetHooks hooks_;
};
} // namespace strix::spec
