#pragma once

#include "strix/draft_proposal_state.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <cstdint>
#include <vector>

namespace strix {

struct MtpDraftRuntimeBinding {
    std::uint64_t manifest_layout_identity = 0;
    ModelContentIdentity model_content_identity;
    std::vector<std::uint32_t> tensor_indices;
    std::uint64_t total_tensor_bytes = 0;

    // Stable across address/allocation relocation. This is the proposal-state
    // identity to bind to DraftProposalStateContext.
    std::uint64_t compatibility_identity = 0;

    // Includes physical storage/address/allocation generation and therefore
    // changes when the concrete runtime binding changes.
    std::uint64_t physical_binding_identity = 0;
};

[[nodiscard]] MtpDraftRuntimeBinding
build_mtp_draft_runtime_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& runtime);

void validate_mtp_draft_runtime_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& runtime,
    const MtpDraftRuntimeBinding& draft);

[[nodiscard]] DraftProposalStateContext
make_mtp_draft_proposal_state(
    const MtpDraftRuntimeBinding& draft,
    const SerialSessionContext& target_session);

void validate_mtp_draft_proposal_state(
    const MtpDraftRuntimeBinding& draft,
    const DraftProposalStateContext& state,
    const SerialSessionContext& target_session);

// Validate the actual lifecycle phase without manufacturing a synchronized copy.
// Reconciliation admits only forward progress with equal position/token deltas.
void validate_mtp_draft_proposal_phase(const MtpDraftRuntimeBinding&,
    const DraftProposalStateContext&,const SerialSessionContext&,DraftProposalStatePhase);

}  // namespace strix
