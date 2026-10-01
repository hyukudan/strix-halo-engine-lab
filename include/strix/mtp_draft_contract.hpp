#pragma once

#include "strix/model_contract.hpp"
#include "strix/mtp_draft_binding.hpp"

#include <cstdint>

namespace strix {

struct MtpDraftStructuralContract {
    std::uint64_t proposal_identity = 0;
    std::uint64_t model_content_bound_identity = 0;

    std::uint64_t vocab_size = 0;
    std::uint64_t hidden_size = 0;
    std::uint64_t num_hidden_layers = 0;
    std::uint64_t full_attention_layers = 0;
    std::uint64_t attention_heads = 0;
    std::uint64_t kv_heads = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t rotary_dim = 0;
    float rope_theta = 0.0F;

    bool uses_dedicated_embeddings = false;
    bool hidden_state_source_layer_explicit = false;

    // Planning-only cache estimate from the declared full-attention geometry.
    // This is not a claim about the unresolved predictor forward semantics.
    std::uint64_t provisional_kv_bytes_per_token = 0;

    // Input fusion has an independent host-side oracle. The complete forward
    // remains blocked until attention/QSA, block-state propagation and output
    // semantics are independently fixed and tested.
    bool input_fusion_semantics_resolved = true;
    bool attention_semantics_resolved = false;
    bool draft_state_semantics_resolved = false;
    bool output_semantics_resolved = false;
    bool forward_semantics_resolved = false;

    std::uint64_t contract_identity = 0;
};

[[nodiscard]] MtpDraftStructuralContract
build_mtp_draft_structural_contract(
    const ModelContract& model,
    const MtpDraftRuntimeBinding& binding);

void validate_mtp_draft_structural_contract(
    const ModelContract& model,
    const MtpDraftRuntimeBinding& binding,
    const MtpDraftStructuralContract& contract);

void require_mtp_draft_execution_semantics(
    const MtpDraftStructuralContract& contract);

}  // namespace strix
