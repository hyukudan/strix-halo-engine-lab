#include "strix/mtp_forward_contract.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("MTP forward composition geometry overflow");
    }
    return left * right;
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t build_identity(
    const MtpForwardCompositionContract& contract) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, contract.proposal_identity);
    hash_u64(hash, contract.structural_contract_identity);
    hash_u64(hash, contract.manifest_layout_identity);
    hash_u64(hash, contract.hidden_size);
    hash_u64(hash, contract.wide_hidden_size);
    hash_u64(hash, contract.attention_query_width);
    hash_u64(hash, contract.attention_kv_width);
    hash_u64(hash, contract.draft_state_width);
    hash_u64(hash, contract.output_wide_input_size);
    hash_u64(hash, contract.output_hidden_size);
    hash_u64(hash, contract.vocab_size);
    hash_u64(hash, contract.tensor_families_complete ? 1ULL : 0ULL);
    hash_u64(hash, contract.attention_geometry_matches_hidden ? 1ULL : 0ULL);
    hash_u64(hash, contract.state_matches_wide_hidden ? 1ULL : 0ULL);
    hash_u64(hash, contract.output_consumes_draft_state ? 1ULL : 0ULL);
    hash_u64(hash, contract.output_head_resolved ? 1ULL : 0ULL);
    hash_u64(hash, contract.semantic_subcontracts_resolved ? 1ULL : 0ULL);
    hash_u64(hash, contract.composition_resolved ? 1ULL : 0ULL);
    return hash;
}

}  // namespace

MtpForwardCompositionContract
build_mtp_forward_composition_contract(
    const ModelContract& model,
    const MtpDraftStructuralContract& structural,
    const MtpAttentionStructuralContract& attention,
    const MtpDraftStateContract& state,
    const MtpOutputStructuralContract& output,
    const MtpDraftTensorAudit& tensors) {
    if (const auto error = model.validate(); !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (structural.contract_identity == 0 ||
        structural.proposal_identity == 0 ||
        tensors.manifest_layout_identity == 0) {
        throw std::invalid_argument(
            "MTP forward composition requires initialized structural/tensor contracts");
    }

    MtpForwardCompositionContract contract;
    contract.proposal_identity = structural.proposal_identity;
    contract.structural_contract_identity = structural.contract_identity;
    contract.manifest_layout_identity = tensors.manifest_layout_identity;
    contract.hidden_size = model.hidden_size;
    contract.wide_hidden_size =
        checked_mul(model.hidden_size, model.hc_count);
    contract.attention_query_width =
        checked_mul(attention.attention_heads, attention.head_dim);
    contract.attention_kv_width =
        checked_mul(attention.kv_heads, attention.head_dim);
    contract.draft_state_width = state.state_width;
    contract.output_wide_input_size = output.wide_input_size;
    contract.output_hidden_size = output.mixed_hidden_size;
    contract.vocab_size = output.vocab_size;

    contract.tensor_families_complete =
        tensors.structurally_complete();
    contract.attention_geometry_matches_hidden =
        attention.attention_heads ==
            model.attention_heads &&
        attention.kv_heads ==
            model.kv_heads &&
        attention.head_dim ==
            model.head_dim &&
        attention.rotary_dim ==
            model.rotary_dim() &&
        contract.attention_query_width ==
            checked_mul(
                model.attention_heads,
                model.head_dim) &&
        contract.attention_kv_width ==
            checked_mul(
                model.kv_heads,
                model.head_dim) &&
        attention.sparse_selection_semantics_resolved;
    contract.state_matches_wide_hidden =
        state.one_state_row_per_request &&
        state.state_width == contract.wide_hidden_size;
    contract.output_consumes_draft_state =
        output.wide_input_size == state.state_width &&
        output.mixed_hidden_size == model.hidden_size &&
        output.vocab_size == model.vocab_size;
    contract.output_head_resolved =
        output.head_binding_resolved &&
        output.output_semantics_resolved;
    contract.semantic_subcontracts_resolved =
        structural.input_fusion_semantics_resolved &&
        structural.attention_semantics_resolved &&
        structural.draft_state_semantics_resolved &&
        structural.output_mixer_semantics_resolved &&
        attention.sparse_selection_semantics_resolved &&
        output.mixer_semantics_resolved;

    contract.composition_resolved =
        contract.tensor_families_complete &&
        contract.attention_geometry_matches_hidden &&
        contract.state_matches_wide_hidden &&
        contract.output_consumes_draft_state &&
        contract.output_head_resolved &&
        contract.semantic_subcontracts_resolved;

    contract.contract_identity = build_identity(contract);
    if (contract.contract_identity == 0) {
        throw std::logic_error(
            "MTP forward composition identity must be non-zero");
    }
    return contract;
}

void validate_mtp_forward_composition_contract(
    const ModelContract& model,
    const MtpDraftStructuralContract& structural,
    const MtpAttentionStructuralContract& attention,
    const MtpDraftStateContract& state,
    const MtpOutputStructuralContract& output,
    const MtpDraftTensorAudit& tensors,
    const MtpForwardCompositionContract& contract) {
    const auto expected =
        build_mtp_forward_composition_contract(
            model, structural, attention, state, output, tensors);

    if (contract.proposal_identity != expected.proposal_identity ||
        contract.structural_contract_identity != expected.structural_contract_identity ||
        contract.manifest_layout_identity != expected.manifest_layout_identity ||
        contract.hidden_size != expected.hidden_size ||
        contract.wide_hidden_size != expected.wide_hidden_size ||
        contract.attention_query_width != expected.attention_query_width ||
        contract.attention_kv_width != expected.attention_kv_width ||
        contract.draft_state_width != expected.draft_state_width ||
        contract.output_wide_input_size != expected.output_wide_input_size ||
        contract.output_hidden_size != expected.output_hidden_size ||
        contract.vocab_size != expected.vocab_size ||
        contract.tensor_families_complete != expected.tensor_families_complete ||
        contract.attention_geometry_matches_hidden != expected.attention_geometry_matches_hidden ||
        contract.state_matches_wide_hidden != expected.state_matches_wide_hidden ||
        contract.output_consumes_draft_state != expected.output_consumes_draft_state ||
        contract.output_head_resolved != expected.output_head_resolved ||
        contract.semantic_subcontracts_resolved != expected.semantic_subcontracts_resolved ||
        contract.composition_resolved != expected.composition_resolved ||
        contract.contract_identity != expected.contract_identity) {
        throw std::invalid_argument(
            "MTP forward composition contract is stale or incompatible");
    }
}

}  // namespace strix
