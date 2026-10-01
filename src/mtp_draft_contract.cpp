#include "strix/mtp_draft_contract.hpp"

#include <bit>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "MTP structural byte calculation overflow");
    }
    return left * right;
}

std::uint64_t build_identity(
    const MtpDraftStructuralContract& contract) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, contract.proposal_identity);
    hash_u64(
        hash,
        contract.model_content_bound_identity);
    hash_u64(hash, contract.vocab_size);
    hash_u64(hash, contract.hidden_size);
    hash_u64(hash, contract.num_hidden_layers);
    hash_u64(hash, contract.full_attention_layers);
    hash_u64(hash, contract.attention_heads);
    hash_u64(hash, contract.kv_heads);
    hash_u64(hash, contract.head_dim);
    hash_u64(hash, contract.rotary_dim);
    hash_u64(
        hash,
        std::bit_cast<std::uint32_t>(
            contract.rope_theta));
    hash_u64(
        hash,
        contract.uses_dedicated_embeddings
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        contract.hidden_state_source_layer_explicit
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        contract.provisional_kv_bytes_per_token);
    hash_u64(
        hash,
        contract.forward_semantics_resolved
            ? 1ULL
            : 0ULL);
    return hash;
}

}  // namespace

MtpDraftStructuralContract
build_mtp_draft_structural_contract(
    const ModelContract& model,
    const MtpDraftRuntimeBinding& binding) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (binding.compatibility_identity == 0 ||
        !binding.model_content_identity.valid() ||
        binding.tensor_indices.empty()) {
        throw std::invalid_argument(
            "MTP structural contract requires a valid checkpoint binding");
    }
    if (model.mtp_full_attention_layers == 0) {
        throw std::invalid_argument(
            "MTP structural contract requires at least one declared full-attention layer");
    }

    MtpDraftStructuralContract contract;
    contract.proposal_identity =
        binding.compatibility_identity;
    contract.model_content_bound_identity =
        binding.compatibility_identity;
    contract.vocab_size =
        model.vocab_size;
    contract.hidden_size =
        model.hidden_size;
    contract.num_hidden_layers =
        model.mtp_full_attention_layers;
    contract.full_attention_layers =
        model.mtp_full_attention_layers;
    contract.attention_heads =
        model.attention_heads;
    contract.kv_heads =
        model.kv_heads;
    contract.head_dim =
        model.head_dim;
    contract.rotary_dim =
        model.rotary_dim();
    contract.rope_theta =
        model.rope_theta;

    contract.uses_dedicated_embeddings =
        model.mtp_use_dedicated_embeddings;
    contract.hidden_state_source_layer_explicit =
        model.mtp_hidden_state_source_layer_explicit;

    const auto kv_elements =
        checked_mul(
            checked_mul(
                contract.full_attention_layers,
                contract.kv_heads),
            checked_mul(
                contract.head_dim,
                2ULL));
    contract.provisional_kv_bytes_per_token =
        checked_mul(
            kv_elements,
            model.bytes_bf16);

    contract.forward_semantics_resolved = false;
    contract.contract_identity =
        build_identity(contract);
    if (contract.contract_identity == 0) {
        throw std::logic_error(
            "MTP structural contract identity must be non-zero");
    }
    return contract;
}

void validate_mtp_draft_structural_contract(
    const ModelContract& model,
    const MtpDraftRuntimeBinding& binding,
    const MtpDraftStructuralContract& contract) {
    const auto expected =
        build_mtp_draft_structural_contract(
            model,
            binding);

    if (contract.proposal_identity !=
            expected.proposal_identity ||
        contract.model_content_bound_identity !=
            expected.model_content_bound_identity ||
        contract.vocab_size != expected.vocab_size ||
        contract.hidden_size != expected.hidden_size ||
        contract.num_hidden_layers !=
            expected.num_hidden_layers ||
        contract.full_attention_layers !=
            expected.full_attention_layers ||
        contract.attention_heads !=
            expected.attention_heads ||
        contract.kv_heads != expected.kv_heads ||
        contract.head_dim != expected.head_dim ||
        contract.rotary_dim !=
            expected.rotary_dim ||
        contract.rope_theta !=
            expected.rope_theta ||
        contract.uses_dedicated_embeddings !=
            expected.uses_dedicated_embeddings ||
        contract.hidden_state_source_layer_explicit !=
            expected.hidden_state_source_layer_explicit ||
        contract.provisional_kv_bytes_per_token !=
            expected.provisional_kv_bytes_per_token ||
        contract.forward_semantics_resolved ||
        contract.contract_identity !=
            expected.contract_identity) {
        throw std::invalid_argument(
            "MTP structural contract is stale, incompatible or overclaims resolved semantics");
    }
}

void require_mtp_draft_execution_semantics(
    const MtpDraftStructuralContract& contract) {
    if (contract.contract_identity == 0) {
        throw std::invalid_argument(
            "MTP execution requires initialized structural contract");
    }
    if (!contract.forward_semantics_resolved) {
        throw std::logic_error(
            "MTP draft forward semantics are unresolved; execution is disabled");
    }
}

}  // namespace strix
