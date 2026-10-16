#include "strix/mtp_forward_semantics.hpp"

#include <limits>
#include <bit>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() /
                left) {
        throw std::overflow_error(
            "MTP forward semantic geometry overflow");
    }
    return left * right;
}

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) &
            0xffULL;
        hash *=
            fnv_prime;
    }
}

std::uint64_t build_identity(
    const MtpForwardSemanticContract& contract) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(hash, contract.semantic_revision);
    hash_u64(hash, static_cast<std::uint64_t>(contract.predictor_position));
    hash_u64(hash, contract.hidden_norm_elements);
    hash_u64(hash, std::bit_cast<std::uint32_t>(contract.rms_norm_epsilon));
    hash_u64(hash, contract.embedding_projected_once_then_broadcast?1U:0U);
    hash_u64(hash, static_cast<std::uint64_t>(contract.hidden_norm_scope));
    hash_u64(hash, static_cast<std::uint64_t>(contract.source_norm_weight_encoding));
    hash_u64(hash, static_cast<std::uint64_t>(contract.input_arithmetic));
    hash_u64(hash, contract.hidden_size);
    hash_u64(hash, contract.hc_count);
    hash_u64(hash, contract.wide_hidden_size);
    hash_u64(hash, contract.eh_proj_input_size);
    hash_u64(hash, contract.eh_proj_output_size);
    hash_u64(
        hash,
        contract.hidden_rms_per_hc_stream
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        contract.token_rms_repeated_across_hc
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            contract.input_concat_order));
    hash_u64(
        hash,
        contract.attention_precedes_mlp
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        contract.attention_combined_before_mlp_mix
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        contract.mlp_combined_before_state_publish
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            contract.next_state_publication));
    hash_u64(
        hash,
        contract.head_mix_after_state_publish
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        contract.head_mix_has_injection
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        contract.lm_head_after_head_mix
            ? 1ULL
            : 0ULL);
    return hash;
}

}  // namespace

MtpForwardSemanticContract
build_mtp_forward_semantic_contract(
    const ModelContract& model, MtpInputFusionArithmetic arithmetic) {
    if(arithmetic!=MtpInputFusionArithmetic::split_projections_then_add &&
       arithmetic!=MtpInputFusionArithmetic::packed_concat_chain)
        throw std::invalid_argument("MTP input arithmetic profile is invalid");
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (model.hc_count == 0 ||
        model.hidden_size == 0) {
        throw std::invalid_argument(
            "MTP forward semantics require hidden/HC geometry");
    }

    MtpForwardSemanticContract
        contract;
    contract.semantic_revision = 3;
    contract.hidden_size =
        model.hidden_size;
    contract.hc_count =
        model.hc_count;
    contract.wide_hidden_size =
        checked_mul(
            model.hidden_size,
            model.hc_count);
    contract.eh_proj_input_size =
        checked_mul(
            model.hidden_size,
            2ULL);
    contract.eh_proj_output_size =
        model.hidden_size;

    contract.hidden_norm_elements = contract.wide_hidden_size;
    contract.rms_norm_epsilon = model.rms_norm_epsilon;
    contract.embedding_projected_once_then_broadcast = true;
    contract.hidden_norm_scope = MtpHiddenNormScope::wide_residual;
    contract.source_norm_weight_encoding = MtpNormWeightEncoding::gemma_offset;
    contract.input_arithmetic = arithmetic;
    contract.hidden_rms_per_hc_stream = false;
    contract.token_rms_repeated_across_hc =
        true;
    contract.input_concat_order =
        MtpInputConcatOrder::
            embedding_then_hidden;
    contract.attention_precedes_mlp =
        true;
    contract.attention_combined_before_mlp_mix =
        true;
    contract.mlp_combined_before_state_publish =
        true;
    contract.next_state_publication =
        MtpNextStatePublicationPoint::
            post_mlp_combine_pre_head_mix;
    contract.head_mix_after_state_publish =
        true;
    contract.head_mix_has_injection =
        false;
    contract.lm_head_after_head_mix =
        true;

    contract.contract_identity =
        build_identity(
            contract);
    if (contract.contract_identity == 0) {
        throw std::logic_error(
            "MTP forward semantic identity must be non-zero");
    }
    return contract;
}

void validate_mtp_forward_semantic_contract(
    const ModelContract& model,
    const MtpForwardSemanticContract& contract) {
    const auto expected =
        build_mtp_forward_semantic_contract(
            model, contract.input_arithmetic);
    if (std::bit_cast<std::uint32_t>(contract.rms_norm_epsilon) != std::bit_cast<std::uint32_t>(expected.rms_norm_epsilon) ||
        contract.embedding_projected_once_then_broadcast != expected.embedding_projected_once_then_broadcast ||
        contract.hidden_norm_elements != expected.hidden_norm_elements ||
        contract.predictor_position != expected.predictor_position ||
        contract.hidden_norm_scope != expected.hidden_norm_scope ||
        contract.source_norm_weight_encoding != expected.source_norm_weight_encoding ||
        contract.input_arithmetic != expected.input_arithmetic ||
        contract.semantic_revision !=
            expected.semantic_revision ||
        contract.hidden_size !=
            expected.hidden_size ||
        contract.hc_count !=
            expected.hc_count ||
        contract.wide_hidden_size !=
            expected.wide_hidden_size ||
        contract.eh_proj_input_size !=
            expected.eh_proj_input_size ||
        contract.eh_proj_output_size !=
            expected.eh_proj_output_size ||
        contract.hidden_rms_per_hc_stream !=
            expected.hidden_rms_per_hc_stream ||
        contract.token_rms_repeated_across_hc !=
            expected.token_rms_repeated_across_hc ||
        contract.input_concat_order !=
            expected.input_concat_order ||
        contract.attention_precedes_mlp !=
            expected.attention_precedes_mlp ||
        contract.attention_combined_before_mlp_mix !=
            expected.attention_combined_before_mlp_mix ||
        contract.mlp_combined_before_state_publish !=
            expected.mlp_combined_before_state_publish ||
        contract.next_state_publication !=
            expected.next_state_publication ||
        contract.head_mix_after_state_publish !=
            expected.head_mix_after_state_publish ||
        contract.head_mix_has_injection !=
            expected.head_mix_has_injection ||
        contract.lm_head_after_head_mix !=
            expected.lm_head_after_head_mix ||
        contract.contract_identity !=
            expected.contract_identity) {
        throw std::invalid_argument(
            "MTP forward semantic contract is stale or incompatible");
    }
}

}  // namespace strix
