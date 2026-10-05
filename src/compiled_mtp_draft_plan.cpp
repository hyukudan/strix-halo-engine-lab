#include "strix/compiled_mtp_draft_plan.hpp"

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

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "MTP draft plan tensor-byte overflow");
    }
    return left + right;
}

void validate_structural_binding(
    const ModelManifest& manifest,
    const MtpDraftRuntimeBinding& binding) {
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(
                manifest) ||
        binding.manifest_layout_identity !=
            manifest.layout_identity ||
        !binding.model_content_identity.valid() ||
        binding.compatibility_identity == 0 ||
        binding.tensor_indices.empty()) {
        throw std::invalid_argument(
            "MTP draft plan requires initialized compatible manifest binding");
    }

    std::vector<std::uint32_t> expected;
    std::uint64_t total = 0;
    for (std::uint32_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        const auto& tensor =
            manifest.tensors[index];
        if (tensor.role !=
            TensorRole::mtp) {
            continue;
        }
        expected.push_back(index);
        total =
            checked_add(
                total,
                tensor.byte_size);
    }

    if (binding.tensor_indices !=
            expected ||
        binding.total_tensor_bytes !=
            total) {
        throw std::invalid_argument(
            "MTP draft plan binding does not exactly cover manifest MTP tensors");
    }
}

std::uint64_t build_plan_identity(
    const CompiledMtpDraftPlan& plan) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, plan.proposal_identity);
    hash_u64(hash, plan.contract_identity);
    hash_u64(
        hash,
        plan.tensor_audit.manifest_layout_identity);
    hash_u64(
        hash,
        plan.attention.kv_row_bytes);
    hash_u64(
        hash,
        plan.attention.raw_index_row_bytes);
    hash_u64(
        hash,
        plan.state.state_width);
    hash_u64(
        hash,
        plan.output.wide_input_size);
    hash_u64(
        hash,
        plan.output.mixed_hidden_size);
    hash_u64(
        hash,
        plan.output.vocab_size);
    hash_u64(
        hash,
        plan.forward.contract_identity);
    hash_u64(hash, plan.steps.size());
    for (const auto step : plan.steps) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                step));
    }
    return hash;
}

}  // namespace

CompiledMtpDraftPlan
compile_mtp_draft_plan(
    const ModelContract& model,
    const ModelManifest& manifest,
    const MtpDraftRuntimeBinding& binding) {
    validate_structural_binding(
        manifest,
        binding);

    CompiledMtpDraftPlan plan;
    plan.proposal_identity =
        binding.compatibility_identity;
    plan.structural =
        build_mtp_draft_structural_contract(
            model,
            binding);
    plan.contract_identity =
        plan.structural.contract_identity;
    plan.attention =
        build_mtp_attention_structural_contract(
            model);
    plan.state =
        build_mtp_draft_state_contract(
            model);
    plan.tensor_audit =
        audit_mtp_draft_tensor_families(
            manifest,
            binding);
    plan.output =
        bind_mtp_output_head(
            model,
            manifest,
            plan.tensor_audit);
    plan.forward =
        build_mtp_forward_composition_contract(
            model,
            plan.structural,
            plan.attention,
            plan.state,
            plan.output,
            plan.tensor_audit);

    plan.steps = {
        MtpDraftPlanStep::input_fusion,
        MtpDraftPlanStep::attention_hyper_mix,
        MtpDraftPlanStep::qsa_indexer,
        MtpDraftPlanStep::sparse_attention,
        MtpDraftPlanStep::attention_combine,
        MtpDraftPlanStep::mlp_hyper_mix,
        MtpDraftPlanStep::routed_and_shared_moe,
        MtpDraftPlanStep::mlp_combine,
        MtpDraftPlanStep::publish_next_state,
        MtpDraftPlanStep::output_head_mix,
        MtpDraftPlanStep::lm_head,
    };

    plan.readiness =
        assess_mtp_draft_readiness(
            plan.structural,
            plan.attention,
            plan.state,
            plan.output,
            plan.tensor_audit);
    plan.executable =
        plan.readiness.executable;
    plan.plan_identity =
        build_plan_identity(plan);

    if (plan.plan_identity == 0) {
        throw std::logic_error(
            "MTP draft plan identity must be non-zero");
    }
    return plan;
}

void require_executable_mtp_draft_plan(
    const CompiledMtpDraftPlan& plan) {
    if (plan.plan_identity == 0 ||
        plan.proposal_identity == 0 ||
        plan.contract_identity == 0) {
        throw std::invalid_argument(
            "MTP draft plan is not initialized");
    }
    if (!plan.executable) {
        throw std::logic_error(
            "MTP draft plan is structural-only; semantic blockers remain");
    }
}

}  // namespace strix
