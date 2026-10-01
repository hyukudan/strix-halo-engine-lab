#include "strix/mtp_draft_readiness.hpp"

#include "strix/model_contract.hpp"

#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::MtpDraftRuntimeBinding binding;
    binding.model_content_identity.bytes[0] = 0x42;
    binding.tensor_indices = {0};
    binding.compatibility_identity = 0x1234ULL;

    const auto structural =
        strix::build_mtp_draft_structural_contract(
            model,
            binding);
    const auto attention =
        strix::build_mtp_attention_structural_contract(
            model);
    const auto state =
        strix::build_mtp_draft_state_contract(
            model);
    const auto output =
        strix::build_mtp_output_structural_contract(
            model);

    strix::MtpDraftTensorAudit tensors;
    tensors.manifest_layout_identity = 1;
    tensors.proposal_identity =
        binding.compatibility_identity;
    tensors.families = {
        {strix::MtpTensorFamily::input_combiner, true, {0}},
        {strix::MtpTensorFamily::embedding_norm, true, {0}},
        {strix::MtpTensorFamily::hidden_norm, true, {0}},
        {strix::MtpTensorFamily::hyper_connection_mixer, true, {0}},
        {strix::MtpTensorFamily::attention, true, {0}},
        {strix::MtpTensorFamily::indexer, true, {0}},
        {strix::MtpTensorFamily::moe, true, {0}},
    };

    const auto readiness =
        strix::assess_mtp_draft_readiness(
            structural,
            attention,
            state,
            output,
            tensors);

    expect(readiness.tensor_families_complete &&
               readiness.input_fusion_semantics_resolved &&
               readiness.draft_state_semantics_resolved,
           "readiness reports independently resolved MTP subcontracts");
    expect(readiness.attention_semantics_resolved &&
               readiness.output_mixer_semantics_resolved &&
               !readiness.output_head_binding_resolved &&
               !readiness.output_semantics_resolved &&
               !readiness.executable,
           "readiness keeps unresolved QSA/head binding from enabling MTP");
    expect(readiness.blockers.size() == 2,
           "readiness exposes LM-head binding and complete-forward blockers");

    auto incomplete = tensors;
    incomplete.missing_required_families.push_back(
        strix::MtpTensorFamily::indexer);
    const auto blocked =
        strix::assess_mtp_draft_readiness(
            structural,
            attention,
            state,
            output,
            incomplete);
    expect(!blocked.tensor_families_complete &&
               blocked.blockers.size() == 3,
           "readiness reports structural tensor blockers separately");

    if (failures == 0) {
        std::cout << "MTP draft readiness tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
