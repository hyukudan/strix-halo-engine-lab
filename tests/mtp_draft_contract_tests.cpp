#include "strix/mtp_draft_contract.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::MtpDraftRuntimeBinding binding() {
    strix::MtpDraftRuntimeBinding result;
    result.manifest_layout_identity = 0x11ULL;
    result.model_content_identity.bytes[0] = 0x42;
    result.tensor_indices = {1, 2, 3};
    result.total_tensor_bytes = 4096;
    result.compatibility_identity = 0x12345678ULL;
    result.physical_binding_identity = 0x87654321ULL;
    return result;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto mtp_binding =
        binding();

    const auto contract =
        strix::build_mtp_draft_structural_contract(
            model,
            mtp_binding);

    expect(contract.num_hidden_layers == 1 &&
               contract.full_attention_layers == 1,
           "MTP contract pins one full-attention predictor layer");
    expect(!contract.uses_dedicated_embeddings &&
               !contract.hidden_state_source_layer_explicit,
           "MTP contract preserves pinned shared-embedding/null-hidden-source config");
    expect(contract.provisional_kv_bytes_per_token ==
               2048,
           "MTP contract reproduces provisional 2 KiB per-token KV planning estimate");
    expect(!contract.forward_semantics_resolved,
           "MTP structural contract does not claim unresolved forward semantics");
    expect(contract.proposal_identity ==
               mtp_binding.compatibility_identity &&
               contract.contract_identity != 0,
           "MTP structural contract binds checkpoint-compatible proposal identity");

    strix::validate_mtp_draft_structural_contract(
        model,
        mtp_binding,
        contract);

    auto overclaimed = contract;
    overclaimed.forward_semantics_resolved = true;
    bool overclaim_rejected = false;
    try {
        strix::validate_mtp_draft_structural_contract(
            model,
            mtp_binding,
            overclaimed);
    } catch (const std::invalid_argument&) {
        overclaim_rejected = true;
    }
    expect(overclaim_rejected,
           "MTP validator rejects claiming resolved forward semantics");

    auto changed_binding =
        mtp_binding;
    ++changed_binding.compatibility_identity;
    bool checkpoint_rejected = false;
    try {
        strix::validate_mtp_draft_structural_contract(
            model,
            changed_binding,
            contract);
    } catch (const std::invalid_argument&) {
        checkpoint_rejected = true;
    }
    expect(checkpoint_rejected,
           "MTP structural contract cannot cross checkpoint binding identity");

    if (failures == 0) {
        std::cout << "MTP draft structural contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
