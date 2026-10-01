#include "strix/mtp_output_contract.hpp"

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

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto contract =
        strix::build_mtp_output_structural_contract(
            model);

    expect(contract.wide_input_size == 10240 &&
               contract.mixed_hidden_size == 2560 &&
               contract.vocab_size == 248320,
           "MTP output collapses wide residual to model hidden width before logits");
    expect(contract.uses_gated_residual_head_mix &&
               !contract.head_mix_has_injection &&
               contract.lm_head_required,
           "MTP output requires non-injecting gated residual head mix followed by LM head");
    expect(contract.mixer_semantics_resolved &&
               !contract.head_binding_resolved &&
               !contract.output_semantics_resolved &&
               contract.lm_head_source ==
                   strix::MtpLmHeadSource::unresolved,
           "MTP output separates resolved mixer formula from unresolved checkpoint head binding");

    strix::validate_mtp_output_structural_contract(
        model,
        contract);

    strix::ModelManifest manifest;
    manifest.architecture = "fixture";
    strix::ManifestTensor head;
    head.name = "mtp.model.shared_head.head.weight";
    head.dimensions = {2560, 248320};
    head.ggml_type = 1;
    head.byte_size = 4096;
    head.absolute_offset = 4096;
    head.role = strix::TensorRole::mtp;
    manifest.total_tensor_bytes =
        head.byte_size;
    manifest.tensors.push_back(head);
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::MtpDraftTensorAudit audit;
    audit.manifest_layout_identity =
        manifest.layout_identity;
    audit.families = {
        {strix::MtpTensorFamily::output_head, false, {0}},
    };

    const auto bound =
        strix::bind_mtp_output_head(
            model,
            manifest,
            audit);
    expect(bound.head_binding_resolved &&
               bound.output_semantics_resolved &&
               bound.lm_head_source ==
                   strix::MtpLmHeadSource::
                       mtp_sidecar_shared_head &&
               bound.lm_head_tensor_index == 0,
           "explicit MTP shared head resolves output binding without target fallback");

    auto overclaimed = contract;
    overclaimed.head_binding_resolved = true;
    bool rejected = false;
    try {
        strix::validate_mtp_output_structural_contract(
            model,
            overclaimed);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected,
           "MTP output validator rejects unverified LM-head binding");

    if (failures == 0) {
        std::cout << "MTP output structural contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
