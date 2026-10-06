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

    strix::ManifestTensor head_norm;
    head_norm.name =
        "mtp.blk.48.nextn.hc_head_norm.weight";
    head_norm.dimensions = {2560, 4};
    head_norm.ggml_type = 30;
    head_norm.byte_size = 20480;
    head_norm.absolute_offset = 4096;
    head_norm.role = strix::TensorRole::mtp;
    manifest.tensors.push_back(
        head_norm);

    strix::ManifestTensor head_down;
    head_down.name =
        "mtp.blk.48.nextn.hc_head_down.weight";
    head_down.dimensions = {10240, 320};
    head_down.ggml_type = 8;
    head_down.byte_size = 1024;
    head_down.absolute_offset = 24576;
    head_down.role = strix::TensorRole::mtp;
    manifest.tensors.push_back(
        head_down);

    strix::ManifestTensor head_up;
    head_up.name =
        "mtp.blk.48.nextn.hc_head_up.weight";
    head_up.dimensions = {320, 10240};
    head_up.ggml_type = 8;
    head_up.byte_size = 1024;
    head_up.absolute_offset = 25600;
    head_up.role = strix::TensorRole::mtp;
    manifest.tensors.push_back(
        head_up);

    strix::ManifestTensor head;
    head.name =
        "mtp.model.shared_head.head.weight";
    head.dimensions = {2560, 248320};
    head.ggml_type = 1;
    head.byte_size = 4096;
    head.absolute_offset = 26624;
    head.role = strix::TensorRole::mtp;
    manifest.tensors.push_back(
        head);

    for (const auto& tensor :
         manifest.tensors) {
        manifest.total_tensor_bytes +=
            tensor.byte_size;
    }
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::MtpDraftTensorAudit audit;
    audit.manifest_layout_identity =
        manifest.layout_identity;
    audit.families = {
        {strix::MtpTensorFamily::output_head, false, {3}},
    };

    const auto bound =
        strix::bind_mtp_output_head(
            model,
            manifest,
            audit);
    expect(bound.head_mix_binding_resolved &&
               bound.head_norm_tensor_index == 0 &&
               bound.head_down_tensor_index == 1 &&
               bound.head_up_tensor_index == 2 &&
               bound.head_binding_resolved &&
               bound.output_semantics_resolved &&
               bound.lm_head_source ==
                   strix::MtpLmHeadSource::
                       mtp_sidecar_shared_head &&
               bound.lm_head_tensor_index == 3,
           "explicit MTP shared head resolves output binding without target fallback");

    auto bad_geometry_manifest =
        manifest;
    bad_geometry_manifest.tensors[1].
        dimensions = {10239, 320};
    bad_geometry_manifest.layout_identity =
        strix::model_manifest_layout_identity(
            bad_geometry_manifest);
    auto bad_geometry_audit =
        audit;
    bad_geometry_audit.manifest_layout_identity =
        bad_geometry_manifest.layout_identity;
    bool bad_geometry_rejected = false;
    try {
        static_cast<void>(
            strix::bind_mtp_output_head(
                model,
                bad_geometry_manifest,
                bad_geometry_audit));
    } catch (const std::invalid_argument&) {
        bad_geometry_rejected = true;
    }
    expect(
        bad_geometry_rejected,
        "MTP output binding rejects incompatible hc_head_down geometry");

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
