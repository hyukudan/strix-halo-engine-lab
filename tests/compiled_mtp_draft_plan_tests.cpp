#include "strix/compiled_mtp_draft_plan.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct Fixture {
    strix::ModelContract model =
        strix::ModelContract::qwen38_flash_next();
    strix::ModelManifest manifest;
    strix::MtpDraftRuntimeBinding binding;

    Fixture() {
        auto add =
            [&](std::string name) {
                strix::ManifestTensor tensor;
                tensor.name =
                    std::move(name);
                tensor.dimensions = {64, 64};
                tensor.ggml_type = 1;
                tensor.byte_size = 4096;
                tensor.absolute_offset =
                    4096 +
                    manifest.total_tensor_bytes;
                tensor.role =
                    strix::TensorRole::mtp;
                manifest.total_tensor_bytes +=
                    tensor.byte_size;
                manifest.tensors.push_back(
                    std::move(tensor));
            };

        manifest.architecture = "fixture";
        add("mtp.fc_embedding.weight");
        add("mtp.fc_hidden.weight");
        add("mtp.enorm.weight");
        add("mtp.hnorm.weight");
        add("mtp.hyper_connection_mixer.down.weight");
        add("mtp.layers.0.self_attn.q_proj.weight");
        add("mtp.layers.0.self_attn.indexer.index_qk_proj.weight");
        add("mtp.layers.0.mlp.experts.gate_up_proj.weight");
        manifest.layout_identity =
            strix::model_manifest_layout_identity(
                manifest);

        binding.manifest_layout_identity =
            manifest.layout_identity;
        binding.model_content_identity.bytes[0] =
            0x42;
        binding.total_tensor_bytes =
            manifest.total_tensor_bytes;
        binding.compatibility_identity =
            0x1234ULL;
        binding.physical_binding_identity =
            0x5678ULL;
        for (std::uint32_t index = 0;
             index < manifest.tensors.size();
             ++index) {
            binding.tensor_indices.push_back(
                index);
        }
    }
};

}  // namespace

int main() {
    Fixture fixture;
    const auto plan =
        strix::compile_mtp_draft_plan(
            fixture.model,
            fixture.manifest,
            fixture.binding);

    expect(plan.plan_identity != 0 &&
               plan.proposal_identity ==
                   fixture.binding.compatibility_identity,
           "compiled MTP structural plan has stable checkpoint-compatible identity");
    expect(plan.steps.size() == 11 &&
               plan.steps.front() ==
                   strix::MtpDraftPlanStep::input_fusion &&
               plan.steps.back() ==
                   strix::MtpDraftPlanStep::lm_head,
           "compiled MTP plan freezes the structural forward step order");
    expect(!plan.executable &&
               !plan.readiness.executable &&
               plan.readiness.input_fusion_semantics_resolved &&
               plan.readiness.draft_state_semantics_resolved &&
               plan.readiness.output_mixer_semantics_resolved &&
               !plan.readiness.output_head_binding_resolved,
           "compiled MTP plan resolves output mixer but remains blocked on head binding and QSA");

    bool blocked = false;
    try {
        strix::require_executable_mtp_draft_plan(
            plan);
    } catch (const std::logic_error&) {
        blocked = true;
    }
    expect(blocked,
           "structural MTP plan cannot be launched");

    auto relocated =
        fixture.binding;
    relocated.physical_binding_identity =
        0x9999ULL;
    const auto relocated_plan =
        strix::compile_mtp_draft_plan(
            fixture.model,
            fixture.manifest,
            relocated);
    expect(relocated_plan.plan_identity ==
               plan.plan_identity,
           "MTP structural plan identity ignores physical relocation");

    auto incomplete =
        fixture.binding;
    incomplete.tensor_indices.pop_back();
    incomplete.total_tensor_bytes -= 4096;
    bool incomplete_rejected = false;
    try {
        static_cast<void>(
            strix::compile_mtp_draft_plan(
                fixture.model,
                fixture.manifest,
                incomplete));
    } catch (const std::invalid_argument&) {
        incomplete_rejected = true;
    }
    expect(incomplete_rejected,
           "MTP structural plan rejects incomplete checkpoint coverage");

    if (failures == 0) {
        std::cout << "compiled MTP draft plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
