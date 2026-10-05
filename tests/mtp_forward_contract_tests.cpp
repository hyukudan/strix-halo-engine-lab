#include "strix/mtp_forward_contract.hpp"

#include "strix/mtp_draft_binding.hpp"

#include <iostream>
#include <string>

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

    strix::ModelManifest manifest;
    manifest.architecture = "fixture";

    auto add = [&](std::string name) {
        strix::ManifestTensor tensor;
        tensor.name = std::move(name);
        tensor.dimensions = {64, 64};
        tensor.ggml_type = 1;
        tensor.byte_size = 4096;
        tensor.absolute_offset =
            4096 + manifest.total_tensor_bytes;
        tensor.role = strix::TensorRole::mtp;
        manifest.total_tensor_bytes += tensor.byte_size;
        manifest.tensors.push_back(std::move(tensor));
    };

    add("mtp.fc_embedding.weight");
    add("mtp.fc_hidden.weight");
    add("mtp.enorm.weight");
    add("mtp.hnorm.weight");
    add("mtp.hyper_connection_mixer.down.weight");
    add("mtp.layers.0.self_attn.q_proj.weight");
    add("mtp.layers.0.self_attn.indexer.index_qk_proj.weight");
    add("mtp.layers.0.mlp.experts.gate_up_proj.weight");
    add("mtp.model.shared_head.head.weight");
    manifest.layout_identity =
        strix::model_manifest_layout_identity(manifest);

    strix::MtpDraftRuntimeBinding binding;
    binding.manifest_layout_identity = manifest.layout_identity;
    binding.model_content_identity.bytes[0] = 0x42;
    binding.total_tensor_bytes = manifest.total_tensor_bytes;
    binding.compatibility_identity = 0x1234ULL;
    for (std::uint32_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        binding.tensor_indices.push_back(index);
    }

    const auto structural =
        strix::build_mtp_draft_structural_contract(model, binding);
    const auto attention =
        strix::build_mtp_attention_structural_contract(model);
    const auto state =
        strix::build_mtp_draft_state_contract(model);
    const auto audit =
        strix::audit_mtp_draft_tensor_families(manifest, binding);
    const auto output =
        strix::bind_mtp_output_head(model, manifest, audit);

    const auto contract =
        strix::build_mtp_forward_composition_contract(
            model, structural, attention, state, output, audit);

    expect(contract.contract_identity != 0 &&
               contract.composition_resolved,
           "MTP forward composition closes cross-stage structural interfaces");
    expect(contract.attention_query_width == model.hidden_size &&
               contract.draft_state_width ==
                   model.hidden_size * model.hc_count &&
               contract.output_wide_input_size ==
                   contract.draft_state_width,
           "MTP forward composition preserves hidden/wide-state geometry");

    strix::validate_mtp_forward_composition_contract(
        model, structural, attention, state, output, audit, contract);

    auto stale = contract;
    ++stale.draft_state_width;
    bool rejected = false;
    try {
        strix::validate_mtp_forward_composition_contract(
            model, structural, attention, state, output, audit, stale);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected,
           "MTP forward composition rejects stale geometry");

    if (failures == 0) {
        std::cout << "MTP forward composition tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
