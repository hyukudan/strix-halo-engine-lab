#include "strix/execution_tensor_binding.hpp"

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

void add_tensor(strix::ModelManifest& manifest,
                std::string name,
                strix::TensorRole role,
                std::optional<std::uint32_t> layer = std::nullopt) {
    strix::ManifestTensor tensor;
    tensor.name = std::move(name);
    tensor.dimensions = {32, 32};
    tensor.ggml_type = 1;
    tensor.byte_size = 2048;
    tensor.absolute_offset =
        static_cast<std::uint64_t>(manifest.tensors.size()) * 4096ULL;
    tensor.role = role;
    tensor.layer = layer;
    manifest.total_tensor_bytes += tensor.byte_size;
    manifest.tensors.push_back(std::move(tensor));
}

strix::ModelManifest fixture_manifest(
    const strix::ModelContract& model) {
    strix::ModelManifest manifest;
    manifest.architecture = "qwen4_exp";

    add_tensor(
        manifest, "token_embd.weight",
        strix::TensorRole::embedding);
    add_tensor(
        manifest, "output.weight",
        strix::TensorRole::output_head);
    add_tensor(
        manifest, "final_norm.weight",
        strix::TensorRole::normalization);
    add_tensor(
        manifest, "ple.ngram_embedding",
        strix::TensorRole::ngram_table);

    for (std::uint32_t layer = 0;
         layer < model.hidden_layers;
         ++layer) {
        add_tensor(
            manifest,
            "blk." + std::to_string(layer) + ".attn_norm.weight",
            strix::TensorRole::normalization,
            layer);
        add_tensor(
            manifest,
            "blk." + std::to_string(layer) + ".attn_q.weight",
            strix::TensorRole::attention_projection,
            layer);
        add_tensor(
            manifest,
            "blk." + std::to_string(layer) + ".ffn_norm.weight",
            strix::TensorRole::normalization,
            layer);
        add_tensor(
            manifest,
            "blk." + std::to_string(layer) + ".router.weight",
            strix::TensorRole::moe_router,
            layer);
        add_tensor(
            manifest,
            "blk." + std::to_string(layer) + ".experts.0.gate_up.weight",
            strix::TensorRole::moe_expert,
            layer);
        add_tensor(
            manifest,
            "blk." + std::to_string(layer) + ".shared_expert.gate_proj.weight",
            strix::TensorRole::dense_projection,
            layer);

        const auto one_based =
            static_cast<std::uint64_t>(layer) + 1ULL;
        if (one_based % model.full_attention_interval != 0) {
            add_tensor(
                manifest,
                "blk." + std::to_string(layer) + ".gdn.weight",
                strix::TensorRole::recurrent_state,
                layer);
        } else {
            add_tensor(
                manifest,
                "blk." + std::to_string(layer) + ".indexer.weight",
                strix::TensorRole::sparse_index,
                layer);
        }
    }

    manifest.layout_identity =
        strix::model_manifest_layout_identity(manifest);
    return manifest;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(model);
    const auto manifest =
        fixture_manifest(model);

    const auto audit =
        strix::audit_execution_tensor_bindings(
            manifest, plan);

    expect(audit.binding_identity != 0,
           "binding audit has stable non-zero identity");
    expect(audit.missing_steps == 0,
           "complete fixture has no missing tensor families");
    expect(audit.ambiguous_steps == 0,
           "complete fixture has no ambiguous unique bindings");

    const auto* embedding =
        strix::find_step_tensor_binding(
            audit, plan.steps.front().id);
    expect(embedding != nullptr &&
               embedding->resolution ==
                   strix::TensorBindingResolution::resolved &&
               embedding->tensor_indices.size() == 1,
           "embedding resolves uniquely");

    auto missing_manifest = manifest;
    missing_manifest.tensors.erase(
        missing_manifest.tensors.begin() + 1);
    missing_manifest.layout_identity =
        strix::model_manifest_layout_identity(
            missing_manifest);
    const auto missing =
        strix::audit_execution_tensor_bindings(
            missing_manifest, plan);
    expect(missing.missing_steps >= 1,
           "missing output head is reported");

    auto ambiguous_manifest = manifest;
    add_tensor(
        ambiguous_manifest,
        "lm_head.weight",
        strix::TensorRole::output_head);
    ambiguous_manifest.layout_identity =
        strix::model_manifest_layout_identity(
            ambiguous_manifest);
    const auto ambiguous =
        strix::audit_execution_tensor_bindings(
            ambiguous_manifest, plan);
    expect(ambiguous.ambiguous_steps >= 1,
           "duplicate output-head candidates are reported");

    const auto second =
        strix::audit_execution_tensor_bindings(
            manifest, plan);
    expect(second.binding_identity ==
               audit.binding_identity,
           "binding identity is deterministic");

    if (failures == 0) {
        std::cout << "execution tensor binding tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
