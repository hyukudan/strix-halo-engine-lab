#include "strix/execution_tensor_access.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::ModelManifest fixture_manifest() {
    strix::ModelManifest manifest;
    manifest.architecture = "fixture";

    const auto add =
        [&](std::string name,
            strix::TensorRole role,
            std::optional<std::uint32_t> layer) {
            strix::ManifestTensor tensor;
            tensor.name = std::move(name);
            tensor.dimensions = {64, 64};
            tensor.ggml_type = 1;
            tensor.byte_size = 8192;
            tensor.absolute_offset =
                4096 +
                manifest.tensors.size() *
                    tensor.byte_size;
            tensor.role = role;
            tensor.layer = layer;
            manifest.total_tensor_bytes +=
                tensor.byte_size;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add(
        "token_embd.weight",
        strix::TensorRole::embedding,
        std::nullopt);
    add(
        "blk.0.experts.0.weight",
        strix::TensorRole::moe_expert,
        0);
    add(
        "blk.0.experts.1.weight",
        strix::TensorRole::moe_expert,
        0);
    add(
        "lm_head.weight",
        strix::TensorRole::output_head,
        std::nullopt);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);
    return manifest;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto manifest =
        fixture_manifest();
    const auto index =
        strix::build_model_tensor_index(
            manifest);


    const auto canonical_requirements =
        strix::build_serial_execution_tensor_requirements(
            execution);
    expect(!canonical_requirements.empty(),
           "canonical requirement generator covers the serial plan");

    bool incomplete_canonical_fixture_rejected = false;
    try {
        static_cast<void>(
            strix::compile_execution_tensor_access_plan(
                execution,
                manifest,
                index,
                canonical_requirements));
    } catch (const std::invalid_argument&) {
        incomplete_canonical_fixture_rejected = true;
    }
    expect(incomplete_canonical_fixture_rejected,
           "canonical requirements reject a deliberately incomplete tensor fixture");

    const auto embedding_requirement =
        std::find_if(
            canonical_requirements.begin(),
            canonical_requirements.end(),
            [](const auto& requirement) {
                return requirement.op ==
                           strix::SerialOp::token_embedding &&
                       requirement.binding_slot == 0;
            });
    expect(embedding_requirement !=
               canonical_requirements.end() &&
               embedding_requirement->selector.role ==
                   strix::TensorRole::embedding &&
               embedding_requirement->cardinality ==
                   strix::TensorRequirementCardinality::exactly_one,
           "canonical requirements make embedding unique");

    const auto routed_requirement =
        std::find_if(
            canonical_requirements.begin(),
            canonical_requirements.end(),
            [](const auto& requirement) {
                return requirement.op ==
                           strix::SerialOp::moe_experts &&
                       requirement.binding_slot == 0 &&
                       requirement.selector.layer.has_value() &&
                       *requirement.selector.layer == 0;
            });
    expect(routed_requirement !=
               canonical_requirements.end() &&
               routed_requirement->selector.role ==
                   strix::TensorRole::moe_expert &&
               routed_requirement->cardinality ==
                   strix::TensorRequirementCardinality::one_or_more,
           "canonical requirements bind routed expert family per layer");

    const auto shared_requirement =
        std::find_if(
            canonical_requirements.begin(),
            canonical_requirements.end(),
            [](const auto& requirement) {
                return requirement.op ==
                           strix::SerialOp::shared_expert &&
                       requirement.binding_slot == 0 &&
                       requirement.selector.layer.has_value() &&
                       *requirement.selector.layer == 0;
            });
    expect(shared_requirement !=
               canonical_requirements.end() &&
               shared_requirement->selector.role ==
                   strix::TensorRole::dense_projection &&
               shared_requirement->selector.name_contains.empty() &&
               shared_requirement->selector.name_contains_any.size() == 3,
           "canonical requirements keep shared expert separate from routed experts");

    const auto inject_requirement =
        std::find_if(
            canonical_requirements.begin(),
            canonical_requirements.end(),
            [](const auto& requirement) {
                return requirement.op ==
                           strix::SerialOp::attn_inject ||
                       requirement.op ==
                           strix::SerialOp::mlp_inject;
            });
    expect(inject_requirement ==
               canonical_requirements.end(),
           "inject application steps require no model tensors");

    const auto hyper_inject_requirement =
        std::find_if(
            canonical_requirements.begin(),
            canonical_requirements.end(),
            [](const auto& requirement) {
                return requirement.op ==
                           strix::SerialOp::attn_hyper_mix &&
                       requirement.binding_slot == 3 &&
                       requirement.selector.name_contains ==
                           "hc_attn_inject";
            });
    expect(hyper_inject_requirement !=
               canonical_requirements.end(),
           "attention hyper-mix owns injection projection weight");

    std::vector<strix::ExecutionTensorRequirement>
        requirements;

    strix::ExecutionTensorRequirement embedding;
    embedding.step_id = 0;
    embedding.op =
        strix::SerialOp::token_embedding;
    embedding.binding_slot = 0;
    embedding.selector.role =
        strix::TensorRole::embedding;
    requirements.push_back(embedding);

    const auto first_moe =
        std::find_if(
            execution.steps.begin(),
            execution.steps.end(),
            [](const auto& step) {
                return step.op ==
                    strix::SerialOp::moe_experts;
            });
    if (first_moe == execution.steps.end()) {
        throw std::runtime_error(
            "fixture execution has no MoE step");
    }

    strix::ExecutionTensorRequirement experts;
    experts.step_id = first_moe->id;
    experts.op =
        strix::SerialOp::moe_experts;
    experts.binding_slot = 0;
    experts.selector.role =
        strix::TensorRole::moe_expert;
    experts.selector.layer = 0;
    experts.cardinality =
        strix::TensorRequirementCardinality::one_or_more;
    requirements.push_back(experts);

    const auto lm =
        execution.steps.back();
    strix::ExecutionTensorRequirement output;
    output.step_id = lm.id;
    output.op = strix::SerialOp::lm_head;
    output.binding_slot = 0;
    output.selector.role =
        strix::TensorRole::output_head;
    requirements.push_back(output);

    const auto plan =
        strix::compile_execution_tensor_access_plan(
            execution,
            manifest,
            index,
            requirements);

    expect(plan.plan_identity != 0 &&
               plan.manifest_layout_identity ==
                   manifest.layout_identity,
           "tensor access plan has stable manifest-bound identity");

    const auto* embedding_access =
        strix::find_execution_tensor_access(
            plan, 0, 0);
    expect(embedding_access != nullptr &&
               embedding_access->tensor_indices.size() == 1 &&
               manifest.tensors[
                   embedding_access->tensor_indices[0]].name ==
                   "token_embd.weight",
           "embedding requirement compiles to one tensor index");

    const auto* expert_access =
        strix::find_execution_tensor_access(
            plan, first_moe->id, 0);
    expect(expert_access != nullptr &&
               expert_access->tensor_indices.size() == 2,
           "one-or-more requirement pre-resolves expert tensor set");

    const auto plan_again =
        strix::compile_execution_tensor_access_plan(
            execution,
            manifest,
            index,
            requirements);
    expect(plan_again.plan_identity ==
               plan.plan_identity,
           "tensor access plan identity is reproducible");

    auto ambiguous = requirements;
    ambiguous[0].selector.role =
        strix::TensorRole::moe_expert;
    ambiguous[0].selector.layer = 0;
    bool ambiguous_rejected = false;
    try {
        static_cast<void>(
            strix::compile_execution_tensor_access_plan(
                execution,
                manifest,
                index,
                ambiguous));
    } catch (const std::invalid_argument&) {
        ambiguous_rejected = true;
    }
    expect(ambiguous_rejected,
           "exactly-one access rejects ambiguous tensor selector");

    auto wrong_step = requirements;
    wrong_step[0].op =
        strix::SerialOp::lm_head;
    bool step_rejected = false;
    try {
        static_cast<void>(
            strix::compile_execution_tensor_access_plan(
                execution,
                manifest,
                index,
                wrong_step));
    } catch (const std::invalid_argument&) {
        step_rejected = true;
    }
    expect(step_rejected,
           "tensor requirement cannot attach to wrong operator");

    if (failures == 0) {
        std::cout << "execution tensor access tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
