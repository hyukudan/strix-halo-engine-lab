#include "strix/execution_resources.hpp"

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
    strix::ModelManifest manifest;
    manifest.architecture = "fixture";

    strix::ManifestTensor tensor;
    tensor.name = "token_embd.weight";
    tensor.dimensions = {64, 128};
    tensor.ggml_type = 1;
    tensor.byte_size = 16'384;
    tensor.absolute_offset = 4096;
    tensor.role = strix::TensorRole::embedding;
    tensor.placement.kind =
        strix::PlacementKind::local;
    tensor.placement.node = 0;
    manifest.tensors.push_back(tensor);
    manifest.total_tensor_bytes =
        tensor.byte_size;
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::ModelRuntimeBinding weights;
    weights.manifest_layout_identity =
        manifest.layout_identity;
    weights.model_content_identity =
        0xabcdefULL;
    weights.tensors.push_back(
        strix::RuntimeTensorBinding{
            0,
            strix::RuntimeTensorStorage::resident,
            static_cast<std::uintptr_t>(
                0x60000000ULL),
            tensor.byte_size,
            7,
            0,
        });

    strix::ExecutionArenaPlan arena;
    arena.alignment = 256;
    arena.total_bytes = 4096;

    strix::ExecutionMemoryBinding scratch;
    scratch.execution_identity =
        0x1234ULL;
    scratch.arena_base =
        static_cast<std::uintptr_t>(
            0x70000000ULL);
    scratch.arena_bytes =
        arena.total_bytes;
    scratch.allocation_generation = 3;

    strix::SerialSessionStateLayout state_layout;
    state_layout.context_tokens = 32;
    state_layout.device_alignment = 256;
    state_layout.host_alignment = 64;
    state_layout.device_total_bytes = 4096;
    state_layout.host_total_bytes = 64;
    state_layout.layout_identity =
        0x5678ULL;

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        static_cast<std::uintptr_t>(
            0x71000000ULL);
    state_memory.device_bytes =
        state_layout.device_total_bytes;
    state_memory.device_allocation_generation = 5;
    state_memory.host_base =
        static_cast<std::uintptr_t>(
            0x72000000ULL);
    state_memory.host_bytes =
        state_layout.host_total_bytes;
    state_memory.host_allocation_generation = 9;

    strix::ExecutionTensorAccessPlan tensor_access;
    tensor_access.manifest_layout_identity =
        manifest.layout_identity;
    tensor_access.plan_identity = 0x4444ULL;

    strix::ExecutionRuntimeResources resources;
    resources.scratch = &scratch;
    resources.model_manifest = &manifest;
    resources.model_weights = &weights;
    resources.tensor_access = &tensor_access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &state_memory;

    strix::validate_execution_runtime_resources(
        scratch.execution_identity,
        arena,
        resources,
        31);

    const auto ids =
        strix::build_runtime_execution_identities(
            scratch.execution_identity,
            arena,
            resources);
    expect(ids.compatibility_identity != 0 &&
               ids.binding_identity != 0 &&
               ids.compatibility_identity !=
                   ids.binding_identity,
           "runtime resources produce separate compatibility and binding identities");

    auto moved_scratch = scratch;
    moved_scratch.arena_base += 0x1000;
    ++moved_scratch.allocation_generation;
    auto moved_resources = resources;
    moved_resources.scratch =
        &moved_scratch;
    const auto moved_ids =
        strix::build_runtime_execution_identities(
            scratch.execution_identity,
            arena,
            moved_resources);
    expect(moved_ids.compatibility_identity ==
               ids.compatibility_identity &&
               moved_ids.binding_identity !=
                   ids.binding_identity,
           "physical scratch relocation changes binding identity only");

    auto changed_content = weights;
    ++changed_content.model_content_identity;
    auto changed_resources = resources;
    changed_resources.model_weights =
        &changed_content;
    const auto changed_ids =
        strix::build_runtime_execution_identities(
            scratch.execution_identity,
            arena,
            changed_resources);
    expect(changed_ids.compatibility_identity !=
               ids.compatibility_identity,
           "model content change invalidates compatibility identity");

    bool position_rejected = false;
    try {
        strix::validate_execution_runtime_resources(
            scratch.execution_identity,
            arena,
            resources,
            32);
    } catch (const std::out_of_range&) {
        position_rejected = true;
    }
    expect(position_rejected,
           "runtime resources enforce allocated context capacity");

    auto missing = resources;
    missing.model_weights = nullptr;
    bool missing_rejected = false;
    try {
        strix::validate_execution_runtime_resources(
            scratch.execution_identity,
            arena,
            missing,
            0);
    } catch (const std::invalid_argument&) {
        missing_rejected = true;
    }
    expect(missing_rejected,
           "runtime resources require model weight binding");

    missing = resources;
    missing.tensor_access = nullptr;
    bool access_rejected = false;
    try {
        strix::validate_execution_runtime_resources(
            scratch.execution_identity,
            arena,
            missing,
            0);
    } catch (const std::invalid_argument&) {
        access_rejected = true;
    }
    expect(access_rejected,
           "runtime resources require pre-resolved tensor access plan");

    if (failures == 0) {
        std::cout << "execution resources tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
