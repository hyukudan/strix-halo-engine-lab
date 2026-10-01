#include "strix/dispatch_resources.hpp"
#include "strix/memory_plan.hpp"

#include <array>
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
    strix::ExecutionBufferUse read_use{
        {
            strix::ExecutionBufferClass::hidden,
            0,
        },
        1,
    };
    strix::ExecutionBufferUse write_use{
        {
            strix::ExecutionBufferClass::logits,
            0,
        },
        1,
    };

    strix::ExecutionArenaPlan arena;
    arena.alignment = 256;
    arena.total_bytes = 2048;
    arena.allocations.push_back(
        {read_use, 0, 1, 0, 512});
    arena.allocations.push_back(
        {write_use, 1, 1, 512, 1024});

    strix::ExecutionMemoryBinding scratch;
    scratch.execution_identity = 0x1111ULL;
    scratch.arena_base =
        static_cast<std::uintptr_t>(
            0x80000000ULL);
    scratch.arena_bytes = arena.total_bytes;
    scratch.allocation_generation = 2;

    strix::SerialSessionStateLayout state_layout;
    state_layout.context_tokens = 8;
    state_layout.device_alignment = 256;
    state_layout.host_alignment = 64;
    state_layout.device_total_bytes = 1024;
    state_layout.host_total_bytes = 64;
    state_layout.layout_identity = 0x2222ULL;
    state_layout.allocations.push_back(
        {
            strix::SessionStateComponent::recurrent_matrix,
            strix::PersistentState::recurrent_matrix,
            strix::SessionStateStorage::device,
            0,
            1024,
            false,
        });

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        static_cast<std::uintptr_t>(
            0x81000000ULL);
    state_memory.device_bytes = 1024;
    state_memory.device_allocation_generation = 3;
    state_memory.host_base =
        static_cast<std::uintptr_t>(
            0x82000000ULL);
    state_memory.host_bytes = 64;
    state_memory.host_allocation_generation = 4;

    strix::ModelManifest manifest;
    manifest.architecture = "fixture";
    strix::ManifestTensor mt;
    mt.name = "token_embd.weight";
    mt.dimensions = {64, 64};
    mt.ggml_type = 1;
    mt.byte_size = 8192;
    mt.absolute_offset = 4096;
    mt.role = strix::TensorRole::embedding;
    manifest.tensors.push_back(mt);
    manifest.total_tensor_bytes = mt.byte_size;
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::ModelRuntimeBinding weights;
    weights.manifest_layout_identity =
        manifest.layout_identity;
    weights.model_content_identity.bytes[0] =
        static_cast<std::uint8_t>(0x3333ULL & 0xffULL);
    weights.model_content_identity.bytes[1] = 0x5a;
    weights.tensors.push_back(
        {
            0,
            strix::RuntimeTensorStorage::resident,
            static_cast<std::uintptr_t>(
                0x83000000ULL),
            mt.byte_size,
            5,
            0,
        });

    strix::ExecutionTensorAccessPlan tensor_access;
    tensor_access.manifest_layout_identity =
        manifest.layout_identity;
    tensor_access.plan_identity = 0x9999ULL;
    strix::ExecutionTensorAccess access;
    access.step_id = 7;
    access.op = strix::SerialOp::token_embedding;
    access.binding_slot = 0;
    access.tensor_indices = {0};
    tensor_access.accesses.push_back(access);

    strix::ExecutionRuntimeResources resources;
    resources.scratch = &scratch;
    resources.model_manifest = &manifest;
    resources.model_weights = &weights;
    resources.tensor_access = &tensor_access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &state_memory;

    const std::array reads{read_use};
    const std::array writes{write_use};

    strix::ExecutionBufferUse temporary_use;
    temporary_use.buffer.buffer_class =
        strix::ExecutionBufferClass::mlp_intermediate;
    temporary_use.buffer.slot = 0;
    temporary_use.generation = 1;
    const std::array temporaries{temporary_use};
    arena.allocations.push_back(
        {temporary_use, 7, 7, 1536, 128});

    strix::ExecutionDispatchContext context;
    context.step_id = 7;
    context.op = strix::SerialOp::token_embedding;
    context.reads = reads;
    context.writes = writes;
    context.temporaries = temporaries;
    context.arena = &arena;
    context.memory = &scratch;
    context.session_state_layout =
        &state_layout;
    context.session_state_memory =
        &state_memory;
    context.resources = &resources;

    const auto input =
        strix::resolve_dispatch_read(
            context, 0);
    const auto output =
        strix::resolve_dispatch_write(
            context, 0);
    const auto temporary =
        strix::resolve_dispatch_temporary(
            context, 0);
    expect(input.address ==
               scratch.arena_base &&
               output.address ==
                   scratch.arena_base + 512,
           "dispatch scratch resolver uses compiled arena offsets");
    expect(
        temporary.address ==
            scratch.arena_base + 1536,
        "dispatch temporary resolver uses compiled arena offsets");

    const auto recurrent =
        strix::resolve_dispatch_state(
            context,
            strix::SessionStateComponent::recurrent_matrix);
    expect(recurrent.address ==
               state_memory.device_base &&
               recurrent.allocation_generation == 3,
           "dispatch state resolver uses persistent device pool");

    const auto& resolved_tensor_access =
        strix::resolve_dispatch_tensor_access(
            context, 0);
    expect(resolved_tensor_access.tensor_indices.size() == 1 &&
               resolved_tensor_access.tensor_indices[0] == 0,
           "dispatch tensor slot resolves precompiled tensor indices");

    const auto& tensor_by_index =
        strix::resolve_dispatch_tensor(
            context, 0);
    const auto& tensor_by_name =
        strix::resolve_dispatch_tensor(
            context,
            "token_embd.weight");
    expect(tensor_by_index.address ==
               weights.tensors[0].address &&
               tensor_by_name.address ==
                   weights.tensors[0].address,
           "dispatch tensor resolver uses model runtime binding");


    const auto model =
        strix::ModelContract::qwen38_flash_next();
    strix::SerialSessionStateLayout gdn_layout;
    gdn_layout.context_tokens = 8;
    gdn_layout.device_alignment = 256;
    gdn_layout.host_alignment = 64;
    gdn_layout.device_total_bytes =
        strix::gdn_matrix_state_bytes_per_session(model) +
        strix::gdn_conv_state_bytes_per_session(model);
    gdn_layout.host_total_bytes = 64;
    gdn_layout.layout_identity = 0x7777ULL;
    gdn_layout.allocations.push_back(
        {
            strix::SessionStateComponent::recurrent_matrix,
            strix::PersistentState::recurrent_matrix,
            strix::SessionStateStorage::device,
            0,
            strix::gdn_matrix_state_bytes_per_session(model),
            false,
        });
    gdn_layout.allocations.push_back(
        {
            strix::SessionStateComponent::gdn_convolution,
            strix::PersistentState::convolution_history,
            strix::SessionStateStorage::device,
            strix::gdn_matrix_state_bytes_per_session(model),
            strix::gdn_conv_state_bytes_per_session(model),
            false,
        });
    strix::SessionStateMemoryBinding gdn_memory;
    gdn_memory.layout_identity =
        gdn_layout.layout_identity;
    gdn_memory.device_base =
        static_cast<std::uintptr_t>(0x90000000ULL);
    gdn_memory.device_bytes =
        gdn_layout.device_total_bytes;
    gdn_memory.device_allocation_generation = 8;
    gdn_memory.host_base =
        static_cast<std::uintptr_t>(0x91000000ULL);
    gdn_memory.host_bytes = 64;
    gdn_memory.host_allocation_generation = 1;

    auto gdn_context = context;
    gdn_context.layer = 4;
    gdn_context.session_state_layout =
        &gdn_layout;
    gdn_context.session_state_memory =
        &gdn_memory;
    auto gdn_resources = resources;
    gdn_resources.session_state_layout =
        &gdn_layout;
    gdn_resources.session_state_memory =
        &gdn_memory;
    gdn_context.resources =
        &gdn_resources;

    const auto gdn_matrix =
        strix::resolve_dispatch_gdn_layer_state(
            gdn_context,
            model,
            strix::SessionStateComponent::recurrent_matrix);
    expect(
        gdn_matrix.bytes ==
            strix::gdn_matrix_state_bytes_per_layer(model) &&
        gdn_matrix.address ==
            gdn_memory.device_base +
            3ULL *
                strix::gdn_matrix_state_bytes_per_layer(model),
        "dispatch GDN resolver slices one recurrent layer");

    bool read_rejected = false;
    try {
        static_cast<void>(
            strix::resolve_dispatch_read(
                context, 1));
    } catch (const std::out_of_range&) {
        read_rejected = true;
    }
    expect(read_rejected,
           "dispatch read resolver checks index bounds");

    auto no_model = resources;
    no_model.model_weights = nullptr;
    context.resources = &no_model;
    bool model_rejected = false;
    try {
        static_cast<void>(
            strix::resolve_dispatch_tensor(
                context, 0));
    } catch (const std::invalid_argument&) {
        model_rejected = true;
    }
    expect(model_rejected,
           "dispatch tensor resolver requires model weights");

    if (failures == 0) {
        std::cout << "dispatch resource tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
