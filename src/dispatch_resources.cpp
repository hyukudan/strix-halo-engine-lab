#include "strix/dispatch_resources.hpp"
#include "strix/distributed_stage_resources.hpp"
#include <algorithm>
#include "strix/prepared_tensor_views.hpp"
#include "strix/prepared_arena_views.hpp"

#include <stdexcept>

namespace strix {
namespace {

// Pair validated by the innermost ValidatedModelBindingScope of this thread.
thread_local const ModelManifest* validated_manifest = nullptr;
thread_local const ModelRuntimeBinding* validated_binding = nullptr;

const ExecutionMemoryBinding& scratch_memory(
    const ExecutionDispatchContext& context) {
    if (context.resources != nullptr &&
        context.resources->scratch != nullptr) {
        return *context.resources->scratch;
    }
    if (context.memory != nullptr) {
        return *context.memory;
    }
    throw std::invalid_argument(
        "dispatch context has no scratch memory binding");
}

const SerialSessionStateLayout& state_layout(
    const ExecutionDispatchContext& context) {
    if (context.resources != nullptr &&
        context.resources->session_state_layout != nullptr) {
        return *context.resources->session_state_layout;
    }
    if (context.session_state_layout != nullptr) {
        return *context.session_state_layout;
    }
    throw std::invalid_argument(
        "dispatch context has no session-state layout");
}

const SessionStateMemoryBinding& state_memory(
    const ExecutionDispatchContext& context) {
    if (context.resources != nullptr &&
        context.resources->session_state_memory != nullptr) {
        return *context.resources->session_state_memory;
    }
    if (context.session_state_memory != nullptr) {
        return *context.session_state_memory;
    }
    throw std::invalid_argument(
        "dispatch context has no session-state memory binding");
}

const PreparedTensorViews* prepared_views(const ExecutionDispatchContext& context) {
    if (!context.prepared_tensor_views) return nullptr;
    if (!context.resources)
        throw std::invalid_argument("prepared tensor lookup requires its source resources");
    context.prepared_tensor_views->validate_stage_scope(context.resources->distributed_stage,context.step_id);
    context.prepared_tensor_views->validate_source_handles(context.resources->model_manifest,
        context.resources->model_weights, context.resources->tensor_access);
    return context.prepared_tensor_views;
}

const ModelManifest& model_manifest(
    const ExecutionDispatchContext& context) {
    if (context.resources == nullptr ||
        context.resources->model_manifest == nullptr) {
        throw std::invalid_argument(
            "dispatch context has no model manifest");
    }
    return *context.resources->model_manifest;
}

const ModelRuntimeBinding& model_weights(
    const ExecutionDispatchContext& context) {
    if (context.resources == nullptr ||
        context.resources->model_weights == nullptr) {
        throw std::invalid_argument(
            "dispatch context has no model weight binding");
    }
    return *context.resources->model_weights;
}

}  // namespace

ExecutionBufferAddress resolve_dispatch_read(
    const ExecutionDispatchContext& context,
    std::size_t index) {
    if (context.arena == nullptr) {
        throw std::invalid_argument(
            "dispatch context has no arena plan");
    }
    if (index >= context.reads.size()) {
        throw std::out_of_range(
            "dispatch read index is out of range");
    }

    const auto& memory =
        scratch_memory(context);
    if (context.prepared_arena_views)
        return context.prepared_arena_views->resolve(context, PreparedBufferKind::read, index, memory);
    return resolve_execution_buffer(
        memory.execution_identity,
        *context.arena,
        memory,
        context.reads[index]);
}

ExecutionBufferAddress resolve_dispatch_write(
    const ExecutionDispatchContext& context,
    std::size_t index) {
    if (context.arena == nullptr) {
        throw std::invalid_argument(
            "dispatch context has no arena plan");
    }
    if (index >= context.writes.size()) {
        throw std::out_of_range(
            "dispatch write index is out of range");
    }

    const auto& memory =
        scratch_memory(context);
    if (context.prepared_arena_views)
        return context.prepared_arena_views->resolve(context, PreparedBufferKind::write, index, memory);
    return resolve_execution_buffer(
        memory.execution_identity,
        *context.arena,
        memory,
        context.writes[index]);
}


ExecutionBufferAddress resolve_dispatch_temporary(
    const ExecutionDispatchContext& context,
    std::size_t index) {
    if (context.arena == nullptr) {
        throw std::invalid_argument(
            "dispatch context has no arena plan");
    }
    if (index >= context.temporaries.size()) {
        throw std::out_of_range(
            "dispatch temporary index is out of range");
    }

    const auto& memory =
        scratch_memory(context);
    if (context.prepared_arena_views)
        return context.prepared_arena_views->resolve(context, PreparedBufferKind::temporary, index, memory);
    return resolve_execution_buffer(
        memory.execution_identity,
        *context.arena,
        memory,
        context.temporaries[index]);
}

namespace {
ExecutionBufferAddress resolve_dispatch_hinted(const ExecutionDispatchContext& context, PreparedBufferKind kind,
    std::span<const ExecutionBufferUse> uses, std::size_t index, std::size_t& hint) {
    if (context.arena == nullptr) throw std::invalid_argument("dispatch context has no arena plan");
    if (index >= uses.size()) throw std::out_of_range("dispatch buffer index is out of range");
    const auto& memory = scratch_memory(context);
    if (context.prepared_arena_views) return context.prepared_arena_views->resolve(context, kind, index, memory);
    return resolve_execution_buffer_hinted(memory.execution_identity, *context.arena, memory, uses[index], hint);
}
}  // namespace

ExecutionBufferAddress resolve_dispatch_read_hinted(const ExecutionDispatchContext& context, std::size_t index, std::size_t& hint) {
    return resolve_dispatch_hinted(context, PreparedBufferKind::read, context.reads, index, hint);
}
ExecutionBufferAddress resolve_dispatch_write_hinted(const ExecutionDispatchContext& context, std::size_t index, std::size_t& hint) {
    return resolve_dispatch_hinted(context, PreparedBufferKind::write, context.writes, index, hint);
}
ExecutionBufferAddress resolve_dispatch_temporary_hinted(const ExecutionDispatchContext& context, std::size_t index, std::size_t& hint) {
    return resolve_dispatch_hinted(context, PreparedBufferKind::temporary, context.temporaries, index, hint);
}

SessionStateAddress resolve_dispatch_state(
    const ExecutionDispatchContext& context,
    SessionStateComponent component) {
    return resolve_session_state(
        state_layout(context),
        state_memory(context),
        component);
}



SessionStateAddress resolve_dispatch_gdn_layer_state(
    const ExecutionDispatchContext& context,
    const ModelContract& model,
    SessionStateComponent component) {
    if (!context.layer.has_value()) {
        throw std::invalid_argument(
            "GDN dispatch state resolution requires a layer id");
    }
    return resolve_gdn_layer_state(
        state_layout(context),
        state_memory(context),
        model,
        *context.layer,
        component);
}

const ExecutionTensorAccess& resolve_dispatch_tensor_access(
    const ExecutionDispatchContext& context,
    std::uint32_t binding_slot) {
    if (const auto* prepared = prepared_views(context))
        return prepared->access(context.execution_identity, context.step_id, binding_slot);
    if (context.resources == nullptr ||
        context.resources->tensor_access == nullptr) {
        throw std::invalid_argument(
            "dispatch context has no tensor access plan");
    }
    const auto* access =
        find_execution_tensor_access(
            *context.resources->tensor_access,
            context.step_id,
            binding_slot);
    if (access == nullptr) {
        throw std::invalid_argument(
            "dispatch tensor binding slot is not present for this step");
    }
    return *access;
}

const RuntimeTensorBinding& resolve_dispatch_tensor(
    const ExecutionDispatchContext& context,
    std::uint32_t tensor_index) {
    if (const auto* prepared = prepared_views(context))
        return prepared->tensor(context.execution_identity, tensor_index);
    const auto& manifest =
        model_manifest(context);
    const auto& weights =
        model_weights(context);
    if(context.resources->distributed_stage) {
        const auto& scope=*context.resources->distributed_stage;
        if(context.execution_identity!=scope.execution_identity() || !scope.contains_step(context.step_id) || !scope.requires_tensor(tensor_index))
            throw std::invalid_argument("dispatch tensor is outside the local stage scope");
        validate_stage_model_runtime_binding(scope,manifest,weights);
    } else validate_model_runtime_binding_unless_scoped(manifest, weights);

    const auto* tensor =
        find_runtime_tensor_binding(
            weights, tensor_index);
    if (tensor == nullptr) {
        throw std::out_of_range(
            "dispatch tensor index is out of range");
    }
    return *tensor;
}

const RuntimeTensorBinding& resolve_dispatch_tensor(
    const ExecutionDispatchContext& context,
    std::string_view tensor_name) {
    if (const auto* prepared = prepared_views(context))
        return prepared->tensor(context.execution_identity, tensor_name);
    const auto& manifest =
        model_manifest(context);
    const auto& weights =
        model_weights(context);

    const RuntimeTensorBinding* tensor=nullptr;
    if(context.resources->distributed_stage) {
        const auto found=std::find_if(manifest.tensors.begin(),manifest.tensors.end(),[&](const auto& t){return t.name==tensor_name;});
        if(found==manifest.tensors.end()) throw std::invalid_argument("dispatch stage tensor name is absent");
        return resolve_dispatch_tensor(context,static_cast<std::uint32_t>(found-manifest.tensors.begin()));
    }
    validate_model_runtime_binding_unless_scoped(manifest, weights);
    for (std::size_t index = 0; index < manifest.tensors.size() && !tensor; ++index)
        if (manifest.tensors[index].name == tensor_name) tensor = &weights.tensors[index];
    if (tensor == nullptr) {
        throw std::invalid_argument(
            "dispatch tensor name is not present in manifest");
    }
    return *tensor;
}

ValidatedModelBindingScope::ValidatedModelBindingScope(
    const ModelManifest& manifest, const ModelRuntimeBinding& binding)
    : previous_manifest_(validated_manifest), previous_binding_(validated_binding) {
    validate_model_runtime_binding(manifest, binding);
    validated_manifest = &manifest;
    validated_binding = &binding;
}

ValidatedModelBindingScope::~ValidatedModelBindingScope() {
    validated_manifest = previous_manifest_;
    validated_binding = previous_binding_;
}

void validate_model_runtime_binding_unless_scoped(
    const ModelManifest& manifest, const ModelRuntimeBinding& binding) {
    if (&manifest != validated_manifest || &binding != validated_binding)
        validate_model_runtime_binding(manifest, binding);
}

}  // namespace strix
