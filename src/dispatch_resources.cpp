#include "strix/dispatch_resources.hpp"
#include "strix/prepared_tensor_views.hpp"

#include <stdexcept>

namespace strix {
namespace {

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
    return resolve_execution_buffer(
        memory.execution_identity,
        *context.arena,
        memory,
        context.temporaries[index]);
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
    validate_model_runtime_binding(
        manifest, weights);

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

    const auto* tensor =
        find_runtime_tensor_binding(
            manifest,
            weights,
            tensor_name);
    if (tensor == nullptr) {
        throw std::invalid_argument(
            "dispatch tensor name is not present in manifest");
    }
    return *tensor;
}

}  // namespace strix
