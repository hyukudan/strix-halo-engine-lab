#include "strix/graph_eligibility.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {
std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("graph instance accounting overflow");
    }
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("graph byte accounting overflow");
    }
    return a + b;
}
void hash_word(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 255ULL;
        hash *= 1099511628211ULL;
    }
}
std::uint64_t identity(const GraphCapturePlan& p) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto& a = p.arguments;
    const auto& m = p.memory;
    for (const auto value : {p.execution.execution_identity, p.execution.kernel_plan_identity,
         static_cast<std::uint64_t>(p.execution.logical_rows), static_cast<std::uint64_t>(p.execution.whole_plan_graph_safe),
         p.compatibility_identity, p.binding_identity, static_cast<std::uint64_t>(a.mode),
         a.memory.execution_identity, static_cast<std::uint64_t>(a.memory.arena_base), a.memory.arena_bytes,
         a.memory.allocation_generation, a.token_offset, a.position_offset, a.device_argument_abi_identity,
         m.sessions, m.shapes_per_session, m.eligibility_classes_per_shape, m.workspace_bytes_per_instance,
         m.executable_bytes_per_instance, m.capture_peak_bytes, m.budget_bytes, p.instance_count, p.charged_bytes}) {
        hash_word(hash, value);
    }
    if (p.execution.rotary_table) {
        hash = append_rotary_identity(hash, p.execution.rotary_table->identity());
        hash_word(hash, reinterpret_cast<std::uintptr_t>(p.execution.rotary_table->data()));
        hash_word(hash, p.execution.rotary_table->generation());
        hash_word(hash, static_cast<std::uint64_t>(p.execution.rotary_table->device()));
    }
    return hash;
}
void validate_arguments(const GraphExecutionContract& execution, const GraphDynamicArguments& a) {
    if (a.mode != GraphArgumentMode::device_data || a.device_argument_abi_identity == 0) {
        throw std::invalid_argument("graph token IDs/absolute positions require explicit device-data ABI; baked scalars are not dynamic");
    }
    if (a.memory.execution_identity != execution.execution_identity || a.memory.arena_base == 0 ||
        a.memory.allocation_generation == 0 || a.memory.arena_bytes == 0) {
        throw std::invalid_argument("graph device arguments require exact stable allocation binding");
    }
    if (a.memory.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - a.memory.arena_base) {
        throw std::overflow_error("graph argument address range overflow");
    }
    const auto token_bytes = multiply(execution.logical_rows, sizeof(std::uint32_t));
    const auto position_bytes = multiply(execution.logical_rows, sizeof(std::uint64_t));
    if (a.token_offset > a.memory.arena_bytes || token_bytes > a.memory.arena_bytes - a.token_offset ||
        a.position_offset > a.memory.arena_bytes || position_bytes > a.memory.arena_bytes - a.position_offset ||
        (a.memory.arena_base + a.token_offset) % alignof(std::uint32_t) != 0 ||
        (a.memory.arena_base + a.position_offset) % alignof(std::uint64_t) != 0 ||
        (a.token_offset < a.position_offset + position_bytes && a.position_offset < a.token_offset + token_bytes)) {
        throw std::invalid_argument("graph token/position device ranges must be bounded aligned and disjoint");
    }
}
void validate_resident(const ExecutionRuntimeResources& resources) {
    // GPU visibility of mapped storage does not prove invariant residency or
    // remove host-side PLE/page/expert decisions from a captured graph.
    if (resources.ple_lookup != nullptr || resources.ple_prefetch != nullptr) {
        throw std::invalid_argument("graph cannot omit live host PLE/residency decisions");
    }
    for (const auto& tensor : resources.model_weights->tensors) {
        if (tensor.storage != RuntimeTensorStorage::resident &&
            tensor.storage != RuntimeTensorStorage::repacked_cache) {
            throw std::invalid_argument("graph assumes local resident or stable repacked weights; mapped/remote storage is ineligible");
        }
    }
}
} // namespace

GraphCapturePlan prepare_graph_capture_plan(
    GraphExecutionContract execution, const ExecutionArenaPlan& arena,
    const ExecutionRuntimeResources& resources, const GraphDynamicArguments& arguments,
    const GraphMemoryRequest& memory) {
    if (execution.execution_identity == 0 || execution.kernel_plan_identity == 0 ||
        execution.logical_rows == 0 || !execution.whole_plan_graph_safe) {
        throw std::invalid_argument("compiled graph contains unresolved dynamic/host/unsafe steps");
    }
    const auto runtime = build_runtime_execution_identities(execution.execution_identity, arena, resources);
    validate_resident(resources);
    validate_arguments(execution, arguments);
    if (memory.sessions == 0 || memory.shapes_per_session == 0 || memory.eligibility_classes_per_shape == 0 ||
        memory.executable_bytes_per_instance == 0 || memory.budget_bytes == 0) {
        throw std::invalid_argument("graph requires explicit bounded instances/executable memory/budget");
    }
    GraphCapturePlan p;
    p.execution = execution;
    p.compatibility_identity = runtime.compatibility_identity;
    p.binding_identity = runtime.binding_identity;
    p.arguments = arguments;
    p.memory = memory;
    p.instance_count = multiply(multiply(memory.sessions, memory.shapes_per_session), memory.eligibility_classes_per_shape);
    p.charged_bytes = add(multiply(p.instance_count,
        add(memory.workspace_bytes_per_instance, memory.executable_bytes_per_instance)), memory.capture_peak_bytes);
    // Dynamic input allocation is retained alongside all graph instances.
    p.charged_bytes = add(p.charged_bytes, multiply(p.instance_count, arguments.memory.arena_bytes));
    if (p.charged_bytes > memory.budget_bytes) {
        throw std::invalid_argument("graph instances/workspace/executable/capture/input memory exceed graph budget");
    }
    p.capture_identity = identity(p);
    if (p.capture_identity == 0) { throw std::logic_error("zero graph capture identity"); }
    return p;
}

GraphEligibility assess_graph_replay_eligibility(
    const GraphCapturePlan& capture, GraphExecutionContract execution,
    const ExecutionArenaPlan& arena, const ExecutionRuntimeResources& current_resources,
    const GraphDynamicArguments& current_arguments, std::uint64_t position) {
    GraphEligibility result;
    result.instance_count = capture.instance_count;
    result.charged_bytes = capture.charged_bytes;
    if (capture.capture_identity == 0 || capture.capture_identity != identity(capture)) {
        result.reasons.push_back("capture metadata integrity mismatch");
        return result;
    }
    if (capture.execution != execution) { result.reasons.push_back("compiled execution/shape/graph-safety contract changed"); }
    if (capture.arguments != current_arguments) { result.reasons.push_back("device token/position argument addresses/generations/ABI changed"); }
    try {
        if (execution.rotary_table) {
            execution.rotary_table->check_position(position);
            if (execution.logical_rows > execution.rotary_table->extent() - position)
                throw std::out_of_range("graph rows exceed rotary table extent");
        }
        validate_execution_runtime_resources(execution.execution_identity, arena, current_resources, position);
        if (execution.logical_rows == 0 || execution.logical_rows > current_resources.session_state_layout->context_tokens - position) {
            throw std::out_of_range("graph row window exceeds current context allocation");
        }
        const auto expected = prepare_graph_capture_plan(execution, arena, current_resources, current_arguments, capture.memory);
        if (capture.compatibility_identity != expected.compatibility_identity || capture.binding_identity != expected.binding_identity) {
            result.reasons.push_back("current model/state/scratch binding or residency generation differs from captured resources");
        }
        if (capture.capture_identity != expected.capture_identity) {
            result.reasons.push_back("current graph instance contract differs from capture");
        }
    } catch (const std::exception& error) { result.reasons.push_back(error.what()); }
    result.candidate_eligible = result.reasons.empty();
    return result;
}
} // namespace strix
