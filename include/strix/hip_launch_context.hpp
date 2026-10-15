#pragma once

#include "strix/graph_inputs.hpp"
#include <hip/hip_runtime_api.h>
#include <limits>
#include <stdexcept>

namespace strix {
// Host-only scope shared by the existing static launchers. One serialized
// scheduler owns the stream and every allocation through actual completion.
struct HipLaunchContext {
    hipStream_t stream = nullptr;
    bool graph_capture = false;
    GraphDynamicArguments arguments;
    ExecutionMemoryBinding status;
    std::uint64_t context_tokens = 0;
    std::uint32_t vocabulary_size = 0;
    // Optional stable device history inputs/outputs for resident-table PLE.
    ExecutionMemoryBinding ngram_input;
    ExecutionMemoryBinding ngram_output;
    // Separate eager first-error record; never selects graph-only kernel paths.
    ExecutionMemoryBinding deferred_moe_status;
};
inline thread_local const HipLaunchContext* active_hip_launch_context = nullptr;
[[nodiscard]] inline const HipLaunchContext* current_hip_launch_context() noexcept {
    return active_hip_launch_context;
}
[[nodiscard]] inline hipStream_t hip_execution_stream() noexcept {
    return active_hip_launch_context ? active_hip_launch_context->stream : nullptr;
}
[[nodiscard]] inline std::uint32_t* hip_graph_status() noexcept {
    return active_hip_launch_context && active_hip_launch_context->graph_capture
        ? reinterpret_cast<std::uint32_t*>(active_hip_launch_context->status.arena_base) : nullptr;
}
[[nodiscard]] inline const std::uint64_t* hip_graph_position() noexcept {
    return active_hip_launch_context && active_hip_launch_context->graph_capture
        ? reinterpret_cast<const std::uint64_t*>(active_hip_launch_context->arguments.memory.arena_base +
              active_hip_launch_context->arguments.position_offset) : nullptr;
}
[[nodiscard]] inline std::uint32_t* hip_deferred_moe_status() noexcept {
    return active_hip_launch_context && !active_hip_launch_context->graph_capture
        ? reinterpret_cast<std::uint32_t*>(active_hip_launch_context->deferred_moe_status.arena_base) : nullptr;
}
class HipLaunchScope {
public:
    explicit HipLaunchScope(const HipLaunchContext& context) : context_(context) {
        if (active_hip_launch_context || !context_.stream || context_.stream == hipStreamPerThread ||
            (context_.graph_capture && context_.stream == hipStreamLegacy)) {
            throw std::invalid_argument("HIP launch scope requires one explicitly owned stream");
        }
        if (context_.graph_capture) {
            const auto expected = bind_graph_inputs(graph_input_layout(1), context_.arguments.memory);
            if (context_.arguments != expected || !context_.context_tokens || !context_.vocabulary_size ||
                context_.status.execution_identity != context_.arguments.memory.execution_identity ||
                !context_.status.allocation_generation || !context_.status.arena_base ||
                context_.status.arena_base % alignof(std::uint32_t) || context_.status.arena_bytes != 4 ||
                context_.status.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - context_.status.arena_base) {
                throw std::invalid_argument("HIP graph scope has invalid dynamic inputs/status");
            }
            if (context_.status.arena_base < context_.arguments.memory.arena_base + context_.arguments.memory.arena_bytes &&
                context_.arguments.memory.arena_base < context_.status.arena_base + context_.status.arena_bytes) {
                throw std::invalid_argument("HIP graph scope arguments and status alias");
            }
        }
        const auto& deferred = context_.deferred_moe_status;
        const bool has_deferred = deferred.arena_base || deferred.arena_bytes ||
            deferred.execution_identity || deferred.allocation_generation;
        if (has_deferred &&
            (context_.graph_capture || !deferred.arena_base || context_.deferred_moe_status.execution_identity == 0 ||
             context_.deferred_moe_status.allocation_generation == 0 ||
             context_.deferred_moe_status.arena_base % alignof(std::uint32_t) != 0 ||
             context_.deferred_moe_status.arena_bytes != 8 ||
             context_.deferred_moe_status.arena_base > std::numeric_limits<std::uintptr_t>::max()-8U))
            throw std::invalid_argument("deferred MoE scope requires a distinct bounded eager first-error record");
        active_hip_launch_context = &context_;
    }
    ~HipLaunchScope() { active_hip_launch_context = nullptr; }
    HipLaunchScope(const HipLaunchScope&) = delete;
    HipLaunchScope& operator=(const HipLaunchScope&) = delete;
private:
    HipLaunchContext context_;
};
} // namespace strix
