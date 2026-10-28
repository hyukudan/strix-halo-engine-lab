#pragma once

#include "strix/execution_resources.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

enum class GraphArgumentMode { baked_scalars, device_data };

struct GraphDynamicArguments {
    GraphArgumentMode mode = GraphArgumentMode::baked_scalars;
    // A separately owned stable device allocation, bound by exact generation.
    ExecutionMemoryBinding memory;
    std::uint64_t token_offset = 0;
    std::uint64_t position_offset = 0;
    std::uint64_t device_argument_abi_identity = 0;
    friend bool operator==(const GraphDynamicArguments& a, const GraphDynamicArguments& b) {
        return a.mode == b.mode && a.memory.execution_identity == b.memory.execution_identity &&
            a.memory.arena_base == b.memory.arena_base && a.memory.arena_bytes == b.memory.arena_bytes &&
            a.memory.allocation_generation == b.memory.allocation_generation && a.token_offset == b.token_offset &&
            a.position_offset == b.position_offset && a.device_argument_abi_identity == b.device_argument_abi_identity;
    }
};

struct GraphMemoryRequest {
    std::uint64_t sessions = 1;
    std::uint64_t shapes_per_session = 1;
    std::uint64_t eligibility_classes_per_shape = 1;
    std::uint64_t workspace_bytes_per_instance = 0;
    std::uint64_t executable_bytes_per_instance = 0;
    std::uint64_t capture_peak_bytes = 0;
    std::uint64_t budget_bytes = 0;
    friend bool operator==(const GraphMemoryRequest&, const GraphMemoryRequest&) = default;
};

struct GraphExecutionContract {
    std::uint64_t execution_identity = 0;
    std::uint64_t kernel_plan_identity = 0;
    std::uint32_t logical_rows = 1;
    bool whole_plan_graph_safe = false;
    std::shared_ptr<const HipQwenRotaryTable> rotary_table = nullptr;
    friend bool operator==(const GraphExecutionContract&, const GraphExecutionContract&) = default;
};

struct GraphCapturePlan {
    GraphExecutionContract execution;
    std::uint64_t compatibility_identity = 0;
    std::uint64_t binding_identity = 0;
    GraphDynamicArguments arguments;
    GraphMemoryRequest memory;
    std::uint64_t instance_count = 0;
    std::uint64_t charged_bytes = 0;
    std::uint64_t capture_identity = 0;
};

struct GraphEligibility {
    bool candidate_eligible = false;
    // Portable eligibility never promotes a native replay implementation.
    bool native_replay_qualified = false;
    std::uint64_t instance_count = 0;
    std::uint64_t charged_bytes = 0;
    std::vector<std::string> reasons;
};

[[nodiscard]] GraphCapturePlan prepare_graph_capture_plan(
    GraphExecutionContract execution, const ExecutionArenaPlan& arena,
    const ExecutionRuntimeResources& resources, const GraphDynamicArguments& arguments,
    const GraphMemoryRequest& memory);

[[nodiscard]] GraphEligibility assess_graph_replay_eligibility(
    const GraphCapturePlan& capture, GraphExecutionContract execution,
    const ExecutionArenaPlan& arena, const ExecutionRuntimeResources& current_resources,
    const GraphDynamicArguments& current_arguments, std::uint64_t position);

} // namespace strix
