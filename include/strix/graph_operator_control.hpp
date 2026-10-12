#pragma once

#include "strix/graph_inputs.hpp"

#include <cstdint>
#include <vector>

namespace strix {

// Portable preflight consumed by device-dynamic operator launchers. Actual
// allocation ownership, stream completion and state publication remain external.
struct GraphOperatorResourceView {
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t allocation_generation = 0;
    std::uint32_t alignment = 1;
};
struct GraphOperatorControlRequest {
    std::uint64_t execution_identity = 0;
    std::uint64_t context_tokens = 0;
    GraphDynamicArguments arguments;
    ExecutionMemoryBinding status;
    std::vector<GraphOperatorResourceView> resources;
};
struct GraphOperatorControlPlan {
    GraphOperatorControlRequest request;
    std::uint64_t identity = 0;
};
[[nodiscard]] GraphOperatorControlPlan prepare_graph_operator_control(
    const GraphOperatorControlRequest& request);
void validate_graph_operator_control(const GraphOperatorControlPlan& plan);

} // namespace strix
