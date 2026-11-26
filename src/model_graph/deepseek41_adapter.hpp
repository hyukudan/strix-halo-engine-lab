#pragma once

#include "strix/model_graph.hpp"
#include "plan.hpp"
#include "strix/deepseek41_model.hpp"
#include "strix/deepseek41_plan.hpp"

#include <cstdint>

namespace strix::deepseek41 {
// Descriptive adapters only. The existing planner/executor remain authoritative.
[[nodiscard]] model_graph::ModelGraph model_graph_from_config(const Config& config);
[[nodiscard]] model_graph::ExecutionPlan common_decode_plan(
    const Plan& plan, const model_graph::ModelGraph& graph, std::uint32_t rank);
}  // namespace strix::deepseek41
