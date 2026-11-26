#pragma once

#include "strix/model_graph.hpp"
#include "glm_rank_plan.hpp"

#include <string>
#include <vector>

namespace strix::glm::load {

// rank_plan consumes arbitrary named split entries, not a compiled layer array.
// Compare the ordered graph with its tensor-name inventory (logical names or
// physical index components), independently of the adapter's declared cadence.
// Throws on missing, extra, conflicting or reordered layer families/counts.
void check_rank_plan_graph(const model_graph::ModelGraph& graph, const std::vector<std::string>& tensor_names);
inline void check_rank_plan_graph(const model_graph::ModelGraph& graph, const Split& split) {
    std::vector<std::string> names;
    for (const auto& [key, entry] : split.entries()) {
        (void)entry;
        names.push_back(key);
    }
    check_rank_plan_graph(graph, names);
}

}  // namespace strix::glm::load
