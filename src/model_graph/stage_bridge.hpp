#pragma once
// Description -> stage graph: the model description (ModelGraph, layers in trunk order) gives the per-layer trait words a
// stage declaration is guarded by. The declaration tables and format rules stay the model's data; this only binds them
// to the described layers and validates the result. Families without a stage trait yet throw (added as they migrate).
#include "model_graph.hpp"
#include "strix/graph/stage_graph.hpp"

#include <vector>

namespace strix::model_graph {

// attention dense -> attn_dense, gdn -> attn_gdn, compressed_indexed -> attn_indexed, mla -> attn_window when the layer
// declares dimensions["window"] > 0 else attn_indexed; MLP dense -> ffn_dense, moe -> ffn_moe (a shared expert rides on
// moe). Other attention families throw std::invalid_argument.
[[nodiscard]] std::vector<graph::Traits> stage_traits(const ModelGraph& graph);
// The declaration with its layer words taken from the description, validated (graph::validate).
[[nodiscard]] graph::StageGraph to_stage_graph(const ModelGraph& graph, graph::StageGraph declaration);

}  // namespace strix::model_graph
