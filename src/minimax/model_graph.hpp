#pragma once

#include "strix/minimax/model_description.hpp"
#include "strix/model_graph.hpp"
#include "../model_graph/plan.hpp"
#include "engine/mm_defaults.hpp"

namespace strix::minimax::graph {

// Configured attachment inventory only; no paths, loader or drafter execution.
struct Attachments {
    bool draft = false, draft_linears = false;
    std::uint32_t draft_heads = 0;
};
[[nodiscard]] model_graph::ModelGraph build_model_graph(
    const model_description::ModelDescription& description, Attachments attachments = {});
void assert_matches(const model_description::ModelDescription& description,
                    const model_graph::ModelGraph& graph, Attachments attachments = {});
// One logical decode row. Describes layers_range/stage_attn/stage_ffn and the
// pure defaults resolver, not a HIP launch plan or a claim of graph capture safety.
[[nodiscard]] model_graph::ExecutionPlan common_decode_plan(
    const model_description::ModelDescription& description, const model_graph::ModelGraph& graph,
    std::uint32_t rank, const defaults::Lookup& environment, Attachments attachments = {});
[[nodiscard]] bool requested(const char* environment);
// No sidecar I/O unless printing or STRIX_MINIMAX_MODEL_GRAPH_CHECK=1. Prints
// exit before transport/HIP; a check alone validates and continues execution.
bool inspect_or_check(const std::string& directory, std::uint32_t rank, Attachments attachments,
                      bool print_graph, bool print_plan);

} // namespace strix::minimax::graph
