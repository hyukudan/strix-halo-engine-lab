#include "strix/indexed_attention_workspace.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto layout =
        strix::build_indexed_attention_workspace_layout(
            model);

    expect(
        layout.maximum_blocks == 65'536,
        "QSA workspace covers every four-token block at 262K context");
    expect(
        layout.maximum_selected_tokens == 2'051,
        "QSA selected-token capacity includes the three-token tail");
    expect(
        layout.index_query_bytes == 2'048 &&
        layout.current_index_key_bytes == 512,
        "QSA workspace sizes current index query/key vectors");
    expect(
        layout.block_scores_bytes == 256ULL * 1024ULL,
        "QSA workspace stores one FP32 score per maximum block");
    expect(
        layout.selected_indices_bytes == 8'204,
        "QSA workspace stores at most budget plus tail uint32 indices");
    expect(
        layout.q_gate_bytes == 49'152 &&
        layout.current_k_bytes == 2'048 &&
        layout.current_v_bytes == 2'048,
        "main attention projection workspace matches pinned heads");
    expect(
        layout.attention_scores_bytes ==
            24ULL * 2'051ULL * 4ULL &&
        layout.attention_output_bytes ==
            24ULL * 256ULL * 4ULL,
        "sparse attention workspace sizes FP32 score and head-output buffers");
    expect(
        layout.total_bytes < 600ULL * 1024ULL,
        "maximum-context QSA scratch remains below 600 KiB");

    const auto second =
        strix::build_indexed_attention_workspace_layout(
            model);
    expect(
        second.total_bytes ==
            layout.total_bytes &&
        second.attention_scores_offset ==
            layout.attention_scores_offset,
        "indexed-attention workspace layout is deterministic");

    bool zero_alignment = false;
    try {
        static_cast<void>(
            strix::build_indexed_attention_workspace_layout(
                model,
                0));
    } catch (const std::invalid_argument&) {
        zero_alignment = true;
    }
    expect(
        zero_alignment,
        "indexed-attention workspace rejects zero alignment");

    const auto hierarchical = strix::build_indexed_attention_workspace_layout(model, 256, true);
    expect(hierarchical.index_query_offset == layout.index_query_offset &&
        hierarchical.attention_output_offset == layout.attention_output_offset &&
        hierarchical.selection_ping_offset >= layout.total_bytes &&
        hierarchical.selection_ping_bytes == 131072 && hierarchical.selection_pong_bytes == 131072 &&
        hierarchical.selection_status_bytes == 4 && hierarchical.selection_status_offset % 256 == 0 &&
        hierarchical.selection_pong_offset >= hierarchical.selection_ping_offset + hierarchical.selection_ping_bytes &&
        hierarchical.selection_status_offset >= hierarchical.selection_pong_offset + hierarchical.selection_pong_bytes,
        "opt-in selection scratch appends bounded aligned regions without moving incumbent views");
    for (const auto heads : {2U, 4U}) {
        const auto grouped = strix::build_indexed_attention_workspace_layout(model, 256, false, heads);
        expect(grouped.attention_output_offset == layout.attention_output_offset && grouped.attention_position_bytes == 8 &&
            grouped.selection_status_bytes == 4 && grouped.selection_ping_bytes == 0 && grouped.selection_pong_bytes == 0,
            "grouped attention appends only bounded control storage without moving incumbent views");
        const auto both = strix::build_indexed_attention_workspace_layout(model, 256, true, heads);
        expect(both.selection_ping_bytes == hierarchical.selection_ping_bytes && both.attention_position_bytes == 8 &&
            both.selection_status_offset >= both.attention_position_offset + both.attention_position_bytes,
            "hierarchical selection and grouped attention share one disjoint local status");
    }
    const auto explicit_off = strix::build_indexed_attention_workspace_layout(model, 256, false);
    expect(explicit_off.total_bytes == layout.total_bytes && explicit_off.selection_status_bytes == 0,
        "OFF workspace bytes remain identical");
    for (const auto alignment : {1ULL, 4ULL}) {
        bool rejected = false;
        try { (void)strix::build_indexed_attention_workspace_layout(model, alignment, true); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "hierarchical workspace rejects Item-misaligned layouts");
    }
    auto wrong_budget = model; wrong_budget.indexer_budget = 1024;
    bool wrong = false;
    try { (void)strix::build_indexed_attention_workspace_layout(wrong_budget, 256, true); }
    catch (const std::invalid_argument&) { wrong = true; }
    expect(wrong, "hierarchical workspace rejects a different selection contract");
    if (failures == 0) {
        std::cout
            << "indexed-attention workspace tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
