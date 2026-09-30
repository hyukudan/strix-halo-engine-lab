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

    if (failures == 0) {
        std::cout
            << "indexed-attention workspace tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
