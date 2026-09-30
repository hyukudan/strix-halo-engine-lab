#include "strix/indexed_attention_state.hpp"

#include "strix/memory_plan.hpp"

#include <iostream>

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
        strix::build_indexed_attention_state_layout(
            model);

    expect(
        layout.kv_key_offset == 0 &&
            layout.kv_key_bytes == 1024 &&
            layout.kv_value_offset == 1024 &&
            layout.kv_value_bytes == 1024 &&
            layout.kv_row_bytes == 2048,
        "indexed KV row is BF16 K then V for two 256-wide heads");
    expect(
        layout.raw_index_row_bytes == 256,
        "indexer state stores one BF16 128-wide raw key per token");

    expect(
        layout.kv_row_bytes ==
            strix::base_kv_bytes_per_token(model) /
                model.full_attention_layers,
        "indexed KV row agrees with memory planner");
    expect(
        layout.raw_index_row_bytes ==
            strix::indexer_bytes_per_token(model) /
                model.full_attention_layers,
        "raw index row agrees with memory planner");

    if (failures == 0) {
        std::cout
            << "indexed-attention state tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
