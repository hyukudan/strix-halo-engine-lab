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

    expect(strix::block_scaled_fp8_row_bytes(512, 128) == 528, "block-scaled FP8 row: codes + F32 scale per block");
    const auto rows = strix::compressed_history_rows(1000, 128, 4);
    expect(rows.history_rows == 128 && rows.compressed_rows == 250, "compressed history rows: window + ceil(ctx/ratio)");
    expect(strix::compressed_history_rows(50, 128, 4).history_rows == 50 &&
               strix::compressed_history_rows(51, 128, 4).compressed_rows == 13,
           "compressed history rows: short context, rounding up");
    expect(strix::compressed_history_carry_bytes(4, 512) == 4U * 512U * 8U, "carry: projection + gate in F32");
    constexpr strix::SparseLatentRowDims dims{512, 128, 64, 128, 32};
    static_assert(dims.valid());
    expect(strix::sparse_latent_row_bytes(dims) == 528 + 64 + 64 + 4, "sparse latent row: latent + rope + E2M1 key + scales");

    if (failures == 0) {
        std::cout
            << "indexed-attention state tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
