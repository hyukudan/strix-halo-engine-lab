#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

struct IndexedAttentionStateLayout {
    std::uint64_t kv_key_offset = 0;
    std::uint64_t kv_key_bytes = 0;
    std::uint64_t kv_value_offset = 0;
    std::uint64_t kv_value_bytes = 0;
    std::uint64_t kv_row_bytes = 0;
    std::uint64_t raw_index_row_bytes = 0;
};

[[nodiscard]] IndexedAttentionStateLayout
build_indexed_attention_state_layout(
    const ModelContract& model);

}  // namespace strix
