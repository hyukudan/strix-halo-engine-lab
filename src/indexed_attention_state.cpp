#include "strix/indexed_attention_state.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() /
                left) {
        throw std::overflow_error(
            "indexed-attention state multiplication overflow");
    }
    return left * right;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() -
            left) {
        throw std::overflow_error(
            "indexed-attention state addition overflow");
    }
    return left + right;
}

}  // namespace

IndexedAttentionStateLayout
build_indexed_attention_state_layout(
    const ModelContract& model) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }

    IndexedAttentionStateLayout layout;
    const auto kv_width =
        checked_mul(
            model.kv_heads,
            model.head_dim);
    layout.kv_key_bytes =
        checked_mul(
            kv_width,
            model.bytes_bf16);
    layout.kv_value_offset =
        layout.kv_key_bytes;
    layout.kv_value_bytes =
        layout.kv_key_bytes;
    layout.kv_row_bytes =
        checked_add(
            layout.kv_key_bytes,
            layout.kv_value_bytes);
    layout.raw_index_row_bytes =
        checked_mul(
            checked_mul(
                model.indexer_kv_heads,
                model.indexer_head_dim),
            model.bytes_bf16);
    return layout;
}

}  // namespace strix
