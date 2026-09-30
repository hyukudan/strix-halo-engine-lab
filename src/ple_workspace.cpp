#include "strix/ple_workspace.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "PLE workspace addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "PLE workspace multiplication overflow");
    }
    return left * right;
}

std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "PLE workspace alignment must be non-zero");
    }
    const auto remainder =
        value % alignment;
    return remainder == 0
        ? value
        : checked_add(
              value,
              alignment - remainder);
}

void append(
    std::uint64_t& cursor,
    std::uint64_t bytes,
    std::uint64_t alignment,
    std::uint64_t& offset,
    std::uint64_t& recorded_bytes) {
    if (bytes == 0) {
        throw std::invalid_argument(
            "PLE workspace component must be non-zero");
    }
    cursor =
        align_up(
            cursor,
            alignment);
    offset = cursor;
    recorded_bytes = bytes;
    cursor =
        checked_add(
            cursor,
            bytes);
}

}  // namespace

PleWorkspaceLayout build_ple_workspace_layout(
    const ModelContract& model,
    std::uint64_t alignment) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (alignment == 0) {
        throw std::invalid_argument(
            "PLE workspace alignment must be non-zero");
    }

    const auto hc_hidden =
        checked_mul(
            model.hidden_size,
            model.hc_count);

    PleWorkspaceLayout layout;
    layout.alignment =
        alignment;

    std::uint64_t cursor = 0;

    append(
        cursor,
        checked_mul(
            model.ple_embed_dim,
            model.bytes_bf16),
        alignment,
        layout.gathered_offset,
        layout.gathered_bytes);

    append(
        cursor,
        checked_mul(
            hc_hidden,
            model.bytes_fp32),
        alignment,
        layout.key_offset,
        layout.key_bytes);

    append(
        cursor,
        checked_mul(
            model.hidden_size,
            model.bytes_fp32),
        alignment,
        layout.value_offset,
        layout.value_bytes);

    append(
        cursor,
        checked_mul(
            hc_hidden,
            model.bytes_fp32),
        alignment,
        layout.gated_offset,
        layout.gated_bytes);

    append(
        cursor,
        checked_mul(
            hc_hidden,
            model.bytes_fp32),
        alignment,
        layout.normalized_offset,
        layout.normalized_bytes);

    append(
        cursor,
        checked_mul(
            model.hc_count,
            model.bytes_fp32),
        alignment,
        layout.gate_offset,
        layout.gate_bytes);

    layout.total_bytes =
        align_up(
            cursor,
            alignment);
    return layout;
}

}  // namespace strix
