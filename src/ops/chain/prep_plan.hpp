#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_CHAIN_PREP_HD __host__ __device__
#else
#define STRIX_CHAIN_PREP_HD
#endif

namespace strix::ops::chain {

inline constexpr unsigned kPrepLanes = 32;
inline constexpr unsigned kPrepFeaturesPerLane = 4;
inline constexpr unsigned kPrepWidth = kPrepLanes * kPrepFeaturesPerLane;

struct PrepShape {
    unsigned rows;
    unsigned query_heads, key_heads, index_query_heads, index_key_heads;
    unsigned rotary_half;
    unsigned rotary_positions;
};

struct PrepSequence {
    unsigned first_row, rows, first_position, capacity;
};

enum class PrepPlane { query, key_value, index_query, index_key };

struct PrepHead {
    PrepPlane plane;
    unsigned index;
};

struct PrepPlan {
    PrepShape shape;
    unsigned heads;
    unsigned layer;
    std::size_t index_layer_stride;  // BF16 elements, including any padding

    // A key warp also copies the corresponding value head.
    [[nodiscard]] STRIX_CHAIN_PREP_HD constexpr PrepHead head(unsigned ordinal) const {
        if (ordinal < shape.query_heads) return {PrepPlane::query, ordinal};
        ordinal -= shape.query_heads;
        if (ordinal < shape.key_heads) return {PrepPlane::key_value, ordinal};
        ordinal -= shape.key_heads;
        if (ordinal < shape.index_query_heads) return {PrepPlane::index_query, ordinal};
        return {PrepPlane::index_key, ordinal - shape.index_query_heads};
    }

    [[nodiscard]] STRIX_CHAIN_PREP_HD static constexpr unsigned position(const PrepSequence& seq, unsigned row) {
        return seq.first_position + (row - seq.first_row);
    }

    [[nodiscard]] STRIX_CHAIN_PREP_HD static constexpr std::size_t row_head(
        std::size_t row, unsigned row_heads, unsigned head_index) {
        return (row * row_heads + head_index) * kPrepWidth;
    }

    [[nodiscard]] STRIX_CHAIN_PREP_HD constexpr std::size_t rotary_offset(unsigned position) const {
        return std::size_t{position} * 2U * shape.rotary_half;
    }

    template <bool Index>
    [[nodiscard]] STRIX_CHAIN_PREP_HD constexpr std::size_t cache_head(
        const PrepSequence& seq, unsigned position, unsigned head_index) const {
        const auto row_heads = Index ? shape.index_key_heads : shape.key_heads;
        const auto stride = Index ? index_layer_stride : row_head(seq.capacity, row_heads, 0);
        return layer * stride + row_head(position, row_heads, head_index);
    }
};

// The same ordered sequences describe the host plan and the device bindings.
// The caller supplies a device row-to-sequence map for these disjoint row ranges.
// A sequence's K/V layer spans its capacity. Its index layer uses the explicit stride.
[[nodiscard]] constexpr std::optional<PrepPlan> plan_prep(
    PrepShape shape, std::span<const PrepSequence> sequences, unsigned layer, std::size_t index_layer_stride) {
    const auto heads = std::uint64_t{shape.query_heads} + shape.key_heads +
                       shape.index_query_heads + shape.index_key_heads;
    if (shape.rows == 0) return sequences.empty() ? std::optional{PrepPlan{shape, 0, layer, index_layer_stride}}
                                                  : std::nullopt;
    if (shape.rows > 0x7fffffffU || heads > 65535U || shape.query_heads == 0 || shape.key_heads == 0 ||
        shape.rotary_half == 0 || shape.rotary_half > kPrepWidth / 2U ||
        shape.rotary_half % kPrepFeaturesPerLane != 0) return std::nullopt;

    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_bf16 = std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
    // Every plane is no wider than all heads combined.
    if (shape.rows > max_floats / (heads * kPrepWidth) ||
        shape.rotary_positions > max_floats / (2U * shape.rotary_half)) return std::nullopt;

    const auto kv_width = std::size_t{shape.key_heads} * kPrepWidth;
    const auto index_width = std::size_t{shape.index_key_heads} * kPrepWidth;
    unsigned next = 0;
    for (const auto& seq : sequences) {
        if (seq.first_row != next || seq.rows == 0 || seq.rows > shape.rows - next ||
            seq.first_position >= seq.capacity || seq.rows > seq.capacity - seq.first_position ||
            seq.first_position >= shape.rotary_positions ||
            seq.rows > shape.rotary_positions - seq.first_position) return std::nullopt;
        if (seq.capacity > max_bf16 / kv_width / (std::uint64_t{layer} + 1U)) return std::nullopt;
        if (index_width != 0) {
            const auto index_rows = seq.first_position + seq.rows;
            if (index_rows > index_layer_stride / index_width) return std::nullopt;
            const auto index_extent = index_rows * index_width;
            if (index_extent > max_bf16 || layer > (max_bf16 - index_extent) / index_layer_stride)
                return std::nullopt;
        }
        next += seq.rows;
    }
    if (next != shape.rows) return std::nullopt;
    return PrepPlan{shape, static_cast<unsigned>(heads), layer, index_layer_stride};
}

}  // namespace strix::ops::chain

#undef STRIX_CHAIN_PREP_HD
