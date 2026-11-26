#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace strix::attn::fp8 {

// Head-major E4M3 K, transposed E4M3 V, one FP32 scale per head and token for each.
// QK and PV use E4M3 operands and accumulate in FP32 around an FP32 online softmax.
inline constexpr std::uint32_t head_dim = 256;
inline constexpr std::uint32_t capacity_alignment = 64;
inline constexpr std::uint32_t split_keys = 2048;
inline constexpr std::uint32_t max_group = 8;

// The aligned shadow allocation may include stale positions. Limit the tile
// before reading its bytes or scales, and use the causal end for per-query work.
[[nodiscard]]
#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
constexpr std::uint32_t live_tile_keys(std::uint32_t first, std::uint32_t end) noexcept {
    if (first >= end) return 0;
    const auto remaining = end - first;
    return remaining < capacity_alignment ? remaining : capacity_alignment;
}

struct Plan {
    std::uint32_t capacity, padded_capacity, heads, kv_heads, group, splits;

    [[nodiscard]] constexpr bool covers(std::uint32_t first, std::uint32_t rows) const {
        return first <= capacity && rows != 0 && rows <= capacity - first;
    }
    [[nodiscard]] constexpr std::size_t shadow_bytes() const {
        return std::size_t{kv_heads} * padded_capacity * (2U * head_dim + 2U * sizeof(float));
    }
    [[nodiscard]] constexpr std::size_t scratch_floats(std::uint32_t rows) const {
        const std::size_t per_row = std::size_t{heads} * splits * (head_dim + 2U);
        return rows > std::numeric_limits<std::size_t>::max() / sizeof(float) / per_row ? 0 : rows * per_row;
    }
};

[[nodiscard]] constexpr std::optional<Plan> make_plan(std::uint32_t width, std::uint32_t heads, std::uint32_t kv_heads,
                                                      std::uint32_t capacity) {
    constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();
    if (width != head_dim || heads == 0 || heads > 65535U || kv_heads == 0 || heads % kv_heads != 0 ||
        heads / kv_heads > max_group || capacity == 0 ||
        capacity > std::numeric_limits<std::uint32_t>::max() - (split_keys - 1U)) return std::nullopt;
    const auto padded = (capacity + capacity_alignment - 1U) / capacity_alignment * capacity_alignment;
    const auto splits = (capacity - 1U) / split_keys + 1U;
    if (kv_heads > max_bytes / padded / (2U * head_dim + 2U * sizeof(float)) ||
        heads > max_bytes / splits / (head_dim + 2U)) return std::nullopt;
    return Plan{capacity, padded, heads, kv_heads, heads / kv_heads, splits};
}

} // namespace strix::attn::fp8
