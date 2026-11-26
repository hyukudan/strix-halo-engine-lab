#pragma once
// Split-K decode plan of a dense attention whose K / V come from an FP32 master or a reduced shadow (backend-neutral
// data; HIP reader trait src/attn_core/shadow_kv_read.hpp, contract strix/attn/cache_shadow.hpp). An engine declares one
// plan per attention geometry; the kernel takes every count from it:
//   per (row, KV head, `tile` absolute key positions) one block of `threads` lanes computes, for the `group` query heads of
//   the KV head, the FP32 maximum, the FP32 exponential sum and the FP32 value sums (partial_floats() floats), then an
//   ordered merge over the tiles in position order. The tiling depends only on positions, never on the number of rows.
#include "strix/attn/cache_shadow.hpp"

#include <cstdint>

namespace strix::attn {

struct SplitDecodePlan {
    std::uint32_t dim = 0;            // head dimension
    std::uint32_t tile = 0;           // keys per split block (absolute positions)
    std::uint32_t threads = 0;        // lanes per split block
    std::uint32_t group = 0;          // query heads per KV head
    std::uint32_t heads = 0;          // query heads
    std::uint32_t kv_heads = 0;       // KV heads
    std::uint32_t max_positions = 0;  // admitted context

    [[nodiscard]] constexpr bool valid() const {
        return dim && dim % 32U == 0 && tile && tile % 64U == 0 && (tile / 4U) % 4U == 0 && threads && threads % 32U == 0 &&
               group && heads == group * kv_heads && max_positions;
    }
    // Per (row, KV head, tile): maxima[group], denominators[group], values[group][dim].
    [[nodiscard]] constexpr std::uint32_t partial_floats() const { return 2U * group + group * dim; }
    [[nodiscard]] constexpr std::uint64_t tiles(std::uint64_t stride) const { return (stride + tile - 1U) / tile; }
    [[nodiscard]] constexpr std::uint32_t max_tiles() const { return std::uint32_t(tiles(max_positions)); }
    // Scratch of `rows` rows at attention stride `stride`: the normed queries plus every tile's partials.
    [[nodiscard]] constexpr std::uint64_t scratch_bytes(std::uint32_t rows, std::uint32_t stride) const {
        return std::uint64_t(rows) * (std::uint64_t(heads) * dim + std::uint64_t(kv_heads) * tiles(stride) * partial_floats()) *
               sizeof(float);
    }
};

// Bytes one decode step reads from the K and V of `layers` attention layers at context `positions` (each layer reads
// its producer's rows once per pass): Form::none = the FP32 master, f16 / kv8 = the shadow (scales included).
[[nodiscard]] constexpr std::uint64_t kv_read_bytes(cache_shadow::Form form, std::uint64_t positions,
                                                    std::uint32_t kv_heads, std::uint32_t dim, std::uint32_t layers) {
    const std::uint32_t width = kv_heads * dim;
    const std::uint64_t one = form == cache_shadow::Form::none ? 2ULL * positions * width * sizeof(float)
                                                               : cache_shadow::shadow_bytes(form, positions, width);
    return one * layers;
}

}  // namespace strix::attn
