#pragma once

#include "strix/device/launch_geometry.hpp"

#include <algorithm>
#include <cstddef>

namespace strix::device::decode_geometry {

constexpr unsigned row_threads = 256;
constexpr unsigned dense_warps = 8;
constexpr unsigned split_keys = 256;

constexpr LaunchGeometry row_groups(unsigned groups, unsigned rows = 1, unsigned threads = row_threads) {
    return {{groups, rows, 1}, {threads, 1, 1}, 0};
}

constexpr LaunchGeometry embedding(unsigned rows) { return row_groups(rows); }

constexpr LaunchGeometry rotary(unsigned groups, unsigned rows) { return row_groups(groups, rows, 32); }

constexpr LaunchGeometry prepared_kv(unsigned heads, unsigned rows, unsigned width) {
    return row_groups(heads, std::min(rows, 65535U), std::min(width, 256U));
}

constexpr LaunchGeometry head_clip(unsigned heads, unsigned rows, unsigned width) {
    return row_groups(heads, std::min(rows, 65535U), width <= 32 ? 32U : width <= 128 ? 128U : 256U);
}

// Caller validates the nonempty image before constructing a launch.
constexpr LaunchGeometry pointwise(std::size_t elements, unsigned rows = 1) {
    const auto blocks = std::min<std::size_t>((elements - 1) / row_threads + 1, 65535U);
    return row_groups(static_cast<unsigned>(blocks), rows);
}

constexpr LaunchGeometry dense_fma(unsigned outputs, unsigned row_tiles = 1) {
    return row_groups((outputs - 1U) / dense_warps + 1U, row_tiles, dense_warps * 32U);
}

constexpr LaunchGeometry mapped_dense(unsigned blocks, unsigned rows) {
    return row_groups(blocks, rows, dense_warps * 32U);
}

constexpr LaunchGeometry dense_mma(unsigned tiles, unsigned row_tiles = 1) { return row_groups(tiles, row_tiles); }

#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
constexpr unsigned attention_chunks(unsigned positions) { return (positions + split_keys - 1U) / split_keys; }

constexpr LaunchGeometry attention_splits(unsigned chunks, unsigned kv_heads, unsigned rows) {
    return {{chunks, kv_heads, rows}, {split_keys, 1, 1}, 0};
}

constexpr LaunchGeometry prepared_split(unsigned kv_heads, unsigned capacity, unsigned rows, unsigned threads) {
    return {{kv_heads, capacity, rows}, {threads, 1, 1}, 0};
}

constexpr std::uint64_t gdn_convolution_heads(unsigned key_heads, unsigned value_heads) {
    return 2ULL * key_heads + value_heads;
}

constexpr LaunchGeometry gdn(unsigned heads, unsigned groups = 1) { return row_groups(heads, groups, 128); }

// Strides, tile size and stages used by the native padded BF16 attention layout.
template <int Width>
struct PaddedFlashStorage {
    static constexpr int D = Width, KB = 2 * D + 32, VB = 2 * D + 16;
    static constexpr int keys = 32, STAGES = 2, K_TILE = keys * KB, STAGE = K_TILE + keys * VB;
};

constexpr LaunchGeometry prepared_flash(unsigned heads, unsigned rows, unsigned shared) {
    return {{heads, rows, 1}, {256, 1, 1}, shared};
}

constexpr LaunchGeometry prepared_flash128(unsigned heads, unsigned rows) {
    using Layout = PaddedFlashStorage<128>;
    return prepared_flash(heads, rows, Layout::STAGES * Layout::STAGE);
}

}  // namespace strix::device::decode_geometry
