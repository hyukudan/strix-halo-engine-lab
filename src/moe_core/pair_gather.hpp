#pragma once
#include <cstddef>

#if defined(__HIPCC__)
#include "exl3_trellis.hpp"
#define STRIX_PAIR_GATHER_INLINE __host__ __device__ __forceinline__
#else
#define STRIX_PAIR_GATHER_INLINE inline
#endif

namespace strix::moe_core {

// Physical [row tile][column tile][row in tile][column in tile] address.
STRIX_PAIR_GATHER_INLINE constexpr std::size_t tiled_at(unsigned row, unsigned column, unsigned width) {
    return ((static_cast<std::size_t>(row >> 4U) * (width >> 4U) + (column >> 4U)) << 8U)
           + ((row & 15U) << 4U) + (column & 15U);
}

// One wave gathers 128 channels for one (routed pair, gate/up matrix).
struct PairGatherBlock {
    unsigned pair, matrix, block;

    STRIX_PAIR_GATHER_INLINE constexpr unsigned channel(unsigned lane, unsigned element) const {
        return block * 128U + lane + 32U * element;
    }

    template <unsigned Width, class Group>
    STRIX_PAIR_GATHER_INLINE constexpr std::size_t output_offset(const Group& group, unsigned padded_pairs, unsigned c) const {
        const unsigned row = group.tbegin + (pair - group.pair_begin);
        return static_cast<std::size_t>(matrix) * padded_pairs * Width + tiled_at(row, c, Width);
    }
};

template <unsigned Width>
STRIX_PAIR_GATHER_INLINE constexpr PairGatherBlock pair_gather_block(unsigned wave) {
    static_assert(Width > 0U && Width % 128U == 0U);
    constexpr unsigned blocks = Width / 128U;
    return {wave / (2U * blocks), (wave / blocks) % 2U, wave % blocks};
}

#if defined(__HIPCC__)
template <unsigned Width> struct ContiguousGatherRows {
    template <class Tables>
    static __device__ __forceinline__ const float* row(const float* x, const Tables&, unsigned token) {
        return x + static_cast<std::size_t>(token) * Width;
    }
};

template <unsigned Width, class Tables, class InputRows = ContiguousGatherRows<Width>>
__global__ __launch_bounds__(256) void gather_transform_tiled_kernel(const float* x, Tables t) {
    const unsigned wave = blockIdx.x * 8U + (threadIdx.x >> 5U), lane = threadIdx.x & 31U;
    if (wave >= t.pairs * 2U * (Width / 128U)) return;
    const PairGatherBlock work = pair_gather_block<Width>(wave);
    const auto& group = t.groups[t.pair_group[work.pair]];
    const auto* suh = work.matrix == 0U ? group.w.gate.suh : group.w.up.suh;
    const float* input = InputRows::row(x, t, t.pair_row[work.pair]);
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        const unsigned c = work.channel(lane, j);
        const float scaled = materialize(input[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j)
        t.xt[work.output_offset<Width>(group, t.tpairs, work.channel(lane, j))] = static_cast<_Float16>(v[j]);
}
#endif

} // namespace strix::moe_core
#undef STRIX_PAIR_GATHER_INLINE
