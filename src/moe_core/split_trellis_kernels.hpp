#pragma once

#include "moe_core/exl3_trellis.hpp"
#include "strix/moe/declared_exp.hpp"
#include <hip/hip_runtime.h>

namespace strix::moe_core::split_trellis {

// Single-row split-K schedule and ordered finishes for quantized expert shards.
// The adapter owns the layer layout and launch geometry.
__device__ __forceinline__ const std::uint32_t* words_at(const std::uint32_t* base, std::uint64_t byte_offset, std::uint64_t back_bytes) {
    // back_bytes: a rank's down matrix holds k tiles [k0, ...); the reader addresses k tiles from 0.
    return reinterpret_cast<const std::uint32_t*>(reinterpret_cast<std::uintptr_t>(base) + byte_offset - back_bytes);
}

// grid (local / 16 / Waves, gate_splits, entries * 2), block 32 * Waves.
template <class Layer, int Bits, int Waves, int Depth, unsigned StageHalves>
__global__ __launch_bounds__(32 * Waves) void gate_up_kernel(Layer L, const float* x, const std::int32_t* ids,
                                                                 std::uint32_t top_k, float* gu_part) {
    __shared__ _Float16 xs[StageHalves];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned entry = blockIdx.z >> 1U, m = blockIdx.z & 1U, split = blockIdx.y;
    const unsigned splits = L.plan.gate_splits, split_ch = L.hidden / splits;
    const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
    const std::uint16_t* suh = (m == 0 ? L.gate_suh : L.up_suh) + static_cast<std::size_t>(e) * L.hidden;
    const float* row = x + static_cast<std::size_t>(entry / top_k) * L.hidden;
    const unsigned c0 = split * split_ch, b0 = c0 / 128U, b1 = (c0 + split_ch + 127U) / 128U;
    for (unsigned b = b0 + wave; b < b1; b += Waves) transform_block_at(row, suh, xs, b * 128U, (b - b0) * 128U, lane);
    __syncthreads();
    const unsigned col = blockIdx.x * Waves + wave;
    const std::uint32_t* w = words_at(m == 0 ? L.gate : L.up, static_cast<std::uint64_t>(e) * L.tile_bytes, 0);
    float* part = gu_part + ((static_cast<std::size_t>(entry) * 2U + m) * splits + split) * L.local;
    expert_split_rows<Bits, 1, Depth>(w, L.local, L.hidden / 16U, splits, col, split, split_ch, xs + (c0 - b0 * 128U), 1, lane,
                                            part, 0);
}

// grid (local / 128, entries), block 32.
template <class Layer>
__global__ __launch_bounds__(32) void finish_gate_up_kernel(Layer L, const std::int32_t* ids, float limit, const float* gu_part,
                                                       _Float16* h) {
    const unsigned lane = threadIdx.x, block = blockIdx.x, entry = blockIdx.y;
    const unsigned splits = L.plan.gate_splits;
    const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
    const float* pg = gu_part + (static_cast<std::size_t>(entry) * 2U) * splits * L.local + block * 128U + lane;
    const float* pu = pg + static_cast<std::size_t>(splits) * L.local;
    float g[4], u[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) g[j] = u[j] = 0.0F;
    for (unsigned s = 0; s < splits; ++s)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            g[j] += pg[static_cast<std::size_t>(s) * L.local + 32U * static_cast<unsigned>(j)];
            u[j] += pu[static_cast<std::size_t>(s) * L.local + 32U * static_cast<unsigned>(j)];
        }
    hadamard128(g, lane);
    hadamard128(u, lane);
    const std::size_t eo = static_cast<std::size_t>(e) * L.local;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned i = block * 128U + lane + 32U * static_cast<unsigned>(j);
        const float act = ::strix::moe::declared_math::swiglu_clamped(g[j] * half_bits(L.gate_svh[eo + i]), u[j] * half_bits(L.up_svh[eo + i]), limit);
        const float scaled = materialize(act * half_bits(L.down_suh[eo + i]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
    _Float16* out = h + static_cast<std::size_t>(entry) * L.local;
#pragma unroll
    for (int j = 0; j < 4; ++j) out[block * 128U + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}

// grid (H / 16 / Waves, down_end - down_first, entries), block 32 * Waves.
template <class Layer, int Bits, int Waves, int Depth, unsigned StageHalves>
__global__ __launch_bounds__(32 * Waves) void down_kernel(Layer L, const std::int32_t* ids, const _Float16* h, float* down_part) {
    __shared__ _Float16 xs[StageHalves];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned entry = blockIdx.z, split = L.plan.down_first + blockIdx.y;
    const unsigned splits = L.plan.down_splits, inter = L.inter;
    const unsigned split_ch = inter / splits, local0 = split * split_ch - L.plan.down_k_tile0 * 16U;
    const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
    const _Float16* src = h + static_cast<std::size_t>(entry) * L.local + local0;
    for (unsigned i = threadIdx.x; i < split_ch; i += 32U * Waves) xs[i] = src[i];
    __syncthreads();
    const unsigned col = blockIdx.x * Waves + wave;
    const std::uint64_t back = static_cast<std::uint64_t>(L.plan.down_k_tile0) * (L.hidden / 16U) * Bits * 32U;
    const std::uint32_t* w = words_at(L.down, static_cast<std::uint64_t>(e) * L.tile_bytes, back);
    float* part = down_part + (static_cast<std::size_t>(entry) * splits + split) * L.hidden;
    expert_split_rows<Bits, 1, Depth>(w, L.hidden, inter / 16U, splits, col, split, split_ch, xs, 1, lane, part, 0);
}

// grid (H / 128, rows), block 32.
template <class Layer>
__global__ __launch_bounds__(32) void finish_down_kernel(Layer L, const std::int32_t* ids, const float* weights,
                                                            std::uint32_t top_k, const float* down_part, float* y) {
    const unsigned lane = threadIdx.x, block = blockIdx.x, row = blockIdx.y;
    const unsigned splits = L.plan.down_splits;
    float out[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned slot = 0; slot < top_k; ++slot) {
        const unsigned entry = row * top_k + slot;
        const unsigned e = static_cast<unsigned>(ids[entry]) - L.expert_first;
        const float* p = down_part + static_cast<std::size_t>(entry) * splits * L.hidden + block * 128U + lane;
        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (unsigned s = 0; s < splits; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] += p[static_cast<std::size_t>(s) * L.hidden + 32U * static_cast<unsigned>(j)];
        hadamard128(acc, lane);
        const float wgt = weights[entry];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = block * 128U + lane + 32U * static_cast<unsigned>(j);
            const float ye = materialize(acc[j] * half_bits(L.down_svh[static_cast<std::size_t>(e) * L.hidden + i]));
            out[j] = out[j] + wgt * ye;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) y[static_cast<std::size_t>(row) * L.hidden + block * 128U + lane + 32U * static_cast<unsigned>(j)] = out[j];
}

}  // namespace strix::moe_core::split_trellis
