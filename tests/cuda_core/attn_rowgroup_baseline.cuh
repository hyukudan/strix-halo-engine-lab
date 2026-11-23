#pragma once

#include <cuda_runtime.h>

#include "cuda/core/attention_rowgroup.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

// Test-only snapshot of qwen_qsa.cu and its layout constants at 5b220d036.
namespace attn_rowgroup_baseline {
namespace {

namespace rg = strix::cuda::core::rowgroup;

constexpr unsigned kHeads = 24, kKvHeads = 2, kHeadDim = 256, kG = 12;
constexpr unsigned kQGateWidth = kHeads * kHeadDim * 2U;
constexpr unsigned kKvWidth = kKvHeads * kHeadDim;
constexpr unsigned kAttnWidth = kHeads * kHeadDim;
constexpr unsigned kRatio = 4, kTopK = 512, kMaxSelected = 2048U + kRatio - 1U;

__host__ __device__ constexpr std::uint32_t selected_count(std::uint64_t position) {
    const auto visible = static_cast<std::uint32_t>(position + 1U);
    const auto complete = visible / kRatio;
    return (complete < kTopK ? complete : kTopK) * kRatio + visible % kRatio;
}

__device__ __forceinline__ float bf16f(std::uint16_t b) {
    return __uint_as_float(static_cast<std::uint32_t>(b) << 16U);
}

__device__ __forceinline__ std::uint16_t to_bf16(float v) {
    const auto b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}

__device__ __forceinline__ float round_bf16(float v) { return bf16f(to_bf16(v)); }

struct FrozenRowgroup {
    static constexpr unsigned D = kHeadDim, Heads = kG, Rows = 4, Cap = kMaxSelected;
    static constexpr float kScaleLog2 = 0.0625F * 1.44269504088896341F;  // 1/sqrt(256) * log2(e)
    const float* q_gate;
    std::uint64_t first_position;
    const std::uint32_t* selected;
    const std::uint16_t* kv;
    float* output;
    std::uint32_t rows;
    __device__ unsigned count(unsigned row) const { return selected_count(first_position + row); }
    __device__ std::uint32_t key(unsigned row, unsigned j) const {
        return selected[static_cast<std::uint64_t>(row) * kMaxSelected + j];
    }
    __device__ const std::uint16_t* key_row(std::uint32_t token, unsigned kvh) const {
        return kv + static_cast<std::uint64_t>(token) * (kKvWidth * 2ULL) + static_cast<std::uint64_t>(kvh) * kHeadDim;
    }
    __device__ const std::uint16_t* value_row(std::uint32_t token, unsigned kvh) const { return key_row(token, kvh) + kKvWidth; }
    __device__ float query(unsigned row, unsigned kvh, unsigned h, unsigned d) const {
        return q_gate[static_cast<std::uint64_t>(row) * kQGateWidth + static_cast<std::uint64_t>(kvh * kG + h) * kHeadDim * 2ULL + d];
    }
    __device__ void store2(unsigned row, unsigned kvh, unsigned h, unsigned d, float a0, float a1) const {
        const auto head = kvh * kG + h;
        const float* g = q_gate + static_cast<std::uint64_t>(row) * kQGateWidth + static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + d;
        const float g0 = round_bf16(1.0F / (1.0F + __expf(-g[0]))), g1 = round_bf16(1.0F / (1.0F + __expf(-g[1])));
        *reinterpret_cast<float2*>(output + static_cast<std::uint64_t>(row) * kAttnWidth + static_cast<std::uint64_t>(head) * kHeadDim + d) =
            make_float2(round_bf16(round_bf16(a0) * g0), round_bf16(round_bf16(a1) * g1));
    }
};

using Shape = rg::Shape<FrozenRowgroup>;
static_assert(Shape::Threads == 128 && Shape::UnionSmem == 49232 && Shape::FlashSmem == 33792);

__global__ __launch_bounds__(256) void frozen_union(FrozenRowgroup p, rg::Union u) { rg::union_body(p, u); }
__global__ __launch_bounds__(Shape::Threads, 2) void frozen_flash(FrozenRowgroup p, rg::Union u) { rg::flash_body(p, u); }

void check(cudaError_t status, const char* stage) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("frozen QSA ") + stage + ": " + cudaGetErrorString(status));
}

void launch(FrozenRowgroup p, void* workspace, cudaStream_t stream) {
    if (!p.rows) return;
    static const bool attr = [] {
        check(cudaFuncSetAttribute(frozen_union, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(Shape::UnionSmem)), "union smem attr");
        return true;
    }();
    (void)attr;
    const unsigned tiles = (p.rows + FrozenRowgroup::Rows - 1U) / FrozenRowgroup::Rows;
    const auto u = rg::carve<FrozenRowgroup>(workspace, tiles);
    frozen_union<<<tiles, 256, Shape::UnionSmem, stream>>>(p, u);
    check(cudaGetLastError(), "union launch");
    frozen_flash<<<dim3(kKvHeads, tiles), Shape::Threads, Shape::FlashSmem, stream>>>(p, u);
    check(cudaGetLastError(), "flash launch");
}

}  // namespace
}  // namespace attn_rowgroup_baseline
