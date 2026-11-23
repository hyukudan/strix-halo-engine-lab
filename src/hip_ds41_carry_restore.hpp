#pragma once
// STRIX_DS41_COMPRESS_RING: the ratio-2 carries of a rolled-back request (batch forward) or window (device forward),
// every layer in one launch (block b: layer b), exactly the copies and memsets of commit / commit_window: open, the kept
// last row's (kv, score) and valid 1; else zeros. One body for both forwards (they had byte-identical copies).
// Home: deepseek41 model (HIP). Bound by: src/hip_deepseek41_batch_forward.hip, src/hip_deepseek41_device_forward.hip.
#include "strix/ds41_attention.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::deepseek41 {
namespace {

struct CarryRestore {
    std::uint32_t* state;  // 2 x kHeadDim words: kv, score
    std::uint32_t* valid;
    const std::uint32_t* kv;
    const std::uint32_t* score;
};
constexpr std::uint32_t max_carry_restores = 64;  // per launch: 2 KiB of kernel arguments
struct CarryRestores {
    CarryRestore layer[max_carry_restores];
};
__global__ void restore_carries(CarryRestores a, std::uint32_t open) {
    const CarryRestore& c = a.layer[blockIdx.x];
    constexpr std::uint32_t D = ds41::attn::kHeadDim;
    for (std::uint32_t i = threadIdx.x; i < 2U * D; i += blockDim.x) c.state[i] = open != 0U ? (i < D ? c.kv[i] : c.score[i - D]) : 0U;
    if (threadIdx.x == 0U) *c.valid = open;
}

}  // namespace
}  // namespace strix::deepseek41
