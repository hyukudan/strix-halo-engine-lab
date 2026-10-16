#pragma once

#include "visibility_contract.hpp"
#include "strix/direct_gpu_stream_commissioning.hpp"
#include <hip/hip_runtime.h>

namespace strix::transport::visibility {
__global__ void initialize_result(Result* result, std::uint64_t nonce,
    std::uint64_t epoch, std::uint64_t sequence, std::uint64_t stage) {
    if (blockIdx.x || threadIdx.x) return;
    result->started = nonce; result->finished = 0;
    result->epoch = epoch; result->sequence = sequence; result->stage = stage;
    result->checked = 0; result->mismatches = 0; result->first_bad = ~0ULL;
}
__global__ void verify_bytes(const std::uint8_t* source, std::uint64_t bytes,
    std::uint64_t seed, std::uint64_t sequence, Result* result) {
    const auto first = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    unsigned long long checked = 0, mismatches = 0, first_bad = ~0ULL;
    for (auto index = first; index < bytes; index += stride) {
        // Normal GPU loads: no cache-bypass/volatile trick to hide stale data.
        const auto actual = source[index];
        if (actual != direct_gpu_commissioning_pattern_byte(index, seed, sequence)) {
            ++mismatches;
            if (index < first_bad) first_bad = index;
        }
        ++checked;
    }
    if (checked) atomicAdd(&result->checked, checked);
    if (mismatches) {
        atomicAdd(&result->mismatches, mismatches);
        atomicMin(&result->first_bad, first_bad);
    }
}
__global__ void finish_result(Result* result, std::uint64_t nonce) {
    if (!blockIdx.x && !threadIdx.x) result->finished = nonce ^ completion_mask;
}
// Used only before IMPORT/ARM for a deliberate one-byte negative control.
__global__ void flip_byte(std::uint8_t* buffer, std::uint64_t index) {
    if (!blockIdx.x && !threadIdx.x) buffer[index] ^= 1U;
}
} // namespace strix::transport::visibility
