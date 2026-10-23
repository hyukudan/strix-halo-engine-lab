#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace strix::dist {
// Dedicated coherent cache line, never overlaid with a payload or its trailer.
// CPU prepares ready/done before exposing a view; the last GPU block resets its
// cacheable counter before publishing done (no memset/event per exchange).
struct alignas(64) StagedReady {
    std::uint64_t done = 0;
    std::uint64_t ready = 0;
    unsigned* block_counter = nullptr; // GPU-cacheable scratch; reset by the last block, never by CPU in the hot path
};
static_assert(sizeof(StagedReady) == 64);
struct ProducerStage {
    void* data = nullptr;
    StagedReady* flag = nullptr; // device alias, not a CPU pointer to be dereferenced by callers
    std::uint64_t token = 0;
};
struct ConsumerStage {
    const void* data = nullptr;
    StagedReady* flag = nullptr;
    std::uint64_t token = 0;
};

#if defined(__HIPCC__)
// All threads of every consumer block call this BEFORE reading its payload.
__device__ __forceinline__ void staged_acquire(const StagedReady* flag, std::uint64_t token) {
    __shared__ std::uint64_t observed;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        observed = __hip_atomic_load(&flag->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    // Broadcast the system acquire through a workgroup barrier; no UC flag load per element/lane.
    __syncthreads();
    if (observed != token) __builtin_trap();
}

// ALL threads of EVERY block call this exactly once at the very end of the
// producing/consuming kernel. Include empty/padding blocks in total_blocks.
// Each lane releases its own global accesses; the last block publishes done.
__device__ __forceinline__ void publish_staged_block(StagedReady* flag, std::uint64_t token, unsigned total_blocks) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const auto grid_blocks = static_cast<unsigned long long>(gridDim.x) * gridDim.y * gridDim.z;
        if (!total_blocks || grid_blocks != total_blocks ||
            __hip_atomic_load(&flag->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != token)
            __builtin_trap();
        auto* counter = flag->block_counter;
        if (!counter) __builtin_trap();
        const auto old = __hip_atomic_fetch_add(counter, 1U, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (old >= total_blocks) __builtin_trap();
        if (old + 1U == total_blocks) {
            __hip_atomic_store(counter, 0U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&flag->done, token, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}
#endif
}
