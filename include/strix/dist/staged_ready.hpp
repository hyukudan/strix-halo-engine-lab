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
    unsigned* counter = nullptr; // == flag->block_counter (device memory), for the lean helpers below
};
struct ConsumerStage {
    const void* data = nullptr;
    StagedReady* flag = nullptr;
    std::uint64_t token = 0;
    unsigned* counter = nullptr;
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

// RX-read lifetime only. ALL threads/blocks call after their final RX read.
// __syncthreads completes/broadcasts the block's accesses; the agent-scoped
// last-block chain then releases done to CPU. Unlike publish_staged_block,
// this does NOT promise CPU/NHI visibility of payload/output writes. Outputs
// must remain GPU-owned and ordered on the communicator's single stream.
// Never use this for a producer or for CPU/DMA-visible output.
__device__ __forceinline__ void publish_staged_reads_block(StagedReady* flag, std::uint64_t token, unsigned total_blocks) {
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
// Lean forms of the two releases (opt-in callers, STRIX_DS41_READY_LEAN=1). The general helpers
// above cost the fused o_b product 66 us of its 91 us at R1 (measured): each
// lane's __threadfence_system() is seq_cst, so every wave of every block also invalidated its
// CU's GL0 and its shader array's GL1 (the activations the other resident blocks reuse), and
// thread 0 of every block made two dependent loads from the coherent control line. Here a lane
// fence only releases (completes the lane's stores at system scope, no invalidation), the
// counter comes from the view and only the last block reads the CPU's token, once, before it
// publishes done. Same last-block chain, same counter reset, same done token.
template <bool Release>
__device__ __forceinline__ void publish_staged_block_lean_impl(StagedReady* flag, unsigned* counter, std::uint64_t token,
                                                              unsigned total_blocks) {
    if constexpr (Release) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const auto grid_blocks = static_cast<unsigned long long>(gridDim.x) * gridDim.y * gridDim.z;
        if (!total_blocks || grid_blocks != total_blocks || !counter) __builtin_trap();
        const auto old = __hip_atomic_fetch_add(counter, 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if (old >= total_blocks) __builtin_trap();
        if (old + 1U == total_blocks) {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            if (__hip_atomic_load(&flag->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != token) __builtin_trap();
            __hip_atomic_store(counter, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&flag->done, token, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}
// Producer: ALL threads of EVERY block, exactly once, after their final payload store.
__device__ __forceinline__ void publish_staged_block_lean(StagedReady* flag, unsigned* counter, std::uint64_t token,
                                                         unsigned total_blocks) {
    publish_staged_block_lean_impl<true>(flag, counter, token, total_blocks);
}
// RX-read lifetime only, as publish_staged_reads_block (outputs GPU-owned, same stream).
__device__ __forceinline__ void publish_staged_reads_block_lean(StagedReady* flag, unsigned* counter, std::uint64_t token,
                                                               unsigned total_blocks) {
    publish_staged_block_lean_impl<false>(flag, counter, token, total_blocks);
}
// Lean consumer entry: a system acquire fence (invalidates this CU's caches) with no load of the
// control line. The view only exists after finish_staged_ready() validated RX on the CPU and set
// ready, before the consumer was enqueued; the last block checks that token in its release. This
// is the visibility the event-based fused path's consumers already rely on (they read the coherent
// receive buffer directly), plus the invalidation.
__device__ __forceinline__ void staged_acquire_lean() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }
// A producer view's release: the lean form when the caller kept the view's counter, else the general one.
__device__ __forceinline__ void publish_staged_view(const ProducerStage& stage, unsigned total_blocks) {
    if (stage.counter) publish_staged_block_lean(stage.flag, stage.counter, stage.token, total_blocks);
    else publish_staged_block(stage.flag, stage.token, total_blocks);
}
#endif
}
