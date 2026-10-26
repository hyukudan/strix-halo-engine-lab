#pragma once
// Device side of the MiniMax exchange module: a producer kernel that writes its message into Exchanger::tx() can
// publish it itself (Publish::producer), so no extra launch or stream packet sits between the last store and the
// progress thread. ALL threads of EVERY block call mmx_publish() exactly once, after their last store to the message;
// `blocks` is the grid size. Each lane releases its own stores at system scope (no cache invalidation), the blocks
// count down on an agent-scope counter, and the last one resets it and stores the sequence into the TX flag with a
// system-scope release (the form of strix::dist::publish_staged_block_lean, without the CPU token check).
#include "minimax_xfer.hpp"

#if defined(__HIPCC__)
namespace strix::minimax::xfer {

__device__ __forceinline__ void mmx_publish(const ProducerTarget& t, unsigned blocks) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const auto old = __hip_atomic_fetch_add(t.counter, 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if (old + 1U == blocks) {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __hip_atomic_store(t.counter, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(t.flag, t.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

// Consumer side of recv_view(): ALL threads of EVERY block of the first kernel that reads the payload call this before
// reading it. One lane per block polls the RX flag with a system-scope acquire (s_sleep between polls); the barrier
// publishes the result to the block. A failed exchange releases the flag with ~0, so the kernel never hangs.
__device__ __forceinline__ void mmx_wait(const RxView& v) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const bool poller = v.gate == nullptr || (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
        if (poller) {
            while (__hip_atomic_load(v.flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < v.sequence) __builtin_amdgcn_s_sleep(2);
            if (v.gate != nullptr) __hip_atomic_store(v.gate, v.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {  // block 0 is dispatched first and never waits on another block: no deadlock
            while (__hip_atomic_load(v.gate, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < v.sequence) __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
}

}  // namespace strix::minimax::xfer
#endif
