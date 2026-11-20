#pragma once
// System-scope producer publication and consumer lifetime gates.
#include "channel.hpp"

#if defined(__HIPCC__)
namespace strix::xport {

__device__ __forceinline__ void publish(const ProducerTarget& t, unsigned blocks) {
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

// True once a bounded view's budget has run out since t0 (wall_clock64 ticks); an unbounded view never expires.
__device__ __forceinline__ bool wait_expired(const RxView& v, std::uint64_t t0) {
    return v.budget != 0 && v.abort != nullptr && static_cast<std::uint64_t>(wall_clock64()) - t0 > v.budget;
}
// First expiry wins: the sticky word keeps the first sequence that timed out (host reads it in the owner's check()).
__device__ __forceinline__ void record_abort(const RxView& v) {
    std::uint64_t expected = 0;
    (void)__hip_atomic_compare_exchange_strong(v.abort, &expected, v.sequence, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_SYSTEM);
}

// Consumer side of recv_view(): ALL threads of EVERY block of the first kernel that reads the payload call this before
// reading it. One lane per block polls the RX flag with a system-scope acquire (s_sleep between polls); the barrier
// publishes the result to the block. A failed exchange releases the flag with ~0; a bounded view (RxView::budget) also
// stops polling when its budget runs out, records the sequence in the sticky abort word and falls through, so the kernel
// never hangs even when nothing on the host releases the flag. Unbounded views run the former loop (the clock is read
// only for a bounded view).
__device__ __forceinline__ void wait_view(const RxView& v) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const bool poller = v.gate == nullptr || (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
        const std::uint64_t t0 = v.budget != 0 ? static_cast<std::uint64_t>(wall_clock64()) : 0U;
        if (poller) {
            while (__hip_atomic_load(v.flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < v.sequence) {
                if (wait_expired(v, t0)) {
                    record_abort(v);
                    break;
                }
                __builtin_amdgcn_s_sleep(2);
            }
            if (v.gate != nullptr) __hip_atomic_store(v.gate, v.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {  // block 0 is dispatched first and never waits on another block: no deadlock
            while (__hip_atomic_load(v.gate, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < v.sequence) {
                if (wait_expired(v, t0)) {  // block 0 opens the gate when its own budget runs out; this is the backstop
                    record_abort(v);
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
}

}  // namespace strix::xport
#endif
