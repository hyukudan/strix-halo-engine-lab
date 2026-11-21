#pragma once
// The HIP runtime comes first: shared headers below define __host__ __device__ helpers that need its device builtins.
#include <hip/hip_runtime.h>
#include "strix/pair/producer_publish.hpp"

namespace strix::pair {

#if defined(__HIP__)
// Called by every thread of each workgroup that stores the partial, after its last store and in workgroup-uniform
// control flow. `done` = the units of the partial this workgroup completed (workgroup-uniform), `total` = the units
// of the whole partial. The workgroup that completes the count publishes.
__device__ __forceinline__ void producer_publish(const ProducerPublish& t, unsigned total, unsigned done = 1U) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const auto old = __hip_atomic_fetch_add(t.counter, done, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if (old + done == total) {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __hip_atomic_store(t.counter, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(t.flag, t.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}
#endif

}  // namespace strix::pair
