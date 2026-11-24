#pragma once
// Host-side executor of the exact tensor-parallel joins on HIP: the swap of the slots through a type-erased gather
// callback (strix/pair/gather_fn.hpp), stream-ordered and counted. No kernels here: the bfloat16-wire join is
// tp_join_bf16_hip.hpp, the join-map joins tp_join_map_hip.hpp (kept apart so an engine's device image only carries the
// kernels it uses). Nothing is model-specific.
// Home: shared (backend core, HIP). Bound by: qwen35 dense HIP exact TP decode (src/qwen35/hip/engine.hip tp_gather) and
// Hy4 HIP TP step joins (src/hy4/hip/hy4_hip_forward.hip tp_gather). Body moved unchanged.
#include "strix/pair/gather_fn.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strix::xport::tp_join_exec {

// The symmetric swap of the slots in `slots`, stream-ordered after its producers; counts the call.
inline void swap(const strix::pair::GatherFn& g, hipStream_t s, void* slots, std::size_t slot_bytes, std::size_t bytes, std::uint64_t& calls) {
    g(static_cast<void*>(s), slots, slot_bytes, bytes);
    ++calls;
}

}  // namespace strix::xport::tp_join_exec
