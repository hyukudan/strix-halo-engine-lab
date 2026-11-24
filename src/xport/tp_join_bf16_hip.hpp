#pragma once
// The exact bfloat16-wire join of a TP step share on HIP (wire contract strix/pair/bf16_wire.hpp, bodies in
// bf16_wire_hip.hpp): pack own share -> swap -> place. Launch order is the engine's own.
// Home: shared (backend core, HIP). Bound by: Hy4 HIP TP step joins (src/hy4/hip/hy4_hip_forward.hip tp_join). Body moved unchanged.
#include "strix/pair/gather_fn.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#include "bf16_wire_hip.hpp"
#include "tp_join_exec_hip.hpp"

namespace strix::xport::tp_join_exec {

// `base` = [2][per_rank] FP32 (slot q = rank q's BF16-rounded share); wbuf = the wire buffer ([2][per_rank] halves). This
// rank's slot packed (flag raised on a word that is not a bfloat16), the halves swapped, both slots widened back into `base`.
// check(hipError_t, const char*) raises on an error (the caller's own).
template <class Check>
inline void join_bf16_wire(const strix::pair::GatherFn& g, hipStream_t s, float* base, std::uint16_t* wbuf, std::size_t per_rank, unsigned rank,
                           unsigned* flag, std::uint64_t& calls, Check&& check) {
    namespace xbw = strix::xport::bf16_wire;
    check(xbw::enqueue_pack(base + rank * per_rank, wbuf + rank * per_rank, per_rank, flag, s), "tp join pack");
    swap(g, s, wbuf, xbw::contract::wire_bytes(per_rank), xbw::contract::wire_bytes(per_rank), calls);
    check(xbw::enqueue_place({wbuf, per_rank, 0U, per_rank, per_rank, 1U, base, 2U * per_rank}, s), "tp join widen");
}

}  // namespace strix::xport::tp_join_exec
