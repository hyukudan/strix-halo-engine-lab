#pragma once
#include "strix/q51_grouped_down_contract.hpp"
#include "strix/q51_slot_parallel_contract.hpp"
#include <hip/hip_runtime.h>
namespace strix::q51_grouped_down {
// Standalone source/ISA prototype, available only with experimental A8 runtime.
// No clear, allocation, synchronization, readback or semantic output publication.
// Caller initializes accumulation to positive FP32 zero and sticky status to0
// before original receipt; grouped overwrites valid output rows after all slots.
// Both enqueue the original route/weight validator (bit1). Caller retains every
// view and stream until quiescent, including after any enqueue exception.
// Nonfinite accumulations are observable scratch; native integration's finite
// scan/semantic commit is deliberately not implemented by this probe.
void enqueue_grouped_probe(const Request&,const Bindings&,hipStream_t stream=nullptr);
void enqueue_original_receipt(const Request&,const Bindings&,hipStream_t stream=nullptr);
} // namespace strix::q51_grouped_down

namespace strix::q51_slot_parallel {
// Eager-only two-kernel probe. Reuses the original route/weight validator,
// then writes unweighted FP32 dots and merges slots in ascending order.
// No clear, allocation, synchronization or semantic output publication.
// Sticky status and invalid routes preserve accumulation and rawdots. The
// caller retains immutable routes/weights/status and every owner until drain.
void enqueue_slot_parallel_probe(const Request&, const Bindings&, hipStream_t stream=nullptr);
} // namespace strix::q51_slot_parallel
