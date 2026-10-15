#pragma once
#include "strix/q51_grouped_down_contract.hpp"
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
