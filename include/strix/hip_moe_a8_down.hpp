#pragma once
#include "strix/moe_a8_down_contract.hpp"
#include <hip/hip_runtime.h>
namespace strix::moe_q4k_a8 {
// Explicit owned non-null eager stream. Reject capture before ANY enqueue.
// No allocation, synchronization, reset, route merge or session publication.
// Full bundle is validated before either enqueue. Rawdots are intermediate:
// caller MUST preserve the existing final finite scan, ordered merge, commit
// and deferred publication. Nonfinite weight/overflow results are not published
// as a successful session by this API.
void enqueue_down_activation_quantize(const DownRequest&,const DownBindings&,hipStream_t);
void enqueue_down_rawdots(const DownRequest&,const DownBindings&,hipStream_t);
}
