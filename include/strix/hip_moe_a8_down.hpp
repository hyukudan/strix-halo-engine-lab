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
// Union rows form (validate_down_union): R <= 8 rows, K-quant down (Q4_K / Q6_K); plan = the one enqueue_probe_union wrote.
// Bitwise those of R enqueue_down_activation_quantize / enqueue_down_rawdots calls; each distinct expert read once.
void enqueue_down_activation_quantize_union(const DownRequest&,const DownBindings&,std::uint32_t rows,DeviceBytes plan,hipStream_t);
void enqueue_down_rawdots_union(const DownRequest&,const DownBindings&,std::uint32_t rows,DeviceBytes plan,hipStream_t);
// Same rawdots, plus the ordered expert-slot merge, finite test, commit and first-error publication inside the
// same launch (strix/moe_slot_merge.hpp): bitwise the incumbent separate serial tail, so the caller enqueues no
// tail. tickets: moe_slot_merge::ticket_words(hidden) words, zero on entry (the caller's prologue clears them);
// the launch leaves them zero. output holds bf16 bits when output_bf16, else f32.
struct SlotMergeTail {
    std::uint32_t* tickets=nullptr;
    const float* route_weights=nullptr;
    float* accumulation=nullptr;
    void* output=nullptr;
    bool output_bf16=false, bf16_commit=false;
    void* record=nullptr; // strix::DeferredMoeFirstError
    std::uint32_t domain=0;
};
void enqueue_down_rawdots(const DownRequest&,const DownBindings&,const SlotMergeTail&,hipStream_t);
}
