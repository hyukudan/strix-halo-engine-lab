#pragma once
#include "strix/moe_q4k_a8_contract.hpp"
#include <hip/hip_runtime.h>
namespace strix::moe_q4k_a8 {
// Standalone caller-owned sticky status; deliberately independent of graph/session status.
inline constexpr unsigned invalid_route = 1U, nonfinite_activation = 2U, invalid_delta = 4U;
// No allocation, synchronization, reset, readback, fallback or session publication.
// Caller retains ALL bindings until its stream has quiesced, including after errors.
void enqueue_probe(const Request&, const Bindings&, hipStream_t stream = nullptr);
// Test-only conversion receipt; exactly65536 uint32 outputs, caller-owned storage.
void enqueue_division_probe(const std::uint32_t* a,const std::uint32_t* b,
    std::uint32_t* output,unsigned count,hipStream_t stream = nullptr);
void enqueue_half_conversion_probe(std::uint32_t* output, hipStream_t stream = nullptr);
}
