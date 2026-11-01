#pragma once
#include "strix/moe_q4k_a8_contract.hpp"
#include <hip/hip_runtime.h>
namespace strix::moe_q4k_a8 {
// Standalone caller-owned sticky status; deliberately independent of graph/session status.
inline constexpr unsigned invalid_route = 1U, nonfinite_activation = 2U, invalid_delta = 4U;
// No allocation, synchronization, reset, readback, fallback or session publication.
// Caller retains ALL bindings until its stream has quiesced, including after errors.
// After an enqueue exception, every writable view is indeterminate even if status is0.
// Read-only data stays immutable; mutable views/status have exclusive stream ownership.
void enqueue_probe(const Request&, const Bindings&, hipStream_t stream = nullptr);
// One serial row in two launches: one CTA clears clear_words floats of the caller's accumulation, validates the routes
// and route weights and quantizes (the same status, codes, deltas and sums as clearing the accumulation and status,
// validate_routes_kernel and enqueue_probe's validation and quantizer), then enqueue_probe's projection.
void enqueue_serial_prologue_probe(const Request&, const Bindings&, float* clear, unsigned clear_words,
    const float* route_weights, hipStream_t stream);
// Exact incumbent route validation and quantizer only. Produces one A8 image;
// projection output is admitted but untouched. No reset/allocation/sync.
void enqueue_quantize_only(const Request&, const Bindings&, hipStream_t stream);
// Standalone diagnostic only. Same validation, quantizer, status and output
// contract; one CTA stages activations once for independent gate/up sums.
// Finite scope only; nonfinite projection payload/sign is not qualified.
inline constexpr std::string_view shared_gate_up_staging_marker = "a8-shared-gateup-staging-v1";
void enqueue_probe_shared_gate_up(const Request&, const Bindings&, hipStream_t stream = nullptr);
// Test-only rounded division receipt; count outputs, caller-owned storage.
void enqueue_division_probe(const std::uint32_t* a,const std::uint32_t* b,
    std::uint32_t* output,unsigned count,hipStream_t stream = nullptr);
// Test-only conversion receipt; exactly65536 uint32 outputs, caller-owned storage.
void enqueue_half_conversion_probe(std::uint32_t* output, hipStream_t stream = nullptr);
}
