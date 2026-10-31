#pragma once
#include "format.hpp"
#include "gemma4/kernels/ops.hpp"

namespace strix::gemma4::exl3 {
// Scratch holds rows*inputs FP16 values, allocated before graph capture.
// shared=false repeats the core weight pass independently, with identical bytes.
void device_linear(LinearView w, const float* x, unsigned rows, float* out, void* scratch,
                   std::uint32_t* failure, bool shared = true, gpu::Queue queue = nullptr);
// g4-exl3-v2 (STRIX_G4_EXL3_IDENTITY=v2): device_linear additionally uses
// scratch bytes [v2_partial_offset, +v2_partial_bytes) for split partials when
// v2_splits(I, O) > 1. Callers size scratch to device_scratch_bytes.
inline constexpr std::size_t v2_partial_offset = 256 * 1024, v2_partial_bytes = 1024 * 1024;
inline constexpr std::size_t device_scratch_bytes = v2_partial_offset + v2_partial_bytes;
// Up to max_group linears over the same input rows (q/k/v, gate/up): one input
// and one output transform launch for the group. Each member's bytes equal
// device_linear. scratch holds count*rows*inputs FP16 values.
// STRIX_G4_EXL3_GROUP=0 issues the members as independent device_linear calls.
inline constexpr unsigned max_group = 3;
void device_linear_group(const LinearView* w, unsigned count, const float* x, unsigned rows, float* const* out,
                         void* scratch, std::uint32_t* failure, bool shared = true, gpu::Queue queue = nullptr);
// Core-only entry over an already transformed FP16 scratch (rows <= 8).
// automatic applies the default route and its OFF switches; the explicit
// routes exist for operator tests and benchmarks and all produce the same bytes.
enum class CoreRoute : unsigned { automatic, legacy, lane8, lane4, staged8, staged4 };
void device_core(LinearView w, const void* transformed, unsigned rows, float* out, std::uint32_t* failure,
                 CoreRoute route, gpu::Queue queue = nullptr);
// Input transform only: rows*inputs FP16 values into scratch (tests/benchmarks).
void device_input(LinearView w, const float* x, unsigned rows, void* scratch, std::uint32_t* failure, gpu::Queue queue = nullptr);
// Writes the lane-tile packed decode of all 65,536 states (FP16 bits) for tests.
void device_codebook_table(void* out65536, gpu::Queue queue = nullptr);
// Optional panels use [N16][K16][N][K] rotated FP16 coefficients. Packed and
// prepared paths issue the identical increasing K16 WMMA instruction sequence.
void device_prefill(LinearView w, const float* x, unsigned rows, float* out, void* scratch,
                    std::uint32_t* failure, std::uint32_t* maximum, const void* panels = nullptr,
                    gpu::Queue queue = nullptr, void* converted_event = nullptr);
} // namespace strix::gemma4::exl3
