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
// Many-row exact linear for prompt chunks (rows > 8): the bytes equal
// device_linear over consecutive slices of at most eight rows under the
// process identity (v1 lane order or v2/v3 split order). scratch holds
// rows_scratch_bytes(w, rows). Returns false (nothing enqueued) for rows <= 8
// or with STRIX_G4_PREFILL_EXACT_MANYROW=0; callers then use device_linear.
[[nodiscard]] std::size_t rows_scratch_bytes(LinearView w, unsigned rows);
bool device_linear_rows(LinearView w, const float* x, unsigned rows, float* out, void* scratch, std::size_t scratch_bytes,
                        std::uint32_t* failure, gpu::Queue queue = nullptr);
// Many-row FFN under the v2/v3 identities: out = down(gelu_tanh(gate(x)) * up(x)),
// the bytes of device_linear_rows(gate), (up), gpu::gelu_product and
// device_linear_rows(down), with the gate/up output transforms, the product and
// down's input transform in one launch. gate_out/up_out (rows * gate.outputs
// floats) hold raw core values. Returns false (nothing enqueued) for rows <= 8,
// the v1 identity, STRIX_G4_PREFILL_EXACT_MANYROW=0 or STRIX_G4_EXL3_FFN_FUSE=0.
[[nodiscard]] std::size_t ffn_rows_scratch_bytes(LinearView gate, LinearView up, LinearView down, unsigned rows);
bool device_ffn_rows(LinearView gate, LinearView up, LinearView down, const float* x, unsigned rows, float* gate_out,
                     float* up_out, float* out, void* scratch, std::size_t scratch_bytes, std::uint32_t* failure,
                     gpu::Queue queue = nullptr);
// Input transform only: rows*inputs FP16 values into scratch (tests/benchmarks).
void device_input(LinearView w, const float* x, unsigned rows, void* scratch, std::uint32_t* failure, gpu::Queue queue = nullptr);
// Writes the lane-tile packed decode of all 65,536 states (FP16 bits) for tests.
void device_codebook_table(void* out65536, gpu::Queue queue = nullptr);
// Pieces of device_prefill for the prefill GEMMs (prefill/linear.hip): the
// input transform into [K16][row][16] FP16 panels (maximum, optional, gets the
// abs bits of x), and the output transform and scale of raw core values in out.
void device_input_panels(LinearView w, const float* x, unsigned rows, void* scratch, std::uint32_t* failure,
                         std::uint32_t* maximum, gpu::Queue queue = nullptr);
void device_output(LinearView w, float* out, unsigned rows, std::uint32_t* failure, gpu::Queue queue = nullptr);
// device_ffn_rows' middle stage on raw gate/up core values (rows x gate.outputs
// floats each): output transforms, gelu_tanh(gate) * up and down's input
// transform, into down's [K16][row][16] FP16 panels. The bytes equal
// device_output(gate), device_output(up), gpu::gelu_product and
// device_input_panels(down); maximum (optional) gets the product's abs bits.
void device_ffn_middle_panels(LinearView gate, LinearView up, LinearView down, const float* gate_core, const float* up_core,
                              unsigned rows, void* product, std::uint32_t* failure, std::uint32_t* maximum,
                              gpu::Queue queue = nullptr);
// Optional panels use [N16][K16][N][K] rotated FP16 coefficients. Packed and
// prepared paths issue the identical increasing K16 WMMA instruction sequence.
void device_prefill(LinearView w, const float* x, unsigned rows, float* out, void* scratch,
                    std::uint32_t* failure, std::uint32_t* maximum, const void* panels = nullptr,
                    gpu::Queue queue = nullptr, void* converted_event = nullptr);
} // namespace strix::gemma4::exl3
