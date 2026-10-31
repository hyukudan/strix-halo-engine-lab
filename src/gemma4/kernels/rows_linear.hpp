#pragma once

// Exact many-row g4-v1 linears for prompt chunks. Every output element keeps
// the shared-pass arithmetic of linear()/gate_up(): lane L accumulates its
// chunks L, L+32, ... and their 16 elements in ascending order from +0 as
// RN(acc + RN(w * x)), the 32 lane values meet in wave_sum's tree, then the
// declared epilogue. The outputs are byte-identical to linear(w, x, n, out)
// called on consecutive slices of at most eight rows; only the reuse of each
// decoded weight across rows changes.
#include "ops.hpp"

namespace strix::gemma4::gpu {
// Startup-cached STRIX_G4_PREFILL_EXACT_MANYROW: unset or 1 = many-row tiles,
// 0 = the eight-row slices of linear()/gate_up() (identical bytes).
[[nodiscard]] bool rows_linear_enabled();
// Tile variants (outputs per wave x rows per workgroup); all have equal bytes.
// -1 selects STRIX_G4_ROWS_TILE, else the measured default per format.
inline constexpr int rows_linear_variants = 8;
// out[r * outputs + o] for rows r < rows; x holds rows * inputs FP32 values.
// failure (optional) receives the finite flag of every stored value.
void linear_rows(LinearView w, const float* x, std::uint32_t rows, float* out, Queue queue = nullptr,
                 std::uint32_t* failure = nullptr, bool many = rows_linear_enabled(), int variant = -1);
// out = gelu_tanh(x . gate) * (x . up), as gate_up(); formats may differ.
// With scratch (rows * outputs floats), split = 1 evaluates gate and up as two
// linear_rows calls and then gelu_product (the unfused decode schedule, same
// bytes); -1 selects STRIX_G4_ROWS_GATE_SPLIT (default 1).
void gate_up_rows(LinearView gate, LinearView up, const float* x, std::uint32_t rows, float* out, Queue queue = nullptr,
                  std::uint32_t* failure = nullptr, bool many = rows_linear_enabled(), int variant = -1,
                  float* scratch = nullptr, int split = -1);
} // namespace strix::gemma4::gpu
