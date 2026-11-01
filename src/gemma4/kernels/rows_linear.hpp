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
// STRIX_G4_PREFILL_ROWS_TILE_V2 (read per call; unset or 1 = on, 0 = off):
// per-shape default tiles (two-tile input staging for calls of at least 256
// rows and more than 512 outputs). Every variant has the same bytes.
[[nodiscard]] bool rows_tile_v2();
// Tile variants (outputs per wave x rows per workgroup); all have equal bytes.
// -1 selects STRIX_G4_ROWS_TILE, else the measured default per format.
inline constexpr int rows_linear_variants = 12;
// out[r * outputs + o] for rows r < rows; x holds rows * inputs FP32 values.
// failure (optional) receives the finite flag of every stored value.
void linear_rows(LinearView w, const float* x, std::uint32_t rows, float* out, Queue queue = nullptr,
                 std::uint32_t* failure = nullptr, bool many = rows_linear_enabled(), int variant = -1);
// out = gelu_tanh(x . gate) * (x . up), as gate_up(); formats may differ.
// With scratch (rows * outputs floats), split = 1 evaluates gate and up as two
// linear_rows calls and then gelu_product (the unfused decode schedule, same
// bytes); -1 selects STRIX_G4_ROWS_GATE_SPLIT (default 0: one fused gate/up tile). fuse_finite: the split
// schedule's product and its finite check run as one launch (same bytes, same flags).
void gate_up_rows(LinearView gate, LinearView up, const float* x, std::uint32_t rows, float* out, Queue queue = nullptr,
                  std::uint32_t* failure = nullptr, bool many = rows_linear_enabled(), int variant = -1,
                  float* scratch = nullptr, int split = -1, bool fuse_finite = false);
} // namespace strix::gemma4::gpu
