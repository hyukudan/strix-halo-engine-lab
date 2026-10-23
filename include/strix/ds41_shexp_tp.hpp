#pragma once

// Shared expert split over the two ranks on the trellis library (identity ds41-trellis-k5-v1:shexp-tp, declared in
// docs/ds41-trellis-k5-v1.md; opt-in STRIX_DS41_SHEXP_TP=1 together with the trellis shexp class).
//
// Rank r holds output rows [1152 r, +1152) of w1 (gate) and w3 (up) (TrellisSlice rows, 5120 -> 1152) and input columns
// [1152 r, +1152) of w2 (down; TrellisSlice columns, 1152 -> 5120). Its partial P_r = L(w2_r, swiglu(L(w1_r, x), L(w3_r, x)))
// is added onto its half sum before the half-sum exchange: y = (S_0 + P_0) + (S_1 + P_1), one FP32 add per value in
// that order. One process (world 1) computes both partials with the same calls, so its bits are the pair's.
// The three slice shapes are one R class for R = 1..8 on v12 (no padding, no staging copies; strix-hip-ds41-trellis-tests
// R-class run), so a row's bits never depend on the window or batch it is in.

#include "strix/ds41_trellis.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41 {

inline constexpr std::uint32_t shexp_tp_ranks = 2;

// One rank's prepared slices of a layer's shared expert.
struct ShexpTpSlices {
    const TrellisSite* gate = nullptr;  // w1 rows [hI r, + hI)
    const TrellisSite* up = nullptr;    // w3 rows [hI r, + hI)
    const TrellisSite* down = nullptr;  // w2 columns [hI r, + hI)
};

// The slices a rank prepares: (row slice, row slice, column slice) of width inter / 2 at inter / 2 * rank.
[[nodiscard]] TrellisSlice shexp_tp_gate_up_slice(std::uint32_t inter, std::uint32_t rank) noexcept;
[[nodiscard]] TrellisSlice shexp_tp_down_slice(std::uint32_t inter, std::uint32_t rank) noexcept;

// partial[r][0, H) = L(down, swiglu_clamped(L(gate, x[r]), L(up, x[r]), limit)) for r < rows; x rows contiguous [rows][H],
// gate / up / act scratch [rows][hI], partial contiguous [rows][H] (all F32 device memory).
void enqueue_shexp_tp_partial(const ShexpTpSlices& s, const float* x, std::uint32_t rows, float limit, float* gate, float* up,
                              float* act, float* partial, void* stream);

// y[i] = y[i] + t[i] for i < n.
void enqueue_add_inplace_f32(float* y, const float* t, std::size_t n, void* stream);
// y[i] = a[i] + b[i] for i < n.
void enqueue_add_f32(const float* a, const float* b, std::size_t n, float* y, void* stream);

}  // namespace strix::ds41
