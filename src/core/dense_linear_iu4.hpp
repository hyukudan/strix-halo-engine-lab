#pragma once
// Dense decode / verify projection over the shared IU4 g64 pack (core/iu4.hpp layout, stored as ggml type
// strix::gguf::type_strix_iu4_g64: packed tiles then FP16 scales [N][K/64]) on the shared W4A16 small-M core
// (core/w4a16.hpp). out[r][o] = sum_k W[o][k] * x[r][k], rows 1..16, F32 output. Third weight format of the
// dense_linear family (bf16, q8_0, iu4); the caller picks it by the tensor type.
//   - Activations: one prep kernel converts the F32 rows to FP16 (finite values saturate, NaN propagates) in a
//     scratch the caller owns; the core reads them straight from global memory (every lane of a wave reads the same
//     address: broadcast, L1 resident).
//   - One block = 8 waves = 64 output columns; the 8 waves split K in contiguous ranges of K / 8 (G / 8 = MaxG K64
//     groups each), reduce their partial sums in LDS in a fixed ascending order (deterministic, no atomics).
//   - Rows are padded to the bucket R (1/2/4/8/16); a core call that holds any real row computes all of its Rc rows
//     (rows past nrows read stale scratch rows: the caller's xh must hold R rows) and only rows < nrows are stored.
//     Each row's dot products are independent of the other rows (same operands, same order), so stored rows are
//     bitwise those of a call with fewer rows. Passing the real count instead took `r >= nrows` exits inside the
//     unrolled row loops of every k-tile: measured on Strix 27B IU4 verify windows rows 3 / 6 / 12 = 127.1 / 130.6 /
//     140.7 ms vs rows 4 / 8 / 16 = 76.5 / 82.6 / 126.8 ms (hip_dense_dec3-STATE.md). Default: core
//     calls of up to 8 rows (R <= 8: one call, every weight word read once; R = 16: two 8-row calls), bitwise equal
//     to the chunked form (per-row op order is identical). One call over 16 rows was measured slower (rows 16
//     verify 272.8 vs 220.7 ms, Strix 27B IU4) and spills (private segment 144 B): not offered.
//     STRIX_IU4_ROW_REUSE=0 selects the old 4-row chunk calls (weights re-streamed per chunk).
// Home: shared (backend core, HIP): no model names; any dense projection can bind it (qwen35 now).
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "strix/env_flags.hpp"
#include "iu4_x_pairs.hpp"
#include "w4a16.hpp"

namespace strix::core::dense_iu4 {

inline constexpr unsigned kWaves = 8U;
inline constexpr unsigned kMaxRows = 16U;

inline constexpr std::size_t packed_bytes(unsigned outputs, unsigned inputs) { return std::size_t(outputs) * inputs / 2U; }
inline constexpr std::size_t tensor_bytes(unsigned outputs, unsigned inputs) {
    return packed_bytes(outputs, inputs) + std::size_t(outputs) * (inputs / 64U) * 2U;
}
// Shapes the kernel takes: N % 64 == 0, K % 512 == 0 and K / 512 in the instantiated set.
inline constexpr bool supported(unsigned outputs, unsigned inputs) {
    const unsigned mg = inputs / 512U;
    return outputs % 64U == 0U && inputs % 512U == 0U && (mg == 10U || mg == 12U || mg == 34U);
}

// Templates (a dummy parameter): a header kernel included by several TUs needs weak linkage (no duplicate symbol).
template <int = 0>
__global__ void to_f16_kernel(const float* x, _Float16* h, std::size_t count) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += std::size_t(gridDim.x) * blockDim.x) {
        const float v = x[i];
        h[i] = static_cast<_Float16>(v != v ? v : fminf(65504.0F, fmaxf(-65504.0F, v)));
    }
}
// Same conversion, stored in the pre-paired layout of the scalar-activation core (permutation inside 8-k blocks; rows
// start on block boundaries because inputs % 64 == 0).
template <int = 0>
__global__ void to_f16_paired_kernel(const float* x, _Float16* h, std::size_t count) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += std::size_t(gridDim.x) * blockDim.x) {
        const float v = x[i];
        h[iu4_pairs::paired_index(i)] = static_cast<_Float16>(v != v ? v : fminf(65504.0F, fmaxf(-65504.0F, v)));
    }
}

template <unsigned R, unsigned MaxG, unsigned Rc = R, bool UX = false>
__global__ void __launch_bounds__(32 * kWaves) kernel(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh,
                                                      float* out, unsigned nrows) {
    constexpr unsigned S = kWaves, KT = MaxG * 4U;
    __shared__ float part[R * S * 64];
    // UX: wave index made wave-uniform (SGPR) so the activation reads of w4a16::wave_rows<..., UniformX> are scalar loads.
    const unsigned lane = threadIdx.x & 31U, quad = blockIdx.x;
    const unsigned wave = UX ? __builtin_amdgcn_readfirstlane(threadIdx.x / 32U) : threadIdx.x / 32U;
    const auto* ws = reinterpret_cast<const std::uint16_t*>(w + packed_bytes(outputs, inputs));
    const unsigned k0 = wave * KT;
    float res[R][2];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) res[r][0] = res[r][1] = 0.0F;
#pragma unroll
    for (unsigned h = 0; h * Rc < R; ++h) {
        if (h * Rc >= nrows) break;
        // all Rc rows of a call that holds a real row (the full-bucket instruction stream); the store keeps r < nrows
        w4a16::wave_rows<Rc, KT, MaxG, false, false, UX>(w, ws, outputs, inputs, k0, k0 + KT, quad * 4U, xh + std::size_t(h * Rc) * inputs + k0 * 16U,
                                       inputs, lane, Rc,
                                       [&](unsigned, unsigned r, unsigned column, float value) { res[h * Rc + r][column & 1U] += value; });
    }
    const unsigned rel = (lane >> 3U) * 16U + 2U * (lane & 7U);
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        part[(r * S + wave) * 64U + rel] = res[r][0];
        part[(r * S + wave) * 64U + rel + 1U] = res[r][1];
    }
    __syncthreads();
    for (unsigned t = threadIdx.x; t < R * 64U; t += 32U * S) {
        const unsigned r = t / 64U, c = t % 64U;
        if (r >= nrows) continue;
        float sum = part[(r * S) * 64U + c];
#pragma unroll
        for (unsigned s = 1; s < S; ++s) sum += part[(r * S + s) * 64U + c];
        out[std::size_t(r) * outputs + quad * 64U + c] = sum;
    }
}

// STRIX_IU4_ROW_REUSE: unset/1 = core calls of up to 8 rows (default), 0 = old 4-row chunks.
inline bool row_reuse() {
    static const bool on = envcfg::flag("STRIX_IU4_ROW_REUSE", true);  // recorded in the effective config
    return on;
}
// STRIX_IU4_SCALAR_X: 1 = pre-paired activations read through scalar loads (w4a16 UniformX: same dot2 operands and
// order, bitwise equal; default since its node A/B: plain 75.73 -> 72.27 ms/token, verify rows 8 118.9 -> 82.6 ms,
// logits bitwise, HDD2-T1), 0 = natural layout through vector broadcast loads (the previous form). One switch for
// prepare() and run() (layout and kernel must agree); off under STRIX_IU4_ROW_REUSE=0.
inline bool scalar_x() {
    static const bool on = envcfg::flag("STRIX_IU4_SCALAR_X", true) && row_reuse();  // recorded in the effective config
    return on;
}

template <unsigned R>
void launch_bucket(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh, float* out, unsigned rows, hipStream_t s) {
    const dim3 grid(outputs / 64U), block(32U * kWaves);
    constexpr unsigned C = R >= 8U ? 4U : R, M = R > 8U ? 8U : R;
    const bool mid = R <= 4U || row_reuse(), sx = scalar_x();
    switch (inputs / 512U) {
    case 10U:
        if (sx) kernel<R, 10U, M, true><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        else if (mid) kernel<R, 10U, M><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        else kernel<R, 10U, C><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        break;
    case 12U:
        if (sx) kernel<R, 12U, M, true><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        else if (mid) kernel<R, 12U, M><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        else kernel<R, 12U, C><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        break;
    default:
        if (sx) kernel<R, 34U, M, true><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        else if (mid) kernel<R, 34U, M><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        else kernel<R, 34U, C><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows);
        break;
    }
}

// Reference form for the exactness gate: dequantises the same pack (w = q * s exact in F32) and sums in double over
// ascending k with the F32 activations, one thread per output column and row. Slow; selected by set_reference(true).
template <int = 0>
__global__ void reference_kernel(const std::uint8_t* w, unsigned inputs, unsigned outputs, const float* x, float* out, unsigned rows) {
    const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= outputs) return;
    const auto* ws = reinterpret_cast<const std::uint16_t*>(w + packed_bytes(outputs, inputs));
    const unsigned groups = inputs / 64U;
    for (unsigned r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (unsigned k = 0; k < inputs; ++k) {
            const std::size_t at = ((std::size_t(k / 16U) * (outputs / 16U) + n / 16U) * 16U + n % 16U) * 8U + (k % 16U) / 2U;
            const unsigned nib = (w[at] >> ((k & 1U) * 4U)) & 15U;
            const float q = float(nib < 8U ? int(nib) : int(nib) - 16);
            const float s = static_cast<float>(__builtin_bit_cast(_Float16, ws[std::size_t(n) * groups + k / 64U]));
            acc += double(q * s) * double(x[std::size_t(r) * inputs + k]);
        }
        out[std::size_t(r) * outputs + n] = float(acc);
    }
}
inline bool& reference_flag() { static bool on = false; return on; }
inline void set_reference(bool on) { reference_flag() = on; }
inline void reference(const std::uint8_t* w, unsigned inputs, unsigned outputs, const float* x, float* out, unsigned rows, hipStream_t s) {
    reference_kernel<0><<<(outputs + 127U) / 128U, 128, 0, s>>>(w, inputs, outputs, x, out, rows);
}

// Converts the F32 activations once; several projections of the same x (q/k/v, gate/up) share the prepared rows.
inline void prepare(const float* x, _Float16* xh, unsigned inputs, unsigned rows, hipStream_t s) {
    if (scalar_x()) to_f16_paired_kernel<0><<<64, 256, 0, s>>>(x, xh, std::size_t(rows) * inputs);
    else to_f16_kernel<0><<<64, 256, 0, s>>>(x, xh, std::size_t(rows) * inputs);
}
// xh: FP16 rows from prepare(); out: F32 [rows][outputs]; rows 1..16.
inline void run(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh, float* out, unsigned rows, hipStream_t s) {
    if (!supported(outputs, inputs) || rows == 0U || rows > kMaxRows) throw std::runtime_error("iu4 dense projection: unsupported shape or rows");
    if (rows <= 1U) launch_bucket<1>(w, inputs, outputs, xh, out, rows, s);
    else if (rows <= 2U) launch_bucket<2>(w, inputs, outputs, xh, out, rows, s);
    else if (rows <= 4U) launch_bucket<4>(w, inputs, outputs, xh, out, rows, s);
    else if (rows <= 8U) launch_bucket<8>(w, inputs, outputs, xh, out, rows, s);
    else launch_bucket<16>(w, inputs, outputs, xh, out, rows, s);
}
inline void linear(const std::uint8_t* w, unsigned inputs, unsigned outputs, const float* x, _Float16* xh, float* out, unsigned rows,
                   hipStream_t s) {
    prepare(x, xh, inputs, rows, s);
    run(w, inputs, outputs, xh, out, rows, s);
}

}  // namespace strix::core::dense_iu4
#endif
