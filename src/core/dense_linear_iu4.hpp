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
//     rows per weight sweep from strix/decode_weight_sweep.hpp (every weight word read once per sweep). Scalar
//     activations (default): one sweep of up to 16 rows, the activation words loaded in 8-row chunks (SGPR budget);
//     the 16-row bucket used to run two 8-row sweeps and stream the weights twice. Vector activations: 8 rows per
//     sweep (one 16-row sweep in that form measured slower, 272.8 vs 220.7 ms, and spilled 144 B). Bitwise equal to
//     the chunked form either way (per-row op order is identical).
//     STRIX_IU4_ROW_REUSE=0 selects the old 4-row chunk calls (weights re-streamed per chunk).
// Home: shared (backend core, HIP): no model names; any dense projection can bind it (qwen35 now).
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>

#include "strix/decode_projection_group.hpp"
#include "strix/decode_weight_sweep.hpp"
#include "strix/env_flags.hpp"
#include "dense_iu4_rows.hpp"
#include "iu4_x_pairs.hpp"
#include "w4a16.hpp"

namespace strix::core::dense_iu4 {

inline constexpr unsigned kWaves = 8U;
inline constexpr unsigned kMaxRows = dense_iu4_rows::kMaxRows;
inline constexpr unsigned kWideRows = dense_iu4_rows::kWideRows;
inline constexpr std::size_t wide_xh_rows(std::size_t rows) { return dense_iu4_rows::xh_rows(rows); }

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
// The same conversion (expression of the two kernels above) for producer epilogues.
__device__ __forceinline__ _Float16 to_f16_value(float v) {
    return static_cast<_Float16>(v != v ? v : fminf(65504.0F, fmaxf(-65504.0F, v)));
}

// One block's 64 output columns (quad) of one tensor; kernel and group_kernel below differ only in how they pick
// (tensor, quad) from blockIdx.
// Grid2 (exact wide windows, rows > 16; kernel only): block (x, y) = rows [x R, x R + R) of the 64-column block y. Same
// instruction stream per block as the 1-D form (only the row / column offsets come from the block index); the R-row blocks
// of one column block are adjacent in dispatch order, so they read its weight slice while it is L2 / MALL resident instead
// of re-streaming it from DRAM once per R rows. Stored rows are bitwise those of run() with <= 16 rows (row independence
// above); xh must hold the rows rounded up to R (pad rows are read, never stored).
// Share = true (cols_kernel only): store at (out_stride, out_quad) instead of (outputs, quad); false compiles to the
// full-launch store (the share parameters are dead).
template <unsigned R, unsigned MaxG, unsigned Rc, bool UX, bool Grid2 = false, bool Share = false>
__device__ __forceinline__ void block_body(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh, float* out,
                                           unsigned nrows, unsigned quad, unsigned out_stride, unsigned out_quad) {
    constexpr unsigned S = kWaves, KT = MaxG * 4U;
    __shared__ float part[R * S * 64];
    if constexpr (Grid2) {
        static_assert(R == dense_iu4_rows::kWideRows, "the wide grid plan owns the row block size");
        const auto b = dense_iu4_rows::block(blockIdx.x, nrows);
        xh += std::size_t(b.row0) * inputs;
        out += std::size_t(b.row0) * outputs;
        nrows = b.rows;
    }
    // UX: wave index made wave-uniform (SGPR) so the activation reads of w4a16::wave_rows<..., UniformX> are scalar loads.
    const unsigned lane = threadIdx.x & 31U;
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
        if constexpr (Share) out[std::size_t(r) * out_stride + out_quad * 64U + c] = sum;
        else out[std::size_t(r) * outputs + quad * 64U + c] = sum;
    }
}

template <unsigned R, unsigned MaxG, unsigned Rc = R, bool UX = false, bool Grid2 = false>
__global__ void __launch_bounds__(32 * kWaves) kernel(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh,
                                                      float* out, unsigned nrows) {
    const unsigned quad = Grid2 ? blockIdx.y : blockIdx.x;
    block_body<R, MaxG, Rc, UX, Grid2>(w, inputs, outputs, xh, out, nrows, quad, outputs, quad);
}

// Column share (exact tensor parallelism, rows 1..16): the launch covers column blocks [quad0, quad0 + gridDim.x) of the
// N = outputs matrix and stores column quad0 * 64 + j into out[r * out_stride + j]. A column's value depends only on the
// pack, N, K, xh and the column (K split and LDS reduction order are per block, block_body), so a share holds bitwise
// the columns of kernel's full launch. kernel and group_kernel are unchanged (they store at their own quad, stride N).
template <unsigned R, unsigned MaxG, unsigned Rc = R, bool UX = false>
__global__ void __launch_bounds__(32 * kWaves) cols_kernel(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh,
                                                           float* out, unsigned nrows, unsigned quad0, unsigned out_stride) {
    block_body<R, MaxG, Rc, UX, false, true>(w, inputs, outputs, xh, out, nrows, quad0 + blockIdx.x, out_stride, blockIdx.x);
}

// Projection group (decode step plan, strix/decode_projection_group.hpp): projections of the same prepared rows xh
// (same inputs) in one launch; block b -> member m, quad b - first[m] (first[] in 64-column blocks). Same block body.
inline constexpr unsigned kGroupMax = 4U;
struct Group {
    const std::uint8_t* w[kGroupMax];
    float* out[kGroupMax];
    unsigned outputs[kGroupMax];
    std::uint32_t first[kGroupMax + 1U];
    unsigned count;
};
template <unsigned R, unsigned MaxG, unsigned Rc = R, bool UX = false>
__global__ void __launch_bounds__(32 * kWaves) group_kernel(Group g, unsigned inputs, const _Float16* xh, unsigned nrows) {
    unsigned m;
    std::uint32_t quad;
    ::strix::decode::locate(g.first, g.count, blockIdx.x, m, quad);  // blockIdx-uniform: scalar
    block_body<R, MaxG, Rc, UX>(g.w[m], inputs, g.outputs[m], xh, g.out[m], nrows, quad, g.outputs[m], quad);
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
// STRIX_IU4_ONE_SWEEP: 1 (default) = scalar-activation windows of 9..16 rows read the weights once (one 16-row sweep,
// strix/decode_weight_sweep.hpp), 0 = the previous two 8-row sweeps. Same rows bitwise either way; rows <= 8 unaffected.
inline bool one_sweep() {
    static const bool on = envcfg::flag("STRIX_IU4_ONE_SWEEP", true);  // recorded in the effective config
    return on;
}

// The template instance of rows bucket R for `inputs`: f.template operator()<MaxG, Rc, UX>() launches it.
template <unsigned R, class F>
void dispatch_bucket(unsigned inputs, F&& f) {
    // Rows per weight sweep (strix/decode_weight_sweep.hpp): scalar activations 16 (one stream for any window), vector 8.
    constexpr unsigned C = R >= 8U ? 4U : R, M = ::strix::decode::sweep_rows(R, ::strix::decode::kHipIu4VectorX),
                       MX = ::strix::decode::sweep_rows(R, ::strix::decode::kHipIu4ScalarX);
    const bool mid = R <= 4U || row_reuse(), sx = scalar_x(), one = MX == M || one_sweep();
    const auto pick = [&]<unsigned G>() {
        if (sx && one) f.template operator()<G, MX, true>();
        else if (sx) f.template operator()<G, M, true>();
        else if (mid) f.template operator()<G, M, false>();
        else f.template operator()<G, C, false>();
    };
    switch (inputs / 512U) {
    case 10U: pick.template operator()<10U>(); break;
    case 12U: pick.template operator()<12U>(); break;
    default: pick.template operator()<34U>(); break;
    }
}
// Grid2 = false: one launch of the R bucket over all rows (rows <= 16). Grid2 = true (R = kWideRows): ceil(rows / R)
// row chunks per column block (block_body above).
template <unsigned R, bool Grid2 = false>
void launch_bucket(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh, float* out, unsigned rows, hipStream_t s) {
    const dim3 grid = Grid2 ? dim3(dense_iu4_rows::blocks(rows), outputs / 64U) : dim3(outputs / 64U), block(32U * kWaves);
    dispatch_bucket<R>(inputs, [&]<unsigned G, unsigned Rc, bool UX>() {
        STRIX_HIP_LAUNCH(Rdna3Layout, kernel<R, G, Rc, UX, Grid2><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows));
    });
}
template <unsigned R>
void launch_cols_bucket(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh, float* out, unsigned rows,
                        hipStream_t s, unsigned quad0, unsigned quads, unsigned out_stride) {
    const dim3 grid(quads), block(32U * kWaves);
    dispatch_bucket<R>(inputs, [&]<unsigned G, unsigned Rc, bool UX>() {
        STRIX_HIP_LAUNCH(Rdna3Layout, cols_kernel<R, G, Rc, UX><<<grid, block, 0, s>>>(w, inputs, outputs, xh, out, rows, quad0, out_stride));
    });
}
template <unsigned R>
void launch_group_bucket(const Group& g, unsigned inputs, const _Float16* xh, unsigned rows, hipStream_t s) {
    const dim3 grid(g.first[g.count]), block(32U * kWaves);
    dispatch_bucket<R>(inputs, [&]<unsigned G, unsigned Rc, bool UX>() {
        STRIX_HIP_LAUNCH(Rdna3Layout, group_kernel<R, G, Rc, UX><<<grid, block, 0, s>>>(g, inputs, xh, rows));
    });
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
    STRIX_HIP_LAUNCH(Rdna3Layout, reference_kernel<0><<<(outputs + 127U) / 128U, 128, 0, s>>>(w, inputs, outputs, x, out, rows));
}

// prepare() as a producer epilogue (decode launch fusion): a kernel that computes x (a norm, a SiLU product) calls
// store(i, x[i]) with the F32 value it stores, i = row * inputs + k; same conversion and layout as prepare(), so run()
// reads the same bytes and the prepare launch is gone. One sweep of rows <= kMaxRows (xh holds those rows).
struct PreparedStore {
    _Float16* h;
    bool paired;
    __device__ void operator()(std::size_t i, float v) const { h[paired ? iu4_pairs::paired_index(i) : i] = to_f16_value(v); }
};
inline PreparedStore prepared_store(_Float16* xh) { return {xh, scalar_x()}; }
// Converts the F32 activations once; several projections of the same x (q/k/v, gate/up) share the prepared rows.
inline void prepare(const float* x, _Float16* xh, unsigned inputs, unsigned rows, hipStream_t s) {
    if (scalar_x()) STRIX_HIP_LAUNCH(Rdna3Layout, to_f16_paired_kernel<0><<<64, 256, 0, s>>>(x, xh, std::size_t(rows) * inputs));
    else STRIX_HIP_LAUNCH(Rdna3Layout, to_f16_kernel<0><<<64, 256, 0, s>>>(x, xh, std::size_t(rows) * inputs));
}
// xh: FP16 rows from prepare(); out: F32 [rows][outputs]. rows 1..16: the decode / verify buckets (unchanged). rows > 16
// (exact wide prefill windows): one Grid2 launch of kWideRows-row blocks (dense_iu4_rows.hpp); xh must hold
// wide_xh_rows(rows) rows.
inline void run(const std::uint8_t* w, unsigned inputs, unsigned outputs, const _Float16* xh, float* out, unsigned rows, hipStream_t s) {
    if (!supported(outputs, inputs) || rows == 0U) throw std::runtime_error("iu4 dense projection: unsupported shape or rows");
    if (rows > kMaxRows) launch_bucket<kWideRows, true>(w, inputs, outputs, xh, out, rows, s);
    else if (rows <= 1U) launch_bucket<1>(w, inputs, outputs, xh, out, rows, s);
    else if (rows <= 2U) launch_bucket<2>(w, inputs, outputs, xh, out, rows, s);
    else if (rows <= 4U) launch_bucket<4>(w, inputs, outputs, xh, out, rows, s);
    else if (rows <= 8U) launch_bucket<8>(w, inputs, outputs, xh, out, rows, s);
    else launch_bucket<16>(w, inputs, outputs, xh, out, rows, s);
}
// Columns [first, first + count) of the N = outputs projection (first, count multiples of 64; rows 1..16): a
// tensor-parallel share. Reads only those columns' tiles and scales; column first + j goes to out[r * out_stride + j]
// (rank-local compact share: out_stride = count; in place in a full-width buffer: out + first, out_stride = outputs).
// Bitwise the same columns as run() (cols_kernel).
inline void run_cols(const std::uint8_t* w, unsigned inputs, unsigned outputs, unsigned first, unsigned count, const _Float16* xh,
                     float* out, unsigned out_stride, unsigned rows, hipStream_t s) {
    if (!supported(outputs, inputs) || rows == 0U || rows > kMaxRows || first % 64U != 0U || count % 64U != 0U ||
        first > outputs || count > outputs - first || out_stride < count)
        throw std::runtime_error("iu4 dense projection: unsupported shape, rows or column range");
    if (count == 0U) return;
    const unsigned q0 = first / 64U, q = count / 64U;
    if (rows <= 1U) launch_cols_bucket<1>(w, inputs, outputs, xh, out, rows, s, q0, q, out_stride);
    else if (rows <= 2U) launch_cols_bucket<2>(w, inputs, outputs, xh, out, rows, s, q0, q, out_stride);
    else if (rows <= 4U) launch_cols_bucket<4>(w, inputs, outputs, xh, out, rows, s, q0, q, out_stride);
    else if (rows <= 8U) launch_cols_bucket<8>(w, inputs, outputs, xh, out, rows, s, q0, q, out_stride);
    else launch_cols_bucket<16>(w, inputs, outputs, xh, out, rows, s, q0, q, out_stride);
}
// Projections of the same prepared rows xh (one `inputs`) in ONE launch (decode step plan: projection group); rows 1..16.
// Every member's output is bitwise run()'s: same block body, operands and order. One member = run().
struct Member {
    const std::uint8_t* w;
    unsigned outputs;
    float* out;
};
inline void run_group(const Member* members, unsigned count, unsigned inputs, const _Float16* xh, unsigned rows, hipStream_t s) {
    if (rows == 0U || rows > kMaxRows) throw std::runtime_error("iu4 projection group: rows must be 1..16");
    Group g{};
    ::strix::decode::GroupIndex<kGroupMax> index;
    for (unsigned i = 0; i < count; ++i) {
        const Member& m = members[i];
        if (!supported(m.outputs, inputs)) throw std::runtime_error("iu4 projection group: unsupported member shape");
        g.w[index.count] = m.w;
        g.out[index.count] = m.out;
        g.outputs[index.count] = m.outputs;
        index.add(m.outputs / 64U);
    }
    if (index.count == 0U) throw std::runtime_error("iu4 projection group: no member");
    if (index.count == 1U) return run(g.w[0], inputs, g.outputs[0], xh, g.out[0], rows, s);
    for (unsigned i = 0; i <= index.count; ++i) g.first[i] = index.first[i];
    g.count = index.count;
    if (rows <= 1U) launch_group_bucket<1>(g, inputs, xh, rows, s);
    else if (rows <= 2U) launch_group_bucket<2>(g, inputs, xh, rows, s);
    else if (rows <= 4U) launch_group_bucket<4>(g, inputs, xh, rows, s);
    else if (rows <= 8U) launch_group_bucket<8>(g, inputs, xh, rows, s);
    else launch_group_bucket<16>(g, inputs, xh, rows, s);
}
inline void run_group(std::initializer_list<Member> members, unsigned inputs, const _Float16* xh, unsigned rows, hipStream_t s) {
    run_group(members.begin(), static_cast<unsigned>(members.size()), inputs, xh, rows, s);
}
inline void linear(const std::uint8_t* w, unsigned inputs, unsigned outputs, const float* x, _Float16* xh, float* out, unsigned rows,
                   hipStream_t s) {
    prepare(x, xh, inputs, rows, s);
    run(w, inputs, outputs, xh, out, rows, s);
}

}  // namespace strix::core::dense_iu4
#endif
