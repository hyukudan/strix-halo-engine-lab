// Gated DeltaNet decode with deferred commit: causal depthwise conv (k=4) + SiLU, l2norm q/k, delta rule with sequential rows
// into a shadow state side, gated RMSNorm. See gdn.hpp for the contract.
#include "strix/cuda_core/gdn.hpp"
#include "core/short_convolution.hpp"

#include <cmath>

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

// 128-thread block sum with the documented 256-thread order (warps 4..7 would hold zeros): butterfly per warp, then 0..3.
__device__ __forceinline__ float block_sum128(float v, float* smem) {
    v = warp_sum_xor(v);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) smem[w] = v;
    __syncthreads();
    float s = __fadd_rn(__fadd_rn(__fadd_rn(smem[0], smem[1]), smem[2]), smem[3]);
    __syncthreads();
    return s;
}

// Rows [r, r + n) of channel c, n <= B (block-uniform; red holds >= B rows), shared by the decode window and prefill kernels.
// Per row the operations and their order are the serial loop's: tap order oldest first ((0 + w0 s0) + w1 s1) + w2 s2) + w3 x,
// silu, and for q/k heads block_sum128's tree (butterfly, then warps 0..3 in order) of y*y, inv, (y * inv) * scale. Only the
// issue order changes: every row's raw value is loaded before the first use (one round trip instead of n dependent ones) and
// the n head sums share one barrier pair (instead of n). s0..s2 carry the taps to the next batch.
constexpr int kConvBatch = 16;  // decode window rows per batch (prefill blocks use 8: no spill at its address terms)
// TREE (token-tree window, strix/spec/tree_window.hpp): a row whose bit is set in `leaf` (window rows < 32) computes its output
// from the current taps and does not shift them (the next row's parent is the last non-leaf row).
template <int B, bool TREE = false>
__device__ __forceinline__ void conv_rows(float w0, float w1, float w2, float w3, float& s0, float& s1, float& s2,
                                          const float* __restrict__ raw, float* qc, std::uint32_t channels, std::uint32_t c,
                                          std::uint32_t r, std::uint32_t n, bool qk, float scale, float (*red)[4],
                                          std::uint32_t leaf = 0) {
    float xv[B], yv[B];
#pragma unroll
    for (int i = 0; i < B; ++i)
        xv[i] = static_cast<std::uint32_t>(i) < n ? raw[static_cast<std::size_t>(r + static_cast<std::uint32_t>(i)) * channels + c] : 0.0f;
    // Rows go in groups of G with no branch inside a group, so the independent per-row chains (silu, the 5-step butterfly,
    // sqrt/div) interleave; rows i >= n of the last group compute on xv = 0, never shift the taps and are never stored.
    constexpr int G = 4;
    static_assert(B % G == 0, "row batch is whole groups");
#pragma unroll
    for (int g0 = 0; g0 < B; g0 += G) {
        if (static_cast<std::uint32_t>(g0) >= n) break;
#pragma unroll
        for (int i = g0; i < g0 + G; ++i) {
            const float samples[] = {s0, s1, s2, xv[i]};
            const float weights[] = {w0, w1, w2, w3};
            const float sum = core::causal_conv::weighted_taps<true, true>(4,
                [&](int tap) { return weights[tap]; }, [&](int tap) { return samples[tap]; });
            yv[i] = cc_silu(sum);
            bool on = static_cast<std::uint32_t>(i) < n;
            if constexpr (TREE) on = on && ((leaf >> (r + static_cast<std::uint32_t>(i))) & 1u) == 0u;
            s0 = on ? s1 : s0;
            s1 = on ? s2 : s1;
            s2 = on ? xv[i] : s2;
        }
    }
    if (qk) {  // block-uniform: every thread reaches both barriers
        const int w = static_cast<int>(threadIdx.x >> 5);
#pragma unroll
        for (int g0 = 0; g0 < B; g0 += G) {
            if (static_cast<std::uint32_t>(g0) >= n) break;
#pragma unroll
            for (int i = g0; i < g0 + G; ++i) {
                const float v = warp_sum_xor(__fmul_rn(yv[i], yv[i]));  // p = 0 + y*y is exact
                if ((threadIdx.x & 31) == 0) red[i][w] = v;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g0 = 0; g0 < B; g0 += G) {
            if (static_cast<std::uint32_t>(g0) >= n) break;
#pragma unroll
            for (int i = g0; i < g0 + G; ++i) {
                const float ss = __fadd_rn(__fadd_rn(__fadd_rn(red[i][0], red[i][1]), red[i][2]), red[i][3]);
                const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, 1e-6f)));
                yv[i] = __fmul_rn(__fmul_rn(yv[i], inv), scale);
            }
        }
        __syncthreads();  // red is reused by the next batch
    }
#pragma unroll
    for (int i = 0; i < B; ++i)
        if (static_cast<std::uint32_t>(i) < n) qc[static_cast<std::size_t>(r + static_cast<std::uint32_t>(i)) * channels + c] = yv[i];
}

// Block = 128 channels. Blocks [0, qk_blocks) are q/k heads (conv + silu + l2norm); the rest are v channels (conv + silu).
// tap order oldest first: ((0 + w0 s0) + w1 s1) + w2 s2) + w3 x. The conv state is read-only here.
template <bool TREE>
__global__ void __launch_bounds__(128) conv_pre_kernel(const float* __restrict__ w, const float* __restrict__ conv_state,
                                                       const float* __restrict__ raw, float* qc, std::uint32_t channels,
                                                       std::uint32_t rows, std::uint32_t q_blocks, std::uint32_t qk_blocks,
                                                       float qscale, GdnSlotTable st, const std::uint32_t* leafp) {
    __shared__ float red[kConvBatch][4];
    if (st.row0) {  // multi-session: this block's group (blockIdx.y)
        pdl_wait();
        const std::uint32_t g = blockIdx.y;
        rows = st.nrows[g];
        conv_state += static_cast<std::size_t>(st.state_slot[g]) * st.conv_stride;
        raw += static_cast<std::size_t>(st.row0[g]) * channels;
        qc += static_cast<std::size_t>(st.row0[g]) * channels;
    }
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float s0 = conv_state[c * 3], s1 = conv_state[c * 3 + 1], s2 = conv_state[c * 3 + 2];  // committed state: written only outside the graph
    pdl_wait();
    pdl_trigger();  // after the wait: at most one dependent grid waits ahead of the running one
    std::uint32_t leaf = 0;
    if constexpr (TREE) leaf = *leafp;  // written by the window's input kernel (complete after the wait)
    const bool qk = blockIdx.x < qk_blocks;
    const float scale = blockIdx.x < q_blocks ? qscale : 1.0f;
    constexpr std::uint32_t kB = kConvBatch;
    for (std::uint32_t r = 0; r < rows; r += kB)
        conv_rows<kConvBatch, TREE>(w0, w1, w2, w3, s0, s1, s2, raw, qc, channels, c, r, min(kB, rows - r), qk, scale, red, leaf);
}

// Row-parallel form of conv_pre_kernel for prefill chunks (the same conv_rows body): block (128 channels, kConvRows rows).
// The conv taps of row r are raw rows r-3..r-1 (negative = committed conv state).
constexpr std::uint32_t kConvRows = 32;
__global__ void __launch_bounds__(128) conv_prefill_kernel(const float* __restrict__ w, const float* __restrict__ conv_state,
                                                           const float* __restrict__ raw, float* qc, std::uint32_t channels,
                                                           std::uint32_t rows, std::uint32_t q_blocks, std::uint32_t qk_blocks,
                                                           float qscale) {
    __shared__ float red[kConvBatch][4];
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const std::uint32_t r0 = blockIdx.y * kConvRows, r1 = min(rows, r0 + kConvRows);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float s[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = static_cast<int>(r0) - 3 + i;
        s[i] = idx >= 0 ? raw[static_cast<std::size_t>(idx) * channels + c] : conv_state[c * 3 + (idx + 3)];
    }
    float s0 = s[0], s1 = s[1], s2 = s[2];
    const bool qk = blockIdx.x < qk_blocks;
    const float scale = blockIdx.x < q_blocks ? qscale : 1.0f;
    for (std::uint32_t r = r0; r < r1; r += 8)
        conv_rows<8>(w0, w1, w2, w3, s0, s1, s2, raw, qc, channels, c, r, min(8u, r1 - r), qk, scale, red);
}

struct DeltaArgs {
    float* base;         // [2][heads][128][128]
    const float* qkvc;   // [rows][channels]
    const float* a_raw;  // [rows][heads]
    const float* b_raw;
    const float* ssm_a;
    const float* dt_bias;
    const float* gain;  // gated norm (window forward only)
    const float* z;     // [rows][v_width]
    float* out;         // [rows][v_width]
    float* conv;        // commit only
    const float* raw;   // commit only
};
struct CommitBatch {
    DeltaArgs a[kMaxGdnLayers];
};

// TREE (token-tree window / path replay, strix/spec/tree_window.hpp): a row whose bit is set in `leaf` does not change S. With OUT
// it runs the same row code (the bits a linear window gives that row) between a backup of S to `bk` (shared memory) and its
// restore; without OUT (commit replay) the row is skipped.
template <bool OUT, bool TREE = false>
__device__ __forceinline__ void delta_body(const DeltaArgs& a, const GdnDims& d, std::uint32_t rows, std::uint32_t parity,
                                           std::uint32_t h, float eps, std::uint32_t leaf = 0,
                                           const std::uint32_t* leafp = nullptr, float* bk = nullptr) {
    constexpr int K = 128;
    __shared__ float ks[K], qs[K];
    __shared__ float red[4];
    const std::uint32_t t = threadIdx.x;
    const std::uint32_t kh = h % d.key_heads;
    const std::uint32_t channels = d.conv_channels();
    const std::size_t side = static_cast<std::size_t>(d.value_heads) * K * d.value_dim;
    const float* sp = a.base + parity * side + static_cast<std::size_t>(h) * K * d.value_dim + t;
    float* dp = a.base + (1 - parity) * side + static_cast<std::size_t>(h) * K * d.value_dim + t;
    float S[K];
    pdl_trigger();
#pragma unroll
    for (int k = 0; k < K; ++k) S[k] = sp[static_cast<std::size_t>(k) * d.value_dim];
    pdl_wait();
    if constexpr (TREE) {
        if (leafp) leaf = *leafp;  // device mask written by the window's input kernel (complete after the wait)
    }
    const float dtb = a.dt_bias[h], sa = a.ssm_a[h];
    // row inputs are prefetched one row ahead (loads only: the arithmetic and its order are unchanged)
    float c_ks = 0.f, c_qs = 0.f, c_v = 0.f, c_b = 0.f, c_a = 0.f, c_z = 0.f;
    auto fetch = [&](std::uint32_t r) {
        const float* row = a.qkvc + static_cast<std::size_t>(r) * channels;
        c_ks = row[d.qk_width() + kh * K + t];
        if constexpr (OUT) {
            c_qs = row[kh * K + t];
            c_z = a.z[static_cast<std::size_t>(r) * d.v_width() + h * d.value_dim + t];
        }
        c_v = row[2 * d.qk_width() + h * d.value_dim + t];
        c_b = a.b_raw[static_cast<std::size_t>(r) * d.value_heads + h];
        c_a = a.a_raw[static_cast<std::size_t>(r) * d.value_heads + h];
    };
    if (rows > 0) fetch(0);
    for (std::uint32_t r = 0; r < rows; ++r) {
        ks[t] = c_ks;
        if constexpr (OUT) qs[t] = c_qs;
        const float vcol = c_v, zv = c_z;
        const float beta = cc_sigmoid(c_b);
        const float xin = __fadd_rn(c_a, dtb);
        const float sp_ = xin > 20.0f ? xin : cc_log1pf(cc_expf(xin));
        const float g = __fmul_rn(sa, sp_);
        const float decay = cc_expf(g);
        __syncthreads();
        if (r + 1 < rows) fetch(r + 1);
        bool leaf_row = false;
        if constexpr (TREE) {
            leaf_row = ((leaf >> r) & 1u) != 0u;  // block-uniform
            if (leaf_row) {
                if constexpr (!OUT) {  // path replay: a skipped leaf
                    __syncthreads();
                    continue;
                } else {
                    // The leaf runs the row code below unchanged (same instructions, same bits) and S is restored after it.
                    // Keeping S intact instead (S*decay as new values) needs > 255 registers and spilled; the backup is
                    // 128 x 128 floats of dynamic shared memory (the window has one block per value head, <= 1 per SM).
#pragma unroll
                    for (int k = 0; k < K; ++k) bk[k * 128 + t] = S[k];
                }
            }
        }
        float kv = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            S[k] = __fmul_rn(S[k], decay);
            kv = __fadd_rn(kv, __fmul_rn(S[k], ks[k]));
        }
        const float dl = __fmul_rn(__fsub_rn(vcol, kv), beta);
        // Register budget: S[] already holds 128 registers. Without this fence the compiler keeps the 128 ks[] values loaded by the
        // kv loop live into the update loop (256 values > 255 registers) and spills inside this serial per-row loop. The fence makes
        // the update loop re-read ks[] from shared memory (same values, no store in between): loads only, arithmetic order unchanged.
        asm volatile("" ::: "memory");
        float acc = 0.0f;
        if constexpr (OUT) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                S[k] = __fadd_rn(S[k], __fmul_rn(ks[k], dl));
                acc = __fadd_rn(acc, __fmul_rn(S[k], qs[k]));
            }
            const float ss = block_sum128(__fmul_rn(acc, acc), red);
            const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, 128.0f), eps)));
            float v = __fmul_rn(acc, inv);
            v = __fmul_rn(v, a.gain[t]);
            v = __fmul_rn(v, cc_silu(zv));
            a.out[static_cast<std::size_t>(r) * d.v_width() + h * d.value_dim + t] = v;
            if constexpr (TREE) {
                if (leaf_row) {
#pragma unroll
                    for (int k = 0; k < K; ++k) S[k] = bk[k * 128 + t];
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < K; ++k) S[k] = __fadd_rn(S[k], __fmul_rn(ks[k], dl));
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < K; ++k) dp[static_cast<std::size_t>(k) * d.value_dim] = S[k];
}

// Multi-session adjustment of one group: offsets the row buffers and the state base, returns the group's rows and parity.
__device__ __forceinline__ void slot_adjust(DeltaArgs& a, const GdnDims& d, const GdnSlotTable& st, std::uint32_t g,
                                            std::uint32_t& rows, std::uint32_t& parity) {
    const std::size_t r0 = st.row0[g];
    const std::uint32_t ss = st.state_slot[g];
    rows = st.nrows[g];
    parity = st.parity[ss];
    a.base += static_cast<std::size_t>(ss) * st.delta_stride;
    a.qkvc += r0 * d.conv_channels();
    a.a_raw += r0 * d.value_heads;
    a.b_raw += r0 * d.value_heads;
    if (a.z) a.z += r0 * d.v_width();
    if (a.out) a.out += r0 * d.v_width();
}

__global__ void __launch_bounds__(128) delta_window_kernel(DeltaArgs a, GdnDims d, std::uint32_t rows,
                                                           const std::uint32_t* parity, float eps, GdnSlotTable st) {
    if (st.row0) {
        pdl_wait();
        std::uint32_t par = 0;
        slot_adjust(a, d, st, blockIdx.y, rows, par);
        delta_body<true>(a, d, rows, par, blockIdx.x, eps);
        return;
    }
    delta_body<true>(a, d, rows, *parity, blockIdx.x, eps);
}

__global__ void __launch_bounds__(128) delta_window_tree_kernel(DeltaArgs a, GdnDims d, std::uint32_t rows,
                                                                const std::uint32_t* parity, float eps, const std::uint32_t* leafp) {
    extern __shared__ float leaf_backup[];  // [128 k][128 t]: S of a leaf row's parent (delta_body TREE)
    delta_body<true, true>(a, d, rows, *parity, blockIdx.x, eps, 0u, leafp, leaf_backup);
}

// Partial accept for multi-session: re-apply the first accept[g] rows (0 < accept < nrows) of group g for layer blockIdx.y.
__global__ void __launch_bounds__(128) delta_apply_slots_kernel(CommitBatch b, GdnDims d, GdnSlotTable st, const std::uint32_t* accept) {
    const std::uint32_t g = blockIdx.z;
    const std::uint32_t k = accept[g];
    if (k == 0 || k >= st.nrows[g]) return;
    DeltaArgs a = b.a[blockIdx.y];
    std::uint32_t rows = 0, par = 0;
    slot_adjust(a, d, st, g, rows, par);
    delta_body<false>(a, d, k, par, blockIdx.x, 0.0f);
}

__global__ void __launch_bounds__(128) conv_commit_slots_kernel(CommitBatch b, std::uint32_t channels, GdnSlotTable st,
                                                                const std::uint32_t* accept) {
    const std::uint32_t g = blockIdx.z;
    const std::uint32_t k = accept ? accept[g] : st.nrows[g];
    if (k == 0) return;
    const DeltaArgs& a = b.a[blockIdx.y];
    float* conv = a.conv + static_cast<std::size_t>(st.state_slot[g]) * st.conv_stride;
    const float* raw = a.raw + static_cast<std::size_t>(st.row0[g]) * channels;
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const float s[3] = {conv[c * 3], conv[c * 3 + 1], conv[c * 3 + 2]};
    float n[3];
    for (int i = 0; i < 3; ++i) {
        const int idx = static_cast<int>(k) - 3 + i;
        n[i] = idx >= 0 ? raw[static_cast<std::size_t>(idx) * channels + c] : s[idx + 3];
    }
    conv[c * 3] = n[0];
    conv[c * 3 + 1] = n[1];
    conv[c * 3 + 2] = n[2];
}

__global__ void flip_slots_kernel(GdnSlotTable st, const std::uint32_t* accept) {
    const std::uint32_t g = threadIdx.x;
    const std::uint32_t k = accept ? accept[g] : st.nrows[g];
    if (k > 0) st.parity[st.state_slot[g]] ^= 1u;
}

__global__ void __launch_bounds__(128) delta_apply_kernel(CommitBatch b, GdnDims d, std::uint32_t rows,
                                                          const std::uint32_t* parity) {
    delta_body<false>(b.a[blockIdx.y], d, rows, *parity, blockIdx.x, 0.0f);
}

// Token-tree path replay: rows [0, rows) except the `skip` rows (leaves not on the accepted path).
__global__ void __launch_bounds__(128) delta_apply_tree_kernel(CommitBatch b, GdnDims d, std::uint32_t rows,
                                                               const std::uint32_t* parity, std::uint32_t skip) {
    delta_body<false, true>(b.a[blockIdx.y], d, rows, *parity, blockIdx.x, 0.0f, skip);
}

// conv state after accepting raw rows: the last three of (s0, s1, s2, accepted raw rows). src[i] = raw row index of the new tap
// i, or -3..-1 = old state tap src[i] + 3 (linear accept of k rows: src = k-3, k-2, k-1).
struct ConvSrc {
    int i[3];
};
__global__ void __launch_bounds__(128) conv_commit_kernel(CommitBatch b, std::uint32_t channels, ConvSrc src) {
    const DeltaArgs& a = b.a[blockIdx.y];
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const float s[3] = {a.conv[c * 3], a.conv[c * 3 + 1], a.conv[c * 3 + 2]};
    float n[3];
    for (int i = 0; i < 3; ++i) {
        const int idx = src.i[i];  // index into raw rows; negative = old state
        n[i] = idx >= 0 ? a.raw[static_cast<std::size_t>(idx) * channels + c] : s[idx + 3];
    }
    a.conv[c * 3] = n[0];
    a.conv[c * 3 + 1] = n[1];
    a.conv[c * 3 + 2] = n[2];
}

__global__ void flip_kernel(std::uint32_t* p) { *p ^= 1u; }

}  // namespace

cudaError_t gdn_window(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                       float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, cudaStream_t stream,
                       const std::uint32_t* leaf_dev) {
    if (d.key_dim != 128 || d.value_dim != 128 || d.conv_kernel != 4 || rows == 0 || rows > kMaxRows) return cudaErrorInvalidValue;
    const std::uint32_t C = d.conv_channels();
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.key_dim)));
    if (auto e = leaf_dev ? launch_pdl(conv_pre_kernel<true>, dim3(C / 128), dim3(128), stream, layer.conv_w, state.conv, win.raw,
                                       win.qkv_c, C, rows, d.key_heads, 2 * d.key_heads, qscale, GdnSlotTable{}, leaf_dev)
                          : launch_pdl(conv_pre_kernel<false>, dim3(C / 128), dim3(128), stream, layer.conv_w, state.conv, win.raw,
                                       win.qkv_c, C, rows, d.key_heads, 2 * d.key_heads, qscale, GdnSlotTable{},
                                       static_cast<const std::uint32_t*>(nullptr));
        e != cudaSuccess)
        return e;
    DeltaArgs a{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, layer.norm_gain, z, out,
                nullptr, nullptr};
    if (leaf_dev) {
        constexpr std::size_t kBackup = 128u * 128u * sizeof(float);  // 64 KiB > the 48 KiB default: opt in once
        static const cudaError_t attr = cudaFuncSetAttribute(delta_window_tree_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                             static_cast<int>(kBackup));
        if (attr != cudaSuccess) return attr;
        return launch_pdl_smem(delta_window_tree_kernel, dim3(d.value_heads), dim3(128), kBackup, stream, a, d, rows, parity_dev, eps,
                               leaf_dev);
    }
    return launch_pdl(delta_window_kernel, dim3(d.value_heads), dim3(128), stream, a, d, rows, parity_dev, eps, GdnSlotTable{});
}

cudaError_t gdn_window_slots(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                             float* out, std::uint32_t groups, const GdnSlotTable& t, float eps, cudaStream_t stream) {
    if (d.key_dim != 128 || d.value_dim != 128 || d.conv_kernel != 4 || groups == 0 || groups > kMaxRows || !t.row0) return cudaErrorInvalidValue;
    const std::uint32_t C = d.conv_channels();
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.key_dim)));
    if (auto e = launch_pdl(conv_pre_kernel<false>, dim3(C / 128, groups), dim3(128), stream, layer.conv_w, state.conv, win.raw, win.qkv_c, C,
                            0u, d.key_heads, 2 * d.key_heads, qscale, t, static_cast<const std::uint32_t*>(nullptr));
        e != cudaSuccess)
        return e;
    DeltaArgs a{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, layer.norm_gain, z, out,
                nullptr, nullptr};
    return launch_pdl(delta_window_kernel, dim3(d.value_heads, groups), dim3(128), stream, a, d, 0u, static_cast<const std::uint32_t*>(nullptr), eps, t);
}

cudaError_t gdn_commit_slots(const GdnDims& d, const GdnCommitLayer* layers, std::uint32_t n, std::uint32_t groups,
                             const GdnSlotTable& t, const std::uint32_t* accept, bool partial, cudaStream_t stream) {
    if (n == 0 || n > kMaxGdnLayers || groups == 0 || groups > kMaxRows || !t.row0 || (partial && !accept)) return cudaErrorInvalidValue;
    CommitBatch b{};
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& L = layers[i];
        b.a[i] = DeltaArgs{L.state.delta_base, L.win.qkv_c, L.win.a_raw, L.win.b_raw, L.layer.ssm_a, L.layer.dt_bias,
                           nullptr, nullptr, nullptr, L.state.conv, L.win.raw};
    }
    const std::uint32_t C = d.conv_channels();
    if (partial) {
        delta_apply_slots_kernel<<<dim3(d.value_heads, n, groups), 128, 0, stream>>>(b, d, t, accept);
        if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    }
    conv_commit_slots_kernel<<<dim3(C / 128, n, groups), 128, 0, stream>>>(b, C, t, accept);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    flip_slots_kernel<<<1, groups, 0, stream>>>(t, accept);
    return cudaGetLastError();
}

cudaError_t gdn_commit_path(const GdnDims& d, const GdnCommitLayer* layers, std::uint32_t n, std::uint32_t window_rows,
                            std::uint32_t end, std::uint32_t skip, bool replay, std::uint32_t* parity_dev, cudaStream_t stream) {
    if (n == 0 || n > kMaxGdnLayers || end > window_rows || window_rows > kMaxRows) return cudaErrorInvalidValue;
    if (end > 0 && (skip >> (end - 1) & 1u)) return cudaErrorInvalidValue;  // the path's last row is never skipped
    if (end == 0) return cudaSuccess;
    if (end < 32) skip &= (1u << end) - 1u;
    CommitBatch b{};
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& L = layers[i];
        b.a[i] = DeltaArgs{L.state.delta_base, L.win.qkv_c, L.win.a_raw, L.win.b_raw, L.layer.ssm_a, L.layer.dt_bias,
                           nullptr, nullptr, nullptr, L.state.conv, L.win.raw};
    }
    const std::uint32_t C = d.conv_channels();
    if (replay) {
        if (skip) delta_apply_tree_kernel<<<dim3(d.value_heads, n), 128, 0, stream>>>(b, d, end, parity_dev, skip);
        else delta_apply_kernel<<<dim3(d.value_heads, n), 128, 0, stream>>>(b, d, end, parity_dev);
        if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    }
    // new taps = the last three of (old taps -3..-1, accepted rows of [0, end) in order)
    ConvSrc src{{-3, -2, -1}};
    for (std::uint32_t r = 0; r < end; ++r) {
        if (skip >> r & 1u) continue;
        src.i[0] = src.i[1];
        src.i[1] = src.i[2];
        src.i[2] = static_cast<int>(r);
    }
    conv_commit_kernel<<<dim3(C / 128, n), 128, 0, stream>>>(b, C, src);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    flip_kernel<<<1, 1, 0, stream>>>(parity_dev);
    return cudaGetLastError();
}

cudaError_t gdn_commit(const GdnDims& d, const GdnCommitLayer* layers, std::uint32_t n, std::uint32_t window_rows,
                       std::uint32_t k, std::uint32_t* parity_dev, cudaStream_t stream) {
    if (n == 0 || n > kMaxGdnLayers || k > window_rows) return cudaErrorInvalidValue;
    if (k == 0) return cudaSuccess;
    return gdn_commit_path(d, layers, n, window_rows, k, 0u, k < window_rows, parity_dev, stream);
}

cudaError_t gdn_prefill_layer(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                              float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, cudaStream_t stream) {
    return gdn_prefill_layer(d, layer, state, win, z, out, rows, eps, parity_dev, GdnChunkWork{}, stream);
}

cudaError_t gdn_prefill_layer(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                              float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, const GdnChunkWork& chunk,
                              cudaStream_t stream) {
    if (d.key_dim != 128 || d.value_dim != 128 || d.conv_kernel != 4 || rows == 0) return cudaErrorInvalidValue;
    if (chunk.defer_norm && chunk.ws == nullptr) return cudaErrorInvalidValue;  // the serial kernel always applies the norm itself
    const std::uint32_t C = d.conv_channels();
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.key_dim)));
    conv_prefill_kernel<<<dim3(C / 128, (rows + kConvRows - 1) / kConvRows), 128, 0, stream>>>(
        layer.conv_w, state.conv, win.raw, win.qkv_c, C, rows, d.key_heads, 2 * d.key_heads, qscale);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    if (chunk.ws != nullptr) {  // FAST gdn-chunk16-f32-v1 (gdn_chunk.cu)
        if (auto e = gdn_chunk_delta(d, layer, state, win, z, out, rows, eps, parity_dev, chunk, stream); e != cudaSuccess) return e;
    } else {
        DeltaArgs a{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, layer.norm_gain, z, out,
                    nullptr, nullptr};
        delta_window_kernel<<<dim3(d.value_heads), 128, 0, stream>>>(a, d, rows, parity_dev, eps, GdnSlotTable{});
        if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    }
    CommitBatch b{};
    b.a[0] = DeltaArgs{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, nullptr, nullptr, nullptr, state.conv, win.raw};
    const int ir = static_cast<int>(rows);  // chunk end: the last three raw rows (negative = old taps)
    conv_commit_kernel<<<dim3(C / 128, 1), 128, 0, stream>>>(b, C, ConvSrc{{ir - 3, ir - 2, ir - 1}});
    return cudaGetLastError();
}

cudaError_t gdn_prefill_flip(std::uint32_t* parity_dev, cudaStream_t stream) {
    flip_kernel<<<1, 1, 0, stream>>>(parity_dev);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
