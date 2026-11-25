#pragma once
// Shared CUDA Gated DeltaNet recurrence in the lanes order (HIP gdn_prefill_lanes_kernel): token-serial
// FP32, G lanes per state column, tree reductions and no barrier per row.
// Row view: float pointers qkv / beta / gate / core with a uint64_t stride (floats between rows).
// The sequential sums and separate multiply/add in delta_body (gdn.cu) use a different exact order.
#include <cstdint>

#include <cuda_runtime.h>

#include "ops/recur/recur_decode_plan.hpp"
#include "strix/cuda/amd_math.cuh"
#include "strix/recurrent_state_type.hpp"
#include <cuda_fp16.h>
#include "../../attn_core/recurrent_state.hpp"

namespace strix::cuda_core::gdn_lanes {

namespace am = strix::cuda::amdmath;
namespace rstate = strix::attn_core::rstate;
inline constexpr unsigned kFull = 0xffffffffU;
// Preserve the incumbent lane binding while callers migrate to explicit layouts.
using DefaultLayout = ops::recur::GdnDecodeLayout<ops::recur::shape::gdn_2560_16x48x128>;
inline constexpr auto kKeyHeads = DefaultLayout::key_heads, kValueHeads = DefaultLayout::value_heads;
inline constexpr auto kHeadDim = DefaultLayout::head_dim, kKeyWidth = DefaultLayout::key_width;

// ---------------------------------------------------------------------------------------------------------------- K5
// gdn_prefill_lanes_kernel (:2896-2994, #pragma clang fp contract(off)). Lane `sub` of a G-lane column group owns keys
// sub + j*G. Per row, in this order: S = S*decay; p = S*k; pred = tree(p); delta = (v - pred)*beta; S = fma(k, delta, S);
// out = tree(S*q). Tree = the pairing of keys i, i+d for d = head_dim/2, ..., 1: in-lane j, j+half (keys differ
// half*G) for half = K/2..1, then xor G/2..1 inside the group (HIP: DPP row_xmask G/2..1, every lane gets the root).
// decay = expf(gate) (computed when the row is staged, :2951). Rows strictly ascending; state read once, written once.
// Mapping (CUDA): a block owns 32 columns of one value head (32*G threads); thread = column*G + sub.
// q/k of Chunk rows are staged in shared memory transposed by key residue (key[r][sub][j] = k[sub + j*G], pitch K+4 so
// the G distinct float4 reads of a warp hit distinct banks), v/beta/decay alongside; double-buffered through registers
// (the next chunk's global loads are in flight while the current chunk computes), as the HIP kernel stages them.
// decay = expf(gate) is computed by decay_kernel for all (row, head) before the recurrence (same value as the HIP commit-time
// expf; an fp64-based amd_expf inside the serial loop cost ~0.7 ms of 3.4 at T=8192). Rows are unrolled by 2 so the
// out tree of row r can overlap the delta chain of row r+1 (the arithmetic of each row is unchanged).
template <unsigned G, unsigned HeadDim = kHeadDim> struct RecShape {
    static constexpr unsigned Columns = 32, Threads = 32 * G, K = HeadDim / G, Pitch = K + 4, Chunk = G >= 16 ? 8 : 16;
    static constexpr unsigned KeyVectors = HeadDim / 4U, PairVectors = 2U * KeyVectors;
    static constexpr unsigned KQ = Chunk * PairVectors / Threads;                                  // float4 of k|q per thread
    static constexpr unsigned VQ = (Chunk * Columns / 4 + Threads - 1) / Threads;          // float4 of v per thread
};
template <unsigned G, unsigned HeadDim = kHeadDim> struct RecStage {
    using S = RecShape<G, HeadDim>;
    alignas(16) float key[S::Chunk][G][S::Pitch];
    alignas(16) float query[S::Chunk][G][S::Pitch];
    alignas(16) float value[S::Chunk][S::Columns];
    float beta[S::Chunk], decay[S::Chunk];
};

// Fused (decode): decays == nullptr and the staging fetch computes decay = amd_expf(gate) itself (the value decay_kernel stores).
// ST = stored element of the state (f32 = incumbent code; f16 / bf16: FP32 arithmetic, widened on load, narrowed on store).
// RR (16-bit only) = round the register state through the stored type after EVERY row (decode / window / commit replay: as
// if stored per token); prefill (RR false) keeps FP32 inside the call and narrows once at the end. Elements of one thread
// are G*head_dim apart, so the 16-bit access is scalar through the codec.
template <unsigned G, RecurrentStateType ST, bool RR, bool Fused = false, class Layout = DefaultLayout, typename RowView>
__device__ __forceinline__ void recurrence_body(const RowView& rows, std::uint32_t count, const float* state_in, float* state_out,
    const float* __restrict__ decays, const am::Tables* tables = nullptr) {
    using Sh = RecShape<G, Layout::head_dim>;
    constexpr unsigned Columns = Sh::Columns, Threads = Sh::Threads, K = Sh::K, Chunk = Sh::Chunk, KQ = Sh::KQ, VQ = Sh::VQ;
    constexpr unsigned KeyVectors = Sh::KeyVectors, PairVectors = Sh::PairVectors;
    static_assert(Chunk * PairVectors % Threads == 0 && 2 * Chunk <= Threads);
    __shared__ RecStage<G, Layout::head_dim> stage[2];
    const unsigned thread = threadIdx.x, sub = thread % G, local = thread / G;
    constexpr unsigned Blocks = Layout::head_dim / Columns;
    const unsigned value_head = blockIdx.x / Blocks, column_base = blockIdx.x % Blocks * Columns, column = column_base + local;
    const unsigned key_offset = value_head % Layout::key_heads * Layout::head_dim;  // value head h reads key head h mod key_heads
    const unsigned value_offset = 2U * Layout::key_width + value_head * Layout::head_dim + column_base;
    float4 kq[KQ], vv[VQ];
    float scalar = 0.0F;
    const auto fetch = [&](unsigned first) {
#pragma unroll
        for (unsigned i = 0; i < KQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / PairVectors, w = q % PairVectors;
            kq[i] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (first + r < count) {
                const float* row = rows.qkv + static_cast<std::uint64_t>(first + r) * rows.stride + key_offset;
                kq[i] = *reinterpret_cast<const float4*>(w < KeyVectors ? row + Layout::key_width + w * 4U : row + (w - KeyVectors) * 4U);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < VQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / (Columns / 4U);
            if (q < Chunk * Columns / 4U && first + r < count)
                vv[i] = *reinterpret_cast<const float4*>(rows.qkv + static_cast<std::uint64_t>(first + r) * rows.stride +
                    value_offset + q % (Columns / 4U) * 4U);
        }
        if (thread < 2U * Chunk && first + thread % Chunk < count)
            scalar = thread < Chunk ? rows.beta[static_cast<std::uint64_t>(first + thread) * rows.stride + value_head]
                   : Fused ? am::amd_expf(*tables, rows.gate[static_cast<std::uint64_t>(first + thread - Chunk) * rows.stride + value_head])
                           : decays[(first + thread - Chunk) * Layout::value_heads + value_head];
    };
    const auto commit = [&](RecStage<G, Layout::head_dim>& s) {
#pragma unroll
        for (unsigned i = 0; i < KQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / PairVectors, w = q % PairVectors, key = (w % KeyVectors) * 4U;
            float (&target)[G][Sh::Pitch] = w < KeyVectors ? s.key[r] : s.query[r];
            const float parts[4] = {kq[i].x, kq[i].y, kq[i].z, kq[i].w};
#pragma unroll
            for (unsigned c = 0; c < 4U; ++c) target[(key + c) % G][(key + c) / G] = parts[c];
        }
#pragma unroll
        for (unsigned i = 0; i < VQ; ++i) {
            const unsigned q = thread + i * Threads;
            if (q < Chunk * Columns / 4U) *reinterpret_cast<float4*>(&s.value[q / (Columns / 4U)][q % (Columns / 4U) * 4U]) = vv[i];
        }
        if (thread < Chunk) s.beta[thread] = scalar;
        else if (thread < 2U * Chunk) s.decay[thread - Chunk] = scalar;
    };
    float S[K];
#pragma unroll
    for (unsigned j = 0; j < K; ++j) {
        const std::uint64_t at = (static_cast<std::uint64_t>(value_head) * Layout::head_dim + sub + j * G) * Layout::head_dim + column;
        if constexpr (ST == RecurrentStateType::f32) S[j] = state_in[at];
        else S[j] = rstate::Codec<ST>::load(reinterpret_cast<const std::uint16_t*>(state_in)[at]);
    }
    const unsigned chunks = (count + Chunk - 1U) / Chunk;
    fetch(0U);
    commit(stage[0]);
    __syncthreads();
    for (unsigned c = 0; c < chunks; ++c) {
        const bool more = c + 1U < chunks;
        if (more) fetch((c + 1U) * Chunk);
        const RecStage<G, Layout::head_dim>& s = stage[c & 1U];
        const unsigned first = c * Chunk, n = min(Chunk, count - first);
        float* core = rows.core + static_cast<std::uint64_t>(first) * rows.stride + value_head * Layout::head_dim + column;
#pragma unroll 2
        for (unsigned r = 0; r < n; ++r) {
            const float4* key = reinterpret_cast<const float4*>(s.key[r][sub]);
            const float4* query = reinterpret_cast<const float4*>(s.query[r][sub]);
            const float decay = s.decay[r], beta = s.beta[r], value = s.value[r][local];
            float p[K];
#pragma unroll
            for (unsigned m = 0; m < K / 4U; ++m) {
                const float4 k4 = key[m];
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
                for (unsigned c4 = 0; c4 < 4U; ++c4) {
                    const unsigned j = 4U * m + c4;
                    S[j] = __fmul_rn(S[j], decay);
                    p[j] = __fmul_rn(S[j], kk[c4]);
                }
            }
#pragma unroll
            for (unsigned half = K / 2U; half; half /= 2U)
#pragma unroll
                for (unsigned j = 0; j < half; ++j) p[j] = __fadd_rn(p[j], p[j + half]);
            float pred = p[0];
#pragma unroll
            for (unsigned m = G / 2U; m; m /= 2U) pred = __fadd_rn(pred, __shfl_xor_sync(kFull, pred, m));
            const float delta = __fmul_rn(__fsub_rn(value, pred), beta);
#pragma unroll
            for (unsigned m = 0; m < K / 4U; ++m) {
                const float4 k4 = key[m], q4 = query[m];
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                for (unsigned c4 = 0; c4 < 4U; ++c4) {
                    const unsigned j = 4U * m + c4;
                    S[j] = __fmaf_rn(kk[c4], delta, S[j]);
                    p[j] = __fmul_rn(S[j], qq[c4]);
                }
            }
#pragma unroll
            for (unsigned half = K / 2U; half; half /= 2U)
#pragma unroll
                for (unsigned j = 0; j < half; ++j) p[j] = __fadd_rn(p[j], p[j + half]);
            float out = p[0];
#pragma unroll
            for (unsigned m = G / 2U; m; m /= 2U) out = __fadd_rn(out, __shfl_xor_sync(kFull, out, m));
            if (sub == 0U) *core = out;
            core += rows.stride;
            if constexpr (RR && ST != RecurrentStateType::f32) {
#pragma unroll
                for (unsigned j = 0; j < K; ++j) S[j] = rstate::round_in_register<ST>(S[j]);
            }
        }
        if (more) commit(stage[(c + 1U) & 1U]);
        __syncthreads();
    }
#pragma unroll
    for (unsigned j = 0; j < K; ++j) {
        const std::uint64_t at = (static_cast<std::uint64_t>(value_head) * Layout::head_dim + sub + j * G) * Layout::head_dim + column;
        if constexpr (ST == RecurrentStateType::f32) state_out[at] = S[j];
        else reinterpret_cast<std::uint16_t*>(state_out)[at] = rstate::Codec<ST>::store(S[j]);
    }
}

}  // namespace strix::cuda_core::gdn_lanes
