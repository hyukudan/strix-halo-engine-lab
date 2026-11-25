#pragma once
// Frozen fixture reference from 453b063b1. Keep independent of K9G decode.
// Donor: include/strix/cuda/qwen_gdn.hpp:24-46 and
// src/cuda/models/qwen/qwen_gdn.cu:48-65,189-202,260-269,283-432,437-476.
// The donor's shared recurrence is frozen here too (recur_lanes.cuh:39-186).
#include "strix/cuda/amd_math.cuh"
#include "strix/recurrent_state_type.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "attn_core/recurrent_state.hpp"

namespace strix::cuda::recur::test::frozen {

namespace am = ::strix::cuda::amdmath;
namespace rstate = ::strix::attn_core::rstate;
constexpr unsigned kFull = 0xffffffffU;

constexpr std::uint32_t kHidden = 2560, kKeyHeads = 16, kValueHeads = 48, kHeadDim = 128, kConvKernel = 4;
constexpr std::uint32_t kKeyWidth = kKeyHeads * kHeadDim;                          // 2048
constexpr std::uint32_t kConvWidth = 2U * kKeyWidth + kValueHeads * kHeadDim;      // 10240
constexpr std::uint32_t kValueWidth = kValueHeads * kHeadDim;                      // 6144
constexpr std::uint64_t kMatrixFloats = std::uint64_t{kValueHeads} * kHeadDim * kHeadDim;  // per layer, key_value
constexpr std::uint64_t kConvStateHalfs = std::uint64_t{kConvWidth} * kConvKernel;          // per layer, BF16 [ch][4]
constexpr std::uint32_t kConvChunk = 32;                                           // rows per conv block
constexpr std::uint32_t kA8Pitch = 256;                                            // fused A8 delta pitch (kDenseA8WideRows)

// A device pointer given directly, or read at kernel start from a device word (`slot`, e.g. an entry of a per-segment
// descriptor table written before a graph replay) plus `offset` elements. Only where the pointer comes from changes; the
// arithmetic is the same.
template <class T> struct DevRef {
    T* direct = nullptr;
    T* const* slot = nullptr;
    std::uint64_t offset = 0;
    __host__ __device__ T* get() const { return slot ? *slot + offset : direct; }
};

struct Rows {
    float* qkv; float* z; float* beta; float* gate; float* core;
    std::uint64_t stride;  // floats between consecutive rows (same for every field)
};

template <unsigned G> struct RecShape {
    static constexpr unsigned Columns = 32, Threads = 32 * G, K = kHeadDim / G, Pitch = K + 4, Chunk = G >= 16 ? 8 : 16;
    static constexpr unsigned KQ = Chunk * 64 / Threads;                                  // float4 of k|q per thread
    static constexpr unsigned VQ = (Chunk * Columns / 4 + Threads - 1) / Threads;          // float4 of v per thread
};
template <unsigned G> struct RecStage {
    using S = RecShape<G>;
    alignas(16) float key[S::Chunk][G][S::Pitch];
    alignas(16) float query[S::Chunk][G][S::Pitch];
    alignas(16) float value[S::Chunk][S::Columns];
    float beta[S::Chunk], decay[S::Chunk];
};

// Fused (decode): decays == nullptr and the staging fetch computes decay = amd_expf(gate) itself (the value decay_kernel stores).
// ST = stored element of the state (f32 = incumbent code; f16 / bf16: FP32 arithmetic, widened on load, narrowed on store).
// RR (16-bit only) = round the register state through the stored type after EVERY row (decode / window / commit replay: as
// if stored per token); prefill (RR false) keeps FP32 inside the call and narrows once at the end. Elements of one thread
// are G*128 apart, so the 16-bit access is scalar through the codec.
template <unsigned G, RecurrentStateType ST, bool RR, bool Fused = false, typename RowView>
__device__ __forceinline__ void recurrence_body(const RowView& rows, std::uint32_t count, const float* state_in, float* state_out,
    const float* __restrict__ decays, const am::Tables* tables = nullptr) {
    using Sh = RecShape<G>;
    constexpr unsigned Columns = Sh::Columns, Threads = Sh::Threads, K = Sh::K, Chunk = Sh::Chunk, KQ = Sh::KQ, VQ = Sh::VQ;
    static_assert(Chunk * 64 % Threads == 0 && 2 * Chunk <= Threads);
    __shared__ RecStage<G> stage[2];
    const unsigned thread = threadIdx.x, sub = thread % G, local = thread / G;
    constexpr unsigned Blocks = kHeadDim / Columns;
    const unsigned value_head = blockIdx.x / Blocks, column_base = blockIdx.x % Blocks * Columns, column = column_base + local;
    const unsigned key_offset = value_head % kKeyHeads * kHeadDim;  // value head h reads key head h mod 16 (:2922)
    const unsigned value_offset = 2U * kKeyWidth + value_head * kHeadDim + column_base;
    float4 kq[KQ], vv[VQ];
    float scalar = 0.0F;
    const auto fetch = [&](unsigned first) {
#pragma unroll
        for (unsigned i = 0; i < KQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / 64U, w = q % 64U;
            kq[i] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (first + r < count) {
                const float* row = rows.qkv + static_cast<std::uint64_t>(first + r) * rows.stride + key_offset;
                kq[i] = *reinterpret_cast<const float4*>(w < 32U ? row + kKeyWidth + w * 4U : row + (w - 32U) * 4U);
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
                           : decays[(first + thread - Chunk) * kValueHeads + value_head];
    };
    const auto commit = [&](RecStage<G>& s) {
#pragma unroll
        for (unsigned i = 0; i < KQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / 64U, w = q % 64U, key = (w % 32U) * 4U;
            float (&target)[G][Sh::Pitch] = w < 32U ? s.key[r] : s.query[r];
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
        const std::uint64_t at = (static_cast<std::uint64_t>(value_head) * kHeadDim + sub + j * G) * kHeadDim + column;
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
        const RecStage<G>& s = stage[c & 1U];
        const unsigned first = c * Chunk, n = min(Chunk, count - first);
        float* core = rows.core + static_cast<std::uint64_t>(first) * rows.stride + value_head * kHeadDim + column;
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
        const std::uint64_t at = (static_cast<std::uint64_t>(value_head) * kHeadDim + sub + j * G) * kHeadDim + column;
        if constexpr (ST == RecurrentStateType::f32) state_out[at] = S[j];
        else reinterpret_cast<std::uint16_t*>(state_out)[at] = rstate::Codec<ST>::store(S[j]);
    }
}

__device__ __forceinline__ float bf16_to_float(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16); }
// hip_gdn_launcher.hip:307-317 (no NaN special case)
__device__ __forceinline__ std::uint16_t float_to_bf16(float v) {
    const std::uint32_t b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float round_bf16(float v) { return bf16_to_float(float_to_bf16(v)); }

template <unsigned Threads = 256U>
__global__ __launch_bounds__(Threads) void conv_rollback_kernel(const float* __restrict__ raw_qkv, std::uint32_t keep,
    const std::uint16_t* __restrict__ before, std::uint16_t* __restrict__ conv_state) {
    const unsigned ch = blockIdx.x * Threads + threadIdx.x;
    if (ch >= kConvWidth) return;
#pragma unroll
    for (unsigned i = 0; i < kConvKernel; ++i) {
        const int row = static_cast<int>(keep) - 4 + static_cast<int>(i);
        conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i] = row >= 0
            ? float_to_bf16(raw_qkv[static_cast<std::uint64_t>(row) * kConvWidth + ch])
            : before[static_cast<std::uint64_t>(ch) * kConvKernel + static_cast<unsigned>(row + 4)];
    }
}

// gdn_window_key_sum (hip_gdn_launcher.hip:2255-2262): lane l holds channels l, l+32, l+64, l+96 as p[0..3];
// s = (p0+p2)+(p1+p3); then lane l (< d) adds lane l+d for d = 16 (permlanex16), 8, 4, 2, 1 (DPP row_shl); the root is
// lane 0's value (readfirstlane). Lane 0's cone only reads lanes l+d with l+d inside the 16-lane row, so the xor
// butterfly gives lane 0 the same operands at every level, and (commutativity) every lane the same root bits.
__device__ __forceinline__ float key_sum(const float (&p)[4]) {
    float s = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
#pragma unroll
    for (int d = 16; d; d >>= 1) s = __fadd_rn(s, __shfl_xor_sync(kFull, s, d));
    return __shfl_sync(kFull, s, 0);
}

__global__ __launch_bounds__(256) void decay_kernel(Rows rows, std::uint32_t count, float* __restrict__ decay, am::Tables t) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x;
    if (i >= count * kValueHeads) return;
    decay[i] = am::amd_expf(t, rows.gate[static_cast<std::uint64_t>(i / kValueHeads) * rows.stride + i % kValueHeads]);
}
template <unsigned G, RecurrentStateType ST, bool RR>
__global__ __launch_bounds__(RecShape<G>::Threads) void recurrence_kernel(Rows rows, std::uint32_t count,
    DevRef<float> in_ref, DevRef<float> out_ref, const float* __restrict__ decays) {
    recurrence_body<G, ST, RR>(rows, count, in_ref.get(), out_ref.get(), decays);
}

constexpr unsigned kDecConvBlocks = kKeyHeads * 2U + kValueHeads;  // 80, 128 channels each
constexpr unsigned kDecGateBlocks = kValueHeads * 2U;               // 96, one head (beta heads then alpha heads) each

// gates_kernel<R,1> arithmetic with R == count rows and every operand load hoisted (kHidden = 10 x 256: lane chain
// f = lane + 256 i, ascending, one fma per step from +0 as in gates_kernel).
template <unsigned R>
__device__ __forceinline__ void gates_dec_body(const std::uint16_t* __restrict__ input, std::uint64_t input_stride, const Rows& rows,
    const float* __restrict__ beta_w, const float* __restrict__ alpha_w, const float* __restrict__ dt, const float* __restrict__ a,
    const am::Tables& t, unsigned head0) {
    constexpr unsigned N = kHidden / 256U;
    static_assert(N * 256U == kHidden);
    __shared__ float scratch[R][256];
    __shared__ float totals[R];
    const unsigned lane = threadIdx.x;
    const bool alpha = head0 >= kValueHeads;
    const unsigned out0 = head0 - (alpha ? kValueHeads : 0U);
    const float* weights = (alpha ? alpha_w : beta_w) + static_cast<std::uint64_t>(out0) * kHidden;
    float w[N], x[N][R];
#pragma unroll
    for (unsigned i = 0; i < N; ++i) {
        w[i] = weights[lane + i * 256U];
#pragma unroll
        for (unsigned r = 0; r < R; ++r) x[i][r] = bf16_to_float(input[static_cast<std::uint64_t>(r) * input_stride + lane + i * 256U]);
    }
    float sum[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) sum[r] = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < N; ++i)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) sum[r] = __fmaf_rn(x[i][r], w[i], sum[r]);
#pragma unroll
    for (unsigned r = 0; r < R; ++r) scratch[r][lane] = sum[r];
    __syncthreads();
    for (unsigned stride = 128U; stride > 32U; stride >>= 1U) {
        if (lane < stride)
#pragma unroll
            for (unsigned i = 0; i < R; ++i) scratch[i][lane] = __fadd_rn(scratch[i][lane], scratch[i][lane + stride]);
        __syncthreads();
    }
    const unsigned wave = lane / 32U, l = lane % 32U;
    for (unsigned i = wave; i < R; i += 8U) {
        float total = __fadd_rn(scratch[i][l], scratch[i][l + 32U]);
#pragma unroll
        for (int d = 16; d; d >>= 1) total = __fadd_rn(total, __shfl_xor_sync(kFull, total, d));
        if (l == 0U) totals[i] = total;
    }
    __syncthreads();
    if (lane < R) {
        const unsigned row = lane;
        const float raw = round_bf16(totals[row]);
        const std::uint64_t offset = static_cast<std::uint64_t>(row) * rows.stride + out0;
        if (alpha) {
            const float biased = __fadd_rn(raw, dt[out0]);
            float sp;
            if (biased > 20.0F) sp = biased;
            else {
                const float e = am::amd_expf(t, biased);
                sp = biased < -20.0F ? e : am::amd_log1pf(t, e);
            }
            rows.gate[offset] = __fmul_rn(a[out0], sp);
        } else {
            rows.beta[offset] = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -raw))));
        }
    }
}

// conv_prepass_kernel + conv_kernel (chunk 0, count <= 8) + the raw_qkv copy for one 128-channel block. The block owns its
// channels exclusively: it stages the 3 history rows and the raw rows in shared memory (and copies raw FP32 rows to
// raw_qkv), synchronises, stores the new history (raw rows count-4..count-1 = raw[count-1+i]), then runs conv_body's math.
template <unsigned R>
__device__ __forceinline__ void conv_dec_body(const Rows& rows, std::uint16_t* __restrict__ conv_state, float* __restrict__ raw_qkv,
    const float* __restrict__ conv_w, float epsilon, const am::Tables& t, unsigned block) {
    __shared__ std::uint16_t raw[R + 3U][kHeadDim];
    const unsigned thread = threadIdx.x, lane = thread % 32U, wave = thread / 32U;
    const unsigned base = block * kHeadDim;
    const bool is_qk = block < kKeyHeads * 2U, is_key = is_qk && block >= kKeyHeads;
    for (unsigned i = thread; i < (R + 3U) * kHeadDim; i += 256U) {
        const unsigned rr = i / kHeadDim, c = i % kHeadDim;
        if (rr < 3U) raw[rr][c] = conv_state[static_cast<std::uint64_t>(base + c) * kConvKernel + rr + 1U];
        else {
            const float v = rows.qkv[static_cast<std::uint64_t>(rr - 3U) * rows.stride + base + c];
            if (raw_qkv) raw_qkv[static_cast<std::uint64_t>(rr - 3U) * kConvWidth + base + c] = v;
            raw[rr][c] = float_to_bf16(v);
        }
    }
    float weight[4][kConvKernel];
#pragma unroll
    for (unsigned part = 0; part < 4U; ++part)
#pragma unroll
        for (unsigned i = 0; i < kConvKernel; ++i)
            weight[part][i] = conv_w[static_cast<std::uint64_t>(base + part * 32U + lane) * kConvKernel + i];
    const float q_scale = is_key ? 1.0F : am::amd_rsqrtf(t, static_cast<float>(kHeadDim));
    __syncthreads();
    if (thread < kHeadDim)
#pragma unroll
        for (unsigned i = 0; i < kConvKernel; ++i)
            conv_state[static_cast<std::uint64_t>(base + thread) * kConvKernel + i] = raw[R - 1U + i][thread];
    for (unsigned local = wave; local < R; local += 8U) {
        float value[4], square[4];
#pragma unroll
        for (unsigned part = 0; part < 4U; ++part) {
            const unsigned channel = part * 32U + lane;
            float sum = __fmaf_rn(bf16_to_float(raw[local][channel]), weight[part][0], 0.0F);
#pragma unroll
            for (unsigned i = 1; i < kConvKernel; ++i) sum = __fmaf_rn(bf16_to_float(raw[local + i][channel]), weight[part][i], sum);
            const float silu = __fdiv_rn(sum, __fadd_rn(1.0F, am::amd_expf(t, -sum)));
            value[part] = round_bf16(silu);
            square[part] = __fmul_rn(value[part], value[part]);
        }
        if (is_qk) {
            const float inverse = am::amd_rsqrtf(t, __fadd_rn(key_sum(square), epsilon));
#pragma unroll
            for (unsigned part = 0; part < 4U; ++part) value[part] = __fmul_rn(__fmul_rn(value[part], inverse), q_scale);
        }
        float* out = rows.qkv + static_cast<std::uint64_t>(local) * rows.stride + base + lane;
#pragma unroll
        for (unsigned part = 0; part < 4U; ++part) out[part * 32U] = value[part];
    }
}
template <unsigned R>
__global__ __launch_bounds__(256) void decode_front_kernel(const std::uint16_t* __restrict__ input, std::uint64_t input_stride,
    Rows rows, const float* __restrict__ beta_w, const float* __restrict__ alpha_w, const float* __restrict__ dt,
    const float* __restrict__ a, const float* __restrict__ conv_w, float epsilon, am::Tables t, DevRef<std::uint16_t> conv_ref,
    float* __restrict__ raw_qkv) {
    if (blockIdx.x < kDecConvBlocks) conv_dec_body<R>(rows, conv_ref.get(), raw_qkv, conv_w, epsilon, t, blockIdx.x);
    else gates_dec_body<R>(input, input_stride, rows, beta_w, alpha_w, dt, a, t, blockIdx.x - kDecConvBlocks);
}

// Recurrence with the decay folded in; blocks >= kValueHeads * 4 (only when launched with them: the commit replay) are the
// conv rollback (conv_rollback_kernel over 32 * G channels per block) of the same commit.
template <unsigned G>
__global__ __launch_bounds__(RecShape<G>::Threads) void recurrence_dec_kernel(Rows rows, std::uint32_t count, DevRef<float> in_ref,
    DevRef<float> out_ref, am::Tables t, const float* __restrict__ raw_qkv, const std::uint16_t* __restrict__ before,
    std::uint16_t* __restrict__ conv_state) {
    constexpr unsigned kRecBlocks = kValueHeads * (kHeadDim / 32U);
    if (blockIdx.x < kRecBlocks) {
        recurrence_body<G, RecurrentStateType::f32, false, true>(rows, count, in_ref.get(), out_ref.get(), nullptr, &t);
        return;
    }
    const unsigned ch = (blockIdx.x - kRecBlocks) * RecShape<G>::Threads + threadIdx.x;
    if (ch >= kConvWidth) return;
#pragma unroll
    for (unsigned i = 0; i < kConvKernel; ++i) {
        const int row = static_cast<int>(count) - 4 + static_cast<int>(i);
        conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i] = row >= 0
            ? float_to_bf16(raw_qkv[static_cast<std::uint64_t>(row) * kConvWidth + ch])
            : before[static_cast<std::uint64_t>(ch) * kConvKernel + static_cast<unsigned>(row + 4)];
    }
}

__device__ __forceinline__ void quantize_lane(float x, std::int8_t* code, float* delta, unsigned lane) {
    const bool bad = __any_sync(kFull, !isfinite(x));
    float amax = fabsf(x);
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(kFull, amax, d));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float scale = bad ? __uint_as_float(0x7fc00000U) : amax == 0.0F || tiny ? 1.0F : __fdiv_rn(amax, 127.0F);
    const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x, scale));
    *code = static_cast<std::int8_t>(static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded))));
    if (lane == 0U) *delta = scale;
}
template <bool A8>
__global__ __launch_bounds__(256) void norm_kernel(Rows rows, std::uint32_t count, const float* __restrict__ norm_w,
    float epsilon, am::Tables t, std::int8_t* codes, float* deltas) {
    const unsigned lane = threadIdx.x % 32U;
    const unsigned item = blockIdx.x * 8U + threadIdx.x / 32U;
    if (item >= count * kValueHeads) return;  // whole warp
    const unsigned row = item / kValueHeads, head = item % kValueHeads;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * rows.stride + head * kHeadDim;
    float value[4], square[4];
#pragma unroll
    for (unsigned part = 0; part < 4U; ++part) {
        value[part] = rows.core[base + part * 32U + lane];
        square[part] = __fmul_rn(value[part], value[part]);
    }
    const float inv = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(key_sum(square), static_cast<float>(kHeadDim)), epsilon));
#pragma unroll
    for (unsigned part = 0; part < 4U; ++part) {
        const unsigned channel = part * 32U + lane;
        const float gate = __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -rows.z[base + channel])));
        const float result = __fmul_rn(__fmul_rn(__fmul_rn(value[part], inv), norm_w[channel]), gate);
        rows.core[base + channel] = result;
        if constexpr (A8) {
            const std::uint64_t feature = head * kHeadDim + channel;
            quantize_lane(result, codes + static_cast<std::uint64_t>(row) * kValueWidth + feature,
                deltas + static_cast<std::uint64_t>(row / kA8Pitch) * (kValueWidth / 32U * kA8Pitch) + feature / 32U * kA8Pitch +
                    row % kA8Pitch, lane);
        }
    }
}

}  // namespace strix::cuda::recur::test::frozen
