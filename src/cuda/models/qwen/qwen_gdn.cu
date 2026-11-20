// CUDA GDN prefill core, bit-identical to the HIP tiled prefill (see include/strix/cuda/qwen_gdn.hpp).
//
// Every floating-point operation is one explicit IEEE intrinsic with the HIP operands; the HIP order comes from the gfx1151
// ISA (gfx1151 disassembly of hip_gdn_launcher, engine bf60d9d2):
//   gates  _ZN5strix12_GLOBAL__N_124gdn_prefill_gates_kernelILj8ELj2EEEvPKtmPKfS5_NS0_14GdnPrefillRowsEPKvNS0_15SmallWeightKindES8_S9_jj
//          (.s:113457): lane chains v_fmac_f32 / v_dual_fmac_f32 (:113561-113585); asm v_add_f32 trees (:113630-113879);
//          softplus: !(x > 20) (:114006), expf (v_mul 0x3fb8aa3b, v_fma, v_exp, v_ldexp), log1pf (v_frexp_mant :114032 ...),
//          a*softplus v_mul; beta: expf(-x) with negated constants (:114180-114185), 1 + e, IEEE division (v_div_fmas :114213);
//   conv   _ZN5strix12_GLOBAL__N_123gdn_prefill_conv_kernelILj8ELj32EEEvNS0_14GdnPrefillRowsEjPtPKvNS0_15SmallWeightKindEjjf
//          (.s:116154): q_scale = v_rsq_f32(128) with subnormal scaling (:117327); tap chain v_fma_f32 x,w,0 then v_fmac
//          (:117473-117503); silu division v_div_fixup (:117617); bf16 RNE v_bfe/v_add3/v_and; squares v_mul; key_sum
//          v_permlanex16 (:117659) + row_shl 8,4,2,1 (:117681) + readfirstlane (:117685); s_add_f32 eps (:117686);
//          v_rsq_f32 (:117694); (inv*value) then *q_scale (:117703);
//   lanes  _ZN5strix12_GLOBAL__N_124gdn_prefill_lanes_kernelILNS_19GdnRecurrentStorageE0ELj4ELj8ELj8EEEvNS0_14GdnPrefillRowsEjPfj
//          (.s:152370): decay = expf(gate) at commit (:152642-152660); v_mul S*decay, v_mul S*k (:152831-152899), in-lane
//          v_add tree, row_xmask:2 then :1 (:152913-152918), v_sub value-pred (:152921), v_mul *beta (:152924),
//          v_fmac S += k*delta (:152927-152971), then S*q and the same tree;
//   norm   _ZN5strix12_GLOBAL__N_128gdn_prefill_norm_rows_kernelILj8ELb0EEEvNS0_14GdnPrefillRowsEjjPKvNS0_15SmallWeightKindEjfPaPf
//          (.s:156427): key_sum (v_permlanex16 :156512, readfirstlane :156539); IEEE division by 128 (v_div_fixup :156644);
//          v_add eps (:156647); v_rsq_f32 (:156656); inv*v (:156667), *w (:156670), gate*(.) (:156674) with
//          gate = 1/(1+expf(-z)).
#include "strix/cuda/qwen_gdn.hpp"
#include "strix/cuda/qwen_gdn_prepass.hpp"
#include "strix/gdn_prefill_overlap.hpp"
#include "../../../attn_core/recurrent_state.hpp"
#include "../../../cuda_core/gdn/gdn_lanes.cuh"

#include "strix/env_flags.hpp"

#include <stdexcept>
#include <type_traits>

namespace strix::cuda::qwen_gdn {
namespace {

namespace am = strix::cuda::amdmath;
namespace rstate = strix::attn_core::rstate;
constexpr unsigned kFull = 0xffffffffU;
// The lanes recurrence (staging shape + per-row body) is the shared CUDA core's (src/cuda_core/gdn/gdn_lanes.cuh).
using strix::cuda_core::gdn_lanes::RecShape;
using strix::cuda_core::gdn_lanes::RecStage;
using strix::cuda_core::gdn_lanes::recurrence_body;
static_assert(strix::cuda_core::gdn_lanes::kHeadDim == kHeadDim && strix::cuda_core::gdn_lanes::kKeyHeads == kKeyHeads &&
              strix::cuda_core::gdn_lanes::kValueHeads == kValueHeads && strix::cuda_core::gdn_lanes::kKeyWidth == kKeyWidth,
              "the shared lanes recurrence is compiled for the Qwen GDN geometry");

__device__ __forceinline__ float bf16_to_float(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16); }
// hip_gdn_launcher.hip:307-317 (no NaN special case)
__device__ __forceinline__ std::uint16_t float_to_bf16(float v) {
    const std::uint32_t b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float round_bf16(float v) { return bf16_to_float(float_to_bf16(v)); }

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

// ---------------------------------------------------------------------------------------------------------------- K3
// gdn_prefill_gates_kernel (hip_gdn_launcher.hip:2750-2802). Per (row, head): lane l of 256 chains features l, l+256, ...
// (sum = fma(x, w, sum) from +0, v_fmac_f32); LDS tree strides 128 then 64 (asm v_add_f32), then per wave
// s[l] + s[l+32] and __shfl_down 16..1 (lane 0's cone == xor butterfly). Heads per block do not change any value.
template <unsigned R, unsigned H>
__global__ __launch_bounds__(256) void gates_kernel(const std::uint16_t* __restrict__ input, std::uint64_t input_stride,
    Rows rows, std::uint32_t count, const float* __restrict__ beta_w, const float* __restrict__ alpha_w,
    const float* __restrict__ dt, const float* __restrict__ a, am::Tables t) {
    constexpr unsigned S = R * H;
    __shared__ float scratch[S][256];
    const unsigned lane = threadIdx.x, first = blockIdx.y * R, head0 = blockIdx.x * H;
    const bool alpha = head0 >= kValueHeads;
    const unsigned out0 = head0 - (alpha ? kValueHeads : 0U);
    const float* weights = (alpha ? alpha_w : beta_w) + static_cast<std::uint64_t>(out0) * kHidden;
    float sum[H][R];
#pragma unroll
    for (unsigned h = 0; h < H; ++h)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) sum[h][r] = 0.0F;
    for (unsigned f = lane; f < kHidden; f += 256U) {
        float x[R];
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
            x[r] = first + r < count ? bf16_to_float(input[static_cast<std::uint64_t>(first + r) * input_stride + f]) : 0.0F;
#pragma unroll
        for (unsigned h = 0; h < H; ++h) {
            const float w = weights[static_cast<std::uint64_t>(h) * kHidden + f];
#pragma unroll
            for (unsigned r = 0; r < R; ++r) sum[h][r] = __fmaf_rn(x[r], w, sum[h][r]);
        }
    }
#pragma unroll
    for (unsigned h = 0; h < H; ++h)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) scratch[h * R + r][lane] = sum[h][r];
    __syncthreads();
    for (unsigned stride = 128U; stride > 32U; stride >>= 1U) {
        if (lane < stride)
#pragma unroll
            for (unsigned i = 0; i < S; ++i) scratch[i][lane] = __fadd_rn(scratch[i][lane], scratch[i][lane + stride]);
        __syncthreads();
    }
    const unsigned wave = lane / 32U, l = lane % 32U;
    __shared__ float totals[S];
    for (unsigned i = wave; i < S; i += 8U) {
        float total = __fadd_rn(scratch[i][l], scratch[i][l + 32U]);
#pragma unroll
        for (int d = 16; d; d >>= 1) total = __fadd_rn(total, __shfl_xor_sync(kFull, total, d));
        if (l == 0U) totals[i] = total;
    }
    __syncthreads();
    // the transforms of the block's S outputs run on S threads at once (HIP: lane 0 of each wave in turn; same values)
    if (lane < S) {
        const unsigned i = lane, out = out0 + i / R, row = first + i % R;
        if (row < count) {
            const float raw = round_bf16(totals[i]);
            const std::uint64_t offset = static_cast<std::uint64_t>(row) * rows.stride + out;
            if (alpha) {
                // softplus (:1141-1151): x > 20 -> x; x < -20 -> expf(x); else log1pf(expf(x)). ISA: one expf shared.
                const float biased = __fadd_rn(raw, dt[out]);
                float sp;
                if (biased > 20.0F) sp = biased;
                else {
                    const float e = am::amd_expf(t, biased);
                    sp = biased < -20.0F ? e : am::amd_log1pf(t, e);
                }
                rows.gate[offset] = __fmul_rn(a[out], sp);
            } else {
                rows.beta[offset] = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -raw))));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------- K4
// The HIP kernel walks all rows of a head in one block with an LDS ring. Rows are independent given their 3 predecessors'
// raw BF16 inputs, so here a block owns (head, 32-row chunk); a prepass saves each chunk's 3 preceding raw rows (from the
// workspace or the old history) and writes the new history, before any row is overwritten in place.
// The body saves the taps of chunks [first, last); `history` = this call owns the old history (reads it, then writes the new
// one). Split rule (include/strix/cuda/qwen_gdn_prepass.hpp): only chunk 0's taps and a history of count < 4 rows read the
// old history, so the caller that covers chunk 0 owns it and every other caller reads workspace rows only.
static_assert(qwen_gdn::prepass::kChunk == kConvChunk && qwen_gdn::prepass::kHistory == kConvKernel && qwen_gdn::prepass::kTaps == 3U,
    "conv prepass split rule matches the conv shape");
__device__ __forceinline__ void conv_prepass_body(const Rows& rows, std::uint32_t count, std::uint16_t* conv_state,
    std::uint16_t* __restrict__ boundary, unsigned first, unsigned last, bool history) {
    const unsigned ch = blockIdx.x * 256U + threadIdx.x;
    if (ch >= kConvWidth) return;
    std::uint16_t old[kConvKernel] = {};
    if (history)
#pragma unroll
        for (unsigned i = 0; i < kConvKernel; ++i) old[i] = conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i];
    const auto raw = [&](std::int64_t row) -> std::uint16_t {  // row >= -3; rows -3..-1 are taps 1..3 (:2823-2824)
        return row >= 0 ? float_to_bf16(rows.qkv[static_cast<std::uint64_t>(row) * rows.stride + ch]) : old[row + 4];
    };
    for (unsigned c = first; c < last; ++c)
#pragma unroll
        for (unsigned j = 0; j < 3U; ++j)
            boundary[(static_cast<std::uint64_t>(c) * 3U + j) * kConvWidth + ch] = raw(qwen_gdn::prepass::tap_row(c, j));
    if (!history) return;
    // history = raw inputs of rows count-4 .. count-1 (:2858-2859; ring slot of row -k is tap 4-k)
#pragma unroll
    for (unsigned i = 0; i < kConvKernel; ++i)
        conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i] = raw(qwen_gdn::prepass::history_row(count, i));
}
// grid (channel blocks, chunks): block y saves chunk y's taps (one DRAM round trip per thread instead of one per chunk);
// the block of chunk 0 owns the history.
__global__ __launch_bounds__(256) void conv_prepass_kernel(Rows rows, std::uint32_t count, DevRef<std::uint16_t> conv_ref,
    std::uint16_t* __restrict__ boundary) {
    conv_prepass_body(rows, count, conv_ref.get(), boundary, blockIdx.y, blockIdx.y + 1U, blockIdx.y == qwen_gdn::prepass::kHistoryChunk);
}
// Segment views: the rows of segment `seg` start at its row (every field shifts by the same stride).
__device__ __forceinline__ Rows segment_rows(const Rows& rows, std::uint32_t row) {
    const std::uint64_t o = static_cast<std::uint64_t>(row) * rows.stride;
    return Rows{rows.qkv + o, rows.z + o, rows.beta + o, rows.gate + o, rows.core + o, rows.stride};
}
// one grid slice (blockIdx.y) per segment; boundary of segment i at boundary + i * 3 * kConvWidth (one chunk: count <= 32)
__global__ __launch_bounds__(256) void conv_prepass_multi_kernel(Rows rows, Segments segs, std::uint16_t* __restrict__ boundary) {
    const Segment& s = segs.s[blockIdx.y];
    conv_prepass_body(segment_rows(rows, s.row), s.count, s.conv.get(), boundary + static_cast<std::uint64_t>(blockIdx.y) * 3U * kConvWidth,
        0U, qwen_gdn::prepass::chunks(s.count), true);
}

// Conv history after the first `keep` rows of a window: the prepass rule (raw inputs of rows keep-4 .. keep-1, older taps
// from the history before the window), from the saved raw rows (launch_core_split raw_qkv).
__global__ __launch_bounds__(256) void conv_rollback_kernel(const float* __restrict__ raw_qkv, std::uint32_t keep,
    const std::uint16_t* __restrict__ before, std::uint16_t* __restrict__ conv_state) {
    const unsigned ch = blockIdx.x * 256U + threadIdx.x;
    if (ch >= kConvWidth) return;
#pragma unroll
    for (unsigned i = 0; i < kConvKernel; ++i) {
        const int row = static_cast<int>(keep) - 4 + static_cast<int>(i);
        conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i] = row >= 0
            ? float_to_bf16(raw_qkv[static_cast<std::uint64_t>(row) * kConvWidth + ch])
            : before[static_cast<std::uint64_t>(ch) * kConvKernel + static_cast<unsigned>(row + 4)];
    }
}

// gdn_prefill_conv_kernel body (:2838-2856): sum = fma(x0,w0,0), fma(x1,w1,.), ... oldest tap first; silu = sum/(1+expf(-sum));
// value = round_bf16(silu); q/k: inv = rsqrtf(key_sum(value^2) + eps), value = (value*inv)*q_scale, q_scale = rsqrtf(128) or 1.
__device__ __forceinline__ void conv_body(const Rows& rows, std::uint32_t count, const std::uint16_t* __restrict__ boundary,
    const float* __restrict__ conv_w, float epsilon, const am::Tables& t, unsigned chunk) {
    __shared__ std::uint16_t raw[kConvChunk + 3U][kHeadDim];
    const unsigned thread = threadIdx.x, lane = thread % 32U, wave = thread / 32U;
    const unsigned block = blockIdx.x, base = block * kHeadDim, first = chunk * kConvChunk;
    const bool is_qk = block < kKeyHeads * 2U, is_key = is_qk && block >= kKeyHeads;
    for (unsigned i = thread; i < (kConvChunk + 3U) * kHeadDim; i += 256U) {
        const unsigned rr = i / kHeadDim, c = i % kHeadDim;
        if (rr < 3U) raw[rr][c] = boundary[(static_cast<std::uint64_t>(chunk) * 3U + rr) * kConvWidth + base + c];
        else if (first + rr - 3U < count)
            raw[rr][c] = float_to_bf16(rows.qkv[static_cast<std::uint64_t>(first + rr - 3U) * rows.stride + base + c]);
    }
    float weight[4][kConvKernel];
#pragma unroll
    for (unsigned part = 0; part < 4U; ++part)
#pragma unroll
        for (unsigned i = 0; i < kConvKernel; ++i)
            weight[part][i] = conv_w[static_cast<std::uint64_t>(base + part * 32U + lane) * kConvKernel + i];
    const float q_scale = is_key ? 1.0F : am::amd_rsqrtf(t, static_cast<float>(kHeadDim));
    __syncthreads();
    for (unsigned local = wave; local < kConvChunk && first + local < count; local += 8U) {
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
        if (is_qk) {  // warp-uniform
            const float inverse = am::amd_rsqrtf(t, __fadd_rn(key_sum(square), epsilon));
#pragma unroll
            for (unsigned part = 0; part < 4U; ++part) value[part] = __fmul_rn(__fmul_rn(value[part], inverse), q_scale);
        }
        float* out = rows.qkv + static_cast<std::uint64_t>(first + local) * rows.stride + base + lane;
#pragma unroll
        for (unsigned part = 0; part < 4U; ++part) out[part * 32U] = value[part];
    }
}
__global__ __launch_bounds__(256) void conv_kernel(Rows rows, std::uint32_t count, const std::uint16_t* __restrict__ boundary,
    const float* __restrict__ conv_w, float epsilon, am::Tables t) {
    conv_body(rows, count, boundary, conv_w, epsilon, t, blockIdx.y);
}
__global__ __launch_bounds__(256) void conv_multi_kernel(Rows rows, Segments segs, const std::uint16_t* __restrict__ boundary,
    const float* __restrict__ conv_w, float epsilon, am::Tables t) {
    const Segment& s = segs.s[blockIdx.y];
    conv_body(segment_rows(rows, s.row), s.count, boundary + static_cast<std::uint64_t>(blockIdx.y) * 3U * kConvWidth, conv_w, epsilon,
        t, 0U);
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
// one grid slice (blockIdx.y) per segment: its rows, its state, its rows' decays (decay rows are window rows)
template <unsigned G, RecurrentStateType ST, bool RR>
__global__ __launch_bounds__(RecShape<G>::Threads) void recurrence_multi_kernel(Rows rows, Segments segs, const float* __restrict__ decays) {
    const Segment& s = segs.s[blockIdx.y];
    recurrence_body<G, ST, RR>(segment_rows(rows, s.row), s.count, s.state_in.get(), s.state_out.get(),
        decays + static_cast<std::uint64_t>(s.row) * kValueHeads);
}


// ---------------------------------------------------------------------------------------------------------------- decode
// Decode windows (1..8 rows, one segment). Same arithmetic as K3/K4/K5, fewer launches and shorter dependent chains
// (STRIX_QRTX_GDN_DECODE): front = gates + conv(+prepass +raw copy) in one kernel, recurrence with decay folded in, and
// at commit the recurrence replay + conv rollback in one launch.
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

// ---------------------------------------------------------------------------------------------------------------- K6
// gdn_prefill_norm_rows_kernel (:3000-3029): one warp per (row, head); inv = rsqrtf(key_sum(v^2)/128 + eps);
// result = ((v*inv)*w[c])*(1/(1+expf(-z))). A8: dense_a8_quantize_lane (hip_q8_rows_gemv.hpp:1147-1159, contract off).
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

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_gdn ") + what + ": " + cudaGetErrorString(e));
}
void check_rows(const Rows& rows) {
    if (!rows.qkv || !rows.z || !rows.beta || !rows.gate || !rows.core || !rows.stride)
        throw std::invalid_argument("qwen_gdn: missing row workspace");
}


bool decode_fused_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_GDN_DECODE", true);
    return on;
}
// prefill two-stream overlap (launch_core_overlap); 0 = the serial K3..K6 order
bool overlap_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_GDN_OVERLAP", true);
    return on;
}
void check_aligned(const Rows& rows) {
    if (rows.stride % 4U || reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U)
        throw std::invalid_argument("qwen_gdn: qkv rows must be 16-byte aligned");
}
// recurrence_dec_kernel launch: the recurrence blocks plus (raw_qkv != nullptr) the conv rollback blocks
void launch_recurrence_dec(const Rows& rows, std::uint32_t count, DevRef<float> state_in, DevRef<float> state_out,
    const float* raw_qkv, const std::uint16_t* before, std::uint16_t* conv_state, const amdmath::Tables& t, cudaStream_t stream,
    Tuning tuning) {
    const auto go = [&](auto kernel, unsigned g) {
        const unsigned rec = kValueHeads * (kHeadDim / 32U), extra = raw_qkv ? (kConvWidth + 32U * g - 1U) / (32U * g) : 0U;
        kernel<<<rec + extra, 32U * g, 0, stream>>>(rows, count, state_in, state_out, t, raw_qkv, before, conv_state);
    };
    switch (tuning.lanes) {
    case 2: go(recurrence_dec_kernel<2>, 2); break;
    case 4: go(recurrence_dec_kernel<4>, 4); break;
    case 8: go(recurrence_dec_kernel<8>, 8); break;
    case 16: go(recurrence_dec_kernel<16>, 16); break;
    default: throw std::invalid_argument("qwen_gdn: lanes must be 2, 4, 8 or 16");
    }
    check(cudaGetLastError(), "recurrence decode launch");
}
template <unsigned R>
void front_go(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, const Weights& w, const amdmath::Tables& t,
    DevRef<std::uint16_t> conv_state, float* raw_qkv, cudaStream_t stream) {
    decode_front_kernel<R><<<kDecConvBlocks + kDecGateBlocks, 256, 0, stream>>>(input, input_stride, rows, w.beta, w.alpha, w.dt,
        w.a, w.conv, w.qk_l2_epsilon, t, conv_state, raw_qkv);
}
// K3 + K4 (+ raw copy) + K5 of a 1..8-row window: front kernel, then the recurrence with the decay folded in
void launch_decode_core(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    DevRef<std::uint16_t> conv_state, DevRef<float> state_in, DevRef<float> state_out, float* raw_qkv, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning) {
    switch (count) {
    case 1: front_go<1>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    case 2: front_go<2>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    case 3: front_go<3>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    case 4: front_go<4>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    case 5: front_go<5>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    case 6: front_go<6>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    case 7: front_go<7>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    default: front_go<8>(input, input_stride, rows, w, t, conv_state, raw_qkv, stream); break;
    }
    check(cudaGetLastError(), "decode front launch");
    launch_recurrence_dec(rows, count, state_in, state_out, nullptr, nullptr, nullptr, t, stream, tuning);
}
}  // namespace

std::size_t conv_scratch_bytes(std::uint32_t count) {
    return static_cast<std::size_t>((count + kConvChunk - 1U) / kConvChunk) * 3U * kConvWidth * sizeof(std::uint16_t);
}

void launch_gates(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    const Weights& w, const amdmath::Tables& t, cudaStream_t stream) {
    check_rows(rows);
    if (!count) return;
    constexpr unsigned R = 8, H = 4;
    gates_kernel<R, H><<<dim3(kValueHeads * 2U / H, (count + R - 1U) / R), 256, 0, stream>>>(input, input_stride, rows, count,
        w.beta, w.alpha, w.dt, w.a, t);
    check(cudaGetLastError(), "gates launch");
}

namespace {
void launch_conv_ref(const Rows& rows, std::uint32_t count, DevRef<std::uint16_t> conv_state, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream);
void launch_recurrence_ref(const Rows& rows, std::uint32_t count, DevRef<float> state_in, DevRef<float> state_out, void* scratch,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning);
}  // namespace

void launch_conv(const Rows& rows, std::uint32_t count, std::uint16_t* conv_state, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream) {
    launch_conv_ref(rows, count, DevRef<std::uint16_t>{conv_state}, scratch, w, t, stream);
}

namespace {
void launch_conv_ref(const Rows& rows, std::uint32_t count, DevRef<std::uint16_t> conv_state, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream) {
    check_rows(rows);
    if (!count) return;
    if (rows.stride % 4U || reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U)
        throw std::invalid_argument("qwen_gdn: qkv rows must be 16-byte aligned");
    auto* boundary = static_cast<std::uint16_t*>(scratch);
    conv_prepass_kernel<<<dim3((kConvWidth + 255U) / 256U, prepass::chunks(count)), 256, 0, stream>>>(rows, count, conv_state,
        boundary);
    check(cudaGetLastError(), "conv prepass launch");
    conv_kernel<<<dim3(kKeyHeads * 2U + kValueHeads, (count + kConvChunk - 1U) / kConvChunk), 256, 0, stream>>>(rows, count,
        boundary, w.conv, w.qk_l2_epsilon, t);
    check(cudaGetLastError(), "conv launch");
}
}  // namespace

void launch_conv_rollback(const float* raw_qkv, std::uint32_t keep, const std::uint16_t* before, std::uint16_t* conv_state,
    cudaStream_t stream) {
    conv_rollback_kernel<<<(kConvWidth + 255U) / 256U, 256, 0, stream>>>(raw_qkv, keep, before, conv_state);
    check(cudaGetLastError(), "conv rollback launch");
}

std::size_t decay_scratch_bytes(std::uint32_t count) { return static_cast<std::size_t>(count) * kValueHeads * sizeof(float); }
std::size_t core_scratch_bytes(std::uint32_t count) {
    return (conv_scratch_bytes(count) + 255U) / 256U * 256U + decay_scratch_bytes(count);
}

void launch_recurrence(const Rows& rows, std::uint32_t count, float* state, void* scratch, const amdmath::Tables& t,
    cudaStream_t stream, Tuning tuning) {
    launch_recurrence_split(rows, count, state, state, scratch, t, stream, tuning);
}

void launch_recurrence_split(const Rows& rows, std::uint32_t count, const float* state_in, float* state_out, void* scratch,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning) {
    launch_recurrence_ref(rows, count, DevRef<float>{const_cast<float*>(state_in)}, DevRef<float>{state_out}, scratch, t, stream, tuning);
}

namespace {
// 0 f32 | 1 f16 prefill | 2 f16 per-row | 3 bf16 prefill | 4 bf16 per-row (round_each_row is meaningless for f32: same code)
unsigned state_mode(const Tuning& tuning) {
    switch (tuning.state_type) {
    case RecurrentStateType::f32: return 0;
    case RecurrentStateType::f16: return tuning.round_each_row ? 2U : 1U;
    case RecurrentStateType::bf16: return tuning.round_each_row ? 4U : 3U;
    }
    throw std::invalid_argument("qwen_gdn: unknown recurrent state type");
}
void launch_recurrence_ref(const Rows& rows, std::uint32_t count, DevRef<float> state_in, DevRef<float> state_out, void* scratch,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning) {
    check_rows(rows);
    if (!count) return;
    if (rows.stride % 4U || reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U)
        throw std::invalid_argument("qwen_gdn: qkv rows must be 16-byte aligned");
    auto* decay = static_cast<float*>(scratch);
    decay_kernel<<<(count * kValueHeads + 255U) / 256U, 256, 0, stream>>>(rows, count, decay, t);
    check(cudaGetLastError(), "decay launch");
    const auto go = [&](auto kernel, unsigned g) {
        kernel<<<kValueHeads * (kHeadDim / 32U), 32U * g, 0, stream>>>(rows, count, state_in, state_out, decay);
    };
    const auto lanes = [&](auto select) {
        switch (tuning.lanes) {
        case 2: go(select(std::integral_constant<unsigned, 2>{}), 2); break;
        case 4: go(select(std::integral_constant<unsigned, 4>{}), 4); break;
        case 8: go(select(std::integral_constant<unsigned, 8>{}), 8); break;
        case 16: go(select(std::integral_constant<unsigned, 16>{}), 16); break;
        default: throw std::invalid_argument("qwen_gdn: lanes must be 2, 4, 8 or 16");
        }
    };
    switch (state_mode(tuning)) {
    case 0: lanes([](auto g) { return recurrence_kernel<decltype(g)::value, RecurrentStateType::f32, false>; }); break;
    case 1: lanes([](auto g) { return recurrence_kernel<decltype(g)::value, RecurrentStateType::f16, false>; }); break;
    case 2: lanes([](auto g) { return recurrence_kernel<decltype(g)::value, RecurrentStateType::f16, true>; }); break;
    case 3: lanes([](auto g) { return recurrence_kernel<decltype(g)::value, RecurrentStateType::bf16, false>; }); break;
    default: lanes([](auto g) { return recurrence_kernel<decltype(g)::value, RecurrentStateType::bf16, true>; }); break;
    }
    check(cudaGetLastError(), "recurrence launch");
}
}  // namespace

void launch_commit_replay(const Rows& rows, std::uint32_t keep, const float* state_in, float* state_out, const float* raw_qkv,
    const std::uint16_t* conv_before, std::uint16_t* conv_state, void* scratch, const amdmath::Tables& t, cudaStream_t stream,
    Tuning tuning) {
    // PROVISIONAL decline for 16-bit state: fused staging reads/writes f32 state directly
    // through recurrence_dec_kernel (src/cuda/models/qwen/qwen_gdn.cu:554); use the typed split path.
    if (decode_fused_enabled() && keep && tuning.state_type == RecurrentStateType::f32) {
        check_rows(rows);
        check_aligned(rows);
        launch_recurrence_dec(rows, keep, DevRef<float>{const_cast<float*>(state_in)}, DevRef<float>{state_out}, raw_qkv, conv_before,
            conv_state, t, stream, tuning);
        return;
    }
    launch_recurrence_split(rows, keep, state_in, state_out, scratch, t, stream, tuning);
    launch_conv_rollback(raw_qkv, keep, conv_before, conv_state, stream);
}

void launch_norm(const Rows& rows, std::uint32_t count, const Weights& w, const amdmath::Tables& t, cudaStream_t stream,
    const A8Out* a8) {
    check_rows(rows);
    if (!count) return;
    const unsigned blocks = (count * kValueHeads + 7U) / 8U;
    if (a8) norm_kernel<true><<<blocks, 256, 0, stream>>>(rows, count, w.norm, w.rms_epsilon, t, a8->codes, a8->deltas);
    else norm_kernel<false><<<blocks, 256, 0, stream>>>(rows, count, w.norm, w.rms_epsilon, t, nullptr, nullptr);
    check(cudaGetLastError(), "norm launch");
}

void launch_core(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    std::uint16_t* conv_state, float* state, void* scratch, const Weights& w, const amdmath::Tables& t, cudaStream_t stream,
    const A8Out* a8, Tuning tuning) {
    launch_core_overlap(input, input_stride, rows, count, conv_state, state, scratch, w, t, stream, a8, tuning, overlap_enabled());
}

namespace {
// The auxiliary stream and events of the calling thread on the current device, created on first use (thread-local: the
// event records and waits of one call never interleave with another thread's). nullptr = run serially.
constexpr int kOverlapDevices = 16;
constexpr unsigned kOverlapEvents = 2U * gdn_overlap::kMaxSegments + 2U;  // fork, ready[], done[], join
struct OverlapLanes {
    cudaStream_t aux = nullptr;
    cudaEvent_t ev[kOverlapEvents] = {};
    bool tried = false;
    OverlapLanes() = default;
    OverlapLanes(const OverlapLanes&) = delete;
    OverlapLanes& operator=(const OverlapLanes&) = delete;
    ~OverlapLanes() { release(); }
    void release() {
        for (cudaEvent_t& e : ev) {
            if (e) (void)cudaEventDestroy(e);
            e = nullptr;
        }
        if (aux) (void)cudaStreamDestroy(aux);
        aux = nullptr;
    }
};
OverlapLanes* overlap_lanes(cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }
    int device = 0;
    if (capture != cudaStreamCaptureStatusNone || cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kOverlapDevices)
        return nullptr;
    thread_local OverlapLanes lanes[kOverlapDevices];
    OverlapLanes& l = lanes[device];
    if (!l.tried) {
        l.tried = true;
        bool ok = cudaStreamCreateWithFlags(&l.aux, cudaStreamNonBlocking) == cudaSuccess;
        for (unsigned i = 0; ok && i < kOverlapEvents; ++i)
            ok = cudaEventCreateWithFlags(&l.ev[i], cudaEventDisableTiming) == cudaSuccess;
        if (!ok) {
            (void)cudaGetLastError();
            l.release();
        }
    }
    return l.aux ? &l : nullptr;
}
}  // namespace

void launch_core_overlap(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    std::uint16_t* conv_state, float* state, void* scratch, const Weights& w, const amdmath::Tables& t, cudaStream_t stream,
    const A8Out* a8, Tuning tuning, bool overlap) {
    const auto plan = gdn_overlap::plan(count, overlap && tuning.state_type == RecurrentStateType::f32);
    OverlapLanes* lanes = plan.n > 1U ? overlap_lanes(stream) : nullptr;
    if (!lanes) {
        launch_core_split(input, input_stride, rows, count, conv_state, state, state, nullptr, scratch, w, t, stream, a8, tuning);
        return;
    }
    check_rows(rows);
    // the whole call's scratch layout: conv boundaries of chunk c at c * 3 rows, decays of row r at r * kValueHeads; a
    // segment starting at a multiple of gdn_overlap::kAlign uses exactly its part of both
    auto* bytes = static_cast<unsigned char*>(scratch);
    auto* decay = reinterpret_cast<float*>(bytes + (conv_scratch_bytes(count) + 255U) / 256U * 256U);
    cudaStream_t aux = lanes->aux;
    cudaEvent_t* ready = lanes->ev + 1;
    cudaEvent_t* done = lanes->ev + 1 + gdn_overlap::kMaxSegments;
    const auto part = [&](std::uint32_t first) {
        const std::uint64_t o = std::uint64_t{first} * rows.stride;
        return Rows{rows.qkv + o, rows.z + o, rows.beta + o, rows.gate + o, rows.core + o, rows.stride};
    };
    const auto front = [&](std::uint32_t i) {  // K3 + K4 of segment i on the auxiliary stream
        const std::uint32_t first = gdn_overlap::first_row(plan, i), n = gdn_overlap::segment_rows(plan, count, i);
        const Rows r = part(first);
        launch_gates(input + std::uint64_t{first} * input_stride, input_stride, r, n, w, t, aux);
        launch_conv(r, n, conv_state, bytes + conv_scratch_bytes(first), w, t, aux);
        check(cudaEventRecord(ready[i], aux), "overlap ready");
    };
    check(cudaEventRecord(lanes->ev[0], stream), "overlap fork");
    check(cudaStreamWaitEvent(aux, lanes->ev[0], 0), "overlap fork wait");
    front(0);
    for (std::uint32_t i = 0; i < plan.n; ++i) {
        const std::uint32_t first = gdn_overlap::first_row(plan, i), n = gdn_overlap::segment_rows(plan, count, i);
        const Rows r = part(first);
        check(cudaStreamWaitEvent(stream, ready[i], 0), "overlap ready wait");
        launch_recurrence_split(r, n, state, state, decay + std::uint64_t{first} * kValueHeads, t, stream, tuning);
        check(cudaEventRecord(done[i], stream), "overlap done");
        if (i + 1U < plan.n) front(i + 1U);
        check(cudaStreamWaitEvent(aux, done[i], 0), "overlap done wait");
        A8Out part_a8{};
        if (a8)
            part_a8 = A8Out{a8->codes + std::uint64_t{first} * kValueWidth,
                            a8->deltas + std::uint64_t{first / kA8Pitch} * (kValueWidth / 32U * kA8Pitch)};
        launch_norm(r, n, w, t, aux, a8 ? &part_a8 : nullptr);
    }
    check(cudaEventRecord(lanes->ev[kOverlapEvents - 1U], aux), "overlap join");
    check(cudaStreamWaitEvent(stream, lanes->ev[kOverlapEvents - 1U], 0), "overlap join wait");
}

void launch_core_split(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    std::uint16_t* conv_state, const float* state_in, float* state_out, float* raw_qkv, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream, const A8Out* a8, Tuning tuning) {
    if (count >= 1 && count <= 8 && decode_fused_enabled() && tuning.state_type == RecurrentStateType::f32) {
        check_rows(rows);
        check_aligned(rows);
        launch_decode_core(input, input_stride, rows, count, DevRef<std::uint16_t>{conv_state}, DevRef<float>{const_cast<float*>(state_in)},
            DevRef<float>{state_out}, raw_qkv, w, t, stream, tuning);
        launch_norm(rows, count, w, t, stream, a8);
        return;
    }
    launch_gates(input, input_stride, rows, count, w, t, stream);
    if (raw_qkv)
        check(cudaMemcpy2DAsync(raw_qkv, std::size_t{kConvWidth} * 4U, rows.qkv, rows.stride * 4U, std::size_t{kConvWidth} * 4U, count,
            cudaMemcpyDeviceToDevice, stream), "raw qkv copy");
    launch_conv(rows, count, conv_state, scratch, w, t, stream);
    launch_recurrence_split(rows, count, state_in, state_out,
        static_cast<unsigned char*>(scratch) + (conv_scratch_bytes(count) + 255U) / 256U * 256U, t, stream, tuning);
    launch_norm(rows, count, w, t, stream, a8);
}

std::size_t multi_scratch_bytes() {
    return (std::size_t{kMaxSegments} * 3U * kConvWidth * sizeof(std::uint16_t) + 255U) / 256U * 256U +
           decay_scratch_bytes(kMaxSegments * kConvChunk);
}

void launch_core_multi(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, const Segments& segs,
    float* raw_qkv, void* scratch, const Weights& w, const amdmath::Tables& t, cudaStream_t stream, Tuning tuning) {
    check_rows(rows);
    if (segs.n == 0 || segs.n > kMaxSegments) throw std::invalid_argument("qwen_gdn multi: 1..8 segments");
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < segs.n; ++i) {
        if (segs.s[i].row != total || segs.s[i].count == 0 || segs.s[i].count > kConvChunk)
            throw std::invalid_argument("qwen_gdn multi: segments must be contiguous from row 0, 1..32 rows each");
        total += segs.s[i].count;
    }
    if (rows.stride % 4U || reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U)
        throw std::invalid_argument("qwen_gdn: qkv rows must be 16-byte aligned");
    // K3 over all rows (row-wise)
    launch_gates(input, input_stride, rows, total, w, t, stream);
    if (raw_qkv)
        check(cudaMemcpy2DAsync(raw_qkv, std::size_t{kConvWidth} * 4U, rows.qkv, rows.stride * 4U, std::size_t{kConvWidth} * 4U, total,
            cudaMemcpyDeviceToDevice, stream), "raw qkv copy");
    // K4 per segment (its conv history)
    auto* boundary = static_cast<std::uint16_t*>(scratch);
    auto* decay = reinterpret_cast<float*>(static_cast<unsigned char*>(scratch) +
        (std::size_t{kMaxSegments} * 3U * kConvWidth * sizeof(std::uint16_t) + 255U) / 256U * 256U);
    conv_prepass_multi_kernel<<<dim3((kConvWidth + 255U) / 256U, segs.n), 256, 0, stream>>>(rows, segs, boundary);
    check(cudaGetLastError(), "conv prepass multi launch");
    conv_multi_kernel<<<dim3(kKeyHeads * 2U + kValueHeads, segs.n), 256, 0, stream>>>(rows, segs, boundary, w.conv, w.qk_l2_epsilon, t);
    check(cudaGetLastError(), "conv multi launch");
    // decay over all rows (row-wise), K5 per segment (its state)
    decay_kernel<<<(total * kValueHeads + 255U) / 256U, 256, 0, stream>>>(rows, total, decay, t);
    check(cudaGetLastError(), "decay launch");
    const auto go = [&](auto kernel, unsigned g) {
        kernel<<<dim3(kValueHeads * (kHeadDim / 32U), segs.n), 32U * g, 0, stream>>>(rows, segs, decay);
    };
    const auto lanes = [&](auto select) {
        switch (tuning.lanes) {
        case 2: go(select(std::integral_constant<unsigned, 2>{}), 2); break;
        case 4: go(select(std::integral_constant<unsigned, 4>{}), 4); break;
        case 8: go(select(std::integral_constant<unsigned, 8>{}), 8); break;
        case 16: go(select(std::integral_constant<unsigned, 16>{}), 16); break;
        default: throw std::invalid_argument("qwen_gdn: lanes must be 2, 4, 8 or 16");
        }
    };
    switch (state_mode(tuning)) {
    case 0: lanes([](auto g) { return recurrence_multi_kernel<decltype(g)::value, RecurrentStateType::f32, false>; }); break;
    case 1: lanes([](auto g) { return recurrence_multi_kernel<decltype(g)::value, RecurrentStateType::f16, false>; }); break;
    case 2: lanes([](auto g) { return recurrence_multi_kernel<decltype(g)::value, RecurrentStateType::f16, true>; }); break;
    case 3: lanes([](auto g) { return recurrence_multi_kernel<decltype(g)::value, RecurrentStateType::bf16, false>; }); break;
    default: lanes([](auto g) { return recurrence_multi_kernel<decltype(g)::value, RecurrentStateType::bf16, true>; }); break;
    }
    check(cudaGetLastError(), "recurrence multi launch");
    // K6 over all rows (row-wise)
    launch_norm(rows, total, w, t, stream, nullptr);
}

void launch_core_split_ref(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    DevRef<std::uint16_t> conv_state, DevRef<float> state_in, DevRef<float> state_out, float* raw_qkv, void* scratch,
    const Weights& w, const amdmath::Tables& t, cudaStream_t stream, const A8Out* a8, Tuning tuning) {
    if (count >= 1 && count <= 8 && decode_fused_enabled() && tuning.state_type == RecurrentStateType::f32) {
        check_rows(rows);
        check_aligned(rows);
        launch_decode_core(input, input_stride, rows, count, conv_state, state_in, state_out, raw_qkv, w, t, stream, tuning);
        launch_norm(rows, count, w, t, stream, a8);
        return;
    }
    launch_gates(input, input_stride, rows, count, w, t, stream);
    if (raw_qkv)
        check(cudaMemcpy2DAsync(raw_qkv, std::size_t{kConvWidth} * 4U, rows.qkv, rows.stride * 4U, std::size_t{kConvWidth} * 4U, count,
            cudaMemcpyDeviceToDevice, stream), "raw qkv copy");
    launch_conv_ref(rows, count, conv_state, scratch, w, t, stream);
    launch_recurrence_ref(rows, count, state_in, state_out,
        static_cast<unsigned char*>(scratch) + (conv_scratch_bytes(count) + 255U) / 256U * 256U, t, stream, tuning);
    launch_norm(rows, count, w, t, stream, a8);
}

}  // namespace strix::cuda::qwen_gdn
