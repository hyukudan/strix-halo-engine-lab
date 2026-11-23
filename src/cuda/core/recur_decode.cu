#include "recur_cuda.cuh"

#include "chain/key.hpp"
#include "common.cuh"
#include "recur_lanes.cuh"

namespace strix::cuda::recur {
namespace {

namespace lanes = cuda_core::gdn_lanes;
namespace am = numeric::gfx1151;
namespace chain = core::chain;
using Layout = Gfx1151Decode;
using chain::Math;

static_assert(Layout::key_heads == lanes::kKeyHeads && Layout::value_heads == lanes::kValueHeads &&
              Layout::head_dim == lanes::kHeadDim);

// The finite-value rounding contract of qwen_gdn.cu. Unlike the state
// codec, the projection and conv boundary uses the raw integer RNE step.
__device__ __forceinline__ std::uint16_t narrow(float x) {
    const auto bits = __float_as_uint(x);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

__device__ __forceinline__ float round_bf16(float x) {
    return cuda_core::bf16_lo(narrow(x));
}

template <Math M>
struct StripedArithmetic {
    static_assert(M == Math::gfx1151_v1);

    __device__ static __forceinline__ float sum_head(const float (&p)[4]) {
        const float local = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
        return __shfl_sync(lanes::kFull, cuda_core::warp_sum_xor(local), 0);
    }

    __device__ static __forceinline__ float log_decay(float projected, float dt, float a, const am::Tables& t) {
        const float x = __fadd_rn(projected, dt);
        if (x > 20.0F) return __fmul_rn(a, x);
        const float e = am::amd_expf(t, x);
        return __fmul_rn(a, x < -20.0F ? e : ::strix::cuda::amdmath::amd_log1pf(t, e));
    }

    __device__ static __forceinline__ float sigmoid(float x, const am::Tables& t) {
        return __fdiv_rn(1.0F, sigmoid_denominator(x, t));
    }

    __device__ static __forceinline__ float sigmoid_denominator(float x, const am::Tables& t) {
        return __fadd_rn(1.0F, am::amd_expf(t, -x));
    }
};

using DecodeArithmetic = StripedArithmetic<Math::gfx1151_v1>;

template <class Arithmetic>
__device__ __forceinline__ void project_gate(const DecodeWeights& w, const DecodeRow& row,
                                             const am::Tables& t, unsigned item) {
    __shared__ float sums[256];
    const unsigned lane = threadIdx.x;
    const bool alpha = item >= Layout::value_heads;
    const unsigned head = item % Layout::value_heads;
    const float* weights = (alpha ? w.alpha : w.beta) + head * Layout::hidden;
    float sum = 0.0F;
#pragma unroll
    for (unsigned i = 0U; i < Layout::hidden / 256U; ++i) {
        const unsigned feature = lane + i * 256U;
        sum = __fmaf_rn(cuda_core::bf16_lo(row.input[feature]), weights[feature], sum);
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned stride = 128U; stride > 32U; stride /= 2U) {
        if (lane < stride) sums[lane] = __fadd_rn(sums[lane], sums[lane + stride]);
        __syncthreads();
    }
    if (lane >= 32U) return;
    sum = cuda_core::warp_sum_xor(__fadd_rn(sums[lane], sums[lane + 32U]));
    if (lane == 0U) {
        const float projected = round_bf16(sum);
        if (alpha) row.log_decay[head] = Arithmetic::log_decay(projected, w.dt[head], w.a[head], t);
        else row.beta[head] = round_bf16(Arithmetic::sigmoid(projected, t));
    }
}

template <class Arithmetic>
__device__ __forceinline__ void convolve_decode(const DecodeWeights& w, const DecodeRow& row,
                                                std::uint16_t* history, const am::Tables& t) {
    const unsigned lane = threadIdx.x;
    if (lane >= 32U) return;
    const unsigned base = blockIdx.x * Layout::head_dim;
    float value[4], square[4];
#pragma unroll
    for (unsigned part = 0U; part < 4U; ++part) {
        const unsigned c = base + part * 32U + lane;
        const float raw = row.qkv[c];
        if (row.raw_capture) row.raw_capture[c] = raw;
        auto* tap = history + c * Layout::taps;
        const std::uint16_t samples[]{tap[1], tap[2], tap[3], narrow(raw)};
        float sum = 0.0F;
#pragma unroll
        for (unsigned i = 0U; i < Layout::taps; ++i) {
            sum = __fmaf_rn(cuda_core::bf16_lo(samples[i]), w.conv[c * Layout::taps + i], sum);
            tap[i] = samples[i];
        }
        const float den = Arithmetic::sigmoid_denominator(sum, t);
        value[part] = round_bf16(__fdiv_rn(sum, den));
        square[part] = __fmul_rn(value[part], value[part]);
    }
    if (blockIdx.x < 2U * Layout::key_heads) {
        const float inverse = am::amd_rsqrtf(t, __fadd_rn(Arithmetic::sum_head(square), w.l2_eps));
        const float q_scale = blockIdx.x < Layout::key_heads
            ? am::amd_rsqrtf(t, static_cast<float>(Layout::head_dim)) : 1.0F;
#pragma unroll
        for (unsigned part = 0U; part < 4U; ++part)
            value[part] = __fmul_rn(__fmul_rn(value[part], inverse), q_scale);
    }
#pragma unroll
    for (unsigned part = 0U; part < 4U; ++part) row.prepared[base + part * 32U + lane] = value[part];
}

template <class Arithmetic>
__global__ __launch_bounds__(256) void decode_front(DecodeWeights w, DecodeRow row, std::uint16_t* history, am::Tables t) {
    constexpr unsigned conv_blocks = 2U * Layout::key_heads + Layout::value_heads;
    if (blockIdx.x < conv_blocks) convolve_decode<Arithmetic>(w, row, history, t);
    else project_gate<Arithmetic>(w, row, t, blockIdx.x - conv_blocks);
}

struct LanesRow {
    const float* qkv;
    const float* beta;
    const float* gate;
    float* core;
    std::uint64_t stride;
};

constexpr unsigned kLanes = 4;

template <RecurrentStateType Type>
__global__ __launch_bounds__(lanes::RecShape<kLanes>::Threads) void decode_delta(
    LanesRow row, const void* matrix_in, void* matrix_out, am::Tables t) {
    lanes::recurrence_body<kLanes, Type, false, true>(row, 1U, static_cast<const float*>(matrix_in),
                                                     static_cast<float*>(matrix_out), nullptr, &t);
}

__device__ __forceinline__ void store_a8(float x, DecodeA8 a8, unsigned feature, unsigned lane) {
    const bool bad = __any_sync(lanes::kFull, !isfinite(x));
    float max = fabsf(x);
#pragma unroll
    for (int d = 16; d > 0; d /= 2) max = fmaxf(max, __shfl_xor_sync(lanes::kFull, max, d));
    const bool tiny = max > 0.0F && max < 0x1p-119F;
    const float delta = bad ? __uint_as_float(0x7fc00000U) : max == 0.0F || tiny ? 1.0F : __fdiv_rn(max, 127.0F);
    const float r = bad || tiny ? 0.0F : rintf(__fdiv_rn(x, delta));
    a8.codes[feature] = static_cast<std::int8_t>(static_cast<int>(fminf(127.0F, fmaxf(-127.0F, r))));
    if (lane == 0U) a8.deltas[feature / 32U * Layout::a8_pitch] = delta;
}

template <class Arithmetic, bool A8>
__global__ __launch_bounds__(256) void decode_norm(DecodeWeights w, DecodeRow row, am::Tables t, DecodeA8 a8) {
    const unsigned lane = threadIdx.x % 32U, head = blockIdx.x * 8U + threadIdx.x / 32U;
    const unsigned base = head * Layout::head_dim;
    float value[4], square[4];
#pragma unroll
    for (unsigned part = 0U; part < 4U; ++part) {
        value[part] = row.out[base + part * 32U + lane];
        square[part] = __fmul_rn(value[part], value[part]);
    }
    const float mean = __fdiv_rn(Arithmetic::sum_head(square), static_cast<float>(Layout::head_dim));
    const float inverse = am::amd_rsqrtf(t, __fadd_rn(mean, w.rms_eps));
#pragma unroll
    for (unsigned part = 0U; part < 4U; ++part) {
        const unsigned c = part * 32U + lane, feature = base + c;
        const float gate = Arithmetic::sigmoid(row.z[feature], t);
        const float result = __fmul_rn(__fmul_rn(__fmul_rn(value[part], inverse), w.gain[c]), gate);
        row.out[feature] = result;
        if constexpr (A8) store_a8(result, a8, feature, lane);
    }
}

}  // namespace

cudaError_t enqueue_decode(Gfx1151Decode, const DecodeWeights& w, DecodeRow row, DecodeState state,
                           const am::Tables& t, DecodeA8 a8, cudaStream_t stream) {
    if (!w.beta || !w.alpha || !w.dt || !w.a || !w.conv || !w.gain || !row.input || !row.qkv || !row.z ||
        !row.prepared || !row.beta || !row.log_decay || !row.out || !state.history || !state.matrix_in ||
        !state.matrix_out || reinterpret_cast<std::uintptr_t>(row.prepared) % 16U ||
        !t.exp_bits || !t.rsq_bits || !t.rcp_bits || (bool(a8.codes) != bool(a8.deltas)))
        return cudaErrorInvalidValue;
    if (state.type != RecurrentStateType::f32 && state.type != RecurrentStateType::f16 &&
        state.type != RecurrentStateType::bf16)
        return cudaErrorInvalidValue;

    decode_front<DecodeArithmetic><<<2U * Layout::key_heads + 3U * Layout::value_heads, 256, 0, stream>>>(
        w, row, state.history, t);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;

    const LanesRow prepared{row.prepared, row.beta, row.log_decay, row.out, Layout::channels};
    const auto launch = [&]<RecurrentStateType Type>() {
        decode_delta<Type><<<Layout::value_heads * (Layout::head_dim / 32U), lanes::RecShape<kLanes>::Threads, 0, stream>>>(
            prepared, state.matrix_in, state.matrix_out, t);
    };
    switch (state.type) {
    case RecurrentStateType::f32: launch.template operator()<RecurrentStateType::f32>(); break;
    case RecurrentStateType::f16: launch.template operator()<RecurrentStateType::f16>(); break;
    case RecurrentStateType::bf16: launch.template operator()<RecurrentStateType::bf16>(); break;
    }
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;

    static_assert(Layout::value_heads % 8U == 0U);
    if (a8.codes) decode_norm<DecodeArithmetic, true><<<Layout::value_heads / 8U, 256, 0, stream>>>(w, row, t, a8);
    else decode_norm<DecodeArithmetic, false><<<Layout::value_heads / 8U, 256, 0, stream>>>(w, row, t, a8);
    return cudaGetLastError();
}

}  // namespace strix::cuda::recur
