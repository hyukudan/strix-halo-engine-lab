#include "recur_cuda.cuh"

#include "chain/key.hpp"
#include "common.cuh"
#include "recur_lanes.cuh"
#include "step_rows.cuh"

#include <limits>
#include <type_traits>

namespace strix::cuda::recur {
namespace {

namespace lanes = cuda_core::gdn_lanes;
namespace am = numeric::gfx1151;
namespace chain = ::strix::cuda::core::chain;
using namespace ::strix::ops::recur;
using chain::Math;

template <GdnDecodeShape Shape>
struct StripedLayout : GdnDecodeLayout<Shape> {
    static_assert(Shape.head_dim == 4U * 32U && Shape.hidden % 256U == 0U && Shape.value_heads % 8U == 0U);
    static_assert(Shape.history_elements() <= std::numeric_limits<unsigned>::max() / 16U);
    static constexpr auto geometry = Shape;
    static constexpr unsigned conv_blocks = 2U * Shape.key_heads + Shape.value_heads;
    static constexpr unsigned gate_blocks = 2U * Shape.value_heads;
    static constexpr unsigned delta_blocks = Shape.value_heads * (Shape.head_dim / 32U);
};

// The finite-value rounding contract of striped GDN. Unlike the state
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

template <class Layout, class Arithmetic>
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

template <class Layout, class Arithmetic>
__device__ __forceinline__ void convolve_decode(const DecodeWeights& w, const DecodeRow& row,
                                                const std::uint16_t* history, std::uint16_t* next,
                                                const am::Tables& t) {
    const unsigned lane = threadIdx.x;
    if (lane >= 32U) return;
    const unsigned base = blockIdx.x * Layout::head_dim;
    float value[4], square[4];
#pragma unroll
    for (unsigned part = 0U; part < 4U; ++part) {
        const unsigned c = base + part * 32U + lane;
        const float raw = row.qkv[c];
        if (row.raw_capture) row.raw_capture[c] = raw;
        const auto* tap = history + c * Layout::taps;
        std::uint16_t samples[Layout::taps];
#pragma unroll
        for (unsigned i = 0U; i < Layout::taps - 1U; ++i) samples[i] = tap[i + 1U];
        samples[Layout::taps - 1U] = narrow(raw);
        float sum = 0.0F;
#pragma unroll
        for (unsigned i = 0U; i < Layout::taps; ++i) {
            sum = __fmaf_rn(cuda_core::bf16_lo(samples[i]), w.conv[c * Layout::taps + i], sum);
            next[c * Layout::taps + i] = samples[i];
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

template <class Layout, class Arithmetic>
__global__ __launch_bounds__(256) void decode_front(DecodeWeights w, DecodeRow row, std::uint16_t* history, am::Tables t) {
    constexpr unsigned conv_blocks = 2U * Layout::key_heads + Layout::value_heads;
    if (blockIdx.x < conv_blocks) convolve_decode<Layout, Arithmetic>(w, row, history, history, t);
    else project_gate<Layout, Arithmetic>(w, row, t, blockIdx.x - conv_blocks);
}

struct LanesRow {
    const float* qkv;
    const float* beta;
    const float* gate;
    float* core;
    std::uint64_t stride;
};

constexpr unsigned kLanes = 4;

template <class Layout, RecurrentStateType Type>
__global__ __launch_bounds__(lanes::RecShape<kLanes, Layout::head_dim>::Threads) void decode_delta(
    LanesRow row, const void* matrix_in, void* matrix_out, am::Tables t) {
    lanes::recurrence_body<kLanes, Type, false, true, Layout>(row, 1U, static_cast<const float*>(matrix_in),
                                                     static_cast<float*>(matrix_out), nullptr, &t);
}

template <class Layout>
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

template <class Layout, class Arithmetic, bool A8, class Binding = DecodeRow>
__global__ __launch_bounds__(256) void decode_norm(DecodeWeights w, Binding binding, am::Tables t, DecodeA8 a8) {
    if constexpr (std::is_same_v<Binding, DecodeStep>) {
        if (!decode_source(binding, blockIdx.y).history) return;
    } else if constexpr (std::is_same_v<Binding, DecodeReplayStep>) {
        if (blockIdx.y >= decode_replay_images<Layout::geometry>(gridDim.y, binding).rows) return;
    }
    if constexpr (!std::is_same_v<Binding, DecodeRow>) {
        if constexpr (A8) {
            a8.codes += blockIdx.y * binding.strides.a8_codes;
            a8.deltas += blockIdx.y;
        }
    }
    const DecodeRow row = [&]() {
        if constexpr (!std::is_same_v<Binding, DecodeRow>) return decode_row(binding, blockIdx.y);
        else return binding;
    }();
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
        if constexpr (A8) store_a8<Layout>(result, a8, feature, lane);
    }
}

template <class Layout, unsigned R, class Binding>
__device__ __forceinline__ DecodeReplayImages replay_images(const Binding& step) {
    if constexpr (std::is_same_v<Binding, DecodeReplayStep>) return decode_replay_images<Layout::geometry>(R, step);
    else return {};
}

template <class Layout, unsigned R, class Binding>
__device__ __forceinline__ void decode_step_front_body(DecodeWeights w, const Binding& step, am::Tables t) {
    constexpr bool replay = std::is_same_v<Binding, DecodeReplayStep>;
    const auto images = replay_images<Layout, R>(step);
    const unsigned block = blockIdx.x;
    if (block >= Layout::conv_blocks) {
        const unsigned gate_block = block - Layout::conv_blocks;
        const unsigned r = gate_block / Layout::gate_blocks;
        const bool valid = [&]() {
            if constexpr (replay) return r < images.rows;
            else return decode_source(step, r).history != nullptr;
        }();
        if (valid)
            project_gate<Layout, DecodeArithmetic>(w, decode_row(step, r), t, gate_block % Layout::gate_blocks);
        return;
    }

    // The first warp owns the same channels in every row. It consumes only
    // the pending taps it wrote for the parent, even when QKV is overwritten.
    if (threadIdx.x >= 32U) return;
    for (unsigned r = 0U; r < (replay ? images.rows : R); ++r) {
        const std::uint16_t* history;
        std::uint16_t* next;
        if constexpr (replay) {
            next = images.pending_history;
            history = r == 0U ? images.history : next;
        } else {
            const auto src = decode_source(step, r);
            if (!src.history) continue;
            next = step.pending_history + r * Layout::history_elements;
            history = src.parent == exec::kStepNoParent
                ? reinterpret_cast<const std::uint16_t*>(src.history->base)
                : step.pending_history + src.parent * Layout::history_elements;
        }
        convolve_decode<Layout, DecodeArithmetic>(w, decode_row(step, r), history, next, t);
    }
}

template <class Layout, unsigned R>
__global__ __launch_bounds__(256) void decode_step_front(DecodeWeights w, DecodeStep step, am::Tables t) {
    decode_step_front_body<Layout, R>(w, step, t);
}

template <class Layout, unsigned R>
__global__ __launch_bounds__(256) void decode_step_front(DecodeWeights w, DecodeReplayStep step, am::Tables t) {
    decode_step_front_body<Layout, R>(w, step, t);
}

template <class Layout, unsigned R, RecurrentStateType Type, class Binding>
__device__ __forceinline__ void decode_step_delta_body(const Binding& step, am::Tables t) {
    constexpr bool replay = std::is_same_v<Binding, DecodeReplayStep>;
    const auto images = replay_images<Layout, R>(step);
    constexpr auto bytes = Layout::template matrix_bytes<Type>;
    for (unsigned r = 0U; r < (replay ? images.rows : R); ++r) {
        const float* matrix;
        float* next;
        if constexpr (replay) {
            next = images.pending_matrix;
            matrix = r == 0U ? images.matrix : next;
        } else {
            const auto src = decode_source(step, r);
            if (!src.matrix) continue;
            auto* pending = static_cast<std::byte*>(step.pending_matrix);
            next = reinterpret_cast<float*>(pending + bytes * r);
            matrix = src.parent == exec::kStepNoParent
                ? reinterpret_cast<const float*>(src.matrix->base)
                : reinterpret_cast<const float*>(pending + bytes * src.parent);
        }
        const auto row = decode_row(step, r);
        const LanesRow prepared{row.prepared, row.beta, row.log_decay, row.out, Layout::channels};
        // The shared body dots in FP32, then stores through the typed codec.
        // Each block reads back only the same cells it owned for the parent.
        lanes::recurrence_body<kLanes, Type, false, true, Layout>(
            prepared, 1U, matrix, next, nullptr, &t);
        __syncthreads();
    }
}

template <class Layout, unsigned R, RecurrentStateType Type>
__global__ __launch_bounds__(lanes::RecShape<kLanes, Layout::head_dim>::Threads) void decode_step_delta(DecodeStep step, am::Tables t) {
    decode_step_delta_body<Layout, R, Type>(step, t);
}

template <class Layout, unsigned R, RecurrentStateType Type>
__global__ __launch_bounds__(lanes::RecShape<kLanes, Layout::head_dim>::Threads) void decode_step_delta(DecodeReplayStep step, am::Tables t) {
    decode_step_delta_body<Layout, R, Type>(step, t);
}

template <class Layout>
cudaError_t enqueue_single(const DecodeWeights& w, DecodeRow row, DecodeState state,
                           const am::Tables& t, DecodeA8 a8, cudaStream_t stream) {
    if (!w.beta || !w.alpha || !w.dt || !w.a || !w.conv || !w.gain || !row.input || !row.qkv || !row.z ||
        !row.prepared || !row.beta || !row.log_decay || !row.out || !state.history || !state.matrix_in ||
        !state.matrix_out || reinterpret_cast<std::uintptr_t>(row.prepared) % 16U ||
        !t.exp_bits || !t.rsq_bits || !t.rcp_bits || (bool(a8.codes) != bool(a8.deltas)))
        return cudaErrorInvalidValue;
    if (state.type != RecurrentStateType::f32 && state.type != RecurrentStateType::f16 &&
        state.type != RecurrentStateType::bf16)
        return cudaErrorInvalidValue;

    decode_front<Layout, DecodeArithmetic><<<2U * Layout::key_heads + 3U * Layout::value_heads, 256, 0, stream>>>(
        w, row, state.history, t);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;

    const LanesRow prepared{row.prepared, row.beta, row.log_decay, row.out, Layout::channels};
    const auto launch = [&]<RecurrentStateType Type>() {
        decode_delta<Layout, Type><<<Layout::value_heads * (Layout::head_dim / 32U), lanes::RecShape<kLanes, Layout::head_dim>::Threads, 0, stream>>>(
            prepared, state.matrix_in, state.matrix_out, t);
    };
    switch (state.type) {
    case RecurrentStateType::f32: launch.template operator()<RecurrentStateType::f32>(); break;
    case RecurrentStateType::f16: launch.template operator()<RecurrentStateType::f16>(); break;
    case RecurrentStateType::bf16: launch.template operator()<RecurrentStateType::bf16>(); break;
    }
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;

    if (a8.codes) decode_norm<Layout, DecodeArithmetic, true><<<Layout::value_heads / 8U, 256, 0, stream>>>(w, row, t, a8);
    else decode_norm<Layout, DecodeArithmetic, false><<<Layout::value_heads / 8U, 256, 0, stream>>>(w, row, t, a8);
    return cudaGetLastError();
}

template <class Layout, class Binding>
cudaError_t enqueue_step(unsigned width, const DecodeWeights& w, const Binding& step,
                         const am::Tables& t, cudaStream_t stream) {
    if (!w.beta || !w.alpha || !w.dt || !w.a || !w.conv || !w.gain ||
        !t.exp_bits || !t.rsq_bits || !t.rcp_bits)
        return cudaErrorInvalidValue;
    return ::strix::cuda::core::dispatch_step_rows(width, [&]<unsigned R>() {
        decode_step_front<Layout, R><<<Layout::conv_blocks + R * Layout::gate_blocks, 256, 0, stream>>>(w, step, t);
        if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;

        const auto launch = [&]<RecurrentStateType Type>() {
            decode_step_delta<Layout, R, Type><<<Layout::delta_blocks, lanes::RecShape<kLanes, Layout::head_dim>::Threads, 0, stream>>>(step, t);
        };
        switch (step.type) {
        case RecurrentStateType::f32: launch.template operator()<RecurrentStateType::f32>(); break;
        case RecurrentStateType::f16: launch.template operator()<RecurrentStateType::f16>(); break;
        case RecurrentStateType::bf16: launch.template operator()<RecurrentStateType::bf16>(); break;
        }
        if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;

        const dim3 grid(Layout::value_heads / 8U, R);
        if (step.a8.codes) decode_norm<Layout, DecodeArithmetic, true><<<grid, 256, 0, stream>>>(w, step, t, step.a8);
        else decode_norm<Layout, DecodeArithmetic, false><<<grid, 256, 0, stream>>>(w, step, t, step.a8);
        return cudaGetLastError();
    });
}

struct CompiledLayout : StripedLayout<shape::gdn_2560_16x48x128> {};

}  // namespace

cudaError_t enqueue_decode(GdnDecodeShape shape, const DecodeWeights& w, DecodeRow row, DecodeState state,
                           const am::Tables& t, DecodeA8 a8, cudaStream_t stream) {
    if (shape != CompiledLayout::geometry) return cudaErrorInvalidValue;
    return enqueue_single<CompiledLayout>(w, row, state, t, a8, stream);
}

cudaError_t enqueue_decode_step(GdnDecodeShape shape, unsigned width, const DecodeWeights& w, const DecodeStep& step,
                                const am::Tables& t, cudaStream_t stream) {
    if (shape != CompiledLayout::geometry || !valid_decode_step(shape, width, step)) return cudaErrorInvalidValue;
    return enqueue_step<CompiledLayout>(width, w, step, t, stream);
}

cudaError_t enqueue_decode_replay_step(GdnDecodeShape shape, unsigned width, const DecodeWeights& w, const DecodeReplayStep& step,
                                       const am::Tables& t, cudaStream_t stream) {
    if (shape != CompiledLayout::geometry || !valid_decode_replay_step(shape, width, step)) return cudaErrorInvalidValue;
    return enqueue_step<CompiledLayout>(width, w, step, t, stream);
}

}  // namespace strix::cuda::recur
