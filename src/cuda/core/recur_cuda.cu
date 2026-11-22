// Native GDN with S1 image selection and retained-input replay.
#include "recur_cuda.cuh"

#include "core/short_convolution.hpp"
#include "recur_math.cuh"

#include <algorithm>
#include <cmath>

namespace strix::cuda::recur {
namespace {

using cuda_core::cc_expf;
using cuda_core::cc_sigmoid;
using cuda_core::cc_silu;

__device__ __forceinline__ float* image(StateView view, std::uint32_t slot) {
    return view.images + view.stride * slot;
}

// Same XOR butterfly and ascending warp fold as the donor's block_sum128.
// Every lane participates, including any warps used only to stage keys.
__device__ __forceinline__ float sum_warps(float v, float* shared, std::uint32_t warps) {
    v = cuda_core::warp_sum_xor(v);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = v;
    __syncthreads();
    float sum = shared[0];
#pragma unroll
    for (std::uint32_t w = 1; w < 4; ++w)
        if (w < warps) sum = __fadd_rn(sum, shared[w]);
    __syncthreads();
    return sum;
}

struct TapSource {
    const float* previous;
    const float* raw;
    std::uint32_t channels, channel, history;

    __device__ __forceinline__ float sample(std::uint32_t index) const {
        if (index < history) return previous[index];
        return raw[std::uint64_t{index - history} * channels + channel];
    }

    __device__ __forceinline__ void store(float* next, std::uint32_t accepted) const {
        // In-place writes happen only at the end. Any history source in
        // the same image is strictly ahead of the destination tap.
        for (std::uint32_t tap = 0; tap < history; ++tap) next[tap] = sample(accepted + tap);
    }
};

template <int FixedTaps = 0>
__global__ void __launch_bounds__(128) conv_taps_kernel(Plan plan, StateView state, const float* weights,
                                                        const float* raw, float* prepared, std::uint32_t prepared_stride,
                                                        std::uint32_t steps, float qscale) {
    __shared__ float red[4];
    const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= plan.channels) return;  // only possible in a V block, which has no collective
    const bool qk = blockIdx.x < 2 * plan.dims.key_heads;
    const float scale = blockIdx.x < plan.dims.key_heads ? qscale : 1.0f;
    const std::uint32_t taps = FixedTaps ? FixedTaps : plan.dims.taps;
    const std::uint32_t history = taps - 1;
    const auto offset = std::uint64_t{c} * history;
    const TapSource source{image(state, state.published) + offset, raw, plan.channels, c, history};
    const float* w = weights + std::uint64_t{c} * taps;
    for (std::uint32_t r = 0; r < steps; ++r) {
        const float sum = core::causal_conv::weighted_taps<true, true>(taps,
            [&](std::uint32_t tap) { return source.sample(r + tap); }, [&](std::uint32_t tap) { return w[tap]; });
        float y = cc_silu(sum);
        if (qk) {
            const float ss = sum_warps(__fmul_rn(y, y), red, blockDim.x / 32);
            const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, 1e-6f)));
            y = __fmul_rn(__fmul_rn(y, inv), scale);
        }
        prepared[std::uint64_t{r} * prepared_stride + c] = y;
        if (plan.history.write == Write::replay_input)
            state.input[std::uint64_t{r} * plan.channels + c] = source.sample(r + history);
        if (plan.history.write == Write::ring_slot || r + 1 == steps) {
            float* next = image(state, image_slot(plan.history.write, state, r)) + offset;
            source.store(next, r + 1);
        }
    }
}

__device__ __forceinline__ void replay_taps(Plan plan, StateView state, std::uint32_t block, std::uint32_t steps) {
    const std::uint32_t c = block * blockDim.x + threadIdx.x;
    if (c >= plan.channels) return;
    const auto history = plan.dims.taps - 1;
    const auto offset = std::uint64_t{c} * history;
    const TapSource source{image(state, state.published) + offset, state.input, plan.channels, c, history};
    source.store(image(state, image_slot(Write::replay_input, state, steps - 1)) + offset, steps);
}

template <std::uint32_t K, bool Replay>
__global__ void __launch_bounds__(128) delta_step_kernel(Plan plan, StateView state, StateView history, Weights w, Rows rows,
                                                         const float* prepared, std::uint32_t stride,
                                                         std::uint32_t steps) {
    __shared__ float ks[K], qs[K], red[4];
    const std::uint32_t h = blockIdx.x, t = threadIdx.x;
    if constexpr (Replay) {
        const auto matrix_blocks = plan.matrix.write == Write::replay_input ? plan.dims.value_heads : 0U;
        if (h >= matrix_blocks) {
            replay_taps(plan, history, h - matrix_blocks, steps);
            return;
        }
    }
    const auto kh = h % plan.dims.key_heads;
    const auto V = plan.dims.value_dim;
    const auto col = h * V + t;
    const auto offset = std::uint64_t{h} * K * V + t;
    const bool value_lane = t < V;
    float S[K];
    if (value_lane) {
        const float* previous = image(state, state.published) + offset;
#pragma unroll
        for (std::uint32_t k = 0; k < K; ++k) S[k] = previous[k * V];
    }
    const auto store = [&](std::uint32_t r) {
        const auto write = Replay ? Write::replay_input : plan.matrix.write;
        float* next = image(state, image_slot(write, state, r)) + offset;
#pragma unroll
        for (std::uint32_t k = 0; k < K; ++k) next[k * V] = S[k];
    };

    for (std::uint32_t r = 0; r < steps; ++r) {
        const float* row = prepared + std::uint64_t{r} * stride;
        if (t < K) {
            ks[t] = row[plan.qk_width + kh * K + t];
            if constexpr (!Replay) qs[t] = row[kh * K + t];
        }
        float a, b;
        if constexpr (Replay) {
            a = row[plan.channels + h];
            b = row[plan.channels + plan.dims.value_heads + h];
        } else {
            const auto gate = std::uint64_t{r} * plan.dims.value_heads + h;
            a = rows.a[gate];
            b = rows.b[gate];
            if (t == 0 && plan.matrix.write == Write::replay_input) {
                float* log = state.input + std::uint64_t{r} * stride + plan.channels;
                log[h] = a;
                log[plan.dims.value_heads + h] = b;
            }
        }
        const float beta = cc_sigmoid(b);
        const float sp = delta_softplus(__fadd_rn(a, w.dt_bias[h]));
        const float decay = cc_expf(__fmul_rn(w.ssm_a[h], sp));
        __syncthreads();

        float acc = 0.0f;
        if (value_lane) {
            float kv = 0.0f;
#pragma unroll
            for (std::uint32_t k = 0; k < K; ++k) {
                S[k] = __fmul_rn(S[k], decay);
                kv = add_product(kv, S[k], ks[k]);
            }
            const float dl = __fmul_rn(__fsub_rn(row[2 * plan.qk_width + col], kv), beta);
            // Keep the donor's reload fence between the two ordered passes.
            asm volatile("" ::: "memory");
#pragma unroll
            for (std::uint32_t k = 0; k < K; ++k) {
                S[k] = add_product(S[k], ks[k], dl);
                if constexpr (!Replay) acc = add_product(acc, S[k], qs[k]);
            }
            if constexpr (!Replay) {
                if (plan.matrix.write == Write::ring_slot || r + 1 == steps) store(r);
            }
        }
        if constexpr (!Replay) {
            const float ss = sum_warps(__fmul_rn(acc, acc), red, V / 32);
            if (value_lane) {
                const auto i = std::uint64_t{r} * plan.value_width + col;
                const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(V)), w.rms_eps)));
                float v = __fmul_rn(acc, inv);
                v = __fmul_rn(v, w.gain[t]);
                rows.out[i] = __fmul_rn(v, cc_silu(rows.z[i]));
            }
        } else {
            __syncthreads();
        }
    }
    if constexpr (Replay) {
        if (value_lane) store(steps - 1);
    }
}

bool supports(StatePlan plan, StateView view, std::uint32_t steps) {
    if (!view.images || view.stride < plan.image_floats || view.published >= view.slots) return false;
    switch (plan.write) {
        case Write::in_place: return true;
        case Write::ring_slot: return steps < view.slots;
        case Write::replay_input: return view.slots == 3 && view.input && steps <= view.input_rows;
    }
    return false;
}

cudaError_t launch_conv(Plan plan, StateView history, const float* w, const float* raw, float* prepared,
                        std::uint32_t stride, std::uint32_t steps, cudaStream_t stream) {
    const auto K = plan.dims.key_dim;
    const auto blocks = (plan.channels + K - 1) / K;
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(K)));
    if (plan.dims.taps == 4)
        conv_taps_kernel<4><<<blocks, K, 0, stream>>>(plan, history, w, raw, prepared, stride, steps, qscale);
    else
        conv_taps_kernel<><<<blocks, K, 0, stream>>>(plan, history, w, raw, prepared, stride, steps, qscale);
    return cudaGetLastError();
}

template <bool Replay>
cudaError_t launch_delta(Plan plan, StateView matrix, StateView history, const Weights& weights, Rows rows,
                         const float* prepared, std::uint32_t stride, std::uint32_t steps, cudaStream_t stream) {
    const auto threads = std::max(plan.dims.key_dim, plan.dims.value_dim);
    auto blocks = plan.dims.value_heads;
    if constexpr (Replay) {
        if (plan.matrix.write != Write::replay_input) blocks = 0;
        // Independent tap CTAs use the retained raw rows while the matrix
        // CTAs use the retained prepared rows. One replay dispatch covers both.
        if (plan.history.write == Write::replay_input) blocks += (plan.channels + threads - 1) / threads;
    }
    const auto launch = [&]<std::uint32_t K>() {
        delta_step_kernel<K, Replay><<<blocks, threads, 0, stream>>>(
            plan, matrix, history, weights, rows, prepared, stride, steps);
    };
    switch (plan.dims.key_dim) {
        case 32: launch.template operator()<32>(); break;
        case 64: launch.template operator()<64>(); break;
        case 96: launch.template operator()<96>(); break;
        case 128: launch.template operator()<128>(); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace

cudaError_t enqueue(const Plan& plan, const Weights& w, Rows rows, StateView matrix, StateView history,
                    std::uint32_t steps, cudaStream_t stream) {
    const bool capture = plan.matrix.write == Write::replay_input;
    if (!steps || steps > max_steps || !supports(plan.matrix, matrix, steps) || !supports(plan.history, history, steps) ||
        !rows.raw || !rows.a || !rows.b || !rows.z || !rows.out || (!capture && !rows.prepared) ||
        !w.conv || !w.ssm_a || !w.dt_bias || !w.gain)
        return cudaErrorInvalidValue;
    float* prepared = capture ? matrix.input : rows.prepared;
    const auto stride = capture ? plan.matrix.replay_floats : plan.channels;
    if (const auto error = launch_conv(plan, history, w.conv, rows.raw, prepared, stride, steps, stream);
        error != cudaSuccess)
        return error;
    return launch_delta<false>(plan, matrix, {}, w, rows, prepared, stride, steps, stream);
}

cudaError_t enqueue_replay(const Plan& plan, const Weights& w, StateView matrix, StateView history,
                           std::uint32_t accepted, cudaStream_t stream) {
    const bool matrix_replay = plan.matrix.write == Write::replay_input;
    const bool history_replay = plan.history.write == Write::replay_input;
    if ((!matrix_replay && !history_replay) || accepted > max_steps) return cudaErrorInvalidValue;
    if (!accepted) return cudaSuccess;
    if ((matrix_replay && (!supports(plan.matrix, matrix, accepted) || !w.ssm_a || !w.dt_bias)) ||
        (history_replay && !supports(plan.history, history, accepted)))
        return cudaErrorInvalidValue;
    return launch_delta<true>(plan, matrix, history, w, {}, matrix.input, plan.matrix.replay_floats, accepted, stream);
}

}  // namespace strix::cuda::recur
