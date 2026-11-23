#pragma once

#include "mix_cuda.cuh"
#include "mix_cuda_latency_plan.hpp"
#include "common.cuh"

#include <type_traits>

namespace strix::cuda::mix {
namespace detail {

// Each physical bank lane has already combined virtual lanes x and x+128.
// Each lower warp lane then evaluates exactly the 64 and 32 levels of that
// tree; the remaining 16..1 levels exchange the same operands via shuffles.
__device__ __forceinline__ float bank_halving_lane_zero(float part, unsigned lane, float* scratch) {
    scratch[lane] = part;
    __syncthreads();
    if (lane < 32U) {
        const float low = part + scratch[lane + 64U];
        const float high = scratch[lane + 32U] + scratch[lane + 96U];
        float sum = low + high;
#pragma unroll
        for (unsigned delta = 16U; delta != 0U; delta >>= 1U)
            sum = sum + __shfl_down_sync(0xffffffffU, sum, delta);
        return sum;
    }
    return 0.0F;
}

// CUDA copies of the register-form loads of the legacy residual readout (src/core/residual_mix.hpp keeps them for
// HIP only since the K8 binding): the same strided order and the same FMA chain (square_partial<kThreads> on the
// loaded values).
__device__ __forceinline__ void latency_load_strided(const float* row, unsigned width, unsigned lane,
                                                     float (&v)[residual::detail::kMixRegs]) {
#pragma unroll
    for (unsigned k = 0; k < residual::detail::kMixRegs; ++k) {
        const unsigned i = lane + k * residual::detail::kThreads;
        v[k] = i < width ? row[i] : 0.0F;
    }
}

__device__ __forceinline__ float latency_square_chain(const float (&v)[residual::detail::kMixRegs], unsigned width,
                                                      unsigned lane) {
    float sum = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < residual::detail::kMixRegs; ++k) {
        if (lane + k * residual::detail::kThreads < width) sum = __builtin_fmaf(v[k], v[k], sum);
    }
    return sum;
}

// Job addressing of the batch: job q = blockIdx.y is (set, index), the single-query readout job_args(args, q). The
// tiled kernel body below is the same for MixArgs (job q = query q of the row) and for the batch, so each job's bits
// are the single-query bits; the MixArgs instance spells its addressing as before (same machine code).
struct BatchJobs {
    __device__ __forceinline__ static const float* query(const residual::BankBatchArgs& args, unsigned q) {
        const auto job = residual::locate(args, q);
        return args.sets[job.set].queries + std::size_t{job.index} * args.width;
    }
    __device__ __forceinline__ static float* output(const residual::BankBatchArgs& args, unsigned q) {
        const auto job = residual::locate(args, q);
        return args.sets[job.set].output + std::size_t{job.index} * args.width;
    }
};

template <bool Normalized, class Args = MixArgs, class Jobs = void>
__global__ void __launch_bounds__(kLatencyBankThreads * residual::kMaxBanks)
bank_softmax_tiled(const __grid_constant__ Args args) {
    static_assert(kLatencyBankThreads == kPairedLanes);
    namespace reference = residual::detail;
    static_assert(kLatencyMaxWidth <= reference::kMixRegisterWidth);
    __shared__ float scratch[residual::kMaxBanks][kPairedLanes];
    __shared__ float inverse[residual::kMaxBanks];
    __shared__ float probabilities[residual::kMaxBanks];
    const unsigned lane = threadIdx.x, bank = threadIdx.y, q = blockIdx.y;
    const float width = static_cast<float>(args.width);
    const float score_scale = 1.0F / sqrtf(width);
    cuda_core::pdl_wait();
    cuda_core::pdl_trigger();
    const auto* source = args.banks[bank];
    const float* query;
    if constexpr (std::is_same_v<Args, MixArgs>) query = args.queries + std::size_t{q} * args.width;
    else query = Jobs::query(args, q);
    float low[reference::kMixRegs], high[reference::kMixRegs];
    latency_load_strided(source, args.width, lane, low);
    latency_load_strided(source, args.width, lane + kPairedLanes, high);
    float scale = 1.0F;
    if constexpr (!Normalized) {
        const float norm_low = latency_square_chain(low, args.width, lane);
        const float norm_high = latency_square_chain(high, args.width, lane + kPairedLanes);
        const float squares = bank_halving_lane_zero(norm_low + norm_high, lane, scratch[bank]);
        if (lane == 0U) inverse[bank] = 1.0F / sqrtf(squares / width + args.epsilon);
        // This also retires all reads of scratch before its use for scores.
        __syncthreads();
        scale = inverse[bank];
    }
    float dot_low = 0.0F, dot_high = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < reference::kMixRegs; ++k) {
        const unsigned i = lane + k * kLanes;
        if (i < args.width) dot_low = fmaf(query[i], bank_value<Normalized>(low[k], scale), dot_low);
        if (i + kPairedLanes < args.width)
            dot_high = fmaf(query[i + kPairedLanes], bank_value<Normalized>(high[k], scale), dot_high);
    }
    const float dot = bank_halving_lane_zero(dot_low + dot_high, lane, scratch[bank]);
    if (lane == 0U) probabilities[bank] = dot * score_scale;
    __syncthreads();
    if (lane == 0U && bank == 0U) {
        float maximum = probabilities[0];
        for (unsigned b = 1; b < args.bank_count; ++b) maximum = fmaxf(maximum, probabilities[b]);
        float denominator = 0.0F;
        for (unsigned b = 0; b < args.bank_count; ++b) {
            probabilities[b] = expf(probabilities[b] - maximum);
            denominator += probabilities[b];
        }
        for (unsigned b = 0; b < args.bank_count; ++b) probabilities[b] /= denominator;
    }
    __syncthreads();
    const unsigned first = blockIdx.x * kLatencyTile;
    const unsigned end = min(first + kLatencyTile, args.width);
    const unsigned stride = blockDim.y * kPairedLanes;
    for (unsigned i = first + bank * kPairedLanes + lane; i < end; i += stride) {
        float values[residual::kMaxBanks];
#pragma unroll
        for (unsigned b = 0; b < residual::kMaxBanks; ++b)
            if (b < args.bank_count) values[b] = args.banks[b][i];
        float value = 0.0F;
#pragma unroll
        for (unsigned b = 0; b < residual::kMaxBanks; ++b)
            if (b < args.bank_count)
                value = fmaf(probabilities[b], bank_value<Normalized>(values[b], Normalized ? 1.0F : inverse[b]), value);
        if constexpr (std::is_same_v<Args, MixArgs>) args.output[q][i] = value;
        else Jobs::output(args, q)[i] = value;
    }
}

} // namespace detail

[[nodiscard]] inline cudaError_t enqueue_tiled(const ::strix::core::residual_mix::MixArgs& args, cudaStream_t stream) {
    const auto grid = latency_grid(args);
    if (grid.tiles == 0U) return cudaErrorInvalidValue;
    const auto kernel = args.normalized ? detail::bank_softmax_tiled<true> : detail::bank_softmax_tiled<false>;
    return cuda_core::launch_pdl(kernel, dim3(grid.tiles, grid.queries), dim3(kLatencyBankThreads, grid.banks), stream, args);
}

// One launch for every job of the batch; each job's bits are those of enqueue_tiled with that single query.
[[nodiscard]] inline cudaError_t enqueue_batch(const ::strix::core::residual_mix::BankBatchArgs& args, cudaStream_t stream) {
    using Args = ::strix::core::residual_mix::BankBatchArgs;
    const auto grid = latency_batch_grid(args);
    if (grid.tiles == 0U) return cudaErrorInvalidValue;
    const auto kernel = args.normalized ? detail::bank_softmax_tiled<true, Args, detail::BatchJobs>
                                        : detail::bank_softmax_tiled<false, Args, detail::BatchJobs>;
    return cuda_core::launch_pdl(kernel, dim3(grid.tiles, grid.queries), dim3(kLatencyBankThreads, grid.banks), stream, args);
}

} // namespace strix::cuda::mix
