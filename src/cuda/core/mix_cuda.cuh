#pragma once

#include "core/residual_mix.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace strix::cuda::mix {
namespace detail {

namespace residual = ::strix::core::residual_mix;
namespace norm = ops::norm;
using residual::MixArgs;

// Preserve the FP32 reduction order in core/residual_mix.hpp. A short chunk
// splits each bank's 256 virtual lanes between 128 threads. The low + high
// sum is the first level of the same halving tree.
constexpr unsigned kLanes = 256U;
constexpr unsigned kPairedLanes = kLanes / 2U;
constexpr unsigned kMaxOutputStride = kPairedLanes * residual::kMaxBanks;
constexpr unsigned kGridLimit = 65535U;
static_assert(kLanes <= kMaxOutputStride && kMaxOutputStride <= 1024U);

template <bool Normalized>
__device__ __forceinline__ float bank_value(float value, float inverse) {
    if constexpr (Normalized) return value;
    return value * inverse;
}

template <bool Normalized>
__device__ __forceinline__ float dot_partial(const float* query, const float* bank, unsigned width,
                                              unsigned lane, float inverse) {
    float dot = 0.0F;
    for (unsigned i = lane; i < width; i += kLanes)
        dot = fmaf(query[i], bank_value<Normalized>(bank[i], inverse), dot);
    return dot;
}

template <bool Normalized, bool ParallelBanks>
__global__ void bank_softmax(MixArgs args) {
    constexpr unsigned bank_lanes = ParallelBanks ? kPairedLanes : kLanes;
    __shared__ float tree[ParallelBanks ? bank_lanes * residual::kMaxBanks : bank_lanes];
    __shared__ float inverse[residual::kMaxBanks];
    __shared__ float probabilities[residual::kMaxQueries][residual::kMaxBanks];
    const unsigned lane = threadIdx.x;
    const unsigned first_bank = ParallelBanks ? threadIdx.y : 0U;
    const unsigned banks_per_slice = ParallelBanks ? 1U : args.bank_count;
    const unsigned out_lane = first_bank * bank_lanes + lane;
    const unsigned out_stride = bank_lanes * blockDim.y;
    auto* scratch = tree + first_bank * bank_lanes;
    const float width = static_cast<float>(args.width);
    const float score_scale = 1.0F / sqrtf(width);

    for (std::size_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        const auto offset = row * args.width;
        for (unsigned local_bank = 0; local_bank < banks_per_slice; ++local_bank) {
            const unsigned bank = first_bank + local_bank;
            const float* source = args.banks[bank] + offset;
            float scale = 1.0F;
            if constexpr (!Normalized) {
                float squares = norm::square_partial<kLanes>(source, args.width, lane);
                if constexpr (ParallelBanks)
                    squares += norm::square_partial<kLanes>(source, args.width, lane + bank_lanes);
                const float sum = norm::lds_halving_sum<bank_lanes>(scratch, squares);
                scale = 1.0F / sqrtf(sum / width + args.epsilon);
                if (lane == 0U) inverse[bank] = scale;
            }
            for (unsigned q = 0; q < args.query_count; ++q) {
                const float* query = args.queries + std::size_t{q} * args.width;
                float dot = dot_partial<Normalized>(query, source, args.width, lane, scale);
                if constexpr (ParallelBanks)
                    dot += dot_partial<Normalized>(query, source, args.width, lane + bank_lanes, scale);
                const float score = norm::lds_halving_sum<bank_lanes>(scratch, dot) * score_scale;
                if (lane == 0U) probabilities[q][bank] = score;
            }
        }
        __syncthreads();
        if (out_lane < args.query_count) {
            float* probs = probabilities[out_lane];
            float maximum = probs[0];
            for (unsigned bank = 1; bank < args.bank_count; ++bank) maximum = fmaxf(maximum, probs[bank]);
            float denominator = 0.0F;
            for (unsigned bank = 0; bank < args.bank_count; ++bank) {
                probs[bank] = expf(probs[bank] - maximum);
                denominator += probs[bank];
            }
            for (unsigned bank = 0; bank < args.bank_count; ++bank) probs[bank] /= denominator;
        }
        __syncthreads();
        for (unsigned q = 0; q < args.query_count; ++q) {
            for (unsigned i = out_lane; i < args.width; i += out_stride) {
                float value = 0.0F;
                for (unsigned bank = 0; bank < args.bank_count; ++bank) {
                    const float scale = Normalized ? 1.0F : inverse[bank];
                    const float source = bank_value<Normalized>(args.banks[bank][offset + i], scale);
                    value = fmaf(probabilities[q][bank], source, value);
                }
                args.output[q][offset + i] = value;
            }
        }
        __syncthreads();
    }
}

template <bool ParallelBanks>
cudaError_t enqueue_schedule(const MixArgs& args, cudaStream_t stream) {
    const auto blocks = std::min(args.rows, kGridLimit);
    const dim3 threads = ParallelBanks ? dim3(kPairedLanes, args.bank_count) : dim3(kLanes);
    if (args.normalized) bank_softmax<true, ParallelBanks><<<blocks, threads, 0, stream>>>(args);
    else bank_softmax<false, ParallelBanks><<<blocks, threads, 0, stream>>>(args);
    return cudaGetLastError();
}

} // namespace detail

// F8 bank_softmax. The caller supplies the live bank prefix in order and owns
// pass reset and capture. The dense FP32 outputs must be disjoint from each
// other and from all inputs.
[[nodiscard]] inline cudaError_t enqueue(const ::strix::core::residual_mix::MixArgs& args, cudaStream_t stream) {
    if (args.rows == 0U) return cudaSuccess;
    // The last lane increment must not wrap the unsigned feature index.
    constexpr auto max_width = std::numeric_limits<unsigned>::max() - detail::kMaxOutputStride + 1U;
    if (!::strix::core::residual_mix::fits(args) || args.width > max_width)
        return cudaErrorInvalidValue;
    if (args.rows <= 2U && args.bank_count > 1U) return detail::enqueue_schedule<true>(args, stream);
    return detail::enqueue_schedule<false>(args, stream);
}

} // namespace strix::cuda::mix
