#pragma once

#include "core/residual_mix.hpp"
#include "step_rows.cuh"

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
    constexpr bool Masked = false;
    [[maybe_unused]] const core::StepRows<1> binding{};
#include "mix_body.inc"
}

template <unsigned Width, bool Normalized, bool ParallelBanks>
__global__ void bank_softmax_step(MixArgs args, core::StepRows<Width> binding) {
    constexpr bool Masked = true;
#include "mix_body.inc"
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

// launch_mix (core/residual_mix.hpp) binds enqueue_selected; every include order of the mix headers completes this
// header's body before the latency form's.
#include "mix_cuda_latency.cuh"
