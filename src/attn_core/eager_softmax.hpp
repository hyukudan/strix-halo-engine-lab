#pragma once
// F4 form "eager selected softmax": the middle pass of a three-pass (scores, softmax, values) decode over a selected
// key list. One block of Threads lanes per head row of `scores` (blockIdx.x = head); the row holds `count` scores at
// scores[head * stride + i]. The pass rewrites each score as its rounded probability in place.
//
// Contract (bits depend on exactly this; host emulation and rule in eager_softmax_rule.hpp):
//   - lane l takes i = l, l + Threads, ... < count: the maximum by fmaxf from -infinity, then (after the maximum's
//     tree) the sum of expf(s_i - maximum) in ascending i from +0.0F;
//   - both trees are the LDS halving tree over Threads lanes, level h = Threads/2 .. 1, lane l < h combining
//     scratch[l] with scratch[l + h] (fmaxf for the maximum, `+=` for the sum), one barrier per level;
//   - probability_i = round(expf(s_i - maximum) / denominator);
//   - a non-finite maximum, or a denominator that is not finite or <= 0, makes lane 0 call fail() and the block
//     returns without writing, when checks() is true.
// Policy `Control` (the only thing an engine supplies; every member is called uniformly by the whole block):
//   bool count(std::uint32_t& count) const;  graph-captured launches update count; false rejects the launch
//   bool checks() const;                     finite checks on
//   void fail() const;                       report a non-finite row (lane 0 only)
//   float round(float) const;                rounding of the stored probability (the query dtype of the engine)
// Publish: each rounded probability is also stored, unchanged, to published[i] when published is not null.
// LDS: the caller owns `scratch` (Threads floats), so each kernel keeps its LDS layout.
// Home: shared (attn_core): no model names; Threads is a template parameter.
#include <cstdint>

#include <hip/hip_runtime.h>

namespace strix::attn_core::eager_softmax {

template <unsigned Threads, bool Publish = false, class Control>
__device__ __forceinline__ void softmax_row(float* scratch, float* scores, std::uint32_t selected_count,
                                            std::uint32_t maximum_selected, const Control& control,
                                            float* published = nullptr) {
    static_assert(Threads >= 1U && (Threads & (Threads - 1U)) == 0U, "eager_softmax: Threads must be a power of two");
    if (!control.count(selected_count)) return;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);

    float local_max = -INFINITY;
    for (std::uint32_t index = lane; index < selected_count; index += Threads) {
        local_max = fmaxf(local_max, scores[static_cast<std::uint64_t>(head) * maximum_selected + index]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride = Threads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    if (control.checks() && !isfinite(maximum)) {
        if (lane == 0) control.fail();
        return;
    }
    // Preserve the reduced maximum in each lane before reusing scratch for
    // the sum reduction (including waves with no active selected element).
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t index = lane; index < selected_count; index += Threads) {
        local_sum += expf(scores[static_cast<std::uint64_t>(head) * maximum_selected + index] - maximum);
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride = Threads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];
    if (control.checks() && (!isfinite(denominator) || denominator <= 0.0F)) {
        if (lane == 0) control.fail();
        return;
    }

    for (std::uint32_t index = lane; index < selected_count; index += Threads) {
        const auto location = static_cast<std::uint64_t>(head) * maximum_selected + index;
        if constexpr (Publish) {
            const float probability = control.round(expf(scores[location] - maximum) / denominator);
            scores[location] = probability;
            if (published != nullptr) published[index] = probability;
        } else
            scores[location] = control.round(expf(scores[location] - maximum) / denominator);
    }
}

}  // namespace strix::attn_core::eager_softmax
