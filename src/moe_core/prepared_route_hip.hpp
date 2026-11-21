#pragma once

#include "moe_core/route_wave_kernel.hpp"

namespace strix::moe_core {

// Checked finite-logit entry for resident device plans. Reuses the existing
// wave sorter. Its ordered keys give the same comparisons and ascending-ID
// tie rule as an ascending finite float scan, including signed zero.
template<const moe::RoutePolicy& P, class Math>
__global__ __launch_bounds__(32) void checked_route_wave_kernel(const float* logits, const float* bias,
                                                               unsigned* ids, float* weights, unsigned* status) {
    static_assert(moe::validate(P) == nullptr && P.score == moe::RouteScore::sigmoid && P.selection_bias &&
                  P.renormalize && (P.top_k > 1U || P.renorm_single));
    using Policy = RouteWavePolicy<P>;
    constexpr unsigned candidates = Policy::candidates, top_k = P.top_k;
    constexpr unsigned all = candidates == 32U ? ~0U : (1U << candidates) - 1U;
    const unsigned lane = threadIdx.x, row = blockIdx.x;
    const auto expert_base = std::size_t(row) * P.experts;
    const auto route_base = std::size_t(row) * top_k;
    __shared__ float scores[P.experts];
    __shared__ unsigned picked[top_k];
    float keys[candidates];
    bool bad = false;
#pragma unroll
    for (unsigned i = 0; i < candidates; ++i) {
        const unsigned e = lane + 32U * i;
        const float logit = logits[expert_base + e], correction = bias[e];
        const float score = moe::sigmoid_score<Math>(logit);
        bad |= !isfinite(logit) || !isfinite(correction);
        scores[e] = score;
        keys[i] = __builtin_bit_cast(float, moe::route_ordered_key(score + correction));
    }
    if (__ballot(bad) != 0U) {
        for (unsigned slot = lane; slot < top_k; slot += 32U) {
            ids[route_base + slot] = P.experts;
            weights[route_base + slot] = 0.0F;
        }
        if (lane == 0U) status[row] = 1U;
        return;
    }
    route_wave_topk(keys, all, P.experts, top_k, lane, Policy{lane, picked});
    __syncthreads();
    if (lane != 0U) return;
    if constexpr (P.order == moe::RouteOrder::ascending_id) {
        for (unsigned a = 1U; a < top_k; ++a) {
            for (unsigned b = a; b > 0U && picked[b - 1U] > picked[b]; --b) {
                const unsigned id = picked[b - 1U];
                picked[b - 1U] = picked[b];
                picked[b] = id;
            }
        }
    }
    float denominator = 0.0F;
    for (unsigned k = 0U; k < top_k; ++k) denominator = __fadd_rn(denominator, scores[picked[k]]);
    denominator = __fadd_rn(denominator, P.renorm_eps);
    for (unsigned k = 0U; k < top_k; ++k) {
        ids[route_base + k] = picked[k];
        weights[route_base + k] = __fmul_rn(__fdiv_rn(scores[picked[k]], denominator), P.scale);
    }
    status[row] = 0U;
}

template<const moe::RoutePolicy& P, class Math>
hipError_t enqueue_checked_route_wave(const float* logits, const float* bias, unsigned* ids,
                                      float* weights, unsigned* status, unsigned rows, hipStream_t stream) {
    if (rows == 0U || !logits || !bias || !ids || !weights || !status) return hipErrorInvalidValue;
    hipLaunchKernelGGL((checked_route_wave_kernel<P, Math>), dim3(rows), dim3(32U), 0, stream, logits, bias,
                       ids, weights, status);
    return hipGetLastError();
}

}  // namespace strix::moe_core
