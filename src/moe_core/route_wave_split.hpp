#pragma once
// Split-plane summed-score router (one wave per row): the device form of moe::route_split_logits_ref with
// SplitRouteMode::summed_scores (include/strix/moe/route_policy.hpp). Two raw F32 logit planes [rows][experts];
// s_e = sigmoid(z1_e) + sigmoid(z2_e) (each score through Math::exp, then one F32 add), selection key s_e + bias_e,
// the k picks of moe_core::route_wave_topk (ties -> lower id), the score sum from +0 in SELECTION order, denominator
// max(sum, renorm_floor), w = ((s / den) * scale) * gain_e. The outputs are the same ids and bits as the host row;
// with AscendingOut the slots are ordered by ascending id (the order of the shared FP32-step combine), else by pick.
// The policy is the checkpoint's RoutePolicy (sigmoid, selection bias, selection order, renormalised).
#include "moe_core/route.hpp"
#include "moe_core/route_wave_kernel.hpp"

#if defined(__HIPCC__)
namespace strix::moe_core {

template <const moe::RoutePolicy& P, class Math, bool AscendingOut>
__global__ void route_wave_split_summed_kernel(const float* plane_select, const float* plane_mix, const float* bias,
                                               const float* gain, float renorm_floor, float scale, std::uint32_t* ids,
                                               float* w) {
    static_assert(moe::validate(P) == nullptr, "route_wave_split_summed_kernel: invalid policy");
    static_assert(P.score == moe::RouteScore::sigmoid && P.selection_bias && P.order == moe::RouteOrder::selection &&
                      P.renormalize && P.top_k > 1U && P.top_k <= 32U,
                  "route_wave_split_summed_kernel: sigmoid + selection bias, selection order, renormalised");
    using Pol = RouteWavePolicy<P>;
    constexpr std::uint32_t experts = P.experts, top_k = P.top_k;
    constexpr std::uint32_t all = Pol::candidates == 32U ? ~0U : (1U << Pol::candidates) - 1U;
    __shared__ float sg[experts];
    __shared__ std::uint32_t pick_s[top_k];
    const unsigned r = blockIdx.x, lane = threadIdx.x;
    const float* l1 = plane_select + static_cast<std::size_t>(r) * experts;
    const float* l2 = plane_mix + static_cast<std::size_t>(r) * experts;
    float key[Pol::candidates];
#pragma unroll
    for (unsigned i = 0; i < Pol::candidates; ++i) {
        const unsigned e = lane + 32U * i;
        const float v = moe::sigmoid_score<Math>(l1[e]) + moe::sigmoid_score<Math>(l2[e]);
        sg[e] = v;
        key[i] = __builtin_bit_cast(float, moe::route_ordered_key(v + bias[e]));
    }
    route_wave_topk(key, all, experts, top_k, lane, Pol{lane, pick_s});
    __syncthreads();
    float sum = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < top_k; ++k) sum = sum + sg[pick_s[k]];
    const float den = renorm_floor > 0.0F ? (renorm_floor > sum ? renorm_floor : sum) : sum + P.renorm_eps;
    if (lane >= top_k) return;
    const std::uint32_t id = pick_s[lane];
    std::uint32_t slot = lane;
    if constexpr (AscendingOut) {  // the picks are distinct ids
        slot = 0U;
#pragma unroll
        for (unsigned k = 0; k < top_k; ++k) slot += pick_s[k] < id ? 1U : 0U;
    }
    float weight = (sg[id] / den) * scale;
    if (gain) weight *= gain[id];
    ids[static_cast<std::size_t>(r) * top_k + slot] = id;
    w[static_cast<std::size_t>(r) * top_k + slot] = weight;
}

// gain == nullptr: no per-expert gain. rows >= 1.
template <const moe::RoutePolicy& P, class Math, bool AscendingOut>
hipError_t enqueue_route_wave_split_summed(const float* plane_select, const float* plane_mix, const float* bias,
                                           const float* gain, float renorm_floor, float scale, std::uint32_t* ids,
                                           float* w, unsigned rows, hipStream_t stream) {
    if (rows == 0U || !plane_select || !plane_mix || !bias || !ids || !w) return hipErrorInvalidValue;
    hipLaunchKernelGGL((route_wave_split_summed_kernel<P, Math, AscendingOut>), dim3(rows), dim3(32U), 0, stream,
                       plane_select, plane_mix, bias, gain, renorm_floor, scale, ids, w);
    return hipGetLastError();
}

}  // namespace strix::moe_core
#endif
