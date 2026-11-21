#pragma once
// MoE router (one wave per row) and expert-output combine: the HIP kernels of the shared route / combine family.
// Contracts and host oracles: include/strix/moe/route_policy.hpp (RoutePolicy, presets, route_select_ref) and
// include/strix/moe/combine.hpp (CombineForm, combine_ref). A family binds a preset (the policy is a template argument
// referring to the constexpr preset object, so every constant is an immediate) and its Math trait (exp with the family's
// bits). Compile with the engine's strict arithmetic flags (-ffp-contract=off -fno-fast-math, wave32).
//
// route_wave_kernel: lane l owns experts l + 32 i (experts % 32 == 0, experts <= 1024). Score s = sigmoid_score<Math>(z),
// key = route_ordered_key(s + bias); the k-pass selection is moe_core::route_wave_topk (XOR tree on the key bits, ties ->
// lower id), so the picks equal route_select_ref with RouteKey::ordered_bits. Lane 0 orders the picks (ascending id when
// the policy says so), sums the scores in output order from +0, adds renorm_eps and writes w = (s / den) * scale.
// Lifted from GLM-5 (src/glm/kernels/glm_tier_a.hip route_kernel / moe_combine_kernel). Checked finite rows may
// also bind float_serial: the ordered keys give the same picks for finite choices, including ties at signed zero.
#include "moe_core/route.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/moe/combine.hpp"
#include "strix/moe/route_policy.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strix::moe_core {

template <const moe::RoutePolicy& P>
struct RouteWavePolicy {
    static constexpr std::uint32_t candidates = P.experts / 32, wave = 32;
    static_assert(P.experts % 32 == 0 && candidates <= 32, "route_wave_kernel: experts must fill the wave candidate mask");
    std::uint32_t lane;
    std::uint32_t* pick;
    template <class T>
    __device__ static T xor32(T value, int mask) { return strix::qwen_dpp::xor32(value, static_cast<unsigned>(mask)); }
    __device__ static bool better(float a, std::uint32_t ai, float b, std::uint32_t bi) {
        const std::uint32_t ka = __builtin_bit_cast(std::uint32_t, a), kb = __builtin_bit_cast(std::uint32_t, b);
        return ka > kb || (ka == kb && ai < bi);
    }
    __device__ void record(std::uint32_t slot, bool, float, std::uint32_t index) const {
        if (lane == 0) pick[slot] = index;  // every slot has a candidate: top_k <= experts
    }
};

// logits / bias: [rows][experts] / [experts]; ids / w: [rows][top_k]. Grid: rows blocks of 32 threads.
template <const moe::RoutePolicy& P, class Math, class Traits>
__device__ __forceinline__ void route_wave_body(const float* logits, const float* bias, std::uint32_t* ids,
                                               float* w, unsigned* status, float* sg, std::uint32_t* pick_s) {
    static_assert(moe::validate(P) == nullptr, "route_wave_kernel: invalid policy");
    static_assert(P.score == moe::RouteScore::sigmoid && P.selection_bias &&
                      (P.key == moe::RouteKey::ordered_bits || Traits::checked_status) &&
                      P.renormalize && (P.top_k > 1U || P.renorm_single),
                  "route_wave_kernel: sigmoid + selection bias, renormalised weights");
    using Pol = RouteWavePolicy<P>;
    constexpr std::uint32_t experts = P.experts, top_k = P.top_k;
    constexpr std::uint32_t all = Pol::candidates == 32U ? ~0U : (1U << Pol::candidates) - 1U;
    const unsigned r = blockIdx.x, lane = threadIdx.x;
    const float* lg = logits + static_cast<std::size_t>(r) * experts;
    float key[Pol::candidates];
    [[maybe_unused]] bool bad = false;
#pragma unroll
    for (unsigned i = 0; i < Pol::candidates; ++i) {
        const unsigned e = lane + 32 * i;
        const float v = moe::sigmoid_score<Math>(lg[e]);
        if constexpr (Traits::checked_status) bad |= !isfinite(lg[e]) || !isfinite(bias[e]);
        sg[e] = v;
        key[i] = __builtin_bit_cast(float, moe::route_ordered_key(v + bias[e]));
    }
    if constexpr (Traits::checked_status) {
        if (__ballot(bad) != 0U) {
            for (unsigned k = lane; k < top_k; k += Pol::wave) {
                ids[static_cast<std::size_t>(r) * top_k + k] = experts;
                w[static_cast<std::size_t>(r) * top_k + k] = 0.0F;
            }
            if (lane == 0U) status[r] = 1U;
            return;
        }
    }
    route_wave_topk(key, all, experts, top_k, lane, Pol{lane, pick_s});
    __syncthreads();
    if (lane != 0) return;
    std::uint32_t pick[top_k];
    for (unsigned k = 0; k < top_k; ++k) pick[k] = pick_s[k];
    if constexpr (P.order == moe::RouteOrder::ascending_id) {
        for (unsigned a = 1; a < top_k; ++a)  // ascending id
            for (unsigned b = a; b > 0 && pick[b - 1] > pick[b]; --b) {
                const std::uint32_t t = pick[b];
                pick[b] = pick[b - 1];
                pick[b - 1] = t;
            }
    }
    float den = 0.0F;
    for (unsigned k = 0; k < top_k; ++k) {
        if constexpr (Traits::round_nearest) den = __fadd_rn(den, sg[pick[k]]);
        else den = den + sg[pick[k]];
    }
    if constexpr (Traits::round_nearest) den = __fadd_rn(den, P.renorm_eps);
    else den = den + P.renorm_eps;
    for (unsigned k = 0; k < top_k; ++k) {
        ids[static_cast<std::size_t>(r) * top_k + k] = pick[k];
        if constexpr (Traits::round_nearest) {
            w[static_cast<std::size_t>(r) * top_k + k] = __fmul_rn(__fdiv_rn(sg[pick[k]], den), P.scale);
        } else {
            w[static_cast<std::size_t>(r) * top_k + k] = (sg[pick[k]] / den) * P.scale;
        }
    }
    if constexpr (Traits::checked_status) status[r] = 0U;
}

// Keep the established four-argument entry and the checked entry's wave32 launch bound.
template <const moe::RoutePolicy& P, class Math>
__global__ void route_wave_kernel(const float* logits, const float* bias, std::uint32_t* ids, float* w) {
    __shared__ float sg[P.experts];
    __shared__ std::uint32_t pick_s[P.top_k];
    route_wave_body<P, Math, moe::RouteWaveTraits<>>(logits, bias, ids, w, nullptr, sg, pick_s);
}

template <const moe::RoutePolicy& P, class Math, class Traits>
__global__ __launch_bounds__(32) void route_wave_status_entry(const float* logits, const float* bias,
                                                             std::uint32_t* ids, float* w, unsigned* status) {
    static_assert(Traits::checked_status);
    __shared__ float scores[P.experts];
    __shared__ std::uint32_t picked[P.top_k];
    route_wave_body<P, Math, Traits>(logits, bias, ids, w, status, scores, picked);
}

__device__ __forceinline__ float combine_bf16(float x) {
    const std::uint32_t bits = __builtin_bit_cast(std::uint32_t, x);
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x7fffffU)) return x;
    return __builtin_bit_cast(float, (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U);
}

// w: [rows][TopK] (bf16_steps: may be null = unit weights); eo: [rows][TopK][Width]; shared (base): [rows][Width]
// (fma_f32: required; bf16_steps: may be null); y: [rows][Width]. Grid: (ceil(Width / 256), rows) x 256 threads.
// RoundEo: each eo operand is read as its bfloat16 rounding (combine_bf16 == ds41::round_to_bf16 bits), i.e. the
// value a separate glu::epilogue<bf16> launch over eo would have stored: that launch folds into the combine.
template <moe::CombineForm Form, unsigned TopK, unsigned Width, bool RoundEo>
__device__ __forceinline__ void combine_body(const float* w, const float* eo, const float* shared, float* y) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= Width) return;
    if constexpr (Form == moe::CombineForm::fma_f32) {
        static_assert(!RoundEo, "moe combine: RoundEo is bound for bf16_steps only");
        float acc = 0.0F;
        for (unsigned k = 0; k < TopK; ++k)
            acc = __builtin_fmaf(w[static_cast<std::size_t>(r) * TopK + k], eo[(static_cast<std::size_t>(r) * TopK + k) * Width + d], acc);
        y[static_cast<std::size_t>(r) * Width + d] = acc + shared[static_cast<std::size_t>(r) * Width + d];
    } else {
        float acc = 0.0F;
        for (unsigned k = 0; k < TopK; ++k) {
            const float raw = eo[(static_cast<std::size_t>(r) * TopK + k) * Width + d];
            const float e = RoundEo ? combine_bf16(raw) : raw;
            const float term = w == nullptr ? e : combine_bf16(e * w[static_cast<std::size_t>(r) * TopK + k]);
            acc = combine_bf16(acc + term);
        }
        const std::size_t at = static_cast<std::size_t>(r) * Width + d;
        y[at] = shared == nullptr ? acc : combine_bf16(acc + shared[at]);
    }
}
template <moe::CombineForm Form, unsigned TopK, unsigned Width>
__global__ void combine_kernel(const float* w, const float* eo, const float* shared, float* y) {
    combine_body<Form, TopK, Width, false>(w, eo, shared, y);
}
template <moe::CombineForm Form, unsigned TopK, unsigned Width>
__global__ void combine_rounded_eo_kernel(const float* w, const float* eo, const float* shared, float* y) {
    combine_body<Form, TopK, Width, true>(w, eo, shared, y);
}

// Host entries (instantiate in the binding TU; the policy / form / shape are template arguments).
template <const moe::RoutePolicy& P, class Math>
void enqueue_route_wave(const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned rows, hipStream_t stream) {
    if (rows == 0U) return;
    hipLaunchKernelGGL((route_wave_kernel<P, Math>), dim3(rows), dim3(32), 0, stream, logits, bias, ids, w);
}
template <const moe::RoutePolicy& P, class Math, class Traits>
hipError_t enqueue_route_wave(const float* logits, const float* bias, std::uint32_t* ids, float* w,
                              unsigned* status, unsigned rows, hipStream_t stream) {
    if (rows == 0U || !logits || !bias || !ids || !w || !status) return hipErrorInvalidValue;
    hipLaunchKernelGGL((route_wave_status_entry<P, Math, Traits>), dim3(rows), dim3(32U), 0, stream, logits, bias,
                       ids, w, status);
    return hipGetLastError();
}
template <moe::CombineForm Form, unsigned TopK, unsigned Width>
void enqueue_combine(const float* w, const float* eo, const float* shared, float* y, unsigned rows, hipStream_t stream) {
    if (Form == moe::CombineForm::fma_f32 && (w == nullptr || shared == nullptr)) throw std::invalid_argument("moe combine: fma_f32 needs w and shared");
    if (rows == 0U) return;
    hipLaunchKernelGGL((combine_kernel<Form, TopK, Width>), dim3((Width + 255U) / 256U, rows), dim3(256), 0, stream, w, eo, shared, y);
}
// The same combine with every eo operand rounded to bfloat16 on read (combine_rounded_eo_kernel): bitwise the
// pair glu::launch_epilogue<bf16>(eo) + enqueue_combine, one launch fewer.
template <moe::CombineForm Form, unsigned TopK, unsigned Width>
void enqueue_combine_rounded_eo(const float* w, const float* eo, const float* shared, float* y, unsigned rows, hipStream_t stream) {
    static_assert(Form == moe::CombineForm::bf16_steps, "moe combine: RoundEo is bound for bf16_steps only");
    if (rows == 0U) return;
    hipLaunchKernelGGL((combine_rounded_eo_kernel<Form, TopK, Width>), dim3((Width + 255U) / 256U, rows), dim3(256), 0, stream, w, eo,
                       shared, y);
}

}  // namespace strix::moe_core
