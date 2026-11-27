#pragma once

#include "strix/device/launch_geometry.hpp"
// MoE router (one wave per row) and expert-output combine.
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
#include "strix/moe/route_policy.hpp"

#if defined(__HIPCC__)
#include "moe_core/route_wave_grid.hpp"
#include "strix/hip_wave32_dpp.hpp"
#include "strix/moe/combine.hpp"
#include <hip/hip_runtime.h>
#else
#include "strix/exec/descriptor.hpp"
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strix::exec { struct StepBlock; }

namespace strix::moe_core {

#if defined(__HIPCC__)
inline constexpr unsigned kCheckedRouteRowsPerBlock = 1U;
#else
inline constexpr unsigned kCheckedRouteRowsPerBlock = 4U;
#endif

constexpr device::LaunchGeometry checked_route_launch(unsigned rows) {
    constexpr auto waves = kCheckedRouteRowsPerBlock;
    return {{(rows - 1U) / waves + 1U, 1, 1}, {waves * 32U, 1, 1}, 0};
}

template <const moe::RoutePolicy& P>
struct RouteWavePolicy {
    static constexpr std::uint32_t candidates = P.experts / 32, wave = 32;
    static_assert(P.experts % 32 == 0 && candidates <= 32, "route_wave_kernel: experts must fill the wave candidate mask");
    std::uint32_t lane;
    std::uint32_t* pick;
    template <class T>
    __device__ static T xor32(T value, int mask) {
#if defined(__HIPCC__)
        return strix::hip_wave32_dpp::xor32(value, static_cast<unsigned>(mask));
#else
        return __shfl_xor_sync(0xffffffffU, value, mask, 32);
#endif
    }
    __device__ static bool better(float a, std::uint32_t ai, float b, std::uint32_t bi) {
#if !defined(__HIPCC__)
        if constexpr (P.key == moe::RouteKey::float_serial) return a > b || (a == b && ai < bi);
#endif
        const std::uint32_t ka = __builtin_bit_cast(std::uint32_t, a), kb = __builtin_bit_cast(std::uint32_t, b);
        return ka > kb || (ka == kb && ai < bi);
    }
    __device__ void record(std::uint32_t slot, bool, float, std::uint32_t index) const {
        if (lane == 0) pick[slot] = index;  // every slot has a candidate: top_k <= experts
    }
};

// Which row each wave of a block routes. BlockRows: the checked route launch (kCheckedRouteRowsPerBlock waves per block,
// wave w of block b routes row b * kCheckedRouteRowsPerBlock + w). A kernel that routes inside a larger launch
// (moe_core/r1_route.hpp) supplies its own site: its wave count (the score scratch rows) and the row of a wave (a value
// >= rows leaves the wave idle), so every route of the R1 family runs this one body.
struct BlockRows {
    static constexpr unsigned kWaves = kCheckedRouteRowsPerBlock;
    __device__ static unsigned row(unsigned wave) { return blockIdx.x * kCheckedRouteRowsPerBlock + wave; }
};

#if defined(__HIPCC__)
struct GridRows {
    static constexpr unsigned kWaves = RouteWaveGrid::rows_per_block;
    __device__ static unsigned row(unsigned wave) { return RouteWaveGrid::row(blockIdx.x, wave); }
};
#endif

// logits / bias: [rows][experts] / [experts]; ids / w: [rows][top_k].
template <const moe::RoutePolicy& P, class Math, class Traits, class Site = BlockRows, bool Masked = false>
__device__ __forceinline__ void route_wave_body(const float* logits, const float* bias, std::uint32_t* ids,
                                               float* w, unsigned* status, unsigned rows, Site site = {},
                                               [[maybe_unused]] const exec::StepBlock* descriptor = nullptr) {
    // moe::validate(P) is checked by the host launchers (constexpr host function).
    static_assert(P.score == moe::RouteScore::sigmoid && P.selection_bias &&
                      (P.key == moe::RouteKey::ordered_bits || Traits::checked_status) &&
                      P.renormalize && (P.top_k > 1U || P.renorm_single),
                  "route_wave_kernel: sigmoid + selection bias, renormalised weights");
    using Pol = RouteWavePolicy<P>;
    constexpr std::uint32_t experts = P.experts, top_k = P.top_k;
    constexpr std::uint32_t all = Pol::candidates == 32U ? ~0U : (1U << Pol::candidates) - 1U;
    __shared__ float score_rows[Site::kWaves][experts];
    __shared__ std::uint32_t picked_rows[Site::kWaves][top_k];
    const unsigned wave = threadIdx.x / Pol::wave;
    const unsigned lane = threadIdx.x % Pol::wave;
    const unsigned r = site.row(wave);
    if (r >= rows) return;
#if defined(__CUDACC__) && !defined(__HIPCC__)
    if constexpr (Masked) {
        if (!exec::step_row_valid(*descriptor, r)) return;
    }
#endif
    auto* sg = score_rows[wave];
    auto* pick_s = picked_rows[wave];
    const float* lg = logits + static_cast<std::size_t>(r) * experts;
    float key[Pol::candidates];
    [[maybe_unused]] bool bad = false;
#pragma unroll
    for (unsigned i = 0; i < Pol::candidates; ++i) {
        const unsigned e = lane + 32 * i;
        const float v = moe::sigmoid_score<Math>(lg[e]);
        const float choice = v + bias[e];
        sg[e] = v;
#if defined(__HIPCC__)
        if constexpr (Traits::checked_status) bad |= !isfinite(lg[e]) || !isfinite(bias[e]);
        key[i] = __builtin_bit_cast(float, moe::route_ordered_key(choice));
#else
        if constexpr (Traits::checked_status) bad |= !isfinite(lg[e]) || !isfinite(bias[e]) || !isfinite(choice);
        if constexpr (P.key == moe::RouteKey::float_serial) key[i] = choice;
        else key[i] = __builtin_bit_cast(float, moe::route_ordered_key(choice));
#endif
    }
    if constexpr (Traits::checked_status) {
#if defined(__HIPCC__)
        if (__ballot(bad) != 0U) {
#else
        if (__any_sync(0xffffffffU, bad)) {
#endif
            for (unsigned k = lane; k < top_k; k += Pol::wave) {
                ids[static_cast<std::size_t>(r) * top_k + k] = experts;
                w[static_cast<std::size_t>(r) * top_k + k] = 0.0F;
            }
            if (lane == 0U) status[r] = 1U;
            return;
        }
    }
    route_wave_topk(key, all, experts, top_k, lane, Pol{lane, pick_s});
#if defined(__HIPCC__)
    if constexpr (Site::kWaves == 1U) {
        __syncthreads();
    } else {
        // Invalid and inactive waves can exit independently. Each live wave publishes only its own LDS row.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
#else
    __syncwarp();
#endif
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

template <const moe::RoutePolicy& P, class Math, class Traits>
__global__ __launch_bounds__(kCheckedRouteRowsPerBlock * 32U) void route_wave_status_entry(
    const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned* status, unsigned rows) {
    static_assert(Traits::checked_status);
    route_wave_body<P, Math, Traits>(logits, bias, ids, w, status, rows);
}

#if defined(__HIPCC__)
template <const moe::RoutePolicy& P, class Math, class Traits>
__global__ __launch_bounds__(RouteWaveGrid::threads_per_block) void route_wave_grid_entry(
    const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned* status, unsigned rows) {
    static_assert(Traits::checked_status);
    route_wave_body<P, Math, Traits, GridRows>(logits, bias, ids, w, status, rows);
}

// The four-argument entry (GLM and HY4) keeps its own body; route_wave_body above is the checked trait form of the
// same algorithm (same score, key, top-k, ordering and renormalisation steps). Its tail runs on the whole wave: lane
// k < top_k takes pick k to its output slot (its rank among the distinct picks for ascending ids, else k), then every
// lane sums the scores in output order from +0 as the serial lane-0 tail did, so ids and w keep their bits.
template <const moe::RoutePolicy& P, class Math>
__global__ void route_wave_kernel(const float* logits, const float* bias, std::uint32_t* ids, float* w) {
    static_assert(moe::validate(P) == nullptr, "route_wave_kernel: invalid policy");
    static_assert(P.score == moe::RouteScore::sigmoid && P.selection_bias && P.key == moe::RouteKey::ordered_bits &&
                      P.renormalize && (P.top_k > 1U || P.renorm_single),
                  "route_wave_kernel: sigmoid + selection bias, ordered keys, renormalised weights");
    using Pol = RouteWavePolicy<P>;
    constexpr std::uint32_t experts = P.experts, top_k = P.top_k;
    constexpr std::uint32_t all = Pol::candidates == 32U ? ~0U : (1U << Pol::candidates) - 1U;
    __shared__ float sg[experts];
    static_assert(top_k <= Pol::wave, "route_wave_kernel: one pick per lane in the tail");
    __shared__ std::uint32_t pick_s[top_k], out_s[top_k];
    const unsigned r = blockIdx.x, lane = threadIdx.x;
    const float* lg = logits + static_cast<std::size_t>(r) * experts;
    float key[Pol::candidates];
#pragma unroll
    for (unsigned i = 0; i < Pol::candidates; ++i) {
        const unsigned e = lane + 32 * i;
        const float v = moe::sigmoid_score<Math>(lg[e]);
        sg[e] = v;
        key[i] = __builtin_bit_cast(float, moe::route_ordered_key(v + bias[e]));
    }
    route_wave_topk(key, all, experts, top_k, lane, Pol{lane, pick_s});
    __syncthreads();
    const bool owner = lane < top_k;
    const std::uint32_t id = pick_s[owner ? lane : 0U];
    std::uint32_t slot = lane;
    if constexpr (P.order == moe::RouteOrder::ascending_id) {  // the picks are distinct ids
        slot = 0U;
#pragma unroll
        for (unsigned k = 0; k < top_k; ++k) slot += pick_s[k] < id ? 1U : 0U;
    }
    if (owner) out_s[slot] = id;
    __syncthreads();
    float den = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < top_k; ++k) den = den + sg[out_s[k]];
    den = den + P.renorm_eps;
    if (!owner) return;
    const std::uint32_t out = out_s[lane];
    ids[static_cast<std::size_t>(r) * top_k + lane] = out;
    w[static_cast<std::size_t>(r) * top_k + lane] = (sg[out] / den) * P.scale;
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
    } else if constexpr (Form == moe::CombineForm::f32_steps_bf16) {
        static_assert(!RoundEo, "moe combine: RoundEo is bound for bf16_steps only");
        float acc = 0.0F;
        for (unsigned k = 0; k < TopK; ++k) {
            const float wk = w == nullptr ? 1.0F : w[static_cast<std::size_t>(r) * TopK + k];
            acc = __fadd_rn(acc, __fmul_rn(wk, eo[(static_cast<std::size_t>(r) * TopK + k) * Width + d]));
        }
        const std::size_t at = static_cast<std::size_t>(r) * Width + d;
        y[at] = combine_bf16(shared == nullptr ? acc : __fadd_rn(acc, shared[at]));
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
    static_assert(moe::validate(P) == nullptr, "route_wave_kernel: invalid policy");
    if (rows == 0U || !logits || !bias || !ids || !w || !status) return hipErrorInvalidValue;
    hipLaunchKernelGGL((route_wave_status_entry<P, Math, Traits>), dim3(rows), dim3(32U), 0, stream, logits, bias,
                       ids, w, status, rows);
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

// Policy-type entry: Policy::value names the route policy. The HIP form forwards to the reference entry above
// (same kernel instantiation). CUDA kernels take the policy as a type because nvcc's host stub cannot name an
// inline variable used as a reference template argument of a __global__ function under MSVC.
template <class Policy, class Math, class Traits>
hipError_t enqueue_route_wave(const float* logits, const float* bias, std::uint32_t* ids, float* w,
                              unsigned* status, unsigned rows, hipStream_t stream) {
    return enqueue_route_wave<Policy::value, Math, Traits>(logits, bias, ids, w, status, rows, stream);
}

// Explicit wave32 policy. Short row counts use the checked one-wave launch.
template <class Policy, class Math, class Traits>
hipError_t enqueue_route_wave_grid(const float* logits, const float* bias, std::uint32_t* ids, float* w,
                                   unsigned* status, unsigned rows, hipStream_t stream) {
    static_assert(moe::validate(Policy::value) == nullptr, "route_wave_kernel: invalid policy");
    if (!RouteWaveGrid::admits(rows)) {
        return enqueue_route_wave<Policy, Math, Traits>(logits, bias, ids, w, status, rows, stream);
    }
    if (!logits || !bias || !ids || !w || !status) return hipErrorInvalidValue;
    hipLaunchKernelGGL((route_wave_grid_entry<Policy::value, Math, Traits>), dim3(RouteWaveGrid::blocks(rows)),
                       dim3(RouteWaveGrid::threads_per_block), 0, stream, logits, bias, ids, w, status, rows);
    return hipGetLastError();
}

#else
template <class Policy, class Math, class Traits>
__global__ __launch_bounds__(kCheckedRouteRowsPerBlock * 32U) void route_wave_policy_entry(
    const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned* status, unsigned rows) {
    static_assert(Traits::checked_status);
    route_wave_body<Policy::value, Math, Traits>(logits, bias, ids, w, status, rows);
}

template <class Policy, class Math, class Traits>
cudaError_t enqueue_route_wave(const float* logits, const float* bias, std::uint32_t* ids, float* w,
                               unsigned* status, unsigned rows, cudaStream_t stream) {
    static_assert(moe::validate(Policy::value) == nullptr, "route_wave_kernel: invalid policy");
    if (rows == 0U) return cudaSuccess;
    if (!logits || !bias || !ids || !w || !status || rows > 0x7fffffffU) return cudaErrorInvalidValue;
    const auto shape = checked_route_launch(rows);
    route_wave_policy_entry<Policy, Math, Traits><<<shape.template grid<dim3>(), shape.template block<dim3>(), shape.shared_bytes, stream>>>(logits, bias, ids, w, status,
                                                                                      rows);
    return cudaGetLastError();
}

template <unsigned Rows, class Policy, class Math, class Traits>
__global__ __launch_bounds__(kCheckedRouteRowsPerBlock * 32U) void route_wave_step_entry(
    const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned* status,
    const exec::StepBlock* step) {
    static_assert(Traits::checked_status);
    route_wave_body<Policy::value, Math, Traits, BlockRows, true>(logits, bias, ids, w, status, Rows, BlockRows{}, step);
}

template <unsigned Rows, class Policy, class Math, class Traits>
cudaError_t enqueue_route_wave_step(const float* logits, const float* bias, std::uint32_t* ids, float* w,
                                   unsigned* status, const exec::StepBlock* step, cudaStream_t stream) {
    static_assert(moe::validate(Policy::value) == nullptr, "route_wave_kernel: invalid policy");
    if (!step || !logits || !bias || !ids || !w || !status) return cudaErrorInvalidValue;
    constexpr auto shape = checked_route_launch(Rows);
    route_wave_step_entry<Rows, Policy, Math, Traits><<<shape.template grid<dim3>(), shape.template block<dim3>(), shape.shared_bytes, stream>>>(
        logits, bias, ids, w, status, step);
    return cudaGetLastError();
}
#endif

}  // namespace strix::moe_core
