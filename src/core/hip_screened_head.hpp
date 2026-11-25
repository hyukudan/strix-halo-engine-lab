#pragma once
// EXACT screened greedy LM head, HIP kernels (gfx1151, wave32). Engine-neutral: rows / width / softcap / plane are
// arguments; nothing here names a model. The bound math and the plane builder are in core/screened_head.hpp, which
// also states the invariant. Five fixed launches, no host read between them (graph-capture safe):
//   1 prep     one block: nx = inflated ||x||_2, the FALLBACK flag (non-finite x or overflow guard), state reset
//   2 pass1    wave per row over the narrow plane: y_hat_i, e_i, U_i (stored), Lstar = max L_i (atomic)
//   3 compact  candidates i with G(U_i) >= G(Lstar) -> ids (wave-aggregated atomic); FALLBACK or too many -> ALL
//   4 pass2    the incumbent tight-row body (dense_linear::mapped_rows_kernel) on the candidate rows (or all rows)
//   5 pick     one block: G(y) and the incumbent first-max / finite-status rule over the candidates
// Cost of the fallback (non-finite x, nx * max A >= 1e37, more than `capacity` candidates): pass 1 is skipped when the
// flag is known at launch 2 (flag) or wasted (overflow of the candidate list), then pass 2 runs the incumbent GEMV on all
// rows with the fixed grid looping (4096 waves with the default grid, each ~49 rows for 200k rows: the incumbent's bytes,
// lower occupancy than its one-wave-per-block launch) and the pick folds all rows in one block (~200k G + compare per
// 256 lanes). The result is the incumbent's by construction.
// Lanes form (enqueue_screened_greedy_lanes, the shared screened_head_plan::Launch the CUDA bodies also take): several
// activation rows against the same head. Only pass 1 changes (one plane read per group of lanes); every lane then runs
// the one-row prep / compact / pass 2 / pick launches on its own scratch part, so each lane's token and status are the
// one-row form's. Engine-neutral like the rest of this file.
#if defined(__HIPCC__)
#include "core/dense_linear.hpp"
#include "core/row_transforms.hpp"
#include "core/screened_head.hpp"
#include "core/screened_head_plan.hpp"
#include "strix/hip_argmax_core.hpp"
#include "strix/env_flags.hpp"
#include "strix/hip_generation_output.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::core::hip_screened_head {

using screened_head_plan::Grids;
using screened_head_plan::HeadSpec;
using screened_head_plan::kAllRows;
using screened_head_plan::kCompactThreads;
using screened_head_plan::kMaxWidth;
using screened_head_plan::kPass1Waves;
using screened_head_plan::PlaneView;
using screened_head_plan::ScratchLayout;
using screened_head_plan::scratch_layout;
using screened_head_plan::State;

// Order-preserving key of a finite or infinite float (larger value -> larger key); 0 is below every key.
__device__ __forceinline__ unsigned order_key(float v) {
    const unsigned b = __float_as_uint(v);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}
__device__ __forceinline__ float key_value(unsigned k) {
    return __uint_as_float((k & 0x80000000U) != 0U ? (k & 0x7fffffffU) : ~k);
}

// The incumbent post-transform: the same function row_transforms::Bf16Softcap applies.
__device__ __forceinline__ float head_transform(float y, float cap) { return row_transforms::bf16_softcap(y, cap); }

// Outward bumps of an RN result by one ulp (the exact sum lies within half an ulp of it).
__device__ __forceinline__ float bump_up(float v) {
    unsigned b = __float_as_uint(v);
    if ((b & 0x7fffffffU) == 0U) return __uint_as_float(1U);
    if ((b & 0x7f800000U) == 0x7f800000U) return v;
    return __uint_as_float((b & 0x80000000U) != 0U ? b - 1U : b + 1U);
}
__device__ __forceinline__ float bump_down(float v) {
    unsigned b = __float_as_uint(v);
    if ((b & 0x7fffffffU) == 0U) return __uint_as_float(0x80000001U);
    if ((b & 0x7f800000U) == 0x7f800000U) return v;
    return __uint_as_float((b & 0x80000000U) != 0U ? b + 1U : b - 1U);
}

__global__ void __launch_bounds__(256) prep_kernel(const float* x, unsigned width, screened_head::Bound bound,
                                                   float max_amplitude, State* state) {
    __shared__ float partial[8];
    float s = 0.0F;
    for (unsigned k = threadIdx.x; k < width; k += 256U) s = __fmaf_rn(x[k], x[k], s);
    s = dense_linear::wave_sum32(s);
    if ((threadIdx.x & 31U) == 0U) partial[threadIdx.x / 32U] = s;
    __syncthreads();
    if (threadIdx.x == 0U) {
        float t = 0.0F;
        for (unsigned w = 0; w < 8U; ++w) t = __fadd_rn(t, partial[w]);
        const float nx = __fsqrt_rn(__fmul_rn(__fadd_rn(t, bound.sum_abs), bound.sum_inflate));
        state->nx = nx;
        state->flag = __fmul_rn(nx, max_amplitude) < screened_head::kOverflowGuard ? 0U : 1U;  // NaN / inf -> 1
        state->count = 0U;
        state->lkey = 0U;
    }
}

template <unsigned Bits>
__global__ void __launch_bounds__(kPass1Waves * 32U) pass1_kernel(PlaneView plane, const float* x,
                                                                  screened_head::Bound bound, State* state,
                                                                  float* upper) {
    extern __shared__ float xs[];
    if (state->flag != 0U) return;  // uniform: the full head runs
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    for (unsigned i = threadIdx.x; i < plane.width / 4U; i += blockDim.x)
        reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x)[i];
    __syncthreads();
    const float nx = state->nx;
    const unsigned units = plane.width / screened_head::kUnit;
    float lmax = -__builtin_inff();
    bool bad = false;
    for (unsigned row = blockIdx.x * kPass1Waves + wave; row < plane.rows; row += gridDim.x * kPass1Waves) {
        const std::uint8_t* base = plane.data + std::size_t{row} * plane.row_bytes;
        const std::uint16_t* scales = reinterpret_cast<const std::uint16_t*>(base + plane.scales_offset);
        float acc = 0.0F;
        for (unsigned u = lane; u < units; u += 32U) {
            float inner = 0.0F;
            const float4 x0 = *reinterpret_cast<const float4*>(xs + 8U * u);
            const float4 x1 = *reinterpret_cast<const float4*>(xs + 8U * u + 4U);
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            if constexpr (Bits == 4U) {
                const unsigned word = *reinterpret_cast<const unsigned*>(base + 4U * u);
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j)
                    inner = __fmaf_rn(static_cast<float>(static_cast<int>((word >> (4U * j)) & 15U) - 8), xv[j], inner);
            } else {
                const uint2 word = *reinterpret_cast<const uint2*>(base + 8U * u);
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) {
                    const unsigned w32 = j < 4U ? word.x : word.y;
                    const int q = static_cast<int>(static_cast<std::int8_t>((w32 >> (8U * (j % 4U))) & 255U));
                    inner = __fmaf_rn(static_cast<float>(q), xv[j], inner);
                }
            }
            const float s = __uint_as_float(static_cast<unsigned>(scales[u / 4U]) << 16U);
            acc = __fmaf_rn(s, inner, acc);
        }
        const float yhat = dense_linear::wave_sum32(acc);
        const float e = __fadd_rn(
            __fmul_rn(__fmul_rn(nx, __fadd_rn(plane.residual[row], __fmul_rn(bound.gamma, plane.amplitude[row]))),
                      bound.margin),
            bound.abs_slack);
        const float hi = bump_up(__fadd_rn(yhat, e)), lo = bump_down(__fsub_rn(yhat, e));
        if (lane == 0U) upper[row] = hi;
        if (!(isfinite(hi) && isfinite(lo))) bad = true; else lmax = fmaxf(lmax, lo);
    }
    if (lane == 0U) {
        if (bad) atomicOr(&state->flag, 1U);
        if (lmax > -__builtin_inff()) atomicMax(&state->lkey, order_key(lmax));
    }
}

__global__ void __launch_bounds__(kCompactThreads) compact_kernel(const float* upper, unsigned rows, float cap,
                                                                  unsigned capacity, State* state,
                                                                  std::uint32_t* ids) {
    const bool flagged = state->flag != 0U;
    if (flagged) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) state->count = kAllRows;
        return;
    }
    const float threshold = head_transform(key_value(state->lkey), cap);
    const unsigned lane = threadIdx.x % static_cast<unsigned>(warpSize);
    for (unsigned base = blockIdx.x * kCompactThreads; base < rows; base += gridDim.x * kCompactThreads) {
        const unsigned i = base + threadIdx.x;
        const bool hit = i < rows && head_transform(upper[i], cap) >= threshold;
        const unsigned long long mask = __ballot(hit);
        if (mask == 0ULL) continue;
        const unsigned leader = static_cast<unsigned>(__ffsll(static_cast<long long>(mask))) - 1U;
        unsigned first = 0U;
        if (lane == leader) first = atomicAdd(&state->count, static_cast<unsigned>(__popcll(mask)));
        first = __shfl(first, static_cast<int>(leader));
        const unsigned slot = first + static_cast<unsigned>(__popcll(mask & ((1ULL << lane) - 1ULL)));
        if (hit && slot < capacity) ids[slot] = i;
    }
}

__global__ void __launch_bounds__(256) pick_kernel(const float* y, const std::uint32_t* ids, const State* state,
                                                   unsigned rows, unsigned capacity, float cap,
                                                   HipGreedyRowsReply* reply) {
    __shared__ argmax::Partial values[256];
    const unsigned count = state->count;
    const bool all = count > capacity;
    const unsigned n = all ? rows : count;
    argmax::Partial best = argmax::empty();
    for (unsigned i = threadIdx.x; i < n; i += 256U)
        best = argmax::choose(best, argmax::load(head_transform(y[i], cap), all ? i : ids[i]));
    best = argmax::block_argmax<256>(values, threadIdx.x, best);
    if (threadIdx.x == 0U) *reply = HipGreedyRowsReply{best.token, argmax::status(best)};
}

// The five launches. x: FP32 [width], 16-byte aligned (the incumbent's requirement too). scratch: scratch_layout bytes,
// device. device_reply: device-visible HipGreedyRowsReply (e.g. the mapped pinned reply of ResidentGreedy).
inline hipError_t enqueue_screened_greedy(const PlaneView& plane, const screened_head::Bound& bound,
                                          const HeadSpec& head, const float* x, std::uint8_t* scratch,
                                          unsigned capacity, HipGreedyRowsReply* device_reply, hipStream_t stream,
                                          const Grids& grids = {}) {
    if (!device_reply || !screened_head_plan::launch_args_valid(plane, bound, head, x, scratch, capacity, grids))
        return hipErrorInvalidValue;
    const auto layout = scratch_layout(head.rows, capacity);
    auto* state = reinterpret_cast<State*>(scratch);
    auto* ids = reinterpret_cast<std::uint32_t*>(scratch + layout.ids);
    auto* upper = reinterpret_cast<float*>(scratch + layout.upper);
    auto* y = reinterpret_cast<float*>(scratch + layout.y);
    prep_kernel<<<1, 256, 0, stream>>>(x, head.width, bound, plane.max_amplitude, state);
    const std::size_t lds = std::size_t{head.width} * sizeof(float);
    if (plane.bits == 4U)
        pass1_kernel<4U><<<grids.pass1, kPass1Waves * 32U, lds, stream>>>(plane, x, bound, state, upper);
    else
        pass1_kernel<8U><<<grids.pass1, kPass1Waves * 32U, lds, stream>>>(plane, x, bound, state, upper);
    compact_kernel<<<grids.compact, kCompactThreads, 0, stream>>>(upper, head.rows, head.softcap, capacity, state, ids);
    const dense_linear::View view{reinterpret_cast<const std::uint8_t*>(head.weights), head.width, head.rows,
                                  std::size_t{head.width} * sizeof(std::uint16_t)};
    dense_linear::mapped_rows_kernel<weight_decode::Format::bf16><<<grids.pass2, 128, 0, stream>>>(
        view, x, y, dense_linear::MappedRows{ids, &state->count, capacity});
    pick_kernel<<<1, 256, 0, stream>>>(y, ids, state, head.rows, capacity, head.softcap, device_reply);
    return hipGetLastError();
}

// Waves per block of the lanes pass 1. A group's activations take Lanes * width * 4 bytes of shared memory per block
// (36,864 B for four lanes at width 2,304), so blocks of kPass1Waves would leave one 8-wave block per compute unit and
// the plane stream latency-bound; one 32-wave block per such footprint keeps the waves of the one-row form.
inline constexpr unsigned kLanesWaves = 32U;

// ---- Lanes: several activation rows against one head (verify rows, cohort lanes), screened_head_plan::Launch. ----
// Pass 1 of a group of Lanes lanes: blockIdx.x serves lanes [x * Lanes, min(lanes, (x + 1) * Lanes)); blockIdx.y
// walks the plane rows. Groups are the fastest grid dimension so blocks reading the same plane rows can share
// those reads through the cache. Their activations live in xs ([Lanes][width]). Each plane row is loaded and
// decoded once for the group; every lane keeps its own
// FMA chain in pass1_kernel's order, so its upper[], its bound flag and its L* are the ones the one-row launch computes on
// that lane. A lane whose prep raised the fallback flag is skipped (its compact runs every row, as in the one-row form).
template <unsigned Bits, unsigned Lanes>
__global__ void __launch_bounds__(kLanesWaves * 32U) pass1_lanes_kernel(PlaneView plane, const float* x, unsigned lanes,
                                                                        unsigned x_stride, screened_head::Bound bound,
                                                                        State* state, float* upper, unsigned upper_stride) {
    extern __shared__ float xs[];
    const unsigned first = blockIdx.x * Lanes;
    const unsigned live = lanes - first < Lanes ? lanes - first : Lanes;
    bool on[Lanes];
    float nx[Lanes];
    bool any = false;
#pragma unroll
    for (unsigned l = 0; l < Lanes; ++l) {
        on[l] = l < live && state[first + l].flag == 0U;
        nx[l] = on[l] ? state[first + l].nx : 0.0F;
        any = any || on[l];
    }
    if (!any) return;  // uniform: every lane of this block row runs the full head
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const unsigned quads = plane.width / 4U;
    for (unsigned i = threadIdx.x; i < Lanes * quads; i += blockDim.x) {
        const unsigned l = i / quads, q = i % quads;
        reinterpret_cast<float4*>(xs)[i] =
            l < live ? reinterpret_cast<const float4*>(x + std::size_t{first + l} * x_stride)[q]
                     : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
    __syncthreads();
    const unsigned units = plane.width / screened_head::kUnit;
    float lmax[Lanes];
    bool bad[Lanes];
#pragma unroll
    for (unsigned l = 0; l < Lanes; ++l) {
        lmax[l] = -__builtin_inff();
        bad[l] = false;
    }
    for (unsigned row = blockIdx.y * kLanesWaves + wave; row < plane.rows; row += gridDim.y * kLanesWaves) {
        const std::uint8_t* base = plane.data + std::size_t{row} * plane.row_bytes;
        const std::uint16_t* scales = reinterpret_cast<const std::uint16_t*>(base + plane.scales_offset);
        float acc[Lanes];
#pragma unroll
        for (unsigned l = 0; l < Lanes; ++l) acc[l] = 0.0F;
        for (unsigned u = lane; u < units; u += 32U) {
            float q[8];
            if constexpr (Bits == 4U) {
                const unsigned word = *reinterpret_cast<const unsigned*>(base + 4U * u);
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) q[j] = static_cast<float>(static_cast<int>((word >> (4U * j)) & 15U) - 8);
            } else {
                const uint2 word = *reinterpret_cast<const uint2*>(base + 8U * u);
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) {
                    const unsigned w32 = j < 4U ? word.x : word.y;
                    q[j] = static_cast<float>(static_cast<int>(static_cast<std::int8_t>((w32 >> (8U * (j % 4U))) & 255U)));
                }
            }
            const float s = __uint_as_float(static_cast<unsigned>(scales[u / 4U]) << 16U);
#pragma unroll
            for (unsigned l = 0; l < Lanes; ++l) {
                // float4 indexing keeps the 16-byte alignment visible (128-bit shared loads, as in pass1_kernel).
                const float4* xl = reinterpret_cast<const float4*>(xs) + l * quads + 2U * u;
                const float4 x0 = xl[0];
                const float4 x1 = xl[1];
                const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                float inner = 0.0F;
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) inner = __fmaf_rn(q[j], xv[j], inner);
                acc[l] = __fmaf_rn(s, inner, acc[l]);
            }
        }
        const float slack = __fmul_rn(bound.gamma, plane.amplitude[row]);
        const float residual = plane.residual[row];
#pragma unroll
        for (unsigned l = 0; l < Lanes; ++l) {
            const float yhat = dense_linear::wave_sum32(acc[l]);
            const float e = __fadd_rn(__fmul_rn(__fmul_rn(nx[l], __fadd_rn(residual, slack)), bound.margin),
                                      bound.abs_slack);
            const float hi = bump_up(__fadd_rn(yhat, e)), lo = bump_down(__fsub_rn(yhat, e));
            if (!on[l]) continue;
            if (lane == 0U) upper[std::size_t{first + l} * upper_stride + row] = hi;
            if (!(isfinite(hi) && isfinite(lo))) bad[l] = true; else lmax[l] = fmaxf(lmax[l], lo);
        }
    }
    if (lane == 0U) {
#pragma unroll
        for (unsigned l = 0; l < Lanes; ++l) {
            if (!on[l]) continue;
            if (bad[l]) atomicOr(&state[first + l].flag, 1U);
            if (lmax[l] > -__builtin_inff()) atomicMax(&state[first + l].lkey, order_key(lmax[l]));
        }
    }
}

// Shared memory per block of this device (pass 1 stages pass1_lanes * width floats).
inline std::size_t shared_limit(int device = 0) {
    int bytes = 0;
    if (hipDeviceGetAttribute(&bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess || bytes <= 0)
        return 32U * 1024U;
    return static_cast<std::size_t>(bytes);
}

// Lanes per pass-1 block row for this device (screened_head_plan::pass1_lanes against its shared-memory limit).
inline unsigned pass1_lanes(unsigned width, unsigned lanes, int device = 0) {
    return screened_head_plan::pass1_lanes(width, lanes, shared_limit(device));
}

// The lanes form of the shared plan: lane l screens x + l * x_stride into device_replies[l] (device-visible, `lanes`
// entries) with the same token and status as enqueue_screened_greedy on that row alone. Scratch: plan::lanes_layout.
// Pass 1 reads the plane once per group of a.pass1_lanes lanes (pass1_lanes_kernel; a group of one is pass1_kernel);
// prep, compact, pass 2 and the pick are the one-row launches above on each lane's part of the scratch. One lane is
// exactly enqueue_screened_greedy. The runner-up trait has no HIP body yet (rejected).
inline hipError_t enqueue_screened_greedy_lanes(const screened_head_plan::Launch& a, HipGreedyRowsReply* device_replies,
                                                hipStream_t stream) {
    if (!device_replies || a.runner_up || !screened_head_plan::launch_valid(a)) return hipErrorInvalidValue;
    if (a.lanes == 1U)
        return enqueue_screened_greedy(a.plane, a.bound, a.head, a.x, a.scratch, a.capacity, device_replies, stream,
                                       a.grids);
    const auto layout = screened_head_plan::lanes_layout(a.head.rows, a.capacity, a.lanes);
    auto* state = reinterpret_cast<State*>(a.scratch);
    auto* ids = reinterpret_cast<std::uint32_t*>(a.scratch + layout.ids);
    auto* upper = reinterpret_cast<float*>(a.scratch + layout.upper);
    auto* y = reinterpret_cast<float*>(a.scratch + layout.y);
    const auto lane_x = [&](unsigned l) { return a.x + std::size_t{l} * a.x_stride; };
    for (unsigned l = 0; l < a.lanes; ++l)
        prep_kernel<<<1, 256, 0, stream>>>(lane_x(l), a.head.width, a.bound, a.plane.max_amplitude, state + l);
    const unsigned per = a.pass1_lanes;
    if (per == 1U) {
        const std::size_t lds = std::size_t{a.head.width} * sizeof(float);
        for (unsigned l = 0; l < a.lanes; ++l) {
            float* lane_upper = upper + std::size_t{l} * layout.upper_stride;
            if (a.plane.bits == 4U)
                pass1_kernel<4U><<<a.grids.pass1, kPass1Waves * 32U, lds, stream>>>(a.plane, lane_x(l), a.bound,
                                                                                   state + l, lane_upper);
            else
                pass1_kernel<8U><<<a.grids.pass1, kPass1Waves * 32U, lds, stream>>>(a.plane, lane_x(l), a.bound,
                                                                                   state + l, lane_upper);
        }
    } else {
        // The same waves in flight as the one-row launch: grids.pass1 blocks of kPass1Waves -> blocks of kLanesWaves.
        const unsigned blocks = std::max(1U, a.grids.pass1 * kPass1Waves / kLanesWaves);
        const dim3 grid(screened_head_plan::pass1_groups(a.lanes, per), blocks);
        const std::size_t lds = std::size_t{per} * a.head.width * sizeof(float);
        const auto stride = static_cast<unsigned>(layout.upper_stride);
        const auto run = [&](auto kernel) {
            kernel<<<grid, kLanesWaves * 32U, lds, stream>>>(a.plane, a.x, a.lanes, a.x_stride, a.bound, state, upper,
                                                             stride);
        };
        if (a.plane.bits == 4U) {
            if (per == 2U) run(pass1_lanes_kernel<4U, 2U>);
            else if (per == 4U) run(pass1_lanes_kernel<4U, 4U>);
            else run(pass1_lanes_kernel<4U, 8U>);
        } else {
            if (per == 2U) run(pass1_lanes_kernel<8U, 2U>);
            else if (per == 4U) run(pass1_lanes_kernel<8U, 4U>);
            else run(pass1_lanes_kernel<8U, 8U>);
        }
    }
    const dense_linear::View view{reinterpret_cast<const std::uint8_t*>(a.head.weights), a.head.width, a.head.rows,
                                  std::size_t{a.head.width} * sizeof(std::uint16_t)};
    for (unsigned l = 0; l < a.lanes; ++l) {
        std::uint32_t* lane_ids = ids + std::size_t{l} * layout.ids_stride;
        float* lane_y = y + std::size_t{l} * layout.y_stride;
        compact_kernel<<<a.grids.compact, kCompactThreads, 0, stream>>>(upper + std::size_t{l} * layout.upper_stride,
                                                                       a.head.rows, a.head.softcap, a.capacity,
                                                                       state + l, lane_ids);
        dense_linear::mapped_rows_kernel<weight_decode::Format::bf16><<<a.grids.pass2, 128, 0, stream>>>(
            view, lane_x(l), lane_y, dense_linear::MappedRows{lane_ids, &state[l].count, a.capacity});
        pick_kernel<<<1, 256, 0, stream>>>(lane_y, lane_ids, state + l, a.head.rows, a.capacity, a.head.softcap,
                                           device_replies + l);
    }
    return hipGetLastError();
}

// ---- Rescore trait (screened_head_plan::RescoreLaunch): exact rows for the largest approximate scores. ----
// Launches (blockIdx.y = lane for every launch but pass 2): clear, digit 1 histogram, digit 1 select, digit 2 histogram,
// digit 2 select, compact, pass 2 (the incumbent mapped_rows_kernel per lane, as in the lanes form above), scatter.
// The histograms count integers, so the threshold and the selected set never depend on the block / atomic order; the
// compacted order of the ids may, and nothing reads it but the scatter, which writes each id's own score.
using screened_head_plan::kRescoreBins;
using screened_head_plan::RescoreState;
inline constexpr unsigned kRescoreThreads = 256U;
inline constexpr unsigned kRescoreBinsPerThread = kRescoreBins / kRescoreThreads;
static_assert(kRescoreBinsPerThread == 16U, "16 bins per select thread");

template <unsigned Digit>
__global__ void __launch_bounds__(kRescoreThreads) rescore_hist_kernel(const float* scores, unsigned pitch,
                                                                       unsigned rows, const RescoreState* states,
                                                                       unsigned* hist, std::size_t hist_stride) {
    __shared__ unsigned counts[kRescoreBins];
    const unsigned lane = blockIdx.y;
    const float* s = scores + std::size_t{lane} * pitch;
    unsigned* h = hist + std::size_t{lane} * hist_stride + (Digit == 1U ? 0U : kRescoreBins);
    const unsigned first = Digit == 1U ? 0U : states[lane].digit;
    for (unsigned b = threadIdx.x; b < kRescoreBins; b += kRescoreThreads) counts[b] = 0U;
    __syncthreads();
    for (unsigned i = blockIdx.x * kRescoreThreads + threadIdx.x; i < rows; i += gridDim.x * kRescoreThreads) {
        const unsigned key = screened_head_plan::rescore_order_key_bits(__float_as_uint(s[i]));
        if (Digit == 1U) atomicAdd(&counts[key >> 20U], 1U);
        else if ((key >> 20U) == first) atomicAdd(&counts[(key >> 8U) & (kRescoreBins - 1U)], 1U);
    }
    __syncthreads();
    for (unsigned b = threadIdx.x; b < kRescoreBins; b += kRescoreThreads)
        if (counts[b] != 0U) atomicAdd(&h[b], counts[b]);
}

// One block per lane: the largest bin whose suffix count (bins >= it) reaches the rank still needed; none -> the
// threshold falls to 0 (every row: the lane has fewer rows than k). Thread t owns bins [16 t, 16 t + 16).
template <unsigned Digit>
__global__ void __launch_bounds__(kRescoreThreads) rescore_select_kernel(RescoreState* states, const unsigned* hist,
                                                                         std::size_t hist_stride, unsigned k) {
    __shared__ unsigned suffix[kRescoreThreads + 1U];
    __shared__ unsigned found;
    const unsigned lane = blockIdx.x;
    RescoreState& st = states[lane];
    const unsigned* h = hist + std::size_t{lane} * hist_stride + (Digit == 1U ? 0U : kRescoreBins);
    const unsigned need = Digit == 1U ? k : st.need;
    const unsigned t = threadIdx.x;
    unsigned own[kRescoreBinsPerThread];
    unsigned sum = 0U;
#pragma unroll
    for (unsigned j = 0; j < kRescoreBinsPerThread; ++j) {
        own[j] = h[t * kRescoreBinsPerThread + j];
        sum += own[j];
    }
    suffix[t] = sum;
    if (t == 0U) {
        suffix[kRescoreThreads] = 0U;
        found = 0U;
    }
    __syncthreads();
    if (t == 0U)  // suffix[u] = count of bins >= 16 u (integer sums: no order question)
        for (unsigned u = kRescoreThreads; u-- > 0U;) suffix[u] += suffix[u + 1U];
    __syncthreads();
    // The one owner of the answer: its range holds the bin where the suffix count first reaches `need` (need >= 1).
    if (need != 0U && suffix[t] >= need && suffix[t + 1U] < need) {
        unsigned above = suffix[t + 1U];
        unsigned bin = t * kRescoreBinsPerThread;
        bool done = false;
#pragma unroll
        for (int j = static_cast<int>(kRescoreBinsPerThread) - 1; j >= 0; --j) {
            if (done) continue;
            if (above + own[j] >= need) {
                bin = t * kRescoreBinsPerThread + static_cast<unsigned>(j);
                done = true;
            } else {
                above += own[j];
            }
        }
        found = 1U;
        if (Digit == 1U) {
            st.digit = bin;
            st.need = need - above;
        } else {
            st.key = (st.digit << 20U) | (bin << 8U);
        }
    }
    __syncthreads();
    if (t == 0U && found == 0U) {  // fewer than `need` rows: select every row
        if (Digit == 1U) {
            st.digit = 0U;
            st.need = 0U;
        } else {
            st.key = 0U;
        }
    }
}

__global__ void __launch_bounds__(kRescoreThreads) rescore_compact_kernel(const float* scores, unsigned pitch,
                                                                          unsigned rows, RescoreState* states,
                                                                          std::uint32_t* ids, std::size_t ids_stride,
                                                                          unsigned capacity) {
    const unsigned l = blockIdx.y;
    const float* s = scores + std::size_t{l} * pitch;
    RescoreState& st = states[l];
    std::uint32_t* lane_ids = ids + std::size_t{l} * ids_stride;
    const unsigned threshold = st.key;
    const unsigned lane = threadIdx.x % static_cast<unsigned>(warpSize);
    for (unsigned base = blockIdx.x * kRescoreThreads; base < rows; base += gridDim.x * kRescoreThreads) {
        const unsigned i = base + threadIdx.x;
        const bool hit = i < rows && screened_head_plan::rescore_order_key_bits(__float_as_uint(s[i])) >= threshold;
        const unsigned long long mask = __ballot(hit);
        if (mask == 0ULL) continue;
        const unsigned leader = static_cast<unsigned>(__ffsll(static_cast<long long>(mask))) - 1U;
        unsigned first = 0U;
        if (lane == leader) first = atomicAdd(&st.count, static_cast<unsigned>(__popcll(mask)));
        first = __shfl(first, static_cast<int>(leader));
        const unsigned slot = first + static_cast<unsigned>(__popcll(mask & ((1ULL << lane) - 1ULL)));
        if (hit && slot < capacity) lane_ids[slot] = i;
    }
}

// Writes the recomputed rows back: output o of pass 2 is row ids[o] (or row o when the count overflowed: every row).
__global__ void __launch_bounds__(kRescoreThreads) rescore_scatter_kernel(float* scores, unsigned pitch, unsigned rows,
                                                                          const RescoreState* states,
                                                                          const std::uint32_t* ids,
                                                                          std::size_t ids_stride, const float* y,
                                                                          std::size_t y_stride, unsigned capacity,
                                                                          float cap) {
    const unsigned l = blockIdx.y;
    const unsigned count = states[l].count;
    const bool all = count > capacity;
    const unsigned n = all ? rows : count;
    float* s = scores + std::size_t{l} * pitch;
    const std::uint32_t* lane_ids = ids + std::size_t{l} * ids_stride;
    const float* lane_y = y + std::size_t{l} * y_stride;
    for (unsigned o = blockIdx.x * kRescoreThreads + threadIdx.x; o < n; o += gridDim.x * kRescoreThreads)
        s[all ? o : lane_ids[o]] = head_transform(lane_y[o], cap);
}

// The rescore trait on `a.lanes` rows of approximate scores (screened_head_plan.hpp states the selection rule). Graph
// capture safe: one memset node and fixed grids, no host read.
inline hipError_t enqueue_rescore_rows(const screened_head_plan::RescoreLaunch& a, hipStream_t stream) {
    if (!screened_head_plan::rescore_launch_valid(a)) return hipErrorInvalidValue;
    const auto layout = screened_head_plan::rescore_layout(a.head.rows, a.capacity, a.lanes);
    auto* states = reinterpret_cast<RescoreState*>(a.scratch);
    auto* hist = reinterpret_cast<unsigned*>(a.scratch + layout.hist);
    auto* ids = reinterpret_cast<std::uint32_t*>(a.scratch + layout.ids);
    auto* y = reinterpret_cast<float*>(a.scratch + layout.y);
    if (const auto e = hipMemsetAsync(a.scratch, 0, layout.cleared, stream); e != hipSuccess) return e;
    const dim3 grid(screened_head_plan::rescore_blocks(a.head.rows), a.lanes);
    rescore_hist_kernel<1U><<<grid, kRescoreThreads, 0, stream>>>(a.scores, a.pitch, a.head.rows, states, hist,
                                                                  layout.hist_stride);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    rescore_select_kernel<1U><<<a.lanes, kRescoreThreads, 0, stream>>>(states, hist, layout.hist_stride, a.k);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    rescore_hist_kernel<2U><<<grid, kRescoreThreads, 0, stream>>>(a.scores, a.pitch, a.head.rows, states, hist,
                                                                  layout.hist_stride);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    rescore_select_kernel<2U><<<a.lanes, kRescoreThreads, 0, stream>>>(states, hist, layout.hist_stride, a.k);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    rescore_compact_kernel<<<grid, kRescoreThreads, 0, stream>>>(a.scores, a.pitch, a.head.rows, states, ids,
                                                                 layout.ids_stride, a.capacity);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    const dense_linear::View view{reinterpret_cast<const std::uint8_t*>(a.head.weights), a.head.width, a.head.rows,
                                  std::size_t{a.head.width} * sizeof(std::uint16_t)};
    for (unsigned l = 0; l < a.lanes; ++l) {
        dense_linear::mapped_rows_kernel<weight_decode::Format::bf16>
            <<<a.pass2_blocks, screened_head_plan::kRescorePass2Threads, 0, stream>>>(
            view, a.x + std::size_t{l} * a.x_stride, y + std::size_t{l} * layout.y_stride,
            dense_linear::MappedRows{ids + std::size_t{l} * layout.ids_stride, &states[l].count, a.capacity});
        if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    }
    rescore_scatter_kernel<<<grid, kRescoreThreads, 0, stream>>>(a.scores, a.pitch, a.head.rows, states, ids,
                                                                 layout.ids_stride, y, layout.y_stride, a.capacity,
                                                                 a.head.softcap);
    return hipGetLastError();
}

// Off switch of every binding (STRIX_SCREENED_HEAD=0): core/screened_head_plan.hpp.
using screened_head_plan::enabled;

// The resident screen of one BF16 softcap head: device plane, per-row norms, scratch. Built once at load from the
// resident weights (one device -> host copy of the live rows); engine-neutral. Device bytes: the plane
// (rows * plane row_bytes, 0.53 x the BF16 head at 8 bits) + 8 B per row of norms + scratch_layout(rows, capacity).
// Default 8 bits: at K 2304 the rigorous 4-bit bound keeps nearly every row as a candidate (CPU test: Gaussian rows
// 1200 of 1200, peaked cos 0.15 1175 of 1500) while 8 bits keeps 1-6, so 4 bits would only reach the fallback.
class ResidentScreen {
public:
    static constexpr unsigned kCapacity = screened_head_plan::kCapacity;

    // Returns an empty screen (usable() false) when the shape is unsupported or a weight is non-finite: the caller keeps
    // the full head. softcap must be finite and > 0 (the transform the screen proves monotone).
    // max_lanes: the most activation rows one enqueue_lanes call screens (1..kMaxLanes); the scratch holds that many
    // lanes (plan::lanes_layout, ~1.7 MB per lane at a 200k-row head). One lane keeps the one-row scratch.
    ResidentScreen(const std::uint16_t* device_weights, unsigned rows, unsigned width, float softcap,
                   hipStream_t stream, unsigned bits = 8U, unsigned max_lanes = 1U) {
        if (!screened_head_plan::shape_supported(device_weights, rows, width, softcap, bits) || max_lanes == 0U ||
            max_lanes > screened_head_plan::kMaxLanes)
            return;
        std::vector<std::uint16_t> host(std::size_t{rows} * width);
        check(hipMemcpyAsync(host.data(), device_weights, host.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost,
                             stream));
        check(hipStreamSynchronize(stream));
        const auto plan = screened_head_plan::plan_resident(host, rows, width, bits);
        host = {};
        if (!plan.usable) return;
        const auto& plane = plan.plane;
        data_ = upload(plane.data.data(), plane.data.size());
        residual_ = upload(plane.residual.data(), plane.residual.size() * sizeof(float));
        amplitude_ = upload(plane.amplitude.data(), plane.amplitude.size() * sizeof(float));
        const std::size_t scratch_bytes =
            max_lanes == 1U ? plan.layout.bytes : screened_head_plan::lanes_layout(rows, kCapacity, max_lanes).bytes;
        scratch_ = allocate(scratch_bytes);
        bytes_ = plan.device_bytes() - plan.layout.bytes + scratch_bytes;
        max_lanes_ = max_lanes;
        int device = 0;
        check(hipGetDevice(&device));
        shared_limit_ = hip_screened_head::shared_limit(device);
        view_ = screened_head_plan::make_view(plan, data_.get(), residual_.get(), amplitude_.get(), rows, width, bits);
        bound_ = plan.bound;
        head_ = {device_weights, rows, width, softcap};
    }

    [[nodiscard]] bool usable() const noexcept { return scratch_ != nullptr; }
    [[nodiscard]] std::size_t device_bytes() const noexcept { return bytes_; }

    // x: FP32 [width] on the device; device_reply: device-visible reply (mapped pinned). Same token and status as the
    // full head + enqueue_hip_greedy_rows over the same rows.
    hipError_t enqueue(const float* x, HipGreedyRowsReply* device_reply, hipStream_t stream) const {
        if (!usable()) return hipErrorNotReady;
        return enqueue_screened_greedy(view_, bound_, head_, x, static_cast<std::uint8_t*>(scratch_.get()), kCapacity,
                                       device_reply, stream);
    }

    [[nodiscard]] unsigned max_lanes() const noexcept { return max_lanes_; }

    // lanes rows x + l * x_stride (FP32, x_stride a multiple of 4 floats >= width), device_replies[lanes]: lane l gets
    // the token and status enqueue() gives on that row alone. One lane is enqueue().
    hipError_t enqueue_lanes(const float* x, unsigned lanes, unsigned x_stride, HipGreedyRowsReply* device_replies,
                             hipStream_t stream) const {
        if (!usable()) return hipErrorNotReady;
        if (lanes == 0U || lanes > max_lanes_) return hipErrorInvalidValue;
        const unsigned per = screened_head_plan::pass1_lanes(head_.width, lanes, shared_limit_);
        const screened_head_plan::Launch launch{view_, bound_, head_, x, static_cast<std::uint8_t*>(scratch_.get()),
                                                kCapacity, lanes, x_stride, Grids{}, per == 0U ? 1U : per};
        return enqueue_screened_greedy_lanes(launch, device_replies, stream);
    }

private:
    struct Free {
        void operator()(void* p) const noexcept { (void)hipFree(p); }
    };
    using Device = std::unique_ptr<void, Free>;
    static void check(hipError_t status) {
        if (status != hipSuccess) throw std::runtime_error(std::string("screened head: ") + hipGetErrorString(status));
    }
    static Device allocate(std::size_t bytes) {
        void* p = nullptr;
        check(hipMalloc(&p, bytes));
        return Device(p);
    }
    static Device upload(const void* host, std::size_t bytes) {
        auto device = allocate(bytes);
        check(hipMemcpy(device.get(), host, bytes, hipMemcpyHostToDevice));
        return device;
    }

    Device data_, residual_, amplitude_, scratch_;
    PlaneView view_{};
    screened_head::Bound bound_{};
    HeadSpec head_{};
    std::size_t bytes_ = 0;
    std::size_t shared_limit_ = 0;
    unsigned max_lanes_ = 0;
};

}  // namespace strix::core::hip_screened_head
#endif
