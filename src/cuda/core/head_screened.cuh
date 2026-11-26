#pragma once
// EXACT screened greedy LM head, CUDA bodies. Engine-neutral: rows / width / softcap / plane are arguments. The plan
// (State, PlaneView, HeadSpec, Grids, Launch, scratch layouts, launch contract, plane build) is core/screened_head_plan.hpp,
// the bound math and the invariant core/screened_head.hpp; the same plan drives the HIP bodies (core/hip_screened_head.hpp).
// Five fixed launches, no host read between them (graph-capture safe), each a separate stage function so a step program
// can bind them as five one-launch leaves:
//   1 prep     one block per lane: nx = inflated ||x||_2, the FALLBACK flag (non-finite x or overflow guard), state reset
//   2 pass1    warp per row tile (plan::pass1_tile_rows rows), up to kMaxPass1Lanes lanes per block (the plane row is
//              read once for all of them; lane groups folded into the block index, one wave of resident blocks):
//              y_hat, e, U (stored per lane), Lstar = max L (ordered-key atomic max per lane)
//   3 compact  per lane: candidates i with G(U_i) >= G(Lstar) -> ids (warp-aggregated atomic); FALLBACK or too many -> ALL
//   4 pass2    the incumbent R=1 FP32 head row body (dense_bf16_fp32_r1_mapped, the mapped trait of
//              dense_bf16_fp32_r1_store) on each lane's candidate rows, storing G(y) = softcap(bf16(y)) like the incumbent
//   5 pick     one block per lane: the shared head pick (strix::head_argmax rule) over the candidates, ids as indices,
//              handed to Emit as row = lane
// The token and the status are the incumbent's by construction: candidate logits are computed by the incumbent body, the
// pick is the incumbent rule (a total order, so any visiting order gives the same Pick), and every row that could be
// the argmax or tie it is a candidate (monotone G, rigorous bound). Fallback (non-finite x, nx * max A >= 1e37, a
// non-finite bound, more than `capacity` candidates) runs pass 2 and the pick on ALL rows of that lane. With finite x
// and the guard met every incumbent logit is finite (|y| <= ||w|| ||x|| (1 + gamma) < 1e37), so the incumbent's
// non-finite count is 0 exactly when the screened one is.
// Lanes: lane l of a multi-row head is bitwise the R=1 head of its row; the incumbent multi-row FP32 rows kernel keeps the
// same per-(output, row) order (dense.cu dot_bf16), so lane l also equals row l of the multi-row head.
// Runner-up trait (Launch::runner_up): the threshold is the second-largest lower bound, so the candidates cover the
// first and the second row and launch_pick2 emits the shared top-2 (head_argmax::Pick2) of the full head.
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <type_traits>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.cuh"
#include "core/row_transforms.hpp"
#include "core/screened_head.hpp"
#include "core/screened_head_plan.hpp"
#include "dense/rows_bf16_epilogue.cuh"
#include "head_argmax.cuh"
#include "strix/head_argmax.hpp"

namespace strix::cuda::core::head_screened {

namespace ha = ::strix::head_argmax;
namespace plan = ::strix::core::screened_head_plan;
namespace sh = ::strix::core::screened_head;

using plan::Grids;
using plan::HeadSpec;
using plan::Launch;
using plan::PlaneView;
using plan::State;

// Order-preserving key of a finite or infinite float (larger value -> larger key); 0 is below every key.
__device__ __forceinline__ unsigned order_key(float v) {
    const unsigned b = __float_as_uint(v);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}
__device__ __forceinline__ float key_value(unsigned k) {
    return __uint_as_float((k & 0x80000000U) != 0U ? (k & 0x7fffffffU) : ~k);
}

// The incumbent post-transform: the same function row_transforms::Bf16Softcap applies.
__device__ __forceinline__ float head_transform(float y, float cap) {
    return ::strix::core::row_transforms::bf16_softcap(y, cap);
}

namespace detail {

// Lane strides of the scratch parts (elements), carried by value into every kernel.
struct Strides {
    unsigned x, ids, upper, y;
};

// Kernels are templates (one implicit instantiation per program, as in head_argmax.cuh) so several TUs may include this.
template <int = 0>
__global__ void __launch_bounds__(256) prep_kernel(const float* x, unsigned x_stride, unsigned width, sh::Bound bound,
                                                   float max_amplitude, State* state, unsigned* key2) {
    __shared__ float partial[8];
    x += std::size_t{blockIdx.x} * x_stride;
    float s = 0.0F;
    for (unsigned k = threadIdx.x; k < width; k += 256U) s = __fmaf_rn(x[k], x[k], s);
    s = ::strix::cuda_core::warp_sum_xor(s);
    if ((threadIdx.x & 31U) == 0U) partial[threadIdx.x / 32U] = s;
    __syncthreads();
    if (threadIdx.x == 0U) {
        float t = 0.0F;
        for (unsigned w = 0; w < 8U; ++w) t = __fadd_rn(t, partial[w]);
        const float nx = __fsqrt_rn(__fmul_rn(__fadd_rn(t, bound.sum_abs), bound.sum_inflate));
        State& st = state[blockIdx.x];
        st.nx = nx;
        st.flag = __fmul_rn(nx, max_amplitude) < sh::kOverflowGuard ? 0U : 1U;  // NaN / inf -> 1
        st.count = 0U;
        st.lkey = 0U;
        key2[blockIdx.x] = 0U;
    }
}

// Lane group g = block % groups serves lanes [g * Lanes, min(lanes, (g + 1) * Lanes)); xs holds their activations
// ([Lanes][width]). Each warp carries TileRows consecutive plane rows (plan::pass1_tile_rows): the activations it reads
// from shared memory for one unit serve all of them. Per (row, lane) the arithmetic is the one-row form's (thread t folds
// units t, t + 32, ... in order, then the warp sum), so every bound is bitwise independent of the schedule.
// Top2 (runner-up trait) also tracks the second-largest lower bound per lane in key2: each warp folds its own top two,
// then max(key1) and max(min(previous key1, own first), own second) give the global second-largest of the multiset.
template <unsigned Bits, unsigned Lanes, bool Top2>
// Two blocks per SM: a group of 8 lanes carries a 4-row tile (32 accumulators + 32 decoded weights per thread); the
// register cap keeps 16 warps resident (one block of 8 warps cannot cover the memory latency).
__global__ void __launch_bounds__(plan::kPass1Waves * 32U, 2) pass1_kernel(PlaneView plane, const float* x, unsigned lanes,
                                                                        unsigned groups, Strides strides, sh::Bound bound,
                                                                        State* state, float* upper, unsigned* key2) {
    constexpr unsigned TileRows = plan::pass1_tile_rows(Lanes);
    extern __shared__ __align__(16) float xs[];
    __shared__ float nxs[Lanes];
    const auto map = plan::pass1_map(blockIdx.x, gridDim.x, groups);
    const unsigned first = map.group * Lanes;
    const unsigned live = lanes - first < Lanes ? lanes - first : Lanes;
    bool any = false;
#pragma unroll
    for (unsigned l = 0; l < Lanes; ++l) {
        const bool on = l < live && state[first + l].flag == 0U;
        any = any || on;
        if (threadIdx.x == 0U) nxs[l] = on ? state[first + l].nx : 0.0F;
    }
    if (!any) return;  // uniform: every lane of this group runs the full head
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const unsigned quads = plane.width / 4U;
    for (unsigned i = threadIdx.x; i < Lanes * quads; i += blockDim.x) {
        const unsigned l = i / quads, q = i % quads;
        reinterpret_cast<float4*>(xs)[i] = l < live ? reinterpret_cast<const float4*>(
                                                          x + std::size_t{first + l} * strides.x)[q]
                                                    : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
    __syncthreads();
    const unsigned units = plane.width / sh::kUnit;
    float lmax[Lanes], lmax2[Lanes];
    unsigned bad = 0U;  // bit l: lane l saw a non-finite bound
#pragma unroll
    for (unsigned l = 0; l < Lanes; ++l) lmax[l] = lmax2[l] = __int_as_float(static_cast<int>(0xff800000U));  // -inf
    const unsigned tiles = (plane.rows + TileRows - 1U) / TileRows, step = plan::pass1_tile_step(map);
    for (unsigned tile = plan::pass1_first_tile(map, warp); tile < tiles; tile += step) {
        const unsigned row0 = tile * TileRows;
        const std::uint8_t* base[TileRows];
#pragma unroll
        for (unsigned r = 0; r < TileRows; ++r)  // a clipped tail row re-reads row0 (in bounds) and is never stored
            base[r] = plane.data + std::size_t{row0 + r < plane.rows ? row0 + r : row0} * plane.row_bytes;
        float acc[TileRows][Lanes];
#pragma unroll
        for (unsigned r = 0; r < TileRows; ++r)
#pragma unroll
            for (unsigned l = 0; l < Lanes; ++l) acc[r][l] = 0.0F;
        for (unsigned u = lane; u < units; u += 32U) {
            float q[TileRows][8], s[TileRows];
#pragma unroll
            for (unsigned r = 0; r < TileRows; ++r) {
                if constexpr (Bits == 4U) {
                    const unsigned word = *reinterpret_cast<const unsigned*>(base[r] + 4U * u);
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j)
                        q[r][j] = static_cast<float>(static_cast<int>((word >> (4U * j)) & 15U) - 8);
                } else {
                    const uint2 word = *reinterpret_cast<const uint2*>(base[r] + 8U * u);
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j) {
                        const unsigned w32 = j < 4U ? word.x : word.y;
                        q[r][j] = static_cast<float>(
                            static_cast<int>(static_cast<std::int8_t>((w32 >> (8U * (j % 4U))) & 255U)));
                    }
                }
                const std::uint16_t* scales = reinterpret_cast<const std::uint16_t*>(base[r] + plane.scales_offset);
                s[r] = __uint_as_float(static_cast<unsigned>(scales[u / 4U]) << 16U);
            }
#pragma unroll
            for (unsigned l = 0; l < Lanes; ++l) {
                const float4 x0 = *reinterpret_cast<const float4*>(xs + l * plane.width + 8U * u);
                const float4 x1 = *reinterpret_cast<const float4*>(xs + l * plane.width + 8U * u + 4U);
                const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (unsigned r = 0; r < TileRows; ++r) {
                    float inner = 0.0F;
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j) inner = __fmaf_rn(q[r][j], xv[j], inner);
                    acc[r][l] = __fmaf_rn(s[r], inner, acc[r][l]);
                }
            }
        }
#pragma unroll
        for (unsigned r = 0; r < TileRows; ++r) {
            const unsigned row = row0 + r;
            if (row >= plane.rows) break;  // warp-uniform
            const float slack = __fmul_rn(bound.gamma, plane.amplitude[row]);
            const float residual = plane.residual[row];
#pragma unroll
            for (unsigned l = 0; l < Lanes; ++l) {
                const float yhat = ::strix::cuda_core::warp_sum_xor(acc[r][l]);
                const float e = __fadd_rn(__fmul_rn(__fmul_rn(nxs[l], __fadd_rn(residual, slack)), bound.margin),
                                          bound.abs_slack);
                // Outward-rounded interval: L <= yhat - e (round down), U >= yhat + e (round up), both as real numbers.
                const float hi = __fadd_ru(yhat, e), lo = __fadd_rd(yhat, -e);
                if (l < live) {
                    if (lane == 0U) upper[std::size_t{first + l} * strides.upper + row] = hi;
                    if (!(isfinite(hi) && isfinite(lo))) {
                        bad |= 1U << l;
                    } else if constexpr (Top2) {
                        if (lo > lmax[l]) {
                            lmax2[l] = lmax[l];
                            lmax[l] = lo;
                        } else if (lo > lmax2[l]) {
                            lmax2[l] = lo;
                        }
                    } else {
                        lmax[l] = fmaxf(lmax[l], lo);
                    }
                }
            }
        }
    }
    if (lane == 0U) {
#pragma unroll
        for (unsigned l = 0; l < Lanes; ++l) {
            if (l >= live) continue;
            if ((bad >> l) & 1U) atomicOr(&state[first + l].flag, 1U);
            const float none = __int_as_float(static_cast<int>(0xff800000U));
            if (lmax[l] > none) {
                const unsigned k1 = order_key(lmax[l]);
                const unsigned previous = atomicMax(&state[first + l].lkey, k1);
                if constexpr (Top2) {
                    atomicMax(&key2[first + l], previous < k1 ? previous : k1);
                    if (lmax2[l] > none) atomicMax(&key2[first + l], order_key(lmax2[l]));
                }
            }
        }
    }
}

// blockIdx.y = lane. Top2: the threshold is the second-largest lower bound; none (fewer than two finite bounds) runs
// every row.
template <bool Top2>
__global__ void __launch_bounds__(plan::kCompactThreads) compact_kernel(const float* upper, unsigned rows, float cap,
                                                                        unsigned capacity, Strides strides, State* state,
                                                                        std::uint32_t* ids, const unsigned* key2) {
    State& st = state[blockIdx.y];
    const unsigned key = Top2 ? key2[blockIdx.y] : st.lkey;
    if (st.flag != 0U || (Top2 && key == 0U)) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) st.count = plan::kAllRows;
        return;
    }
    upper += std::size_t{blockIdx.y} * strides.upper;
    ids += std::size_t{blockIdx.y} * strides.ids;
    const float threshold = head_transform(key_value(key), cap);
    const unsigned lane = threadIdx.x & 31U;
    for (unsigned base = blockIdx.x * plan::kCompactThreads; base < rows; base += gridDim.x * plan::kCompactThreads) {
        const unsigned i = base + threadIdx.x;
        const bool hit = i < rows && head_transform(upper[i], cap) >= threshold;
        const unsigned mask = __ballot_sync(0xffffffffU, hit);
        if (mask == 0U) continue;
        const unsigned leader = static_cast<unsigned>(__ffs(static_cast<int>(mask))) - 1U;
        unsigned first = 0U;
        if (lane == leader) first = atomicAdd(&st.count, static_cast<unsigned>(__popc(mask)));
        first = __shfl_sync(0xffffffffU, first, static_cast<int>(leader));
        const unsigned slot = first + static_cast<unsigned>(__popc(mask & ((1U << lane) - 1U)));
        if (hit && slot < capacity) ids[slot] = i;
    }
}

// y already holds G(y) (stored by the mapped pass 2 like the incumbent's softcap store): fold with the shared rule
// (P = ha::Pick, or ha::Pick2 for the runner-up trait). blockIdx.x = lane; emit(lane, P).
template <class Emit, class P>
__global__ void __launch_bounds__(256) pick_kernel(const float* y, const std::uint32_t* ids, const State* state,
                                                   unsigned rows, unsigned capacity, Strides strides, Emit emit) {
    const unsigned count = state[blockIdx.x].count;
    y += std::size_t{blockIdx.x} * strides.y;
    ids += std::size_t{blockIdx.x} * strides.ids;
    const bool all = count > capacity;
    const unsigned n = all ? rows : count;
    P p;
    for (unsigned i = threadIdx.x; i < n; i += 256U) {
        const float v = y[i];
        const std::uint32_t id = all ? i : ids[i];
        if constexpr (std::is_same_v<P, ha::Pick2>) {
            if (!isfinite(v)) ++p.first.nonfinite;
            else ha::push2(p, v, id);
        } else {
            if (!isfinite(v)) ++p.nonfinite;
            else if (ha::better(v, id, p.value, p.index)) { p.value = v; p.index = id; }
        }
    }
    p = ::strix::cuda::core::head_argmax::detail::block_merge(p);
    if (threadIdx.x == 0U) emit(blockIdx.x, p);
}

struct Parts {
    State* state;
    std::uint32_t* ids;
    float* upper;
    float* y;
    unsigned* key2;
    Strides strides;
};

inline Parts parts(const Launch& a) {
    const auto layout = plan::lanes_layout(a.head.rows, a.capacity, a.lanes);
    const auto narrow = [](std::size_t v) { return static_cast<unsigned>(v); };
    return {reinterpret_cast<State*>(a.scratch), reinterpret_cast<std::uint32_t*>(a.scratch + layout.ids),
            reinterpret_cast<float*>(a.scratch + layout.upper), reinterpret_cast<float*>(a.scratch + layout.y),
            reinterpret_cast<unsigned*>(a.scratch + layout.key2),
            {a.lanes > 1U ? a.x_stride : a.head.width, narrow(layout.ids_stride), narrow(layout.upper_stride),
             narrow(layout.y_stride)}};
}

inline std::size_t shared_limit(int device) {
    int optin = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess || optin <= 0)
        return 48U * 1024U;
    return static_cast<std::size_t>(optin);
}

template <unsigned Bits, unsigned Lanes, bool Top2>
cudaError_t ensure_pass1_shared(std::size_t bytes) {
    // Raised once per instantiation to the largest request so far (host call; never inside a capture after the first
    // prepare(), which every resident owner runs at load).
    static std::atomic<std::size_t> granted{48U * 1024U};
    std::size_t current = granted.load();
    if (bytes <= current) return cudaSuccess;
    if (const auto e = cudaFuncSetAttribute(pass1_kernel<Bits, Lanes, Top2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(bytes));
        e != cudaSuccess)
        return e;
    while (bytes > current && !granted.compare_exchange_weak(current, bytes)) {}
    return cudaSuccess;
}

template <unsigned Bits, bool Top2>
cudaError_t pass1_form(const Launch& a, const Parts& p, cudaStream_t stream, bool launch, unsigned* resident = nullptr) {
    // The kernel form, the group count and the geometry all come from the launch (a.pass1_lanes), the bound one
    // (plan::pass1_grid): a caller cannot pass a lane count that disagrees with the grid.
    const unsigned per_group = a.pass1_lanes;
    const auto geometry = plan::pass1_grid(a);
    const std::size_t smem = geometry.shared_bytes;
    const unsigned groups = plan::pass1_groups(a.lanes, per_group);
    const dim3 grid(geometry.blocks), block(geometry.threads);
    const auto run = [&](auto kernel, cudaError_t ready) {
        if (ready != cudaSuccess) return ready;
        if (resident) {  // host query at load: SMs x blocks per SM of this instantiation at this shared size
            int device = 0, sms = 0, per_sm = 0;
            if (const auto e = cudaGetDevice(&device); e != cudaSuccess) return e;
            if (const auto e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); e != cudaSuccess)
                return e;
            if (const auto e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                                             static_cast<int>(block.x), smem);
                e != cudaSuccess)
                return e;
            *resident = static_cast<unsigned>(sms > 0 ? sms : 0) * static_cast<unsigned>(per_sm > 0 ? per_sm : 0);
        }
        if (!launch) return cudaSuccess;
        kernel<<<grid, block, smem, stream>>>(a.plane, a.x, a.lanes, groups, p.strides, a.bound, p.state, p.upper,
                                              p.key2);
        return cudaGetLastError();
    };
    switch (per_group) {
    case 1U: return run(pass1_kernel<Bits, 1U, Top2>, ensure_pass1_shared<Bits, 1U, Top2>(smem));
    case 2U: return run(pass1_kernel<Bits, 2U, Top2>, ensure_pass1_shared<Bits, 2U, Top2>(smem));
    case 4U: return run(pass1_kernel<Bits, 4U, Top2>, ensure_pass1_shared<Bits, 4U, Top2>(smem));
    case 8U: return run(pass1_kernel<Bits, 8U, Top2>, ensure_pass1_shared<Bits, 8U, Top2>(smem));
    default: return cudaErrorInvalidValue;
    }
}

template <unsigned Bits>
cudaError_t pass1_bits(const Launch& a, const Parts& p, cudaStream_t stream, bool launch,
                       unsigned* resident = nullptr) {
    return a.runner_up ? pass1_form<Bits, true>(a, p, stream, launch, resident)
                       : pass1_form<Bits, false>(a, p, stream, launch, resident);
}

}  // namespace detail

// Lanes per pass-1 block row on this device (0: the width does not fit; the caller keeps the full head).
inline unsigned pass1_lanes(unsigned width, unsigned lanes, int device = 0) {
    return plan::pass1_lanes(width, lanes, detail::shared_limit(device));
}

// ---- The five stages (one kernel launch each). `a` must satisfy plan::launch_valid; enqueue() checks it once. ----
inline cudaError_t launch_prep(const Launch& a, cudaStream_t stream) {
    const auto p = detail::parts(a);
    detail::prep_kernel<0> <<<a.lanes, 256, 0, stream>>>(a.x, p.strides.x, a.head.width, a.bound, a.plane.max_amplitude,
                                                     p.state, p.key2);
    return cudaGetLastError();
}

// Raises the pass-1 shared-memory limit of the launch's instantiation (host only; the owner runs it once at load for
// every lane count it will launch, so no enqueue inside a capture has to).
inline cudaError_t prepare_pass1(const Launch& a) {
    const auto p = detail::parts(a);
    return a.plane.bits == 4U ? detail::pass1_bits<4U>(a, p, nullptr, false)
                              : detail::pass1_bits<8U>(a, p, nullptr, false);
}

// prepare_pass1 plus the pass-1 blocks the device holds at once for the launch's instantiation (Launch::pass1_resident;
// host only, at load).
inline cudaError_t prepare_pass1(const Launch& a, unsigned& resident) {
    const auto p = detail::parts(a);
    resident = 0U;
    return a.plane.bits == 4U ? detail::pass1_bits<4U>(a, p, nullptr, false, &resident)
                              : detail::pass1_bits<8U>(a, p, nullptr, false, &resident);
}

inline cudaError_t launch_pass1(const Launch& a, cudaStream_t stream) {
    const auto p = detail::parts(a);
    return a.plane.bits == 4U ? detail::pass1_bits<4U>(a, p, stream, true)
                              : detail::pass1_bits<8U>(a, p, stream, true);
}

inline cudaError_t launch_compact(const Launch& a, cudaStream_t stream) {
    const auto p = detail::parts(a);
    const dim3 grid(a.grids.compact, a.lanes);
    if (a.runner_up)
        detail::compact_kernel<true><<<grid, plan::kCompactThreads, 0, stream>>>(
            p.upper, a.head.rows, a.head.softcap, a.capacity, p.strides, p.state, p.ids, p.key2);
    else
        detail::compact_kernel<false><<<grid, plan::kCompactThreads, 0, stream>>>(
            p.upper, a.head.rows, a.head.softcap, a.capacity, p.strides, p.state, p.ids, p.key2);
    return cudaGetLastError();
}

inline cudaError_t launch_pass2(const Launch& a, cudaStream_t stream) {
    const auto p = detail::parts(a);
    const ::strix::cuda_core::MappedLaneStrides lanes{p.strides.x, p.strides.y, p.strides.ids,
                                                      sizeof(State) / sizeof(std::uint32_t)};
    return ::strix::cuda_core::dense_bf16_fp32_r1_mapped_epilogue(
        a.head.weights, a.x, p.y, p.ids, &p.state->count, a.capacity, a.head.rows, a.head.width,
        ::strix::cuda_core::Bf16Fp32Store::softcap, a.head.softcap, a.grids.pass2, stream, a.lanes, lanes);
}

// emit(lane, ha::Pick) runs on the device once per lane (head_argmax Emit contract, the engine's adapter); the Pick's
// value is G(y) of the winner.
template <class Emit>
cudaError_t launch_pick(const Launch& a, Emit emit, cudaStream_t stream) {
    const auto p = detail::parts(a);
    detail::pick_kernel<Emit, ha::Pick>
        <<<a.lanes, 256, 0, stream>>>(p.y, p.ids, p.state, a.head.rows, a.capacity, p.strides, emit);
    return cudaGetLastError();
}

// Runner-up trait of the pick: emit(lane, ha::Pick2), the shared top-2 (first == launch_pick's Pick bit for bit, second
// == the incumbent runner-up). Requires a runner_up launch (its candidates cover the second row).
template <class Emit>
cudaError_t launch_pick2(const Launch& a, Emit emit, cudaStream_t stream) {
    if (!a.runner_up) return cudaErrorInvalidValue;
    const auto p = detail::parts(a);
    detail::pick_kernel<Emit, ha::Pick2>
        <<<a.lanes, 256, 0, stream>>>(p.y, p.ids, p.state, a.head.rows, a.capacity, p.strides, emit);
    return cudaGetLastError();
}

// The device State of lane `lane` (diagnostics: copy it after the pick to read the candidate count / fallback).
inline const State* lane_state(const Launch& a, unsigned lane = 0U) {
    return reinterpret_cast<const State*>(a.scratch) + lane;
}

// The five launches in order (TopTwo: the runner-up pick, which needs a.runner_up).
template <bool TopTwo = false, class Emit>
cudaError_t enqueue(const Launch& a, Emit emit, cudaStream_t stream) {
    if (!plan::launch_valid(a) || (TopTwo && !a.runner_up)) return cudaErrorInvalidValue;
    if (const auto e = launch_prep(a, stream); e != cudaSuccess) return e;
    if (const auto e = launch_pass1(a, stream); e != cudaSuccess) return e;
    if (const auto e = launch_compact(a, stream); e != cudaSuccess) return e;
    if (const auto e = launch_pass2(a, stream); e != cudaSuccess) return e;
    if constexpr (TopTwo) return launch_pick2(a, emit, stream);
    else return launch_pick(a, emit, stream);
}

// Grids from the device: pass 1 / compact / pass 2 blocks as multiples of the SM count (no laboratory value compiled in).
inline Grids default_grids(int device = 0) {
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms <= 0) return Grids{};
    const auto n = static_cast<unsigned>(sms);
    Grids g;
    g.pass1 = n * 8U;
    g.compact = n * 2U;
    g.pass2 = n * 4U;
    return g;
}

// The R=1 form over explicit device buffers. x: FP32 [width], 16-byte aligned. scratch: scratch_layout bytes, device.
template <class Emit>
cudaError_t enqueue_screened_greedy_cuda(const PlaneView& view, const sh::Bound& bound, const HeadSpec& head,
                                         const float* x, std::uint8_t* scratch, unsigned capacity, Emit emit,
                                         cudaStream_t stream, const Grids& grids) {
    return enqueue(Launch{view, bound, head, x, scratch, capacity, 1U, head.width, grids, 1U}, emit, stream);
}

// The resident screen of one BF16 softcap head: device plane, per-row norms, scratch for up to `max_lanes` lanes. Built
// once at load from the resident weights (one device -> host copy of the live rows). Plan, bound and view come from
// core/screened_head_plan.hpp.
class ResidentScreen {
public:
    static constexpr unsigned kCapacity = plan::kCapacity;

    // Empty (usable() false) when the shape is unsupported, a weight is non-finite or pass 1 cannot stage one lane:
    // the caller keeps the full head.
    ResidentScreen(const std::uint16_t* device_weights, unsigned rows, unsigned width, float softcap,
                   cudaStream_t stream, unsigned bits = 8U, unsigned max_lanes = 1U) {
        if (!plan::shape_supported(device_weights, rows, width, softcap, bits) || max_lanes == 0U ||
            max_lanes > plan::kMaxLanes)
            return;
        int device = 0;
        check(cudaGetDevice(&device));
        shared_limit_ = detail::shared_limit(device);
        if (plan::pass1_lanes(width, 1U, shared_limit_) == 0U) return;
        std::vector<std::uint16_t> host(std::size_t{rows} * width);
        check(cudaMemcpyAsync(host.data(), device_weights, host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                              stream));
        check(cudaStreamSynchronize(stream));
        const auto built = plan::plan_resident(host, rows, width, bits);
        host = {};
        if (!built.usable) return;
        const auto& plane = built.plane;
        data_ = upload(plane.data.data(), plane.data.size());
        residual_ = upload(plane.residual.data(), plane.residual.size() * sizeof(float));
        amplitude_ = upload(plane.amplitude.data(), plane.amplitude.size() * sizeof(float));
        const auto layout = plan::lanes_layout(rows, kCapacity, max_lanes);
        scratch_ = allocate(layout.bytes);
        bytes_ = plane.bytes() + layout.bytes;
        max_lanes_ = max_lanes;
        view_ = plan::make_view(built, data_.get(), residual_.get(), amplitude_.get(), rows, width, bits);
        bound_ = built.bound;
        head_ = {device_weights, rows, width, softcap};
        grids_ = default_grids(device);
        for (const bool second : {false, true}) {
            const auto prepare = [&](unsigned lanes) {
                const auto a = launch({}, lanes, width, second);
                check(prepare_pass1(a, resident_[second ? 1 : 0][form_index(a.pass1_lanes)]));
            };
            for (unsigned lanes = 1U; lanes <= max_lanes; lanes *= 2U) prepare(lanes);
            prepare(max_lanes);
        }
    }

    [[nodiscard]] bool usable() const noexcept { return scratch_ != nullptr; }
    [[nodiscard]] std::size_t device_bytes() const noexcept { return bytes_; }
    [[nodiscard]] unsigned max_lanes() const noexcept { return max_lanes_; }
    void set_grids(const Grids& g) noexcept { grids_ = g; }
    [[nodiscard]] const Grids& grids() const noexcept { return grids_; }

    // The launch record of `lanes` activation rows (x + l * x_stride), for callers that bind the five stages themselves.
    // runner_up: candidates cover the second row too (launch_pick2 / enqueue<true>).
    [[nodiscard]] Launch launch(const float* x, unsigned lanes = 1U, unsigned x_stride = 0U,
                                bool runner_up = false) const noexcept {
        const unsigned per_group = plan::pass1_lanes(head_.width, lanes, shared_limit_);
        return {view_,
                bound_,
                head_,
                x,
                static_cast<std::uint8_t*>(scratch_.get()),
                kCapacity,
                lanes,
                lanes > 1U ? x_stride : head_.width,
                grids_,
                per_group,
                runner_up,
                resident_[runner_up ? 1 : 0][form_index(per_group)]};
    }

    // x: FP32 [width] (or lanes rows at x_stride floats) on the device. Same Pick per lane as the full head
    // (dense_bf16_fp32_r1 softcap + head_argmax) over the same rows.
    template <class Emit>
    cudaError_t enqueue(const float* x, Emit emit, cudaStream_t stream, unsigned lanes = 1U, unsigned x_stride = 0U) const {
        if (!usable()) return cudaErrorNotReady;
        if (lanes == 0U || lanes > max_lanes_) return cudaErrorInvalidValue;
        return head_screened::enqueue(launch(x, lanes, x_stride), emit, stream);
    }

    // Runner-up form: emit(lane, ha::Pick2) == the shared top-2 of the full head, per lane.
    template <class Emit>
    cudaError_t enqueue2(const float* x, Emit emit, cudaStream_t stream, unsigned lanes = 1U, unsigned x_stride = 0U) const {
        if (!usable()) return cudaErrorNotReady;
        if (lanes == 0U || lanes > max_lanes_) return cudaErrorInvalidValue;
        return head_screened::enqueue<true>(launch(x, lanes, x_stride, true), emit, stream);
    }

private:
    struct Free {
        void operator()(void* p) const noexcept { (void)cudaFree(p); }
    };
    using Device = std::unique_ptr<void, Free>;
    static void check(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(std::string("screened head: ") + cudaGetErrorString(status));
    }
    static Device allocate(std::size_t bytes) {
        void* p = nullptr;
        check(cudaMalloc(&p, bytes));
        return Device(p);
    }
    static Device upload(const void* host, std::size_t bytes) {
        auto device = allocate(bytes);
        check(cudaMemcpy(device.get(), host, bytes, cudaMemcpyHostToDevice));
        return device;
    }

    Device data_, residual_, amplitude_, scratch_;
    PlaneView view_{};
    sh::Bound bound_{};
    HeadSpec head_{};
    Grids grids_{};
    std::size_t bytes_ = 0;
    unsigned max_lanes_ = 0;
    std::size_t shared_limit_ = 48U * 1024U;
    // Pass-1 resident blocks per instantiation: [runner_up][log2 lanes per group], measured at load (0: not prepared).
    static constexpr unsigned kForms = 4U;  // 1, 2, 4, 8 lanes per group (plan::kMaxPass1Lanes)
    static_assert(plan::kMaxPass1Lanes == 1U << (kForms - 1U), "one resident entry per pass-1 lane form");
    static constexpr unsigned form_index(unsigned per_group) noexcept {
        unsigned i = 0U;
        while (i + 1U < kForms && (1U << i) < per_group) ++i;
        return i;
    }
    unsigned resident_[2][kForms] = {};
};

}  // namespace strix::cuda::core::head_screened
