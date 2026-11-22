#pragma once
// EXACT screened greedy LM head, CUDA bodies. Engine-neutral: rows / width / softcap / plane are arguments. The plan
// (State, PlaneView, HeadSpec, Grids, scratch layout, launch contract, plane build) is core/screened_head_plan.hpp, the bound
// math and the invariant core/screened_head.hpp; the same plan drives the HIP bodies (core/hip_screened_head.hpp).
// Five fixed launches, no host read between them (graph-capture safe):
//   1 prep     one block: nx = inflated ||x||_2, the FALLBACK flag (non-finite x or overflow guard), state reset
//   2 pass1    warp per row over the narrow plane: y_hat_i, e_i, U_i (stored), Lstar = max L_i (ordered-key atomic max)
//   3 compact  candidates i with G(U_i) >= G(Lstar) -> ids (warp-aggregated atomic); FALLBACK or too many -> ALL rows
//   4 pass2    the incumbent R=1 FP32 head row body (dense_bf16_fp32_r1_mapped, the mapped trait of
//              dense_bf16_fp32_r1_store) on the candidate rows, storing G(y) = softcap(bf16(y)) like the incumbent
//   5 pick     one block: the shared head pick (strix::head_argmax rule) over the candidates, ids as indices, handed to Emit
// The token and the status are the incumbent's by construction: candidate logits are computed by the incumbent body, the
// pick is the incumbent rule (a total order, so any visiting order gives the same Pick), and every row that could be
// the argmax or tie it is a candidate (monotone G, rigorous bound). Fallback (non-finite x, nx * max A >= 1e37, a
// non-finite bound, more than `capacity` candidates) runs pass 2 and the pick on ALL rows.
#include <cuda_runtime.h>

#include <cmath>
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

__global__ void __launch_bounds__(256) prep_kernel(const float* x, unsigned width, sh::Bound bound, float max_amplitude,
                                                   State* state) {
    __shared__ float partial[8];
    float s = 0.0F;
    for (unsigned k = threadIdx.x; k < width; k += 256U) s = __fmaf_rn(x[k], x[k], s);
    s = ::strix::cuda_core::warp_sum_xor(s);
    if ((threadIdx.x & 31U) == 0U) partial[threadIdx.x / 32U] = s;
    __syncthreads();
    if (threadIdx.x == 0U) {
        float t = 0.0F;
        for (unsigned w = 0; w < 8U; ++w) t = __fadd_rn(t, partial[w]);
        const float nx = __fsqrt_rn(__fmul_rn(__fadd_rn(t, bound.sum_abs), bound.sum_inflate));
        state->nx = nx;
        state->flag = __fmul_rn(nx, max_amplitude) < sh::kOverflowGuard ? 0U : 1U;  // NaN / inf -> 1
        state->count = 0U;
        state->lkey = 0U;
    }
}

template <unsigned Bits>
__global__ void __launch_bounds__(plan::kPass1Waves * 32U) pass1_kernel(PlaneView plane, const float* x, sh::Bound bound,
                                                                        State* state, float* upper) {
    extern __shared__ __align__(16) float xs[];
    if (state->flag != 0U) return;  // uniform: the full head runs
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    for (unsigned i = threadIdx.x; i < plane.width / 4U; i += blockDim.x)
        reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x)[i];
    __syncthreads();
    const float nx = state->nx;
    const unsigned units = plane.width / sh::kUnit;
    float lmax = __int_as_float(static_cast<int>(0xff800000U));  // -inf
    bool bad = false;
    for (unsigned row = blockIdx.x * plan::kPass1Waves + warp; row < plane.rows; row += gridDim.x * plan::kPass1Waves) {
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
        const float yhat = ::strix::cuda_core::warp_sum_xor(acc);
        const float e = __fadd_rn(
            __fmul_rn(__fmul_rn(nx, __fadd_rn(plane.residual[row], __fmul_rn(bound.gamma, plane.amplitude[row]))),
                      bound.margin),
            bound.abs_slack);
        // Outward-rounded interval: L <= yhat - e (round down), U >= yhat + e (round up), both as real numbers.
        const float hi = __fadd_ru(yhat, e), lo = __fadd_rd(yhat, -e);
        if (lane == 0U) upper[row] = hi;
        if (!(isfinite(hi) && isfinite(lo))) bad = true; else lmax = fmaxf(lmax, lo);
    }
    if (lane == 0U) {
        if (bad) atomicOr(&state->flag, 1U);
        if (lmax > __int_as_float(static_cast<int>(0xff800000U))) atomicMax(&state->lkey, order_key(lmax));
    }
}

__global__ void __launch_bounds__(plan::kCompactThreads) compact_kernel(const float* upper, unsigned rows, float cap,
                                                                        unsigned capacity, State* state,
                                                                        std::uint32_t* ids) {
    if (state->flag != 0U) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) state->count = plan::kAllRows;
        return;
    }
    const float threshold = head_transform(key_value(state->lkey), cap);
    const unsigned lane = threadIdx.x & 31U;
    for (unsigned base = blockIdx.x * plan::kCompactThreads; base < rows; base += gridDim.x * plan::kCompactThreads) {
        const unsigned i = base + threadIdx.x;
        const bool hit = i < rows && head_transform(upper[i], cap) >= threshold;
        const unsigned mask = __ballot_sync(0xffffffffU, hit);
        if (mask == 0U) continue;
        const unsigned leader = static_cast<unsigned>(__ffs(static_cast<int>(mask))) - 1U;
        unsigned first = 0U;
        if (lane == leader) first = atomicAdd(&state->count, static_cast<unsigned>(__popc(mask)));
        first = __shfl_sync(0xffffffffU, first, static_cast<int>(leader));
        const unsigned slot = first + static_cast<unsigned>(__popc(mask & ((1U << lane) - 1U)));
        if (hit && slot < capacity) ids[slot] = i;
    }
}

// y already holds G(y) (stored by the mapped pass 2 like the incumbent's softcap store): fold with the shared rule.
template <class Emit>
__global__ void __launch_bounds__(256) pick_kernel(const float* y, const std::uint32_t* ids, const State* state,
                                                   unsigned rows, unsigned capacity, Emit emit) {
    const unsigned count = state->count;
    const bool all = count > capacity;
    const unsigned n = all ? rows : count;
    ha::Pick p;
    for (unsigned i = threadIdx.x; i < n; i += 256U) {
        const float v = y[i];
        const std::uint32_t id = all ? i : ids[i];
        if (!isfinite(v)) ++p.nonfinite;
        else if (ha::better(v, id, p.value, p.index)) { p.value = v; p.index = id; }
    }
    p = ::strix::cuda::core::head_argmax::detail::block_merge(p);
    if (threadIdx.x == 0U) emit(0U, p);
}

}  // namespace detail

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

// The five launches. x: FP32 [width], 16-byte aligned. scratch: scratch_layout bytes, device. emit(0, ha::Pick) runs on the
// device once (head_argmax Emit contract, the engine's adapter); the Pick's value is G(y) of the winner.
template <class Emit>
cudaError_t enqueue_screened_greedy_cuda(const PlaneView& view, const sh::Bound& bound, const HeadSpec& head,
                                         const float* x, std::uint8_t* scratch, unsigned capacity, Emit emit,
                                         cudaStream_t stream, const Grids& grids) {
    if (!plan::launch_args_valid(view, bound, head, x, scratch, capacity, grids)) return cudaErrorInvalidValue;
    const auto layout = plan::scratch_layout(head.rows, capacity);
    auto* state = reinterpret_cast<State*>(scratch);
    auto* ids = reinterpret_cast<std::uint32_t*>(scratch + layout.ids);
    auto* upper = reinterpret_cast<float*>(scratch + layout.upper);
    auto* y = reinterpret_cast<float*>(scratch + layout.y);
    detail::prep_kernel<<<1, 256, 0, stream>>>(x, head.width, bound, view.max_amplitude, state);
    const std::size_t smem = std::size_t{head.width} * sizeof(float);
    if (view.bits == 4U)
        detail::pass1_kernel<4U><<<grids.pass1, plan::kPass1Waves * 32U, smem, stream>>>(view, x, bound, state, upper);
    else
        detail::pass1_kernel<8U><<<grids.pass1, plan::kPass1Waves * 32U, smem, stream>>>(view, x, bound, state, upper);
    detail::compact_kernel<<<grids.compact, plan::kCompactThreads, 0, stream>>>(upper, head.rows, head.softcap, capacity,
                                                                                state, ids);
    if (const auto e = ::strix::cuda_core::dense_bf16_fp32_r1_mapped_epilogue(
            head.weights, x, y, ids, &state->count, capacity, head.rows, head.width,
            ::strix::cuda_core::Bf16Fp32Store::softcap, head.softcap, grids.pass2, stream);
        e != cudaSuccess)
        return e;
    detail::pick_kernel<<<1, 256, 0, stream>>>(y, ids, state, head.rows, capacity, emit);
    return cudaGetLastError();
}

// The resident screen of one BF16 softcap head: device plane, per-row norms, scratch. Built once at load from the resident
// weights (one device -> host copy of the live rows). Plan, bound and view come from core/screened_head_plan.hpp.
class ResidentScreen {
public:
    static constexpr unsigned kCapacity = plan::kCapacity;

    // Empty (usable() false) when the shape is unsupported or a weight is non-finite: the caller keeps the full head.
    ResidentScreen(const std::uint16_t* device_weights, unsigned rows, unsigned width, float softcap,
                   cudaStream_t stream, unsigned bits = 8U) {
        if (!plan::shape_supported(device_weights, rows, width, softcap, bits)) return;
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
        scratch_ = allocate(built.layout.bytes);
        bytes_ = built.device_bytes();
        view_ = plan::make_view(built, data_.get(), residual_.get(), amplitude_.get(), rows, width, bits);
        bound_ = built.bound;
        head_ = {device_weights, rows, width, softcap};
        grids_ = default_grids();
    }

    [[nodiscard]] bool usable() const noexcept { return scratch_ != nullptr; }
    [[nodiscard]] std::size_t device_bytes() const noexcept { return bytes_; }
    void set_grids(const Grids& g) noexcept { grids_ = g; }
    [[nodiscard]] const Grids& grids() const noexcept { return grids_; }

    // x: FP32 [width] on the device. Same Pick as the full head (dense_bf16_fp32_r1 softcap + head_argmax) over the same rows.
    template <class Emit>
    cudaError_t enqueue(const float* x, Emit emit, cudaStream_t stream) const {
        if (!usable()) return cudaErrorNotReady;
        return enqueue_screened_greedy_cuda(view_, bound_, head_, x, static_cast<std::uint8_t*>(scratch_.get()), kCapacity,
                                            emit, stream, grids_);
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
};

}  // namespace strix::cuda::core::head_screened
