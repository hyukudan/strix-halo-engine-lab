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
#if defined(__HIPCC__)
#include "core/dense_linear.hpp"
#include "core/row_transforms.hpp"
#include "core/screened_head.hpp"
#include "strix/hip_argmax_core.hpp"
#include "strix/env_flags.hpp"
#include "strix/hip_generation_output.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::core::hip_screened_head {

inline constexpr unsigned kCompactThreads = 256U;
inline constexpr unsigned kPass1Waves = 8U;      // waves (rows in flight) per pass-1 block
inline constexpr unsigned kAllRows = 0xffffffffU;
inline constexpr unsigned kMaxWidth = 8192U;     // x is staged in LDS (width * 4 bytes <= 32 KiB)

struct State {
    float nx;
    unsigned flag, count, lkey;
};

// Device view of a built plane (core/screened_head.hpp Plane uploaded as three buffers).
struct PlaneView {
    const std::uint8_t* data;
    const float* residual;
    const float* amplitude;
    std::size_t row_bytes, scales_offset;
    float max_amplitude;
    unsigned rows, width, bits;
};

struct HeadSpec {
    const std::uint16_t* weights;  // incumbent BF16 rows [>= rows][width], 16-byte aligned
    unsigned rows, width;
    float softcap;
};

struct Grids {
    unsigned pass1 = 640U, compact = 256U, pass2 = 1024U;
};

// Scratch: State | ids[capacity] | upper[rows] | y[max(capacity, rows)], each part 256-byte aligned.
struct ScratchLayout {
    std::size_t ids, upper, y, bytes;
};
constexpr ScratchLayout scratch_layout(unsigned rows, unsigned capacity) noexcept {
    const auto up = [](std::size_t v) { return (v + 255U) / 256U * 256U; };
    ScratchLayout l{};
    l.ids = 256U;
    l.upper = l.ids + up(std::size_t{capacity} * 4U);
    l.y = l.upper + up(std::size_t{rows} * 4U);
    l.bytes = l.y + up(std::size_t{rows > capacity ? rows : capacity} * 4U);
    return l;
}

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
    if (!x || !scratch || !device_reply || !head.weights || capacity == 0U || plane.rows != head.rows ||
        plane.width != head.width || bound.width != head.width || head.width > kMaxWidth ||
        !screened_head::layout_supported(head.width, plane.bits) || head.width % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(x) % 16U != 0U || reinterpret_cast<std::uintptr_t>(head.weights) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(plane.data) % 16U != 0U || plane.row_bytes % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(scratch) % 256U != 0U || grids.pass1 == 0U || grids.compact == 0U ||
        grids.pass2 == 0U || head.rows >= (1U << 30U))
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

// Off switch of every binding (EXACT: same token and status either way). STRIX_SCREENED_HEAD=0 restores the full head.
inline bool enabled() { return env_flag_default_on("STRIX_SCREENED_HEAD"); }

// The resident screen of one BF16 softcap head: device plane, per-row norms, scratch. Built once at load from the
// resident weights (one device -> host copy of the live rows); engine-neutral. Device bytes: the plane
// (rows * plane row_bytes, 0.53 x the BF16 head at 8 bits) + 8 B per row of norms + scratch_layout(rows, capacity).
// Default 8 bits: at K 2304 the rigorous 4-bit bound keeps nearly every row as a candidate (CPU test: Gaussian rows
// 1200 of 1200, peaked cos 0.15 1175 of 1500) while 8 bits keeps 1-6, so 4 bits would only reach the fallback.
class ResidentScreen {
public:
    static constexpr unsigned kCapacity = 16384U;

    // Returns an empty screen (usable() false) when the shape is unsupported or a weight is non-finite: the caller keeps
    // the full head. softcap must be finite and > 0 (the transform the screen proves monotone).
    ResidentScreen(const std::uint16_t* device_weights, unsigned rows, unsigned width, float softcap,
                   hipStream_t stream, unsigned bits = 8U) {
        if (!device_weights || rows == 0U || width > kMaxWidth || width % 16U != 0U || !(softcap > 0.0F) ||
            !std::isfinite(softcap) || !screened_head::layout_supported(width, bits) || rows >= (1U << 30U))
            return;
        std::vector<std::uint16_t> host(std::size_t{rows} * width);
        check(hipMemcpyAsync(host.data(), device_weights, host.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost,
                             stream));
        check(hipStreamSynchronize(stream));
        const auto plane = screened_head::build_plane(host, width, rows, bits);
        host = {};
        if (!plane.usable) return;
        data_ = upload(plane.data.data(), plane.data.size());
        residual_ = upload(plane.residual.data(), plane.residual.size() * sizeof(float));
        amplitude_ = upload(plane.amplitude.data(), plane.amplitude.size() * sizeof(float));
        const auto layout = scratch_layout(rows, kCapacity);
        scratch_ = allocate(layout.bytes);
        bytes_ = plane.bytes() + layout.bytes;
        view_ = {static_cast<const std::uint8_t*>(data_.get()), static_cast<const float*>(residual_.get()),
                 static_cast<const float*>(amplitude_.get()), plane.layout.row_bytes, plane.layout.scales_offset,
                 plane.max_amplitude, rows, width, bits};
        bound_ = screened_head::make_bound(width);
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
};

}  // namespace strix::core::hip_screened_head
#endif
