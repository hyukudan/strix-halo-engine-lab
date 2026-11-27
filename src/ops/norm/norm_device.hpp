#pragma once

#include "strix/device/decode_geometry.hpp"
#include "norm.hpp"
#include "norm_numeric_traits.hpp"

#include "core/pointwise.hpp"
#include "row_reduction.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <cstddef>
#include <type_traits>

namespace strix::ops::norm {

// No second store (the plain norm).
struct NoStore {
    __device__ void operator()(std::size_t, float) const {}
};

template <unsigned L, SquareForm F = SquareForm::fma>
__device__ __forceinline__ float square_partial(const float* input, unsigned width, unsigned lane) {
    float sum = 0.0F;
    for (unsigned i = lane; i < width; i += L) sum = square_step<F>(sum, input[i]);
    return sum;
}

// A second store whose image is the only one its consumer reads declares `static constexpr bool kReplacesRow = true`:
// the FP32 row store is then skipped (yr is not written and may be null); the value handed to the store is unchanged.
template <class Store>
inline constexpr bool replaces_row_store = requires { requires Store::kReplacesRow; };

// The output expression of rmsnorm_row for element i (v = the row value, r = the RMS factor, wi = the weight when the
// Traits read one), then the optional second store. Traits::numerator(v), when declared, replaces the row value in the
// output only (the RMS factor keeps the unrounded sum of squares); Traits without it emit the same code as before.
template <class Traits, class Store>
__device__ __forceinline__ void rmsnorm_emit(const typename Traits::Scale* w, typename Traits::ScaleFormat wt, float* yr,
                                             std::size_t base, const Store& store, unsigned i, float v, float r, float wi) {
    constexpr OutputForm form = output_form_of<Traits>();  // the Traits' output rounding (norm.hpp OutputForm)
    if constexpr (requires { Traits::numerator(v); }) v = Traits::numerator(v);
    float y = 0.0F;
    if constexpr (requires { Traits::finish(v * r, wt); }) y = Traits::finish(v * r, wt);
    else if constexpr (form == OutputForm::plain) y = w ? (v * r) * wi : v * r;
    else y = finish<form>(v * r, w ? wi : 1.0F, w != nullptr);
    if constexpr (!replaces_row_store<Store>) yr[i] = y;
    if constexpr (!std::is_same_v<Store, NoStore>) {
#if defined(__HIPCC__)
        asm volatile("" : "+v"(y));  // y rounded to FP32 before the consumer's conversion (no mixed-precision fold)
#else
        asm volatile("" : "+f"(y));
#endif
        store(base + i, y);
    }
}

// Register form of rmsnorm_row (n <= L * Regs): a lane's row values are all loaded before its FMA chain and its weights
// before the reduction, so one block pays about two memory round trips instead of 2 n / L dependent ones (the loop form
// waits for each load before its FMA). The FMA chain, the tree and every expression are the loop form's: the same bits.
template <class Traits, unsigned L, bool Residual, unsigned Regs, class Store>
__device__ __forceinline__ void rmsnorm_row_regs(float* s, const float* xr, float* xw, const float* rr,
                                                 const typename Traits::Scale* w, typename Traits::ScaleFormat wt, unsigned n,
                                                 float eps, float* yr, std::size_t base, const Store& store) {
    constexpr bool weighted = !requires(float x, typename Traits::ScaleFormat f) { Traits::finish(x, f); };
    float v[Regs], wv[Regs];
#pragma unroll
    for (unsigned k = 0; k < Regs; ++k) {
        const unsigned i = threadIdx.x + k * L;
        v[k] = 0.0F;
        if (i < n) {
            if constexpr (Residual) v[k] = xr[i] + rr[i];
            else v[k] = xr[i];
        }
    }
    float a = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < Regs; ++k) {
        const unsigned i = threadIdx.x + k * L;
        if (i < n) {
            if constexpr (Residual) xw[i] = v[k];
            a = square_step<square_form_of<Traits>()>(a, v[k]);
        }
    }
#pragma unroll
    for (unsigned k = 0; k < Regs; ++k) {
        const unsigned i = threadIdx.x + k * L;
        wv[k] = 1.0F;
        if constexpr (weighted) {
            if (w && i < n) wv[k] = Traits::load(w, wt, i);
        }
    }
    const float ss = Traits::template reduce<L>(s, a);
    const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
#pragma unroll
    for (unsigned k = 0; k < Regs; ++k) {
        const unsigned i = threadIdx.x + k * L;
        if (i < n) rmsnorm_emit<Traits>(w, wt, yr, base, store, i, v[k], r, wv[k]);
    }
}

// Row values a lane holds in registers in the register form (n <= L * kRowRegs).
inline constexpr unsigned kRowRegs = 32;

// One row of the norm: L lane-strided FMA chains, then the engine's LDS halving tree. Traits adapters must inline
// without changing evaluation order.
// Residual (decode launch fusion): x[i] = x[i] + r[i] is stored back before the reduction, the same FP32 add as a
// separate residual-add kernel, so x and y keep their bits and one launch per norm is saved.
// Store (decode launch fusion): store(base + i, y[i]) after the F32 store, so a consumer's input conversion (e.g. the
// prepared FP16 rows of dense_iu4) is produced here instead of by its own launch. NoStore = the plain norm.
// Rows up to L * Regs values take the register form (same bits); Regs = 0 keeps the loop form for every width (the
// identity test compares the two).
template <class Traits, unsigned L, bool Residual, class Store, unsigned Regs = kRowRegs>
__device__ __forceinline__ void rmsnorm_row_strided(float* s, const float* xr, float* xw, const float* rr,
                                                    const typename Traits::Scale* w, typename Traits::ScaleFormat wt, unsigned n,
                                                    float eps, float* yr, std::size_t base, const Store& store) {
    static_assert(Traits::reduction_order == ReductionOrder::lane_strided_fma_lds_halving);
    if constexpr (Regs > 0U) {
        if (n <= L * Regs) {
            rmsnorm_row_regs<Traits, L, Residual, Regs>(s, xr, xw, rr, w, wt, n, eps, yr, base, store);
            return;
        }
    }
    float a = 0.0F;
    if constexpr (Residual) {
        for (unsigned i = threadIdx.x; i < n; i += L) {
            const float v = xr[i] + rr[i];
            xw[i] = v;
            a = square_step<square_form_of<Traits>()>(a, v);
        }
    } else {
        a = square_partial<L, square_form_of<Traits>()>(xr, n, threadIdx.x);
    }
    const float ss = Traits::template reduce<L>(s, a);
    const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
    constexpr bool weighted = !requires(float x, typename Traits::ScaleFormat f) { Traits::finish(x, f); };
    for (unsigned i = threadIdx.x; i < n; i += L) {
        const float v = Residual ? xw[i] : xr[i];  // this thread's own store of the first loop
        float wi = 1.0F;
        if constexpr (weighted) wi = w ? Traits::load(w, wt, i) : 1.0F;
        rmsnorm_emit<Traits>(w, wt, yr, base, store, i, v, r, wi);
    }
}

// The xor butterfly of ReductionOrder::wave_chunk_serial: v = v + shfl_xor(v, d) for d = 16 .. 1 (every lane ends with the
// same bits: each level adds the same two partial sums).
__device__ __forceinline__ float wave_butterfly_sum(float v) {
    for (int d = static_cast<int>(kWaveLanes) / 2; d != 0; d /= 2) {
#if defined(__HIPCC__)
        v = v + __shfl_xor(v, d, static_cast<int>(kWaveLanes));
#else
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffU, v, d, static_cast<int>(kWaveLanes)));
#endif
    }
    return v;
}

// One row in ReductionOrder::wave_chunk_serial (norm_numeric_traits.hpp): chunk c = [c * C, c * C + C) of C =
// wave_chunk_of<Traits>() values (C = 0: the whole row), summed by wave c % (L / 32) with lane chains of
// square_form_of<Traits>() and the xor butterfly; s[c] = that sum; the row sum is s[0] + s[1] + ... in chunk order.
// Requires L a multiple of 32 (L = 0: the block size blockDim.x, a multiple of 32, read at run time) and at most as many
// chunks as s holds floats. Residual: x[i] = x[i] + r[i] is stored by the lane that sums i, before the barrier. The output
// loop is the lane-strided loop of rmsnorm_row_strided (thread t owns i = t + k L), so the factor, the output expression
// and Store are the same as the other order's.
template <class Traits, unsigned L, bool Residual, class Store>
__device__ __forceinline__ void rmsnorm_row_chunked(float* s, const float* xr, float* xw, const float* rr,
                                                    const typename Traits::Scale* w, typename Traits::ScaleFormat wt, unsigned n,
                                                    float eps, float* yr, std::size_t base, const Store& store) {
    static_assert(L % kWaveLanes == 0U);
    const unsigned lanes = L != 0U ? L : blockDim.x;
    constexpr unsigned chunk = wave_chunk_of<Traits>();
    constexpr SquareForm form = square_form_of<Traits>();
    const unsigned lane = threadIdx.x % kWaveLanes, wave = threadIdx.x / kWaveLanes;
    const unsigned span = chunk == 0U ? n : chunk, chunks = wave_chunk_count(n, chunk);
    for (unsigned c = wave; c < chunks; c += lanes / kWaveLanes) {
        const unsigned first = c * span, end = n - first < span ? n : first + span;
        float a = 0.0F;
        for (unsigned i = first + lane; i < end; i += kWaveLanes) {
            float v = 0.0F;
            if constexpr (Residual) {
                v = xr[i] + rr[i];
                xw[i] = v;
            } else {
                v = xr[i];
            }
            a = square_step<form>(a, v);
        }
        a = wave_butterfly_sum(a);
        if (lane == 0U) s[c] = a;
    }
    __syncthreads();
    float ss = s[0];
    for (unsigned c = 1; c < chunks; ++c) ss = ss + s[c];
    __syncthreads();
    const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
    constexpr bool weighted = !requires(float x, typename Traits::ScaleFormat f) { Traits::finish(x, f); };
    for (unsigned i = threadIdx.x; i < n; i += lanes) {
        const float v = Residual ? xw[i] : xr[i];  // stored before the barrier (possibly by another lane)
        float wi = 1.0F;
        if constexpr (weighted) wi = w ? Traits::load(w, wt, i) : 1.0F;
        rmsnorm_emit<Traits>(w, wt, yr, base, store, i, v, r, wi);
    }
}

// One row of the norm in Traits::reduction_order (norm.hpp ReductionOrder): the lane-strided body above (default, every
// existing binding) or the wave-chunk body.
template <class Traits, unsigned L, bool Residual, class Store, unsigned Regs = kRowRegs>
__device__ __forceinline__ void rmsnorm_row(float* s, const float* xr, float* xw, const float* rr,
                                            const typename Traits::Scale* w, typename Traits::ScaleFormat wt, unsigned n,
                                            float eps, float* yr, std::size_t base, const Store& store) {
    if constexpr (Traits::reduction_order == ReductionOrder::wave_chunk_serial)
        rmsnorm_row_chunked<Traits, L, Residual>(s, xr, xw, rr, w, wt, n, eps, yr, base, store);
    else
        rmsnorm_row_strided<Traits, L, Residual, Store, Regs>(s, xr, xw, rr, w, wt, n, eps, yr, base, store);
}

// One block per row.
template <class Traits, unsigned L>
__global__ void rmsnorm_kernel(const typename Traits::Input* x, const typename Traits::Scale* w,
                              typename Traits::ScaleFormat wt, unsigned n, float eps,
                              typename Traits::Output* y) {
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    __shared__ float s[L];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;
    rmsnorm_row<Traits, L, false>(s, x + base, nullptr, nullptr, w, wt, n, eps, y + base, base, NoStore{});
}

// The norm of the first n values of rows at a row stride (x row r at x + r * x_stride, y row r at y + r * y_stride; y may
// be x): the plain kernel's row function on the same values, without gathering the rows to a tight buffer. One block per row.
template <class Traits, unsigned L>
__global__ void rmsnorm_strided_kernel(const typename Traits::Input* x, std::size_t x_stride, const typename Traits::Scale* w,
                                       typename Traits::ScaleFormat wt, unsigned n, float eps, typename Traits::Output* y,
                                       std::size_t y_stride) {
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    __shared__ float s[L];
    const std::size_t row = blockIdx.x;
    rmsnorm_row<Traits, L, false>(s, x + row * x_stride, nullptr, nullptr, w, wt, n, eps, y + row * y_stride, row * n, NoStore{});
}

// Rows of at most 32 J values (J a power of two), one wave32 per row and kWaveRows rows per block: the bits of
// rmsnorm_strided_kernel<Traits, L> for any power-of-two L >= 32 J. Lane t holds the values of the L-lane form's lanes
// t + 32 j (one value each), so every partial is that lane's one-step chain (+0 where i >= n). The L-lane halving tree's
// steps at h >= 32 J only add those +0 partials (x + +0 = x: a partial is a sum of squares, never -0), its steps
// 16 J .. 32 pair the lane's registers (a[j] += a[j + h / 32]) and its steps 16 .. 1 are lane shuffles (s[t] = s[t] +
// s[t + h], every lane reads before any lane writes); then the factor and rmsnorm_emit per value as in the register form.
// No LDS, no barrier. HIP body (wave32 lane shuffles).
#if defined(__HIPCC__)
inline constexpr unsigned kWaveRows = 8;

template <class Traits, unsigned J>
__global__ __launch_bounds__(32U * kWaveRows) void rmsnorm_wave_rows_kernel(
    const typename Traits::Input* x, std::size_t x_stride, const typename Traits::Scale* w,
    typename Traits::ScaleFormat wt, unsigned n, unsigned rows, float eps, typename Traits::Output* y,
    std::size_t y_stride) {
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    static_assert(Traits::reduction_order == ReductionOrder::lane_strided_fma_lds_halving);
    static_assert(J > 0U && (J & (J - 1U)) == 0U);
    constexpr bool weighted = !requires(float value, typename Traits::ScaleFormat f) { Traits::finish(value, f); };
    const unsigned lane = threadIdx.x % 32U;
    const std::size_t row = std::size_t{blockIdx.x} * kWaveRows + threadIdx.x / 32U;
    if (row >= rows) return;  // whole waves only: no barrier follows
    const float* xr = x + row * x_stride;
    float v[J], a[J], wv[J];
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = lane + 32U * j;
        v[j] = i < n ? xr[i] : 0.0F;
    }
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = lane + 32U * j;
        a[j] = i < n ? square_step<square_form_of<Traits>()>(0.0F, v[j]) : 0.0F;
        wv[j] = 1.0F;
        if constexpr (weighted) {
            if (w && i < n) wv[j] = Traits::load(w, wt, i);
        }
    }
#pragma unroll
    for (unsigned h = J / 2U; h > 0U; h >>= 1U) {
#pragma unroll
        for (unsigned j = 0; j < h; ++j) a[j] = a[j] + a[j + h];
    }
    float sum = a[0];
#pragma unroll
    for (unsigned h = 16U; h > 0U; h >>= 1U) {
        const float upper = __shfl_down(sum, h, 32);
        if (lane < h) sum = sum + upper;
    }
    const float ss = __shfl(sum, 0, 32);
    const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
    float* yr = y + row * y_stride;
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = lane + 32U * j;
        if (i < n) rmsnorm_emit<Traits>(w, wt, yr, row * n, NoStore{}, i, v[j], r, wv[j]);
    }
}
#endif

// x += r (Residual), then the norm of x into y and store(row * n + i, y) per element. One block per row.
template <class Traits, unsigned L, bool Residual, class Store>
__global__ void rmsnorm_fused_kernel(float* x, const float* r, const typename Traits::Scale* w,
                                     typename Traits::ScaleFormat wt, unsigned n, float eps, float* y, Store store) {
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    __shared__ float s[L];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;
    rmsnorm_row<Traits, L, Residual>(s, x + base, x + base, Residual ? r + base : nullptr, w, wt, n, eps, y + base, base,
                                     store);
}

struct LogPositionScale {
    unsigned first = 0;
    unsigned floor = 0;  // Zero disables the positional scale.
    float alpha = 0.0F;
};

struct Bf16HeadClipArgs {
    const float* input;
    const std::uint16_t* gain;  // [group][width], optional
    float* output;
    unsigned rows;
    unsigned groups;
    unsigned width;
    float epsilon;
    float threshold;
    LogPositionScale position;
};

// Use the shared LDS reduction with an L2 clip epilogue on BF16-rounded inputs. The
// optional per-head gain follows the BF16 projection boundary and precedes
// clipping. The clip output rounds to BF16 before optional position scaling.
template <unsigned Threads, class RowBinding = std::nullptr_t>
__device__ __forceinline__ void clip_body(Bf16HeadClipArgs args, float (&reductions)[Threads],
                                          [[maybe_unused]] RowBinding binding = nullptr) {
    const auto group = blockIdx.x;
    const auto lane = threadIdx.x;
    for (unsigned row = blockIdx.y; row < args.rows; row += gridDim.y) {
        if constexpr (!std::is_same_v<RowBinding, std::nullptr_t>) {
            if (!binding.valid(row)) continue;
        }
        const auto base = (std::size_t{row} * args.groups + group) * args.width;
        float value = 0.0F;
        if (lane < args.width) {
            const auto gain_offset = std::size_t{group} * args.width + lane;
            const float gain = args.gain
                ? weight::scalar_float::u2f(std::uint32_t{args.gain[gain_offset]} << 16U) : 1.0F;
            value = __fmul_rn(weight::scalar_float::bf16_round(args.input[base + lane]), gain);
        }
        const float square = lds_halving_sum<Threads>(reductions, __fmul_rn(value, value));
        const float scale = fminf(1.0F, __fdiv_rn(args.threshold, __fadd_rn(__fsqrt_rn(square), args.epsilon)));
        if (lane < args.width) {
            float clipped = weight::scalar_float::bf16_round(__fmul_rn(value, scale));
            const unsigned absolute = [&] {
                if constexpr (std::is_same_v<RowBinding, std::nullptr_t>) return args.position.first + row;
                else return binding.position(row);
            }();
            if (args.position.floor != 0 && absolute >= args.position.floor) {
                const float ratio = __fdiv_rn(static_cast<float>(absolute + 1U), static_cast<float>(args.position.floor));
                const float log_scale = __fadd_rn(1.0F, __fmul_rn(args.position.alpha, logf(ratio)));
                clipped = weight::scalar_float::bf16_round(__fmul_rn(clipped, log_scale));
            }
            args.output[base + lane] = clipped;
        }
    }
}

template <unsigned Threads>
__global__ void clip_kernel(Bf16HeadClipArgs args) {
    __shared__ float reductions[Threads];
    clip_body(args, reductions);
}

template <unsigned Threads>
__global__ void clip_step_kernel(Bf16HeadClipArgs args, const std::uint32_t* position) {
    __shared__ float reductions[Threads];
    args.position.first = *position;
    clip_body(args, reductions);
}

// Row r's absolute position is positions[r * stride] (one query row per lane of a multi-request step, each at
// its own position): the clip of row r is exactly the one-row clip at first = positions[r * stride], because
// clip_body adds the row index to `first` with wrapping unsigned arithmetic and gridDim.y == rows here.
template <unsigned Threads>
__global__ void clip_rows_kernel(Bf16HeadClipArgs args, const std::uint32_t* positions, unsigned stride) {
    __shared__ float reductions[Threads];
    args.position.first = positions[std::size_t{blockIdx.y} * stride] - blockIdx.y;
    clip_body(args, reductions);
}

// Contiguous rows. In-place operation is safe when input and output match exactly.
inline core::pointwise::Status l2_clip_bf16(const Bf16HeadClipArgs& args, core::pointwise::Stream stream,
                                           const std::uint32_t* device_position = nullptr) {
    constexpr unsigned max_rows = std::numeric_limits<unsigned>::max() - 65535U;
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.output || args.rows == 0 || args.rows > max_rows || args.groups == 0 ||
        args.groups > 0x7FFFFFFFU || args.width == 0 || args.width > 256 ||
        args.groups > max_floats / args.rows / args.width ||
        !(args.epsilon > 0.0F) || !std::isfinite(args.epsilon) ||
        !(args.threshold > 0.0F) || !std::isfinite(args.threshold) ||
        (args.position.floor != 0 && (!std::isfinite(args.position.alpha) ||
                                     args.rows > std::numeric_limits<unsigned>::max() - args.position.first))) {
        return core::pointwise::invalid;
    }
    const device::LaunchGeometry geometry = device::decode_geometry::head_clip(args.groups, args.rows, args.width);
    const auto grid = geometry.grid<dim3>();
    const auto launch = [&]<unsigned Threads> {
        if (device_position) clip_step_kernel<Threads><<<grid, Threads, 0, stream>>>(args, device_position);
        else clip_kernel<Threads><<<grid, Threads, 0, stream>>>(args);
    };
    if (geometry.threads[0] == 32U) launch.template operator()<32>();
    else if (geometry.threads[0] == 128U) launch.template operator()<128>();
    else launch.template operator()<256>();
    return core::pointwise::last_error();
}

// One row per lane with a device position per row (positions[r * stride], e.g. a StepRow column): the bits of
// l2_clip_bf16 on row r alone at first = that position. args.position.first is ignored. At most 65535 rows.
inline core::pointwise::Status l2_clip_bf16_rows(const Bf16HeadClipArgs& args, core::pointwise::Stream stream,
                                                const std::uint32_t* positions, unsigned stride) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.output || args.rows == 0 || args.rows > 65535U || args.groups == 0 ||
        args.groups > 0x7FFFFFFFU || args.width == 0 || args.width > 256 ||
        args.groups > max_floats / args.rows / args.width ||
        !(args.epsilon > 0.0F) || !std::isfinite(args.epsilon) ||
        !(args.threshold > 0.0F) || !std::isfinite(args.threshold) ||
        (args.position.floor != 0 && (!std::isfinite(args.position.alpha) || !positions))) {
        return core::pointwise::invalid;
    }
    if (args.position.floor == 0) return l2_clip_bf16(args, stream);  // position-free: the plain row clip
    const device::LaunchGeometry geometry = device::decode_geometry::head_clip(args.groups, args.rows, args.width);
    const auto grid = geometry.grid<dim3>();
    const auto launch = [&]<unsigned Threads> {
        clip_rows_kernel<Threads><<<grid, Threads, 0, stream>>>(args, positions, stride);
    };
    if (geometry.threads[0] == 32U) launch.template operator()<32>();
    else if (geometry.threads[0] == 128U) launch.template operator()<128>();
    else launch.template operator()<256>();
    return core::pointwise::last_error();
}

}  // namespace strix::ops::norm
