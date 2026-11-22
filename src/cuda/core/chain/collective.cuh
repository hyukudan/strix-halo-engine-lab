#pragma once

#include "element.cuh"
#include "../norms/row_math.cuh"
#include "cuda/core/head_argmax.cuh"
#include "ops/norm/row_reduction.hpp"
#include <math_constants.h>

namespace strix::cuda::core::chain {

inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kMaxRegisters = 32;

// A row is striped across exactly 256 x-lanes. For a full row, all lanes
// participate, including lanes with no live values. Padding registers are +0.
template <unsigned Registers>
struct RegisterRow {
    static_assert(Registers >= 1 && Registers <= kMaxRegisters);
    float values[Registers]{};
};

namespace detail {

namespace nr = ::strix::ops::norm;
namespace head = ::strix::cuda::core::head_argmax;
namespace pick = ::strix::head_argmax;

__device__ __forceinline__ unsigned column(unsigned reg) { return threadIdx.x + kThreads * reg; }

template <unsigned R>
__device__ __forceinline__ float squared(const RegisterRow<R>& row, const TileCtx& ctx, RmsOrder order,
                                         float mean = 0.0f, bool centered = false) {
    float sum = 0.0f;
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const float x = centered ? __fsub_rn(row.values[r], mean) : row.values[r];
        if (column(r) < ctx.width) {
            sum = order == RmsOrder::bank_halving ? __fmaf_rn(x, x, sum) : __fadd_rn(sum, __fmul_rn(x, x));
        }
    }
    return order == RmsOrder::bank_halving ? nr::lds_halving_sum<kThreads>(ctx.scratch, sum)
                                         : cc::block_sum256(sum, ctx.scratch);
}

template <Form F, unsigned R>
__device__ __forceinline__ void normalize(RegisterRow<R>& row, const TileCtx& ctx, const Norm& n) {
    float mean = 0.0f;
    constexpr bool centered = F == Form::norm_layer || F == Form::norm_group;
    const auto width = static_cast<float>(ctx.width);
    if constexpr (centered) {
        float sum = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
            if (column(r) < ctx.width) sum = __fadd_rn(sum, row.values[r]);
        mean = __fdiv_rn(cc::block_sum256(sum, ctx.scratch), width);
    }
    const auto ss = squared(row, ctx, n.order, mean, centered);
    const float inv = STRIX_CUDA_RMS_INVERSE_RN(ss, width, n.epsilon);
    if (n.inverse_out && threadIdx.x == 0) n.inverse_out[ctx.row] = inv;
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const auto i = column(r);
        if (i >= ctx.width) continue;
        const auto group_i = std::size_t{ctx.group} * ctx.width + i;
        float x = centered ? __fsub_rn(row.values[r], mean) : row.values[r];
        x = __fmul_rn(x, inv);
        if (n.bf16_gain) x = __fmul_rn(x, cache::decode(n.bf16_gain[group_i]));
        else if (n.gain) x = __fmul_rn(x, n.gain[F == Form::norm_group ? group_i : i]);
        if (n.bias) x = __fadd_rn(x, n.bias[F == Form::norm_group ? group_i : i]);
        if (n.bf16_output)
            x = n.order == RmsOrder::cuda_warp ? __bfloat162float(__float2bfloat16_rn(x)) : sf::bf16_round(x);
        row.values[r] = x;
    }
}

template <unsigned R>
__device__ __forceinline__ void clip_l2(RegisterRow<R>& row, const TileCtx& ctx, const Clip& clip) {
    const auto lane = threadIdx.x;
    float x = 0.0f;
    if (lane < ctx.width) {
        const auto i = std::size_t{ctx.group} * ctx.width + lane;
        const float gain = clip.gain ? cache::decode(clip.gain[i]) : 1.0f;
        x = __fmul_rn(sf::bf16_round(row.values[0]), gain);
    }
    // Extra upper levels add +0 squares and preserve the incumbent tree
    // over the live lanes for the 32-, 128- and 256-thread clip variants.
    const float square = nr::lds_halving_sum<kThreads>(ctx.scratch, __fmul_rn(x, x));
    const float factor = fminf(1.0f, __fdiv_rn(clip.threshold, __fadd_rn(__fsqrt_rn(square), clip.epsilon)));
    if (lane < ctx.width) {
        x = sf::bf16_round(__fmul_rn(x, factor));
        if (clip.position_floor != 0 && ctx.position >= clip.position_floor) {
            const float ratio = __fdiv_rn(static_cast<float>(ctx.position + 1U), static_cast<float>(clip.position_floor));
            const float scale = __fadd_rn(1.0f, __fmul_rn(clip.position_alpha, logf(ratio)));
            x = sf::bf16_round(__fmul_rn(x, scale));
        }
        row.values[0] = x;
    }
}

template <bool Inverse, unsigned R>
__device__ __forceinline__ void rotate(RegisterRow<R>& row, const TileCtx& ctx, const Rope& rope) {
    const auto offset = std::size_t{ctx.position} * rope.half;
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        if (column(r) < 2U * rope.half) ctx.scratch[column(r)] = row.values[r];
    __syncthreads();
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const auto i = column(r);
        if (i >= 2U * rope.half) continue;
        const bool upper = i >= rope.half;
        const auto j = upper ? i - rope.half : i;
        const float c = rope.cosine[offset + j], s = rope.sine[offset + j];
        row.values[r] = row_math::rotate_half(row.values[r], ctx.scratch[upper ? j : j + rope.half],
                                             c, s, upper != Inverse);
    }
    __syncthreads();
}

template <unsigned R>
__device__ __forceinline__ void hadamard(RegisterRow<R>& row, const TileCtx& ctx) {
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const auto lane = threadIdx.x;
        float x = row.values[r];
#pragma unroll
        for (unsigned d = 1; d < 32; d *= 2) {
            const float peer = __shfl_xor_sync(0xffffffffU, x, d);
            x = (lane & d) ? __fsub_rn(peer, x) : __fadd_rn(x, peer);
        }
#pragma unroll
        for (unsigned d = 32; d < 128; d *= 2) {
            ctx.scratch[lane] = x;
            __syncthreads();
            const float peer = ctx.scratch[lane ^ d];
            __syncthreads();
            x = (lane & d) ? __fsub_rn(peer, x) : __fadd_rn(x, peer);
        }
        row.values[r] = __fmul_rn(x, 0x1.6a09e6p-4f);
    }
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (unsigned d = 16; d > 0; d /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffffU, x, d));
    return x;
}

__device__ __forceinline__ float row_max(float x, float* scratch) {
    x = warp_max(x);
    if ((threadIdx.x & 31U) == 0) scratch[threadIdx.x / 32U] = x;
    __syncthreads();
    float result = scratch[0];
#pragma unroll
    for (unsigned w = 1; w < kThreads / 32U; ++w) result = fmaxf(result, scratch[w]);
    __syncthreads();
    return result;
}

template <int Limit>
__device__ __forceinline__ void quantize_lane(float x, unsigned i, const TileCtx& ctx, const Quant& q) {
    const auto base = row_base(ctx);
    const auto group = (base + i) / 32U;
    const float max = warp_max(fabsf(x));
    const bool tiny = Limit == 127 && max > 0.0f && max < 0x1p-119f;
    const float step = __fdiv_rn(max, static_cast<float>(Limit));
    const float delta = step == 0.0f || tiny ? 1.0f : step;
    const float quotient = __fdiv_rn(x, delta);
    const int code = (tiny || (Limit == 7 && isnan(quotient))) ? 0 : __float2int_rn(
        fminf(static_cast<float>(Limit), fmaxf(static_cast<float>(-Limit), quotient)));
    if (i < ctx.width) static_cast<std::int8_t*>(q.codes)[base + i] = static_cast<std::int8_t>(code);
    int sum = code;
    if (q.sums) {
#pragma unroll
        for (unsigned d = 16; d > 0; d /= 2) sum += __shfl_xor_sync(0xffffffffU, sum, d);
    }
    if (i < ctx.width && (i & 31U) == 0) {
        q.scales[group] = delta;
        if (q.sums) q.sums[group] = sum;
    }
}

template <unsigned R>
__device__ __forceinline__ void quantize_e4m3(const RegisterRow<R>& row, const TileCtx& ctx, const Quant& q) {
    float max = 0.0f;
#pragma unroll
    for (unsigned r = 0; r < R; ++r) if (column(r) < ctx.width) max = fmaxf(max, fabsf(row.values[r]));
    max = row_max(max, ctx.scratch);
    const float scale = max != 0.0f ? __fdiv_rn(max, 448.0f) : 1.0f;
    const float inverse = scale != 0.0f ? __fdiv_rn(1.0f, scale) : 0.0f;
    if (threadIdx.x == 0) q.scales[ctx.row] = scale;
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const auto i = column(r);
        if (i < ctx.width)
            static_cast<std::uint8_t*>(q.codes)[row_base(ctx) + i] = sf::e4m3_encode(__fmul_rn(row.values[r], inverse));
    }
}

template <Form F, std::uint8_t Slot>
__device__ __forceinline__ void warp_step(Step<F, Slot> step, float& value, unsigned i,
                                          const ChainArgs& args, const TileCtx& ctx) {
    if constexpr (form_scope<F> == Scope::element) {
        if (i < ctx.width) apply_element(step, value, i, args, ctx);
    } else if constexpr (F == Form::quant_a8g32 || F == Form::store_a8g32) {
        quantize_lane<127>(value, i, ctx, descriptor<Quant>(args, Slot));
    } else if constexpr (F == Form::blocksum) {
        float sum = value;
#pragma unroll
        for (unsigned d = 8; d > 0; d /= 2) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffU, sum, d));
        if (i < ctx.width && (i & 15U) == 0) descriptor<float*>(args, Slot)[(row_base(ctx) + i) / 16U] = sum;
    }
}

template <Form F, unsigned R>
__device__ __forceinline__ void emit(RegisterRow<R> row, const TileCtx& ctx, const Emit& out) {
    if constexpr (F == Form::emit_top2) {
        pick::Pick2 p;
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
            if (column(r) < ctx.width && cache::finite(row.values[r]))
                pick::push2(p, row.values[r], ctx.first_index + column(r));
        p = head::detail::block_merge(p);
        if (threadIdx.x == 0) {
            const auto i = std::size_t{ctx.row} * 2U;
            out.ids[i] = p.first.index;
            out.ids[i + 1] = p.index2;
            if (out.values) {
                out.values[i] = p.first.value;
                out.values[i + 1] = p.value2;
            }
        }
        __syncthreads();
    } else {
        const unsigned count = F == Form::emit_topn ? out.count : 1;
        for (unsigned rank = 0; rank < count; ++rank) {
            pick::Pick p;
#pragma unroll
            for (unsigned r = 0; r < R; ++r) {
                const float x = row.values[r];
                const auto i = ctx.first_index + column(r);
                if (column(r) < ctx.width && cache::finite(x) && pick::better(x, i, p.value, p.index))
                    p = {x, i, 0};
            }
            p = head::detail::block_merge(p);
            if (threadIdx.x == 0) {
                const auto i = std::size_t{ctx.row} * count + rank;
                out.ids[i] = p.index;
                if (out.values) out.values[i] = p.value;
                if constexpr (F == Form::emit_topn) ctx.scratch[0] = __uint_as_float(p.index);
            }
            __syncthreads();
            if constexpr (F == Form::emit_topn) {
                const auto selected = __float_as_uint(ctx.scratch[0]);
                __syncthreads();
#pragma unroll
                for (unsigned r = 0; r < R; ++r)
                    if (ctx.first_index + column(r) == selected) row.values[r] = -CUDART_INF_F;
            }
        }
    }
}

template <unsigned R>
__device__ __forceinline__ void emit_lse(const RegisterRow<R>& row, const TileCtx& ctx, float* out) {
    float max = -CUDART_INF_F;
#pragma unroll
    for (unsigned r = 0; r < R; ++r) if (column(r) < ctx.width) max = fmaxf(max, row.values[r]);
    max = row_max(max, ctx.scratch);
    float sum = 0.0f;
    if (cache::finite(max)) {
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
            if (column(r) < ctx.width) sum = __fadd_rn(sum, expf(__fsub_rn(row.values[r], max)));
        sum = cc::block_sum256(sum, ctx.scratch);
    }
    if (threadIdx.x == 0) out[ctx.row] = cache::finite(max) ? __fadd_rn(max, logf(sum)) : max;
}

template <Form F, std::uint8_t Slot, unsigned R>
__device__ __forceinline__ void apply_step(Step<F, Slot> step, RegisterRow<R>& row,
                                           const ChainArgs& args, const TileCtx& ctx) {
    if constexpr (form_scope<F> <= Scope::warp) {
#pragma unroll
        for (unsigned r = 0; r < R; ++r) warp_step(step, row.values[r], column(r), args, ctx);
    } else if constexpr (F == Form::norm_rms || F == Form::norm_layer || F == Form::norm_group) {
        normalize<F>(row, ctx, descriptor<Norm>(args, Slot));
    } else if constexpr (F == Form::clip_l2) {
        clip_l2(row, ctx, descriptor<Clip>(args, Slot));
    } else if constexpr (F == Form::rope || F == Form::rope_inverse) {
        rotate<F == Form::rope_inverse>(row, ctx, descriptor<Rope>(args, Slot));
    } else if constexpr (F == Form::rotate_hadamard) {
        hadamard(row, ctx);
    } else if constexpr (F == Form::quant_a4had || F == Form::quant_i8rot) {
        auto rotated = row;
        hadamard(rotated, ctx);
        const auto& q = descriptor<Quant>(args, Slot);
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
            quantize_lane<F == Form::quant_a4had ? 7 : 127>(rotated.values[r], column(r), ctx, q);
    } else if constexpr (F == Form::quant_e4m3) {
        quantize_e4m3(row, ctx, descriptor<Quant>(args, Slot));
    } else if constexpr (F == Form::emit_argmax || F == Form::emit_top2 || F == Form::emit_topn) {
        emit<F>(row, ctx, descriptor<Emit>(args, Slot));
    } else if constexpr (F == Form::emit_lse) {
        emit_lse(row, ctx, descriptor<float*>(args, Slot));
    }
}

}  // namespace detail

// Every lane participates, holding the column of its lane in one aligned
// 32-value group. Padding lanes pass +0 and do not write an output.
template <class... Steps>
__device__ __forceinline__ void apply_warp(List<Steps...>, float& value, unsigned column,
                                           const ChainArgs& args, const TileCtx& ctx) {
    static_assert(chain_scope<List<Steps...>> <= Scope::warp);
    (detail::warp_step(Steps{}, value, column, args, ctx), ...);
}

// Entry point for opened reductions and recompute prologues. A producer may
// call it on its register fragment only when it owns the required full row.
// Use apply_element for tile-local epilogues, including individual state stores.
template <unsigned R, class... Steps>
__device__ __forceinline__ void apply(List<Steps...>, RegisterRow<R>& row, const ChainArgs& args, const TileCtx& ctx) {
    static_assert(chain_scope<List<Steps...>> != Scope::unsupported);
    (detail::apply_step(Steps{}, row, args, ctx), ...);
}

}  // namespace strix::cuda::core::chain
