#pragma once

#include "args.hpp"
#include "../common.cuh"
#include "moe_core/expert_activation.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/moe/declared_exp.hpp"
#include "strix/weight/formats/scalar_float.hpp"

namespace strix::cuda::core::chain {

namespace cc = ::strix::cuda_core;
namespace sf = ::strix::weight::scalar_float;
namespace cache = ::strix::attn::cache_rows;

namespace detail {

__device__ __forceinline__ std::size_t row_base(const TileCtx& ctx) {
    return std::size_t{ctx.row} * ctx.width;
}

template <class T>
__device__ __forceinline__ const T& descriptor(const ChainArgs& args, unsigned slot) {
    return *static_cast<const T*>(args.aux[slot]);
}

__device__ __forceinline__ void finite_flag(float value, unsigned* flag) {
    if (flag && !cache::finite(value)) atomicOr(flag, 1U);
}

}  // namespace detail

// Per-register entry point. column is the logical coordinate of the register
// supplied by the producer. This path has no CTA scheduling requirements.
template <Form F, std::uint8_t Slot>
__device__ __forceinline__ void apply_element(Step<F, Slot>, float& value, unsigned column,
                                              const ChainArgs& args, const TileCtx& ctx) {
    static_assert(chain_scope<List<Step<F, Slot>>> == Scope::element);
    using detail::descriptor;
    const auto base = detail::row_base(ctx);
    const auto i = base + column;

    if constexpr (F == Form::store_f32 || F == Form::store_bf16 || F == Form::store_f16 ||
                  F == Form::store_e4m3 || F == Form::store_iu4prep) {
        void* dst = args.aux[Slot] ? descriptor<ScalarStore>(args, Slot).data : nullptr;
        if constexpr (F == Form::store_f32) {
            if (dst) static_cast<float*>(dst)[i] = value;
        } else if constexpr (F == Form::store_bf16) {
            const auto bits = sf::bf16_encode(value);
            if (dst) static_cast<std::uint16_t*>(dst)[i] = bits;
            value = cache::decode(bits);
        } else if constexpr (F == Form::store_e4m3) {
            const auto bits = sf::e4m3_encode(value);
            if (dst) static_cast<std::uint8_t*>(dst)[i] = bits;
            value = sf::e4m3(bits);
        } else {
            const auto half = __float2half_rn(value);
            if (dst) static_cast<std::uint16_t*>(dst)[i] = __half_as_ushort(half);
            if constexpr (F == Form::store_f16) value = __half2float(half);
        }
    } else if constexpr (F == Form::residual_add || F == Form::residual_add_round_bf16) {
        value = __fadd_rn(value, static_cast<const float*>(args.aux[Slot])[i]);
        if constexpr (F == Form::residual_add_round_bf16) value = sf::bf16_round(value);
    } else if constexpr (F == Form::scale_const || F == Form::scale_layer) {
        value = __fmul_rn(value, args.scalar[Slot]);
    } else if constexpr (F == Form::route_weight_mul) {
        value = __fmul_rn(value, static_cast<const float*>(args.aux[Slot])[ctx.row]);
    } else if constexpr (F == Form::bias_add) {
        value = __fadd_rn(value, static_cast<const float*>(args.aux[Slot])[column]);
    } else if constexpr (F == Form::act_sigmoid) {
        value = cc::cc_sigmoid(value);
    } else if constexpr (F == Form::act_silu_mul) {
        value = __fmul_rn(cc::cc_silu(value), static_cast<const float*>(args.aux[Slot])[i]);
    } else if constexpr (F == Form::act_swiglu_clamp) {
        const auto& g = descriptor<Gate>(args, Slot);
        value = ::strix::moe::declared_math::swiglu_clamped(value, g.operand[i], g.limit);
    } else if constexpr (F == Form::act_swiglu_softcap) {
        const auto& g = descriptor<Softcaps>(args, Slot);
        namespace activation = ::strix::moe_core::expert_activation::detail;
        const ::strix::moe::SoftcappedSwiglu<activation::DeviceMath> act{g.gate_cap, g.up_cap};
        value = g.bf16 ? cache::decode(activation::activate_bf16(act, value, g.up[i])) : act(value, g.up[i]);
    } else if constexpr (F == Form::act_gelu_tanh_mul) {
        const float square = __fmul_rn(value, value);
        const float cube = __fmul_rn(square, value);
        const float t = __fmul_rn(0.7978845608f, __fadd_rn(value, __fmul_rn(0.044715f, cube)));
        value = __fmul_rn(__fmul_rn(__fmul_rn(0.5f, value), __fadd_rn(1.0f, tanhf(t))),
                          static_cast<const float*>(args.aux[Slot])[i]);
    } else if constexpr (F == Form::act_swiglu_oai) {
        const auto& g = descriptor<Gate>(args, Slot);
        const auto c = ::strix::moe::swiglu_clamp(value, g.operand[i], g.limit);
        value = __fmul_rn(__fmul_rn(c.gate, cc::cc_sigmoid(__fmul_rn(1.702f, c.gate))), __fadd_rn(c.up, 1.0f));
    } else if constexpr (F == Form::gate_sigmoid_mul) {
        const auto& g = descriptor<Gate>(args, Slot);
        const float operand = g.bf16 ? sf::bf16_round(g.operand[i]) : g.operand[i];
        if (g.bf16) value = sf::bf16_round(value);
        value = __fmul_rn(value, cc::cc_sigmoid(operand));
        if (g.bf16) value = sf::bf16_round(value);
    } else if constexpr (F == Form::softcap) {
        const auto& cap = descriptor<Softcap>(args, Slot);
        if (cap.bf16_input) value = sf::bf16_round(value);
        if (!cap.bf16_input || cache::finite(value))
            value = __fmul_rn(cap.cap, tanhf(__fdiv_rn(value, cap.cap)));
    } else if constexpr (F == Form::load_round_bf16) {
        value = sf::bf16_round(value);
    } else if constexpr (F == Form::mod_affine) {
        const auto& a = descriptor<Affine>(args, Slot);
        value = __fadd_rn(__fmul_rn(value, a.scale[column]), a.shift[column]);
    } else if constexpr (F == Form::bias_by_class) {
        const auto& b = descriptor<ClassBias>(args, Slot);
        value = __fadd_rn(value, b.bias[std::size_t{b.classes[ctx.row]} * ctx.width + column]);
    } else if constexpr (F == Form::finite_status) {
        auto* flag = descriptor<unsigned*>(args, Slot);
        detail::finite_flag(value, flag + ctx.row);
    } else if constexpr (F == Form::kv_append || F == Form::index_pool_append ||
                         F == Form::conv_tap || F == Form::recur_commit) {
        const auto& pending = ctx.pending[args.state[Slot]];
        if constexpr (F == Form::kv_append || F == Form::index_pool_append) {
            const auto bits = cache::encode(value);
            if (pending.nonfinite && (!cache::finite(value) || !cache::finite(bits)))
                atomicOr(pending.nonfinite, 1U);
            static_cast<std::uint16_t*>(pending.data)[column] = bits;
        } else {
            static_cast<float*>(pending.data)[column] = value;
        }
    } else if constexpr (F == Form::mix_expand) {
        const auto& m = descriptor<Mix>(args, Slot);
        const auto* mix = m.coefficients + std::size_t{ctx.row} * 24U;
        const auto start = base * 4U + column;
        float residual[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) residual[j] = m.residual[start + std::size_t{j} * ctx.width];
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            float sum = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) sum = __fadd_rn(sum, __fmul_rn(mix[8U + j * 4U + k], residual[j]));
            m.output[start + std::size_t{k} * ctx.width] = __fadd_rn(__fmul_rn(mix[4U + k], value), sum);
            if (m.pre_carry && column == 0) m.pre_carry[std::size_t{ctx.row} * 4U + k] = mix[k];
        }
    }
}

template <class... Steps>
__device__ __forceinline__ void apply_element(List<Steps...>, float& value, unsigned column,
                                              const ChainArgs& args, const TileCtx& ctx) {
    static_assert(chain_scope<List<Steps...>> == Scope::element);
    (apply_element(Steps{}, value, column, args, ctx), ...);
}

}  // namespace strix::cuda::core::chain
