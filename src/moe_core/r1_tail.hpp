#pragma once

// Tail policies of the microscaled R1 down-combine kernel (B8 "combine in the epilogue", MoE tail fold).
//
// The R1 down kernel already applies the route weights and adds the experts in ascending order. A tail continues from
// the stored combined value of one hidden channel inside the CTA that owns the channel, so a per-channel consumer of the
// MoE output needs no node of its own. Each tail repeats the FP32 operations of the nodes it replaces, in their order,
// on the same operands:
//   NoTail            nothing (the incumbent kernel).
//   ResidualTail      sum = [bf16_round] (residual + moe)                      == core::residual_mix add node.
//   ConvResidualTail  conv = [bf16_round] ((sum_lag x_lag * w_lag + rw * x) / d), history shift, optional
//                     sum = [bf16_round] (residual + conv)                     == causal_conv decode node + add node.
// The kernel stages a tail's operands (thread 0, after its own expert dot, before the CTA barrier) and finishes each
// live channel after the incumbent store, so the stored MoE output itself is unchanged.
//
// Legality is part of the plan: a convolution reads the previous row of the same channel, which another CTA owns in a
// rows launch, so ConvResidualTail admits one routed row only. ResidualTail admits every rows count.

#include "core/short_convolution.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_R1_TAIL_HD __host__ __device__ inline
#else
#define STRIX_R1_TAIL_HD inline
#endif

namespace strix::moe_core::r1_tail {

// FP32 operations with the rounding of the replaced nodes: round-to-nearest device intrinsics (never contracted) and the
// plain host operations of the CPU reference.
namespace fp {
STRIX_R1_TAIL_HD float add(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
STRIX_R1_TAIL_HD float mul(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
STRIX_R1_TAIL_HD float div(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
STRIX_R1_TAIL_HD float maybe_round(float value, bool round) {
    return round ? weight::scalar_float::bf16_round(value) : value;
}
}  // namespace fp

struct NoTail {
    template <unsigned Columns>
    struct Staged {};
    static constexpr bool enabled = false;
    template <unsigned Columns>
    STRIX_R1_TAIL_HD Staged<Columns> stage(unsigned, unsigned, unsigned) const { return {}; }
    template <unsigned Columns>
    STRIX_R1_TAIL_HD void finish(const Staged<Columns>&, unsigned, unsigned, unsigned, float) const {}
};

// residual and sum are [row][stride] FP32 rows; sum may alias residual (in-place residual stream).
struct ResidualTail {
    const float* residual;
    float* sum;
    std::size_t stride;
    bool round;

    template <unsigned Columns>
    struct Staged {
        float residual[Columns];
    };
    static constexpr bool enabled = true;
    template <unsigned Columns>
    STRIX_R1_TAIL_HD Staged<Columns> stage(unsigned row, unsigned first, unsigned live) const {
        Staged<Columns> staged{};
        for (unsigned i = 0; i < Columns && i < live; ++i) staged.residual[i] = residual[row * stride + first + i];
        return staged;
    }
    // slot = channel - first of the staged range.
    template <unsigned Columns>
    STRIX_R1_TAIL_HD void finish(const Staged<Columns>& staged, unsigned slot, unsigned row, unsigned channel,
                                 float moe) const {
        sum[row * stride + channel] = fp::maybe_round(fp::add(staged.residual[slot], moe), round);
    }
};

// Decode-only causal short convolution of the MoE output (static weights, no dynamic groups) and an optional residual add.
// weights [lag][channel] (lag 0 = current row), history [taps - 1][channel] oldest first, exactly as
// core::causal_short_convolution. Taps is a compile-time constant equal to plan.taps, so every staged operand stays in
// registers. conv_output may be null (the convolved row is then not stored); residual null means no add (sum unused).
// The history update is the decode commit of causal_conv (shift by one, append the current input).
template <unsigned Taps>
struct ConvResidualTail {
    static_assert(Taps >= 1U);
    static constexpr unsigned kHistory = Taps - 1U;
    core::ShortConvolution plan;
    const float* weights;
    float* history;
    float* conv_output;
    const float* residual;
    float* sum;
    bool round_conv;
    bool round_sum;

    template <unsigned Columns>
    struct Staged {
        float weight[Columns][Taps];
        float history[Columns][kHistory > 0U ? kHistory : 1U];
        float residual[Columns];
    };
    static constexpr bool enabled = true;

    template <unsigned Columns>
    STRIX_R1_TAIL_HD Staged<Columns> stage(unsigned, unsigned first, unsigned live) const {
        Staged<Columns> staged{};
        const auto channels = plan.channels;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
        for (unsigned i = 0; i < Columns; ++i) {
            if (i >= live) break;
            const std::size_t channel = first + i;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
            for (unsigned lag = 0; lag < Taps; ++lag) staged.weight[i][lag] = weights[lag * channels + channel];
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
            for (unsigned h = 0; h < kHistory; ++h) staged.history[i][h] = history[h * channels + channel];
            staged.residual[i] = residual ? residual[channel] : 0.0F;
        }
        return staged;
    }

    template <unsigned Columns>
    STRIX_R1_TAIL_HD void finish(const Staged<Columns>& staged, unsigned slot, unsigned, unsigned channel,
                                 float moe) const {
        const auto channels = plan.channels;
        float total = 0.0F;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
        for (unsigned lag = 0; lag < Taps; ++lag) {
            const float sample = lag == 0U ? moe : staged.history[slot][kHistory - lag];
            total = fp::add(total, fp::mul(sample, staged.weight[slot][lag]));
        }
        const float residual_term = fp::mul(plan.residual_weight, moe);
        const float conv = fp::maybe_round(fp::div(fp::add(total, residual_term), plan.output_divisor), round_conv);
        if (conv_output) conv_output[channel] = conv;
        if constexpr (kHistory > 0U) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
            for (unsigned h = 0; h + 1U < kHistory; ++h) history[h * channels + channel] = staged.history[slot][h + 1U];
            history[std::size_t{kHistory - 1U} * channels + channel] = moe;
        }
        if (residual) sum[channel] = fp::maybe_round(fp::add(staged.residual[slot], conv), round_sum);
    }
};

// Staged operands of an optional tail (empty pack = no tail).
template <unsigned Columns, class... Tail>
struct StagedSelect {
    using type = NoTail::Staged<Columns>;
};
template <unsigned Columns, class Tail>
struct StagedSelect<Columns, Tail> {
    using type = typename Tail::template Staged<Columns>;
};
template <unsigned Columns, class... Tail>
using StagedOf = typename StagedSelect<Columns, Tail...>::type;

// Plan legality (host). rows = routed rows of the launch, hidden = the R1 hidden width.
inline bool admits(const NoTail&, unsigned, unsigned) { return true; }
inline bool admits(const ResidualTail& t, unsigned rows, unsigned hidden) {
    return rows >= 1U && t.residual && t.sum && t.stride >= hidden;
}
template <unsigned Taps>
inline bool admits(const ConvResidualTail<Taps>& t, unsigned rows, unsigned hidden) {
    const auto& p = t.plan;
    const bool disjoint_history = t.history != t.conv_output && t.history != t.sum && t.history != t.residual &&
                                  t.history != t.weights;
    return rows == 1U && core::valid_short_convolution_plan(p) && p.dynamic_groups == 0 && p.channels == hidden &&
           p.taps == Taps && t.weights && (p.taps == 1U || t.history) &&
           (p.taps == 1U || disjoint_history) && (t.residual == nullptr || t.sum != nullptr) &&
           (t.conv_output != nullptr || t.residual != nullptr);
}

}  // namespace strix::moe_core::r1_tail
