#pragma once
// Shared gated elementwise ops (one family per backend; HIP bodies in glu_device.hpp), FP32 storage:
//   swiglu:       (g, u) = clamp(gate, up, limit) (strix::moe::swiglu_clamp: gate from above, up both sides; limit
//                 +infinity = no clamp), a = R(R(silu(g)) * u); Round::none drops both R (no rounding).
//                 Form::div uses g / (1 + exp(-g)); Form::mul_recip uses g * (1 / (1 + exp(-g))).
//   sigmoid gate: y = R(R(x) * R(1 / (1 + exp(-gate)))); Round::none: y = x * (1 / (1 + exp(-gate))).
// R = BF16 RNE (ds41::round_to_bf16). exp is the binder's Exp trait; DeclaredExp = the declared exp of ds41_moe.hpp (basic
// IEEE operations only: host and device give the same bits when the binder's TU does not contract).
// Lifted from the DS41 trellis (swiglu_f32_kernel, Round::none + DeclaredExp = its former bits). Binders: DS41 (dense
// SwiGLU of the device / batch forward and the TP shared expert), and Round::bf16 bindings (clamped routed experts, unclamped dense
// and shared experts, attention output gate).
#include "strix/ds41_layer.hpp"  // round_to_bf16
#include "strix/ds41_moe.hpp"    // declared_exp

#include <cmath>  // expf (NativeExp)
#include "strix/moe/activation_policy.hpp"

#include <cstdint>
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_OPS_GLU_HD __host__ __device__
#else
#define STRIX_OPS_GLU_HD
#endif

namespace strix::ops::glu {

enum class Round : std::uint32_t { none, bf16 };
enum class Form : std::uint32_t { div, mul_recip };

inline constexpr float no_clamp = std::numeric_limits<float>::infinity();

struct DeclaredExp {
    STRIX_OPS_GLU_HD static float exp(float x) { return ds41::declared_exp(x); }
};

// Native exp, used by disjoint tensor rows that round only after SiLU * up.
struct NativeExp {
    STRIX_OPS_GLU_HD static float exp(float x) { return expf(x); }
};

template <class Exp, Round R, Form F = Form::div>
STRIX_OPS_GLU_HD inline float swiglu(float gate, float up, float limit) {
    const auto c = strix::moe::swiglu_clamp(gate, up, limit);
    if constexpr (F == Form::mul_recip) {
        const float silu = c.gate * (1.0F / (1.0F + Exp::exp(-c.gate)));
        if constexpr (R == Round::none) return silu * c.up;
        else return ds41::round_to_bf16(ds41::round_to_bf16(silu) * c.up);
    } else if constexpr (R == Round::none) {
        return (c.gate / (1.0F + Exp::exp(-c.gate))) * c.up;
    } else {
        return ds41::round_to_bf16(ds41::round_to_bf16(c.gate / (1.0F + Exp::exp(-c.gate))) * c.up);
    }
}

// A8 consumes the completed FP32 activation as a lane of a 32-value group.
// The backend supplies its group quantizer, including its lane exchange.
struct A8Epilogue {
    template <class Exp, Round R, Form F = Form::div, class Quantize>
#if defined(__HIPCC__)
    __device__ inline __attribute__((always_inline))  // = HIP's __forceinline__, without needing hip_runtime.h.
#elif defined(__CUDACC__)
    __device__ __forceinline__  // nvcc defines it on every host compiler (MSVC rejects __attribute__).
#endif
    static auto apply(float gate, float up, float limit, Quantize quantize) {
        return quantize(swiglu<Exp, R, F>(gate, up, limit));
    }
};

template <class Exp, Round R>
STRIX_OPS_GLU_HD inline float sigmoid_gate(float x, float gate) {
    const float g = 1.0F / (1.0F + Exp::exp(-gate));
    if constexpr (R == Round::none) {
        return x * g;
    } else {
        return ds41::round_to_bf16(ds41::round_to_bf16(x) * ds41::round_to_bf16(g));
    }
}

// Linear epilogue: the stored output of a Linear layer, y = R(x + bias), bias == nullptr: y = R(x); Round::none: y = x + bias
// (unit: y = x). One rounding of (dot + bias), as a BF16 Linear module stores its output. Binders: BF16-storing
// models (every Q8_0 / BF16 projection output, Engram kv with its bias, the embedding rows' BF16 storage).
template <Round R>
STRIX_OPS_GLU_HD inline float epilogue(float x, const float* bias, std::size_t column) {
    const float v = bias ? x + bias[column] : x;
    if constexpr (R == Round::none) return v;
    else return ds41::round_to_bf16(v);
}

}  // namespace strix::ops::glu
