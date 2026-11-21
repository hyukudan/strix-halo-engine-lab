#pragma once
// MoE activation policy: the SwiGLU clamp trait shared by every family that clamps (DS41 ds41_moe.hpp swiglu_clamped,
// GLM-5 glm_host.hpp swiglu). gate is clamped from above, up on both sides; NaN inputs pass through.
// A limit of +infinity is NONE (every comparison is false), e.g. an unclamped shared expert. The silu expression and its exp
// stay the family's math (they differ in bits between families); only the clamp is the shared trait.
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MOE_ACT_HD __host__ __device__
#else
#define STRIX_MOE_ACT_HD
#endif

namespace strix::moe {

inline constexpr float kSwigluNoClamp = std::numeric_limits<float>::infinity();

struct SwigluClamped {
    float gate;
    float up;
};
STRIX_MOE_ACT_HD inline SwigluClamped swiglu_clamp(float gate, float up, float limit) {
    return {gate > limit ? limit : gate, up < -limit ? -limit : (up > limit ? limit : up)};
}

// The sigmoid operates on the raw gate. Only the linear gate factor
// and the up projection are softcapped.
template <class Math>
struct SoftcappedSwiglu {
    float gate_cap;
    float up_cap;

    STRIX_MOE_ACT_HD float operator()(float gate, float up) const {
        const float capped_gate = gate_cap * Math::tanh(gate / gate_cap);
        const float capped_up = up_cap * Math::tanh(up / up_cap);
        const float sigmoid = 1.0F / (1.0F + Math::exp(-gate));
        return (capped_gate * sigmoid) * capped_up;
    }
};

}  // namespace strix::moe
