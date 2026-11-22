#pragma once

#include "core/hc_core.hpp"
#include "glm/common/glm_math.hpp"

namespace strix::glm {

struct HcArithmetic {
    __host__ __device__ static float exp(float x) { return declared_exp(x); }
    __host__ __device__ static float sigmoid(float x) { return glm::sigmoid(x); }
};

using HcR6 = core::hc::PackedForm<HcArithmetic, core::hc::PackedInput::projected, kHcIters>;
static_assert(HcR6::streams == kHc && HcR6::mixes == kHcMix);

}  // namespace strix::glm
