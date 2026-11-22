#pragma once

#include "core/hc_core.hpp"
#include "glm/common/glm_math.hpp"

namespace strix::glm {

struct HcArithmetic {
    __host__ __device__ static float exp(float x) { return declared_exp(x); }
    __host__ __device__ static float sigmoid(float x) { return glm::sigmoid(x); }
    __host__ __device__ static float rstd(float ss, unsigned n, float eps) { return inv_sqrt(ss / static_cast<float>(n) + eps); }
};

using HcR6 = core::hc::PackedForm<HcArithmetic, core::hc::PackedInput::projected, kHcIters>;
static_assert(HcR6::streams == kHc && HcR6::mixes == kHcMix);
// R1b producer of the R6 input: 256 lane-strided squares, f = x * rr, 64 blocks of 256 (glm_math r1b_dot order).
using HcR1bProjection = core::hc::NormalizedProjection<256, kR1bBlock, kHc * kHidden / kR1bBlock>;
static_assert(HcR1bProjection::columns == kHc * kHidden);

}  // namespace strix::glm
