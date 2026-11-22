#pragma once

#include "core/pointwise.hpp"
#include "glu.hpp"
#include "strix/weight/formats/scalar_float.hpp"

namespace strix::ops::glu {

struct Bf16SwiGluProduct {
    float* gate;
    const float* up;
    __device__ void operator()(std::size_t i) const {
        gate[i] = weight::scalar_float::bf16_round(swiglu<NativeExp, Round::none>(gate[i], up[i], no_clamp));
    }
};

}  // namespace strix::ops::glu
