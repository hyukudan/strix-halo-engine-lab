#pragma once
// Device scalar steps of the a8g32-v1 group rule (strix/core/a8g32.hpp is the ONE CPU reference and owns the constants):
//   amax  = max |x[j]| from +0 (NaN ignored by fmaxf),  tiny = 0 < amax < 2^-119
//   delta = amax == 0 || tiny ? 1 : amax / 127,  code = tiny ? 0 : clamp(rint(x / delta), -127, 127),  sum = sum of codes
// The group reduction (which lanes hold the 32 values and how the maximum and the sum are exchanged) is the caller's
// topology: wave32 one value per lane (quantize_group below), half-wave pairs, 16 values per lane, DPP exchanges. Every
// topology applies group_scale to the group maximum and code_of to each value, so the bits are those of the reference
// for finite inputs and are order-free (maximum and integer sum).
#include "strix/core/a8g32.hpp"

#include <hip/hip_runtime.h>

namespace strix::ops::a8 {

inline constexpr float code_max = core::a8g32::code_max;
inline constexpr float tiny_below = core::a8g32::tiny_below;
static_assert(core::a8g32::group == 32U && code_max == 127.0F && tiny_below == 0x1p-119F);

struct Scale {
    float delta;
    bool tiny;
};

__device__ __forceinline__ Scale group_scale(float amax) {
    const bool tiny = amax > 0.0F && amax < tiny_below;
    return {amax == 0.0F || tiny ? 1.0F : amax / code_max, tiny};
}

// The clamped integer code of one value.
__device__ __forceinline__ int code_of(float x, const Scale& s) {
    const float rounded = s.tiny ? 0.0F : rintf(x / s.delta);
    return static_cast<int>(fminf(code_max, fmaxf(-code_max, rounded)));
}

struct Group {
    int code;
    float delta;
    int sum;
};

// x: this lane's value; all 32 lanes of the wave call (shuffles over the wave32).
__device__ __forceinline__ Group quantize_group(float x) {
    float amax = fabsf(x);
    for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
    const Scale s = group_scale(amax);
    const int code = code_of(x, s);
    int sum = code;
    for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
    return {code, s.delta, sum};
}

}  // namespace strix::ops::a8
