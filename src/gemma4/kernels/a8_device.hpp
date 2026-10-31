#pragma once

// Device-side A8G32 group quantization shared by the quantize kernel and fused producers.
#include "internal.hpp"

namespace strix::gemma4::gpu {
// The calling wave holds the 32 values of one group in lane order: lane l owns element i = 32*k + l
// of the row. Same operations as a8::quantize_group (exact maximum, FP32 division, rint, clamp).
__device__ inline void a8_quantize_lane(float v, const A8Rows& q, unsigned row, unsigned i) {
    float amax = bit_cast<float>(bit_cast<unsigned>(v) & 0x7fffffffU);
    for (int d = 16; d; d >>= 1) {
        const float other = __shfl_xor(amax, d, 32);
        amax = other > amax ? other : amax;
    }
    const float delta = amax / 127.0F;
    int code = 0;
    if (delta != 0.0F) {
        float c = __builtin_rintf(v / delta);
        c = c > 127.0F ? 127.0F : (c < -127.0F ? -127.0F : c);
        code = int(c);
    }
    int sum = code;
    for (int d = 16; d; d >>= 1)
        sum += __shfl_xor(sum, d, 32);
    q.codes[std::size_t(row) * q.width + i] = std::int8_t(code);
    if (i % 32U == 0)
        q.meta[std::size_t(row) * (q.width / 32U) + i / 32U] = {delta != 0.0F ? delta : 1.0F, delta != 0.0F ? sum : 0};
}
} // namespace strix::gemma4::gpu
