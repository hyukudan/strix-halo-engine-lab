#pragma once
#include "hip_runtime.h"
inline unsigned short __ushort_as_half(unsigned short v) { return v; }
inline float __half2float(unsigned short h) {
    const unsigned sign = unsigned(h & 0x8000U) << 16;
    const unsigned e = (h >> 10) & 31U, m = h & 1023U;
    if (e == 31) return std::bit_cast<float>(sign | 0x7f800000U | (m << 13));
    const float mag = e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(1024U + m), int(e) - 25);
    return (h & 0x8000U) ? -mag : mag;
}
