#pragma once
// Sensitivity null (diagnostic, never a lever): move every finite nonzero FP32 value of a stream by ONE ulp. It is the
// smallest perturbation a stream can carry, so the logit distance it causes downstream is the floor of any FAST gate
// that compares against the exact path (DS41 dsnull, MiniMax mmgate, Hy4 hy4_q4). Every engine's null probe runs this
// one function (host reference = device kernel).
// seed == 0: the last mantissa bit flipped (XOR 1). seed != 0: one ulp up or down (magnitude +-1 in the bit pattern) by
// a hash of (seed, index), so seeds sample the noise. Zero and non-finite values are left untouched.
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define STRIX_NULL_HD __host__ __device__
#else
#define STRIX_NULL_HD
#endif

namespace strix::core::null_ulp {

STRIX_NULL_HD inline std::uint32_t bits_of(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}
STRIX_NULL_HD inline float float_of(std::uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// The null value of element i.
STRIX_NULL_HD inline float value(float f, std::size_t i, std::uint32_t seed) {
    const std::uint32_t b = bits_of(f);
    if ((b & 0x7FFFFFFFU) == 0U || (b & 0x7F800000U) == 0x7F800000U) return f;  // +-0, Inf, NaN
    if (seed == 0U) return float_of(b ^ 1U);
    std::uint64_t x = (std::uint64_t{seed} << 40) ^ static_cast<std::uint64_t>(i);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return float_of((x & 1U) || (b & 0x7FFFFFU) == 0U ? b + 1U : b - 1U);
}

#if defined(__HIPCC__)
template <int Unused = 0>
__global__ void flip_kernel(float* p, std::size_t n, std::uint32_t seed) {
    const std::size_t i = blockIdx.x * 256ULL + threadIdx.x;
    if (i < n) p[i] = value(p[i], i, seed);
}
// p[0, n) in place on stream s.
inline hipError_t enqueue(float* p, std::size_t n, std::uint32_t seed, hipStream_t s) {
    if (n == 0U) return hipSuccess;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(flip_kernel<0>), dim3(static_cast<unsigned>((n + 255U) / 256U)), dim3(256), 0, s, p, n, seed);
    return hipGetLastError();
}
#endif

}  // namespace strix::core::null_ulp
