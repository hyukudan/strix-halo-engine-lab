#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::cuda::core::dense::detail {

inline constexpr unsigned full_mask = 0xffffffffU;

inline void require(bool ok, const char* what) {
    // Preserve the existing error prefix for callers of the lifted implementation.
    if (!ok) throw std::invalid_argument(std::string("ds41 cuda: ") + what);
}

inline void check(cudaError_t status, const char* what, const char* file, int line) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorName(status) + " (" + cudaGetErrorString(status) + ") in " + what +
                             " at " + file + ":" + std::to_string(line));
}

// A8 of one 32-value group held one value per lane (quantize_a8): returns the code; delta and sum are uniform.
// amax ignores NaN as std::max(amax, |v|) from 0 does; the clamp keeps std::min(127, std::max(-127, q))'s comparisons.
__device__ __forceinline__ std::int32_t a8_group(float v, float& delta, std::int32_t& sum) {
    const float a = fabsf(v);
    float amax = a > 0.0F ? a : 0.0F;
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) {
        const float o = __shfl_xor_sync(full_mask, amax, d);
        amax = o > amax ? o : amax;
    }
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(v / delta);
    const float lo = -127.0F < rounded ? rounded : -127.0F;
    const auto code = static_cast<std::int32_t>(lo < 127.0F ? lo : 127.0F);
    std::int32_t s = code;
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) s += __shfl_xor_sync(full_mask, s, d);
    sum = s;
    return code;
}

}  // namespace strix::cuda::core::dense::detail
