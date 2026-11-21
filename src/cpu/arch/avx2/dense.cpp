#include "cpu/core/chunk16_reduce.hpp"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>

#if defined(_MSC_VER)
#pragma warning(disable : 4752)  // Explicit intrinsics, reached only through the ISA dispatcher.
__declspec(noinline)
#endif
float strix::cpu::detail::chunk16_avx2(const float* weights, const float* input, std::uint32_t width) noexcept {
    return chunk16_reduce(weights, input, width, [](const float* w, const float* x, float* products) {
        _mm256_storeu_ps(products, _mm256_mul_ps(_mm256_loadu_ps(w), _mm256_loadu_ps(x)));
        _mm256_storeu_ps(products + 8, _mm256_mul_ps(_mm256_loadu_ps(w + 8), _mm256_loadu_ps(x + 8)));
    });
}
#endif
