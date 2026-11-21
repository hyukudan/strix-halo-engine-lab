#include "cpu/core/chunk16_reduce.hpp"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>

#if defined(_MSC_VER)
#pragma warning(disable : 4752)  // Explicit intrinsics, reached only through the ISA dispatcher.
__declspec(noinline)
#endif
float strix::cpu::detail::chunk16_avx512(const float* weights, const float* input, std::uint32_t width) noexcept {
    return chunk16_reduce(weights, input, width, [](const float* w, const float* x, float* products) {
        _mm512_storeu_ps(products, _mm512_mul_ps(_mm512_loadu_ps(w), _mm512_loadu_ps(x)));
    });
}
#endif
