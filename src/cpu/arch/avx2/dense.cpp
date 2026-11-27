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

namespace strix::cpu::detail {

void ordered_matvec_avx2(const float* weights, const float* input, std::size_t width, float* out,
                         std::size_t outputs) noexcept {
    std::size_t row = 0;
    for (; row + 4 <= outputs; row += 4) {
        const auto* w = weights + row * width;
        auto a = _mm_setzero_ps(), b = a, c = a, d = a;
        for (std::size_t k = 0; k < width; ++k) {
            const auto x = _mm_load_ss(input + k);
            a = _mm_fmadd_ss(_mm_load_ss(w + k), x, a);
            b = _mm_fmadd_ss(_mm_load_ss(w + width + k), x, b);
            c = _mm_fmadd_ss(_mm_load_ss(w + 2 * width + k), x, c);
            d = _mm_fmadd_ss(_mm_load_ss(w + 3 * width + k), x, d);
        }
        _mm_store_ss(out + row, a);
        _mm_store_ss(out + row + 1, b);
        _mm_store_ss(out + row + 2, c);
        _mm_store_ss(out + row + 3, d);
    }
    for (; row < outputs; ++row) {
        auto sum = _mm_setzero_ps();
        for (std::size_t k = 0; k < width; ++k)
            sum = _mm_fmadd_ss(_mm_load_ss(weights + row * width + k), _mm_load_ss(input + k), sum);
        _mm_store_ss(out + row, sum);
    }
}

void ordered_dot8_avx2(const float* weights, const float* inputs, std::size_t width, float* out) noexcept {
    auto sum = _mm256_setzero_ps();
    for (std::size_t k = 0; k < width; ++k)
        sum = _mm256_fmadd_ps(_mm256_broadcast_ss(weights + k), _mm256_loadu_ps(inputs + k * 8), sum);
    _mm256_storeu_ps(out, sum);
}

} // namespace strix::cpu::detail
#endif
