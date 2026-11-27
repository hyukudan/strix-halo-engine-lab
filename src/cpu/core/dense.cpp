#include "cpu/core/dense.hpp"
#include "cpu/core/chunk16_reduce.hpp"

#include <cmath>

namespace strix::cpu::detail {
void ordered_matvec_avx2(const float*, const float*, std::size_t, float*, std::size_t) noexcept;
void ordered_dot8_avx2(const float*, const float*, std::size_t, float*) noexcept;
}

namespace strix::cpu {
namespace {

float chunk16_scalar(const float* weights, const float* input, std::uint32_t width) noexcept {
    return detail::chunk16_reduce(weights, input, width, [](const float* w, const float* x, float* products) {
        for (unsigned i = 0; i < 16; ++i) products[i] = w[i] * x[i];
    });
}

void ordered_matvec_scalar(const float* weights, const float* input, std::size_t width, float* out,
                           std::size_t outputs) noexcept {
    for (std::size_t row = 0; row < outputs; ++row) {
        float sum = 0;
        for (std::size_t k = 0; k < width; ++k) sum = std::fma(weights[row * width + k], input[k], sum);
        out[row] = sum;
    }
}

void ordered_dot8_scalar(const float* weights, const float* inputs, std::size_t width, float* out) noexcept {
    for (std::size_t row = 0; row < 8; ++row) {
        float sum = 0;
        for (std::size_t k = 0; k < width; ++k) sum = std::fma(weights[k], inputs[k * 8 + row], sum);
        out[row] = sum;
    }
}

}  // namespace

Chunk16Dot chunk16_kernel(Isa maximum) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    const auto usable = caps();
    // The 512-bit TU may also use AVX2 for narrower operations.
    const auto isa = select_isa(usable, {maximum == Isa::avx2 || maximum == Isa::avx512,
                                       maximum == Isa::avx512 && usable.avx2});
    if (isa == Isa::avx512) return {isa, detail::chunk16_avx512};
    if (isa == Isa::avx2) return {isa, detail::chunk16_avx2};
#else
    (void)maximum;
#endif
    return {Isa::scalar, chunk16_scalar};
}

OrderedFma ordered_fma_kernel(bool allow_simd) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    if (allow_simd) {
        const auto usable = caps();
        if (usable.avx2 && usable.fma) return {Isa::avx2, detail::ordered_matvec_avx2, detail::ordered_dot8_avx2};
    }
#else
    (void)allow_simd;
#endif
    return {Isa::scalar, ordered_matvec_scalar, ordered_dot8_scalar};
}

}  // namespace strix::cpu
