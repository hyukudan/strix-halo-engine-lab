#include "cpu/core/dense.hpp"
#include "cpu/core/chunk16_reduce.hpp"

namespace strix::cpu {
namespace {

float chunk16_scalar(const float* weights, const float* input, std::uint32_t width) noexcept {
    return detail::chunk16_reduce(weights, input, width, [](const float* w, const float* x, float* products) {
        for (unsigned i = 0; i < 16; ++i) products[i] = w[i] * x[i];
    });
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

}  // namespace strix::cpu
