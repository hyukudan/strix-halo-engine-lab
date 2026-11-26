#include "core/strixpack_cpu_impl.hpp"

#if defined(STRIX_SP_X86_SIMD) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strix::weight::sp {
namespace {

template <class S, bool Project>
void scalar(const Header& h, const std::uint8_t* bytes, std::size_t rows, std::size_t k, const float* x, float* out) {
    float levels[2U * S::levels];
    for (unsigned c = 0; c < 2U * S::levels; ++c) levels[c] = cpu_detail::level<S>(h, c);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t col = 0; col < k; col += S::elements, bytes += S::bytes)
            for (unsigned g = 0; g < S::groups; ++g) {
                const float s = scale_value(S::scale_byte(bytes, g), h.base);
                for (unsigned j = g * S::group; j < (g + 1U) * S::group; ++j) {
                    const float w = levels[S::code(bytes, j)] * s;
                    if constexpr (Project) sum = std::fma(x[col + j], w, sum);
                    else *out++ = w;
                }
            }
        if constexpr (Project) *out++ = sum;
    }
}

template <class S>
constexpr cpu_detail::Operations scalar_ops{
    [](const Header& h, const std::uint8_t* p, std::size_t rows, std::size_t k, float* out) {
        scalar<S, false>(h, p, rows, k, nullptr, out);
    }, scalar<S, true>
};

constexpr cpu_detail::Operations scalar_table[]{scalar_ops<SP2Storage>, scalar_ops<SP3Storage>, scalar_ops<SP4Storage>};

CpuIsa detect() noexcept {
#if defined(STRIX_SP_X86_SIMD)
#if defined(_MSC_VER)
    int regs[4]{};
    __cpuid(regs, 0);
    if (regs[0] < 7) return CpuIsa::scalar;
    __cpuid(regs, 1);
    constexpr int required = (1 << 12) | (1 << 27) | (1 << 28);  // FMA, OSXSAVE, AVX
    if ((regs[2] & required) != required) return CpuIsa::scalar;
    const auto xcr0 = _xgetbv(0);
    if ((xcr0 & 6U) != 6U) return CpuIsa::scalar;
    __cpuidex(regs, 7, 0);
    if ((regs[1] & (1 << 5)) == 0) return CpuIsa::scalar;
    constexpr int avx512 = (1 << 16) | (1 << 17) | (1 << 28) | (1 << 30) | static_cast<int>(1U << 31);
    if ((xcr0 & 0xe6U) == 0xe6U && (regs[1] & avx512) == avx512) return CpuIsa::avx512;
    return CpuIsa::avx2;
#else
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
            __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vl")) return CpuIsa::avx512;
        return CpuIsa::avx2;
    }
#endif
#endif
    return CpuIsa::scalar;
}

const cpu_detail::Operations* ops(unsigned bits, CpuIsa isa) noexcept {
    if (bits < 2U || bits > 4U) return nullptr;
#if defined(STRIX_SP_X86_SIMD)
    if (isa == CpuIsa::automatic) isa = selected_cpu_isa();
    if (cpu_supports(isa)) {
        if (isa == CpuIsa::avx512) return &cpu_detail::avx512[bits - 2U];
        if (isa == CpuIsa::avx2) return &cpu_detail::avx2[bits - 2U];
    }
#else
    (void)isa;
#endif
    return &scalar_table[bits - 2U];
}

}  // namespace

CpuIsa selected_cpu_isa() noexcept {
    static const auto isa = detect();
    return isa;
}

bool cpu_supports(CpuIsa isa) noexcept {
    return isa <= selected_cpu_isa();
}

DecodeRows cpu_decode(unsigned bits, CpuIsa isa) noexcept {
    const auto* p = ops(bits, isa);
    return p ? p->decode : nullptr;
}

ProjectRows cpu_project(unsigned bits, CpuIsa isa) noexcept {
    const auto* p = ops(bits, isa);
    return p ? p->project : nullptr;
}

}  // namespace strix::weight::sp
