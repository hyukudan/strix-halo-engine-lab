#include "cpu/core/isa.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define STRIX_CPU_X86 1
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#define STRIX_CPU_X86 1
#endif

namespace strix::cpu {
namespace {

#if defined(STRIX_CPU_X86)
void cpuid(unsigned leaf, unsigned sub, unsigned (&registers)[4]) noexcept {
#if defined(_MSC_VER)
    int values[4];
    __cpuidex(values, static_cast<int>(leaf), static_cast<int>(sub));
    for (unsigned i = 0; i < 4; ++i) registers[i] = static_cast<unsigned>(values[i]);
#else
    __cpuid_count(leaf, sub, registers[0], registers[1], registers[2], registers[3]);
#endif
}

std::uint64_t xcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned low, high;
    __asm__("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<std::uint64_t>(high) << 32) | low;
#endif
}
#endif

Caps detect() noexcept {
    Caps usable;
#if defined(STRIX_CPU_X86)
    detail::X86Features features;
    unsigned registers[4];
    cpuid(0, 0, registers);
    const unsigned max_leaf = registers[0];
    if (max_leaf >= 1) {
        cpuid(1, 0, registers);
        features.leaf1_ecx = registers[2];
        if (detail::has_os_avx(features.leaf1_ecx)) features.xcr0 = xcr0();
    }
    if (max_leaf >= 7) {
        cpuid(7, 0, registers);
        features.leaf7_ebx = registers[1];
        features.leaf7_ecx = registers[2];
        features.leaf7_edx = registers[3];
        if (registers[0] >= 1) {
            cpuid(7, 1, registers);
            features.leaf7_1_eax = registers[0];
        }
    }
    usable = detail::x86_caps(features);
#endif
#if defined(__ARM_NEON) || defined(_M_ARM64)
    usable.neon = true;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    usable.dotprod = true;
#endif
#if defined(__ARM_FEATURE_SVE)
    usable.sve = true;
#endif
    return usable;
}

}  // namespace

Caps caps() noexcept {
    static const Caps usable = detect();
    return usable;
}

const char* isa_name(Isa isa) noexcept {
    switch (isa) {
    case Isa::scalar: return "scalar";
    case Isa::avx2: return "avx2";
    case Isa::avx512: return "avx512";
    case Isa::amx: return "amx";
    case Isa::neon: return "neon";
    case Isa::sve: return "sve";
    }
    return "unknown";
}

}  // namespace strix::cpu
