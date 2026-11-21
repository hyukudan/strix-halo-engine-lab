#pragma once

#include <cstdint>

namespace strix::cpu {

enum class Isa : std::uint8_t { scalar, avx2, avx512, amx, neon, sve };
struct Caps {
    bool avx2 = false, avx512f = false, avx512bw = false, avx512vnni = false, amx_int8 = false;
    bool neon = false, sve = false, dotprod = false;
    unsigned sve_bits = 0;
    bool avx_vnni = false, f16c = false;
};

Caps caps() noexcept;
const char* isa_name(Isa isa) noexcept;

struct IsaVariants {
    bool avx2 = false;
    bool avx512 = false;
};

// Select only a body provided by this kernel and usable with the given capabilities.
constexpr Isa select_isa(const Caps& usable, IsaVariants variants) noexcept {
    if (variants.avx512 && usable.avx512f) return Isa::avx512;
    if (variants.avx2 && usable.avx2) return Isa::avx2;
    return Isa::scalar;
}

namespace detail {

struct X86Features {
    unsigned leaf1_ecx = 0, leaf7_ebx = 0, leaf7_ecx = 0, leaf7_edx = 0, leaf7_1_eax = 0;
    std::uint64_t xcr0 = 0;
};

constexpr bool has_os_avx(unsigned leaf1_ecx) noexcept {
    constexpr unsigned mask = (1U << 26) | (1U << 27) | (1U << 28);  // XSAVE, OSXSAVE, AVX
    return (leaf1_ecx & mask) == mask;
}

constexpr Caps x86_caps(const X86Features& features) noexcept {
    const bool ymm = has_os_avx(features.leaf1_ecx) && (features.xcr0 & 0x06U) == 0x06U;
    const bool zmm = ymm && (features.xcr0 & 0xe0U) == 0xe0U;
    Caps usable;
    usable.avx2 = ymm && (features.leaf7_ebx & (1U << 5));
    usable.f16c = ymm && (features.leaf1_ecx & (1U << 29));
    usable.avx_vnni = usable.avx2 && (features.leaf7_1_eax & (1U << 4));
    usable.avx512f = zmm && (features.leaf7_ebx & (1U << 16));
    usable.avx512bw = usable.avx512f && (features.leaf7_ebx & (1U << 30));
    usable.avx512vnni = usable.avx512f && (features.leaf7_ecx & (1U << 11));
    usable.amx_int8 = (features.leaf7_edx & (1U << 25)) != 0;  // Hardware bit. No AMX body is selected here.
    return usable;
}

}  // namespace detail
}  // namespace strix::cpu
