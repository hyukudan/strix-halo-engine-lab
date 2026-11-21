#include "cpu/core/dense.hpp"

#include <bit>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace strix::cpu;

void check(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

void forced_caps() {
    constexpr IsaVariants both{true, true};
    detail::X86Features features;
    constexpr unsigned leaf1 = (7U << 26) | (1U << 29);
    constexpr unsigned leaf7 = (1U << 5) | (1U << 16) | (1U << 30);
    features.leaf1_ecx = leaf1;
    features.leaf7_ebx = leaf7;
    features.leaf7_ecx = 1U << 11;
    features.leaf7_1_eax = 1U << 4;
    features.xcr0 = 0xe7;
    const auto full = detail::x86_caps(features);
    check(select_isa(full, both) == Isa::avx512 && full.avx512bw && full.avx512vnni && full.avx_vnni && full.f16c,
          "AVX-512 with OS state, optional features");
    check(select_isa(full, {true, false}) == Isa::avx2, "trait has only AVX2");
    check(select_isa(full, {}) == Isa::scalar, "trait has only scalar");
    features.leaf7_ebx = leaf7 & ~(1U << 16);
    const auto no_512f = detail::x86_caps(features);
    check(select_isa(no_512f, both) == Isa::avx2 && !no_512f.avx512bw && !no_512f.avx512vnni, "no AVX-512F");
    features.leaf7_ebx = 0;
    check(select_isa(detail::x86_caps(features), both) == Isa::scalar, "no vector feature bits");
    features.leaf7_ebx = leaf7;
    for (const auto mask : {0x20U, 0x40U, 0x80U}) {
        features.xcr0 = 0xe7U & ~mask;
        check(select_isa(detail::x86_caps(features), both) == Isa::avx2, "missing ZMM/opmask state");
        check(select_isa(detail::x86_caps(features), {false, true}) == Isa::scalar, "missing usable trait body");
    }
    for (const auto mask : {0x02U, 0x04U}) {
        features.xcr0 = 0xe7U & ~mask;
        const auto usable = detail::x86_caps(features);
        check(select_isa(usable, both) == Isa::scalar && !usable.avx_vnni && !usable.f16c, "missing XMM/YMM state");
    }
    features.xcr0 = 0xe7;
    for (const auto bit : {26U, 27U, 28U}) {
        features.leaf1_ecx = leaf1 & ~(1U << bit);
        check(select_isa(detail::x86_caps(features), both) == Isa::scalar, "missing CPUID AVX/XSAVE/OSXSAVE bit");
    }
}

void native_bodies() {
    const auto scalar = chunk16_kernel(Isa::scalar);
    check(scalar.isa == Isa::scalar, "scalar cap");
    std::mt19937 rng(127);
    std::vector<float> input(1030), weight(input.size());
    for (auto* values : {&input, &weight}) {
        for (auto& value : *values) {
            // Finite arbitrary significands, signs and exponents, including subnormals.
            const auto bits = static_cast<std::uint32_t>(rng());
            value = std::bit_cast<float>(bits & 0xbeffffffU);
        }
    }
    input[513] = -0.0F;
    for (const auto ceiling : {Isa::avx2, Isa::avx512}) {
        const auto kernel = chunk16_kernel(ceiling);
#if defined(_M_X64) || defined(__x86_64__)
        const auto usable = caps();
        const auto expected = usable.avx2 ? (usable.avx512f && ceiling == Isa::avx512 ? Isa::avx512 : Isa::avx2) : Isa::scalar;
        check(kernel.isa == expected, "native selector obeys capability and cap");
#endif
        std::printf("chunk16 cap %s executes %s\n", isa_name(ceiling), isa_name(kernel.isa));
        for (std::uint32_t width = 0; width <= 1025; ++width) {
            const auto want = scalar.run(weight.data() + 1, input.data() + 1, width);
            const auto got = kernel.run(weight.data() + 1, input.data() + 1, width);
            check(std::bit_cast<std::uint32_t>(want) == std::bit_cast<std::uint32_t>(got), "chunk16 bitwise agreement");
        }
    }
}
}  // namespace

int main() {
    try {
        forced_caps();
        native_bodies();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
