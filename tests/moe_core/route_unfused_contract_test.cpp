// Proves that the unfused host route accumulator stays exact with fast contraction enabled.
// GCC x86 additionally calls from a function compiled with the FMA target feature.
#include "strix/moe/route_project.hpp"
#include <bit>
#include <cstdint>
#include <cstdio>

#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("fma"), optimize("fp-contract=fast"), noinline))
float fma_enabled_caller(float acc, float w, float x) {
    return strix::moe::RouteAccMulAdd::step(acc, w, x);
}
#endif

int main() {
    // (1 + 2^-23)^2 = 1 + 2^-22 + 2^-46; separate product rounding cancels to +0,
    // while the fused step keeps +2^-46. Volatile prevents constant folding.
    volatile float w = 0x1.000002p+0F;
    volatile float acc = -0x1.000004p+0F;
    const float unfused = strix::moe::RouteAccMulAdd::step(acc, w, w);
    const float fused = strix::moe::RouteAccFma::step(acc, w, w);
    if (std::bit_cast<std::uint32_t>(unfused) != 0U || fused != 0x1p-46F) {
        std::fprintf(stderr, "route contract expected +0 and FMA residual: %a / %a\\n", unfused, fused);
        return 1;
    }
#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("fma")) {
        const float via_fast = fma_enabled_caller(acc, w, w);
        if (std::bit_cast<std::uint32_t>(via_fast) != 0U) {
            std::fprintf(stderr, "GCC fast caller contracted the unfused route to %a\\n", via_fast);
            return 2;
        }
    }
#endif
    return 0;
}
