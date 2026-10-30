#pragma once
#include <cstdint>
#include <string_view>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace strix {
// Integer RNE also preserves infinities and never turns a NaN into an infinity.
#if defined(__HIPCC__)
__host__ __device__
#endif
inline constexpr std::uint16_t moe_route_bf16_bits(std::uint32_t bits) {
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
inline bool moe_route_bf16_environment_requested(const char* request, const char* baseline, const char* defaults) {
    return request && request[0] && request[0] != '0' &&
        !(baseline && std::string_view(baseline) == "1") && !(defaults && defaults[0] == '0');
}
// Storage identity is separate from the prefill/decode profile-pair identity.
inline constexpr std::string_view moe_prefill_storage_name(bool bf16) {
    return bf16 ? "fp16-v1+moeroute16" : "fp16-v1";
}
inline constexpr std::uint64_t moe_prefill_storage_identity(bool bf16) {
    std::uint64_t h=0xcbf29ce484222325ULL;
    for(const char c:moe_prefill_storage_name(bf16)) {h^=static_cast<unsigned char>(c);h*=0x100000001b3ULL;}
    return h;
}
} // namespace strix
