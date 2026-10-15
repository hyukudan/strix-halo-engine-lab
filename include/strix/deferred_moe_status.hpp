#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix {

enum class DeferredMoeDomain : std::uint32_t { none = 0, legacy = 1, q4k_a8 = 2 };
struct DeferredMoeFirstError {
    std::uint32_t domain = 0;
    std::uint32_t bits = 0;
};
static_assert(sizeof(DeferredMoeFirstError) == 8);

inline void check_deferred_moe_status(const DeferredMoeFirstError& error) {
    if (error.domain == 0 && error.bits == 0) return;
    if (error.domain == static_cast<std::uint32_t>(DeferredMoeDomain::legacy) && error.bits != 0)
        throw std::invalid_argument("HIP routed experts rejected invalid routes or nonfinite weights");
    if (error.domain == static_cast<std::uint32_t>(DeferredMoeDomain::q4k_a8) && error.bits != 0)
        throw std::invalid_argument("Q4K A8 rejected status bits (route/weight=1,input=2,delta=4,pipeline=8): " + std::to_string(error.bits));
    throw std::runtime_error("deferred MoE completion record is malformed");
}

} // namespace strix
