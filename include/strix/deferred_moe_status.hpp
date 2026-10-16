#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix {

enum class DeferredMoeDomain : std::uint32_t { none = 0, legacy = 1, q4k_a8 = 2, qsa_selection = 3, qsa_attention = 4 };
struct DeferredMoeFirstError {
    std::uint32_t domain = 0;
    std::uint32_t bits = 0;
};
static_assert(sizeof(DeferredMoeFirstError) == 8);

// The caller serializes publishers in one stream. Preserve the complete first
// nonzero status; a successful layer or a later error cannot overwrite it.
#if defined(__HIPCC__)
__host__ __device__
#endif
inline void merge_deferred_moe_first_error(DeferredMoeFirstError& record,
    std::uint32_t local_bits, DeferredMoeDomain domain) noexcept {
    if(local_bits != 0U && record.domain == 0U) {
        record.bits = local_bits;
        record.domain = static_cast<std::uint32_t>(domain);
    }
}

inline void check_deferred_moe_status(const DeferredMoeFirstError& error) {
    if (error.domain == 0 && error.bits == 0) return;
    if (error.domain == static_cast<std::uint32_t>(DeferredMoeDomain::legacy) && error.bits != 0)
        throw std::invalid_argument("HIP routed experts rejected invalid routes or nonfinite weights");
    if (error.domain == static_cast<std::uint32_t>(DeferredMoeDomain::q4k_a8) && error.bits != 0)
        throw std::invalid_argument("Q4K A8 rejected status bits (route/weight=1,input=2,delta=4,pipeline=8): " + std::to_string(error.bits));
    if (error.domain == static_cast<std::uint32_t>(DeferredMoeDomain::qsa_selection) && error.bits != 0)
        throw std::invalid_argument("hierarchical QSA selection rejected nonfinite scores");
    if (error.domain == static_cast<std::uint32_t>(DeferredMoeDomain::qsa_attention) && error.bits != 0)
        throw std::invalid_argument("QSA grouped attention rejected invalid selection or nonfinite arithmetic");
    throw std::runtime_error("deferred MoE completion record is malformed");
}

} // namespace strix
