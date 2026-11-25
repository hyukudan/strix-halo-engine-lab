#pragma once

#include "prefill.hpp"
#include "strix/profiles/gemma4.hpp"

namespace strix::gemma4::prefill {

inline constexpr char check_linear_policy_lever[] = "prefill_linear_policy";

inline profile::Resolution resolve_check_profile(profile::Name name, const std::string& cli_policy, bool allow_pending,
                                                 profile::Lookup env = {}) {
    profile::Pairs cli;
    if (!cli_policy.empty()) cli.emplace_back(check_linear_policy_lever, cli_policy);
    return profile::resolve(name, profile::App::prefill_bench,
                            {.cli = std::move(cli), .assets = {}, .env = std::move(env), .allow_pending = allow_pending});
}

struct CheckLinearReceipt {
    std::string_view policy;
    const char* identity;
};

// Only WMMA consumes this policy. Its canonical text comes from the backend after setup.
inline CheckLinearReceipt check_linear_receipt(profile::Resolution& product, std::string_view backend_policy,
                                               const char* non_wmma_identity) {
    const auto applied = non_wmma_identity ? std::string_view{} : backend_policy;
    for (auto it = product.entries.begin(); it != product.entries.end(); ++it) {
        if (it->lever->name != check_linear_policy_lever) continue;
        if (applied.empty()) product.entries.erase(it);
        else it->value = applied;
        break;
    }
    const auto* applied_identity = non_wmma_identity ? non_wmma_identity
        : applied.empty() ? identity : applied == v2_policy ? v2_identity : "g4-wmma-pf-policy";
    return {applied, applied_identity};
}

} // namespace strix::gemma4::prefill
