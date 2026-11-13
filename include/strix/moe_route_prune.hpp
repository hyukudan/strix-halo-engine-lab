#pragma once
// RoutePrune (FAST, default off): host declaration of the shared route-pruning policy (device side:
// src/moe_core/route_prune.hpp). Governed by the profile table row "moe_route_prune"
// (STRIX_QPF2K_MOE_ROUTE_PRUNE / --prefill-moe-route-prune). Spec grammar:
//   ""  | "off" | "0"      no pruning (the default; identity bytes unchanged)
//   rel<f>[n]              keep assignments whose weight is >= f * the row's weight sum      (0 < f < 1)
//   topp<f>[n]             keep the best assignments until their mass reaches f * the sum     (0 < f < 1)
//   cap<k>[n]              keep the k best assignments of each row                           (1 <= k <= 31)
// A trailing "n" renormalises the kept weights to the row's original sum. The best assignment is always kept.
#include "strix/env_flags.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::moe_core {

struct RoutePrune {
    enum Mode : unsigned { off = 0U, relative = 1U, top_p = 2U, cap = 3U };
    unsigned mode = off;
    float value = 0.0F;
    unsigned renorm = 0U;
    constexpr bool active() const { return mode != off; }
};

inline RoutePrune parse_route_prune(std::string_view text) {
    if (text.empty() || text == "off" || text == "0") return {};
    RoutePrune p{};
    const auto take = [&](std::string_view head, unsigned mode) {
        if (text.substr(0, head.size()) != head) return false;
        text.remove_prefix(head.size());
        p.mode = mode;
        return true;
    };
    if (!take("rel", RoutePrune::relative) && !take("topp", RoutePrune::top_p) && !take("cap", RoutePrune::cap))
        throw std::invalid_argument("MoE route prune spec must be off, rel<f>[n], topp<f>[n] or cap<k>[n]");
    if (!text.empty() && text.back() == 'n') { p.renorm = 1U; text.remove_suffix(1); }
    const std::string number(text);
    char* end = nullptr;
    p.value = number.empty() ? 0.0F : std::strtof(number.c_str(), &end);
    const bool whole = !number.empty() && end == number.c_str() + number.size();
    const bool range = p.mode == RoutePrune::cap ? p.value >= 1.0F && p.value <= 31.0F && p.value == static_cast<float>(static_cast<unsigned>(p.value))
                                                 : p.value > 0.0F && p.value < 1.0F;
    if (!whole || !range)
        throw std::invalid_argument("MoE route prune spec has an invalid value");
    return p;
}

// Canonical identity tag ("" when off): the spec text itself, validated.
inline std::string route_prune_tag(std::string_view text) {
    return parse_route_prune(text).active() ? std::string(text) : std::string{};
}

// Process-wide request (restart to change it), as the other prefill drafts.
inline const std::string& route_prune_active_tag() {
    static const std::string tag = [] {
        const char* value = strix::envcfg::text("STRIX_QPF2K_MOE_ROUTE_PRUNE");
        return route_prune_tag(value ? value : "");
    }();
    return tag;
}
inline RoutePrune route_prune_active() { return parse_route_prune(route_prune_active_tag()); }

}  // namespace strix::moe_core
