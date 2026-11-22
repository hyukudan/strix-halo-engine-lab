#pragma once

#include "strix/profiles/inkling2_mini.hpp"

namespace strix::inkling2 {

struct MiniWeightOptions {
    profile::Inputs tier;
    bool qualify_q8{};
    bool disable_q8{};
};

template <class Value>
bool parse_mini_weight_option(MiniWeightOptions& options, std::string_view argument, const Value& value,
                              bool native) {
    if (argument == "--profile") {
        const std::string_view tier(value());
        if (tier != "exact" && tier != "fast" && tier != "turbo")
            throw std::invalid_argument("--profile must be exact, fast or turbo");
        options.tier.profile = profile::parse(tier);
    } else if (native && argument == "--qualify-q8") {
        options.qualify_q8 = true;
    } else if (native && argument == "--no-weight-q8") {
        options.disable_q8 = true;
    } else {
        return false;
    }
    return true;
}

inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution) {
    auto tier = profile::resolve(kMiniProfile, options.tier);
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    return tier;
}

inline const char* mini_profile_name(profile::Name tier) {
    return tier == profile::Name::standard ? "exact" : profile::name(tier);
}

inline weight::WeightLoadPlan resolve_mini_resident_weight_load(profile::Name tier, const MiniWeightOptions& options,
                                                              const weight::WeightLoadScope& scope) {
    constexpr std::array consumers{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    const auto plan = resolve_mini_weight_load(tier, scope, consumers, !options.disable_q8, options.qualify_q8);
    if (options.qualify_q8 && plan.storage != weight::kQ8WeightStorage)
        throw std::invalid_argument("native Q8_0 weights were not admitted");
    return plan;
}

} // namespace strix::inkling2
