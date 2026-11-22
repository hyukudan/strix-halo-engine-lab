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

// Resolved state of the dense prefill mma1 lever (true: multi-row prefill chunks use DenseArith::mma1). The value
// domain is 0/1; an explicit 1 is refused while no column of the row is runnable in this build.
inline bool mini_dense_prefill_mma1(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniDensePrefillMma1);
    if (entry == nullptr) return false;
    if (entry->value != "0" && entry->value != "1")
        throw std::invalid_argument("STRIX_INK_DENSE_MMA1 must be 0 or 1");
    if (entry->value == "0") return false;
    bool runnable = false;
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        runnable = runnable || entry->lever->ready[column] || entry->lever->runnable[column];
    if (!runnable)
        throw std::invalid_argument("STRIX_INK_DENSE_MMA1=1: this build has no BF16 mma1 prefill route");
    return true;
}

// Resolved state of the W4A4 expert lever (true: multi-row prefill chunks run NVFP4 expert layers as W4A4 on CUDA).
inline bool mini_moe_w4a4(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniMoeW4a4);
    if (entry == nullptr) return false;
    if (entry->value != "0" && entry->value != "1") throw std::invalid_argument("STRIX_INK_MOE_W4A4 must be 0 or 1");
    return entry->value == "1";
}

// Resolved state of the decode attention split lever (kMiniDecodeSplit).
enum class MiniDecodeSplit : unsigned char { off, automatic, every_visit };

inline MiniDecodeSplit mini_decode_split(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniDecodeSplit);
    if (entry == nullptr || entry->value == "0") return MiniDecodeSplit::off;
    if (entry->value == "auto") return MiniDecodeSplit::automatic;
    if (entry->value == "1") return MiniDecodeSplit::every_visit;
    throw std::invalid_argument(std::string(kMiniDecodeSplitEnv) + " must be 0, auto or 1");
}

inline const char* mini_decode_split_name(MiniDecodeSplit split) {
    return split == MiniDecodeSplit::off ? "off" : split == MiniDecodeSplit::automatic ? "auto" : "every-visit";
}

// The legacy switch STRIX_INKLING_DECODE_SPLIT answers for the generic one when the generic one is unset.
inline profile::Inputs with_mini_switch_aliases(profile::Inputs inputs) {
    inputs.env = [base = inputs.env](const char* key) -> const char* {
        const char* value = profile::environment(base, key);
        if (value == nullptr && std::string_view(key) == kMiniDecodeSplitEnv)
            value = profile::environment(base, kMiniDecodeSplitLegacyEnv);
        return value;
    };
    return inputs;
}

inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution) {
    auto tier = profile::resolve(kMiniProfile, with_mini_switch_aliases(options.tier));
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    static_cast<void>(mini_dense_prefill_mma1(tier));
    static_cast<void>(mini_moe_w4a4(tier));  // refuse a malformed or unrunnable explicit switch early
    static_cast<void>(mini_decode_split(tier));
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
