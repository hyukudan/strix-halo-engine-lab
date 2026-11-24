#pragma once

#include "strix/profiles/inkling2_mini.hpp"

#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::inkling2 {

struct MiniWeightOptions {
    profile::Inputs tier;
    bool qualify_q8{};  // the Q8_0 qualification arm (--qualify-q8, the alias of --qualify-weights <Q8 policy>)
    bool disable_q8{};  // off switch of every compact weight storage
    // The shared policy of an explicit qualification arm (--qualify-weights KEY); nullptr: none.
    const weight::WeightLoadPolicy* qualify{};
};

// The bindings a qualification key may name: the model's Q8_0 binding (legacy id included), the shared low-bit mixes and
// the shared head-only low-bit rows (attention BF16).
inline constexpr std::array<weight::WeightLoadPolicy, 6> kMiniQualifiablePolicies{
    kMiniWeightLoadPolicy, weight::kAttentionHeadQ6Policy, weight::kAttentionQ4HeadQ6Policy,
    weight::kAttentionHeadQ4Policy, weight::kHeadQ6Policy, weight::kHeadQ4Policy};

// The binding a load resolves under: an explicit arm's policy (the model's own row for the Q8_0 policy, so its legacy
// records match); else the TURBO mix for --profile turbo and the Q8_0 binding for the other tiers.
inline const weight::WeightLoadPolicy& mini_weight_load_policy(const MiniWeightOptions& options, profile::Name tier) {
    if (options.qualify == nullptr)
        return tier == profile::Name::turbo ? kMiniTurboWeightPolicy : kMiniWeightLoadPolicy;
    if (weight::same_id(options.qualify->id, kMiniWeightLoadPolicy.id)) return kMiniWeightLoadPolicy;
    return *options.qualify;
}

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
        options.qualify = &weight::kAttentionHeadQ8Policy;
    } else if (native && argument == "--qualify-weights") {
        const std::string_view key(value());
        options.qualify = weight::find_weight_load_policy(key, kMiniQualifiablePolicies);
        bool bound = false;
        for (const auto& binding : kMiniQualifiablePolicies)
            bound = bound || (options.qualify != nullptr && weight::same_id(binding.id, options.qualify->id));
        if (!bound) throw std::invalid_argument("--qualify-weights: unknown or unbound weight policy " + std::string(key));
        options.qualify_q8 = weight::same_id(options.qualify->id, kMiniWeightLoadPolicy.id);
    } else if (native && argument == "--no-weight-q8") {
        options.disable_q8 = true;
    } else {
        return false;
    }
    return true;
}

// Resolved state of the W4A4 expert lever (true: multi-row prefill chunks run NVFP4 expert layers as W4A4 on CUDA).
inline bool mini_moe_w4a4(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniMoeW4a4);
    if (entry == nullptr) return false;
    if (entry->value != "0" && entry->value != "1") throw std::invalid_argument("STRIX_INK_MOE_W4A4 must be 0 or 1");
    return entry->value == "1";
}

inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution) {
    auto tier = profile::resolve(kMiniProfile, options.tier);
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    // A low-bit arm is a FAST candidate or, failing the FAST gate, a TURBO one.
    if (options.qualify != nullptr &&
        ((tier.profile != profile::Name::fast && tier.profile != profile::Name::turbo) || options.disable_q8 ||
         !native_execution))
        throw std::invalid_argument("--qualify-weights requires --profile fast or turbo and native model execution");
    static_cast<void>(mini_moe_w4a4(tier));  // refuse a malformed or unrunnable explicit switch early
    return tier;
}

inline const char* mini_profile_name(profile::Name tier) {
    return tier == profile::Name::standard ? "exact" : profile::name(tier);
}

inline weight::WeightLoadPlan resolve_mini_resident_weight_load(profile::Name tier, const MiniWeightOptions& options,
                                                              const weight::WeightLoadScope& scope) {
    // Consumers of this backend's resident routes: the HIP projections and head also read the low-bit storages.
    constexpr std::array consumers{weight::kBf16WeightStorage, weight::kQ8WeightStorage,
                                   weight::kQ4G64WeightStorage, weight::kQ6G64WeightStorage};
    const std::span<const weight::FormatId> formats(consumers.data(), scope.backend == weight::Backend::hip ? 4U : 2U);
    const auto& policy = mini_weight_load_policy(options, tier);
    const auto plan = resolve_mini_weight_load(tier, scope, formats, !options.disable_q8, options.qualify != nullptr,
                                               kMiniWeightLoadDecisions, policy);
    if (options.qualify != nullptr && (plan.storage != policy.target || plan.head_storage != policy.head_target()))
        throw std::invalid_argument("the qualified compact weights were not admitted on this backend");
    return plan;
}

} // namespace strix::inkling2
