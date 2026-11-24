#pragma once

#include "moe_core/grouped_w4a4.hpp"
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
    } else if (argument == profile::kIdentityVariantOption) {
        // Identity variants: one entry (profile::request_variant) for the CLI and strix-serve; admission in resolve.
        const std::string token(value());
        profile::request_variant(kMiniProfile, options.tier, token, profile::VariantOrigin::cli);
    } else if (argument == "--screened-head") {
        profile::request_variant(kMiniProfile, options.tier, argument, profile::VariantOrigin::cli);
    } else if (native && argument == "--qualify-q8") {
        options.qualify_q8 = true;
    } else if (native && argument == "--no-weight-q8") {
        options.disable_q8 = true;
    } else {
        return false;
    }
    return true;
}

// Resolved operand class of the W4A4 expert lever (multi-row prefill chunks run NVFP4 expert layers through the shared
// grouped W4A4 class on CUDA): 0 = bf16 (off), 1 = nvfp4 (nearest@1), mse = nvfp4_mse, mse-gate-up = nvfp4_mse_gate_up.
// mse / mse-gate-up are candidate values of the route-flip re-qualification, not yet gated.
inline moe_core::grouped_w4a4::Activations mini_moe_w4a4_class(const profile::Resolution& tier) {
    using A = moe_core::grouped_w4a4::Activations;
    const auto* entry = tier.find(kMiniMoeW4a4);
    if (entry == nullptr || entry->value == "0") return A::bf16;
    if (entry->value == "1") return A::nvfp4;
    if (entry->value == "mse") return A::nvfp4_mse;
    if (entry->value == "mse-gate-up") return A::nvfp4_mse_gate_up;
    throw std::invalid_argument("STRIX_INK_MOE_W4A4 must be 0, 1, mse or mse-gate-up");
}

// Resolved state of the W4A4 expert lever (true: multi-row prefill chunks run NVFP4 expert layers as W4A4 on CUDA).
inline bool mini_moe_w4a4(const profile::Resolution& tier) {
    return mini_moe_w4a4_class(tier) != moe_core::grouped_w4a4::Activations::bf16;
}

// --grouped-decode names the incumbent serving form; beside a packed variant request (--packed-decode or
// --identity-variant inkling2-mini-moe-packed, either order) it is a conflict, refused on every entry.
inline void mini_require_one_decode_form(const profile::Inputs& inputs, bool grouped_decode) {
    if (!grouped_decode) return;
    for (const auto& request : inputs.variants)
        if (request.id == kMiniPackedVariant)
            throw std::invalid_argument("--packed-decode and --grouped-decode are mutually exclusive");
}

// `backend` ("hip", "cuda"; empty on the CPU app) admits identity variants: a variant request without a backend that
// declares it is refused here, before any load.
inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution,
                                                std::string_view backend = {}) {
    auto inputs = options.tier;
    inputs.backend = std::string(backend);
    auto tier = profile::resolve(kMiniProfile, inputs);
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    static_cast<void>(mini_moe_w4a4(tier));  // refuse a malformed or unrunnable explicit switch early
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
