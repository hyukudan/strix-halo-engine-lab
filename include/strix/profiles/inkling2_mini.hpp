#pragma once

#include "strix/engine_profile.hpp"
#include "strix/weight/levers.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <string_view>

namespace strix::inkling2 {

// The resident binding supplies its consumer formats and complete-model receipts.
inline constexpr weight::WeightLoadPolicy kMiniWeightLoadPolicy{
    {"inkling2-mini-attention-head-bf16-q8_0", 1}, weight::kQ8WeightStorage,
};

// The resident engine is the one regime of these rows (serial CLI and served requests share its routes).
inline constexpr std::string_view kMiniResidentRegime = "resident";

// The existing gate passed for 8K prose. It does not cover a resident load, whose
// storage is fixed before the content class or context of any request is known.
inline constexpr std::span<const weight::WeightLoadDecision> kMiniWeightLoadDecisions{};

inline weight::WeightLoadPlan resolve_mini_weight_load(
    profile::Name tier, const weight::WeightLoadScope& scope,
    std::span<const weight::FormatId> consumer_formats, bool enabled, bool qualify,
    std::span<const weight::WeightLoadDecision> decisions = kMiniWeightLoadDecisions) {
    return weight::resolve_weight_load(weight::builtin_registry(), kMiniWeightLoadPolicy,
        {tier, scope, consumer_formats, enabled, qualify}, decisions);
}

// FAST lever: W4A4 experts. Multi-row prompt-prefill chunks run the NVFP4 expert layers through the shared grouped W4A4
// operand class (moe_core/grouped_w4a4.hpp, identity grouped-nvfp4-w4a4-f32-v1: activations quantised once per projection
// input by the family NVFP4 rule, sm_120 block-scaled FP4 MMA, weights used as stored). MXFP8 layers, decode, verify and
// one-row calls keep the BF16 tile. CUDA sm_120 only (HIP ignores it). Runnable, not ready: opt in with
// STRIX_INK_MOE_W4A4=1 until the FAST gate (prefill regime) and a prefill receipt exist. Off switch: STRIX_INK_MOE_W4A4=0.
inline constexpr std::string_view kMiniMoeW4a4 = "inkling2-mini-moe-w4a4@1";
inline constexpr std::array<profile::Lever, 1> kMiniLevers{{
    {kMiniMoeW4a4, "STRIX_INK_MOE_W4A4", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for the FAST gate (prefill regime, routes replayed) and an RTX prefill receipt",
     false, {}, true},
}};

inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
