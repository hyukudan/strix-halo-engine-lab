#pragma once

#include "strix/engine_profile.hpp"
#include "strix/weight/levers.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <string_view>

namespace strix::inkling2 {

// The resident binding supplies its consumer formats and complete-model receipts. The lever is the shared
// generic policy weights.attention_head.q8_0@1; records under its former model-named id still resolve to it.
inline constexpr weight::WeightLoadPolicy kMiniWeightLoadPolicy = weight::with_legacy(
    weight::kAttentionHeadQ8Policy, {"inkling2-mini-attention-head-bf16-q8_0", 1});

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

// FAST lever: BF16-rounded dense on tensor cores (one BF16 activation plane, DenseArith::mma1, instead of the
// three-plane exact mma3) for the dense projections of a multi-row prompt-prefill chunk on CUDA. Decode, verify and
// one-row calls keep mma3. The row name is its versioned identity. Pending, not runnable: the shared CUDA BF16 rows
// entry routes tall rows to the dense_big kernel for mma3 only (three planes) and refuses BF16-rounded outputs
// outside mma3, so this build has no mma1 prefill route to run; no FAST gate or gain receipt exists either.
// Off switch: STRIX_INK_DENSE_MMA1=0.
inline constexpr std::string_view kMiniDensePrefillMma1 = "inkling2-mini-dense-prefill-bf16-mma1@1";
inline constexpr std::array<profile::Lever, 1> kMiniLevers{{
    {kMiniDensePrefillMma1, "STRIX_INK_DENSE_MMA1", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {}, "1", "0",
     "waits for the shared CUDA BF16 rows mma1 tall route, then the FAST gate (prefill regime) and a prefill receipt",
     false, {}, true},
}};

inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
