#pragma once

#include "strix/engine_profile.hpp"
#include "strix/spec/block_draft_policy.hpp"
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

// FAST lever: W4A4 experts. Multi-row prompt-prefill chunks run the NVFP4 expert layers through the shared grouped W4A4
// operand class (moe_core/grouped_w4a4.hpp, identity grouped-nvfp4-w4a4-f32-v1: activations quantised once per projection
// input by the family NVFP4 rule, sm_120 block-scaled FP4 MMA, weights used as stored). MXFP8 layers, decode, verify and
// one-row calls keep the BF16 tile. CUDA sm_120 only (HIP ignores it). Runnable, not ready: opt in with
// STRIX_INK_MOE_W4A4=1 until the FAST gate (prefill regime) and a prefill receipt exist. Off switch: STRIX_INK_MOE_W4A4=0.
inline constexpr std::string_view kMiniMoeW4a4 = "inkling2-mini-moe-w4a4@1";

// Exact lever, opt-in: the grouped MoE down projection of the HIP resident prefill stores a BF16 image of its sums and the
// combine reads it (moe_core grouped_rows combine, ImageOperand), with a sticky NaN word sending NaN sums to the FP32 sum:
// the bits of the FP32 down rows for every sum. Off in every profile column (all cells empty, builtin 0): select it with
// STRIX_INK_COMBINE_IMAGE=1 (any profile). Runs only on the qualified HIP target (mini_combine_image_arch).
inline constexpr std::string_view kMiniCombineImage = "inkling2-mini-combine-down-image@1";
inline constexpr std::array<profile::Lever, 3> kMiniLevers{{
    {kMiniMoeW4a4, "STRIX_INK_MOE_W4A4", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for the FAST gate (prefill regime, routes replayed) and an RTX prefill receipt",
     false, {}, true},
    {kMiniCombineImage, "STRIX_INK_COMBINE_IMAGE", "", {"", "", "", ""}, "0", profile::Kind::exact,
     {true, true, true, true}, {true, true, true, true}, "", "0",
     "BF16 down image + combine: output bits of the FP32 down rows (device A/B through the grouped down producer, NaN "
     "operand-position fixture); gfx1151 only",
     false, {}, true},
    spec::kBlockDraftLever,
}};

// The lever table must stay bound here: an unbound row is never resolved, so its switch is silently ignored.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
