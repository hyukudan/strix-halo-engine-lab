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
// Exact schedule trait of the HIP prepared attention (identity: the bits of the separate BF16 round passes): the stage's
// plain BF16 rounds fold into their producers, the K/V convolutions round their reads and stores and, at prefill
// heights, the output projection rounds each completed sum in its store with the separate pass's own function
// (core::RoundStore::separate). Opt-in: off in every profile column (all cells empty, builtin 0); select it with
// STRIX_INK_PREPARED_ROUND_FOLD=1 (any profile), through the one resolver the CLI, the gate and the server share.
inline constexpr std::string_view kMiniPreparedRoundFold = "inkling2-mini-prepared-round-fold@1";
// Exact lever: the CUDA prompt flash projects the relative bias itself (cuda_core::PreparedBf16Projection) instead of
// reading the [row][head][distance] image the prepare stage produces. Same FP32 FMA chain in feature order, same BF16
// round per distance, same failure contract: bitwise the image route for every launch it admits (single prompt chunk of
// two or more rows, host positions, no null-order count). A launch it cannot admit (e.g. the device's dynamic
// shared-memory opt-in below the 1024-column projection's need) keeps the image route; real errors stay visible. CUDA only
// (HIP selects its projected route in its own binding). On by default in every tier (value cells "1", builtin "1": the
// bits are the image route's, so the run identity at the default is the legacy one; an explicit 0 selects the image route
// and is a distinct identity). Switch: STRIX_INK_ATTN_PROJECTED_BIAS=0/1.
inline constexpr std::string_view kMiniAttnProjectedBias = "inkling2-mini-attn-projected-bias@1";
inline constexpr std::array<profile::Lever, 4> kMiniLevers{{
    {kMiniMoeW4a4, "STRIX_INK_MOE_W4A4", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for the FAST gate (prefill regime, routes replayed) and an RTX prefill receipt",
     false, {}, true},
    {kMiniCombineImage, "STRIX_INK_COMBINE_IMAGE", "", {"", "", "", ""}, "0", profile::Kind::exact,
     {true, true, true, true}, {true, true, true, true}, "", "0",
     "BF16 down image + combine: output bits of the FP32 down rows (device A/B through the grouped down producer, NaN "
     "operand-position fixture); gfx1151 only",
     false, {}, true},
    {.name = kMiniPreparedRoundFold, .env = "STRIX_INK_PREPARED_ROUND_FOLD", .option = "",
     .value = {"", "", "", ""}, .builtin = "0",
     .kind = profile::Kind::exact, .ready = {true, true, true, true}, .runnable = {true, true, true, true},
     .target = "", .off = "0",
     .source = "tests/core/round_fold_hip.hip (every store: the separate round's bits) and the isolated fold off/on "
               "identity ticket of the HIP prepared attention",
     .omit_builtin_fingerprint = true},
    {kMiniAttnProjectedBias, "STRIX_INK_ATTN_PROJECTED_BIAS", "", {"1", "1", "1", "1"}, "1", profile::Kind::exact,
     {true, true, true, true}, {true, true, true, true}, "", "0",
     "relative bias projected inside the CUDA prompt flash: bitwise the image route (lever on vs off logits identity, EXACT and FAST)",
     false, {}, true},
}};

// The lever table must stay bound here: an unbound row is never resolved, so its switch is silently ignored.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
