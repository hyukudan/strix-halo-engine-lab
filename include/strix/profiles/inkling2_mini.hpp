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

// FAST weight mix (data): the head alone in q4_g64 from the calibrated pack, every other role BF16. It passed gate v2
// (prefill and decode) and loads only through its FAST row below; without the pack FAST keeps the BF16 image.
inline constexpr weight::WeightLoadPolicy kMiniFastWeightPolicy = weight::kHeadQ4Policy;
// TURBO weight mix (data): the shared low-bit policy --profile turbo resolves when no explicit arm is named. It loads
// only through a TURBO row of kMiniWeightLoadDecisions (measured gain, quality against BF16 stated in the row);
// without one TURBO keeps the BF16 image.
inline constexpr weight::WeightLoadPolicy kMiniTurboWeightPolicy = weight::kAttentionHeadQ4Policy;
// The resident engine is the one regime of these rows (serial CLI and served requests share its routes).
inline constexpr std::string_view kMiniResidentRegime = "resident";

// The calibrated pack both rows were measured with (strix-lowbit-calib-pack, technique both, damp 0.1, q4 for the
// attention projections and the head; docs/engine/lowbit-calibration.md). A FAST or TURBO load whose row matched reads
// it from STRIX_LOWBIT_PACK or, unset, from this file next to the checkpoint. Without the file the rows do not apply
// (BF16 image, the receipt says why); a file with other content is refused: the rows' quality is this pack's.
inline constexpr std::string_view kMiniLowBitPackFile = "inkling2-mini-q4.lowbit-pack";
inline constexpr weight::Digest256 kMiniLowBitPackIdentity =
    weight::digest_from_hex("6c2dfa3505a9e23993967006b3d2f1c44a22ba9cd4e7a9184ab141eebd5a2465");

// The resident load fixes its storage before the content class or context of any request is known, so a row covers
// every request of the resident regime. Served receipt (gfx1151, 16 slots, context 16384, default profiles):
// EXACT B=1 decode 20.81 tok/s, C8 44.15 tok/s, 8K prefill 18.95 s.
// - FAST row, q4 head: B=1 21.27 tok/s, C8 50.36 tok/s, 8K prefill 18.95 s; gate v2 PASS in prefill and decode
//   (16 held-out prose windows, 0 route flips).
// - TURBO row, q4 attention and head: B=1 24.09 tok/s, C8 52.50 tok/s, 8K prefill 19.01 s; gate v2 fails the FAST bar
//   on route flips (4.07x the null) and one / two windows. Quality stated in the row.
inline constexpr weight::Digest256 kMiniLowBitBuild =
    weight::digest_from_hex("89665a6147dd8ca5eefcf4fbc4306f56a4c9e5e35bea774943d22ea08dd60f03");
inline constexpr weight::Digest256 kMiniCheckpoint =
    weight::digest_from_hex("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1");
inline constexpr std::array<weight::WeightLoadDecision, 2> kMiniWeightLoadDecisionRows{{
    {kMiniFastWeightPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-clang-24.0", kMiniResidentRegime, kMiniLowBitBuild, kMiniCheckpoint},
     weight::digest_from_hex("8897c15c3422e4c7b8fc7d46d38aeba2c8b2367c196b20a04896092fbdbdc845"),
     weight::digest_from_hex("b9e0ba8dbf6bd1b4153621d97894980a7ff9fbb17d928fae32c2175d3633f0e3"),
     {"docs/engine/lowbit-calibration.md", "2026-11-24", 1008, 0},
     {}},
    {kMiniTurboWeightPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-clang-24.0", kMiniResidentRegime, kMiniLowBitBuild, kMiniCheckpoint},
     {},
     weight::digest_from_hex("b9e0ba8dbf6bd1b4153621d97894980a7ff9fbb17d928fae32c2175d3633f0e3"),
     {"docs/engine/lowbit-calibration.md", "2026-11-24", 1008, 0},
     "vs BF16 prefill kl=0.0064 top1=0.958 dnll=-0.0016 flips=4.07x decode kl=0.0063 top1=0.963 dnll=-0.0025 "
     "flips=4.06x pack=6c2dfa3505a9e239"},
}};
inline constexpr std::span<const weight::WeightLoadDecision> kMiniWeightLoadDecisions{kMiniWeightLoadDecisionRows};

inline weight::WeightLoadPlan resolve_mini_weight_load(
    profile::Name tier, const weight::WeightLoadScope& scope,
    std::span<const weight::FormatId> consumer_formats, bool enabled, bool qualify,
    std::span<const weight::WeightLoadDecision> decisions = kMiniWeightLoadDecisions,
    const weight::WeightLoadPolicy& policy = kMiniWeightLoadPolicy) {
    return weight::resolve_weight_load(weight::builtin_registry(), policy,
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

// The lever table must stay bound here: an unbound row is never resolved, so its switch is silently ignored.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
