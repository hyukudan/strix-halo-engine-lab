#pragma once

#include "strix/engine_profile.hpp"
#include "strix/weight/levers.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace strix::inkling2 {

// The resident binding supplies its consumer formats and complete-model receipts. The lever is the shared
// generic policy weights.attention_head.q8_0@1; records under its former model-named id still resolve to it.
inline constexpr weight::WeightLoadPolicy kMiniWeightLoadPolicy = weight::with_legacy(
    weight::kAttentionHeadQ8Policy, {"inkling2-mini-attention-head-bf16-q8_0", 1});

// FAST weight mix (data): the head alone in q4_g64 from the calibrated pack, every other role BF16. It loads only
// through a FAST row, and there is none: on the held-out decode gate (39 windows, prose / code / structured, 8K-64K) the
// head row passed every class and the pooled bars but failed dNLL on two prose windows, so FAST keeps the BF16 image
// and the q4 head is part of the TURBO mix only.
inline constexpr weight::WeightLoadPolicy kMiniFastWeightPolicy = weight::kHeadQ4Policy;
// TURBO weight mix (data): the shared low-bit policy --profile turbo resolves when no explicit arm is named. It loads
// only through a TURBO row of kMiniWeightLoadDecisions (measured gain, quality against BF16 stated in the row);
// without one TURBO keeps the BF16 image.
inline constexpr weight::WeightLoadPolicy kMiniTurboWeightPolicy = weight::kAttentionHeadQ4Policy;
// The regime of these rows: decode steps (one row per lane, and the rows of a verify step), CLI and server alike. A
// load that takes a row keeps the BF16 image resident and adds the low-bit copies of the row's roles; prompt-prefill
// chunks, and steps wider than the row's band (evidence max_rows), read the BF16 image.
inline constexpr std::string_view kMiniWeightRegime = weight::kDecodeRegime;

// The calibrated pack the TURBO row was measured with (strix-lowbit-calib-pack, technique both, damp 0.1, q4 for the
// attention projections and the head; docs/engine/lowbit-calibration.md). A load whose row matched reads
// it from STRIX_LOWBIT_PACK or, unset, from this file next to the checkpoint. Without the file the rows do not apply
// (BF16 image, the receipt says why); a file with other content is refused: the rows' quality is this pack's.
inline constexpr std::string_view kMiniLowBitPackFile = "inkling2-mini-q4.lowbit-pack";
inline constexpr weight::Digest256 kMiniLowBitPackIdentity =
    weight::digest_from_hex("b0bd0103dcdd9ffd89e65ae06a3d86663d516c5eb09f263346555d2376bf658e");

// Served receipt (gfx1151, 16 slots, context 16384, 256 tokens per request): EXACT (= FAST) B=1 decode 23.36 tok/s,
// C8 53.76, C16 66.74; TURBO row, q4 attention and head, 27.03 / 68.48 / 86.60 tok/s. 8K prefill 14.15 s EXACT,
// 14.38 s TURBO (neutral to slower), so prompt chunks keep the BF16 image and the row is decode-only, banded to the
// step widths the receipt measured a gain at (kMiniLowBitStepBand rows). Its quality against BF16 (decode gate, 39
// held-out windows, 2457 rows) is stated in the row; it fails the FAST bar on two windows.
inline constexpr weight::Digest256 kMiniLowBitBuild =
    weight::digest_from_hex("89665a6147dd8ca5eefcf4fbc4306f56a4c9e5e35bea774943d22ea08dd60f03");
inline constexpr weight::Digest256 kMiniCheckpoint =
    weight::digest_from_hex("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1");
inline constexpr std::uint32_t kMiniLowBitStepBand = 16U;
inline constexpr std::array<weight::WeightLoadDecision, 1> kMiniWeightLoadDecisionRows{{
    {kMiniTurboWeightPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-clang-24.0", kMiniWeightRegime, kMiniLowBitBuild, kMiniCheckpoint},
     {},
     weight::digest_from_hex("54182047338baf7005882cf2062d334aea4b8bb662d6cb484f1ccfce6926508b"),
     {"docs/engine/lowbit-calibration.md", "2026-11-24", 2457, 0, kMiniLowBitStepBand},
     "vs BF16 decode kl=0.0080 top1=0.978 dnll=+0.0076 flips=1.88x windows_failed=2/39 pack=b0bd0103dcdd9ffd"},
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
