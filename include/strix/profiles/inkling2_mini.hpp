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

// FAST lever: BF16-rounded dense on tensor cores (one BF16 activation plane, DenseArith::mma1, instead of the
// three-plane exact mma3) for the dense projections of a multi-row prompt-prefill chunk on CUDA. Decode, verify and
// one-row calls keep mma3, and so does every BF16-rounded projection group (the rounded epilogue exists only on the
// three-term chain; the attention relative-bias projection). The row name is its versioned identity. Runnable, not
// ready: the shared tall BF16 rows route runs mma1, but no FAST gate or gain receipt exists, so fast/turbo report it
// pending and run mma3; STRIX_INK_DENSE_MMA1=1 opts in (receipt "unqualified"). Off switch: STRIX_INK_DENSE_MMA1=0.
inline constexpr std::string_view kMiniDensePrefillMma1 = "inkling2-mini-dense-prefill-bf16-mma1@1";
// FAST lever: W4A4 experts. Multi-row prompt-prefill chunks run the NVFP4 expert layers through the shared grouped W4A4
// operand class (moe_core/grouped_w4a4.hpp, identity grouped-nvfp4-w4a4-f32-v1: activations quantised once per projection
// input by the family NVFP4 rule, sm_120 block-scaled FP4 MMA, weights used as stored). MXFP8 layers, decode, verify and
// one-row calls keep the BF16 tile. CUDA sm_120 only (HIP ignores it). Runnable, not ready: opt in with
// STRIX_INK_MOE_W4A4=1 until the FAST gate (prefill regime) and a prefill receipt exist. Off switch: STRIX_INK_MOE_W4A4=0.
inline constexpr std::string_view kMiniMoeW4a4 = "inkling2-mini-moe-w4a4@1";
// FAST lever, generic catalogue id attn.decode.split@1: one-row decode attention runs the position-canonical key split of
// the prepared D128 flash (strix/attention_split_plan.hpp; exec leaves prepared.flash.split.d128 and its host-position
// form, identity cuda-sm120-prepared-flash-decode-split-d128-v1) instead of the one-pass prepared flash, on the CUDA step
// program. Values: 0 = one-pass flash (builtin, the EXACT path), auto = split from the plan's crossover on (a visit whose
// plan has two chunks or more, i.e. more than 256 visible keys; below it the visit keeps the one-pass flash and its bits),
// 1 = split on every visit (A/B and crossover measurements). FAST, not EXACT: the split keeps P in FP32 where the one-pass
// flash feeds a BF16 MMA (host check in strix-attention-split-plan-host-tests). Runnable, not ready: the fast cell is
// pending until gate v2 (decode regime, routes replayed) passes; then the cell flips to ready. Switch:
// STRIX_ATTN_DECODE_SPLIT (legacy alias STRIX_INKLING_DECODE_SPLIT), off = 0. HIP keeps its own position-only one-row
// key split (measured: the absolute plan is slower on gfx1151), so this row binds CUDA only.
inline constexpr std::string_view kMiniDecodeSplit = "attn.decode.split@1";
inline constexpr std::string_view kMiniDecodeSplitEnv = "STRIX_ATTN_DECODE_SPLIT";
inline constexpr std::string_view kMiniDecodeSplitLegacyEnv = "STRIX_INKLING_DECODE_SPLIT";
inline constexpr std::array<profile::Lever, 3> kMiniLevers{{
    {kMiniDensePrefillMma1, "STRIX_INK_DENSE_MMA1", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "runnable (shared tall BF16 rows mma1 route); waits for the FAST gate (prefill regime) and a prefill gain receipt",
     false, {}, true},
    {kMiniMoeW4a4, "STRIX_INK_MOE_W4A4", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for the FAST gate (prefill regime, routes replayed) and an RTX prefill receipt",
     false, {}, true},
    {kMiniDecodeSplit, kMiniDecodeSplitEnv, "", {"", "auto", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "auto", "0",
     "runnable (CUDA step program, graph and route-tap forms); waits for the FAST gate (decode regime, routes replayed); "
     "gain receipts: 2.5K context +28 % tok/s, 7-token context -3.5 % (below the crossover, which keeps the one-pass flash)",
     false, {}, true},
}};

inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
