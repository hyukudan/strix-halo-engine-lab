#pragma once

#include <array>

#include "strix/engine_profile.hpp"

// Product-profile rows of the two-node DeepSeek-V4.1 apps (rank / batch / prefill / serve), one per lever:
// value per profile {default, fast, quality, turbo}, ready = qualified (qualification receipt id in `source`), runnable = the
// code is in this build, provisional = applied ahead of its receipt and stamped "qualification in progress". Every
// exact production default stays in deepseek41_pair_defaults (on in every profile); a row here is a lever whose value
// differs between profiles or must agree across ranks. A bit-changing column flips ready only with its qualification receipt
// (docs/optimizations.json). No quality pack: 'quality' runs default with one notice.
namespace strix::deepseek41::pair_defaults {

inline constexpr std::array<strix::profile::Lever, 14> profile_table{{
    // The historical bundle and the individual arithmetic switches remain explicit experiments.
    {"attention_f16wmma", "STRIX_DS41_FAST_ATTN", "", {}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit FAST attention: 1 FP16 WMMA, 2 BF16 flash WMMA",
     false, {}, false, {}, {{{"0", "v1"}, {"1", "ds41-attn-f16wmma-v1"}, {"2", "ds41-attn-bf16flash-v1"}}}},
    {"moe_relaxed", "STRIX_DS41_FAST_MOE", "", {}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit relaxed routed MoE (qualification in progress)",
     false, {}, false, {}, {{{"0", "v1"}, {"1", "ds41-moe-relaxed-v1"}}}},
    {"dense_f16", "STRIX_DS41_FAST_DENSE", "", {}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit FAST FP16 dense",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "ds41-dense-f16-v1"}}}},
    {"fast_prefill", "STRIX_DS41_FAST_PREFILL", "", {}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit ds41-fast-pf-v1 bundle (attention, relaxed MoE, dense)",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "ds41-fast-pf-v1"}}}},
    {"dense_min_rows", "STRIX_DS41_FAST_DENSE_MIN_ROWS", "", {}, "128", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "", "FAST dense dispatch threshold", false, {}, true, {}, {{{"128", "v1"}}}},
    // One decision for the complete D1 composition, before replay so the replay receipt binds over D1. A benchmark outside the receipt must explicitly
    // set the constituent switches. Pending startup never exports request-specific arithmetic.
    {"d1_prefill", "", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, true, true, false}, {true, false, true, false}, "D1 full prefill", "0",
     "D1: BF16 flash attention, FP16 dense, relaxed MoE with FP16 down, FP16 Engram and shared expert. Replay OFF. "
     "Unmatched requests retain the previous prefill route (explicit switches still apply).",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "ds41-prefill-d1-v1"}}}},
    // Prefill only: FAST policy ds41-decoder-replay-bounded-v1 (value R rows, a multiple of 128): every decoder layer runs
    // its last R rows, window keys floored at the cut, its dense products in the exact form (bounded run == full run
    // with the floor, bit for bit; not bit-equal to the full decode). 'fast' runs R=256 (key floor
    // round_down(T - R, 256)) on an admitted D1 request inside its band receipts; TURBO nests it (empty cell). OFF in default / quality.
    // Qualified by FAST gate v2 D1R (8K and 32K classes). A run that needs every decoder row (a deferred serve piece,
    // in-run prefix snapshots, a group pass) takes the exact plan instead. Set, it enters the profile fingerprint (rank
    // agreement, prefix-cache identity).
    {"decoder_replay", "STRIX_DS41_DECODER_REPLAY", "", {"", "256", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, true, true, false}, {true, false, true, false}, "", "0",
     "FAST bounded decoder replay R=256 (ds41-replay-r256-v1, key floor round_down(T - R, 256)) on the resolved fast set; "
     "FAST gate v2 D1R on fast set + R256: qualified (8K and 32K classes, 28 windows, 1,764 rows, final PASS: KL 0.0101, "
     "top-1 0.972, dNLL +0.0001 se 0.0037, route flips 0.37 x null; 8K class KL 0.0121 top-1 0.962, 32K class KL 0.0080 "
     "top-1 0.982 dNLL +0.0060 se 0.0044)",
     false, {false, false, false, false}, true, {},
     {{{"0", "v1"}, {"256", "ds41-replay-r256-v1"}}}},
    {"engram_f16wmma", "STRIX_DS41_FAST_ENGRAM", "", {}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit FAST FP16 Engram projection",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "ds41-engram-f16wmma-v1"}}}},
    {"shared_f16", "STRIX_DS41_FAST_SHARED", "", {}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit FAST FP16 shared expert",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "ds41-shared-f16-v1"}}}},
    {"moe_f16wmma", "STRIX_DS41_FAST_MOE_F16", "", {}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit FAST FP16 routed MoE down (requires relaxed MoE)",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "ds41-moe-f16wmma-v1"}}}},
    // Prefill only: routed experts as IU4 codes x A4 activations (gguf-iu4-direct-v1; the stored IQ2_XXS / Q2_K weights
    // are decoded inside the product, no pack). No profile selects it: on the pair prefill it is slower than the
    // relaxed MoE it replaces (source below). Explicit STRIX_DS41_FAST_I4=1 displaces the profile-selected relaxed MoE.
    {"moe_iu4", "STRIX_DS41_FAST_I4", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0",
     "explicit only: no profile selects gguf-iu4-direct-v1 (pair prefill 8,192 tokens over the resolved fast set: "
     "4,975 ms FAST, 18,116 ms scope all, 15,110 ms scope gu; slower than FAST, no TURBO row)", false,
     {}, true, {}, {{{"0", "v1"}, {"1", "gguf-iu4-direct-v1"}}}},
    {"moe_iu4_scope", "STRIX_DS41_FAST_I4_SCOPE", "", {"", "", "", ""}, "", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "", "explicit IU4 projection scope: all (default) or gu", false, {}, true, {}, {{{"", "all-v1"}, {"all", "all-v1"}, {"gu", "gu-v1"}}}},
    // Explicit diagnostics (construct-time switches of the device prefill): listed so receipts name them.
    {"prefill_bf16_attention", "STRIX_DS41_PREFILL_BF16_ATTN", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit BF16 hi/lo attention diagnostic", false, {}, true, {}, {{{"0", "v1"}, {"1", "bf16-hi-lo-v1"}}}},
    {"prefill_fp16_dense_lt", "STRIX_DS41_PREFILL_FP16_DENSE", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {}, {}, "", "0", "explicit FP16 dense diagnostic; requires the optional hipBLASLt build", false, {}, true, {}, {{{"0", "v1"}, {"1", "fp16-hipblaslt-v1"}}}},
}};

inline constexpr std::string_view kFastGateD1 =
    "FAST gate v2 D1, complete D1 set without replay, pair, prompts <= 8,192 and 32,768: 28 windows, 1,764 rows, final PASS "
    "(KL 0.0093, top-1 0.972, dNLL -0.0037 se 0.0035, route flips 1.05 x null)";
inline constexpr std::string_view kFastGateD1R =
    "FAST gate v2 D1R, complete D1 set + decoder replay R=256, pair, prompts <= 8,192 and 32,768: 28 windows, 1,764 "
    "rows, final PASS (KL 0.0101, top-1 0.972, dNLL +0.0001 se 0.0037, route flips 0.37 x null)";
inline constexpr strix::profile::FastGate fast_gates[] = {{"d1_prefill", "1", kFastGateD1},
                                                          {"decoder_replay", "256", kFastGateD1R}};

inline constexpr strix::profile::LadderRow ladder_rows[] = {
    // Band receipts (bind_band_receipts per request): the complete D1 composition and, on top of it, the bounded
    // decoder replay R=256. Timed on the pair CLI (gfx1151, one request) at 8,192 and 32,768 prompt tokens; the quality
    // arms at 8,128 gave the same direction (16.148 / 16.179 s EXACT, 12.564 / 12.695 s D1). Outside the bands, or
    // when either rank declines, the request keeps the previous prefill route. The 8,128 timing is D1 only. Lower edges
    // timed on the pair CLI (ABBA, warm run, the FAST product D1 + R=256 vs EXACT; D1 alone is not a product execution:
    // any explicit lever leaves the measured composition): 4,096 x1.60, 6,000 x1.57, 8,193 x1.58, 16,384 x1.55 -> the
    // nominal edges stay.
    {.lever = "d1_prefill", .value = "1", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 4096, .max_context = 8192, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1614.0 / 1128.5, .noise = 0.05,
     .receipt = "D1 final 8K PASS. Pair CLI 8,192 tokens: EXACT 1,128.5 tok/s, D1 1,614 / 1,647 tok/s (no replay). Quality runner 8,128 tokens: EXACT 16.148 / 16.179 s, D1 12.564 / 12.695 s. "
                "Lower edge 4,096: EXACT 4,089 ms, FAST (D1 + R256) 2,557 ms (x1.60); 6,000: x1.57.",
     .identity = "ds41-prefill-d1-v1", .band_bound = true, .band_evidence = {{4096, 6000, 8128, 8192}, {8128, 32768}, kFastGateD1, false}},
    {.lever = "d1_prefill", .value = "1", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 8193, .max_context = 32768, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1808.6 / 1257.2, .noise = 0.05,
     .receipt = "D1 final 32K PASS. Pair CLI 32,768 tokens: EXACT 1,257.2 tok/s, D1 1,808.6 tok/s (no replay). "
                "Lower edge 8,193: EXACT 7,172 ms, FAST (D1 + R256) 4,537 ms (x1.58); 16,384: x1.55.",
     .identity = "ds41-prefill-d1-v1", .band_bound = true, .band_evidence = {{8193, 16384, 32768}, {8128, 32768}, kFastGateD1, false}},
    {.lever = "decoder_replay", .value = "256", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 4096, .max_context = 8192, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1855.0 / 1647.0, .noise = 0.05,
     .receipt = "D1R final PASS. Pair CLI 8,192 tokens over D1: 1,614 / 1,647 tok/s without replay, 1,855 tok/s with R=256. "
                "Lower edge 4,096: FAST 2,557 ms vs EXACT 4,089 ms (x1.60); 6,000: x1.57.",
     .identity = "ds41-replay-r256-v1", .band_bound = true, .band_evidence = {{4096, 6000, 8192}, {8128, 32768}, kFastGateD1R, false}},
    {.lever = "decoder_replay", .value = "256", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 8193, .max_context = 32768, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1981.0 / 1808.6, .noise = 0.05,
     .receipt = "D1R final PASS. Pair CLI 32,768 tokens over D1: 1,808.6 tok/s without replay, 1,981 tok/s with R=256. "
                "Lower edge 8,193: FAST 4,537 ms vs EXACT 7,172 ms (x1.58); 16,384: x1.55.",
     .identity = "ds41-replay-r256-v1", .band_bound = true, .band_evidence = {{8193, 16384, 32768}, {8128, 32768}, kFastGateD1R, false}},
};

}  // namespace strix::deepseek41::pair_defaults
