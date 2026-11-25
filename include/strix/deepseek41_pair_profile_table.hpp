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
    // Prefill only: FAST policy ds41-decoder-replay-bounded-v1 (value R rows, a multiple of 128): every decoder layer runs
    // its last R rows, window keys floored at the cut, its dense products in the exact form (bounded run == full run
    // with the floor, bit for bit; not bit-equal to the full decode). Explicit only, OFF in default / fast / quality;
    // set, it enters the profile fingerprint (rank agreement, prefix-cache identity). TURBO R=256 is pending a
    // matched timing and quality receipt on the executing key floor.
    {"decoder_replay", "STRIX_DS41_DECODER_REPLAY", "", {"", "", "", "256"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, true}, "ds41-replay-r256-v1", "0",
     "pending: TURBO timing over effective FAST and quality for R=256 with key floor round_down(T - R, 256); "
     "the historical R=256 gate ran on floor T - R", false, {}, true, {},
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
    // are decoded inside the product, no pack). Automatic TURBO admission needs the actual request prefill regime
    // and must replace relaxed MoE atomically for execution and prefix identity. Explicit STRIX_DS41_FAST_I4=1
    // already displaces the profile-selected relaxed MoE.
    {"moe_iu4", "STRIX_DS41_FAST_I4", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "gguf-iu4-direct-v1", "0",
     "pending: TURBO prefill timing over FAST and quality (KL / top-1 / dNLL vs BF16) for gguf-iu4-direct-v1; "
     "automatic admission requires the request prefill regime and atomic replacement of relaxed MoE", false,
     {}, true, {}, {{{"0", "v1"}, {"1", "gguf-iu4-direct-v1"}}}},
    {"moe_iu4_scope", "STRIX_DS41_FAST_I4_SCOPE", "", {"", "", "", ""}, "", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "", "explicit IU4 projection scope: all (default) or gu", false, {}, true, {}, {{{"", "all-v1"}, {"all", "all-v1"}, {"gu", "gu-v1"}}}},
    // Explicit diagnostics (construct-time switches of the device prefill): listed so receipts name them.
    {"prefill_bf16_attention", "STRIX_DS41_PREFILL_BF16_ATTN", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {}, {true, true, true, true}, "", "0", "explicit BF16 hi/lo attention diagnostic", false, {}, true, {}, {{{"0", "v1"}, {"1", "bf16-hi-lo-v1"}}}},
    {"prefill_fp16_dense_lt", "STRIX_DS41_PREFILL_FP16_DENSE", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {}, {}, "", "0", "explicit FP16 dense diagnostic; requires the optional hipBLASLt build", false, {}, true, {}, {{{"0", "v1"}, {"1", "fp16-hipblaslt-v1"}}}},
    // One decision for the complete D1 composition. A benchmark outside the receipt must explicitly
    // set the constituent switches. Pending startup never exports request-specific arithmetic.
    {"d1_prefill", "", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, true, true, false}, {true, false, true, false}, "D1 full prefill", "0",
     "D1: BF16 flash attention, FP16 dense, relaxed MoE with FP16 down, FP16 Engram and shared expert. Replay OFF. "
     "Unmatched requests retain the previous prefill route (explicit switches still apply).",
     false, {}, true, {}, {{{"0", "v1"}, {"1", "ds41-prefill-d1-v1"}}}},
}};

inline constexpr std::string_view kFastGateD1 =
    "FAST gate v2 D1, complete set without replay: 8K class 14 windows, 882 rows, final PASS "
    "(KL 0.0116, top-1 0.962, dNLL -0.0072); 32K class 126 rows, provisional";
inline constexpr strix::profile::FastGate fast_gates[] = {{"d1_prefill", "1", kFastGateD1}};

inline constexpr strix::profile::LadderRow ladder_rows[] = {
    {.lever = "d1_prefill", .value = "1", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 8128, .max_context = 8128, .min_lanes = 1, .max_lanes = 1,
     .speedup = 16.179274 / 12.694894, .noise = 0.05,
     .receipt = "D1 final 8K PASS. Pair quality-arm timings at 8128 were 16.148/16.179 s EXACT and 12.564/12.695 s D1. "
                "single full prompt, no row split or cone, head64, chunk 1024. Other executions keep the previous route.",
     .band_bound = true},
};

}  // namespace strix::deepseek41::pair_defaults
