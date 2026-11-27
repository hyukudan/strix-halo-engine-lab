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
    // Prefill only: FAST policy ds41-decoder-replay-bounded-v1 (value R rows, a multiple of 128): every decoder layer
    // runs its last R rows, window keys floored at the cut round_down(T - R, 256), its dense products in the exact form
    // (not bit-equal to the full prefill). 'fast' runs R=512 on an admitted D1 request inside its band receipt
    // (4,096..8,192); TURBO nests it (empty cell); OFF in default / quality. Held-out gate v2 (heldout/v1): R=256 fails FAST in both
    // bands by design, not a bug: the key floor truncates the window of the
    // first 127 rows above the cut in each of the 20 decoder layers and the error grows layer by layer (largest on
    // fragile rows, not detectable before the pass). R=512 passes FAST at 8,128; 8,193..32,768 has no replay row until
    // its gate completes (11 of 13 windows scored). R=1,024 passes at 8,128 too, but its speed over D1 alone is within
    // the ladder noise (20,480 flat decoder row-layers against the 24,150-row exact cone). A run that needs every decoder
    // row (a deferred serve piece, in-run prefix snapshots, a group pass) takes the exact plan instead. Set, it enters
    // the profile fingerprint (rank agreement, prefix-cache identity).
    {"decoder_replay", "STRIX_DS41_DECODER_REPLAY", "", {"", "512", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, true, true, false}, {true, false, true, false}, "", "0",
     "FAST bounded decoder replay R=512 (ds41-replay-r512-v1, key floor round_down(T - R, 256)) over D1, prompts "
     "4,096..8,192. Held-out gate v2 (heldout/v1, 13 windows / 819 rows): FAST PASS 8,128 (KL 0.0104, top-1 0.970, "
     "dNLL +0.0010 se 0.0067, flips 0.42 x null). R=256 (explicit only) fails FAST in both bands",
     false, {false, false, false, false}, true, {},
     {{{"0", "v1"}, {"256", "ds41-replay-r256-v1"}, {"512", "ds41-replay-r512-v1"}, {"1024", "ds41-replay-r1024-v1"}}}},
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

// Held-out re-score (gate_r 2026-11-27, origin/main d179ad0d08 quality binary fd050a5a, corrected gate v2, no null
// floor, window set tools/strix-gate/heldout/v1: prose en / es, code, chat; reference-only pre-filter drops windows
// with label == reference top-1 > 0.95). Receipts: gate_r score/{ds8,ds32,all}.
inline constexpr std::string_view kFastGateD1 =
    "FAST gate v2 D1 (complete D1 set without replay, pair, quality head 64), held-out windows heldout/v1: 8,128-token "
    "class 13 windows / 819 rows (2 of 15 dropped by the reference pre-filter), final PASS (KL 0.0105, top-1 0.961, "
    "dNLL +0.0003 se 0.0064, route flips 1.07 x null); 32,768-token class 13 windows / 819 rows (2 dropped), final "
    "PASS (KL 0.0142, top-1 0.965, dNLL -0.0024 se 0.0090, flips 1.06 x null); pooled 1,638 rows PASS (KL 0.0124, "
    "top-1 0.963, dNLL -0.0011 se 0.0055).";
// ds_replay 2026-11-27: same quality binary, windows and gate as gate_r (exact / null hashes equal to gate_r's).
inline constexpr std::string_view kFastGateD1R512 =
    "FAST gate v2 D1 + decoder replay R=512 (pair, quality head 64), held-out windows heldout/v1: 8,128-token class 13 "
    "windows / 819 rows (the gate_r kept set), final PASS (KL 0.0104, top-1 0.970, dNLL +0.0010 se 0.0067, route flips "
    "0.42 x null); TURBO tier PASS. R=256 on the same windows: FAST FAIL (dNLL +0.0122 se 0.0080 at 8,128; +0.0225 se "
    "0.0111 at 32,768, windows en32-a +0.117 / chat32-b +0.126; R=512 there +0.067 / +0.043, R=1,024 +0.0085 / +0.032).";
inline constexpr strix::profile::FastGate fast_gates[] = {{"d1_prefill", "1", kFastGateD1},
                                                          {"decoder_replay", "512", kFastGateD1R512}};

inline constexpr strix::profile::LadderRow ladder_rows[] = {
    // Band receipts (bind_band_receipts per request): the complete D1 composition (FAST) and, on top of it, the bounded
    // decoder replay R=512 (FAST, 4,096..8,192; TURBO nests it). Timed on the pair CLI (gfx1151, one request) at 8,192 and
    // 32,768 prompt tokens; the quality arms at 8,128 gave the same direction (16.148 / 16.179 s EXACT, 12.564 / 12.695 s
    // D1). Outside the bands, or when either rank declines, the request keeps the previous prefill route. Lower edges
    // of D1 alone timed on the quality runner (gate_r, no row split, no cone, head 64, two runs, bit-equal hashes):
    // 4,096 EXACT 9.84 / 9.01 s, D1 7.55 / 7.56 s (x1.19..1.30); 8,193 EXACT 17.81 / 17.68 s, D1 14.61 / 14.58 s (x1.21)
    // -> the nominal edges stay. A drafter seed (served DSpark) rides only on the replay (seed <= R): seeded requests
    // above 8,192 (no replay row) decline D1.
    {.lever = "d1_prefill", .value = "1", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 4096, .max_context = 8192, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1614.0 / 1128.5, .noise = 0.05,
     .receipt = "D1 held-out 8K PASS. Pair CLI 8,192 tokens: EXACT 1,128.5 tok/s, D1 1,614 / 1,647 tok/s (no replay). Quality runner 8,128 tokens: EXACT 16.148 / 16.179 s, D1 12.564 / 12.695 s. "
                "Lower edge 4,096 (quality runner, D1 alone): EXACT 9.84 / 9.01 s, D1 7.55 / 7.56 s (x1.19..1.30).",
     .identity = "ds41-prefill-d1-v1", .band_bound = true, .band_evidence = {{4096, 8128, 8192}, {8128, 32768}, kFastGateD1, false}},
    {.lever = "d1_prefill", .value = "1", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 8193, .max_context = 32768, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1808.6 / 1257.2, .noise = 0.05,
     .receipt = "D1 held-out 32K PASS. Pair CLI 32,768 tokens: EXACT 1,257.2 tok/s, D1 1,808.6 tok/s (no replay). "
                "Lower edge 8,193 (quality runner, D1 alone): EXACT 17.81 / 17.68 s, D1 14.61 / 14.58 s (x1.21).",
     .identity = "ds41-prefill-d1-v1", .band_bound = true, .band_evidence = {{8193, 32768}, {8128, 32768}, kFastGateD1, false}},
    {.lever = "decoder_replay", .value = "512", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 4096, .max_context = 8192, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1773.4 / 1647.0, .noise = 0.05,
     .receipt = "D1 + R512 FAST 8K PASS (held-out). Pair CLI 8,192 tokens (ds_replay, warm run): D1 + R512 1,773.4 tok/s "
                "(4.619 s), D1 + R256 1,832.9 tok/s the same hour (4.470 s; 1,855 in gate_r), D1 alone 1,614 / 1,647 tok/s. "
                "Quality runner 8,128 tokens (13 windows): D1 + R512 8.28..8.65 s, D1 + R1024 8.76..9.07 s, EXACT "
                "18.1..19.5 s, the same hours. Lower edge 4,096 untimed for R512.",
     .identity = "ds41-replay-r512-v1", .band_bound = true,
     .band_evidence = {{8128, 8192}, {8128}, kFastGateD1R512, true}},
};

}  // namespace strix::deepseek41::pair_defaults
