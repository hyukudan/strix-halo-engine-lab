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

inline constexpr std::array<strix::profile::Lever, 13> profile_table{{
    // Prefill only (device prefill of the serve and the prefill app). FP32 softmax statistics.
    {"attention_f16wmma", "STRIX_DS41_FAST_ATTN", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, true}, "ds41-attn-f16wmma-v1", "0",
     "pending: qualification receipt for ds41-attn-f16wmma-v1 (f16 WMMA sparse attention); runnable with --allow-pending; fast and turbo run 2 (ds41-attn-bf16flash-v1) through the FAST bundle",
     false, {false, false, false, false}, false, {}, {{{"0", "v1"}, {"1", "ds41-attn-f16wmma-v1"}, {"2", "ds41-attn-bf16flash-v1"}}}},
    // Prefill only: routed MoE IQ2_XXS gate/up + Q2_K down with FMA accumulation in ascending group order. Applied by
    // 'fast' while its qualification runs (provisional, never stamped qualified); 'default' and 'quality' never set it.
    {"moe_relaxed", "STRIX_DS41_FAST_MOE", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "selected for fast ahead of its qualification receipt (ds41-moe-relaxed-v1, in progress); changes logits",
     false, {false, true, false, false}, false, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    {"dense_f16", "STRIX_DS41_FAST_DENSE", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, true}, "", "0", "FAST dense (ds41-dense-f16-v1), a member of the FAST bundle (fast and turbo run it through the bundle); the exact dense guard stays (dense_min_rows 128)", false,
     {false, false, false, false}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    // FAST prefill bundle ds41-fast-pf-v1 (attention bf16flash-v1 + moe-relaxed-v1 + dense-f16-v1, strix/ds41_fast_profile.hpp).
    // 'fast' selects it (expanded exactly as the switch: src/deepseek41_pair_defaults.cpp resolve); a set switch wins.
    // The fast cells of this row and of engram_f16wmma / shared_f16 / moe_f16wmma are selected for fast ahead of a full
    // receipt (provisional): the FAST gate v2 of exactly this resolved set (no replay, tools/strix-gate/gate_v2.py) is
    // final PASS for prompts <= 8,192 (882 rows) and provisional above (32K class 126 rows < 800). The ledger
    // `fast_gates` below holds the receipt; the row flips to qualified when the 32K class is final. TURBO nests FAST: its
    // empty cells run the same bundle and members, each set once (by the bundle).
    {"fast_prefill", "STRIX_DS41_FAST_PREFILL", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "FAST bundle ds41-fast-pf-v1 with the three levers below; FAST gate v2 D1 on the resolved fast set (no replay): qualified for prompts <= 8,192 (8K class 14 windows, 882 rows, final PASS: KL 0.0116, top-1 0.962, dNLL -0.0072 se 0.0058); provisional above (32K class 126 rows < 800)",
     false, {false, true, false, false}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    {"dense_min_rows", "STRIX_DS41_FAST_DENSE_MIN_ROWS", "", {"", "", "", ""}, "128", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "", "FAST dense dispatch threshold", false, {}, true, {}, {{{"128", "v1"}}}},
    // Prefill only: FAST policy ds41-decoder-replay-bounded-v1 (value R rows, a multiple of 128): every decoder layer runs
    // its last R rows, window keys floored at the cut, its dense products in the exact form (bounded run == full run
    // with the floor, bit for bit; not bit-equal to the full decode). Explicit only, OFF in default / fast / quality;
    // set, it enters the profile fingerprint (rank agreement, prefix-cache identity). The turbo cell (R=128 under
    // ds41-replay-r128) stays pending: v1 refuses bounds below 256.
    {"decoder_replay", "STRIX_DS41_DECODER_REPLAY", "", {"", "", "", "128"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "explicit FAST policy ds41-decoder-replay-bounded-v1, R >= 256 (smaller bounds refused); not qualified on its key floor round_down(T - R, 256) (the R = 256 gate ran on the floor T - R), so no profile sets it; turbo R=128 (ds41-replay-r128) pending", false, {}, true, {}, {{{"0", "v1"}, {"128", "v1"}}}},
    // Prefill only: the Engram KV projection on the shared FP16 WMMA core (ds41-engram-f16wmma-v1); views whose x would overflow FP16
    // keep the exact kernel. Not a bundle member (the env bundle ds41-fast-pf-v1 keeps its identity); 'fast' selects it.
    {"engram_f16wmma", "STRIX_DS41_FAST_ENGRAM", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "FAST Engram projection ds41-engram-f16wmma-v1 in the fast set; FAST gate v2 D1 on the resolved fast set (no replay): qualified for prompts <= 8,192 (8K class 14 windows, 882 rows, final PASS: KL 0.0116, top-1 0.962, dNLL -0.0072 se 0.0058); provisional above (32K class 126 rows < 800); TURBO inherits FAST",
     false, {false, true, false, false}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    // Prefill only: the shared expert's three Q8_0 x A8 products on the FAST dense core (ds41-shared-f16-v1). Not a bundle
    // member; 'fast' selects it.
    {"shared_f16", "STRIX_DS41_FAST_SHARED", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "FAST shared expert ds41-shared-f16-v1 (#720) in the fast set; FAST gate v2 D1 on the resolved fast set (no replay): qualified for prompts <= 8,192 (8K class 14 windows, 882 rows, final PASS: KL 0.0116, top-1 0.962, dNLL -0.0072 se 0.0058); provisional above (32K class 126 rows < 800)",
     false, {false, true, false, false}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    // Prefill only, with fast_moe: the relaxed routed-expert kernels' FP16-operand form (ds41-moe-f16wmma-v1); 'fast' selects it.
    {"moe_f16wmma", "STRIX_DS41_FAST_MOE_F16", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "FAST MoE down FP16 operands ds41-moe-f16wmma-v1 (#721) in the fast set; FAST gate v2 D1 on the resolved fast set (no replay): qualified for prompts <= 8,192 (8K class 14 windows, 882 rows, final PASS: KL 0.0116, top-1 0.962, dNLL -0.0072 se 0.0058); provisional above (32K class 126 rows < 800)",
     false, {false, true, false, false}, true, {}, {{{"0", "v1"}, {"1", "v1"}}}},
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
}};

// FAST gate ledger (strix::profile::FastGate, tests/profile_fast_set_tests.cpp): the gate receipt behind each fast cell.
inline constexpr std::string_view kFastGateD1 =
    "FAST gate v2 D1, resolved fast set, pair, prompts <= 8,192: 14 windows, 882 rows, final PASS (32K: provisional)";
inline constexpr strix::profile::FastGate fast_gates[] = {
    {"fast_prefill", "1", kFastGateD1}, {"engram_f16wmma", "1", kFastGateD1},
    {"shared_f16", "1", kFastGateD1},   {"moe_f16wmma", "1", kFastGateD1}};

// Ladder rows (strix::profile::LadderRow): incumbent grants for the cells the resolver ran before the ladder rule.
// TURBO nests FAST: its column is empty for every member, so it resolves over the admitted FAST set.
namespace ladder_detail {
namespace sp = ::strix::profile;
inline constexpr auto kFast = sp::Name::fast;
inline constexpr auto kTurbo = sp::Name::turbo;
}  // namespace ladder_detail
inline constexpr strix::profile::LadderRow ladder_rows[] = {
    strix::profile::incumbent_cell("fast_prefill", "1", ladder_detail::kFast, kFastGateD1, "v1"),
    strix::profile::incumbent_cell("engram_f16wmma", "1", ladder_detail::kFast, kFastGateD1, "v1"),
    strix::profile::incumbent_cell("shared_f16", "1", ladder_detail::kFast, kFastGateD1, "v1"),
    strix::profile::incumbent_cell("moe_f16wmma", "1", ladder_detail::kFast, kFastGateD1, "v1"),
    strix::profile::incumbent_cell("moe_relaxed", "1", ladder_detail::kFast, strix::profile::kProvisionalBeforeLadder, "v1"),
    strix::profile::incumbent_cell("moe_relaxed", "1", ladder_detail::kTurbo, strix::profile::kProvisionalBeforeLadder, "v1")};

}  // namespace strix::deepseek41::pair_defaults
