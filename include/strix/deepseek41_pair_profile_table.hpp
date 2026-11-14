#pragma once

#include <array>

#include "strix/engine_profile.hpp"

// Product-profile rows of the two-node DeepSeek-V4.1 apps (rank / batch / prefill / serve), one per lever:
// value per profile {default, fast, quality}, ready = qualified (qualification receipt id in `source`), runnable = the
// code is in this build, provisional = applied ahead of its receipt and stamped "qualification in progress". Every
// exact production default stays in deepseek41_pair_defaults (on in every profile); a row here is a lever whose value
// differs between profiles or must agree across ranks. A bit-changing column flips ready only with its qualification receipt
// (docs/optimizations.json). No quality pack: 'quality' runs default with one notice.
namespace strix::deepseek41::pair_defaults {

inline constexpr std::array<strix::profile::Lever, 6> profile_table{{
    // Prefill only (device prefill of the serve and the prefill app). FP32 softmax statistics.
    {"attention_f16wmma", "STRIX_DS41_FAST_ATTN", "", {"", "1", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "ds41-attn-f16wmma-v1", "0",
     "pending: qualification receipt for ds41-attn-f16wmma-v1 (f16 WMMA sparse attention); runnable with --allow-pending"},
    // Prefill only: routed MoE IQ2_XXS gate/up + Q2_K down with FMA accumulation in ascending group order. Applied by
    // 'fast' while its qualification runs (provisional, never stamped qualified); 'default' and 'quality' never set it.
    {"moe_relaxed", "STRIX_DS41_FAST_MOE", "", {"", "1", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "", "0",
     "selected for fast ahead of its qualification receipt (ds41-moe-relaxed-v1, in progress); changes logits",
     false, {false, true, false}},
    {"dense_f16", "STRIX_DS41_FAST_DENSE", "", {"", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "", "0", "explicit FAST dense; native qualification required", false, {}, true},
    {"fast_prefill", "STRIX_DS41_FAST_PREFILL", "", {"", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "", "0", "explicit FAST bundle; native qualification required", false, {}, true},
    {"dense_min_rows", "STRIX_DS41_FAST_DENSE_MIN_ROWS", "", {"", "", ""}, "128", strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "", "", "FAST dense dispatch threshold", false, {}, true},
    // Prefill only: FAST policy ds41-decoder-replay-bounded-v1 (value R rows, a multiple of 128): every decoder layer runs
    // its last R rows, window keys floored at the cut, its dense products in the exact form (bounded run == full run
    // with the floor, bit for bit; not bit-equal to the full decode). Explicit only, OFF in every
    // profile; set, it enters the profile fingerprint (rank agreement, prefix-cache identity).
    {"decoder_replay_bounded", "STRIX_DS41_DECODER_REPLAY", "", {"", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "", "0", "explicit FAST policy ds41-decoder-replay-bounded-v1; quality gate (DS41 FAST user rule) required", false, {}, true},
}};

}  // namespace strix::deepseek41::pair_defaults
