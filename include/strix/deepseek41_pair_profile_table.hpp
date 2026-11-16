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

inline constexpr std::array<strix::profile::Lever, 7> profile_table{{
    // Prefill only (device prefill of the serve and the prefill app). FP32 softmax statistics.
    {"attention_f16wmma", "STRIX_DS41_FAST_ATTN", "", {"", "1", "", "2"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, true}, "ds41-attn-f16wmma-v1", "0",
     "pending: qualification receipt for ds41-attn-f16wmma-v1 (f16 WMMA sparse attention); runnable with --allow-pending; turbo = 2 (ds41-attn-bf16flash-v1, the FAST bundle member)",
     false, {false, false, false, true}},
    // Prefill only: routed MoE IQ2_XXS gate/up + Q2_K down with FMA accumulation in ascending group order. Applied by
    // 'fast' while its qualification runs (provisional, never stamped qualified); 'default' and 'quality' never set it.
    {"moe_relaxed", "STRIX_DS41_FAST_MOE", "", {"", "1", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "selected for fast ahead of its qualification receipt (ds41-moe-relaxed-v1, in progress); changes logits",
     false, {false, true, false, false}},
    {"dense_f16", "STRIX_DS41_FAST_DENSE", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, true}, "", "0", "FAST dense (ds41-dense-f16-v1), selected by turbo ahead of qualification; the exact dense guard stays (dense_min_rows 128)", false,
     {false, false, false, true}, true},
    {"fast_prefill", "STRIX_DS41_FAST_PREFILL", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0", "explicit FAST bundle; native qualification required", false, {}, true},
    {"dense_min_rows", "STRIX_DS41_FAST_DENSE_MIN_ROWS", "", {"", "", "", ""}, "128", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "", "FAST dense dispatch threshold", false, {}, true},
    {"decoder_replay", "STRIX_DS41_DECODER_REPLAY", "", {"", "", "", "128"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: decoder replay R=128 under the turbo identity ds41-replay-r128 (open PR #605 admits R >= 256 only; R=256 stays the alternative)", false, {}, true},
    // Prefill only: the Engram KV projection on the shared FP16 WMMA core (ds41-engram-f16wmma-v1); views whose x would overflow FP16
    // keep the exact kernel. Explicit only (not a FAST bundle member) until its qualification receipt.
    {"engram_f16wmma", "STRIX_DS41_FAST_ENGRAM", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0", "explicit FAST Engram projection; native qualification required; not a turbo member until its qualification receipt", false, {}, true},
}};

}  // namespace strix::deepseek41::pair_defaults
