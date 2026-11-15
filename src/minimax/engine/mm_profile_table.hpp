#pragma once
// Product-profile rows of the MiniMax M3.1 pair apps (strix-minimax-rank / strix-minimax-serve): the levers that are
// OFF by default (the defaults-ON items stay in mm_defaults.cpp kItems) and that both ranks must agree on. Resolved by
// the one MiniMax resolver (mm_defaults.cpp prefill_policy / profile_resolution): engine, session-cache identity,
// effective config and rank agreement read the same values. A row at its builtin value is left out of the profile
// fingerprint (omit_builtin_fingerprint), so with every row OFF these rows add no fingerprint inputs.
// Complete cache compatibility also depends on the executable and raw environment identity.
// No row has a 'fast' column yet: MiniMax rank's --profile is the per-phase timing session, so the product profile is
// selected by the FAST prefill bundle switch only.
#include "strix/engine_profile.hpp"
#include "strix/fast_profile.hpp"

#include <array>
#include <string_view>

namespace strix::minimax::defaults {

// STRIX_MM_FAST_PREFILL=1 (default off) = profile mm-fast-pf-v1:
//   MSA attention  STRIX_MM_PF_MSA_FLASH=1  msa-flash-bf16-l40-v1 (one-term BF16 operands on the shared key-split
//                  WMMA flash core, layers STRIX_MM_PF_MSA_FLASH_LAYERS = 40-59), on the exact EXL3 MoE base only:
//                  the bundle does not select it with STRIX_MMX_MODEL_FORMAT=fast_i4 (gate fails there, KL 0.079).
// A set member variable wins over the bundle ("0" = off beats it). Changes logits.
inline constexpr strix::fast_profile::Member kFastMembers[] = {
    {"STRIX_MM_PF_MSA_FLASH", "1", "msa-flash-bf16-l40-v1", "1 2"},
};
inline constexpr strix::fast_profile::Bundle kFastBundle{"STRIX_MM_FAST_PREFILL", "mm-fast-pf-v1", kFastMembers};
inline constexpr const char* kMsaFlashLayers = "40-59";  // passes the FAST gate on EXL3 at 8K and 32K; 30-59 8K only
inline constexpr unsigned kMsaLayers = 60;

// Named one-term policies (canonical layer ranges -> policy id), unchanged FAST gate (KL <= 0.02, top-1 >= 93 %,
// dNLL <= 0.01) on EXL3, 256 rows per prompt:
//   l40-v1  8K PASS (3 prompts, KL 0.0012); 32K PASS (b32k KL 0.0029, c32k KL 0.0029, top-1 99.6 / 100 %): the bundle's.
//   l30-v1  8K PASS (KL 0.0076, top-1 98.2 %, dNLL -0.0096); 32K FAIL (b32k KL 0.042, dNLL +0.012; c32k dNLL +0.016):
//           explicit STRIX_MM_PF_MSA_FLASH_LAYERS=30-59 only, for prompts of at most 8K tokens. The layer set is
//           resolved once per process, so the bundle cannot switch it by prompt length.
// Any other mask / the two-term mode is reported by its ranges.
struct MsaFlashPolicy {
    const char* layers;
    const char* policy;
};
inline constexpr MsaFlashPolicy kMsaFlashPolicies[] = {
    {"40-59", "msa-flash-bf16-l40-v1"},
    {"30-59", "msa-flash-bf16-l30-v1"},
};
inline constexpr const char* msa_flash_policy(std::string_view layers) {
    for (const auto& p : kMsaFlashPolicies)
        if (layers == p.layers) return p.policy;
    return nullptr;
}
static_assert(msa_flash_policy(kMsaFlashLayers) != nullptr &&
                  std::string_view(msa_flash_policy(kMsaFlashLayers)) == std::string_view(kFastMembers[0].policy),
              "the bundle member's policy id names the bundle's layer set");

// Bundle selection is requested provenance; only effective members enter agreement.
inline constexpr std::array<strix::profile::Lever, 2> profile_table{{
    // Prefill only: MSA attention of windows > 8 rows; selection, block lists and K/V caches unchanged.
    {"msa_flash", "STRIX_MM_PF_MSA_FLASH", "", {"", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "", "0",
     "msa-flash-bf16-l40-v1 (EXL3 gate PASS 8K KL 0.0012, 32K KL 0.0029); 2 = two-term operands", false, {}, true},
    {"msa_flash_layers", "STRIX_MM_PF_MSA_FLASH_LAYERS", "", {"", "", ""}, kMsaFlashLayers, strix::profile::Kind::bit_changing,
     {true, false, true}, {true, true, true}, "", "", "FAST MSA flash layer set (read only while msa_flash is on)", false, {}, true},
}};

}  // namespace strix::minimax::defaults
