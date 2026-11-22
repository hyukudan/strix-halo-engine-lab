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

// A receipt row is scoped to the consumers' implementation identity (weight::implementation_identity of
// core::kResidentQ8Arithmetic<Backend>), checkpoint SHA-256, backend, device, compiler and regime.
// Populate this table only from passing gate and complete workload evidence.
//   gfx1151 / hip-7.1 = HIP_VERSION_MAJOR.MINOR the binary is compiled with (cluster1: the clang HIP wrapper
//   pre-includes /usr/include/hip, 7.1, ahead of the 7.16 core tree) (cluster1, checkpoint model-00001-of-00001.safetensors 19,526,019,168 bytes): gate v2 PASS in
//   decode mode (KL 0.00012, top-1 0.996) and prefill mode (16 windows, KL 0.00010, top-1 0.998, dNLL +0.00042);
//   same binary, gen 256: prefill FAST 1,127 / 921-930 / 716.5 vs EXACT 1,111-1,122 / 917-928 / 715.0 tok/s and
//   decode 26.4 / 23.9 / 21.46 vs 22.2 / 20.4 / 18.56 tok/s at 8K / 32K / 64K. The digests name the receipt manifests.
inline constexpr std::array<weight::WeightLoadDecision, 1> kMiniWeightLoadDecisionRows{{
    {kMiniWeightLoadPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-7.1", kMiniResidentRegime,
      weight::digest_from_hex("06b924b4f4116787df43a3668795417070ca22669e6314fd3bb8250eb29bd9c6"),
      weight::digest_from_hex("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1")},
     weight::digest_from_hex("b1514b2d81938783f3949aa42e0949d4083ea959db317c532b316c0b5113de7f"),
     weight::digest_from_hex("5edd127c94574ace19d23af230e722993233cfdf04c895fb674ddc4b8c339db5")},
}};
inline constexpr std::span<const weight::WeightLoadDecision> kMiniWeightLoadDecisions{kMiniWeightLoadDecisionRows};

inline weight::WeightLoadPlan resolve_mini_weight_load(
    profile::Name tier, const weight::WeightLoadScope& scope,
    std::span<const weight::FormatId> consumer_formats, bool enabled, bool qualify,
    std::span<const weight::WeightLoadDecision> decisions = kMiniWeightLoadDecisions) {
    return weight::resolve_weight_load(weight::builtin_registry(), kMiniWeightLoadPolicy,
        {tier, scope, consumer_formats, enabled, qualify}, decisions);
}

inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
