#pragma once

#include "moe_core/grouped_w4a4.hpp"
#include "strix/env_flags.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/resolve.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <string_view>

namespace strix::inkling2 {

struct MiniWeightOptions {
    profile::Inputs tier;
    bool qualify_q8{};  // the Q8_0 qualification arm (--qualify-q8, the alias of --qualify-weights <Q8 policy>)
    bool disable_q8{};  // off switch of every compact weight storage
    // The shared policy of an explicit qualification arm (--qualify-weights KEY); nullptr: none.
    const weight::WeightLoadPolicy* qualify{};
};

// The bindings a qualification key may name: the model's Q8_0 binding (legacy id included), the shared low-bit mixes and
// the shared head-only low-bit rows (attention BF16).
inline constexpr std::array<weight::WeightLoadPolicy, 6> kMiniQualifiablePolicies{
    kMiniWeightLoadPolicy, weight::kAttentionHeadQ6Policy, weight::kAttentionQ4HeadQ6Policy,
    weight::kAttentionHeadQ4Policy, weight::kHeadQ6Policy, weight::kHeadQ4Policy};

// The binding a load resolves under: an explicit arm's policy (the model's own row for the Q8_0 policy, so its legacy
// records match); else the TURBO mix for --profile turbo, the FAST mix for --profile fast and the Q8_0 binding for
// EXACT (which keeps BF16 whatever the policy).
inline const weight::WeightLoadPolicy& mini_weight_load_policy(const MiniWeightOptions& options, profile::Name tier) {
    if (options.qualify == nullptr)
        return tier == profile::Name::turbo ? kMiniTurboWeightPolicy
            : tier == profile::Name::fast   ? kMiniFastWeightPolicy
                                            : kMiniWeightLoadPolicy;
    if (weight::same_id(options.qualify->id, kMiniWeightLoadPolicy.id)) return kMiniWeightLoadPolicy;
    return *options.qualify;
}

// The calibrated low-bit pack file of a checkpoint: STRIX_LOWBIT_PACK, else kMiniLowBitPackFile next to it.
inline std::string mini_lowbit_pack_file(const std::string& checkpoint) {
    if (const char* env = envcfg::text("STRIX_LOWBIT_PACK"); env && *env) return env;
    return (std::filesystem::path(checkpoint).parent_path() / kMiniLowBitPackFile).string();
}

// A regular file at the pack path (no throw: a filesystem error reads as absent).
inline bool mini_lowbit_pack_present(const std::string& checkpoint) {
    std::error_code error;
    return std::filesystem::is_regular_file(mini_lowbit_pack_file(checkpoint), error);
}

inline bool mini_lowbit_storage(weight::FormatId f) {
    return f == weight::kQ4G64WeightStorage || f == weight::kQ6G64WeightStorage;
}

// How the resident image carries a low-bit plan. A decision row (FAST or TURBO) keeps the BF16 image, adds the low-bit
// copies of its roles from the pack it was measured with (`pack`, its identity), and runs them only in decode-regime
// steps of at most `band` rows (0: every step width); prompt-prefill chunks read BF16. An explicit qualification arm
// (`every_chunk`) loads the low-bit roles in place of BF16 and runs them in every chunk, so the gate scores the arm in
// prefill and in decode.
// `checkpoint`: the raw SHA-256 of the checkpoint (the load scope's), which a pack must be bound to.
struct MiniLowBitRows {
    const weight::Digest256* pack = nullptr;
    std::uint32_t band = 0;
    bool every_chunk = false;
    weight::Digest256 checkpoint{};
};

inline MiniLowBitRows mini_lowbit_rows(const weight::WeightLoadPlan& plan, const MiniWeightOptions& options,
                                       const weight::Digest256& checkpoint) {
    if (!mini_lowbit_storage(plan.storage) && !mini_lowbit_storage(plan.head_storage)) return {};
    if (options.qualify != nullptr) return {nullptr, 0U, true, checkpoint};
    return {&kMiniLowBitPackIdentity, plan.row_band(kMiniWeightRegime), false, checkpoint};
}

// A committed decision row exists for the policy this tier resolves (a TURBO row never makes FAST or EXACT compute the
// device scope at load).
inline bool mini_has_decision_rows(const MiniWeightOptions& options, profile::Name tier) {
    const auto& policy = mini_weight_load_policy(options, tier);
    return std::ranges::any_of(kMiniWeightLoadDecisions,
                               [&](const weight::WeightLoadDecision& row) { return weight::policy_has_id(policy, row.policy); });
}

template <class Value>
bool parse_mini_weight_option(MiniWeightOptions& options, std::string_view argument, const Value& value,
                              bool native) {
    if (argument == "--profile") {
        const std::string_view tier(value());
        if (tier != "exact" && tier != "fast" && tier != "turbo")
            throw std::invalid_argument("--profile must be exact, fast or turbo");
        options.tier.profile = profile::parse(tier);
    } else if (native && argument == "--qualify-q8") {
        options.qualify_q8 = true;
        options.qualify = &weight::kAttentionHeadQ8Policy;
    } else if (native && argument == "--qualify-weights") {
        const std::string_view key(value());
        options.qualify = weight::find_weight_load_policy(key, kMiniQualifiablePolicies);
        bool bound = false;
        for (const auto& binding : kMiniQualifiablePolicies)
            bound = bound || (options.qualify != nullptr && weight::same_id(binding.id, options.qualify->id));
        if (!bound) throw std::invalid_argument("--qualify-weights: unknown or unbound weight policy " + std::string(key));
        options.qualify_q8 = weight::same_id(options.qualify->id, kMiniWeightLoadPolicy.id);
    } else if (native && argument == "--no-weight-q8") {
        options.disable_q8 = true;
    } else {
        return false;
    }
    return true;
}

// Resolved operand class of the W4A4 expert lever (multi-row prefill chunks run NVFP4 expert layers through the shared
// grouped W4A4 class on CUDA): 0 = bf16 (off), 1 = nvfp4 (nearest@1), mse = nvfp4_mse, mse-gate-up = nvfp4_mse_gate_up.
// mse / mse-gate-up are candidate values of the route-flip re-qualification, not yet gated.
inline moe_core::grouped_w4a4::Activations mini_moe_w4a4_class(const profile::Resolution& tier) {
    using A = moe_core::grouped_w4a4::Activations;
    const auto* entry = tier.find(kMiniMoeW4a4);
    if (entry == nullptr || entry->value == "0") return A::bf16;
    if (entry->value == "1") return A::nvfp4;
    if (entry->value == "mse") return A::nvfp4_mse;
    if (entry->value == "mse-gate-up") return A::nvfp4_mse_gate_up;
    throw std::invalid_argument("STRIX_INK_MOE_W4A4 must be 0, 1, mse or mse-gate-up");
}

// Resolved state of the W4A4 expert lever (true: multi-row prefill chunks run NVFP4 expert layers as W4A4 on CUDA).
inline bool mini_moe_w4a4(const profile::Resolution& tier) {
    return mini_moe_w4a4_class(tier) != moe_core::grouped_w4a4::Activations::bf16;
}

inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution) {
    auto tier = profile::resolve(kMiniProfile, options.tier);
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    // A low-bit arm is a FAST candidate or, failing the FAST gate, a TURBO one.
    if (options.qualify != nullptr &&
        ((tier.profile != profile::Name::fast && tier.profile != profile::Name::turbo) || options.disable_q8 ||
         !native_execution))
        throw std::invalid_argument("--qualify-weights requires --profile fast or turbo and native model execution");
    static_cast<void>(mini_moe_w4a4(tier));  // refuse a malformed or unrunnable explicit switch early
    return tier;
}

inline const char* mini_profile_name(profile::Name tier) {
    return tier == profile::Name::standard ? "exact" : profile::name(tier);
}

// Admission of the calibrated pack a decision row was measured with, before the plan is committed: the file must open
// (framing, bounds, identity read through one open file), be bound to this checkpoint's bytes (scope.checkpoint) and
// carry the row's pack identity. Empty: admitted; otherwise why not.
inline std::string admit_mini_lowbit_pack(const std::string& file, const weight::Digest256& checkpoint) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(file, error)) return "no pack at " + file;
    weight::LowbitPack pack;
    if (const auto opened = weight::LowbitPack::open(file, checkpoint, pack); !opened) return opened.detail;
    if (!(pack.identity() == kMiniLowBitPackIdentity)) return file + " is not the measured pack (other content)";
    return {};
}

// `checkpoint` (when known): a decision row that selected low-bit roles applies only when its calibrated pack is
// admitted (admit_mini_lowbit_pack); otherwise the load keeps BF16, the plan's reason says so and `pack_note` (when
// given) receives the refusal detail for the receipt. No sidecar I/O happens for a plan without such a row (EXACT,
// lever off, no matching row). An explicit qualification arm keeps its failures explicit (the decoder throws).
inline weight::WeightLoadPlan resolve_mini_resident_weight_load(profile::Name tier, const MiniWeightOptions& options,
                                                              const weight::WeightLoadScope& scope,
                                                              const std::string& checkpoint = {},
                                                              std::string* pack_note = nullptr) {
    // Consumers of this backend's resident routes: the HIP projections and head also read the low-bit storages.
    constexpr std::array consumers{weight::kBf16WeightStorage, weight::kQ8WeightStorage,
                                   weight::kQ4G64WeightStorage, weight::kQ6G64WeightStorage};
    const std::span<const weight::FormatId> formats(consumers.data(), scope.backend == weight::Backend::hip ? 4U : 2U);
    const auto& policy = mini_weight_load_policy(options, tier);
    const auto resolve = [&](std::span<const weight::WeightLoadDecision> rows) {
        return resolve_mini_weight_load(tier, scope, formats, !options.disable_q8, options.qualify != nullptr, rows, policy);
    };
    auto plan = resolve(kMiniWeightLoadDecisions);
    if (options.qualify == nullptr && !checkpoint.empty() && mini_lowbit_rows(plan, options, scope.checkpoint).pack != nullptr) {
        if (auto refused = admit_mini_lowbit_pack(mini_lowbit_pack_file(checkpoint), scope.checkpoint); !refused.empty()) {
            plan = resolve({});
            plan.reason = "calibrated low-bit pack refused (strix-lowbit-pack/v2 line): BF16 image";
            if (pack_note) *pack_note = "refused " + std::move(refused) + "; BF16 image";
        }
    }
    if (options.qualify != nullptr && (plan.storage != policy.target || plan.head_storage != policy.head_target()))
        throw std::invalid_argument("the qualified compact weights were not admitted on this backend");
    return plan;
}

} // namespace strix::inkling2
