#pragma once

#include "moe_core/grouped_w4a4.hpp"
#include "strix/profiles/inkling2_mini.hpp"

#include <cstdint>
#include <string>

namespace strix::inkling2 {

struct MiniWeightOptions {
    profile::Inputs tier;
    bool qualify_q8{};
    bool disable_q8{};
};

template <class Value>
bool parse_mini_weight_option(MiniWeightOptions& options, std::string_view argument, const Value& value,
                              bool native) {
    if (argument == "--profile") {
        const std::string_view tier(value());
        if (tier != "exact" && tier != "fast" && tier != "turbo")
            throw std::invalid_argument("--profile must be exact, fast or turbo");
        options.tier.profile = profile::parse(tier);
    } else if (native && argument == "--controller-planes") {
        options.tier.cli.emplace_back(kMiniControllerPlanes, value());
    } else if (argument == "--block-drafter") {
        options.tier.cli.emplace_back(std::string(spec::kBlockDraftLeverName), value());
    } else if (native && argument == "--qualify-q8") {
        options.qualify_q8 = true;
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

// Resolved state of the prepared-attention round fold (HIP: the stage's plain BF16 rounds fold into their producers,
// same bits; off unless STRIX_INK_PREPARED_ROUND_FOLD=1 selects it, in any profile). The serial CLI and the server
// read it from the same resolution.
inline bool mini_prepared_round_fold(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniPreparedRoundFold);
    if (entry == nullptr || entry->value == "0") return false;
    if (entry->value == "1") return true;
    throw std::invalid_argument("STRIX_INK_PREPARED_ROUND_FOLD must be 0 or 1");
}

// Resolved state of the narrow expert image lever (true: the checkpoint's MXFP8 expert layers load as NVFP4).
inline bool mini_experts_nvfp4(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniExpertsNvfp4);
    if (entry == nullptr || entry->value == "0") return false;
    if (entry->value == "1") return true;
    throw std::invalid_argument("STRIX_MOE_EXPERTS_NVFP4 must be 0 or 1");
}

inline void require_mini_expert_consumer(const profile::Resolution& tier, bool supports_narrow_experts) {
    if (!mini_experts_nvfp4(tier)) return;
    if (tier.profile == profile::Name::standard)
        throw std::invalid_argument("STRIX_MOE_EXPERTS_NVFP4=1 is forbidden under --profile exact");
    if (!supports_narrow_experts)
        throw std::invalid_argument("STRIX_MOE_EXPERTS_NVFP4=1 requires native resident model execution");
}

// Resolved state of the exact down-image lever (true: the HIP resident prefill stores the BF16 down image on the qualified
// target). 0 or absent = off.
inline bool mini_combine_image(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniCombineImage);
    if (entry == nullptr || entry->value == "0") return false;
    if (entry->value == "1") return true;
    throw std::invalid_argument("STRIX_INK_COMBINE_IMAGE must be 0 or 1");
}

// Resolved state of the projected-bias attention lever (true: the CUDA prompt flash projects the relative bias; bitwise the
// image route). One value for the CLI, the gate and the server: every entry point reads it from the resolution.
inline bool mini_attn_projected_bias(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniAttnProjectedBias);
    if (entry == nullptr || entry->value == "0") return false;
    if (entry->value == "1") return true;
    throw std::invalid_argument("STRIX_INK_ATTN_PROJECTED_BIAS must be 0 or 1");
}

// The observed route of the projected-bias lever, one line for both entry points (with the lever selected): the
// projected-bias appends the decoder submitted so far (0: every prompt chunk took the image route, whose bits are
// equal). The command line prints it after each prefill, the server once when it stops.
inline std::string mini_projected_bias_dispatches_line(std::uint64_t dispatches) {
    return "Inkling attn_projected_bias dispatches " + std::to_string(dispatches);
}

// Resolved state of the served route lever (true: a HIP server runs the command line's route, kMiniServedCliRoute).
// 0 or absent = off: the served incumbent route. Read (and validated) by the server only: the command line always runs
// its own route and reports the lever as not applicable (mini_served_route_cli_line).
inline bool mini_served_cli_route(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniServedCliRoute);
    if (entry == nullptr || entry->value == "0") return false;
    if (entry->value == "1") return true;
    throw std::invalid_argument("STRIX_INK_SERVED_CLI_ROUTE must be 0 or 1");
}

// The command line's receipt of the served route lever: the requested value as the resolution holds it (unparsed, so a
// malformed server setting never aborts the command line) and that it does not apply here.
inline std::string mini_served_route_cli_line(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniServedCliRoute);
    const std::string_view requested = entry == nullptr ? std::string_view("0")
                                       : entry->requested.empty() ? std::string_view(entry->value)
                                                                  : std::string_view(entry->requested);
    return "strix-served-route/v1 requested " + std::string(requested) +
           " effective n/a (strix-serve only; the command line runs its own route)";
}

inline bool mini_controller_one_plane(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniControllerPlanes);
    if (entry == nullptr || entry->value == "3") return false;
    if (entry->value != "1") throw std::invalid_argument("controller planes must be 1 or 3");
    if (tier.profile != profile::Name::fast && tier.profile != profile::Name::turbo)
        throw std::invalid_argument("one controller plane requires --profile fast or turbo");
    return true;
}

// "on", "off" or "declined" (selected, but the target does not run it): the state name of a lever in the startup lines
// (down_image, prepared_round_fold), the same text in the CLI, the gate and the server.
inline const char* mini_down_image_name(bool active, bool selected) { return active ? "on" : selected ? "declined" : "off"; }
inline const char* mini_round_fold_name(bool active, bool selected) { return mini_down_image_name(active, selected); }

// What the decoder's backend admitted of the resolved levers: the effective state every entry point reports.
struct MiniDecoderLevers {
    bool down_image = false;           // the BF16 down image route is active
    bool prepared_round_fold = false;  // the prepared-attention round fold runs
};

// The one place a resident decoder takes its resolved profile levers; the serial CLI, the gate and the server all call
// it, so a lever cannot reach one entry point and not the others. Returns what the backend admitted (a lever the target
// does not run is declined, not silently reported as on).
template <class Decoder>
MiniDecoderLevers configure_mini_decoder(Decoder& decoder, moe_core::grouped_w4a4::Activations moe_w4a4,
                                         bool combine_image, bool prepared_round_fold, bool attn_projected_bias) {
    decoder.set_moe_w4a4(moe_w4a4);
    decoder.set_attention_projected_bias(attn_projected_bias);
    MiniDecoderLevers levers;
    levers.prepared_round_fold = decoder.set_prepared_round_fold(prepared_round_fold);
    levers.down_image = decoder.set_combine_image(combine_image);
    return levers;
}

template <class Decoder>
MiniDecoderLevers configure_mini_decoder(Decoder& decoder, const profile::Resolution& tier) {
    return configure_mini_decoder(decoder, mini_moe_w4a4_class(tier), mini_combine_image(tier),
                                  mini_prepared_round_fold(tier), mini_attn_projected_bias(tier));
}


inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution) {
    auto tier = profile::resolve(kMiniProfile, options.tier);
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    static_cast<void>(mini_moe_w4a4(tier));  // refuse a malformed or unrunnable explicit switch early
    static_cast<void>(mini_prepared_round_fold(tier));
    static_cast<void>(mini_combine_image(tier));
    static_cast<void>(mini_attn_projected_bias(tier));
    require_mini_expert_consumer(tier, native_execution);
    static_cast<void>(mini_controller_one_plane(tier));
    spec::require_block_draft_execution(spec::block_draft_selection(tier));
    return tier;
}

// The command line's resolution never runs the served route lever (its own route is the command line's): an explicit
// request stays visible as `requested` with the reason, and its effective value is the builtin, so the native summary,
// qualification and fingerprint carry no server-only setting (mini_served_route_cli_line prints the request).
inline void mini_cli_decline_served_route(profile::Resolution& tier) {
    for (auto& entry : tier.entries) {
        if (entry.lever == nullptr || entry.lever->name != kMiniServedCliRoute) continue;
        if (entry.value == entry.lever->builtin && entry.requested.empty()) continue;
        if (entry.requested.empty()) entry.requested = entry.value;
        entry.value = std::string(entry.lever->builtin);
        entry.status = profile::Status::off;
        entry.reason = "strix-serve only: the command line runs its own route";
    }
}

inline const char* mini_profile_name(profile::Name tier) {
    return tier == profile::Name::standard ? "exact" : profile::name(tier);
}

// The one resolver of the CLI and the server: `extent` is what the instance holds (resident context, decode lanes;
// see resolve_mini_weight_load); the receipt line weight::weight_load_regime_receipt says which regime rows are active.
inline weight::WeightLoadPlan resolve_mini_resident_weight_load(profile::Name tier, const MiniWeightOptions& options,
                                                              const weight::WeightLoadScope& scope,
                                                              weight::WeightLoadExtent extent) {
    constexpr std::array consumers{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    const auto plan = resolve_mini_weight_load(tier, scope, consumers, !options.disable_q8, options.qualify_q8,
                                               kMiniWeightLoadDecisions, extent);
    if (options.qualify_q8 && plan.storage != weight::kQ8WeightStorage)
        throw std::invalid_argument("native Q8_0 weights were not admitted");
    return plan;
}

} // namespace strix::inkling2
