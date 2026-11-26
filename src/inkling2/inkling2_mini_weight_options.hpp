#pragma once

#include "core/resident_row_plan.hpp"
#include "inkling2_mini_resident_route.hpp"
#include "moe_core/grouped_w4a4.hpp"
#include "strix/env_flags.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/serve/row_budget.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/resolve.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::inkling2 {

struct MiniWeightOptions {
    profile::Inputs tier;
    bool qualify_q8{};  // the Q8_0 qualification arm (--qualify-q8, the alias of --qualify-weights <Q8 policy>)
    bool disable_q8{};  // off switch of every compact weight storage
    // The shared policy of an explicit qualification arm (--qualify-weights KEY); nullptr: none.
    const weight::WeightLoadPolicy* qualify{};
    bool qualify_decode_only{};
    bool lowbit_turbo{};  // resolved from the shared profile lever
};

// The bindings a qualification key may name: the model's Q8_0 binding (legacy id included), the shared low-bit mixes and
// the shared head-only low-bit rows (attention BF16).
inline constexpr std::array<weight::WeightLoadPolicy, 7> kMiniQualifiablePolicies{
    kMiniWeightLoadPolicy, weight::kAttentionQ8Policy, weight::kAttentionHeadQ6Policy, weight::kAttentionQ4HeadQ6Policy,
    weight::kAttentionHeadQ4Policy, weight::kHeadQ6Policy, weight::kHeadQ4Policy};

inline const weight::WeightLoadPolicy& mini_weight_load_policy(const MiniWeightOptions& options) {
    if (options.qualify == nullptr || weight::same_id(options.qualify->id, kMiniWeightLoadPolicy.id))
        return kMiniWeightLoadPolicy;
    return *options.qualify;
}

// The PR's calibrated and qualification routes used the shared FP16 tile arithmetic.
inline constexpr auto kMiniLowBitArithmetic = core::LowBitArithmetic::f16_group_scale;
inline constexpr auto& kMiniLowBitConsumers = core::kResidentLowBitTileArithmeticHip;

inline bool mini_lowbit_storage(weight::FormatId f) {
    return f == weight::kQ4G64WeightStorage || f == weight::kQ6G64WeightStorage;
}

// The native load binds this only when the profile has timing rows. Each append then supplies the workload
// it actually launches. The same decoder supplies the key on the CLI and in a served cohort.
struct MiniLowBitTiming {
    profile::Inputs inputs;
    std::string device, product, weights;
    std::uint32_t capacity;

    core::LowBitArithmetic select(profile::Phase phase, std::uint64_t context, unsigned lanes, unsigned rows,
                                  unsigned chunk, std::string_view route) const {
        auto request = inputs;
        // A receipt prices one prompt chunk or one decode pass over an already resident context.
        const auto execution = profile::ladder_execution({"hip", product,
            {context, capacity, lanes, chunk, phase == profile::Phase::prefill ? rows : 0U,
             phase == profile::Phase::decode ? rows : 0U, 0U, phase == profile::Phase::decode ? 1U : 0U},
            weights + std::string(route)});
        request.regime = {phase, context, lanes, device, execution};
        return core::resident_lowbit_arithmetic(profile::resolve(kMiniProfile, request));
    }
};

// The decode-only qualification arm keeps the incumbent BF16 tensors beside the compact
// tensors. Ordinary weight qualification replaces the selected tensors.
struct MiniLowBitRows {
    bool decode_only = false;
    std::shared_ptr<const weight::LowbitPack> admitted{};
    core::LowBitArithmetic arithmetic = kMiniLowBitArithmetic;
    std::optional<MiniLowBitTiming> timing{};
};

inline weight::WeightLoadPlan mini_lowbit_receipt_plan(
    profile::Name tier, const weight::WeightLoadScope& scope, weight::WeightLoadExtent extent,
    std::span<const weight::WeightLoadDecision> decisions = kMiniLowBitDecisions) {
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ4G64WeightStorage};
    const auto& policy = tier == profile::Name::turbo ? kMiniTurboWeightPolicy : kMiniFastWeightPolicy;
    const bool enabled = scope.backend == weight::Backend::hip && scope.regime == kMiniWeightRegime &&
                         extent.context != 0U && extent.lanes != 0U;
    return weight::resolve_weight_load(weight::builtin_registry(), policy, {tier, scope, formats, enabled, false,
                                                                             {}, extent, {1, 1}}, decisions);
}

inline bool mini_lowbit_turbo(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniLowBitTurbo);
    if (entry == nullptr || entry->value == "0") return false;
    if (entry->value != "1") throw std::invalid_argument("STRIX_INK_LOWBIT_TURBO must be 0 or 1");
    return tier.profile == profile::Name::turbo;
}

inline bool mini_may_load_lowbit_row(profile::Name tier, const MiniWeightOptions& options,
                                    weight::Backend backend, weight::WeightLoadExtent extent) {
    return tier == profile::Name::turbo && options.lowbit_turbo && !options.disable_q8 &&
           !options.qualify && !options.qualify_q8 &&
           std::ranges::any_of(kMiniLowBitDecisions, [backend, extent](const auto& row) {
               return backend == row.scope.backend && weight::weight_load_gain_selects(row, extent);
           });
}

// The HTTP held-back route sends each prompt's last token through decode_slots. It is still prefill work and must use
// the BF16 route used to qualify the calibrated row. Only an emits-first served route can admit that row safely.
inline constexpr bool mini_server_calibrated_row_allowed(bool requested, bool emits_first) noexcept {
    return requested && emits_first;
}

// Lookup and drafter speculation must verify against the same incumbent target for every verify width. Until the
// calibrated target supports both serial and multi-row verification, keep the whole speculative run on incumbent weights.
inline constexpr bool mini_calibrated_row_allowed_for_cli(bool requested, bool speculation_requested,
                                                          bool drafter_requested) noexcept {
    return requested && !speculation_requested && !drafter_requested;
}

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
        options.qualify_decode_only = false;
        options.qualify_q8 = true;
        options.qualify = &weight::kAttentionHeadQ8Policy;
    } else if (native && (argument == "--qualify-weights" || argument == "--qualify-decode-weights")) {
        options.qualify_decode_only = argument == "--qualify-decode-weights";
        const std::string_view key(value());
        options.qualify = weight::find_weight_load_policy(key, kMiniQualifiablePolicies);
        const bool strixpack = options.qualify && weight::is_strixpack_storage(options.qualify->target);
        bool bound = strixpack;
        for (const auto& binding : kMiniQualifiablePolicies)
            bound = bound || (options.qualify != nullptr && weight::same_id(binding.id, options.qualify->id));
        if (!bound) throw std::invalid_argument("--qualify-weights: unknown or unbound weight policy " + std::string(key));
        options.qualify_q8 = weight::same_id(options.qualify->id, kMiniWeightLoadPolicy.id);
        if (options.qualify_decode_only && !strixpack && !mini_lowbit_storage(options.qualify->head_target()))
            throw std::invalid_argument("--qualify-decode-weights requires a low-bit head");
    } else if (native && argument == "--no-weight-q8") {
        options.disable_q8 = true;
    } else {
        return false;
    }
    return true;
}

// Resolved operand class of the W4A4 expert lever (multi-row prefill chunks run NVFP4 expert layers through the shared
// grouped W4A4 class on CUDA): 0 = bf16 (off), 1 = nvfp4 (nearest@1), mse = nvfp4_mse, mse-gate-up = nvfp4_mse_gate_up,
// mse-r2-gate-up = nvfp4_mse_r2_gate_up, mse-r2 = nvfp4_mse_r2 (two-term activation rule r2@1 of moe_core/nvfp4_act_rule.hpp),
// r2 = nvfp4_r2 (two nearest@1 terms on both projections).
// Every value but 0 / 1 is a candidate of the route-flip re-qualification, not yet gated.
inline moe_core::grouped_w4a4::Activations mini_moe_w4a4_class(const profile::Resolution& tier) {
    using A = moe_core::grouped_w4a4::Activations;
    const auto* entry = tier.find(kMiniMoeW4a4);
    if (entry == nullptr || entry->value == "0") return A::bf16;
    if (entry->value == "1") return A::nvfp4;
    if (entry->value == "mse") return A::nvfp4_mse;
    if (entry->value == "mse-gate-up") return A::nvfp4_mse_gate_up;
    if (entry->value == "mse-r2-gate-up") return A::nvfp4_mse_r2_gate_up;
    if (entry->value == "mse-r2") return A::nvfp4_mse_r2;
    if (entry->value == "r2") return A::nvfp4_r2;
    throw std::invalid_argument("STRIX_INK_MOE_W4A4 must be 0, 1, mse, mse-gate-up, mse-r2-gate-up, mse-r2 or r2");
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

// Resolved state of the screened-head lever (true: CUDA greedy picks without a score row run the EXACT screened head;
// the same token and status as the full head). One value for the CLI, the gate and the server.
inline bool mini_screened_head(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniScreenedHead);
    if (entry == nullptr || entry->value == "0") return false;
    if (entry->value == "1") return true;
    throw std::invalid_argument("STRIX_INK_SCREENED_HEAD must be 0 or 1");
}

// Resolved level of the verify rows lever (kMiniVerifyRows). 0 or absent = off: the per-row incumbent. 1: a HIP verify
// window takes the pitched rows head, one lanes attention launch set and the convolution tape. 2: level 1, and greedy
// picks whose scores nobody reads (verify windows without a sampled or observing reader, served cohorts whose groups
// declare no score read) go through the lanes form of the EXACT screened head (kMiniVerifyLanesPick).
inline constexpr unsigned kMiniVerifyLanesPick = 2U;
inline unsigned mini_verify_rows(const profile::Resolution& tier) {
    const auto* entry = tier.find(kMiniVerifyRows);
    if (entry == nullptr || entry->value == "0") return 0U;
    if (entry->value == "1") return 1U;
    if (entry->value == "2") return kMiniVerifyLanesPick;
    throw std::invalid_argument("STRIX_INK_VERIFY_ROWS must be 0, 1 or 2");
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
// Effective verify rows level against the selected one: off, on (level 1), on+lanes-pick (level 2), declined (nothing
// admitted), on+lanes-pick-declined (level 2 selected, the decoder has no screened head: compact head or no screen).
inline const char* mini_verify_rows_name(unsigned active, unsigned selected) {
    if (active == 0U) return selected == 0U ? "off" : "declined";
    if (active >= kMiniVerifyLanesPick) return "on+lanes-pick";
    return selected >= kMiniVerifyLanesPick ? "on+lanes-pick-declined" : "on";
}
// The level a server runs of the decoder's admitted one (its startup receipt): the server stages no verify window, so
// its lanes pick screens only cohorts of two or more distinct admitted slots on the incumbent served route. The served
// CLI route (packed passes, decode_slots_served) and a one-slot server run level 1 (on+lanes-pick-declined); a one-slot
// command line keeps the decoder's level (its verify windows screen through the lanes pick).
inline unsigned mini_served_verify_rows(unsigned active, bool served_cli_route, std::size_t admitted_slots) {
    return active >= kMiniVerifyLanesPick && (served_cli_route || admitted_slots < 2U) ? 1U : active;
}

// What the decoder's backend admitted of the resolved levers: the effective state every entry point reports.
struct MiniDecoderLevers {
    bool down_image = false;           // the BF16 down image route is active
    bool prepared_round_fold = false;  // the prepared-attention round fold runs
    MiniHead head{};                   // the greedy head the decoder runs (screened-head lever) and why
    unsigned verify_rows = 0U;         // verify rows level admitted (mini_verify_rows: 0 off, 1 rows, 2 rows + lanes pick)
};

// The one place a resident decoder takes its resolved profile levers; the serial CLI, the gate and the server all call
// it, so a lever cannot reach one entry point and not the others. Returns what the backend admitted (a lever the target
// does not run is declined, not silently reported as on).
template <class Decoder>
MiniDecoderLevers configure_mini_decoder(Decoder& decoder, moe_core::grouped_w4a4::Activations moe_w4a4,
                                         bool combine_image, bool prepared_round_fold, bool attn_projected_bias,
                                         bool screened_head, unsigned verify_rows) {
    decoder.set_moe_w4a4(moe_w4a4);
    decoder.set_attention_projected_bias(attn_projected_bias);
    MiniDecoderLevers levers;
    levers.prepared_round_fold = decoder.set_prepared_round_fold(prepared_round_fold);
    levers.down_image = decoder.set_combine_image(combine_image);
    levers.verify_rows = decoder.set_verify_rows(verify_rows);
    levers.head = decoder.set_screened_head(screened_head);
    return levers;
}

template <class Decoder>
MiniDecoderLevers configure_mini_decoder(Decoder& decoder, const profile::Resolution& tier) {
    return configure_mini_decoder(decoder, mini_moe_w4a4_class(tier), mini_combine_image(tier),
                                  mini_prepared_round_fold(tier), mini_attn_projected_bias(tier),
                                  mini_screened_head(tier), mini_verify_rows(tier));
}

// The W4A4 class a run on `backend` ("hip", "cuda") executes. Only the CUDA backend holds the grouped W4A4 tile: a
// selected class that the backend cannot run is refused, never replaced by the BF16 tile under the class's name. Every
// entry point (generation CLI, gate, step program, server) admits through here before it reports or binds the class.
inline moe_core::grouped_w4a4::Activations admit_mini_moe_w4a4(moe_core::grouped_w4a4::Activations activations,
                                                                std::string_view backend) {
    if (activations != moe_core::grouped_w4a4::Activations::bf16 && backend != "cuda")
        throw std::invalid_argument(std::string("STRIX_INK_MOE_W4A4 selects ") +
                                    moe_core::grouped_w4a4::identity(activations) + ": the " + std::string(backend) +
                                    " backend has no grouped W4A4 tile (CUDA only); select 0 to run the BF16 tile");
    return activations;
}

inline moe_core::grouped_w4a4::Activations admit_mini_moe_w4a4(const profile::Resolution& tier, std::string_view backend) {
    return admit_mini_moe_w4a4(mini_moe_w4a4_class(tier), backend);
}

// Backend of this translation unit's device build, for the entry points compiled once per backend.
#if defined(__HIPCC__)
inline constexpr std::string_view kMiniBuildBackend = "hip";
#else
inline constexpr std::string_view kMiniBuildBackend = "cuda";
#endif

inline profile::Resolution resolve_mini_weight_profile(const MiniWeightOptions& options, profile::Inputs inputs) {
    if (options.qualify && weight::is_strixpack_storage(options.qualify->target)) {
        const auto entry = std::ranges::find_if(kMiniProfile.table, [&](const profile::Lever& lever) {
            return weight::id_has_key(options.qualify->id, lever.name);
        });
        if (entry != kMiniProfile.table.end()) inputs.cli.emplace_back(entry->name, entry->target);
    }
    return profile::resolve(kMiniProfile, inputs);
}

inline std::optional<weight::StrixPackLoad> admit_mini_strixpack_receipt(profile::Resolution& tier) {
    const auto load = weight::bind_strixpack_load(tier);
    if (!tier.allow_pending)
        std::erase_if(tier.entries, [](const profile::Entry& entry) {
            const auto* policy = entry.lever ? weight::find_weight_load_policy(entry.lever->name) : nullptr;
            return policy && weight::is_strixpack_storage(policy->target) && entry.reason.empty();
        });
    return load;
}

inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution) {
    auto tier = resolve_mini_weight_profile(options, options.tier);
    // Native CLI (including the CUDA step program), gate and server admit the weight request before routing.
    if (const auto load = admit_mini_strixpack_receipt(tier); native_execution && load)
        load->require_loadable();
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    // A low-bit arm is a FAST candidate or, failing the FAST gate, a TURBO one.
    if (options.qualify != nullptr &&
        ((tier.profile != profile::Name::fast && tier.profile != profile::Name::turbo) || options.disable_q8 ||
         !native_execution))
        throw std::invalid_argument("--qualify-weights requires --profile fast or turbo and native model execution");
    static_cast<void>(mini_moe_w4a4(tier));  // refuse a malformed or unrunnable explicit switch early
    static_cast<void>(mini_prepared_round_fold(tier));
    static_cast<void>(mini_combine_image(tier));
    static_cast<void>(mini_attn_projected_bias(tier));
    static_cast<void>(mini_screened_head(tier));
    static_cast<void>(mini_verify_rows(tier));
    mini_defer_vision_recipe(tier);  // image input only: re-admitted and checked by the image opener
    require_mini_expert_consumer(tier, native_execution);
    static_cast<void>(mini_controller_one_plane(tier));
    static_cast<void>(mini_lowbit_turbo(tier));
    if (core::resident_lowbit_arithmetic(tier) != kMiniLowBitArithmetic && !native_execution)
        throw std::invalid_argument("the low-bit arithmetic candidate requires native HIP execution");
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

// One scoped selection for serial CLI, the gate and each served prompt. Explicit
// selection was already validated during parsing and keeps its precedence. `receipt` (the command line's resolution)
// declines the served route lever as the command line's own resolution does.
inline moe_core::grouped_w4a4::Activations mini_prompt_moe(const MiniWeightOptions& options,
                                                         std::string_view device, std::uint64_t tokens,
                                                         bool print = true, profile::Resolution* receipt = nullptr) {
    auto inputs = options.tier;
    inputs.regime = {profile::Phase::prefill, tokens, 1, device};
    auto selected = resolve_mini_weight_profile(options, inputs);
    admit_mini_strixpack_receipt(selected);
    if (print)
        for (const auto& entry : selected.entries)
            if (!entry.ladder.empty()) std::cerr << entry.ladder << '\n';
    const auto arithmetic = admit_mini_moe_w4a4(mini_moe_w4a4_class(selected), kMiniBuildBackend);
    if (receipt) {
        mini_defer_vision_recipe(selected);
        mini_cli_decline_served_route(selected);
        *receipt = std::move(selected);
    }
    return arithmetic;
}

inline const char* mini_profile_name(profile::Name tier) {
    return tier == profile::Name::standard ? "exact" : profile::name(tier);
}

// Open once against the checkpoint identity. The same source is used for tensor validation
// and staging, which retains the shared stale-source checks.
inline std::string admit_mini_lowbit_pack(const std::string& file, const weight::Digest256& checkpoint,
                                          std::shared_ptr<const weight::LowbitPack>& admitted,
                                          const weight::Digest256* measured = nullptr) {
    admitted.reset();
    auto pack = std::make_shared<weight::LowbitPack>();
    if (const auto opened = weight::LowbitPack::open(file, checkpoint, *pack); !opened) return opened.detail;
    if (measured && !(pack->identity() == *measured)) return "pack content does not match the receipt";
    admitted = std::move(pack);
    return {};
}

inline weight::WeightLoadPlan resolve_mini_resident_weight_load(profile::Name tier, const MiniWeightOptions& options,
                                                              const weight::WeightLoadScope& scope,
                                                              weight::WeightLoadExtent extent = {}) {
    auto inputs = options.tier;
    inputs.profile = tier;
    auto resolved = resolve_mini_weight_profile(options, std::move(inputs));
    const auto strixpack = weight::bind_strixpack_load(resolved);
    if (strixpack && strixpack->refused) {
        weight::WeightLoadPlan refused{strixpack->policy, weight::kBf16WeightStorage, profile::Status::pending,
                                       strixpack->reason()};
        refused.strixpack = strixpack;
        return refused;
    }
    constexpr std::array consumers{weight::kBf16WeightStorage, weight::kQ8WeightStorage,
                                   weight::kQ4G64WeightStorage, weight::kQ6G64WeightStorage};
    const std::span<const weight::FormatId> formats(consumers.data(), scope.backend == weight::Backend::hip ? 4U : 2U);
    const auto& policy = mini_weight_load_policy(options);
    const bool qualify = options.qualify_q8 || options.qualify != nullptr;
    if (mini_may_load_lowbit_row(tier, options, scope.backend, extent)) {
        auto candidate = mini_lowbit_receipt_plan(tier, scope, extent);
        if (mini_lowbit_storage(candidate.head_storage)) {
            candidate.strixpack = strixpack;
            return candidate;
        }
    }
    // The resident image serves every request from one position and one lane up to extent.
    auto plan = weight::resolve_weight_load(weight::builtin_registry(), policy,
        {tier, scope, formats, !options.disable_q8, qualify, kMiniWeightLoadRegimes, extent, {1, 1}}, kMiniWeightLoadDecisions);
    if (qualify && (plan.storage != policy.target || plan.head_storage != policy.head_target()))
        throw std::invalid_argument("the qualified compact weights were not admitted on this backend");
    plan.strixpack = strixpack;
    return plan;
}

inline MiniLowBitRows open_mini_weight_pack(const std::string& file, const weight::Digest256& checkpoint,
                                           bool decode_only, const weight::Digest256* measured = nullptr) {
    MiniLowBitRows rows{decode_only, {}};
    if (auto detail = admit_mini_lowbit_pack(file, checkpoint, rows.admitted, measured); !detail.empty())
        throw std::runtime_error("calibrated low-bit pack refused: " + detail);
    return rows;
}

inline weight::WeightLoadPlan mini_plain_weight_fallback(profile::Name tier, const MiniWeightOptions& options,
                                                        const weight::WeightLoadScope& scope,
                                                        weight::WeightLoadExtent extent, std::string_view reason) {
    auto incumbent = options;
    incumbent.qualify = nullptr;
    incumbent.qualify_q8 = false;
    incumbent.qualify_decode_only = false;
    incumbent.lowbit_turbo = false;
    auto plan = resolve_mini_resident_weight_load(tier, incumbent, scope, extent);
    plan.reason = reason;
    return plan;
}

// Called after the resident image and first slot are allocated; free bytes already account for
// both BF16 and any compact copies. A calibrated load must retain the requested slot capacity.
inline void require_mini_weight_slot_capacity(std::uint32_t slots, std::uint32_t requested, bool calibrated) {
    if (calibrated && slots < requested)
        throw std::runtime_error("calibrated image cannot retain the requested slot capacity");
    if (!slots) throw std::runtime_error("Inkling KV budget holds no session slots");
}

inline std::uint32_t mini_weight_slot_capacity(std::uint64_t free, std::uint64_t total, std::uint64_t device_limit,
                                               std::uint64_t kv_limit, std::uint64_t slot_bytes,
                                               std::uint32_t requested, bool calibrated) {
    const auto slots = serve::resident_kv_slots(free, total, device_limit, kv_limit, slot_bytes, requested);
    require_mini_weight_slot_capacity(slots, requested, calibrated);
    return slots;
}

// Keep the admitted handle through staging and verification. Destroy any failed image before
// rebuilding the incumbent, and publish the fallback plan with the weights that actually loaded.
template <class Load>
auto load_mini_weight_pack(weight::WeightLoadPlan& plan, profile::Name tier, const MiniWeightOptions& options,
                           const weight::WeightLoadScope& incumbent_scope, weight::WeightLoadExtent extent,
                           const std::string& path, const weight::Digest256* measured, std::ostream& receipt, Load load) {
    plan.require_loadable();
    const bool decode_only = measured || options.qualify_decode_only;
    if (path.empty() && !decode_only) return load(MiniLowBitRows{});
    try {
        if (path.empty()) throw std::runtime_error("--qualify-decode-weights requires STRIX_LOWBIT_PACK");
        const auto rows = open_mini_weight_pack(path, incumbent_scope.checkpoint, decode_only, measured);
        auto loaded = load(rows);
        if (const auto checked = rows.admitted->verify_identity(); !checked) throw std::runtime_error(checked.detail);
        return loaded;
    } catch (const std::runtime_error& error) {
        receipt << "strix-lowbit-pack/v2 declined (" << error.what() << "); retrying incumbent weights\n";
        plan = mini_plain_weight_fallback(tier, options, incumbent_scope, extent,
                                          "calibrated pack declined; incumbent weights");
        return load(MiniLowBitRows{});
    }
}

} // namespace strix::inkling2
