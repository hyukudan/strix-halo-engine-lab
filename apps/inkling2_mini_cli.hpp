#pragma once

#include "core/dense_linear.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/head_gate_cli.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::inkling2::cli {

struct Options {
    serve::SerialGenerationOptions generation;
    const char* dspark{};
    const char* gate_manifest{};
    const char* gate_out{};
    const char* gate_arm{};
    unsigned gate_rows{};
    std::optional<session::HeadGateMode> gate_mode;
    std::optional<bool> step_graphs;
    bool qualify_q8{};
    bool disable_q8{};
    bool gate_order_null{};
    profile::Resolution tier;
};

inline Options parse(int argc, char** argv, const serve::SerialGenerationFamily& family) {
    Options options;
    profile::Inputs tier_inputs;
    options.generation = serve::parse_serial_generation_cli(argc, argv, family,
        [&](std::string_view argument, const auto& value) {
            if (argument == "--profile") {
                const std::string_view tier(value());
                if (tier != "exact" && tier != "fast" && tier != "turbo")
                    throw std::invalid_argument("--profile must be exact, fast or turbo");
                tier_inputs.profile = profile::parse(tier);
                return true;
            }
            if (argument == "--dspark") {
                options.dspark = value();
                return true;
            }
            if (!family.default_prefill_chunk) return false;
            if (argument == "--step-graph" || argument == "--no-step-graph")
                options.step_graphs = argument == "--step-graph";
            else if (argument == "--qualify-q8") options.qualify_q8 = true;
            else if (argument == "--no-weight-q8") options.disable_q8 = true;
            else if (argument == "--gate-manifest") options.gate_manifest = value();
            else if (argument == "--gate-out") options.gate_out = value();
            else if (argument == "--gate-arm") options.gate_arm = value();
            else if (argument == "--gate-order-null") options.gate_order_null = true;
            else if (argument == "--gate-rows") {
                options.gate_rows = core::parse_u32(value());
                if (options.gate_rows < 2U) throw std::invalid_argument("--gate-rows must be at least 2");
            } else if (argument == "--gate-mode") {
                const std::string_view mode(value());
                if (mode == "prefill") options.gate_mode = session::HeadGateMode::prefill;
                else if (mode == "decode") options.gate_mode = session::HeadGateMode::decode;
                else throw std::invalid_argument("--gate-mode must be prefill or decode");
            } else {
                return false;
            }
            return true;
        });
    const bool gate = options.gate_manifest || options.gate_out || options.gate_arm ||
                      options.gate_rows != 0U || options.gate_mode.has_value();
    if (gate && !(options.gate_manifest && options.gate_out && options.gate_arm &&
                  options.gate_rows != 0U && options.gate_mode))
        throw std::invalid_argument("gate mode requires --gate-manifest, --gate-out, --gate-arm, --gate-rows and --gate-mode");
    if (gate && !session::valid_head_gate_name(options.gate_arm))
        throw std::invalid_argument("invalid --gate-arm name");
    const auto& generation = options.generation;
    if (options.dspark) {
        if (gate || generation.tokenize_file || generation.speculation != serve::CliSpeculation::automatic ||
            token_sampling_enabled(generation.sampling))
            throw std::invalid_argument("--dspark requires greedy --speculation auto with model generation");
        if (generation.logits && std::filesystem::exists(generation.logits) && std::filesystem::exists(options.dspark) &&
            std::filesystem::equivalent(generation.logits, options.dspark))
            throw std::invalid_argument("logits output must differ from the DSpark sidecar");
    }
    if (gate && (!generation.prompt.empty() || generation.text || generation.messages ||
                 generation.tokenize_file || generation.generate_requested || generation.logits ||
                 generation.sampling_requested))
        throw std::invalid_argument("gate mode takes no prompt, --tokenize, --generate, --logits or sampling options");
    options.tier = profile::resolve(kMiniProfile, tier_inputs);
    options.tier.weights.require_honoured();
    if (options.qualify_q8 && (options.tier.profile != profile::Name::fast || options.disable_q8 ||
                               generation.tokenize_file))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    if (options.gate_order_null && (!gate || options.tier.profile != profile::Name::standard))
        throw std::invalid_argument("--gate-order-null requires an exact gate arm");
    return options;
}

inline void write_profile(const profile::Resolution& tier, std::ostream& output, bool device) {
    auto levers = " none";
    output << "profile " << (tier.profile == profile::Name::standard ? "exact" : profile::name(tier.profile))
           << " levers";
    for (const auto& entry : tier.entries) {
        if (entry.value == entry.lever->builtin) continue;
        output << ' ' << entry.lever->name << '=' << entry.value;
        levers = "";
    }
    output << levers << (device ? " reference BASE\n" : "\n");
}

inline void write_weight_load_plan(std::ostream& output, const weight::WeightLoadPlan& plan) {
    output << "strix-weight-load/v1 " << plan.policy.name << '@' << plan.policy.revision
           << ' ' << plan.storage.name << '@' << plan.storage.revision << '\n';
}

}  // namespace strix::inkling2::cli
