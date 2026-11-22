#pragma once

#include "core/dense_linear.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
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
    MiniWeightOptions weights;
    bool gate_order_null{};
    profile::Resolution tier;
};

inline Options parse(int argc, char** argv, const serve::SerialGenerationFamily& family) {
    Options options;
    options.generation = serve::parse_serial_generation_cli(argc, argv, family,
        [&](std::string_view argument, const auto& value) {
            if (parse_mini_weight_option(options.weights, argument, value, family.default_prefill_chunk != 0))
                return true;
            if (argument == "--dspark") {
                options.dspark = value();
                return true;
            }
            if (!family.default_prefill_chunk) return false;
            if (argument == "--step-graph" || argument == "--no-step-graph")
                options.step_graphs = argument == "--step-graph";
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
    options.tier = resolve_mini_profile(options.weights, !generation.tokenize_file);
    if (options.gate_order_null && (!gate || options.tier.profile != profile::Name::standard))
        throw std::invalid_argument("--gate-order-null requires an exact gate arm");
    return options;
}

struct CudaStepRoute {
    bool program;
    serve::CliSpeculation speculation;
    std::string_view speculation_reason;
};

inline CudaStepRoute resolve_cuda_step_route(const Options& options, bool requested_program,
                                             bool default_program, bool route_taps,
                                             weight::FormatId storage) {
    using Mode = serve::CliSpeculation;
    const auto requested_mode = options.generation.speculation;
    const bool has_decode = options.gate_manifest ? options.gate_mode == session::HeadGateMode::decode
                                                  : options.generation.generate >= 2U;
    if (requested_program) {
        if (options.step_graphs)
            throw std::invalid_argument("--step-program is exclusive with --step-graph and --no-step-graph");
        if (!options.generation.packed_decode)
            throw std::invalid_argument("--step-program requires packed decode");
        if (!has_decode)
            throw std::invalid_argument("--step-program requires generation of at least two tokens or a decode gate");
        if (route_taps)
            throw std::invalid_argument("--step-program requires route tap and replay to be disabled");
    }
    const bool program = requested_program || (default_program && has_decode && !options.step_graphs &&
                                               options.generation.packed_decode && !route_taps);
    if (!program) return {false, requested_mode, {}};
    if (options.dspark)
        throw std::invalid_argument("CUDA step program does not support --dspark; choose resident replay "
                                    "with --step-graph or --no-step-graph");
    if (requested_mode == Mode::verify || requested_mode == Mode::lookup)
        throw std::invalid_argument("CUDA step program has no verifier; choose resident replay "
                                    "with --step-graph or --no-step-graph for forced drafting");
    if (storage != weight::kBf16WeightStorage)
        throw std::invalid_argument("CUDA step program requires BF16 weights; choose resident replay "
                                    "with --step-graph or --no-step-graph for Q8_0");
    if (requested_mode == Mode::automatic)
        return {true, Mode::off, "CUDA step program has no verifier"};
    return {true, Mode::off, {}};
}

inline void write_profile(const profile::Resolution& tier, std::ostream& output, bool device) {
    auto levers = " none";
    output << "profile " << mini_profile_name(tier.profile)
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
