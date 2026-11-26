#pragma once

#include "core/dense_linear.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
#include "strix/exec/route_capabilities.hpp"
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/head_gate_cli.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::inkling2::cli {

struct Options {
    serve::SerialGenerationOptions generation;
    const char* dspark{};
    const char* host_pack{};
    const char* host_pack_hot{};
    bool dspark_off{};  // --dspark off: no drafter, not even the model's sidecar under auto
    const char* gate_manifest{};
    const char* gate_out{};
    const char* gate_arm{};
    unsigned gate_rows{};
    std::optional<session::HeadGateMode> gate_mode;
    std::optional<bool> step_graphs;
    MiniWeightOptions weights;
    bool gate_order_null{};
    profile::Resolution tier;
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
};

inline Options parse(int argc, char** argv, const serve::SerialGenerationFamily& family, std::string_view backend = {}) {
    const bool native = !backend.empty();
    Options options;
    options.generation = serve::parse_serial_generation_cli(argc, argv, family,
        [&](std::string_view argument, const auto& value) {
            if (parse_mini_weight_option(options.weights, argument, value, native))
                return true;
            if (argument == "--dspark") {
                const char* file = value();
                options.dspark_off = std::string_view(file) == "off";
                options.dspark = options.dspark_off ? nullptr : file;
                return true;
            }
            if (!native) {
                if (argument != "--host-pack" && argument != "--host-pack-hot") return false;
                auto& path = argument == "--host-pack" ? options.host_pack : options.host_pack_hot;
                if (path) throw std::invalid_argument("duplicate " + std::string(argument));
                path = value();
                return true;
            }
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
    if (gate && (!generation.prompt.empty() || generation.text || generation.messages ||
                 generation.tokenize_file || generation.generate_requested || generation.logits ||
                 generation.sampling_requested))
        throw std::invalid_argument("gate mode takes no prompt, --tokenize, --generate, --logits or sampling options");
    options.tier = resolve_mini_profile(options.weights, native && !generation.tokenize_file);
    if (options.host_pack_hot && !options.host_pack)
        throw std::invalid_argument("--host-pack-hot requires --host-pack");
    if (options.host_pack) {
        if ((options.tier.profile != profile::Name::fast && options.tier.profile != profile::Name::turbo) ||
            generation.tokenize_file)
            throw std::invalid_argument("--host-pack requires CPU generation with --profile fast or --profile turbo");
        if (generation.logits && std::filesystem::exists(generation.logits) && std::filesystem::exists(options.host_pack) &&
            std::filesystem::equivalent(generation.logits, options.host_pack))
            throw std::invalid_argument("logits output must differ from the host pack");
    }
    // The speculation row (kMiniSpeculation) is the default of a generating run; an explicit --speculation wins.
    if (!generation.speculation_requested && !gate && !generation.tokenize_file && family.supports_speculation &&
        mini_speculation(options.tier))
        options.generation.speculation = serve::CliSpeculation::automatic;
    if (options.dspark) {
        if (gate || generation.tokenize_file || generation.speculation != serve::CliSpeculation::automatic ||
            token_sampling_enabled(generation.sampling))
            throw std::invalid_argument("--dspark requires greedy --speculation auto with model generation");
        if (generation.logits && std::filesystem::exists(generation.logits) && std::filesystem::exists(options.dspark) &&
            std::filesystem::equivalent(generation.logits, options.dspark))
            throw std::invalid_argument("logits output must differ from the DSpark sidecar");
    }
    options.weights.lowbit_turbo = mini_lowbit_turbo(options.tier);
    if (native) apply_mini_lowbit_head(options.weights, options.tier);
    mini_cli_decline_served_route(options.tier);
    if (options.gate_order_null && (!gate || options.tier.profile != profile::Name::standard))
        throw std::invalid_argument("--gate-order-null requires an exact gate arm");
    if (!backend.empty()) options.moe_w4a4 = admit_mini_moe_w4a4(options.tier, backend);
    return options;
}

// A gate holds one sequence at a time and reuses one weight image across all windows.
inline weight::WeightLoadExtent gate_weight_extent(std::span<const session::HeadGateWindow> windows) {
    unsigned positions = 0;
    for (const auto& window : windows)
        positions = std::max(positions, static_cast<unsigned>(window.ids.size()));
    return {positions, 1U};
}

// Exact R1 latency forms of the CUDA step program (mask: 1 tiled bank readout, 2 controller output split,
// 4 packed MoE pipe); the values mirror kMiniFormsMix/Moe/All (checked in the CUDA main).
inline constexpr unsigned kLatencyFormsMix = 3U, kLatencyFormsMoe = 4U, kLatencyFormsAll = 7U;
inline constexpr unsigned kLatencyFormsBatch = kLatencyFormsMix | 8U, kLatencyFormsB3 = kLatencyFormsBatch | 16U;

struct StepProgramOptions {
    bool requested = false;
    bool epilogue_fusion = true;
    unsigned latency_forms = kLatencyFormsMix;  // default: mix; --no-latency-forms restores the incumbent leaves
    bool latency_forms_set = false;
};

// Leave option payloads in place for the shared serial and gate parsers.
inline StepProgramOptions extract_step_program(std::vector<char*>& args) {
    constexpr std::string_view value_options[]{
        "--generate", "--logits", "--temperature", "--top-k", "--top-p", "--min-p", "--seed", "--prompt", "--messages",
        "--tokenize", "--tokenizer", "--prefill-chunk", "--profile", "--gate-manifest", "--gate-out",
        "--gate-arm", "--gate-rows", "--gate-mode", "--dspark", "--speculation",
    };
    StepProgramOptions options;
    auto output = args.begin() + 2;
    for (auto arg = output; arg != args.end(); ++arg) {
        const std::string_view value(*arg);
        if (value == "--step-program") {
            if (options.requested) throw std::invalid_argument("duplicate --step-program");
            options.requested = true;
            continue;
        }
        if (value == "--no-epilogue-fusion") {
            if (!options.epilogue_fusion) throw std::invalid_argument("duplicate --no-epilogue-fusion");
            options.epilogue_fusion = false;
            continue;
        }
        if (value == "--latency-forms" || value == "--latency-forms=mix" || value == "--latency-forms=moe" ||
            value == "--latency-forms=batch" || value == "--latency-forms=b3" ||
            value == "--no-latency-forms") {
            if (options.latency_forms_set) throw std::invalid_argument("duplicate latency forms option");
            options.latency_forms_set = true;
            options.latency_forms = value == "--no-latency-forms" ? 0U
                                  : value == "--latency-forms=mix" ? kLatencyFormsMix
                                  : value == "--latency-forms=batch" ? kLatencyFormsBatch
                                  : value == "--latency-forms=b3" ? kLatencyFormsB3
                                  : value == "--latency-forms=moe" ? kLatencyFormsMoe : kLatencyFormsAll;
            continue;
        }
        *output++ = *arg;
        if (std::ranges::find(value_options, value) != std::end(value_options) && arg + 1 != args.end())
            *output++ = *++arg;
    }
    args.erase(output, args.end());
    return options;
}
struct CudaStepRoute {
    bool program;
    serve::CliSpeculation speculation;
    std::string_view speculation_reason;
    // The verifying route's plain rows (the speculator's one-row steps) run on the CUDA step program over the resident
    // decoder's own state (it records the DSpark taps); drafting passes stay on the resident verify windows.
    bool plain_program = false;
};

enum class CudaDecoder { program, resident };
inline constexpr exec::DecodeCandidate<CudaDecoder> kProgramDecoder{CudaDecoder::program, exec::DecodeCapabilities{.sampling = true}};
inline constexpr exec::DecodeCandidate<CudaDecoder> kCudaDecoders[]{
    kProgramDecoder,
    {CudaDecoder::resident, {.verifier = true, .taps = true, .sampling = true}},
};

inline CudaStepRoute resolve_cuda_step_route(const Options& options, const StepProgramOptions& program_options,
                                             weight::FormatId storage, bool resident_k1) {
    using Mode = serve::CliSpeculation;
    const auto requested_mode = options.generation.speculation;
    const bool has_decode = options.gate_manifest ? options.gate_mode == session::HeadGateMode::decode
                                                  : options.generation.generate >= 2U;
    if (program_options.requested) {
        if (options.step_graphs)
            throw std::invalid_argument("--step-program is exclusive with --step-graph and --no-step-graph");
        if (!has_decode)
            throw std::invalid_argument("--step-program requires generation of at least two tokens or a decode gate");
    }
    if (resident_k1 && storage != weight::kBf16WeightStorage)
        throw std::invalid_argument("STRIX_INKLING_K1BIND=1 requires native BF16 weights");

    // The speculation row default (auto) on a decode run takes the VERIFYING route: the resident replay holds the
    // drafter, the arbiter and the verify window program, and the step program has no verifier (it would decline auto
    // and run plain). An explicit step-program option (--step-program, latency forms, --no-epilogue-fusion), an explicit
    // replay choice or a gate run keeps the former resolution. A run that can never draft (token sampling: the shared CLI
    // loads it with speculation off) keeps the step program too; an embeddings prompt, known only at load, binds the
    // plain rows of the verifying route to the step program instead (DeviceCliDecoder::decode).
    const bool automatic_route = requested_mode == Mode::automatic && has_decode && !options.gate_manifest &&
                           !options.step_graphs && !program_options.requested && !program_options.latency_forms_set &&
                           program_options.epilogue_fusion;
    const bool sampled = token_sampling_enabled(options.generation.sampling);
    const auto verifying_route = automatic_route
        ? exec::resolve_decode_route(std::span{kCudaDecoders},
              {.verifier = true, .taps = true, .sampling = sampled, .sampled_verifier = sampled})
        : std::nullopt;
    const bool verifying = verifying_route == CudaDecoder::resident;
    // E1 handles both packed and grouped decode. Its backend resolves the
    // K1 projection batch and the eager fallback for tapped recipes.
    const bool program = verifying_route ? *verifying_route == CudaDecoder::program :
                         program_options.requested || (has_decode && !options.step_graphs);
    if (!program) {
        if (!program_options.epilogue_fusion)
            throw std::invalid_argument("--no-epilogue-fusion requires the CUDA step program");
        if (program_options.latency_forms_set)
            throw std::invalid_argument("latency forms select step-program leaves: decode through --step-program");
        // q52c (RTX, 11-26): plain prose rows ran 101.5 tok/s on the step program vs 91.7 on the resident auto route; the
        // step program needs BF16 weights and no route tap, like its own route.
        return {false, requested_mode, {},
                verifying && storage == weight::kBf16WeightStorage && !resident_k1 && !options.gate_order_null};
    }
    require_mini_expert_consumer(options.tier, false);
    if (options.gate_order_null)
        throw std::invalid_argument("--gate-order-null requires resident replay with --step-graph or --no-step-graph");
    if (options.dspark)
        throw std::invalid_argument("CUDA step program does not support --dspark; choose resident replay "
                                    "with --step-graph or --no-step-graph");
    const auto selected = std::span{&kProgramDecoder, 1U};
    const auto can_verify = exec::resolve_decode_route(selected, {.verifier = true, .sampled_verifier = sampled});
    if ((requested_mode == Mode::verify || requested_mode == Mode::lookup) && !can_verify)
        throw std::invalid_argument("CUDA step program has no verifier; choose resident replay "
                                    "with --step-graph or --no-step-graph for forced drafting");
    if (storage != weight::kBf16WeightStorage)
        throw std::invalid_argument("CUDA step program requires BF16 weights; choose resident replay "
                                    "with --step-graph or --no-step-graph for Q8_0");
    if (requested_mode == Mode::automatic && !can_verify)
        return {true, Mode::off, "CUDA step program has no verifier"};
    return {true, requested_mode, {}};
}

inline void write_profile(const profile::Resolution& tier, std::ostream& output, bool device) {
    output << tier.summary() << (device ? " reference BASE\n" : "\n");
    if (const auto notice = tier.tier_notice(); !notice.empty()) output << notice << '\n';
    for (const auto& entry : tier.entries)
        if (!entry.ladder.empty()) output << entry.ladder << '\n';
}

// The one writer of the weight receipts of a command-line run (device stderr and gate receipt files): the plan line,
// the quality receipt when present, and exactly one regime line.
inline void write_weight_load_plan(std::ostream& output, const weight::WeightLoadPlan& plan) {
    output << "strix-weight-load/v1 " << plan.policy.name << '@' << plan.policy.revision
           << ' ' << plan.storage.name << '@' << plan.storage.revision;
    if (plan.head_storage != plan.storage)
        output << " head " << plan.head_storage.name << '@' << plan.head_storage.revision;
    output << '\n';
    // A TURBO row states its measured quality against BF16 (it did not pass the FAST gate).
    if (!plan.turbo_quality.empty())
        output << "strix-weight-load-quality/v1 " << plan.policy.name << '@' << plan.policy.revision << ' '
               << plan.turbo_quality << '\n';
    output << weight::weight_load_regime_receipt(plan) << '\n';
}

// The receipts the GPU command line writes, composed here so a host fixture exercises the same path: the run's profile
// receipt (profile summary, ladder decisions, weight plan) and one gate window's route receipt. Each holds exactly one
// weight plan and one weight regime record (write_weight_load_plan is their only writer).
inline void write_run_receipt(const profile::Resolution& tier, const weight::WeightLoadPlan& plan, std::ostream& output) {
    write_profile(tier, output, true);
    write_weight_load_plan(output, plan);
}
inline void write_gate_receipt(std::ostream& output, const weight::WeightLoadPlan& plan, std::string_view routes) {
    write_weight_load_plan(output, plan);
    output << "strix-route-file/v1 " << routes << '\n';
}

// The evaluation-order null receipt of one gate window. A null arm whose window
// never reached the reversed relative-bias order is not a null: fail closed
// instead of writing a receipt the gate would have to reject.
inline std::string gate_null_receipt(std::string routes, std::uint64_t applications, std::string_view window) {
    if (applications == 0U)
        throw std::runtime_error("--gate-order-null was not applied in gate window " + std::string(window));
    if (routes.empty() || routes.back() != '}')
        throw std::runtime_error("gate route receipt is not a JSON object");
    routes.pop_back();
    routes += ",\"null\":{\"order\":\"relative-bias-reverse-fma@1\",\"applications\":" +
              std::to_string(applications) + "}}";
    return routes;
}

}  // namespace strix::inkling2::cli
