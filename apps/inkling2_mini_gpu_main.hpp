#pragma once

#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
#include "inkling2/inkling2_mini_program_cuda.hpp"
#endif
#include "strix/env_flags.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/session/cli_output.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace strix::inkling2::cli {

// Prefill chunk rows. The logits are chunk-invariant (same bits for every chunk), so this is a schedule choice only.
// gfx1151 (prose prompts, one chunk size per run, 2048 -> 8192 rows): 4K 3.47 -> 3.13 s, 8K 7.27 -> 6.23 s,
// 16K 15.69 -> 14.06 s, 32K 35.6 -> 35.8 s. CUDA keeps 2048 rows until it is measured there.
// Tail absorption (serve::prefill_tail_rows): a remainder of up to chunk / 8 rows rides in the final chunk. Every
// chunk re-streams all expert weights of every layer, so a short tail chunk is weight-bound: gfx1151 8K prose
// (8236 = 8192 + 44 rows) 5,021 -> 4,810 ms with one chunk, logits bitwise equal; equal halves (4160 + 4076) 5,069 ms.
// The decoder admits only a tail that takes the same route forms in both schedules (inkling2::mini_prefill_tail_rows:
// more rows than the key-split bound, chunk a multiple of the 16-position wave tile, enlarged chunk admitted by the
// device); any other tail runs as its own chunk. CUDA keeps greedy chunks until it is measured there.
#if defined(__HIPCC__)
inline constexpr unsigned kMiniDefaultPrefillChunk = 8192U;
inline constexpr unsigned kMiniPrefillTailDivisor = 8U;
#else
inline constexpr unsigned kMiniDefaultPrefillChunk = 2048U;
inline constexpr unsigned kMiniPrefillTailDivisor = 0U;
#endif

class DeviceCliDecoder final : public serve::SerialGenerationDecoder {
public:
    DeviceCliDecoder(const serve::SerialGenerationLoad& load, bool step_graphs)
        : decoder_(load.checkpoint, load.positions, load.prefill_chunk, load.packed_decode, step_graphs,
                   load.prefill_tail) {}

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
        return decoder_.prefill(ids, logits);
    }

    unsigned decode(unsigned token, std::span<float> logits) override {
        return decoder_.decode(token, logits);
    }

private:
    MiniResidentDecoder decoder_;
};

template <bool StepGraphs>
std::unique_ptr<serve::SerialGenerationDecoder> load_device_decoder(const serve::SerialGenerationLoad& load) {
    return std::make_unique<DeviceCliDecoder>(load, StepGraphs);
}

#if defined(STRIX_INKLING2_STEP_PROGRAM)
// Leave option payloads in place for the shared serial and gate parsers.
inline bool extract_step_program(std::vector<char*>& args) {
    constexpr std::string_view value_options[]{
        "--generate", "--logits", "--temperature", "--top-k", "--seed", "--prompt", "--messages",
        "--tokenize", "--tokenizer", "--prefill-chunk", "--profile", "--gate-manifest", "--gate-out",
        "--gate-arm", "--gate-rows", "--gate-mode",
    };
    bool requested = false;
    auto output = args.begin() + 2;
    for (auto arg = output; arg != args.end(); ++arg) {
        const std::string_view value(*arg);
        if (value == "--step-program") {
            if (requested) throw std::invalid_argument("duplicate --step-program");
            requested = true;
            continue;
        }
        *output++ = *arg;
        if (std::ranges::find(value_options, value) != std::end(value_options) && arg + 1 != args.end())
            *output++ = *++arg;
    }
    args.erase(output, args.end());
    return requested;
}
#endif

inline void run_gate(const char* checkpoint, const Options& options
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     , bool step_program
#endif
) {
    namespace fs = std::filesystem;
    const auto mode = *options.gate_mode;
    const auto windows = session::read_head_gate_manifest(options.gate_manifest, mode, options.gate_rows,
                                                          kMaxContext, kOutputVocabulary);
    const auto route_config = moe::route_tap_config_from_env();
    constexpr std::string_view case_marker = "{case}";
    const bool per_window_routes = route_config.in_path.find(case_marker) != std::string::npos ||
                                   route_config.out_path.find(case_marker) != std::string::npos;
    if (per_window_routes &&
        ((!route_config.in_path.empty() && route_config.in_path.find(case_marker) == std::string::npos) ||
         (!route_config.out_path.empty() && route_config.out_path.find(case_marker) == std::string::npos)))
        throw std::invalid_argument("use {case} in every route path for per-window evidence");
    const auto expand = [&](std::string path, const std::string& name) {
        for (auto at = path.find(case_marker); at != std::string::npos; at = path.find(case_marker, at + name.size()))
            path.replace(at, case_marker.size(), name);
        return path;
    };

    const fs::path out(options.gate_out);
    fs::create_directories(out);
    if (per_window_routes) fs::create_directories(out / "log");
    unsigned positions = 0;
    for (const auto& window : windows) positions = std::max(positions, static_cast<unsigned>(window.ids.size()));
    const auto prefix_positions = mode == session::HeadGateMode::decode ? positions - options.gate_rows : positions;
    const auto max_rows = std::min(options.generation.prefill_chunk, prefix_positions);
    const auto load_start = session::CliClock::now();
    MiniResidentDecoder decoder(checkpoint, positions, max_rows, options.generation.packed_decode,
                                 options.step_graphs.value_or(MiniResidentBackend::default_step_graphs));
#if defined(STRIX_INKLING2_STEP_PROGRAM)
    std::optional<MiniCudaProgramDecoder> program;
    if (step_program) program.emplace(decoder);
#endif
    session::write_load_timing(std::cout, load_start);
    std::vector<float> logits(std::size_t{options.gate_rows} * kOutputVocabulary);
    for (const auto& window : windows) {
        const auto dump = out / (window.name + "-" + options.gate_arm + ".f32");
        const auto log = out / "log" / (window.name + "." + options.gate_arm + ".stderr");
        if (fs::exists(dump) || (per_window_routes && fs::exists(log)))
            throw std::runtime_error("gate output already exists for " + window.name);
        std::optional<moe::RouteTap> tap;
        if (per_window_routes) {
            auto config = route_config;
            config.in_path = expand(config.in_path, window.name);
            config.out_path = expand(config.out_path, window.name);
            if (!config.out_path.empty() && fs::exists(config.out_path))
                throw std::runtime_error("gate route output already exists: " + config.out_path);
            tap.emplace(std::move(config));
        }
        decoder.set_gate_route_tap(tap ? &*tap : nullptr);
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        if (program)
            session::run_head_gate_window(*program, window.ids, mode, options.gate_rows, kOutputVocabulary, logits);
        else
#endif
        session::run_head_gate_window(decoder, window.ids, mode, options.gate_rows, kOutputVocabulary, logits);
        decoder.set_gate_route_tap(nullptr);
        if (!std::ranges::all_of(logits, [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("non-finite gate logits for " + window.name);
        session::write_logits(dump, logits);
        if (tap) {
            std::ofstream receipt(log);
            receipt << "strix-route-file/v1 " << tap->receipt() << '\n';
            receipt.close();
            if (!receipt) throw std::runtime_error("could not write gate route receipt: " + log.string());
        }
    }
}

}  // namespace strix::inkling2::cli

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    namespace serve = strix::serve;
#if defined(__HIPCC__)
    constexpr auto backend = "HIP";
    constexpr auto executable = "strix-inkling2-mini-hip";
    // Single-token decode runs the experts on the one-row packed GEMV (gfx1151, 128 tokens:
    // 19.7 tok/s against 16.0 tok/s on the grouped tile kernel); --grouped-decode keeps the tile kernel.
#else
    constexpr auto backend = "CUDA";
    constexpr auto executable = "strix-inkling2-mini-cuda";
    // RTX PRO 6000, ink_cuda_opt p3 (128 tokens): with graphs, R1 ran at 58.1 vs 43.4 tok/s short and 45.2 vs 36.6 at 4K. Eager short was 44.3 vs 34.1.
#endif
    constexpr bool packed_decode_default = true;
    if (argc < 3) {
        std::cerr << "usage: " << executable
                  << " MODEL.safetensors (TOKEN [TOKEN ...] | --prompt TEXT | --messages FILE) [--tokenizer FILE]"
                     " [--prefill-chunk N] [--packed-decode | --grouped-decode] [--generate N] [--logits FILE]"
                     " [--step-graph | --no-step-graph"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     " | --step-program"
#endif
                     "] [--temperature T] [--top-k K] [--seed N]\n"
                     "       MODEL.safetensors --gate-manifest FILE --gate-out DIR --gate-arm NAME"
                     " --gate-rows N --gate-mode prefill|decode [--prefill-chunk N] [--packed-decode | --grouped-decode]\n"
                     "       MODEL.safetensors --tokenize FILE [--tokenizer FILE]\n"
                     "       --profile exact|fast|turbo (default exact)\n";
        return 2;
    }
    try {
        serve::SerialGenerationFamily family{
            executable, "Inkling 2 Mini", "MODEL.safetensors",
            kVocabulary, kOutputVocabulary, kMaxContext, kMiniEosToken,
            cli::load_device_decoder<MiniResidentBackend::default_step_graphs>,
            strix::inkling2::cli::kMiniDefaultPrefillChunk, true, true,
            [](const std::string& path) {
                return serve::BpeTokenizer::load_hf_json(path, serve::inkling2_mini_tokenizer_identity);
            },
            packed_decode_default, serve::render_inkling2_chat_request};
        // Off switch: STRIX_INKLING_PREFILL_TAIL=0 restores greedy chunks (same logits).
        family.prefill_tail_divisor =
            strix::env_flag_default_on("STRIX_INKLING_PREFILL_TAIL") ? strix::inkling2::cli::kMiniPrefillTailDivisor : 0U;
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        std::vector<char*> arguments(argv, argv + argc);
        const bool step_program = cli::extract_step_program(arguments);
        auto options = cli::parse(static_cast<int>(arguments.size()), arguments.data(), family);
        if (step_program) {
            if (options.step_graphs)
                throw std::invalid_argument("--step-program is exclusive with --step-graph and --no-step-graph");
            if (!options.generation.packed_decode)
                throw std::invalid_argument("--step-program requires packed decode");
            if (options.gate_manifest ? options.gate_mode != strix::session::HeadGateMode::decode
                                      : options.generation.generate < 2U)
                throw std::invalid_argument("--step-program requires generation of at least two tokens or a decode gate");
            if (strix::moe::route_tap_config_from_env().active())
                throw std::invalid_argument("--step-program requires route tap and replay to be disabled");
            family.load = load_mini_cuda_program;
        }
#else
        auto options = cli::parse(argc, argv, family);
#endif
        if (options.step_graphs)
            family.load = *options.step_graphs ? cli::load_device_decoder<true> : cli::load_device_decoder<false>;
        if (options.gate_manifest || options.generation.tokenize_file)
            cli::write_profile(options.tier, std::cerr, true);
        if (options.gate_manifest) {
            cli::run_gate(argv[1], options
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                          , step_program
#endif
            );
            return 0;
        }
        serve::run_serial_generation(argv[1], std::move(options.generation), family, std::cout,
            [&] { cli::write_profile(options.tier, std::cerr, true); });
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini " << backend << ": " << error.what() << '\n';
        return 1;
    }
    return 0;
}
