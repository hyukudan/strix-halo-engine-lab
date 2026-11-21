#pragma once

#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/session/cli_output.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace strix::inkling2::cli {

class DeviceCliDecoder final : public serve::SerialGenerationDecoder {
public:
    explicit DeviceCliDecoder(const serve::SerialGenerationLoad& load)
        : decoder_(load.checkpoint, load.positions, load.prefill_chunk, load.packed_decode) {}

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
        return decoder_.prefill(ids, logits);
    }

    unsigned decode(unsigned token, std::span<float> logits) override {
        return decoder_.decode(token, logits);
    }

private:
    MiniResidentDecoder decoder_;
};

inline void run_gate(const char* checkpoint, const Options& options) {
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
    MiniResidentDecoder decoder(checkpoint, positions, max_rows, options.generation.packed_decode);
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
    constexpr bool packed_decode_default = true;
#else
    constexpr auto backend = "CUDA";
    constexpr auto executable = "strix-inkling2-mini-cuda";
    constexpr bool packed_decode_default = false;  // grouped until a CUDA receipt says otherwise
#endif
    if (argc < 3) {
        std::cerr << "usage: " << executable
                  << " MODEL.safetensors (TOKEN [TOKEN ...] | --prompt TEXT | --messages FILE) [--tokenizer FILE]"
                     " [--prefill-chunk N] [--packed-decode | --grouped-decode] [--generate N] [--logits FILE]"
                     " [--temperature T] [--top-k K] [--seed N]\n"
                     "       MODEL.safetensors --gate-manifest FILE --gate-out DIR --gate-arm NAME"
                     " --gate-rows N --gate-mode prefill|decode [--prefill-chunk N] [--packed-decode | --grouped-decode]\n"
                     "       MODEL.safetensors --tokenize FILE [--tokenizer FILE]\n"
                     "       --profile exact|fast|turbo (default exact)\n";
        return 2;
    }
    try {
        // Larger chunks give each expert more rows per grouped tile; 2048 was the fastest
        // 4096-token prefill on gfx1151 (128: 12.8 s, 512: 6.7 s, 2048: 4.8 s, 4096: 6.2 s).
        const serve::SerialGenerationFamily family{
            executable, "Inkling 2 Mini", "MODEL.safetensors",
            kVocabulary, kOutputVocabulary, kMaxContext, kMiniEosToken,
            [](const serve::SerialGenerationLoad& load) -> std::unique_ptr<serve::SerialGenerationDecoder> {
                return std::make_unique<cli::DeviceCliDecoder>(load);
            },
            2048U, true, true,
            [](const std::string& path) {
                return serve::BpeTokenizer::load_hf_json(path, serve::inkling2_mini_tokenizer_identity);
            },
            packed_decode_default, serve::render_inkling2_chat_request};
        auto options = cli::parse(argc, argv, family);
        if (options.gate_manifest || options.generation.tokenize_file)
            cli::write_profile(options.tier, std::cerr, true);
        if (options.gate_manifest) {
            cli::run_gate(argv[1], options);
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
