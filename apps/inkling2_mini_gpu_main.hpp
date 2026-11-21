#pragma once

#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "inkling2/inkling2_mini_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>

namespace strix::inkling2::cli {

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
    const auto max_rows = std::min(options.prefill_chunk, prefix_positions);
    const auto load_start = session::CliClock::now();
    MiniResidentDecoder decoder(checkpoint, positions, max_rows, options.packed_decode);
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
    namespace session = strix::session;
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
                  << " MODEL.safetensors (TOKEN [TOKEN ...] | --prompt TEXT [--tokenizer FILE])"
                     " [--prefill-chunk N] [--packed-decode | --grouped-decode] [--generate N] [--logits FILE]\n"
                     "       MODEL.safetensors --gate-manifest FILE --gate-out DIR --gate-arm NAME"
                     " --gate-rows N --gate-mode prefill|decode [--prefill-chunk N] [--packed-decode | --grouped-decode]\n"
                     "       MODEL.safetensors --tokenize FILE [--tokenizer FILE]\n"
                     "       --profile exact|fast|turbo (default exact)\n";
        return 2;
    }
    try {
        // Larger chunks give each expert more rows per grouped tile; 2048 was the fastest
        // 4096-token prefill on gfx1151 (128: 12.8 s, 512: 6.7 s, 2048: 4.8 s, 4096: 6.2 s).
        auto options = cli::parse(argc, argv, 2048U);
        if (options.packed_decode && options.grouped_decode)
            throw std::invalid_argument("--packed-decode and --grouped-decode are exclusive");
        options.packed_decode = options.packed_decode || (packed_decode_default && !options.grouped_decode);
        if (options.gate_manifest || options.tokenize_file) cli::write_profile(options.tier, std::cerr, true);
        if (options.gate_manifest) {
            cli::run_gate(argv[1], options);
            return 0;
        }
        std::shared_ptr<const strix::serve::BpeTokenizer> tokenizer;
        if (options.text || options.tokenize_file) {
            namespace fs = std::filesystem;
            const fs::path asset = options.tokenizer ? fs::path(options.tokenizer) :
                fs::path(argv[1]).parent_path() / "tokenizer.json";
            if (options.logits_path && fs::exists(options.logits_path) && fs::equivalent(asset, options.logits_path))
                throw std::invalid_argument("logits output must differ from the tokenizer");
            tokenizer = load_mini_tokenizer(asset.string());
            if (options.tokenize_file) {
                std::ifstream input(options.tokenize_file, std::ios::binary);
                if (!input) throw std::runtime_error("could not open text for tokenization");
                const std::string plain{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
                if (input.bad()) throw std::runtime_error("could not read text for tokenization");
                const auto ids = tokenizer->encode(plain, false);
                for (const auto id : ids) std::cout << id << '\n';
                if (!std::cout) throw std::runtime_error("could not write token IDs");
                return 0;
            }
            options.prompt = tokenizer->encode(options.text, false);
        }
        const auto positions = cli::context(options);
        const auto max_rows = std::min(options.prefill_chunk, static_cast<unsigned>(options.prompt.size()));
        const auto load_start = session::CliClock::now();
        MiniResidentDecoder decoder(argv[1], positions, max_rows, options.packed_decode);
        session::write_load_timing(std::cout, load_start);
        std::vector<unsigned> generated;
        cli::run(decoder, options, tokenizer ? &generated : nullptr, true);
        if (tokenizer && !generated.empty())
            session::write_generated_text(std::cout, tokenizer->decode(generated, true));
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini " << backend << ": " << error.what() << '\n';
        return 1;
    }
    return 0;
}
