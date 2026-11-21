#pragma once

#include "inkling2/inkling2_mini_config.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/session/greedy_cli.hpp"
#include "strix/session/head_gate_cli.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::inkling2::cli {

struct Options {
    std::vector<unsigned> prompt;
    unsigned generate{};
    const char* logits_path{};
    unsigned prefill_chunk{};
    bool packed_decode{};
    bool grouped_decode{};
    const char* text{};
    const char* tokenizer{};
    const char* tokenize_file{};
    const char* gate_manifest{};
    const char* gate_out{};
    const char* gate_arm{};
    unsigned gate_rows{};
    std::optional<session::HeadGateMode> gate_mode;
    profile::Resolution tier;
    bool generation_requested{};
};

inline unsigned parse_number(std::string_view value) {
    unsigned number{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("expected an unsigned integer");
    return number;
}

inline Options parse(int argc, char** argv, unsigned default_prefill_chunk = 0) {
    Options options;
    profile::Inputs tier_inputs;
    options.prefill_chunk = default_prefill_chunk;
    for (int i = 2; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        const bool device_option = default_prefill_chunk != 0 &&
            (argument == "--prefill-chunk" || argument == "--prompt" || argument == "--tokenizer" ||
             argument == "--tokenize" || argument == "--gate-manifest" || argument == "--gate-out" ||
             argument == "--gate-arm" || argument == "--gate-rows" || argument == "--gate-mode");
        if (default_prefill_chunk != 0 && argument == "--packed-decode") {
            options.packed_decode = true;
        } else if (default_prefill_chunk != 0 && argument == "--grouped-decode") {
            options.grouped_decode = true;
        } else if (argument == "--generate" || argument == "--logits" || argument == "--profile" || device_option) {
            if (++i == argc) throw std::invalid_argument("missing option value");
            if (argument == "--generate") {
                options.generate = parse_number(argv[i]);
                options.generation_requested = true;
            } else if (argument == "--logits") options.logits_path = argv[i];
            else if (argument == "--profile") {
                const std::string_view tier(argv[i]);
                if (tier != "exact" && tier != "fast" && tier != "turbo")
                    throw std::invalid_argument("--profile must be exact, fast or turbo");
                tier_inputs.profile = profile::parse(tier);
            }
            else if (argument == "--prompt") options.text = argv[i];
            else if (argument == "--tokenizer") options.tokenizer = argv[i];
            else if (argument == "--tokenize") options.tokenize_file = argv[i];
            else if (argument == "--gate-manifest") options.gate_manifest = argv[i];
            else if (argument == "--gate-out") options.gate_out = argv[i];
            else if (argument == "--gate-arm") options.gate_arm = argv[i];
            else if (argument == "--gate-rows") {
                options.gate_rows = parse_number(argv[i]);
                if (options.gate_rows < 2U) throw std::invalid_argument("--gate-rows must be at least 2");
            } else if (argument == "--gate-mode") {
                const std::string_view mode(argv[i]);
                if (mode == "prefill") options.gate_mode = session::HeadGateMode::prefill;
                else if (mode == "decode") options.gate_mode = session::HeadGateMode::decode;
                else throw std::invalid_argument("--gate-mode must be prefill or decode");
            } else {
                options.prefill_chunk = parse_number(argv[i]);
                if (options.prefill_chunk == 0 || options.prefill_chunk > kMaxContext)
                    throw std::invalid_argument("prefill chunk size must be within the Mini context");
            }
        } else {
            const auto token = parse_number(argument);
            if (token >= kVocabulary) throw std::out_of_range("token id exceeds the Mini vocabulary");
            options.prompt.push_back(token);
        }
    }
    if (options.text && !options.prompt.empty())
        throw std::invalid_argument("choose either token ids or --prompt");
    const bool gate = options.gate_manifest || options.gate_out || options.gate_arm ||
                      options.gate_rows != 0U || options.gate_mode.has_value();
    if (gate && !(options.gate_manifest && options.gate_out && options.gate_arm &&
                  options.gate_rows != 0U && options.gate_mode))
        throw std::invalid_argument("gate mode requires --gate-manifest, --gate-out, --gate-arm, --gate-rows and --gate-mode");
    if (gate && !session::valid_head_gate_name(options.gate_arm))
        throw std::invalid_argument("invalid --gate-arm name");
    if ((gate || options.tokenize_file) &&
        (!options.prompt.empty() || options.text || options.generation_requested || options.logits_path))
        throw std::invalid_argument("gate and tokenize modes take no prompt, --generate or --logits");
    if (gate && options.tokenize_file) throw std::invalid_argument("choose either gate mode or --tokenize");
    if (options.tokenizer && !options.text && !options.tokenize_file)
        throw std::invalid_argument("--tokenizer requires --prompt or --tokenize");
    if (options.logits_path && std::filesystem::exists(options.logits_path) &&
        std::filesystem::equivalent(argv[1], options.logits_path)) {
        throw std::invalid_argument("logits output must differ from the checkpoint");
    }
    options.tier = profile::resolve(kMiniProfile, tier_inputs);
    options.tier.weights.require_honoured();
    return options;
}

inline unsigned context(const Options& options) {
    return session::admit_cli_context(options.prompt.size(), options.generate, kMaxContext,
                                      "prompt and generation must fit the Mini context");
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

template <class Decoder>
void run(Decoder& decoder, const Options& options, std::vector<unsigned>* generated = nullptr, bool device = false) {
    write_profile(options.tier, std::cerr, device);
    session::run_greedy_cli(decoder, {options.prompt, options.generate, options.logits_path,
                                      kOutputVocabulary, kMiniEosToken}, std::cout, generated);
}

}  // namespace strix::inkling2::cli
