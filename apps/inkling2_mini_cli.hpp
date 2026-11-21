#pragma once

#include "inkling2/inkling2_mini_config.hpp"
#include "strix/session/greedy_cli.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>
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
    const char* text{};
    const char* tokenizer{};
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
    options.prefill_chunk = default_prefill_chunk;
    for (int i = 2; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        const bool device_option = default_prefill_chunk != 0 &&
            (argument == "--prefill-chunk" || argument == "--prompt" || argument == "--tokenizer");
        if (default_prefill_chunk != 0 && argument == "--packed-decode") {
            options.packed_decode = true;
        } else if (argument == "--generate" || argument == "--logits" || device_option) {
            if (++i == argc) throw std::invalid_argument("missing option value");
            if (argument == "--generate") options.generate = parse_number(argv[i]);
            else if (argument == "--logits") options.logits_path = argv[i];
            else if (argument == "--prompt") options.text = argv[i];
            else if (argument == "--tokenizer") options.tokenizer = argv[i];
            else {
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
    if (options.tokenizer && !options.text) throw std::invalid_argument("--tokenizer requires --prompt");
    if (options.logits_path && std::filesystem::exists(options.logits_path) &&
        std::filesystem::equivalent(argv[1], options.logits_path)) {
        throw std::invalid_argument("logits output must differ from the checkpoint");
    }
    return options;
}

inline unsigned context(const Options& options) {
    return session::admit_cli_context(options.prompt.size(), options.generate, kMaxContext,
                                      "prompt and generation must fit the Mini context");
}

template <class Decoder>
void run(Decoder& decoder, const Options& options, std::vector<unsigned>* generated = nullptr) {
    session::run_greedy_cli(decoder, {options.prompt, options.generate, options.logits_path,
                                      kOutputVocabulary, kMiniEosToken}, std::cout, generated);
}

}  // namespace strix::inkling2::cli
