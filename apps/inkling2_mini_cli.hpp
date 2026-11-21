#pragma once

#include "inkling2/inkling2_mini_config.hpp"

#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
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
    if (options.prompt.empty() || options.prompt.size() > kMaxContext ||
        options.generate > kMaxContext - options.prompt.size())
        throw std::invalid_argument("prompt and generation must fit the Mini context");
    return static_cast<unsigned>(options.prompt.size()) + options.generate;
}

inline void write_logits(const char* path, std::span<const float> logits) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("could not open logits output");
    for (float value : logits) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        unsigned char encoded[4];
        for (unsigned i = 0; i < 4; ++i) encoded[i] = static_cast<unsigned char>(bits >> (8U * i));
        output.write(reinterpret_cast<const char*>(encoded), sizeof(encoded));
    }
    output.close();
    if (!output) throw std::runtime_error("could not write logits output");
}

template <class Decoder>
void run(Decoder& decoder, const Options& options, std::vector<unsigned>* generated = nullptr) {
    using clock = std::chrono::steady_clock;
    const auto milliseconds = [](clock::duration duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    };
    std::vector<float> logits(options.logits_path ? kOutputVocabulary : 0U);
    const auto prefill_start = clock::now();
    auto token = decoder.prefill(options.prompt, logits);
    const double prefill_ms = milliseconds(clock::now() - prefill_start);
    if (options.logits_path) write_logits(options.logits_path, logits);
    unsigned decoded = 0;
    clock::duration decode_time{};
    const auto report = [&] {
        const double decode_ms = milliseconds(decode_time);
        std::cout << "timing prompt_tokens " << options.prompt.size() << " prefill_ms " << prefill_ms
                  << " decode_tokens " << decoded << " decode_ms " << decode_ms << " decode_tok_s "
                  << (decoded != 0U && decode_ms > 0.0 ? 1000.0 * decoded / decode_ms : 0.0) << '\n';
    };

    std::cout << "next_token " << token << '\n';
    if (options.generate == 0) {
        report();
        return;
    }

    std::cout << "generated" << std::flush;
    for (unsigned i = 0; i < options.generate; ++i) {
        std::cout << ' ' << token << std::flush;
        if (generated) generated->push_back(token);
        if (token == kMiniEosToken || i + 1U == options.generate) break;
        const auto decode_start = clock::now();
        token = decoder.decode(token);
        decode_time += clock::now() - decode_start;
        ++decoded;
    }
    std::cout << '\n';
    report();
}

}  // namespace strix::inkling2::cli
