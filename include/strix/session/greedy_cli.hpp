#pragma once

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::session {

inline unsigned admit_cli_context(std::size_t prompt, unsigned generate, unsigned capacity,
                                  const char* error) {
    if (prompt == 0U || prompt > capacity || generate > capacity - prompt)
        throw std::invalid_argument(error);
    return static_cast<unsigned>(prompt) + generate;
}

using CliClock = std::chrono::steady_clock;

inline double cli_milliseconds(CliClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

inline void write_load_timing(std::ostream& output, CliClock::time_point start) {
    output << "timing load_ms " << cli_milliseconds(CliClock::now() - start) << '\n';
}

inline void write_generated_text(std::ostream& output, std::string_view text) {
    output << "text " << text << '\n';
}

struct GreedyCliRequest {
    std::span<const unsigned> prompt;
    unsigned generate;
    const char* logits_path;
    unsigned output_vocabulary;
    unsigned eos;
};

inline void write_logits(const std::filesystem::path& path, std::span<const float> logits) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("could not open logits output");
    if constexpr (std::endian::native == std::endian::little) {
        output.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size_bytes()));
    } else {
        for (float value : logits) {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            unsigned char encoded[4];
            for (unsigned i = 0; i < 4; ++i) encoded[i] = static_cast<unsigned char>(bits >> (8U * i));
            output.write(reinterpret_cast<const char*>(encoded), sizeof(encoded));
        }
    }
    output.close();
    if (!output) throw std::runtime_error("could not write logits output");
}

template <class Decoder>
void run_greedy_cli(Decoder& decoder, const GreedyCliRequest& request, std::ostream& output,
                    std::vector<unsigned>* generated = nullptr) {
    std::vector<float> logits(request.logits_path ? request.output_vocabulary : 0U);
    const auto prefill_start = CliClock::now();
    auto token = decoder.prefill(request.prompt, logits);
    const double prefill_ms = cli_milliseconds(CliClock::now() - prefill_start);
    if (request.logits_path) write_logits(request.logits_path, logits);
    unsigned decoded = 0;
    CliClock::duration decode_time{};
    const auto report = [&] {
        const double decode_ms = cli_milliseconds(decode_time);
        output << "timing prompt_tokens " << request.prompt.size() << " prefill_ms " << prefill_ms
               << " decode_tokens " << decoded << " decode_ms " << decode_ms << " decode_tok_s "
               << (decoded != 0U && decode_ms > 0.0 ? 1000.0 * decoded / decode_ms : 0.0) << '\n';
    };

    output << "next_token " << token << '\n';
    if (request.generate == 0) {
        report();
        return;
    }

    output << "generated" << std::flush;
    for (unsigned i = 0; i < request.generate; ++i) {
        output << ' ' << token << std::flush;
        if (generated) generated->push_back(token);
        if (token == request.eos || i + 1U == request.generate) break;
        const auto decode_start = CliClock::now();
        token = decoder.decode(token);
        decode_time += CliClock::now() - decode_start;
        ++decoded;
    }
    output << '\n';
    report();
}

}  // namespace strix::session
