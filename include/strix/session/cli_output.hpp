#pragma once

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <span>
#include <stdexcept>

namespace strix::session {

using CliClock = std::chrono::steady_clock;

inline double cli_milliseconds(CliClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

inline void write_load_timing(std::ostream& output, CliClock::time_point start) {
    output << "timing load_ms " << cli_milliseconds(CliClock::now() - start) << '\n';
}

inline void protect_logits_input(const char* output, const std::filesystem::path& input, const char* error) {
    if (output && std::filesystem::exists(output) && std::filesystem::equivalent(input, output))
        throw std::invalid_argument(error);
}

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

}  // namespace strix::session
