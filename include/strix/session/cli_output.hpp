#pragma once

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>

namespace strix::session {

using CliClock = std::chrono::steady_clock;

inline double cli_milliseconds(CliClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

inline void write_load_timing(std::ostream& output, CliClock::time_point start) {
    output << "timing load_ms " << cli_milliseconds(CliClock::now() - start) << '\n';
}

// The --logits file format: the row's float32 values, little-endian.
inline std::string logits_bytes(std::span<const float> logits) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::string bytes(logits.size_bytes(), '\0');
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const auto bits = std::bit_cast<std::uint32_t>(logits[i]);
        for (unsigned b = 0; b < 4; ++b) bytes[4U * i + b] = static_cast<char>(bits >> (8U * b));
    }
    return bytes;
}

inline void write_logits(const std::filesystem::path& path, std::span<const float> logits) {
    const auto bytes = logits_bytes(logits);
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("could not open logits output");
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("could not write logits output");
}

}  // namespace strix::session
