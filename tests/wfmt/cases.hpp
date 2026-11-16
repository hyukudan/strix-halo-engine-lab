#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight::test {
struct Record {
    std::string arithmetic;
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> bytes;
};
inline std::uint64_t fnv1a64(std::span<const std::byte> bytes) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (auto byte : bytes) {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}
inline bool compare_records(const Record& a, const Record& b) noexcept {
    return a.arithmetic == b.arithmetic && a.shape == b.shape &&
           a.bytes.size() == b.bytes.size() && a.bytes == b.bytes;
}
bool compare_case();
bool capture_case(std::string_view output_directory);
}
