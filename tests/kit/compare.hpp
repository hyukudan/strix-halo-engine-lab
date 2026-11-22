#pragma once

#include "capture.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace strix::kit {

// Empty diagnostics denote equality. Names make unrelated buffer order immaterial.
inline std::string compare_exact(const Buffers& expected, const Buffers& actual) {
    if (expected.size() != actual.size()) return "buffer count differs";
    std::set<std::string_view> seen;
    for (const auto& [name, bytes] : expected) {
        if (!seen.insert(name).second) return name + ": duplicate expected buffer";
        const auto match = std::ranges::find_if(actual, [&](const auto& row) { return row.first == name; });
        if (match == actual.end()) return name + ": missing buffer";
        if (bytes.size() != match->second.size()) return name + ": byte count differs";
        const auto mismatch = std::ranges::mismatch(bytes, match->second);
        if (mismatch.in1 != bytes.end())
            return name + ": byte " + std::to_string(mismatch.in1 - bytes.begin()) + " differs";
    }
    return {};
}

// Error is absolute + relative * abs(reference), with an explicit numerical identity.
inline std::string compare_bounded(std::span<const float> reference, std::span<const float> actual,
                                   std::string_view recorded, std::string_view observed, double absolute, double relative) {
    if (recorded.empty() || recorded != observed) return "numerical identity differs";
    if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0.0 || relative < 0.0)
        return "invalid error bound";
    if (reference.size() != actual.size()) return "element count differs";
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double ref = reference[i];
        const double value = actual[i];
        const double bound = absolute + relative * std::abs(ref);
        if (!std::isfinite(ref) || !std::isfinite(value) || !std::isfinite(bound) || std::abs(value - ref) > bound)
            return "element " + std::to_string(i) + " exceeds error bound";
    }
    return {};
}

}  // namespace strix::kit
