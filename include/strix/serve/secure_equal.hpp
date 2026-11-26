#pragma once
// Credential comparison whose running time does not depend on where the inputs first differ.
#include <algorithm>
#include <cstddef>
#include <string_view>

namespace strix::serve {

[[nodiscard]] inline bool secure_equal(std::string_view a, std::string_view b) noexcept {
    const std::size_t n = std::max(a.size(), b.size());
    volatile unsigned diff = a.size() == b.size() ? 0U : 1U;
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0U;
        const unsigned y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0U;
        diff = diff | (x ^ y);
    }
    return diff == 0U;
}

}  // namespace strix::serve
