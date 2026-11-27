#pragma once

#include <array>

namespace strix::core {

struct LinearQuadraticHinges {
    // Lower and upper linear hinges, then lower and upper quadratic hinges.
    std::array<float, 4> thresholds{};
    std::array<float, 4> gains{};
};

} // namespace strix::core
