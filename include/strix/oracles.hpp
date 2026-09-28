#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace strix::oracle {

// Candidate tie contract for experiments only: higher score first, then lower
// original index. The canonical QSA tie contract remains P-014.
[[nodiscard]] std::vector<std::size_t> top_k_lower_index(
    std::span<const float> scores,
    std::size_t k);

}  // namespace strix::oracle
