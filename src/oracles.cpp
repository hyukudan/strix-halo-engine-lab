#include "strix/oracles.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace strix::oracle {

std::vector<std::size_t> top_k_lower_index(std::span<const float> scores,
                                           std::size_t k) {
    if (k > scores.size()) {
        throw std::invalid_argument("top-k exceeds score count");
    }
    for (const auto score : scores) {
        if (!std::isfinite(score)) {
            throw std::invalid_argument("top-k input must be finite");
        }
    }

    std::vector<std::size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    const auto before = [&](std::size_t left, std::size_t right) {
        if (scores[left] != scores[right]) {
            return scores[left] > scores[right];
        }
        return left < right;
    };
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k),
                      indices.end(), before);
    indices.resize(k);
    return indices;
}

}  // namespace strix::oracle
