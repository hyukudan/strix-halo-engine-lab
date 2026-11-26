#pragma once

#include "core/linear_reference.hpp"

#include <vector>

namespace strix::attn_core {

// V[h,r] contracts the head axis and U[h,r] projects it back. The input and
// output use [head][channel]; each channel is mixed independently.
inline void residual_value_head_mix(std::span<float> values, unsigned heads, unsigned rank,
                                    std::span<const float> u, std::span<const float> v) {
    if (heads == 0 || rank == 0 || values.empty() || values.size() % heads != 0 ||
        u.size() != std::size_t{heads} * rank || v.size() != u.size())
        throw std::invalid_argument("value head mix: incompatible dimensions");
    const auto width = values.size() / heads;
    std::vector<float> coefficients(heads), column(heads), latent(rank);
    for (std::size_t d = 0; d < width; ++d) {
        for (unsigned h = 0; h < heads; ++h) column[h] = values[h * width + d];
        for (unsigned r = 0; r < rank; ++r) {
            for (unsigned h = 0; h < heads; ++h) coefficients[h] = v[std::size_t{h} * rank + r];
            latent[r] = core::dot_f32(coefficients, column);
        }
        for (unsigned h = 0; h < heads; ++h)
            values[h * width + d] += core::dot_f32(u.subspan(std::size_t{h} * rank, rank), latent);
    }
}

} // namespace strix::attn_core
