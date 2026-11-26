#pragma once

#include "core/linear_reference.hpp"
#include "strix/attn/group_indexer.hpp"

#include <vector>

namespace strix::attn_core {

inline std::vector<float> fold_value_head_mix_reference(unsigned heads, unsigned rank,
    std::span<const float> output_factor, std::span<const float> input_factor, bool residual) {
    if (heads == 0 || rank == 0 || output_factor.size() != std::size_t{heads} * rank ||
        input_factor.size() != output_factor.size())
        throw std::invalid_argument("value head mix: incompatible factors");
    std::vector<float> folded(std::size_t{heads} * heads);
    for (unsigned h = 0; h < heads; ++h) {
        const auto out_row = output_factor.subspan(std::size_t{h} * rank, rank);
        for (unsigned g = 0; g < heads; ++g) {
            const auto in_row = input_factor.subspan(std::size_t{g} * rank, rank);
            const float entry = core::dot_f32(out_row, in_row) + static_cast<float>(residual && h == g);
            folded[std::size_t{h} * heads + g] = attn::group_indexer::bf16_round(entry);
        }
    }
    return folded;
}

inline void apply_folded_value_head_mix_reference(std::span<float> values, unsigned heads,
                                                  std::span<const float> folded) {
    if (heads == 0 || values.empty() || values.size() % heads != 0 || folded.size() != std::size_t{heads} * heads)
        throw std::invalid_argument("value head mix: incompatible folded dimensions");
    const auto width = values.size() / heads;
    std::vector<float> result(values.size());
    for (unsigned h = 0; h < heads; ++h) {
        auto out = std::span(result).subspan(h * width, width);
        for (unsigned g = 0; g < heads; ++g) {
            const auto in = values.subspan(g * width, width);
            const float coefficient = folded[std::size_t{h} * heads + g];
            for (std::size_t d = 0; d < width; ++d) out[d] = std::fma(coefficient, in[d], out[d]);
        }
    }
    std::ranges::copy(result, values.begin());
}

// V[h,r] contracts the head axis and U[h,r] projects it back. The input and
// output use [head][channel]; each channel is mixed independently.
inline void residual_value_head_mix(std::span<float> values, unsigned heads, unsigned rank,
                                    std::span<const float> u, std::span<const float> v, bool residual = true) {
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
        for (unsigned h = 0; h < heads; ++h) {
            const auto mixed = core::dot_f32(u.subspan(std::size_t{h} * rank, rank), latent);
            if (residual) values[h * width + d] += mixed;
            else values[h * width + d] = mixed;
        }
    }
}

} // namespace strix::attn_core
