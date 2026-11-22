#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::serve {

// The control plane publishes row indices. Only the lead owns sampler state.
using SamplingRowMask = std::uint32_t;

[[nodiscard]] constexpr SamplingRowMask sampling_row_bit(std::size_t row) noexcept {
    return SamplingRowMask{1} << row;
}

template<class Rows, class Emits, class Samplers>
[[nodiscard]] SamplingRowMask sampling_row_mask(const Rows& rows, const Emits& emits, const Samplers& samplers) {
    SamplingRowMask mask = 0;
    for (std::size_t row = 0; row < rows.size(); ++row)
        if (emits[row] && samplers[rows[row].slot]) mask |= sampling_row_bit(row);
    return mask;
}

// A row consumes its input token and decides the token at the next position.
[[nodiscard]] constexpr std::uint64_t sampled_position(std::uint64_t input_position, std::uint32_t row_in_request) noexcept {
    return input_position + row_in_request + 1U;
}

// Consume a compact rank-0 result in ascending step-row order.
template<class Write>
void apply_sampled_tokens(SamplingRowMask mask, std::span<const std::uint32_t> tokens, Write write) {
    std::size_t next = 0;
    for (std::size_t row = 0; mask != 0U; ++row, mask >>= 1U)
        if ((mask & 1U) != 0U) write(row, tokens[next++]);
}

}  // namespace strix::serve
