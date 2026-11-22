#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix::serve {

// The control plane publishes row indices. Only the lead owns sampler state.
using SamplingRowMask = std::uint32_t;
inline constexpr std::size_t kSamplingRowCapacity =
    static_cast<std::size_t>(std::numeric_limits<SamplingRowMask>::digits);

[[nodiscard]] constexpr SamplingRowMask sampling_row_bit(std::size_t row) {
    if (row >= kSamplingRowCapacity) throw std::invalid_argument("sampling row index exceeds mask capacity");
    return SamplingRowMask{1} << row;
}

template<class Rows, class Emits, class Samplers>
[[nodiscard]] SamplingRowMask sampling_row_mask(const Rows& rows, const Emits& emits, const Samplers& samplers) {
    // Reject malformed batches before computing shifts or indexing caller-owned arrays.
    if (rows.size() > kSamplingRowCapacity || emits.size() != rows.size())
        throw std::invalid_argument("sampling row mask capacity or emits size mismatch");
    SamplingRowMask mask = 0;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (rows[row].slot >= samplers.size())
            throw std::invalid_argument("sampling row slot exceeds sampler count");
        if (emits[row] && samplers[rows[row].slot]) mask |= sampling_row_bit(row);
    }
    return mask;
}

// A row consumes its input token and decides the token at the next position.
[[nodiscard]] constexpr std::uint64_t sampled_position(std::uint64_t input_position, std::uint32_t row_in_request) noexcept {
    return input_position + row_in_request + 1U;
}

// Consume a compact rank-0 result in ascending step-row order.
template<class Write>
void apply_sampled_tokens(SamplingRowMask mask, std::span<const std::uint32_t> tokens, Write write) {
    // Check the entire compact result before any write: a short result must not partially publish a step.
    if (tokens.size() != static_cast<std::size_t>(std::popcount(mask)))
        throw std::invalid_argument("sampled token count differs from row mask");
    std::size_t next = 0;
    for (std::size_t row = 0; mask != 0U; ++row, mask >>= 1U)
        if ((mask & 1U) != 0U) write(row, tokens[next++]);
}

}  // namespace strix::serve
