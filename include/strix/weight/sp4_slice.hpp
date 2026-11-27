#pragma once
// Rank slices of a Strix pack SP4 matrix (host only, no device types). A matrix is `rows` outputs of `columns` inputs;
// one row is columns / 128 superblocks of SP4Storage::bytes (strix/weight/host_lowbit.hpp), laid back to back.
//  - an output slice [first, first + count) of rows is one contiguous byte range (a view, no copy);
//  - an input slice [first, first + count) of columns (multiples of 128) is the same superblock range of every row: the
//    rows of the slice are gathered into one compact matrix of count columns (what a tensor-parallel input split reads).
// Both keep the stored codes and scales byte for byte, so each output's value is the one of the whole matrix.
#include "strix/weight/host_lowbit.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::weight::sp4_slice {

inline constexpr std::uint64_t kBlock = sp::SP4Storage::elements;  // inputs per superblock

[[nodiscard]] constexpr std::uint64_t row_bytes(std::uint64_t columns) noexcept { return sp::SP4Storage::row_bytes(columns); }

// True when `bytes` holds `rows` x `columns` SP4 rows exactly.
[[nodiscard]] constexpr bool admitted(std::uint64_t bytes, std::uint64_t rows, std::uint64_t columns) noexcept {
    return rows != 0U && columns != 0U && columns % kBlock == 0U && bytes == rows * row_bytes(columns);
}

// Output slice: a view of rows [first, first + count) of the matrix. Throws std::invalid_argument on a bad geometry.
[[nodiscard]] inline std::span<const std::uint8_t> output_slice(std::span<const std::uint8_t> bytes, std::uint64_t rows,
                                                                 std::uint64_t columns, std::uint64_t first,
                                                                 std::uint64_t count) {
    if (!admitted(bytes.size(), rows, columns)) throw std::invalid_argument("sp4 slice: matrix bytes do not match rows x columns");
    if (count == 0U || first > rows || count > rows - first) throw std::invalid_argument("sp4 slice: output range outside the matrix");
    const auto rb = row_bytes(columns);
    return bytes.subspan(static_cast<std::size_t>(first * rb), static_cast<std::size_t>(count * rb));
}

// Input slice: the superblocks of columns [first, first + count) of every row, gathered row by row (count x rows).
[[nodiscard]] inline std::vector<std::uint8_t> input_slice(std::span<const std::uint8_t> bytes, std::uint64_t rows,
                                                           std::uint64_t columns, std::uint64_t first,
                                                           std::uint64_t count) {
    if (!admitted(bytes.size(), rows, columns)) throw std::invalid_argument("sp4 slice: matrix bytes do not match rows x columns");
    if (count == 0U || first % kBlock != 0U || count % kBlock != 0U || first > columns || count > columns - first)
        throw std::invalid_argument("sp4 slice: input range must be superblock aligned and inside the matrix");
    const auto rb = row_bytes(columns), sb = row_bytes(count), at = row_bytes(first);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(rows * sb));
    for (std::uint64_t r = 0; r < rows; ++r)
        std::memcpy(out.data() + r * sb, bytes.data() + r * rb + at, static_cast<std::size_t>(sb));
    return out;
}

}  // namespace strix::weight::sp4_slice
