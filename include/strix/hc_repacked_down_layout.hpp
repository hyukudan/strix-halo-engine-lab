#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace strix::hc_repacked_down {

inline constexpr std::string_view kLayoutId = "hc-down-q8-j16s2-rpad32-v1";

struct Geometry {
    std::uint32_t width;
    std::uint32_t rank;
    std::uint32_t blocks;
    std::uint32_t rank_pad;
    std::uint64_t codes_bytes;
    std::uint64_t scales_bytes;
    std::uint64_t scales_offset;
    std::uint64_t plane_bytes;
    std::uint64_t padding_bytes;
    std::uint64_t original_bytes;
    bool operator==(const Geometry&) const = default;
};

namespace detail {
constexpr std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("HC repack size addition overflow");
    }
    return a + b;
}
constexpr std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("HC repack size multiplication overflow");
    }
    return a * b;
}
constexpr void validate_output(std::uint32_t output, std::uint32_t rank_pad) {
    if (rank_pad == 0 || rank_pad > 512 || rank_pad % 32 != 0 || output >= rank_pad) {
        throw std::invalid_argument("HC repack output or padded rank is invalid");
    }
}
}  // namespace detail

[[nodiscard]] constexpr Geometry geometry(std::uint32_t width, std::uint32_t rank) {
    if (width < 256 || width > 10240 || width % 256 != 0) {
        throw std::invalid_argument("HC repack width must be a multiple of 256 in [256,10240]");
    }
    if (rank == 0 || rank > 512) {
        throw std::invalid_argument("HC repack rank must be in [1,512]");
    }
    const auto blocks = width / 32;
    const auto rank_pad = (rank + 31U) / 32U * 32U;
    const auto padded_blocks = detail::multiply(blocks, rank_pad);
    const auto codes = detail::multiply(32, padded_blocks);
    const auto scales = detail::multiply(2, padded_blocks);
    const auto payload = detail::add(codes, scales);
    const auto plane = detail::multiply(detail::add(payload, 255) / 256, 256);
    const auto original = detail::multiply(detail::multiply(rank, blocks), 34);
    return {width, rank, blocks, rank_pad, codes, scales, codes, plane,
            plane - original, original};
}

// Unit indices, not byte offsets. Geometry-bound callers must also check block < blocks.
[[nodiscard]] constexpr std::uint64_t code_unit_index(
    std::uint64_t block, std::uint32_t half, std::uint32_t output,
    std::uint32_t rank_pad) {
    detail::validate_output(output, rank_pad);
    if (half > 1) {
        throw std::invalid_argument("HC repack code half must be 0 or 1");
    }
    return detail::add(detail::multiply(detail::add(detail::multiply(2, block), half),
                                        rank_pad), output);
}

[[nodiscard]] constexpr std::uint64_t scale_word_index(
    std::uint64_t block, std::uint32_t output, std::uint32_t rank_pad) {
    detail::validate_output(output, rank_pad);
    return detail::add(detail::multiply(block / 2, rank_pad), output);
}

}  // namespace strix::hc_repacked_down
