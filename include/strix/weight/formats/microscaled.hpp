#pragma once

#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MICROSCALED_HD __host__ __device__ inline
#else
#define STRIX_MICROSCALED_HD inline
#endif

namespace strix::weight::microscaled {

enum class MatrixFormat { nvfp4, mxfp8 };
enum class RowOrder { contiguous, interleaved_halves };

STRIX_MICROSCALED_HD constexpr std::size_t physical_row(RowOrder order, std::size_t rows, std::size_t logical_row) {
    if (order == RowOrder::contiguous) return logical_row;
    const auto half = rows / 2U;  // Valid only for an even nonzero row count.
    return logical_row < half ? 2U * logical_row : 2U * (logical_row - half) + 1U;
}

inline constexpr std::size_t kNvfp4BlockColumns = 16;
inline constexpr std::size_t kMxfp8BlockColumns = 32;

STRIX_MICROSCALED_HD float nvfp4_value(std::uint8_t code, float scale, float outer_scale) {
    return (scalar_float::e2m1(code) * scale) * outer_scale;
}

// Non-owning data addresses. For a device launch, the caller supplies the
// corresponding device addresses. The same row order applies to codes and scales.
struct PackedMatrix {
    const std::uint8_t* codes{};
    const std::uint8_t* block_scales{};
    std::size_t rows{};
    std::size_t columns{};
    float outer_scale{1.0F};
    RowOrder row_order{RowOrder::contiguous};
};

// Shared K16 operand for prefetch and LDS stages. At admission the caller
// validates its matrix format, span sizes and geometry. A tile starts on a
// 16-column boundary and stays fully inside the admitted matrix.
template <MatrixFormat Format>
struct PackedOperand {
    static_assert(Format == MatrixFormat::nvfp4 || Format == MatrixFormat::mxfp8);
    static constexpr bool kNvfp4 = Format == MatrixFormat::nvfp4;
    static constexpr std::size_t kBlockColumns = kNvfp4 ? kNvfp4BlockColumns : kMxfp8BlockColumns;
    static constexpr std::size_t kTileColumns = 16;
    static constexpr std::size_t kCodes = kNvfp4 ? kTileColumns / 2U : kTileColumns;

    struct Tile16 {
        std::uint8_t codes[kCodes];
        std::uint8_t scale_code;
    };

    PackedMatrix matrix;

    STRIX_MICROSCALED_HD Tile16 load_tile16(std::size_t logical_row, std::size_t k_start) const {
        const auto row = physical_row(matrix.row_order, matrix.rows, logical_row);
        const auto codes_per_row = kNvfp4 ? matrix.columns / 2U : matrix.columns;
        const auto code_start = kNvfp4 ? k_start / 2U : k_start;
        const auto* codes = matrix.codes + row * codes_per_row + code_start;
        Tile16 tile;
        for (std::size_t i = 0; i < kCodes; ++i) tile.codes[i] = codes[i];
        tile.scale_code = matrix.block_scales[row * (matrix.columns / kBlockColumns) + k_start / kBlockColumns];
        return tile;
    }

    // Eight LDS dwords: each packs two neighbouring RNE BF16 values, low half
    // first. NVFP4 applies its outer scale after the FP32 code × block scale.
    STRIX_MICROSCALED_HD void decode_bf16(const Tile16& tile, std::uint32_t (&out)[8]) const {
        const float scale = kNvfp4
            ? (tile.scale_code <= 0x7eU ? scalar_float::e4m3(tile.scale_code) : scalar_float::u2f(0x7fc00000U))
            : scalar_float::e8m0(tile.scale_code);
        for (std::size_t pair = 0; pair < 8; ++pair) {
            float lo, hi;
            if constexpr (kNvfp4) {
                const auto byte = tile.codes[pair];
                lo = nvfp4_value(static_cast<std::uint8_t>(byte & 0x0fU), scale, matrix.outer_scale);
                hi = nvfp4_value(static_cast<std::uint8_t>(byte >> 4U), scale, matrix.outer_scale);
            } else {
                lo = scalar_float::e4m3(tile.codes[2U * pair]) * scale;
                hi = scalar_float::e4m3(tile.codes[2U * pair + 1U]) * scale;
            }
            out[pair] = scalar_float::bf16_encode(lo) |
                        (std::uint32_t{scalar_float::bf16_encode(hi)} << 16U);
        }
    }
};

// Decode one contiguous source row into FP32. The caller selects the expert's
// amax, and performs any operand rounding or matrix layout conversion separately.
// On malformed data the caller must discard the output row.
inline void decode_nvfp4_row(std::span<const std::uint8_t> codes,
                             std::span<const std::uint8_t> block_scales,
                             float amax, float denominator, std::span<float> output) {
    constexpr auto block_columns = kNvfp4BlockColumns;
    if (output.empty() || output.size() % block_columns != 0 ||
        codes.size() != output.size() / 2 || block_scales.size() != output.size() / block_columns) {
        throw std::invalid_argument("NVFP4 row has inconsistent dimensions");
    }
    if (!std::isfinite(amax) || amax < 0.0F || !std::isfinite(denominator) || denominator <= 0.0F) {
        throw std::invalid_argument("NVFP4 outer scale requires nonnegative amax and positive denominator");
    }
    const float outer_scale = amax / denominator;
    if (!std::isfinite(outer_scale)) {
        throw std::invalid_argument("NVFP4 outer scale exceeds FP32 range");
    }
    for (std::size_t block = 0; block < block_scales.size(); ++block) {
        const auto encoded_scale = block_scales[block];
        if (encoded_scale > 0x7eU) {
            throw std::invalid_argument("NVFP4 block scale must be nonnegative finite E4M3");
        }
        const float scale = scalar_float::e4m3(encoded_scale);
        const std::size_t first = block * block_columns;
        for (std::size_t i = first; i < first + block_columns; i += 2) {
            const auto byte = codes[i / 2];
            const auto low_code = static_cast<std::uint8_t>(byte & 0x0fU);
            const auto high_code = static_cast<std::uint8_t>(byte >> 4U);
            const float even = nvfp4_value(low_code, scale, outer_scale);
            const float odd = nvfp4_value(high_code, scale, outer_scale);
            if (!std::isfinite(even) || !std::isfinite(odd)) {
                throw std::invalid_argument("NVFP4 weight exceeds FP32 range");
            }
            output[i] = even;
            output[i + 1] = odd;
        }
    }
}

// MXFP8 has one E8M0 scale per 32 columns of each row. Each E4M3 value is a byte.
inline void decode_mxfp8_row(std::span<const std::uint8_t> codes,
                             std::span<const std::uint8_t> block_scales,
                             std::span<float> output) {
    constexpr auto block_columns = kMxfp8BlockColumns;
    if (output.empty() || output.size() % block_columns != 0 ||
        codes.size() != output.size() || block_scales.size() != output.size() / block_columns) {
        throw std::invalid_argument("MXFP8 row has inconsistent dimensions");
    }
    for (std::size_t block = 0; block < block_scales.size(); ++block) {
        const float scale = scalar_float::e8m0(block_scales[block]);
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("MXFP8 block scale must be finite E8M0");
        }
        const std::size_t first = block * block_columns;
        for (std::size_t i = first; i < first + block_columns; ++i) {
            const float weight = scalar_float::e4m3(codes[i]) * scale;
            if (!std::isfinite(weight)) {
                throw std::invalid_argument("MXFP8 weight must be finite in FP32");
            }
            output[i] = weight;
        }
    }
}

} // namespace strix::weight::microscaled

#undef STRIX_MICROSCALED_HD
