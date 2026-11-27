#pragma once

#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

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
inline constexpr std::size_t kMxfp4BlockColumns = 32;

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

    struct Tile16Address {
        const std::uint8_t* codes;
        const std::uint8_t* scale_code;
    };

    PackedMatrix matrix;

    STRIX_MICROSCALED_HD Tile16Address tile16_address(std::size_t logical_row, std::size_t k_start) const {
        const auto row = physical_row(matrix.row_order, matrix.rows, logical_row);
        const auto codes_per_row = kNvfp4 ? matrix.columns / 2U : matrix.columns;
        const auto code_start = kNvfp4 ? k_start / 2U : k_start;
        return {matrix.codes + row * codes_per_row + code_start,
                matrix.block_scales + row * (matrix.columns / kBlockColumns) + k_start / kBlockColumns};
    }

    // Aligned device loads require kCodes-aligned code rows and their tile starts.
    template <bool Aligned = false>
    STRIX_MICROSCALED_HD Tile16 load_tile16(std::size_t logical_row, std::size_t k_start) const {
        const auto address = tile16_address(logical_row, k_start);
        Tile16 tile;
#if defined(__HIP_DEVICE_COMPILE__)
        if constexpr (Aligned) {
            __builtin_memcpy(tile.codes, __builtin_assume_aligned(address.codes, kCodes), kCodes);
        } else
#endif
        {
            for (std::size_t i = 0; i < kCodes; ++i) tile.codes[i] = address.codes[i];
        }
        tile.scale_code = *address.scale_code;
        return tile;
    }

    STRIX_MICROSCALED_HD static float decode_scale(std::uint8_t scale_code) {
        if constexpr (kNvfp4) {
            return scale_code <= 0x7eU ? scalar_float::e4m3(scale_code) : scalar_float::u2f(0x7fc00000U);
        } else {
            return scalar_float::e8m0(scale_code);
        }
    }

    // First byte contains both NVFP4 nibbles. MXFP8 reads two bytes.
    STRIX_MICROSCALED_HD std::uint32_t decode_pair(const std::uint8_t* codes, float scale) const {
        float lo, hi;
        if constexpr (kNvfp4) {
            const auto byte = codes[0];
            lo = nvfp4_value(static_cast<std::uint8_t>(byte & 0x0fU), scale, matrix.outer_scale);
            hi = nvfp4_value(static_cast<std::uint8_t>(byte >> 4U), scale, matrix.outer_scale);
        } else {
            lo = scalar_float::e4m3(codes[0]) * scale;
            hi = scalar_float::e4m3(codes[1]) * scale;
        }
        return scalar_float::bf16_encode(lo) |
               (std::uint32_t{scalar_float::bf16_encode(hi)} << 16U);
    }

    // Each dword packs two neighbouring RNE BF16 values, low half
    // first. NVFP4 applies its outer scale after the FP32 code × block scale.
    template <std::size_t Pairs>
    STRIX_MICROSCALED_HD void decode_bf16(const std::uint8_t* codes, std::uint8_t scale_code,
                                         std::uint32_t (&out)[Pairs]) const {
        static_assert(Pairs <= kTileColumns / 2U);
        const float scale = decode_scale(scale_code);
        for (std::size_t pair = 0; pair < Pairs; ++pair) {
            out[pair] = decode_pair(codes + (kNvfp4 ? pair : 2U * pair), scale);
        }
    }

    STRIX_MICROSCALED_HD void decode_bf16(const Tile16& tile, std::uint32_t (&out)[8]) const {
        decode_bf16(tile.codes, tile.scale_code, out);
    }

    // NVFP4 exponent-shift decode: bitwise decode_bf16 for every input, a few integer operations per value.
    // Every nonzero E2M1 magnitude m = code & 7 is M x 2^E with M = 1.5 for m in {3, 5, 7} (else 1) and E = (m >> 1) - 1,
    // and E2M1 x E4M3 is exact in FP32. While the products stay normal, fl(fl(v x scale) x outer) = 2^E x fl(M x scale x
    // outer) and its BF16 RNE is the BF16 of fl(M x scale x outer) with E added to the exponent field (the rounding
    // increment only reads bits below the exponent). So one tile rounds two products (M = 1 and 1.5), keeps the 8 magnitudes
    // as byte planes, and each code is a byte-permute lookup plus its sign bit (negation is exact; RNE is symmetric).
    // Tiles outside that domain (NaN block scale; negative, infinite or NaN outer scale; a product within a factor of 4 of
    // the subnormal or overflow boundary) take decode_bf16. codes[w] holds code bytes 4w .. 4w + 3 (little endian), out[p]
    // the BF16 pair of code byte p exactly as decode_bf16.
    template <std::size_t Words>
    STRIX_MICROSCALED_HD void decode_bf16_shift(const std::uint32_t (&codes)[Words], std::uint8_t scale_code,
                                               std::uint32_t (&out)[4U * Words]) const {
        static_assert(kNvfp4 && Words >= 1U && Words <= 2U);
        std::uint32_t low[2]{}, high[2]{};
        if (!shift_table(scale_code, low, high)) {
            std::uint8_t bytes[4U * Words];
            for (std::size_t i = 0; i < 4U * Words; ++i)
                bytes[i] = static_cast<std::uint8_t>(codes[i / 4U] >> (8U * (i % 4U)));
            decode_bf16(bytes, scale_code, out);
            return;
        }
        decode_bf16_shifted(codes, low, high, out);
    }

    // The in-domain half of decode_bf16_shift: the codes through a table shift_table() accepted (a caller may compute the
    // table once per scale code and reuse it; decode_bf16_shift = shift_table, then this or the decode_bf16 fallback).
    template <std::size_t Words>
    STRIX_MICROSCALED_HD static void decode_bf16_shifted(const std::uint32_t (&codes)[Words], const std::uint32_t (&low)[2],
                                                        const std::uint32_t (&high)[2], std::uint32_t (&out)[4U * Words]) {
        static_assert(kNvfp4 && Words >= 1U && Words <= 2U);
        for (std::size_t w = 0; w < Words; ++w) {
            for (unsigned half = 0; half < 2U; ++half) {
                const std::uint32_t q = codes[w] >> (16U * half);  // values 0..3 in nibbles 0..3
                const std::uint32_t select = q & 0x7777U;
                const std::uint32_t l = byte_perm(low[0], low[1], select), h = byte_perm(high[0], high[1], select);
                out[4U * w + 2U * half] = byte_perm(l, h, 0x5140U) | ((q & 0x8U) << 12U) | ((q & 0x80U) << 24U);
                out[4U * w + 2U * half + 1U] =
                    byte_perm(l, h, 0x7362U) | ((q & 0x800U) << 4U) | ((q & 0x8000U) << 16U);
            }
        }
    }

    // The 8 BF16 magnitudes of an NVFP4 tile as byte planes (entry m: byte m % 4 of low / high[m / 4]); false outside the
    // exponent-shift domain of decode_bf16_shift.
    STRIX_MICROSCALED_HD bool shift_table(std::uint8_t scale_code, std::uint32_t (&low)[2],
                                         std::uint32_t (&high)[2]) const {
        static_assert(kNvfp4);
        if (scale_code > 0x7eU) return false;
        const float scale = decode_scale(scale_code), outer = matrix.outer_scale;
        if (scalar_float::f2u(outer) >= 0x7f800000U) return false;  // sign bit set, infinite or NaN
        low[0] = low[1] = high[0] = high[1] = 0U;
        if (scale == 0.0F || outer == 0.0F) return true;  // every product is a zero with the code's sign
        const float one = scale * outer, half_up = (1.5F * scale) * outer;
        const std::uint32_t one_exponent = scalar_float::f2u(one) >> 23U;
        const std::uint32_t half_exponent = scalar_float::f2u(half_up) >> 23U;
        if (one_exponent < 2U || one_exponent > 251U || half_exponent < 2U || half_exponent > 251U) return false;
        const std::uint32_t base[2] = {scalar_float::bf16_encode(one), scalar_float::bf16_encode(half_up)};
        for (unsigned m = 1; m < 8U; ++m) {
            const std::uint32_t entry = base[(m & 1U) != 0U && m >= 3U ? 1U : 0U] + ((m >> 1U) << 7U) - 128U;
            low[m / 4U] |= (entry & 0xffU) << (8U * (m % 4U));
            high[m / 4U] |= (entry >> 8U) << (8U * (m % 4U));
        }
        return true;
    }

    // Default-mode byte permute (PTX prmt) for selector nibbles 0..7: result byte i = byte (s >> 4i) & 7 of {y, x}.
    STRIX_MICROSCALED_HD static std::uint32_t byte_perm(std::uint32_t x, std::uint32_t y, std::uint32_t s) {
#if defined(__CUDA_ARCH__)
        return __byte_perm(x, y, s);
#else
        const std::uint64_t v = (std::uint64_t{y} << 32U) | x;
        std::uint32_t r = 0;
        for (unsigned i = 0; i < 4U; ++i)
            r |= static_cast<std::uint32_t>((v >> (8U * ((s >> (4U * i)) & 7U))) & 0xffU) << (8U * i);
        return r;
#endif
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

// Split MXFP4 storage: one E8M0 per 32 columns of each row, low nibble first.
// Widen to FP32; the caller owns any later operand rounding.
inline void decode_mxfp4_row(std::span<const std::uint8_t> codes,
                             std::span<const std::uint8_t> block_scales,
                             std::span<float> output) {
    constexpr auto block_columns = kMxfp4BlockColumns;
    if (output.empty() || output.size() % block_columns != 0 ||
        codes.size() != output.size() / 2U || block_scales.size() != output.size() / block_columns)
        throw std::invalid_argument("MXFP4 row has inconsistent dimensions");
    for (std::size_t block = 0; block < block_scales.size(); ++block) {
        const float scale = scalar_float::e8m0(block_scales[block]);
        if (!std::isfinite(scale))
            throw std::invalid_argument("MXFP4 block scale must be finite E8M0");
        const auto first = block * block_columns;
        for (std::size_t i = first; i < first + block_columns; i += 2U) {
            const auto byte = codes[i / 2U];
            const float even = scalar_float::e2m1(static_cast<std::uint8_t>(byte & 0x0fU)) * scale;
            const float odd = scalar_float::e2m1(static_cast<std::uint8_t>(byte >> 4U)) * scale;
            if (!std::isfinite(even) || !std::isfinite(odd))
                throw std::invalid_argument("MXFP4 weight exceeds FP32 range");
            output[i] = even;
            output[i + 1U] = odd;
        }
    }
}

// Most scales produce finite FP32 for every E2M1 code. Validate extreme
// blocks through the scalar decoder and use a small table for ordinary blocks.
inline void decode_mxfp4_row_table(std::span<const std::uint8_t> codes,
                                  std::span<const std::uint8_t> block_scales,
                                  std::span<float> output) {
    constexpr auto columns = kMxfp4BlockColumns;
    if (output.empty() || output.size() % columns || codes.size() != output.size() / 2U ||
        block_scales.size() != output.size() / columns)
        throw std::invalid_argument("MXFP4 row has inconsistent dimensions");
    static const auto table = [] {
        std::array<std::array<float, 16>, 252> values{};
        for (std::size_t s = 0; s < values.size(); ++s)
            for (std::size_t c = 0; c < values[s].size(); ++c)
                values[s][c] = scalar_float::e2m1(static_cast<std::uint8_t>(c)) *
                               scalar_float::e8m0(static_cast<std::uint8_t>(s));
        return values;
    }();
    for (std::size_t block = 0; block < block_scales.size(); ++block) {
        const auto scale = block_scales[block];
        const auto bytes = codes.subspan(block * columns / 2U, columns / 2U);
        auto row = output.subspan(block * columns, columns);
        if (scale < 2 || scale >= table.size()) {
            decode_mxfp4_row(bytes, block_scales.subspan(block, 1), row);
            continue;
        }
        const auto& lut = table[scale];
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            const auto c = bytes[i];
            row[i * 2U] = lut[c & 0x0fU];
            row[i * 2U + 1U] = lut[c >> 4U];
        }
    }
}

// Row-local image of one split MXFP4 row for the device row sweeps (core/weight_decode.hpp Format::mxfp4_b32): per 32
// columns, the E8M0 scale byte then the 16 nibble bytes (low nibble first), 17 bytes. Same weights as decode_mxfp4_row.
constexpr std::size_t mxfp4_row_block_bytes(std::size_t columns) { return columns / kMxfp4BlockColumns * 17U; }
inline void pack_mxfp4_row_blocks(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> block_scales,
                                  std::span<std::uint8_t> out) {
    constexpr auto half = kMxfp4BlockColumns / 2U;
    if (block_scales.empty() || codes.size() != block_scales.size() * half || out.size() != block_scales.size() * (half + 1U))
        throw std::invalid_argument("MXFP4 row-block image: inconsistent dimensions");
    for (std::size_t block = 0; block < block_scales.size(); ++block) {
        auto* dst = out.data() + block * (half + 1U);
        *dst++ = block_scales[block];
        for (std::size_t i = 0; i < half; ++i) *dst++ = codes[block * half + i];
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
