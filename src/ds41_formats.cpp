#include "strix/ds41_formats.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strix::ds41 {

// Format data of the IQ2_XXS layout, as published with the GGUF quantization types
// (ggml-org/llama.cpp, ggml/src/ggml-common.h, MIT; see docs/reference/sources-ledger.md).
const std::array<std::uint64_t, 256> iq2_xxs_grid{
#include "strix/ds41_iq2_xxs_grid.inc"
};

namespace {

float half_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU, mantissa = bits & 0x3ffU;
    if (exponent == 0U) {
        const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -magnitude : magnitude;
    }
    if (exponent == 31U) return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}
std::uint16_t load_u16(const std::uint8_t* p) noexcept { return static_cast<std::uint16_t>(p[0] | (p[1] << 8U)); }
std::uint32_t load_u32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8U) |
        (static_cast<std::uint32_t>(p[2]) << 16U) | (static_cast<std::uint32_t>(p[3]) << 24U);
}
// The device butterfly: v[l] = v[l] + v[l ^ d] for d = 16 .. 1; lane 0.
float butterfly(std::array<float, product_lanes> lanes) noexcept {
    for (unsigned d = 16U; d; d /= 2U) {
        std::array<float, product_lanes> next{};
        for (unsigned l = 0; l < product_lanes; ++l) next[l] = lanes[l] + lanes[l ^ d];
        lanes = next;
    }
    return lanes[0];
}
template <class Term>
float lane_product(std::size_t groups, Term&& term) {
    std::array<float, product_lanes> lanes{};
    for (std::size_t g = 0; g < groups; ++g) lanes[g % product_lanes] = lanes[g % product_lanes] + term(g);
    return butterfly(lanes);
}
void require_row(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width, std::size_t block_elements,
                 std::size_t block_bytes) {
    if (width == 0U || width % block_elements != 0U || width % a8_group != 0U ||
        row.size() < width / block_elements * block_bytes || x.codes.size() < width || x.deltas.size() < width / a8_group)
        throw std::invalid_argument("ds41 product row, activation or width is inconsistent");
}

} // namespace

// The scalar codecs are the shared host/device definitions (ds41_scalar_codecs.hpp).
float e8m0_to_float(std::uint8_t code) noexcept { return codec::e8m0(code); }
float e8m0_half_to_float(std::uint8_t code) noexcept { return codec::e8m0_half(code); }
float e4m3_to_float(std::uint8_t code) noexcept { return codec::e4m3(code); }
const std::array<float, 256>& e4m3_table() noexcept {
    static const auto table = [] {
        std::array<float, 256> out{};
        for (unsigned c = 0; c < 256U; ++c) out[c] = e4m3_to_float(static_cast<std::uint8_t>(c));
        return out;
    }();
    return table;
}
std::uint8_t float_to_e4m3(float value) noexcept { return codec::e4m3_encode(value); }
float e2m1_to_float(std::uint8_t nibble) noexcept { return codec::e2m1(static_cast<std::uint8_t>(nibble & 15U)); }

Iq2XxsGroup iq2_xxs_group(std::span<const std::uint8_t, iq2_xxs_block_bytes> block, unsigned group) {
    Iq2XxsGroup out;
    const float d = half_to_float(load_u16(block.data()));
    const auto* q = block.data() + 2U + 8U * group;
    const std::uint32_t aux0 = load_u32(q), aux1 = load_u32(q + 4U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    for (unsigned l = 0; l < 4U; ++l) {
        const std::uint64_t grid = iq2_xxs_grid[(aux0 >> (8U * l)) & 0xffU];
        const std::uint8_t signs = iq2_xxs_sign_byte(aux1 >> (7U * l));
        for (unsigned j = 0; j < 8U; ++j) {
            const auto magnitude = static_cast<std::int8_t>((grid >> (8U * j)) & 0xffU);
            out.codes[8U * l + j] = (signs >> j) & 1U ? static_cast<std::int8_t>(-magnitude) : magnitude;
        }
    }
    return out;
}
Mxfp4Group mxfp4_group(std::span<const std::uint8_t, mxfp4_block_bytes> block) {
    Mxfp4Group out;
    out.scale = e8m0_half_to_float(block[0]);
    for (unsigned j = 0; j < 16U; ++j) {
        out.codes[j] = e2m1_doubled[block[1U + j] & 15U];
        out.codes[j + 16U] = e2m1_doubled[block[1U + j] >> 4U];
    }
    return out;
}
Q8Group q8_0_group(std::span<const std::uint8_t, q8_0_block_bytes> block) {
    Q8Group out;
    out.scale = half_to_float(load_u16(block.data()));
    for (unsigned j = 0; j < 32U; ++j) out.codes[j] = static_cast<std::int8_t>(block[2U + j]);
    return out;
}
Q2kGroup q2_k_group(std::span<const std::uint8_t, q2_k_block_bytes> block, unsigned group) {
    Q2kGroup out;
    const unsigned half = group / 4U, shift = 2U * (group % 4U);
    for (unsigned l = 0; l < 32U; ++l) out.codes[l] = static_cast<std::uint8_t>((block[16U + 32U * half + l] >> shift) & 3U);
    out.scale[0] = block[8U * half + 2U * (group % 4U)];
    out.scale[1] = block[8U * half + 2U * (group % 4U) + 1U];
    out.d = half_to_float(load_u16(block.data() + 80U));
    out.dmin = half_to_float(load_u16(block.data() + 82U));
    return out;
}

void decode_iq2_xxs(std::span<const std::uint8_t, iq2_xxs_block_bytes> block, std::span<float, super_block_elements> out) {
    for (unsigned g = 0; g < 8U; ++g) {
        const auto group = iq2_xxs_group(block, g);
        for (unsigned j = 0; j < 32U; ++j) out[32U * g + j] = group.scale * static_cast<float>(group.codes[j]);
    }
}
void decode_q2_k(std::span<const std::uint8_t, q2_k_block_bytes> block, std::span<float, super_block_elements> out) {
    for (unsigned g = 0; g < 8U; ++g) {
        const auto group = q2_k_group(block, g);
        for (unsigned j = 0; j < 32U; ++j) {
            const auto sc = group.scale[j / 16U];
            const float dl = group.d * static_cast<float>(sc & 15U), ml = group.dmin * static_cast<float>(sc >> 4U);
            out[32U * g + j] = dl * static_cast<float>(group.codes[j]) - ml;
        }
    }
}
void decode_mxfp4(std::span<const std::uint8_t, mxfp4_block_bytes> block, std::span<float, mxfp4_block_elements> out) {
    const auto group = mxfp4_group(block);
    for (unsigned j = 0; j < 32U; ++j) out[j] = static_cast<float>(group.codes[j]) * group.scale;
}
void decode_q8_0(std::span<const std::uint8_t, q8_0_block_bytes> block, std::span<float, q8_0_block_elements> out) {
    const auto group = q8_0_group(block);
    for (unsigned j = 0; j < 32U; ++j) out[j] = static_cast<float>(group.codes[j]) * group.scale;
}
void decode_engram_row(std::span<const std::uint8_t, engram_row_bytes> row, std::span<float, engram_row_elements> out) {
    const auto& table = e4m3_table();
    for (unsigned j = 0; j < engram_row_elements; ++j) out[j] = table[row[j]] * e8m0_to_float(row[256U + j / 32U]);
}

A8Activation quantize_a8(std::span<const float> values) {
    if (values.size() % a8_group != 0U) throw std::invalid_argument("ds41 A8 activation width must be a multiple of 32");
    A8Activation out;
    out.codes.resize(values.size()); out.deltas.resize(values.size() / a8_group); out.sums.resize(values.size() / a8_group);
    for (std::size_t g = 0; g < out.deltas.size(); ++g) {
        float amax = 0.0F;
        for (std::size_t j = 0; j < a8_group; ++j) amax = std::max(amax, std::fabs(values[g * a8_group + j]));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        std::int32_t sum = 0;
        for (std::size_t j = 0; j < a8_group; ++j) {
            const float rounded = tiny ? 0.0F : std::nearbyint(values[g * a8_group + j] / delta);
            const auto code = static_cast<std::int8_t>(std::min(127.0F, std::max(-127.0F, rounded)));
            out.codes[g * a8_group + j] = code;
            sum += code;
        }
        out.deltas[g] = delta; out.sums[g] = sum;
    }
    return out;
}

float product_iq2_xxs(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width) {
    require_row(row, x, width, super_block_elements, iq2_xxs_block_bytes);
    return lane_product(width / a8_group, [&](std::size_t g) {
        const auto group = iq2_xxs_group(
            std::span<const std::uint8_t, iq2_xxs_block_bytes>(row.data() + g / 8U * iq2_xxs_block_bytes, iq2_xxs_block_bytes),
            static_cast<unsigned>(g % 8U));
        std::int32_t dot = 0;
        for (unsigned j = 0; j < 32U; ++j) dot += group.codes[j] * x.codes[g * a8_group + j];
        return static_cast<float>(dot) * (group.scale * x.deltas[g]);
    });
}
float product_mxfp4(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width) {
    require_row(row, x, width, mxfp4_block_elements, mxfp4_block_bytes);
    return lane_product(width / a8_group, [&](std::size_t g) {
        const auto group = mxfp4_group(
            std::span<const std::uint8_t, mxfp4_block_bytes>(row.data() + g * mxfp4_block_bytes, mxfp4_block_bytes));
        std::int32_t dot = 0;
        for (unsigned j = 0; j < 32U; ++j) dot += group.codes[j] * x.codes[g * a8_group + j];
        return static_cast<float>(dot) * (group.scale * x.deltas[g]);
    });
}
float product_q8_0(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width) {
    require_row(row, x, width, q8_0_block_elements, q8_0_block_bytes);
    return lane_product(width / a8_group, [&](std::size_t g) {
        const auto group = q8_0_group(
            std::span<const std::uint8_t, q8_0_block_bytes>(row.data() + g * q8_0_block_bytes, q8_0_block_bytes));
        std::int32_t dot = 0;
        for (unsigned j = 0; j < 32U; ++j) dot += group.codes[j] * x.codes[g * a8_group + j];
        return static_cast<float>(dot) * (group.scale * x.deltas[g]);
    });
}
float product_q2_k(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width) {
    require_row(row, x, width, super_block_elements, q2_k_block_bytes);
    return lane_product(width / a8_group, [&](std::size_t g) {
        const auto group = q2_k_group(
            std::span<const std::uint8_t, q2_k_block_bytes>(row.data() + g / 8U * q2_k_block_bytes, q2_k_block_bytes),
            static_cast<unsigned>(g % 8U));
        std::int32_t a = 0, b = 0;
        for (unsigned h = 0; h < 2U; ++h) {
            std::int32_t qx = 0, sx = 0;
            for (unsigned j = 16U * h; j < 16U * h + 16U; ++j) {
                qx += group.codes[j] * x.codes[g * a8_group + j];
                sx += x.codes[g * a8_group + j];
            }
            a += static_cast<std::int32_t>(group.scale[h] & 15U) * qx;
            b += static_cast<std::int32_t>(group.scale[h] >> 4U) * sx;
        }
        const float scaled = group.d * static_cast<float>(a);
        const float offset = group.dmin * static_cast<float>(b);
        return (scaled - offset) * x.deltas[g];
    });
}

} // namespace strix::ds41
