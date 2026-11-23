// Shared HC lane chain: the host-callable chain of core/hc_lane_chain.hpp equals, bit for bit, the per-lane serial
// chain the HC launchers write inline (scalar, row-tile, row-split and eight-code-word forms), the coefficient is
// exact for every finite FP16 scale and every 8-bit code, and LaneChainTile gives the launchers' tile byte sizes.

#include "core/hc_geometry.hpp"
#include "core/hc_lane_chain.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
namespace hc = strix::core::hc;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

float half_to_float(std::uint16_t h) {
    const int exponent = (h >> 10) & 31;
    const int mantissa = h & 1023;
    const float sign = (h & 0x8000U) ? -1.0F : 1.0F;
    if (exponent == 0) return sign * std::ldexp(static_cast<float>(mantissa), -24);
    return sign * std::ldexp(static_cast<float>(1024 + mantissa), exponent - 25);
}

std::uint32_t lcg(std::uint32_t& s) { return s = s * 1664525U + 1013904223U; }

// The private chain as the launchers write it: per block, coefficient = scale * float(code), then
// gate += low[block * 32 + k] * coefficient in ascending k.
float private_chain(const std::vector<unsigned char>& row, const float* low, std::uint32_t rank) {
    float gate = 0.0F;
    for (std::uint32_t block = 0; block < rank / 32U; ++block) {
        const unsigned char* weight = row.data() + static_cast<std::uint64_t>(block) * 34U;
        const auto scale_bits = static_cast<unsigned short>(static_cast<unsigned>(weight[0]) | (static_cast<unsigned>(weight[1]) << 8U));
        const float scale = half_to_float(scale_bits);
        for (std::uint32_t k = 0; k < 32U; ++k) {
            const float coefficient = scale * static_cast<float>(static_cast<signed char>(weight[2U + k]));
            gate += low[static_cast<std::uint64_t>(block) * 32U + k] * coefficient;
        }
    }
    return gate;
}

template <bool Unrolled, std::uint32_t Rows>
void shared_chain(const std::vector<unsigned char>& row, const float* const* low, std::uint32_t rank, float* gate) {
    for (std::uint32_t t = 0; t < Rows; ++t) gate[t] = 0.0F;
    for (std::uint32_t block = 0; block < rank / 32U; ++block) {
        const unsigned char* weight = row.data() + static_cast<std::uint64_t>(block) * 34U;
        const float scale = half_to_float(hc::lane_chain_q8_scale_bits(weight));
        hc::lane_chain_block<Unrolled, Rows>(
            scale, [weight](std::uint32_t k) { return static_cast<signed char>(weight[2U + k]); },
            [&](std::uint32_t t, std::uint32_t k) { return low[t][static_cast<std::uint64_t>(block) * 32U + k]; },
            [&](std::uint32_t t) -> float& { return gate[t]; });
    }
}

template <bool Unrolled>
float shared_chain8(const std::vector<unsigned char>& row, const float* low, std::uint32_t rank) {
    float gate = 0.0F;
    for (std::uint32_t block = 0; block < rank / 32U; ++block) {
        const unsigned char* weight = row.data() + static_cast<std::uint64_t>(block) * 34U;
        const float scale = half_to_float(hc::lane_chain_q8_scale_bits(weight));
        hc::lane_chain_block8<Unrolled, 1U>(
            scale,
            [weight](std::uint32_t g) {
                std::uint64_t w = 0;
                for (std::uint32_t l = 0; l < 8U; ++l) w |= static_cast<std::uint64_t>(weight[2U + 8U * g + l]) << (8U * l);
                return w;
            },
            [&](std::uint32_t, std::uint32_t g, std::uint32_t l) { return low[static_cast<std::uint64_t>(block) * 32U + 8U * g + l]; },
            [&](std::uint32_t) -> float& { return gate; });
    }
    return gate;
}

void test_coefficient_exact() {
    std::uint64_t inexact = 0;
    for (std::uint32_t h = 0; h < 65536U; ++h) {
        if (((h >> 10) & 31U) == 31U) continue;  // Inf / NaN scales are not finite blocks
        const float scale = half_to_float(static_cast<std::uint16_t>(h));
        for (int c = -128; c < 128; ++c) {
            const float coefficient = hc::lane_chain_coefficient(scale, static_cast<signed char>(c));
            if (static_cast<double>(coefficient) != static_cast<double>(scale) * static_cast<double>(c)) ++inexact;
        }
    }
    check(inexact == 0U, "coefficient exact for every finite FP16 scale and 8-bit code");
}

void test_chains() {
    std::uint32_t seed = 12345U;
    const std::uint32_t ranks[] = {32U, 64U, 320U};
    for (std::uint32_t rank : ranks) {
        for (int trial = 0; trial < 64; ++trial) {
            constexpr std::uint32_t kRows = 4U;
            std::vector<unsigned char> row(static_cast<std::size_t>(rank / 32U) * 34U);
            for (std::uint32_t block = 0; block < rank / 32U; ++block) {
                // finite scales of mixed magnitude, random codes
                const std::uint32_t h = (lcg(seed) >> 16) & 0x7bffU;
                row[block * 34U] = static_cast<unsigned char>(h & 255U);
                row[block * 34U + 1U] = static_cast<unsigned char>((h >> 8) | ((lcg(seed) >> 31) << 7));
                for (std::uint32_t k = 0; k < 32U; ++k) row[block * 34U + 2U + k] = static_cast<unsigned char>(lcg(seed) >> 24);
            }
            std::vector<float> low(static_cast<std::size_t>(kRows) * rank);
            for (float& v : low) v = static_cast<float>(static_cast<std::int32_t>(lcg(seed))) * 0x1p-31F * 4.0F;
            const float* rows[kRows];
            float reference[kRows];
            for (std::uint32_t t = 0; t < kRows; ++t) {
                rows[t] = low.data() + static_cast<std::size_t>(t) * rank;
                reference[t] = private_chain(row, rows[t], rank);
            }
            float one_plain = 0.0F, one_unrolled = 0.0F, tile_plain[kRows], tile_unrolled[kRows];
            shared_chain<false, 1U>(row, rows, rank, &one_plain);
            shared_chain<true, 1U>(row, rows, rank, &one_unrolled);
            shared_chain<false, kRows>(row, rows, rank, tile_plain);
            shared_chain<true, kRows>(row, rows, rank, tile_unrolled);
            bool same = std::bit_cast<std::uint32_t>(one_plain) == std::bit_cast<std::uint32_t>(reference[0]) &&
                        std::bit_cast<std::uint32_t>(one_unrolled) == std::bit_cast<std::uint32_t>(reference[0]);
            for (std::uint32_t t = 0; t < kRows; ++t)
                same = same && std::bit_cast<std::uint32_t>(tile_plain[t]) == std::bit_cast<std::uint32_t>(reference[t]) &&
                       std::bit_cast<std::uint32_t>(tile_unrolled[t]) == std::bit_cast<std::uint32_t>(reference[t]);
            same = same && std::bit_cast<std::uint32_t>(shared_chain8<false>(row, rows[0], rank)) == std::bit_cast<std::uint32_t>(reference[0]) &&
                   std::bit_cast<std::uint32_t>(shared_chain8<true>(row, rows[0], rank)) == std::bit_cast<std::uint32_t>(reference[0]);
            check(same, "shared chain forms equal the private chain bit for bit");
        }
    }
}

void test_tile() {
    static_assert(hc::LaneChainTile<false>::row_bytes(320U) == 340U && hc::LaneChainTile<false>::tile_bytes(320U) == 10880U);
    static_assert(hc::LaneChainTile<true>::row_bytes(320U) == 260U && hc::LaneChainTile<true>::tile_bytes(320U) == 8320U);
    for (std::uint32_t rank = 32U; rank <= 4096U; rank += 32U) {
        check(hc::LaneChainTile<false>::tile_bytes(rank) == 32U * hc::LaneChainTile<false>::row_bytes(rank), "Q8 tile is 32 rows");
        check(hc::LaneChainTile<true>::tile_bytes(rank) == 32U * hc::LaneChainTile<true>::row_bytes(rank), "Q6P32 tile is 32 rows");
        check(hc::LaneChainTile<false>::tile_bytes(rank) % 16U == 0U && hc::LaneChainTile<true>::tile_bytes(rank) % 16U == 0U,
              "tiles are 16-byte multiples");
    }
    const unsigned char bytes[2] = {0x34U, 0x3cU};
    check(hc::lane_chain_q8_scale_bits(bytes) == 0x3c34U, "scale bits are little endian");
}

}  // namespace

int main() {
    test_coefficient_exact();
    test_chains();
    test_tile();
    if (failures != 0) return 1;
    std::printf("hc lane chain tests: PASS\n");
    return 0;
}
