#include "strix/ds41_formats.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_packed.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>  // _mm_cvtps_ph (F16C)
#endif

// Packed forms of Q8_0-embeddable weights: choice, both layouts round-trip to the exact Q8_0 blocks, planar plane
// alignment, products through the decoded blocks equal the Q8_0 products, and rejection of unrepresentable codes.
namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

// rows x width Q8_0 with codes from `codes` and random finite f16 scales near 2^-10.
std::vector<std::uint8_t> q8_from(std::mt19937& rng, std::size_t rows, std::size_t width, const std::vector<int>& codes) {
    std::vector<std::uint8_t> q8(rows * width / 32U * 34U);
    std::uniform_int_distribution<std::size_t> pick(0, codes.size() - 1U);
    std::uniform_int_distribution<int> mant(0, 0x3ff);
    for (std::size_t b = 0; b < q8.size() / 34U; ++b) {
        const auto d = static_cast<std::uint16_t>(0x1400U | static_cast<unsigned>(mant(rng)));
        std::memcpy(q8.data() + b * 34U, &d, 2);
        for (std::size_t j = 0; j < 32U; ++j) q8[b * 34U + 2U + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(codes[pick(rng)]));
    }
    return q8;
}

std::vector<int> range(int lo, int hi) {
    std::vector<int> v;
    for (int c = lo; c <= hi; ++c) v.push_back(c);
    return v;
}

void check_kind(std::mt19937& rng, const std::vector<int>& codes, strix::ds41::PackedKind want, const std::string& label) {
    using namespace strix::ds41;
    for (const std::size_t width : {1280U, 2304U, 5120U, 8192U}) {
        const std::size_t rows = 37, groups = width / 32U;
        const auto q8 = q8_from(rng, rows, width, codes);
        const auto choice = choose_packing(q8);
        expect(choice && choice->kind == want, label + ": chosen packing, width " + std::to_string(width));
        if (!choice) continue;
        for (const auto layout : {PackedLayout::interleaved, PackedLayout::planar}) {
            const auto packed = pack_rows(q8, width, *choice, layout);
            const std::size_t row_bytes = packed_row_bytes(choice->kind, width);
            expect(packed.size() == rows * row_bytes, label + ": packed size");
            expect(row_bytes % 16U == 0U, label + ": row bytes are a multiple of 16 for width " + std::to_string(width));
            std::size_t bad = 0;
            std::uint8_t block[34];
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t g = 0; g < groups; ++g) {
                    unpack_group(packed.data() + r * row_bytes, width, g, *choice, layout, block);
                    bad += std::memcmp(block, q8.data() + (r * groups + g) * 34U, 34U) != 0;
                }
            expect(bad == 0U, label + (layout == PackedLayout::planar ? " planar" : " interleaved") + ": every group decodes to its Q8_0 block (" +
                                  std::to_string(bad) + " differ), width " + std::to_string(width));
            if (layout == PackedLayout::planar)  // high and scale planes start 16-B aligned inside the (aligned) row
                expect((16U * groups) % 16U == 0U && ((16U + packed_high_bytes(choice->kind)) * groups) % 16U == 0U,
                       label + ": planar planes 16-B aligned, width " + std::to_string(width));
        }
        // products through the decoded blocks are the Q8_0 products (A8 activation of a random row)
        std::vector<float> x(width);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        for (auto& v : x) v = nd(rng);
        const auto a8 = quantize_a8(x);
        const auto packed = pack_rows(q8, width, *choice, PackedLayout::planar);
        const std::size_t row_bytes = packed_row_bytes(choice->kind, width);
        std::size_t bad = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            std::vector<std::uint8_t> row(groups * 34U);
            for (std::size_t g = 0; g < groups; ++g)
                unpack_group(packed.data() + r * row_bytes, width, g, *choice, PackedLayout::planar, row.data() + g * 34U);
            const float a = product_q8_0(std::span<const std::uint8_t>(q8.data() + r * groups * 34U, groups * 34U), a8, width);
            const float b = product_q8_0(row, a8, width);
            bad += std::memcmp(&a, &b, 4) != 0;
        }
        expect(bad == 0U, label + ": products of decoded planar rows = Q8_0 products, width " + std::to_string(width));
    }
}

}  // namespace

int main() try {
    using namespace strix::ds41;
    std::mt19937 rng(20261020);
    check_kind(rng, {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}, PackedKind::cb4, "CB4 nl4");
    check_kind(rng, range(-8, 7), PackedKind::cb4, "CB4 u4");
    check_kind(rng, range(-16, 15), PackedKind::u5, "U5");
    check_kind(rng, range(-32, 31), PackedKind::u6, "U6");
    expect(!choose_packing(q8_from(rng, 4, 1280, range(-40, 40))), "codes beyond 6 bits: no packing");
    bool threw = false;
    try {
        PackedChoice c;
        c.kind = PackedKind::u5;
        (void)pack_rows(q8_from(rng, 2, 1280, range(-32, 31)), 1280, c, PackedLayout::planar);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "U5 packing of 6-bit codes is rejected");

    // F16 rows that are E4M3 x 2^k per 32 values (signed zeros, subnormal E4M3, both exponent ends) -> F8 rows
    {
        const std::size_t width = 6144, rows = 9;
        std::uniform_int_distribution<int> code(0, 255), kd(-15, -4);
        std::vector<std::uint16_t> f16(rows * width);
        for (std::size_t b = 0; b < f16.size() / 32U; ++b) {
            const int k = kd(rng);
            for (std::size_t j = 0; j < 32U; ++j) {
                auto c = static_cast<std::uint8_t>(code(rng));
                if ((c & 0x7fU) == 0x7fU) c = 0x80U;  // no NaN; -0 instead
                const float v = std::ldexp(e4m3_to_float(c), k);
                // the F16 of an E4M3 value times 2^k (|k| small) is exact: 4 significant bits, in range
#if defined(_MSC_VER) && !defined(__clang__)
                const auto h = static_cast<std::uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT), 0));  // F16C, RNE
#else
                const auto h = std::bit_cast<std::uint16_t>(static_cast<_Float16>(v));
#endif
                f16[b * 32U + j] = h;
            }
        }
        const auto f8 = pack_f16_as_f8(f16, width);
        expect(f8.has_value(), "F16 of E4M3 x 2^k blocks packs to F8");
        if (f8) {
            std::size_t bad = 0;
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t i = 0; i < width; ++i) {
                    const float a = unpack_f8(f8->data() + r * f8_row_bytes(width), width, i);
                    const float b = f16_to_float(f16[r * width + i]);
                    bad += std::bit_cast<std::uint32_t>(a) != std::bit_cast<std::uint32_t>(b);
                }
            expect(bad == 0U, "F8 rows decode to the F16 bits (" + std::to_string(bad) + " differ)");
        }
        std::uniform_int_distribution<int> any(0, 0x7bff);
        std::vector<std::uint16_t> plain(width);
        for (auto& h : plain) h = static_cast<std::uint16_t>(any(rng) | 0x0001);  // odd mantissas: not E4M3 x 2^k
        expect(!pack_f16_as_f8(plain, width).has_value(), "plain F16 rows are not packed to F8");
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ds41 packed tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
