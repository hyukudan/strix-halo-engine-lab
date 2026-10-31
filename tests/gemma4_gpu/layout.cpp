#include "gemma4/kv/layout.hpp"
#include "gemma4/kernels/ops.hpp"
#include "gemma4/kernels/norm_bounds.hpp"
#include "test.hpp"
#include <random>
using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
void bounds_tests() {
    for (unsigned index = 0; index < 512; ++index) {
        const int power = index < 256 ? 70 : 71;
        const auto first = 0x800000U + ((index % 256U) << 15U), last = first + 0x7fffU;
        const auto interval = interval_detail::bounds[index];
        CHECK(strix::gemma4::detail::square_compare(last, interval.lower, -power) <= 0);
        CHECK(strix::gemma4::detail::square_compare(last, interval.lower + 1U, -power) > 0);
        CHECK(strix::gemma4::detail::square_compare(first, interval.upper, -power) >= 0);
        CHECK(strix::gemma4::detail::square_compare(first, interval.upper - 1U, -power) < 0);
    }
    const auto check = [](unsigned bits) {
        if (!bits)
            return;
        std::uint32_t mx;
        int ex;
        strix::gemma4::detail::dyadic(bits, mx, ex);
        const int er = rsqrt_exponent(mx, ex);
        CHECK(er >= -64 && er <= 74);
        // Independent exact inequalities bracket the real result, before rounding.
        CHECK(strix::gemma4::detail::square_compare(mx, 0x800000U, ex + 2 * (er - 23)) <= 0);
        CHECK(strix::gemma4::detail::square_compare(mx, 0x800000U, ex + 2 * (er - 22)) > 0);
        const auto lo = unsigned(er + 127) << 23U, hi = lo | 0x7fffffU;
        const auto ref = bit_cast<unsigned>(pow32_rsqrt(bit_cast<float>(bits)));
        CHECK(ref >= lo && ref <= hi + 1U);
        unsigned narrowed_lo, narrowed_hi;
        rsqrt_interval_bounds(mx, ex, er, narrowed_lo, narrowed_hi);
        CHECK(narrowed_lo >= lo && narrowed_hi <= hi && narrowed_lo <= narrowed_hi);
        CHECK(ref >= narrowed_lo && ref <= narrowed_hi + 1U);
        const auto lower = (narrowed_lo & 0x7fffffU) | 0x800000U;
        const auto upper = (narrowed_hi & 0x7fffffU) | 0x800000U;
        CHECK(strix::gemma4::detail::square_compare(mx, lower, ex + 2 * (er - 23)) <= 0);
        CHECK(strix::gemma4::detail::square_compare(mx, upper + 1U, ex + 2 * (er - 23)) > 0);
    };
    for (unsigned exponent = 0; exponent < 255; ++exponent)
        for (unsigned mantissa : {0U, 1U, 0x3fffffU, 0x400000U, 0x7ffffeU, 0x7fffffU})
            check((exponent << 23U) | mantissa);
    for (unsigned bit = 0; bit < 23; ++bit)
        for (unsigned value : {(1U << bit) - 1U, 1U << bit, (1U << bit) + 1U})
            check(value);
    std::mt19937 rng(8117);
    for (unsigned i = 0; i < 32768; ++i)
        check(1U + unsigned(rng() % 0x7f7ffffeU));
}
} // namespace
int main() {
    try {
        bounds_tests();
        ModelConfig c;
        for (unsigned l = 0; l < layers; ++l) {
            const bool global = l % 6 == 5;
            c.layer.push_back({global ? AttentionKind::global : AttentionKind::sliding, global ? 512U : 256U,
                               l < 24 ? l : (global ? 23U : 22U)});
        }
        validate_config(c);
        CHECK(cache_bytes(c, 65536) == 20ULL * 512 * (2 * 512 * 4 + 8) + 4ULL * 65536 * (2 * 1024 * 4 + 8));
        CHECK(cache_bytes(c, 8) == 20ULL * 512 * (2 * 512 * 4 + 8) + 4ULL * 8 * (2 * 1024 * 4 + 8));
        for (std::uint64_t p = 0; p < 65536; ++p) {
            const auto s = visible(p, false, 512), g = visible(p, true, 512);
            CHECK(s.first == (p < 512 ? 0 : p - 511));
            CHECK(s.last == p);
            CHECK(g.first == 0);
            CHECK(g.last == p);
        }
        auto bad = c;
        bad.layer[24].producer = 23;
        refuses([&] { validate_config(bad); });
        bad = c;
        bad.layer[23].producer = 24;
        refuses([&] { validate_config(bad); });
        bad = c;
        bad.heads = 6;
        refuses([&] { validate_config(bad); });
        refuses([&] { (void)cache_bytes(c, 0); });
        refuses([&] { (void)cache_bytes(c, 65537); });
        CHECK(same_config(c, c));
        bad = c;
        bad.layer[41].producer = 17;
        CHECK(!same_config(c, bad));
        std::uint8_t b[] = {0x80, 0x3f, 0, 0x40};
        const LinearView v{b, 1, 2, 4, WeightFormat::bf16};
        CHECK(coefficient(v, 0, 0) == 1.0F);
        CHECK(coefficient(v, 0, 1) == 2.0F);
        std::cout << "device layout: aliases, admitted bytes, absolute visibility, packed lookup and exact rsqrt bounds passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
