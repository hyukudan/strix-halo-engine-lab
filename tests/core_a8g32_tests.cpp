// a8g32-v1 contract (include/strix/core/a8g32.hpp): the shared CPU reference gives the bits of the reference it
// replaced (ds41::quantize_a8 / cpu ref_quantize_a8 as of origin/main 7f1f3fac1, frozen below) on adversarial groups:
// zeros, the tiny boundary, exact rounding ties, clamp edges, signed zeros, infinities and NaN; the row layout leaves
// pitch padding untouched; the tier rule admits a lever edge only in a lossy tier.
#include "strix/core/a8g32.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

namespace a8 = strix::core::a8g32;

// Frozen copy of the replaced reference (src/ds41_formats.cpp quantize_a8 before the move). Do not edit.
void frozen(const std::vector<float>& values, std::vector<std::int8_t>& codes, std::vector<float>& deltas,
            std::vector<std::int32_t>& sums) {
    codes.resize(values.size()); deltas.resize(values.size() / 32U); sums.resize(values.size() / 32U);
    for (std::size_t g = 0; g < deltas.size(); ++g) {
        float amax = 0.0F;
        for (std::size_t j = 0; j < 32U; ++j) amax = std::max(amax, std::fabs(values[g * 32U + j]));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        std::int32_t sum = 0;
        for (std::size_t j = 0; j < 32U; ++j) {
            const float rounded = tiny ? 0.0F : std::nearbyint(values[g * 32U + j] / delta);
            const auto code = static_cast<std::int8_t>(std::min(127.0F, std::max(-127.0F, rounded)));
            codes[g * 32U + j] = code;
            sum += code;
        }
        deltas[g] = delta; sums[g] = sum;
    }
}

std::uint32_t bits(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, sizeof u);
    return u;
}
float from_bits(std::uint32_t u) {
    float v = 0.0F;
    std::memcpy(&v, &u, sizeof v);
    return v;
}
std::uint32_t next(std::uint32_t& s) {
    s = s * 1664525U + 1013904223U;
    return s;
}

// Adversarial groups, 32 values each.
std::vector<float> cases() {
    std::vector<float> v;
    auto group = [&](auto make) { for (std::uint32_t j = 0; j < 32U; ++j) v.push_back(make(j)); };
    group([](std::uint32_t) { return 0.0F; });                                             // all zero
    group([](std::uint32_t j) { return j % 2U ? -0.0F : 0.0F; });                          // signed zeros
    group([](std::uint32_t j) { return j == 5U ? 0x1.fffffep-120F : 0.0F; });              // tiny: below 2^-119
    group([](std::uint32_t j) { return j == 5U ? -0x1p-119F : 1e-38F; });                  // exactly 2^-119: not tiny
    group([](std::uint32_t j) { return std::numeric_limits<float>::denorm_min() * static_cast<float>(j); });
    group([](std::uint32_t j) { return (static_cast<float>(j) - 15.5F) * 2.0F; });         // x / delta near ties
    group([](std::uint32_t j) { return j == 0U ? 127.0F : static_cast<float>(j) + 0.5F; });// delta 1: exact .5 ties
    group([](std::uint32_t j) { return j == 0U ? -254.0F : static_cast<float>(j) * 2.0F + 1.0F; });  // delta 2: ties
    group([](std::uint32_t j) { return j == 3U ? std::numeric_limits<float>::infinity() : static_cast<float>(j); });
    group([](std::uint32_t j) { return j == 7U ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(j) - 9.0F; });
    group([](std::uint32_t j) { return j == 0U ? std::numeric_limits<float>::quiet_NaN() : 0.0F; });   // NaN only
    group([](std::uint32_t j) { return j == 9U ? std::numeric_limits<float>::max() : -1.0F; });
    std::uint32_t s = 0x1234567U;
    for (int g = 0; g < 4000; ++g)                                                          // random exponents / signs
        group([&](std::uint32_t) {
            const std::uint32_t r = next(s);
            const std::uint32_t exponent = 64U + (r >> 9) % 128U;
            return from_bits((r & 0x80000000U) | (exponent << 23) | (next(s) & 0x7fffffU));
        });
    return v;
}

void test_matches_frozen_reference() {
    const std::vector<float> x = cases();
    std::vector<std::int8_t> want_codes, codes(x.size());
    std::vector<float> want_deltas, deltas(x.size() / 32U);
    std::vector<std::int32_t> want_sums, sums(x.size() / 32U);
    frozen(x, want_codes, want_deltas, want_sums);
    std::fill(sums.begin(), sums.end(), -1);
    a8::quantize_row(x, codes.data(), deltas.data(), sums.data());
    expect(codes == want_codes, "codes bitwise == frozen reference");
    bool same_deltas = true;
    for (std::size_t g = 0; g < deltas.size(); ++g) same_deltas = same_deltas && bits(deltas[g]) == bits(want_deltas[g]);
    expect(same_deltas, "deltas bitwise == frozen reference");
    expect(sums == want_sums, "sums == frozen reference");
    expect(want_deltas[2] == 1.0F && want_codes[64 + 5] == 0, "tiny group: delta 1, code 0");
    expect(bits(want_deltas[3]) == bits(0x1p-119F / 127.0F), "2^-119 group is not tiny");

    std::vector<float> no_sum_deltas(deltas.size());
    std::vector<std::int8_t> no_sum_codes(codes.size());
    a8::quantize_row(x, no_sum_codes.data(), no_sum_deltas.data(), nullptr);
    expect(no_sum_codes == want_codes, "null sums: same codes");
}

void test_layout_and_padding() {
    constexpr std::uint32_t rows = 3U, width = 4096U;
    const a8::Layout layout = a8::Layout::padded(rows, width);
    expect(layout.valid() && layout.pitch == width + strix::kA8PitchPadBytes, "padded pitch avoids the 4 KiB alias");
    expect(layout.code_count() == strix::a8_codes_bytes(width, rows) &&
               layout.group_count() == strix::a8_group_count(width, rows),
           "layout counts == a8_row_pitch helpers");
    expect(!a8::Layout{1U, 48U, 64U}.valid() && !a8::Layout{1U, 64U, 32U}.valid(), "invalid layouts refused");

    std::vector<float> x(std::size_t{rows} * width);
    std::uint32_t s = 99U;
    for (float& v : x) v = static_cast<float>(static_cast<std::int32_t>(next(s) >> 16) - 32768) * 0x1p-9F;
    std::vector<std::int8_t> codes(layout.code_count(), std::int8_t{0x55});
    std::vector<float> deltas(layout.group_count(), -7.0F);
    std::vector<std::int32_t> sums(layout.group_count(), -7);
    a8::quantize_rows(x.data(), width, layout, codes.data(), deltas.data(), sums.data());
    bool rows_ok = true, pad_ok = true;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::int8_t> c(width);
        std::vector<float> d(width / 32U);
        std::vector<std::int32_t> m(width / 32U);
        a8::quantize_row(std::span<const float>(x.data() + std::size_t{r} * width, width), c.data(), d.data(), m.data());
        rows_ok = rows_ok && std::equal(c.begin(), c.end(), codes.begin() + std::ptrdiff_t{r} * layout.pitch) &&
                  std::equal(m.begin(), m.end(), sums.begin() + std::ptrdiff_t{r} * layout.group_pitch());
        for (std::uint32_t k = width; k < layout.pitch; ++k) pad_ok = pad_ok && codes[std::size_t{r} * layout.pitch + k] == 0x55;
        for (std::uint32_t g = width / 32U; g < layout.group_pitch(); ++g)
            pad_ok = pad_ok && deltas[std::size_t{r} * layout.group_pitch() + g] == -7.0F;
    }
    expect(rows_ok, "rows at pitch == per-row reference");
    expect(pad_ok, "pitch padding untouched");

    bool threw = false;
    try {
        a8::quantize_row(std::span<const float>(x.data(), 33U), codes.data(), deltas.data(), nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "width not a multiple of 32 refused");
}

void test_tier_rule() {
    expect(a8::admitted(a8::edge_class(true), false), "reference A8 edge admitted in EXACT");
    expect(!a8::admitted(a8::edge_class(false), false), "lever A8 edge refused in EXACT");
    expect(a8::admitted(a8::edge_class(false), true), "lever A8 edge admitted in FAST/TURBO");
    expect(a8::identity == "a8g32-v1", "identity string");
}

}  // namespace

int main() {
    test_matches_frozen_reference();
    test_layout_and_padding();
    test_tier_rule();
    if (failures != 0) return 1;
    std::puts("core a8g32 tests passed");
    return 0;
}
