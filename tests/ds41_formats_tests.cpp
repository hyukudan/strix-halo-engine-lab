#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace {

int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

void test_tables() {
    std::set<std::uint64_t> distinct(strix::ds41::iq2_xxs_grid.begin(), strix::ds41::iq2_xxs_grid.end());
    expect(distinct.size() == 256U, "IQ2_XXS grid entries are distinct");
    bool magnitudes = true;
    for (const auto entry : strix::ds41::iq2_xxs_grid)
        for (unsigned j = 0; j < 8U; ++j) {
            const auto byte = (entry >> (8U * j)) & 0xffU;
            magnitudes = magnitudes && (byte == 8U || byte == 25U || byte == 43U);
        }
    expect(magnitudes, "IQ2_XXS grid bytes are 8, 25 or 43");
    bool even = true;
    for (std::uint32_t index = 0; index < 128U; ++index) {
        const auto byte = strix::ds41::iq2_xxs_sign_byte(index);
        even = even && (std::popcount(static_cast<unsigned>(byte)) % 2 == 0) && (byte & 127U) == index;
    }
    expect(even, "IQ2_XXS sign bytes keep the index and have even parity");
    expect(strix::ds41::iq2_xxs_sign_byte(1U) == 129U && strix::ds41::iq2_xxs_sign_byte(3U) == 3U, "IQ2_XXS sign byte examples");
}

void test_scalars() {
    using namespace strix::ds41;
    expect(e4m3_to_float(0x38U) == 1.0F && e4m3_to_float(0x7eU) == 448.0F && e4m3_to_float(0xfeU) == -448.0F,
        "E4M3 normal values");
    expect(e4m3_to_float(0x01U) == std::ldexp(1.0F, -9) && e4m3_to_float(0x08U) == std::ldexp(1.0F, -6),
        "E4M3 smallest subnormal and normal");
    expect(std::isnan(e4m3_to_float(0x7fU)) && std::isnan(e4m3_to_float(0xffU)) && bits(e4m3_to_float(0x80U)) == 0x80000000U,
        "E4M3 NaN codes and negative zero");
    bool round_trip = true;
    for (unsigned c = 0; c < 256U; ++c) {
        if ((c & 0x7fU) == 0x7fU || c == 0x80U) continue;
        round_trip = round_trip && float_to_e4m3(e4m3_to_float(static_cast<std::uint8_t>(c))) == c;
    }
    expect(round_trip, "E4M3 encode inverts decode on every finite code");
    // 1.0625 lies halfway between 1.0 (0x38, even) and 1.125 (0x39): ties to even.
    expect(float_to_e4m3(1.0625F) == 0x38U && float_to_e4m3(1.1875F) == 0x3aU, "E4M3 encode rounds ties to even");
    expect(float_to_e4m3(1000.0F) == 0x7eU && float_to_e4m3(-1e30F) == 0xfeU && float_to_e4m3(NAN) == 0x7fU,
        "E4M3 encode saturates and maps NaN");
    expect(e8m0_to_float(127U) == 1.0F && e8m0_to_float(0U) == std::ldexp(1.0F, -127) && std::isnan(e8m0_to_float(255U)),
        "E8M0 values");
    expect(e8m0_half_to_float(128U) == 1.0F && e8m0_half_to_float(1U) == std::ldexp(1.0F, -127) &&
           e8m0_half_to_float(0U) == std::ldexp(1.0F, -128), "E8M0/2 values");
    expect(e2m1_to_float(7U) == 6.0F && e2m1_to_float(9U) == -0.5F && e2m1_to_float(1U) == 0.5F, "E2M1 values");
}

void test_blocks() {
    using namespace strix::ds41;
    // IQ2_XXS: d = 1, group 0: grid entry 0 (all 8), sign index 1 (element 0 and parity element 7), scale nibble 15.
    std::array<std::uint8_t, iq2_xxs_block_bytes> iq{};
    iq[0] = 0x00U; iq[1] = 0x3cU;
    const std::uint32_t aux1 = (15U << 28U) | 1U;
    for (unsigned b = 0; b < 4U; ++b) iq[6U + b] = static_cast<std::uint8_t>(aux1 >> (8U * b));
    std::array<float, 256> values{};
    decode_iq2_xxs(iq, values);
    expect(values[0] == -31.0F && values[7] == -31.0F && values[1] == 31.0F && values[8] == 31.0F,
        "IQ2_XXS decodes d (0.5 + s) / 4 times signed grid values");
    // Q2_K: d = 1, dmin = 0.5, scale byte of half 0 group 0: scale 3, min 2; first code 3 -> 3*3 - 0.5*2 = 8.
    std::array<std::uint8_t, q2_k_block_bytes> q2{};
    q2[0] = 0x23U; q2[16] = 0x03U; q2[80] = 0x00U; q2[81] = 0x3cU; q2[82] = 0x00U; q2[83] = 0x38U;
    decode_q2_k(q2, values);
    expect(values[0] == 8.0F && values[1] == -1.0F, "Q2_K decodes d sc q - dmin m");
    // Group 1 of half 0 reads bits 2-3 of the same bytes with scale bytes 2 and 3.
    q2[2] = 0x01U; q2[16] = 0x0cU;
    decode_q2_k(q2, values);
    expect(values[32] == 3.0F, "Q2_K group 1 uses the next shift and scale pair");
    // MXFP4: e = 128 is 2^1 for E2M1 values, i.e. 2^0 for the doubled codes: nibbles 7 (12) and 9 (-1).
    std::array<std::uint8_t, mxfp4_block_bytes> mx{};
    mx[0] = 128U; mx[1] = 0x97U;
    std::array<float, 32> small{};
    decode_mxfp4(mx, small);
    expect(small[0] == 12.0F && small[16] == -1.0F, "MXFP4 low nibbles are elements 0-15, high nibbles 16-31");
}

void test_products() {
    using namespace strix::ds41;
    std::mt19937 random(7U);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const auto fill = [&](std::vector<std::uint8_t>& row, std::size_t block, std::size_t half_offset) {
        for (auto& b : row) b = static_cast<std::uint8_t>(byte(random));
        for (std::size_t at = half_offset; at + 1U < row.size(); at += block) { row[at] = 0x00U; row[at + 1U] = 0x2cU; }
    };
    std::vector<float> activation(5120);
    for (auto& v : activation) v = normal(random);
    const auto x = quantize_a8(activation);
    const auto oracle = [&](const std::vector<float>& weights) {
        double sum = 0.0, magnitude = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            const double a = static_cast<double>(x.codes[i]) * static_cast<double>(x.deltas[i / 32U]);
            sum += a * weights[i]; magnitude += std::fabs(a * weights[i]);
        }
        return std::pair{sum, magnitude};
    };
    {
        std::vector<std::uint8_t> row(20U * iq2_xxs_block_bytes);
        fill(row, iq2_xxs_block_bytes, 0U);
        std::vector<float> weights(5120);
        for (std::size_t b = 0; b < 20U; ++b)
            decode_iq2_xxs(std::span<const std::uint8_t, iq2_xxs_block_bytes>(row.data() + b * iq2_xxs_block_bytes, iq2_xxs_block_bytes),
                std::span<float, 256>(weights.data() + 256U * b, 256U));
        const auto [sum, magnitude] = oracle(weights);
        expect(std::fabs(product_iq2_xxs(row, x, 5120U) - sum) <= 1e-5 * magnitude, "IQ2_XXS product agrees with the decoded dot");
    }
    {
        std::vector<std::uint8_t> row(9U * q2_k_block_bytes);
        for (auto& b : row) b = static_cast<std::uint8_t>(byte(random));
        for (std::size_t b = 0; b < 9U; ++b) {
            row[b * q2_k_block_bytes + 80U] = 0x00U; row[b * q2_k_block_bytes + 81U] = 0x2cU;
            row[b * q2_k_block_bytes + 82U] = 0x00U; row[b * q2_k_block_bytes + 83U] = 0x28U;
        }
        std::vector<float> weights(2304);
        for (std::size_t b = 0; b < 9U; ++b)
            decode_q2_k(std::span<const std::uint8_t, q2_k_block_bytes>(row.data() + b * q2_k_block_bytes, q2_k_block_bytes),
                std::span<float, 256>(weights.data() + 256U * b, 256U));
        const auto narrow = quantize_a8(std::span<const float>(activation.data(), 2304U));
        double sum = 0.0, magnitude = 0.0;
        for (std::size_t i = 0; i < 2304U; ++i) {
            const double a = static_cast<double>(narrow.codes[i]) * static_cast<double>(narrow.deltas[i / 32U]);
            sum += a * weights[i]; magnitude += std::fabs(a * weights[i]);
        }
        expect(std::fabs(product_q2_k(row, narrow, 2304U) - sum) <= 1e-5 * magnitude, "Q2_K product agrees with the decoded dot");
    }
    {
        std::vector<std::uint8_t> row(160U * mxfp4_block_bytes);
        for (auto& b : row) b = static_cast<std::uint8_t>(byte(random));
        for (std::size_t b = 0; b < 160U; ++b) row[b * mxfp4_block_bytes] = static_cast<std::uint8_t>(120 + byte(random) % 10);
        std::vector<float> weights(5120);
        for (std::size_t b = 0; b < 160U; ++b)
            decode_mxfp4(std::span<const std::uint8_t, mxfp4_block_bytes>(row.data() + b * mxfp4_block_bytes, mxfp4_block_bytes),
                std::span<float, 32>(weights.data() + 32U * b, 32U));
        const auto [sum, magnitude] = oracle(weights);
        expect(std::fabs(product_mxfp4(row, x, 5120U) - sum) <= 1e-5 * magnitude, "MXFP4 product agrees with the decoded dot");
    }
    bool rejected = false;
    try { static_cast<void>(product_mxfp4(std::vector<std::uint8_t>(10), x, 5120U)); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "a short row is rejected");
}

void test_a8() {
    using namespace strix::ds41;
    std::vector<float> values(96, 0.0F);
    values[32] = 127.0F; values[33] = -0.5F; values[34] = 1.5F;
    values[64] = 1e-37F;
    const auto x = quantize_a8(values);
    expect(x.deltas[0] == 1.0F && x.sums[0] == 0, "A8 zero group has delta 1 and zero codes");
    expect(x.deltas[1] == 1.0F && x.codes[32] == 127 && x.codes[33] == 0 && x.codes[34] == 2 && x.sums[1] == 129,
        "A8 codes round to nearest even");
    expect(x.deltas[2] == 1.0F && x.codes[64] == 0, "A8 tiny groups quantize to zero");
}

void test_moe_arithmetic() {
    using namespace strix::ds41;
    double worst = 0.0;
    for (int i = -200000; i <= 175000; ++i) {
        const float x = static_cast<float>(i) * 5e-4F;
        const double expected = std::exp(static_cast<double>(x));
        const double got = declared_exp(x);
        if (expected > 1e-37) worst = std::max(worst, std::fabs(got - expected) / expected);
    }
    expect(worst < 4.0 * 0x1p-24, "declared exp is within 4 ulp of exp on [-100, 87.5]");
    expect(declared_exp(0.0F) == 1.0F && declared_exp(89.0F) == INFINITY && declared_exp(-104.0F) == 0.0F &&
           std::isnan(declared_exp(NAN)), "declared exp special values");
    expect(swiglu_clamped(20.0F, 1.0F, 10.0F) == swiglu_clamped(10.0F, 1.0F, 10.0F) &&
           swiglu_clamped(1.0F, 50.0F, 10.0F) == swiglu_clamped(1.0F, 10.0F, 10.0F) &&
           swiglu_clamped(1.0F, -50.0F, 10.0F) == swiglu_clamped(1.0F, -10.0F, 10.0F) &&
           swiglu_clamped(-30.0F, 1.0F, 10.0F) < 0.0F && swiglu_clamped(-30.0F, 1.0F, 10.0F) > -1e-11F,
        "SwiGLU clamps gate from above and up on both sides");
    expect(std::fabs(swiglu_clamped(1.0F, 2.0F, 10.0F) - 2.0 / (1.0 + std::exp(-1.0))) < 1e-6, "SwiGLU value");
    // Two rows of top-3: row 0 {200, 5, 191}, row 1 {5, 300, 7}; split 192.
    const std::vector<std::uint32_t> ids{200, 5, 191, 5, 300, 7};
    const auto routes = build_window_routes(ids, 2U, 3U, 192U);
    expect(routes.experts == std::vector<std::uint32_t>{5, 7, 191, 200, 300}, "routes list the union in ascending id");
    expect(routes.begins == std::vector<std::uint32_t>{0, 2, 3, 4, 5, 6} && routes.max_rows == 2U, "routes CSR begins");
    expect(routes.rows == std::vector<std::uint32_t>{0, 1, 1, 0, 0, 1} && routes.slots == std::vector<std::uint32_t>{1, 3, 5, 2, 0, 4},
        "routes entries carry rows and slots in ascending row");
    expect(routes.order == std::vector<std::uint32_t>{1, 2, 0, 3, 5, 4} && routes.low_counts == std::vector<std::uint32_t>{2, 2},
        "combine order is ascending expert id with the low-half count");
    bool rejected = false;
    try { static_cast<void>(build_window_routes(std::vector<std::uint32_t>{1, 1}, 1U, 2U, 192U)); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "an expert repeated within a row is rejected");
}

void test_router_and_halves() {
    using namespace strix::ds41;
    double worst = 0.0;
    for (int i = -4000; i <= 4000; ++i) {
        const float y = std::ldexp(1.0F + static_cast<float>(i + 4000) / 8001.0F, i / 100);
        worst = std::max(worst, std::fabs(declared_log(y) - std::log(static_cast<double>(y))) /
            std::max(1.0, std::fabs(std::log(static_cast<double>(y)))));
    }
    expect(worst < 4.0 * 0x1p-24, "declared log is within 4 ulp (relative to max(1, |log y|))");
    double worst1p = 0.0;
    for (int i = -60; i <= 20; ++i) {
        const float e = std::ldexp(1.37F, i);
        worst1p = std::max(worst1p, std::fabs(declared_log1p(e) - std::log1p(static_cast<double>(e))) / std::log1p(static_cast<double>(e)));
    }
    expect(worst1p < 8.0 * 0x1p-24, "declared log1p keeps its relative accuracy for small arguments");
    expect(router_score(25.0F) == std::sqrt(25.0F) && std::fabs(router_score(0.0F) - std::sqrt(std::log(2.0))) < 1e-7,
        "router score is sqrt(softplus) with the threshold");
    // Two rows, four experts, hidden 32. Row 0: scores tie on experts 1 and 2 after bias; row 1 uses the image bias.
    std::vector<float> weight(4U * 32U, 0.0F), x(2U * 32U, 0.0F);
    for (unsigned e = 0; e < 4U; ++e) weight[e * 32U] = 1.0F;
    x[0] = 0.5F; x[32] = 2.0F;
    const std::vector<float> bias{0.0F, 0.1F, 0.1F, -1.0F}, image{-1.0F, -1.0F, -1.0F, 0.5F};
    const std::vector<std::uint8_t> image_rows{0, 1};
    const auto out = reference_router(x, 2U, 32U, weight, 4U, bias, image, image_rows, 2U, 1.5F);
    expect(out.ids == std::vector<std::uint32_t>{1, 2, 3, 0}, "router picks by score + bias, lower id first on ties, image bias per row");
    const float s0 = router_score(0.5F);
    expect(out.weights[0] == (s0 / ((s0 + s0) + 1e-20F)) * 1.5F, "router weights are normalized unbiased scores times the scale");
    // Halves reproduce the combine.
    const std::vector<std::uint32_t> ids{3, 0, 2, 1};
    const auto routes = build_window_routes(ids, 2U, 2U, 2U);
    const auto lower = restrict_routes(routes, 0U, 2U);
    expect(lower.experts == std::vector<std::uint32_t>{0, 1} && lower.order == routes.order, "restricted routes keep the owned experts");
    const std::vector<float> raw{1.5F, -2.0F, 0.25F, 3.0F, 7.0F, -1.0F, 2.5F, 0.125F}, w{0.3F, 0.7F, 0.6F, 0.4F};
    const auto low = reference_half_sum(routes, w, raw, 2U, 2U, 2U, 0U), high = reference_half_sum(routes, w, raw, 2U, 2U, 2U, 1U);
    // Row 0: slots 0 (expert 3) and 1 (expert 0): low = w1 * raw1, high = w0 * raw0.
    expect(low[0] == 0.7F * 0.25F && high[0] == 0.3F * 1.5F, "half sums split a row's slots by expert id");
}

} // namespace

int main() {
    test_tables();
    test_scalars();
    test_blocks();
    test_products();
    test_a8();
    test_moe_arithmetic();
    test_router_and_halves();
    if (failures == 0) std::cout << "deepseek41 format tests passed\n";
    return failures == 0 ? 0 : 1;
}
