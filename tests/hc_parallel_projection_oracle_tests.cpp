#include "strix/hc_parallel_projection_oracle.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>

namespace oracle = strix::hc_parallel_projection_oracle;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejects(F function) {
    bool caught = false;
    try { function(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "invalid oracle input accepted");
}
void half_classification() {
    using oracle::half_to_float;
    require(std::bit_cast<std::uint32_t>(half_to_float(0U)) == 0U, "positive zero");
    require(std::bit_cast<std::uint32_t>(half_to_float(0x8000U)) == 0x80000000U, "negative zero");
    require(half_to_float(1U) == std::ldexp(1.0F, -24), "smallest subnormal");
    require(half_to_float(0x83ffU) == -std::ldexp(1023.0F, -24), "negative largest subnormal");
    require(half_to_float(0x400U) == std::ldexp(1.0F, -14), "smallest normal");
    require(half_to_float(0x3c00U) == 1.0F && half_to_float(0x7bffU) == 65504.0F, "finite normal");
    require(std::isinf(half_to_float(0x7c00U)) && std::signbit(half_to_float(0xfc00U)), "infinity");
    require(std::bit_cast<std::uint32_t>(half_to_float(0xfe11U)) == 0xffc22000U, "quiet NaN payload/sign");
    require(std::bit_cast<std::uint32_t>(half_to_float(0x7c01U)) == 0x7f802000U, "signaling NaN payload");
    for (unsigned word = 0; word < 65536U; ++word) {
        const float value = half_to_float(static_cast<std::uint16_t>(word));
        const auto exponent = (word >> 10U) & 31U, fraction = word & 1023U;
        require(std::signbit(value) == ((word & 0x8000U) != 0U), "exhaustive half sign");
        require(std::isnan(value) == (exponent == 31U && fraction != 0U), "exhaustive half NaN");
        require(std::isinf(value) == (exponent == 31U && fraction == 0U), "exhaustive half infinity");
        if (exponent != 31U) {
            const double magnitude = exponent == 0U ? std::ldexp(static_cast<double>(fraction), -24) :
                std::ldexp(1024.0 + fraction, static_cast<int>(exponent) - 25);
            require(static_cast<double>(value) == ((word & 0x8000U) ? -magnitude : magnitude), "exhaustive finite half value");
        }
    }
}
std::vector<std::byte> weights(oracle::Geometry g, std::uint16_t scale = 0x3c00U) {
    std::vector<std::byte> data(oracle::q8_bytes(g));
    for (std::size_t i = 0; i < data.size(); i += 34U) {
        data[i] = static_cast<std::byte>(scale & 255U);
        data[i + 1U] = static_cast<std::byte>(scale >> 8U);
    }
    return data;
}
void set_code(std::vector<std::byte>& data, std::size_t width, std::size_t row,
    std::size_t feature, int code) {
    const auto at = (row * (width / 32U) + feature / 32U) * 34U + 2U + feature % 32U;
    data.at(at) = static_cast<std::byte>(static_cast<unsigned>(code) & 255U);
}
void hand_cases_and_mutations() {
    const oracle::Geometry g{64, 2, 2, 2};
    auto w = weights(g);
    set_code(w, g.width, 0, 0, -128); set_code(w, g.width, 0, 32, 127);
    set_code(w, g.width, 1, 0, 1); set_code(w, g.width, 1, 63, -2);
    require(oracle::q8_coefficient(w, 64, 0, 0) == -128.0F &&
        oracle::q8_coefficient(w, 64, 0, 32) == 127.0F, "signed int8 decode/block stride");
    std::vector<float> n(g.width * g.rows), inject(g.streams * g.width);
    n[0] = 1; n[32] = 2; n[63] = 3; n[64] = 4; n[96] = 5; n[127] = 6;
    inject[0] = 1; inject[64 + 63] = -2;
    const auto result = oracle::evaluate(g, n, w, inject);
    const std::array<double, 4> expected{126, -5, 123, -8};
    for (std::size_t i = 0; i < expected.size(); ++i) require(result.down[i].dot == expected[i], "hand dot rows/output/block stride");
    require(result.down[0].sum_abs == 382 && result.down[2].sum_abs == 1147, "hand sum absolute products");
    require(result.down[0].scaled == 63 && result.injection[0].dot == 1 &&
        result.injection[1].dot == -6 && result.injection[2].dot == 4 && result.injection[3].dot == -12, "scale/injection row stride");
    require(result.down[1].postactivation == oracle::stable_silu(-2.5), "full postactivation");
    auto broken = w;
    // Deliberately substitute the other output row; a row-stride mutation.
    std::copy(w.begin() + 68, w.end(), broken.begin());
    require(oracle::evaluate(g, n, broken, inject).down[0].dot != result.down[0].dot, "broken row stride unobservable");
    auto dropped = w; set_code(dropped, 64, 1, 63, 0);
    require(oracle::evaluate(g, n, dropped, inject).down[3].dot != result.down[3].dot, "last feature dropped unobservable");
    auto input_stride = n; std::copy(n.begin(), n.begin() + 64, input_stride.begin() + 64);
    require(oracle::evaluate(g, input_stride, w, inject).down[2].dot != result.down[2].dot, "broken input row stride unobservable");
    require(oracle::evaluate(g, n, w).injection.empty(), "optional injection");
    rejects([&] { oracle::evaluate(g, std::span<const float>(n).first(n.size() - 1U), w); });
    rejects([&] { oracle::evaluate(g, n, std::span<const std::byte>(w).first(w.size() - 1U)); });
    rejects([&] { oracle::evaluate(g, n, w, std::span<const float>(inject).first(1)); });
    rejects([&] { oracle::q8_coefficient(w, 64, 2, 0); });
    rejects([&] { oracle::q8_bytes({33, 1, 1, 1}); });
    rejects([&] { oracle::q8_bytes({32, 0, 1, 1}); });
    rejects([&] { oracle::checked_product(std::numeric_limits<std::size_t>::max(), 2); });
}
void cancellation_metrics_and_nonfinite() {
    const oracle::Geometry g{32, 1, 1, 1};
    auto w = weights(g); for (unsigned k = 0; k < 3; ++k) set_code(w, 32, 0, k, 1);
    std::vector<float> n(32); n[0] = 16777216.0F; n[1] = 1; n[2] = -16777216.0F;
    const auto result = oracle::evaluate(g, n, w);
    require(result.down[0].dot == 1.0 && result.down[0].sum_abs == 33554433.0, "FP64 cancellation oracle");
    float serial = 0; for (unsigned k = 0; k < 3; ++k) serial = std::fma(n[k], 1.0F, serial);
    require(serial == 0, "FP32 cancellation witness");
    const std::array<float, 1> s{serial}, c{1};
    const auto errors = oracle::compare_pair(s, c, result.down, oracle::Stage::dot);
    require(errors.differing_fp32_words == 1 && errors.serial.max_absolute == 1 &&
        errors.serial.rmse == 1 && errors.serial.max_relative == 1 && errors.candidate.max_absolute == 0 &&
        errors.candidate.outputs[0].sum_abs == 33554433.0, "independent serial/candidate oracle metrics");
    auto dropped = w; set_code(dropped, 32, 0, 1, 0);
    require(oracle::evaluate(g, n, dropped).down[0].dot == 0, "drop-term mutant distinguishes cancellation oracle");
    require(oracle::float_ulp_distance(1.0F, std::nextafter(1.0F, 2.0F)) == 1, "one ULP");
    require(oracle::float_ulp_distance(-0.0F, 0.0F) == 0, "zero ULP convention");
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    require(oracle::stable_sigmoid(1000) == 1 && oracle::stable_sigmoid(-1000) == 0 &&
        oracle::stable_silu(1000) == 1000 && std::signbit(oracle::stable_silu(-1000)), "stable finite extremes");
    require(oracle::stable_sigmoid(-inf) == 0 && oracle::stable_sigmoid(inf) == 1 &&
        oracle::stable_silu(-inf) == 0 && std::signbit(oracle::stable_silu(-inf)) &&
        std::isnan(oracle::stable_silu(nan)), "mathematical nonfinite limits");
    std::array<oracle::Output, 4> expected{};
    expected[0].dot = nan; expected[1].dot = inf; expected[2].dot = -0.0; expected[3].dot = 0;
    const std::array<float, 4> actual{0, -std::numeric_limits<float>::infinity(), 0, 1};
    const auto mixed = oracle::compare(actual, expected, oracle::Stage::dot);
    require(mixed.classification_mismatches == 2 && mixed.reference_nan == 1 &&
        mixed.reference_inf == 1 && mixed.observed_inf == 1 && mixed.finite_pairs == 2 &&
        mixed.signed_zero_mismatches == 1 && mixed.zero_reference_nonzero_observed == 1, "nonfinite/signedzero metrics segregated");
    rejects([&] { oracle::compare(s, expected); });
    rejects([&] { oracle::float_ulp_distance(std::numeric_limits<float>::infinity(), 0); });
    auto nw = weights(g, 0x7c00U); set_code(nw, 32, 0, 0, 1);
    require(std::isinf(oracle::q8_coefficient(nw, 32, 0, 0)) &&
        std::isnan(oracle::q8_coefficient(nw, 32, 0, 1)), "Inf scale and zero code FP32 product");
    auto tiny = weights(g, 1U); set_code(tiny, 32, 0, 0, -128);
    require(oracle::q8_coefficient(tiny, 32, 0, 0) == -std::ldexp(128.0F, -24), "subnormal half signed coefficient");
}
} // namespace

int main() {
    try {
        half_classification(); hand_cases_and_mutations(); cancellation_metrics_and_nonfinite();
        std::cout << "HC parallel CPU oracle: all65536half patterns, hand rows/stride/cancellation, mutants and error metrics PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
