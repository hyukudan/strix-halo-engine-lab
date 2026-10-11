#include "strix/low_precision.hpp"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template<class Exception, class Function>
void rejects(Function function, const char* message) {
    try { function(); expect(false, message); }
    catch (const Exception&) {}
    catch (...) { expect(false, "wrong exception category"); }
}

// Independent constants/calculations: no call to operand decoding or corrections.
constexpr std::array<int, 16> nonlinear{
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
int signed_byte(unsigned code) { return code < 128 ? static_cast<int>(code) : static_cast<int>(code) - 256; }
int signed_nibble(unsigned code) { return code < 8 ? static_cast<int>(code) : static_cast<int>(code) - 16; }

void exhaustive_arithmetic() {
    using namespace strix::low_precision;
    for (int q = -128; q <= 127; ++q) {
        const auto digits = split_int8(q);
        const auto expected_high = static_cast<int>(std::floor(static_cast<double>(q) / 16.0));
        expect(digits.high == expected_high && digits.low == q - 16 * expected_high,
               "all I8 digits agree with mathematical floor");
        expect(reconstruct_int8(digits) == q, "all I8 digits reconstruct");
    }
    for (unsigned w = 0; w < 16; ++w) {
        for (unsigned a = 0; a < 256; ++a) {
            const std::array<std::uint8_t, 1> wc{static_cast<std::uint8_t>(w)};
            const std::array<std::uint8_t, 1> ac{static_cast<std::uint8_t>(a)};
            const auto av = signed_byte(a);
            for (const int az : {-128, 0, 127}) {
                const auto dot = affine_group_dot({CodeEncoding::unsigned_i4, wc, 0.25, 8},
                    {CodeEncoding::signed_i8, ac, 2.0, az});
                const auto direct = (static_cast<int>(w) - 8) * (av - az);
                expect(dot.corrected_dot == direct && dot.scaled_dot == 0.5 * direct,
                       "all U4/I8 codes and extreme offsets equal direct dequantized math");
            }
            const auto iq = affine_group_dot({CodeEncoding::iq4_index, wc, -0.25, 0},
                {CodeEncoding::signed_i8, ac, 2.0, 0});
            expect(iq.corrected_dot == nonlinear[w] * av,
                   "all nonlinear indices map to exact signed I8 values");
            const std::array<std::int8_t, 1> sw{static_cast<std::int8_t>(signed_nibble(w))};
            const std::array<std::int8_t, 1> sa{static_cast<std::int8_t>(av)};
            expect(two_i4_dot(sw, sa) == signed_nibble(w) * av,
                   "all 4096 signed I4/I8 products reconstruct using two I4 products");
        }
        for (unsigned a = 0; a < 16; ++a) {
            const std::array<std::uint8_t, 1> wc{static_cast<std::uint8_t>(w)};
            const std::array<std::uint8_t, 1> ac{static_cast<std::uint8_t>(a)};
            const auto dot = affine_group_dot({CodeEncoding::signed_i4, wc, 0.5, -8},
                {CodeEncoding::unsigned_i4, ac, 0.25, 15});
            expect(dot.corrected_dot == (signed_nibble(w) + 8) * (static_cast<int>(a) - 15),
                   "all signed/unsigned I4 products and offsets");
        }
    }
    const std::array<std::uint8_t, 4> w{0, 15, 9, 7};
    const std::array<std::uint8_t, 4> a{128, 127, 3, 255};
    std::int64_t direct = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        direct += (static_cast<int>(w[i]) - 8) * (signed_byte(a[i]) + 2);
    }
    const auto first = affine_group_dot({CodeEncoding::unsigned_i4, w, 0.25, 8},
        {CodeEncoding::signed_i8, a, 2.0, -2});
    const auto second = affine_group_dot({CodeEncoding::unsigned_i4, w, -2.0, 8},
        {CodeEncoding::signed_i8, a, 0.25, -2});
    const std::array groups{first, second};
    expect(first.corrected_dot == direct && ordered_group_sum(groups) == 0.0,
           "unequal group scales are applied before canonical group combination");
    const std::array<GroupDot, 3> ordered{{{0, 0, 0, 0, 1e16}, {0, 0, 0, 0, -1e16}, {0, 0, 0, 0, 1.0}}};
    expect(ordered_group_sum(ordered) == 1.0, "floating groups preserve declared order");
}

void quantization_and_rows() {
    using namespace strix::low_precision;
    expect(quantization_identity(ActivationEncoding::symmetric_a8, 32) ==
        quantization_identity(ActivationEncoding::symmetric_a8, 32), "quantization identity is reproducible");
    expect(quantization_identity(ActivationEncoding::symmetric_a8, 32) !=
        quantization_identity(ActivationEncoding::symmetric_a4, 32) &&
        quantization_identity(ActivationEncoding::symmetric_a8, 32) !=
        quantization_identity(ActivationEncoding::symmetric_a8, 64), "encoding and scale groups distinguish quantization identities");
    for (const auto encoding : {ActivationEncoding::symmetric_a8, ActivationEncoding::symmetric_a4}) {
        const int maximum = encoding == ActivationEncoding::symmetric_a8 ? 127 : 7;
        const auto code_encoding = encoding == ActivationEncoding::symmetric_a8 ? CodeEncoding::signed_i8 : CodeEncoding::signed_i4;
        const std::array<double, 8> values{-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, static_cast<double>(maximum), 0};
        const auto row = quantize_row(values, encoding, 8);
        const std::array<int, 8> expected{-2, -2, 0, 0, 2, 2, maximum, 0};
        expect(row.scales[0] == 1.0, "ties fixture has unit scale");
        for (std::size_t i = 0; i < values.size(); ++i) {
            expect(decode_code(code_encoding, row.codes[i]) == expected[i], "positive/negative ties round even");
        }
        const std::array<double, 8> zero{-0.0, 0, 0, 0, 0, 0, 0, 0};
        const auto z = quantize_row(zero, encoding, 8);
        expect(z.scales[0] == 1.0 && z.codes == std::vector<std::uint8_t>(8, 0), "zero group has unit scale and zero codes");
        const std::array<double, 8> other{-12.3, 200, 1e-300, -10, 3, 6, -999, 31};
        const auto alone = quantize_row(other, encoding, 8);
        // Independent nearbyint reference rather than the implementation's tie routine.
        for (std::size_t i = 0; i < other.size(); ++i) {
            const auto expected_code = static_cast<int>(std::nearbyint(other[i] / (999.0 / maximum)));
            expect(decode_code(code_encoding, alone.codes[i]) == expected_code, "independent row quantization calculation");
        }
        const std::array<std::span<const double>, 5> batch{values, other, zero, other, values};
        for (const auto input : batch) {
            const auto b = quantize_row(input, encoding, 8);
            if (input.data() == other.data()) {
                expect(b.codes == alone.codes && b.scales == alone.scales,
                       "row alone equals duplicate/reordered/unrelated batch rows");
            }
        }
        const std::array<double, 2> tiny{1e-300, -1e-300};
        expect(quantize_row(tiny, encoding, 2).scales[0] > 0.0, "small finite scales retained");
        const std::array<double, 2> huge{std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
        expect(quantize_row(huge, encoding, 2).codes.size() == 2, "extreme finite rows avoid reciprocal overflow");
    }
    const std::array<double, 1> nonfinite{std::numeric_limits<double>::infinity()};
    rejects<std::invalid_argument>([&] { (void)quantize_row(nonfinite, ActivationEncoding::symmetric_a8, 1); }, "infinite activations rejected");
    const std::array<double, 1> nan{std::numeric_limits<double>::quiet_NaN()};
    rejects<std::invalid_argument>([&] { (void)quantize_row(nan, ActivationEncoding::symmetric_a4, 1); }, "NaN activations rejected");
    const std::array<double, 1> underflow{std::numeric_limits<double>::denorm_min()};
    rejects<std::overflow_error>([&] { (void)quantize_row(underflow, ActivationEncoding::symmetric_a8, 1); }, "unrepresentable scale rejected");
    const std::array<double, 3> bad_group{1, 2, 3};
    rejects<std::invalid_argument>([&] { (void)quantize_row(bad_group, ActivationEncoding::symmetric_a8, 2); }, "activation group tails rejected");
    const auto old_round = std::fegetround();
    (void)std::fesetround(FE_DOWNWARD);
    rejects<std::invalid_argument>([&] { (void)quantize_row(bad_group, ActivationEncoding::symmetric_a8, 1); }, "undeclared floating rounding mode rejected");
    (void)std::fesetround(old_round);
}

void iq4_block_reference() {
    using namespace strix::low_precision;
    std::array<std::byte, 136> block{};
    block[0] = std::byte{0x00}; block[1] = std::byte{0x38}; // global=0.5
    const std::array<unsigned, 8> locals{0, 1, 31, 32, 33, 48, 62, 63};
    unsigned high = 0;
    for (std::size_t g = 0; g < 8; ++g) {
        high |= (locals[g] >> 4U) << static_cast<unsigned>(g * 2);
        const auto shift = static_cast<unsigned>((g % 2) * 4);
        block[4 + g / 2] |= static_cast<std::byte>((locals[g] & 15U) << shift);
        for (std::size_t j = 0; j < 16; ++j) {
            block[8 + g * 16 + j] = static_cast<std::byte>(j | ((15 - j) << 4U));
        }
    }
    block[2] = static_cast<std::byte>(high & 255U);
    block[3] = static_cast<std::byte>(high >> 8U);
    for (std::size_t g = 0; g < 8; ++g) {
        for (std::size_t j = 0; j < 32; ++j) {
            const auto nibble = j < 16 ? j : 31 - j;
            const auto direct_weight = 0.5 * (static_cast<int>(locals[g]) - 32) * nonlinear[nibble];
            expect(strix::iq4_xs::dequantized_value(block, g * 32 + j) == direct_weight,
                   "all packed subgroups agree with existing floating dequantizer and independent math");
        }
    }
    std::array<double, 256> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<double>(static_cast<int>(i % 29) - 14);
    }
    const auto row = quantize_row(values, ActivationEncoding::symmetric_a8);
    double expected = 0;
    for (std::size_t g = 0; g < 8; ++g) {
        std::int64_t dot = 0;
        for (std::size_t j = 0; j < 32; ++j) {
            const auto nibble = j < 16 ? j : 31 - j;
            dot += nonlinear[nibble] * signed_byte(row.codes[g * 32 + j]);
        }
        expected += (0.5 * (static_cast<int>(locals[g]) - 32) * row.scales[g]) * static_cast<double>(dot);
    }
    expect(iq4_xs_quantized_dot(block, row) == expected,
           "packed IQ4_XS eight local scales and nonlinear indices equal independent scalar formula");
    block[1] = std::byte{0xb8}; // negative global=-0.5
    expect(iq4_xs_quantized_dot(block, row) == -expected, "negative global scale preserves packed group interpretation");
    for (std::size_t g = 0; g < 8; ++g) {
        for (std::size_t j = 0; j < 32; ++j) {
            const auto nibble = j < 16 ? j : 31 - j;
            const auto direct_weight = -0.5 * (static_cast<int>(locals[g]) - 32) * nonlinear[nibble];
            expect(strix::iq4_xs::dequantized_value(block, g * 32 + j) == direct_weight,
                   "negative global and local scales cross-check existing dequantizer");
        }
    }
    block[1] = std::byte{0x38};
    auto broken = row;
    broken.scales[0] = std::numeric_limits<double>::quiet_NaN();
    rejects<std::invalid_argument>([&] { (void)iq4_xs_quantized_dot(block, broken); }, "IQ4 dot rejects nonfinite activation scale");
    broken = row; broken.scales[0] = 0;
    rejects<std::invalid_argument>([&] { (void)iq4_xs_quantized_dot(block, broken); }, "IQ4 dot rejects zero activation scale");
    broken = row; broken.codes[0] = 128;
    rejects<std::invalid_argument>([&] { (void)iq4_xs_quantized_dot(block, broken); }, "IQ4 dot enforces symmetric A8 excludes minus128");
    block[1] = std::byte{0x7c};
    rejects<std::invalid_argument>([&] { (void)iq4_xs_quantized_dot(block, row); }, "IQ4 dot rejects nonfinite checkpoint scale");
    broken = row; broken.codes.resize(640); broken.scales.resize(20);
    rejects<std::invalid_argument>([&] { (void)iq4_xs_quantized_dot(block, broken); }, "K640 cannot fabricate IQ4_XS tail blocks");
}

void rejection_and_bounds() {
    using namespace strix::low_precision;
    rejects<std::invalid_argument>([] { (void)decode_code(CodeEncoding::unsigned_i4, 16); }, "invalid nibble rejected");
    rejects<std::invalid_argument>([] { (void)decode_code(static_cast<CodeEncoding>(99), 0); }, "invalid interpretation rejected");
    rejects<std::invalid_argument>([] { (void)split_int8(-129); }, "I8 split input domain enforced");
    rejects<std::invalid_argument>([] { (void)reconstruct_int8({-8, 16}); }, "low digit domain enforced");
    rejects<std::overflow_error>([] { (void)correct_dot(0, std::numeric_limits<std::int64_t>::min(), 0, 1, 0, 1); }, "negative INT64 correction cannot wrap");
    rejects<std::overflow_error>([] { (void)correct_dot(0, 0, 0, std::numeric_limits<std::uint64_t>::max(), 1, 1); }, "group length overflow rejected");
    rejects<std::overflow_error>([] { (void)correct_dot(0, 0, 0, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()), 127, 127); }, "wide correction multiplication overflow rejected");
    rejects<std::overflow_error>([] { (void)correct_dot(1, 0, std::numeric_limits<std::int64_t>::max(), 1, -1, 0); }, "wide correction addition overflow rejected");
    std::vector<std::uint8_t> large(140'000, 127);
    rejects<std::overflow_error>([&] { (void)affine_group_dot({CodeEncoding::signed_i8, large, 1, 0}, {CodeEncoding::signed_i8, large, 1, 0}); }, "INT32 raw accumulation checked without saturation");
    large.assign(140'000, 0);
    rejects<std::overflow_error>([&] { (void)affine_group_dot({CodeEncoding::signed_i8, large, 1, 127}, {CodeEncoding::signed_i8, large, 1, 127}); }, "corrected INT32 overflow checked despite zero raw dot");
    const std::array<std::uint8_t, 1> code{1};
    rejects<std::invalid_argument>([&] { (void)affine_group_dot({CodeEncoding::iq4_index, code, 1, 1}, {CodeEncoding::signed_i8, code, 1, 0}); }, "IQ4 codebook has no integer zero-point reinterpretation");
    rejects<std::overflow_error>([&] { (void)affine_group_dot({CodeEncoding::unsigned_i4, code, std::numeric_limits<double>::max(), 0}, {CodeEncoding::signed_i8, code, 2, 0}); }, "scale product overflow rejected");
    const std::array<std::uint8_t, 1> zero{0};
    expect(affine_group_dot({CodeEncoding::signed_i8, zero, 1, 0}, {CodeEncoding::signed_i8, zero, 1, 0}).scaled_dot == 0,
           "operand zero gives exact mathematical zero");
    expect(decode_code(CodeEncoding::iq4_index, 0) != 0, "stored nonlinear nibble zero is not padding zero");
}

void candidate_accounting() {
    using namespace strix::low_precision;
    CandidateRequest r;
    const auto one = plan_candidate(r);
    expect(one.eligible && one.weight_bytes == 10 * 136 * 2560 && one.storage_block_elements == 256,
           "IQ4_XS actual physical bytes and storage groups");
    expect(one.issued_products == 16 * one.useful_products && one.matrix_tiles == 160 * 160,
           "one actual row only uses one of sixteen tile rows");
    expect(one.preparation_elements == 2560 && one.activation_scale_bytes == 80 * 8,
           "row preparation and binary64 scales counted separately");
    r.actual_rows = 16;
    const auto sixteen = plan_candidate(r);
    expect(sixteen.issued_products == sixteen.useful_products && sixteen.ideal_weight_read_bytes == one.weight_bytes,
           "16 actual rows fill tile and reuse one ideal weight read");
    expect(sixteen.serial_weight_read_bytes == 16 * one.weight_bytes, "serial weight upper bound counts actual rows");
    r.input_features = 640;
    expect(!plan_candidate(r).eligible, "K640 is incompatible with 256-element IQ4_XS storage");
    r.storage = StorageEncoding::iq4_nl;
    expect(plan_candidate(r).eligible && plan_candidate(r).weight_bytes == 20 * 18 * 2560,
           "K640 fits existing 32-element IQ4_NL storage");
    r.path = ExecutionPath::integer4; r.activation = ActivationEncoding::symmetric_a4;
    expect(!plan_candidate(r).eligible, "nonlinear index IU4 candidate rejected");
    r.storage = StorageEncoding::q4_0;
    const auto four = plan_candidate(r);
    expect(four.eligible && four.prepared_activation_bytes == 16 * 640 / 2,
           "Q4_0 affine codes plus explicit A4 are representable");
    r.path = ExecutionPath::two_integer4; r.activation = ActivationEncoding::symmetric_a8;
    const auto two = plan_candidate(r);
    expect(two.issued_products == 2 * four.issued_products && two.matrix_tiles == 2 * four.matrix_tiles,
           "two I4 products count twice the physical arithmetic");
    r.actual_rows = 3; r.output_features = 17; r.tile = {16, 16, 48};
    const auto padded = plan_candidate(r);
    expect(padded.useful_products == 3 * 17 * 640 && padded.issued_products == 2 * 16 * 32 * 672,
           "M/N/K tile padding independent of exact storage geometry");
    expect(padded.masked_output_elements == 16 * 32 - 3 * 17, "masked output elements counted");
    r.path = ExecutionPath::packed_float; r.activation = ActivationEncoding::float32_control;
    expect(plan_candidate(r).issued_products == 3 * 17 * 640, "packed scalar control does not invent matrix padding");
    r.actual_rows = std::numeric_limits<std::uint64_t>::max();
    rejects<std::overflow_error>([&] { (void)plan_candidate(r); }, "candidate byte/work overflow rejected");
    r.actual_rows = 0;
    rejects<std::invalid_argument>([&] { (void)plan_candidate(r); }, "empty expert is skipped rather than zero-sized candidate");
    r.actual_rows = 1; r.storage = static_cast<StorageEncoding>(99);
    rejects<std::invalid_argument>([&] { (void)plan_candidate(r); }, "unsupported encoding fails closed");
}
} // namespace

int main() {
    exhaustive_arithmetic();
    quantization_and_rows();
    iq4_block_reference();
    rejection_and_bounds();
    candidate_accounting();
    if (failures != 0) { return 1; }
    std::cout << "low-precision portable references passed; no native/model qualification\n";
    return 0;
}
