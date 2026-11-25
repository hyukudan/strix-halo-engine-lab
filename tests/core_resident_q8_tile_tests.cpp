#include "core/iu4_onfly.hpp"
#include "core/q8_f16_staging.hpp"
#include "core/resident_row_plan.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
namespace core = strix::core;
namespace staging = core::q8_f16_staging;
namespace weight = strix::weight;

double half_value(std::uint16_t bits) {
    const auto exponent = static_cast<int>((bits >> 10U) & 31U);
    const unsigned mantissa = bits & 1023U;
    const double value = exponent == 31 ? std::numeric_limits<double>::infinity() :
        std::ldexp(static_cast<double>(mantissa + (exponent == 0 ? 0U : 1024U)),
                   exponent == 0 ? -24 : exponent - 25);
    return (bits & 0x8000U) != 0U ? -value : value;
}

// Independent scalar RNE oracle using adjacent representable FP16 values.
std::uint16_t reference_half(double value, const std::vector<double>& positives) {
    if (std::isnan(value)) return 0x7e00U;
    const auto sign = std::signbit(value) ? 0x8000U : 0U;
    value = std::abs(value);
    if (value >= 65520.0) return static_cast<std::uint16_t>(sign | 0x7c00U);
    const auto it = std::lower_bound(positives.begin(), positives.end(), value);
    auto high = static_cast<unsigned>(it - positives.begin());
    if (it == positives.end()) --high;
    else if (high != 0U) {
        const double down = value - positives[high - 1U], up = positives[high] - value;
        if (down < up || (down == up && (high & 1U) != 0U)) --high;
    }
    return static_cast<std::uint16_t>(sign | high);
}

bool check_decode() {
    constexpr std::array<std::uint16_t, 14> edges{0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
                                                   0x3c01, 0xbc01, 0x7bff, 0xfbff, 0x7c00, 0xfc00};
    std::vector<double> positives(0x7c00);
    for (std::size_t i = 0; i < positives.size(); ++i) positives[i] = half_value(static_cast<std::uint16_t>(i));
    std::mt19937 rng(20261122U);
    for (unsigned group = 0; group < 4096U; ++group) {
        auto scale_bits = group < edges.size() ? edges[group] : static_cast<std::uint16_t>(rng());
        if ((scale_bits & 0x7fffU) > 0x7c00U) scale_bits = 0x7c00U;
        std::array<std::uint8_t, weight::Q8Storage::bytes> block{};
        block[0] = static_cast<std::uint8_t>(scale_bits);
        block[1] = static_cast<std::uint8_t>(scale_bits >> 8U);
        for (unsigned k = 2; k < block.size(); ++k) block[k] = static_cast<std::uint8_t>(rng());
        block[2] = 0;
        block[3] = 128;
        block[18] = 127;
        block[19] = 255;

        const float scale = core::onfly::f16_bits_to_f32(weight::Q8Storage::scale_bits(block.data()));
        const auto pair = [scale](std::uint32_t biased, unsigned high) {
            const auto operand = [scale, biased](unsigned byte) {
                const auto code = static_cast<int>((biased >> (8U * byte)) & 255U) - 128;
                const float value = static_cast<float>(code) * scale;
                // NaN payload and sign are outside the operand contract.
                return std::isnan(value) ? std::uint16_t{0x7e00U} : core::onfly::f32_to_f16_bits(value);
            };
            return std::uint32_t{operand(2U * high)} | (std::uint32_t{operand(2U * high + 1U)} << 16U);
        };
        for (unsigned half = 0; half < 2U; ++half) {
            std::array<std::uint32_t, 4> codes{};
            for (unsigned i = 0; i < staging::kHalfCodes; ++i)
                codes[i / 4U] |= std::uint32_t{block[staging::code_offset(half) + i]} << (8U * (i % 4U));
            std::array<std::uint32_t, 8> words{};
            staging::decode(codes, words, pair);
            for (unsigned i = 0; i < staging::kHalfCodes; ++i) {
                const auto actual = static_cast<std::uint16_t>(words[i / 2U] >> (16U * (i % 2U)));
                const std::size_t lane = half * 16U + i;
                const int code = weight::Q8Storage::code(block.data(), lane);
                if (actual != reference_half(static_cast<double>(code) * half_value(scale_bits), positives)) {
                    std::cerr << "staging mismatch at block " << group << " code lane " << lane << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

bool check_admission() {
    // Admission reads the pointers' values only (alignment, byte ranges), never the memory: the activation and output
    // addresses stand for buffers large enough for every row count below and far apart (2^20 / 2^44 bytes).
    const auto* input = reinterpret_cast<const float*>(std::uintptr_t{1} << 20U);
    auto* output = reinterpret_cast<float*>(std::uintptr_t{1} << 44U);
    alignas(16) std::array<std::uint8_t, 2U * 68U> packed{};
    const weight::ResidentMatrixView matrix{packed, weight::kQ8WeightStorage, 2, 64, 68};
    const auto admits = [&](const weight::ResidentMatrixView& view, unsigned rows, unsigned outputs = 1U) {
        return core::valid_resident_q8_rows(view, outputs, input, rows, output);
    };
    constexpr auto row_limit = 65535U * core::f16_tile_gemm::kBM;
    for (unsigned rows : {1U, 2U, 4U, 8U, 16U})
        if (!admits(matrix, rows) || core::use_resident_q8_tile(rows)) return false;
    for (unsigned rows : {17U, 128U, 129U, 2048U, row_limit})
        if (!admits(matrix, rows) || !core::use_resident_q8_tile(rows)) return false;
    if (!admits(matrix, row_limit + 1U) || core::use_resident_q8_tile(row_limit + 1U)) return false;

    auto bad = matrix;
    bad.columns = 48U;
    if (admits(bad, 17U) || admits(matrix, 0U) || admits(matrix, 17U, 3U)) return false;
    bad = matrix;
    --bad.row_bytes;
    if (admits(bad, 17U)) return false;
    bad = matrix;
    bad.bytes = bad.bytes.subspan(1);
    if (admits(bad, 17U)) return false;
    bad.bytes = std::span(packed).first(packed.size() - 1U);
    // A misaligned activation, and an output inside the activation rows (aligned partial overlap), are refused.
    return !admits(bad, 17U) && !core::valid_resident_q8_rows(matrix, 1U, input + 1, 17U, output) &&
        !core::valid_resident_q8_rows(matrix, 1U, input, 17U, const_cast<float*>(input) + 4) &&
        core::valid_resident_q8_rows(matrix, 1U, input, 17U, const_cast<float*>(input) + 17U * 64U);
}
}  // namespace

int main() {
    if (!check_decode() || !check_admission()) {
        std::cerr << "FAIL resident Q8_0 tile staging or admission\n";
        return 1;
    }
    std::cout << "PASS resident Q8_0 tile staging and admission\n";
}
