#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::hc_parallel_projection_oracle {

// CPU diagnostic mathematics only. No numerical acceptance threshold or device
// reduction order is implied. Inputs are already-normalized FP32 values.
struct Geometry {
    std::size_t width = 0, streams = 0, rank = 0, rows = 1;
};

inline float half_to_float(std::uint16_t bits) {
    const auto sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const auto exponent = (bits >> 10U) & 31U;
    const auto fraction = bits & 1023U;
    if (exponent == 0U) {
        if (fraction == 0U) return std::bit_cast<float>(sign);
        const float magnitude = std::ldexp(static_cast<float>(fraction), -24);
        return (sign != 0U) ? -magnitude : magnitude;
    }
    if (exponent == 31U)
        return std::bit_cast<float>(sign | 0x7f800000U |
            (static_cast<std::uint32_t>(fraction) << 13U));
    return std::bit_cast<float>(sign |
        ((static_cast<std::uint32_t>(exponent) + 112U) << 23U) |
        (static_cast<std::uint32_t>(fraction) << 13U));
}

inline std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0U && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("HC oracle geometry overflow");
    return a * b;
}

inline std::size_t q8_bytes(Geometry g) {
    if (g.width == 0U || g.width % 32U != 0U || g.streams == 0U ||
        g.rank == 0U || g.rows == 0U)
        throw std::invalid_argument("HC oracle invalid geometry");
    return checked_product(checked_product(g.rank, g.width / 32U), 34U);
}

// Original row-major Q8_0: little-endian FP16 scale, then 32 signed codes.
inline float q8_coefficient(std::span<const std::byte> weights,
    std::size_t width, std::size_t output, std::size_t feature) {
    if (width == 0U || width % 32U != 0U || feature >= width)
        throw std::invalid_argument("HC oracle invalid coefficient index");
    const auto row_bytes = checked_product(width / 32U, 34U);
    if (output >= weights.size() / row_bytes)
        throw std::invalid_argument("HC oracle coefficient owner extent");
    const auto at = output * row_bytes + (feature / 32U) * 34U;
    const auto lo = std::to_integer<unsigned>(weights[at]);
    const auto hi = std::to_integer<unsigned>(weights[at + 1U]);
    const auto byte = std::to_integer<unsigned>(weights[at + 2U + feature % 32U]);
    const int code = byte < 128U ? static_cast<int>(byte) : static_cast<int>(byte) - 256;
    // Materialize the FP32 coefficient before its promotion into the FP64 dot.
    const volatile float coefficient = half_to_float(
        static_cast<std::uint16_t>(lo | (hi << 8U))) * static_cast<float>(code);
    return coefficient;
}

inline double stable_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

inline double stable_silu(double x) {
    // Mathematical limit; nonfinite device classification is recorded separately.
    if (x == -std::numeric_limits<double>::infinity()) return -0.0;
    return x * stable_sigmoid(x);
}

struct Output {
    double dot = 0.0;
    double sum_abs = 0.0;
    double scaled = 0.0;
    double postactivation = 0.0;
};
struct Result {
    Geometry geometry;
    // Row-major [rows][rank] and [rows][streams], respectively.
    std::vector<Output> down, injection;
};

inline Result evaluate(Geometry g, std::span<const float> normalized,
    std::span<const std::byte> down, std::span<const float> injection = {}) {
    const auto required_down = q8_bytes(g);
    const auto normalized_count = checked_product(g.rows, g.width);
    const auto injection_count = checked_product(g.streams, g.width);
    if (normalized.size() != normalized_count || down.size() != required_down ||
        (!injection.empty() && injection.size() != injection_count))
        throw std::invalid_argument("HC oracle input extent differs");
    Result result{g, std::vector<Output>(checked_product(g.rows, g.rank)), {}};
    if (!injection.empty()) result.injection.resize(checked_product(g.rows, g.streams));
    const auto dot = [&](std::size_t row, std::size_t out, bool inject) {
        Output value;
        for (std::size_t k = 0; k < g.width; ++k) {
            const double x = normalized[row * g.width + k];
            const double w = inject ? static_cast<double>(injection[out * g.width + k]) :
                static_cast<double>(q8_coefficient(down, g.width, out, k));
            value.dot = std::fma(x, w, value.dot);
            value.sum_abs += std::abs(x * w);
        }
        value.scaled = value.dot / static_cast<double>(g.streams);
        value.postactivation = inject ? 2.0 * stable_sigmoid(value.scaled) :
            stable_silu(value.scaled);
        return value;
    };
    for (std::size_t row = 0; row < g.rows; ++row) {
        for (std::size_t out = 0; out < g.rank; ++out)
            result.down[row * g.rank + out] = dot(row, out, false);
        if (!injection.empty())
            for (std::size_t out = 0; out < g.streams; ++out)
                result.injection[row * g.streams + out] = dot(row, out, true);
    }
    return result;
}

enum class Stage { dot, scaled, postactivation };
inline double value_at(const Output& value, Stage stage) {
    switch (stage) {
    case Stage::dot: return value.dot;
    case Stage::scaled: return value.scaled;
    case Stage::postactivation: return value.postactivation;
    }
    throw std::invalid_argument("HC oracle invalid comparison stage");
}

inline std::uint64_t float_ulp_distance(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b))
        throw std::invalid_argument("HC oracle ULP requires finite floats");
    if (a == b) return 0U; // Signed zero is tracked separately, not as one ULP.
    const auto ordered = [](float x) {
        const auto word = std::bit_cast<std::uint32_t>(x);
        return (word & 0x80000000U) != 0U ? ~word : word | 0x80000000U;
    };
    const auto x = ordered(a), y = ordered(b);
    return x > y ? static_cast<std::uint64_t>(x) - y : static_cast<std::uint64_t>(y) - x;
}

struct Error {
    double reference = 0.0, observed = 0.0, sum_abs = 0.0;
    double absolute = std::numeric_limits<double>::quiet_NaN();
    double relative = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t ulp = 0;
    bool finite_pair = false, ulp_available = false;
};
struct Comparison {
    std::vector<Error> outputs;
    std::size_t finite_pairs = 0, classification_mismatches = 0;
    std::size_t reference_nan = 0, observed_nan = 0;
    std::size_t reference_inf = 0, observed_inf = 0;
    std::size_t zero_reference_nonzero_observed = 0, signed_zero_mismatches = 0;
    double max_absolute = 0.0, max_relative = 0.0, rmse = 0.0;
    std::uint64_t max_ulp = 0;
};

inline Comparison compare(std::span<const float> observed,
    std::span<const Output> expected, Stage stage = Stage::postactivation) {
    if (observed.size() != expected.size())
        throw std::invalid_argument("HC oracle comparison extent differs");
    Comparison result;
    double sum_squared = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        Error error;
        error.reference = value_at(expected[i], stage);
        error.observed = observed[i];
        error.sum_abs = expected[i].sum_abs;
        const bool rn = std::isnan(error.reference), on = std::isnan(error.observed);
        const bool ri = std::isinf(error.reference), oi = std::isinf(error.observed);
        result.reference_nan += rn; result.observed_nan += on;
        result.reference_inf += ri; result.observed_inf += oi;
        if (rn != on || ri != oi || (ri && oi &&
            std::signbit(error.reference) != std::signbit(error.observed)))
            ++result.classification_mismatches;
        error.finite_pair = std::isfinite(error.reference) && std::isfinite(error.observed);
        if (error.finite_pair) {
            ++result.finite_pairs;
            error.absolute = std::abs(error.observed - error.reference);
            sum_squared += error.absolute * error.absolute;
            if (error.absolute > result.max_absolute) result.max_absolute = error.absolute;
            if (error.reference != 0.0) {
                error.relative = error.absolute / std::abs(error.reference);
                if (error.relative > result.max_relative) result.max_relative = error.relative;
            } else if (error.observed != 0.0) {
                ++result.zero_reference_nonzero_observed;
            }
            if (error.reference == 0.0 && error.observed == 0.0 &&
                std::signbit(error.reference) != std::signbit(error.observed))
                ++result.signed_zero_mismatches;
            if (std::abs(error.reference) <= std::numeric_limits<float>::max()) {
                error.ulp_available = true;
                error.ulp = float_ulp_distance(observed[i], static_cast<float>(error.reference));
                if (error.ulp > result.max_ulp) result.max_ulp = error.ulp;
            }
        }
        result.outputs.push_back(error);
    }
    if (result.finite_pairs != 0U)
        result.rmse = std::sqrt(sum_squared / static_cast<double>(result.finite_pairs));
    return result;
}

struct PairComparison {
    Comparison serial, candidate;
    std::size_t differing_fp32_words = 0;
};
inline PairComparison compare_pair(std::span<const float> serial,
    std::span<const float> candidate, std::span<const Output> expected,
    Stage stage = Stage::postactivation) {
    PairComparison result{compare(serial, expected, stage), compare(candidate, expected, stage), 0};
    for (std::size_t i = 0; i < serial.size(); ++i)
        result.differing_fp32_words += std::bit_cast<std::uint32_t>(serial[i]) !=
            std::bit_cast<std::uint32_t>(candidate[i]);
    return result;
}

} // namespace strix::hc_parallel_projection_oracle
