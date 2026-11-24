#pragma once
// Host encoder of NVFP4 rows (E2M1 codes, one E4M3 scale per 16 columns, one FP32 outer scale per matrix; decode:
// microscaled.hpp nvfp4_value = (e2m1 x scale) x outer) and the MXFP8 -> NVFP4 transcode of a matrix: the narrow pack of a
// microscaled bank (4.5 instead of 8.25 bits per weight). Bit-changing: a packed bank is a FAST or TURBO weight lever, decided by
// gate v2 on the model; the per-matrix error returned here is the load-time evidence (and the per-bank fallback input).
//
// Block scale rule (mse@2): the candidates are the E4M3 rounding of amax_block / (6 outer) and its two neighbours on each
// side (codes clamped to [0, 0x7e]); the candidate with the smallest squared error of the decoded block wins, ties to the
// lower code. Codes: nearest E2M1 of value / (scale outer), ties to even, saturating at +-6.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/formats/microscaled_matrix.hpp"
#include "strix/weight/formats/scalar_float.hpp"
#include "strix/weight/source.hpp"

namespace strix::weight::microscaled {

namespace detail {

// Compare in the input domain. The scaled midpoints are exact in FP64 for these formats
// and a FP32 outer scale, so midpoint ties do not depend on the caller's rounding mode.
template <auto Decode, int Last>
inline std::uint8_t nearest_scaled_code(double magnitude, double multiplier) {
    int lo = 0, hi = Last;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        const double midpoint = (double{Decode(static_cast<std::uint8_t>(mid))} +
                                 double{Decode(static_cast<std::uint8_t>(mid + 1))}) * 0.5 * multiplier;
        if (magnitude > midpoint || (magnitude == midpoint && (mid & 1) != 0)) lo = mid + 1;
        else hi = mid;
    }
    return static_cast<std::uint8_t>(lo);
}

}  // namespace detail

// Outer scale that lets the largest block scale (448) times the largest code (6) reach amax.
inline float nvfp4_outer_scale(float amax) {
    if (!(amax > 0.0F) || !std::isfinite(amax)) return 1.0F;
    return std::max(amax / (6.0F * 448.0F), std::numeric_limits<float>::denorm_min());
}

// Squared error of one 16-column block encoded with scale code `scale`; writes the 8 code bytes when `codes` is set.
inline double nvfp4_block_error(const float* x, float outer, std::uint8_t scale, std::uint8_t* codes) {
    const float block_scale = scalar_float::e4m3(scale);
    const double step = double{block_scale} * double{outer};
    double error = 0.0;
    for (std::size_t i = 0; i < kNvfp4BlockColumns; ++i) {
        std::uint8_t code = 0;
        if (step > 0.0) {
            const auto magnitude = detail::nearest_scaled_code<scalar_float::e2m1, 0x07>(std::fabs(double{x[i]}), step);
            code = static_cast<std::uint8_t>(magnitude | (std::signbit(x[i]) ? 8U : 0U));
        }
        const double d = double{nvfp4_value(code, block_scale, outer)} - double{x[i]};
        error += d * d;
        if (codes) {
            auto& byte = codes[i / 2U];
            byte = i % 2U == 0U ? code : static_cast<std::uint8_t>(byte | (code << 4U));
        }
    }
    return error;
}

// Encode one row of finite values (size a multiple of 16) with the matrix outer scale.
inline void encode_nvfp4_row(std::span<const float> x, float outer, std::span<std::uint8_t> codes,
                             std::span<std::uint8_t> scales) {
    if (x.empty() || x.size() % kNvfp4BlockColumns != 0U || codes.size() != x.size() / 2U ||
        scales.size() != x.size() / kNvfp4BlockColumns || !(outer > 0.0F) || !std::isfinite(outer))
        throw std::invalid_argument("NVFP4 encode: inconsistent row");
    for (std::size_t b = 0; b < scales.size(); ++b) {
        const float* block = x.data() + b * kNvfp4BlockColumns;
        float amax = 0.0F;
        for (std::size_t i = 0; i < kNvfp4BlockColumns; ++i) {
            if (!std::isfinite(block[i])) throw std::invalid_argument("NVFP4 encode: non-finite value");
            amax = std::fmax(amax, std::fabs(block[i]));
        }
        const int center = detail::nearest_scaled_code<scalar_float::e4m3, 0x7e>(amax, 6.0 * double{outer});
        std::uint8_t best = 0U;
        double best_error = -1.0;
        for (int s = center - 2; s <= center + 2; ++s) {
            if (s < 0 || s > 0x7E) continue;
            const double error = nvfp4_block_error(block, outer, static_cast<std::uint8_t>(s), nullptr);
            if (best_error < 0.0 || error < best_error) {
                best = static_cast<std::uint8_t>(s);
                best_error = error;
            }
        }
        scales[b] = best;
        nvfp4_block_error(block, outer, best, codes.data() + b * kNvfp4BlockColumns / 2U);
    }
}

// Error of a packed matrix against its source: sum of squared errors, source energy, largest absolute error.
struct PackError {
    double sse{}, energy{};
    float max_abs{};
    double relative() const { return energy > 0.0 ? std::sqrt(sse / energy) : 0.0; }
};

struct Nvfp4Matrix {
    std::vector<std::uint8_t> codes, scales;  // rows x columns / 2, rows x columns / 16, the source row order
    float outer{1.0F};
    PackError error;
};

// MXFP8 matrix (rows x columns E4M3 codes, rows x columns / 32 E8M0 scales) -> NVFP4 with one outer scale (from the
// matrix amax). Throws on malformed input (decode_mxfp8_row).
inline Nvfp4Matrix transcode_mxfp8_to_nvfp4(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
                                           std::size_t rows, std::size_t columns) {
    if (rows == 0U || columns == 0U || columns % kMxfp8BlockColumns != 0U || codes.size() / columns != rows ||
        codes.size() % columns != 0U || scales.size() != rows * (columns / kMxfp8BlockColumns))
        throw std::invalid_argument("MXFP8 transcode: inconsistent matrix");
    std::vector<float> values(rows * columns);
    float amax = 0.0F;
    for (std::size_t r = 0; r < rows; ++r) {
        std::span<float> row(values.data() + r * columns, columns);
        decode_mxfp8_row(codes.subspan(r * columns, columns),
                         scales.subspan(r * (columns / kMxfp8BlockColumns), columns / kMxfp8BlockColumns), row);
        for (const float v : row) amax = std::fmax(amax, std::fabs(v));
    }
    Nvfp4Matrix out;
    out.outer = nvfp4_outer_scale(amax);
    out.codes.resize(rows * columns / 2U);
    out.scales.resize(rows * (columns / kNvfp4BlockColumns));
    for (std::size_t r = 0; r < rows; ++r) {
        const std::span<const float> row(values.data() + r * columns, columns);
        const std::span<std::uint8_t> row_codes(out.codes.data() + r * columns / 2U, columns / 2U);
        const std::span<std::uint8_t> row_scales(out.scales.data() + r * (columns / kNvfp4BlockColumns),
                                                 columns / kNvfp4BlockColumns);
        encode_nvfp4_row(row, out.outer, row_codes, row_scales);
        for (std::size_t c = 0; c < columns; ++c) {
            const auto byte = row_codes[c / 2U];
            const auto code = static_cast<std::uint8_t>(c % 2U == 0U ? byte & 0x0FU : byte >> 4U);
            const float decoded = nvfp4_value(code, scalar_float::e4m3(row_scales[c / kNvfp4BlockColumns]), out.outer);
            const double d = double{decoded} - double{row[c]};
            out.error.sse += d * d;
            out.error.energy += double{row[c]} * double{row[c]};
            out.error.max_abs = std::fmax(out.error.max_abs, static_cast<float>(std::fabs(d)));
        }
    }
    return out;
}

// A microscaled expert slab stored as NVFP4: [expert][row][column / 2] codes and [expert][row][column / 16] E4M3 scales in
// the source row order, one outer scale per expert (from that expert's amax, nvfp4_outer_scale). This is the narrow image
// of a MXFP8 bank (4.5 instead of 8.25 bits per weight); a device copy of `codes` / `scales` binds exactly like a
// checkpoint NVFP4 slab, so every NVFP4 consumer reads it unchanged.
struct Nvfp4Slab {
    std::size_t experts{}, rows{}, columns{};
    RowOrder row_order{};
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> outer;
    PackError error;  // summed over the experts in expert order; max_abs over the slab

    [[nodiscard]] std::size_t code_stride() const { return rows * columns / 2U; }
    [[nodiscard]] std::size_t scale_stride() const { return rows * (columns / kNvfp4BlockColumns); }

    // One PackedMatrix per expert over device (or host) copies of `codes` and `scales`, byte for byte.
    [[nodiscard]] Status bind_storage(std::span<const std::uint8_t> resident_codes,
                                      std::span<const std::uint8_t> resident_scales, std::span<PackedMatrix> out) const {
        std::size_t code_bytes = 0, scale_bytes = 0;
        if (const auto status = detail::slab_strides(MatrixFormat::nvfp4, row_order, experts, rows, columns,
                                                     code_bytes, scale_bytes); !status) return status;
        if (codes.size() != experts * code_bytes || scales.size() != experts * scale_bytes ||
            outer.size() != experts)
            return {Error::malformed, "NVFP4 slab is not a complete transcode"};
        if (resident_codes.size() != codes.size() || resident_scales.size() != scales.size() || out.size() != experts)
            return {Error::unsupported_shape, "resident NVFP4 slab dimensions disagree with the transcode"};
        for (std::size_t expert = 0; expert < experts; ++expert)
            out[expert] = {resident_codes.data() + expert * code_bytes, resident_scales.data() + expert * scale_bytes,
                           rows, columns, outer[expert], row_order};
        return {Error::ok, {}};
    }
};

// Reads each expert of a MXFP8 slab from its source and transcodes it (transcode_mxfp8_to_nvfp4) on up to `threads`
// host threads (0: hardware concurrency). Each expert is encoded alone, so the bytes do not depend on `threads`. Refuses a
// slab that is not MXFP8 (a NVFP4 slab is already narrow) and any malformed or non-finite source.
[[nodiscard]] inline Status transcode_slab_to_nvfp4(const ExpertSlabView& slab, Nvfp4Slab& out, unsigned threads = 0U) try {
    if (slab.format != MatrixFormat::mxfp8) return {Error::unknown_format, "NVFP4 transcode reads a MXFP8 slab"};
    // matrix() validates the geometry and both complete parent slices without reading their payloads.
    MatrixView first;
    if (auto status = slab.matrix(0, first); !status) return status;
    Nvfp4Slab result;
    result.experts = slab.experts;
    result.rows = slab.rows;
    result.columns = slab.columns;
    result.row_order = slab.row_order;
    result.codes.resize(slab.experts * result.code_stride());
    result.scales.resize(slab.experts * result.scale_stride());
    result.outer.resize(slab.experts);
    struct ExpertResult {
        PackError error;
        Status status{Error::ok, {}};
        std::exception_ptr failure;
    };
    std::vector<ExpertResult> experts(slab.experts);

    const auto encode = [&](std::size_t expert, PackError& error) -> Status {
        MatrixView view;
        if (auto status = slab.matrix(expert, view); !status) return status;
        std::vector<std::uint8_t> codes(view.codes.range.bytes), scales(view.block_scales.range.bytes);
        if (auto status = read_slice(view.codes, std::as_writable_bytes(std::span(codes))); !status) return status;
        if (auto status = read_slice(view.block_scales, std::as_writable_bytes(std::span(scales))); !status) return status;
        auto matrix = transcode_mxfp8_to_nvfp4(codes, scales, slab.rows, slab.columns);
        std::ranges::copy(matrix.codes, result.codes.begin() + static_cast<std::ptrdiff_t>(expert * result.code_stride()));
        std::ranges::copy(matrix.scales, result.scales.begin() + static_cast<std::ptrdiff_t>(expert * result.scale_stride()));
        result.outer[expert] = matrix.outer;
        error = matrix.error;
        return {Error::ok, {}};
    };
    if (threads == 0U) threads = std::max(1U, std::thread::hardware_concurrency());
    threads = static_cast<unsigned>(std::min<std::size_t>(threads, slab.experts));
    std::atomic<std::size_t> next{0};
    // Join before inspecting results, including when launching a later worker throws.
    try {
        std::vector<std::jthread> pool;
        pool.reserve(threads);
        for (unsigned t = 0; t < threads; ++t)
            pool.emplace_back([&] {
                for (std::size_t expert = next++; expert < slab.experts; expert = next++) {
                    auto& entry = experts[expert];
                    try {
                        entry.status = encode(expert, entry.error);
                    } catch (...) {
                        entry.failure = std::current_exception();
                    }
                }
            });
    } catch (const std::system_error& e) {
        return {Error::memory_budget, e.what()};
    }

    for (auto& entry : experts) {
        if (entry.failure) std::rethrow_exception(entry.failure);
        if (!entry.status) return std::move(entry.status);
        result.error.sse += entry.error.sse;
        result.error.energy += entry.error.energy;
        result.error.max_abs = std::fmax(result.error.max_abs, entry.error.max_abs);
    }
    out = std::move(result);
    return {Error::ok, {}};
} catch (const std::bad_alloc&) {
    return {Error::memory_budget, {}};
} catch (const std::length_error& e) {
    return {Error::memory_budget, e.what()};
} catch (const std::exception& e) {
    return {Error::malformed, std::string("MXFP8 slab transcode: ") + e.what()};
} catch (...) {
    return {Error::malformed, "MXFP8 slab transcode: unknown exception"};
}

}  // namespace strix::weight::microscaled
