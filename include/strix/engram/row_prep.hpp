#pragma once
// Engram row preparation: the shared contract (backend-neutral; the HIP kernel is src/engram/hip/row_kernel.hpp).
//
// Input: stored rows packed as the row reader writes them (include/strix/engram/row_reader.hpp: request i at
// rows[i * fmt.bytes()]), in the RowFormat of include/strix/engram/spec.hpp. The gather (which table rows, read from
// file / host / device) is the reader's; this stage turns packed rows into the projection input. Two forms:
//
// decode (DeepSeek V4.1): out[i][j] = the decode_row value (E4M3 x scale, exact), FP32. DS41 rounds to BF16 in its
//   own following launch (unchanged; enqueue_engram_rows_bf16).
// head_rms_bf16: rows are (token, head) pairs, head-minor: head = i % heads. Per row, the model's
//   torch.nn.RMSNorm(elements) with weight w[head] and eps = the dtype epsilon, in the declared arithmetic:
//     x_j = bf(decode_j)                      (exact for unscaled E4M3: every E4M3 value is a BF16 value)
//     ss  = chunked_dot(x, x)                 (core::chunked_dot_of: lane l sums j = l, l + 32, ... in order, then the
//                                              xor butterfly 16..1; one chunk for elements <= 640)
//     r   = 1 / declared_sqrt(ss / elements + eps)
//     out[i][j] = bf(w[head][j] * bf(x_j * r))
//   A binding's CPU oracle runs these statements in the same order. Token t's concatenated heads out[t * heads .. + heads)
//   are the heads * width projection input.
// Rows are independent; NaN codes propagate.
#include "strix/core/chunked_dot.hpp"
#include "strix/numeric/declared.hpp"
#include "strix/weight/formats/scalar_float.hpp"
#include "strix/engram/spec.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::engram {

enum class RowPrepForm : std::uint8_t { decode, head_rms_bf16 };

// Device launch view (pointers are device memory; the same fields drive the host reference below).
struct RowPrepView {
    RowPrepForm form = RowPrepForm::decode;
    RowFormat fmt;
    const std::uint8_t* rows = nullptr;  // [row_count][fmt.bytes()]
    std::uint64_t row_count = 0;
    float* out = nullptr;                // [row_count][fmt.elements]
    // head_rms_bf16 only
    const float* norm_weight = nullptr;  // [heads][fmt.elements]
    std::uint32_t heads = 0;
    float eps = 0.0F;
};

// Largest row of the head_rms_bf16 form: one lane-dot chunk (core::dot_chunk).
inline constexpr std::uint32_t row_prep_max_elements = 640U;

// nullptr when the view is supported, else the reason (pointers are not inspected).
constexpr const char* row_prep_invalid(const RowPrepView& v) {
    if (v.fmt.elements == 0U) return "Engram row prep: elements == 0";
    if (v.fmt.scale == RowScale::e8m0_per32 && v.fmt.elements % 32U != 0U) return "Engram row prep: e8m0_per32 needs elements % 32 == 0";
    if (v.form == RowPrepForm::head_rms_bf16) {
        if (v.fmt.elements > row_prep_max_elements) return "Engram row prep: head_rms_bf16 elements > 640";
        if (v.heads == 0U) return "Engram row prep: head_rms_bf16 needs heads > 0";
        if (v.row_count % v.heads != 0U) return "Engram row prep: row_count is not a whole number of tokens";
    }
    return nullptr;
}

// One stored value j of a row (the decode_row arithmetic of spec.cpp).
inline float row_value(const RowFormat& fmt, const std::uint8_t* row, std::uint32_t j) {
    const float v = weight::scalar_float::e4m3(row[j]);
    if (fmt.scale == RowScale::e8m0_per32) return v * weight::scalar_float::e8m0(row[fmt.elements + j / 32U]);
    if (fmt.scale == RowScale::f32_per_row) {
        float s = 0.0F;
        std::memcpy(&s, row + fmt.elements, sizeof(float));
        return v * s;
    }
    return v;
}

// Host reference of one row of the head_rms_bf16 form (x: the decoded row; w: this head's weight).
inline void head_rms_row_reference(const float* x, std::uint32_t n, const float* w, float eps, float* out) {
    std::vector<float> xb(n);
    for (std::uint32_t j = 0; j < n; ++j) xb[j] = numeric::round_to_bf16(x[j]);
    const float ss = core::chunked_dot_of([&](std::size_t i) { return xb[i]; }, [&](std::size_t i) { return xb[i]; }, n);
    const float r = 1.0F / numeric::declared_sqrt(ss / static_cast<float>(n) + eps);
    for (std::uint32_t j = 0; j < n; ++j) out[j] = numeric::round_to_bf16(w[j] * numeric::round_to_bf16(xb[j] * r));
}

// Host reference of the whole view (host pointers). Throws std::invalid_argument on an unsupported view.
inline void row_prep_reference(const RowPrepView& v) {
    if (const char* why = row_prep_invalid(v)) throw std::invalid_argument(why);
    const std::uint32_t n = v.fmt.elements, bytes = v.fmt.bytes();
    std::vector<float> x(n);
    for (std::uint64_t i = 0; i < v.row_count; ++i) {
        const std::uint8_t* row = v.rows + static_cast<std::size_t>(i) * bytes;
        float* o = v.out + static_cast<std::size_t>(i) * n;
        if (v.form == RowPrepForm::decode) {
            for (std::uint32_t j = 0; j < n; ++j) o[j] = row_value(v.fmt, row, j);
            continue;
        }
        for (std::uint32_t j = 0; j < n; ++j) x[j] = row_value(v.fmt, row, j);
        head_rms_row_reference(x.data(), n, v.norm_weight + static_cast<std::size_t>(i % v.heads) * n, v.eps, o);
    }
}

}  // namespace strix::engram
