#include "strix/engram/spec.hpp"

#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace strix::engram {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

} // namespace

void validate(const Spec& s) {
    const std::size_t layers = s.layer_ids.size(), cols = s.columns();
    require(s.max_ngram_size >= 2U && s.heads_per_order >= 1U && layers != 0U, "Engram spec geometry");
    require(s.moduli.size() == layers * cols, "Engram spec: moduli are [layer][order][head]");
    require(s.multipliers.size() == layers * s.max_ngram_size, "Engram spec: multipliers are [layer][lookback]");
    for (const auto m : s.moduli) require(m != 0U, "Engram spec: zero modulus");
    require(!s.compress.empty() && s.valid_ids() <= s.compress.size(), "Engram spec: compress map");
    require(s.pad_token < s.valid_ids(), "Engram spec: pad token outside the compress map");
    require(s.row.elements != 0U && (s.row.scale != RowScale::e8m0_per32 || s.row.elements % 32U == 0U),
            "Engram spec: row format");
}

std::vector<std::uint64_t> column_offsets(const Spec& s) {
    const std::size_t layers = s.layer_ids.size(), cols = s.columns();
    std::vector<std::uint64_t> out(layers * cols);
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t sum = 0;
        for (std::size_t k = 0; k < cols; ++k) {
            out[e * cols + k] = sum;
            sum += s.moduli[e * cols + k];
        }
    }
    return out;
}

std::uint64_t layer_rows(const Spec& s, std::uint32_t e) {
    const std::size_t cols = s.columns();
    require(e < s.layers(), "Engram layer index");
    std::uint64_t sum = 0;
    for (std::size_t k = 0; k < cols; ++k) sum += s.moduli[e * cols + k];
    return sum;
}

std::uint64_t layer_row_base(const Spec& s, std::uint32_t e) {
    std::uint64_t base = 0;
    for (std::uint32_t i = 0; i < e; ++i) base += layer_rows(s, i);
    return base;
}

bool products_fit_int64(const Spec& s) {
    std::uint64_t max_id = s.compress[s.pad_token], max_mult = 0;
    for (std::uint32_t t = 0; t < s.valid_ids(); ++t) max_id = std::max<std::uint64_t>(max_id, s.compress[t]);
    for (const auto m : s.multipliers) max_mult = std::max(max_mult, m);
    constexpr std::uint64_t int64_max = 0x7FFF'FFFF'FFFF'FFFFULL;
    return max_mult == 0U || max_id <= int64_max / max_mult;
}

void decode_row(const RowFormat& fmt, std::span<const std::uint8_t> bytes, std::span<float> out) {
    require(bytes.size() == fmt.bytes() && out.size() == fmt.elements, "Engram row decode size");
    const std::uint8_t* scales = bytes.data() + fmt.elements;
    float row_scale = 1.0F;
    if (fmt.scale == RowScale::f32_per_row) std::memcpy(&row_scale, scales, sizeof(float));
    for (std::uint32_t j = 0; j < fmt.elements; ++j) {
        const float v = weight::scalar_float::e4m3(bytes[j]);
        out[j] = fmt.scale == RowScale::e8m0_per32 ? v * weight::scalar_float::e8m0(scales[j / 32U])
               : fmt.scale == RowScale::f32_per_row ? v * row_scale
                                                    : v;
    }
}

} // namespace strix::engram
