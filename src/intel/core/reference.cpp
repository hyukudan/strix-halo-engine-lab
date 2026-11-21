#include "reference.hpp"

#include <limits>
#include <stdexcept>

namespace strix::intel {

std::string_view reference_refusal(Status status, const Capabilities& caps) noexcept {
    switch (status) {
        case Status::not_built: return "Intel reference requires a SYCL build";
        case Status::unsupported: return "no Level Zero GPU at ordinal 0";
        case Status::ok: break;
        default: return "failed to open the Level Zero GPU queue";
    }
    if (caps.arch != "xe2-bmg" || caps.integrated) return "Intel reference requires a discrete Xe2 Arc GPU";
    if (!caps.device_usm) return "Intel reference requires device USM allocations";
    return {};
}

namespace {
std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Intel reference layout overflow");
    return a * b;
}
}

std::size_t q8_matrix_bytes(core::Q8Rows matrix) {
    if (!matrix.data || !matrix.rows || !matrix.width || matrix.width % 32U ||
        matrix.row_bytes != weight::Q8Storage::row_bytes(matrix.width))
        throw std::invalid_argument("Intel reference requires packed Q8_0 matrix rows");
    return checked_product(matrix.rows, matrix.row_bytes);
}

std::size_t attention_score_elements(const core::DenseAttentionRow& row, std::size_t output_elements) {
    if (!row.width || !row.positions || !row.kv_heads || !row.heads || row.heads % row.kv_heads)
        throw std::invalid_argument("invalid Intel dense attention geometry");
    const auto queries = checked_product(row.heads, row.width);
    const auto cache = checked_product(row.positions, checked_product(row.kv_heads, row.width));
    (void)checked_product(queries, sizeof(float));
    (void)checked_product(cache, sizeof(float));
    if (row.query.size() != queries || output_elements != queries ||
        row.keys.size() != cache || row.values.size() != cache)
        throw std::invalid_argument("invalid Intel dense attention extents");
    const auto scores = checked_product(row.heads, row.positions);
    (void)checked_product(scores, sizeof(float));
    return scores;
}

} // namespace strix::intel
