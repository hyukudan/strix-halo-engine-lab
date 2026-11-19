#include "strix/hy4/engram_spec.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix::hy4 {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

} // namespace

engram::Spec engram_spec(const EngramPackMeta& m) {
    engram::Spec s;
    s.layer_ids = m.layers;
    s.max_ngram_size = m.ngram_size;
    s.heads_per_order = m.heads_per_ngram;
    s.moduli = m.head_vocab_sizes;
    s.multipliers.reserve(m.layer_multipliers.size());
    for (const auto v : m.layer_multipliers) {
        require(v > 0 && (v & 1) == 1, "hy4 engram: multipliers are positive odd int64");
        s.multipliers.push_back(static_cast<std::uint64_t>(v));
    }
    s.compress.reserve(m.token_map.size());
    for (const auto v : m.token_map) {
        require(v >= 0, "hy4 engram: negative token map entry");
        s.compress.push_back(static_cast<std::uint32_t>(v));
    }
    s.compress_valid = std::min(static_cast<std::uint32_t>(s.compress.size()), m.lookup_table_size);
    s.unmapped = engram::Unmapped::pad;
    s.pad_token = m.pad_token_id;
    // GGUF F8_E4M3 table without a scale: the reference's per-row amax/448 scale cancels in the per-head RMSNorm.
    s.row = {m.row_elements, engram::RowScale::none};
    s.residency = engram::Residency::file_direct;  // placement may choose host_pinned / device per node
    engram::validate(s);
    // The reference hashes in int64 with a floor modulo; the shared hash is unsigned. Equal when no product wraps.
    require(engram::products_fit_int64(s), "hy4 engram: an id x multiplier product overflows int64");
    if (!m.head_offsets.empty()) {
        require(m.head_offsets.size() == s.moduli.size(), "hy4 engram: head offsets per head");
        const auto cols = s.columns();
        const auto local = engram::column_offsets(s);
        for (std::uint32_t e = 0; e < s.layers(); ++e) {
            const auto base = engram::layer_row_base(s, e);
            for (std::uint32_t k = 0; k < cols; ++k)
                require(m.head_offsets[e * cols + k] == base + local[e * cols + k], "hy4 engram: head offsets");
        }
    }
    return s;
}

} // namespace strix::hy4
