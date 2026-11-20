#pragma once

// Engram (hashed n-gram memory), shared host contract. Vendor-free: a family adapter only builds a Spec
// (DS41 from its admitted EngramContract, other packs from their metadata); hashing, row reading and row decoding
// are the shared code in strix/engram/{hasher,row_reader}.hpp.
//
// Hash of position t at Engram layer e (one definition for every family):
//   c_s = compressed id of token t - s (s = 0 .. max_ngram_size - 1); the compressed pad when t - s is before the
//         sequence start, or when any of t .. t - s is a dead position (a row whose embedding was provided);
//   p_s = c_s * multipliers[e][s] (64-bit), r_1 = p_0 ^ p_1, r_s = r_(s-1) ^ p_s;
//   row of the (s + 1)-gram, head h = r_s % moduli[e][(s - 1) * heads_per_order + h] + offset of that column,
//   offsets being the running sum of the layer's moduli in column order (n-gram size major, head minor).
// Rows are relative to the layer's table; layer_row_base() places a layer inside one flat table.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::engram {

// Scale carried by a stored row. e8m0_per32: one E8M0 byte per 32 elements after the codes (DS41, 264-byte rows);
// none: raw E4M3 codes (packs storing bare codes: the per-row amax/448 scale of the reference cancels in the per-head RMSNorm up to
// its eps term); f32_per_row: one little-endian float after the codes.
enum class RowScale : std::uint8_t { none, e8m0_per32, f32_per_row };

struct RowFormat {
    std::uint32_t elements = 0;  // E4M3 codes per row (= head_dim)
    RowScale scale = RowScale::none;
    [[nodiscard]] constexpr std::uint32_t bytes() const noexcept {
        return elements + (scale == RowScale::e8m0_per32 ? elements / 32U : scale == RowScale::f32_per_row ? 4U : 0U);
    }
};

// Where the table lives. Placement (byte budget per node) decides; the reader binds file_direct / file_cached.
enum class Residency : std::uint8_t { file_direct, file_cached, host_pinned, device };

// Token ids at or past `compress_valid`: reject (throw, DS41 admits only mapped ids) or read the compressed pad.
enum class Unmapped : std::uint8_t { reject, pad };

struct Spec {
    std::vector<std::uint32_t> layer_ids;    // model layers (0-based) that run Engram
    std::uint32_t max_ngram_size = 0;        // n-gram orders 2 .. max_ngram_size
    std::uint32_t heads_per_order = 0;
    std::vector<std::uint64_t> moduli;       // [layer][order - 2][head]
    std::vector<std::uint64_t> multipliers;  // [layer][lookback 0 .. max_ngram_size - 1], odd
    std::vector<std::uint32_t> compress;     // token id -> compressed id
    std::uint32_t compress_valid = 0;        // ids >= this are unmapped (0: compress.size())
    Unmapped unmapped = Unmapped::reject;
    std::uint32_t pad_token = 0;             // token id whose compressed id pads (index into compress)
    RowFormat row;
    Residency residency = Residency::file_direct;

    [[nodiscard]] std::uint32_t layers() const noexcept { return static_cast<std::uint32_t>(layer_ids.size()); }
    [[nodiscard]] std::uint32_t columns() const noexcept { return (max_ngram_size - 1U) * heads_per_order; }
    [[nodiscard]] std::uint32_t valid_ids() const noexcept {
        return compress_valid == 0U ? static_cast<std::uint32_t>(compress.size()) : compress_valid;
    }
};

// Throws std::invalid_argument naming the first inconsistent field.
void validate(const Spec& spec);
// [layer][column] start row of each column inside its layer's table.
[[nodiscard]] std::vector<std::uint64_t> column_offsets(const Spec& spec);
// Rows of layer e's table (sum of its moduli).
[[nodiscard]] std::uint64_t layer_rows(const Spec& spec, std::uint32_t e);
// First row of layer e when all layers are one flat table in layer order (one per-layer token-embedding tensor).
[[nodiscard]] std::uint64_t layer_row_base(const Spec& spec, std::uint32_t e);

// True when every product compressed id x multiplier fits in a signed int64 (ids of valid tokens and the pad). Then
// the unsigned 64-bit hash equals a reference computed in int64 with a floor modulo (torch / numpy `%`).
[[nodiscard]] bool products_fit_int64(const Spec& spec);

// Host reference decode of one stored row to FP32 (exact: E4M3 x scale). bytes.size() == fmt.bytes(),
// out.size() == fmt.elements.
void decode_row(const RowFormat& fmt, std::span<const std::uint8_t> bytes, std::span<float> out);

} // namespace strix::engram
