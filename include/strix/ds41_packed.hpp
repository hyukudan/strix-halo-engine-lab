#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// Packed forms of Q8_0-embeddable weights (requantization candidates). A tensor whose Q8_0 codes take at most 16
// values packs to CB4 (4-bit indices into a 16-entry int8 table); codes within [-16, 15] / [-32, 31] pack to U5 / U6
// (uniform, code = v - 16 / v - 32). Every packed group decodes to its Q8_0 block {d, codes} exactly, so a kernel
// that computes the Q8_0 product from the decoded codes gives the bits of the Q8_0 form.
//
// Group payload: 16 bytes of low nibbles (byte j = weight j | weight j + 16 << 4), the high bits (U5: u32, bit j =
// bit 4 of weight j; U6: u64, bits 2j..2j+1 = bits 4..5 of weight j; CB4: none) and the f16 scale d.
// Layouts of one weight row of G = width / 32 groups:
//   interleaved  G blocks of {d, nibbles, high} (18 / 22 / 26 B), the form of the current site kernels;
//   planar       [nibbles: G x 16 B][high: G x 0 / 4 / 8 B][scales: G x 2 B], every plane 16-B aligned when G % 4 == 0
//                and the row starts 16-B aligned (the form for one-row-per-wave decode kernels).
// Both layouts have the same row bytes, so offsets of whole rows are layout-independent.
namespace strix::ds41 {

enum class PackedKind : std::uint32_t { cb4 = 4, u5 = 5, u6 = 6 };
enum class PackedLayout : std::uint32_t { interleaved = 0, planar = 1 };

[[nodiscard]] constexpr std::size_t packed_high_bytes(PackedKind k) noexcept {
    return k == PackedKind::u5 ? 4U : k == PackedKind::u6 ? 8U : 0U;
}
[[nodiscard]] constexpr std::size_t packed_group_bytes(PackedKind k) noexcept { return 18U + packed_high_bytes(k); }
[[nodiscard]] constexpr std::size_t packed_row_bytes(PackedKind k, std::size_t width) noexcept {
    return width / 32U * packed_group_bytes(k);
}

struct PackedChoice {
    PackedKind kind = PackedKind::cb4;
    std::array<std::int8_t, 16> table{};  // CB4 only: entry i = the code of index i (sorted, padded with the last)
};

// The packing the codes of a Q8_0 tensor allow (CB4 first, then U5, then U6), or none.
[[nodiscard]] std::optional<PackedChoice> choose_packing(std::span<const std::uint8_t> q8);

// Q8_0 rows (width weights each, row-major 34-byte blocks) to packed rows in the given layout; multi-threaded.
// Throws std::invalid_argument if a code is not representable in the choice.
[[nodiscard]] std::vector<std::uint8_t> pack_rows(std::span<const std::uint8_t> q8, std::size_t width, const PackedChoice& choice,
                                                  PackedLayout layout);

// Group g of one packed row back to its 34-byte Q8_0 block: the reference of every packed decode kernel.
void unpack_group(const std::uint8_t* row, std::size_t width, std::size_t g, const PackedChoice& choice, PackedLayout layout,
                  std::uint8_t* q8_block);

// F16 weights that are MX FP8 underneath (every 32-value block of a row is E4M3 x 2^k exactly, e.g. engram_kv and
// indexer.attn_q_b of the 1017 model) as F8 rows: [width E4M3 codes][width / 32 E8M0 exponents] per row, 8.25 bpw.
// The decoded weight e4m3 * 2^(e - 127) is exactly the F16 value (signed zeros included), so an F16 kernel that
// widens these instead gives the same bits. Empty if a block is not exact.
[[nodiscard]] std::optional<std::vector<std::uint8_t>> pack_f16_as_f8(std::span<const std::uint16_t> f16, std::size_t width);
[[nodiscard]] constexpr std::size_t f8_row_bytes(std::size_t width) noexcept { return width + width / 32U; }
// Weight i of one F8 row (the reference of the F8 kernels).
[[nodiscard]] float unpack_f8(const std::uint8_t* row, std::size_t width, std::size_t i);

}  // namespace strix::ds41
