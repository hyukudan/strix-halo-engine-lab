#pragma once
// Column map of the shared tall core's pair launches (f16_dense_tall.hpp pair store mode): one CTA of TC column tiles
// holds TC / 2 OUTPUT tiles, each computed twice (first = e.g. the gate projection, second = the up projection of the
// same outputs). Launch tile i of a TC-tile launch: CTA column block i / TC, slot i % TC; slots [0, TC / 2) are the first
// member, slots [TC / 2, TC) the second member of the same output tiles. Every output tile of a launch with
// out_tiles % (TC / 2) == 0 is produced exactly once per member and the store sees both members of one output in the
// same lane slot (tile t and t + TC / 2).
// x16 tile layout (f16_dense_tall.hpp x16 form, the A operand of the next GEMM): element (row r, input k) at
// (k / 64 * rows + r) * 64 + k % 64.
// Pure index arithmetic, host and device (tests/dense_pair_map_host_tests.cpp). Home: shared (backend core, any
// backend's pair launch can use it); nothing here names a model.
#include <cstddef>

#if defined(__HIPCC__)
#define STRIX_PAIR_HD __host__ __device__
#else
#define STRIX_PAIR_HD
#endif

namespace strix::core::dense_pair_map {

// Output tile of launch tile i (TC tiles per CTA, TC even).
template <unsigned TC> STRIX_PAIR_HD constexpr unsigned out_tile(unsigned i) {
    static_assert(TC % 2U == 0U, "pair launches need an even tile count");
    return (i / TC) * (TC / 2U) + (i % TC) % (TC / 2U);
}
// true: launch tile i is the second member (e.g. up) of its output tile.
template <unsigned TC> STRIX_PAIR_HD constexpr bool second(unsigned i) { return i % TC >= TC / 2U; }
// CTA column blocks of a pair launch over out_tiles output tiles (callers require out_tiles % (TC / 2) == 0).
template <unsigned TC> STRIX_PAIR_HD constexpr unsigned column_blocks(unsigned out_tiles) { return out_tiles / (TC / 2U); }
// Element offset of (row r, input k) in x16 tiles of `rows` rows.
STRIX_PAIR_HD constexpr std::size_t x16_index(unsigned rows, unsigned r, unsigned k) {
    return (static_cast<std::size_t>(k / 64U) * rows + r) * 64U + k % 64U;
}

}  // namespace strix::core::dense_pair_map
