// core/dense_pair_map.hpp: column map of the shared tall core's pair launches (GLU pair sweep of f16_dense_iu4.hpp).
// Invariants: (1) over every launch tile of a TC = 16 pair launch with out_tiles % 8 == 0, each (output tile, member) is
// produced exactly once; (2) the store's tile t and t + TC / 2 of one CTA are the two members of the SAME output tile
// (the core passes acc[t] and acc[t + TC / 2] to put_pair); (3) the pair launch stages exactly the column of the
// incumbent single-member launch (output tile o of member m <-> launch tile o of that member's own x16 launch, i.e. the
// same weight column and so the same WMMA chain); (4) x16_index is the x16 tile layout of f16_dense_tall (the map2
// producer's address (c / 8 * rows + r) * 64 + (c % 8) * 8 + j for k = 8 c + j), a bijection onto [0, rows * in).
#include "core/dense_pair_map.hpp"

#include <cstdio>
#include <vector>

namespace pm = strix::core::dense_pair_map;

static int failures = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s at %d\n", #c, __LINE__); ++failures; } } while (0)

template <unsigned TC> static void check_cover(unsigned out_tiles) {
    const unsigned blocks = pm::column_blocks<TC>(out_tiles);
    EXPECT(blocks * (TC / 2U) == out_tiles);
    std::vector<int> seen(std::size_t(out_tiles) * 2U, 0);
    for (unsigned i = 0; i < blocks * TC; ++i) {
        const unsigned o = pm::out_tile<TC>(i);
        EXPECT(o < out_tiles);
        if (o < out_tiles) ++seen[std::size_t(o) * 2U + (pm::second<TC>(i) ? 1U : 0U)];
    }
    for (int s : seen) EXPECT(s == 1);
    for (unsigned b = 0; b < blocks; ++b)
        for (unsigned t = 0; t < TC / 2U; ++t) {
            const unsigned a = b * TC + t, c = a + TC / 2U;
            EXPECT(!pm::second<TC>(a) && pm::second<TC>(c));
            EXPECT(pm::out_tile<TC>(a) == pm::out_tile<TC>(c));
            EXPECT(pm::out_tile<TC>(a) == b * (TC / 2U) + t);  // the store's output tile of slot t
        }
}

int main() {
    check_cover<16>(8);
    check_cover<16>(1088);  // 17408 outputs (a 27B-class FFN width)
    check_cover<16>(136);
    check_cover<4>(10);
    check_cover<2>(7);
    static_assert(pm::out_tile<16>(0) == 0 && pm::out_tile<16>(8) == 0 && pm::out_tile<16>(15) == 7 && pm::out_tile<16>(16) == 8);
    static_assert(!pm::second<16>(7) && pm::second<16>(8) && !pm::second<16>(16));
    // (4) x16 layout = the map2 producer's address, and a bijection
    for (unsigned rows : {1U, 130U, 512U}) {
        const unsigned in = 384U;
        std::vector<int> hit(std::size_t(rows) * in, 0);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned k = 0; k < in; ++k) {
                const unsigned c = k / 8U, j = k % 8U;
                const std::size_t want = (std::size_t(c / 8U) * rows + r) * 64U + (c % 8U) * 8U + j;
                const std::size_t got = pm::x16_index(rows, r, k);
                EXPECT(got == want);
                if (got < hit.size()) ++hit[got];
            }
        for (int h : hit) EXPECT(h == 1);
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
