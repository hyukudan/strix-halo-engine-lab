// core/lds_rows.hpp: slot32 is a bijection inside its tile; under the gfx11 LDS model (32 banks x 4 B; a 16-B access
// serves 8 lanes per cycle) the fragment reads and the row stores of its user hit distinct 4-bank groups, it repeats
// every 16 rows up to a constant (one per-lane base + immediate offsets), and the plain 32-B rows it replaces reproduce
// the measured 2-way read conflicts.
#include "core/lds_rows.hpp"

#include <cstdio>
#include <set>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what, unsigned a, unsigned b) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s (%u, %u)\n", what, a, b);
    }
}
namespace lr = strix::core::lds_rows;

// Largest number of distinct 16-B addresses of one LDS cycle that share a 4-bank group (1 = conflict-free).
unsigned ways(const std::vector<unsigned>& byte_addresses) {
    std::set<unsigned> groups[8];
    for (unsigned a : byte_addresses) groups[(a / 16U) % 8U].insert(a / 16U);
    unsigned w = 0;
    for (const auto& g : groups) w = g.size() > w ? static_cast<unsigned>(g.size()) : w;
    return w;
}

}  // namespace

int main() {
    static_assert(lr::slot32(0, 0) == 0 && lr::slot32(4, 0) == 9 && lr::slot32(4, 1) == 8, "slot32 formula");
    for (unsigned rows : {16U, 64U, 128U, 256U, 512U}) {
        std::vector<int> seen32(2U * rows, 0);
        for (unsigned r = 0; r < rows; ++r) {
            for (unsigned h = 0; h < 2; ++h) {
                const unsigned sl = lr::slot32(r, h);
                check(sl < 2U * rows, "slot32 inside tile", r, h);
                if (sl < 2U * rows) seen32[sl] += 1;
            }
            if (r + 16U < rows) {  // 16-row period: one per-lane base address plus an immediate per fragment
                check(lr::slot32(r + 16U, 0) == lr::slot32(r, 0) + 32U && lr::slot32(r + 16U, 1) == lr::slot32(r, 1) + 32U, "slot32 period", r, 0);
            }
        }
        for (unsigned i = 0; i < 2U * rows; ++i) check(seen32[i] == 1, "slot32 bijection", rows, i);
        // Fragment reads: one cycle = rows R .. R + 7 (R % 8 == 0), one 16-B half each.
        for (unsigned R = 0; R < rows; R += 8)
            for (unsigned h = 0; h < 2; ++h) {
                std::vector<unsigned> s32, plain32;
                for (unsigned r = R; r < R + 8; ++r) {
                    s32.push_back(16U * lr::slot32(r, h));
                    plain32.push_back(32U * r + 16U * h);
                }
                check(ways(s32) == 1, "slot32 read conflict-free", R, h);
                check(ways(plain32) == 2, "plain 32-B read 2-way (model)", R, h);
            }
        // Row stores (dense V^T stage): 8 lanes = rows 4 j .. 4 j + 3, both halves, 16 B each.
        for (unsigned j = 0; j + 4 <= rows; j += 4) {
            std::vector<unsigned> s32;
            for (unsigned r = j; r < j + 4; ++r)
                for (unsigned h = 0; h < 2; ++h) s32.push_back(16U * lr::slot32(r, h));
            check(ways(s32) == 1, "slot32 row store conflict-free", j, 0);
        }
    }
    if (failures) {
        std::fprintf(stderr, "lds_rows: %d failures\n", failures);
        return 1;
    }
    std::printf("lds_rows: ok\n");
    return 0;
}
