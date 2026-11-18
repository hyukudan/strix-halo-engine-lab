// core::dense_row_sweeps (src/core/dense_linear.hpp), the row split of every small-row dense projection call.
//   1. rows 1..16: exactly one call (offset 0, the real count): one weight pass per window. The power-of-two split it
//      replaces streamed the weights 2-4 times for rows 3 / 5 / 6 / 7 / 9..15 (jagged verify cost, hip_dense_dec4).
//   2. rows > 16: 16-row sweeps then a power-of-two tail (unchanged); every call <= 16 rows; calls tile [0, rows)
//      contiguously with no overlap or gap; no call extends past the window.
//   3. rows 0: no call.
#include "core/dense_linear.hpp"

#include <cstdio>
#include <utility>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what, unsigned rows) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL rows %u: %s\n", rows, what);
    }
}
std::vector<std::pair<unsigned, unsigned>> calls_for(unsigned rows) {
    std::vector<std::pair<unsigned, unsigned>> calls;
    strix::core::dense_row_sweeps(rows, [&](unsigned offset, unsigned count) { calls.emplace_back(offset, count); });
    return calls;
}
// The previous split (powers of two from 16 down), for the weight-pass count.
unsigned power_of_two_passes(unsigned rows) {
    unsigned passes = 0;
    for (unsigned tile = 16, left = rows; tile; tile /= 2)
        while (left >= tile) left -= tile, ++passes;
    return passes;
}
}  // namespace

int main() {
    check(calls_for(0).empty(), "no call for an empty window", 0);
    unsigned saved = 0;
    for (unsigned rows = 1; rows <= 16; ++rows) {
        const auto c = calls_for(rows);
        check(c.size() == 1 && c[0].first == 0 && c[0].second == rows, "one call of the real count", rows);
        saved += power_of_two_passes(rows) - 1U;
    }
    // rows 3,5,6,7,9..15 were split; the extra weight passes of the old form over rows 1..16.
    check(saved == 1 + 1 + 1 + 2 + 1 + 1 + 2 + 1 + 2 + 2 + 3, "old split extra passes (3:1 5:1 6:1 7:2 9:1 10:1 11:2 12:1 13:2 14:2 15:3)", 16);
    for (unsigned rows = 17; rows <= 300; ++rows) {
        const auto c = calls_for(rows);
        unsigned next = 0;
        bool ok = !c.empty();
        for (const auto& [offset, count] : c) {
            ok = ok && offset == next && count >= 1 && count <= 16 && (count & (count - 1U)) == 0U && offset + count <= rows;
            next = offset + count;
        }
        check(ok && next == rows, "16-row sweeps + power-of-two tail tile the window exactly", rows);
        check(c.size() == power_of_two_passes(rows), "wide windows keep the previous split", rows);
    }
    if (failures == 0) std::printf("core dense_row_sweeps tests: PASS\n");
    return failures == 0 ? 0 : 1;
}
