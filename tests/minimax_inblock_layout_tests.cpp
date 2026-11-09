#include "../src/minimax/attn/msa_inblock_layout.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

namespace {
constexpr unsigned kGroup = 16, kMaxChunks = 16, kTopK = 16;
constexpr unsigned kQHeads = 64, kLanes = 32, kDimsPerLane = 4;
int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

void test_inblock_layout() {
    using ib = strix::minimax::msa::fused::inblock::Layout<128, 16, 16, 16>;
    std::array<unsigned, ib::kStatsCount> hits{};
    for (unsigned head = 0; head < kGroup; ++head)
        for (unsigned field = 0; field < 2; ++field)
            for (unsigned count = 1; count <= kMaxChunks; ++count) {
                // Every legal selected-list length, including a partial last list.
                for (unsigned chunk = 0; chunk < count; ++chunk) {
                    const unsigned index = ib::stats_index(head, field, chunk);
                    check(index < hits.size(), "in-block scalar LDS bounds");
                    if (index < hits.size() && count == kMaxChunks) ++hits[index];
                    if (chunk != 0)
                        check(index == ib::stats_index(head, field, chunk - 1) + 1,
                              "in-block consecutive combine lanes");
                }
            }
    check(std::all_of(hits.begin(), hits.end(), [](unsigned n) { return n == 1; }),
          "in-block scalar LDS has no aliases or holes");
    check(ib::kStatsOffset >= ib::kTileBytes + kTopK * sizeof(std::uint32_t) &&
              ib::kSharedBytes == ib::kStatsOffset + hits.size() * sizeof(float),
          "in-block IDs, scalars and double-buffered tiles are disjoint");

    // Include the next 16-row scratch window and multiple KV groups. Each
    // numerator must match mode 8's public partial layout, excluding m/l slots.
    bool ok = true;
    for (unsigned row = 0; row <= 16; ++row)
        for (unsigned head = 0; head < kQHeads; ++head)
            for (unsigned chunk = 0; chunk < kMaxChunks; ++chunk) {
                const std::size_t base = ((std::size_t{row} * kQHeads + head) * kMaxChunks + chunk) * 130;
                for (unsigned lane = 0; lane < kLanes; ++lane)
                    for (unsigned dim = 0; dim < kDimsPerLane; ++dim)
                        ok = ok && ib::numerator_index(std::size_t{row} * kQHeads + head, chunk, lane, dim) ==
                                       base + 2 + lane * kDimsPerLane + dim;
            }
    check(ok, "in-block numerator scratch matches mode 8 across rows/heads/chunks/lanes");
    static_assert(ib::numerator_index(std::size_t{1} << 32, 15, 31, 3) > (std::size_t{1} << 32));
}

}  // namespace

int main() {
    test_inblock_layout();
    if (g_failures != 0) return 1;
    std::puts("strix-minimax-inblock-layout-tests: all passed");
    return 0;
}
