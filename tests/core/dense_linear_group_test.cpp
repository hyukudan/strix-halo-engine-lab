// Grouped tight-row decode launch (src/core/dense_linear.hpp) on the shared projection-group index
// (strix/decode_projection_group.hpp): every block of the one launch runs exactly one block of exactly one projection's
// own launch, and every block of every own launch is run once. With the shared body this makes each stored value
// bitwise the single-launch value.
#include "core/dense_linear.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using strix::core::dense_linear::kTightGroupMax;
using strix::core::dense_linear::plan_tight_blocks;
using strix::core::dense_linear::TightBlocks;

int failures = 0;

struct Slot {
    unsigned projection;
    std::uint32_t block;
};
// The device kernel's lookup: the shared decode::locate over the plan's prefix table.
Slot locate(const TightBlocks& plan, unsigned block) {
    Slot slot{};
    strix::decode::locate(plan.first, plan.count, block, slot.projection, slot.block);
    return slot;
}

void expect(bool ok, const char* what, unsigned a, unsigned b) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s (%u, %u)\n", what, a, b);
}

unsigned own_blocks(unsigned outputs, unsigned block_outputs) { return (outputs + block_outputs - 1U) / block_outputs; }

void check_group(const std::vector<unsigned>& outputs, unsigned block_outputs) {
    TightBlocks plan{};
    const auto count = static_cast<unsigned>(outputs.size());
    expect(plan_tight_blocks(outputs.data(), count, block_outputs, plan), "plan accepted", count, block_outputs);
    std::vector<std::vector<unsigned>> seen(count);
    for (unsigned p = 0; p < count; ++p) seen[p].assign(own_blocks(outputs[p], block_outputs), 0U);
    unsigned total = 0U;
    for (unsigned p = 0; p < count; ++p) total += own_blocks(outputs[p], block_outputs);
    expect(plan.first[count] == total, "launch blocks = sum of own grids", plan.first[count], total);
    for (unsigned block = 0; block < plan.first[count]; ++block) {
        const auto slot = locate(plan, block);
        expect(slot.projection < count, "projection in range", block, slot.projection);
        if (slot.projection >= count) continue;
        expect(slot.block < seen[slot.projection].size(), "local block inside own grid", block, slot.block);
        if (slot.block < seen[slot.projection].size()) ++seen[slot.projection][slot.block];
    }
    for (unsigned p = 0; p < count; ++p)
        for (unsigned b = 0; b < seen[p].size(); ++b) expect(seen[p][b] == 1U, "own block run exactly once", p, b);
}

}  // namespace

int main() {
    // Decode tight-row block widths: 4 waves x tile 2 (rows <= 8) or 4 (rows 16).
    const unsigned widths[] = {8U, 16U};
    const std::vector<std::vector<unsigned>> groups = {
        {2304U, 2304U},          // residual controller forget + gain
        {3072U, 1024U, 1024U},   // attention prepare
        {24U, 384U, 128U},       // small projections in one group
        {1U},                    {7U, 9U},     {8U, 8U, 8U, 8U}, {17U, 1U, 33U, 4097U},
    };
    for (const auto width : widths)
        for (const auto& group : groups) check_group(group, width);

    TightBlocks plan{};
    const unsigned zero[] = {8U, 0U};
    expect(!plan_tight_blocks(zero, 2U, 8U, plan), "zero outputs rejected", 0U, 0U);
    const unsigned five[] = {1U, 1U, 1U, 1U, 1U};
    expect(!plan_tight_blocks(five, kTightGroupMax + 1U, 8U, plan), "count above max rejected", kTightGroupMax + 1U, 0U);
    expect(!plan_tight_blocks(five, 0U, 8U, plan), "empty group rejected", 0U, 0U);
    const unsigned huge[] = {0xffffffffU, 0xffffffffU};
    expect(!plan_tight_blocks(huge, 2U, 1U, plan), "32-bit block overflow rejected", 0U, 0U);

    static_assert([] {
        TightBlocks p{};
        const unsigned outputs[] = {2304U, 2304U};
        return plan_tight_blocks(outputs, 2U, 8U, p) && p.count == 2U && p.first[0] == 0U && p.first[1] == 288U &&
               p.first[2] == 576U;
    }());
    {
        TightBlocks p{};
        const unsigned outputs[] = {2304U, 2304U};
        expect(plan_tight_blocks(outputs, 2U, 8U, p), "pair plan", 0U, 0U);
        expect(locate(p, 287U).projection == 0U && locate(p, 288U).projection == 1U && locate(p, 288U).block == 0U,
               "pair boundary", 287U, 288U);
        // A rejected plan leaves the caller's table untouched.
        const auto before = p;
        expect(!plan_tight_blocks(zero, 2U, 8U, p) && p.count == before.count && p.first[2] == before.first[2],
               "rejected plan untouched", 0U, 0U);
    }

    if (failures != 0) return 1;
    std::printf("dense_linear grouped tight rows: every launch block maps to one own block, each own block once "
                "(%zu groups x 2 widths)\n", groups.size());
    return 0;
}
