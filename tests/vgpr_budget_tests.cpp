// core/vgpr_budget.hpp + attn_core/dense_flash_budget.hpp: the gfx11 wave32 VGPR budget per occupancy (24-register
// granule of a 1,536-register file) and the dense flash core's occupancy targets for the shapes the engines launch,
// next to the VGPR counts the compiler reported for them (gfx1151, tests/hip_dense_flash_bench.hip) so that a change of
// the estimate that moves an engine shape across a budget line fails here first.
#include "attn_core/dense_flash_budget.hpp"
#include "core/vgpr_budget.hpp"

#include <cstdio>

namespace {

int failures = 0;
void check(bool ok, const char* what, unsigned a, unsigned b) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s (%u, %u)\n", what, a, b);
    }
}
namespace vb = strix::core::vgpr_budget;
namespace dn = strix::attn_core::dense;

struct Shape {
    const char* name;
    unsigned dim, group, split, positions, terms;
    unsigned target;     // expected occupancy target
    unsigned measured;   // VGPRs the compiler used at that target (gfx1151, release toolchain, bench build), 0 = not built
};

}  // namespace

int main() {
    // Budget lines: {largest VGPR count of an occupancy, that occupancy, the occupancy one register later}.
    const unsigned lines[][3] = {{192U, 8U, 7U}, {216U, 7U, 6U}, {240U, 6U, 5U}, {96U, 16U, 12U}, {120U, 12U, 10U}, {168U, 9U, 8U}};
    for (const auto& l : lines) {
        check(vb::waves_for(l[0]) == l[1], "waves_for at the line", l[0], vb::waves_for(l[0]));
        check(vb::waves_for(l[0] + 1U) == l[2], "one register more", l[0] + 1U, vb::waves_for(l[0] + 1U));
    }
    check(vb::waves_for(vb::kMaxVgprs) == 5U, "wave32 maximum", vb::kMaxVgprs, vb::waves_for(vb::kMaxVgprs));
    check(vb::waves_for(0U) == 16U && vb::waves_for(1U) == 16U, "small kernels: 16 waves", vb::waves_for(1U), 16U);
    for (unsigned v = 1U; v < 256U; ++v) check(vb::waves_for(v) >= vb::waves_for(v + 1U), "monotone", v, vb::waves_for(v));

    // Engine shapes (Shape<D, Group, Split, Positions>: frags = D / Split / 16, per_thread = ceil(2 D / threads)).
    const Shape shapes[] = {
        {"MHA sliding <256, 1, 2, 64> (Hy4-flash)", 256U, 1U, 2U, 64U, 1U, 6U, 232U},
        {"D256 group 4 <256, 4, 2, 16> (gemma4 policy keeps 8)", 256U, 4U, 2U, 16U, 1U, 6U, 213U},
        {"gemma4 global <512, 4, 4, 16> (16 waves)", 512U, 4U, 4U, 16U, 1U, 8U, 192U},
        {"qwen35 wide window <256, 6, 2, 16>", 256U, 6U, 2U, 16U, 1U, 6U, 233U},
        {"vision two-term <80, 1, 1, 64>", 80U, 1U, 1U, 64U, 2U, 5U, 229U},
        {"vision two-term <64, 1, 1, 64>", 64U, 1U, 1U, 64U, 2U, 7U, 216U},
        {"vision one-term <80, 1, 1, 64>", 80U, 1U, 1U, 64U, 1U, 8U, 0U},
    };
    for (const auto& s : shapes) {
        const unsigned waves = s.group * s.positions / 16U * s.split, threads = 32U * waves;
        const unsigned frags = s.dim / s.split / 16U, per_thread = (2U * s.dim + threads - 1U) / threads;
        const unsigned live = dn::live_vgprs(frags, per_thread, s.terms), target = dn::occupancy_target(frags, per_thread, s.terms, waves);
        check(target == s.target, s.name, target, s.target);
        check(target <= 8U && (vb::waves_for(live) >= target || waves == 16U), "target within the live budget", live, target);
        if (s.measured) check(vb::waves_for(s.measured) >= target, "measured VGPRs fit the target", s.measured, target);
        std::printf("%-40s frags %u per_thread %u terms %u live %u -> %u waves (measured %u VGPRs)\n", s.name, frags, per_thread,
                    s.terms, live, target, s.measured);
    }
    std::printf("vgpr budget tests: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
