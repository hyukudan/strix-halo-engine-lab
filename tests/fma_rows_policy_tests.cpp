// FMA-rows form policy (include/strix/fma_rows_policy.hpp): on the measured part (40 CUs) the CU-scaled rule picks
// exactly the forms of the former fixed rule (fill 80 work groups, 40 CUs) over the whole shape grid; other CU counts
// scale the fill target; the runtime report maps to CUs per architecture family.
#include "strix/fma_rows_policy.hpp"

#include <cstdio>
#include <cstdlib>

namespace F = strix::core::fma_rows;

namespace {

int failures = 0;
void require(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// The selection as it was before the policy took the CU count (constants of the gfx1151 table).
F::Form fixed_rule(unsigned in, unsigned rows, unsigned blocks, unsigned projections) {
    constexpr unsigned kFill = 80U, kCu = 40U;
    if (in <= F::kXresIn) {
        if (rows < F::kLongRows) return F::Form::Pipe128K16;
        const unsigned tiles = (rows + 63U) / 64U;
        if (tiles * ((blocks + 15U) / 16U) >= kFill) return F::Form::Xres16;
        if (tiles * ((blocks + 7U) / 8U) >= kFill) return F::Form::Xres8;
        return F::Form::Xres4;
    }
    if (rows >= F::kLongRows) return projections > 1U ? F::Form::PipeNp3 : F::Form::Pipe64;
    return ((rows + 127U) / 128U) * blocks < kCu ? F::Form::Pipe64 : F::Form::Pipe128;
}

void test_measured_part_unchanged() {
    const unsigned ins[] = {32U, 64U, 96U, 128U, 160U, 512U, 4096U, 6144U};
    unsigned checked = 0, diffs = 0;
    for (unsigned in : ins)
        for (unsigned rows = 1U; rows <= 16384U; rows += (rows < 512U ? 1U : 61U))
            for (unsigned blocks = 1U; blocks <= 400U; ++blocks)
                for (unsigned p = 1U; p <= F::kGroupMax; ++p) {
                    ++checked;
                    if (F::select_form(in, rows, blocks, p, F::kMeasuredComputeUnits) != fixed_rule(in, rows, blocks, p)) ++diffs;
                }
    std::printf("measured part: %u shapes, %u differences\n", checked, diffs);
    require(checked > 1000000U && diffs == 0U, "40 CUs reproduce the fixed rule on every shape");
}

void test_scaling() {
    require(F::fill_groups(40U) == 80U, "fill target 80 on 40 CUs");
    require(F::fill_groups(20U) == 40U, "fill target scales with the CU count");
    // 4096 rows, 128 inputs: 64 tiles. 16 blocks -> 64 x 1 = 64 groups: below 80 (Xres8 needs 64 x 2 = 128 >= 80),
    // above a 20-CU part's 40.
    require(F::select_form(128U, 4096U, 16U, 1U, 40U) == F::Form::Xres8, "40 CUs: 64 groups do not fill");
    require(F::select_form(128U, 4096U, 16U, 1U, 20U) == F::Form::Xres16, "20 CUs: 64 groups fill");
    // Short chains, 1024 rows: 8 x blocks groups against the CU count.
    require(F::select_form(4096U, 1024U, 4U, 1U, 40U) == F::Form::Pipe64, "40 CUs: 32 groups < 40");
    require(F::select_form(4096U, 1024U, 4U, 1U, 32U) == F::Form::Pipe128, "32 CUs: 32 groups fill");
    require(F::select_form(4096U, 1024U, 4U, 1U, 304U) == F::Form::Pipe64, "304 CUs: 32 groups do not");
}

void test_compute_units_from() {
    require(F::compute_units_from("gfx1151", 20U) == 40U, "gfx1151: 20 WGPs = 40 CUs");
    require(F::compute_units_from("gfx1100", 48U) == 96U, "gfx1100: WGPs");
    require(F::compute_units_from("gfx1201", 32U) == 64U, "gfx1201: WGPs");
    require(F::compute_units_from("gfx1030", 40U) == 80U, "gfx1030: WGPs");
    require(F::compute_units_from("gfx942", 304U) == 304U, "gfx942: CUs");
    require(F::compute_units_from("gfx90a", 104U) == 104U, "gfx90a: CUs");
    require(F::compute_units_from("gfx1151", 0U) == 0U, "no report: unknown");
    require(F::compute_units_from("sm_120", 188U) == 0U, "not a HIP arch name: unknown");
    require(F::compute_units_from("gfx8", 10U) == 0U, "short name: unknown");
    require(F::compute_units_from("gfx803", 64U) == 0U, "unmapped family: unknown");
}

}  // namespace

int main() {
    test_measured_part_unchanged();
    test_scaling();
    test_compute_units_from();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("fma_rows policy tests PASS\n");
    return EXIT_SUCCESS;
}
