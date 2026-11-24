#include "strix/serve/cohort_passes.hpp"

#include <cstdio>

namespace {

using strix::serve::CohortPasses;
using strix::serve::plan_cohort_passes;

int failures = 0;

void check(bool ok, const char* what, unsigned lanes, unsigned cap) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s (lanes %u, cap %u)\n", what, lanes, cap);
        ++failures;
    }
}

constexpr bool splits(CohortPasses plan, unsigned count, unsigned first_rows, unsigned last_rows) {
    return plan.count == count && plan.rows(0U) == first_rows && plan.rows(count - 1U) == last_rows;
}

static_assert(plan_cohort_passes(0U, 16U).count == 0U);
static_assert(splits(plan_cohort_passes(1U, 16U), 1U, 1U, 1U));
static_assert(splits(plan_cohort_passes(16U, 16U), 1U, 16U, 16U));
static_assert(splits(plan_cohort_passes(17U, 16U), 2U, 9U, 8U));
static_assert(splits(plan_cohort_passes(24U, 16U), 2U, 12U, 12U));
static_assert(splits(plan_cohort_passes(32U, 16U), 2U, 16U, 16U));
static_assert(splits(plan_cohort_passes(33U, 16U), 3U, 11U, 11U));
static_assert(splits(plan_cohort_passes(34U, 16U), 3U, 12U, 11U));
static_assert(splits(plan_cohort_passes(40U, 0U), 1U, 40U, 40U));   // no cap: one pass
static_assert(splits(plan_cohort_passes(40U, 64U), 1U, 40U, 40U));  // cap above the cohort: one pass
static_assert(splits(plan_cohort_passes(65535U, 16U), 4096U, 16U, 15U));

// Every plan covers the lanes in order with contiguous passes, no pass exceeds the cap, larger passes come first and
// differ by at most one row, the pass count is the minimum ceil(lanes / cap), and with a cap of at least 3 a cohort
// above the cap never has a pass of one row (cap 2 with an odd cohort cannot avoid one at the minimum count).
void coverage() {
    for (unsigned cap = 0U; cap <= 64U; ++cap) {
        for (unsigned lanes = 1U; lanes <= 512U; ++lanes) {
            const auto plan = plan_cohort_passes(lanes, cap);
            const unsigned limit = cap == 0U ? lanes : cap;
            const unsigned minimum = (lanes + limit - 1U) / limit;
            check(plan.count == minimum, "minimum pass count", lanes, cap);
            unsigned next = 0U;
            for (unsigned pass = 0U; pass < plan.count; ++pass) {
                const auto rows = plan.rows(pass);
                check(plan.first(pass) == next, "contiguous in order", lanes, cap);
                check(rows >= 1U && rows <= limit, "pass within the cap", lanes, cap);
                check(pass == 0U || rows <= plan.rows(pass - 1U), "larger passes first", lanes, cap);
                check(rows + 1U >= plan.rows(0U), "passes differ by at most one row", lanes, cap);
                check(lanes <= limit || limit < 3U || rows >= 2U, "no one-row pass above the cap", lanes, cap);
                next += rows;
            }
            check(next == lanes, "all lanes covered once", lanes, cap);
        }
    }
}

}  // namespace

int main() {
    coverage();
    if (failures) {
        std::fprintf(stderr, "cohort passes tests: %d failures\n", failures);
        return 1;
    }
    std::printf("cohort passes tests: PASS\n");
    return 0;
}
