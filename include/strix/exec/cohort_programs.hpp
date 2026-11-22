#pragma once

// Resolved policy of the multi-request decode programs (strix/device/width_graphs.hpp, strix/exec/step_lanes.hpp).
// EXACT schedule change: a program replays the eager lane stages with the same bits per lane, so it is on by
// default wherever an engine binds it; the switches only choose the schedule.
//   STRIX_COHORT_GRAPHS=0            every multi-request step runs eagerly (the incumbent launch sequence)
//   STRIX_COHORT_WIDTHS=exact        one program per request count (default; no padded rows)
//   STRIX_COHORT_WIDTHS=classes      width classes 2/4/8/16/32 with padded rows (fewer programs)

#include "strix/env_flags.hpp"
#include "strix/exec/step_lanes.hpp"

#include <cstdint>
#include <string_view>

namespace strix::exec {

struct CohortProgramPolicy {
    bool enabled = true;
    CohortWidths widths = CohortWidths::exact;

    [[nodiscard]] std::uint32_t width(std::uint32_t lanes, std::uint32_t capacity) const noexcept {
        return enabled ? cohort_program_width(lanes, capacity, widths) : 0U;
    }
};

[[nodiscard]] inline std::string_view cohort_widths_name(CohortWidths widths) noexcept {
    return widths == CohortWidths::exact ? "exact" : "classes";
}

// Read once by the owner at construction (server-level setting; a change needs a restart).
[[nodiscard]] inline CohortProgramPolicy resolve_cohort_programs() {
    CohortProgramPolicy policy;
    policy.enabled = envcfg::flag("STRIX_COHORT_GRAPHS", true);
    policy.widths = envcfg::choice("STRIX_COHORT_WIDTHS", {"exact", "classes"}, 0) == 1 ? CohortWidths::classes
                                                                                       : CohortWidths::exact;
    return policy;
}

}  // namespace strix::exec
