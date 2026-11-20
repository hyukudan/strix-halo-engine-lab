#pragma once
// Physical form selection of the shared exact FMA-rows core (src/core/fma_rows.hpp), as a pure host policy: every form
// gives the same bytes, so the choice is scheduling only and depends on the shape and on the compute-unit count of the
// device that runs it. The thresholds were measured on gfx1151 (40 CUs: fill target 80 work groups = 2 per CU, see the
// form table in fma_rows.hpp); a device with another CU count gets the same rule scaled by its CU count (unmeasured
// there: a second target re-measures the table and may replace the rule, it never inherits gfx1151's absolute counts).
// Home: shared (backend core policy, backend-neutral host code). Bound by: src/core/fma_rows.hpp launch_auto.

#include <string_view>

namespace strix::core::fma_rows {

inline constexpr unsigned kGroupMax = 3;
enum class Form : unsigned { Auto = 0, Pipe128 = 1, Pipe64 = 2, PipeNp3 = 3, Xres16 = 4, Xres8 = 5, Xres4 = 6, Pipe128K16 = 7 };
inline constexpr unsigned kFormCount = 8;
inline constexpr unsigned kXresIn = 128, kLongRows = 4096;
// CU count of the part the table was measured on (gfx1151) and the work groups per CU of its fill target.
inline constexpr unsigned kMeasuredComputeUnits = 40U, kFillGroupsPerCu = 2U;

// Work groups that fill a device of `cu` compute units (80 on the measured part).
constexpr unsigned fill_groups(unsigned cu) { return kFillGroupsPerCu * cu; }

// CUs of a device from the runtime's report: arch = name without feature suffix (HIP gcnArchName up to ':'),
// multiprocessors = HIP multiProcessorCount, which counts work-group processors (2 CUs each) on RDNA (gfx10 / gfx11 /
// gfx12) and CUs on GCN / CDNA (gfx9). 0 = unknown (the caller keeps the measured part's count).
constexpr unsigned compute_units_from(std::string_view arch, unsigned multiprocessors) {
    if (multiprocessors == 0U || arch.size() < 5U || arch.substr(0, 3) != "gfx") return 0U;
    const char major = arch[3];
    if (major == '1' && (arch[4] == '0' || arch[4] == '1' || arch[4] == '2')) return 2U * multiprocessors;  // RDNA: WGPs
    if (major == '9') return multiprocessors;                                                                // GCN / CDNA
    return 0U;
}

// The form for a group of `blocks` 64-column blocks (sum over its projections), `in` inputs, `rows` rows on a device of
// `cu` compute units (cu == kMeasuredComputeUnits reproduces the measured selection exactly).
constexpr Form select_form(unsigned in, unsigned rows, unsigned blocks, unsigned projections, unsigned cu) {
    const unsigned fill = fill_groups(cu);
    if (in <= kXresIn) {
        if (rows < kLongRows) return Form::Pipe128K16;
        const unsigned tiles = (rows + 63U) / 64U;
        if (tiles * ((blocks + 15U) / 16U) >= fill) return Form::Xres16;
        if (tiles * ((blocks + 7U) / 8U) >= fill) return Form::Xres8;
        return Form::Xres4;
    }
    if (rows >= kLongRows) return projections > 1U ? Form::PipeNp3 : Form::Pipe64;
    return ((rows + 127U) / 128U) * blocks < cu ? Form::Pipe64 : Form::Pipe128;
}

}  // namespace strix::core::fma_rows
