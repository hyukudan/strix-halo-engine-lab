#pragma once

// Multi-request decode steps on the X3 step ABI (strix/exec/descriptor.hpp), for programs captured once
// per program width and replayed for any request set of that width (strix/exec/width_graphs.hpp).
//
// Lane l of a width-W program is StepBlock row l: one query row of its own session (StepRow::slot) at its
// own absolute position (StepRow::position) with its own input id (StepRow::token). Kernels see the rows
// through the backend-neutral lane view (strix/state/lane_view.hpp): strided columns of the device block
// plus the slot directory. Nothing a kernel reads is a host position or a host slot address, so one
// host-to-device copy of the block (1,056 B) per step is the whole per-step input of a replayed program.
//
// Padding: a program of width W serves R <= W requests. Rows [R, W) repeat row 0's slot, position and id
// with every flag clear: they are never picked and every kernel that WRITES session state skips them.

#include "strix/exec/descriptor.hpp"
#include "strix/state/lane_view.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::exec {

// 32-bit words between consecutive rows: the stride of a StepRow column.
inline constexpr std::uint32_t kStepRowWords = static_cast<std::uint32_t>(sizeof(StepRow) / sizeof(std::uint32_t));
static_assert(sizeof(StepRow) % sizeof(std::uint32_t) == 0 && (step_valid & state::kLaneWrites) != 0U);

// The lane view of a device step block (address arithmetic only: the block is device memory). A row's
// write flag is StepRow::flags & step_valid; make_cohort_step sets it exactly on the requests.
[[nodiscard]] inline state::LaneView step_lane_view(const StepBlock* device_block, const state::SlotEntry* entries,
                                                    std::uint32_t slots, std::uint32_t lanes) {
    const auto* base = reinterpret_cast<const unsigned char*>(device_block) + offsetof(StepBlock, row);
    const auto column = [&](std::size_t field) { return reinterpret_cast<const std::uint32_t*>(base + field); };
    return {column(offsetof(StepRow, position)), column(offsetof(StepRow, slot)), column(offsetof(StepRow, flags)),
            entries, kStepRowWords, slots, lanes};
}

// Width classes of captured cohort programs. A request set of R lanes runs in the smallest class >= R;
// classes are data so a model or device may narrow them (ascending powers of two <= kStepMaxRows).
inline constexpr std::array<std::uint32_t, 5> kCohortWidthClasses{2U, 4U, 8U, 16U, 32U};

[[nodiscard]] constexpr bool valid_width_classes(std::span<const std::uint32_t> classes) noexcept {
    std::uint32_t previous = 0;
    for (const auto width : classes) {
        if (width == 0U || width > kStepMaxRows || (width & (width - 1U)) != 0U || width <= previous) return false;
        previous = width;
    }
    return true;
}

// The class width for `lanes` requests: the smallest class >= lanes that is <= capacity (the rows the
// engine's workspaces hold). 0 = no class fits: run the step without a captured program.
[[nodiscard]] constexpr std::uint32_t cohort_width_class(std::uint32_t lanes, std::uint32_t capacity,
                                                         std::span<const std::uint32_t> classes =
                                                             kCohortWidthClasses) noexcept {
    if (lanes == 0U || !valid_width_classes(classes)) return 0U;
    for (const auto width : classes) {
        if (width >= lanes) return width <= capacity ? width : 0U;
    }
    return 0U;
}

// How a request count maps to a program width. exact: one program per request count (no padded rows: no
// expert, head or attention row work is spent on padding; at most kStepMaxRows captures). classes:
// kCohortWidthClasses with padded rows (fewer programs; padded rows cost row work).
enum class CohortWidths : std::uint8_t { exact, classes };

// 0 = run the step without a program. A lone lane always runs at width 1.
[[nodiscard]] constexpr std::uint32_t cohort_program_width(std::uint32_t lanes, std::uint32_t capacity,
                                                           CohortWidths widths) noexcept {
    if (lanes == 0U || lanes > capacity || lanes > kStepMaxRows) return 0U;
    if (lanes == 1U || widths == CohortWidths::exact) return lanes;
    return cohort_width_class(lanes, capacity);
}

struct CohortLane {
    std::uint32_t slot, position, token;
};

// The step block of one cohort decode step: rows [0, R) are the requests in caller order (valid, sampled
// by the program), rows [R, width) repeat row 0 with every flag clear. Refuses an empty or too wide set, a
// width outside [1, kStepMaxRows], a slot outside the directory, a position at or past `context`, and two
// rows of one slot (each lane writes its own state).
[[nodiscard]] inline StepBlock make_cohort_step(std::span<const CohortLane> lanes, std::uint32_t width,
                                                std::uint32_t slots, std::uint32_t context, std::uint32_t step = 0U) {
    if (lanes.empty() || width == 0U || width > kStepMaxRows || lanes.size() > width)
        throw std::invalid_argument("invalid cohort step width");
    StepBlock block{};
    block.h.rows = width;
    block.h.step = step;
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        const auto& lane = lanes[i];
        if (lane.slot >= slots || lane.position >= context)
            throw std::out_of_range("cohort step lane outside the slot directory or the context");
        for (std::size_t j = 0; j < i; ++j) {
            if (lanes[j].slot == lane.slot) throw std::invalid_argument("two cohort step lanes share a slot");
        }
        block.row[i] = {lane.token, lane.position, lane.slot, kStepNoParent, 0U, step_valid | step_sample, 0U, 0U};
        block.h.valid_mask |= std::uint32_t{1} << i;
    }
    for (std::size_t i = lanes.size(); i < width; ++i) {
        block.row[i] = block.row[0];
        block.row[i].flags = 0U;
    }
    return block;
}

}  // namespace strix::exec
