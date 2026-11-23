// Multi-request decode programs per width (strix/exec/step_lanes.hpp, strix/exec/width_replay.hpp,
// strix/state/lane_view.hpp):
// the width class of a request set, the X3 step block of one cohort step (valid rows in caller order, padded
// rows repeating row 0 with every flag clear, refusals), the device-side lane resolution evaluated on a host
// directory (slot directory state-major then slot-major, plane offset in bytes, padded rows never write), the
// strided position column, and the per-width capture/replay policy.
#include "strix/exec/cohort_programs.hpp"
#include "strix/exec/step_lanes.hpp"
#include "strix/exec/width_replay.hpp"
#include "strix/state/lane_view.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace ex = strix::exec;
namespace st = strix::state;

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void width_classes() {
    CHECK(ex::cohort_width_class(0, 32) == 0U);
    CHECK(ex::cohort_width_class(1, 32) == 2U);
    CHECK(ex::cohort_width_class(2, 32) == 2U);
    CHECK(ex::cohort_width_class(3, 32) == 4U);
    CHECK(ex::cohort_width_class(4, 32) == 4U);
    CHECK(ex::cohort_width_class(5, 32) == 8U);
    CHECK(ex::cohort_width_class(8, 32) == 8U);
    CHECK(ex::cohort_width_class(9, 32) == 16U);
    CHECK(ex::cohort_width_class(16, 32) == 16U);
    CHECK(ex::cohort_width_class(17, 32) == 32U);
    CHECK(ex::cohort_width_class(32, 32) == 32U);
    CHECK(ex::cohort_width_class(33, 64) == 0U);
    // The class must fit the engine's rows: 5 lanes on 6 rows has no class (eager), on 8 rows it is 8.
    CHECK(ex::cohort_width_class(5, 6) == 0U);
    CHECK(ex::cohort_width_class(5, 8) == 8U);
    CHECK(ex::cohort_width_class(9, 12) == 0U);
    const std::array<std::uint32_t, 2> narrow{4U, 8U};
    CHECK(ex::cohort_width_class(1, 32, narrow) == 4U);
    CHECK(ex::cohort_width_class(9, 32, narrow) == 0U);
    const std::array<std::uint32_t, 2> unsorted{8U, 4U};
    const std::array<std::uint32_t, 2> odd{3U, 8U};
    const std::array<std::uint32_t, 1> wide{64U};
    CHECK(!ex::valid_width_classes(unsorted) && ex::cohort_width_class(2, 32, unsorted) == 0U);
    CHECK(!ex::valid_width_classes(odd) && !ex::valid_width_classes(wide));
    CHECK(ex::valid_width_classes(ex::kCohortWidthClasses));
    static_assert(ex::cohort_width_class(3, 8) == 4U);
    // Program width: exact request counts by default, classes on request; a lone lane runs at width 1.
    using W = ex::CohortWidths;
    CHECK(ex::cohort_program_width(0, 32, W::exact) == 0U && ex::cohort_program_width(1, 32, W::classes) == 1U);
    CHECK(ex::cohort_program_width(5, 32, W::exact) == 5U && ex::cohort_program_width(5, 32, W::classes) == 8U);
    CHECK(ex::cohort_program_width(5, 4, W::exact) == 0U && ex::cohort_program_width(33, 64, W::exact) == 0U);
    CHECK(ex::cohort_program_width(5, 6, W::classes) == 0U && ex::cohort_program_width(6, 6, W::exact) == 6U);
    const ex::CohortProgramPolicy off{false, W::exact};
    const ex::CohortProgramPolicy classes{true, W::classes};
    CHECK(off.width(4, 32) == 0U && classes.width(3, 32) == 4U && ex::CohortProgramPolicy{}.width(3, 32) == 3U);
    CHECK(ex::cohort_widths_name(W::exact) == "exact" && ex::cohort_widths_name(W::classes) == "classes");
}

void step_block() {
    const std::vector<ex::CohortLane> lanes{{5, 100, 11}, {2, 7, 12}, {0, 0, 13}};
    const auto block = ex::make_cohort_step(lanes, 8, 6, 4096, 9);
    CHECK(block.h.rows == 8U && block.h.step == 9U && block.h.valid_mask == 0x7U);
    for (std::uint32_t i = 0; i < 3; ++i) {
        CHECK(block.row[i].slot == lanes[i].slot && block.row[i].position == lanes[i].position &&
              block.row[i].token == lanes[i].token);
        CHECK(ex::step_row_valid(block, i) && (block.row[i].flags & ex::step_sample) != 0U);
        CHECK(block.row[i].parent == ex::kStepNoParent && block.row[i].ancestors == 0U);
    }
    for (std::uint32_t i = 3; i < 8; ++i) {
        CHECK(!ex::step_row_valid(block, i) && block.row[i].flags == 0U);
        CHECK(block.row[i].slot == 5U && block.row[i].position == 100U && block.row[i].token == 11U);
    }
    for (std::uint32_t i = 8; i < ex::kStepMaxRows; ++i) CHECK(!ex::step_row_valid(block, i));
    // The exact width: no padding.
    const auto full = ex::make_cohort_step(std::span(lanes).first(2), 2, 6, 4096);
    CHECK(full.h.rows == 2U && full.h.valid_mask == 0x3U);

    CHECK(throws([&] { (void)ex::make_cohort_step({}, 4, 6, 4096); }));
    CHECK(throws([&] { (void)ex::make_cohort_step(lanes, 2, 6, 4096); }));   // wider than the program
    CHECK(ex::make_cohort_step(lanes, 6, 6, 4096).h.rows == 6U);              // an exact width
    CHECK(throws([&] { (void)ex::make_cohort_step(lanes, 0, 6, 4096); }));   // no width
    CHECK(throws([&] { (void)ex::make_cohort_step(lanes, 64, 6, 4096); }));  // past the step ABI
    CHECK(throws([&] { (void)ex::make_cohort_step(lanes, 4, 5, 4096); }));   // slot 5 outside 5 slots
    CHECK(throws([&] { (void)ex::make_cohort_step(lanes, 4, 6, 100); }));    // position 100 at the context
    const std::vector<ex::CohortLane> shared{{1, 3, 1}, {1, 9, 2}};
    CHECK(throws([&] { (void)ex::make_cohort_step(shared, 2, 6, 4096); }));  // one slot, two lanes
}

void lane_resolution() {
    // Directory of 2 states x 4 slots over host storage; the same arithmetic runs on the device.
    constexpr std::uint32_t kSlots = 4;
    std::vector<std::uint8_t> storage(2U * kSlots * 4096U);
    std::vector<strix::state::SlotEntry> entries(2U * kSlots);
    for (std::uint32_t state = 0; state < 2; ++state) {
        for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
            entries[state * kSlots + slot].base =
                reinterpret_cast<std::uint64_t>(storage.data() + (std::size_t{state} * kSlots + slot) * 4096U);
        }
    }
    const std::vector<ex::CohortLane> lanes{{3, 70, 1}, {1, 5, 2}};
    const auto block = ex::make_cohort_step(lanes, 4, kSlots, 4096);
    const auto view = ex::step_lane_view(&block, entries.data(), kSlots, 4);
    CHECK(view.stride == ex::kStepRowWords && view.slots == kSlots && view.lanes == 4U && view.entries == entries.data());
    const st::LanePlane history{1U, 256U};
    CHECK(st::lane_plane<std::uint8_t>(view, history, 0) == storage.data() + (kSlots + 3U) * 4096U + 256U);
    CHECK(st::lane_plane<std::uint8_t>(view, history, 1) == storage.data() + (kSlots + 1U) * 4096U + 256U);
    CHECK(st::lane_plane<std::uint8_t>(view, {0U, 0U}, 1) == storage.data() + 1U * 4096U);
    // A padded row reads the slot it repeats but never writes.
    CHECK(st::lane_plane<std::uint8_t>(view, history, 3) == st::lane_plane<std::uint8_t>(view, history, 0));
    const st::LaneState<float> state{view, history};
    CHECK(state.writes(0) && state.writes(1) && !state.writes(2) && !state.writes(3));
    CHECK(reinterpret_cast<std::uint8_t*>(state(1)) == storage.data() + (kSlots + 1U) * 4096U + 256U);
    CHECK(st::lane_position(view, 0) == 70U && st::lane_position(view, 1) == 5U && st::lane_position(view, 2) == 70U);
    // The columns are the X3 row fields at row r * kStepRowWords.
    for (std::uint32_t r = 0; r < 4; ++r) {
        const auto at = std::size_t{r} * ex::kStepRowWords;
        CHECK(view.position[at] == block.row[r].position && view.slot[at] == block.row[r].slot &&
              view.flags[at] == block.row[r].flags);
    }
    // clip_rows_kernel: first = position - row, then clip_body adds row back (wrapping unsigned arithmetic).
    for (const std::uint32_t position : {0U, 1U, 5U, 4095U, 70000U}) {
        for (const std::uint32_t row : {0U, 1U, 7U, 31U}) CHECK((position - row) + row == position);
    }
}

void width_policy() {
    ex::WidthReplayPolicy policy;
    static_assert(ex::WidthReplayPolicy::kWidths == 32U);
    CHECK(ex::WidthReplayPolicy::index(1) == 0U && ex::WidthReplayPolicy::index(32) == 31U);
    CHECK(ex::WidthReplayPolicy::index(3) == 2U && ex::WidthReplayPolicy::index(0) == ex::WidthReplayPolicy::kWidths);
    CHECK(ex::WidthReplayPolicy::index(33) == ex::WidthReplayPolicy::kWidths);
    CHECK(policy.select(true, 0, true) == ex::WidthReplayAction::eager);
    CHECK(policy.select(true, 3, true) == ex::WidthReplayAction::capture);
    CHECK(policy.select(true, 64, true) == ex::WidthReplayAction::eager);
    CHECK(policy.select(true, 4, true) == ex::WidthReplayAction::capture);
    policy.captured(4);
    CHECK(policy.ready(4) && policy.select(true, 4, true) == ex::WidthReplayAction::replay);
    // Widths are independent.
    CHECK(policy.select(true, 8, true) == ex::WidthReplayAction::capture);
    policy.fail(8);
    CHECK(policy.failed(8) && policy.select(true, 8, true) == ex::WidthReplayAction::eager);
    CHECK(policy.select(true, 4, true) == ex::WidthReplayAction::replay);
    // Off or unstable: eager and the width must capture again.
    CHECK(policy.select(false, 4, true) == ex::WidthReplayAction::eager && !policy.ready(4));
    CHECK(policy.select(true, 4, true) == ex::WidthReplayAction::capture);
    policy.captured(4);
    CHECK(policy.select(true, 4, false) == ex::WidthReplayAction::eager);
    policy.captured(4);
    policy.captured(16);
    policy.invalidate();
    CHECK(!policy.ready(4) && !policy.ready(16) && policy.failed(8));
    policy.reset();
    CHECK(!policy.failed(8) && policy.select(true, 8, true) == ex::WidthReplayAction::capture);
}

}  // namespace

int main() {
    width_classes();
    step_block();
    lane_resolution();
    width_policy();
    if (failures != 0) {
        std::fprintf(stderr, "step-lanes-tests: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("step-lanes-tests: PASS\n");
    return 0;
}
