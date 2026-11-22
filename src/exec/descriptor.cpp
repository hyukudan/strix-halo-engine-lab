#include "strix/exec/descriptor.hpp"

#include <bit>
#include <limits>
#include <stdexcept>

namespace strix::exec {

StepAdvance advance_step(const StepBlock& current, std::span<const RowOut> picks, const StepStopTable& host_stops) {
    const auto& header = current.h;
    if (!std::has_single_bit(header.rows) || header.rows > kStepMaxRows || picks.size() != header.rows)
        throw std::invalid_argument("invalid step descriptor width or readback size");
    if (header.stop_count > kStepMaxStops) throw std::invalid_argument("invalid step stop count");

    StepAdvance result{.next = current, .readback = {}, .commit_mask = 0};
    ++result.next.h.step;
    result.next.h.valid_mask = 0;
    result.next.h.feed_mask = 0;
    for (std::uint32_t i = 0; i < header.rows; ++i) {
        const auto bit = std::uint32_t{1} << i;
        auto& next = result.next.row[i];
        if (!step_row_valid(current, i)) {
            next.flags &= ~step_valid;
            result.readback[i] = {0, 0.0F, 0.0F, RowStatus::masked};
            continue;
        }
        if (next.parent != kStepNoParent || (next.ancestors & ~bit) != 0 || (next.flags & step_pending_state) != 0)
            throw std::invalid_argument("step advance requires independent decode rows");
        if (next.position == std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("step frontier overflow");

        const auto& pick = picks[i];
        if (pick.status != RowStatus::picked && pick.status != RowStatus::stop)
            throw std::invalid_argument("active step row has no pick");
        const bool stopped = pick.status == RowStatus::stop || step_stops_on(header, host_stops, pick.id);
        result.readback[i] = pick;
        ++next.position;
        result.commit_mask |= bit;
        if (stopped) result.readback[i].status = RowStatus::stop;
        if (stopped || (header.feed_mask & bit) == 0) {
            next.flags &= ~step_valid;
        } else {
            next.token = pick.id;
            result.next.h.valid_mask |= bit;
            result.next.h.feed_mask |= bit;
        }
    }
    return result;
}

}  // namespace strix::exec
