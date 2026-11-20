#include "strix/mtp_selection_transaction.hpp"

#include <algorithm>

namespace strix {

MtpSelectionPrefix walk_mtp_selection(std::span<const std::uint32_t> proposals,
    std::size_t row_limit, std::size_t output_limit, std::span<const std::uint32_t> eos_ids,
    const std::function<std::uint32_t(std::size_t)>& select_row) {
    MtpSelectionPrefix plan;
    plan.selected_ids.reserve(std::min(row_limit, output_limit));
    if (!output_limit) { plan.stop = MtpSelectionStop::output_limit; return plan; }
    for (std::size_t row = 0; row < row_limit; ++row) {
        std::uint32_t id;
        try { id = select_row(row); }
        catch(const MtpSelectionError& e) {
            plan.stop=MtpSelectionStop::selection_failure;plan.selection_failure=e.what();break;
        }
        plan.selected_ids.push_back(id);
        ++plan.decided_rows;
        if (std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end()) plan.stop = MtpSelectionStop::eos;
        else if (plan.decided_rows == output_limit) plan.stop = MtpSelectionStop::output_limit;
        else if (plan.decided_rows == row_limit) plan.stop = MtpSelectionStop::row_limit;
        else if (row == proposals.size()) plan.stop = MtpSelectionStop::no_draft;
        else if (proposals[row] != id) plan.stop = MtpSelectionStop::rejection;
        else { ++plan.accepted_draft_inputs; continue; }
        break;
    }
    return plan;
}

GreedyWindowSelection select_greedy_window(MtpSelectionTransaction& transaction,
    std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> argmax) {
    if (argmax.empty() || drafts.size() + 1 != argmax.size())
        throw std::invalid_argument("greedy window: one argmax reply per input row (the pending token and every draft)");
    transaction.reset();
    transaction.expect(drafts);
    auto plan = walk_mtp_selection(drafts, argmax.size(), argmax.size(), {}, [&](std::size_t row) {
        if (const auto known = transaction.select_known(argmax[row], 0)) return *known;
        throw MtpSelectionError("greedy window: row " + std::to_string(row) +
                                " needs host logits; this engine replies with each row's argmax only");
    });
    if (plan.selection_failure) {
        transaction.discard();
        throw MtpSelectionError(*plan.selection_failure);
    }
    return {std::move(plan.selected_ids), transaction.seal()};
}

}  // namespace strix
