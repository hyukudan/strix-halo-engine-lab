#pragma once

#include "strix/exec/graph_submission.hpp"
#include "strix/exec/width_replay.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace strix::exec {

enum class GraphRows { cohort, single, inclusive_window };

// Captured device arguments can only name a different request via the uploaded block.
// Refuse a half-initialized mask before a kernel can read id zero or skip a state write.
inline void validate_graph_rows(const StepBlock& block, GraphRows layout = GraphRows::cohort) {
    if (block.h.rows == 0U || block.h.rows > kStepMaxRows)
        throw std::invalid_argument("invalid graph descriptor width");
    if (block.h.stop_count > kStepMaxStops || (block.h.stop_count && !step_stop_address(block.h)))
        throw std::invalid_argument("invalid graph stop table");
    if (layout == GraphRows::single && block.h.rows != 1U)
        throw std::invalid_argument("single-row graph requires one row");
    std::uint32_t valid = 0;
    for (std::uint32_t i = 0; i < block.h.rows; ++i) {
        const auto bit = std::uint32_t{1} << i;
        const auto& row = block.row[i];
        if (row.flags & step_valid) valid |= bit;
        if (layout == GraphRows::inclusive_window &&
            (row.flags != step_valid || row.slot != block.row[0].slot ||
             row.position != std::uint64_t{block.row[0].position} + i ||
             row.parent != (i == 0U ? kStepNoParent : i - 1U) || row.ancestors != ((bit - 1U) | bit)))
            throw std::invalid_argument("invalid inclusive verify window");
    }
    if (valid != block.h.valid_mask || (block.h.feed_mask & ~valid) != 0U)
        throw std::invalid_argument("inconsistent graph descriptor masks");
}

// A binding contains only the owner's capture-time dependencies. Device-read values belong
// in the StepBlock. A fully indirect owner with stable allocations needs no additional key.
struct IndirectGraphBinding {
    bool operator==(const IndirectGraphBinding&) const = default;
};

// One serial owner per program family and stream. A runtime records work with capture(),
// instantiates and inspects the candidate, uploads into stable storage, then launches it.
// The serial owner retires staging reads before submitting again. Only a pre-launch decline
// may fall back to eager work; an uncertain submission is never retried here.
template <class Binding = IndirectGraphBinding>
class StepGraphRunner {
public:
    explicit StepGraphRunner(GraphRows layout = GraphRows::cohort) : layout_(layout) {}

    template <class Runtime, class Stream, class Work>
    bool run(Runtime& runtime, Stream stream, bool enabled, bool stable, const StepBlock& block,
             const Binding& binding, Work& work) {
        const auto width = block.h.rows;
        const auto index = WidthReplayPolicy::index(width);
        if (index >= WidthReplayPolicy::kWidths) return false;
        if (enabled && stable) validate_graph_rows(block, layout_);
        if (bindings_[index] && *bindings_[index] != binding) {
            runtime.discard(index);
            policy_.invalidate_width(width);
            bindings_[index].reset();
        }
        const auto action = policy_.select(enabled, width, stable);
        if (action == WidthReplayAction::eager) {
            runtime.discard(index);
            return false;
        }
        return submit_graph(action == WidthReplayAction::capture, [&] {
            bindings_[index] = binding;
            if (!runtime.capture(stream, index, work)) return false;
            policy_.captured(width);
            return true;
        }, [&] {
            runtime.upload(stream, block);
            return true;
        }, [&] { runtime.launch(stream, index); }, [&] {
            runtime.discard(index);
            policy_.fail(width);
        });
    }

    template <class Runtime>
    void invalidate(Runtime& runtime) noexcept {
        for (std::uint32_t i = 0; i < WidthReplayPolicy::kWidths; ++i) runtime.discard(i);
        policy_.invalidate();
    }

    template <class Runtime>
    void reset(Runtime& runtime) noexcept {
        invalidate(runtime);
        bindings_ = {};
        policy_.reset();
    }

    [[nodiscard]] const WidthReplayPolicy& policy() const noexcept { return policy_; }

private:
    GraphRows layout_;
    WidthReplayPolicy policy_;
    std::array<std::optional<Binding>, WidthReplayPolicy::kWidths> bindings_{};
};

}  // namespace strix::exec
