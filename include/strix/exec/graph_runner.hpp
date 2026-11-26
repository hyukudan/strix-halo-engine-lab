#pragma once

#include "strix/exec/width_replay.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace strix::exec {

// Captured device arguments can only name a different request via the uploaded block.
// Refuse a half-initialized mask before a kernel can read id zero or skip a state write.
inline void validate_graph_rows(const StepBlock& block) {
    if (block.h.rows == 0U || block.h.rows > kStepMaxRows)
        throw std::invalid_argument("invalid graph descriptor width");
    std::uint32_t valid = 0;
    for (std::uint32_t i = 0; i < block.h.rows; ++i)
        if (block.row[i].flags & step_valid) valid |= std::uint32_t{1} << i;
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
    template <class Runtime, class Stream, class Work>
    bool run(Runtime& runtime, Stream stream, bool enabled, bool stable, const StepBlock& block,
             const Binding& binding, Work& work) {
        const auto width = block.h.rows;
        const auto index = WidthReplayPolicy::index(width);
        if (index >= WidthReplayPolicy::kWidths) return false;
        if (enabled && stable) validate_graph_rows(block);
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
        if (action == WidthReplayAction::capture) {
            bindings_[index] = binding;
            if (!runtime.capture(stream, index, work)) {
                runtime.discard(index);
                policy_.fail(width);
                return false;
            }
            policy_.captured(width);
        }
        try {
            runtime.upload(stream, block);
            runtime.launch(stream, index);
        } catch (...) {
            runtime.discard(index);
            policy_.fail(width);
            throw;
        }
        return true;
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
    WidthReplayPolicy policy_;
    std::array<std::optional<Binding>, WidthReplayPolicy::kWidths> bindings_{};
};

}  // namespace strix::exec
