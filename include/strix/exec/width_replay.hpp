#pragma once

#include "strix/exec/descriptor.hpp"

#include <array>
#include <cstdint>

namespace strix::exec {

enum class WidthReplayAction { eager, capture, replay };

// Replay state of the multi-request decode programs, one per program width in [1, kStepMaxRows]
// (a width class, or an exact request count). Each width captures once and then replays; a failed capture keeps
// only that width eager until reset. The owner invalidates every width before it replaces a
// captured allocation (workspace, slot directory, device step block) or changes a host-side
// route the capture baked in.
class WidthReplayPolicy {
public:
    static constexpr std::uint32_t kWidths = kStepMaxRows;

    [[nodiscard]] static constexpr std::uint32_t index(std::uint32_t width) noexcept {
        return width != 0U && width <= kWidths ? width - 1U : kWidths;
    }

    [[nodiscard]] WidthReplayAction select(bool enabled, std::uint32_t width, bool stable_layout) noexcept {
        const auto at = index(width);
        if (at >= kWidths) return WidthReplayAction::eager;
        auto& state = states_[at];
        if (!enabled || !stable_layout || state.failed) {
            state.ready = false;
            return WidthReplayAction::eager;
        }
        return state.ready ? WidthReplayAction::replay : WidthReplayAction::capture;
    }

    void captured(std::uint32_t width) noexcept {
        if (const auto at = index(width); at < kWidths) states_[at].ready = true;
    }
    void fail(std::uint32_t width) noexcept {
        if (const auto at = index(width); at < kWidths) states_[at] = {false, true};
    }
    // A different baked binding is a new candidate even after a prior capture decline.
    void invalidate_width(std::uint32_t width) noexcept {
        if (const auto at = index(width); at < kWidths) states_[at] = {};
    }
    void invalidate() noexcept {
        for (auto& state : states_) state.ready = false;
    }
    void reset() noexcept { states_ = {}; }

    [[nodiscard]] bool ready(std::uint32_t width) const noexcept {
        const auto at = index(width);
        return at < kWidths && states_[at].ready;
    }
    [[nodiscard]] bool failed(std::uint32_t width) const noexcept {
        const auto at = index(width);
        return at < kWidths && states_[at].failed;
    }

private:
    struct State {
        bool ready = false;
        bool failed = false;
    };
    std::array<State, kWidths> states_{};
};

}  // namespace strix::exec
