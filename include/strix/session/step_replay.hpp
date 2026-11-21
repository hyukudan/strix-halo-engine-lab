#pragma once

#include <cstdint>
#include <type_traits>

namespace strix::session {

// One stream-ordered upload. Cache slots, relative offsets and split geometry
// are derived from the absolute position against the captured layout.
struct StepDescriptor {
    std::uint32_t position;
    std::uint32_t token;
};
static_assert(sizeof(StepDescriptor) == 8U && std::is_trivially_copyable_v<StepDescriptor>);

enum class StepReplayAction { eager, capture, replay };

// The owning session invalidates this state before replacing any captured
// allocation or rolling back. An unsupported capture stays eager until reset.
class StepReplayPolicy {
public:
    [[nodiscard]] StepReplayAction select(bool enabled, unsigned rows, bool decode, bool stable_layout) noexcept {
        if (!enabled || rows != 1U || !decode || !stable_layout || failed_) {
            invalidate();
            return StepReplayAction::eager;
        }
        return ready_ ? StepReplayAction::replay : StepReplayAction::capture;
    }
    void captured() noexcept { ready_ = true; }
    void invalidate() noexcept { ready_ = false; }
    void fail() noexcept { invalidate(); failed_ = true; }
    void reset() noexcept { *this = {}; }

private:
    bool ready_{};
    bool failed_{};
};

}  // namespace strix::session
