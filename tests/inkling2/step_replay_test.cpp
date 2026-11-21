#include "attn_core/dense_flash_keysplit.hpp"
#include "strix/session/step_replay.hpp"

int main() {
    using strix::attn_core::dense::small_row_key_splits;
    if (small_row_key_splits(127U, 1U, 0U) != 1U ||
        small_row_key_splits(128U, 1U, 0U) != 2U ||
        small_row_key_splits(2048U, 1U, 1024U) != 8U ||
        small_row_key_splits(2048U, 1U, 0U) != 16U) return 11;
    using strix::session::StepReplayAction;
    strix::session::StepReplayPolicy policy;
    const auto next = [&] { return policy.select(true, 1U, true, true); };
    if (next() != StepReplayAction::capture) return 1;
    policy.captured();
    if (next() != StepReplayAction::replay) return 2;
    if (policy.select(true, 2U, true, true) != StepReplayAction::eager ||
        next() != StepReplayAction::capture) return 3;
    policy.captured();
    if (policy.select(true, 1U, false, true) != StepReplayAction::eager ||
        next() != StepReplayAction::capture) return 4;
    policy.captured();
    if (policy.select(true, 1U, true, false) != StepReplayAction::eager ||
        next() != StepReplayAction::capture) return 5;
    policy.captured();
    if (policy.select(false, 1U, true, true) != StepReplayAction::eager ||
        next() != StepReplayAction::capture) return 6;
    policy.captured();
    policy.invalidate(); // KV restore
    if (next() != StepReplayAction::capture) return 7;
    policy.fail();
    policy.invalidate();
    if (next() != StepReplayAction::eager) return 8;
    policy.reset();
    if (next() != StepReplayAction::capture) return 9;
    policy.captured();
    policy.reset();
    return next() == StepReplayAction::capture ? 0 : 10;
}
