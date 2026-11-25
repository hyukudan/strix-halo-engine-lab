#pragma once

#include "strix/exec/descriptor.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix::session {

// Row convention of a linear (chain) verify window. Both conventions chain row i to parent i - 1 at
// position first + i; they differ in what a row may see in-step and in how its state write lands.
enum class LinearVerifyRows : std::uint8_t {
    // Each row sees only the rows before it (ancestors exclusive of itself) and its state write is
    // staged until acceptance (step_valid | step_pending_state). The draft-verify runner's window.
    exclusive_pending,
    // Each row sees itself and the rows before it (ancestors inclusive of itself) and is a plain valid
    // row (step_valid). A consumer that resolves the current row's own K/V through the in-step
    // ancestor mask (decode_state_rows.hpp pending_row) needs this convention.
    inclusive_valid,
};

struct LinearVerifyWindow {
    std::uint32_t width;           // Program rows, including masked padding rows.
    std::uint32_t first_position;  // Absolute position of row 0.
    std::uint32_t slot;            // StateTable session lane of every row.
    std::uint32_t step;            // Header step word.
    LinearVerifyRows rows;
};

// One linear verify descriptor: ids.size() valid rows, then masked padding up to the program width.
// The caller owns its model limits (token range, context, admitted widths); this builder only refuses
// a window the descriptor cannot express.
[[nodiscard]] constexpr exec::StepBlock linear_verify_step(const LinearVerifyWindow& window,
                                                           std::span<const std::uint32_t> ids) {
    const auto valid = ids.size();
    if (valid == 0U || valid > window.width || window.width > exec::kStepMaxRows ||
        window.first_position > std::numeric_limits<std::uint32_t>::max() - (valid - 1U))
        throw std::invalid_argument("linear verify window exceeds its width or position room");
    const bool inclusive = window.rows == LinearVerifyRows::inclusive_valid;
    const std::uint32_t flags = inclusive ? exec::step_valid : exec::step_valid | exec::step_pending_state;
    exec::StepBlock step{};
    step.h.rows = window.width;
    step.h.step = window.step;
    for (std::uint32_t i = 0; i < valid; ++i) {
        const auto earlier = (std::uint32_t{1} << i) - 1U;
        step.row[i] = {ids[i], window.first_position + i, window.slot, i ? i - 1U : exec::kStepNoParent,
                       inclusive ? earlier | (std::uint32_t{1} << i) : earlier, flags, 0U, 0U};
        step.h.valid_mask |= std::uint32_t{1} << i;
    }
    return step;
}

}  // namespace strix::session
