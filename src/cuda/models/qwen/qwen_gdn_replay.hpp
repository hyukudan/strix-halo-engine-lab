#pragma once

#include "ops/recur/recur_decode_step.hpp"

#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::cuda::qwen_gdn {

inline constexpr std::uint32_t kMaxReplayRows = 8;

[[nodiscard]] constexpr ops::recur::GdnDecodeShape replay_shape_for(ops::recur::GdnDecodeShape configured) {
    constexpr auto supported = ops::recur::shape::gdn_2560_16x48x128;
    if (configured != supported)
        throw std::invalid_argument("Qwen GDN replay: unsupported shape (requires gdn_2560_16x48x128)");
    return supported;
}

// A single-session window uses transient slot 0. The host replaces this
// packet before each eager submission or graph launch, including parity flips.
struct ReplayWindow {
    exec::StepBlock descriptor;
    state::SlotEntry published[2]; // taps, matrix
    std::uint16_t* pending_history;
    float* pending_matrix;
};

[[nodiscard]] inline ReplayWindow stage_replay_window(std::span<const std::uint32_t> tokens,
    std::uint32_t position, std::uint32_t capacity, const std::uint16_t* history, const float* matrix,
    std::uint16_t* pending_history, float* pending_matrix) noexcept {
    // verify_cohort has checked the position range and the replay route
    // admits one through kMaxReplayRows tokens.
    ReplayWindow window{};
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    window.descriptor.h.rows = std::bit_ceil(rows);
    window.descriptor.h.valid_mask = (1U << rows) - 1U;
    for (std::uint32_t r = 0; r < rows; ++r) {
        window.descriptor.row[r] = {tokens[r], position + r, 0, r == 0 ? exec::kStepNoParent : r - 1U,
            (2U << r) - 1U, exec::step_valid | exec::step_pending_state, 0, 0};
    }
    window.published[0] = {reinterpret_cast<std::uintptr_t>(history), position, rows, capacity, 0};
    window.published[1] = {reinterpret_cast<std::uintptr_t>(matrix), position, rows, capacity, 0};
    window.pending_history = pending_history;
    window.pending_matrix = pending_matrix;
    return window;
}

[[nodiscard]] inline ops::recur::DecodeReplayStep bind_replay_layer(ops::recur::GdnDecodeShape shape,
    ReplayWindow& device, ops::recur::DecodeRow rows, ops::recur::DecodeRowStrides strides, std::uint32_t layer,
    RecurrentStateType type, ops::recur::DecodeA8 a8 = {}) noexcept {
    const state::StateTable table{device.published, 2, 1};
    const auto history_offset = std::uint64_t{layer} * shape.history_elements() * sizeof(std::uint16_t);
    const auto matrix_offset = std::uint64_t{layer} * shape.matrix_bytes(type);
    return {&device.descriptor, rows, strides, {table, 0}, {table, 1},
        {nullptr, &device.pending_history, history_offset}, {nullptr, &device.pending_matrix, matrix_offset},
        type, a8, history_offset, matrix_offset};
}

}  // namespace strix::cuda::qwen_gdn
