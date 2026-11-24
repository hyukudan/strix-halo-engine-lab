#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/exec/descriptor.hpp"
#include "strix/session/step_replay.hpp"

#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::inkling2 {

inline constexpr unsigned kMiniVerifyMaxRows = 16U;

[[nodiscard]] constexpr bool mini_verify_width(unsigned rows) noexcept {
    return rows <= kMiniVerifyMaxRows && session::is_step_replay_width(rows);
}

// Session geometry of a request that may stage verify windows of up to verify_rows rows on the step program: the
// shared rule (session::step_replay_session_geometry) at Inkling's graph-width cap and context. Both the serving
// decoder and the --verify-rows CLI build their resident session from it.
[[nodiscard]] constexpr session::StepReplaySessionGeometry mini_step_session_geometry(
    unsigned positions, unsigned chunk_rows, unsigned verify_rows) noexcept {
    return session::step_replay_session_geometry(positions, chunk_rows,
                                                 verify_rows < kMiniVerifyMaxRows ? verify_rows : kMiniVerifyMaxRows,
                                                 kMaxContext);
}

// The plain verify ticket targets the existing resident slot. A token is
// forced for every active row, so the program has no device-fed rows.
[[nodiscard]] inline exec::StepBlock mini_verify_descriptor(unsigned width, std::span<const unsigned> ids,
                                                            unsigned first_position) {
    if (!mini_verify_width(width) || ids.empty() || ids.size() > width || first_position >= kMaxContext ||
        ids.size() > kMaxContext - first_position)
        throw std::invalid_argument("invalid Inkling verify width, row count or position");

    exec::StepBlock block{};
    block.h.rows = width;
    block.h.step = first_position;
    for (unsigned row = 0; row < ids.size(); ++row) {
        if (ids[row] >= kVocabulary) throw std::invalid_argument("Inkling verify token out of range");
        block.h.valid_mask |= 1U << row;
        block.row[row] = {ids[row], first_position + row, 0U, row == 0U ? exec::kStepNoParent : row - 1U,
                          block.h.valid_mask, exec::step_valid, 0U, 0U};
    }
    return block;
}

enum class MiniVerifyMissing : unsigned {
    dense, controller, residual_mix, bank_norm, convolution, head_clip, relative, kv, attention, finite,
    route, gate_up, down_combine, argmax, embedding, rms, add, round, sigmoid, softcap, draft_tap, count
};

inline constexpr std::array<std::string_view, static_cast<unsigned>(MiniVerifyMissing::count)> kMiniVerifyReasons{
    "dense (width-bound exact BF16 projections)",
    "controller (width-bound low-rank controller)",
    "residual_read (width-bound residual mix)",
    "message_bank (width-bound bank normalization)",
    "causal_convolution (slot- and parent-indexed history)",
    "head_clip (per-row descriptor positions)",
    "relative_logits (width-bound relative projection)",
    "kv_append (descriptor positions and StateTable slots)",
    "prepared_attention (descriptor slots and ancestor mask)",
    "prepared_finite (valid-mask-aware row validation)",
    "checked_wave_route (width-bound route and row status)",
    "packed_gate_up (exact NVFP4/MXFP8 rows)",
    "packed_down_combine (exact NVFP4/MXFP8 rows)",
    "greedy_head (width-bound row picks and masked status)",
    "embedding (descriptor tokens)",
    "rms (descriptor rows)",
    "residual_add (descriptor rows)",
    "bf16_round (descriptor rows)",
    "sigmoid_product (descriptor rows)",
    "logit_softcap (descriptor rows)",
    "draft_tap (descriptor-positioned BF16 row store)",
};

[[nodiscard]] constexpr unsigned mini_verify_bit(MiniVerifyMissing reason) noexcept {
    return 1U << static_cast<unsigned>(reason);
}

inline void require_mini_verify_leaves(unsigned rows, unsigned missing) {
    if (!mini_verify_width(rows)) throw std::invalid_argument("Inkling --verify-rows requires 1, 2, 4, 8 or 16");
    if (missing == 0U) return;
    std::string message = "Inkling verify R=" + std::to_string(rows) + " needs shared CUDA leaves: ";
    for (unsigned i = 0; i < kMiniVerifyReasons.size(); ++i) {
        if ((missing & (1U << i)) == 0U) continue;
        message += kMiniVerifyReasons[i];
        missing &= ~(1U << i);
        if (missing != 0U) message += ", ";
    }
    throw std::invalid_argument(message);
}

inline void require_mini_verify_width(unsigned rows) {
    require_mini_verify_leaves(rows, 0U);
}

}  // namespace strix::inkling2
