#pragma once

#include "strix/serial_request.hpp"
#include "strix/spec/arm_selector.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/row_rule.hpp"

namespace strix::spec {

// Run a fresh request's BlockSpeculator over a target with staged forward and
// prefix commit. The prompt is already in the target; initial_token is pending.
// The caller owns the measured verify costs and may supply a BlockDrafter with
// a ChainDrafter (e.g. MtpChain) without changing the verify target.
// A sampled RowRule decides every row with the serial draw at its position, so
// the emitted ids are those of serial sampled decode with the same policy and
// request identity; the target must support it (supports_row_rule(true)).
// An optional RoundSink (not owned) sees every verify round after its commit (BlockSpeculator::set_round_sink).
[[nodiscard]] SerialGreedyRequestResult execute_block_token_continuation(
    const SerialTokenContinuationRequest& request, std::span<const std::uint32_t> prompt,
    BlockDrafter& drafter, VerifyTarget& target, std::uint32_t max_rows,
    const std::function<void(std::uint32_t)>& emit = {}, const RowRule& rule = {}, RoundSink* sink = nullptr);
// The same continuation with several drafters under the shared per-regime selector (spec::ArmSelector): every arm
// sees the prompt and the anchor, the selector picks the drafter of each pass, the verify decides every id.
[[nodiscard]] SerialGreedyRequestResult execute_block_token_continuation(
    const SerialTokenContinuationRequest& request, std::span<const std::uint32_t> prompt,
    ArmSelector& arms, VerifyTarget& target, std::uint32_t max_rows,
    const std::function<void(std::uint32_t)>& emit = {}, const RowRule& rule = {}, RoundSink* sink = nullptr);

}  // namespace strix::spec
