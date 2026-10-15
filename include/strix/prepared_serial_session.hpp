#pragma once

#include "strix/prepared_execution.hpp"
#include "strix/serial_session.hpp"

namespace strix {
// Experimental prepared submission, with the original C1 session transaction.
// Retain prepared and all resource owners through device quiescence, including
// after failure. The returned order view belongs to prepared, not to the GPU.
[[nodiscard]] PreparedEagerExecutionResult execute_prepared_serial_token(
    const PreparedEagerExecution& prepared,
    SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher,
    const std::function<void()>& before_commit = {});
} // namespace strix
