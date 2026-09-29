#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_runtime.hpp"

#include <cstdint>

namespace strix {

struct SerialSessionContext {
    std::uint64_t execution_identity = 0;
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    PersistentStateGenerations state;
};

[[nodiscard]] EagerExecutionResult execute_serial_token(
    const CompiledSerialExecution& compiled,
    SerialSessionContext& session,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher);

}  // namespace strix
