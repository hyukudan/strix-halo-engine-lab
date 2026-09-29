#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>

namespace strix {

enum class SerialSessionHealth {
    ready,
    poisoned,
};

struct SerialSessionContext {
    std::uint64_t execution_identity = 0;
    std::uint64_t state_layout_identity = 0;
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    SerialSessionHealth health = SerialSessionHealth::ready;
    PersistentStateGenerations state;
};

[[nodiscard]] EagerExecutionResult execute_serial_token(
    const CompiledSerialExecution& compiled,
    SerialSessionContext& session,
    const ExecutionMemoryBinding& memory,
    const SerialSessionStateLayout& session_state_layout,
    const SessionStateMemoryBinding& session_state_memory,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher);

}  // namespace strix
