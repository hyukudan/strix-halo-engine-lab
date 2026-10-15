#pragma once

#include "strix/arithmetic_lineage.hpp"

#include "strix/compiled_execution.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>
#include <functional>

namespace strix {

enum class SerialSessionHealth {
    ready,
    poisoned,
};

struct SerialSessionContext {
    std::uint64_t execution_identity = 0;
    std::uint64_t state_layout_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    SerialSessionHealth health = SerialSessionHealth::ready;
    PersistentStateGenerations state;
    ArithmeticLineage arithmetic_lineage;
};

// Optional native completion runs after dispatch and PLE consumption, inside
// the poisoning boundary and before state generations/lineage/token commit.
// Its owner must retain async resources and publish any shadow host history
// only after successful completion. An empty hook preserves eager semantics.
[[nodiscard]] EagerExecutionResult execute_serial_token(
    const CompiledSerialExecution& compiled,
    SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher,
    const std::function<void()>& before_commit = {});

[[nodiscard]] EagerExecutionResult execute_serial_token(
    const CompiledSerialExecution& compiled,
    SerialSessionContext& session,
    const ExecutionMemoryBinding& memory,
    const SerialSessionStateLayout& session_state_layout,
    const SessionStateMemoryBinding& session_state_memory,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher,
    const std::function<void()>& before_commit = {});

}  // namespace strix
