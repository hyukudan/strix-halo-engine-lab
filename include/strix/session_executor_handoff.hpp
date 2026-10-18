#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_resources.hpp"
#include "strix/serial_session.hpp"

namespace strix {

// One serial execution and the runtime bindings it runs with.
struct SessionExecutor {
    const CompiledSerialExecution* compiled = nullptr;
    const ExecutionRuntimeResources* resources = nullptr;
};

// Two serial executions are siblings when they differ only in arithmetic: the same steps,
// dependencies, state reads/writes, buffer IO, arena layout and geometry, with different
// kernel bindings and therefore different execution identities. A session state produced by
// one is a valid input state for the other (split prefill/decode numerical profiles).
void validate_sibling_serial_executions(const CompiledSerialExecution& a, const CompiledSerialExecution& b);

// The serial scratch binding of a sibling execution: the same arena allocation viewed under the
// sibling's execution identity. Siblings have byte-identical arena layouts and the two executions
// never run concurrently, so they share one scratch arena.
[[nodiscard]] ExecutionMemoryBinding sibling_scratch_binding(const CompiledSerialExecution& sibling,
    const CompiledSerialExecution& owner, const ExecutionMemoryBinding& owner_scratch);

// Moves a ready session from one sibling executor to the other. Only the executor identities
// change: positions, state generations and the arithmetic lineage (which keeps the producer
// identity of every row) are untouched. Both executors must bind the same model, tensor access,
// state layout/memory and PLE resources. Throws std::invalid_argument and leaves the session
// unchanged when any binding differs.
void hand_off_serial_session(SerialSessionContext& session, SessionExecutor from, SessionExecutor to);

} // namespace strix
