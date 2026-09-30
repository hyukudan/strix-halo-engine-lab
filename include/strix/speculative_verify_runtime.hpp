#pragma once

#include "strix/compiled_speculative_verify.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/serial_session.hpp"

#include <cstdint>
#include <span>

namespace strix {

struct SpeculativeVerifyExecutionResult {
    EagerExecutionResult execution;
    std::uint64_t base_position = 0;
    std::uint64_t post_position = 0;
    std::uint64_t base_committed_tokens = 0;
    std::uint64_t post_committed_tokens = 0;
};

[[nodiscard]] SpeculativeVerifyExecutionResult
execute_compiled_speculative_verify_eager(
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    std::span<const std::uint32_t> candidate_tokens,
    EagerExecutionDispatcher& dispatcher);

}  // namespace strix
