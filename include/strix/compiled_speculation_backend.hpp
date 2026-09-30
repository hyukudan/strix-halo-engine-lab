#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/serial_session.hpp"
#include "strix/speculation_executor.hpp"
#include "strix/speculative_verify_runtime.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

class GreedyTargetTokenCollector {
public:
    virtual ~GreedyTargetTokenCollector() = default;

    [[nodiscard]] virtual std::uint32_t collect_base_target_token(
        const CompiledSerialExecution& serial,
        const ExecutionRuntimeResources& serial_resources,
        const SerialSessionContext& session) = 0;

    [[nodiscard]] virtual std::vector<std::uint32_t>
    collect_verified_target_tokens(
        const CompiledSpeculativeVerifyExecution& verify,
        const ExecutionRuntimeResources& verify_resources,
        const SerialSessionContext& session) = 0;
};

class CompiledGreedySpeculationBackend final
    : public GreedySpeculationExecutionBackend {
public:
    CompiledGreedySpeculationBackend(
        const CompiledSerialExecution& serial,
        const ExecutionRuntimeResources& serial_resources,
        EagerExecutionDispatcher& serial_dispatcher,
        const CompiledSpeculativeVerifyExecution& verify,
        const ExecutionRuntimeResources& verify_resources,
        EagerExecutionDispatcher& verify_dispatcher,
        GreedyTargetTokenCollector& collector);

    [[nodiscard]] std::vector<std::uint32_t>
    verify_target_candidates(
        SerialSessionContext& session,
        const ExecutionRuntimeResources& resources,
        std::span<const std::uint32_t> candidate_tokens) override;

    void replay_target_token(
        SerialSessionContext& session,
        const ExecutionRuntimeResources& resources,
        std::uint32_t token) override;

private:
    void validate_shared_target_resources(
        const ExecutionRuntimeResources& caller_resources) const;

    const CompiledSerialExecution& serial_;
    const ExecutionRuntimeResources& serial_resources_;
    EagerExecutionDispatcher& serial_dispatcher_;

    const CompiledSpeculativeVerifyExecution& verify_;
    const ExecutionRuntimeResources& verify_resources_;
    EagerExecutionDispatcher& verify_dispatcher_;

    GreedyTargetTokenCollector& collector_;
};

}  // namespace strix
