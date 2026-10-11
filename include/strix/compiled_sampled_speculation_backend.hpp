#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_resources.hpp"
#include "strix/serial_session.hpp"
#include "strix/speculation_sampling_executor.hpp"
#include "strix/speculative_verify_runtime.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

class TargetLogitsReader {
public:
    virtual ~TargetLogitsReader() = default;

    [[nodiscard]] virtual std::vector<float> read_fp32_logits(
        const ExecutionBufferAddress& buffer,
        std::uint32_t rows,
        std::uint64_t vocab_size) = 0;
};

class SampledTargetDistributionCollector {
public:
    virtual ~SampledTargetDistributionCollector() = default;

    [[nodiscard]] virtual std::vector<double>
    collect_base_target_distribution(
        const CompiledSerialExecution& serial,
        const ExecutionRuntimeResources& serial_resources,
        const SerialSessionContext& session) = 0;

    [[nodiscard]] virtual std::vector<std::vector<double>>
    collect_verified_target_distributions(
        const CompiledSpeculativeVerifyExecution& verify,
        const ExecutionRuntimeResources& verify_resources,
        const SerialSessionContext& session) = 0;
};

class Fp32SoftmaxTargetDistributionCollector final
    : public SampledTargetDistributionCollector {
public:
    explicit Fp32SoftmaxTargetDistributionCollector(
        TargetLogitsReader& reader)
        : reader_(reader) {}

    [[nodiscard]] std::vector<double>
    collect_base_target_distribution(
        const CompiledSerialExecution& serial,
        const ExecutionRuntimeResources& serial_resources,
        const SerialSessionContext& session) override;

    [[nodiscard]] std::vector<std::vector<double>>
    collect_verified_target_distributions(
        const CompiledSpeculativeVerifyExecution& verify,
        const ExecutionRuntimeResources& verify_resources,
        const SerialSessionContext& session) override;

private:
    TargetLogitsReader& reader_;
};

class CompiledSampledSpeculationBackend final
    : public SampledSpeculationExecutionBackend {
public:
    CompiledSampledSpeculationBackend(
        const CompiledSerialExecution& serial,
        const ExecutionRuntimeResources& serial_resources,
        EagerExecutionDispatcher& serial_dispatcher,
        const CompiledSpeculativeVerifyExecution& verify,
        const ExecutionRuntimeResources& verify_resources,
        EagerExecutionDispatcher& verify_dispatcher,
        SampledTargetDistributionCollector& collector);

    [[nodiscard]] std::vector<std::vector<double>>
    verify_target_distributions(
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
    SampledTargetDistributionCollector& collector_;
};

}  // namespace strix
