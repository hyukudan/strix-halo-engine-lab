#include "strix/compiled_sampled_speculation_backend.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {

bool same_state_memory(
    const SessionStateMemoryBinding& left,
    const SessionStateMemoryBinding& right) noexcept {
    return left.layout_identity == right.layout_identity &&
           left.device_base == right.device_base &&
           left.device_bytes == right.device_bytes &&
           left.device_allocation_generation == right.device_allocation_generation &&
           left.host_base == right.host_base &&
           left.host_bytes == right.host_bytes &&
           left.host_allocation_generation == right.host_allocation_generation;
}

void require_complete(
    const ExecutionRuntimeResources& resources,
    const char* name) {
    if (resources.scratch == nullptr ||
        resources.model_manifest == nullptr ||
        resources.model_weights == nullptr ||
        resources.tensor_access == nullptr ||
        resources.session_state_layout == nullptr ||
        resources.session_state_memory == nullptr) {
        throw std::invalid_argument(
            std::string(name) +
            " runtime resources are incomplete");
    }
}

ExecutionBufferAddress logits_buffer(
    std::uint64_t execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionMemoryBinding& memory,
    const SerialExecutionIoPlan& io) {
    if (io.steps.empty()) {
        throw std::invalid_argument(
            "target distribution collector requires execution IO");
    }
    const auto& last = io.steps.back();
    if (last.op != SerialOp::lm_head ||
        last.writes.size() != 1U ||
        last.writes[0].buffer.buffer_class !=
            ExecutionBufferClass::logits) {
        throw std::invalid_argument(
            "target distribution collector requires final LM-head logits write");
    }
    return resolve_execution_buffer(
        execution_identity,
        arena,
        memory,
        last.writes[0]);
}

std::vector<std::vector<double>> softmax_rows(
    std::span<const float> logits,
    std::uint32_t rows,
    std::uint64_t vocab_size) {
    if (rows == 0 ||
        vocab_size == 0 ||
        vocab_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(
            "target logits geometry is invalid");
    }
    const auto vocab =
        static_cast<std::size_t>(
            vocab_size);
    if (vocab >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(rows) ||
        logits.size() !=
            vocab *
            static_cast<std::size_t>(rows)) {
        throw std::invalid_argument(
            "target logits payload does not match rows/vocabulary");
    }

    std::vector<std::vector<double>> result;
    result.reserve(rows);
    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        const auto offset =
            static_cast<std::size_t>(row) *
            vocab;
        float maximum =
            -std::numeric_limits<float>::infinity();
        for (std::size_t index = 0;
             index < vocab;
             ++index) {
            const auto value =
                logits[offset + index];
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "target logits must be finite");
            }
            if (value > maximum) {
                maximum = value;
            }
        }

        std::vector<double> probabilities(
            vocab);
        long double total = 0.0L;
        for (std::size_t index = 0;
             index < vocab;
             ++index) {
            const auto value =
                std::exp(
                    static_cast<double>(
                        logits[offset + index] -
                        maximum));
            probabilities[index] = value;
            total +=
                static_cast<long double>(
                    value);
        }
        if (!(total > 0.0L) ||
            !std::isfinite(
                static_cast<double>(total))) {
            throw std::runtime_error(
                "target softmax produced invalid probability mass");
        }
        for (auto& value :
             probabilities) {
            value =
                static_cast<double>(
                    static_cast<long double>(value) /
                    total);
        }
        result.push_back(
            std::move(probabilities));
    }
    return result;
}

void validate_distribution(
    std::span<const double> probabilities,
    std::uint64_t vocab_size,
    const char* name) {
    if (probabilities.size() !=
        static_cast<std::size_t>(
            vocab_size)) {
        throw std::runtime_error(
            std::string(name) +
            " distribution has wrong vocabulary size");
    }
    long double mass = 0.0L;
    for (const auto value :
         probabilities) {
        if (!std::isfinite(value) ||
            value < 0.0) {
            throw std::runtime_error(
                std::string(name) +
                " distribution contains invalid probability");
        }
        mass +=
            static_cast<long double>(
                value);
    }
    if (!(mass > 0.0L)) {
        throw std::runtime_error(
            std::string(name) +
            " distribution has zero mass");
    }
}

}  // namespace

std::vector<double>
Fp32SoftmaxTargetDistributionCollector::
collect_base_target_distribution(
    const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    const SerialSessionContext& session) {
    if (session.next_position == 0 ||
        serial.vocab_size == 0 ||
        serial_resources.scratch == nullptr) {
        throw std::invalid_argument(
            "base target distribution requires materialized serial logits");
    }
    const auto buffer =
        logits_buffer(
            serial.execution_identity,
            serial.arena,
            *serial_resources.scratch,
            serial.io);
    auto logits =
        reader_.read_fp32_logits(
            buffer,
            1,
            serial.vocab_size);
    auto rows =
        softmax_rows(
            logits,
            1,
            serial.vocab_size);
    return std::move(rows.front());
}

std::vector<std::vector<double>>
Fp32SoftmaxTargetDistributionCollector::
collect_verified_target_distributions(
    const CompiledSpeculativeVerifyExecution& verify,
    const ExecutionRuntimeResources& verify_resources,
    const SerialSessionContext&) {
    if (verify.candidate_depth == 0 ||
        verify.vocab_size == 0 ||
        verify_resources.scratch == nullptr) {
        throw std::invalid_argument(
            "verified target distributions require initialized verify resources");
    }
    const auto buffer =
        logits_buffer(
            verify.execution_identity,
            verify.arena,
            *verify_resources.scratch,
            verify.io);
    auto logits =
        reader_.read_fp32_logits(
            buffer,
            verify.candidate_depth,
            verify.vocab_size);
    return softmax_rows(
        logits,
        verify.candidate_depth,
        verify.vocab_size);
}

CompiledSampledSpeculationBackend::
CompiledSampledSpeculationBackend(
    const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    EagerExecutionDispatcher& serial_dispatcher,
    const CompiledSpeculativeVerifyExecution& verify,
    const ExecutionRuntimeResources& verify_resources,
    EagerExecutionDispatcher& verify_dispatcher,
    SampledTargetDistributionCollector& collector)
    : serial_(serial),
      serial_resources_(serial_resources),
      serial_dispatcher_(serial_dispatcher),
      verify_(verify),
      verify_resources_(verify_resources),
      verify_dispatcher_(verify_dispatcher),
      collector_(collector) {
    require_complete(
        serial_resources_,
        "serial");
    require_complete(
        verify_resources_,
        "verify");

    validate_execution_runtime_resources(
        serial_.execution_identity,
        serial_.arena,
        serial_resources_,
        0);
    validate_execution_runtime_resources(
        verify_.execution_identity,
        verify_.arena,
        verify_resources_,
        0);
    if (serial_.vocab_size == 0 ||
        serial_.vocab_size !=
            verify_.vocab_size) {
        throw std::invalid_argument(
            "serial/verify vocabularies differ");
    }
    validate_shared_target_resources(
        serial_resources_);
}

void CompiledSampledSpeculationBackend::
validate_shared_target_resources(
    const ExecutionRuntimeResources& caller_resources) const {
    require_complete(
        caller_resources,
        "caller");

    const auto& serial_manifest =
        *serial_resources_.model_manifest;
    const auto& verify_manifest =
        *verify_resources_.model_manifest;
    const auto& caller_manifest =
        *caller_resources.model_manifest;

    if (serial_manifest.layout_identity !=
            verify_manifest.layout_identity ||
        serial_manifest.layout_identity !=
            caller_manifest.layout_identity) {
        throw std::invalid_argument(
            "serial/verify model manifest layouts differ");
    }
    if (serial_resources_.model_weights->model_content_identity !=
            verify_resources_.model_weights->model_content_identity ||
        serial_resources_.model_weights->model_content_identity !=
            caller_resources.model_weights->model_content_identity) {
        throw std::invalid_argument(
            "serial/verify model content identities differ");
    }
    if (serial_resources_.session_state_layout->layout_identity !=
            verify_resources_.session_state_layout->layout_identity ||
        serial_resources_.session_state_layout->layout_identity !=
            caller_resources.session_state_layout->layout_identity) {
        throw std::invalid_argument(
            "serial/verify state layouts differ");
    }
    if (!same_state_memory(
            *serial_resources_.session_state_memory,
            *verify_resources_.session_state_memory) ||
        !same_state_memory(
            *serial_resources_.session_state_memory,
            *caller_resources.session_state_memory)) {
        throw std::invalid_argument(
            "serial/verify must share persistent target-state allocation");
    }
}

std::vector<std::vector<double>>
CompiledSampledSpeculationBackend::
verify_target_distributions(
    SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,
    std::span<const std::uint32_t> candidate_tokens) {
    validate_shared_target_resources(
        resources);

    auto base =
        collector_.collect_base_target_distribution(
            serial_,
            serial_resources_,
            session);
    validate_distribution(
        base,
        serial_.vocab_size,
        "base target");

    if (candidate_tokens.empty()) {
        return {std::move(base)};
    }
    if (candidate_tokens.size() !=
        verify_.candidate_depth) {
        throw std::invalid_argument(
            "candidate depth does not match compiled sampled verify backend");
    }

    static_cast<void>(
        execute_compiled_speculative_verify_eager(
            verify_,
            verify_resources_,
            session,
            candidate_tokens,
            verify_dispatcher_));

    auto suffix =
        collector_.collect_verified_target_distributions(
            verify_,
            verify_resources_,
            session);
    if (suffix.size() !=
        verify_.candidate_depth) {
        throw std::runtime_error(
            "verified distribution collector returned wrong row count");
    }
    for (const auto& distribution :
         suffix) {
        validate_distribution(
            distribution,
            serial_.vocab_size,
            "verified target");
    }

    std::vector<std::vector<double>> result;
    result.reserve(
        suffix.size() + 1U);
    result.push_back(
        std::move(base));
    for (auto& distribution :
         suffix) {
        result.push_back(
            std::move(distribution));
    }
    return result;
}

void CompiledSampledSpeculationBackend::
replay_target_token(
    SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,
    std::uint32_t token) {
    validate_shared_target_resources(
        resources);
    static_cast<void>(
        execute_serial_token(
            serial_,
            session,
            serial_resources_,
            token,
            serial_dispatcher_));
}

}  // namespace strix
