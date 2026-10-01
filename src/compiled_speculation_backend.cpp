#include "strix/compiled_speculation_backend.hpp"

#include <stdexcept>
#include <string>

namespace strix {
namespace {

bool same_state_memory(
    const SessionStateMemoryBinding& left,
    const SessionStateMemoryBinding& right) noexcept {
    return left.layout_identity ==
               right.layout_identity &&
           left.device_base ==
               right.device_base &&
           left.device_bytes ==
               right.device_bytes &&
           left.device_allocation_generation ==
               right.device_allocation_generation &&
           left.host_base ==
               right.host_base &&
           left.host_bytes ==
               right.host_bytes &&
           left.host_allocation_generation ==
               right.host_allocation_generation;
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

}  // namespace

CompiledGreedySpeculationBackend::
CompiledGreedySpeculationBackend(
    const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    EagerExecutionDispatcher& serial_dispatcher,
    const CompiledSpeculativeVerifyExecution& verify,
    const ExecutionRuntimeResources& verify_resources,
    EagerExecutionDispatcher& verify_dispatcher,
    GreedyTargetTokenCollector& collector)
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

    validate_shared_target_resources(
        serial_resources_);
}

void CompiledGreedySpeculationBackend::
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
            "serial/verify must operate on the same persistent target-state allocation");
    }
}

std::vector<std::uint32_t>
CompiledGreedySpeculationBackend::
verify_target_candidates(
    SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,
    std::span<const std::uint32_t> candidate_tokens) {
    validate_shared_target_resources(
        resources);

    if (candidate_tokens.size() !=
        verify_.candidate_depth) {
        throw std::invalid_argument(
            "candidate depth does not match compiled verify backend");
    }

    const auto base_token =
        collector_.collect_base_target_token(
            serial_,
            serial_resources_,
            session);
    if (serial_.vocab_size == 0 ||
        static_cast<std::uint64_t>(base_token) >=
            serial_.vocab_size) {
        throw std::runtime_error(
            "base target token is outside compiled vocabulary");
    }

    static_cast<void>(
        execute_compiled_speculative_verify_eager(
            verify_,
            verify_resources_,
            session,
            candidate_tokens,
            verify_dispatcher_));

    auto suffix =
        collector_.collect_verified_target_tokens(
            verify_,
            verify_resources_,
            session);

    if (suffix.size() !=
        verify_.candidate_depth) {
        throw std::runtime_error(
            "verified target token collector returned wrong row count");
    }
    for (const auto token : suffix) {
        if (serial_.vocab_size == 0 ||
            static_cast<std::uint64_t>(token) >=
                serial_.vocab_size) {
            throw std::runtime_error(
                "verified target token is outside compiled vocabulary");
        }
    }

    std::vector<std::uint32_t> target_tokens;
    target_tokens.reserve(
        suffix.size() + 1U);
    target_tokens.push_back(
        base_token);
    target_tokens.insert(
        target_tokens.end(),
        suffix.begin(),
        suffix.end());
    return target_tokens;
}

void CompiledGreedySpeculationBackend::
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
