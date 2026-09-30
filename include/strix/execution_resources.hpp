#pragma once

#include "strix/execution_arena.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>

namespace strix {

struct ExecutionRuntimeResources {
    const ExecutionMemoryBinding* scratch = nullptr;
    const ModelManifest* model_manifest = nullptr;
    const ModelRuntimeBinding* model_weights = nullptr;
    const ExecutionTensorAccessPlan* tensor_access = nullptr;
    const SerialSessionStateLayout* session_state_layout = nullptr;
    const SessionStateMemoryBinding* session_state_memory = nullptr;
    PleLookupBackend* ple_lookup = nullptr;
};

struct RuntimeExecutionIdentities {
    std::uint64_t compatibility_identity = 0;
    std::uint64_t binding_identity = 0;
};

void validate_execution_runtime_resources(
    std::uint64_t compiled_execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionRuntimeResources& resources,
    std::uint64_t position);

[[nodiscard]] RuntimeExecutionIdentities
build_runtime_execution_identities(
    std::uint64_t compiled_execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionRuntimeResources& resources);

}  // namespace strix
