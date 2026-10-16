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
class DistributedStageResourcePlan;

struct ExecutionRuntimeResources {
    const ExecutionMemoryBinding* scratch = nullptr;
    const ModelManifest* model_manifest = nullptr;
    const ModelRuntimeBinding* model_weights = nullptr;
    const ExecutionTensorAccessPlan* tensor_access = nullptr;
    const SerialSessionStateLayout* session_state_layout = nullptr;
    const SessionStateMemoryBinding* session_state_memory = nullptr;
    PleLookupBackend* ple_lookup = nullptr;
    const PleHashContract* ple_hash_contract = nullptr;
    PleLookupPrefetchState* ple_prefetch = nullptr;
    // Explicit capacity-only materialization scope; ordinary execution rejects it.
    const DistributedStageResourcePlan* distributed_stage = nullptr;
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

// Explicit scoped admissions. Full-model entrypoints above remain strict.
void validate_stage_execution_runtime_resources(std::uint64_t,const ExecutionArenaPlan&,
    const ExecutionRuntimeResources&,std::uint64_t position);
[[nodiscard]] RuntimeExecutionIdentities build_stage_runtime_execution_identities(std::uint64_t,
    const ExecutionArenaPlan&,const ExecutionRuntimeResources&);
}  // namespace strix
