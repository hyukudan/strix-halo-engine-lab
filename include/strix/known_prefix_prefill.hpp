#pragma once

#include "strix/compiled_speculative_verify.hpp"
#include "strix/speculative_verify_runtime.hpp"

namespace strix {
struct KnownPrefillRequest {
    std::uint32_t logical_rows=16;
    bool r2_shared_expert=false,r2_gdn=false,r2_gdn_f32_gates=false,routed_expert_union=false;
    KnownWindowOutputPolicy output_policy=KnownWindowOutputPolicy::all_rows;
    // Explicit declared storage limits, not inferred free/physical memory.
    // The native allocation consumer separately admits full owner growth.
    std::uint64_t workspace_budget_bytes=0,host_metadata_payload_budget_bytes=0;
    bool router_rows=false,combine_inject_rows=false;
    // Rows HC mixers and QSA rows in 8-row tiles.
    bool rows_hyper_mix=false,qsa_rows=false;
    // PLE rows from host-staged lookups (all T rows chained, see HipPleKnownWindow prefill).
    bool rows_ple=false;
};
struct CompiledKnownPrefixPrefill {
    std::uint32_t logical_rows=0;
    std::uint64_t prefill_execution_identity=0,prefill_kernel_plan_identity=0;
    KnownWindowArena arena;
    // Closed selection/format contract only. Its row plans are never relabelled
    // as a T-row window or executed as complete eight-row forwards.
    CompiledKnownPrefixWindow tile8;
    KnownWindowOutputPolicy output_policy=KnownWindowOutputPolicy::all_rows;
    std::vector<std::uint32_t> logit_row_indices;
    std::uint64_t workspace_budget_bytes=0,host_metadata_payload_budget_bytes=0;
    std::uint64_t host_metadata_payload_bytes=0;
};
[[nodiscard]] CompiledKnownPrefixPrefill compile_known_prefix_prefill(
    const ModelContract&,const CompiledSerialExecution&,const ModelManifest&,
    const ExecutionTensorAccessPlan&,const KnownPrefillRequest&);
void validate_compiled_known_prefix_prefill(const CompiledKnownPrefixPrefill&,
    const CompiledSerialExecution&);

// Whole T step-major execution. Every promoted callback still receives exactly
// its original closed 8/4/2 rows. Odd suffixes use full R1 launchers. All T-row
// preflight completes before PLE/mutation; contexts use fixed packs of eight.
// One completed transaction publishes T rows; no rollback is implied on error.
[[nodiscard]] KnownWindowResult execute_known_prefix_prefill_eager(
    const CompiledKnownPrefixPrefill&,const CompiledSerialExecution&,
    const ExecutionRuntimeResources&,const ExecutionMemoryBinding&,SerialSessionContext&,
    std::span<const std::uint32_t>,const CompiledLaunchPlan&,
    std::span<const KnownWindowR2Entry>,std::span<const KnownWindowRowsEntry>,
    const KnownWindowExecutionHooks&,
    // Optional entries that take all T rows of a step in one call (selected by the step's variant).
    std::span<const KnownWindowRowsEntry> full={});
} // namespace strix
