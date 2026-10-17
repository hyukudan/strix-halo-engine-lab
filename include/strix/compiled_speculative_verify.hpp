#pragma once

#include "strix/execution_arena.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/execution_binding.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/execution_io.hpp"
#include "strix/kernel_registry.hpp"
#include "strix/graph_eligibility.hpp"
#include "strix/model_contract.hpp"
#include "strix/speculative_verify_binding.hpp"
#include "strix/speculative_verify_plan.hpp"
#include "strix/known_window_output_policy.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace strix {

enum class KnownWindowStepMode : std::uint8_t {
    r1_row_loop,
    r2_shared_q8,
    r2_gdn_staged,
    shared_rows,
    gdn_rows,
    routed_expert_union,
    hyper_mix_rows,
};

// Closed implementation table. These values identify window entries; they do
// not change any serial KernelVariant or claim hardware qualification.
enum class KnownWindowR2Variant : std::uint32_t {
    none = 0,
    shared_q8_virtual256_r2_v1 = 1,
    gdn_q8_virtual256_r2_v1 = 2,
    // Large projections share Q8 weights; both small F32 gates retain R1 math.
    gdn_q8_rows2_f32_gates_r1_v1 = 3,
    shared_q8_virtual256_r4_v1 = 4,
    shared_q8_virtual256_r8_v1 = 5,
    gdn_q8_rows4_f32_gates_r1_v1 = 6,
    gdn_q8_rows8_f32_gates_r1_v1 = 7,
    routed_q51_union_rows2_v1 = 8,
    routed_q51_union_rows4_v1 = 9,
    routed_q51_union_rows8_v1 = 10,
    // attn/mlp/final HC: typed prefetched Q8 path, R1 order per row.
    hyper_mix_rows2_v1 = 11,
    hyper_mix_rows4_v1 = 12,
    hyper_mix_rows8_v1 = 13,
};
inline constexpr std::uint32_t known_window_r2_variant_slots = 14;

struct KnownWindowRequest {
    bool r2_shared_expert = false;
    bool r2_gdn = false;
    bool r2_gdn_f32_gates = false;
    // Native shared and mixed-GDN rows4/8 when selected, all-Q8 GDN pair tiles and R1
    // fallback elsewhere. Does not register new serial kernel IDs.
    std::uint32_t logical_rows = 2;
    // Explicit grouped Q5_1 down candidate; unmatched formats retain R1.
    bool routed_expert_union = false;
    KnownWindowOutputPolicy output_policy = KnownWindowOutputPolicy::all_rows;
    // Multi-row HC mixers for the typed prefetched Q8 serial selection; others retain R1.
    bool rows_hyper_mix = false;
};

struct CompiledKnownPrefixWindow {
    std::uint64_t serial_execution_identity = 0;
    std::uint64_t serial_kernel_plan_identity = 0;
    std::uint64_t window_execution_identity = 0;
    std::uint64_t window_kernel_plan_identity = 0;
    std::uint64_t manifest_layout_identity = 0;
    std::uint64_t tensor_access_plan_identity = 0;
    KnownWindowArena arena;
    // Fresh launch plans/dispatchers must be built for these row plans. Their
    // kernel IDs and serial kernel-plan identity stay intact; arena and
    // execution identity belong to the complete window.
    std::vector<CompiledSerialExecution> row_plan;
    std::vector<KnownWindowStepMode> mode;
    std::vector<KnownWindowR2Variant> r2_variant;
    std::uint32_t r2_steps = 0;
    std::uint32_t row_loop_steps = 0;
    std::uint32_t runtime_row_steps = 0;
    // Logical repeated projection payload/launches removed by the closed R2
    // or wide mapping; neither counter measures DRAM traffic or elapsed time.
    std::uint64_t logical_projection_weight_bytes_avoided = 0;
    std::uint32_t projection_launches_saved = 0;
    std::uint32_t logical_rows = 2;
    std::uint32_t shared_rows_steps = 0;
    std::uint32_t gdn_rows_steps = 0;
    std::uint32_t routed_expert_union_steps = 0;
    std::uint32_t hyper_mix_rows_steps = 0;
    KnownWindowOutputPolicy output_policy = KnownWindowOutputPolicy::all_rows;
    // Compact result.logits entries correspond to these original source rows.
    std::vector<std::uint32_t> logit_row_indices;
};

// Complete wide entries do not contribute to the legacy paired-step counter.
// Admission must cover every closed mode without adding counters that could overflow.
[[nodiscard]] inline bool known_window_has_promoted_steps(const CompiledKnownPrefixWindow& window) noexcept {
    return window.r2_steps || window.shared_rows_steps || window.gdn_rows_steps || window.routed_expert_union_steps ||
        window.hyper_mix_rows_steps;
}

[[nodiscard]] CompiledKnownPrefixWindow compile_known_prefix_window(
    const ModelContract& model, const CompiledSerialExecution& serial,
    const ModelManifest& manifest, const ExecutionTensorAccessPlan& tensor_access,
    const KnownWindowRequest& request = {});

// Reconstruct row geometry and identities from the original serial source.
// Call before callbacks/enqueues: this is admission, not a per-step hot check.
void validate_compiled_known_prefix_window(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial);

// Bind each promoted numerical variant to the actual selected tensor formats.
void validate_known_prefix_window_formats(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ModelManifest& manifest, const ExecutionTensorAccessPlan& tensor_access);

struct CompiledSpeculativeVerifyExecution {
    SerialExecutionPlan topology;
    SpeculativeVerifyPlan verify;
    ExecutionBindingPlan bindings;
    SerialExecutionIoPlan io;
    ExecutionArenaPlan arena;

    std::uint64_t kernel_plan_identity = 0;
    std::uint64_t execution_identity = 0;

    std::uint64_t vocab_size = 0;
    std::uint32_t candidate_depth = 0;
    std::uint32_t request_concurrency = 1;
    std::uint32_t sequential_window_steps = 0;
    std::uint32_t causal_window_steps = 0;
    std::uint32_t dynamic_steps = 0;
    bool whole_plan_graph_safe = false;
};

[[nodiscard]] CompiledSpeculativeVerifyExecution
compile_speculative_verify_execution(
    const ModelContract& model,
    std::uint32_t candidate_depth,
    const kernels::KernelRegistry& registry,
    const SpeculativeVerifyBindingRequest& binding_request = {},
    const ExecutionBufferRequest& buffer_request = {},
    std::uint64_t arena_alignment = 256);

[[nodiscard]] GraphCapturePlan prepare_compiled_verify_graph_capture(
    const CompiledSpeculativeVerifyExecution& compiled, const ExecutionRuntimeResources& resources,
    const GraphDynamicArguments& arguments, const GraphMemoryRequest& memory);

[[nodiscard]] GraphEligibility assess_compiled_verify_graph_replay(
    const CompiledSpeculativeVerifyExecution& compiled, const GraphCapturePlan& capture,
    const ExecutionRuntimeResources& resources, const GraphDynamicArguments& arguments,
    std::uint64_t position);

}  // namespace strix
