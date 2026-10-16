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

#include <array>
#include <cstdint>

namespace strix {

enum class KnownWindowStepMode : std::uint8_t {
    r1_row_loop,
    r2_shared_q8,
    r2_gdn_staged,
};

// Closed implementation table. These values identify a new R2 entry; they do
// not change any serial KernelVariant or claim hardware qualification.
enum class KnownWindowR2Variant : std::uint32_t {
    none = 0,
    shared_q8_virtual256_r2_v1 = 1,
    gdn_q8_virtual256_r2_v1 = 2,
};

struct KnownWindowRequest {
    bool r2_shared_expert = false;
    bool r2_gdn = false;
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
    std::array<CompiledSerialExecution, 2> row_plan;
    std::vector<KnownWindowStepMode> mode;
    std::vector<KnownWindowR2Variant> r2_variant;
    std::uint32_t r2_steps = 0;
    std::uint32_t row_loop_steps = 0;
    std::uint32_t runtime_row_steps = 0;
    // Logical repeated projection payload/launches removed by the closed R2
    // mapping; neither counter measures DRAM traffic or elapsed time.
    std::uint64_t logical_projection_weight_bytes_avoided = 0;
    std::uint32_t projection_launches_saved = 0;
};

[[nodiscard]] CompiledKnownPrefixWindow compile_known_prefix_window(
    const ModelContract& model, const CompiledSerialExecution& serial,
    const ModelManifest& manifest, const ExecutionTensorAccessPlan& tensor_access,
    const KnownWindowRequest& request = {});

// Reconstruct row geometry and identities from the original serial source.
// Call before callbacks/enqueues: this is admission, not a per-step hot check.
void validate_compiled_known_prefix_window(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial);

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
