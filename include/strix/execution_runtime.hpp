#pragma once

#include "strix/execution_arena.hpp"
#include "strix/execution_binding.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace strix {

class PreparedTensorViews;

struct ExecutionDispatchContext {
    // Zero denotes an uncompiled execution and cannot use a compiled launcher.
    std::uint64_t execution_identity = 0;
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::optional<std::uint32_t> layer;
    std::optional<std::string_view> kernel_id;
    std::span<const ExecutionBufferUse> reads;
    std::span<const ExecutionBufferUse> writes;
    std::span<const ExecutionBufferUse> temporaries;
    const ExecutionArenaPlan* arena = nullptr;
    const ExecutionMemoryBinding* memory = nullptr;
    const SerialSessionStateLayout* session_state_layout = nullptr;
    const SessionStateMemoryBinding* session_state_memory = nullptr;
    const ExecutionRuntimeResources* resources = nullptr;
    // Published only after the immutable prepared plan admits this run. The
    // launcher borrows this snapshot; caller retains it through quiescence.
    const PreparedTensorViews* prepared_tensor_views = nullptr;
    PersistentState reads_state = PersistentState::none;
    PersistentState writes_state = PersistentState::none;
    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
    std::uint32_t input_token_id = 0;
    std::span<const std::uint32_t> input_tokens;
    std::uint64_t position = 0;
};

class EagerExecutionDispatcher {
public:
    virtual ~EagerExecutionDispatcher() = default;

    virtual void dispatch_kernel(
        const ExecutionDispatchContext& context) = 0;

    virtual void dispatch_runtime_step(
        const ExecutionDispatchContext& context) = 0;
};

struct PersistentStateGenerations {
    PersistentState valid = PersistentState::none;
    std::array<std::uint64_t, 6> generations{};

    [[nodiscard]] bool contains(
        PersistentState required) const noexcept;

    void publish_write(
        PersistentState written);
};

struct EagerExecutionRequest {
    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
    std::uint32_t input_token_id = 0;
    std::span<const std::uint32_t> input_tokens;
    std::uint64_t position = 0;

    // Callers explicitly declare which persistent-state objects have been
    // created for this request. Fresh zero/empty state still counts as valid.
    PersistentStateGenerations state;
};

struct EagerExecutionResult {
    std::uint32_t completed_steps = 0;
    std::uint32_t kernel_steps = 0;
    std::uint32_t runtime_steps = 0;
    PersistentStateGenerations state;
    std::vector<std::uint32_t> completion_order;
};

void validate_eager_execution_preflight(
    const SerialExecutionPlan& plan,
    const ExecutionBindingPlan& bindings,
    const SerialExecutionIoPlan& io_plan,
    const ExecutionArenaPlan& arena_plan,
    const EagerExecutionRequest& request,
    const SerialSessionStateLayout* session_state_layout = nullptr,
    const SessionStateMemoryBinding* session_state_memory = nullptr,
    const ExecutionRuntimeResources* resources = nullptr);

[[nodiscard]] EagerExecutionResult execute_serial_eager(
    const SerialExecutionPlan& plan,
    const ExecutionBindingPlan& bindings,
    const SerialExecutionIoPlan& io_plan,
    const ExecutionArenaPlan& arena_plan,
    const EagerExecutionRequest& request,
    EagerExecutionDispatcher& dispatcher,
    const ExecutionMemoryBinding* memory = nullptr,
    const SerialSessionStateLayout* session_state_layout = nullptr,
    const SessionStateMemoryBinding* session_state_memory = nullptr,
    const ExecutionRuntimeResources* resources = nullptr,
    std::uint64_t execution_identity = 0);

}  // namespace strix
