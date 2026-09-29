#pragma once

#include "strix/execution_binding.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_plan.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace strix {

struct ExecutionDispatchContext {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::optional<std::uint32_t> layer;
    std::optional<std::string_view> kernel_id;
    std::span<const ExecutionBufferUse> reads;
    std::span<const ExecutionBufferUse> writes;
    PersistentState reads_state = PersistentState::none;
    PersistentState writes_state = PersistentState::none;
    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
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
    const EagerExecutionRequest& request);

[[nodiscard]] EagerExecutionResult execute_serial_eager(
    const SerialExecutionPlan& plan,
    const ExecutionBindingPlan& bindings,
    const SerialExecutionIoPlan& io_plan,
    const EagerExecutionRequest& request,
    EagerExecutionDispatcher& dispatcher);

}  // namespace strix
