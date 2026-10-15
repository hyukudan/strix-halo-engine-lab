#pragma once

#include "strix/compiled_execution.hpp"

#include <memory>
#include <optional>
#include <span>

namespace strix {
struct PreparedEagerExecutionResult {
    std::uint32_t completed_steps = 0;
    std::uint32_t kernel_steps = 0;
    std::uint32_t runtime_steps = 0;
    PersistentStateGenerations state;
    // Immutable prepared submission order, not evidence of GPU completion.
    // Valid across subsequent runs until the PreparedEagerExecution is destroyed.
    std::span<const std::uint32_t> completion_order;
};

// Bounded eager preparation: privately owns an immutable deep copy of the
// compiled DAG/IO/arena and its validated submission order. It caches neither
// tensor addresses nor launcher functions. Request spans are never retained.
//
// actual_owner must retain the actual prepared allocations. The caller must
// additionally retain every borrowed resource/dispatcher and exclude mutation
// until device quiescence. A returned dispatch is not a completion fence; the
// session owner remains responsible for drain, commit and failure poisoning.
class PreparedEagerExecution final {
public:
    PreparedEagerExecution(CompiledSerialExecution compiled,
        const ExecutionRuntimeResources& prepare_resources,
        const EagerExecutionRequest& prototype,
        std::shared_ptr<const void> actual_owner);
    ~PreparedEagerExecution();
    PreparedEagerExecution(const PreparedEagerExecution&) = delete;
    PreparedEagerExecution& operator=(const PreparedEagerExecution&) = delete;
    PreparedEagerExecution(PreparedEagerExecution&&) = delete;
    PreparedEagerExecution& operator=(PreparedEagerExecution&&) = delete;

    // Successful runs allocate no result/order/completed-vector storage. Dynamic
    // resource validation and each launcher's own validation remain in place.
    [[nodiscard]] PreparedEagerExecutionResult run(
        const ExecutionRuntimeResources& resources,
        const EagerExecutionRequest& request,
        EagerExecutionDispatcher& dispatcher) const;
    [[nodiscard]] const CompiledSerialExecution& compiled() const noexcept;
    // Frozen compatibility metadata only; this accessor does not admit a run.
    [[nodiscard]] std::uint64_t compatibility_identity() const noexcept;
    [[nodiscard]] std::optional<std::uint32_t> ple_step_id() const noexcept;
    // No dispatch and no static DAG/IO traversal. Session wrappers may use this
    // before starting PLE work or entering their dispatch-failure poison scope.
    void validate_dynamic(const ExecutionRuntimeResources& resources,
                          const EagerExecutionRequest& request) const;
private:
    struct Impl;
    std::unique_ptr<const Impl> impl_;
};
} // namespace strix
