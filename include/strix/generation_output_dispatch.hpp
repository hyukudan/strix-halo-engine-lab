#pragma once

#include "strix/compiled_execution.hpp"

#include <limits>
#include <stdexcept>

namespace strix {

// Only the terminal, state-free LM head may be omitted. Every state-producing
// prefix step still dispatches normally. Full-output execution is the default.
class GenerationOutputDispatcher final : public EagerExecutionDispatcher {
public:
    GenerationOutputDispatcher(EagerExecutionDispatcher& delegate, const CompiledSerialExecution& compiled)
        : delegate_(delegate), execution_identity_(compiled.execution_identity),
          logical_rows_(compiled.logical_rows), concurrency_(compiled.request_concurrency) {
        if (!execution_identity_ || !logical_rows_ || !concurrency_ || compiled.execution.steps.empty() ||
            compiled.execution.steps.back().id != compiled.execution.steps.size() - 1U ||
            compiled.execution.steps.back().op != SerialOp::lm_head ||
            (compiled.execution.steps.back().reads_state != PersistentState::none ||
             compiled.execution.steps.back().writes_state != PersistentState::none))
            throw std::invalid_argument("state-only prefix requires a terminal state-free LM head");
        terminal_head_ = compiled.execution.steps.back().id;
    }
    class StateOnlyScope {
    public:
        explicit StateOnlyScope(GenerationOutputDispatcher& owner) : owner_(owner) {
            if (owner_.state_only_) throw std::logic_error("nested prefix output intent");
            owner_.state_only_ = true;
        }
        ~StateOnlyScope() { owner_.state_only_ = false; }
        StateOnlyScope(const StateOnlyScope&) = delete;
        StateOnlyScope& operator=(const StateOnlyScope&) = delete;
    private:
        GenerationOutputDispatcher& owner_;
    };
    void dispatch_kernel(const ExecutionDispatchContext& context) override {
        if (state_only_ && context.op == SerialOp::lm_head) {
            if (context.execution_identity != execution_identity_ || context.logical_rows != logical_rows_ ||
                context.request_concurrency != concurrency_ || context.step_id != terminal_head_ ||
                context.reads_state != PersistentState::none || context.writes_state != PersistentState::none)
                throw std::invalid_argument("prefix output intent cannot omit a nonterminal or stateful head");
            if (omitted_heads_ != std::numeric_limits<std::uint64_t>::max()) ++omitted_heads_;
            return;
        }
        delegate_.dispatch_kernel(context);
    }
    void dispatch_runtime_step(const ExecutionDispatchContext& context) override {
        delegate_.dispatch_runtime_step(context);
    }
    [[nodiscard]] bool state_only_active() const noexcept { return state_only_; }
    [[nodiscard]] std::uint64_t omitted_heads() const noexcept { return omitted_heads_; }
private:
    EagerExecutionDispatcher& delegate_;
    const std::uint64_t execution_identity_;
    const std::uint32_t logical_rows_, concurrency_;
    std::uint32_t terminal_head_ = 0;
    bool state_only_ = false;
    std::uint64_t omitted_heads_ = 0;
};
} // namespace strix
