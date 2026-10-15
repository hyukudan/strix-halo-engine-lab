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
        : delegate_(delegate) {
        if (compiled.execution.steps.empty() || compiled.execution.steps.back().op != SerialOp::lm_head ||
            compiled.execution.steps.back().writes_state != PersistentState::none)
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
            if (context.step_id != terminal_head_ || context.writes_state != PersistentState::none)
                throw std::invalid_argument("prefix output intent cannot omit a nonterminal or stateful head");
            if (omitted_heads_ != std::numeric_limits<std::uint64_t>::max()) ++omitted_heads_;
            return;
        }
        delegate_.dispatch_kernel(context);
    }
    void dispatch_runtime_step(const ExecutionDispatchContext& context) override {
        delegate_.dispatch_runtime_step(context);
    }
    [[nodiscard]] std::uint64_t omitted_heads() const noexcept { return omitted_heads_; }
private:
    EagerExecutionDispatcher& delegate_;
    std::uint32_t terminal_head_ = 0;
    bool state_only_ = false;
    std::uint64_t omitted_heads_ = 0;
};
} // namespace strix
