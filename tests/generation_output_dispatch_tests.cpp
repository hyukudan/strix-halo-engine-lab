#include "strix/generation_output_dispatch.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }
struct Delegate final : strix::EagerExecutionDispatcher {
    unsigned state_calls = 0, head_calls = 0, runtime_calls = 0;
    bool fail = false;
    void dispatch_kernel(const strix::ExecutionDispatchContext& context) override {
        if (fail) throw std::runtime_error("injected dispatch failure");
        if (context.op == strix::SerialOp::lm_head) ++head_calls;
        else ++state_calls;
    }
    void dispatch_runtime_step(const strix::ExecutionDispatchContext&) override { ++runtime_calls; }
};
}
int main() try {
    strix::CompiledSerialExecution compiled;
    compiled.execution_identity = 17;
    compiled.execution.steps.resize(2);
    compiled.execution.steps[0].id = 0;
    compiled.execution.steps[0].op = strix::SerialOp::gated_delta_net;
    compiled.execution.steps[1].id = 1;
    compiled.execution.steps[1].op = strix::SerialOp::lm_head;
    Delegate delegate;
    strix::GenerationOutputDispatcher dispatcher(delegate, compiled);
    strix::ExecutionDispatchContext state, head;
    state.op = strix::SerialOp::gated_delta_net;
    state.writes_state = strix::PersistentState::recurrent_matrix;
    head.op = strix::SerialOp::lm_head;
    head.execution_identity = 17;
    head.step_id = 1;
    require(!dispatcher.state_only_active(), "default output intent is full output");
    dispatcher.dispatch_kernel(head);
    {
        strix::GenerationOutputDispatcher::StateOnlyScope intent(dispatcher);
        require(dispatcher.state_only_active(), "state-only intent is visible to the completion owner");
        bool nested_rejected = false;
        try { strix::GenerationOutputDispatcher::StateOnlyScope nested(dispatcher); }
        catch (const std::logic_error&) { nested_rejected = true; }
        require(nested_rejected && dispatcher.state_only_active(), "nested rejection preserves outer intent");
        dispatcher.dispatch_kernel(state);
        dispatcher.dispatch_runtime_step(state);
        dispatcher.dispatch_kernel(head);
    }
    dispatcher.dispatch_kernel(head);
    require(delegate.head_calls == 2 && delegate.state_calls == 1 && delegate.runtime_calls == 1 &&
            dispatcher.omitted_heads() == 1, "state-only prefix prunes only its terminal head");
    try {
        strix::GenerationOutputDispatcher::StateOnlyScope intent(dispatcher);
        delegate.fail = true;
        dispatcher.dispatch_kernel(state);
    } catch (const std::runtime_error&) {}
    delegate.fail = false;
    require(!dispatcher.state_only_active(), "state-only intent resets after exception");
    dispatcher.dispatch_kernel(head);
    require(delegate.head_calls == 3, "intent is restored after a producer exception");
    for (unsigned mode = 0; mode < 4; ++mode) {
        auto invalid = head;
        if (mode == 0) ++invalid.execution_identity;
        if (mode == 1) ++invalid.logical_rows;
        if (mode == 2) ++invalid.request_concurrency;
        if (mode == 3) invalid.reads_state = strix::PersistentState::recurrent_matrix;
        bool rejected_context = false;
        try {
            strix::GenerationOutputDispatcher::StateOnlyScope intent(dispatcher);
            dispatcher.dispatch_kernel(invalid);
        } catch (const std::invalid_argument&) { rejected_context = true; }
        require(rejected_context, "head omission cannot bypass execution or shape admission");
    }
    bool rejected = false;
    try {
        strix::GenerationOutputDispatcher::StateOnlyScope intent(dispatcher);
        head.writes_state = strix::PersistentState::recurrent_matrix;
        dispatcher.dispatch_kernel(head);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "stateful head cannot be silently pruned");
    for (unsigned mode = 0; mode < 5; ++mode) {
        auto malformed = compiled;
        if (mode == 0) malformed.execution_identity = 0;
        if (mode == 1) malformed.execution.steps.back().id = 0;
        if (mode == 2) malformed.execution.steps.back().op = strix::SerialOp::moe_experts;
        if (mode == 3) malformed.execution.steps.back().writes_state = strix::PersistentState::recurrent_matrix;
        if (mode == 4) malformed.execution.steps.back().reads_state = strix::PersistentState::recurrent_matrix;
        bool bad_constructor = false;
        try { strix::GenerationOutputDispatcher invalid(delegate, malformed); }
        catch (const std::invalid_argument&) { bad_constructor = true; }
        require(bad_constructor, "prefix intent requires an identified terminal state-free head");
    }
    std::cout << "generation output dispatch source fixtures PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
