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
    head.step_id = 1;
    dispatcher.dispatch_kernel(head);
    {
        strix::GenerationOutputDispatcher::StateOnlyScope intent(dispatcher);
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
    dispatcher.dispatch_kernel(head);
    require(delegate.head_calls == 3, "intent is restored after a producer exception");
    bool rejected = false;
    try {
        strix::GenerationOutputDispatcher::StateOnlyScope intent(dispatcher);
        head.writes_state = strix::PersistentState::recurrent_matrix;
        dispatcher.dispatch_kernel(head);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "stateful head cannot be silently pruned");
    std::cout << "generation output dispatch source fixtures PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
