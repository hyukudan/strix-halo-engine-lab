#include "strix/graph_operator_control.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures = 0;
void expect(bool value) { if (!value) ++failures; }
template<class F> void rejects(F f) { try { f(); ++failures; } catch (const std::invalid_argument&) {} }
strix::GraphOperatorControlRequest fixture() {
    strix::GraphOperatorControlRequest r;
    r.execution_identity = 9; r.context_tokens = 32;
    r.arguments = strix::bind_graph_inputs(strix::graph_input_layout(1), {9, 0x1000, 16, 1});
    r.status = {9, 0x2000, 4, 2};
    r.resources = {{0x3000, 256, 3, 4}, {0x4000, 32, 4, 2}, {0x5000, 65536, 5, 4}};
    return r;
}
}
int main() {
    const auto original = fixture();
    auto plan = strix::prepare_graph_operator_control(original);
    strix::validate_graph_operator_control(plan);
    for (unsigned fault = 0; fault < 10; ++fault) {
        auto bad = original;
        switch (fault) {
        case 0: bad.execution_identity = 0; break;
        case 1: bad.context_tokens = 0; break;
        case 2: ++bad.arguments.device_argument_abi_identity; break;
        case 3: bad.status.arena_bytes = 8; break;
        case 4: bad.status.arena_base = 0x1008; break;
        case 5: bad.resources[0].address = 0x1000; break;
        case 6: bad.resources[1].address = 0x2000; break;
        case 7: bad.resources[0].allocation_generation = 0; break;
        case 8: bad.resources[0].address = std::numeric_limits<std::uintptr_t>::max() - 3; break;
        case 9: bad.resources[0].alignment = 3; break;
        }
        rejects([&] { (void)strix::prepare_graph_operator_control(bad); });
    }
    auto changed = original; ++changed.resources[0].allocation_generation;
    expect(strix::prepare_graph_operator_control(changed).identity != plan.identity);
    ++plan.request.context_tokens;
    rejects([&] { strix::validate_graph_operator_control(plan); });
    if (!failures) std::cout << "graph operator control portable contracts passed\n";
    return failures ? 1 : 0;
}
