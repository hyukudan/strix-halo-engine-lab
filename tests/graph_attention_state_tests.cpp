#include "strix/graph_attention_state.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures = 0;
void expect(bool value, const char* text) { if (!value) { std::cerr << text << '\n'; ++failures; } }
template<class Error = std::invalid_argument, class F> void rejects(F f) {
    try { f(); expect(false, "attention state should reject"); }
    catch (const Error&) {}
    catch (...) { expect(false, "wrong attention state error category"); }
}
strix::GraphAttentionStateRequest fixture() {
    strix::GraphAttentionStateRequest r;
    r.execution_identity = 7; r.query_heads = 4; r.kv_heads = 2; r.head_dim = 8;
    r.query_head_stride = 16; r.rotary_dim = 4; r.epsilon = 1e-6F; r.theta = 10000;
    r.context_tokens = 32;
    r.arguments = strix::bind_graph_inputs(strix::graph_input_layout(1), {7, 0x1000, 16, 1});
    r.query.address = 0x2000; r.query.bytes = 256; r.query.allocation_generation = 2; r.query.use.generation = 1;
    r.key.address = 0x3000; r.key.bytes = 64; r.key.allocation_generation = 3; r.key.use.generation = 1;
    r.value.address = 0x4000; r.value.bytes = 64; r.value.allocation_generation = 4; r.value.use.generation = 1;
    r.query_norm = {0, strix::RuntimeTensorStorage::resident, 0x5000, 32, 5, 0};
    r.key_norm = {1, strix::RuntimeTensorStorage::resident, 0x6000, 32, 6, 0};
    r.kv_state.address = 0x10000; r.kv_state.bytes = 2048; r.kv_state.allocation_generation = 7;
    r.kv_state.state_flag = strix::PersistentState::attention_kv;
    r.status = {7, 0x7000, 4, 8};
    return r;
}
}
int main() {
    const auto original = fixture();
    const auto plan = strix::prepare_graph_attention_state(original);
    strix::validate_graph_attention_state_plan(plan);
    expect(plan.query_elements == 56 && plan.kv_elements == 16 && plan.required_state_bytes == 2048,
           "strided query and context KV byte geometry");
    expect(plan.retained_view_bytes == 2516, "all retained disjoint views are accounted");
    auto shared = original; shared.key_norm = shared.query_norm;
    expect(strix::prepare_graph_attention_state(shared).retained_view_bytes == plan.retained_view_bytes - 32,
           "read-only shared norm physical bytes counted once");
    shared.key_norm.allocation_generation++;
    rejects([&] { (void)strix::prepare_graph_attention_state(shared); });
    auto changed = original; changed.position_mode = strix::AttentionPositionMode::scalar_control;
    expect(strix::prepare_graph_attention_state(changed).identity != plan.identity, "scalar mode has distinct identity");
    changed = original; ++changed.arguments.memory.allocation_generation;
    expect(strix::prepare_graph_attention_state(changed).identity != plan.identity, "same-address allocation change invalidates plan");
    for (int fault = 0; fault < 16; ++fault) {
        auto bad = original;
        switch (fault) {
        case 0: bad.query_heads = 0; break;
        case 1: bad.kv_heads = 3; break;
        case 2: bad.head_dim = 6; break;
        case 3: bad.rotary_dim = 3; break;
        case 4: bad.query_head_stride = 7; break;
        case 5: bad.epsilon = std::numeric_limits<float>::quiet_NaN(); break;
        case 6: bad.theta = 0; break;
        case 7: bad.status.execution_identity++; break;
        case 8: bad.key.bytes--; break;
        case 9: bad.kv_state.bytes--; break;
        case 10: bad.kv_state.provisional = true; break;
        case 11: bad.kv_state.storage = strix::SessionStateStorage::host; break;
        case 12: bad.value.address = bad.key.address; break;
        case 13: bad.query_norm.storage = strix::RuntimeTensorStorage::file_mapped; break;
        case 14: bad.arguments.device_argument_abi_identity++; break;
        case 15: bad.norm_ggml_type = 23; break;
        }
        rejects([&] { (void)strix::prepare_graph_attention_state(bad); });
    }
    auto huge = original; huge.context_tokens = std::numeric_limits<std::uint64_t>::max();
    rejects<std::overflow_error>([&] { (void)strix::prepare_graph_attention_state(huge); });
    auto tampered = plan; tampered.required_state_bytes--;
    rejects([&] { strix::validate_graph_attention_state_plan(tampered); });
    tampered = plan; tampered.request.key.allocation_generation++;
    rejects([&] { strix::validate_graph_attention_state_plan(tampered); });
    if (failures == 0) std::cout << "graph attention state portable contracts passed\n";
    return failures == 0 ? 0 : 1;
}
