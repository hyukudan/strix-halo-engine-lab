#include "strix/session_executor_handoff.hpp"

#include "strix/arithmetic_lineage.hpp"

#include <stdexcept>

namespace strix {
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
bool same_arena(const ExecutionArenaPlan& a, const ExecutionArenaPlan& b) {
    if (a.alignment != b.alignment || a.total_bytes != b.total_bytes || a.allocations.size() != b.allocations.size())
        return false;
    for (std::size_t i = 0; i < a.allocations.size(); ++i) {
        const auto& x = a.allocations[i];
        const auto& y = b.allocations[i];
        if (!(x.use == y.use) || x.first_step != y.first_step || x.last_step != y.last_step ||
            x.offset != y.offset || x.bytes != y.bytes) return false;
    }
    return true;
}
bool same_io(const ExecutionStepIo& a, const ExecutionStepIo& b) {
    return a.step_id == b.step_id && a.op == b.op && a.reads == b.reads && a.writes == b.writes &&
        a.temporaries == b.temporaries;
}
bool same_step(const ExecutionStep& a, const ExecutionStep& b) {
    return a.id == b.id && a.op == b.op && a.layer == b.layer && a.dependencies == b.dependencies &&
        a.reads_state == b.reads_state && a.writes_state == b.writes_state;
}
} // namespace

void validate_sibling_serial_executions(const CompiledSerialExecution& a, const CompiledSerialExecution& b) {
    require(a.execution_identity && b.execution_identity && a.execution_identity != b.execution_identity &&
        a.kernel_plan_identity && b.kernel_plan_identity && a.kernel_plan_identity != b.kernel_plan_identity,
        "sibling serial executions require distinct non-zero arithmetic identities");
    require(a.logical_rows == 1U && b.logical_rows == 1U && a.request_concurrency == 1U && b.request_concurrency == 1U &&
        !a.device_graph_inputs && !b.device_graph_inputs && a.vocab_size == b.vocab_size &&
        a.max_position_embeddings == b.max_position_embeddings,
        "sibling serial executions require eager one-row plans of one model geometry");
    require(a.execution.steps.size() == b.execution.steps.size() && a.io.steps.size() == b.io.steps.size() &&
        a.bindings.bindings.size() == b.bindings.bindings.size(),
        "sibling serial executions differ in step count");
    for (std::size_t i = 0; i < a.execution.steps.size(); ++i)
        require(same_step(a.execution.steps[i], b.execution.steps[i]),
            "sibling serial executions differ in a step, dependency or persistent state access");
    for (std::size_t i = 0; i < a.io.steps.size(); ++i)
        require(same_io(a.io.steps[i], b.io.steps[i]), "sibling serial executions differ in buffer IO");
    for (std::size_t i = 0; i < a.bindings.bindings.size(); ++i)
        require(a.bindings.bindings[i].step_id == b.bindings.bindings[i].step_id &&
            a.bindings.bindings[i].op == b.bindings.bindings[i].op &&
            a.bindings.bindings[i].host_or_runtime_step == b.bindings.bindings[i].host_or_runtime_step,
            "sibling serial executions differ in a binding's step or runtime role");
    require(same_arena(a.arena, b.arena), "sibling serial executions differ in arena layout");
}

ExecutionMemoryBinding sibling_scratch_binding(const CompiledSerialExecution& sibling,
    const CompiledSerialExecution& owner, const ExecutionMemoryBinding& owner_scratch) {
    validate_sibling_serial_executions(sibling, owner);
    validate_execution_memory_binding(owner.execution_identity, owner.arena, owner_scratch);
    auto binding = owner_scratch;
    binding.execution_identity = sibling.execution_identity;
    validate_execution_memory_binding(sibling.execution_identity, sibling.arena, binding);
    return binding;
}

void hand_off_serial_session(SerialSessionContext& session, SessionExecutor from, SessionExecutor to) {
    require(from.compiled && from.resources && to.compiled && to.resources,
        "serial session handoff requires both executors");
    const auto& a = *from.resources;
    const auto& b = *to.resources;
    require(a.model_manifest && a.model_weights && a.tensor_access && a.session_state_layout && a.session_state_memory &&
        !a.distributed_stage && !b.distributed_stage &&
        a.model_manifest == b.model_manifest && a.model_weights == b.model_weights && a.tensor_access == b.tensor_access &&
        a.session_state_layout == b.session_state_layout && a.session_state_memory == b.session_state_memory &&
        a.ple_lookup == b.ple_lookup && a.ple_hash_contract == b.ple_hash_contract && a.ple_prefetch == b.ple_prefetch,
        "serial session handoff requires one model, tensor access, state and PLE binding");
    validate_sibling_serial_executions(*from.compiled, *to.compiled);
    require(session.health == SerialSessionHealth::ready && !session.arithmetic_lineage.tainted &&
        (session.next_position == 0U ? session.arithmetic_lineage.intervals.empty()
                                     : arithmetic_lineage_valid(session.arithmetic_lineage, session.next_position)),
        "serial session handoff requires a ready session with a valid arithmetic lineage");
    const auto source = build_runtime_execution_identities(from.compiled->execution_identity, from.compiled->arena, a);
    require(session.execution_identity == from.compiled->execution_identity &&
        session.runtime_compatibility_identity == source.compatibility_identity &&
        session.state_layout_identity == a.session_state_layout->layout_identity,
        "serial session handoff source executor does not own the session");
    const auto target = build_runtime_execution_identities(to.compiled->execution_identity, to.compiled->arena, b);
    session.execution_identity = to.compiled->execution_identity;
    session.runtime_compatibility_identity = target.compatibility_identity;
}

} // namespace strix
