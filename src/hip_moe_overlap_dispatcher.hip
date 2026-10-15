#include "strix/hip_moe_overlap_dispatcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {
void require(bool ok, const char* text) { if (!ok) throw std::logic_error(text); }
void check(hipError_t result, const char* text) {
    if (result != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(result));
}
bool same_memory(const ExecutionMemoryBinding& a, const ExecutionMemoryBinding& b) {
    return a.execution_identity == b.execution_identity && a.arena_base == b.arena_base &&
           a.arena_bytes == b.arena_bytes && a.allocation_generation == b.allocation_generation;
}
bool branch_op(SerialOp op) {
    return op == SerialOp::moe_route || op == SerialOp::shared_expert ||
           op == SerialOp::moe_experts || op == SerialOp::moe_combine;
}
} // namespace
struct HipMoeSiblingStreams::Impl {
    enum class Phase { idle, forking, shared_recorded, routed };
    hipStream_t main = nullptr, side = nullptr;
    hipEvent_t fork = nullptr, join = nullptr;
    std::shared_ptr<const void> resources_owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    HipMoeOverlapStats counters;
    Phase phase = Phase::idle;
    std::uint64_t position = 0, committed_before = 0;
    std::uint32_t expected = 0, forks = 0, joins = 0, layer = 0;
    bool active = false, verified = false, broken = false;
    Impl(hipStream_t borrowed, std::shared_ptr<const void> owner,
         std::shared_ptr<const HipSetupQuiescenceGuard> setup)
        : main(borrowed), resources_owner(std::move(owner)), guard(std::move(setup)) {
        require(main && main != hipStreamLegacy && main != hipStreamPerThread && resources_owner && guard,
                "MoE sibling streams require actual main/resources/guard owners");
        guard->require_original();
        unsigned flags = 0;
        check(hipStreamGetFlags(main, &flags), "validate borrowed MoE main stream");
        try {
            check(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), "create MoE sibling stream");
            check(hipEventCreateWithFlags(&fork, hipEventDisableTiming), "create MoE fork event");
            check(hipEventCreateWithFlags(&join, hipEventDisableTiming), "create MoE join event");
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        if (!guard) return;
        guard->recover_or_fail_stop();
        const auto release = [](hipError_t e) { if (e != hipSuccess) std::_Exit(3); };
        if (join) { release(hipEventDestroy(join)); join = nullptr; }
        if (fork) { release(hipEventDestroy(fork)); fork = nullptr; }
        if (side) { release(hipStreamDestroy(side)); side = nullptr; }
    }
    ~Impl() { cleanup(); }
    void abort() noexcept {
        guard->recover_or_fail_stop();
        if (active && counters.aborted_tokens != std::numeric_limits<std::uint64_t>::max()) ++counters.aborted_tokens;
        active = false; verified = false; broken = true; phase = Phase::idle;
    }
    void begin(std::uint64_t p, std::uint64_t committed, std::uint32_t layers) {
        require(!active && !broken && phase == Phase::idle && layers &&
                p < std::numeric_limits<std::uint64_t>::max() && committed < std::numeric_limits<std::uint64_t>::max(),
                "MoE sibling token is invalid, already active or failed");
        require(layers <= std::numeric_limits<std::uint64_t>::max() - counters.fork_records &&
                layers <= std::numeric_limits<std::uint64_t>::max() - counters.join_waits &&
                layers <= std::numeric_limits<std::uint64_t>::max() - counters.committed_layers &&
                counters.verified_tokens < std::numeric_limits<std::uint64_t>::max() &&
                counters.committed_tokens < std::numeric_limits<std::uint64_t>::max(), "MoE sibling counters exhausted");
        guard->require_original();
        hipStreamCaptureStatus main_capture{}, side_capture{};
        check(hipStreamIsCapturing(main, &main_capture), "query MoE main capture");
        check(hipStreamIsCapturing(side, &side_capture), "query MoE side capture");
        require(main_capture == hipStreamCaptureStatusNone && side_capture == hipStreamCaptureStatusNone,
                "MoE sibling phase one rejects graph capture");
        position = p; committed_before = committed; expected = layers; forks = joins = 0;
        active = true; verified = false;
    }
    void expected_phase(Phase wanted, std::uint32_t current_layer) {
        require(active && !verified && !broken && phase == wanted &&
                (wanted == Phase::idle || layer == current_layer), "MoE sibling branch phase/layer differs");
    }
};
HipMoeSiblingStreams::HipMoeSiblingStreams(hipStream_t main, std::shared_ptr<const void> owner,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard)
    : impl_(std::make_unique<Impl>(main, std::move(owner), std::move(guard))) {}
HipMoeSiblingStreams::~HipMoeSiblingStreams() = default;
void HipMoeSiblingStreams::begin_token(std::uint64_t position, std::uint64_t committed, std::uint32_t layers) {
    impl_->begin(position, committed, layers);
}
void HipMoeSiblingStreams::fork_shared(std::uint32_t layer) {
    auto& s = *impl_; s.expected_phase(Impl::Phase::idle, layer);
    require(s.forks < s.expected, "too many MoE sibling forks");
    s.layer = layer; s.phase = Impl::Phase::forking;
    try {
        check(hipEventRecord(s.fork, s.main), "record shared producer-ready");
        check(hipStreamWaitEvent(s.side, s.fork, 0), "side waits for shared input producer");
    } catch (...) { s.abort(); throw; }
}
void HipMoeSiblingStreams::record_shared_done(std::uint32_t layer) {
    auto& s = *impl_; s.expected_phase(Impl::Phase::forking, layer);
    try { check(hipEventRecord(s.join, s.side), "record shared branch done"); }
    catch (...) { s.abort(); throw; }
    ++s.forks; ++s.counters.fork_records; s.phase = Impl::Phase::shared_recorded;
}
void HipMoeSiblingStreams::begin_experts(std::uint32_t layer) {
    auto& s = *impl_; s.expected_phase(Impl::Phase::shared_recorded, layer);
    s.phase = Impl::Phase::routed;
}
void HipMoeSiblingStreams::join_before_combine(std::uint32_t layer) {
    auto& s = *impl_; s.expected_phase(Impl::Phase::routed, layer);
    try { check(hipStreamWaitEvent(s.main, s.join, 0), "main joins shared before combine"); }
    catch (...) { s.abort(); throw; }
    ++s.joins; ++s.counters.join_waits; s.phase = Impl::Phase::idle;
}
void HipMoeSiblingStreams::verify_before_commit() {
    auto& s = *impl_;
    require(s.active && !s.broken && !s.verified && s.phase == Impl::Phase::idle &&
            s.forks == s.expected && s.joins == s.expected, "incomplete MoE sibling token cannot commit");
    try {
        check(hipStreamSynchronize(s.main), "drain main including shared joins");
        check(hipStreamQuery(s.side), "prove shared stream complete before publication");
        s.guard->require_original();
    } catch (...) { s.abort(); throw; }
    s.verified = true; ++s.counters.verified_tokens;
}
void HipMoeSiblingStreams::mark_committed(std::uint64_t next, std::uint64_t committed) {
    auto& s = *impl_;
    require(s.active && s.verified && !s.broken && next == s.position + 1U && committed == s.committed_before + 1U,
            "MoE sibling completion requires actual serial-token commit");
    ++s.counters.committed_tokens; s.counters.committed_layers += s.expected;
    s.active = false; s.verified = false;
}
void HipMoeSiblingStreams::abort() noexcept { impl_->abort(); }
hipStream_t HipMoeSiblingStreams::main_stream() const noexcept { return impl_->main; }
hipStream_t HipMoeSiblingStreams::side_stream() const noexcept { return impl_->side; }
HipMoeOverlapStats HipMoeSiblingStreams::stats() const noexcept { return impl_->counters; }
bool HipMoeSiblingStreams::failed() const noexcept { return impl_->broken; }

struct HipMoeOverlapDispatcher::Impl {
    const CompiledSerialExecution compiled; // Deep immutable admission snapshot.
    std::vector<std::vector<std::size_t>> accessed_allocations;
    const ExecutionRuntimeResources& resources;
    const ExecutionRuntimeResources resource_snapshot;
    MoeForkJoinArenaPlan plan;
    CompiledLaunchPlan launches;
    DirectLaunchDispatcher direct;
    ExecutionMemoryBinding scratch;
    RuntimeExecutionIdentities runtime;
    std::shared_ptr<const void> launcher_owner;
    HipMoeSiblingStreams streams;
    std::size_t cursor = 0, layer_cursor = 0;
    std::uint64_t position = 0;
    bool active = false;
    Impl(const CompiledSerialExecution& execution, const MoeForkJoinArenaPlan& arena,
         const CompiledLaunchPlan& launch, const ExecutionRuntimeResources& bound, hipStream_t main,
         std::shared_ptr<const void> owner, std::shared_ptr<const void> configs,
         std::shared_ptr<const HipSetupQuiescenceGuard> guard, ExecutionTelemetry* telemetry)
        : compiled(execution), resources(bound), resource_snapshot(bound),
          plan(arena), launches(launch), direct(launches, telemetry),
          scratch(bound.scratch ? *bound.scratch : ExecutionMemoryBinding{}), launcher_owner(std::move(configs)),
          streams(main, std::move(owner), std::move(guard)) {
        plan.validate_execution(compiled);
        plan.validate_binding(scratch);
        require(launcher_owner && !compiled.device_graph_inputs && compiled.logical_rows == 1 &&
                compiled.request_concurrency == 1 && launches.execution_identity == compiled.execution_identity &&
                launches.plan_identity && launches.steps.size() == compiled.execution.steps.size() &&
                compiled.io.steps.size() == compiled.execution.steps.size() &&
                compiled.bindings.bindings.size() == compiled.execution.steps.size(), "MoE overlap requires fresh expanded C1 launch plan/resources");
        runtime = build_runtime_execution_identities(compiled.execution_identity, compiled.arena, resources);
        for (std::size_t i = 0; i < launches.steps.size(); ++i) {
            const auto& step = compiled.execution.steps[i];
            const auto& launcher = launches.steps[i];
            require(step.id == i && launcher.step_id == step.id && launcher.op == step.op && launcher.function &&
                    launcher.implementation_identity && launcher.runtime_step == compiled.bindings.bindings[i].host_or_runtime_step,
                    "MoE overlap compiled launcher differs");
        }
        accessed_allocations.resize(compiled.io.steps.size());
        for (std::size_t step = 0; step < compiled.io.steps.size(); ++step) {
            const auto cache = [&](const auto& uses) {
                for (const auto use : uses) {
                    const auto found = std::find_if(compiled.arena.allocations.begin(), compiled.arena.allocations.end(),
                        [&](const auto& allocation) { return allocation.use == use; });
                    require(found != compiled.arena.allocations.end(), "MoE overlap IO has no arena allocation");
                    accessed_allocations[step].push_back(static_cast<std::size_t>(found - compiled.arena.allocations.begin()));
                }
            };
            cache(compiled.io.steps[step].reads); cache(compiled.io.steps[step].writes); cache(compiled.io.steps[step].temporaries);
        }
        for (const auto& layer : plan.layers()) {
            require(layer.route_step + 1U == layer.shared_step && layer.shared_step + 1U == layer.experts_step &&
                    layer.experts_step + 1U == layer.combine_step && !launches.steps[layer.shared_step].runtime_step &&
                    !launches.steps[layer.experts_step].runtime_step && !launches.steps[layer.combine_step].runtime_step,
                    "MoE overlap requires unchanged route/shared/experts/combine host order");
        }
    }
    void dispatch(const ExecutionDispatchContext& c, bool runtime_step) {
        try {
            require(active && cursor < compiled.execution.steps.size() && c.step_id == cursor &&
                    c.execution_identity == compiled.execution_identity && c.position == position &&
                    c.logical_rows == 1 && c.request_concurrency == 1 && c.arena &&
                    c.arena->alignment == compiled.arena.alignment && c.arena->total_bytes == compiled.arena.total_bytes &&
                    c.arena->allocations.size() == compiled.arena.allocations.size() &&
                    c.resources == &resources && c.memory && resources.scratch &&
                    same_memory(*c.memory, scratch) && same_memory(*resources.scratch, scratch) &&
                    c.session_state_layout == resources.session_state_layout && c.session_state_memory == resources.session_state_memory,
                    "MoE overlap live context differs from retained execution/resources");
            for (const auto index : accessed_allocations[cursor]) {
                // Prepared eager owns an equivalent immutable arena copy. Validate
                // every accessed mapping, without rejecting its distinct address.
                const auto& actual = c.arena->allocations[index];
                const auto& expected = compiled.arena.allocations[index];
                require(actual.use == expected.use && actual.first_step == expected.first_step && actual.last_step == expected.last_step &&
                        actual.offset == expected.offset && actual.bytes == expected.bytes, "MoE overlap arena mapping changed");
            }
            require(resources.model_manifest == resource_snapshot.model_manifest && resources.model_weights == resource_snapshot.model_weights &&
                    resources.tensor_access == resource_snapshot.tensor_access && resources.session_state_layout == resource_snapshot.session_state_layout &&
                    resources.session_state_memory == resource_snapshot.session_state_memory && resources.ple_lookup == resource_snapshot.ple_lookup &&
                    resources.ple_hash_contract == resource_snapshot.ple_hash_contract && resources.ple_prefetch == resource_snapshot.ple_prefetch,
                    "MoE overlap immutable resource pointers changed");
            const auto& step = compiled.execution.steps[cursor];
            const auto& io = compiled.io.steps[cursor];
            const auto& binding = compiled.bindings.bindings[cursor];
            const auto* launch_context = current_hip_launch_context();
            require(launch_context && !launch_context->graph_capture && launch_context->stream == streams.main_stream() &&
                    launch_context->deferred_moe_status.execution_identity == compiled.execution_identity &&
                    launch_context->deferred_moe_status.arena_bytes == 8 && launch_context->deferred_moe_status.allocation_generation &&
                    launch_context->deferred_moe_status.arena_base,
                    "MoE overlap requires unchanged outer eager D1 scope");
            require(c.op == step.op && c.layer == step.layer && c.reads_state == step.reads_state &&
                    c.writes_state == step.writes_state && binding.host_or_runtime_step == runtime_step &&
                    std::equal(c.reads.begin(),c.reads.end(),io.reads.begin(),io.reads.end()) &&
                    std::equal(c.writes.begin(),c.writes.end(),io.writes.begin(),io.writes.end()) &&
                    std::equal(c.temporaries.begin(),c.temporaries.end(),io.temporaries.begin(),io.temporaries.end()) &&
                    (binding.kernel_id ? (c.kernel_id && *c.kernel_id == *binding.kernel_id) : !c.kernel_id),
                    "MoE overlap step/IO/binding changed");
            require(!runtime_step || !branch_op(c.op), "MoE overlap branches cannot be runtime dispatches");
            if (c.op == SerialOp::shared_expert || c.op == SerialOp::moe_experts || c.op == SerialOp::moe_combine) {
                require(layer_cursor < plan.layers().size(), "unexpected MoE overlap layer");
                const auto& layer = plan.layers()[layer_cursor];
                require(c.layer && *c.layer == layer.layer, "MoE overlap layer identity differs");
                if (c.op == SerialOp::shared_expert) {
                    require(c.step_id == layer.shared_step && c.writes_state == PersistentState::none,
                            "unexpected shared branch or persistent state writer");
                    streams.fork_shared(layer.layer);
                    {
                        HipSharedExpertSideScope side(streams.main_stream(), streams.side_stream());
                        direct.dispatch_kernel(c);
                    }
                    streams.record_shared_done(layer.layer);
                } else if (c.op == SerialOp::moe_experts) {
                    require(c.step_id == layer.experts_step, "unexpected routed branch");
                    streams.begin_experts(layer.layer); direct.dispatch_kernel(c);
                } else {
                    require(c.step_id == layer.combine_step, "unexpected MoE combine");
                    streams.join_before_combine(layer.layer); direct.dispatch_kernel(c); ++layer_cursor;
                }
            } else {
                if (runtime_step) direct.dispatch_runtime_step(c); else direct.dispatch_kernel(c);
            }
            ++cursor;
        } catch (...) { streams.abort(); active = false; throw; }
    }
};
HipMoeOverlapDispatcher::HipMoeOverlapDispatcher(const CompiledSerialExecution& compiled,
    const MoeForkJoinArenaPlan& plan, const CompiledLaunchPlan& launches, const ExecutionRuntimeResources& resources,
    hipStream_t main, std::shared_ptr<const void> owner, std::shared_ptr<const void> launchers,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard, ExecutionTelemetry* telemetry)
    : impl_(std::make_unique<Impl>(compiled,plan,launches,resources,main,std::move(owner),std::move(launchers),std::move(guard),telemetry)) {}
HipMoeOverlapDispatcher::~HipMoeOverlapDispatcher() = default;
void HipMoeOverlapDispatcher::begin_token(const SerialSessionContext& session) {
    auto& s = *impl_;
    require(!s.active && session.health == SerialSessionHealth::ready && !session.arithmetic_lineage.tainted &&
            session.execution_identity == s.compiled.execution_identity &&
            session.runtime_compatibility_identity == s.runtime.compatibility_identity && s.resources.scratch &&
            same_memory(*s.resources.scratch,s.scratch), "MoE overlap token session/binding differs");
    s.streams.begin_token(session.next_position,session.committed_tokens,static_cast<std::uint32_t>(s.plan.layers().size()));
    s.position = session.next_position; s.cursor = 0; s.layer_cursor = 0; s.active = true;
}
void HipMoeOverlapDispatcher::dispatch_kernel(const ExecutionDispatchContext& c) { impl_->dispatch(c,false); }
void HipMoeOverlapDispatcher::dispatch_runtime_step(const ExecutionDispatchContext& c) { impl_->dispatch(c,true); }
void HipMoeOverlapDispatcher::verify_before_commit(bool terminal_head_omitted) {
    auto& s = *impl_;
    require(s.active && s.layer_cursor == s.plan.layers().size(),
            "MoE overlap missing compiled layers cannot publish");
    if (terminal_head_omitted) {
        require(!s.compiled.execution.steps.empty() && s.cursor == s.compiled.execution.steps.size() - 1U,
                "MoE overlap prefix may omit exactly one terminal head");
        const auto& head = s.compiled.execution.steps.back();
        const auto& binding = s.compiled.bindings.bindings.back();
        require(head.id == s.cursor && head.op == SerialOp::lm_head && !head.layer &&
                head.reads_state == PersistentState::none && head.writes_state == PersistentState::none &&
                !binding.host_or_runtime_step && !s.launches.steps.back().runtime_step,
                "MoE overlap omitted head must be a terminal stateless kernel");
        // Logical traversal only: the exterior output dispatcher admitted the
        // omission. No launcher, telemetry or device-work counter is advanced.
        ++s.cursor;
    }
    require(s.cursor == s.compiled.execution.steps.size(),
            "MoE overlap missing compiled steps cannot publish");
    s.streams.verify_before_commit();
}
void HipMoeOverlapDispatcher::mark_committed(const SerialSessionContext& session) {
    auto& s = *impl_;
    require(s.active && session.execution_identity == s.compiled.execution_identity && session.health == SerialSessionHealth::ready,
            "MoE overlap commit session differs");
    s.streams.mark_committed(session.next_position,session.committed_tokens); s.active = false;
}
void HipMoeOverlapDispatcher::abort() noexcept { impl_->streams.abort(); impl_->active = false; }
HipMoeOverlapStats HipMoeOverlapDispatcher::stats() const noexcept { return impl_->streams.stats(); }
} // namespace strix
