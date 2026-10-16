#include "strix/hip_distributed_stage.hpp"

#include "strix/deferred_moe_status.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_tensor_views.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {
void stage_check(hipError_t value,const char* operation) {
    if(value!=hipSuccess) throw std::runtime_error(std::string(operation)+": "+hipGetErrorString(value));
}
bool stage_empty(const ExecutionMemoryBinding& value) {
    return !value.execution_identity && !value.arena_base && !value.arena_bytes && !value.allocation_generation;
}

}
struct HipDistributedStage::Impl {
    std::shared_ptr<const void> owner;
    HipSetupQuiescenceGuard guard;
    CompiledSerialExecution serial;
    DistributedPlacementPlan placement;
    std::uint32_t first,last,cursor;
    ExecutionRuntimeResources resources;
    HipLaunchContext scope;
    HipLauncherCatalog catalog;
    ExecutionLauncherRegistry registry;
    CompiledLaunchPlan launch;
    std::unique_ptr<DirectLaunchDispatcher> dispatcher;
    std::unique_ptr<const PreparedTensorViews> metadata;
    RuntimeExecutionIdentities identities;
    SerialSessionStateLayout state_layout;
    std::vector<ExecutionDispatchContext> contexts;
    HipDistributedStageToken token{};
    PersistentStateGenerations local;
    DeferredMoeFirstError status{};
    bool deferred=false,active=false,broken=false;
    std::uint64_t last_generation=0,last_position=0,request_id=0,epoch=0;

    Impl(CompiledSerialExecution source,DistributedPlacementPlan plan,std::uint32_t begin,std::uint32_t end,
        ModelContract model,HipLauncherCatalogConfig config,const ExecutionRuntimeResources& binding,
        HipLaunchContext native,std::shared_ptr<const void> retained)
        :owner(std::move(retained)),serial(std::move(source)),placement(std::move(plan)),
         first(begin),last(end),cursor(begin),resources(binding),scope(native),catalog(model,config),
         deferred(config.deferred_moe_status) {
        if(!owner || !serial.execution_identity || !serial.kernel_plan_identity || serial.logical_rows!=1U ||
            serial.request_concurrency!=1U || serial.device_graph_inputs || config.enable_device_graph_inputs ||
            serial.execution.steps.size()>=std::numeric_limits<std::uint32_t>::max() ||
            first>last || last>=serial.execution.steps.size() ||
            serial.execution.steps.size()!=serial.bindings.bindings.size() ||
            !placement.executable || !placement.plan_identity || placement.compiled_execution_identity!=serial.execution_identity ||
            placement.steps.size()!=serial.execution.steps.size())
            throw std::invalid_argument("HIP distributed stage requires one complete eager C1 snapshot and contiguous placement");
        validate_serial_execution_io_plan(serial.execution,serial.io);
        validate_execution_runtime_resources(serial.execution_identity,serial.arena,resources,0);
        std::vector<std::uint32_t> nodes;nodes.reserve(placement.steps.size());
        for(const auto& step:placement.steps) nodes.push_back(step.node);
        const auto derived=build_distributed_placement_plan(serial,*binding.model_manifest,*binding.tensor_access,nodes,placement.local_node);
        bool same=placement.plan_identity==derived.plan_identity && placement.executable==derived.executable &&
            placement.has_remote_work==derived.has_remote_work && placement.remote_nodes==derived.remote_nodes &&
            placement.boundary_payload_bytes_per_token==derived.boundary_payload_bytes_per_token &&
            placement.boundaries.size()==derived.boundaries.size();
        for(std::size_t i=0;same && i<placement.boundaries.size();++i) {
            const auto& a=placement.boundaries[i];const auto& b=derived.boundaries[i];
            same=a.boundary_id==b.boundary_id && a.source_step==b.source_step && a.destination_step==b.destination_step &&
                a.source_node==b.source_node && a.destination_node==b.destination_node &&
                a.payload_bytes==b.payload_bytes && a.payloads==b.payloads;
        }
        if(!same) throw std::invalid_argument("HIP distributed stage placement is not derived from its actual serial IO/arena");
        if(placement.manifest_layout_identity!=resources.model_manifest->layout_identity ||
            placement.tensor_access_identity!=resources.tensor_access->plan_identity)
            throw std::invalid_argument("HIP distributed stage placement provenance differs");
        for(const auto& weight:resources.model_weights->tensors) {
            const auto& tensor=resources.model_manifest->tensors[weight.tensor_index];
            const bool local_ple=tensor.role==TensorRole::ngram_table &&
                tensor.residency==ResidencyPolicy::file_backed_candidate && weight.storage==RuntimeTensorStorage::file_mapped &&
                dynamic_cast<const FileMappedPleLookupBackend*>(resources.ple_lookup)!=nullptr;
            if((!runtime_tensor_gpu_visible(weight.storage) && !local_ple) || !weight.address || !weight.allocation_generation)
                throw std::invalid_argument("HIP distributed stage requires local GPU weights or its retained file-mapped PLE source");
        }
        for(std::size_t i=0;i<placement.steps.size();++i) {
            const auto& p=placement.steps[i];const auto& s=serial.execution.steps[i];
            const bool local_step=i>=first && i<=last;
            if(s.id!=i || p.step_id!=s.id || p.op!=s.op || p.placement_ambiguous ||
                p.remote!=(p.node!=placement.local_node) || local_step!=(p.node==placement.local_node))
                throw std::invalid_argument("HIP distributed stage local steps must be exactly one contiguous interval");
        }
        const auto overlap=[](std::uintptr_t a,std::uint64_t an,std::uintptr_t b,std::uint64_t bn) {
            if((an && (!a || an>std::numeric_limits<std::uintptr_t>::max()-a)) ||
                (bn && (!b || bn>std::numeric_limits<std::uintptr_t>::max()-b)))
                throw std::invalid_argument("HIP distributed stage full owner extent overflows");
            return an && bn && a<b+bn && b<a+an;
        };
        const auto& arena=*resources.scratch;const auto& state=*resources.session_state_memory;
        if(overlap(arena.arena_base,arena.arena_bytes,state.device_base,state.device_bytes) ||
            overlap(arena.arena_base,arena.arena_bytes,state.host_base,state.host_bytes) ||
            overlap(state.device_base,state.device_bytes,state.host_base,state.host_bytes))
            throw std::invalid_argument("HIP distributed stage scratch and persistent owners must be disjoint");
        for(const auto& w:resources.model_weights->tensors)
            if(overlap(arena.arena_base,arena.arena_bytes,w.address,w.bytes) ||
                overlap(state.device_base,state.device_bytes,w.address,w.bytes) ||
                overlap(state.host_base,state.host_bytes,w.address,w.bytes))
                throw std::invalid_argument("HIP distributed stage writable owner aliases model weights");
        validate_scope();
        identities=build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        state_layout=*resources.session_state_layout;
        metadata=std::make_unique<const PreparedTensorViews>(serial.execution_identity,serial.execution,
            *resources.model_manifest,*resources.model_weights,*resources.tensor_access,owner);
        catalog.register_for_compiled_execution(serial,registry,&resources);
        launch=compile_launch_plan(serial,registry);
        dispatcher=std::make_unique<DirectLaunchDispatcher>(launch);
        contexts.resize(static_cast<std::size_t>(last-first)+1U);
        if(deferred) guard.run([&]{finish_step();});
    }
    void original_device() const {
        stage_check(hipSetDevice(guard.device_index()),"select distributed stage device");
        guard.require_original();
    }
    ~Impl() {
        if(hipSetDevice(guard.device_index())!=hipSuccess) std::terminate();
        guard.recover_or_fail_stop();
    }
    void validate_scope() const {
        guard.require_original();
        if(current_hip_launch_context() || !scope.stream || scope.stream==hipStreamLegacy || scope.stream==hipStreamPerThread ||
            scope.graph_capture || scope.arguments!=GraphDynamicArguments{} || !stage_empty(scope.status) ||
            !stage_empty(scope.ngram_input) || !stage_empty(scope.ngram_output) || scope.context_tokens || scope.vocabulary_size)
            throw std::invalid_argument("HIP distributed stage requires an unnested owned eager scope");
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        stage_check(hipStreamIsCapturing(scope.stream,&capture),"query distributed stage capture");
        if(capture!=hipStreamCaptureStatusNone) throw std::invalid_argument("HIP distributed stage stream is captured");
        const auto& d=scope.deferred_moe_status;
        if(!deferred) {
            if(!stage_empty(d)) throw std::invalid_argument("HIP distributed stage D1 parity differs");
        } else {
            if(d.execution_identity!=serial.execution_identity || !d.arena_base || d.arena_base%8U || d.arena_bytes!=8U ||
                !d.allocation_generation || d.arena_base>std::numeric_limits<std::uintptr_t>::max()-8U)
                throw std::invalid_argument("HIP distributed stage requires the exact original D1 record");
            const auto disjoint=[&](std::uintptr_t base,std::uint64_t bytes) {
                if(!bytes) return;
                if(!base || bytes>std::numeric_limits<std::uintptr_t>::max()-base ||
                    (d.arena_base<base+bytes && base<d.arena_base+8U))
                    throw std::invalid_argument("HIP distributed stage D1 aliases a retained owner");
            };
            disjoint(resources.scratch->arena_base,resources.scratch->arena_bytes);
            disjoint(resources.session_state_memory->device_base,resources.session_state_memory->device_bytes);
            disjoint(resources.session_state_memory->host_base,resources.session_state_memory->host_bytes);
            for(const auto& w:resources.model_weights->tensors) disjoint(w.address,w.bytes);
        }
    }
    void validate_metadata() const {
        metadata->validate_source_metadata(*resources.model_manifest,*resources.model_weights,*resources.tensor_access);
        const auto now=build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        if(now.compatibility_identity!=identities.compatibility_identity || now.binding_identity!=identities.binding_identity)
            throw std::invalid_argument("HIP distributed stage resource binding changed");
        const auto& current=*resources.session_state_layout;
        if(current.context_tokens!=state_layout.context_tokens || current.layout_identity!=state_layout.layout_identity ||
            current.device_alignment!=state_layout.device_alignment || current.host_alignment!=state_layout.host_alignment ||
            current.device_total_bytes!=state_layout.device_total_bytes || current.host_total_bytes!=state_layout.host_total_bytes ||
            current.recurrent_storage!=state_layout.recurrent_storage || current.allocations.size()!=state_layout.allocations.size())
            throw std::invalid_argument("HIP distributed stage state layout changed");
        for(std::size_t i=0;i<current.allocations.size();++i) {
            const auto& a=current.allocations[i];const auto& b=state_layout.allocations[i];
            if(a.component!=b.component || a.state_flag!=b.state_flag || a.storage!=b.storage ||
                a.offset!=b.offset || a.bytes!=b.bytes || a.provisional!=b.provisional)
                throw std::invalid_argument("HIP distributed stage state allocation changed");
        }
    }
    void finish_step() {
        if(deferred) stage_check(hipMemcpyAsync(&status,reinterpret_cast<const void*>(scope.deferred_moe_status.arena_base),
            sizeof(status),hipMemcpyDeviceToHost,scope.stream),"read distributed stage D1");
        stage_check(hipStreamSynchronize(scope.stream),"complete distributed stage callback before ACK");
        guard.require_original();
        if(deferred) check_deferred_moe_status(status);
    }
    void drain_failure() noexcept {
        broken=true;active=false;
        if(resources.ple_lookup && resources.ple_prefetch)
            cancel_ple_lookup_prefetch(*resources.ple_lookup,*resources.ple_prefetch);
        if(hipSetDevice(guard.device_index())!=hipSuccess) std::terminate();
        guard.recover_or_fail_stop();
    }
    void begin(const HipDistributedStageToken& value) {
        if(broken || active) throw std::logic_error("HIP distributed stage is failed or already active");
        original_device();validate_scope();validate_metadata();
        if(!value.request.connection_epoch || !value.request.request_id || !value.request.generation || !value.request.dtype_id ||
            value.input_token>=serial.vocab_size || value.position>=serial.max_position_embeddings ||
            value.position>=resources.session_state_layout->context_tokens ||
            (last_generation && (request_id!=value.request.request_id || epoch!=value.request.connection_epoch ||
                value.request.generation<=last_generation || value.position!=last_position+1U)))
            throw std::invalid_argument("HIP distributed stage token controller binding is invalid or stale");
        if(resources.ple_prefetch && resources.ple_prefetch->active)
            throw std::invalid_argument("HIP distributed stage cannot borrow an active PLE lookup");
        auto generations=value.initial_state;
        const HipLaunchScope launch_scope(scope);
        bool local_position_prepared=false;
        for(std::uint32_t i=first;i<=last;++i) {
            const auto& step=serial.execution.steps[i];const auto& io=serial.io.steps[i];
            const auto& binding=serial.bindings.bindings[i];auto& c=contexts[i-first];c={};
            if(!generations.contains(step.reads_state)) throw std::invalid_argument("HIP distributed stage initial local state is incomplete");
            generations.publish_write(step.writes_state); // Overflow admission only; no actual publication.
            c.execution_identity=serial.execution_identity;c.step_id=step.id;c.op=step.op;c.layer=step.layer;
            c.reads=io.reads;c.writes=io.writes;c.temporaries=io.temporaries;c.arena=&serial.arena;
            c.memory=resources.scratch;c.resources=&resources;c.session_state_layout=resources.session_state_layout;
            c.session_state_memory=resources.session_state_memory;c.reads_state=step.reads_state;c.writes_state=step.writes_state;
            c.position=value.position;c.input_token_id=value.input_token;
            if(!binding.host_or_runtime_step) c.kernel_id=*binding.kernel_id;
            const auto& l=launch.steps[i];
            if(l.step_id!=step.id || l.op!=step.op || l.runtime_step!=binding.host_or_runtime_step ||
                !l.function || !l.preflight || !l.implementation_identity)
                throw std::invalid_argument("HIP distributed stage local launcher has no exact deep preflight");
            l.preflight(c,l.user_data);
            if(step.op==SerialOp::position_prepare) local_position_prepared=true;
            else if(has_state(step.reads_state,PersistentState::position_history) && !local_position_prepared) {
                const auto history=resolve_dispatch_state(c,SessionStateComponent::position_history);
                if(history.storage!=SessionStateStorage::host || !history.address || history.bytes<sizeof(std::uint64_t))
                    throw std::invalid_argument("HIP distributed stage requires a prepared host position word");
                std::uint64_t word=0;std::memcpy(&word,reinterpret_cast<const void*>(history.address),sizeof(word));
                if(word!=value.position) throw std::invalid_argument("HIP distributed stage controller did not materialize the peer position");
            }
        }
        token=value;local=value.initial_state;cursor=first;active=true;
    }
    void execute(const DistributedStepPlacement& incoming) {
        if(broken || !active) throw std::logic_error("HIP distributed stage has no healthy active token");
        original_device();validate_scope();
        try {
            validate_metadata();
            // remote is relative to the sending plan's origin. Receiver-side
            // control callbacks legitimately pass remote=true; target node and
            // exact serial step/op, not that origin-relative bit, authorize us.
            if(cursor>last || incoming.step_id!=cursor || incoming.op!=placement.steps[cursor].op ||
                incoming.node!=placement.local_node || incoming.placement_ambiguous)
                throw std::invalid_argument("HIP distributed stage callback is foreign, duplicated or out of order");
            const HipLaunchScope launch_scope(scope);const auto& c=contexts[cursor-first];
            if(c.op==SerialOp::ple && resources.ple_lookup && resources.ple_hash_contract && resources.ple_prefetch)
                start_ple_lookup_prefetch(*resources.ple_hash_contract,c.step_id,token.input_token,token.position,
                    *resources.model_manifest,*resources.model_weights,*resources.tensor_access,
                    *resources.session_state_layout,*resources.session_state_memory,*resources.ple_lookup,*resources.ple_prefetch);
            if(serial.bindings.bindings[cursor].host_or_runtime_step) dispatcher->dispatch_runtime_step(c);
            else dispatcher->dispatch_kernel(c);
            if(c.op==SerialOp::ple && resources.ple_prefetch && resources.ple_prefetch->active)
                throw std::logic_error("HIP distributed stage PLE consumer did not retire its lookup");
            finish_step(); // Returning this callback allows transport ACK: always device-complete.
            local.publish_write(c.writes_state);++cursor;
        } catch(...) {drain_failure();throw;}
    }
};
HipDistributedStage::HipDistributedStage(CompiledSerialExecution serial,DistributedPlacementPlan placement,
    std::uint32_t first,std::uint32_t last,ModelContract model,HipLauncherCatalogConfig config,
    const ExecutionRuntimeResources& resources,HipLaunchContext scope,std::shared_ptr<const void> owner)
    :impl_(std::make_unique<Impl>(std::move(serial),std::move(placement),first,last,std::move(model),
        std::move(config),resources,scope,std::move(owner))) {}
HipDistributedStage::~HipDistributedStage()=default;
void HipDistributedStage::begin_token(const HipDistributedStageToken& token) {impl_->begin(token);}
void HipDistributedStage::execute_local_step(const DistributedStepPlacement& step) {impl_->execute(step);}
HipDistributedStageResult HipDistributedStage::complete_stage() {
    if(impl_->broken || !impl_->active || impl_->cursor!=impl_->last+1U)
        throw std::logic_error("HIP distributed stage is not completely retired");
    impl_->original_device();impl_->validate_scope();impl_->validate_metadata();
    impl_->last_generation=impl_->token.request.generation;impl_->last_position=impl_->token.position;
    impl_->request_id=impl_->token.request.request_id;
    impl_->epoch=impl_->token.request.connection_epoch;impl_->active=false;
    return {impl_->token,impl_->local,impl_->last-impl_->first+1U};
}
void HipDistributedStage::abort() noexcept {
    // Local retirement is provisional; peer failure after complete_stage still
    // invalidates this owner. No global commit or rollback is invented here.
    if(impl_->active || impl_->token.request.generation) impl_->drain_failure();
}
bool HipDistributedStage::failed() const noexcept {return impl_->broken;}
std::vector<ExecutionBufferAddress> HipDistributedStage::completed_boundary_views(
    const DistributedExecutionBoundary& boundary) const {
    const auto& s=*impl_;
    s.original_device();s.validate_scope();s.validate_metadata();
    if(s.broken || !s.token.request.generation || boundary.source_node!=s.placement.local_node ||
        boundary.source_step<s.first || boundary.source_step>s.last || boundary.source_step>=s.cursor ||
        boundary.destination_step>=s.serial.execution.steps.size())
        throw std::invalid_argument("HIP distributed stage boundary producer has not completed");
    const auto found=std::find_if(s.placement.boundaries.begin(),s.placement.boundaries.end(),
        [&](const auto& b){return b.boundary_id==boundary.boundary_id && b.source_step==boundary.source_step &&
            b.destination_step==boundary.destination_step && b.source_node==boundary.source_node &&
            b.destination_node==boundary.destination_node && b.payloads==boundary.payloads && b.payload_bytes==boundary.payload_bytes;});
    if(found==s.placement.boundaries.end()) throw std::invalid_argument("HIP distributed stage boundary differs from its retained plan");
    std::vector<ExecutionBufferAddress> views;
    std::uint64_t bytes=0;
    for(const auto use:boundary.payloads) {
        const auto& writes=s.serial.io.steps[boundary.source_step].writes;
        const auto& reads=s.serial.io.steps[boundary.destination_step].reads;
        if(std::find(writes.begin(),writes.end(),use)==writes.end() || std::find(reads.begin(),reads.end(),use)==reads.end())
            throw std::invalid_argument("HIP distributed stage boundary is not an actual source write/destination read");
        const auto view=resolve_execution_buffer(s.serial.execution_identity,s.serial.arena,*s.resources.scratch,use);
        if(view.bytes>std::numeric_limits<std::uint64_t>::max()-bytes) throw std::overflow_error("HIP distributed stage boundary byte overflow");
        bytes+=view.bytes;views.push_back(view);
    }
    if(bytes!=boundary.payload_bytes) throw std::invalid_argument("HIP distributed stage boundary payload extent differs");
    return views;
}
} // namespace strix
