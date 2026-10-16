#include "strix/hip_distributed_transaction.hpp"
#include "strix/hip_distributed_stage_resources.hpp"
#include "strix/distributed_stage_state.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/session_snapshot_memory.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {
const ModelManifest& checked_manifest(const ExecutionRuntimeResources& r) {
    if(!r.model_manifest) throw std::invalid_argument("distributed transaction has no manifest");
    return *r.model_manifest;
}
const ExecutionTensorAccessPlan& checked_access(const ExecutionRuntimeResources& r) {
    if(!r.tensor_access) throw std::invalid_argument("distributed transaction has no tensor access plan");
    return *r.tensor_access;
}
bool transaction_layout_equal(const SerialSessionStateLayout& a,const SerialSessionStateLayout& b) {
    if(a.context_tokens!=b.context_tokens || a.device_alignment!=b.device_alignment || a.host_alignment!=b.host_alignment ||
        a.device_total_bytes!=b.device_total_bytes || a.host_total_bytes!=b.host_total_bytes || a.layout_identity!=b.layout_identity ||
        a.recurrent_storage!=b.recurrent_storage || a.allocations.size()!=b.allocations.size()) return false;
    for(std::size_t i=0;i<a.allocations.size();++i) {
        const auto& x=a.allocations[i];const auto& y=b.allocations[i];
        if(x.component!=y.component || x.state_flag!=y.state_flag || x.storage!=y.storage || x.offset!=y.offset ||
            x.bytes!=y.bytes || x.provisional!=y.provisional) return false;
    }
    return true;
}
void require_disjoint(std::uintptr_t a,std::uint64_t an,std::uintptr_t b,std::uint64_t bn) {
    if(!an || !bn) return;
    const auto maximum=std::numeric_limits<std::uintptr_t>::max();
    if(!a || !b || an>maximum-a || bn>maximum-b || (a<b+bn && b<a+an))
        throw std::invalid_argument("distributed transaction snapshot aliases a retained owner or overflows");
}
// Physical stage snapshot storage has its own type; never a forged full
// SerialSessionSnapshotPlan. The participant retains the allocator owner and
// drains its original device before this member is released.
struct StageSnapshotOwner {
    RuntimeMemoryAllocator* allocator=nullptr;
    DistributedStageSnapshotMemoryBinding memory;
    StageSnapshotOwner(const DistributedStageSnapshotPlan& plan,RuntimeMemoryAllocator& a,AllocationGenerationSource& generations)
        :allocator(&a) {
        memory.snapshot_plan_identity=plan.plan_identity;memory.allocation_generation=generations.next();
        memory.device_section_bytes=plan.device_bytes;memory.host_section_bytes=plan.host_bytes;
        try {
            if(plan.device_bytes) memory.device_section_base=a.allocate_device(plan.device_bytes,256U);
            if(plan.host_bytes) memory.host_section_base=a.allocate_host(plan.host_bytes,256U);
            validate_stage_snapshot_binding(plan,memory);
        } catch(...) {release();throw;}
    }
    StageSnapshotOwner(const StageSnapshotOwner&)=delete;
    StageSnapshotOwner& operator=(const StageSnapshotOwner&)=delete;
    ~StageSnapshotOwner() {release();}
    void release() noexcept {
        if(!allocator) return;
        if(memory.host_section_base) allocator->free_host(memory.host_section_base,memory.host_section_bytes);
        if(memory.device_section_base) allocator->free_device(memory.device_section_base,memory.device_section_bytes);
        allocator=nullptr;
    }
};
bool transaction_state_equal(const PersistentStateGenerations& a,const PersistentStateGenerations& b) {
    return a.valid==b.valid && a.generations==b.generations;
}
}
struct HipDistributedTokenParticipant::Impl {
    std::shared_ptr<const void> owner;
    HipSetupQuiescenceGuard guard;
    CompiledSerialExecution serial;
    DistributedPlacementPlan plan;
    ModelContract model;
    HipLauncherCatalogConfig config;
    ExecutionRuntimeResources resources;
    SerialSessionContext* ordinary;
    std::shared_ptr<HipDistributedStageResources> stage_owner;
    SerialSessionContext saved;
    HipLaunchContext scope;
    std::uint32_t node;
    DistributedTokenTransaction protocol;
    RuntimeExecutionIdentities identities;
    SerialSessionStateLayout retained_layout;
    SessionStateAddress position_word;
    std::unique_ptr<const PreparedTensorViews> metadata;
    SerialSessionSnapshotPlan snapshot;
    SerialSessionRestorePlan restore;
    SessionSnapshotMemoryOwner snapshot_memory;
    SessionSnapshotMemoryBinding active_snapshot_memory;
    std::optional<DistributedStageSnapshotPlan> stage_snapshot;
    std::unique_ptr<StageSnapshotOwner> stage_snapshot_memory;
    DistributedStageSnapshotMemoryBinding active_stage_snapshot_memory;
    std::array<HipRuntimeAllocationDescription,2> snapshot_inventory_leases{};
    std::uint64_t snapshot_inventory_generation=0;
    std::optional<DistributedSequenceFrontier> sequence;
    HipSessionSnapshotTransferBackend transfer;
    std::unique_ptr<HipDistributedStage> stage;
    bool prepared=false,ready=false,restored=false,native_failed=false,quarantined=false,rollback_published=false,position_materialized=false;
    bool position_completion_issued=false;
    std::uint64_t last_sent=0;
    std::optional<DistributedTokenMessage> last_emitted;

    Impl(CompiledSerialExecution source,DistributedPlacementPlan placement,ModelContract contract,HipLauncherCatalogConfig launch_config,
        const ExecutionRuntimeResources& r,SerialSessionContext* session,HipLaunchContext native,std::shared_ptr<const void> retained,
        RuntimeMemoryAllocator& allocator,AllocationGenerationSource& generations,DistributedTokenBinding binding,
        std::uint32_t local_node,std::uint64_t budget,std::shared_ptr<HipDistributedStageResources> scoped={})
        :owner(std::move(retained)),serial(std::move(source)),plan(std::move(placement)),model(std::move(contract)),
         config(std::move(launch_config)),resources(r),ordinary(session),stage_owner(std::move(scoped)),
         saved(session?*session:SerialSessionContext{}),scope(native),node(local_node),
         protocol(serial,plan,checked_manifest(r),checked_access(r),std::move(binding)) {
        if(!owner || node>1U || !budget || !resources.model_manifest || !resources.model_weights || !resources.tensor_access ||
            !resources.session_state_layout || !resources.session_state_memory || !resources.scratch)
            throw std::invalid_argument("distributed native transaction requires retained complete local resources");
        // Reject foreign/nested capture before allocating either snapshot kind.
        if(!scope.stream || scope.stream==hipStreamLegacy || scope.stream==hipStreamPerThread)
            throw std::invalid_argument("distributed transaction requires its owned eager stream before snapshot allocation");
        original();
        retained_layout=*resources.session_state_layout;
        position_word=resolve_session_state(*resources.session_state_layout,*resources.session_state_memory,SessionStateComponent::position_history);
        if(position_word.storage!=SessionStateStorage::host || !position_word.address || position_word.bytes<sizeof(std::uint64_t) ||
            position_word.bytes>std::numeric_limits<std::uintptr_t>::max()-position_word.address)
            throw std::invalid_argument("distributed transaction requires a complete bounded host input-position word");
        const auto& b=protocol.binding();const auto& peer=b.peers[node];
        identities=resources.distributed_stage?build_stage_runtime_execution_identities(serial.execution_identity,serial.arena,resources):
            build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        if(stage_owner) {
            if(ordinary || !resources.distributed_stage || node!=stage_owner->plan().local_node())
                throw std::invalid_argument("capacity participant has mismatched typed owners");
            stage_owner->validate_initial_binding(b);
        } else {
            if(!ordinary || resources.distributed_stage) throw std::invalid_argument("ordinary participant requires the original complete serial anchor");
        if(ordinary->health!=SerialSessionHealth::ready || ordinary->arithmetic_lineage.tainted ||
            !arithmetic_lineage_valid(ordinary->arithmetic_lineage,b.position) ||
            ordinary->arithmetic_lineage.history_identity!=b.previous_history_identity ||
            ordinary->execution_identity!=serial.execution_identity || ordinary->next_position!=b.position ||
            ordinary->committed_tokens!=b.committed_tokens || ordinary->runtime_compatibility_identity!=identities.compatibility_identity ||
            ordinary->state_layout_identity!=resources.session_state_layout->layout_identity ||
            peer.runtime_compatibility_identity!=identities.compatibility_identity || peer.state_layout_identity!=ordinary->state_layout_identity ||
            peer.context_capacity!=resources.session_state_layout->context_tokens ||
            !transaction_state_equal(ordinary->state,peer.initial_state) ||
            resources.model_weights->model_content_identity!=b.model_content_identity)
            throw std::invalid_argument("distributed native transaction does not bind the original physical frontier");
        }
        metadata=std::make_unique<const PreparedTensorViews>(serial.execution_identity,serial.execution,
            *resources.model_manifest,*resources.model_weights,*resources.tensor_access,owner,resources.distributed_stage);
        SerialSessionSnapshotRequest request;request.compiled_execution_identity=serial.execution_identity;
        request.runtime_compatibility_identity=identities.compatibility_identity;request.next_position=b.position;
        request.committed_tokens=b.committed_tokens;request.state=b.peers[node].initial_state;
        if(const auto* mapping=resources.session_state_memory->stage_mapping) {
            if(!stage_owner) throw std::invalid_argument("local state snapshot requires its actual typed stage owner");
            stage_snapshot=build_stage_snapshot_plan(*mapping,protocol.binding_identity());
            if(stage_snapshot->device_bytes>std::numeric_limits<std::uint64_t>::max()-stage_snapshot->host_bytes ||
                stage_snapshot->device_bytes+stage_snapshot->host_bytes>budget)
                throw std::invalid_argument("distributed native transaction local snapshot exceeds budget");
            stage_snapshot_memory=std::make_unique<StageSnapshotOwner>(*stage_snapshot,allocator,generations);
            active_stage_snapshot_memory=stage_snapshot_memory->memory;
        } else {
        snapshot=build_serial_session_snapshot_plan(model,*resources.session_state_layout,request);
        if(snapshot.ranges.size()!=7U || snapshot.device_bytes>std::numeric_limits<std::uint64_t>::max()-snapshot.host_bytes ||
            snapshot.device_bytes+snapshot.host_bytes>budget)
            throw std::invalid_argument("distributed native transaction requires a budgeted complete seven-state snapshot");
        std::array<bool,7> components{};
        for(const auto& range:snapshot.ranges) {
            const auto index=static_cast<std::size_t>(range.component);
            const auto* allocation=find_session_state_allocation(*resources.session_state_layout,range.component);
            if(index>=components.size() || components[index] || !allocation || range.bytes!=allocation->bytes ||
                range.storage!=allocation->storage || range.source_offset!=allocation->offset)
                throw std::invalid_argument("distributed native transaction snapshot omits or truncates a state component");
            components[index]=true;
        }
        restore=build_serial_session_restore_plan(model,*resources.session_state_layout,identities.compatibility_identity,snapshot);
        // Allocate both snapshot sections before any target state mutation.
        snapshot_memory=SessionSnapshotMemoryOwner::allocate(snapshot,allocator,generations);
        active_snapshot_memory=snapshot_memory.binding();
        validate_snapshot_memory_binding(snapshot,active_snapshot_memory);
        }
        // Only physical addresses/bytes are shared for alias admission. Types
        // and validation remain distinct for full serial and local snapshots.
        const auto device_section=stage_snapshot?
            std::pair{active_stage_snapshot_memory.device_section_base,active_stage_snapshot_memory.device_section_bytes}:
            std::pair{active_snapshot_memory.device_section_base,active_snapshot_memory.device_section_bytes};
        const auto host_section=stage_snapshot?
            std::pair{active_stage_snapshot_memory.host_section_base,active_stage_snapshot_memory.host_section_bytes}:
            std::pair{active_snapshot_memory.host_section_base,active_snapshot_memory.host_section_bytes};
        if(stage_owner) {
            auto* actual_allocator=dynamic_cast<HipRuntimeAllocator*>(&allocator);
            if(!actual_allocator || actual_allocator!=&stage_owner->snapshot_allocator())
                throw std::invalid_argument("stage snapshot inventory requires its actual HIP allocator");
            if(device_section.second) snapshot_inventory_leases[0]=actual_allocator->describe_allocation(
                device_section.first,device_section.second,HipRuntimeAllocationSpace::device);
            if(host_section.second) snapshot_inventory_leases[1]=actual_allocator->describe_allocation(
                host_section.first,host_section.second,HipRuntimeAllocationSpace::host);
            snapshot_inventory_generation=stage_snapshot?active_stage_snapshot_memory.allocation_generation:
                active_snapshot_memory.allocation_generation;
        }
        require_disjoint(device_section.first,device_section.second,host_section.first,host_section.second);
        for(const auto& section:std::array<std::pair<std::uintptr_t,std::uint64_t>,2>{{device_section,host_section}}) {
            require_disjoint(section.first,section.second,resources.session_state_memory->device_base,resources.session_state_memory->device_bytes);
            require_disjoint(section.first,section.second,resources.session_state_memory->host_base,resources.session_state_memory->host_bytes);
            require_disjoint(section.first,section.second,resources.scratch->arena_base,resources.scratch->arena_bytes);
            require_disjoint(section.first,section.second,scope.deferred_moe_status.arena_base,scope.deferred_moe_status.arena_bytes);
            for(const auto& weight:resources.model_weights->tensors)
                if(!resources.distributed_stage || resources.distributed_stage->requires_tensor(weight.tensor_index))
                    require_disjoint(section.first,section.second,weight.address,weight.bytes);
        }
        std::vector<std::uint32_t> nodes;for(const auto& step:plan.steps) nodes.push_back(step.node);
        const auto local_plan=build_distributed_placement_plan(serial,*resources.model_manifest,*resources.tensor_access,nodes,node);
        stage=std::make_unique<HipDistributedStage>(serial,local_plan,peer.first_step,peer.last_step,model,config,resources,scope,owner);
    }
    ~Impl() {
        if(hipSetDevice(guard.device_index())!=hipSuccess) std::terminate();
        if(stage && protocol.phase()!=DistributedTokenPhase::committed && protocol.phase()!=DistributedTokenPhase::rolled_back) stage->abort();
        guard.recover_or_fail_stop();
        // Snapshot release here proves only local allocation quiescence. An
        // in-doubt request remains quarantined; there is no crash recovery claim.
    }
    void original() const {
        if(current_hip_launch_context()) throw std::invalid_argument("distributed transaction cannot nest a HIP scope");
        if(hipSetDevice(guard.device_index())!=hipSuccess) throw std::runtime_error("distributed transaction original device unavailable");
        guard.require_original();
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        if(hipStreamIsCapturing(scope.stream,&capture)!=hipSuccess || capture!=hipStreamCaptureStatusNone)
            throw std::invalid_argument("distributed transaction stream is captured or unavailable");
    }
    void validate() const {
        metadata->validate_source_metadata(*resources.model_manifest,*resources.model_weights,*resources.tensor_access);
        if(stage_owner) {
            stage_owner->validate();
            if(!transaction_layout_equal(retained_layout,*resources.session_state_layout))
                throw std::invalid_argument("capacity participant full-state layout changed");
        } else {
        if(!transaction_layout_equal(retained_layout,*resources.session_state_layout) || !ordinary ||
            ordinary->execution_identity!=saved.execution_identity || ordinary->next_position!=saved.next_position ||
            ordinary->committed_tokens!=saved.committed_tokens || ordinary->state_layout_identity!=saved.state_layout_identity ||
            ordinary->runtime_compatibility_identity!=saved.runtime_compatibility_identity ||
            !transaction_state_equal(ordinary->state,saved.state) ||
            ordinary->arithmetic_lineage.intervals!=saved.arithmetic_lineage.intervals ||
            ordinary->arithmetic_lineage.history_identity!=saved.arithmetic_lineage.history_identity ||
            ordinary->health!=(quarantined?SerialSessionHealth::poisoned:saved.health) ||
            ordinary->arithmetic_lineage.tainted!=(quarantined?true:saved.arithmetic_lineage.tainted))
            throw std::invalid_argument("distributed transaction layout or quarantined session frontier changed");
        }
        const auto current_word=resolve_session_state(*resources.session_state_layout,*resources.session_state_memory,SessionStateComponent::position_history);
        if(current_word.address!=position_word.address || current_word.bytes!=position_word.bytes ||
            current_word.storage!=position_word.storage || current_word.allocation_generation!=position_word.allocation_generation)
            throw std::invalid_argument("distributed transaction input-position owner changed");
        const auto now=resources.distributed_stage?build_stage_runtime_execution_identities(serial.execution_identity,serial.arena,resources):
            build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        if(now.compatibility_identity!=identities.compatibility_identity || now.binding_identity!=identities.binding_identity)
            throw std::invalid_argument("distributed transaction physical owner binding changed");
    }
    void quarantine() noexcept {if(ordinary) {ordinary->health=SerialSessionHealth::poisoned;ordinary->arithmetic_lineage.tainted=true;}quarantined=true;}
    void uncertain() noexcept {
        protocol.transport_failed();quarantine();
        if(stage) stage->abort();
    }
    DistributedTokenMessage message(DistributedTokenMessageKind kind,DistributedTokenReceipt receipt={},std::uint64_t decision=0) {
        DistributedTokenMessage m;m.binding=protocol.binding();m.binding_identity=protocol.binding_identity();
        m.sequence=protocol.next_sequence(node);m.sender_node=node;m.kind=kind;m.receipt=receipt;m.decision_identity=decision;
        try {protocol.observe(m);last_emitted=m;return m;}
        catch(...) {uncertain();throw;}
    }
    DistributedTokenReceipt snapshot_receipt(PersistentStateGenerations state,std::uint32_t completed=0) const {
        return stage_snapshot?DistributedTokenReceipt{stage_snapshot->plan_identity,active_stage_snapshot_memory.allocation_generation,state,completed}:
            DistributedTokenReceipt{snapshot.plan_identity,active_snapshot_memory.allocation_generation,state,completed};
    }
};
HipDistributedTokenParticipant::HipDistributedTokenParticipant(CompiledSerialExecution serial,DistributedPlacementPlan plan,
    ModelContract model,HipLauncherCatalogConfig config,const ExecutionRuntimeResources& resources,SerialSessionContext& session,
    HipLaunchContext scope,std::shared_ptr<const void> owner,RuntimeMemoryAllocator& allocator,AllocationGenerationSource& generations,
    DistributedTokenBinding binding,std::uint32_t node,std::uint64_t budget)
    :impl_(std::make_unique<Impl>(std::move(serial),std::move(plan),std::move(model),std::move(config),resources,&session,scope,
        std::move(owner),allocator,generations,std::move(binding),node,budget)) {}
HipDistributedTokenParticipant::HipDistributedTokenParticipant(std::shared_ptr<HipDistributedStageResources> owner,
    DistributedTokenBinding binding) {
    if(!owner) throw std::invalid_argument("capacity participant requires the actual stage resource owner");
    owner->validate_initial_binding(binding);
    impl_=std::make_unique<Impl>(owner->serial(),owner->plan().canonical_placement(),owner->model(),owner->config(),
        owner->resources(),nullptr,owner->scope(),owner,owner->snapshot_allocator(),owner->generations(),
        std::move(binding),owner->plan().local_node(),owner->snapshot_budget_bytes(),owner);
    owner->claim_initial(impl_->protocol.binding());
}
HipDistributedTokenParticipant::~HipDistributedTokenParticipant()=default;
void HipDistributedTokenParticipant::begin_next(std::uint32_t input,std::uint64_t generation,
    NativeDistributedTokenRetirement&& retirement) {
    auto& s=*impl_;
    const auto phase=s.protocol.phase();
    if(s.native_failed || (phase!=DistributedTokenPhase::committed &&
        !(phase==DistributedTokenPhase::rolled_back && s.rollback_published && (s.sequence || s.stage_owner))))
        throw std::logic_error("distributed continuation requires a committed or bilaterally restored distributed frontier");
    s.original();s.validate();
    const auto& current=s.protocol.binding();
    if(current.generation==std::numeric_limits<std::uint64_t>::max() || generation!=current.generation+1U)
        throw std::invalid_argument("distributed continuation generation must advance exactly once");
    auto frontier=phase==DistributedTokenPhase::committed?s.protocol.committed_sequence_frontier():s.sequence;
    if(!frontier && !(s.stage_owner && phase==DistributedTokenPhase::rolled_back))
        throw std::logic_error("distributed continuation has no admitted typed frontier");
    auto binding=frontier?frontier->derive_next_binding(input,generation):current;
    if(!frontier) {
        // Healthy initial scoped rollback returns to the genuine initialized
        // p0 seed, never an ordinary serial session. The attempt still advances.
        if(input>=s.serial.vocab_size) throw std::invalid_argument("distributed retry input exceeds vocabulary");
        binding.input_token=input;binding.generation=generation;
    }
    DistributedTokenTransaction next(s.serial,s.plan,*s.resources.model_manifest,*s.resources.tensor_access,binding);
    SerialSessionSnapshotRequest request;
    request.compiled_execution_identity=s.serial.execution_identity;
    request.runtime_compatibility_identity=s.identities.compatibility_identity;
    request.next_position=binding.position;request.committed_tokens=binding.committed_tokens;
    request.state=binding.peers[s.node].initial_state;
    SerialSessionSnapshotPlan snapshot;
    SerialSessionRestorePlan restore;
    SessionSnapshotMemoryBinding memory;
    std::optional<DistributedStageSnapshotPlan> stage_snapshot;
    DistributedStageSnapshotMemoryBinding stage_memory;
    if(s.stage_snapshot) {
        const auto* mapping=s.resources.session_state_memory->stage_mapping;
        if(!mapping || !s.stage_snapshot_memory) throw std::invalid_argument("distributed local snapshot owner changed");
        stage_snapshot=build_stage_snapshot_plan(*mapping,next.binding_identity());
        if(stage_snapshot->mapping_identity!=s.stage_snapshot->mapping_identity ||
            stage_snapshot->device_bytes!=s.stage_snapshot->device_bytes || stage_snapshot->host_bytes!=s.stage_snapshot->host_bytes ||
            stage_snapshot->ranges!=s.stage_snapshot->ranges)
            throw std::invalid_argument("distributed continuation changed local snapshot capacity or ranges");
        stage_memory=s.stage_snapshot_memory->memory;
        stage_memory.snapshot_plan_identity=stage_snapshot->plan_identity;
        validate_stage_snapshot_binding(*stage_snapshot,stage_memory);
    } else {
    snapshot=build_serial_session_snapshot_plan(s.model,*s.resources.session_state_layout,request);
    // Logical snapshot metadata may change; the complete physical storage,
    // allocation generation and retained allocator owner never do. Rebinding is
    // admitted only for an identical canonical full-seven range layout.
    if(snapshot.policy!=SessionSnapshotPolicy::full_components || s.snapshot.policy!=snapshot.policy ||
        snapshot.ranges.size()!=7U || snapshot.ranges.size()!=s.snapshot.ranges.size() ||
        snapshot.device_bytes!=s.snapshot.device_bytes || snapshot.host_bytes!=s.snapshot.host_bytes ||
        snapshot.alignment!=s.snapshot.alignment || snapshot.state_layout_identity!=s.snapshot.state_layout_identity ||
        snapshot.recurrent_storage!=s.snapshot.recurrent_storage)
        throw std::invalid_argument("distributed continuation changed the retained snapshot capacity or format");
    for(std::size_t i=0;i<snapshot.ranges.size();++i) {
        const auto& a=snapshot.ranges[i];const auto& b=s.snapshot.ranges[i];
        if(a.component!=b.component || a.storage!=b.storage || a.source_offset!=b.source_offset ||
            a.destination_offset!=b.destination_offset || a.bytes!=b.bytes || a.provisional!=b.provisional)
            throw std::invalid_argument("distributed continuation changed a retained snapshot range");
    }
    memory=s.snapshot_memory.binding();memory.snapshot_plan_identity=snapshot.plan_identity;
    validate_snapshot_memory_binding(snapshot,memory);
    restore=build_serial_session_restore_plan(s.model,*s.resources.session_state_layout,
        s.identities.compatibility_identity,snapshot);
    }
    std::unique_ptr<HipDistributedStage> replacement;
    if(phase==DistributedTokenPhase::rolled_back) {
        // abort is sticky; never clear that latch. Build a genuinely new stage
        // using the same actual owners after the complete bilateral restore.
        std::vector<std::uint32_t> nodes;for(const auto& step:s.plan.steps) nodes.push_back(step.node);
        const auto local_plan=build_distributed_placement_plan(s.serial,*s.resources.model_manifest,*s.resources.tensor_access,nodes,s.node);
        const auto& peer=binding.peers[s.node];
        replacement=std::make_unique<HipDistributedStage>(s.serial,local_plan,peer.first_step,peer.last_step,
            s.model,s.config,s.resources,s.scope,s.owner);
    }
    const DistributedRuntimeRequest request_id{current.connection_epoch,current.request_id,current.generation,current.boundary_dtype_id};
    try {
        retirement.consume_matches(s.protocol.binding_identity(),current.placement_plan_identity,request_id,s.node,
            phase==DistributedTokenPhase::committed?NativeDistributedRetirementOutcome::completed_boundaries:
                NativeDistributedRetirementOutcome::aborted_attempt);
    } catch(...) {s.uncertain();throw;}
    // No fallible allocation follows consumed retirement proof. Replaced-stage
    // teardown retains its original-device quiescence/fail-stop contract.
    s.sequence=std::move(frontier);s.protocol=std::move(next);
    s.snapshot=std::move(snapshot);s.restore=std::move(restore);s.active_snapshot_memory=memory;
    s.stage_snapshot=std::move(stage_snapshot);s.active_stage_snapshot_memory=stage_memory;
    if(replacement) s.stage=std::move(replacement);
    s.prepared=false;s.ready=false;s.restored=false;s.rollback_published=false;s.position_materialized=false;
    s.position_completion_issued=false;
    s.last_sent=0;s.last_emitted.reset();
}
const DistributedTokenBinding& HipDistributedTokenParticipant::current_binding() const noexcept {return impl_->protocol.binding();}
std::uint64_t HipDistributedTokenParticipant::binding_identity() const noexcept {return impl_->protocol.binding_identity();}
std::uint32_t HipDistributedTokenParticipant::local_node() const noexcept {return impl_->node;}
const HipDistributedStageResources* HipDistributedTokenParticipant::stage_resources() const noexcept {return impl_->stage_owner.get();}
NativeDistributedPositionCompletion HipDistributedTokenParticipant::mint_position_dependency_completion() {
    auto& s=*impl_;
    s.original();s.validate();
    const auto committed=s.protocol.committed_frontier();
    if(s.position_completion_issued || !s.ready || s.native_failed || !committed ||
        s.protocol.phase()!=DistributedTokenPhase::committed || !s.protocol.both_ready() ||
        committed->binding_identity!=s.protocol.binding_identity() ||
        !committed->decision_identity || committed->decision_identity!=s.protocol.decision_identity())
        throw std::logic_error("distributed position completion requires an unissued two-ACK native commit");
    std::uint64_t position=0;
    std::memcpy(&position,reinterpret_cast<const void*>(s.position_word.address),sizeof(position));
    const auto& b=s.protocol.binding();
    if(position!=b.position)
        throw std::invalid_argument("distributed position completion word differs from committed input position");
    NativeDistributedPositionCompletion proof;
    for(const auto& boundary:s.plan.boundaries) {
        if(boundary.payload_bytes!=0U) continue;
        if(!is_materialized_input_position_dependency(s.plan,boundary) ||
            proof.count_==NativeDistributedPositionCompletion::maximum_dependencies)
            throw std::invalid_argument("distributed position completion has unsupported logical dependencies");
        for(std::size_t i=0;i<proof.count_;++i)
            if(proof.boundaries_[i]==boundary.boundary_id)
                throw std::invalid_argument("distributed position completion has duplicate logical dependencies");
        proof.boundaries_[proof.count_++]=boundary.boundary_id;
    }
    if(proof.count_==0U)
        throw std::invalid_argument("distributed position completion has no logical dependencies");
    proof.request_={b.connection_epoch,b.request_id,b.generation,b.boundary_dtype_id};
    proof.binding_=s.protocol.binding_identity();proof.placement_=b.placement_plan_identity;
    proof.decision_=committed->decision_identity;proof.position_=b.position;proof.node_=s.node;
    proof.owner_=s.owner;
    proof.valid_=true;
    s.position_completion_issued=true;
    return proof;
}
HipDistributedOwnerInventory HipDistributedTokenParticipant::owner_inventory(
    std::shared_ptr<const HipDistributedTokenParticipant> actual_owner,UmaMemoryObservation observation) const {
    const auto self=weak_from_this().lock();const auto& s=*impl_;
    if(!self || actual_owner.get()!=this || actual_owner.owner_before(self) || self.owner_before(actual_owner) ||
        !s.stage_owner || current_hip_launch_context())
        throw std::invalid_argument("participant inventory requires its exact shared stage-owned participant outside a HIP scope");
    s.guard.require_original();s.validate();
    auto* allocator=dynamic_cast<HipRuntimeAllocator*>(&s.stage_owner->snapshot_allocator());
    if(!allocator) throw std::invalid_argument("participant inventory requires its actual HIP snapshot allocator");
    if(s.stage_snapshot) validate_stage_snapshot_binding(*s.stage_snapshot,s.active_stage_snapshot_memory);
    else validate_snapshot_memory_binding(s.snapshot,s.active_snapshot_memory);
    const auto generation=s.stage_snapshot?s.active_stage_snapshot_memory.allocation_generation:s.active_snapshot_memory.allocation_generation;
    if(!generation || generation!=s.snapshot_inventory_generation)
        throw std::invalid_argument("participant inventory snapshot allocation generation changed");
    auto result=s.stage_owner->owner_inventory(s.stage_owner);
    auto views=std::move(result.retained.pinned_views);
    const std::shared_ptr<const void> pin=std::move(actual_owner);
    for(std::size_t i=0;i<2U;++i) {
        const auto address=s.stage_snapshot?(i?s.active_stage_snapshot_memory.host_section_base:s.active_stage_snapshot_memory.device_section_base):
            (i?s.active_snapshot_memory.host_section_base:s.active_snapshot_memory.device_section_base);
        const auto bytes=s.stage_snapshot?(i?s.active_stage_snapshot_memory.host_section_bytes:s.active_stage_snapshot_memory.device_section_bytes):
            (i?s.active_snapshot_memory.host_section_bytes:s.active_snapshot_memory.device_section_bytes);
        const auto& expected=s.snapshot_inventory_leases[i];
        if(!bytes) {
            if(address || expected.lease_identity) throw std::invalid_argument("participant inventory absent snapshot section changed");
            continue;
        }
        const auto actual=allocator->describe_allocation(address,bytes,i?HipRuntimeAllocationSpace::host:HipRuntimeAllocationSpace::device);
        if(actual!=expected) throw std::invalid_argument("participant inventory snapshot allocator lease changed");
        views.push_back({next_uma_owner_view_identity(),{HipRuntimeAllocator::inventory_issuer,actual.lease_identity,generation},
            UmaBackingKind::allocated_storage,i?"stage.snapshot.host":"stage.snapshot.device",actual.raw_bytes,
            actual.view_offset,actual.view_bytes,pin,i?UmaAddressSpace::host:UmaAddressSpace::device,address});
    }
    result.retained=build_uma_owner_inventory(views,std::move(observation));result.snapshot_included=true;
    std::erase(result.unobserved_categories,std::string("snapshot.unattached_participant"));
    return result;
}
std::optional<DistributedSequenceFrontier> HipDistributedTokenParticipant::frontier() const {
    if(impl_->native_failed || impl_->protocol.phase()==DistributedTokenPhase::in_doubt) return std::nullopt;
    const auto committed=impl_->protocol.committed_sequence_frontier();
    return committed?committed:impl_->sequence;
}

DistributedTokenMessage HipDistributedTokenParticipant::prepare() {
    auto& s=*impl_;
    if(s.prepared || s.protocol.phase()!=DistributedTokenPhase::preparing) throw std::logic_error("distributed participant cannot prepare twice");
    s.original();s.validate();
    const auto& b=s.protocol.binding();HipDistributedStageToken token;
    token.request={b.connection_epoch,b.request_id,b.generation,b.boundary_dtype_id};token.position=b.position;token.input_token=b.input_token;
    token.initial_state=b.peers[s.node].initial_state;
    const auto& peer=b.peers[s.node];
    token.controller_materializes_position=std::none_of(s.serial.execution.steps.begin()+peer.first_step,
        s.serial.execution.steps.begin()+peer.last_step+1U,[](const auto& step){return step.op==SerialOp::position_prepare;});
    s.stage->begin_token(token);
    try {
        s.guard.run([&] {
            // The snapshot backend uses blocking D2D copies; explicitly join
            // this owned nonblocking producer stream BEFORE the first copy.
            if(hipStreamSynchronize(s.scope.stream)!=hipSuccess)
                throw std::runtime_error("distributed snapshot producer fence failed");
            if(s.stage_snapshot) capture_stage_snapshot_payload(*s.stage_snapshot,*s.resources.session_state_layout,
                *s.resources.session_state_memory,s.active_stage_snapshot_memory,s.transfer);
            else capture_serial_session_snapshot(s.snapshot,*s.resources.session_state_layout,*s.resources.session_state_memory,
                s.active_snapshot_memory,s.transfer);
        });
        // Both peers must publish prepared before either can mutate the target.
        // Quarantine the ordinary serial entry before publishing that receipt.
        s.quarantine();s.prepared=true;
        return s.message(DistributedTokenMessageKind::prepared,s.snapshot_receipt(token.initial_state));
    } catch(...) {s.native_failed=true;s.uncertain();throw;}
}
void HipDistributedTokenParticipant::execute_local_step(const DistributedStepPlacement& step) {
    auto& s=*impl_;
    if(!s.protocol.both_prepared() || s.protocol.phase()!=DistributedTokenPhase::executing || s.ready || s.restored)
        throw std::logic_error("distributed target execution requires both complete snapshot preparations");
    s.original(); // Pure scope/capture rejection must not drain a foreign capture.
    try {
        s.validate();
        // All stage callbacks were preflighted by begin_token before snapshot.
        // Only after BOTH complete snapshot receipts may this input word change.
        if(!s.position_materialized) {
            const auto& peer=s.protocol.binding().peers[s.node];
            const bool suffix=std::none_of(s.serial.execution.steps.begin()+peer.first_step,
                s.serial.execution.steps.begin()+peer.last_step+1U,[](const auto& x){return x.op==SerialOp::position_prepare;});
            if(suffix) {
                const auto word=resolve_session_state(*s.resources.session_state_layout,*s.resources.session_state_memory,
                    SessionStateComponent::position_history);
                if(word.storage!=SessionStateStorage::host || !word.address || word.bytes<sizeof(std::uint64_t))
                    throw std::invalid_argument("distributed transaction requires a complete host input-position word");
                const auto position=s.protocol.binding().position;
                std::memcpy(reinterpret_cast<void*>(word.address),&position,sizeof(position));
            }
            s.position_materialized=true;
        }
        s.stage->execute_local_step(step);
    } catch(...) {s.native_failed=true;s.uncertain();throw;}
}
DistributedTokenMessage HipDistributedTokenParticipant::finish_local() {
    auto& s=*impl_;
    if(!s.prepared || s.ready || s.protocol.phase()!=DistributedTokenPhase::executing)
        throw std::logic_error("distributed local readiness is outside execution phase");
    s.original();
    try {
        const auto result=s.stage->complete_stage();s.ready=true;
        return s.message(DistributedTokenMessageKind::ready,s.snapshot_receipt(result.local_state,result.completed_steps));
    } catch(...) {s.native_failed=true;s.uncertain();throw;}
}
void HipDistributedTokenParticipant::observe_peer(const DistributedTokenMessage& m) {
    auto& s=*impl_;s.original();
    try {
        if(m.sender_node==s.node) throw std::invalid_argument("distributed peer message claims our local sender");
        s.protocol.observe(m);
    } catch(...) {s.uncertain();throw;}
}
DistributedTokenMessage HipDistributedTokenParticipant::decide_commit() {
    auto& s=*impl_;s.original();return s.message(DistributedTokenMessageKind::commit_decision,{},s.protocol.expected_commit_decision());
}
DistributedTokenMessage HipDistributedTokenParticipant::decide_abort() {impl_->original();return impl_->message(DistributedTokenMessageKind::abort_decision);}
DistributedTokenMessage HipDistributedTokenParticipant::acknowledge_commit() {impl_->original();return impl_->message(DistributedTokenMessageKind::committed,{},impl_->protocol.expected_commit_decision());}
DistributedTokenMessage HipDistributedTokenParticipant::restore_local() {
    auto& s=*impl_;
    if(s.protocol.phase()!=DistributedTokenPhase::abort_decided || !s.prepared || s.restored)
        throw std::logic_error("distributed rollback requires the agreed precommit abort decision");
    s.original();s.validate();s.stage->abort();
    try {
        s.guard.run([&] {
            if(s.stage_snapshot) restore_stage_snapshot_payload(*s.stage_snapshot,*s.resources.session_state_layout,
                *s.resources.session_state_memory,s.active_stage_snapshot_memory,s.transfer);
            else restore_serial_session_snapshot_payload(s.restore,*s.resources.session_state_layout,
                *s.resources.session_state_memory,s.active_snapshot_memory,s.transfer);
        });
        s.validate();s.restored=true;
        // No original history/session publication until BOTH restored receipts.
        return s.message(DistributedTokenMessageKind::restored,s.snapshot_receipt(s.protocol.binding().peers[s.node].initial_state));
    } catch(...) {s.native_failed=true;s.uncertain();throw;}
}
void HipDistributedTokenParticipant::finalize_rollback() {
    auto& s=*impl_;
    if(s.protocol.phase()!=DistributedTokenPhase::rolled_back || !s.restored || s.native_failed || s.rollback_published)
        throw std::logic_error("distributed rollback cannot republish before both complete healthy restores");
    s.original();
    try {
        s.validate();
        s.guard.run([&] {
            if(s.scope.deferred_moe_status.arena_base) {
                DeferredMoeFirstError record{};
                if(hipMemcpy(&record,reinterpret_cast<const void*>(s.scope.deferred_moe_status.arena_base),sizeof(record),hipMemcpyDeviceToHost)!=hipSuccess)
                    throw std::runtime_error("distributed rollback D1 read failed");
                check_deferred_moe_status(record);
            }
        });
        if(!s.sequence && s.ordinary) {
            // Only the initial, never-distributed serial anchor is eligible.
            *s.ordinary=std::move(s.saved);s.quarantined=false;
        }
        // With a previous distributed commit the typed sequence stays intact;
        // all seven local bytes were restored but ordinary remains quarantined.
        s.rollback_published=true;
    } catch(...) {s.native_failed=true;s.uncertain();throw;}
}
void HipDistributedTokenParticipant::send(FramedDistributedTokenWire& wire,const DistributedTokenMessage& m) {
    auto& s=*impl_;s.original();
    try {
        if(!s.last_emitted || m.sender_node!=s.node || m.binding_identity!=s.protocol.binding_identity() || m.sequence==std::numeric_limits<std::uint64_t>::max() ||
            m.sequence<=s.last_sent || m.sequence+1U!=s.protocol.next_sequence(s.node))
            throw std::invalid_argument("distributed send is not the most recent locally applied message");
        if(encode_distributed_token_message(m)!=encode_distributed_token_message(*s.last_emitted))
            throw std::invalid_argument("distributed send changed a locally applied receipt or decision");
        wire.send(m);s.last_sent=m.sequence;
    } catch(...) {s.uncertain();throw;}
}
void HipDistributedTokenParticipant::receive(FramedDistributedTokenWire& wire) {
    impl_->original();
    try {observe_peer(wire.receive());} catch(...) {impl_->uncertain();throw;}
}
std::vector<ExecutionBufferAddress> HipDistributedTokenParticipant::completed_boundary_views(const DistributedExecutionBoundary& b) const {
    impl_->original();impl_->validate();
    if(impl_->protocol.phase()!=DistributedTokenPhase::executing || !impl_->ready)
        throw std::logic_error("distributed transaction boundary is not a completed provisional stage");
    return impl_->stage->completed_boundary_views(b);
}
void HipDistributedTokenParticipant::communication_failed() noexcept {impl_->uncertain();}
DistributedTokenPhase HipDistributedTokenParticipant::phase() const noexcept {return impl_->protocol.phase();}
std::uint64_t HipDistributedTokenParticipant::decision_identity() const noexcept {return impl_->protocol.decision_identity();}
std::optional<DistributedTokenCommit> HipDistributedTokenParticipant::committed_frontier() const {return impl_->protocol.committed_frontier();}
} // namespace strix
