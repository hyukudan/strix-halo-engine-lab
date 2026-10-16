#include "strix/hip_distributed_transaction.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
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
    ExecutionRuntimeResources resources;
    SerialSessionContext& ordinary;
    SerialSessionContext saved,restore_shadow;
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
    HipSessionSnapshotTransferBackend transfer;
    std::unique_ptr<HipDistributedStage> stage;
    bool prepared=false,ready=false,restored=false,native_failed=false,quarantined=false,rollback_published=false,position_materialized=false;
    std::uint64_t last_sent=0;
    std::optional<DistributedTokenMessage> last_emitted;

    Impl(CompiledSerialExecution source,DistributedPlacementPlan placement,ModelContract contract,HipLauncherCatalogConfig config,
        const ExecutionRuntimeResources& r,SerialSessionContext& session,HipLaunchContext native,std::shared_ptr<const void> retained,
        RuntimeMemoryAllocator& allocator,AllocationGenerationSource& generations,DistributedTokenBinding binding,
        std::uint32_t local_node,std::uint64_t budget)
        :owner(std::move(retained)),serial(std::move(source)),plan(std::move(placement)),model(std::move(contract)),
         resources(r),ordinary(session),saved(session),restore_shadow(session),scope(native),node(local_node),
         protocol(serial,plan,checked_manifest(r),checked_access(r),std::move(binding)) {
        if(!owner || node>1U || !budget || !resources.model_manifest || !resources.model_weights || !resources.tensor_access ||
            !resources.session_state_layout || !resources.session_state_memory || !resources.scratch)
            throw std::invalid_argument("distributed native transaction requires retained complete local resources");
        retained_layout=*resources.session_state_layout;
        position_word=resolve_session_state(*resources.session_state_layout,*resources.session_state_memory,SessionStateComponent::position_history);
        if(position_word.storage!=SessionStateStorage::host || !position_word.address || position_word.bytes<sizeof(std::uint64_t) ||
            position_word.bytes>std::numeric_limits<std::uintptr_t>::max()-position_word.address)
            throw std::invalid_argument("distributed transaction requires a complete bounded host input-position word");
        const auto& b=protocol.binding();const auto& peer=b.peers[node];
        identities=build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        if(ordinary.health!=SerialSessionHealth::ready || ordinary.arithmetic_lineage.tainted ||
            !arithmetic_lineage_valid(ordinary.arithmetic_lineage,b.position) ||
            ordinary.arithmetic_lineage.history_identity!=b.previous_history_identity ||
            ordinary.execution_identity!=serial.execution_identity || ordinary.next_position!=b.position ||
            ordinary.committed_tokens!=b.committed_tokens || ordinary.runtime_compatibility_identity!=identities.compatibility_identity ||
            ordinary.state_layout_identity!=resources.session_state_layout->layout_identity ||
            peer.runtime_compatibility_identity!=identities.compatibility_identity || peer.state_layout_identity!=ordinary.state_layout_identity ||
            peer.context_capacity!=resources.session_state_layout->context_tokens ||
            !transaction_state_equal(ordinary.state,peer.initial_state) ||
            resources.model_weights->model_content_identity!=b.model_content_identity)
            throw std::invalid_argument("distributed native transaction does not bind the original physical frontier");
        metadata=std::make_unique<const PreparedTensorViews>(serial.execution_identity,serial.execution,
            *resources.model_manifest,*resources.model_weights,*resources.tensor_access,owner);
        SerialSessionSnapshotRequest request;request.compiled_execution_identity=serial.execution_identity;
        request.runtime_compatibility_identity=identities.compatibility_identity;request.next_position=b.position;
        request.committed_tokens=b.committed_tokens;request.state=ordinary.state;
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
        validate_snapshot_memory_binding(snapshot,snapshot_memory.binding());
        const auto& snap=snapshot_memory.binding();
        require_disjoint(snap.device_section_base,snap.device_section_bytes,snap.host_section_base,snap.host_section_bytes);
        for(const auto& section:std::array<std::pair<std::uintptr_t,std::uint64_t>,2>{{
                {snap.device_section_base,snap.device_section_bytes},{snap.host_section_base,snap.host_section_bytes}}}) {
            require_disjoint(section.first,section.second,resources.session_state_memory->device_base,resources.session_state_memory->device_bytes);
            require_disjoint(section.first,section.second,resources.session_state_memory->host_base,resources.session_state_memory->host_bytes);
            require_disjoint(section.first,section.second,resources.scratch->arena_base,resources.scratch->arena_bytes);
            require_disjoint(section.first,section.second,scope.deferred_moe_status.arena_base,scope.deferred_moe_status.arena_bytes);
            for(const auto& weight:resources.model_weights->tensors)
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
        if(!transaction_layout_equal(retained_layout,*resources.session_state_layout) ||
            ordinary.execution_identity!=saved.execution_identity || ordinary.next_position!=saved.next_position ||
            ordinary.committed_tokens!=saved.committed_tokens || ordinary.state_layout_identity!=saved.state_layout_identity ||
            ordinary.runtime_compatibility_identity!=saved.runtime_compatibility_identity ||
            !transaction_state_equal(ordinary.state,saved.state) ||
            ordinary.arithmetic_lineage.intervals!=saved.arithmetic_lineage.intervals ||
            ordinary.arithmetic_lineage.history_identity!=saved.arithmetic_lineage.history_identity ||
            ordinary.health!=(quarantined?SerialSessionHealth::poisoned:saved.health) ||
            ordinary.arithmetic_lineage.tainted!=(quarantined?true:saved.arithmetic_lineage.tainted))
            throw std::invalid_argument("distributed transaction layout or quarantined session frontier changed");
        const auto current_word=resolve_session_state(*resources.session_state_layout,*resources.session_state_memory,SessionStateComponent::position_history);
        if(current_word.address!=position_word.address || current_word.bytes!=position_word.bytes ||
            current_word.storage!=position_word.storage || current_word.allocation_generation!=position_word.allocation_generation)
            throw std::invalid_argument("distributed transaction input-position owner changed");
        const auto now=build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        if(now.compatibility_identity!=identities.compatibility_identity || now.binding_identity!=identities.binding_identity)
            throw std::invalid_argument("distributed transaction physical owner binding changed");
    }
    void quarantine() noexcept {ordinary.health=SerialSessionHealth::poisoned;ordinary.arithmetic_lineage.tainted=true;quarantined=true;}
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
        return {snapshot.plan_identity,snapshot_memory.binding().allocation_generation,state,completed};
    }
};
HipDistributedTokenParticipant::HipDistributedTokenParticipant(CompiledSerialExecution serial,DistributedPlacementPlan plan,
    ModelContract model,HipLauncherCatalogConfig config,const ExecutionRuntimeResources& resources,SerialSessionContext& session,
    HipLaunchContext scope,std::shared_ptr<const void> owner,RuntimeMemoryAllocator& allocator,AllocationGenerationSource& generations,
    DistributedTokenBinding binding,std::uint32_t node,std::uint64_t budget)
    :impl_(std::make_unique<Impl>(std::move(serial),std::move(plan),std::move(model),std::move(config),resources,session,scope,
        std::move(owner),allocator,generations,std::move(binding),node,budget)) {}
HipDistributedTokenParticipant::~HipDistributedTokenParticipant()=default;
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
            capture_serial_session_snapshot(s.snapshot,*s.resources.session_state_layout,*s.resources.session_state_memory,
                s.snapshot_memory.binding(),s.transfer);
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
        s.guard.run([&] {restore_serial_session_snapshot(s.restore_shadow,s.restore,*s.resources.session_state_layout,
            *s.resources.session_state_memory,s.snapshot_memory.binding(),s.transfer);});
        s.validate();s.restored=true;
        // No original history/session publication until BOTH restored receipts.
        return s.message(DistributedTokenMessageKind::restored,s.snapshot_receipt(s.saved.state));
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
        s.ordinary=std::move(s.saved);s.quarantined=false;s.rollback_published=true;
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
