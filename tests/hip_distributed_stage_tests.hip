// Opt-in real-model COMPONENT fixture. No network, peer, transport transfer,
// distributed token commit, latency measurement or sharded-memory claim.
#include "strix/hip_distributed_stage.hpp"
#include "strix/hip_distributed_transaction.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/serial_session.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const char* text) {if(!value) throw std::runtime_error(text);}
void check(hipError_t value,const char* text) {if(value!=hipSuccess) throw std::runtime_error(text);}
std::string digest(const std::filesystem::path& path) {return strix::model_content_identity_hex(strix::sha256_file_content(path));}
struct Owner {
    strix::HipSetupQuiescenceGuard guard;
    strix::AllocationGenerationSource generations;
    hipStream_t stream=nullptr;
    strix::ModelContract model=strix::ModelContract::qwen38_flash_next();
    strix::ModelManifest manifest;
    strix::CompiledSerialExecution serial;
    strix::ExecutionTensorAccessPlan access;
    strix::SerialSessionStateLayout layout;
    std::shared_ptr<strix::MappedModelFile> mapped;
    std::shared_ptr<strix::HipModelResidency> resident;
    std::shared_ptr<strix::HipRuntimeAllocator> allocator;
    std::shared_ptr<strix::RuntimeExecutionMemoryOwner> memory;
    strix::FileMappedPleLookupBackend ple;
    strix::PleHashContract hash;
    strix::PleLookupPrefetchState prefetch;
    strix::ExecutionRuntimeResources resources;
    strix::ExecutionMemoryBinding status;
    strix::HipLauncherCatalogConfig config;
    ~Owner() {
        if(hipSetDevice(guard.device_index())!=hipSuccess) std::terminate();
        guard.recover_or_fail_stop();
        if(stream && hipStreamDestroy(stream)!=hipSuccess) std::terminate();
        if(status.arena_base) allocator->free_device(status.arena_base,status.arena_bytes);
    }
    strix::HipLaunchContext scope() const {
        strix::HipLaunchContext value;value.stream=stream;value.deferred_moe_status=status;return value;
    }
};
std::shared_ptr<Owner> load(const std::filesystem::path& path) {
    auto o=std::make_shared<Owner>();
    o->manifest=strix::build_model_manifest(strix::gguf::read_file(path));
    o->mapped=std::make_shared<strix::MappedModelFile>(strix::MappedModelFile::open_read_only(path,"model",o->generations));
    strix::ExecutionBindingRequest request;request.minimum_qualification=strix::kernels::QualificationStatus::experimental;
    request.deferred_moe_status=o->config.deferred_moe_status=true;
    strix::kernels::KernelRegistry variants;
    strix::register_complete_hip_kernel_catalog(o->model,request,o->config,variants);
    o->serial=strix::compile_serial_execution(o->model,variants,request,{},256);
    o->access=strix::compile_execution_tensor_access_plan(o->serial.execution,o->manifest,
        strix::build_model_tensor_index(o->manifest),strix::build_serial_execution_tensor_requirements(o->serial.execution));
    strix::SerialSessionStateLayoutRequest state;state.context_tokens=16;
    o->layout=strix::build_serial_session_state_layout(o->model,state);
    require(o->layout.allocations.size()==7U,"fixture requires all seven state components");
    o->resident=std::make_shared<strix::HipModelResidency>(strix::HipModelResidency::create(*o->mapped,o->manifest,o->generations));
    o->allocator=std::make_shared<strix::HipRuntimeAllocator>(strix::HipHostAllocationMode::pageable_aligned);
    o->memory=std::make_shared<strix::RuntimeExecutionMemoryOwner>(strix::RuntimeExecutionMemoryOwner::allocate(
        o->serial,o->layout,*o->allocator,o->generations));
    o->status={o->serial.execution_identity,o->allocator->allocate_device(8U,256U),8U,o->generations.next()};
    o->guard.run([&] {
        check(hipStreamCreateWithFlags(&o->stream,hipStreamNonBlocking),"create component stream");
        strix::initialize_hip_fresh_runtime_memory(o->memory->scratch(),o->memory->state());
        check(hipMemsetAsync(reinterpret_cast<void*>(o->status.arena_base),0,8U,o->stream),"initialize external D1 record");
    });
    o->hash=strix::build_ple_hash_contract(o->model);
    o->resources={&o->memory->scratch(),&o->manifest,&o->resident->binding(),&o->access,&o->layout,
        &o->memory->state(),&o->ple,&o->hash,&o->prefetch};
    return o;
}
// Two local peers share only immutable mapped/resident weights. Each keeps
// genuinely distinct seven-state, scratch, status, PLE and stream owners.
std::shared_ptr<Owner> clone_local(const std::shared_ptr<Owner>& source) {
    auto o=std::make_shared<Owner>();
    o->model=source->model;o->manifest=source->manifest;o->serial=source->serial;o->access=source->access;
    o->layout=source->layout;o->mapped=source->mapped;o->resident=source->resident;o->config=source->config;
    o->allocator=std::make_shared<strix::HipRuntimeAllocator>(strix::HipHostAllocationMode::pageable_aligned);
    o->memory=std::make_shared<strix::RuntimeExecutionMemoryOwner>(strix::RuntimeExecutionMemoryOwner::allocate(
        o->serial,o->layout,*o->allocator,o->generations));
    o->status={o->serial.execution_identity,o->allocator->allocate_device(8U,256U),8U,o->generations.next()};
    o->guard.run([&] {
        check(hipStreamCreateWithFlags(&o->stream,hipStreamNonBlocking),"create second local peer stream");
        strix::initialize_hip_fresh_runtime_memory(o->memory->scratch(),o->memory->state());
        check(hipMemsetAsync(reinterpret_cast<void*>(o->status.arena_base),0,8U,o->stream),"initialize second peer D1 record");
    });
    o->hash=strix::build_ple_hash_contract(o->model);
    o->resources={&o->memory->scratch(),&o->manifest,&o->resident->binding(),&o->access,&o->layout,
        &o->memory->state(),&o->ple,&o->hash,&o->prefetch};
    return o;
}
strix::SerialSessionContext fresh_session(const Owner& o) {
    strix::SerialSessionContext s;s.execution_identity=o.serial.execution_identity;
    s.state_layout_identity=o.layout.layout_identity;
    s.runtime_compatibility_identity=strix::build_runtime_execution_identities(o.serial.execution_identity,o.serial.arena,o.resources).compatibility_identity;
    for(const auto& a:o.layout.allocations) s.state.valid=s.state.valid|a.state_flag;
    return s;
}
struct Snapshot {
    std::vector<std::byte> scratch,device,host;
};
std::vector<std::byte> read(std::uintptr_t address,std::uint64_t bytes,bool device) {
    require(bytes<=std::numeric_limits<std::size_t>::max(),"snapshot host extent overflow");
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    if(bytes) {
        if(device) check(hipMemcpy(result.data(),reinterpret_cast<const void*>(address),result.size(),hipMemcpyDeviceToHost),"read component snapshot");
        else std::memcpy(result.data(),reinterpret_cast<const void*>(address),result.size());
    }
    return result;
}
Snapshot snapshot(const Owner& o) {
    check(hipStreamSynchronize(o.stream),"snapshot component fence");
    const auto& s=o.memory->state();return {read(o.memory->scratch().arena_base,o.memory->scratch().arena_bytes,true),
        read(s.device_base,s.device_bytes,true),read(s.host_base,s.host_bytes,false)};
}
void restore(const Owner& o,const Snapshot& s) {
    require(!o.prefetch.active,"cannot restore an active PLE lookup");
    o.guard.run([&] {
        check(hipMemcpy(reinterpret_cast<void*>(o.memory->scratch().arena_base),s.scratch.data(),s.scratch.size(),hipMemcpyHostToDevice),"restore scratch");
        if(!s.device.empty()) check(hipMemcpy(reinterpret_cast<void*>(o.memory->state().device_base),s.device.data(),s.device.size(),hipMemcpyHostToDevice),"restore all device states");
        if(!s.host.empty()) std::memcpy(reinterpret_cast<void*>(o.memory->state().host_base),s.host.data(),s.host.size());
        check(hipMemsetAsync(reinterpret_cast<void*>(o.status.arena_base),0,8U,o.stream),"initialize fresh external status phase");
    });
}
struct Reference final : strix::EagerExecutionDispatcher {
    strix::DirectLaunchDispatcher direct;
    Owner& owner;std::uint32_t cut;Snapshot captured;bool observed=false;
    Reference(const strix::CompiledLaunchPlan& plan,Owner& o,std::uint32_t end):direct(plan),owner(o),cut(end) {}
    void after(const strix::ExecutionDispatchContext& c) {
        if(c.step_id==cut) {captured=snapshot(owner);observed=true;}
    }
    void dispatch_kernel(const strix::ExecutionDispatchContext& c) override {direct.dispatch_kernel(c);after(c);}
    void dispatch_runtime_step(const strix::ExecutionDispatchContext& c) override {direct.dispatch_runtime_step(c);after(c);}
};
class NoControl final : public strix::DistributedNativeControlPlane {
    void execute_remote_step(const strix::DistributedStepPlacement&) override {throw std::logic_error("no peer in component fixture");}
    void publish_boundary_transfer(const strix::DistributedExecutionBoundary&,const strix::DistributedBoundaryTransfer&) override {throw std::logic_error("no transport in component fixture");}
    strix::DistributedBoundaryTransfer await_boundary_transfer(const strix::DistributedExecutionBoundary&) override {throw std::logic_error("no transport in component fixture");}
};
class NoStaging final : public strix::DistributedBoundaryStagingProvider {
    strix::transport::TransportBufferView stage_outbound(const strix::DistributedExecutionBoundary&,const strix::DistributedBoundaryTransfer&) override {throw std::logic_error("no transport in component fixture");}
    strix::transport::TransportBufferView stage_inbound(const strix::DistributedExecutionBoundary&,const strix::DistributedBoundaryTransfer&) override {throw std::logic_error("no transport in component fixture");}
    void consume_inbound(const strix::DistributedExecutionBoundary&,const strix::transport::TransportBufferView&) override {throw std::logic_error("no transport in component fixture");}
    std::unique_ptr<strix::DistributedInboundCompletion> finish_inbound(const strix::DistributedExecutionBoundary&,const strix::DistributedBoundaryTransfer&) override {throw std::logic_error("no transport in component fixture");}
};
strix::DistributedPlacementPlan placement(const Owner& o,std::uint32_t cut) {
    std::vector<std::uint32_t> nodes(o.serial.execution.steps.size(),1U);
    for(std::uint32_t i=0;i<=cut;++i) nodes[i]=0U;
    auto p=strix::build_distributed_placement_plan(o.serial,o.manifest,o.access,nodes,0U);
    require(p.executable && !p.boundaries.empty(),"component cut has no derived real IO placement");
    return p;
}
enum class TransactionOutcome {rollback,commit,lost_decision};
void transaction_component(const std::shared_ptr<Owner>& first,const strix::SerialSessionContext& initial_session,
    std::uint32_t cut,std::uint32_t input,TransactionOutcome outcome) {
    auto second=clone_local(first);
    const auto before0=snapshot(*first);restore(*second,before0);
    const auto before1=snapshot(*second);
    auto session0=initial_session,session1=initial_session;
    session1.runtime_compatibility_identity=fresh_session(*second).runtime_compatibility_identity;
    const auto saved0=session0,saved1=session1;
    const auto plan=placement(*first,cut);
    strix::DistributedTokenBinding binding;
    binding.serial_execution_identity=first->serial.execution_identity;binding.kernel_plan_identity=first->serial.kernel_plan_identity;
    binding.placement_plan_identity=plan.plan_identity;binding.model_content_identity=first->resident->binding().model_content_identity;
    binding.request_id=17U;binding.connection_epoch=29U;binding.generation=1U;binding.input_token=input;binding.boundary_dtype_id=1U;
    binding.position=session0.next_position;binding.committed_tokens=session0.committed_tokens;
    binding.previous_history_identity=session0.arithmetic_lineage.history_identity;
    binding.initial_global_state=session0.state;
    for(std::uint32_t node=0;node<2U;++node) {
        const auto& o=node?*second:*first;const auto& session=node?session1:session0;
        auto& peer=binding.peers[node];peer.node=node;peer.first_step=node?cut+1U:0U;
        peer.last_step=node?static_cast<std::uint32_t>(o.serial.execution.steps.size()-1U):cut;
        peer.runtime_compatibility_identity=session.runtime_compatibility_identity;
        peer.state_layout_identity=session.state_layout_identity;peer.context_capacity=o.layout.context_tokens;peer.initial_state=session.state;
    }
    const auto budget=[](const Owner& o) {
        require(o.layout.device_total_bytes<=std::numeric_limits<std::uint64_t>::max()-o.layout.host_total_bytes,"snapshot budget overflow");
        // Snapshot sections include alignment padding; budget is the complete
        // existing owner extents plus one alignment quantum per component.
        const auto bytes=o.layout.device_total_bytes+o.layout.host_total_bytes;
        require(bytes<=std::numeric_limits<std::uint64_t>::max()-4096U,"snapshot padding budget overflow");
        return bytes+4096U;
    };
    strix::HipDistributedTokenParticipant p0(first->serial,plan,first->model,first->config,first->resources,session0,
        first->scope(),first,*first->allocator,first->generations,binding,0U,budget(*first));
    strix::HipDistributedTokenParticipant p1(second->serial,plan,second->model,second->config,second->resources,session1,
        second->scope(),second,*second->allocator,second->generations,binding,1U,budget(*second));
    auto capacity_binding=binding;capacity_binding.position=first->layout.context_tokens;
    bool capacity_rejected=false;
    try {
        strix::HipDistributedTokenParticipant invalid(first->serial,plan,first->model,first->config,first->resources,session0,
            first->scope(),first,*first->allocator,first->generations,capacity_binding,0U,budget(*first));
    } catch(const std::invalid_argument&) {capacity_rejected=true;}
    require(capacity_rejected && session0.health==strix::SerialSessionHealth::ready,
        "out-of-capacity transaction changed the original session before admission");
    const auto prepared0=p0.prepare(),prepared1=p1.prepare();p0.observe_peer(prepared1);p1.observe_peer(prepared0);
    require(session0.health==strix::SerialSessionHealth::poisoned && session1.health==strix::SerialSessionHealth::poisoned,
        "prepared physical frontiers were exposed to an ordinary serial caller");
    std::array<strix::transport::DirectTransportNativeLinkSession*,1> unavailable{nullptr};NoControl control;NoStaging staging;
    strix::NativeDistributedBackend backend(unavailable,control,p0,staging);
    for(std::uint32_t i=0;i<=cut;++i) backend.execute_local_step(plan.steps[i]);
    const auto ready=p0.finish_local();p1.observe_peer(ready);
    // Component-only local D2D transfer of the REAL derived arena boundary.
    // This deliberately does not exercise a physical/network transport.
    for(const auto& boundary:plan.boundaries) {
        const auto source=p0.completed_boundary_views(boundary);
        require(source.size()==boundary.payloads.size(),"transaction boundary arity changed");
        for(std::size_t i=0;i<source.size();++i) {
            const auto target=strix::resolve_execution_buffer(second->serial.execution_identity,second->serial.arena,
                second->memory->scratch(),boundary.payloads[i]);
            require(target.bytes==source[i].bytes,"transaction boundary source/destination extents differ");
            check(hipMemcpy(reinterpret_cast<void*>(target.address),reinterpret_cast<const void*>(source[i].address),
                static_cast<std::size_t>(target.bytes),hipMemcpyDeviceToDevice),"component local peer boundary transfer");
        }
    }
    strix::NativeDistributedBackend peer_backend(unavailable,control,p1,staging);
    for(std::uint32_t i=cut+1U;i<plan.steps.size();++i) peer_backend.execute_local_step(plan.steps[i]);
    const auto ready1=p1.finish_local();p0.observe_peer(ready1);
    const auto* position_allocation=strix::find_session_state_allocation(second->layout,strix::SessionStateComponent::position_history);
    require(position_allocation!=nullptr,"transaction fixture has no position word");
    std::uint64_t word=0;std::memcpy(&word,reinterpret_cast<const void*>(second->memory->state().host_base+
        static_cast<std::uintptr_t>(position_allocation->offset)),sizeof(word));
    require(word==binding.position,"suffix did not materialize the explicit position after both preparations");
    auto expected1=saved1.state;
    for(std::uint32_t i=cut+1U;i<plan.steps.size();++i) expected1.publish_write(first->serial.execution.steps[i].writes_state);
    require(ready1.receipt.local_state.generations==expected1.generations,
        "controller position materialization fabricated a generation");
    if(outcome!=TransactionOutcome::rollback) {
        const auto commit=p0.decide_commit();
        require(p0.decision_identity()!=0U && !p0.committed_frontier(),"native decision alone published global commit");
        if(outcome==TransactionOutcome::lost_decision) {
            const auto irreversible=p0.decision_identity();
            // Decision was made locally but deliberately never delivered.
            p0.communication_failed();p1.communication_failed();
            bool blocked0=false,blocked1=false;
            try {(void)p0.restore_local();} catch(const std::logic_error&) {blocked0=true;}
            try {(void)p1.restore_local();} catch(const std::logic_error&) {blocked1=true;}
            require(blocked0 && blocked1 && p0.phase()==strix::DistributedTokenPhase::in_doubt &&
                p1.phase()==strix::DistributedTokenPhase::in_doubt && p0.decision_identity()==irreversible &&
                session0.health==strix::SerialSessionHealth::poisoned && session1.health==strix::SerialSessionHealth::poisoned &&
                !p0.committed_frontier() && !p1.committed_frontier(),"lost decision allowed unilateral rollback/reuse or fabricated acknowledgement");
            return;
        }
        p1.observe_peer(commit);
        const auto ack0=p0.acknowledge_commit();p1.observe_peer(ack0);
        require(!p0.committed_frontier() && !p1.committed_frontier(),"one peer ACK published global commit");
        const auto ack1=p1.acknowledge_commit();p0.observe_peer(ack1);
        const auto done0=p0.committed_frontier(),done1=p1.committed_frontier();
        require(done0 && done1 && done0->decision_identity==done1->decision_identity &&
            done0->next_position==binding.position+1U && done1->history_identity==done0->history_identity &&
            done0->committed_tokens==binding.committed_tokens+1U &&
            done0->local_state[1].generations==ready1.receipt.local_state.generations &&
            session0.health==strix::SerialSessionHealth::poisoned && session1.health==strix::SerialSessionHealth::poisoned,
            "native global frontier did not require both receipts or retagged a partial peer as a healthy serial session");
        const auto sequence0=p0.frontier(),sequence1=p1.frontier();
        require(sequence0 && sequence1 && sequence0->commit().history_identity==done0->history_identity &&
            sequence1->commit().local_state[1].generations==ready1.receipt.local_state.generations &&
            p0.current_binding().generation==binding.generation && p0.binding_identity()==done0->binding_identity &&
            session0.next_position==saved0.next_position && session1.next_position==saved1.next_position &&
            session0.arithmetic_lineage.tainted && session1.arithmetic_lineage.tainted,
            "typed distributed frontier mutated or revalidated an ordinary partial-state serial session");
        if(binding.position+1U<first->layout.context_tokens) {
            const auto next=sequence0->derive_next_binding(input,binding.generation+1U);
            require(next.position==binding.position+1U && next.peers[0].initial_state.generations==done0->local_state[0].generations &&
                next.peers[1].initial_state.generations==done0->local_state[1].generations,
                "native-derived next logical frontier copied global generations into local peers");
        }
        // This local-D2D component cannot mint an actual transport retirement
        // capability. Full multi-token begin_next is exercised only by the
        // separate real-physical-setup sequence fixture, never a forged permit.
        strix::NativeDistributedTokenRetirement missing;
        const auto committed_bytes=snapshot(*first);
        bool invalid_next=false;
        try {p0.begin_next(static_cast<std::uint32_t>(first->model.vocab_size),binding.generation+1U,std::move(missing));}
        catch(const std::invalid_argument&) {invalid_next=true;}
        require(invalid_next && p0.phase()==strix::DistributedTokenPhase::committed && p0.frontier(),
            "invalid next input destroyed the previously committed distributed frontier");
        invalid_next=false;
        try {p0.begin_next(input,binding.generation,std::move(missing));}
        catch(const std::invalid_argument&) {invalid_next=true;}
        require(invalid_next && p0.phase()==strix::DistributedTokenPhase::committed,
            "repeated generation admitted or altered the committed frontier");
        invalid_next=false;
        try {p0.begin_next(input,binding.generation+1U,std::move(missing));}
        catch(const std::exception&) {invalid_next=true;}
        const auto refused_bytes=snapshot(*first);
        require(invalid_next && p0.phase()==strix::DistributedTokenPhase::in_doubt && !p0.frontier() &&
            session0.health==strix::SerialSessionHealth::poisoned && session0.arithmetic_lineage.tainted &&
            refused_bytes.device==committed_bytes.device && refused_bytes.host==committed_bytes.host,
            "empty retirement capability permitted a next token or changed local state before submission");

        return;
    }
    const auto abort=p0.decide_abort();p1.observe_peer(abort);
    const auto restored0=p0.restore_local();p1.observe_peer(restored0);
    bool rejected=false;try {p0.finalize_rollback();} catch(const std::logic_error&) {rejected=true;}
    require(rejected && session0.health==strix::SerialSessionHealth::poisoned,"one restored peer republished history before the other restored");
    const auto restored1=p1.restore_local();p0.observe_peer(restored1);
    p0.finalize_rollback();p1.finalize_rollback();
    const auto after0=snapshot(*first),after1=snapshot(*second);
    require(after0.device==before0.device && after0.host==before0.host && after1.device==before1.device && after1.host==before1.host,
        "bilateral rollback did not restore all seven complete physical state components");
    const auto same=[](const strix::SerialSessionContext& a,const strix::SerialSessionContext& b) {
        return a.health==b.health && a.execution_identity==b.execution_identity && a.next_position==b.next_position &&
            a.state_layout_identity==b.state_layout_identity && a.runtime_compatibility_identity==b.runtime_compatibility_identity &&
            a.committed_tokens==b.committed_tokens && a.state.generations==b.state.generations && a.state.valid==b.state.valid &&
            a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
            a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals && a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted;
    };
    require(same(session0,saved0) && same(session1,saved1),"rollback failed to republish exact saved history only after both restores");
    rejected=false;try {p0.finalize_rollback();} catch(const std::logic_error&) {rejected=true;}
    require(rejected,"rollback publication was repeated using a moved history");
    // This local component exercised REAL target callbacks and all7 snapshot
    // transfers. There is no network, distributed commit or continuation claim.
}
int run(const std::filesystem::path& model,const std::string& expected,std::uint32_t input,const std::filesystem::path& output) {
    require(!std::filesystem::exists(output),"output must be fresh");
    require(digest(model)==expected,"model SHA before differs");
    auto o=load(model);require(input<o->model.vocab_size,"input token outside model vocabulary");
    const auto found=std::find_if(o->serial.execution.steps.begin(),o->serial.execution.steps.end(),[](const auto& s) {
        return s.op==strix::SerialOp::mlp_inject && s.layer==0U;
    });
    require(found!=o->serial.execution.steps.end(),"no layer0 stage boundary");const auto cut=found->id;
    auto catalog=std::make_shared<strix::HipLauncherCatalog>(o->model,o->config);strix::ExecutionLauncherRegistry registry;
    catalog->register_for_compiled_execution(o->serial,registry,&o->resources);
    const auto launch=strix::compile_launch_plan(o->serial,registry);
    strix::SerialSessionContext session;session.execution_identity=o->serial.execution_identity;
    session.state_layout_identity=o->layout.layout_identity;
    session.runtime_compatibility_identity=strix::build_runtime_execution_identities(o->serial.execution_identity,o->serial.arena,o->resources).compatibility_identity;
    for(const auto& a:o->layout.allocations) session.state.valid=session.state.valid|a.state_flag;
    const auto base=session;const auto initial=snapshot(*o);Reference reference(launch,*o,cut);
    o->guard.run([&] {
        strix::HipLaunchScope scope(o->scope());
        (void)strix::execute_serial_token(o->serial,session,o->resources,input,reference,[&] {
            strix::DeferredMoeFirstError record{};
            check(hipMemcpyAsync(&record,reinterpret_cast<const void*>(o->status.arena_base),sizeof(record),hipMemcpyDeviceToHost,o->stream),"read reference status");
            check(hipStreamSynchronize(o->stream),"complete reference token");strix::check_deferred_moe_status(record);
        });
    });
    require(reference.observed,"reference did not observe the actual intermediate stage");
    const auto full_reference=snapshot(*o);const auto full_reference_session=session;
    restore(*o,initial);session=base;
    const auto plan=placement(*o,cut);
    strix::HipDistributedStage stage(o->serial,plan,0,cut,o->model,o->config,o->resources,o->scope(),o);
    strix::HipDistributedStageToken token;token.request={1,1,1,1};token.position=0;token.input_token=input;token.initial_state=base.state;
    auto wrong_controller=token;wrong_controller.controller_materializes_position=true;
    bool controller_rejected=false;try {stage.begin_token(wrong_controller);} catch(const std::invalid_argument&) {controller_rejected=true;}
    require(controller_rejected,"prefix containing position_prepare accepted suffix controller materialization");
    stage.begin_token(token);
    std::array<strix::transport::DirectTransportNativeLinkSession*,1> unavailable{nullptr};
    NoControl control;NoStaging staging;
    strix::NativeDistributedBackend backend(unavailable,control,stage,staging);
    for(std::uint32_t i=0;i<=cut;++i) backend.execute_local_step(plan.steps[i]);
    const auto result=stage.complete_stage();const auto actual=snapshot(*o);
    auto expected_state=base.state;
    for(std::uint32_t i=0;i<=cut;++i) expected_state.publish_write(o->serial.execution.steps[i].writes_state);
    require(actual.scratch==reference.captured.scratch && actual.device==reference.captured.device &&
        actual.host==reference.captured.host,"actual contiguous HIP stage differs from serial intermediate arena/all seven states");
    require(result.local_state.valid==expected_state.valid && result.local_state.generations==expected_state.generations &&
        result.completed_steps==cut+1U && session.next_position==base.next_position && session.committed_tokens==base.committed_tokens &&
        session.arithmetic_lineage.history_identity==base.arithmetic_lineage.history_identity,
        "stage published a global session commit or incorrect local state generations");
    std::uint64_t boundary_bytes=0;
    for(const auto& b:plan.boundaries) for(const auto& view:stage.completed_boundary_views(b)) boundary_bytes+=view.bytes;
    require(boundary_bytes==plan.boundary_payload_bytes_per_token,"actual boundary view bytes differ");
    bool rejected=false;try {stage.begin_token(token);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"completed stage accepted a stale generation");
    stage.abort();require(stage.failed(),"peer abort after local completion did not invalidate provisional owner");
    auto next=token;next.request.generation=2U;next.position=1U;
    rejected=false;try {stage.begin_token(next);} catch(const std::logic_error&) {rejected=true;}
    require(rejected,"peer-aborted retired stage was reused without fresh/restored owner");
    restore(*o,initial);
    // A newly constructed owner is mandatory after an error. Inject sticky D1
    // AFTER begin admission, then dispatch a genuine GPU embedding callback.
    strix::HipDistributedStage failed(o->serial,plan,0,cut,o->model,o->config,o->resources,o->scope(),o);
    failed.begin_token(token);const strix::DeferredMoeFirstError injected{1U,1U};
    check(hipMemcpy(reinterpret_cast<void*>(o->status.arena_base),&injected,sizeof(injected),hipMemcpyHostToDevice),"inject component status");
    rejected=false;try {failed.execute_local_step(plan.steps[0]);} catch(const std::exception&) {rejected=true;}
    require(rejected && failed.failed(),"queued GPU callback error did not drain and disable owner");
    bool sticky=false;try {failed.execute_local_step(plan.steps[1]);} catch(const std::logic_error&) {sticky=true;}
    require(sticky,"failed stage was reused");
    restore(*o,full_reference);session=full_reference_session;
    require(session.next_position>0U,"transaction rollback must exercise a nonzero input position and retained history");
    transaction_component(o,session,cut,input,TransactionOutcome::rollback);
    restore(*o,full_reference);session=full_reference_session;
    transaction_component(o,session,cut,input,TransactionOutcome::commit);
    restore(*o,full_reference);session=full_reference_session;
    transaction_component(o,session,cut,input,TransactionOutcome::lost_decision);
    require(digest(model)==expected,"model SHA after differs");
    std::ofstream out(output,std::ios::binary);out.exceptions(std::ios::failbit|std::ios::badbit);
    out<<"{\"schema\":\"hip-distributed-stage-component-v1\",\"finished\":true,\"component_passed\":true,"
        "\"transport_exercised\":false,\"distributed_model_qualified\":false,\"global_commit_published\":false,"
        "\"bilateral_snapshot_rollback_component\":true,\"transaction_commit_and_lost_decision_components\":true,\"all_weights_local\":true,\"stage_steps\":"<<result.completed_steps<<",\"boundary_bytes\":"<<boundary_bytes
        <<",\"model_sha256\":\""<<expected<<"\"}\n";
    return 0;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=9 || std::string(argv[1])!="--model" || std::string(argv[3])!="--model-sha256" ||
            std::string(argv[5])!="--token" || std::string(argv[7])!="--output")
            throw std::invalid_argument("opt-in only: --model PATH --model-sha256 HEX64 --token ID --output FRESH_JSON");
        const std::string hash=argv[4],text=argv[6];std::uint32_t token=0;
        const auto parsed=std::from_chars(text.data(),text.data()+text.size(),token);
        require(hash.size()==64U && std::all_of(hash.begin(),hash.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');}) &&
            parsed.ec==std::errc{} && parsed.ptr==text.data()+text.size(),"invalid model pin/token");
        return run(argv[2],hash,token,argv[8]);
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
