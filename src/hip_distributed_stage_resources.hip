#include "strix/hip_distributed_stage_resources.hpp"
#include "strix/distributed_stage_state.hpp"
#include <cstring>
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_tensor_views.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace strix {
namespace {
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("stage resource budget overflows");
    return a+b;
}
ModelManifest selected_manifest(const ModelManifest& source,const DistributedStageResourcePlan& plan) {
    ModelManifest selected;selected.architecture=source.architecture;
    for(const auto index:plan.tensor_indices()) {
        const auto& t=source.tensors[index];selected.tensors.push_back(t);
        selected.total_tensor_bytes=add(selected.total_tensor_bytes,t.byte_size);
        if(t.residency==ResidencyPolicy::file_backed_candidate) selected.file_backed_candidate_bytes=add(selected.file_backed_candidate_bytes,t.byte_size);
        if(t.residency==ResidencyPolicy::bounded_cache_candidate) selected.bounded_cache_candidate_bytes=add(selected.bounded_cache_candidate_bytes,t.byte_size);
        if(t.role==TensorRole::unknown) ++selected.unknown_tensor_count;
    }
    selected.layout_identity=model_manifest_layout_identity(selected);return selected;
}
bool same_state(const PersistentStateGenerations& a,const PersistentStateGenerations& b) {
    return a.valid==b.valid && a.generations==b.generations;
}
}
struct HipDistributedStageResources::Impl {
    HipSetupQuiescenceGuard guard;
    AllocationGenerationSource generations;
    std::shared_ptr<const MappedModelFile> mapped;
    CompiledSerialExecution serial;
    ModelContract model;
    HipLauncherCatalogConfig config;
    ModelManifest manifest,selected;
    ExecutionTensorAccessPlan access;
    std::optional<DistributedStageResourcePlan> plan;
    HipDistributedStageBudget budget;
    HipDistributedStageReservation reservation;
    std::shared_ptr<HipModelResidency> backing;
    ModelRuntimeBinding weights;
    SerialSessionStateLayout layout;
    HipRuntimeAllocator allocator{HipHostAllocationMode::pageable_aligned};
    RuntimeExecutionMemoryOwner memory;
    std::optional<DistributedStageStateMapping> state_mapping;
    ExecutionMemoryBinding scratch;
    SessionStateMemoryBinding state;
    bool packed_pools=false;
    FileMappedPleLookupBackend ple;
    PleHashContract hash;
    PleLookupPrefetchState prefetch;
    ExecutionMemoryBinding d1;
    HipLaunchContext scope;
    ExecutionRuntimeResources resources;
    std::optional<HipDistributedStageInitialFrontier> initial;
    std::uintptr_t mapped_base=0;
    std::uint64_t mapped_bytes=0,mapped_generation=0;
    bool claimed=false;
    ~Impl() {
        if(hipSetDevice(guard.device_index())!=hipSuccess) std::terminate();
        if(resources.ple_prefetch && prefetch.active) cancel_ple_lookup_prefetch(ple,prefetch);
        guard.recover_or_fail_stop();
        if(scope.stream && hipStreamDestroy(scope.stream)!=hipSuccess) std::terminate();
        if(d1.arena_base) allocator.free_device(d1.arena_base,d1.arena_bytes);
        if(packed_pools) {
            if(state.host_base) allocator.free_host(state.host_base,state.host_bytes);
            if(state.device_base) allocator.free_device(state.device_base,state.device_bytes);
            if(scratch.arena_base) allocator.free_device(scratch.arena_base,scratch.arena_bytes);
        }
        // memory is destroyed before allocator; backing before mapped source.
    }
};
HipDistributedStageResources::HipDistributedStageResources():impl_(std::make_unique<Impl>()) {}
HipDistributedStageResources::~HipDistributedStageResources()=default;
std::shared_ptr<HipDistributedStageResources> HipDistributedStageResources::create(
    std::shared_ptr<const MappedModelFile> mapped,const CompiledSerialExecution& serial,const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& access,const DistributedPlacementPlan& placement,std::uint32_t node,
    const ModelContract& model,const HipLauncherCatalogConfig& config,std::uint64_t capacity,
    HipDistributedStageBudget budget,HipModelResidencyPolicy policy,HipDistributedStageStatePolicy state_policy) {
    if(current_hip_launch_context()) throw std::invalid_argument("stage resource creation cannot nest a HIP scope");
    if(!model.validate().empty() || serial.vocab_size!=model.vocab_size || serial.max_position_embeddings!=model.max_position_embeddings ||
        (state_policy!=HipDistributedStageStatePolicy::full && state_policy!=HipDistributedStageStatePolicy::local_owned_ranges) ||
        !mapped || !mapped->valid() || !mapped->content_identity().valid() || !mapped->allocation_generation() ||
        !capacity || capacity>serial.max_position_embeddings || !budget.retained_bytes || !budget.setup_peak_bytes ||
        !budget.headroom_bytes || !budget.snapshot_bytes || config.enable_device_graph_inputs || !config.deferred_moe_status)
        throw std::invalid_argument("stage resources require actual source, eager D1 and explicit complete budgets");
    const auto plan=DistributedStageResourcePlan::derive(serial,manifest,access,placement,node);
    auto selected=selected_manifest(manifest,plan);
    SerialSessionStateLayoutRequest state_request;state_request.context_tokens=capacity;state_request.recurrent_storage=config.gdn_recurrent_storage;
    const auto layout=build_serial_session_state_layout(model,state_request);
    if(layout.allocations.size()!=7U) throw std::invalid_argument("stage resources require the complete seven-component global logical layout");
    std::optional<DistributedStageStateMapping> mapping;
    if(state_policy==HipDistributedStageStatePolicy::local_owned_ranges)
        mapping=DistributedStageStateMapping::derive(serial,model,layout,plan);
    std::uint64_t snapshot_device=0,snapshot_host=0;
    for(const auto& a:layout.allocations) {
        auto& bytes=a.storage==SessionStateStorage::device?snapshot_device:snapshot_host;
        bytes=add(bytes,(64U-bytes%64U)%64U);bytes=add(bytes,a.bytes);
    }
    if(mapping) {
        const auto local_snapshot=build_stage_snapshot_plan(*mapping,plan.identity());
        snapshot_device=local_snapshot.device_bytes;snapshot_host=local_snapshot.host_bytes;
    }
    if(add(snapshot_device,snapshot_host)>budget.snapshot_bytes)
        throw std::invalid_argument("stage snapshot budget cannot retain all actually owned snapshot ranges");
    const auto weight_reservation=plan_hip_model_residency_reservation(*mapped,selected,policy);
    HipDistributedStageReservation reservation;reservation.weights=weight_reservation;
    reservation.full_state_bytes=add(layout.device_total_bytes,layout.host_total_bytes);
    reservation.state_partitioned=mapping.has_value();
    reservation.retained_state_bytes=mapping?add(mapping->device_bytes(),mapping->host_bytes()):reservation.full_state_bytes;
    reservation.full_scratch_bytes=serial.arena.total_bytes;reservation.snapshot_budget_bytes=budget.snapshot_bytes;
    reservation.headroom_bytes=budget.headroom_bytes;
    // HipRuntimeAllocator owns bytes+(alignment-1) for device pools. Host
    // pools use pageable aligned allocations. Full snapshot device sections use
    //64B; typed local snapshot sections are both256B-aligned.
    reservation.allocator_padding_bytes=add(add(serial.arena.alignment-1U,layout.device_alignment-1U),
        mapping?255U+255U+255U:255U+63U);
    auto retained=add(weight_reservation.retained_reserved_bytes,reservation.retained_state_bytes);
    retained=add(retained,reservation.full_scratch_bytes);retained=add(retained,8U);
    retained=add(retained,reservation.snapshot_budget_bytes);retained=add(retained,reservation.allocator_padding_bytes);
    retained=add(retained,reservation.headroom_bytes);reservation.retained_bytes=retained;
    reservation.setup_peak_bytes=add(retained,weight_reservation.setup_peak_reserved_bytes-weight_reservation.retained_reserved_bytes);
    if(reservation.retained_bytes>budget.retained_bytes || reservation.setup_peak_bytes>budget.setup_peak_bytes)
        throw std::invalid_argument("stage resource complete reservation exceeds its explicit budget");
    auto result=std::shared_ptr<HipDistributedStageResources>(new HipDistributedStageResources);
    auto& p=*result->impl_;p.mapped=std::move(mapped);p.serial=serial;p.model=model;p.config=config;
    p.manifest=manifest;p.selected=std::move(selected);p.access=access;p.plan=plan;p.layout=layout;p.budget=budget;p.reservation=reservation;
    p.state_mapping=std::move(mapping);
    p.mapped_base=p.mapped->base_address();p.mapped_bytes=p.mapped->bytes();p.mapped_generation=p.mapped->allocation_generation();
    p.backing=std::make_shared<HipModelResidency>(HipModelResidency::create(p.mapped,p.selected,p.generations,policy));
    p.weights.manifest_layout_identity=manifest.layout_identity;p.weights.model_content_identity=p.mapped->content_identity();
    p.weights.tensors.resize(manifest.tensors.size());
    for(std::size_t i=0;i<manifest.tensors.size();++i)
        p.weights.tensors[i]={static_cast<std::uint32_t>(i),RuntimeTensorStorage::remote,0,manifest.tensors[i].byte_size,0,1U-node};
    std::size_t slot=0;
    for(const auto index:plan.tensor_indices()) {
        auto actual=p.backing->binding().tensors[slot++];actual.tensor_index=index;
        // Only the backing owner's private filtered index is remapped. Public
        // model/step/manifest indices and content SHA remain canonical.
        p.weights.tensors[index]=actual;
    }
    validate_stage_model_runtime_binding(plan,p.manifest,p.weights);
    if(p.state_mapping) {
        p.packed_pools=true; // Partial setup cleanup owns every successfully allocated pool.
        p.scratch.execution_identity=p.serial.execution_identity;
        p.scratch.arena_bytes=p.serial.arena.total_bytes;
        p.scratch.allocation_generation=p.generations.next();
        p.scratch.arena_base=p.allocator.allocate_device(p.scratch.arena_bytes,p.serial.arena.alignment);
        p.state.layout_identity=p.layout.layout_identity;p.state.stage_mapping=&*p.state_mapping;
        p.state.device_bytes=p.state_mapping->device_bytes();p.state.host_bytes=p.state_mapping->host_bytes();
        if(p.state.device_bytes) {
            p.state.device_allocation_generation=p.generations.next();
            p.state.device_base=p.allocator.allocate_device(p.state.device_bytes,p.layout.device_alignment);
        }
        if(p.state.host_bytes) {
            p.state.host_allocation_generation=p.generations.next();
            p.state.host_base=p.allocator.allocate_host(p.state.host_bytes,p.layout.host_alignment);
        }
    } else {
        p.memory=RuntimeExecutionMemoryOwner::allocate(p.serial,p.layout,p.allocator,p.generations);
        p.scratch=p.memory.scratch();p.state=p.memory.state();
    }
    p.d1={p.serial.execution_identity,p.allocator.allocate_device(8U,256U),8U,p.generations.next()};
    if(hipStreamCreateWithFlags(&p.scope.stream,hipStreamNonBlocking)!=hipSuccess)
        throw std::runtime_error("stage resource owned stream creation failed");
    p.scope.deferred_moe_status=p.d1;p.hash=build_ple_hash_contract(model);
    bool needs_ple=false;
    for(std::uint32_t i=plan.first_step();i<=plan.last_step();++i) needs_ple|=serial.execution.steps[i].op==SerialOp::ple;
    p.resources={&p.scratch,&p.manifest,&p.weights,&p.access,&p.layout,&p.state,
        needs_ple?&p.ple:nullptr,needs_ple?&p.hash:nullptr,needs_ple?&p.prefetch:nullptr,&*p.plan};
    result->validate();
    // Own fresh buffers only; no borrowed serial context is created or retagged.
    p.guard.run([&] {
        if(p.state_mapping) {
            if(hipMemsetAsync(reinterpret_cast<void*>(p.scratch.arena_base),0,p.scratch.arena_bytes,p.scope.stream)!=hipSuccess ||
                (p.state.device_bytes && hipMemsetAsync(reinterpret_cast<void*>(p.state.device_base),0,p.state.device_bytes,p.scope.stream)!=hipSuccess))
                throw std::runtime_error("stage packed-state initialization failed");
            if(p.state.host_bytes) std::memset(reinterpret_cast<void*>(p.state.host_base),0,static_cast<std::size_t>(p.state.host_bytes));
        } else initialize_hip_fresh_runtime_memory(p.scratch,p.state);
        if(hipMemsetAsync(reinterpret_cast<void*>(p.d1.arena_base),0,8U,p.scope.stream)!=hipSuccess)
            throw std::runtime_error("stage resource fresh D1 initialization failed");
    });
    // Global logical flags describe the transaction frontier, not physical
    // possession of absent state classes in the packed local policy.
    PersistentStateGenerations global;for(const auto& a:p.layout.allocations) global.valid=global.valid|a.state_flag;
    const auto ids=build_stage_runtime_execution_identities(serial.execution_identity,serial.arena,p.resources);
    DistributedTokenPeer peer;peer.node=node;peer.first_step=plan.first_step();peer.last_step=plan.last_step();
    peer.runtime_compatibility_identity=ids.compatibility_identity;peer.state_layout_identity=p.layout.layout_identity;
    peer.context_capacity=capacity;peer.initial_state=global;
    p.initial=HipDistributedStageInitialFrontier(peer,global);
    return result;
}
void HipDistributedStageResources::validate() const {
    const auto& p=*impl_;p.guard.require_original();
    if(!p.mapped || !p.backing || !p.plan || p.mapped->base_address()!=p.mapped_base || p.mapped->bytes()!=p.mapped_bytes ||
        p.mapped->allocation_generation()!=p.mapped_generation || p.mapped->content_identity()!=p.weights.model_content_identity)
        throw std::invalid_argument("stage resource source owner changed");
    p.plan->validate_source(p.serial,p.manifest,p.access);
    if(p.state_mapping) {
        p.state_mapping->validate_source(p.serial,p.model,p.layout,*p.plan);
        if(p.state.stage_mapping!=&*p.state_mapping)
            throw std::invalid_argument("stage packed-state mapping owner changed");
    } else if(p.state.stage_mapping) throw std::invalid_argument("full stage state cannot carry a local mapping");
    validate_stage_execution_runtime_resources(p.serial.execution_identity,p.serial.arena,p.resources,0);
    std::size_t slot=0;
    for(const auto index:p.plan->tensor_indices()) {
        const auto& r=p.weights.tensors[index];const auto& actual=p.backing->binding().tensors[slot++];
        if(r.storage!=actual.storage || r.address!=actual.address || r.bytes!=actual.bytes ||
            r.allocation_generation!=actual.allocation_generation || r.owner_node!=actual.owner_node)
            throw std::invalid_argument("stage resource remapped backing differs from actual retained allocation");
        const auto& t=p.manifest.tensors[index];
        if(runtime_tensor_gpu_visible(r.storage)) {
            if(!p.backing->covers_device_range(r.address,r.bytes)) throw std::invalid_argument("stage weight is outside actual GPU backing");
        } else if(t.role!=TensorRole::ngram_table || r.storage!=RuntimeTensorStorage::file_mapped ||
            t.absolute_offset>p.mapped_bytes || r.bytes>p.mapped_bytes-t.absolute_offset || r.address!=p.mapped_base+t.absolute_offset)
            throw std::invalid_argument("stage host weight is not the actual retained PLE mapping");
    }
}
void HipDistributedStageResources::validate_initial_binding(const DistributedTokenBinding& b) const {
    validate();const auto& p=*impl_;
    if(p.claimed) throw std::logic_error("stage initial frontier already belongs to a participant");
    const auto& seed=initial_frontier();const auto& peer=b.peers[p.plan->local_node()];
    if(b.position || b.committed_tokens || b.previous_history_identity || b.serial_execution_identity!=p.serial.execution_identity ||
        b.kernel_plan_identity!=p.serial.kernel_plan_identity || b.placement_plan_identity!=p.plan->placement_identity() ||
        b.model_content_identity!=p.weights.model_content_identity || !same_state(b.initial_global_state,seed.global_state()) ||
        peer.node!=seed.peer().node || peer.first_step!=seed.peer().first_step || peer.last_step!=seed.peer().last_step ||
        peer.runtime_compatibility_identity!=seed.peer().runtime_compatibility_identity || peer.state_layout_identity!=seed.peer().state_layout_identity ||
        peer.context_capacity!=seed.peer().context_capacity || !same_state(peer.initial_state,seed.peer().initial_state))
        throw std::invalid_argument("stage participant requires its genuine initialized position-zero frontier");
}
void HipDistributedStageResources::claim_initial(const DistributedTokenBinding& b) {
    validate_initial_binding(b);
    if(impl_->claimed) throw std::logic_error("stage resources already belong to a participant");
    impl_->claimed=true;
}
const CompiledSerialExecution& HipDistributedStageResources::serial() const noexcept {return impl_->serial;}
const ModelContract& HipDistributedStageResources::model() const noexcept {return impl_->model;}
const HipLauncherCatalogConfig& HipDistributedStageResources::config() const noexcept {return impl_->config;}
const DistributedStageResourcePlan& HipDistributedStageResources::plan() const noexcept {return *impl_->plan;}
const ExecutionRuntimeResources& HipDistributedStageResources::resources() const noexcept {return impl_->resources;}
const HipLaunchContext& HipDistributedStageResources::scope() const noexcept {return impl_->scope;}
const HipDistributedStageInitialFrontier& HipDistributedStageResources::initial_frontier() const {
    if(!impl_->initial) throw std::logic_error("stage initial frontier has not completed initialization");
    return *impl_->initial;
}
const HipDistributedStageReservation& HipDistributedStageResources::reservation() const noexcept {return impl_->reservation;}
RuntimeMemoryAllocator& HipDistributedStageResources::snapshot_allocator() noexcept {return impl_->allocator;}
AllocationGenerationSource& HipDistributedStageResources::generations() noexcept {return impl_->generations;}
std::uint64_t HipDistributedStageResources::snapshot_budget_bytes() const noexcept {return impl_->budget.snapshot_bytes;}
} // namespace strix
