#include "strix/distributed_stage_state.hpp"
#include "strix/memory_plan.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
void require_stage(bool good,const char* why) {if(!good) throw std::invalid_argument(why);}
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a) throw std::overflow_error("stage state byte arithmetic overflow");
    return a+b;
}
std::uint64_t mul(std::uint64_t a,std::uint64_t b) {
    if(a && b>UINT64_MAX/a) throw std::overflow_error("stage state byte multiplication overflow");
    return a*b;
}
std::uint64_t align(std::uint64_t n,std::uint64_t a) {
    require_stage(a && !(a&(a-1U)),"stage state alignment must be a nonzero power of two");
    return add(n,a-1U)&~(a-1U);
}
void word(std::uint64_t& h,std::uint64_t v) noexcept {
    for(unsigned i=0;i<64U;i+=8U) {h^=(v>>i)&255U;h*=1099511628211ULL;}
}
std::size_t index(SessionStateComponent c) {
    const auto i=static_cast<unsigned>(c);require_stage(i<7U,"unknown stage state component");return i;
}
bool same_layout(const SerialSessionStateLayout& a,const SerialSessionStateLayout& b) {
    if(a.layout_identity!=b.layout_identity || a.context_tokens!=b.context_tokens || a.device_alignment!=b.device_alignment ||
        a.host_alignment!=b.host_alignment || a.device_total_bytes!=b.device_total_bytes || a.host_total_bytes!=b.host_total_bytes ||
        a.recurrent_storage!=b.recurrent_storage || a.allocations.size()!=b.allocations.size()) return false;
    for(std::size_t i=0;i<a.allocations.size();++i) {
        const auto& x=a.allocations[i];const auto& y=b.allocations[i];
        if(x.component!=y.component || x.state_flag!=y.state_flag || x.storage!=y.storage || x.offset!=y.offset ||
            x.bytes!=y.bytes || x.provisional!=y.provisional) return false;
    }
    return true;
}
void pool(std::uintptr_t base,std::uint64_t bytes,std::uint64_t need,std::uint64_t gen,std::uint64_t alignment) {
    if(!need) {require_stage(!base && !bytes && !gen,"absent stage state pool must be explicitly absent");return;}
    require_stage(base && bytes>=need && gen && alignment && base%alignment==0U && bytes<=UINTPTR_MAX-base,
        "stage state pool has invalid alignment, extent or actual allocation generation");
}
void disjoint(std::uintptr_t a,std::uint64_t an,std::uintptr_t b,std::uint64_t bn) {
    if(!an || !bn) return;
    require_stage(a && b && an<=UINTPTR_MAX-a && bn<=UINTPTR_MAX-b && !(a<b+bn && b<a+an),
        "stage state/snapshot declared pools alias or overflow");
}
std::uint64_t snapshot_identity(const DistributedStageSnapshotPlan& p) {
    std::uint64_t h=14695981039346656037ULL;word(h,0x535447534e415031ULL);
    word(h,p.mapping_identity);word(h,p.logical_plan_identity);word(h,p.device_bytes);word(h,p.host_bytes);
    word(h,p.ranges.size());for(const auto& r:p.ranges) {
        word(h,static_cast<unsigned>(r.storage));word(h,r.source_offset);word(h,r.destination_offset);word(h,r.bytes);
    }return h;
}
}
DistributedStageStateMapping DistributedStageStateMapping::derive(const CompiledSerialExecution& serial,
    const ModelContract& model,const SerialSessionStateLayout& layout,const DistributedStageResourcePlan& stage) {
    require_stage(model.validate().empty() && serial.execution_identity && stage.identity() &&
        stage.execution_identity()==serial.execution_identity && serial.logical_rows==1U &&
        serial.execution.steps.size()<=UINT32_MAX && serial.execution.steps.size()==stage.canonical_placement().steps.size(),
        "stage state source does not match serial execution/model");
    validate_serial_session_state_layout_identity(model,layout);
    const auto canonical=build_serial_session_state_layout(model,{layout.context_tokens,layout.device_alignment,
        layout.host_alignment,find_session_state_allocation(layout,SessionStateComponent::sparse_index)!=nullptr,layout.recurrent_storage});
    require_stage(same_layout(canonical,layout),"stage state requires exact canonical complete global layout geometry");
    DistributedStageStateMapping result;result.layout_=layout;result.stage_=stage.identity();
    result.execution_=serial.execution_identity;result.node_=stage.local_node();
    const auto append=[&](SessionStateComponent component,std::uint64_t offset,std::uint64_t bytes) {
        const auto* allocation=find_session_state_allocation(layout,component);
        require_stage(allocation && bytes && offset<=allocation->bytes && bytes<=allocation->bytes-offset,
            "owned stage state slice exceeds its global logical component");
        result.ranges_.push_back({component,allocation->state_flag,allocation->storage,offset,bytes,0,allocation->provisional});
    };
    for(std::size_t i=0;i<serial.execution.steps.size();++i) {
        const auto& step=serial.execution.steps[i];
        if(!stage.contains_step(static_cast<std::uint32_t>(i))) continue;
        require_stage(i<stage.canonical_placement().steps.size() && stage.canonical_placement().steps[i].step_id==i &&
            stage.canonical_placement().steps[i].op==step.op && stage.canonical_placement().steps[i].node==stage.local_node() &&
            step.id==i,"stage state step census differs from canonical placement");
        if(step.op==SerialOp::gated_delta_net) {
            require_stage(step.layer.has_value(),"stage GDN state step lacks layer");
            const auto ordinal=gdn_layer_ordinal(model,*step.layer);
            append(SessionStateComponent::recurrent_matrix,mul(ordinal,gdn_matrix_state_bytes_per_layer(model)),gdn_matrix_state_bytes_per_layer(model));
            append(SessionStateComponent::gdn_convolution,mul(ordinal,gdn_conv_state_bytes_per_layer(model)),gdn_conv_state_bytes_per_layer(model));
        } else if(step.op==SerialOp::indexed_attention) {
            require_stage(step.layer.has_value() && model.full_attention_layers,"stage QSA state step lacks layer");
            const auto ordinal=indexed_attention_layer_ordinal(model,*step.layer);
            const auto kv=mul(base_kv_bytes_per_token(model)/model.full_attention_layers,layout.context_tokens);
            const auto sparse=mul(provisional_indexer_bytes_per_token(model)/model.full_attention_layers,layout.context_tokens);
            append(SessionStateComponent::attention_kv,mul(ordinal,kv),kv);
            append(SessionStateComponent::sparse_index,mul(ordinal,sparse),sparse);
        } else if(step.op==SerialOp::ple) {
            for(const auto component:{SessionStateComponent::ple_convolution,SessionStateComponent::ngram_history}) {
                const auto* allocation=find_session_state_allocation(layout,component);
                require_stage(allocation,"stage PLE state component is absent");append(component,0,allocation->bytes);
            }
        }
    }
    const auto* position=find_session_state_allocation(layout,SessionStateComponent::position_history);
    require_stage(position && position->storage==SessionStateStorage::host && position->bytes==sizeof(std::uint64_t),
        "stage state requires the replicated complete host position word");
    append(SessionStateComponent::position_history,0,position->bytes);
    std::sort(result.ranges_.begin(),result.ranges_.end(),[](const auto& a,const auto& b) {
        return a.component==b.component?a.logical_offset<b.logical_offset:a.component<b.component;
    });
    for(std::size_t i=0;i<result.ranges_.size();++i) {
        auto& r=result.ranges_[i];
        if(i && result.ranges_[i-1U].component==r.component)
            require_stage(result.ranges_[i-1U].logical_offset+result.ranges_[i-1U].bytes<=r.logical_offset,
                "stage state ownership contains duplicate/overlapping logical slices");
        auto& total=r.storage==SessionStateStorage::device?result.device_:result.host_;
        total=align(total,r.storage==SessionStateStorage::device?layout.device_alignment:layout.host_alignment);
        r.local_offset=total;total=add(total,r.bytes);
    }
    result.device_=align(result.device_,layout.device_alignment);result.host_=align(result.host_,layout.host_alignment);
    for(const auto& allocation:layout.allocations) {
        std::uint64_t bytes=0;for(const auto& r:result.ranges_) if(r.component==allocation.component) bytes=add(bytes,r.bytes);
        result.presence_[index(allocation.component)]=!bytes?DistributedStageStatePresence::absent:
            bytes==allocation.bytes?DistributedStageStatePresence::complete:DistributedStageStatePresence::partial;
    }
    std::uint64_t h=14695981039346656037ULL;word(h,0x5354475354415431ULL);word(h,result.stage_);word(h,result.execution_);
    word(h,layout.layout_identity);word(h,result.node_);word(h,result.device_);word(h,result.host_);
    for(const auto& r:result.ranges_) {word(h,static_cast<unsigned>(r.component));word(h,static_cast<unsigned>(r.storage));
        word(h,r.logical_offset);word(h,r.local_offset);word(h,r.bytes);word(h,static_cast<unsigned>(r.state_flag));word(h,r.provisional);}
    require_stage(h!=0U,"stage state identity is zero");result.identity_=h;return result;
}
void DistributedStageStateMapping::validate_source(const CompiledSerialExecution& serial,const ModelContract& model,
    const SerialSessionStateLayout& layout,const DistributedStageResourcePlan& stage) const {
    const auto expected=derive(serial,model,layout,stage);
    require_stage(expected.identity_==identity_ && expected.ranges_==ranges_ && expected.presence_==presence_ &&
        expected.device_==device_ && expected.host_==host_,"stage state mapping source/census changed");validate_layout(layout);
}
void DistributedStageStateMapping::validate_layout(const SerialSessionStateLayout& layout) const {
    require_stage(identity_ && same_layout(layout_,layout),"stage state global layout metadata changed");
}
void DistributedStageStateMapping::validate_memory(const SerialSessionStateLayout& layout,const SessionStateMemoryBinding& memory) const {
    validate_layout(layout);require_stage(memory.stage_mapping==this && memory.layout_identity==layout.layout_identity,
        "stage state mapping handle/layout binding changed");
    pool(memory.device_base,memory.device_bytes,device_,memory.device_allocation_generation,layout.device_alignment);
    pool(memory.host_base,memory.host_bytes,host_,memory.host_allocation_generation,layout.host_alignment);
    disjoint(memory.device_base,memory.device_bytes,memory.host_base,memory.host_bytes);
}
SessionStateAddress DistributedStageStateMapping::resolve(const SerialSessionStateLayout& layout,const SessionStateMemoryBinding& memory,
    SessionStateComponent component,std::uint64_t offset,std::uint64_t bytes) const {
    validate_memory(layout,memory);
    const auto* allocation=find_session_state_allocation(layout,component);
    require_stage(allocation && bytes && offset<=allocation->bytes && bytes<=allocation->bytes-offset,
        "stage state requested logical slice is empty, absent or outside full component");
    std::uint64_t remaining=bytes,current=offset,start_local=0,next_local=0;
    bool found=false;
    for(const auto& r:ranges_) {
        if(r.component!=component) continue;
        if(current<r.logical_offset) break;
        if(current-r.logical_offset>=r.bytes) continue;
        const auto local=add(r.local_offset,current-r.logical_offset);
        if(!found) {start_local=local;found=true;}
        else require_stage(local==next_local,
            "owned logical state crosses noncontiguous packed ranges; resolve exact layer slices");
        const auto take=std::min(remaining,r.bytes-(current-r.logical_offset));
        remaining-=take;current=add(current,take);next_local=add(local,take);
        if(!remaining) {
            const auto base=r.storage==SessionStateStorage::device?memory.device_base:memory.host_base;
            const auto gen=r.storage==SessionStateStorage::device?memory.device_allocation_generation:memory.host_allocation_generation;
            require_stage(start_local<=UINTPTR_MAX-base && bytes<=UINTPTR_MAX-(base+start_local),
                "stage state resolved address overflows");
            return {r.component,r.state_flag,r.storage,base+static_cast<std::uintptr_t>(start_local),bytes,gen,r.provisional};
        }
    }
    throw std::invalid_argument("requested logical state slice is absent or crosses partial stage ownership");
}
DistributedStageStatePresence DistributedStageStateMapping::presence(SessionStateComponent component) const {return presence_[index(component)];}
DistributedStageSnapshotPlan build_stage_snapshot_plan(const DistributedStageStateMapping& mapping,std::uint64_t logical) {
    require_stage(mapping.identity() && logical,"typed stage snapshot requires source mapping and logical transaction identity");
    DistributedStageSnapshotPlan p;p.mapping_identity=mapping.identity();p.logical_plan_identity=logical;
    p.device_bytes=mapping.device_bytes();p.host_bytes=mapping.host_bytes();
    for(const auto& r:mapping.ranges()) p.ranges.push_back({r.storage,r.local_offset,r.local_offset,r.bytes});
    p.plan_identity=snapshot_identity(p);require_stage(p.plan_identity,"typed stage snapshot identity is zero");return p;
}
void validate_stage_snapshot_binding(const DistributedStageSnapshotPlan& p,const DistributedStageSnapshotMemoryBinding& memory) {
    require_stage(p.mapping_identity && p.logical_plan_identity && p.plan_identity==snapshot_identity(p) &&
        !p.ranges.empty() && memory.snapshot_plan_identity==p.plan_identity && memory.allocation_generation,
        "typed stage snapshot identity/generation changed");
    // Ranges already inherit source alignment; snapshot backing is at least256B
    // aligned, including the independent host pool. No fake zero-sized owner.
    pool(memory.device_section_base,memory.device_section_bytes,p.device_bytes,
        p.device_bytes?memory.allocation_generation:0U,256);
    pool(memory.host_section_base,memory.host_section_bytes,p.host_bytes,
        p.host_bytes?memory.allocation_generation:0U,256);
    disjoint(memory.device_section_base,memory.device_section_bytes,memory.host_section_base,memory.host_section_bytes);
}
namespace {
std::vector<SessionSnapshotTransfer> prepare(const DistributedStageSnapshotPlan& p,const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& state,const DistributedStageSnapshotMemoryBinding& snapshot,bool restore) {
    require_stage(state.stage_mapping,"typed stage snapshot requires an actual stage mapping");
    state.stage_mapping->validate_memory(layout,state);
    const auto expected=build_stage_snapshot_plan(*state.stage_mapping,p.logical_plan_identity);
    require_stage(p.plan_identity==expected.plan_identity && p.mapping_identity==expected.mapping_identity &&
        p.ranges==expected.ranges && p.device_bytes==expected.device_bytes && p.host_bytes==expected.host_bytes,
        "typed stage snapshot does not cover every owned range exactly");
    validate_stage_snapshot_binding(p,snapshot);
    for(const auto& a:{std::pair{state.device_base,state.device_bytes},std::pair{state.host_base,state.host_bytes}})
        for(const auto& b:{std::pair{snapshot.device_section_base,snapshot.device_section_bytes},std::pair{snapshot.host_section_base,snapshot.host_section_bytes}})
            disjoint(a.first,a.second,b.first,b.second);
    std::vector<SessionSnapshotTransfer> transfers;transfers.reserve(p.ranges.size());
    for(const auto& r:p.ranges) {
        const auto device=r.storage==SessionStateStorage::device;
        const auto src=device?state.device_base:state.host_base;
        const auto dst=device?snapshot.device_section_base:snapshot.host_section_base;
        const auto n=device?state.device_bytes:state.host_bytes;
        const auto m=device?snapshot.device_section_bytes:snapshot.host_section_bytes;
        require_stage(r.bytes && r.source_offset<=n && r.bytes<=n-r.source_offset &&
            r.destination_offset<=m && r.bytes<=m-r.destination_offset,"typed stage snapshot range exceeds actual pools");
        const auto a=src+static_cast<std::uintptr_t>(r.source_offset),b=dst+static_cast<std::uintptr_t>(r.destination_offset);
        transfers.push_back({r.storage,restore?b:a,restore?a:b,r.bytes});
    }
    return transfers;
}
}
void capture_stage_snapshot_payload(const DistributedStageSnapshotPlan& p,const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& state,const DistributedStageSnapshotMemoryBinding& snapshot,SessionSnapshotTransferBackend& backend) {
    const auto batch=prepare(p,layout,state,snapshot,false);backend.capture_batch(batch);
}
void restore_stage_snapshot_payload(const DistributedStageSnapshotPlan& p,const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& state,const DistributedStageSnapshotMemoryBinding& snapshot,SessionSnapshotTransferBackend& backend) {
    const auto batch=prepare(p,layout,state,snapshot,true);backend.restore_batch(batch);
}
} // namespace strix
