#include "strix/distributed_stage_state.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/model_tensor_index.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix;
namespace {
int failures=0;
void expect(bool value,const char* why) {if(!value) {++failures;std::cerr<<why<<'\n';}}
template<class F> bool rejects(F&& f) {try {f();} catch(const std::exception&) {return true;}return false;}
template<class F> bool typed_guard(F&& f) {
    try {f();} catch(const std::invalid_argument& e) {return std::string(e.what()).find("distinct typed stage snapshot")!=std::string::npos;}
    return false;
}
struct Fixture {
    ModelContract model=ModelContract::qwen38_flash_next();
    CompiledSerialExecution serial;
    ModelManifest manifest;
    ExecutionTensorAccessPlan access;
    DistributedPlacementPlan placement;
    SerialSessionStateLayout layout;
    Fixture() {
        model.hidden_layers=8;model.full_attention_layers=2;model.gdn_layers=6;
        model.gdn_key_heads=model.gdn_value_heads=1;model.gdn_key_head_dim=model.gdn_value_head_dim=2;
        serial.execution=build_serial_decode_execution_plan(model);
        serial.io=build_serial_execution_io_plan(serial.execution);
        serial.arena=build_serial_execution_arena_plan(model,{},serial.execution,serial.io);
        serial.execution_identity=101;serial.kernel_plan_identity=102;
        serial.vocab_size=model.vocab_size;serial.max_position_embeddings=model.max_position_embeddings;
        serial.bindings.bindings.resize(serial.execution.steps.size());
        const auto requirements=build_serial_execution_tensor_requirements(serial.execution);
        std::set<std::string> names;
        for(const auto& r:requirements) {
            std::string tag(r.selector.name_contains);
            if(tag.empty() && !r.selector.name_contains_any.empty()) tag=r.selector.name_contains_any.front();
            const auto name=std::to_string(static_cast<unsigned>(*r.selector.role))+"."+
                (r.selector.layer?std::to_string(*r.selector.layer):"global")+"."+tag;
            if(!names.insert(name).second) continue;
            ManifestTensor t;t.name=name;t.role=*r.selector.role;t.layer=r.selector.layer;
            t.dimensions={1};t.ggml_type=0;t.byte_size=4;t.absolute_offset=256U+manifest.tensors.size()*256U;
            manifest.total_tensor_bytes+=t.byte_size;manifest.tensors.push_back(std::move(t));
        }
        manifest.layout_identity=model_manifest_layout_identity(manifest);
        access=compile_execution_tensor_access_plan(serial.execution,manifest,build_model_tensor_index(manifest),requirements);
        std::vector<std::uint32_t> nodes(serial.execution.steps.size());
        for(std::size_t i=0;i<nodes.size();++i) {
            const auto& s=serial.execution.steps[i];
            if((s.layer && *s.layer>=2U) || s.op==SerialOp::final_hyper_mix || s.op==SerialOp::lm_head) nodes[i]=1U;
        }
        placement=build_distributed_placement_plan(serial,manifest,access,nodes,0U);
        layout=build_serial_session_state_layout(model,{4,256,64,true});
    }
};
// CPU-owned aligned vectors only. These are NOT GPU or native allocation proof.
struct Pool {
    std::vector<std::byte> storage;
    std::uintptr_t base=0;
    std::uint64_t bytes=0;
    explicit Pool(std::uint64_t n):storage(static_cast<std::size_t>(n+512U),std::byte{0xa5}),bytes(n) {
        if(n) base=(reinterpret_cast<std::uintptr_t>(storage.data())+255U)&~std::uintptr_t{255U};
    }
};
struct Batch final:SessionSnapshotTransferBackend {
    unsigned captures=0,restores=0,copies=0;
    void capture_batch(std::span<const SessionSnapshotTransfer> t) override {++captures;SessionSnapshotTransferBackend::capture_batch(t);}
    void restore_batch(std::span<const SessionSnapshotTransfer> t) override {++restores;SessionSnapshotTransferBackend::restore_batch(t);}
    void copy(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) {++copies;std::memcpy(reinterpret_cast<void*>(b),reinterpret_cast<const void*>(a),static_cast<std::size_t>(n));}
    void capture_device(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {copy(a,b,n);}
    void capture_host(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {copy(a,b,n);}
    void restore_device(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {copy(a,b,n);}
    void restore_host(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {copy(a,b,n);}
};
}
int main() {
    Fixture f;
    const auto p0=DistributedStageResourcePlan::derive(f.serial,f.manifest,f.access,f.placement,0);
    const auto p1=DistributedStageResourcePlan::derive(f.serial,f.manifest,f.access,f.placement,1);
    const auto m0=DistributedStageStateMapping::derive(f.serial,f.model,f.layout,p0);
    const auto m1=DistributedStageStateMapping::derive(f.serial,f.model,f.layout,p1);
    m0.validate_source(f.serial,f.model,f.layout,p0);m1.validate_source(f.serial,f.model,f.layout,p1);
    expect(m0.identity()!=m1.identity() && m0.layout_identity()==f.layout.layout_identity &&
        m1.execution_identity()==f.serial.execution_identity,"stage mapping retains global IDs and distinct actual local ownership");
    expect(m0.presence(SessionStateComponent::attention_kv)==DistributedStageStatePresence::absent &&
        m1.presence(SessionStateComponent::attention_kv)==DistributedStageStatePresence::complete &&
        m0.presence(SessionStateComponent::recurrent_matrix)==DistributedStageStatePresence::partial,
        "absence is explicit and QSA owns full capacity only on its actual layer stage");
    expect(m0.presence(SessionStateComponent::ple_convolution)==DistributedStageStatePresence::complete &&
        m1.presence(SessionStateComponent::ple_convolution)==DistributedStageStatePresence::absent &&
        m0.presence(SessionStateComponent::position_history)==DistributedStageStatePresence::complete &&
        m1.presence(SessionStateComponent::position_history)==DistributedStageStatePresence::complete,
        "PLE state is single-owner and full position word is replicated");
    for(const auto* mapping:{&m0,&m1}) {
        Pool device(mapping->device_bytes()),host(mapping->host_bytes());
        SessionStateMemoryBinding memory{f.layout.layout_identity,device.base,device.bytes,device.bytes?11U:0U,
            host.base,host.bytes,12,mapping};
        validate_session_state_memory_binding(f.layout,memory);
        for(const auto& r:mapping->ranges()) {
            const auto address=mapping->resolve(f.layout,memory,r.component,r.logical_offset,r.bytes);
            expect(address.address==(r.storage==SessionStateStorage::device?device.base:host.base)+r.local_offset &&
                address.bytes==r.bytes,"direct logical slice resolves exact packed offset, extent and generation");
            std::memset(reinterpret_cast<void*>(address.address),static_cast<int>(static_cast<unsigned>(r.component)+17U),static_cast<std::size_t>(r.bytes));
        }
        expect(rejects([&]{(void)resolve_session_state(f.layout,memory,SessionStateComponent::recurrent_matrix);}),
            "whole component resolver never clips partial recurrent state");
        const auto own_layer=mapping==&m0?0U:2U;const auto foreign_layer=mapping==&m0?2U:0U;
        const auto local=resolve_gdn_layer_state(f.layout,memory,f.model,own_layer,SessionStateComponent::recurrent_matrix);
        expect(local.bytes>0U && rejects([&]{(void)resolve_gdn_layer_state(f.layout,memory,f.model,foreign_layer,SessionStateComponent::recurrent_matrix);}),
            "layer resolver selects actual local logical slice and rejects foreign layer");
        if(mapping==&m1) {
            const auto first=resolve_indexed_attention_layer_state(f.layout,memory,f.model,3U,SessionStateComponent::attention_kv);
            const auto last=resolve_indexed_attention_layer_state(f.layout,memory,f.model,7U,SessionStateComponent::attention_kv);
            const auto whole=resolve_session_state(f.layout,memory,SessionStateComponent::attention_kv);
            expect(first.bytes && whole.bytes==first.bytes+last.bytes && last.address==first.address+first.bytes,
                "complete physically contiguous multilayer QSA component resolves without clipping");
        }
        else expect(rejects([&]{(void)resolve_indexed_attention_layer_state(f.layout,memory,f.model,3U,SessionStateComponent::attention_kv);}),
            "nonlocal QSA layer has no invented state owner");
        const auto plan=build_stage_snapshot_plan(*mapping,301);
        Pool saved_device(plan.device_bytes),saved_host(plan.host_bytes);
        DistributedStageSnapshotMemoryBinding saved{plan.plan_identity,saved_device.base,saved_device.bytes,saved_host.base,saved_host.bytes,21};
        Batch backend;capture_stage_snapshot_payload(plan,f.layout,memory,saved,backend);
        expect(backend.captures==1U && backend.copies==mapping->ranges().size(),"typed capture admits all owned ranges then submits one completed batch");
        std::memset(reinterpret_cast<void*>(device.base),0,static_cast<std::size_t>(device.bytes));
        std::memset(reinterpret_cast<void*>(host.base),0,static_cast<std::size_t>(host.bytes));
        restore_stage_snapshot_payload(plan,f.layout,memory,saved,backend);
        for(const auto& r:mapping->ranges()) {
            const auto address=mapping->resolve(f.layout,memory,r.component,r.logical_offset,r.bytes);
            const auto* data=reinterpret_cast<const std::byte*>(address.address);
            expect(std::all_of(data,data+r.bytes,[&](std::byte b){return b==static_cast<std::byte>(static_cast<unsigned>(r.component)+17U);}),
                "typed restore reproduces every actually owned byte without full-session publication");
        }
        expect(backend.restores==1U,"typed restore is one completed batch");
        const auto next=build_stage_snapshot_plan(*mapping,302);
        expect(next.plan_identity!=plan.plan_identity && next.device_bytes==plan.device_bytes,"logical snapshot rebind preserves actual physical capacity");
        for(unsigned fault=0;fault<9U;++fault) {
            auto bad=memory;auto corrupt=plan;auto wrong=saved;
            if(fault==0) bad.stage_mapping=mapping==&m0?&m1:&m0;
            if(fault==1) ++bad.layout_identity;
            if(fault==2) ++bad.device_base;
            if(fault==3) bad.device_allocation_generation=0;
            if(fault==4) --bad.device_bytes;
            if(fault==5) corrupt.ranges.pop_back();
            if(fault==6) ++corrupt.ranges.back().source_offset;
            if(fault==7) wrong.allocation_generation=0;
            if(fault==8) wrong.host_section_base=wrong.device_section_base;
            const auto before=backend.captures;
            expect(rejects([&]{capture_stage_snapshot_payload(corrupt,f.layout,bad,wrong,backend);}) && backend.captures==before,
                "mapping/identity/alignment/extent/census/snapshot alias mutations reject before any batch");
        }
        const auto before=backend.copies;
        for(const auto policy:{SessionSnapshotPolicy::full_components,SessionSnapshotPolicy::occupied_kv_v1}) {
            SerialSessionSnapshotPlan ordinary;ordinary.policy=policy;
            SerialSessionRestorePlan restore;restore.policy=policy;SerialSessionContext session;
            const auto health=session.health;
            expect(typed_guard([&]{capture_serial_session_snapshot(ordinary,f.layout,memory,{},backend);}) &&
                typed_guard([&]{restore_serial_session_snapshot_payload(restore,f.layout,memory,{},backend);}) &&
                typed_guard([&]{restore_serial_session_snapshot(session,restore,f.layout,memory,{},backend);}) &&
                backend.copies==before && session.health==health,
                "full/compact capture and both restore entrypoints explicitly reject stage format before work/publication");
        }
        auto changed=f.layout;changed.allocations.back().bytes+=8U;
        expect(rejects([&]{mapping->validate_layout(changed);}),"stale layout metadata rejects even with retained layout ID");
        expect(rejects([&]{mapping->validate_source(f.serial,f.model,f.layout,mapping==&m0?p1:p0);}),"foreign stage source rejects immutable mapping reuse");
    }
    return failures?1:0;
}
