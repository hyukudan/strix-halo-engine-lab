#include "strix/hip_mtp_target_checkpoint.hpp"
#include "strix/indexed_attention_state.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void independent_rollback_payload_fixture() {
    using namespace strix;
    auto model=ModelContract::qwen38_flash_next();
    model.hidden_layers=8;model.full_attention_layers=2;model.gdn_layers=6;
    model.gdn_key_heads=model.gdn_value_heads=1;model.gdn_key_head_dim=model.gdn_value_head_dim=2;
    const auto layout=build_serial_session_state_layout(model,{32,256,64,true});
    const auto attention=build_indexed_attention_state_layout(model);
    const auto require=[](bool pass) {if(!pass)throw std::logic_error("independent rollback byte fixture failed");};
    for(const std::uint32_t rows:{2U,4U,8U}) {
        const std::uint64_t position=3;
        const auto plan=plan_hip_mtp_target_window_rollback(model,layout,position,rows,true);
        std::vector<std::byte> device(static_cast<std::size_t>(layout.device_total_bytes)),host(static_cast<std::size_t>(layout.host_total_bytes));
        for(std::size_t i=0;i<device.size();++i)device[i]=static_cast<std::byte>((i*17U+3U)&255U);
        for(std::size_t i=0;i<host.size();++i)host[i]=static_cast<std::byte>((i*29U+5U)&255U);
        std::array<std::byte,8> d1{std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0}};
        const auto before_device=device,before_host=host;const auto before_d1=d1;
        std::vector<std::byte> snapshot_device(static_cast<std::size_t>(plan.reservation.device_payload_bytes));
        std::vector<std::byte> snapshot_host(static_cast<std::size_t>(plan.reservation.host_payload_bytes));
        const auto copy=[&](bool restoring) {
            for(const auto& range:plan.ranges) {
                auto* state=range.deferred_status?d1.data():(range.storage==SessionStateStorage::device?device.data():host.data())+range.source_offset;
                auto* snapshot=(range.storage==SessionStateStorage::device?snapshot_device.data():snapshot_host.data())+range.snapshot_offset;
                std::memcpy(restoring?state:snapshot,restoring?snapshot:state,static_cast<std::size_t>(range.bytes));
            }
        };
        copy(false);
        // Independent mutation oracle follows actual store kernels: per-layer
        // K/V and raw-index rows only, every nonappend state component whole.
        for(const auto& allocation:layout.allocations) {
            auto& pool=allocation.storage==SessionStateStorage::device?device:host;
            if(allocation.component==SessionStateComponent::attention_kv || allocation.component==SessionStateComponent::sparse_index) {
                const auto row=allocation.component==SessionStateComponent::attention_kv?attention.kv_row_bytes:attention.raw_index_row_bytes;
                for(std::uint64_t layer=0;layer<model.full_attention_layers;++layer) {
                    const auto offset=allocation.offset+(layer*layout.context_tokens+position)*row;
                    std::fill_n(pool.data()+offset,static_cast<std::size_t>(rows*row),std::byte{0xee});
                }
            } else std::fill_n(pool.data()+allocation.offset,static_cast<std::size_t>(allocation.bytes),std::byte{0xda});
        }
        d1.fill(std::byte{0x7e});copy(true);
        require(device==before_device && host==before_host && d1==before_d1);
        const auto* kv=find_session_state_allocation(layout,SessionStateComponent::attention_kv);
        require(kv!=nullptr);
        device[static_cast<std::size_t>(kv->offset)]^=std::byte{1};copy(true);
        require(device!=before_device); // Oracle detects writes outside claimed footprint.
        require(plan.reservation.device_payload_bytes<plan_hip_mtp_target_checkpoint(model,layout,SessionSnapshotPolicy::full_components).device_payload_bytes);
    }
    bool rejected=false;
    try{(void)plan_hip_mtp_target_window_rollback(model,layout,31,2,false);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected);rejected=false;
    try{(void)plan_hip_mtp_target_window_rollback(model,layout,3,3,false);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected);
}
}

// Geometry-only source fixture: no HIP call, allocation or model execution.
int main() {
    using namespace strix;
    try {
        independent_rollback_payload_fixture();
        const auto model=ModelContract::qwen38_flash_next();
        const auto layout=build_serial_session_state_layout(model,{2048,256,64,true});
        const auto full=plan_hip_mtp_target_checkpoint(model,layout,SessionSnapshotPolicy::full_components);
        std::uint64_t device=0,host=0;
        for(const auto& allocation:layout.allocations) {
            if(!allocation.bytes) continue;
            auto& total=allocation.storage==SessionStateStorage::device?device:host;
            total=(total+255)/256*256+allocation.bytes;
        }
        device=(device+255)/256*256;host=(host+255)/256*256;
        const auto require=[](bool pass,const char* message) { if(!pass) throw std::logic_error(message); };
        require(full.device_payload_bytes==device && full.host_payload_bytes==host,"full checkpoint section endpoints must be aligned");
        require(full.retained_bytes==device+host+(device?255:0)+(host?255:0),"checkpoint retained budget must include terminal and allocator alignment");
        for(const auto policy:{SessionSnapshotPolicy::full_components,SessionSnapshotPolicy::occupied_kv_v1}) {
            const auto capacity=plan_hip_mtp_target_checkpoint(model,layout,policy);
            for(const std::uint64_t position:{0ULL,1ULL,255ULL,256ULL,2048ULL}) {
                SerialSessionSnapshotRequest request;
                request.compiled_execution_identity=17; request.runtime_compatibility_identity=23;
                request.next_position=position; request.committed_tokens=position;
                request.alignment=256; request.policy=policy;
                request.state.valid=PersistentState::attention_kv|PersistentState::recurrent_matrix|
                    PersistentState::convolution_history|PersistentState::sparse_index|
                    PersistentState::ngram_history|PersistentState::position_history;
                request.state.generations.fill(3);
                const auto plan=build_serial_session_snapshot_plan(model,layout,request);
                if(plan.device_bytes>capacity.device_payload_bytes || plan.host_bytes>capacity.host_payload_bytes)
                    throw std::logic_error("checkpoint reservation smaller than actual snapshot: policy="+
                        std::to_string(static_cast<unsigned>(policy))+" position="+std::to_string(position)+
                        " device="+std::to_string(plan.device_bytes)+"/"+std::to_string(capacity.device_payload_bytes)+
                        " host="+std::to_string(plan.host_bytes)+"/"+std::to_string(capacity.host_payload_bytes));
                if(policy==SessionSnapshotPolicy::full_components)
                    require(plan.device_bytes==capacity.device_payload_bytes && plan.host_bytes==capacity.host_payload_bytes,
                        "full checkpoint reservation must match actual all-component format exactly");
                // This request is a range-planner input, never a claimed ready
                // SerialSessionContext or native arithmetic qualification.
                const auto restore=build_serial_session_restore_plan(model,layout,23,plan);
                require(restore.snapshot_plan_identity==plan.plan_identity,"restore identity must bind actual snapshot");
            }
        }
        bool rejected=false;
        try { (void)plan_hip_mtp_target_checkpoint(model,layout,static_cast<SessionSnapshotPolicy>(99)); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"unknown checkpoint snapshot policy must reject");
        auto stale=layout; ++stale.context_tokens;
        rejected=false;
        try { (void)plan_hip_mtp_target_checkpoint(model,stale,SessionSnapshotPolicy::full_components); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"stale layout identity must reject");
        std::cout<<"checkpoint reservation geometry passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
