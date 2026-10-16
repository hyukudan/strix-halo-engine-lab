#include "strix/hip_mtp_target_checkpoint.hpp"
#include <iostream>
#include <stdexcept>

// Geometry-only source fixture: no HIP call, allocation or model execution.
int main() {
    using namespace strix;
    try {
        const auto model=ModelContract::qwen38_flash_next();
        const auto layout=build_serial_session_state_layout(model,{2048,256,64,true});
        const auto full=plan_hip_mtp_target_checkpoint(model,layout,SessionSnapshotPolicy::full_components);
        std::uint64_t device=0,host=0;
        for(const auto& allocation:layout.allocations) {
            if(!allocation.bytes) continue;
            auto& total=allocation.storage==SessionStateStorage::device?device:host;
            total=(total+255)/256*256+allocation.bytes;
        }
        const auto require=[](bool pass) { if(!pass) throw std::logic_error("checkpoint geometry fixture failed"); };
        require(full.device_payload_bytes==device && full.host_payload_bytes==host);
        require(full.retained_bytes==device+host+(device?255:0)+(host?255:0));
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
                require(plan.device_bytes<=capacity.device_payload_bytes && plan.host_bytes<=capacity.host_payload_bytes);
                // This request is a range-planner input, never a claimed ready
                // SerialSessionContext or native arithmetic qualification.
                const auto restore=build_serial_session_restore_plan(model,layout,23,plan);
                require(restore.snapshot_plan_identity==plan.plan_identity);
            }
        }
        bool rejected=false;
        try { (void)plan_hip_mtp_target_checkpoint(model,layout,static_cast<SessionSnapshotPolicy>(99)); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected);
        auto stale=layout; ++stale.context_tokens;
        rejected=false;
        try { (void)plan_hip_mtp_target_checkpoint(model,stale,SessionSnapshotPolicy::full_components); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected);
        std::cout<<"checkpoint reservation geometry passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
