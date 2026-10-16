#include "hip_mtp_target_checkpoint_components.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strix::tests {
namespace {
void require(bool pass,const char* message){if(!pass)throw std::invalid_argument(message);}
void check(hipError_t result){if(result!=hipSuccess)throw std::runtime_error("MTP checkpoint component fixture HIP readback failed");}
bool same(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity && a.next_position==b.next_position &&
        a.committed_tokens==b.committed_tokens && a.health==b.health && a.state.valid==b.state.valid && a.state.generations==b.state.generations &&
        a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity && a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted &&
        a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
HipMtpTargetCheckpointComponentReceipt qualify_hip_mtp_target_window_checkpoint(
    std::shared_ptr<HipMtpTargetHidden> target,std::shared_ptr<HipKnownPrefixWindow> window,SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,std::shared_ptr<const void> actual_owner,std::span<const std::uint32_t> inputs,
    HipMtpTargetWindowCapturePolicy capture_policy,std::uint64_t checkpoint_budget,std::uint64_t readback_budget,AllocationGenerationSource& generations) {
    require(target && window && actual_owner && resources.session_state_layout &&
        resources.session_state_memory==&target->memory_owner()->state() && inputs.size()==window->logical_rows(),
        "MTP checkpoint component fixture needs actual target/window resources");
    auto capture=window->prepare_mtp_target_capture(capture_policy);target->validate_window_capture(*capture);
    const auto state=target->memory_owner()->state();const auto d1=capture->deferred_status_binding();
    require(d1.arena_base && d1.arena_bytes==8 && d1.allocation_generation,
        "MTP native checkpoint fixture requires an actual D1 record for its explicit D1 coverage");
    constexpr std::uint64_t chunk=65536;
    require(state.device_bytes<=UINT64_MAX-state.host_bytes && state.device_bytes+state.host_bytes<=UINT64_MAX-chunk-8 &&
        readback_budget && state.device_bytes+state.host_bytes+chunk+8<=readback_budget &&
        state.device_bytes<=SIZE_MAX && state.host_bytes<=SIZE_MAX,"MTP checkpoint fixture exceeds explicit baseline/readback byte budget");
    std::vector<std::byte> baseline_device(static_cast<std::size_t>(state.device_bytes)),baseline_host(static_cast<std::size_t>(state.host_bytes));
    std::array<std::byte,65536> readback{};std::array<std::byte,8> baseline_status{};
    auto recorded=session;
    auto guard=target->guard();const auto stream=static_cast<hipStream_t>(target->stream());
    // Baseline reads use the completed target stream; the rollback snapshot
    // itself must follow the distinct actual window-producer stream and owner.
    const auto checkpoint_stream=static_cast<hipStream_t>(capture->source_stream());
    if(checkpoint_stream!=stream) {
        bool wrong_stream_rejected=false;
        try {
            HipMtpTargetCheckpoint wrong(target->model(),target->compiled(),*resources.session_state_layout,
                target->memory_owner(),target->weights_owner(),target->allocator(),stream,target,guard,
                SessionSnapshotPolicy::full_components,checkpoint_budget,generations,capture);
        }catch(const std::invalid_argument&){wrong_stream_rejected=true;}
        require(wrong_stream_rejected,"bounded rollback accepted a foreign serial-target stream");
    }
    HipMtpTargetCheckpoint checkpoint(target->model(),target->compiled(),*resources.session_state_layout,
        target->memory_owner(),target->weights_owner(),target->allocator(),checkpoint_stream,window,guard,
        SessionSnapshotPolicy::full_components,checkpoint_budget,generations,capture);
    bool attempted=false;
    try {
        guard->require_original();check(hipStreamSynchronize(stream));guard->require_original();
        if(state.device_bytes)check(hipMemcpyAsync(baseline_device.data(),reinterpret_cast<const void*>(state.device_base),static_cast<std::size_t>(state.device_bytes),hipMemcpyDeviceToHost,stream));
        if(d1.arena_bytes)check(hipMemcpyAsync(baseline_status.data(),reinterpret_cast<const void*>(d1.arena_base),8,hipMemcpyDeviceToHost,stream));
        check(hipStreamSynchronize(stream));
        if(state.host_bytes)std::memcpy(baseline_host.data(),reinterpret_cast<const void*>(state.host_base),static_cast<std::size_t>(state.host_bytes));
        checkpoint.capture_window(session);
        attempted=true;
        auto execution=window->execute_mtp_captured(session,inputs,capture);
        checkpoint.restore_window(session,execution.batch);
        HipMtpTargetCheckpointComponentReceipt receipt;
        receipt.exact_session_metadata_restored=same(session,recorded);
        for(std::uint64_t offset=0;offset<state.device_bytes;) {
            const auto count=std::min(chunk,state.device_bytes-offset);
            check(hipMemcpyAsync(readback.data(),reinterpret_cast<const void*>(state.device_base+offset),static_cast<std::size_t>(count),hipMemcpyDeviceToHost,stream));
            check(hipStreamSynchronize(stream));
            require(std::equal(readback.begin(),readback.begin()+static_cast<std::ptrdiff_t>(count),baseline_device.begin()+static_cast<std::ptrdiff_t>(offset)),
                "bounded checkpoint failed exact FULL device allocation restoration");
            offset+=count;
        }
        receipt.full_device_bytes_restored=true;receipt.device_bytes_checked=state.device_bytes;
        require(!state.host_bytes || std::memcmp(baseline_host.data(),reinterpret_cast<const void*>(state.host_base),static_cast<std::size_t>(state.host_bytes))==0,
            "bounded checkpoint failed exact FULL host allocation restoration");
        receipt.full_host_bytes_restored=true;receipt.host_bytes_checked=state.host_bytes;
        if(d1.arena_bytes) {
            check(hipMemcpyAsync(readback.data(),reinterpret_cast<const void*>(d1.arena_base),8,hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
            require(std::equal(baseline_status.begin(),baseline_status.end(),readback.begin()),"bounded checkpoint failed exact D1 restoration");
        }
        receipt.deferred_status_restored=true;receipt.deferred_bytes_checked=d1.arena_bytes;
        receipt.checkpoint_reserved_bytes=checkpoint.reserved_bytes();
        require(receipt.exact_session_metadata_restored,"bounded checkpoint failed recorded metadata/lineage restoration");
        guard->require_original();return receipt;
    }catch(...) {
        if(attempted){session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;}
        guard->recover_or_fail_stop();throw;
    }
}
}
