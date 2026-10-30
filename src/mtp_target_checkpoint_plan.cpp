#include "strix/mtp_target_checkpoint_plan.hpp"
#include "strix/indexed_attention_state.hpp"
#include <limits>
#include <stdexcept>
namespace strix {
namespace {
constexpr std::uint64_t alignment=256;
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a) throw std::overflow_error("MTP checkpoint reservation overflow");
    return a+b;
}
std::uint64_t aligned(std::uint64_t n) { return n%alignment?add(n,alignment-n%alignment):n; }
}
HipMtpTargetCheckpointReservation plan_hip_mtp_target_checkpoint(
    const ModelContract& model,const SerialSessionStateLayout& layout,SessionSnapshotPolicy policy) {
    validate_session_snapshot_policy(policy);
    validate_serial_session_state_layout_identity(model,layout);
    HipMtpTargetCheckpointReservation result;
    // Sum every component irrespective of any current valid-state mask. Compact
    // KV has one aligned range per layer, so include its worst-case padding.
    for(const auto& allocation:layout.allocations) {
        if(!allocation.bytes) continue;
        auto& total=allocation.storage==SessionStateStorage::device?
            result.device_payload_bytes:result.host_payload_bytes;
        total=add(aligned(total),allocation.bytes);
        if(policy==SessionSnapshotPolicy::occupied_kv_v1 &&
           allocation.component==SessionStateComponent::attention_kv && model.full_attention_layers>1) {
            for(std::uint64_t i=1;i<model.full_attention_layers;++i) total=add(total,alignment-1);
        }
    }
    // Serial snapshot plans align the END of each section as well as each
    // range start. Reserve that actual format extent before allocator padding.
    result.device_payload_bytes=aligned(result.device_payload_bytes);
    result.host_payload_bytes=aligned(result.host_payload_bytes);
    result.retained_bytes=add(result.device_payload_bytes,result.host_payload_bytes);
    if(result.device_payload_bytes) result.retained_bytes=add(result.retained_bytes,alignment-1);
    if(result.host_payload_bytes) result.retained_bytes=add(result.retained_bytes,alignment-1);
    return result;
}

HipMtpTargetRollbackPlan plan_hip_mtp_target_window_rollback(const ModelContract& model,
    const SerialSessionStateLayout& layout,std::uint64_t position,std::uint32_t rows,bool status) {
    validate_serial_session_state_layout_identity(model,layout);
    const auto canonical=build_serial_session_state_layout(model,
        {layout.context_tokens,layout.device_alignment,layout.host_alignment,true,layout.recurrent_storage});
    if(canonical.layout_identity!=layout.layout_identity || layout.allocations.size()!=7 ||
       (rows!=2 && rows!=4 && rows!=8) || position>layout.context_tokens || rows>layout.context_tokens-position ||
       position>model.effective_max_positions() || rows>model.effective_max_positions()-position)
        throw std::invalid_argument("MTP bounded rollback requires a canonical complete layout and closed in-capacity producer");
    HipMtpTargetRollbackPlan plan;plan.first_position=position;plan.logical_rows=rows;
    const auto attention=build_indexed_attention_state_layout(model);
    const auto append=[&](const SessionStateAllocation& allocation,std::uint64_t offset,std::uint64_t bytes) {
        if(offset<allocation.offset || offset-allocation.offset>allocation.bytes || bytes>allocation.bytes-(offset-allocation.offset))
            throw std::invalid_argument("MTP rollback slice exceeds the actual component");
        auto& total=allocation.storage==SessionStateStorage::device?plan.reservation.device_payload_bytes:plan.reservation.host_payload_bytes;
        const auto destination=aligned(total);total=add(destination,bytes);
        plan.ranges.push_back({allocation.component,allocation.storage,offset,destination,bytes,false});
    };
    for(const auto& allocation:layout.allocations) {
        if(allocation.component==SessionStateComponent::attention_kv || allocation.component==SessionStateComponent::sparse_index) {
            const auto row=allocation.component==SessionStateComponent::attention_kv?attention.kv_row_bytes:attention.raw_index_row_bytes;
            if(!row || layout.context_tokens>UINT64_MAX/row || model.full_attention_layers>UINT64_MAX/(layout.context_tokens*row))
                throw std::overflow_error("MTP rollback layer stride overflow");
            const auto stride=layout.context_tokens*row;
            if(allocation.storage!=SessionStateStorage::device || allocation.bytes!=stride*model.full_attention_layers)
                throw std::invalid_argument("MTP rollback append-row geometry differs from actual canonical QSA storage");
            for(std::uint64_t layer=0;layer<model.full_attention_layers;++layer)
                append(allocation,add(allocation.offset,add(layer*stride,position*row)),rows*row);
        } else {
            // Mutable recurrent matrix, BOTH convolutions and BOTH host words
            // are not append-only: preserve their complete declared extents.
            append(allocation,allocation.offset,allocation.bytes);
        }
    }
    if(status) {
        auto& total=plan.reservation.device_payload_bytes;
        const auto offset=aligned(total);total=add(offset,8);
        plan.ranges.push_back({SessionStateComponent::position_history,SessionStateStorage::device,0,offset,8,true});
    }
    plan.reservation.retained_bytes=add(plan.reservation.device_payload_bytes,plan.reservation.host_payload_bytes);
    if(plan.reservation.device_payload_bytes) plan.reservation.retained_bytes=add(plan.reservation.retained_bytes,alignment-1);
    if(plan.reservation.host_payload_bytes) plan.reservation.retained_bytes=add(plan.reservation.retained_bytes,alignment-1);
    return plan;
}

} // namespace strix
