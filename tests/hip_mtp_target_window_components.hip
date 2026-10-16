#include "hip_mtp_target_window_components.hpp"
#include <hip/hip_runtime.h>
#include <cstring>
#include <stdexcept>
namespace strix::testing {
namespace {
void require(bool v,const char* message){if(!v)throw std::invalid_argument(message);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
}
void qualify_hip_mtp_target_window_capture(HipKnownPrefixWindow& window,SerialSessionContext& session,
    const MtpTargetWindowReference& reference,HipMtpTargetWindowCapturePolicy policy){
    require(reference.independent_reference_content.valid() && reference.target_content.valid() && reference.chunks.size()>=2U &&
        !window.supports_tiled_prefill() && window.output_policy()==KnownWindowOutputPolicy::all_rows &&
        session.next_position==reference.initial_position,"MTP window component requires independent all-head native chunks");
    const auto rows=window.logical_rows();
    for(const auto& chunk:reference.chunks){require(chunk.inputs.size()==rows && chunk.predictions.size()==rows && chunk.wide_hidden.size()==rows,
        "MTP window independent reference row count differs");for(const auto& hidden:chunk.wide_hidden)require(hidden.size()==10240U,"MTP window reference must contain every BF16 wide value");}
    auto capture=window.prepare_mtp_target_capture(policy);require(capture->ready_for_reuse(),"fresh capture slots are not available");
    std::vector<MtpDeviceView> original_views;original_views.reserve(rows);std::uint64_t previous_epoch=0;
    std::vector<std::uint16_t> actual(10240U);
    for(const auto& chunk:reference.chunks){
        const auto base=session;
        auto result=window.execute_mtp_captured(session,chunk.inputs,capture);
        require(result.batch.valid() && !capture->ready_for_reuse() && result.batch.logical_rows()==rows &&
            result.batch.epoch()>previous_epoch && result.batch.chunk_begin()==base.next_position &&
            result.batch.chunk_end()==session.next_position && result.batch.target_content_identity()==reference.target_content &&
            result.batch.base_session().next_position==base.next_position && result.batch.base_session().committed_tokens==base.committed_tokens &&
            result.batch.base_session().state.generations==base.state.generations &&
            result.batch.base_session().arithmetic_lineage.intervals==base.arithmetic_lineage.intervals &&
            result.batch.committed_session().next_position==session.next_position &&
            result.batch.committed_session().state.generations==session.state.generations &&
            result.batch.committed_session().arithmetic_lineage.intervals==session.arithmetic_lineage.intervals,
            "capture fabricated a row session or failed to preserve actual whole-chunk metadata");
        for(std::uint32_t row=0;row<rows;++row){
            const auto view=result.batch.view(row);require(result.batch.source_position(row)==base.next_position+row &&
                result.batch.input_token(row)==chunk.inputs[row] && result.batch.predicted_token(row)==chunk.predictions[row],
                "captured target row position/token/finite prediction differs");
            if(!previous_epoch)original_views.push_back(view);
            else require(view==original_views[row],"capture pool fabricated a new physical generation or reallocated between chunks");
            check(hipMemcpy(actual.data(),reinterpret_cast<const void*>(view.address),20480U,hipMemcpyDeviceToHost));
            require(actual==chunk.wide_hidden[row],"captured wide row differs from independent complete BF16 golden");
        }
        auto retained=result.batch;result.batch={};
        require(!capture->ready_for_reuse(),"copied published batch failed to pin its physical slots");
        previous_epoch=retained.epoch();retained={};
        require(capture->ready_for_reuse(),"expired batch lease did not release capture slots for reuse");
    }
}
} // namespace strix::testing
