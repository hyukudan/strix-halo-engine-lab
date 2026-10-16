#include "hip_mtp_shared_expert_components.hpp"
#include <hip/hip_runtime.h>
#include <cstring>
#include <stdexcept>
namespace strix::testing {
namespace {
void require(bool x,const char* text){if(!x)throw std::invalid_argument(text);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
}
void qualify_hip_mtp_shared_expert_component(std::shared_ptr<HipMtpResources> owner,const MtpSharedComponentReference& ref){
    require(owner && ref.independent_reference_content.valid() && ref.proposal_identity==owner->plan().proposal_identity &&
        ref.position<owner->context_capacity() && ref.numerical_profile==mtp_shared_q8_profile_v1 &&
        ref.input.size()==2560U && ref.output.size()==2560U && ref.workspace.size()==5120U,
        "MTP shared component needs independently bound complete golden payloads");
    MtpNativeContext context{};context.model=&owner->model();context.plan=&owner->plan();context.tensors=&owner->tensors();
    context.resources=owner.get();context.position=ref.position;
    preflight_hip_mtp_shared_expert(context);owner->require_idle();
    const auto stream=static_cast<hipStream_t>(owner->stream());
    std::vector<std::uint16_t> output(2560U);std::vector<std::byte> workspace(5120U);
    try{
        for(unsigned repeat=0;repeat<2U;++repeat){
            check(hipMemcpyAsync(reinterpret_cast<void*>(owner->view(HipMtpBuffer::hidden0).address),ref.input.data(),5120U,hipMemcpyHostToDevice,stream));
            owner->mark_submitted();launch_hip_mtp_shared_expert(context,stream);owner->complete();
            check(hipMemcpyAsync(output.data(),reinterpret_cast<const void*>(owner->view(HipMtpBuffer::hidden1).address),5120U,hipMemcpyDeviceToHost,stream));
            check(hipMemcpyAsync(workspace.data(),reinterpret_cast<const void*>(owner->view(HipMtpBuffer::moe_workspace).address),5120U,hipMemcpyDeviceToHost,stream));
            check(hipStreamSynchronize(stream));
            require(output==ref.output && workspace==ref.workspace,"MTP shared component differs from independent output/full-workspace golden");
        }
        auto wrong=context;wrong.position=owner->context_capacity();bool rejected=false;
        try{preflight_hip_mtp_shared_expert(wrong);}catch(const std::out_of_range&){rejected=true;}
        require(rejected,"MTP shared out-of-capacity context admitted");owner->require_idle();
    }catch(...){owner->fail();throw;}
}
} // namespace strix::testing
