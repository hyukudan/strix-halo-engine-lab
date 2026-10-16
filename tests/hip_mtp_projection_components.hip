#include "hip_mtp_projection_components.hpp"
#include <hip/hip_runtime.h>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strix::testing {
namespace {
void require(bool x,const char* text){if(!x)throw std::invalid_argument(text);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
void upload(MtpDeviceView v,const void* p,std::size_t bytes,hipStream_t stream) {
    require(v.bytes>=bytes && p,"MTP component fixture upload extent");
    check(hipMemcpyAsync(reinterpret_cast<void*>(v.address),p,bytes,hipMemcpyHostToDevice,stream));
}
void compare(MtpDeviceView v,const void* expected,std::size_t bytes,hipStream_t stream) {
    require(v.bytes>=bytes && expected,"MTP component golden extent");
    std::vector<std::byte> actual(bytes);check(hipMemcpyAsync(actual.data(),reinterpret_cast<const void*>(v.address),bytes,hipMemcpyDeviceToHost,stream));
    check(hipStreamSynchronize(stream));require(std::memcmp(actual.data(),expected,bytes)==0,"MTP native component differs from independent full-bit golden");
}
}
void qualify_hip_mtp_projection_components(std::shared_ptr<HipMtpResources> owner,const MtpProjectionComponentReference& ref) {
    require(owner && ref.independent_reference_content.valid() && ref.proposal_identity==owner->plan().proposal_identity &&
        ref.position<owner->context_capacity() && ref.hyper_profile==mtp_hyper_q8_profile_v1 &&
        ref.routed_profile==mtp_routed_q8_profile_v1,"MTP component fixture needs actual owner and bound independent golden");
    require(ref.hyper[0].role==MtpHyperRole::attention && ref.hyper[1].role==MtpHyperRole::mlp && ref.hyper[2].role==MtpHyperRole::head,
        "MTP golden must cover all three HC roles");
    for(const auto& c:ref.hyper)require(c.input.size()==10240U && c.output.size()==2560U && c.workspace.size()==42240U &&
        c.injection.size()==(c.role==MtpHyperRole::head?0U:4U),"MTP HC independent golden is incomplete");
    require(ref.routed.input.size()==2560U && ref.routed.output.size()==2560U && ref.routed.workspace.size()==15360U,
        "MTP routed independent golden is incomplete");
    MtpNativeContext context{};context.model=&owner->model();context.plan=&owner->plan();
    context.tensors=&owner->tensors();context.resources=owner.get();context.position=ref.position;
    const auto stream=static_cast<hipStream_t>(owner->stream());
    // All role admission precedes the first fixture write or kernel.
    for(auto role:{MtpHyperRole::attention,MtpHyperRole::mlp,MtpHyperRole::head})preflight_hip_mtp_hyper_mix(context,role);
    preflight_hip_mtp_routed_experts(context);owner->require_idle();
    try {
        for(const auto& c:ref.hyper)for(unsigned repeat=0;repeat<2U;++repeat){
            const auto input=c.role==MtpHyperRole::attention?HipMtpBuffer::fused_wide:
                c.role==MtpHyperRole::mlp?HipMtpBuffer::wide_work:HipMtpBuffer::wide_current;
            upload(owner->view(input),c.input.data(),c.input.size()*2U,stream);
            owner->mark_submitted();launch_hip_mtp_hyper_mix(context,c.role,stream);owner->complete();
            compare(owner->view(HipMtpBuffer::hidden0),c.output.data(),c.output.size()*2U,stream);
            compare(owner->view(HipMtpBuffer::hyper_workspace),c.workspace.data(),c.workspace.size(),stream);
            if(!c.injection.empty())compare(owner->view(HipMtpBuffer::injection),c.injection.data(),16U,stream);
        }
        for(unsigned repeat=0;repeat<2U;++repeat){
            const auto& c=ref.routed;upload(owner->view(HipMtpBuffer::hidden0),c.input.data(),c.input.size()*2U,stream);
            upload(owner->view(HipMtpBuffer::routes),c.routes.data(),40U,stream);
            upload(owner->view(HipMtpBuffer::route_weights),c.route_weights.data(),40U,stream);
            owner->mark_submitted();launch_hip_mtp_routed_experts(context,stream);owner->complete();
            compare(owner->view(HipMtpBuffer::hidden2),c.output.data(),c.output.size()*2U,stream);
            compare(owner->view(HipMtpBuffer::moe_workspace),c.workspace.data(),c.workspace.size(),stream);
        }
        // Exact same retained owner context, no forged serial execution IDs.
        auto bad=context;bad.position=owner->context_capacity();bool rejected=false;
        try{preflight_hip_mtp_routed_experts(bad);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"MTP out-of-capacity preflight accepted");
        rejected=false;try{preflight_hip_mtp_hyper_mix(context,static_cast<MtpHyperRole>(99U));}
        catch(const std::invalid_argument&){rejected=true;}require(rejected,"MTP unknown HC role accepted");
        owner->require_idle();
    } catch(...) {owner->fail();throw;}
}
} // namespace strix::testing
