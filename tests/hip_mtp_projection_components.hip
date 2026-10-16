#include "hip_mtp_projection_components.hpp"
#include <hip/hip_runtime.h>
#include <cstring>
#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
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
std::string compare(MtpDeviceView v,const void* expected,std::size_t bytes,hipStream_t stream,
    const char* role,const char* array,const char* stage,bool bf16,std::size_t offset=0) {
    const std::size_t element_bytes=bf16?2U:4U;
    require(offset<=v.bytes && bytes<=v.bytes-offset && expected && bytes%element_bytes==0,
        "MTP component golden extent");
    std::vector<std::byte> actual(bytes);
    check(hipMemcpyAsync(actual.data(),reinterpret_cast<const void*>(v.address+offset),bytes,hipMemcpyDeviceToHost,stream));
    check(hipStreamSynchronize(stream));
    if(std::memcmp(actual.data(),expected,bytes)==0)return {};
    const auto* reference=static_cast<const std::byte*>(expected);
    std::size_t first=bytes/element_bytes,mismatches=0,nonfinite_pairs=0;
    std::uint32_t first_actual=0,first_expected=0;
    float first_actual_value=0,first_expected_value=0;
    double maximum_absolute_error=0;
    for(std::size_t index=0;index<bytes/element_bytes;++index){
        std::uint32_t actual_bits=0,expected_bits=0;
        if(bf16){
            std::uint16_t a=0,e=0;std::memcpy(&a,actual.data()+index*element_bytes,2U);
            std::memcpy(&e,reference+index*element_bytes,2U);actual_bits=a;expected_bits=e;
        }else{
            std::memcpy(&actual_bits,actual.data()+index*element_bytes,4U);
            std::memcpy(&expected_bits,reference+index*element_bytes,4U);
        }
        if(actual_bits==expected_bits)continue;
        const auto actual_value=std::bit_cast<float>(bf16?actual_bits<<16U:actual_bits);
        const auto expected_value=std::bit_cast<float>(bf16?expected_bits<<16U:expected_bits);
        if(!mismatches){first=index;first_actual=actual_bits;first_expected=expected_bits;
            first_actual_value=actual_value;first_expected_value=expected_value;}
        ++mismatches;
        if(std::isfinite(actual_value) && std::isfinite(expected_value))
            maximum_absolute_error=std::max(maximum_absolute_error,std::fabs(static_cast<double>(actual_value)-expected_value));
        else ++nonfinite_pairs;
    }
    std::ostringstream message;
    message<<"\nrole="<<role<<" array="<<array<<" tensor_stage="<<stage<<" dtype="<<(bf16?"bf16":"fp32")
        <<" actual_view_bytes="<<v.bytes<<" view_offset="<<offset<<" expected_bytes="<<bytes<<" compared_bytes="<<bytes
        <<" elements="<<bytes/element_bytes<<" mismatch_count="<<mismatches<<" first_index="<<first
        <<" first_actual_bits=0x"<<std::hex<<std::setw(bf16?4:8)<<std::setfill('0')<<first_actual
        <<" first_expected_bits=0x"<<std::setw(bf16?4:8)<<first_expected<<std::dec<<std::setfill(' ')
        <<std::setprecision(17)<<" first_actual_value="<<first_actual_value<<" first_expected_value="<<first_expected_value
        <<" max_abs_error_finite_pairs="<<maximum_absolute_error<<" nonfinite_mismatch_pairs="<<nonfinite_pairs;
    return message.str();
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
            const auto* role=c.role==MtpHyperRole::attention?"hc-attention":c.role==MtpHyperRole::mlp?"hc-mlp":"hc-head";
            std::string mismatch;
            // Compare every produced array in this completed role before
            // reporting. Workspace is exactly [normalized FP32 wide, FP32 low].
            // Splitting the report preserves complete byte coverage and exposes
            // the earliest arithmetic stage without dumping expected arrays.
            mismatch+=compare(owner->view(HipMtpBuffer::hidden0),c.output.data(),c.output.size()*2U,stream,
                role,"hidden-output","q8-up-and-normalized-wide-combine",true);
            const auto normalized_bytes=c.input.size()*sizeof(float);
            mismatch+=compare(owner->view(HipMtpBuffer::hyper_workspace),c.workspace.data(),normalized_bytes,stream,
                role,"normalized-wide","norm",false);
            mismatch+=compare(owner->view(HipMtpBuffer::hyper_workspace),c.workspace.data()+normalized_bytes,
                c.workspace.size()-normalized_bytes,stream,role,"lowrank","q8-down",false,normalized_bytes);
            if(!c.injection.empty())mismatch+=compare(owner->view(HipMtpBuffer::injection),c.injection.data(),16U,stream,
                role,"injection","q8-inject",false);
            if(!mismatch.empty())throw std::invalid_argument("MTP native component differs from independent full-bit golden"+mismatch);
        }
        for(unsigned repeat=0;repeat<2U;++repeat){
            const auto& c=ref.routed;upload(owner->view(HipMtpBuffer::hidden0),c.input.data(),c.input.size()*2U,stream);
            upload(owner->view(HipMtpBuffer::routes),c.routes.data(),40U,stream);
            upload(owner->view(HipMtpBuffer::route_weights),c.route_weights.data(),40U,stream);
            owner->mark_submitted();launch_hip_mtp_routed_experts(context,stream);owner->complete();
            std::string mismatch;
            mismatch+=compare(owner->view(HipMtpBuffer::hidden2),c.output.data(),c.output.size()*2U,stream,
                "routed-experts","hidden-output","slot-ascending-route-weighted-down",true);
            // The existing independent reference covers last-slot activated
            // gate/up and the complete FP32 accumulation, with no padding.
            const auto intermediate=static_cast<std::size_t>(owner->model().moe_intermediate_size);
            const auto activation_bytes=intermediate*sizeof(float);
            mismatch+=compare(owner->view(HipMtpBuffer::moe_workspace),c.workspace.data(),activation_bytes,stream,
                "routed-experts","last-activation","q8-gate-silu-times-q8-up",false);
            mismatch+=compare(owner->view(HipMtpBuffer::moe_workspace),c.workspace.data()+activation_bytes,activation_bytes,stream,
                "routed-experts","last-up","q8-up",false,activation_bytes);
            mismatch+=compare(owner->view(HipMtpBuffer::moe_workspace),c.workspace.data()+2U*activation_bytes,
                c.workspace.size()-2U*activation_bytes,stream,"routed-experts","accumulator","slot-ascending-q8-down",false,2U*activation_bytes);
            if(!mismatch.empty())throw std::invalid_argument("MTP native component differs from independent full-bit golden"+mismatch);
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
