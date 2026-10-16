#include "hip_mtp_lm_head_components.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strix::tests {
namespace {
void require(bool value,const char* text){if(!value)throw std::invalid_argument(text);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
void append(std::vector<std::byte>& frame,std::uint64_t v){for(unsigned shift=0;shift<64U;shift+=8U)frame.push_back(static_cast<std::byte>((v>>shift)&255U));}
void append(std::vector<std::byte>& frame,const ModelContentIdentity& d){for(auto b:d.bytes)frame.push_back(static_cast<std::byte>(b));}
template<class T>ModelContentIdentity digest(const std::vector<T>& v){return sha256_bytes(std::as_bytes(std::span<const T>(v)));}
}
ModelContentIdentity hip_mtp_lm_head_golden_identity(const HipMtpLmHeadGolden& golden){
    static_assert(sizeof(float)==4U&&std::numeric_limits<float>::is_iec559);
    require(std::endian::native==std::endian::little,"MTP head golden requires little-endian storage");
    std::vector<std::byte> frame;frame.reserve(256U);
    append(frame,0x4d54504c4d474f31ULL);append(frame,golden.independent_source_content);append(frame,golden.target_content);append(frame,golden.sidecar_content);
    append(frame,golden.independent_reference_identity);append(frame,golden.plan_identity);append(frame,golden.position);
    append(frame,golden.head_profile_identity);append(frame,static_cast<std::uint64_t>(golden.recipe));
    append(frame,golden.input_bf16.size());append(frame,digest(golden.input_bf16));append(frame,golden.full_logits.size());append(frame,digest(golden.full_logits));
    return sha256_bytes(frame);
}
HipMtpLmHeadComponentReceipt qualify_hip_mtp_lm_head_component(std::shared_ptr<HipMtpResources> owner,
    const HipMtpLmHeadGolden& golden,const HipMtpLmHeadComponentRequest& request){
    require(owner&&golden.independent_source_content.valid()&&golden.independent_reference_identity&&request.expected_payload.valid(),
        "MTP head component requires actual owner and independently pinned reference");
    owner->require_idle();require(!current_hip_launch_context(),"MTP head component cannot borrow a launch scope");
    const auto stream=static_cast<hipStream_t>(owner->stream());hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check(hipStreamIsCapturing(stream,&capture));require(capture==hipStreamCaptureStatusNone,"MTP head component cannot enter capture");
    const auto h=owner->model().hidden_size,v=owner->model().vocab_size;
    require(h&&h<=UINT32_MAX&&v&&v<=UINT32_MAX&&golden.input_bf16.size()==h&&golden.full_logits.size()==v&&
        golden.target_content==owner->shared_target().target_model_content_identity&&golden.sidecar_content==owner->shared_target().sidecar_model_content_identity&&
        golden.plan_identity==owner->plan().plan_identity&&golden.position<owner->context_capacity(),"MTP head golden geometry/content/plan differs");
    // Bound supplied payload, actual full-vocabulary capture and hash frame
    // before allocating captures or submitting any input transfer.
    const auto bytes=h*2U+v*8U+256U;
    require(request.host_payload_budget_bytes>=bytes,"MTP head component host payload budget exceeded");
    for(auto bits:golden.input_bf16)require(std::isfinite(std::bit_cast<float>(static_cast<std::uint32_t>(bits)<<16U)),"MTP head input golden is nonfinite");
    for(auto value:golden.full_logits)require(std::isfinite(value),"MTP head output golden is nonfinite");
    const auto recipe=select_hip_mtp_lm_head_recipe(owner->model(),owner->tensors().lm_head);
    const auto profile=hip_mtp_lm_head_profile_identity(owner->model(),owner->tensors().lm_head);
    require(recipe==golden.recipe&&profile==golden.head_profile_identity,"MTP head golden numerical recipe/profile differs");
    const auto payload=hip_mtp_lm_head_golden_identity(golden);require(payload==request.expected_payload,"MTP head complete golden digest differs");
    std::vector<float> actual(static_cast<std::size_t>(v));
    auto validation=owner->admit_native_entry();
    const MtpNativeContext context{&owner->model(),&owner->plan(),&owner->tensors(),owner.get(),golden.position};
    preflight_hip_mtp_lm_head(context);check(hipPeekAtLastError());
    try{
        for(unsigned repeat=0;repeat<2U;++repeat){
            owner->mark_submitted();
            check(hipMemcpyAsync(reinterpret_cast<void*>(owner->view(HipMtpBuffer::hidden0).address),golden.input_bf16.data(),
                static_cast<std::size_t>(h*2U),hipMemcpyHostToDevice,stream));
            launch_hip_mtp_lm_head(context,stream);owner->complete();
            check(hipMemcpyAsync(actual.data(),reinterpret_cast<const void*>(owner->view(HipMtpBuffer::logits).address),
                static_cast<std::size_t>(v*4U),hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
            for(auto value:actual)require(std::isfinite(value),"MTP native head full output is nonfinite");
            require(std::memcmp(actual.data(),golden.full_logits.data(),static_cast<std::size_t>(v*4U))==0,
                "MTP native head full vocabulary differs from independent recipe-bound golden");
        }
        owner->require_idle();return {recipe,profile,v,payload,digest(actual)};
    }catch(...){owner->fail();throw;}
}
} // namespace strix::tests
