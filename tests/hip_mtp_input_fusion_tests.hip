// Native component SOURCE fixture; no checkpoint/upstream equivalence claim.
#include "strix/hip_mtp_input_fusion.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
struct Buffer {
    strix::HipSetupQuiescenceGuard guard;
    void* p=nullptr;std::uint64_t bytes;
    explicit Buffer(std::uint64_t n):bytes(n){guard.run([&]{check(hipMalloc(&p,static_cast<std::size_t>(n)));});}
    ~Buffer(){guard.recover_or_fail_stop();if(p)static_cast<void>(hipFree(p));}
    strix::HipMtpInputFusionView view()const{return {reinterpret_cast<std::uintptr_t>(p),bytes,1U};}
};
struct Stream {
    strix::HipSetupQuiescenceGuard guard;hipStream_t value=nullptr;
    Stream(){guard.run([&]{check(hipStreamCreateWithFlags(&value,hipStreamNonBlocking));});}
    ~Stream(){guard.recover_or_fail_stop();if(value)static_cast<void>(hipStreamDestroy(value));}
};
std::uint16_t bf16(float f){const auto b=std::bit_cast<std::uint32_t>(f);return static_cast<std::uint16_t>((b+0x7fffU+((b>>16U)&1U))>>16U);}
std::vector<unsigned char> identity_weights(unsigned h,bool embedding) {
    const unsigned stride=2U*h/32U*34U;std::vector<unsigned char> w(h*stride);
    for(unsigned row=0;row<h;++row) {
        for(unsigned block=0;block<2U*h/32U;++block) {const std::uint16_t one=0x3c00U;std::memcpy(w.data()+row*stride+block*34U,&one,2U);}
        const unsigned col=embedding?row:h+row;
        w[row*stride+(col/32U)*34U+2U+col%32U]=1U;
    }
    return w;
}
template<class T>void upload(Buffer& b,const std::vector<T>& values,Stream& s){require(values.size()*sizeof(T)==b.bytes,"fixture upload extent");check(hipMemcpyAsync(b.p,values.data(),static_cast<std::size_t>(b.bytes),hipMemcpyHostToDevice,s.value));}
void qualify() {
    constexpr unsigned h=32U,hc=2U,wide=h*hc;
    Stream stream;Buffer e(h*4U),x(wide*4U),en(h*4U),hn(wide*4U),weights(h*2U*h/32U*34U);
    const auto scratch=strix::hip_mtp_input_fusion_scratch_bytes(h,hc);
    Buffer n(scratch),p(scratch),out(wide*2U);
    std::vector<float> ev(h,0.0F),hv(wide,1.0F),ew(h,1.0F),hw(wide,1.0F);
    for(unsigned i=h;i<wide;++i)hv[i]=3.0F;
    auto w=identity_weights(h,false);
    upload(e,ev,stream);upload(x,hv,stream);upload(en,ew,stream);upload(hn,hw,stream);upload(weights,w,stream);
    strix::HipMtpInputFusionRequest r;r.hidden_size=h;r.hc_count=hc;r.epsilon=4.0F;
    r.embedding_storage=r.hidden_storage=strix::HipMtpInputStorage::fp32;r.output_storage=strix::HipMtpInputStorage::bf16;
    r.normalized_rounding=r.projection_rounding=strix::HipMtpInputRounding::bf16_rne;
    r.norm_encoding_admitted=true;r.norm_weight_encoding=strix::MtpNormWeightEncoding::effective_scale;
    r.embedding=e.view();r.wide_hidden=x.view();r.embedding_norm=en.view();r.hidden_norm=hn.view();
    r.packed_projection=weights.view();r.projection_row_stride_bytes=2U*h/32U*34U;
    r.normalized_scratch=n.view();r.projection_scratch=p.view();r.output=out.view();
    check(hipStreamSynchronize(stream.value));
    strix::preflight_hip_mtp_input_fusion(r);
    const auto identity=strix::hip_mtp_input_fusion_profile_identity(r);
    auto profile=r;profile.normalized_rounding=strix::HipMtpInputRounding::fp32;
    require(identity!=strix::hip_mtp_input_fusion_profile_identity(profile),"normalization cast boundaries need distinct identity");
    std::vector<std::uint16_t> actual(wide);
    for(unsigned repeat=0;repeat<2U;++repeat) {
        strix::launch_hip_mtp_input_fusion(r,stream.value);
        check(hipMemcpyAsync(actual.data(),out.p,out.bytes,hipMemcpyDeviceToHost,stream.value));check(hipStreamSynchronize(stream.value));
        // All-wide mean5 plus epsilon4 -> denominator3. Separate hidden
        // identity projection, zero embedding, explicit BF16 stage casts.
        for(unsigned i=0;i<wide;++i)require(actual[i]==bf16(i<h?1.0F/3.0F:1.0F),"wide RMS/split/broadcast component mismatch");
    }
    // Zero raw offsets represent the SAME effective gain1, not gain0 or2.
    upload(en,std::vector<float>(h,0.0F),stream);upload(hn,std::vector<float>(wide,0.0F),stream);
    auto raw=r;raw.norm_weight_encoding=strix::MtpNormWeightEncoding::gemma_offset;
    strix::launch_hip_mtp_input_fusion(raw,stream.value);
    std::vector<std::uint16_t> converted(wide);check(hipMemcpyAsync(converted.data(),out.p,out.bytes,hipMemcpyDeviceToHost,stream.value));check(hipStreamSynchronize(stream.value));
    require(converted==actual,"raw-offset versus effective-scale encoding differs");
    // Every host rejection precedes a kernel and preserves the full output.
    for(unsigned fault=0;fault<7U;++fault) {
        auto bad=r;
        if(fault==0U)bad.hidden_norm.bytes-=4U;
        if(fault==1U)bad.projection_row_stride_bytes/=2U;
        if(fault==2U)bad.output=bad.wide_hidden;
        if(fault==3U)bad.norm_encoding_admitted=false;
        if(fault==4U)bad.arithmetic=strix::MtpInputFusionArithmetic::packed_concat_chain;
        if(fault==5U)bad.normalized_scratch.allocation_generation=0U;
        if(fault==6U)bad.projection_scratch=bad.normalized_scratch;
        bool rejected=false;try{strix::launch_hip_mtp_input_fusion(bad,stream.value);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid input-fusion descriptor accepted");
        check(hipMemcpyAsync(converted.data(),out.p,out.bytes,hipMemcpyDeviceToHost,stream.value));check(hipStreamSynchronize(stream.value));
        require(converted==actual,"host admission failure changed output");
    }
    check(hipStreamBeginCapture(stream.value,hipStreamCaptureModeThreadLocal));
    bool capture_rejected=false;try{strix::launch_hip_mtp_input_fusion(r,stream.value);}catch(const std::invalid_argument&){capture_rejected=true;}
    hipGraph_t graph=nullptr;check(hipStreamEndCapture(stream.value,&graph));std::size_t nodes=0;check(hipGraphGetNodes(graph,nullptr,&nodes));
    static_cast<void>(hipGraphDestroy(graph));require(capture_rejected&&nodes==0U,"active capture must reject before nodes");
    // Independent projection side: nonzero embedding is projected once and
    // broadcast. Both column halves address the original complete Q8 row.
    upload(en,ew,stream);upload(hn,hw,stream);upload(e,std::vector<float>(h,1.0F),stream);upload(x,std::vector<float>(wide,1.0F),stream);
    upload(weights,identity_weights(h,true),stream);r.epsilon=3.0F;
    strix::launch_hip_mtp_input_fusion(r,stream.value);check(hipMemcpyAsync(actual.data(),out.p,out.bytes,hipMemcpyDeviceToHost,stream.value));check(hipStreamSynchronize(stream.value));
    for(auto value:actual)require(value==bf16(0.5F),"embedding projection was not broadcast to each stream");
}
// Production geometry takes the staged kernel: its projection scratch must equal
// host coefficient-first ascending fmaf chains over the device-normalized input.
void qualify_staged() {
    constexpr unsigned h=2560U,hc=4U,wide=h*hc,stride=2U*h/32U*34U;
    Stream stream;Buffer e(h*4U),x(wide*4U),en(h*4U),hn(wide*4U),weights(static_cast<std::uint64_t>(h)*stride);
    const auto scratch=strix::hip_mtp_input_fusion_scratch_bytes(h,hc);
    Buffer n(scratch),p(scratch),out(wide*2U);
    std::uint32_t seed=12345U;const auto next=[&]{seed=seed*1664525U+1013904223U;return seed;};
    const auto uniform=[&]{return static_cast<float>(next()>>8U)/16777216.0F*2.0F-1.0F;};
    std::vector<float> ev(h),hv(wide),ew(h),hw(wide);
    for(auto& v:ev)v=uniform();
    for(auto& v:hv)v=uniform();
    for(auto& v:ew)v=uniform();
    for(auto& v:hw)v=uniform();
    std::vector<unsigned char> w(static_cast<std::size_t>(h)*stride);
    for(unsigned row=0;row<h;++row)for(unsigned block=0;block<2U*h/32U;++block) {
        auto* b=w.data()+static_cast<std::size_t>(row)*stride+block*34U;
        const std::uint16_t scale=static_cast<std::uint16_t>(0x2000U+(next()%0x1000U));std::memcpy(b,&scale,2U);
        for(unsigned k=0;k<32U;++k)b[2U+k]=static_cast<unsigned char>(next());
    }
    upload(e,ev,stream);upload(x,hv,stream);upload(en,ew,stream);upload(hn,hw,stream);upload(weights,w,stream);
    for(const auto rounding:{strix::HipMtpInputRounding::fp32,strix::HipMtpInputRounding::bf16_rne}) {
        strix::HipMtpInputFusionRequest r;r.hidden_size=h;r.hc_count=hc;r.epsilon=1e-6F;
        r.embedding_storage=r.hidden_storage=strix::HipMtpInputStorage::fp32;r.output_storage=strix::HipMtpInputStorage::bf16;
        r.normalized_rounding=strix::HipMtpInputRounding::fp32;r.projection_rounding=rounding;
        r.norm_encoding_admitted=true;r.norm_weight_encoding=strix::MtpNormWeightEncoding::effective_scale;
        r.embedding=e.view();r.wide_hidden=x.view();r.embedding_norm=en.view();r.hidden_norm=hn.view();
        r.packed_projection=weights.view();r.projection_row_stride_bytes=stride;
        r.normalized_scratch=n.view();r.projection_scratch=p.view();r.output=out.view();
        strix::launch_hip_mtp_input_fusion(r,stream.value);
        std::vector<float> normalized(scratch/4U),projected(scratch/4U);
        check(hipMemcpyAsync(normalized.data(),n.p,scratch,hipMemcpyDeviceToHost,stream.value));
        check(hipMemcpyAsync(projected.data(),p.p,scratch,hipMemcpyDeviceToHost,stream.value));check(hipStreamSynchronize(stream.value));
        const auto coefficient=[&](unsigned row,unsigned column){
            const auto* b=w.data()+static_cast<std::size_t>(row)*stride+(column/32U)*34U;std::uint16_t bits;std::memcpy(&bits,b,2U);
            return static_cast<float>(std::bit_cast<_Float16>(bits))*static_cast<float>(static_cast<signed char>(b[2U+column%32U]));};
        const auto round=[&](float v){return rounding==strix::HipMtpInputRounding::bf16_rne?std::bit_cast<float>(static_cast<std::uint32_t>(bf16(v))<<16U):v;};
        for(unsigned row=0;row<h;++row) {
            float embedding=0.0F,sums[hc]={};
            for(unsigned col=0;col<h;++col)embedding=std::fma(coefficient(row,col),normalized[col],embedding);
            for(unsigned col=0;col<h;++col){const float c=coefficient(row,h+col);for(unsigned s=0;s<hc;++s)sums[s]=std::fma(c,normalized[h+s*h+col],sums[s]);}
            require(std::bit_cast<std::uint32_t>(projected[row])==std::bit_cast<std::uint32_t>(round(embedding)),"staged embedding chain differs");
            for(unsigned s=0;s<hc;++s)
                require(std::bit_cast<std::uint32_t>(projected[h+s*h+row])==std::bit_cast<std::uint32_t>(round(sums[s])),"staged HC chain differs");
        }
    }
}
}
int main(){try{qualify();qualify_staged();std::cout<<"MTP input fusion component passed; checkpoint qualification separate\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
