#include "strix/hip_hc_deferred_norm_tests.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "hc_deferred_norm_oracle.hpp"
#include <algorithm>
#include <iostream>
#include <random>

namespace {
namespace api=strix::hc_deferred_norm_diagnostic;
namespace oracle=hc_deferred_norm_oracle;
using api::Span;
void expect(bool value,const char* message) {if (!value) throw std::runtime_error(message);}
void check(hipError_t e) {if (e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e));}
struct Buffer {
    void* p=nullptr;std::size_t bytes;const strix::HipSetupQuiescenceGuard& guard;
    Buffer(std::size_t n,const strix::HipSetupQuiescenceGuard& g):bytes(n),guard(g) {check(hipMalloc(&p,n+32U));}
    ~Buffer() {if (p) {guard.recover_or_fail_stop();(void)hipFree(p);}}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void* data() const {return static_cast<unsigned char*>(p)+16U;}
    Span span() const {return bytes?Span{reinterpret_cast<std::uintptr_t>(data()),bytes}:Span{};}
    void poison(hipStream_t stream) const {check(hipMemsetAsync(p,0xa5,bytes+32U,stream));}
    void upload(const void* source) const {if (bytes) check(hipMemcpy(data(),source,bytes,hipMemcpyHostToDevice));}
    std::vector<unsigned char> owner() const {
        std::vector<unsigned char> value(bytes+32U);
        guard.run([&] {check(hipMemcpy(value.data(),p,value.size(),hipMemcpyDeviceToHost));});return value;
    }
    template<class T> std::vector<T> read() const {
        std::vector<T> value(bytes/sizeof(T));
        if (bytes) guard.run([&] {check(hipMemcpy(value.data(),data(),bytes,hipMemcpyDeviceToHost));});
        return value;
    }
    void guards() const {
        const auto value=owner();
        for (unsigned i=0;i<16;++i) expect(value[i]==0xa5 && value[value.size()-16U+i]==0xa5,"HC-F actual owner guard changed");
    }
};
struct Stream {
    hipStream_t value=nullptr;const strix::HipSetupQuiescenceGuard& guard;
    explicit Stream(const strix::HipSetupQuiescenceGuard& g):guard(g) {check(hipStreamCreate(&value));}
    ~Stream() {if (value) {guard.recover_or_fail_stop();(void)hipStreamDestroy(value);}}
    Stream(const Stream&)=delete;Stream& operator=(const Stream&)=delete;
};
void fixture(unsigned pattern,bool with_inject,const strix::HipSetupQuiescenceGuard& guard) {
    Stream stream(guard);
    std::vector<std::uint16_t> raw(10240U,0x3f00U);
    std::vector<float> norm(10240U,1.0F),inject(with_inject?40960U:0U,0x1p-10F);
    std::vector<std::byte> down(3481600U,std::byte{1}),up(3481600U,std::byte{0});
    std::mt19937 random(0x484346U+pattern);
    float epsilon=0.75F;
    if (pattern!=0) {
        epsilon=0x1p-12F;
        for (unsigned i=0;i<raw.size();++i) {
            const auto value=std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),-7);
            raw[i]=static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value)>>16U);
            norm[i]=std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),-6);
            if (i%257U==0) raw[i]=(i&1U)?0x8000U:0U;
        }
        for (auto& value:inject) value=pattern==1?0.0F:
            std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),-10);
    }
    for (std::size_t block=0;block<down.size()/34U;++block) {
        down[block*34U]=std::byte{0};down[block*34U+1U]=std::byte{0x14};
        for (unsigned j=0;j<32;++j) down[block*34U+2U+j]=pattern==0?std::byte{1}:
            pattern==1?std::byte{0}:static_cast<std::byte>(random()&255U);
    }
    Buffer raw_b(20480,guard),norm_b(40960,guard),down_b(down.size(),guard),inject_b(inject.size()*4U,guard),
        up_b(up.size(),guard),inverse_b(16,guard),low_b(1280,guard),injection_b(with_inject?16:0,guard),
        out_b(5120,guard),inverse_audit((with_inject?81U:80U)*16U,guard),scaled_b((with_inject?324U:320U)*4U,guard);
    const std::array<const Buffer*,11> buffers{&raw_b,&norm_b,&down_b,&inject_b,&up_b,&inverse_b,&low_b,
        &injection_b,&out_b,&inverse_audit,&scaled_b};
    const auto poison=[&] {for (const auto* b:{&inverse_b,&low_b,&injection_b,&out_b,&inverse_audit,&scaled_b}) b->poison(stream.value);};
    guard.run([&] {
        for (const auto* b:buffers) b->poison(stream.value);
        check(hipStreamSynchronize(stream.value));
        raw_b.upload(raw.data());norm_b.upload(norm.data());down_b.upload(down.data());inject_b.upload(inject.data());up_b.upload(up.data());
    });
    const std::array read_before{raw_b.owner(),norm_b.owner(),down_b.owner(),inject_b.owner(),up_b.owner()};
    const auto spans=[&] {return std::array{raw_b.span(),norm_b.span(),down_b.span(),inject_b.span(),up_b.span(),inverse_b.span(),
        low_b.span(),injection_b.span(),out_b.span(),inverse_audit.span(),scaled_b.span()};};
    const auto launch=[&](const std::array<Span,11>& b,float eps) {
        api::enqueue(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],eps,stream.value);
    };
    const auto den=oracle::rms_denominators(raw,epsilon);
    std::vector<unsigned> first_inverse,first_scaled;
    for (unsigned repeat=0;repeat<3;++repeat) {
        guard.run([&] {poison();launch(spans(),epsilon);});
        const auto inverse=inverse_b.read<float>();const auto audit=inverse_audit.read<unsigned>();
        for (unsigned s=0;s<4;++s) {
            const auto actual=std::bit_cast<unsigned>(inverse[s]);
            expect(std::isfinite(inverse[s]) && inverse[s]>0,"HC-F inverse is not positive finite");
            if (pattern==0) expect(actual==std::bit_cast<unsigned>(1.0F),"HC-F exact RMS denominator1 must give inverse1");
            else {
                // Predeclared diagnostic bound for these well-scaled finite
                // fixtures only; no ISA-wide rsqrt accuracy/qualification claim.
                const float cpu=static_cast<float>(1.0/std::sqrt(static_cast<double>(den[s])));
                const auto reference=std::bit_cast<unsigned>(cpu);
                const auto distance=actual>reference?actual-reference:reference-actual;
                expect(distance<=2U,"HC-F fixture inverse exceeds predeclared two-ULP diagnostic bound");
            }
            for (unsigned block=0;block<(with_inject?81U:80U);++block)
                expect(audit[block*4U+s]==actual,"HC-F block did not reproduce the same inverse bits");
        }
        const auto expected=oracle::scaled(raw,norm,down,inject,inverse);
        const auto actual_scaled=scaled_b.read<unsigned>();
        expect(expected.size()==actual_scaled.size(),"HC-F scaled audit geometry");
        for (unsigned row=0;row<expected.size();++row)
            expect(actual_scaled[row]==std::bit_cast<unsigned>(expected[row]),"HC-F raw-dot tree differs bitwise at fixed observed inverse");
        if (pattern==0) for (auto value:actual_scaled) expect(value==std::bit_cast<unsigned>(1.25F),"HC-F independent constant projection1.25");
        const auto low=low_b.read<float>(),injection=injection_b.read<float>();
        for (float value:low) expect(std::isfinite(value),"HC-F finite scaled dot produced nonfinite SiLU");
        for (float value:injection) expect(std::isfinite(value) && value>=0 && value<=2,"HC-F finite injection range");
        if (pattern==1) {
            for (float value:low) expect(std::bit_cast<unsigned>(value)==0,"HC-F SiLU zero exact epilogue");
            for (float value:injection) expect(std::bit_cast<unsigned>(value)==std::bit_cast<unsigned>(1.0F),"HC-F twice-sigmoid zero exact epilogue");
        }
        expect(out_b.read<std::uint16_t>()==oracle::output_with_zero_up(raw,norm,inverse),
            "HC-F recomputed normalized output differs at exact sigmoid0=.5");
        if (repeat==0) {first_inverse=inverse_b.read<unsigned>();first_scaled=actual_scaled;}
        else expect(inverse_b.read<unsigned>()==first_inverse && actual_scaled==first_scaled,"HC-F repeat changed inverse/scaled bits");
        for (const auto* b:buffers) b->guards();
        expect(raw_b.owner()==read_before[0] && norm_b.owner()==read_before[1] && down_b.owner()==read_before[2] &&
            inject_b.owner()==read_before[3] && up_b.owner()==read_before[4],"HC-F changed read-only owner bytes");
    }
    if (pattern!=2) return;
    const auto all=[&] {
        std::array<std::vector<unsigned char>,11> result;
        for (unsigned i=0;i<buffers.size();++i) result[i]=buffers[i]->owner();
        return result;
    };
    // All late owners, including both optional audits, reject before mutation.
    for (unsigned field=0;field<11;++field) {
        if (!with_inject && (field==3 || field==7)) continue;
        auto b=spans();--b[field].bytes;
        guard.run([&] {poison();});const auto before=all();bool rejected=false;
        try {launch(b,epsilon);} catch (const std::invalid_argument&) {rejected=true;}
        expect(rejected && all()==before,"HC-F undersized owner accepted or mutated data");
    }
    for (unsigned mutation=0;mutation<7;++mutation) {
        auto b=spans();float eps=epsilon;
        if (mutation==0) b[10]=b[2]; // Late audit aliases original down read.
        if (mutation==1) b[9]=b[6];
        if (mutation==2) ++b[2].address;
        if (mutation==3) eps=0;
        if (mutation==4) eps=std::numeric_limits<float>::quiet_NaN();
        if (mutation==5) eps=std::numeric_limits<float>::infinity();
        if (mutation==6) {b[10].address=0;b[10].bytes=4;}
        guard.run([&] {poison();});const auto before=all();bool rejected=false;
        try {launch(b,eps);} catch (const std::invalid_argument&) {rejected=true;}
        expect(rejected && all()==before,"HC-F alias/alignment/epsilon/empty-audit rejection mutated data");
    }
    guard.run([&] {poison();});const auto before_capture=all();hipGraph_t graph=nullptr;
    guard.run([&] {
        check(hipStreamBeginCapture(stream.value,hipStreamCaptureModeGlobal));bool rejected=false;
        try {launch(spans(),epsilon);} catch (const std::invalid_argument&) {rejected=true;}
        check(hipStreamEndCapture(stream.value,&graph));expect(rejected,"HC-F capture admitted");
    });
    std::size_t nodes=1;check(hipGraphGetNodes(graph,nullptr,&nodes));check(hipGraphDestroy(graph));
    expect(nodes==0 && all()==before_capture,"HC-F rejected capture recorded nodes or mutated owners");
    // Optional audit absence is admitted without a dummy owner. Compare actual
    // outputs against the last audited call, not against inferred counters.
    guard.run([&] {poison();launch(spans(),epsilon);});
    const auto reference_low=low_b.read<unsigned>();
    const auto reference_inverse=inverse_b.read<unsigned>();
    const auto reference_injection=injection_b.read<unsigned>();
    const auto reference_out=out_b.read<std::uint16_t>();
    auto no_audit=spans();no_audit[9]={};no_audit[10]={};
    guard.run([&] {poison();launch(no_audit,epsilon);});
    expect(low_b.read<unsigned>()==reference_low && inverse_b.read<unsigned>()==reference_inverse &&
        injection_b.read<unsigned>()==reference_injection && out_b.read<std::uint16_t>()==reference_out,
        "HC-F audit observations altered normal outputs");
    for (const auto* unused:{&inverse_audit,&scaled_b}) {
        const auto value=unused->owner();
        expect(std::all_of(value.begin(),value.end(),[](auto byte){return byte==0xa5;}),
            "HC-F disabled audit wrote an unbound owner");
    }
}
}
int main() try {
    strix::HipSetupQuiescenceGuard guard;expect(guard.architecture().starts_with("gfx1151"),"HC-F diagnostic requires gfx1151");
    strix::moe_q4k_a8::require_f32_environment();
    for (bool injection:{false,true}) for (unsigned pattern:{0U,1U,2U}) fixture(pattern,injection,guard);
    std::cout<<"HC-F inverse replication/scaled-dot tree/output normalization/owner/capture fixtures PASS; no expf bitwise or model-quality claim\n";
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
