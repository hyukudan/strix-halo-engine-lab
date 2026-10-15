#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "q4k_a8_oracle.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
extern "C" __global__ void strix_q4k_a8_native_dot4_compile_probe(const int*,const unsigned*,int*);
using namespace strix::moe_q4k_a8;
namespace {
void expect(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void check(hipError_t error) {if(error!=hipSuccess) throw std::runtime_error(hipGetErrorString(error));}
struct Buffer {
    void* owner=nullptr;std::size_t bytes;const strix::HipSetupQuiescenceGuard& quiescence;
    Buffer(std::size_t n,const strix::HipSetupQuiescenceGuard& guard):bytes(n),quiescence(guard) {check(hipMalloc(&owner,n+32U));}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(owner) (void)hipFree(owner);}
    void* data() const {return static_cast<unsigned char*>(owner)+16U;}
    DeviceBytes view() const {return {reinterpret_cast<std::uintptr_t>(data()),bytes};}
    void upload(const void* src) {check(hipMemcpy(data(),src,bytes,hipMemcpyHostToDevice));}
    template<class T> std::vector<T> read() const {
        std::vector<T> out(bytes/sizeof(T));
        quiescence.run([&]{check(hipMemcpy(out.data(),data(),bytes,hipMemcpyDeviceToHost));});return out;
    }
    void poison(hipStream_t stream=nullptr) {check(hipMemsetAsync(owner,0xa5,bytes+32U,stream));}
    void guards(unsigned char pattern=0xa5) const {
        std::array<unsigned char,16> a{},b{};
        quiescence.run([&]{check(hipMemcpy(a.data(),owner,16,hipMemcpyDeviceToHost));
            check(hipMemcpy(b.data(),static_cast<unsigned char*>(data())+bytes,16,hipMemcpyDeviceToHost));});
        for(unsigned i=0;i<16;++i) expect(a[i]==pattern && b[i]==pattern,"output guard overwritten");
    }
    void unchanged() const {
        const auto value=read<unsigned char>();for(auto x:value) expect(x==0xa5,"failed call published data");guards();
    }
};
void dot_receipt(const strix::HipSetupQuiescenceGuard& guard) {
    std::vector<int> a(256);std::vector<unsigned> b(256);std::vector<int> expected(256);
    bool unsigned_mutant_detected=false;
    for(unsigned i=0;i<256;++i) {
        unsigned packed=0;int correct=0,wrong=0;
        for(unsigned lane=0;lane<4;++lane) {
            const int code=static_cast<int>((i+lane*61U)%255U)-127;
            const unsigned weight=(i+lane*7U)&15U;
            packed|=static_cast<unsigned>(static_cast<unsigned char>(code))<<(8U*lane);
            b[i]|=weight<<(8U*lane);correct+=code*static_cast<int>(weight);
            wrong+=static_cast<unsigned char>(code)*static_cast<int>(weight);
        }
        a[i]=std::bit_cast<int>(packed);expected[i]=correct;unsigned_mutant_detected|=wrong!=correct;
    }
    expect(unsigned_mutant_detected,"signed activation dot control ineffective");
    Buffer da(1024,guard),db(1024,guard),out(1024,guard);
    guard.run([&]{da.upload(a.data());db.upload(b.data());out.poison();
        strix_q4k_a8_native_dot4_compile_probe<<<1,256>>>(static_cast<int*>(da.data()),static_cast<unsigned*>(db.data()),static_cast<int*>(out.data()));check(hipGetLastError());});
    expect(out.read<int>()==expected,"native signed-A unsigned-B dot4 INT64-bounded oracle");out.guards();
}
void conversion_receipts(const strix::HipSetupQuiescenceGuard& guard) {
    Buffer output(65536U*4U,guard);
    guard.run([&]{output.poison();enqueue_half_conversion_probe(static_cast<unsigned*>(output.data()));});
    auto actual=output.read<unsigned>();unsigned nan_count=0;
    for(unsigned bits=0;bits<65536U;++bits) {
        const auto expected=q4k_a8_oracle::half(static_cast<std::uint16_t>(bits));
        if(std::isnan(expected)) {++nan_count;expect(std::isnan(std::bit_cast<float>(actual[bits])),"FP16 NaN classification");}
        else expect(actual[bits]==std::bit_cast<unsigned>(expected),"FP16 all finite/Inf/zero decode bits");
    }
    output.guards();std::cout<<"half65536 finite/Inf/zero exact; NaN classification only "<<nan_count<<'\n';
    std::vector<unsigned> a,b;std::mt19937 random(0x12345U);
    for(unsigned i=0;i<16384U;++i) {
        unsigned x=static_cast<unsigned>(random()),y=static_cast<unsigned>(random());
        if((x&0x7fffffffU)>=0x7f800000U || (y&0x7fffffffU)>=0x7f800000U || (y&0x7fffffffU)==0) {--i;continue;}
        a.push_back(x);b.push_back(y);
    }
    for(unsigned tiny:{1U,63U,64U,190U,0x800000U,0x7f7fffffU}) {a.push_back(tiny);b.push_back(0x42fe0000U);}
    a.push_back(0x3a8be27eU);b.push_back(0x42fe0000U);
    a.push_back(0x37b03b95U);b.push_back(0x370cfc77U);
    a.push_back(0xb7b03b95U);b.push_back(0x370cfc77U);
    bool reciprocal_mutant_detected=false;
    for(unsigned i=0;i<a.size();++i) {
        const float x=std::bit_cast<float>(a[i]),y=std::bit_cast<float>(b[i]);
        volatile float reciprocal=1.0F/y;volatile float product=x*reciprocal;
        const auto exact=q4k_a8_oracle::divide(a[i],b[i]);
        if(std::isfinite(reciprocal) && std::isfinite(product) && (exact&0x7fffffffU)<0x7f800000U)
            reciprocal_mutant_detected|=std::bit_cast<unsigned>(static_cast<float>(product))!=exact;
    }
    expect(reciprocal_mutant_detected,"reciprocal-multiply mutant must differ on finite receipt vectors");
    Buffer da(a.size()*4U,guard),db(b.size()*4U,guard),result(a.size()*4U,guard);
    guard.run([&]{da.upload(a.data());db.upload(b.data());result.poison();enqueue_division_probe(
        static_cast<unsigned*>(da.data()),static_cast<unsigned*>(db.data()),static_cast<unsigned*>(result.data()),static_cast<unsigned>(a.size()));});
    const auto divided=result.read<unsigned>();
    for(unsigned i=0;i<a.size();++i) expect(divided[i]==q4k_a8_oracle::divide(a[i],b[i]),"OCML RN division independent integer oracle");
    result.guards();
}
void projection_case(unsigned hidden,unsigned rows,unsigned top_k,bool bf16,unsigned special,const strix::HipSetupQuiescenceGuard& guard) {
    Request req;req.enabled=true;req.approved_numerical_identity=numerical_identity;
    req.input=bf16?InputKind::bf16:InputKind::fp32;req.shape={hidden,rows,2,top_k,1};
    const std::size_t wb=2ULL*rows*(hidden/256U)*144U;
    Buffer input(hidden*(bf16?2U:4U),guard),gate(wb,guard),up(wb,guard),routes(top_k*4U,guard),codes(hidden,guard),deltas(hidden/8U,guard),sums(hidden/8U,guard),status(4,guard),out(top_k*2U*rows*4U,guard);
    Bindings bind{input.view(),gate.view(),up.view(),routes.view(),codes.view(),deltas.view(),sums.view(),status.view(),out.view()};
    std::mt19937 random(0x314159U+hidden+rows+top_k+special);
    std::vector<float> x(hidden);std::vector<unsigned short> bx(hidden);std::vector<unsigned> rt(top_k);
    for(unsigned i=0;i<hidden;++i) {
        x[i]=std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),-6);
        if(special==8) x[i]=0.0F;
        if(special==1) x[i]=(i&1U)?-0.0F:0.0F;
        if(special==2) x[i]=std::bit_cast<float>((i&1U)?64U:190U);
        if(special==3) x[i]=std::bit_cast<float>(0x00010000U); // minimum BF16 subnormal
        if(special==4) x[i]=std::numeric_limits<float>::max();
        if(special==7) {
            const unsigned maximum=64U+static_cast<unsigned>((i/32U*373U+31U)%221U)*0x800000U;
            const float amax=std::bit_cast<float>(maximum);
            const float delta=std::bit_cast<float>(q4k_a8_oracle::divide(maximum,0x42fe0000U));
            x[i]=i%32U==0?amax:std::min(amax,delta*(static_cast<float>(i%32U)+0.5F));
            if(i&2U) x[i]=-x[i];
        }
        if(special==5) x[i]=i==0?std::bit_cast<float>(0x3a8be27eU):i==1?std::bit_cast<float>(0x37b03b95U):0.0F;
        bx[i]=static_cast<unsigned short>(std::bit_cast<unsigned>(x[i])>>16U);
        if(bf16) x[i]=std::bit_cast<float>(static_cast<unsigned>(bx[i])<<16U);
    }
    const auto quant=q4k_a8_oracle::quantize(x);
    std::vector<std::byte> gw(wb),uw(wb);
    for(auto* w:{&gw,&uw}) {
        for(auto& value:*w) value=static_cast<std::byte>(random()&255U);
        for(std::size_t block=0;block<wb/144U;++block) {
            const std::array<unsigned,10> headers{0,0x8000,1,0x8001,0x03ff,0x83ff,0x3c00,0xbc00,0x7bff,0xfbff};
            const unsigned d=special==6?0x7e01U:special==8?0xbc00U:headers[random()%headers.size()];
            const unsigned dm=special==8?0U:headers[random()%headers.size()];
            (*w)[block*144U]=static_cast<std::byte>(d&255U);(*w)[block*144U+1U]=static_cast<std::byte>(d>>8U);
            (*w)[block*144U+2U]=static_cast<std::byte>(dm&255U);(*w)[block*144U+3U]=static_cast<std::byte>(dm>>8U);
        }
    }
    for(unsigned i=0;i<top_k;++i) rt[i]=i&1U;
    const auto poison=[&]{status.poison();for(auto* b:{&codes,&deltas,&sums,&out}) b->poison();check(hipMemsetAsync(status.data(),0,4));};
    guard.run([&]{check(hipMemsetAsync(input.owner,0xff,input.bytes+32U));check(hipMemsetAsync(gate.owner,0xff,gate.bytes+32U));check(hipMemsetAsync(up.owner,0xff,up.bytes+32U));check(hipMemsetAsync(routes.owner,0xff,routes.bytes+32U));input.upload(bf16?static_cast<const void*>(bx.data()):static_cast<const void*>(x.data()));gate.upload(gw.data());up.upload(uw.data());routes.upload(rt.data());poison();enqueue_probe(req,bind);});
    const auto guards=[&]{status.guards();input.guards(0xff);gate.guards(0xff);up.guards(0xff);routes.guards(0xff);};
    const auto verify=[&] {
        expect(status.read<unsigned>()[0]==0,"valid probe status");
        expect(codes.read<signed char>()==quant.codes,"quantized code mismatch");
        const auto actual_delta=deltas.read<unsigned>();
        for(unsigned i=0;i<quant.deltas.size();++i) expect(actual_delta[i]==std::bit_cast<unsigned>(quant.deltas[i]),"quantized delta bits mismatch");
        expect(sums.read<int>()==quant.sums,"quantized sum mismatch");
        const auto values=out.read<unsigned>();unsigned qualified=0,unqualified=0;
        for(unsigned slot=0;slot<top_k;++slot) for(unsigned which=0;which<2;++which) for(unsigned row=0;row<rows;++row) {
            const auto& weights=which?uw:gw;
            const auto packed=std::span<const std::byte>(weights).subspan((static_cast<std::size_t>(rt[slot])*rows+row)*(hidden/256U)*144U,(hidden/256U)*144U);
            try {
                // Host production predicate rejects EVERY nonfinite intermediate before oracle equality.
                const float finite=evaluate_row(packed,quant);const float reference=q4k_a8_oracle::row(packed,quant);
                expect(std::bit_cast<unsigned>(finite)==std::bit_cast<unsigned>(reference),"host independent combine disagreement");
                expect(values[(slot*2U+which)*rows+row]==std::bit_cast<unsigned>(reference),"GPU independent affine/combine/tree bits mismatch");++qualified;
            } catch(const std::invalid_argument&) {++unqualified;}
        }
        const unsigned expected=top_k*2U*rows;
        expect(qualified+unqualified==expected,"finite-scope accounting incomplete");
        if(special==6) expect(qualified==0 && unqualified==expected,"NaN-header scope must be wholly unqualified");
        else if(special==4) expect(qualified>0 && unqualified>0,"FLT_MAX activation overflow control must be exercised");
        else expect(unqualified==0 && qualified==expected,"finite oracle unexpectedly skipped GPU equality checks");
        guards();
        if(unqualified) {std::cout<<"unqualified_output_bits";for(auto bits:values) std::cout<<' '<<bits;std::cout<<'\n';}
        for(auto* b:{&codes,&deltas,&sums,&out}) b->guards();
        std::cout<<"H="<<hidden<<" I="<<rows<<" K="<<top_k<<" bf16="<<bf16<<" special="<<special<<" qualified="<<qualified<<" unqualified="<<unqualified<<'\n';
    };
    verify();
    hipStream_t stream{};hipGraph_t graph{};hipGraphExec_t executable{};
    check(hipStreamCreate(&stream));
    try {
        guard.run([&]{check(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));enqueue_probe(req,bind,stream);check(hipStreamEndCapture(stream,&graph));});
        check(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(unsigned replay=0;replay<3;++replay) {guard.run([&]{
            for(auto* b:{&codes,&deltas,&sums,&out}) b->poison(stream);
            check(hipMemsetAsync(status.data(),0,4,stream));check(hipGraphLaunch(executable,stream));});verify();}
        guard.run([&]{for(auto* b:{&codes,&deltas,&sums,&out}) b->poison(stream);
            check(hipMemsetAsync(status.data(),0x40,4,stream));check(hipGraphLaunch(executable,stream));});
        for(auto* b:{&codes,&deltas,&sums,&out}) b->unchanged();
        expect(status.read<unsigned>()[0]==0x40404040U,"captured sticky status was evaluated at capture time");guards();
        const unsigned invalid=2;
        guard.run([&]{check(hipMemcpyAsync(routes.data(),&invalid,4,hipMemcpyHostToDevice,stream));
            for(auto* b:{&codes,&deltas,&sums,&out}) b->poison(stream);
            check(hipMemsetAsync(status.data(),0,4,stream));check(hipGraphLaunch(executable,stream));});
        expect(status.read<unsigned>()[0]==invalid_route,"captured route was evaluated at capture time");
        for(auto* b:{&codes,&deltas,&sums,&out}) b->unchanged();guards();
        guard.run([&]{routes.upload(rt.data());});
        const auto old_exec=executable;executable=nullptr;check(hipGraphExecDestroy(old_exec));
        const auto old_graph=graph;graph=nullptr;check(hipGraphDestroy(old_graph));
        const auto old_stream=stream;stream=nullptr;check(hipStreamDestroy(old_stream));
    } catch(...) {guard.recover_or_fail_stop();if(executable) (void)hipGraphExecDestroy(executable);if(graph) (void)hipGraphDestroy(graph);if(stream) (void)hipStreamDestroy(stream);throw;}
    // Caller sticky status is never reset by the API; writes remain poison.
    guard.run([&]{poison();check(hipMemsetAsync(status.data(),0x40,4));enqueue_probe(req,bind);});
    for(auto* b:{&codes,&deltas,&sums,&out}) b->unchanged();expect(status.read<unsigned>()[0]==0x40404040U,"sticky status reset");guards();
    // Full route validation precedes any activation/output publication.
    rt.back()=2;guard.run([&]{routes.upload(rt.data());poison();enqueue_probe(req,bind);});
    expect(status.read<unsigned>()[0]==invalid_route,"invalid route status");for(auto* b:{&codes,&deltas,&sums,&out}) b->unchanged();guards();
    rt.back()=0;guard.run([&]{routes.upload(rt.data());});
    // Invalid activation in LAST group must leave every external workspace/output unchanged.
    for(unsigned error=0;error<4;++error) {
        if(error==0) {x.back()=std::numeric_limits<float>::quiet_NaN();bx.back()=0x7fc1U;}
        else if(error==1) {for(auto& value:x)value=std::bit_cast<float>(1U);if(bf16) continue;}
        else {for(auto& value:x) value=1.0F;for(auto& value:bx) value=0x3f80U;x.back()=error==2?std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity();bx.back()=error==2?0x7f80U:0xff80U;}
        guard.run([&]{input.upload(bf16?static_cast<const void*>(bx.data()):static_cast<const void*>(x.data()));poison();enqueue_probe(req,bind);});
        expect(status.read<unsigned>()[0]==(error==1?invalid_delta:nonfinite_activation),"invalid activation status");
        for(auto* b:{&codes,&deltas,&sums,&out}) b->unchanged();guards();
    }
    // Pure host rejection must occur before any launch/data mutation.
    for(unsigned bad=0;bad<3;++bad) {
        guard.run([&]{poison();});auto q=req;auto b=bind;
        if(bad==0) q.enabled=false;if(bad==1) q.gate_type=13;if(bad==2) b.output.bytes-=4;
        bool rejected=false;try{enqueue_probe(q,b);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"host rejection missing");for(auto* buffer:{&codes,&deltas,&sums,&out}) buffer->unchanged();expect(status.read<unsigned>()[0]==0,"host rejection changed status");
    }
}
}
int main() try {
    strix::HipSetupQuiescenceGuard guard;
    expect(guard.architecture().starts_with("gfx1151"),"real gfx1151 required");require_f32_environment();
    dot_receipt(guard);conversion_receipts(guard);
    for(bool bf16:{false,true}) for(unsigned hidden:{256U,512U,2560U}) projection_case(hidden,32,2,bf16,0,guard);
    projection_case(2560,640,10,true,0,guard);
    projection_case(256,32,1,false,1,guard);projection_case(256,32,1,false,2,guard);
    projection_case(256,32,1,false,8,guard);
    projection_case(2560,32,1,false,7,guard);
    projection_case(256,32,1,false,5,guard);projection_case(256,32,1,false,6,guard);
    projection_case(256,32,1,true,3,guard);projection_case(256,32,1,false,4,guard);
    std::cout<<"Standalone Q4K A8G32 GPU probe PASS; intentionally quantized, no legacy equality/full-model/performance claim\n";
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
