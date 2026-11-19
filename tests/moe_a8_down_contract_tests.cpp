#include "strix/moe_a8_down_contract.hpp"
#include "strix/ggml_dequant.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace strix::moe_q4k_a8;
// Latent MoE geometry at the C1 envelope edge (1536 in and out, top-16 of 512).
constexpr Geometry kLatent{1536, 1536, 512, 16};
static_assert(fits_c1(kLatent));
namespace {
void expect(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F&& f) {bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}expect(caught,"expected contract rejection");}
void put16(std::vector<std::byte>& w,std::size_t at,unsigned v) {w[at]=std::byte(v&255U);w[at+1]=std::byte((v>>8U)&255U);}
void tests() {
    require_f32_environment();
    // Exhaust all byte/nibble/qh positions against a separate bit-address oracle.
    for(unsigned position=0;position<32;++position) for(unsigned packed=0;packed<256;++packed) for(unsigned high=0;high<2;++high) {
        std::array<std::byte,24> w{};w[8U+position%16U]=std::byte(packed);
        w[4U+position/8U]=std::byte(high<<(position%8U));
        const auto q=decode_down_q51_codes(w);
        for(unsigned j=0;j<32;++j) {
            unsigned expected=0;
            if(j%16U==position%16U) expected=j<16U?packed%16U:packed/16U;
            if(j==position) expected+=16U*high;
            expect(q[j]==expected,"Q5_1 all nibble/highbit/position decode");
        }
    }
    rejects([]{std::array<std::byte,23> bad{};static_cast<void>(decode_down_q51_codes(bad));});
    std::vector<float> values(128,127.0F);
    auto image=quantize_down_values(values);
    expect(std::all_of(image.codes.begin(),image.codes.end(),[](auto x){return x==127;}),"maximum code");
    for(unsigned g=0;g<4;++g) expect(image.deltas[g]==1 && image.sums[g]==4064,"exact delta/code sum");
    std::vector<std::byte> q51(4U*24U),q80(4U*34U);
    for(unsigned g=0;g<4;++g) {
        put16(q51,g*24U,0x3400);put16(q51,g*24U+2U,0xb800); // d=.25,m=-.5
        std::fill(q51.begin()+g*24U+4U,q51.begin()+(g+1U)*24U,std::byte{255});
        put16(q80,g*34U,0x3c00);std::fill(q80.begin()+g*34U+2U,q80.begin()+(g+1U)*34U,std::byte{128});
    }
    expect(evaluate_down_row(q51,image,7)==117856.0F,"Q5_1 positive offset correction/INT64 independent bound");
    expect(evaluate_down_row(q80,image,8)==-2080768.0F,"Q8_0 -128 original weights signed dot bound");
    auto ties=values;ties[0]=0.5F;ties[1]=1.5F;ties[2]=-0.5F;ties[3]=-1.5F;
    const auto tied=quantize_down_values(ties);
    expect(tied.codes[0]==0 && tied.codes[1]==2 && tied.codes[2]==0 && tied.codes[3]==-2,"signed half ties round to even");
    for(unsigned i=0;i<ties.size();++i) ties[i]=(i&1U)?-127.0F:127.0F;
    const auto symmetric=quantize_down_values(ties);
    expect(symmetric.codes[0]==127 && symmetric.codes[1]==-127 && symmetric.sums[0]==0,"signed quantization endpoints");
    auto mutated=image;mutated.sums[0]--;rejects([&]{static_cast<void>(evaluate_down_row(q51,mutated,7));});
    mutated=image;mutated.codes[0]=-128;rejects([&]{static_cast<void>(evaluate_down_row(q51,mutated,7));});
    std::fill(values.begin(),values.end(),0x1p-120F);image=quantize_down_values(values);
    expect(std::all_of(image.codes.begin(),image.codes.end(),[](auto x){return x==0;}),"tiny policy zeroes codes");
    std::fill(values.begin(),values.end(),0x1p-119F);image=quantize_down_values(values);
    expect(image.codes[0]==127 && image.deltas[0]!=1,"threshold equality is NOT tiny");
    std::fill(values.begin(),values.end(),-0.0F);image=quantize_down_values(values);
    expect(image.deltas[0]==1 && image.sums[0]==0,"signed zero scale=1");
    values[3]=std::numeric_limits<float>::infinity();rejects([&]{static_cast<void>(quantize_down_values(values));});
    values[3]=std::numeric_limits<float>::quiet_NaN();rejects([&]{static_cast<void>(quantize_down_values(values));});
    // Explicit cancellation witness distinguishes declared tree from ascending scalar sum.
    image.codes.assign(128,1);image.sums.assign(4,32);image.deltas.assign(4,1024.0F);
    std::fill(q51.begin(),q51.end(),std::byte{0});
    const std::array<unsigned,4> offsets{0x7800,0x0200,0xf800,0x0200};
    for(unsigned g=0;g<4;++g) put16(q51,g*24U+2U,offsets[g]);
    expect(evaluate_down_row(q51,image,7)==0,"four lane tree cancellation witness");
    volatile float sequential=0;for(float term:{0x1p30F,1.0F,-0x1p30F,1.0F}) sequential=sequential+term;
    expect(sequential==1,"ascending reduction mutant actually distinguished");
    DownRequest request;request.enabled=true;request.approved_numerical_identity=q51_a8_down_numerical_identity;request.shape={64,128,2,2};
    DownBindings b;std::uintptr_t at=0x100000;
    auto next=[&](std::uint64_t bytes){DeviceBytes v{at,bytes};at+=static_cast<std::uintptr_t>((bytes+255U)&~255ULL);return v;};
    b.intermediates=next(2U*2U*128U*4U);b.weights=next(2U*64U*4U*24U);b.routes=next(8);
    b.codes=next(256);b.deltas=next(32);b.sums=next(32);b.rawdots=next(512);b.status=next(4);b.aggregate_status=next(4);
    validate_down(request,b);
    for(unsigned which=0;which<8;++which) {
        auto bad=b;
        std::array<DeviceBytes*,8> v{&bad.intermediates,&bad.weights,&bad.routes,&bad.codes,&bad.deltas,&bad.sums,&bad.rawdots,&bad.status};
        v[which]->bytes--;rejects([&]{validate_down(request,bad);});
    }
    auto bad=b;bad.codes.address=b.intermediates.address;rejects([&]{validate_down(request,bad);});
    bad=b;bad.weights.address=std::numeric_limits<std::uintptr_t>::max()-1U;rejects([&]{validate_down(request,bad);});
    bad=b;bad.aggregate_status={0,4};rejects([&]{validate_down(request,bad);});
    bad=b;bad.aggregate_status={};validate_down(request,bad);
    auto invalid=request;invalid.enabled=false;rejects([&]{validate_down(invalid,b);});
    invalid=request;invalid.weight_type=8;rejects([&]{validate_down(invalid,b);});
    invalid=request;invalid.shape.intermediate=129;rejects([&]{validate_down(invalid,b);});
    invalid=request;invalid.shape.top_k=11;rejects([&]{validate_down(invalid,b);});
    invalid=request;invalid.approved_numerical_identity="unknown";rejects([&]{validate_down(invalid,b);});
    const int saved=std::fegetround();expect(std::fesetround(FE_UPWARD)==0,"round mode setup");
    rejects([&]{static_cast<void>(quantize_down_values(std::vector<float>(128,1)));});
    expect(std::fesetround(saved)==0,"round mode restore");
}
// K-quant down classes (model-B preset: Q4_K / Q6_K down, I = H = 1536, top-16, clamp 10).
std::uint32_t mix(std::uint32_t x) {x^=x>>16U;x*=0x7feb352dU;x^=x>>15U;x*=0x846ca68bU;x^=x>>16U;return x;}
std::vector<std::byte> kquant_row(std::uint32_t type,std::uint32_t width,std::uint32_t seed) {
    std::vector<std::byte> w(static_cast<std::size_t>(down_row_bytes(type,width)));
    for(std::size_t i=0;i<w.size();++i) w[i]=std::byte(mix(seed*2654435761U+static_cast<std::uint32_t>(i))&255U);
    const std::size_t bytes=type==12?144U:210U;
    for(std::size_t b=0;b<w.size()/bytes;++b) {  // finite FP16 scales of both signs, magnitude ~2^-8..2^-6
        const auto at=b*bytes+(type==12?0U:208U);
        put16(w,at,0x1c00U|(mix(seed+static_cast<std::uint32_t>(b))&0x83ffU));
        if(type==12) put16(w,at+2U,0x1800U|(mix(seed*7U+static_cast<std::uint32_t>(b))&0x03ffU));
    }
    return w;
}
void kquant_tests() {
    expect(down_numerical_identity(12)==q4k_a8_down_numerical_identity && down_numerical_identity(14)==q6k_a8_down_numerical_identity,
        "K-quant down identities");
    expect(down_row_bytes(12,1536)==864 && down_row_bytes(14,1536)==1260,"K-quant down row bytes");
    expect(std::uint64_t{1536}*down_row_bytes(14,1536)==1935360 && std::uint64_t{1536}*down_row_bytes(12,1536)==1327104,
        "per-expert bytes match pack-tensors.tsv (990,904,320 / 679,477,248 over 512 experts)");
    // Oracle == exact dot of the shared llama.cpp-bitwise dequantizer and the dequantised A8 codes (only FP32 rounding of
    // the declared order differs): proves the Q6_K element / scale / pair mapping and the reuse of the Q4_K term.
    constexpr std::uint32_t width=kLatent.intermediate;
    for(const std::uint32_t type:{12U,14U}) for(std::uint32_t seed=1;seed<=24;++seed) {
        const auto w=kquant_row(type,width,seed);
        std::vector<float> h(width);
        for(std::uint32_t i=0;i<width;++i) h[i]=(static_cast<float>(mix(seed*31U+i)%2001U)-1000.0F)/97.0F;
        const auto q=quantize_down_values(h);
        std::vector<float> deq(width);strix::ggml_dequant::dequantize_row(type,w,deq);
        double exact=0,magnitude=0;
        for(std::uint32_t i=0;i<width;++i) {
            const double term=static_cast<double>(deq[i])*static_cast<double>(q.codes[i])*static_cast<double>(q.deltas[i/32U]);
            exact+=term;magnitude+=std::fabs(term);
        }
        const double got=evaluate_down_row(w,q,type);
        expect(std::fabs(got-exact)<=1e-5*magnitude+1e-30,"K-quant oracle vs shared dequantizer dot");
        if(type==14 && w[192]!=w[193]) {  // a swapped scale half (s0 <-> s1 of one group) must be visible
            auto mutant=w;std::swap(mutant[192],mutant[193]);
            expect(evaluate_down_row(mutant,q,type)!=got,"Q6_K scale half mutant has power");
        }
    }
    // Geometry presets: model-B admitted for the K-quant classes only; Qwen bounds unchanged for the incumbent classes.
    DownRequest request;request.enabled=true;request.weight_type=14;request.approved_numerical_identity=q6k_a8_down_numerical_identity;
    request.shape={kLatent.hidden,kLatent.intermediate,kLatent.experts,kLatent.top_k};request.swiglu_limit=10.0F;
    const auto s=request.shape;
    DownBindings b;std::uintptr_t at=0x100000;
    auto next=[&](std::uint64_t bytes){DeviceBytes v{at,bytes};at+=static_cast<std::uintptr_t>((bytes+255U)&~255ULL);return v;};
    b.intermediates=next(std::uint64_t{s.top_k}*2U*s.intermediate*4U);b.weights=next(std::uint64_t{s.experts}*s.hidden*1260U);
    b.routes=next(std::uint64_t{s.top_k}*4U);b.codes=next(std::uint64_t{s.top_k}*s.intermediate);
    b.deltas=next(std::uint64_t{s.top_k}*(s.intermediate/32U)*4U);b.sums=next(std::uint64_t{s.top_k}*(s.intermediate/32U)*4U);
    b.rawdots=next(std::uint64_t{s.top_k}*s.hidden*4U);b.status=next(4);b.aggregate_status=next(4);
    validate_down(request,b);
    auto invalid=request;invalid.shape.top_k=17;rejects([&]{validate_down(invalid,b);});
    invalid=request;invalid.shape.intermediate=1664;rejects([&]{validate_down(invalid,b);});
    for(const float limit:{0.0F,-1.0F,std::numeric_limits<float>::quiet_NaN(),-std::numeric_limits<float>::infinity()}) {
        invalid=request;invalid.swiglu_limit=limit;rejects([&]{validate_down(invalid,b);});
    }
    invalid=request;invalid.swiglu_limit=strix::moe::kSwigluNoClamp;validate_down(invalid,b);
    invalid=request;invalid.weight_type=7;invalid.approved_numerical_identity=q51_a8_down_numerical_identity;
    rejects([&]{validate_down(invalid,b);});  // Q5_1 keeps I <= 640, K <= 10
    invalid=request;invalid.weight_type=12;invalid.approved_numerical_identity=q4k_a8_down_numerical_identity;
    rejects([&]{validate_down(invalid,b);});  // Q4_K rows are 864 B, the weight extent above is Q6_K
    auto q4=b;q4.weights.bytes=std::uint64_t{s.experts}*s.hidden*864U;validate_down(invalid,q4);
    q4.weights.address+=2U;rejects([&]{validate_down(invalid,q4);});  // Q4_K needs 16-byte rows
    // Clamp trait = model-B oracle semantics (modelb_host_ref.cpp: g = min(g, L), u = clamp(u, -L, L)); NaN propagates.
    const auto c=strix::moe::swiglu_clamp(12.5F,-11.0F,10.0F);expect(c.gate==10.0F && c.up==-10.0F,"clamp gate max / up both sides");
    const auto d=strix::moe::swiglu_clamp(-50.0F,3.0F,10.0F);expect(d.gate==-50.0F && d.up==3.0F,"gate has no lower clamp");
    const auto e=strix::moe::swiglu_clamp(std::numeric_limits<float>::quiet_NaN(),1.0F,10.0F);expect(std::isnan(e.gate),"NaN propagates");
}
}
int main() try {tests();kquant_tests();std::cout<<"A8 down portable contract PASS\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
