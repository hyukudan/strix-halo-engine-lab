#include "strix/moe_a8_down_contract.hpp"
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
}
int main() try {tests();std::cout<<"A8 down portable contract PASS\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
