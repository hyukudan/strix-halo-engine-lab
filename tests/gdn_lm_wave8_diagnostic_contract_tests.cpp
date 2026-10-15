#include "strix/gdn_lm_wave8_diagnostic_contract.hpp"
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace strix::gdn_lm_wave8_diagnostic;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Case {
    Shape shape{2560,19};
    Span input{0x10000,5120}, weights{0x100000,80U*34U*19U}, output{0x10000000,76};
    Kind kind=Kind::lm_bf16;
    Arm arm=Arm::wave8;
    Launch run() const { return validate(shape,input,weights,output,kind,arm); }
};
template<class F> void reject(F mutate, const char* message) {
    Case c; mutate(c); bool rejected=false;
    try { (void)c.run(); } catch(const std::invalid_argument& e) {
        rejected=std::string(e.what())==message;
    }
    require(rejected,"invalid contract did not reject with expected reason");
}
constexpr const char* shape_error="wave8 diagnostic shape outside admitted bounds";
constexpr const char* extent_error="wave8 diagnostic view has invalid alignment or extent";
}
int main() try {
    using namespace strix::gdn_lm_wave8_diagnostic;
    unsigned cases=0;
    for(auto width:{32U,96U,256U,288U,640U,2560U,10240U})
        for(auto outputs:{1U,7U,8U,9U,19U,6144U,10240U,248320U})
            for(auto kind:{Kind::gdn_bf16,Kind::lm_bf16})
                for(auto arm:{Arm::reference,Arm::prefetch1,Arm::wave8}) {
                    Case c; c.shape={width,outputs};c.kind=kind;c.arm=arm;
                    c.input.bytes=static_cast<std::uint64_t>(width)*2U;
                    c.weights.bytes=static_cast<std::uint64_t>(width/32U)*34U*outputs;
                    c.output={0x100000000ULL,static_cast<std::uint64_t>(outputs)*4U};
                    const auto launch=c.run();const unsigned waves=arm==Arm::wave8?8U:1U;
                    require(launch.grid_x==(outputs+waves-1U)/waves && launch.block_x==waves*32U,"tail launch geometry mismatch");
                    ++cases;
                }
    for(auto width:{0U,1U,31U,33U,10241U,std::numeric_limits<unsigned>::max()})
        reject([&](Case& c){c.shape.width=width;},shape_error);
    for(auto outputs:{0U,248321U,std::numeric_limits<unsigned>::max()})
        reject([&](Case& c){c.shape.outputs=outputs;},shape_error);
    reject([](Case& c){c.kind=static_cast<Kind>(-1);},"wave8 diagnostic kind is invalid");
    reject([](Case& c){c.arm=static_cast<Arm>(99);},"wave8 diagnostic arm is invalid");
    for(unsigned which=0;which<3;++which) {
        for(unsigned mutation=0;mutation<5;++mutation)
            reject([&](Case& c){
                Span* views[]{&c.input,&c.weights,&c.output};auto& v=*views[which];
                if(mutation==0)v.address=0;
                if(mutation==1)++v.address;
                if(mutation==2)--v.bytes;
                if(mutation==3)++v.bytes;
                if(mutation==4)v.address=std::numeric_limits<std::uintptr_t>::max()&~std::uintptr_t{15};
            },extent_error);
    }
    for(unsigned mode=0;mode<4;++mode)
        reject([&](Case& c){
            c.output.address=mode<2?c.input.address:c.weights.address;
            if(mode==1)c.output.address+=c.input.bytes-16;
            if(mode==3)c.output.address+=c.weights.bytes-16;
        },"wave8 diagnostic output aliases a read-only view");
    // Read/read overlap is legal; exact adjacent writable boundaries are legal.
    Case read_alias;read_alias.weights.address=read_alias.input.address;(void)read_alias.run();
    Case adjacent;adjacent.output.address=adjacent.input.address+adjacent.input.bytes;(void)adjacent.run();
    Case reverse;reverse.output.address=reverse.input.address-80U;(void)reverse.run();
    std::cout<<"wave8 diagnostic host contracts PASS valid_cases="<<cases<<"\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
