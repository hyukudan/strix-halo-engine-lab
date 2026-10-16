#include "hc_deferred_norm_oracle.hpp"
#include <iostream>
#include <random>

namespace {
namespace oracle=hc_deferred_norm_oracle;
void expect(bool value,const char* text) {if (!value) throw std::runtime_error(text);}
std::uint32_t bits(float x) {return std::bit_cast<std::uint32_t>(x);}
void tests() {
    std::vector<std::uint16_t> raw(oracle::width,0x3f00U); // .5
    std::vector<float> norm(oracle::width,1.0F),injection(4U*oracle::width,0x1p-10F);
    std::vector<std::byte> down(3481600U,std::byte{1});
    for (std::size_t block=0;block<down.size()/34U;++block) {
        down[block*34U]=std::byte{0};down[block*34U+1U]=std::byte{0x14}; // half2^-10
    }
    const auto denom=oracle::rms_denominators(raw,0.75F);
    for (float d:denom) expect(bits(d)==bits(1.0F),"HC-F known RMS denominator is exact1");
    const std::array<float,4> inverse{1,1,1,1};
    const auto projection=oracle::scaled(raw,norm,down,injection,inverse);
    expect(projection.size()==324,"HC-F canonical down plus injection geometry");
    for (float p:projection) expect(bits(p)==bits(1.25F),"HC-F independent constant dot value");
    const auto output=oracle::output_with_zero_up(raw,norm,inverse);
    for (auto v:output) expect(v==0x3e80U,"HC-F zero-up sigmoid and normalized merge exact .25");
    raw[0]=0x8000U;raw[1]=0;
    expect(std::isfinite(oracle::rms_denominators(raw,0.75F)[0]),"HC-F signed zeros in RMS");
    std::mt19937 random(0x484346U);
    bool gain_mutation=false,cross_fma_mutation=false;
    for (unsigned trial=0;trial<8;++trial) {
        for (unsigned i=0;i<oracle::width;++i) {
            raw[i]=static_cast<std::uint16_t>(0x3c00U+(random()%1024U));
            norm[i]=std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),-6);
        }
        for (auto& byte:down) byte=static_cast<std::byte>(random()&255U);
        for (std::size_t block=0;block<down.size()/34U;++block) {
            down[block*34U]=static_cast<std::byte>(random()&255U);down[block*34U+1U]=std::byte{0x24};
        }
        std::array<float,4> inv{0.73F,1.19F,0.92F,1.41F};
        const auto actual=oracle::scaled(raw,norm,down,{},inv);
        const auto bad_gain=oracle::scaled(raw,norm,down,{},inv,true);
        const auto bad_cross=oracle::scaled(raw,norm,down,{},inv,false,true);
        for (unsigned row=0;row<actual.size();++row) {
            gain_mutation|=bits(actual[row])!=bits(bad_gain[row]);
            cross_fma_mutation|=bits(actual[row])!=bits(bad_cross[row]);
        }
    }
    expect(gain_mutation && cross_fma_mutation,"HC-F gain and cross-tile-FMA mutation witnesses");
    bool bad=false;try {(void)oracle::rms_denominators(raw,0);} catch (const std::invalid_argument&) {bad=true;}
    expect(bad,"HC-F rejects nonpositive epsilon");
}
}
int main() try {
    strix::moe_q4k_a8::require_f32_environment();tests();
    std::cout<<"HC-F independent RMS/tree/scaled-dot oracle source fixture PASS; no native rsqrt/expf equality claim\n";
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
