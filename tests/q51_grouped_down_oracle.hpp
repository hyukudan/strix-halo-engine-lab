#pragma once
// Independent packed-bit CPU oracle. No production loader or HIP helper calls.
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
namespace q51_down_test {
inline float half(std::uint16_t h) {
    const auto sign=static_cast<std::uint32_t>(h&0x8000U)<<16U;
    const auto exponent=(h>>10U)&31U;
    std::uint32_t fraction=h&1023U;
    if(exponent==31U) return std::bit_cast<float>(sign|0x7f800000U|(fraction<<13U));
    if(exponent==0U) {
        if(fraction==0U) return std::bit_cast<float>(sign);
        unsigned shift=0;
        while((fraction&1024U)==0U) {fraction<<=1U;++shift;}
        return std::bit_cast<float>(sign|((113U-shift)<<23U)|((fraction&1023U)<<13U));
    }
    return std::bit_cast<float>(sign|((exponent+112U)<<23U)|(fraction<<13U));
}
inline float add(float a,float b) {volatile float value=a+b;return value;}
inline float mul(float a,float b) {volatile float value=a*b;return value;}
enum class Mutation {none,tree32,descending,unfused_leaf,ignore_high,ignore_minimum};
inline float weight(std::span<const std::byte> row,unsigned feature,Mutation m=Mutation::none) {
    const auto block=row.subspan(static_cast<std::size_t>(feature/32U)*24U,24);
    const auto byte=[&](unsigned i){return static_cast<unsigned>(std::to_integer<unsigned char>(block[i]));};
    const auto lane=feature%32U;
    const unsigned packed=byte(8U+lane%16U);
    unsigned code=lane<16U ? packed&15U : packed>>4U;
    if(m!=Mutation::ignore_high) code|=((byte(4U+lane/8U)>>(lane%8U))&1U)<<4U;
    const auto d=half(static_cast<std::uint16_t>(byte(0)|(byte(1)<<8U)));
    const auto minimum=m==Mutation::ignore_minimum ? 0.0F : half(static_cast<std::uint16_t>(byte(2)|(byte(3)<<8U)));
    return std::fma(d,static_cast<float>(code),minimum);
}
inline float legacy(std::span<const float> input,std::span<const std::byte> row,Mutation m=Mutation::none) {
    const unsigned width=static_cast<unsigned>(input.size());
    if(width==0 || width%32U || row.size()!=input.size()/32U*24U) throw std::invalid_argument("Q5_1 oracle row shape");
    const unsigned leaves=m==Mutation::tree32 ? 32U : 256U;
    std::array<float,256> sums{};
    for(unsigned leaf=0;leaf<leaves;++leaf) {
        if(m==Mutation::descending && leaf<width) {
            unsigned feature=leaf+(width-1U-leaf)/leaves*leaves;
            for(;;) {
                sums[leaf]=std::fma(input[feature],weight(row,feature,m),sums[leaf]);
                if(feature<leaves) break;
                feature-=leaves;
            }
        } else for(unsigned feature=leaf;feature<width;feature+=leaves) {
            const auto w=weight(row,feature,m);
            sums[leaf]=m==Mutation::unfused_leaf ? add(sums[leaf],mul(input[feature],w)) : std::fma(input[feature],w,sums[leaf]);
        }
    }
    for(unsigned stride=leaves/2U;stride>0;stride>>=1U)
        for(unsigned leaf=0;leaf<stride;++leaf) sums[leaf]=add(sums[leaf],sums[leaf+stride]);
    return sums[0];
}
inline float virtual256(std::span<const float> input,std::span<const std::byte> row) {
    std::array<std::array<float,8>,32> registers{};
    for(unsigned lane=0;lane<32;++lane) for(unsigned j=0;j<8;++j)
        for(unsigned feature=8U*lane+j;feature<input.size();feature+=256U)
            registers[lane][j]=std::fma(input[feature],weight(row,feature),registers[lane][j]);
    for(unsigned distance=16;distance>0;distance>>=1U)
        for(unsigned lane=0;lane<distance;++lane) for(unsigned j=0;j<8;++j)
            registers[lane][j]=add(registers[lane][j],registers[lane+distance][j]);
    for(unsigned stride=4;stride>0;stride>>=1U)
        for(unsigned j=0;j<stride;++j) registers[0][j]=add(registers[0][j],registers[0][j+stride]);
    return registers[0][0];
}
inline float combine(std::span<const float> dots,std::span<const float> weights,bool reverse=false,bool first_product=false) {
    float current=0.0F;
    for(unsigned slot=0;slot<dots.size();++slot) {
        const unsigned s=reverse ? static_cast<unsigned>(dots.size())-1U-slot : slot;
        current=first_product && slot==0 ? mul(dots[s],weights[s]) : std::fma(dots[s],weights[s],current);
    }
    return current;
}
} // namespace q51_down_test
