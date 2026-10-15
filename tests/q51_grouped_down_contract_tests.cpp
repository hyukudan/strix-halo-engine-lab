#include "strix/q51_grouped_down_contract.hpp"
#include "q51_grouped_down_oracle.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
namespace {
int failures=0;
void check(bool value,const char* message) {if(!value){++failures;std::cerr<<"FAIL: "<<message<<'\n';}}
template<class F> void rejects(F fn,const char* message){try{fn();check(false,message);}catch(const std::invalid_argument&) {}}
void half_at(std::span<std::byte> row,unsigned block,unsigned offset,std::uint16_t bits){row[block*24U+offset]=std::byte(bits&255U);row[block*24U+offset+1U]=std::byte(bits>>8U);}
void code_at(std::span<std::byte> row,unsigned feature,unsigned code){
    const auto base=feature/32U*24U;const auto lane=feature%32U;
    auto bits=std::to_integer<unsigned>(row[base+8U+lane%16U]);
    const auto shift=(lane/16U)*4U;bits=(bits&~(15U<<shift))|((code&15U)<<shift);
    row[base+8U+lane%16U]=std::byte(bits);
    if(code&16U)row[base+4U+lane/8U]|=std::byte(1U<<(lane%8U));
}
}
int main(){
    using namespace strix::q51_grouped_down;
    using q51_down_test::Mutation;
    check(std::fegetround()==FE_TONEAREST && std::numeric_limits<float>::is_iec559,"CPU oracle requires binary32 RNE");
    volatile float tiny=std::numeric_limits<float>::denorm_min(),one=1.0F;
    check(std::bit_cast<unsigned>(static_cast<float>(tiny*one))==1U,"CPU oracle requires gradual subnormals");
    const auto n=required_bytes({2560,640,512,10});
    check(n.intermediates==51200 && n.weights==629145600 && n.accumulation==10240 && n.routes==40,"canonical Q5_1 byte contracts");
    Request r{true,numerical_identity,{67,288,3,3}};
    const auto need=required_bytes(r.shape);
    Bindings b{{0x100000,need.intermediates},{0x200000,need.weights},{0x300000,need.routes},
        {0x400000,need.route_weights},{0x500000,4},{0x600000,need.accumulation}};
    validate(r,b);
    auto bad=r;bad.enabled=false;rejects([&]{validate(bad,b);},"explicit enable");
    bad=r;bad.approved_numerical_identity="other";rejects([&]{validate(bad,b);},"numerical identity");
    for(unsigned field=0;field<4;++field) {
        bad=r;
        if(field==0) bad.shape.hidden=0;
        if(field==1) bad.shape.intermediate=33;
        if(field==2) bad.shape.experts=513;
        if(field==3) bad.shape.top_k=4;
        rejects([&]{validate(bad,b);},"invalid geometry");
    }
    const auto at=[](Bindings& q,unsigned i)->DeviceBytes& {
        switch(i){case 0:return q.intermediates;case 1:return q.weights;case 2:return q.routes;
        case 3:return q.route_weights;case 4:return q.status;default:return q.accumulation;}
    };
    for(unsigned i=0;i<6;++i) {
        auto v=b;--at(v,i).bytes;rejects([&]{validate(r,v);},"one byte short");
        v=b;++at(v,i).bytes;rejects([&]{validate(r,v);},"exact size required");
        v=b;++at(v,i).address;rejects([&]{validate(r,v);},"absolute alignment");
        v=b;at(v,i).address=0;rejects([&]{validate(r,v);},"null view");
        v=b;at(v,i).address=std::numeric_limits<std::uintptr_t>::max()&~std::uintptr_t{15};
        if(at(v,i).bytes>15U)rejects([&]{validate(r,v);},"address overflow");
    }
    for(unsigned write=4;write<6;++write)for(unsigned read=0;read<write;++read){
        auto v=b;at(v,write).address=at(v,read).address;rejects([&]{validate(r,v);},"mutable alias");
    }
    auto aliases=b;aliases.routes.address=aliases.route_weights.address;validate(r,aliases);
    std::mt19937 random(0x5132256U);
    constexpr std::array<Mutation,6> mutants{Mutation::tree32,Mutation::descending,Mutation::unfused_leaf,Mutation::ignore_high,Mutation::ignore_minimum,Mutation::none};
    std::array<bool,5> detected{};
    unsigned compared=0;
    for(const unsigned width:{32U,64U,256U,288U,512U,640U}) for(unsigned trial=0;trial<80;++trial){
        std::vector<float> input(width);
        std::vector<std::byte> row(static_cast<std::size_t>(width/32U)*24U);
        for(unsigned block=0;block<width/32U;++block){
            const auto d=static_cast<std::uint16_t>((random()&0x8000U)|((random()%26U+1U)<<10U)|(random()&1023U));
            const auto m=static_cast<std::uint16_t>((random()&0x8000U)|((random()%26U+1U)<<10U)|(random()&1023U));
            half_at(row,block,0,d);half_at(row,block,2,m);
        }
        for(unsigned f=0;f<width;++f){
            code_at(row,f,random()%32U);
            const float sign=(random()&1U)?1.0F:-1.0F;
            input[f]=sign*std::ldexp(1.0F+static_cast<float>(random()%1024U)/1024.0F,static_cast<int>(random()%41U)-20);
        }
        const auto expected=std::bit_cast<unsigned>(q51_down_test::legacy(input,row));
        check(std::isfinite(std::bit_cast<float>(expected)),"random oracle scope remains finite");
        check(expected==std::bit_cast<unsigned>(q51_down_test::virtual256(input,row)),"virtual256 reproduces original tree bits");
        ++compared;
        for(unsigned i=0;i<detected.size();++i)detected[i]=detected[i]||expected!=std::bit_cast<unsigned>(q51_down_test::legacy(input,row,mutants[i]));
    }
    for(const bool v:detected)check(v,"host executed numerical/packed mutant is detected");
    unsigned coefficient_pairs=0;
    constexpr std::array<std::uint16_t,8> offsets{0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0xfbff};
    for(unsigned header=0;header<65536;++header) {
        if((header&0x7c00U)==0x7c00U) continue;
        const auto d=q51_down_test::half(static_cast<std::uint16_t>(header));
        for(unsigned code=0;code<32;++code) {
            const auto m=q51_down_test::half(offsets[code%offsets.size()]);
            check(std::bit_cast<unsigned>(std::fma(d,static_cast<float>(code),m))==
                std::bit_cast<unsigned>(q51_down_test::add(q51_down_test::mul(d,static_cast<float>(code)),m)),
                "specific finite Q5_1 exact product permits one affine rounding");
            ++coefficient_pairs;
        }
    }
    // Actual FP32 activation subnormals, signed zeros and extreme finite Q5_1 codes.
    for(const unsigned width:{32U,288U,640U}) {
        std::vector<float> input(width);
        std::vector<std::byte> row(static_cast<std::size_t>(width/32U)*24U);
        for(unsigned block=0;block<width/32U;++block) {
            half_at(row,block,0,block%2 ? 0xfbffU : 0x0001U);
            half_at(row,block,2,block%2 ? 0x8000U : 0x8001U);
        }
        for(unsigned i=0;i<width;++i) {
            input[i]=std::bit_cast<float>((i%2 ? 0x80000000U : 0U)|(i%4==0 ? 0U : i%128U));
            code_at(row,i,i%2 ? 31U : 0U);
        }
        check(std::bit_cast<unsigned>(q51_down_test::legacy(input,row))==
              std::bit_cast<unsigned>(q51_down_test::virtual256(input,row)),
              "subnormal signed-zero extreme-code CPU tree equality");
        ++compared;
    }
    const std::array dots{33554432.0F,-33554432.0F,1.0F},weights{1.0F,1.0F,1.0F};
    check(q51_down_test::combine(dots,weights)==1.0F && q51_down_test::combine(dots,weights,true)==0.0F,"ascending slot cancellation mutant");
    const std::array negative{-1.0F},zero{0.0F};
    check(std::bit_cast<unsigned>(q51_down_test::combine(negative,zero))==0U &&
          std::bit_cast<unsigned>(q51_down_test::combine(negative,zero,false,true))==0x80000000U,"positive zero seed versus first-product mutant");
    std::cout<<"Q5_1 host oracle rows="<<compared<<" finite affine pairs="<<coefficient_pairs<<" mutants=7 detected; GPU prototype not qualified\n";
    return failures?1:0;
}
