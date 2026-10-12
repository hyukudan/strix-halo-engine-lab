#include "integer_projection_runtime_fixture.hpp"
#include <bit>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ip=strix::integer_projection;
namespace {
int failures=0;
void expect(bool value,const char* message){if(!value){++failures;std::cerr<<message<<'\n';}}
template<class F>void rejects(F f,const char* message){try{f();expect(false,message);}catch(const std::exception&) {}}
using integer_runtime_fixture::Fixture;
void target_tiling(){
    Fixture f;const auto plan=f.bind();expect(plan.accounting().checkpoint_bytes==208896000ULL,"exact target IQ4 checkpoint bytes");
    expect(plan.accounting().tile_count==150,"target requires 150 real column tiles");
    expect(plan.accounting().internal_peak_bytes<16ULL*1024*1024,"target staging stays bounded below full weight expansion");
    std::uint64_t columns=0;for(std::uint32_t first=0;first<f.config.vocabulary_size;first+=f.config.tile_columns){const auto tile=plan.tile(first,first+1);
        columns+=tile.request().shape.columns;expect(tile.request().shape.inner==2560,"target K2560 unchanged");}
    expect(columns==153600,"tiles cover full target vocabulary exactly");
    auto snap=plan.validate_dispatch(f.context(),f.owners);expect(snap.retained_bytes==f.arena_pin->value.bytes+f.weight_pin->value.bytes+plan.accounting().internal_peak_bytes,"each external allocation counted once");
    ++f.arena_pin->value.access_generation;rejects([&]{plan.validate_snapshot(f.owners,snap);},"stale arena epoch rejected");
    (void)plan.validate_dispatch(f.context(),f.owners); // Fresh invocation may use new content epoch.
    Fixture tail(2051);expect(tail.bind().tile(2048,3).request().shape.columns==3,"partial last tile covers exactly three columns");
    rejects([&]{(void)plan.tile(1,7);},"noncanonical tiles rejected");
    Fixture affine(19,640,ip::WeightEncoding::q4_0);expect(affine.bind().accounting().checkpoint_bytes==19ULL*20*18,"affine K640 exact storage accepted");
    rejects([]{Fixture invalid(19,640);(void)invalid.bind();},"IQ4 K640 rejected without fabricated storage block");
}
void negative_controls(){
    Fixture f(19);const auto plan=f.bind();
    auto c=f.context();c.op=strix::SerialOp::moe_route;rejects([&]{(void)plan.validate_dispatch(c,f.owners);},"wrong operator");
    c=f.context();c.logical_rows=2;rejects([&]{(void)plan.validate_dispatch(c,f.owners);},"phase/row substitution");
    ++f.weights.tensors[0].allocation_generation;rejects([&]{(void)plan.validate_dispatch(f.context(),f.owners);},"changed weight binding generation");--f.weights.tensors[0].allocation_generation;
    f.weights.model_content_identity.bytes[1]=2;rejects([&]{(void)plan.validate_dispatch(f.context(),f.owners);},"changed actual model content binding");f.weights.model_content_identity.bytes[1]=0;
    ++f.memory.allocation_generation;rejects([&]{(void)plan.validate_dispatch(f.context(),f.owners);},"changed scratch generation");--f.memory.allocation_generation;
    ++f.arena_pin->value.identity;rejects([&]{(void)plan.validate_dispatch(f.context(),f.owners);},"changed owning allocation");--f.arena_pin->value.identity;
    f.weight_pin->throwing=true;rejects([&]{(void)plan.validate_dispatch(f.context(),f.owners);},"throwing owner fails closed");f.weight_pin->throwing=false;
    f.config.maximum_retained_bytes=plan.accounting().internal_peak_bytes;rejects([&]{(void)f.bind();},"full allocation capacity admission");f.config.maximum_retained_bytes=2ULL<<30;
    f.config.experimental_opt_in=false;rejects([&]{(void)f.bind();},"explicit opt-in required");f.config.experimental_opt_in=true;
    f.config.approved_numerical_identity="fp32";rejects([&]{(void)f.bind();},"generic precision cannot authorize lossy activation");f.config.approved_numerical_identity=ip::runtime_numerical_identity(f.config);
    f.config.input=ip::RuntimeInput::bf16;f.config.approved_numerical_identity=ip::runtime_numerical_identity(f.config);
    rejects([&]{(void)f.bind();},"FP32 arena cannot be relabeled BF16 storage");
    f.config.input=ip::RuntimeInput::fp32;f.config.approved_numerical_identity=ip::runtime_numerical_identity(f.config);
    f.manifest.tensors[0].dimensions.push_back(1);rejects([&]{(void)f.bind();},"rank-three original format not accepted");f.manifest.tensors[0].dimensions.pop_back();
    f.weights.tensors[0].storage=strix::RuntimeTensorStorage::repacked_cache;rejects([&]{(void)f.bind();},"generic repacked cache not assumed original bytes");f.weights.tensors[0].storage=strix::RuntimeTensorStorage::resident;
    ++f.weights.tensors[0].address;++f.weight_pin->value.base;
    rejects([&]{(void)f.bind();},"odd half-scale checkpoint address must reject before native dereference");
    --f.weights.tensors[0].address;--f.weight_pin->value.base;
    f.config.instruction=ip::Instruction::iu4;rejects([&]{(void)ip::runtime_numerical_identity(f.config);},"nonlinear nibbles cannot raw IU4");
    f.config.instruction=ip::Instruction::iu8;f.config.input=static_cast<ip::RuntimeInput>(99);
    rejects([&]{(void)ip::runtime_numerical_identity(f.config);},"unknown source storage rejected");
}
void conversion(){
    expect(std::fesetround(FE_TONEAREST)==0,"set independent reference rounding");
    for(unsigned half=0;half<65536;++half){
        const auto exponent=(half>>10U)&31U,fraction=half&1023U;
        const auto widened=ip::binary64_widen_half_bits(static_cast<std::uint16_t>(half));
        if(exponent==31U){
            expect(((widened>>52U)&2047U)==2047U&&((widened&0x000fffffffffffffULL)!=0)==(fraction!=0)&&
                (widened>>63U)==(half>>15U),"exhaustive FP16 NaN/Inf/sign encodings stay rejectable");
        }else{const double sign=(half&0x8000U)?-1.0:1.0;
            const double direct=exponent?sign*std::ldexp(1.0+static_cast<double>(fraction)/1024.0,static_cast<int>(exponent)-15):
                sign*std::ldexp(static_cast<double>(fraction),-24);
            expect(widened==std::bit_cast<std::uint64_t>(direct),"exhaustive FP16 finite/signedzero scale widening against independent mathematical formula");}
    }
    for(std::uint32_t exponent=0;exponent<255;++exponent)for(std::uint32_t fraction:{0U,1U,0x3fffffU,0x7ffffeU,0x7fffffU})
    for(std::uint32_t sign:{0U,0x80000000U}){
        const auto bits=sign|(exponent<<23U)|fraction;
        expect(ip::binary64_widen_bits(bits)==std::bit_cast<std::uint64_t>(static_cast<double>(std::bit_cast<float>(bits))),"independent binary32 widening all exponents and boundary significands");
    }
    for(std::uint32_t fraction=1;fraction<0x00800000U;++fraction){
        expect(ip::binary64_widen_bits(fraction)==std::bit_cast<std::uint64_t>(static_cast<double>(std::bit_cast<float>(fraction))),"exhaustive positive float32 subnormal widening");
        expect(ip::binary64_widen_bits(fraction|0x80000000U)==std::bit_cast<std::uint64_t>(static_cast<double>(std::bit_cast<float>(fraction|0x80000000U))),"exhaustive negative float32 subnormal widening");
    }
    for(std::uint32_t bf16=0;bf16<65536;++bf16)if((bf16&0x7f80U)!=0x7f80U){
        const auto bits=bf16<<16U;expect(ip::binary64_widen_bits(bits)==std::bit_cast<std::uint64_t>(static_cast<double>(std::bit_cast<float>(bits))),"exhaustive finite BF16 raw-bit widening");
    }
    expect(ip::binary64_widen_bits(0x7f800000U)==0x7ff0000000000000ULL&&
        (ip::binary64_widen_bits(0x7fc00001U)&0x7ff0000000000000ULL)==0x7ff0000000000000ULL,"nonfinite source remains rejectable");
    for(auto x:{0.0,-0.0,std::numeric_limits<double>::denorm_min(),-std::numeric_limits<double>::denorm_min(),
        std::ldexp(1.0,-150),std::ldexp(1.0,-149),std::ldexp(3.0,-150),std::ldexp(1.0,-126),1.0+std::ldexp(1.0,-24),
        -1.0-std::ldexp(1.0,-24),static_cast<double>(std::numeric_limits<float>::max()),std::ldexp(1.0,128),
        std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){
        const auto expected=std::bit_cast<std::uint32_t>(static_cast<float>(x));
        expect(ip::binary32_rne_bits(std::bit_cast<std::uint64_t>(x))==expected,"independent RNE conversion edges");}
    for(std::uint32_t exponent=0;exponent<255;++exponent)for(std::uint32_t fraction:{0U,1U,0x3fffffU,0x7ffffeU,0x7fffffU}){
        const auto bits=(exponent<<23U)|fraction;const double x=std::bit_cast<float>(bits);
        for(double d:{x,std::nextafter(x,INFINITY),std::nextafter(x,-INFINITY),-x})
            expect(ip::binary32_rne_bits(std::bit_cast<std::uint64_t>(d))==std::bit_cast<std::uint32_t>(static_cast<float>(d)),"independent float exponent and boundary conversion");}
    expect(std::fesetround(FE_DOWNWARD)==0,"alter caller FENV");
    expect(ip::binary32_rne_bits(std::bit_cast<std::uint64_t>(1.0+std::ldexp(3.0,-24)))==0x3f800002U,"integer conversion ignores altered caller FENV");
    (void)std::fesetround(FE_TONEAREST);
}
void subnormal_projection_observable(){
    for(auto encoding:{ip::WeightEncoding::q4_0,ip::WeightEncoding::iq4_xs})
    for(auto input:{ip::RuntimeInput::fp32,ip::RuntimeInput::bf16})
    for(auto activation:{strix::low_precision::ActivationEncoding::symmetric_a8,strix::low_precision::ActivationEncoding::symmetric_a4}){
        constexpr unsigned columns=35;const unsigned k=encoding==ip::WeightEncoding::q4_0?640U:2560U;
        const unsigned block=encoding==ip::WeightEncoding::q4_0?18U:136U,storage=encoding==ip::WeightEncoding::q4_0?32U:256U;
        std::vector<std::byte> checkpoint(static_cast<std::size_t>(columns)*(k/storage)*block);
        for(std::size_t i=0;i<checkpoint.size();i+=block){const unsigned short scale=i/block<k/storage?0x0001U:((i/block)%3==0)?0xb800U:0x3400U;
            checkpoint[i]=static_cast<std::byte>(scale&255U);checkpoint[i+1]=static_cast<std::byte>(scale>>8U);
            if(encoding==ip::WeightEncoding::iq4_xs){checkpoint[i+2]=std::byte{0xe4};checkpoint[i+3]=std::byte{0x1b};
                for(unsigned j=0;j<4;++j)checkpoint[i+4+j]=static_cast<std::byte>(0x10U+j*0x22U);}
            for(unsigned j=encoding==ip::WeightEncoding::q4_0?2U:8U;j<block;++j)checkpoint[i+j]=static_cast<std::byte>((i/block+j*37U)&255U);
        }
        const auto source_bits=input==ip::RuntimeInput::fp32?1U:0x00010000U;
        std::vector<double> values(k,static_cast<double>(std::bit_cast<float>(source_bits)));
        const auto weights=ip::stage_weights(encoding,columns,k,checkpoint);
        const auto prepared=ip::prepare({1,columns,k},activation,values);
        const auto result=ip::reference(weights,prepared,encoding==ip::WeightEncoding::q4_0&&activation==strix::low_precision::ActivationEncoding::symmetric_a4?ip::Instruction::iu4:ip::Instruction::iu8);
        bool nonzero=false;for(double value:result.output)nonzero=nonzero||(std::bit_cast<std::uint32_t>(static_cast<float>(value))&0x7fffffffU)!=0;
        expect(nonzero,"native all-subnormal fixture has independently observable nonzero projected output");
        for(unsigned i=0;i<k;++i)values[i]=static_cast<double>(static_cast<int>(i%23)-11)/16.0;
        const auto regular_prepared=ip::prepare({1,columns,k},activation,values);
        const auto regular=ip::reference(weights,regular_prepared,encoding==ip::WeightEncoding::q4_0&&activation==strix::low_precision::ActivationEncoding::symmetric_a4?ip::Instruction::iu4:ip::Instruction::iu8);
        expect(static_cast<float>(regular.output[0])!=0.0F,"minimum subnormal FP16 scale column has observable nonzero native reference");
    }
}
}
int main()try{target_tiling();negative_controls();conversion();subnormal_projection_observable();if(failures)return 1;std::cout<<"integer runtime tests passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
