#include "strix/moe_q4k_a8_contract.hpp"
#include "q5k_a8_oracle.hpp"
#include <iostream>
#include <random>
#include <string>

namespace {
using namespace strix::moe_q4k_a8;
void expect(bool ok,const char* message) {if (!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {
    bool rejected = false;
    try {f();} catch (const std::invalid_argument&) {rejected = true;}
    expect(rejected,"invalid Q5 contract accepted");
}
std::uint32_t bits(float value) {return std::bit_cast<std::uint32_t>(value);}
void tests() {
    require_f32_environment();
    expect(numerical_identity_for_gate_type(12) == numerical_identity &&
        numerical_identity_for_gate_type(13) == q5k_numerical_identity &&
        q5k_numerical_identity != numerical_identity,"per-type numerical identity closed");
    rejects([] {static_cast<void>(numerical_identity_for_gate_type(14));});
    // Exhaustive byte/plane/lane census: untouched bits deliberately all ones.
    // Tests all8 groups,32 lanes,256 low-byte values and both high-plane bits.
    std::array<std::byte,176> packed{};
    for (unsigned group = 0; group < 8; ++group) for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned low = 0; low < 256; ++low) for (unsigned high = 0; high < 2; ++high) {
            packed.fill(std::byte{0xff});
            packed[48U+(group/2U)*32U+lane] = static_cast<std::byte>(low);
            packed[16U+lane] = static_cast<std::byte>((255U&~(1U<<group))|(high<<group));
            const auto codes = decode_q5k_codes(packed,group);
            const auto expected = ((low>>(4U*(group&1U)))&15U)|(high<<4U);
            expect(codes[lane] == expected && codes[lane] == q5k_a8_oracle::code(packed,group*32U+lane),
                "Q5 original-byte exhaustive unsigned code assembly");
            for (unsigned other = 0; other < 32; ++other)
                if (other != lane) expect(codes[other] == 31,"Q5 lane extraction crosses byte boundary");
        }
    rejects([&] {static_cast<void>(decode_q5k_codes(std::span<const std::byte>(packed).first(175),0));});
    rejects([&] {static_cast<void>(decode_q5k_codes(packed,8));});
    std::array<std::uint8_t,32> q{};q.fill(31);
    std::array<std::int8_t,32> a{};
    for (int sign : {-1,1}) {
        a.fill(static_cast<std::int8_t>(sign*127));
        const auto actual = integer_group_q5k(q,a,{63,63});
        expect(actual.raw == sign*125984 && actual.sum == sign*4064 &&
            actual.scaled == sign*7936992 && actual.minimum == sign*256032,
            "Q5 maximum exact integer bound and affine correction");
        expect(static_cast<std::int32_t>(static_cast<float>(actual.scaled)) == actual.scaled,
            "Q5 scaled integer remains exactly representable in binary32");
        rejects([&] {static_cast<void>(integer_group(q,a,{63,63}));}); // Q4 remains closed.
    }
    q[0]=32;rejects([&] {static_cast<void>(integer_group_q5k(q,a,{1,1}));});
    q[0]=31;a[0]=-128;rejects([&] {static_cast<void>(integer_group_q5k(q,a,{1,1}));});a[0]=0;
    rejects([&] {static_cast<void>(integer_group_q5k(q,a,{64,1}));});
    std::mt19937 random(0x513250U);
    std::array<bool,6> detected{};
    for (unsigned width : {256U,512U,2560U}) for (unsigned trial = 0; trial < 256; ++trial) {
        std::vector<float> x(width);
        for (auto& value : x) value = std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),
            static_cast<int>(random()%15U)-8);
        const auto image = q4k_a8_oracle::quantize(x);
        std::vector<std::byte> weights(width/256U*176U);
        for (auto& value : weights) value = static_cast<std::byte>(random()&255U);
        for (std::size_t block = 0; block < weights.size()/176U; ++block) {
            const std::array<unsigned,10> headers{0,0x8000,1,0x8001,0x03ff,0x83ff,0x3c00,0xbc00,0x7bff,0xfbff};
            const auto d=headers[random()%headers.size()],dm=headers[random()%headers.size()];
            weights[block*176U]=static_cast<std::byte>(d&255U);weights[block*176U+1U]=static_cast<std::byte>(d>>8U);
            weights[block*176U+2U]=static_cast<std::byte>(dm&255U);weights[block*176U+3U]=static_cast<std::byte>(dm>>8U);
        }
        const auto expected = bits(q5k_a8_oracle::row(weights,image));
        expect(bits(evaluate_q5k_row(weights,image)) == expected,"Q5 independent full-row FP32 epilogue and pair4c0 tree");
        for (unsigned m = 0; m < detected.size(); ++m)
            detected[m] = detected[m] || bits(q5k_a8_oracle::row(weights,image,
                static_cast<q5k_a8_oracle::Mutation>(m+1U))) != expected;
        if (trial == 0) for (unsigned header : {0x7c00U,0xfc00U,0x7e01U,0x7c01U}) {
            auto bad=weights;bad[0]=static_cast<std::byte>(header&255U);bad[1]=static_cast<std::byte>(header>>8U);
            rejects([&] {static_cast<void>(evaluate_q5k_row(bad,image));});
            rejects([&] {static_cast<void>(q5k_a8_oracle::row(bad,image));});
        }
    }
    expect(std::all_of(detected.begin(),detected.end(),[](bool v){return v;}),
        "Q5 high-bit/signedness/block-order/tree/FMA mutations all have finite witnesses");
    std::vector<float> zero(256,-0.0F);auto z=q4k_a8_oracle::quantize(zero);
    std::vector<std::byte> zw(176,std::byte{0xff});zw[0]=std::byte{0};zw[1]=std::byte{0xbc};zw[2]=zw[3]=std::byte{0};
    expect(bits(evaluate_q5k_row(zw,z)) == 0 && bits(q5k_a8_oracle::row(zw,z)) == 0,"Q5 +0 writer seed");
    // F32 tiny activation uses the incumbent quantizer (no new tinyzero rule).
    std::fill(zero.begin(),zero.end(),std::ldexp(1.0F,-120));
    const auto tiny=quantize_f32(zero),tiny_oracle=q4k_a8_oracle::quantize(zero);
    expect(tiny.codes==tiny_oracle.codes && tiny.sums==tiny_oracle.sums &&
        bits(tiny.deltas[0])==bits(tiny_oracle.deltas[0]),"Q5 retains incumbent gradual tiny input image");
    Request request;request.enabled=true;request.gate_type=request.up_type=13;
    request.approved_numerical_identity=q5k_numerical_identity;
    Bindings b;std::array<DeviceBytes*,9> fields{&b.input,&b.gate,&b.up,&b.routes,&b.codes,&b.deltas,&b.sums,&b.status,&b.output};
    const std::array<std::uint64_t,9> sizes{512,5632,5632,4,256,32,32,4,256};
    for (unsigned i=0;i<fields.size();++i) *fields[i]={0x10000ULL*(i+1U),sizes[i]};
    validate(request,b);
    auto maximum=request;maximum.input=InputKind::fp32;maximum.shape={2560,640,512,10,1};
    Bindings large;std::array<DeviceBytes*,9> big{&large.input,&large.gate,&large.up,&large.routes,&large.codes,&large.deltas,&large.sums,&large.status,&large.output};
    const std::array<std::uint64_t,9> lengths{10240,576716800,576716800,40,2560,320,320,4,51200};
    for (unsigned i=0;i<big.size();++i) *big[i]={0x100000000ULL*(i+1U),lengths[i]};
    validate(maximum,large);
    for (unsigned mutation=0;mutation<10;++mutation) {
        auto req=request;auto v=b;
        if (mutation==0) req.approved_numerical_identity=numerical_identity;
        if (mutation==1) req.up_type=12;
        if (mutation==2) req.gate_type=req.up_type=12;
        if (mutation==3) v.gate.bytes=4608;
        if (mutation==4) ++v.up.bytes;
        if (mutation==5) ++v.gate.address;
        if (mutation==6) v.codes=v.output;
        if (mutation==7) req.fused_gate_up=true;
        if (mutation==8) req.shape.logical_rows=2;
        if (mutation==9) req.enabled=false;
        rejects([&] {validate(req,v);});
    }
}
}
int main() try {
    tests();std::cout<<"Q5K A8 unsigned original-byte/INT64/FP32/admission oracle PASS; no GPU or quality claim\n";
} catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
