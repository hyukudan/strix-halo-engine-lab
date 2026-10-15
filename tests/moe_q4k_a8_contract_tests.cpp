#include "strix/moe_q4k_a8_contract.hpp"
#include "strix/low_precision.hpp"
#include "q4k_a8_oracle.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>

namespace {
using namespace strix::moe_q4k_a8;
void expect(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
std::uint32_t bits(float v) { return std::bit_cast<std::uint32_t>(v); }
void check_quant(std::span<const float> values) {
    const auto a = quantize_f32(values), b = q4k_a8_oracle::quantize(values);
    expect(a.codes == b.codes && a.sums == b.sums && a.deltas.size() == b.deltas.size(),"independent quantizer codes/sums");
    for (std::size_t g = 0; g < a.deltas.size(); ++g) expect(bits(a.deltas[g]) == bits(b.deltas[g]),"integer-rational binary32 delta");
}
void reject_quant(std::span<const float> values) {
    bool a = false,b = false;
    try { static_cast<void>(quantize_f32(values)); } catch (const std::invalid_argument&) { a = true; }
    try { static_cast<void>(q4k_a8_oracle::quantize(values)); } catch (const std::invalid_argument&) { b = true; }
    expect(a && b,"production and independent quantizer reject invalid group");
}
void tests(bool exhaustive) {
    require_f32_environment();
    const int original_round=std::fegetround();
    expect(std::fesetround(FE_UPWARD)==0,"test can set rounding mode");
    bool wrong_round_rejected=false;
    try { require_f32_environment(); } catch(const std::invalid_argument&) { wrong_round_rejected=true; }
    expect(std::fesetround(original_round)==0,"test restored rounding mode");
    expect(wrong_round_rejected,"non-nearest host environment rejected before arithmetic");
    std::mt19937 random(0x4a832U);
    bool reciprocal_mutant_detected=false;
    for (unsigned trial = 0; trial < 100000; ++trial) {
        const auto a = static_cast<std::uint32_t>(random()&0x7fffffffU), b = static_cast<std::uint32_t>(random()&0x7fffffffU);
        if (a >= 0x7f800000U || b == 0 || b >= 0x7f800000U) continue;
        const float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b);
        volatile float quotient = x/y;
        volatile float reciprocal=1.0F/y;volatile float product=x*reciprocal;
        if(std::isfinite(quotient) && std::isfinite(reciprocal) && std::isfinite(product) && quotient!=0 && product!=0 && reciprocal!=0)
            reciprocal_mutant_detected|=bits(quotient)!=bits(product);
        expect(q4k_a8_oracle::divide(a,b) == bits(quotient),"exact rational division agrees with independently executed binary32 hardware");
    }
    expect(reciprocal_mutant_detected,"finite reciprocal-multiply division mutant actually detected");
    // Wide-exponent activation/quantization boundary sweep; includes delta underflow.
    for(unsigned trial=0;trial<1000U;++trial) {
        const unsigned maximum=1U+static_cast<unsigned>(random()%0x7f7fffffU);
        const float amax=std::bit_cast<float>(maximum);
        const auto dbits=q4k_a8_oracle::divide(maximum,0x42fe0000U);
        std::vector<float> sweep(32);sweep[0]=amax;
        if(dbits==0) {reject_quant(sweep);continue;}
        const float delta=std::bit_cast<float>(dbits);
        for(unsigned lane=1;lane<32U;++lane) {
            const float boundary=delta*(static_cast<float>(lane)+0.5F);
            sweep[lane]=std::min(amax,std::nextafter(boundary,(lane&1U)?0.0F:amax));
            if(lane&2U) sweep[lane]=-sweep[lane];
        }
        check_quant(sweep);
    }
    std::vector<float> values(32,0.0F); check_quant(values);
    std::fill(values.begin(),values.end(),-0.0F); check_quant(values);
    values[0] = 127; values[1] = 2.5F; values[2] = 3.5F; values[3] = -2.5F; values[4] = -3.5F;
    check_quant(values); const auto ties = quantize_f32(values);
    expect(ties.codes[1] == 2 && ties.codes[2] == 4 && ties.codes[3] == -2 && ties.codes[4] == -4,"ties round even under delta1");
    values[1] = std::nextafter(2.5F,std::numeric_limits<float>::infinity());
    expect(quantize_f32(values).codes[1] == 3,"one-ULP activation crossing the exact rounding boundary changes code");
    check_quant(values);
    // This exact rational quotient is slightly ABOVE2.5, but binary32 division
    // first rounds it TO2.5; RN integer is2. Direct rational->integer is wrongly3.
    std::fill(values.begin(),values.end(),0);values[0]=std::bit_cast<float>(0x3a8be27eU);
    values[1]=std::bit_cast<float>(0x37b03b95U);check_quant(values);
    const auto boundary=quantize_f32(values);
    const double direct=static_cast<double>(values[1])/static_cast<double>(boundary.deltas[0]);
    expect(bits(boundary.deltas[0])==0x370cfc77U && boundary.codes[1]==2 &&
        std::nearbyint(direct)==3.0,"direct-rational integer-round mutant actually detected");
    for(unsigned half=0;half<65536U;++half) {
        const float expected=q4k_a8_oracle::half(static_cast<std::uint16_t>(half));
        const float actual=half_bits_to_float(static_cast<std::uint16_t>(half));
        expect(std::isnan(expected)?std::isnan(actual):bits(actual)==bits(expected),"all65536 half values; NaN classification scope");
    }
    for (const auto magnitude : {std::numeric_limits<float>::max(),std::ldexp(1.0F,-133),
        std::bit_cast<float>(std::uint32_t{64}),std::bit_cast<float>(std::uint32_t{190})}) {
        std::fill(values.begin(),values.end(),0); values[0] = magnitude; values[1] = -magnitude; check_quant(values);
    }
    std::fill(values.begin(),values.end(),0); values[0] = std::bit_cast<float>(std::uint32_t{64});
    expect(quantize_f32(values).codes[0] == 64,"subnormal delta rounding does not imply maximum maps127");
    for (const auto invalid : {std::numeric_limits<float>::denorm_min(), std::bit_cast<float>(std::uint32_t{63}), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        std::fill(values.begin(),values.end(),0); values[31] = invalid; reject_quant(values);
    }
    std::vector<std::uint16_t> bf16(32,1U); const auto bf = quantize_bf16(bf16);
    std::fill(values.begin(),values.end(),std::ldexp(1.0F,-133)); check_quant(values);
    expect(bf.codes == quantize_f32(values).codes,"BF16 minimum subnormal expands exactly");
    std::array<std::byte,12> packed{};
    for (unsigned group = 0; group < 8; ++group)
        for (unsigned s = 0; s < 64; ++s) for (unsigned m = 0; m < 64; ++m) {
            for (auto& b : packed) b = static_cast<std::byte>(random()&255U);
            const auto lo = group&3U;
            if (group < 4) {
                packed[lo] = static_cast<std::byte>((std::to_integer<unsigned>(packed[lo])&192U)|s);
                packed[lo+4U] = static_cast<std::byte>((std::to_integer<unsigned>(packed[lo+4U])&192U)|m);
            } else {
                packed[lo] = static_cast<std::byte>((std::to_integer<unsigned>(packed[lo])&63U)|((s>>4U)<<6U));
                packed[lo+4U] = static_cast<std::byte>((std::to_integer<unsigned>(packed[lo+4U])&63U)|((m>>4U)<<6U));
                packed[lo+8U] = static_cast<std::byte>((s&15U)|((m&15U)<<4U));
            }
            const auto actual = decode_factors(packed,group); const auto expected = q4k_a8_oracle::factors(packed,group);
            expect(actual.scale == s && actual.minimum == m && expected.first == static_cast<int>(s) && expected.second == static_cast<int>(m),
                "every six-bit factor pair and group matches independent original formula");
        }
    if (exhaustive)
        for (unsigned lo = 0; lo < 4; ++lo) for (unsigned tuple = 0; tuple < (1U<<24U); ++tuple) {
            packed[lo] = static_cast<std::byte>(tuple&255U); packed[lo+4U] = static_cast<std::byte>((tuple>>8U)&255U);
            packed[lo+8U] = static_cast<std::byte>(tuple>>16U);
            for (const auto group : {lo,lo+4U}) {
                const auto a = decode_factors(packed,group); const auto b = q4k_a8_oracle::factors(packed,group);
                expect(a.scale == static_cast<unsigned>(b.first) && a.minimum == static_cast<unsigned>(b.second),"exhaustive byte-triple decoder");
            }
        }
    std::array<bool,13> detected{};
    for (const unsigned width : {256U,512U,2560U}) for (unsigned trial = 0; trial < 100; ++trial) {
        values.resize(width);
        for (auto& x : values) x = std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),static_cast<int>(random()%15U)-8);
        check_quant(values); const auto quantized = quantize_f32(values);
        std::vector<std::byte> weights(width/256U*144U);
        for (auto& x : weights) x = static_cast<std::byte>(random()&255U);
        for (unsigned block = 0; block < width/256U; ++block) {
            const std::array<std::uint16_t,10> headers{0,0x8000,1,0x8001,0x03ff,0x83ff,0x3c00,0xbc00,0x7bff,0xfbff};
            const auto d=headers[random()%headers.size()],dm=headers[random()%headers.size()];
            weights[block*144U] = static_cast<std::byte>(d&255U); weights[block*144U+1U] = static_cast<std::byte>(d>>8U);
            weights[block*144U+2U] = static_cast<std::byte>(dm&255U); weights[block*144U+3U] = static_cast<std::byte>(dm>>8U);
        }
        const auto expected = bits(q4k_a8_oracle::row(weights,quantized));
        expect(bits(evaluate_row(weights,quantized)) == expected,"independent separate-FP32 affine combine and writer tree");
        for (unsigned mutation = 0; mutation < 4; ++mutation) {
            auto w = weights; auto a = quantized;
            if (mutation == 0) w[w.size()-1U] ^= std::byte{0x10};
            if (mutation == 1) w[(width/256U-1U)*144U+15U] ^= std::byte{0xc0};
            if (mutation == 2) { w[2] = std::byte{0}; w[3] = std::byte{0}; }
            if (mutation == 3) { auto x = values; x[1] *= -1; a = quantize_f32(x); }
            detected[mutation] = detected[mutation] || bits(q4k_a8_oracle::row(w,a)) != expected;
        }
        const std::array mutations{q4k_a8_oracle::Mutation::fused,q4k_a8_oracle::Mutation::reverse_blocks,q4k_a8_oracle::Mutation::wrong_tree,q4k_a8_oracle::Mutation::nibble_swap,q4k_a8_oracle::Mutation::hp_swap,q4k_a8_oracle::Mutation::header_swap,q4k_a8_oracle::Mutation::factor_swap,q4k_a8_oracle::Mutation::unsigned_a,q4k_a8_oracle::Mutation::missing_xor4};
        for (unsigned m = 0; m < mutations.size(); ++m)
            detected[m+4U] = detected[m+4U] || bits(q4k_a8_oracle::row(weights,quantized,mutations[m])) != expected;
        std::array<std::uint8_t,32> q{},a_bytes{},biased{}; std::array<std::int8_t,32> a{};
        std::int32_t raw_biased = 0; std::int64_t sumq = 0, suma = 0;
        for (unsigned l = 0; l < 32; ++l) {
            q[l] = static_cast<std::uint8_t>(random()%16U); a[l] = static_cast<std::int8_t>(static_cast<int>(random()%255U)-127);
            a_bytes[l] = static_cast<std::uint8_t>(a[l]); biased[l] = static_cast<std::uint8_t>(static_cast<int>(a[l])+127);
            raw_biased += q[l]*biased[l]; sumq += q[l]; suma += biased[l];
        }
        const auto integers = integer_group(q,a,{63,63});
        const auto affine = strix::low_precision::affine_group_dot({strix::low_precision::CodeEncoding::unsigned_i4,q,1,0},
            {strix::low_precision::CodeEncoding::signed_i8,a_bytes,1,0});
        expect(integers.raw == affine.corrected_dot && integers.scaled == integers.raw*63 && integers.minimum == integers.sum*63,
            "INT64-backed affine oracle and bounds");
        expect(strix::low_precision::correct_dot(raw_biased,sumq,suma,32,0,127) == integers.raw,"unsigned bias127 correction exact");
    }
    expect(std::all_of(detected.begin(),detected.end(),[](bool x){return x;}),"all packed/activation/FMA/order mutants detected");
    std::vector<float> zero_values(256,0.0F);const auto zero_quant=quantize_f32(zero_values);
    std::vector<std::byte> zero_weights(144,std::byte{0xff});
    zero_weights[0]=std::byte{0};zero_weights[1]=std::byte{0xbc};zero_weights[2]=std::byte{0};zero_weights[3]=std::byte{0};
    expect(bits(evaluate_row(zero_weights,zero_quant))==0 &&
        bits(q4k_a8_oracle::row(zero_weights,zero_quant,q4k_a8_oracle::Mutation::negative_zero_seed))==0x80000000U,
        "negative zero seed mutant detected through full qualified combine tree");
    // Pure admission: default disabled, wrong approval, dtype, C>1 and bad views reject before any HIP call.
    Request request; request.enabled = true; request.approved_numerical_identity = numerical_identity;
    Bindings b; std::array<DeviceBytes*,9> fields{&b.input,&b.gate,&b.up,&b.routes,&b.codes,&b.deltas,&b.sums,&b.status,&b.output};
    const std::array<std::uint64_t,9> lengths{512,4608,4608,4,256,32,32,4,256};
    for (unsigned i = 0; i < fields.size(); ++i) *fields[i] = {0x10000ULL+i*0x10000ULL,lengths[i]};
    validate(request,b);
    auto f32=request;f32.input=InputKind::fp32;auto f32b=b;f32b.input.bytes=1024;validate(f32,f32b);
    auto alias=b;alias.input.address=alias.routes.address;validate(request,alias);
    alias=b;alias.routes.address=alias.gate.address;validate(request,alias);
    Request maximum=request;maximum.input=InputKind::fp32;maximum.shape={2560,640,512,10,1};
    Bindings large;std::array<DeviceBytes*,9> large_fields{&large.input,&large.gate,&large.up,&large.routes,&large.codes,&large.deltas,&large.sums,&large.status,&large.output};
    const std::array<std::uint64_t,9> large_lengths{10240,471859200,471859200,40,2560,320,320,4,51200};
    for(unsigned i=0;i<large_fields.size();++i) *large_fields[i]={0x100000000ULL*(i+1U),large_lengths[i]};
    validate(maximum,large);
    for (unsigned change = 0; change < 25; ++change) {
        auto q = request; auto v = b;
        if (change == 0) q.enabled = false;
        if (change == 1) q.approved_numerical_identity = "wrong";
        if (change == 2) q.gate_type = 13;
        if (change == 3) q.fused_gate_up = true;
        if (change == 4) q.shape.logical_rows = 2;
        if (change == 5) v.gate.address += 1;
        if (change == 6) --v.output.bytes;
        if (change == 7) v.codes.address = v.output.address;
        if (change == 8) q.shape.hidden=0;
        if (change == 9) q.shape.hidden=257;
        if (change == 10) q.shape.hidden=2816;
        if (change == 11) q.shape.rows=0;
        if (change == 12) q.shape.rows=31;
        if (change == 13) q.shape.rows=672;
        if (change == 14) q.shape.experts=0;
        if (change == 15) q.shape.experts=513;
        if (change == 16) q.shape.top_k=0;
        if (change == 17) q.shape.top_k=11;
        if (change == 18) q.input=static_cast<InputKind>(99);
        if (change == 19) ++v.gate.bytes;
        if (change == 20) --v.routes.bytes;
        if (change == 21) v.codes.address=std::numeric_limits<std::uintptr_t>::max()-15U;
        if (change == 22) v.up=v.gate;
        if (change == 23) v.input.address=v.codes.address;
        if (change == 24) v.status.address+=4;
        bool rejected = false; try { validate(q,v); } catch (const std::invalid_argument& error) {
            rejected = true;
            if(change>=8 && change<=17) expect(std::string_view(error.what()).find("bounded C1 dimensions")!=std::string_view::npos,
                "invalid dimensions must reject at dimension gate before incidental byte mismatch");
        }
        expect(rejected,"invalid probe admission fails before enqueue");
    }
}
}
int main(int argc,char** argv) try {
    tests(argc == 2 && std::string_view(argv[1]) == "--exhaustive");
    std::cout << "Q4K A8G32 independent binary32 quantizer/integer-affine/combine/admission tests PASS; no GPU\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
