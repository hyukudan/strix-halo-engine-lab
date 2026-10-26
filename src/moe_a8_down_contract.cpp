#include "strix/moe_a8_down_contract.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace strix::moe_q4k_a8 {
namespace {
void need(bool value,const char* why) { if(!value) throw std::invalid_argument(why); }
unsigned byte(std::byte x) { return std::to_integer<unsigned>(x); }
unsigned u16(std::span<const std::byte> b,std::size_t at) {return byte(b[at])|(byte(b[at+1])<<8U);}
float mul(float x,float y) {volatile float z=x*y;return z;}
float add(float x,float y) {volatile float z=x+y;return z;}
}
std::string_view down_numerical_identity(std::uint32_t type) {
    need(type==7 || type==8 || ((type==42 || type==20) && q20_a8_enabled()),
        "A8 down weights must be original Q5_1 or Q8_0 (or Q2_0 / IQ4_NL with STRIX_GSQ_Q20_A8=1)");
    return type==7?q51_a8_down_numerical_identity:type==42?q20_a8_down_numerical_identity:
        type==20?iq4nl_a8_down_numerical_identity:q80_a8_down_numerical_identity;
}
std::uint64_t down_row_bytes(std::uint32_t type,std::uint32_t intermediate) {
    if(type==20) return static_cast<std::uint64_t>(intermediate/32U)*18U;
    return type==42?static_cast<std::uint64_t>(intermediate/64U)*18U:
        static_cast<std::uint64_t>(intermediate/32U)*(type==7?24U:34U);
}
void validate_down(const DownRequest& r,const DownBindings& b) {
    need(r.enabled && r.approved_numerical_identity==down_numerical_identity(r.weight_type),
        "A8 down requires explicit matching numerical identity");
    const auto s=r.shape;
    need(s.hidden>0 && s.hidden<=max_hidden && s.hidden%64U==0 &&
        s.intermediate>0 && s.intermediate<=max_rows && s.intermediate%128U==0 &&
        s.experts>0 && s.experts<=max_experts && s.top_k>0 && s.top_k<=max_top_k,
        "A8 down admits bounded H%64 I%128 E<=512 K<=10 only");
    const std::uint64_t groups=static_cast<std::uint64_t>(s.top_k)*(s.intermediate/32U);
    const std::array<DeviceBytes,9> views{b.intermediates,b.weights,b.routes,b.codes,b.deltas,b.sums,b.rawdots,b.status,b.aggregate_status};
    const std::array<std::uint64_t,9> bytes{static_cast<std::uint64_t>(s.top_k)*2U*s.intermediate*4U,
        static_cast<std::uint64_t>(s.experts)*s.hidden*down_row_bytes(r.weight_type,s.intermediate),
        static_cast<std::uint64_t>(s.top_k)*4U,static_cast<std::uint64_t>(s.top_k)*s.intermediate,
        groups*4U,groups*4U,static_cast<std::uint64_t>(s.top_k)*s.hidden*4U,4,4};
    const std::array<unsigned,9> align{16,2,4,16,16,16,16,4,4};
    for(std::size_t i=0;i<views.size();++i) {
        if(i==8 && views[i].address==0 && views[i].bytes==0) continue;
        need(views[i].address!=0 && views[i].address%align[i]==0 && views[i].bytes==bytes[i] &&
            bytes[i]<=std::numeric_limits<std::uintptr_t>::max()-views[i].address,
            "A8 down view alignment/address/exact extent invalid");
        for(std::size_t j=0;j<i;++j)
            need(views[i].address+views[i].bytes<=views[j].address ||
                views[j].address+views[j].bytes<=views[i].address,"A8 down views must be disjoint");
    }
}
std::array<std::uint8_t,32> decode_down_q51_codes(std::span<const std::byte> b) {
    need(b.size()==24,"Q5_1 block requires24bytes");std::array<std::uint8_t,32> result{};
    for(unsigned i=0;i<32;++i) {
        const auto q=byte(b[8U+i%16U]);
        const auto low=(i<16?q:q>>4U)&15U;
        const auto high=(byte(b[4U+i/8U])>>(i%8U))&1U;
        result[i]=static_cast<std::uint8_t>(low|(high<<4U));
    }
    return result;
}
QuantizedRow quantize_down_values(std::span<const float> x) {
    require_f32_environment();need(!x.empty() && x.size()<=max_rows && x.size()%128U==0,"A8 down row length invalid");
    QuantizedRow q;q.codes.resize(x.size());q.deltas.resize(x.size()/32U);q.sums.resize(x.size()/32U);
    for(std::size_t g=0;g<q.deltas.size();++g) {
        float a=0;for(unsigned j=0;j<32;++j) {need(std::isfinite(x[g*32U+j]),"A8 down rejects nonfinite h");a=std::max(a,std::fabs(x[g*32U+j]));}
        const bool tiny=a>0 && a<0x1p-119F;
        volatile float divided=a/127.0F;
        const float d=a==0 || tiny?1.0F:divided;
        need(d>0 && std::isfinite(d),"A8 down delta invalid");q.deltas[g]=d;
        for(unsigned j=0;j<32;++j) {
            volatile float ratio=tiny?0.0F:x[g*32U+j]/d;
            const auto c=static_cast<std::int8_t>(std::clamp(std::nearbyint(static_cast<float>(ratio)),-127.0F,127.0F));
            q.codes[g*32U+j]=c;q.sums[g]+=c;
        }
    }
    return q;
}
float evaluate_down_row(std::span<const std::byte> w,const QuantizedRow& q,std::uint32_t type) {
    require_f32_environment();static_cast<void>(down_numerical_identity(type));
    need(!q.codes.empty() && q.codes.size()<=max_rows && q.codes.size()%128U==0 &&
        q.deltas.size()==q.codes.size()/32U && q.sums.size()==q.deltas.size(),"A8 down oracle image shape invalid");
    need(w.size()==down_row_bytes(type,static_cast<std::uint32_t>(q.codes.size())),"A8 down packed row size invalid");
    if(type==20) {
        std::array<float,4> lanes{};
        for(std::size_t g=0;g<q.deltas.size();++g) {
            const auto group=gsq_iq_group(20,w,static_cast<std::uint32_t>(g));
            std::int64_t sum=0,integer=0;
            for(unsigned j=0;j<32;++j){const auto c=q.codes[g*32U+j];need(c!=-128,"A8 down code -128 invalid");sum+=c;
                integer+=static_cast<std::int64_t>(c)*group.v[j];}
            need(sum==q.sums[g],"A8 down integer sum mismatch");
            const float contribution=mul(q.deltas[g],mul(group.s0,static_cast<float>(integer)));
            lanes[g%4U]=add(lanes[g%4U],contribution);
        }
        return add(add(lanes[0],lanes[1]),add(lanes[2],lanes[3]));
    }
    if(type==42) {
        std::array<float,4> lanes{};
        for(std::size_t g=0;g<q.deltas.size();++g) {
            need(q.deltas[g]>0 && std::isfinite(q.deltas[g]),"A8 down oracle delta invalid");
            const auto b=w.subspan((g/2U)*18U,18U);
            std::int64_t sum=0,integer=0;
            for(unsigned j=0;j<32;++j) {
                const auto c=q.codes[g*32U+j];need(c!=-128,"A8 down code -128 invalid");sum+=c;
                const auto e=(g%2U)*32U+j;
                integer+=static_cast<std::int64_t>(c)*static_cast<int>((byte(b[2U+e/4U])>>(2U*(e%4U)))&3U);
            }
            need(sum==q.sums[g] && std::abs(integer-sum)<=16256,"A8 down integer sum/bound mismatch");
            const float d=half_bits_to_float(static_cast<std::uint16_t>(u16(b,0)));
            const float contribution=mul(q.deltas[g],mul(d,static_cast<float>(integer-sum)));
            lanes[g%4U]=add(lanes[g%4U],contribution);
        }
        return add(add(lanes[0],lanes[1]),add(lanes[2],lanes[3]));
    }
    const auto block_bytes=type==7?24U:34U;
    std::array<float,4> lanes{};
    for(std::size_t g=0;g<q.deltas.size();++g) {
        need(q.deltas[g]>0 && std::isfinite(q.deltas[g]),"A8 down oracle delta invalid");
        const auto b=w.subspan(g*block_bytes,block_bytes);
        const auto codes=type==7?decode_down_q51_codes(b):std::array<std::uint8_t,32>{};
        std::int64_t sum=0,integer=0;
        for(unsigned j=0;j<32;++j) {
            const auto c=q.codes[g*32U+j];need(c!=-128,"A8 down code -128 invalid");sum+=c;
            const int weight=type==7?static_cast<int>(codes[j]):static_cast<int>(std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(byte(b[2U+j]))));
            integer+=static_cast<std::int64_t>(c)*weight;
        }
        need(sum==q.sums[g] && std::abs(integer)<=(type==7?125984:520192),"A8 down integer sum/bound mismatch");
        const float d=half_bits_to_float(static_cast<std::uint16_t>(u16(b,0)));
        float contribution=mul(d,static_cast<float>(integer));
        if(type==7) contribution=add(contribution,mul(half_bits_to_float(static_cast<std::uint16_t>(u16(b,2))),static_cast<float>(sum)));
        contribution=mul(q.deltas[g],contribution);
        lanes[g%4U]=add(lanes[g%4U],contribution);
    }
    return add(add(lanes[0],lanes[1]),add(lanes[2],lanes[3]));
}
} // namespace strix::moe_q4k_a8
