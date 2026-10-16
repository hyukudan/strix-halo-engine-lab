#include "strix/hip_moe_a8_down.hpp"
#include "strix/moe_batched_activation_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace strix::moe_q4k_a8;
namespace {
void expect(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
void check(hipError_t e){if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e));}
struct Stream {
    const strix::HipSetupQuiescenceGuard& guard;hipStream_t value=nullptr;
    explicit Stream(const strix::HipSetupQuiescenceGuard& g):guard(g){guard.run([&]{check(hipStreamCreateWithFlags(&value,hipStreamNonBlocking));});}
    ~Stream(){if(value)guard.run([&]{check(hipStreamDestroy(value));});}
};
struct Buffer {
    const strix::HipSetupQuiescenceGuard& guard;void* owner=nullptr;std::size_t size;
    Buffer(std::size_t n,const strix::HipSetupQuiescenceGuard& g):guard(g),size(n){guard.run([&]{check(hipMalloc(&owner,n+32U));check(hipMemset(owner,0xa5,n+32U));});}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(owner)guard.run([&]{check(hipFree(owner));});}
    void* data()const{return static_cast<unsigned char*>(owner)+16U;}
    DeviceBytes view()const{return {reinterpret_cast<std::uintptr_t>(data()),size};}
    void upload(const void* p){guard.run([&]{check(hipMemcpy(data(),p,size,hipMemcpyHostToDevice));});}
    void poison(){guard.run([&]{check(hipMemset(data(),0xa5,size));});}
    template<class T> std::vector<T> read()const {
        expect(size%sizeof(T)==0,"fixture read geometry");std::vector<T> out(size/sizeof(T));
        guard.run([&]{check(hipMemcpy(out.data(),data(),size,hipMemcpyDeviceToHost));});return out;
    }
    void guards()const {
        std::array<unsigned char,32> b{};
        guard.run([&]{check(hipMemcpy(b.data(),owner,16,hipMemcpyDeviceToHost));check(hipMemcpy(b.data()+16,static_cast<unsigned char*>(data())+size,16,hipMemcpyDeviceToHost));});
        for(auto x:b) expect(x==0xa5,"buffer canary overwritten");
    }
    void unchanged()const {for(auto x:read<unsigned char>())expect(x==0xa5,"rejected work published rawdots");guards();}
};
void put16(std::vector<std::byte>& w,std::size_t at,unsigned bits){w[at]=std::byte(bits&255U);w[at+1]=std::byte((bits>>8U)&255U);}
void finite_case(unsigned type,unsigned hidden,unsigned width,unsigned slots,
    const strix::HipSetupQuiescenceGuard& guard,hipStream_t stream) {
    constexpr unsigned experts=3;
    const auto block_bytes=type==7?24U:34U;
    const std::size_t weight_size=static_cast<std::size_t>(experts)*hidden*(width/32U)*block_bytes;
    std::vector<std::byte> weights(weight_size);
    // Distinct expert/row/block headers and code bytes, including all qh bits,
    // both Q5 nibbles, Q8 -128 and misaligned34-byte block starts.
    for(unsigned e=0;e<experts;++e) for(unsigned row=0;row<hidden;++row) for(unsigned g=0;g<width/32U;++g) {
        const auto at=(static_cast<std::size_t>(e)*hidden+row)*(width/32U)*block_bytes+g*block_bytes;
        put16(weights,at,(row+g)%3U==0?0x3000U:0xb000U);
        if(type==7) {
            put16(weights,at+2U,(g+e)%2U?0x2800U:0xa800U);
            for(unsigned j=4;j<24;++j) weights[at+j]=std::byte((row*29U+e*43U+g*19U+j*71U)&255U);
        } else for(unsigned j=0;j<32;++j) weights[at+2U+j]=std::byte((row*17U+g*31U+e*53U+j*23U)&255U);
    }
    std::vector<float> input(static_cast<std::size_t>(slots)*2U*width);
    for(unsigned slot=0;slot<slots;++slot) for(unsigned j=0;j<width;++j) {
        const auto at=static_cast<std::size_t>(slot)*2U*width+j;
        input[at]=static_cast<float>(static_cast<int>((j*7U+slot*11U)%65U)-32)/8.0F;
        input[at+width]=static_cast<float>(static_cast<int>((j*13U+slot*5U)%127U)-63)/16.0F;
        if(j<32U) {input[at]=1.0F;input[at+width]=0x1p-125F;} // deliberate tiny group
        if(j>=32U && j<64U) input[at]=(j&1U)?-0.0F:0.0F;
        if(j>=64U && j<96U) {input[at]=(j&1U)?-2.0F:2.0F;input[at+width]=0x1p100F;}
    }
    std::vector<unsigned> routes(slots);for(unsigned slot=0;slot<slots;++slot)routes[slot]=(slot*2U+1U)%experts;
    Buffer intermediates(input.size()*4U,guard),reference(input.size()*4U,guard),w(weight_size,guard),r(slots*4U,guard),
        codes(slots*width,guard),deltas(slots*(width/32U)*4U,guard),sums(slots*(width/32U)*4U,guard),
        raw(slots*hidden*4U,guard),status(4,guard),aggregate(4,guard);
    const unsigned zero=0;intermediates.upload(input.data());reference.upload(input.data());w.upload(weights.data());r.upload(routes.data());status.upload(&zero);aggregate.upload(&zero);
    DownRequest req;req.enabled=true;req.approved_numerical_identity=down_numerical_identity(type);req.shape={hidden,width,experts,slots};req.weight_type=type;
    DownBindings b{intermediates.view(),w.view(),r.view(),codes.view(),deltas.view(),sums.view(),raw.view(),status.view(),aggregate.view()};
    guard.run([&]{
        // Call the ACTUAL production TU kernel with its own compilation flags,
        // not an expression clone built under this fixture's strict flags.
        strix::moe_batched_activation_diagnostic::enqueue({width,slots},
            {{reference.view().address,reference.view().bytes},{status.view().address,status.view().bytes}},
            strix::moe_batched_activation_diagnostic::Mode::batched,stream);
        enqueue_down_activation_quantize(req,b,stream);enqueue_down_rawdots(req,b,stream);
    });
    expect(status.read<unsigned>()[0]==0 && aggregate.read<unsigned>()[0]==0,"finite producer status");
    const auto actual_h=intermediates.read<float>(),old_h=reference.read<float>();
    expect(std::memcmp(actual_h.data(),old_h.data(),input.size()*4U)==0,"producer h/up complete bytes vs actual runtime batched activation");
    const auto actual_codes=codes.read<std::int8_t>();
    const auto actual_sums=sums.read<std::int32_t>();
    const auto actual_delta=deltas.read<float>(),actual_raw=raw.read<float>();
    bool q51_offset_power=false,q8_signed_power=false;
    for(unsigned slot=0;slot<slots;++slot) {
        const auto q=quantize_down_values(std::span<const float>(actual_h).subspan(static_cast<std::size_t>(slot)*2U*width,width));
        for(unsigned g=0;g<width/32U;++g) {
            const auto index=slot*(width/32U)+g;
            expect(actual_sums[index]==q.sums[g] && std::bit_cast<unsigned>(actual_delta[index])==std::bit_cast<unsigned>(q.deltas[g]),"delta/C independent CPU quantizer");
        }
        expect(std::equal(q.codes.begin(),q.codes.end(),actual_codes.begin()+slot*width),"full codes independent CPU quantizer");
        for(unsigned row=0;row<hidden;++row) {
            const auto at=(static_cast<std::size_t>(routes[slot])*hidden+row)*(width/32U)*block_bytes;
            const auto weight_row=std::span<const std::byte>(weights).subspan(at,(width/32U)*block_bytes);
            const auto expected=evaluate_down_row(weight_row,q,type);
            expect(std::bit_cast<unsigned>(expected)==std::bit_cast<unsigned>(actual_raw[slot*hidden+row]),"full ordered rawdot FP32 bits vs original-byte CPU oracle");
            if(type==7) {auto mutant=std::vector<std::byte>(weight_row.begin(),weight_row.end());
                for(unsigned g=0;g<width/32U;++g){mutant[g*24U+2U]=std::byte{0};mutant[g*24U+3U]=std::byte{0};}
                q51_offset_power|=std::bit_cast<unsigned>(evaluate_down_row(mutant,q,type))!=std::bit_cast<unsigned>(expected);
            } else q8_signed_power|=expected<0;
        }
    }
    expect(type==7?q51_offset_power:q8_signed_power,"signed/offset mutant fixture has power");
    expect(w.read<std::byte>()==weights && r.read<unsigned>()==routes,"read-only original weight/routes unchanged");
    for(auto* view:{&intermediates,&reference,&w,&r,&codes,&deltas,&sums,&raw,&status,&aggregate})view->guards();
    // Sticky status is not cleared and no raw result is published.
    const unsigned prior=64,prior_aggregate=32;status.upload(&prior);aggregate.upload(&prior_aggregate);raw.poison();
    guard.run([&]{enqueue_down_activation_quantize(req,b,stream);enqueue_down_rawdots(req,b,stream);});
    expect(status.read<unsigned>()[0]==prior && aggregate.read<unsigned>()[0]==prior_aggregate,"sticky first status preserved");raw.unchanged();
    // Nonfinite input must be caught before abs-max can mask NaN; raw then skips.
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        auto invalid=input;invalid[0]=bad;intermediates.upload(invalid.data());status.upload(&zero);aggregate.upload(&prior_aggregate);raw.poison();
        guard.run([&]{enqueue_down_activation_quantize(req,b,stream);enqueue_down_rawdots(req,b,stream);});
        expect((status.read<unsigned>()[0]&down_nonfinite_activation)!=0 && aggregate.read<unsigned>()[0]==(prior_aggregate|down_aggregate_error),"nonfinite local/aggregate OR status");raw.unchanged();
    }
    // Even standalone consumer cannot read a foreign expert row.
    std::fill(routes.begin(),routes.end(),experts);r.upload(routes.data());status.upload(&zero);aggregate.upload(&zero);raw.poison();
    guard.run([&]{enqueue_down_rawdots(req,b,stream);});
    expect(status.read<unsigned>()[0]==down_invalid_route && aggregate.read<unsigned>()[0]==down_aggregate_error,"invalid route consumer bounds");raw.unchanged();
    // Host metadata rejection, then capture rejection, both before any node.
    auto bad_binding=b;bad_binding.sums.address=bad_binding.codes.address;
    bool rejected=false;try{enqueue_down_rawdots(req,bad_binding,stream);}catch(const std::invalid_argument&){rejected=true;}
    expect(rejected,"pre-enqueue aliases rejected");
    check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
    bool rejected_producer=false,rejected_consumer=false;
    try{enqueue_down_activation_quantize(req,b,stream);}catch(const std::invalid_argument&){rejected_producer=true;}
    try{enqueue_down_rawdots(req,b,stream);}catch(const std::invalid_argument&){rejected_consumer=true;}
    hipGraph_t graph=nullptr;check(hipStreamEndCapture(stream,&graph));
    std::size_t nodes=0;check(hipGraphGetNodes(graph,nullptr,&nodes));check(hipGraphDestroy(graph));
    expect(rejected_producer && rejected_consumer && nodes==0,"capture rejected with zero enqueued nodes");
    for(auto* view:{&intermediates,&reference,&w,&r,&codes,&deltas,&sums,&raw,&status,&aggregate})view->guards();
    expect(w.read<std::byte>()==weights,"weights remain read-only after failures/capture");
}
}
int main() try {
    strix::HipSetupQuiescenceGuard guard;expect(guard.architecture().starts_with("gfx1151"),"actual gfx1151 required");
    require_f32_environment();Stream stream(guard);
    for(unsigned type:{7U,8U}) for(unsigned width:{128U,640U}) for(unsigned slots:{1U,10U})
        finite_case(type,width==128U?64U:2560U,width,slots,guard,stream.value);
    std::cout<<"A8 down component fixtures PASS; no model quality/performance qualification\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
