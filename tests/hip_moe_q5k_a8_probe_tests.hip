#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "q5k_a8_oracle.hpp"
#include <cstring>
#include <iostream>
#include <random>
#include <utility>

namespace {
using namespace strix::moe_q4k_a8;
void expect(bool value,const char* message) {if (!value) throw std::runtime_error(message);}
void check(hipError_t error) {if (error!=hipSuccess) throw std::runtime_error(hipGetErrorString(error));}
struct Buffer {
    void* allocation=nullptr;
    std::size_t bytes;
    const strix::HipSetupQuiescenceGuard& guard;
    Buffer(std::size_t n,const strix::HipSetupQuiescenceGuard& g):bytes(n),guard(g) {
        guard.require_original();check(hipMalloc(&allocation,n+32U));
    }
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer() {if (allocation) {guard.recover_or_fail_stop();(void)hipFree(allocation);}}
    void* data() const {return static_cast<unsigned char*>(allocation)+16U;}
    DeviceBytes view() const {return {reinterpret_cast<std::uintptr_t>(data()),bytes};}
    void poison(hipStream_t stream) const {check(hipMemsetAsync(allocation,0xa5,bytes+32U,stream));}
    void upload(const void* source,hipStream_t stream) const {
        check(hipMemcpyAsync(data(),source,bytes,hipMemcpyHostToDevice,stream));
    }
    std::vector<unsigned char> owner() const {
        std::vector<unsigned char> result(bytes+32U);
        guard.run([&] {check(hipMemcpy(result.data(),allocation,result.size(),hipMemcpyDeviceToHost));});
        return result;
    }
    template<class T> std::vector<T> read() const {
        std::vector<T> result(bytes/sizeof(T));
        guard.run([&] {check(hipMemcpy(result.data(),data(),bytes,hipMemcpyDeviceToHost));});
        return result;
    }
    void guards() const {
        const auto all=owner();
        for (unsigned i=0;i<16;++i) expect(all[i]==0xa5 && all[all.size()-16U+i]==0xa5,"Q5 probe guard overwritten");
    }
    void untouched() const {
        const auto all=owner();expect(std::all_of(all.begin(),all.end(),[](auto b){return b==0xa5;}),
            "Q5 rejected/sticky call published workspace/output");
    }
};
struct Stream {
    hipStream_t value=nullptr;
    const strix::HipSetupQuiescenceGuard& guard;
    explicit Stream(const strix::HipSetupQuiescenceGuard& g):guard(g) {check(hipStreamCreate(&value));}
    ~Stream() {if (value) {guard.recover_or_fail_stop();(void)hipStreamDestroy(value);}}
    Stream(const Stream&)=delete;Stream& operator=(const Stream&)=delete;
};
void fixture(unsigned hidden,unsigned rows,unsigned top_k,bool bf16,unsigned special,
    bool negatives,const strix::HipSetupQuiescenceGuard& guard) {
    Stream stream(guard);
    Request request;request.enabled=true;request.approved_numerical_identity=q5k_numerical_identity;
    request.gate_type=request.up_type=13;request.input=bf16?InputKind::bf16:InputKind::fp32;
    const unsigned experts=std::max(top_k,3U);request.shape={hidden,rows,experts,top_k,1};
    const std::size_t row_bytes=hidden/256U*176U,weight_bytes=experts*rows*row_bytes;
    Buffer input(hidden*(bf16?2U:4U),guard),gate(weight_bytes,guard),up(weight_bytes,guard),routes(top_k*4U,guard),
        codes(hidden,guard),deltas(hidden/8U,guard),sums(hidden/8U,guard),status(4,guard),out(top_k*2U*rows*4U,guard);
    const Bindings binding{input.view(),gate.view(),up.view(),routes.view(),codes.view(),deltas.view(),sums.view(),status.view(),out.view()};
    std::mt19937 random(0x5a832U+hidden+rows+top_k+special);
    std::vector<float> activation(hidden);std::vector<std::uint16_t> bf(hidden);
    for (unsigned i=0;i<hidden;++i) {
        activation[i]=std::ldexp(static_cast<float>(static_cast<int>(random()%255U)-127),-6);
        if (special==1) activation[i]=(i&1U)?-0.0F:0.0F;
        if (special==2) activation[i]=std::ldexp((i&1U)?-1.0F:1.0F,-120);
        if (special==3) activation[i]=i%32U==0?127.0F:i%32U==1?2.5F:i%32U==2?3.5F:-2.5F;
        bf[i]=static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(activation[i])>>16U);
        if (bf16) activation[i]=std::bit_cast<float>(static_cast<std::uint32_t>(bf[i])<<16U);
    }
    const auto image=q4k_a8_oracle::quantize(activation);
    std::vector<std::byte> gw(weight_bytes),uw(weight_bytes);
    for (auto* weights:{&gw,&uw}) {
        for (auto& value:*weights) value=static_cast<std::byte>(random()&255U);
        for (std::size_t block=0;block<weight_bytes/176U;++block) {
            const std::array<unsigned,10> headers{0,0x8000,1,0x8001,0x03ff,0x83ff,0x3c00,0xbc00,0x7bff,0xfbff};
            const auto d=headers[random()%headers.size()],dm=headers[random()%headers.size()];
            (*weights)[block*176U]=static_cast<std::byte>(d&255U);(*weights)[block*176U+1U]=static_cast<std::byte>(d>>8U);
            (*weights)[block*176U+2U]=static_cast<std::byte>(dm&255U);(*weights)[block*176U+3U]=static_cast<std::byte>(dm>>8U);
        }
    }
    std::vector<unsigned> rt(top_k);
    for (unsigned slot=0;slot<top_k;++slot) rt[slot]=experts-1U-(slot%experts); // Last expert plus distinct/duplicate slots.
    const auto upload=[&] {
        input.upload(bf16?static_cast<const void*>(bf.data()):static_cast<const void*>(activation.data()),stream.value);
        gate.upload(gw.data(),stream.value);up.upload(uw.data(),stream.value);routes.upload(rt.data(),stream.value);
    };
    const auto reset=[&](unsigned preset=0) {
        for (const auto* buffer:{&codes,&deltas,&sums,&out,&status}) buffer->poison(stream.value);
        check(hipMemcpyAsync(status.data(),&preset,4,hipMemcpyHostToDevice,stream.value));
        // This host scalar stays alive until completion inside this helper.
        check(hipStreamSynchronize(stream.value));
    };
    guard.run([&] {for (const auto* buffer:{&input,&gate,&up,&routes}) buffer->poison(stream.value);upload();reset();});
    const std::array readonly{input.owner(),gate.owner(),up.owner(),routes.owner()};
    const auto readonly_unchanged=[&] {
        expect(input.owner()==readonly[0] && gate.owner()==readonly[1] && up.owner()==readonly[2] && routes.owner()==readonly[3],
            "Q5 projection changed read-only data/guards");
    };
    for (unsigned repeat=0;repeat<3;++repeat) {
        guard.run([&] {reset();enqueue_probe(request,binding,stream.value);});
        expect(status.read<unsigned>()[0]==0,"Q5 finite native status");
        expect(codes.read<std::int8_t>()==image.codes && sums.read<int>()==image.sums,"Q5 incumbent activation image codes/C");
        const auto ds=deltas.read<unsigned>();
        for (unsigned g=0;g<ds.size();++g) expect(ds[g]==std::bit_cast<unsigned>(image.deltas[g]),"Q5 activation delta bits");
        const auto actual=out.read<unsigned>();
        for (unsigned slot=0;slot<top_k;++slot) for (unsigned which=0;which<2;++which) for (unsigned row=0;row<rows;++row) {
            const auto& weights=which==0?gw:uw;
            const auto raw=std::span<const std::byte>(weights).subspan((static_cast<std::size_t>(rt[slot])*rows+row)*row_bytes,row_bytes);
            const float expected=q5k_a8_oracle::row(raw,image);
            expect(actual[(slot*2U+which)*rows+row]==std::bit_cast<unsigned>(expected),
                "Q5 native full output differs from independent original-byte epilogue/tree oracle");
        }
        for (const auto* buffer:{&codes,&deltas,&sums,&out,&status}) buffer->guards();
        readonly_unchanged();
    }
    if (!negatives) return;
    const auto all_owners=[&] {
        return std::array{input.owner(),gate.owner(),up.owner(),routes.owner(),codes.owner(),
            deltas.owner(),sums.owner(),status.owner(),out.owner()};
    };
    const auto no_publication=[&] {
        for (const auto* buffer:{&codes,&deltas,&sums,&out}) buffer->untouched();
        status.guards();
    };
    guard.run([&] {reset(0x40000000U);enqueue_probe(request,binding,stream.value);});
    expect(status.read<unsigned>()[0]==0x40000000U,"Q5 caller sticky status reset");no_publication();readonly_unchanged();
    rt.back()=experts;
    guard.run([&] {routes.upload(rt.data(),stream.value);reset();enqueue_probe(request,binding,stream.value);});
    expect(status.read<unsigned>()[0]==invalid_route,"Q5 last-route bound error");no_publication();
    guard.run([&] {reset(0x40000000U);enqueue_probe(request,binding,stream.value);});
    expect(status.read<unsigned>()[0]==(0x40000000U|invalid_route),"Q5 invalid route erased prior sticky bits");no_publication();
    rt.back()=0;guard.run([&] {routes.upload(rt.data(),stream.value);});
    const auto finite_activation=activation;const auto finite_bf=bf;
    for (unsigned bad=0;bad<4;++bad) {
        const auto old=activation.back();const auto old_bf=bf.back();
        activation.back()=bad==0?std::numeric_limits<float>::quiet_NaN():bad==1?
            std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity();
        bf.back()=bad==0?0x7fc1U:bad==1?0x7f80U:0xff80U;
        if (bad==3) {
            if (bf16) {activation.back()=old;bf.back()=old_bf;continue;}
            for (unsigned i=hidden-32U;i<hidden;++i) activation[i]=std::numeric_limits<float>::denorm_min();
        }
        guard.run([&] {upload();reset();enqueue_probe(request,binding,stream.value);});
        expect(status.read<unsigned>()[0]==(bad==3?invalid_delta:nonfinite_activation),"Q5 invalid activation status");
        no_publication();activation.back()=old;bf.back()=old_bf;
    }
    // Restore exact original source bytes, including the BF16 input boundary.
    activation=finite_activation;bf=finite_bf;
    guard.run([&] {upload();reset();});
    for (unsigned change=0;change<9;++change) {
        auto q=request;auto b=binding;hipStream_t selected_stream=stream.value;
        if (change==0) q.approved_numerical_identity=numerical_identity;
        if (change==1) q.up_type=12;
        if (change==2) --b.gate.bytes;
        if (change==3) ++b.output.address;
        if (change==4) b.codes.address=b.output.address;
        if (change==5) q.shape.logical_rows=2;
        if (change==6) selected_stream=nullptr;
        if (change==7) q.enabled=false;
        if (change==8) b.up=b.gate;
        guard.run([&] {reset();});const auto before=all_owners();bool rejected=false;
        try {enqueue_probe(q,b,selected_stream);} catch (const std::invalid_argument&) {rejected=true;}
        expect(rejected,"Q5 metadata/stream no-write rejection missing");
        no_publication();expect(status.read<unsigned>()[0]==0,"Q5 metadata rejection changed status");
        expect(all_owners()==before,"Q5 metadata rejection changed an actual owner byte");
    }
    // Shared Q4-only staging must reject the newly valid Q5 contract before work.
    guard.run([&] {reset();});const auto before_shared=all_owners();bool shared_rejected=false;
    try {enqueue_probe_shared_gate_up(request,binding,stream.value);} catch (const std::invalid_argument&) {shared_rejected=true;}
    expect(shared_rejected,"Q5 accidentally admitted to Q4 shared staging");no_publication();
    expect(all_owners()==before_shared,"Q5 rejected shared path changed an actual owner byte");
    // Eager scope with graph-only data and a foreign stream both reject before work.
    for (unsigned mode=0;mode<3;++mode) {
        guard.run([&] {reset();});strix::HipLaunchContext context;context.stream=stream.value;
        if (mode==0) context.status.arena_base=16;
        if (mode==1) context.arguments.memory.arena_base=16;
        Stream foreign(guard);
        if (mode==2) context.stream=foreign.value;
        const auto before=all_owners();bool rejected=false;
        {strix::HipLaunchScope scope(context);
            try {enqueue_probe(request,binding,stream.value);} catch (const std::invalid_argument&) {rejected=true;}}
        expect(rejected,"Q5 graph/foreign context guard missing");no_publication();
        expect(all_owners()==before,"Q5 rejected launch context changed an actual owner byte");
    }
    hipGraph_t graph=nullptr;
    guard.run([&] {reset();});const auto before_capture=all_owners();
    guard.run([&] {
        check(hipStreamBeginCapture(stream.value,hipStreamCaptureModeGlobal));bool rejected=false;
        try {enqueue_probe(request,binding,stream.value);} catch (const std::invalid_argument&) {rejected=true;}
        check(hipStreamEndCapture(stream.value,&graph));expect(rejected,"Q5 capture must reject before enqueue");
    });
    std::size_t nodes=99;check(hipGraphGetNodes(graph,nullptr,&nodes));
    check(hipGraphDestroy(graph));expect(nodes==0,"Q5 rejected capture recorded nodes");no_publication();
    expect(all_owners()==before_capture,"Q5 rejected capture changed an actual owner byte");
    // Nonfinite projection headers are deliberately UNQUALIFIED propagation,
    // not an activation/status error or a finite-equality loophole.
    const auto finite_image=q4k_a8_oracle::quantize(activation);
    for (unsigned header:{0x7c00U,0x7e01U,0x7c01U,0xfc00U}) {
        const auto offset=(static_cast<std::size_t>(rt[0])*rows+rows-1U)*row_bytes;
        const auto old0=uw[offset],old1=uw[offset+1U];
        uw[offset]=static_cast<std::byte>(header&255U);uw[offset+1U]=static_cast<std::byte>(header>>8U);
        guard.run([&] {upload();reset();enqueue_probe(request,binding,stream.value);});
        expect(status.read<unsigned>()[0]==0,"Q5 projection header must not invent activation error status");
        const auto values=out.read<float>();
        for (unsigned slot=0;slot<top_k;++slot) for (unsigned which=0;which<2;++which) for (unsigned row=0;row<rows;++row) {
            const float actual=values[(slot*2U+which)*rows+row];
            if (which==1 && row==rows-1U && rt[slot]==rt[0]) expect(!std::isfinite(actual),"Q5 nonfinite header failed to propagate");
            else {
                const auto& weights=which==0?gw:uw;
                const auto raw=std::span<const std::byte>(weights).subspan((static_cast<std::size_t>(rt[slot])*rows+row)*row_bytes,row_bytes);
                expect(std::bit_cast<unsigned>(actual)==std::bit_cast<unsigned>(q5k_a8_oracle::row(raw,finite_image)),
                    "Q5 nonfinite header contaminated an independent finite row/projection");
            }
        }
        uw[offset]=old0;uw[offset+1U]=old1;
        for (const auto* buffer:{&codes,&deltas,&sums,&out,&status}) buffer->guards();
    }
}
}
int main() try {
    strix::HipSetupQuiescenceGuard guard;expect(guard.architecture().starts_with("gfx1151"),"Q5 native probe requires gfx1151");
    require_f32_environment();
    for (bool bf16:{false,true}) for (unsigned hidden:{256U,512U,2560U})
        for (unsigned rows:{32U,64U,640U}) for (unsigned top_k:{1U,2U,10U})
            fixture(hidden,rows,top_k,bf16,0,false,guard);
    for (bool bf16:{false,true}) for (unsigned special:{1U,2U,3U}) fixture(512,64,2,bf16,special,false,guard);
    fixture(512,64,2,false,0,true,guard);fixture(512,64,2,true,0,true,guard);
    std::cout<<"Q5K A8 native original-byte FP32 oracle/status/guards/capture PASS; not model quality or performance qualification\n";
} catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
