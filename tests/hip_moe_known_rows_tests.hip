#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_plan.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/moe_plan.hpp"
#include "q51_grouped_down_oracle.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <exception>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix;
void require(bool yes,const char* why) { if(!yes) throw std::runtime_error(why); }
void check(hipError_t e,const char* why) { if(e!=hipSuccess) throw std::runtime_error(std::string(why)+": "+hipGetErrorString(e)); }
struct View { std::uintptr_t address; std::uint64_t bytes; };
std::vector<std::byte> read(View v) {
    std::vector<std::byte> out(static_cast<std::size_t>(v.bytes));
    check(hipMemcpy(out.data(),reinterpret_cast<const void*>(v.address),out.size(),hipMemcpyDeviceToHost),"read known routed view"); return out;
}
void upload(View v,const void* data,std::size_t bytes) {
    require(bytes<=v.bytes,"known routed upload exceeds view");
    check(hipMemcpy(reinterpret_cast<void*>(v.address),data,bytes,hipMemcpyHostToDevice),"upload known routed view");
}
struct Owner {
    HipSetupQuiescenceGuard guard;
    hipStream_t stream=nullptr;
    std::vector<void*> allocations;
    Owner() { guard.run([&] {check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create known routed stream");}); }
    ~Owner() {
        guard.recover_or_fail_stop();
        for(auto* p:allocations) if(hipFree(p)!=hipSuccess) std::_Exit(3);
        if(hipStreamDestroy(stream)!=hipSuccess) std::_Exit(3);
    }
    View allocate(std::uint64_t bytes) {
        require(bytes && bytes < (1ULL<<31U),"known routed fixture owner exceeds bound");
        void* p=nullptr; check(hipMalloc(&p,static_cast<std::size_t>(bytes+512U)),"allocate guarded known routed owner");
        try { allocations.push_back(p); } catch(...) { guard.recover_or_fail_stop(); (void)hipFree(p); throw; }
        check(hipMemset(p,0xa5,static_cast<std::size_t>(bytes+512U)),"poison known routed owner");
        return {reinterpret_cast<std::uintptr_t>(p)+256U,bytes};
    }
    void drain() {check(hipStreamSynchronize(stream),"drain known routed stream");guard.require_original();}
};
void guards(View v) {
    for(auto p:{v.address-256U,v.address+v.bytes}) {
        const auto bytes=read({p,256U});
        require(std::all_of(bytes.begin(),bytes.end(),[](auto b){return b==std::byte{0xa5};}),"known routed outer guard changed");
    }
}
// Independent packed patterns vary expert/output/block and gate/up. Small
// finite scales keep the complete prefix within the incumbent finite domain.
// Q5_1 (7) and Q8_0 (8) carry 32 features per block, Q4_K (12) and Q5_K (13) 256.
std::vector<std::byte> packed(unsigned width,unsigned outputs,unsigned experts,unsigned type,unsigned salt) {
    const auto block_elements=type==7U||type==8U?32U:256U;
    const auto block_bytes=type==7U?24U:type==8U?34U:type==12U?144U:176U;
    std::vector<std::byte> out(static_cast<std::size_t>(width/block_elements)*outputs*experts*block_bytes);
    const auto set16=[](std::byte* p,unsigned x) {p[0]=std::byte(x&255U);p[1]=std::byte(x>>8U);};
    for(std::size_t b=0;b<out.size()/block_bytes;++b) {
        auto* p=out.data()+b*block_bytes;
        set16(p,0x1400U+static_cast<unsigned>((b+salt)%3U)*0x400U);
        if(type==8U){for(unsigned i=2U;i<block_bytes;++i) p[i]=std::byte((b*37U+i*19U+salt)%256U);continue;}
        set16(p+2,type==7U?0x8800U:0x1000U);
        if(type!=7U) for(unsigned i=0;i<12U;++i) p[4U+i]=std::byte((b*17U+i*23U+salt)%256U);
        else for(unsigned i=0;i<4U;++i) p[4U+i]=std::byte((b*13U+i*29U+salt)%256U);
        const auto start=type==7U?8U:16U;
        for(unsigned i=start;i<block_bytes;++i) p[i]=std::byte((b*37U+i*19U+salt)%256U);
    }
    return out;
}
// I640 exercises all three incumbent 256-feature chunks and the final
// partial chunk. E96 also permits an eight-row, all-distinct route union.
// The benchmark instantiates the actual H2560/E512 model geometry.
// Serial arithmetic profile: float Q5_1 slot-parallel down, or the A8 profile
// (Q5_1/Q8_0 A8G32 down, Q4_K or Q5_K A8G32 gate/up; canonical geometry only).
struct Profile { bool a8=false; unsigned gate=12U, down=7U, layer=0U; };
template<unsigned H,unsigned E> struct Fixture {
    static constexpr unsigned I=640,K=10;
    std::shared_ptr<Owner> owner=std::make_shared<Owner>();
    ModelContract model=ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution;
    SerialExecutionIoPlan io;
    ExecutionStep step;
    KnownWindowArena window;
    View scratch{},csr{},record{};
    View combine_buffer{};
    ExecutionMemoryBinding memory;
    q51_known_rows::Workspace workspace;
    ModelManifest manifest;
    ModelRuntimeBinding weights;
    ExecutionTensorAccessPlan access;
    HipMoeExpertsLauncherConfig config;
    ExecutionLauncher incumbent;
    std::string id;
    std::vector<View> weight_views;
    std::vector<std::vector<std::byte>> packed_weights;
    std::vector<ExecutionRuntimeResources> resources;
    std::vector<ExecutionDispatchContext> rows;
    std::vector<ExecutionBufferAddress> hidden,routes,mix,output,temp;
    std::vector<std::byte> seed;
    std::vector<std::uint32_t> selected;

    Profile profile;
    explicit Fixture(unsigned count,bool batched,Profile selected_profile={}):profile(selected_profile) {
        require(owner->guard.architecture()=="gfx1151","known routed fixture requires actual gfx1151");
        model.hidden_size=H;model.moe_intermediate_size=I;model.num_experts=E;model.num_experts_per_token=K;
        execution=build_serial_decode_execution_plan(model);io=build_serial_execution_io_plan(execution);
        const auto found=std::find_if(execution.steps.begin(),execution.steps.end(),[&](const auto& s){return s.op==SerialOp::moe_experts&&s.layer==profile.layer;});
        require(found!=execution.steps.end(),"known routed step missing");step=*found;
        const auto& uses=io.steps[step.id];
        ExecutionArenaPlan serial;serial.alignment=256U;
        const auto alloc=[&](ExecutionBufferUse use,std::uint64_t bytes) {
            const auto offset=(serial.total_bytes+255U)/256U*256U;
            serial.allocations.push_back({use,step.id,step.id,offset,bytes});serial.total_bytes=offset+bytes;
        };
        const auto layout=moe_q4k_a8::build_runtime_workspace(H,I,E,K,true,profile.a8);
        alloc(uses.reads[0],H*2U);alloc(uses.reads[1],K*4U);alloc(uses.reads[2],K*4U);
        alloc(uses.writes[0],H*2U);alloc(uses.temporaries[0],layout.total_bytes);
        serial.naive_nonaliased_bytes=serial.total_bytes;window=build_known_window_arena(serial,256U,count);
        scratch=owner->allocate(window.physical.total_bytes);memory={0x4d4f45524f575301ULL,scratch.address,scratch.bytes,31U};
        csr=owner->allocate(q51_known_rows::required_scratch_bytes(count,E,K));
        workspace={{memory.execution_identity,csr.address,csr.bytes,43U},owner};record=owner->allocate(8U);
        manifest.architecture="qwen4_exp";
        const auto prefix="blk."+std::to_string(profile.layer)+".";
        add(prefix+"ffn_gate_exps.weight",profile.gate,{H,I,E},packed(H,I,E,profile.gate,3U));
        add(prefix+"ffn_up_exps.weight",profile.gate,{H,I,E},packed(H,I,E,profile.gate,7U));
        add(prefix+"ffn_down_exps.weight",profile.down,{I,H,E},packed(I,H,E,profile.down,13U));
        manifest.layout_identity=model_manifest_layout_identity(manifest);weights.manifest_layout_identity=manifest.layout_identity;
        weights.model_content_identity.bytes[0]=1U;
        if(profile.a8) weights.model_content_identity.bytes={0x37,0x26,0xd6,0xf0,0xd9,0xba,0x30,0xb9,0xd6,0x55,0xba,0x30,0x35,0x95,0x11,0x89,
            0x41,0xba,0xec,0xa5,0xe6,0x94,0x5f,0xc8,0x78,0x5f,0x93,0x8e,0x0f,0x54,0x55,0xe4};
        access.manifest_layout_identity=manifest.layout_identity;access.plan_identity=71U;
        access.accesses.push_back({step.id,step.op,0U,{0U,1U,2U}});
        config.hidden_size=H;config.intermediate_size=I;config.expert_count=E;config.top_k=K;
        config.q4k_a8g32_gate_up=true;config.grouped_exact_q51_down=true;config.q51_slot_parallel_down=true;
        config.deferred_moe_status=true;config.batched_slot_activation=batched;
        config.a8_down_q51=config.a8_down_q80=config.q5k_a8g32_gate_up=profile.a8;
        config.q4k_a8_quiescence=std::make_shared<HipSetupQuiescenceGuard>();
        config.q4k_a8_stats=std::make_shared<moe_q4k_a8::RuntimeDispatchStats>();
        ExecutionBindingRequest request;request.q4k_a8g32_gate_up=true;request.grouped_exact_q51_down=true;
        request.q51_slot_parallel_down=true;request.deferred_moe_status=true;request.batched_moe_activation=batched;
        request.a8_down_q51=request.a8_down_q80=request.q5k_a8g32_gate_up=profile.a8;
        kernels::KernelRegistry registry;register_hip_moe_experts_kernel_variant(model,request,config,registry);
        id=registry.variants().front().id;incumbent=make_hip_moe_experts_launcher(id,&config);
        rows.resize(count);resources.resize(count);hidden.resize(count);routes.resize(count);mix.resize(count);output.resize(count);temp.resize(count);
        for(unsigned r=0;r<count;++r) {
            resources[r]={&memory,&manifest,&weights,&access};auto& c=rows[r];
            c.execution_identity=memory.execution_identity;c.step_id=step.id;c.op=step.op;c.layer=step.layer;c.kernel_id=id;
            c.reads=uses.reads;c.writes=uses.writes;c.temporaries=uses.temporaries;c.arena=&window.row_view[r];c.memory=&memory;c.resources=&resources[r];c.position=29U+r;
            hidden[r]=resolve_dispatch_read(c,0);routes[r]=resolve_dispatch_read(c,1);mix[r]=resolve_dispatch_read(c,2);
            output[r]=resolve_dispatch_write(c,0);temp[r]=resolve_dispatch_temporary(c,0);
        }
    }
    void add(std::string name,unsigned type,std::vector<std::uint64_t> dims,std::vector<std::byte> data) {
        ManifestTensor t;t.name=std::move(name);t.ggml_type=type;t.dimensions=std::move(dims);t.byte_size=data.size();
        t.absolute_offset=manifest.total_tensor_bytes;t.layer=profile.layer;t.role=TensorRole::moe_expert;
        manifest.total_tensor_bytes+=t.byte_size;manifest.tensors.push_back(std::move(t));
        const auto v=owner->allocate(data.size());upload(v,data.data(),data.size());
        weights.tensors.push_back({static_cast<std::uint32_t>(weight_views.size()),RuntimeTensorStorage::resident,v.address,v.bytes,37U,0U});
        weight_views.push_back(v);packed_weights.push_back(std::move(data));
    }
    HipLaunchContext scope() const {
        HipLaunchContext s;s.stream=owner->stream;s.deferred_moe_status={memory.execution_identity,record.address,8U,41U};return s;
    }
    void prepare(unsigned mode) {
        seed.assign(static_cast<std::size_t>(scratch.bytes),std::byte{0xa5});selected.resize(rows.size()*K);
        for(unsigned r=0;r<rows.size();++r) {
            std::vector<std::uint16_t> x(H);
            for(unsigned i=0;i<H;++i) {
                const float value=static_cast<float>(static_cast<int>((i*31U+r*17U)%127U)-63)/64.0F;
                auto b=std::bit_cast<std::uint32_t>(value);b+=0x7fffU+((b>>16U)&1U);x[i]=static_cast<std::uint16_t>(b>>16U);
            }
            std::array<std::uint32_t,K> ids{};std::array<float,K> w{};
            for(unsigned s=0;s<K;++s) {
                ids[s]=mode==0U?s:(mode==1U?(s+r*3U)%E:r*K+s);
                if(mode==3U) ids[s]=(K-1U-s+r*3U)%E;
                if(mode==4U) ids[s]=(r+s/3U)%4U;
                if(mode==5U) ids[s]=(s<5U?7U*s+3U:40U+r*5U+s)%E; // five shared, five private experts per row
                selected[r*K+s]=ids[s];w[s]=(s%2U?-1.0F:1.0F)*static_cast<float>(s+1U)/64.0F;
            }
            std::memcpy(seed.data()+hidden[r].address-scratch.address,x.data(),x.size()*2U);
            std::memcpy(seed.data()+routes[r].address-scratch.address,ids.data(),K*4U);
            std::memcpy(seed.data()+mix[r].address-scratch.address,w.data(),K*4U);
        }
    }
    void reset() {
        upload(scratch,seed.data(),seed.size());check(hipMemset(reinterpret_cast<void*>(csr.address),0xa5,static_cast<std::size_t>(csr.bytes)),"poison CSR payload");
        check(hipMemset(reinterpret_cast<void*>(record.address),0,8U),"clear known routed D1");
    }
    std::vector<std::byte> run(bool candidate) {
        reset();const auto s=scope();
        {HipLaunchScope active(s);
         if(candidate) {moe_experts_known_window_rows_preflight(rows,config,workspace);moe_experts_known_window_rows_launch(rows,config,workspace);}
         else for(const auto& c:rows) incumbent.function(c,incumbent.user_data);}
        owner->drain();const auto error=read(record);require(std::all_of(error.begin(),error.end(),[](auto b){return b==std::byte{0};}),"valid known routed D1 nonzero");
        return read(scratch);
    }
    // The routed step with the MoE combine folded into its commit: the arena equals the routed launch and each
    // combined row is BF16(shared + routed output) with the combine kernel's rounding.
    void check_commit_combine(const std::vector<std::byte>& expected) {
        const auto n=static_cast<unsigned>(rows.size());
        if(!combine_buffer.bytes) combine_buffer=owner->allocate(2ULL*n*H*2U);
        std::vector<std::uint16_t> shared(static_cast<std::size_t>(n)*H);
        for(std::size_t i=0;i<shared.size();++i) {
            const float value=static_cast<float>(static_cast<int>((i*13U)%201U)-100)/128.0F;
            auto b=std::bit_cast<std::uint32_t>(value);b+=0x7fffU+((b>>16U)&1U);shared[i]=static_cast<std::uint16_t>(b>>16U);
        }
        reset();upload(combine_buffer,shared.data(),shared.size()*2U);
        check(hipMemset(reinterpret_cast<void*>(combine_buffer.address+shared.size()*2U),0xa5,shared.size()*2U),"poison combined rows");
        std::vector<const void*> shared_rows(n),routed_rows(n);std::vector<void*> combined_rows(n);
        for(unsigned r=0;r<n;++r) {
            shared_rows[r]=reinterpret_cast<const void*>(combine_buffer.address+static_cast<std::uint64_t>(r)*H*2U);
            combined_rows[r]=reinterpret_cast<void*>(combine_buffer.address+static_cast<std::uint64_t>(n+r)*H*2U);
            routed_rows[r]=reinterpret_cast<const void*>(output[r].address);
        }
        bool joined=false;const auto s=scope();
        {HipLaunchScope active(s);
         moe_experts_known_window_rows_preflight(rows,config,workspace);
         require(moe_experts_known_window_rows_launch_combined(rows,config,workspace,shared_rows,routed_rows,combined_rows,[&]{joined=true;}),
             "commit + combine fold declined disjoint rows");}
        owner->drain();require(joined,"commit + combine fold did not run its pre-commit hook");
        const auto error=read(record);require(std::all_of(error.begin(),error.end(),[](auto b){return b==std::byte{0};}),"valid known routed D1 nonzero");
        require(read(scratch)==expected,"commit + combine fold changed the routed arena");
        const auto all=read(combine_buffer);
        for(unsigned r=0;r<n;++r)for(unsigned i=0;i<H;++i) {
            std::uint16_t routed=0,got=0;
            std::memcpy(&routed,expected.data()+(output[r].address-scratch.address)+2U*i,2U);
            std::memcpy(&got,all.data()+(static_cast<std::size_t>(n+r)*H+i)*2U,2U);
            const float sum=std::bit_cast<float>(static_cast<std::uint32_t>(shared[static_cast<std::size_t>(r)*H+i])<<16U)+
                std::bit_cast<float>(static_cast<std::uint32_t>(routed)<<16U);
            auto b=std::bit_cast<std::uint32_t>(sum);b+=0x7fffU+((b>>16U)&1U);
            require(got==static_cast<std::uint16_t>(b>>16U),"commit + combine row differs from BF16(shared + routed)");
        }
    }
    void check_csr() const {
        // The host planner forbids duplicates within one routing row, while
        // the incumbent native kernel admits them. Treat each assignment as
        // one planner row, then restore the original row/slot coordinates.
        const auto plan=build_moe_route_plan({static_cast<std::uint32_t>(selected.size()),1U,E,0U,selected});
        const auto l=q51_known_rows::scratch_layout(static_cast<std::uint32_t>(rows.size()),E,K);
        const auto bytes=read(csr);std::vector<bool> written(bytes.size(),false);
        const auto word=[&](std::uint64_t offset,std::uint32_t expected) {
            std::uint32_t actual=0;std::memcpy(&actual,bytes.data()+offset,4U);require(actual==expected,"device CSR differs from existing host planner");
            for(unsigned b=0;b<4U;++b)written[static_cast<std::size_t>(offset+b)]=true;
        };
        word(0,0U);word(4,plan.distinct_experts);
        std::vector<unsigned> batch_experts,batch_begins;
        for(unsigned e=0;e<E;++e)for(auto a=plan.expert_offsets[e];a<plan.expert_offsets[e+1U];a+=rows.size()) {
            batch_experts.push_back(e);batch_begins.push_back(static_cast<unsigned>(a));
        }
        batch_begins.push_back(static_cast<unsigned>(selected.size()));word(8,static_cast<unsigned>(batch_experts.size()));
        std::array<unsigned,8> row_hist{},batch_hist{};unsigned duplicates=0U,maximum=0U;
        for(unsigned e=0;e<E;++e) {
            unsigned unique=0U,count=0U;
            for(unsigned r=0;r<rows.size();++r) {bool found=false;
                for(unsigned s=0;s<K;++s)if(selected[r*K+s]==e){found=true;++count;}
                if(found)++unique;
            }
            if(unique){++row_hist[unique-1U];duplicates+=count-unique;maximum=std::max(maximum,count);}
        }
        for(unsigned b=0;b<batch_experts.size();++b)++batch_hist[batch_begins[b+1U]-batch_begins[b]-1U];
        word(12,static_cast<unsigned>(selected.size()));word(16,duplicates);word(20,maximum);
        word(24,static_cast<unsigned>(rows.size()));word(28,1U);
        for(unsigned n=0;n<8U;++n){word(32U+n*4U,row_hist[n]);word(64U+n*4U,batch_hist[n]);word(96U+n*4U,0U);}
        for(unsigned e=0;e<=E;++e)word(l.offsets+e*4U,static_cast<std::uint32_t>(plan.expert_offsets[e]));
        for(unsigned t=0;t<plan.touched_experts.size();++t)word(l.touched+t*4U,plan.touched_experts[t]);
        for(unsigned a=0;a<plan.assignments;++a) {word(l.rows+a*4U,plan.assignment_rows[a]/K);word(l.slots+a*4U,plan.assignment_rows[a]%K);}
        for(unsigned b=0;b<batch_experts.size();++b)word(l.batch_experts+b*4U,batch_experts[b]);
        for(unsigned b=0;b<batch_begins.size();++b)word(l.batch_begins+b*4U,batch_begins[b]);
        for(std::size_t b=0;b<bytes.size();++b)if(!written[b])require(bytes[b]==std::byte{0xa5},"CSR padding or unused assignment changed");
    }
    void check_raw_oracle() const {
        const auto layout=moe_q4k_a8::build_runtime_workspace(H,I,E,K,true);
        const auto down_row_bytes=I/32U*24U;
        for(unsigned r=0;r<rows.size();++r) {
            const auto bytes=read({temp[r].address,temp[r].bytes});
            for(unsigned slot=0;slot<K;++slot) {
                std::array<float,I> intermediate{};
                std::memcpy(intermediate.data(),bytes.data()+layout.gate_up_offset+slot*2U*I*4U,I*4U);
                for(unsigned out=0;out<H;++out) {
                    const auto weight_offset=(static_cast<std::size_t>(selected[r*K+slot])*H+out)*down_row_bytes;
                    const auto expected=q51_down_test::legacy(intermediate,std::span<const std::byte>(packed_weights[2]).subspan(weight_offset,down_row_bytes));
                    std::uint32_t actual=0;std::memcpy(&actual,bytes.data()+layout.rawdots_offset+(slot*H+out)*4U,4U);
                    require(actual==std::bit_cast<std::uint32_t>(expected),"CSR raw dot differs from independent original256 oracle");
                }
            }
        }
    }
    void reject_metadata(unsigned fault,bool capture) {
        reset();auto mutated=rows;auto extra=workspace;auto cfg=config;
        auto alternate_memory=memory;auto alternate_resources=resources.back();
        if(fault==0U)extra.memory.arena_bytes-=256U;
        if(fault==1U)extra.memory.execution_identity++;
        if(fault==2U)extra.memory.allocation_generation=0U;
        if(fault==3U)extra.owner.reset();
        if(fault==4U)extra.memory.arena_base=scratch.address;
        if(fault==5U)mutated.back().position++;
        if(fault==6U) {alternate_resources.scratch=&alternate_memory;mutated.back().resources=&alternate_resources;}
        if(fault==7U)cfg.q51_slot_parallel_down=false;
        if(fault==8U)mutated.back().kernel_id="forged";
        if(fault==9U)mutated.back().prepared_arena_views=reinterpret_cast<const PreparedArenaViews*>(std::uintptr_t{1});
        const auto before=read(scratch),before_csr=read(csr),before_d1=read(record);
        if(capture)check(hipStreamBeginCapture(owner->stream,hipStreamCaptureModeThreadLocal),"begin rejected known routed capture");
        const char* expected=fault<4U?"known-window routed rows CSR owner identity/extent differs":
            (fault==4U?"known-window routed rows writable owner aliases storage":
             (fault==7U?"known-window routed rows require BF16 A8 slot-parallel Q5_1 D1 configuration":
                         "known-window routed rows source/position/owner differs"));
        if(capture&&fault!=7U)expected="slot-parallel Q5_1 down admits eager execution only";
        std::exception_ptr unexpected;
        bool rejected=false;{const auto s=scope();HipLaunchScope active(s);
            try {moe_experts_known_window_rows_launch(mutated,cfg,extra);}catch(const std::invalid_argument& error){rejected=std::string(error.what())==expected;}
            catch(...){unexpected=std::current_exception();}}
        if(capture) {hipGraph_t graph=nullptr;check(hipStreamEndCapture(owner->stream,&graph),"end rejected known routed capture");
            std::size_t nodes=0;check(hipGraphGetNodes(graph,nullptr,&nodes),"count rejected known routed capture nodes");
            require(nodes==0U,"rejected known routed callback enqueued capture work");check(hipGraphDestroy(graph),"destroy rejected known routed graph");}
        if(unexpected)std::rethrow_exception(unexpected);
        require(rejected,"malformed known routed metadata was admitted");
        owner->drain();require(before==read(scratch)&&before_csr==read(csr)&&before_d1==read(record),"host rejection mutated a payload");
    }
    // mode 0: NaN input row 0 + invalid route in the last row; 1: invalid route
    // only; 2: NaN input only (the CSR is built, the quantizer fails).
    void reject_device(unsigned mode) {
        prepare(1U);
        // If a late bad route aborts the shared producer, stale NaN projection
        // scratch must not create an artificial earlier-row pipeline error.
        const auto l=moe_q4k_a8::build_runtime_workspace(H,I,E,K,true);
        const std::uint32_t poison=0x7fc1c0deU;
        for(const auto& t:temp)for(std::uint64_t offset=0;offset<l.gate_up_bytes;offset+=4U)
            std::memcpy(seed.data()+t.address-scratch.address+l.gate_up_offset+offset,&poison,4U);
        const std::uint16_t nan=0x7fc1U;const std::uint32_t invalid=E;
        if(mode!=1U)std::memcpy(seed.data()+hidden.front().address-scratch.address,&nan,2U);
        if(mode!=2U)std::memcpy(seed.data()+routes.back().address-scratch.address,&invalid,4U);
        const std::uint32_t bits=mode==1U?1U:2U,status=mode==0U?3U:bits;
        reset();const auto s=scope();{HipLaunchScope active(s);moe_experts_known_window_rows_launch(rows,config,workspace);}owner->drain();
        for(unsigned r=0;r<rows.size();++r) {
            const auto got=read({output[r].address,output[r].bytes});require(std::all_of(got.begin(),got.end(),[](auto b){return b==std::byte{0xa5};}),"invalid row partially published window output");
            const auto raw=read({temp[r].address+l.rawdots_offset,l.rawdots_bytes});require(std::all_of(raw.begin(),raw.end(),[](auto b){return b==std::byte{0xa5};}),"invalid CSR/prefix changed raw dots");
        }
        DeferredMoeFirstError error{};const auto bytes=read(record);std::memcpy(&error,bytes.data(),sizeof(error));
        require(error.domain==static_cast<std::uint32_t>(DeferredMoeDomain::q4k_a8)&&error.bits==bits,"D1 does not preserve first logical row failure");
        q51_known_rows::Summary summary{};const auto metadata=read(csr);std::memcpy(&summary,metadata.data(),sizeof(summary));
        require(summary.version==1U&&summary.logical_rows==rows.size()&&summary.assignments==rows.size()*K&&
            summary.status==status&&summary.touched==0U&&summary.batches==0U&&
            summary.duplicate_assignments==0U&&summary.maximum_assignments==0U,"failed group summary is not bounded fail-closed metadata");
        for(unsigned n=0;n<8U;++n)require(summary.rows_per_expert_histogram[n]==0U&&summary.batch_size_histogram[n]==0U&&summary.reserved[n]==0U,
            "failed group summary leaked stale histogram/reserved bytes");
    }
    void readonly_and_guards() const {
        guards(scratch);guards(csr);guards(record);
        for(unsigned w=0;w<weight_views.size();++w){guards(weight_views[w]);require(read(weight_views[w])==packed_weights[w],"known routed backend changed weights");}
    }
};
__global__ void spin_kernel(long long ticks) {const auto t=wall_clock64();while(wall_clock64()-t<ticks){}}
__global__ void flush_kernel(const uint4* p,std::size_t count,unsigned* sink) {
    uint4 a{};for(auto i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){const auto v=p[i];a.x^=v.x;a.y^=v.y;a.z^=v.z;a.w^=v.w;}
    if((a.x^a.y^a.z^a.w)==0x9e3779b9U)*sink=1U;
}
// Real-geometry speed: whole-arena equality first, then median GPU time with
// cold weights. A 256 MiB READ evicts L2/MALL without leaving dirty lines (a
// write flush makes the timed reads pay its writebacks); a spin kernel hides
// host enqueue cost.
template<unsigned H,unsigned E> void bench(unsigned count,unsigned mode,Profile profile={}) {
    Fixture<H,E> f(count,true,profile);f.prepare(mode);
    const auto expected=f.run(false);require(f.run(true)==expected,"real-geometry candidate differs from per-row incumbent");f.check_csr();
    void* flush=nullptr;check(hipMalloc(&flush,(256U<<20U)+256U),"allocate flush");
    check(hipMemset(flush,1,(256U<<20U)+256U),"fill flush");
    hipEvent_t a,b;check(hipEventCreate(&a),"event");check(hipEventCreate(&b),"event");
    // Candidate and per-row incumbent alternate so shared-node load affects both.
    std::vector<float> ms[2];
    for(unsigned i=0;i<62U;++i) {
        const bool candidate=i%2U!=0U;
        f.reset();const auto s=f.scope();
        {HipLaunchScope active(s);
         hipLaunchKernelGGL(flush_kernel,dim3(2048),dim3(256),0,f.owner->stream,static_cast<const uint4*>(flush),
             static_cast<std::size_t>(16U<<20U),reinterpret_cast<unsigned*>(static_cast<char*>(flush)+(256U<<20U)));
         hipLaunchKernelGGL(spin_kernel,dim3(1),dim3(1),0,f.owner->stream,200000LL);
         check(hipEventRecord(a,f.owner->stream),"record");
         if(candidate) moe_experts_known_window_rows_launch(f.rows,f.config,f.workspace);
         else for(const auto& c:f.rows) f.incumbent.function(c,f.incumbent.user_data);
         check(hipEventRecord(b,f.owner->stream),"record");}
        f.owner->drain();float t=0;check(hipEventElapsedTime(&t,a,b),"elapsed");if(i>=6U)ms[candidate].push_back(t*1000.0F);
    }
    for(auto& v:ms)std::sort(v.begin(),v.end());
    const auto rows=ms[0][ms[0].size()/2U],candidate=ms[1][ms[1].size()/2U];
    std::set<unsigned> distinct(f.selected.begin(),f.selected.end());
    std::cout<<"bench "<<(profile.a8?"a8":"f32")<<" gate="<<profile.gate<<" down="<<profile.down<<" T="<<count<<" mode="<<mode<<" distinct="<<distinct.size()<<" union_us="<<candidate
             <<" rows_x_1row_us="<<rows<<" ratio="<<candidate/rows<<'\n';
    (void)hipFree(flush);(void)hipEventDestroy(a);(void)hipEventDestroy(b);
}
} // namespace

Profile parse_profile(std::string_view name) {
    if(name=="a8q51")return {true,12U,7U};
    // Canonical A8 census: Q8_0 down on layers 2/4/30/46/47, Q5_K gate/up on layer 2.
    if(name=="a8q80")return {true,12U,8U,4U};
    if(name=="a8q5k")return {true,13U,8U,2U};
    require(name=="f32","profile must be f32/a8q51/a8q80/a8q5k");return {};
}
// A8 profile: canonical-geometry fixture, whole-arena equality with the serial
// A8 launcher per routing pattern, CSR, and the device failure paths.
void a8_fixture(unsigned count,Profile profile) {
    Fixture<2560U,512U> f(count,true,profile);
    for(unsigned mode=0;mode<6U;++mode) {
        f.prepare(mode);const auto expected=f.run(false);
        for(unsigned repeat=0;repeat<2U;++repeat){require(f.run(true)==expected,"A8 union whole arena differs from serial A8 rows");f.check_csr();}
        f.check_commit_combine(expected);
    }
    for(unsigned mode=0;mode<3U;++mode)f.reject_device(mode);
    f.readonly_and_guards();
}
int main(int argc,char** argv) try {
    if(argc>1&&std::string(argv[1])=="bench") {
        const auto profile=parse_profile(argc>4?argv[4]:"f32");
        if(argc>3) {bench<2560U,512U>(static_cast<unsigned>(std::stoul(argv[2])),static_cast<unsigned>(std::stoul(argv[3])),profile);return 0;}
        for(unsigned count:{2U,4U,8U})for(unsigned mode:{5U,1U,2U})bench<2560U,512U>(count,mode,profile);
        return 0;
    }
    if(argc>3&&std::string(argv[3])!="f32") {
        const unsigned rows=static_cast<unsigned>(std::stoul(argv[1]));
        require(rows==2U||rows==4U||rows==8U,"known routed fixture rows must be 2/4/8");
        a8_fixture(rows,parse_profile(argv[3]));
        std::cout<<"known routed rows "<<rows<<' '<<argv[3]<<" A8 fixture completed\n";return 0;
    }
    const unsigned count=argc>1?static_cast<unsigned>(std::stoul(argv[1])):2U;
    require(count==2U||count==4U||count==8U,"known routed fixture rows must be 2/4/8");
    const bool batched=argc>2&&std::string(argv[2])=="1";
    Fixture<256U,96U> f(count,batched);
    for(unsigned mode=0;mode<6U;++mode) {
        f.prepare(mode);const auto expected=f.run(false);
        for(unsigned repeat=0;repeat<3U;++repeat) {require(f.run(true)==expected,"native CSR whole arena differs from per-row incumbent");f.check_csr();if(repeat==0U)f.check_raw_oracle();}
        f.check_commit_combine(expected);
    }
    f.prepare(1U);for(unsigned fault=0;fault<10U;++fault){f.reject_metadata(fault,false);f.reject_metadata(fault,true);}
    for(unsigned mode=0;mode<3U;++mode)f.reject_device(mode);
    f.readonly_and_guards();
    std::cout<<"known routed rows "<<count<<" source fixture completed\n";return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
