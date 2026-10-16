// Test-only same-TU inclusion exposes the unchanged original kernel and the
// candidate without adding a public diagnostic API. Actual app/fixture object
// linkage must be checked independently before device qualification.
#include "../src/hip_gdn_launcher.hip"
#include "strix/hip_setup_quiescence.hpp"
#include "virtual_q8_oracle.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <vector>
namespace {
using virtual_q8_test::Mutation;
constexpr std::array mutations{Mutation::local_first,Mutation::cross_reverse,Mutation::local_reverse,
    Mutation::wrong_mapping,Mutation::descending_leaves,Mutation::negative_zero,
    Mutation::first_product,Mutation::unfused,Mutation::lane_accumulator};
std::array<bool,mutations.size()> detected{};
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void check(hipError_t value) {if(value!=hipSuccess) throw std::runtime_error(hipGetErrorString(value));}
float half_value(std::uint16_t bits) {
    const unsigned e=(bits>>10U)&31U,m=bits&1023U;
    require(e!=31U,"oracle requires finite binary16 scale");
    const float value=e?std::ldexp(static_cast<float>(1024U+m),static_cast<int>(e)-25):std::ldexp(static_cast<float>(m),-24);
    return (bits&0x8000U)?-value:value;
}
std::uint16_t bf16(float value) {
    const auto bits=std::bit_cast<std::uint32_t>(value);return static_cast<std::uint16_t>((bits+0x7fffU+((bits>>16U)&1U))>>16U);
}
void set_scale(strix::Q8Block& block,std::uint16_t bits) {std::memcpy(&block.scale,&bits,2);}
std::uint16_t scale_bits(const strix::Q8Block& block) {std::uint16_t bits=0;std::memcpy(&bits,&block.scale,2);return bits;}
struct Memory {
    strix::HipSetupQuiescenceGuard guard;void* p=nullptr;
    explicit Memory(std::size_t bytes) {guard.run([&]{check(hipMalloc(&p,bytes));});}
    ~Memory() {guard.recover_or_fail_stop();if(p) static_cast<void>(hipFree(p));}
};
struct Stream {
    strix::HipSetupQuiescenceGuard guard;hipStream_t stream=nullptr;
    Stream() {guard.run([&]{check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));});}
    ~Stream() {guard.recover_or_fail_stop();if(stream) static_cast<void>(hipStreamDestroy(stream));}
};
struct Graph {
    hipGraph_t graph=nullptr;hipGraphExec_t executable=nullptr;hipKernelNodeParams params{};
    ~Graph() {if(executable) static_cast<void>(hipGraphExecDestroy(executable));if(graph) static_cast<void>(hipGraphDestroy(graph));}
};
void capture(Graph& graph,Stream& stream,unsigned arm,const float* input,
    const strix::RuntimeTensorBinding& weights,unsigned width,unsigned rows,std::uint16_t* output) {
    strix::HipLaunchContext context{};context.stream=stream.stream;
    const strix::HipLaunchScope scope(context);
    check(hipStreamBeginCapture(stream.stream,hipStreamCaptureModeThreadLocal));
    try {strix::launch_output_by_kind(strix::WeightKind::q8_0,input,weights,width,rows,output,arm!=0U,arm==2U);}
    catch(...) {static_cast<void>(hipStreamEndCapture(stream.stream,&graph.graph));throw;}
    check(hipStreamEndCapture(stream.stream,&graph.graph));
    std::size_t count=0;check(hipGraphGetNodes(graph.graph,nullptr,&count));require(count==1,"projection receipt must contain exactly one kernel");
    hipGraphNode_t node=nullptr;check(hipGraphGetNodes(graph.graph,&node,&count));
    check(hipGraphKernelNodeGetParams(node,&graph.params));
    require(graph.params.blockDim.x==(arm==1U?32U:256U) && graph.params.gridDim.x==(arm==2U?(rows+7U)/8U:rows),
        "kernel geometry must prove original256/candidate32 selection");
    // Deliberately overlaunch rows to exercise both uniform output bounds guards.
    auto extra=graph.params;extra.gridDim.x+=3U;
    const void* raw_weights=reinterpret_cast<const void*>(weights.address);
    void* args[]={&input,&raw_weights,&width,&rows,&output};
    extra.kernelParams=args;extra.extra=nullptr;
    check(hipGraphKernelNodeSetParams(node,&extra));
    check(hipGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
}
void run(unsigned width,unsigned rows,unsigned kind,bool cpu_only=false) {
    require(width%32U==0,"fixture Q8 width must align");
    std::vector<float> input(width);std::vector<strix::Q8Block> weights(static_cast<std::size_t>(rows)*(width/32));
    for(unsigned i=0;i<width;++i) input[i]=std::ldexp(static_cast<float>(static_cast<int>((i*73U+i/17U)%2047U)-1023)/1024.0F,static_cast<int>(i%17U)-8);
    constexpr std::array<std::uint16_t,8> scales{0x0001,0x03ff,0x0400,0x3401,0x3c01,0x4bff,0x8000,0xb801};
    for(unsigned row=0;row<rows;++row) for(unsigned block=0;block<width/32;++block) {
        auto& q=weights[static_cast<std::size_t>(row)*(width/32)+block];set_scale(q,scales[(row*5U+block*3U)%scales.size()]);
        for(unsigned j=0;j<32;++j) q.values[j]=static_cast<std::int8_t>(static_cast<int>((row*47U+block*29U+j*11U)%255U)-127);
    }
    if(kind==1) {
        require(width>=512,"FMA witness needs two terms of one original leaf");
        std::fill(input.begin(),input.end(),0.0F);
        const float x=std::nextafter(1.0F,2.0F),w=half_value(0x3c01)*127.0F;
        volatile float rounded=x*w;input[0]=-rounded;input[256]=x;
        for(auto& q:weights) {set_scale(q,0x3c00);std::fill(std::begin(q.values),std::end(q.values),0);}
        for(unsigned row=0;row<rows;++row) {
            auto* q=weights.data()+static_cast<std::size_t>(row)*(width/32);q[0].values[0]=1;
            set_scale(q[8],0x3c01);q[8].values[0]=127;
        }
    }
    if(kind==2 || kind==3) {
        std::fill(input.begin(),input.end(),kind==2?-std::numeric_limits<float>::denorm_min():-0.0F);
        for(auto& q:weights) {set_scale(q,kind==2?0x3800:0x3c00);std::fill(std::begin(q.values),std::end(q.values),1);}
    }
    if(kind==4) input[0]=std::bit_cast<float>(0x7fc12345U);
    if(kind==5) input[0]=std::bit_cast<float>(0x7fa23456U);
    if(kind==6) input[0]=std::bit_cast<float>(0xffc54321U);
    if(kind==7) input[3]=std::numeric_limits<float>::infinity();
    if(kind==8) {
        for(unsigned i=0;i<width;++i) input[i]=(i%3U)?std::numeric_limits<float>::max():-std::numeric_limits<float>::max();
        for(auto& q:weights) set_scale(q,0x0001);
    }
    if(kind>=9 && kind<=12) {
        std::fill(input.begin(),input.end(),0.0F);
        for(auto& q:weights) {set_scale(q,0x3c00);std::fill(std::begin(q.values),std::end(q.values),0);}
        if(kind==9) { input[0]=8.0F;input[128]=4.0F;input[16]=std::ldexp(1.0F,-15); }
        if(kind==10) std::fill(input.begin(),input.end(),std::ldexp(1.0F,-126));
        if(kind==11) {
            std::fill(input.begin(),input.end(),1.0F);
            std::fill_n(input.begin(),256,-std::ldexp(1.0F,-133));
        }
        if(kind==12) {input[0]=std::ldexp(1.0F,26);input[256]=-std::ldexp(1.0F,26);input[512]=1.0F;}
        for(unsigned row=0;row<rows;++row) {
            auto* q=weights.data()+static_cast<std::size_t>(row)*(width/32);
            if(kind==9) {
                set_scale(q[0],0x7800);q[0].values[0]=64;q[0].values[16]=1;
                set_scale(q[4],0x7800);q[4].values[0]=-128;
            }
            if(kind==10) for(unsigned i=0;i<width/32;++i) {set_scale(q[i],0x0001);std::fill(std::begin(q[i].values),std::end(q[i].values),127);}
            if(kind==11) for(unsigned i=0;i<width/32;++i) {set_scale(q[i],i<8?0x0001:0x8000);std::fill(std::begin(q[i].values),std::end(q[i].values),1);}
            if(kind==12) {q[0].values[0]=1;q[8].values[0]=1;q[16].values[0]=1;}
        }
    }
    if(kind>=13 && kind<=15) {
        const unsigned negative=kind==13?128U:kind==14?4U:128U;
        const unsigned small=kind==13?1U:kind==14?2U:64U;
        std::fill(input.begin(),input.end(),0.0F);
        input[0]=8.0F;input[negative]=4.0F;input[small]=std::ldexp(1.0F,-15);
        for(auto& q:weights) {set_scale(q,0x7800);std::fill(std::begin(q.values),std::end(q.values),0);}
        for(unsigned row=0;row<rows;++row) {
            auto* q=weights.data()+static_cast<std::size_t>(row)*(width/32);
            q[0].values[0]=64;q[negative/32].values[negative%32]=-128;q[small/32].values[small%32]=1;
        }
    }
    if(kind==16) {
        std::fill(input.begin(),input.end(),0.0F);
        input[0]=std::ldexp(1.0F,26);input[1]=-std::ldexp(1.0F,26);input[8]=1.0F;
        for(auto& q:weights) {set_scale(q,0x3c00);std::fill(std::begin(q.values),std::end(q.values),1);}
    }
    // Actual-device comparison covers NaN coefficient decode and operand priority;
    // the finite CPU oracle is deliberately not used for these payload cases.
    if(kind>=17) {
        std::fill(input.begin(),input.end(),1.0F);
        for(auto& q:weights) {set_scale(q,0x3c00);std::fill(std::begin(q.values),std::end(q.values),1);}
        if(kind==17 || kind==18) for(auto& q:weights) set_scale(q,kind==17?0x7e01U:0x7c01U);
        if(kind==19) for(auto& q:weights) {set_scale(q,0x7c00U);std::fill(std::begin(q.values),std::end(q.values),0);}
        if(kind==20) {
            std::fill(input.begin(),input.end(),std::numeric_limits<float>::max());
            for(auto& q:weights) {set_scale(q,0x7bffU);std::fill(std::begin(q.values),std::end(q.values),-128);}
        }
        if(kind==21) {
            input[0]=std::bit_cast<float>(0xffc12345U);
            for(unsigned row=0;row<rows;++row) set_scale(weights[static_cast<std::size_t>(row)*(width/32)],0x7e02U);
        }
        if(kind==22) {input[0]=std::bit_cast<float>(0xffc12345U);input[256]=std::bit_cast<float>(0x7fc54321U);}
        if(kind>=23) {
            const unsigned pair=(kind-23U)/2U;require(pair<8U,"invalid NaN tree witness");
            const unsigned stride=128U>>pair;const bool swap=(kind&1U)==0U;
            input[0]=std::bit_cast<float>(swap?0x7fc54321U:0xffc12345U);
            input[stride]=std::bit_cast<float>(swap?0xffc12345U:0x7fc54321U);
        }
    }
    const bool finite_oracle=kind<4 || (kind>=8 && kind<=16);
    std::vector<std::uint16_t> expected(rows);
    if(finite_oracle) for(unsigned row=0;row<rows;++row) {
        std::vector<float> decoded(width);
        for(unsigned i=0;i<width;++i) {const auto& q=weights[static_cast<std::size_t>(row)*(width/32)+i/32];decoded[i]=half_value(scale_bits(q))*static_cast<float>(q.values[i%32]);}
        const float value=virtual_q8_test::legacy_oracle(input,decoded);expected[row]=bf16(value);
        for(std::size_t m=0;rows<64U && m<mutations.size();++m)
            detected[m]=detected[m] || expected[row]!=bf16(virtual_q8_test::legacy_oracle(input,decoded,mutations[m]));
        if(kind==16) require(expected[row]==0 && bf16(virtual_q8_test::legacy_oracle(input,decoded,Mutation::lane_accumulator))==0x3f80U,
            "contiguous-eight one-accumulator mutant must differ after BF16");
        if(kind==10) {
            std::vector<float> premature(width),codes(width,127.0F);
            for(unsigned i=0;i<width;++i) {volatile float product=input[i]*half_value(0x0001);premature[i]=product;}
            require(expected[row]==0x0002U && bf16(virtual_q8_test::legacy_oracle(premature,codes))==0,
                "decode-first witness must reject reassociation and expose subnormal FMA results after BF16");
        }
        if(kind==11) {
            auto wrong=decoded;std::fill(wrong.begin()+256,wrong.end(),0.0F);
            require(expected[row]==0x8000U && bf16(virtual_q8_test::legacy_oracle(input,wrong))==0,
                "negative-zero coefficient witness must reject erased sign after BF16");
        }
        if(kind==1) require(expected[row]!=bf16(virtual_q8_test::legacy_oracle(input,decoded,virtual_q8_test::Mutation::unfused)),
            "FMA witness must distinguish unfused arithmetic AFTER BF16 storage");
        if(kind==2) require(expected[row]==0x8000U,"all256 negative underflows must retain negative zero");
        if(kind==3) require(expected[row]!=bf16(virtual_q8_test::legacy_oracle(input,decoded,virtual_q8_test::Mutation::negative_zero)),
            "signed-zero witness must detect changed accumulator initialization AFTER BF16");
    }
    if(cpu_only) return;
    Memory d_input(input.size()*4),d_weights(weights.size()*sizeof(strix::Q8Block)),d_output((rows+16U)*2U);Stream stream;
    d_input.guard.run([&]{check(hipMemcpy(d_input.p,input.data(),input.size()*4,hipMemcpyHostToDevice));check(hipMemcpy(d_weights.p,weights.data(),weights.size()*sizeof(strix::Q8Block),hipMemcpyHostToDevice));});
    strix::RuntimeTensorBinding binding;binding.address=reinterpret_cast<std::uintptr_t>(d_weights.p);binding.bytes=weights.size()*sizeof(strix::Q8Block);
    auto* output=static_cast<std::uint16_t*>(d_output.p)+8;
    Graph old_graph,new_graph,wave_graph;capture(old_graph,stream,0,static_cast<float*>(d_input.p),binding,width,rows,output);
    capture(new_graph,stream,1,static_cast<float*>(d_input.p),binding,width,rows,output);
    capture(wave_graph,stream,2,static_cast<float*>(d_input.p),binding,width,rows,output);
    require(wave_graph.params.func!=new_graph.params.func && old_graph.params.func!=new_graph.params.func,"original and candidate must be distinct actual functions");
    const auto observe=[&](Graph& graph,int poison,bool execute) {
        std::vector<std::uint16_t> values(rows+16);
        d_output.guard.run([&]{check(hipMemsetAsync(d_output.p,poison,values.size()*2,stream.stream));
            if(execute) check(hipGraphLaunch(graph.executable,stream.stream));
            check(hipMemcpyAsync(values.data(),d_output.p,values.size()*2,hipMemcpyDeviceToHost,stream.stream));});
        const auto bits=static_cast<std::uint16_t>(poison*257);
        require(std::all_of(values.begin(),values.begin()+8,[&](auto v){return v==bits;}) &&
            std::all_of(values.end()-8,values.end(),[&](auto v){return v==bits;}),"output overlaunch wrote guards");
        std::vector<std::uint16_t> result(values.begin()+8,values.end()-8);
        require(std::any_of(result.begin(),result.end(),[&](auto v){return v!=bits;})==execute,"no-op poison control failed");return result;
    };
    (void)observe(old_graph,0xa5,false);
    const auto old=observe(old_graph,0xa5,true),candidate=observe(new_graph,0x5a,true);
    if(kind<17) require(old==candidate,"virtual candidate differs from actual original-kernel BF16 receipt");
    const auto exact_wave=[&](const std::vector<std::uint16_t>& actual,const char* phase,unsigned replay) {
        for(std::size_t row=0;row<actual.size();++row) if(actual[row]!=candidate[row]) {
            std::cerr<<"GDN_OUTPUT_BIT_MISMATCH width="<<width<<" rows="<<rows<<" case="<<kind
                <<" phase="<<phase<<" replay="<<replay<<" row="<<row<<" virtual=0x"<<std::hex<<candidate[row]
                <<" wave8=0x"<<actual[row]<<std::dec<<'\n';
            throw std::runtime_error("wave8 output differs from actual virtual-output BF16 payload receipt");
        }
    };
    for(unsigned replay=0;replay<3;++replay) exact_wave(observe(wave_graph,replay%2U?0xa5:0x5a,true),"graph",replay);
    if(finite_oracle) require(old==expected,"actual original kernel differs from independent256-leaf FP32 FMA/tree oracle");
    for(unsigned replay=0;replay<3;++replay) {
        std::vector<std::uint16_t> eager(rows+16U);
        d_output.guard.run([&] {
            check(hipMemsetAsync(d_output.p,0xa5,eager.size()*2U,stream.stream));
            strix::HipLaunchContext context{};context.stream=stream.stream;
            const strix::HipLaunchScope scope(context);
            strix::launch_output_by_kind(strix::WeightKind::q8_0,static_cast<float*>(d_input.p),binding,width,rows,output,true,true);
            check(hipMemcpyAsync(eager.data(),d_output.p,eager.size()*2U,hipMemcpyDeviceToHost,stream.stream));
        });
        require(std::all_of(eager.begin(),eager.begin()+8,[](auto v){return v==0xa5a5U;}) &&
            std::all_of(eager.end()-8,eager.end(),[](auto v){return v==0xa5a5U;}),"eager wave8 output guards");
        exact_wave(std::vector<std::uint16_t>(eager.begin()+8,eager.end()-8),"eager",replay);
    }
    std::vector<float> retained(input.size());std::vector<strix::Q8Block> retained_weights(weights.size());
    d_input.guard.run([&]{check(hipMemcpy(retained.data(),d_input.p,retained.size()*4,hipMemcpyDeviceToHost));check(hipMemcpy(retained_weights.data(),d_weights.p,retained_weights.size()*sizeof(strix::Q8Block),hipMemcpyDeviceToHost));});
    require(std::memcmp(retained.data(),input.data(),input.size()*4)==0 && std::memcmp(retained_weights.data(),weights.data(),weights.size()*sizeof(strix::Q8Block))==0,"borrowed input/weight bytes mutated");
    if(kind>=23) {
        // Report hardware discrimination without making it a success condition.
        // GDN's incumbent for the new output overlay is the virtual kernel.
        static std::array<std::uint16_t,8> virtual_payload{};
        const auto pair=(kind-23U)/2U;
        if((kind-23U)%2U==0) virtual_payload[pair]=candidate.front();
        else std::cout<<"GDN_TREE_ROLE_RECEIPT stride="<<(128U>>pair)
            <<" baseline_virtual_swapped_payload_discriminates="<<(virtual_payload[pair]!=candidate.front())
            <<" baseline_virtual_first_bits="<<virtual_payload[pair]<<" swapped_bits="<<candidate.front()<<'\n';
    }
    for(const auto bad: {strix::WeightKind::f32,strix::WeightKind::q8_0}) {
        check(hipMemset(d_output.p,0xa5,(rows+16U)*2U));bool rejected=false;
        try {strix::launch_output_by_kind(bad,static_cast<float*>(d_input.p),binding,bad==strix::WeightKind::q8_0?33U:width,rows,output,true);}
        catch(const std::invalid_argument& e) {rejected=std::string(e.what())=="virtual Q8 GDN output requires aligned Q8_0 weights";}
        require(rejected,"wrong format/alignment must reject exact entry guard");
        std::vector<std::uint16_t> preserved(rows+16);check(hipMemcpy(preserved.data(),d_output.p,preserved.size()*2,hipMemcpyDeviceToHost));
        require(std::all_of(preserved.begin(),preserved.end(),[](auto v){return v==0xa5a5U;}),"wrong format/alignment mutated output");
    }
    for(unsigned fault=0;fault<9;++fault) {
        auto bad_binding=binding;auto* bad_input=static_cast<float*>(d_input.p);auto* bad_output=output;
        auto bad_width=width;auto bad_rows=rows;bool virtual_enabled=true;
        if(fault==0) virtual_enabled=false;
        if(fault==1) bad_width=0;
        if(fault==2) bad_rows=0;
        if(fault==3) bad_width=10272;
        if(fault==4) bad_binding.bytes-=1;
        if(fault==5) bad_binding.address+=2;
        if(fault==6) bad_input=reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(bad_input)+4U);
        if(fault==7) bad_output+=1;
        if(fault==8) bad_width=33;
        check(hipMemset(d_output.p,0xa5,(rows+16U)*2U));
        strix::HipLaunchContext context{};context.stream=stream.stream;Graph rejected_graph;bool rejected=false;
        {
            const strix::HipLaunchScope scope(context);
            check(hipStreamBeginCapture(stream.stream,hipStreamCaptureModeThreadLocal));
            try {strix::launch_output_by_kind(strix::WeightKind::q8_0,bad_input,bad_binding,bad_width,bad_rows,bad_output,virtual_enabled,true);}
            catch(const std::invalid_argument&) {rejected=true;}
            check(hipStreamEndCapture(stream.stream,&rejected_graph.graph));
        }
        std::size_t nodes=0;check(hipGraphGetNodes(rejected_graph.graph,nullptr,&nodes));
        require(rejected && nodes==0,"wave8 malformed view must reject before enqueue");
        std::vector<std::uint16_t> preserved(rows+16);check(hipMemcpy(preserved.data(),d_output.p,preserved.size()*2,hipMemcpyDeviceToHost));
        require(std::all_of(preserved.begin(),preserved.end(),[](auto v){return v==0xa5a5U;}),"wave8 malformed view mutated output");
    }
    std::cout<<"GDN_OUTPUT_RECEIPT width="<<width<<" rows="<<rows<<" case="<<kind<<" original_threads=256 virtual_threads=32 wave8_threads=256 outputs_per_cta=8 virtual_wave8_exact_bf16=true cpu_finite_oracle="<<finite_oracle<<'\n';
}
}
int main() try {
    // Host registration receipts cover output suffix independence from each
    // original input mode, including the existing prefetch suffix parser.
    for(unsigned input_mode=0;input_mode<3;++input_mode) {
        strix::HipGdnLauncherConfig baseline;baseline.virtual_q8_input_projections=input_mode!=0;
        baseline.prefetch_q8_input_projections=input_mode==2;
        auto candidate=baseline;candidate.virtual_q8_output_projection=true;
        strix::kernels::KernelRegistry old_registry,new_registry;strix::ExecutionBindingRequest request;
        strix::register_hip_gdn_kernel_variant(baseline.model,request,baseline,old_registry);
        strix::register_hip_gdn_kernel_variant(candidate.model,request,candidate,new_registry);
        const auto old = old_registry.variants()[0];const auto changed = new_registry.variants()[0];
        require(old.id!=changed.id && old.compiler_identity!=changed.compiler_identity,"output compiler/kernel identities must be distinct");
        const auto old_launcher=strix::make_hip_gdn_launcher(old.id,&baseline),new_launcher=strix::make_hip_gdn_launcher(changed.id,&candidate);
        require(old_launcher.implementation_identity!=new_launcher.implementation_identity,"output launcher identity must differ");
        strix::CompiledSerialExecution compiled;compiled.execution_identity=1;
        strix::ExecutionStep step;step.id=1;step.op=strix::SerialOp::gated_delta_net;compiled.execution.steps.push_back(step);
        strix::ExecutionKernelBinding binding;binding.step_id=1;binding.op=step.op;binding.kernel_id=changed.id;compiled.bindings.bindings.push_back(binding);
        strix::HipGdnLauncherBundle bundle(candidate);strix::ExecutionLauncherRegistry launchers;
        bundle.register_for_compiled_execution(compiled,launchers);
        require(launchers.find_kernel_launcher(changed.id)!=nullptr,"positive selected output identity must register");
        compiled.bindings.bindings[0].kernel_id=old.id;strix::ExecutionLauncherRegistry rejected_registry;bool rejected=false;
        try {bundle.register_for_compiled_execution(compiled,rejected_registry);}
        catch(const std::invalid_argument& error) {rejected=std::string(error.what())=="HIP GDN selected kernel disagrees with virtual Q8 output configuration";}
        require(rejected && rejected_registry.find_kernel_launcher(old.id)==nullptr,"wrong selected output identity must reject before registration");
        compiled.bindings.bindings[0].kernel_id=changed.id;
        strix::HipGdnLauncherBundle old_bundle(baseline);strix::ExecutionLauncherRegistry reverse_registry;rejected=false;
        try {old_bundle.register_for_compiled_execution(compiled,reverse_registry);}
        catch(const std::invalid_argument& error) {rejected=std::string(error.what())=="HIP GDN selected kernel disagrees with virtual Q8 output configuration";}
        require(rejected && reverse_registry.find_kernel_launcher(changed.id)==nullptr,"output-off config must reject suffixed identity before registration");
    }
    for(unsigned input_mode=0;input_mode<3;++input_mode) {
        strix::HipGdnLauncherConfig base;base.virtual_q8_output_projection=true;
        base.virtual_q8_input_projections=input_mode!=0;base.prefetch_q8_input_projections=input_mode==2;
        auto wave=base;wave.wave8_q8_output_projection=true;
        strix::ExecutionBindingRequest request;request.wave8_q8_gdn_output=true;
        strix::kernels::KernelRegistry registry;
        strix::register_hip_gdn_kernel_variant(wave.model,request,wave,registry);
        const auto variant=registry.variants().front();
        require(variant.id.find(".wave8-q8-output-v1")!=std::string::npos &&
            variant.id.ends_with(".virtual256.q8-output"),"wave8 output marker preserves terminal selector");
        require(strix::make_hip_gdn_launcher(variant.id,&wave).implementation_identity!=
            strix::make_hip_gdn_launcher("test.base",&base).implementation_identity,"wave8 output hash distinct");
        strix::CompiledSerialExecution compiled;compiled.execution_identity=1;
        strix::ExecutionStep step;step.id=1;step.op=strix::SerialOp::gated_delta_net;compiled.execution.steps.push_back(step);
        strix::ExecutionKernelBinding binding;binding.step_id=1;binding.op=step.op;binding.kernel_id=variant.id;compiled.bindings.bindings.push_back(binding);
        strix::HipGdnLauncherBundle bundle(wave);strix::ExecutionLauncherRegistry launchers;
        bundle.register_for_compiled_execution(compiled,launchers);
        require(launchers.find_kernel_launcher(variant.id)!=nullptr,"wave8 selected identity registers");
        strix::HipGdnLauncherBundle wrong_bundle(base);strix::ExecutionLauncherRegistry wrong_launchers;bool wrong_rejected=false;
        try {wrong_bundle.register_for_compiled_execution(compiled,wrong_launchers);}
        catch(const std::invalid_argument&) {wrong_rejected=true;}
        require(wrong_rejected && wrong_launchers.find_kernel_launcher(variant.id)==nullptr,"output-off rejects selected wave8 id atomically");
        for(unsigned fault=0;fault<3;++fault) {
            auto bad=wave;auto bad_request=request;
            if(fault==0) bad.virtual_q8_output_projection=false;
            if(fault==1) bad_request.wave8_q8_gdn_output=false;
            if(fault==2) bad_request.require_graph_safe=true;
            bool rejected=false;strix::kernels::KernelRegistry invalid;
            try {strix::register_hip_gdn_kernel_variant(bad.model,bad_request,bad,invalid);}
            catch(const std::invalid_argument&) {rejected=true;}
            require(rejected && invalid.variants().empty(),"wave8 output invalid configuration registers nothing");
        }
    }
    for(const unsigned width:{32U,96U,256U,288U,640U,6144U,10240U}) run(width,17,0);
    run(512,3,1);run(256,3,2);run(256,3,3);run(640,7,8);
    run(640,3,9);run(2560,3,10);run(2560,3,11);run(768,3,12);run(640,3,13);run(640,3,14);run(640,3,15);
    run(256,3,16);
    for(const unsigned rows:{1U,4U,7U,8U,9U,2560U,2563U}) run(6144,rows,0);
    for(unsigned kind=17;kind<=38;++kind) run(640,9,kind);
    for(std::size_t m=0;m<detected.size();++m) {
        if(!detected[m]) throw std::runtime_error("fixture BF16-invisible mutant "+std::to_string(m));
    }
    for(unsigned mode=4;mode<=7;++mode) run(288,7,mode);
    std::cout<<"GDN original/virtual/wave8 output projection fixtures passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
