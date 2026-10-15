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
void capture(Graph& graph,Stream& stream,bool candidate,const float* input,
    const strix::RuntimeTensorBinding& weights,unsigned width,unsigned rows,std::uint16_t* output) {
    strix::HipLaunchContext context{};context.stream=stream.stream;
    const strix::HipLaunchScope scope(context);
    check(hipStreamBeginCapture(stream.stream,hipStreamCaptureModeThreadLocal));
    try {strix::launch_output_by_kind(strix::WeightKind::q8_0,input,weights,width,rows,output,candidate);}
    catch(...) {static_cast<void>(hipStreamEndCapture(stream.stream,&graph.graph));throw;}
    check(hipStreamEndCapture(stream.stream,&graph.graph));
    std::size_t count=0;check(hipGraphGetNodes(graph.graph,nullptr,&count));require(count==1,"projection receipt must contain exactly one kernel");
    hipGraphNode_t node=nullptr;check(hipGraphGetNodes(graph.graph,&node,&count));
    check(hipGraphKernelNodeGetParams(node,&graph.params));
    require(graph.params.blockDim.x==(candidate?32U:256U) && graph.params.gridDim.x==rows,
        "kernel geometry must prove original256/candidate32 selection");
    // Deliberately overlaunch rows to exercise both uniform output bounds guards.
    auto extra=graph.params;extra.gridDim.x+=3U;check(hipGraphKernelNodeSetParams(node,&extra));
    check(hipGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
}
void run(unsigned width,unsigned rows,unsigned kind) {
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
    const bool finite_oracle=kind<4 || kind==8;
    std::vector<std::uint16_t> expected(rows);
    if(finite_oracle) for(unsigned row=0;row<rows;++row) {
        std::vector<float> decoded(width);
        for(unsigned i=0;i<width;++i) {const auto& q=weights[static_cast<std::size_t>(row)*(width/32)+i/32];decoded[i]=half_value(scale_bits(q))*static_cast<float>(q.values[i%32]);}
        const float value=virtual_q8_test::legacy_oracle(input,decoded);expected[row]=bf16(value);
        if(kind==1) require(expected[row]!=bf16(virtual_q8_test::legacy_oracle(input,decoded,virtual_q8_test::Mutation::unfused)),
            "FMA witness must distinguish unfused arithmetic AFTER BF16 storage");
        if(kind==2) require(expected[row]==0x8000U,"all256 negative underflows must retain negative zero");
        if(kind==3) require(expected[row]!=bf16(virtual_q8_test::legacy_oracle(input,decoded,virtual_q8_test::Mutation::negative_zero)),
            "signed-zero witness must detect changed accumulator initialization AFTER BF16");
    }
    Memory d_input(input.size()*4),d_weights(weights.size()*sizeof(strix::Q8Block)),d_output((rows+16U)*2U);Stream stream;
    d_input.guard.run([&]{check(hipMemcpy(d_input.p,input.data(),input.size()*4,hipMemcpyHostToDevice));check(hipMemcpy(d_weights.p,weights.data(),weights.size()*sizeof(strix::Q8Block),hipMemcpyHostToDevice));});
    strix::RuntimeTensorBinding binding;binding.address=reinterpret_cast<std::uintptr_t>(d_weights.p);binding.bytes=weights.size()*sizeof(strix::Q8Block);
    auto* output=static_cast<std::uint16_t*>(d_output.p)+8;
    Graph old_graph,new_graph;capture(old_graph,stream,false,static_cast<float*>(d_input.p),binding,width,rows,output);
    capture(new_graph,stream,true,static_cast<float*>(d_input.p),binding,width,rows,output);
    require(old_graph.params.func!=new_graph.params.func,"original and candidate must be distinct actual functions");
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
    require(old==candidate,"candidate differs from actual original-kernel BF16 receipt");
    if(finite_oracle) require(old==expected,"actual original kernel differs from independent256-leaf FP32 FMA/tree oracle");
    std::vector<float> retained(input.size());std::vector<strix::Q8Block> retained_weights(weights.size());
    d_input.guard.run([&]{check(hipMemcpy(retained.data(),d_input.p,retained.size()*4,hipMemcpyDeviceToHost));check(hipMemcpy(retained_weights.data(),d_weights.p,retained_weights.size()*sizeof(strix::Q8Block),hipMemcpyDeviceToHost));});
    require(std::memcmp(retained.data(),input.data(),input.size()*4)==0 && std::memcmp(retained_weights.data(),weights.data(),weights.size()*sizeof(strix::Q8Block))==0,"borrowed input/weight bytes mutated");
    for(const auto bad: {strix::WeightKind::f32,strix::WeightKind::q8_0}) {
        check(hipMemset(d_output.p,0xa5,(rows+16U)*2U));bool rejected=false;
        try {strix::launch_output_by_kind(bad,static_cast<float*>(d_input.p),binding,bad==strix::WeightKind::q8_0?33U:width,rows,output,true);}
        catch(const std::invalid_argument& e) {rejected=std::string(e.what())=="virtual Q8 GDN output requires aligned Q8_0 weights";}
        require(rejected,"wrong format/alignment must reject exact entry guard");
        std::vector<std::uint16_t> preserved(rows+16);check(hipMemcpy(preserved.data(),d_output.p,preserved.size()*2,hipMemcpyDeviceToHost));
        require(std::all_of(preserved.begin(),preserved.end(),[](auto v){return v==0xa5a5U;}),"wrong format/alignment mutated output");
    }
    std::cout<<"GDN_OUTPUT_RECEIPT width="<<width<<" rows="<<rows<<" case="<<kind<<" original_threads=256 candidate_threads=32 exact_bf16=true cpu_finite_oracle="<<finite_oracle<<'\n';
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
        const auto& old=old_registry.variants()[0];const auto& changed=new_registry.variants()[0];
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
    }
    for(const unsigned width:{32U,96U,256U,288U,640U,6144U}) run(width,17,0);
    run(512,3,1);run(256,3,2);run(256,3,3);run(640,7,8);
    for(unsigned mode=4;mode<=7;++mode) run(288,7,mode);
    std::cout<<"GDN original/virtual output projection fixtures passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
