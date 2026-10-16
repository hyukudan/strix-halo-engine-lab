// Test-only same-TU inclusion exposes the unchanged original kernel and the
// candidate plus a test-only explicit-stream probe. Actual app/fixture object
// linkage must be checked independently before device qualification.
#ifndef STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC
#error "QSA output fixture requires the private diagnostic build definition"
#endif
#include "../src/hip_indexed_attention_launcher.hip"
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
void capture(Graph& graph,Stream& stream,bool candidate,const float* input,
    const strix::RuntimeTensorBinding& weights,unsigned width,unsigned rows,std::uint16_t* output) {
    strix::HipLaunchContext context{};context.stream=stream.stream;
    const strix::HipLaunchScope scope(context);
    check(hipStreamBeginCapture(stream.stream,hipStreamCaptureModeThreadLocal));
    try {strix::launch_hip_qsa_output_projection_probe(candidate?strix::HipQsaOutputProjectionMode::wave8:
            strix::HipQsaOutputProjectionMode::legacy256,
        {reinterpret_cast<std::uintptr_t>(input),static_cast<std::uint64_t>(width)*4U},
        {weights.address,weights.bytes},{reinterpret_cast<std::uintptr_t>(output),static_cast<std::uint64_t>(rows)*2U},
        width,rows,stream.stream);}
    catch(...) {static_cast<void>(hipStreamEndCapture(stream.stream,&graph.graph));throw;}
    check(hipStreamEndCapture(stream.stream,&graph.graph));
    std::size_t count=0;check(hipGraphGetNodes(graph.graph,nullptr,&count));require(count==1,"projection receipt must contain exactly one kernel");
    hipGraphNode_t node=nullptr;check(hipGraphGetNodes(graph.graph,&node,&count));
    check(hipGraphKernelNodeGetParams(node,&graph.params));
    require(graph.params.blockDim.x==256U && graph.params.gridDim.x==(candidate?(rows+7U)/8U:rows),
        "kernel geometry must prove original256 versus eight independent output waves");
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
    if(kind>=16) {
        std::fill(input.begin(),input.end(),0.0F);
        for(auto& q:weights) {set_scale(q,0x3c00);std::fill(std::begin(q.values),std::end(q.values),0);}
        for(unsigned row=0;row<rows;++row) {
            auto* q=weights.data()+static_cast<std::size_t>(row)*(width/32);
            if(kind==16) {
                input[0]=std::ldexp(1.0F,26);input[1]=-std::ldexp(1.0F,26);input[8]=1.0F;
                std::fill(std::begin(q[0].values),std::end(q[0].values),1);
            } else if(kind>=17 && kind<=20) {
                std::fill(input.begin(),input.end(),kind==20?std::numeric_limits<float>::max():1.0F);
                for(unsigned b=0;b<width/32;++b) {
                    set_scale(q[b],kind==17?0x7c00:kind==18?0x7e01:kind==19?0x7c01:0x7bff);
                    std::fill(std::begin(q[b].values),std::end(q[b].values),kind==17?0:kind==20?-128:1);
                    if(kind==20) q[b].values[0]=127;
                }
            } else if(kind==21) {
                input[0]=std::bit_cast<float>(0x7fc12345U);set_scale(q[0],0x7e23);q[0].values[0]=1;
            } else if(kind==22) {
                require(width>256,"two-FMA-role case needs a repeated original leaf");
                input[0]=std::bit_cast<float>(0x7fc12345U);input[256]=std::bit_cast<float>(0xffc54321U);
                q[0].values[0]=1;q[8].values[0]=1;
            } else {
                const unsigned pair=(kind-23U)/2U,stride=128U>>pair;
                require(pair<8,"tree role case outside eight original strides");
                const bool swap=(kind-23U)%2U;
                input[0]=std::bit_cast<float>(swap?0xffc54321U:0x7fc12345U);
                input[stride]=std::bit_cast<float>(swap?0x7fc12345U:0xffc54321U);
                q[0].values[0]=1;q[stride/32U].values[stride%32U]=1;
            }
        }
    }
    const bool finite_oracle=kind<4 || (kind>=8 && kind<=16);
    std::vector<std::uint16_t> expected(rows);
    if(finite_oracle) for(unsigned row=0;row<rows;++row) {
        std::vector<float> decoded(width);
        for(unsigned i=0;i<width;++i) {const auto& q=weights[static_cast<std::size_t>(row)*(width/32)+i/32];decoded[i]=half_value(scale_bits(q))*static_cast<float>(q.values[i%32]);}
        const float value=virtual_q8_test::legacy_oracle(input,decoded);expected[row]=bf16(value);
        for(std::size_t m=0;m<mutations.size();++m)
            detected[m]=detected[m] || expected[row]!=bf16(virtual_q8_test::legacy_oracle(input,decoded,mutations[m]));
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
        if(kind==16) require(expected[row]!=bf16(virtual_q8_test::legacy_oracle(input,decoded,Mutation::lane_accumulator)),
            "one accumulator per lane must differ after BF16 storage");
    }
    if(cpu_only) return;
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
    if(kind>=23) {
        static std::array<std::uint16_t,8> original_payload{};
        const auto pair=(kind-23U)/2U;
        if((kind-23U)%2U==0) original_payload[pair]=old.front();
        else std::cout<<"QSA_TREE_ROLE_RECEIPT stride="<<(128U>>pair)
            <<" original_swapped_payload_discriminates="<<(original_payload[pair]!=old.front())
            <<" original_first_bits="<<original_payload[pair]<<" swapped_bits="<<old.front()<<'\n';
    }
    for(unsigned failure=0;failure<6;++failure) {
        check(hipMemset(d_output.p,0xa5,(rows+16U)*2U));bool rejected=false;
        auto in=strix::HipQsaInputProjectionView{reinterpret_cast<std::uintptr_t>(d_input.p),input.size()*4U};
        auto w=strix::HipQsaInputProjectionView{binding.address,binding.bytes};
        auto out=strix::HipQsaInputProjectionView{reinterpret_cast<std::uintptr_t>(output),rows*2U};
        try {
            if(failure==0) {
                strix::ManifestTensor tensor;tensor.ggml_type=0;
                const strix::TensorPair pair{&tensor,&binding};
                strix::ExecutionBufferAddress destination;
                destination.address=out.address;destination.bytes=out.bytes;
                strix::launch_linear_float_to_bf16(pair,static_cast<const float*>(d_input.p),width,rows,destination,true);
            } else {
                if(failure==4) out.address=in.address;
                if(failure==5) --w.bytes;
                strix::launch_hip_qsa_output_projection_probe(strix::HipQsaOutputProjectionMode::wave8,
                    in,w,out,failure==1?33U:failure==2?10272U:width,failure==3?0U:rows,stream.stream);
            }
        } catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"invalid output entry must reject before enqueue");
        std::vector<std::uint16_t> preserved(rows+16);check(hipMemcpy(preserved.data(),d_output.p,preserved.size()*2,hipMemcpyDeviceToHost));
        require(std::all_of(preserved.begin(),preserved.end(),[](auto v){return v==0xa5a5U;}),"rejected output entry mutated output");
    }
    std::cout<<"QSA_OUTPUT_RECEIPT width="<<width<<" rows="<<rows<<" case="<<kind<<" original_threads=256 candidate_threads=256 candidate_rows_per_block=8 exact_bf16=true cpu_finite_oracle="<<finite_oracle<<'\n';
}
}
int main() try {
    // Public registration, exact make-id and input-suffix composition admission.
    for(bool input_prefetch:{false,true}) {
        strix::HipIndexedAttentionLauncherConfig base;
        base.prefetch_q8_input=input_prefetch;
        if(input_prefetch)base.prefetch_eligible_layers={3};
        auto candidate=base;candidate.wave8_q8_output=true;
        strix::ExecutionBindingRequest request;request.prefetch_q8_qsa_input=input_prefetch;
        request.qsa_input_prefetch_layers=base.prefetch_eligible_layers;
        strix::kernels::KernelRegistry old_registry,new_registry;
        strix::register_hip_indexed_attention_kernel_variant(base.model,request,base,old_registry);
        request.wave8_q8_qsa_output=true;
        strix::register_hip_indexed_attention_kernel_variant(candidate.model,request,candidate,new_registry);
        const auto old=old_registry.variants().front(),changed=new_registry.variants().front();
        require(old.id!=changed.id && old.compiler_identity!=changed.compiler_identity,"output identities differ");
        require(changed.id.ends_with(strix::qsa_input_prefetch::kernel_suffix)==input_prefetch,"output marker preserves input suffix");
        auto old_launcher=strix::make_hip_indexed_attention_launcher(old.id,&base);
        auto new_launcher=strix::make_hip_indexed_attention_launcher(changed.id,&candidate);
        require(old_launcher.implementation_identity!=new_launcher.implementation_identity,"implementation identity differs");
        // A QSA input fallback is still an output candidate. Exercise the
        // bundle's actual two-config ownership rather than only string suffixes.
        auto fallback=candidate;fallback.prefetch_q8_input=false;fallback.prefetch_eligible_layers.clear();
        auto fallback_request=request;fallback_request.prefetch_q8_qsa_input=false;
        fallback_request.qsa_input_prefetch_layers.clear();
        strix::kernels::KernelRegistry fallback_registry;
        strix::register_hip_indexed_attention_kernel_variant(base.model,fallback_request,fallback,fallback_registry);
        const auto fallback_variant=fallback_registry.variants().front();
        strix::CompiledSerialExecution compiled;compiled.execution_identity=1;
        const unsigned steps=input_prefetch?2U:1U;
        for(unsigned i=0;i<steps;++i) {
            strix::ExecutionStep step;step.id=i;step.op=strix::SerialOp::indexed_attention;step.layer=i?7U:3U;
            compiled.execution.steps.push_back(step);
            strix::ExecutionKernelBinding binding;binding.step_id=i;binding.op=step.op;
            binding.kernel_id=i?fallback_variant.id:changed.id;compiled.bindings.bindings.push_back(binding);
        }
        strix::HipIndexedAttentionLauncherBundle bundle(candidate);strix::ExecutionLauncherRegistry launchers;
        bundle.register_for_compiled_execution(compiled,launchers);
        require(launchers.find_kernel_launcher(changed.id)!=nullptr &&
            (!input_prefetch || launchers.find_kernel_launcher(fallback_variant.id)!=nullptr),
            "selected output bit survives the actual input fallback bundle");
        for(bool reverse:{false,true}) {
            bool rejected=false;
            try {(void)strix::make_hip_indexed_attention_launcher(reverse?changed.id:old.id,reverse?&base:&candidate);}
            catch(const std::invalid_argument&) {rejected=true;}
            require(rejected,"make rejects both directions of output marker mismatch");
        }
        for(unsigned failure=0;failure<3;++failure) {
            auto bad=request;
            if(failure==0)bad.wave8_q8_qsa_output=false;
            if(failure==1)bad.device_graph_inputs=true;
            if(failure==2)bad.require_graph_safe=true;
            strix::kernels::KernelRegistry registry;bool rejected=false;
            try {strix::register_hip_indexed_attention_kernel_variant(candidate.model,bad,candidate,registry);}
            catch(const std::invalid_argument&) {rejected=true;}
            require(rejected && registry.variants().empty(),"public registration rejects before insertion");
        }
    }
    for(const unsigned width:{32U,96U,256U,288U,640U,6144U,10240U}) run(width,17,0);
    run(512,3,1);run(256,3,2);run(256,3,3);run(640,7,8);
    run(640,3,9);run(2560,3,10);run(2560,3,11);run(768,3,12);run(640,3,13);run(640,3,14);run(640,3,15);
    run(256,3,16);
    for(std::size_t m=0;m<detected.size();++m) {
        if(!detected[m]) throw std::runtime_error("fixture BF16-invisible mutant "+std::to_string(m));
    }
    for(unsigned mode=4;mode<=7;++mode) run(288,7,mode);
    for(unsigned rows:{1U,4U,7U,8U,9U,2560U,2563U}) run(6144,rows,0);
    run(10240,9,0);
    for(unsigned kind=17;kind<=38;++kind)run(640,3,kind);
    std::cout<<"QSA original/wave8 output projection fixtures passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
