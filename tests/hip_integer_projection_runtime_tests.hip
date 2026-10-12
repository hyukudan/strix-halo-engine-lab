#include "integer_projection_runtime_fixture.hpp"
#include "strix/hip_integer_projection_runtime.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/dispatch_resources.hpp"
#include <hip/hip_runtime.h>
#include <bit>
#include <iostream>
#include <limits>

namespace ip=strix::integer_projection;
namespace lp=strix::low_precision;
namespace {
void check(hipError_t e,const char* message){if(e!=hipSuccess)throw std::runtime_error(std::string(message)+": "+hipGetErrorString(e));}
void expect(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Allocation {
    void* pointer=nullptr;std::size_t bytes=0;
    explicit Allocation(std::size_t count):bytes(count){check(hipMalloc(&pointer,count),"fixture allocation");}
    ~Allocation(){int active=-1;if(hipGetDevice(&active)==hipSuccess&&active==0&&hipStreamQuery(hipStreamLegacy)==hipSuccess)(void)hipFree(pointer);}
};
struct Stream {
    hipStream_t value=nullptr;
    Stream(){check(hipStreamCreateWithFlags(&value,hipStreamNonBlocking),"fixture owned stream");}
    ~Stream(){if(hipStreamQuery(value)==hipSuccess)(void)hipStreamDestroy(value);}
};
void run(ip::WeightEncoding encoding,lp::ActivationEncoding activation,ip::Instruction instruction,unsigned k,unsigned columns,bool bf16,bool explicit_stream=false){
    integer_runtime_fixture::Fixture f(columns,k,encoding,bf16?ip::RuntimeInput::bf16:ip::RuntimeInput::fp32,activation,instruction,17);
    // Other compiled steps are not executed. The selected step still consumes
    // the actual compiler/registry binding for this exact alternate profile.
    const auto step=f.access.accesses.front().step_id;
    auto arena=std::make_shared<Allocation>(static_cast<std::size_t>(f.memory.arena_bytes));
    auto weights=std::make_shared<Allocation>(static_cast<std::size_t>(f.weights.tensors[0].bytes));
    f.memory.arena_base=reinterpret_cast<std::uintptr_t>(arena->pointer);f.weights.tensors[0].address=reinterpret_cast<std::uintptr_t>(weights->pointer);
    auto ap=ip::HipIntegerRuntimeAllocationOwner::pin(f.config.device,f.memory.arena_base,f.memory.allocation_generation,arena);
    auto wp=ip::HipIntegerRuntimeAllocationOwner::pin(f.config.device,f.weights.tensors[0].address,f.weights.tensors[0].allocation_generation,weights);
    f.owners={ap,wp};auto context=f.context();
    const auto input=strix::resolve_dispatch_read(context,0),output=strix::resolve_dispatch_write(context,0);
    const auto block_bytes=encoding==ip::WeightEncoding::q4_0?18U:136U;
    std::vector<std::byte> checkpoint(static_cast<std::size_t>(weights->bytes));
    for(std::size_t i=0;i<checkpoint.size();i+=block_bytes){
        const auto blocks_per_column=k/(encoding==ip::WeightEncoding::q4_0?32U:256U);
        const unsigned short scale=i/block_bytes<blocks_per_column?0x0001U:((i/block_bytes)%3U==0)?0xb800U:0x3400U;
        checkpoint[i]=static_cast<std::byte>(scale&255U);checkpoint[i+1]=static_cast<std::byte>(scale>>8U);
        if(encoding==ip::WeightEncoding::iq4_xs){checkpoint[i+2]=std::byte{0xe4};checkpoint[i+3]=std::byte{0x1b};
            for(unsigned j=0;j<4;++j)checkpoint[i+4+j]=static_cast<std::byte>(0x10U+j*0x22U);}
        const auto start=encoding==ip::WeightEncoding::q4_0?2U:8U;
        for(unsigned j=start;j<block_bytes;++j)checkpoint[i+j]=static_cast<std::byte>((i/block_bytes+j*37U)&255U);
    }
    std::vector<float> values(k);std::vector<unsigned short> bfvalues(k);std::vector<double> reference_input(k);
    for(unsigned i=0;i<k;++i){values[i]=static_cast<float>(static_cast<int>(i%23)-11)/16.0F;
        bfvalues[i]=static_cast<unsigned short>(std::bit_cast<unsigned>(values[i])>>16U);
        reference_input[i]=bf16?static_cast<double>(std::bit_cast<float>(static_cast<unsigned>(bfvalues[i])<<16U)):static_cast<double>(values[i]);}
    ap->advance_access(2);wp->advance_access(2);
    check(hipMemcpy(weights->pointer,checkpoint.data(),checkpoint.size(),hipMemcpyHostToDevice),"original checkpoint upload");
    check(hipMemcpy(reinterpret_cast<void*>(input.address),bf16?static_cast<const void*>(bfvalues.data()):static_cast<const void*>(values.data()),k*(bf16?2U:4U),hipMemcpyHostToDevice),"input upload");
    const auto staged=ip::stage_weights(encoding,columns,k,checkpoint);
    const auto prepared=ip::prepare({1,columns,k},activation,reference_input);
    const auto mathematical=ip::reference(staged,prepared,instruction);
    std::vector<float> expected(columns);for(unsigned i=0;i<columns;++i)expected[i]=static_cast<float>(mathematical.output[i]);
    expect(expected[0]!=0.0F,"minimum subnormal FP16 scale column must produce observable nonzero reference");
    const auto plan=ip::RuntimePlan::bind(f.compiled,f.resources,f.config,f.owners);
    strix::HipLmHeadLauncherConfig default_config;default_config.hidden_size=k;default_config.vocab_size=columns;
    default_config.input=bf16?strix::HipActivationStorage::bf16:strix::HipActivationStorage::fp32;
    strix::HipLmHeadLauncherBundle default_bundle(default_config);strix::ExecutionLauncherRegistry rejected_registry;
    bool default_rejected=false;
    try{default_bundle.register_for_compiled_execution(f.compiled,rejected_registry);}catch(const std::invalid_argument&){default_rejected=true;}
    expect(default_rejected,"default floating launcher must reject alternate integer numerical binding");
    auto stream_owner=explicit_stream?std::make_shared<Stream>():std::shared_ptr<Stream>{};
    const auto stream=stream_owner?stream_owner->value:hipStreamLegacy;
    ip::HipIntegerProjectionRuntimeBundle bundle(plan,f.owners,stream,stream_owner);
    if(explicit_stream){const auto unscoped=bundle.execute(context);expect(!unscoped.output_committed&&!bundle.failed(),"nonlegacy runtime cannot guess upstream producer stream");}
    strix::HipLaunchContext launch_context;launch_context.stream=stream;
    auto scope=explicit_stream?std::make_unique<strix::HipLaunchScope>(launch_context):std::unique_ptr<strix::HipLaunchScope>{};
    if(explicit_stream){check(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal),"fixture capture begin");
        const auto captured=bundle.execute(context);expect(!captured.output_committed&&!bundle.failed(),"blocking runtime must reject actual active capture");
        hipGraph_t graph=nullptr;check(hipStreamEndCapture(stream,&graph),"fixture capture end");check(hipGraphDestroy(graph),"fixture empty graph destroy");}
    strix::ExecutionLauncherRegistry registry;bundle.register_for_compiled_execution(f.compiled,registry);
    strix::CompiledLaunchPlan launch;launch.execution_identity=f.compiled.execution_identity;launch.plan_identity=1;
    const auto* registered=registry.find_kernel_launcher(ip::runtime_kernel_id(f.config));expect(registered,"actual runtime launcher registered");
    launch.steps.resize(step+1);launch.steps[step]={step,strix::SerialOp::lm_head,false,registered->implementation_identity,registered->function,registered->user_data};
    strix::DirectLaunchDispatcher dispatcher(launch);
    dispatcher.dispatch_kernel(context);
    std::vector<float> observed(columns);check(hipMemcpy(observed.data(),reinterpret_cast<const void*>(output.address),columns*4U,hipMemcpyDeviceToHost),"logits readback");
    for(unsigned i=0;i<columns;++i)expect(std::bit_cast<unsigned>(observed[i])==std::bit_cast<unsigned>(expected[i]),"actual dispatcher logits differ from independent scalar oracle");
    // Every source element is subnormal, so a source conversion that flushes
    // before exact widening would turn the real projection into zero.
    ap->advance_access(3);std::fill(values.begin(),values.end(),std::numeric_limits<float>::denorm_min());
    std::fill(bfvalues.begin(),bfvalues.end(),1U);
    std::fill(reference_input.begin(),reference_input.end(),bf16?static_cast<double>(std::bit_cast<float>(0x00010000U)):
        static_cast<double>(std::numeric_limits<float>::denorm_min()));
    check(hipMemcpy(reinterpret_cast<void*>(input.address),bf16?static_cast<const void*>(bfvalues.data()):static_cast<const void*>(values.data()),k*(bf16?2U:4U),hipMemcpyHostToDevice),"all-subnormal raw source upload");
    const auto subnormal_prepared=ip::prepare({1,columns,k},activation,reference_input);
    const auto subnormal_math=ip::reference(staged,subnormal_prepared,instruction);
    dispatcher.dispatch_kernel(context);
    check(hipMemcpy(observed.data(),reinterpret_cast<const void*>(output.address),columns*4U,hipMemcpyDeviceToHost),"subnormal source logits readback");
    bool nonzero=false;for(unsigned i=0;i<columns;++i){
        expect(std::bit_cast<unsigned>(observed[i])==std::bit_cast<unsigned>(static_cast<float>(subnormal_math.output[i])),"all-subnormal source lost exact widening or gradual output bits");
        nonzero=nonzero||(std::bit_cast<unsigned>(observed[i])&0x7fffffffU)!=0;}
    expect(nonzero,"subnormal native projection has an observable nonzero mathematical result");
    // Last-tile failure must not publish earlier successful column tiles.
    std::vector<float> sentinel(columns,37.0F);ap->advance_access(4);wp->advance_access(3);
    check(hipMemcpy(reinterpret_cast<void*>(output.address),sentinel.data(),columns*4U,hipMemcpyHostToDevice),"sentinel logits upload");
    const auto last_column=columns-1;const auto block=static_cast<std::size_t>(last_column)*(k/(encoding==ip::WeightEncoding::q4_0?32U:256U))*block_bytes;
    checkpoint[block]=std::byte{0x00};checkpoint[block+1]=std::byte{0x7c};
    check(hipMemcpy(weights->pointer,checkpoint.data(),checkpoint.size(),hipMemcpyHostToDevice),"late nonfinite checkpoint upload");
    const auto failed=bundle.execute(context);expect(!failed.output_committed&&!failed.destination_poisoned&&failed.stream_quiescent&&failed.completed_tiles>0,"late tile failure lacks uncommitted quiescent semantics");
    check(hipMemcpy(observed.data(),reinterpret_cast<const void*>(output.address),columns*4U,hipMemcpyDeviceToHost),"rejected logits readback");expect(observed==sentinel,"late failed tile overwrote full logits destination");
    expect(bundle.failed()&&!bundle.execute(context).output_committed,"failed runtime bundle cannot be replayed");
    auto rejects_without_commit=[&](const char* message){
        ap->advance_access(ap->current_binding().access_generation+1);wp->advance_access(wp->current_binding().access_generation+1);
        check(hipMemcpy(reinterpret_cast<void*>(output.address),sentinel.data(),columns*4U,hipMemcpyHostToDevice),"failure sentinel reset");
        check(hipMemcpy(weights->pointer,checkpoint.data(),checkpoint.size(),hipMemcpyHostToDevice),"failure checkpoint upload");
        check(hipMemcpy(reinterpret_cast<void*>(input.address),bf16?static_cast<const void*>(bfvalues.data()):static_cast<const void*>(values.data()),k*(bf16?2U:4U),hipMemcpyHostToDevice),"failure input upload");
        ip::HipIntegerProjectionRuntimeBundle rejected(plan,f.owners,stream,stream_owner);const auto result=rejected.execute(context);
        expect(!result.output_committed&&!result.destination_poisoned&&result.stream_quiescent,message);
        check(hipMemcpy(observed.data(),reinterpret_cast<const void*>(output.address),columns*4U,hipMemcpyDeviceToHost),"failure logits readback");
        expect(observed==sentinel,"rejected preparation/conversion overwrote published logits");
    };
    checkpoint[block]=std::byte{0x00};checkpoint[block+1]=std::byte{0x34};
    values[0]=std::numeric_limits<float>::quiet_NaN();bfvalues[0]=0x7fc0U;
    rejects_without_commit("nonfinite source must fail before logits commit");
    for(std::size_t i=0;i<checkpoint.size();i+=block_bytes){checkpoint[i]=std::byte{0xff};checkpoint[i+1]=std::byte{0x7b};
        for(unsigned j=2;j<block_bytes;++j)checkpoint[i+j]=std::byte{0xff};}
    std::fill(values.begin(),values.end(),std::numeric_limits<float>::max());std::fill(bfvalues.begin(),bfvalues.end(),0x7f7fU);
    rejects_without_commit("finite binary64 projection overflow in FP32 conversion must reject whole output");
}
}
int main()try{
    int count=0;if(hipGetDeviceCount(&count)!=hipSuccess||count==0){std::cout<<"SKIP: native gfx1151 unavailable\n";return 77;}
    check(hipSetDevice(0),"fixture select device");hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,0),"fixture device properties");
    const std::string architecture=props.gcnArchName;if(architecture.substr(0,architecture.find(':'))!="gfx1151"){std::cout<<"SKIP: native gfx1151 unavailable\n";return 77;}
    run(ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8,640,35,false);
    run(ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu4,2560,35,true);
    run(ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8,2560,35,true);
    run(ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu8,2560,35,false,true);
    std::cout<<"Native experimental integer runtime synthetic correctness passed; no model qualification.\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
