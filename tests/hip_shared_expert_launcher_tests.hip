#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

static_assert(sizeof(Q8Block) == 34);

const strix::ExecutionStep& first_shared_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::shared_expert) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no shared-expert step");
}

std::vector<Q8Block> make_q8(
    std::uint32_t rows,
    std::uint32_t width,
    float scale_base) {
    const auto blocks_per_row =
        width / 32U;
    std::vector<Q8Block> result(
        static_cast<std::size_t>(rows) *
        blocks_per_row);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        for (std::uint32_t block_index = 0;
             block_index < blocks_per_row;
             ++block_index) {
            auto& block =
                result[
                    static_cast<std::size_t>(row) *
                        blocks_per_row +
                    block_index];
            block.scale =
                __float2half(
                    scale_base *
                    static_cast<float>(
                        row % 7U + 1U));
            for (std::uint32_t offset = 0;
                 offset < 32U;
                 ++offset) {
                block.values[offset] =
                    static_cast<std::int8_t>(
                        static_cast<int>(
                            (row * 3U +
                             block_index * 5U +
                             offset) %
                            11U) -
                        5);
            }
        }
    }
    return result;
}

float q8_value(
    const std::vector<Q8Block>& blocks,
    std::uint32_t row,
    std::uint32_t width,
    std::uint32_t feature) {
    const auto blocks_per_row =
        width / 32U;
    const auto& block =
        blocks[
            static_cast<std::size_t>(row) *
                blocks_per_row +
            feature / 32U];
    return __half2float(block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

float silu(float x) {
    return x / (1.0F + std::exp(-x));
}

float sigmoid(float x) {
    return 1.0F / (1.0F + std::exp(-x));
}

}  // namespace

int main() try {
    constexpr std::uint32_t hidden = 64;
    constexpr std::uint32_t intermediate = 32;

    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        first_shared_step(plan);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.activation_element_bytes = 4;
    buffer_request.mlp_intermediate_element_bytes = 4;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            buffer_request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc shared-expert arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x5348455854455354ULL;
    memory.arena_base =
        reinterpret_cast<std::uintptr_t>(
            arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto input =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto output =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    std::vector<float> hidden_values(hidden);
    for (std::size_t i = 0;
         i < hidden_values.size();
         ++i) {
        hidden_values[i] =
            static_cast<float>(
                static_cast<int>(i % 17U) -
                8) /
            19.0F;
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input.address),
            hidden_values.data(),
            hidden_values.size() *
                sizeof(float),
            hipMemcpyHostToDevice),
        "copy shared-expert input");

    const auto gate =
        make_q8(
            intermediate,
            hidden,
            0.0007F);
    std::vector<float> up(
        static_cast<std::size_t>(
            intermediate) *
        hidden);
    for (std::size_t i = 0;
         i < up.size();
         ++i) {
        up[i] =
            static_cast<float>(
                static_cast<int>(i % 13U) -
                6) *
            0.0009F;
    }
    const auto down =
        make_q8(
            hidden,
            intermediate,
            0.0006F);
    std::vector<float> shared_gate(
        hidden);
    for (std::size_t i = 0;
         i < shared_gate.size();
         ++i) {
        shared_gate[i] =
            static_cast<float>(
                static_cast<int>(i % 9U) -
                4) *
            0.0011F;
    }

    std::vector<float> expected_intermediate(
        intermediate);
    for (std::uint32_t inner = 0;
         inner < intermediate;
         ++inner) {
        double gate_sum = 0.0;
        double up_sum = 0.0;
        for (std::uint32_t feature = 0;
             feature < hidden;
             ++feature) {
            gate_sum +=
                static_cast<double>(
                    hidden_values[feature]) *
                q8_value(
                    gate,
                    inner,
                    hidden,
                    feature);
            up_sum +=
                static_cast<double>(
                    hidden_values[feature]) *
                static_cast<double>(
                    up[
                        static_cast<std::size_t>(
                            inner) *
                            hidden +
                        feature]);
        }
        expected_intermediate[inner] =
            silu(
                static_cast<float>(
                    gate_sum)) *
            static_cast<float>(
                up_sum);
    }

    double shared_gate_sum = 0.0;
    for (std::uint32_t feature = 0;
         feature < hidden;
         ++feature) {
        shared_gate_sum +=
            static_cast<double>(
                hidden_values[feature]) *
            static_cast<double>(
                shared_gate[feature]);
    }
    const float expected_scalar =
        sigmoid(
            static_cast<float>(
                shared_gate_sum));

    std::vector<float> expected(hidden);
    for (std::uint32_t feature = 0;
         feature < hidden;
         ++feature) {
        double value = 0.0;
        for (std::uint32_t inner = 0;
             inner < intermediate;
             ++inner) {
            value +=
                static_cast<double>(
                    expected_intermediate[
                        inner]) *
                q8_value(
                    down,
                    feature,
                    intermediate,
                    inner);
        }
        expected[feature] =
            static_cast<float>(
                value) *
            expected_scalar;
    }

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    auto add_tensor =
        [&](std::string name,
            std::uint32_t type,
            std::vector<std::uint64_t> dims,
            std::uint64_t bytes) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.ggml_type =
                type;
            tensor.dimensions =
                std::move(dims);
            tensor.byte_size =
                bytes;
            tensor.role =
                strix::TensorRole::
                    dense_projection;
            tensor.layer =
                step.layer;
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add_tensor(
        "blk.0.shared_expert.gate_proj.weight",
        8,
        {hidden, intermediate},
        gate.size() * sizeof(Q8Block));
    add_tensor(
        "blk.0.shared_expert.up_proj.weight",
        0,
        {hidden, intermediate},
        up.size() * sizeof(float));
    add_tensor(
        "blk.0.shared_expert.down_proj.weight",
        8,
        {intermediate, hidden},
        down.size() * sizeof(Q8Block));
    add_tensor(
        "blk.0.shared_expert_gate.weight",
        0,
        {hidden, 1},
        shared_gate.size() * sizeof(float));

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        manifest.tensors.size(),
        nullptr);
    const auto upload =
        [&](std::size_t index,
            const void* source,
            std::size_t bytes) {
            check(
                hipMalloc(
                    &device_weights[index],
                    bytes),
                "hipMalloc shared-expert weight");
            check(
                hipMemcpy(
                    device_weights[index],
                    source,
                    bytes,
                    hipMemcpyHostToDevice),
                "copy shared-expert weight");
        };
    upload(
        0,
        gate.data(),
        gate.size() *
            sizeof(Q8Block));
    upload(
        1,
        up.data(),
        up.size() *
            sizeof(float));
    upload(
        2,
        down.data(),
        down.size() *
            sizeof(Q8Block));
    upload(
        3,
        shared_gate.data(),
        shared_gate.size() *
            sizeof(float));

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    for (std::size_t index = 0;
         index <
            manifest.tensors.size();
         ++index) {
        runtime.tensors.push_back(
            {
                static_cast<std::uint32_t>(
                    index),
                strix::RuntimeTensorStorage::
                    resident,
                reinterpret_cast<std::uintptr_t>(
                    device_weights[index]),
                manifest.tensors[index].
                    byte_size,
                1,
                0,
            });
    }

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x5348455854455355ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {0, 1, 2, 3},
        });

    strix::ExecutionRuntimeResources resources;
    resources.scratch =
        &memory;
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &runtime;
    resources.tensor_access =
        &access;

    strix::ExecutionDispatchContext context;
    context.step_id =
        step.id;
    context.op =
        step.op;
    context.layer =
        step.layer;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.temporaries =
        step_io.temporaries;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.resources =
        &resources;
    context.logical_rows = 1;
    context.request_concurrency = 1;

    strix::HipSharedExpertLauncherConfig config;
    config.hidden_size =
        hidden;
    config.intermediate_size =
        intermediate;
    config.activation =
        strix::HipActivationStorage::fp32;

    auto launcher =
        strix::make_hip_shared_expert_launcher(
            "test.shared_expert",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "shared-expert synchronize");

    std::vector<float> actual(hidden);
    check(
        hipMemcpy(
            actual.data(),
            reinterpret_cast<const void*>(
                output.address),
            actual.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy shared-expert output");

    float maximum_error = 0.0F;
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    actual[index] -
                    expected[index]));
    }
    expect(
        maximum_error <= 2e-4F,
        "mixed-format shared expert matches CPU reference");


    {
        const auto temporary=strix::resolve_execution_buffer(memory.execution_identity,arena,memory,step_io.temporaries[0]);
        hipStream_t stream=nullptr;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create shared comparison stream");
        strix::HipLaunchContext scope;scope.stream=stream;
        // This operator has no device-position/status consumer. The owned stream
        // scope is sufficient for both eager enqueue and external HIP capture.
        const auto read_bytes=[&](std::uintptr_t address,std::uint64_t bytes){std::vector<std::byte> result(bytes);
            check(hipMemcpy(result.data(),reinterpret_cast<void*>(address),bytes,hipMemcpyDeviceToHost),"read shared comparison bytes");return result;};
        const auto bf16=[](float value){auto bits=std::bit_cast<std::uint32_t>(value);
            bits+=0x7fffU+((bits>>16U)&1U);return static_cast<std::uint16_t>(bits>>16U);};
        for(unsigned mixed=0;mixed<2;++mixed){
            auto& up_tensor=manifest.tensors[1];
            if(mixed){up_tensor.ggml_type=0;up_tensor.byte_size=up.size()*4;
                check(hipMemcpy(device_weights[1],up.data(),up_tensor.byte_size,hipMemcpyHostToDevice),"restore shared F32 up fallback");}
            else{const auto up_q8=make_q8(intermediate,hidden,0.0009F);up_tensor.ggml_type=8;
                up_tensor.byte_size=up_q8.size()*sizeof(Q8Block);
                check(hipMemcpy(device_weights[1],up_q8.data(),up_tensor.byte_size,hipMemcpyHostToDevice),"set shared Q8 up");}
            runtime.tensors[1].bytes=up_tensor.byte_size;
            manifest.layout_identity=strix::model_manifest_layout_identity(manifest);
            runtime.manifest_layout_identity=manifest.layout_identity;access.manifest_layout_identity=manifest.layout_identity;
            for(auto dtype:{strix::HipActivationStorage::bf16,strix::HipActivationStorage::fp32}){
                auto off_config=config;off_config.activation=dtype;
                auto on_config=off_config;on_config.prefetch_q8=true;on_config.prefetch_eligible_layers={*step.layer};
                on_config.prefetch_stats=std::make_shared<strix::shared_expert_prefetch::RuntimeDispatchStats>();
                auto off_launch=strix::make_hip_shared_expert_launcher("test.shared.off",&off_config);
                auto on_launch=strix::make_hip_shared_expert_launcher("test.shared.on",&on_config);
                hipGraph_t graphs[2]{};hipGraphExec_t executables[2]{};
                for(unsigned mode=0;mode<2;++mode){auto& candidate=mode?on_launch:off_launch;
                    const auto before=on_config.prefetch_stats->dispatched_projections.load();
                    check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture shared ON/OFF");
                    try{const strix::HipLaunchScope owned(scope);candidate.function(context,candidate.user_data);}
                    catch(...){static_cast<void>(hipStreamEndCapture(stream,&graphs[mode]));throw;}
                    check(hipStreamEndCapture(stream,&graphs[mode]),"end shared ON/OFF capture");
                    std::size_t count=0;check(hipGraphGetNodes(graphs[mode],nullptr,&count),"shared graph node count");
                    expect(count==5,"shared exact schedule retains all five kernels");
                    std::vector<hipGraphNode_t> nodes(count);check(hipGraphGetNodes(graphs[mode],nodes.data(),&count),"shared graph nodes");
                    unsigned candidate_nodes=0;for(auto node:nodes){hipGraphNodeType type;
                        check(hipGraphNodeGetType(node,&type),"shared graph node type");
                        if(type==hipGraphNodeTypeKernel){hipKernelNodeParams params{};
                            check(hipGraphKernelNodeGetParams(node,&params),"shared node params");
                            if(params.blockDim.x==256 && (params.gridDim.x==4 || params.gridDim.x==8))++candidate_nodes;}}
                    expect(candidate_nodes==(mode?3U-mixed:0U),"shared graph contains exact row-group candidate nodes");
                    expect(on_config.prefetch_stats->dispatched_projections.load()-before==(mode?3U-mixed:0U),
                           "shared capture counter matches eligible projections");
                    check(hipGraphInstantiate(&executables[mode],graphs[mode],nullptr,nullptr,0),"instantiate shared comparison");
                }
                const auto reset=[&](unsigned replay){
                    check(hipMemset(reinterpret_cast<void*>(output.address),0xff,output.bytes),"poison shared output");
                    check(hipMemset(reinterpret_cast<void*>(temporary.address),0xff,temporary.bytes),"poison shared workspace");
                    for(unsigned f=0;f<hidden;++f)hidden_values[f]=static_cast<float>(static_cast<int>((f+7*replay)%19)-9)/23.0F;
                    if(dtype==strix::HipActivationStorage::fp32)check(hipMemcpy(reinterpret_cast<void*>(input.address),hidden_values.data(),hidden*4,hipMemcpyHostToDevice),"set shared FP32 input");
                    else{std::vector<std::uint16_t> encoded(hidden);for(unsigned f=0;f<hidden;++f)encoded[f]=bf16(hidden_values[f]);
                        check(hipMemcpy(reinterpret_cast<void*>(input.address),encoded.data(),hidden*2,hipMemcpyHostToDevice),"set shared BF16 input");}
                };
                for(unsigned replay=0;replay<3;++replay){std::vector<std::byte> expected_output,expected_workspace;
                    for(unsigned path=0;path<4;++path){reset(replay);const auto before=on_config.prefetch_stats->dispatched_projections.load();
                        if(path<2){auto& candidate=path?on_launch:off_launch;
                            {const strix::HipLaunchScope owned(scope);candidate.function(context,candidate.user_data);}
                            expect(on_config.prefetch_stats->dispatched_projections.load()-before==(path?3U-mixed:0U),"shared eager counter exact");}
                        else{check(hipGraphLaunch(executables[path-2],stream),"replay shared ON/OFF");
                            expect(on_config.prefetch_stats->dispatched_projections.load()==before,"shared graph replay leaves host capture count unchanged");}
                        check(hipStreamSynchronize(stream),"complete shared comparison");
                        auto actual_output=read_bytes(output.address,output.bytes),actual_workspace=read_bytes(temporary.address,temporary.bytes);
                        for(unsigned f=0;f<hidden;++f){float value;
                            if(dtype==strix::HipActivationStorage::fp32)std::memcpy(&value,actual_output.data()+f*4,4);
                            else{std::uint16_t bits;std::memcpy(&bits,actual_output.data()+f*2,2);value=std::bit_cast<float>(static_cast<std::uint32_t>(bits)<<16U);}
                            expect(std::isfinite(value),"shared full operator output finite");}
                        if(path==0){expected_output=std::move(actual_output);expected_workspace=std::move(actual_workspace);}
                        else expect(actual_output==expected_output && actual_workspace==expected_workspace,
                            "shared full operator ON/OFF eager/graph exact output and intermediate bytes");
                    }
                }
                for(unsigned failure=0;failure<2;++failure){reset(0);auto bad_arena=arena;
                    const auto old_address=runtime.tensors[0].address;
                    if(failure==0){const auto* allocation=strix::find_arena_allocation(arena,step_io.temporaries[0]);
                        for(auto& output_allocation:bad_arena.allocations)if(output_allocation.use==step_io.writes[0]){output_allocation.offset=allocation->offset;output_allocation.bytes=hidden*static_cast<std::uint32_t>(dtype);}
                        context.arena=&bad_arena;
                    }else runtime.tensors[0].address+=2;
                    const auto before=read_bytes(memory.arena_base,memory.arena_bytes);
                    const auto before_count=on_config.prefetch_stats->dispatched_projections.load();
                    hipGraph_t rejected_graph=nullptr;bool rejected=false;
                    check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture shared rejected preflight");
                    try{const strix::HipLaunchScope owned(scope);on_launch.function(context,on_launch.user_data);}
                    catch(const std::invalid_argument& error){rejected=std::string(error.what()).find(failure?
                        "weight alignment/extent invalid":"writable alias")!=std::string::npos;}
                    check(hipStreamEndCapture(stream,&rejected_graph),"end shared rejected preflight");
                    std::size_t count=1;check(hipGraphGetNodes(rejected_graph,nullptr,&count),"rejected shared nodes");
                    expect(rejected && count==0 && before==read_bytes(memory.arena_base,memory.arena_bytes) &&
                        before_count==on_config.prefetch_stats->dispatched_projections.load(),"shared preflight rejects precise cause before any node or mutation");
                    check(hipGraphDestroy(rejected_graph),"destroy shared rejected graph");
                    context.arena=&arena;runtime.tensors[0].address=old_address;
                }
                for(unsigned mode=0;mode<2;++mode){check(hipGraphExecDestroy(executables[mode]),"destroy shared executable");
                    check(hipGraphDestroy(graphs[mode]),"destroy shared graph");}
            }
        }
        check(hipStreamDestroy(stream),"destroy shared comparison stream");
    }

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree shared-expert weight");
    }
    check(
        hipFree(arena_pointer),
        "hipFree shared-expert arena");

    if (failures == 0) {
        std::cout
            << "HIP shared-expert launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP shared-expert launcher tests: "
        << error.what() << '\n';
    return 2;
}
