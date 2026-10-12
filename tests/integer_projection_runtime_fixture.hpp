#pragma once
#include "strix/integer_projection_runtime.hpp"
#include <stdexcept>

namespace integer_runtime_fixture {
namespace ip=strix::integer_projection;
struct Pin:ip::AllocationPin {
    ip::AllocationBinding value;
    bool throwing=false;
    ip::AllocationBinding current_binding()const override{if(throwing)throw std::runtime_error("owner unavailable");return value;}
};
struct Fixture {
    strix::ModelContract model=strix::ModelContract::qwen38_flash_next();
    ip::RuntimeConfig config;
    strix::CompiledSerialExecution compiled;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionTensorAccessPlan access;
    strix::ExecutionMemoryBinding memory;
    strix::ExecutionRuntimeResources resources;
    std::shared_ptr<Pin> arena_pin=std::make_shared<Pin>(),weight_pin=std::make_shared<Pin>();
    ip::RuntimeOwners owners{arena_pin,weight_pin};
    Fixture(std::uint32_t columns=153600,std::uint32_t k=2560,ip::WeightEncoding encoding=ip::WeightEncoding::iq4_xs,
        ip::RuntimeInput input=ip::RuntimeInput::fp32,
        strix::low_precision::ActivationEncoding activation=strix::low_precision::ActivationEncoding::symmetric_a8,
        ip::Instruction instruction=ip::Instruction::iu8,std::uint32_t tile=1024){
        model.vocab_size=columns;model.hidden_size=k;model.ple_eos_token_id=0;
        config.experimental_opt_in=true;config.hidden_size=k;config.vocabulary_size=columns;config.weights=encoding;
        config.input=input;config.activations=activation;config.instruction=instruction;config.tile_columns=tile;
        config.device={0,17,"gfx1151"};config.maximum_internal_bytes=ip::max_working_bytes;
        config.maximum_retained_bytes=2ULL*1024*1024*1024;config.approved_numerical_identity=ip::runtime_numerical_identity(config);
        strix::ExecutionBindingRequest request;request.minimum_qualification=strix::kernels::QualificationStatus::experimental;
        strix::kernels::KernelRegistry registry;
        const auto full=strix::build_serial_decode_execution_plan(model);
        for(const auto& step:full.steps){const auto family=strix::operator_family_for(step.op);if(!family||step.op==strix::SerialOp::lm_head)continue;
            strix::kernels::KernelVariant v;v.id="fixture."+std::string(strix::to_string(step.op));v.family=*family;v.phase=strix::kernels::ExecutionPhase::decode;
            v.shape=strix::execution_shape_for(model,step.op,request);v.resources.workgroup_threads=32;v.numerics={"fp32","fp32","fp32",true,true};
            v.input_layout="fixture";v.output_layout="fixture";v.compiler_identity="fixture";
            if(!registry.find(v.id))registry.add(v);}
        ip::register_integer_projection_runtime_variant(model,request,config,registry);
        strix::ExecutionBufferRequest buffers;buffers.activation_element_bytes=input==ip::RuntimeInput::bf16?2:4;
        compiled=strix::compile_serial_execution(model,registry,request,buffers);
        const auto count=static_cast<std::uint64_t>(columns)*(k/(encoding==ip::WeightEncoding::q4_0?32U:256U));
        strix::ManifestTensor tensor;tensor.name="output.weight";tensor.dimensions={k,columns};tensor.ggml_type=encoding==ip::WeightEncoding::q4_0?2:23;
        tensor.byte_size=count*(encoding==ip::WeightEncoding::q4_0?18U:136U);tensor.role=strix::TensorRole::output_head;
        manifest.tensors={tensor};manifest.total_tensor_bytes=tensor.byte_size;manifest.layout_identity=strix::model_manifest_layout_identity(manifest);
        weights.manifest_layout_identity=manifest.layout_identity;weights.model_content_identity.bytes[0]=1;
        weights.tensors.push_back({0,strix::RuntimeTensorStorage::resident,0x100000000ULL,tensor.byte_size,29,0});
        access.manifest_layout_identity=manifest.layout_identity;access.plan_identity=31;
        for(const auto& step:compiled.execution.steps)if(step.op==strix::SerialOp::lm_head)access.accesses.push_back({step.id,step.op,0,{0}});
        memory={compiled.execution_identity,0x200000000ULL,compiled.arena.total_bytes,37};
        resources.scratch=&memory;resources.model_manifest=&manifest;resources.model_weights=&weights;resources.tensor_access=&access;
        arena_pin->value={memory.arena_base,memory.arena_bytes,41,37,config.device,1};
        weight_pin->value={weights.tensors[0].address,weights.tensors[0].bytes,43,29,config.device,1};
    }
    strix::ExecutionDispatchContext context(){
        const auto step=access.accesses.front().step_id;const auto& io=compiled.io.steps[step];
        strix::ExecutionDispatchContext c;c.execution_identity=compiled.execution_identity;c.step_id=step;c.op=strix::SerialOp::lm_head;
        c.kernel_id=*compiled.bindings.bindings[step].kernel_id;c.reads=io.reads;c.writes=io.writes;c.arena=&compiled.arena;c.resources=&resources;return c;
    }
    ip::RuntimePlan bind(){return ip::RuntimePlan::bind(compiled,resources,config,owners);}
};
}
