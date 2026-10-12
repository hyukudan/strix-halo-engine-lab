#include "strix/integer_projection_runtime.hpp"
#include "strix/dispatch_resources.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix::integer_projection {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
std::uint64_t add(std::uint64_t a,std::uint64_t b){if(b>UINT64_MAX-a)throw std::overflow_error("runtime projection accounting overflow");return a+b;}
std::uint64_t multiply(std::uint64_t a,std::uint64_t b){if(a&&b>UINT64_MAX/a)throw std::overflow_error("runtime projection accounting overflow");return a*b;}
const char* input_name(RuntimeInput input){switch(input){case RuntimeInput::bf16:return "bf16";case RuntimeInput::fp32:return "fp32";}throw std::invalid_argument("unknown runtime projection input encoding");}
void validate(const RuntimeConfig& c){
    require(c.experimental_opt_in&&c.hidden_size&&c.vocabulary_size,"explicit experimental runtime projection and nonzero shape required");
    require(c.tile_columns&&c.tile_columns<=4096,"runtime projection column tile must be in [1,4096]");
    (void)input_name(c.input);
    require(c.maximum_internal_bytes&&c.maximum_internal_bytes<=max_working_bytes&&c.maximum_retained_bytes,
        "runtime projection requires explicit internal and full retained admission ceilings");
    require(c.approved_numerical_identity==runtime_numerical_identity(c),"runtime projection numeric approval must match exact alternate target");
}
ExecutionRequest tile_request(const RuntimeConfig& c,std::uint32_t columns,std::uint64_t generation){
    ExecutionRequest r;r.experimental_opt_in=c.experimental_opt_in;r.phase=Phase::decode;r.operation_generation=generation;
    r.shape={1,columns,c.hidden_size};r.weights=c.weights;r.activations=c.activations;r.instruction=c.instruction;r.device=c.device;
    r.approved_numerical_identity=numerical_identity(c.weights,c.activations,c.instruction);r.maximum_device_bytes=c.maximum_internal_bytes;return r;
}
bool same_physical(AllocationBinding a,AllocationBinding b){a.access_generation=0;b.access_generation=0;return a==b;}
void view(const AllocationBinding& a,std::uintptr_t pointer,std::uint64_t bytes,std::uint64_t generation,const DeviceIdentity& device){
    require(a.base&&a.bytes&&a.identity&&a.generation==generation&&a.access_generation&&a.device==device,
        "runtime projection allocation owner/generation/device mismatch");
    require(a.bytes<=UINTPTR_MAX-a.base&&pointer>=a.base&&pointer-a.base<=a.bytes&&bytes<=a.bytes-(pointer-a.base),
        "runtime projection source/destination view exceeds owning allocation");
}
std::uint64_t hash(std::string_view text){std::uint64_t h=14695981039346656037ULL;for(char c:text){h^=static_cast<unsigned char>(c);h*=1099511628211ULL;}return h?h:1;}
std::string content(const ModelContentIdentity& c){std::string s;constexpr char digits[]="0123456789abcdef";for(auto b:c.bytes){s+=digits[b>>4];s+=digits[b&15];}return s;}
bool same_tensor(const ManifestTensor& a,const ManifestTensor& b){return a.name==b.name&&a.dimensions==b.dimensions&&a.ggml_type==b.ggml_type&&
    a.byte_size==b.byte_size&&a.absolute_offset==b.absolute_offset&&a.role==b.role&&a.placement.kind==b.placement.kind&&a.placement.node==b.placement.node&&a.layer==b.layer;}
bool same_runtime(const RuntimeTensorBinding& a,const RuntimeTensorBinding& b){return a.tensor_index==b.tensor_index&&a.storage==b.storage&&a.address==b.address&&
    a.bytes==b.bytes&&a.allocation_generation==b.allocation_generation&&a.owner_node==b.owner_node;}
}
std::string runtime_numerical_identity(const RuntimeConfig& c){return "strix-experimental-lm-head-v1:decode:C1:M1:"+std::string(input_name(c.input))+
    "-exact-widen-f64:"+numerical_identity(c.weights,c.activations,c.instruction)+
    ":f64-group-order:binary32-integer-RNE-gradual-signed-zero:reject-nonfinite:whole-logits-status-gated-commit";}
std::string runtime_kernel_id(const RuntimeConfig& c){validate(c);return "hip.experimental.integer_lm_head.h"+std::to_string(c.hidden_size)+".v"+
    std::to_string(c.vocabulary_size)+".tile"+std::to_string(c.tile_columns)+".profile"+std::to_string(hash(c.approved_numerical_identity));}
void register_integer_projection_runtime_variant(const ModelContract& model,const ExecutionBindingRequest& request,const RuntimeConfig& c,kernels::KernelRegistry& registry){
    validate(c);require(model.validate().empty(),"invalid runtime projection model contract");
    require(c.hidden_size==model.hidden_size&&c.vocabulary_size==model.vocab_size,"runtime projection config differs from model shape");
    require(request.logical_rows==1&&request.request_concurrency==1&&!request.require_graph_safe&&
        request.minimum_qualification==kernels::QualificationStatus::experimental,"runtime projection variant requires explicit experimental C1 non-graph selection");
    const auto core=ExecutionPlan::build(tile_request(c,std::min(c.tile_columns,c.vocabulary_size),1));
    const auto bytes=add(core.accounting().used_device_bytes,add(multiply(c.vocabulary_size,4),4));
    require(bytes<=c.maximum_internal_bytes,"runtime projection internal peak exceeds admission ceiling");
    kernels::KernelVariant v;v.id=runtime_kernel_id(c);v.family=kernels::OperatorFamily::lm_head;v.phase=kernels::ExecutionPhase::decode;
    v.shape=execution_shape_for(model,SerialOp::lm_head,request);v.wave=kernels::WaveSize::wave32;v.resources.workgroup_threads=32;
    v.resources.scratch_bytes=bytes;v.numerics={c.approved_numerical_identity,"checked-int32/group-f64","fp32-integer-RNE",true,true};
    v.qualification=kernels::QualificationStatus::experimental;v.graph_safe=false;v.input_layout="row-major-explicit-source-storage";
    v.output_layout="fp32-logits";v.compiler_identity="hip-gfx1151-experimental-tiled-integer-lm-head-v1";registry.add(std::move(v));
}
RuntimePlan RuntimePlan::bind(const CompiledSerialExecution& compiled,const ExecutionRuntimeResources& resources,const RuntimeConfig& c,const RuntimeOwners& owners){
    validate(c);require(compiled.execution_identity&&compiled.logical_rows==1&&compiled.request_concurrency==1&&compiled.vocab_size==c.vocabulary_size,
        "runtime projection requires matching compiled C1 execution");
    require(compiled.execution.steps.size()==compiled.bindings.bindings.size()&&resources.scratch&&resources.model_manifest&&resources.model_weights&&resources.tensor_access,
        "runtime projection requires compiled bindings and concrete model/scratch resources");
    RuntimePlan p;p.config_=c;p.execution_identity_=compiled.execution_identity;
    const ExecutionStep* selected=nullptr;
    for(const auto& step:compiled.execution.steps)if(step.op==SerialOp::lm_head){require(!selected,"runtime projection requires exactly one LM-head step");selected=&step;}
    require(selected&&selected->id<compiled.bindings.bindings.size()&&!selected->layer,"compiled runtime projection LM-head step invalid");
    p.step_id_=selected->id;const auto& binding=compiled.bindings.bindings[p.step_id_];
    require(binding.op==SerialOp::lm_head&&binding.step_id==p.step_id_&&!binding.host_or_runtime_step&&binding.kernel_id==runtime_kernel_id(c),
        "compiled runtime projection must bind its exact experimental kernel/profile");
    const ExecutionStepIo* io=nullptr;for(const auto& entry:compiled.io.steps)if(entry.step_id==p.step_id_){require(!io,"duplicate LM-head IO");io=&entry;}
    require(io&&io->op==SerialOp::lm_head&&io->reads.size()==1&&io->writes.size()==1&&io->temporaries.empty(),"runtime projection expects exactly one hidden read and logits write");
    ExecutionDispatchContext context;context.execution_identity=p.execution_identity_;context.step_id=p.step_id_;context.op=SerialOp::lm_head;
    context.kernel_id=*binding.kernel_id;context.reads=io->reads;context.writes=io->writes;context.arena=&compiled.arena;context.resources=&resources;
    p.memory_=*resources.scratch;p.input_=resolve_dispatch_read(context,0);p.output_=resolve_dispatch_write(context,0);
    require(p.memory_.execution_identity==p.execution_identity_,"runtime scratch belongs to another compiled execution");
    require(p.input_.use.buffer.buffer_class==ExecutionBufferClass::hidden&&p.output_.use.buffer.buffer_class==ExecutionBufferClass::logits,
        "runtime projection requires hidden/logits buffers");
    require(p.input_.bytes==multiply(c.hidden_size,c.input==RuntimeInput::bf16?2:4)&&p.output_.bytes==multiply(c.vocabulary_size,4),
        "runtime projection logical buffer storage must match exact declared source/logits dtype");
    const auto& access=resolve_dispatch_tensor_access(context,0);
    require(access.op==SerialOp::lm_head&&access.tensor_indices.size()==1,"runtime projection requires one authoritative output tensor");
    p.tensor_index_=access.tensor_indices[0];require(p.tensor_index_<resources.model_manifest->tensors.size(),"runtime projection tensor index out of range");
    p.tensor_=resources.model_manifest->tensors[p.tensor_index_];p.checkpoint_=resolve_dispatch_tensor(context,p.tensor_index_);
    require(p.tensor_.role==TensorRole::output_head&&p.tensor_.dimensions==std::vector<std::uint64_t>{c.hidden_size,c.vocabulary_size}&&
        p.tensor_.ggml_type==(c.weights==WeightEncoding::q4_0?2U:23U)&&p.tensor_.placement.kind==PlacementKind::local&&
        p.tensor_.placement.node==0&&p.checkpoint_.owner_node==0,"runtime projection exact local GGUF encoding/rank/geometry mismatch");
    require(p.checkpoint_.storage==RuntimeTensorStorage::resident||p.checkpoint_.storage==RuntimeTensorStorage::registered_file_mapped,
        "runtime projection requires original resident or registered GGUF blocks; repacked caches are not checkpoint bytes");
    const auto core=ExecutionPlan::build(tile_request(c,std::min(c.tile_columns,c.vocabulary_size),1));
    p.accounting_.checkpoint_bytes=multiply(c.vocabulary_size,core.accounting().source_checkpoint_bytes/core.request().shape.columns);
    require(p.tensor_.byte_size==p.accounting_.checkpoint_bytes&&p.checkpoint_.bytes==p.tensor_.byte_size,"runtime projection checkpoint extent differs from exact storage blocks");
    p.accounting_.staged_logits_bytes=multiply(c.vocabulary_size,4);p.accounting_.internal_peak_bytes=add(core.accounting().used_device_bytes,add(p.accounting_.staged_logits_bytes,4));
    require(p.accounting_.internal_peak_bytes<=c.maximum_internal_bytes,"runtime projection tile and full logits staging exceed admission ceiling");
    p.accounting_.tile_count=c.vocabulary_size/c.tile_columns+static_cast<std::uint64_t>(c.vocabulary_size%c.tile_columns!=0);
    p.accounting_.preparation_elements=multiply(p.accounting_.tile_count,c.hidden_size);p.accounting_.repacked_groups=multiply(c.vocabulary_size,c.hidden_size/32);
    p.manifest_identity_=resources.model_manifest->layout_identity;p.access_identity_=resources.tensor_access->plan_identity;p.content_identity_=resources.model_weights->model_content_identity;
    require(p.content_identity_.valid()&&p.manifest_identity_&&p.access_identity_&&resources.tensor_access->manifest_layout_identity==p.manifest_identity_,"runtime projection requires bound model content/layout/access identities");
    require(owners.arena&&owners.weights,"runtime projection requires real owning allocation pins");
    p.arena_owner_=owners.arena->current_binding();p.weight_owner_=owners.weights->current_binding();
    p.identity_="strix-integer-runtime-v1:execution="+std::to_string(p.execution_identity_)+":step="+std::to_string(p.step_id_)+":model="+content(p.content_identity_)+
        ":manifest="+std::to_string(p.manifest_identity_)+":access="+std::to_string(p.access_identity_)+":tensor="+std::to_string(p.tensor_index_)+
        ":weight-generation="+std::to_string(p.checkpoint_.allocation_generation)+":arena-generation="+std::to_string(p.memory_.allocation_generation)+":"+runtime_kernel_id(c)+":"+c.approved_numerical_identity;
    p.implementation_identity_=hash(p.identity_);(void)p.validate_dispatch(context,owners);return p;
}
ExecutionPlan RuntimePlan::tile(std::uint32_t first,std::uint64_t generation)const{
    require(first<config_.vocabulary_size&&first%config_.tile_columns==0,"runtime projection tile offset is not canonical");
    return ExecutionPlan::build(tile_request(config_,std::min(config_.tile_columns,config_.vocabulary_size-first),generation));
}
RuntimeSnapshot RuntimePlan::validate_dispatch(const ExecutionDispatchContext& context,const RuntimeOwners& owners)const{
    require(context.execution_identity==execution_identity_&&context.step_id==step_id_&&context.op==SerialOp::lm_head&&!context.layer&&
        context.kernel_id==runtime_kernel_id(config_)&&context.logical_rows==1&&context.request_concurrency==1&&context.reads_state==PersistentState::none&&
        context.writes_state==PersistentState::none&&context.reads.size()==1&&context.writes.size()==1&&context.temporaries.empty()&&
        context.reads[0]==input_.use&&context.writes[0]==output_.use,"runtime projection dispatch/phase/IO differs from approved compiled step");
    require(context.resources&&context.resources->scratch&&context.resources->model_manifest&&context.resources->model_weights&&context.resources->tensor_access,"runtime projection resources missing");
    const auto& r=*context.resources;const auto& m=*r.scratch;
    require(m.execution_identity==memory_.execution_identity&&m.arena_base==memory_.arena_base&&m.arena_bytes==memory_.arena_bytes&&m.allocation_generation==memory_.allocation_generation,
        "runtime projection scratch binding changed");
    require(r.model_manifest->layout_identity==manifest_identity_&&r.tensor_access->plan_identity==access_identity_&&
        r.tensor_access->manifest_layout_identity==manifest_identity_&&r.model_weights->model_content_identity==content_identity_&&
        tensor_index_<r.model_manifest->tensors.size()&&same_tensor(r.model_manifest->tensors[tensor_index_],tensor_),"runtime projection model content/layout changed");
    const auto& access=resolve_dispatch_tensor_access(context,0);require(access.op==SerialOp::lm_head&&access.tensor_indices==std::vector<std::uint32_t>{tensor_index_},"runtime projection tensor access changed");
    require(same_runtime(resolve_dispatch_tensor(context,tensor_index_),checkpoint_),"runtime projection checkpoint binding changed");
    const auto input=resolve_dispatch_read(context,0),output=resolve_dispatch_write(context,0);
    require(input.address==input_.address&&input.bytes==input_.bytes&&input.allocation_generation==input_.allocation_generation&&
        output.address==output_.address&&output.bytes==output_.bytes&&output.allocation_generation==output_.allocation_generation,"runtime projection arena view changed");
    require(owners.arena&&owners.weights,"runtime projection owning pins absent");RuntimeSnapshot s{owners.arena->current_binding(),owners.weights->current_binding(),0};
    require(same_physical(s.arena,arena_owner_)&&same_physical(s.weights,weight_owner_),"runtime projection owning allocation identity changed");
    view(s.arena,memory_.arena_base,memory_.arena_bytes,memory_.allocation_generation,config_.device);
    view(s.weights,checkpoint_.address,checkpoint_.bytes,checkpoint_.allocation_generation,config_.device);
    require(checkpoint_.address%2U==0,"runtime projection original half-scale checkpoint must be aligned to two bytes");
    require(input_.address%(config_.input==RuntimeInput::bf16?2U:4U)==0&&output_.address%4U==0,"runtime projection IO alignment mismatch");
    require(s.arena.identity!=s.weights.identity&&!(s.arena.base<s.weights.base+s.weights.bytes&&s.weights.base<s.arena.base+s.arena.bytes),"runtime projection model and writable arena alias");
    s.retained_bytes=add(add(s.arena.bytes,s.weights.bytes),accounting_.internal_peak_bytes);
    require(s.retained_bytes<=config_.maximum_retained_bytes,"runtime projection full retained allocations exceed explicit admission ceiling");
    validate_snapshot(owners,s);return s;
}
void RuntimePlan::validate_snapshot(const RuntimeOwners& owners,const RuntimeSnapshot& s)const{
    require(owners.arena&&owners.weights&&owners.arena->current_binding()==s.arena&&owners.weights->current_binding()==s.weights,"runtime projection input/output epochs changed during execution");
}
} // namespace strix::integer_projection
