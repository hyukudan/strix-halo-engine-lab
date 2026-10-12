#include "strix/hip_serial_graph.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/serial_prefix_cache.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace {
void check(hipError_t e,const char* operation){if(e!=hipSuccess)throw std::runtime_error(std::string(operation)+": "+hipGetErrorString(e));}
void expect(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F>void rejects(F f,const char* message){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}expect(rejected,message);}
strix::ModelContract model(){
    auto m=strix::ModelContract::qwen38_flash_next();m.vocab_size=128;m.hidden_size=128;m.max_position_embeddings=64;
    m.hidden_layers=2;m.full_attention_interval=2;m.full_attention_layers=1;m.gdn_layers=1;
    m.ple_layer_1based=1;m.ple_embed_dim=2560;m.ple_conv_kernel_size=4;m.ngram_size=3;m.heads_per_ngram=8;
    m.ngram_vocab_size_base=101;m.ple_eos_token_id=1;m.hc_count=4;m.hc_lowrank=16;
    m.attention_heads=1;m.kv_heads=1;m.head_dim=256;m.partial_rotary_factor=0.5F;
    m.gdn_key_heads=1;m.gdn_key_head_dim=128;m.gdn_value_heads=2;m.gdn_value_head_dim=128;m.gdn_conv_kernel=4;
    m.indexer_n_heads=2;m.indexer_kv_heads=1;m.indexer_head_dim=128;m.indexer_compress_ratio=4;m.indexer_budget=8;
    m.moe_intermediate_size=32;m.shared_expert_intermediate_size=32;m.num_experts=4;m.num_experts_per_token=2;
    expect(m.validate().empty(),"complete graph fixture model must satisfy existing geometry contracts");return m;
}
struct TensorData {
    strix::ModelManifest manifest;
    std::vector<std::vector<std::byte>> data;
    void add(std::string name,strix::TensorRole role,std::vector<std::uint64_t> dims,std::optional<std::uint32_t> layer={},float scale=0.02F,bool bf16=false){
        std::uint64_t count=1;for(auto d:dims)count*=d;expect(count<=4ULL*1024*1024,"bounded synthetic graph tensor");
        strix::ManifestTensor t;t.name=std::move(name);t.role=role;t.layer=layer;t.dimensions=dims;t.ggml_type=bf16?30U:0U;
        t.byte_size=count*(bf16?2U:4U);t.absolute_offset=manifest.total_tensor_bytes;
        manifest.total_tensor_bytes+=t.byte_size;manifest.tensors.push_back(t);data.emplace_back(static_cast<std::size_t>(t.byte_size));
        auto& payload=data.back();
        if(bf16){for(std::size_t i=0;i<count;++i){const auto v=static_cast<std::uint16_t>(0x3d00U+static_cast<unsigned>(i%47U));std::memcpy(payload.data()+i*2,&v,2);}}
        else {std::vector<float> values(static_cast<std::size_t>(count),0.0F);
            if(dims.size()==1||t.name.find("norm")!=std::string::npos||t.name.find("conv1d")!=std::string::npos||t.name.find("ssm_a")!=std::string::npos){
                std::fill(values.begin(),values.end(),scale);
            }else{const auto width=static_cast<std::size_t>(dims[0]);for(std::size_t row=0;row<values.size()/width;++row){
                values[row*width+(row*7+3)%width]=scale*static_cast<float>(row%7+1);
                values[row*width+(row*11+17)%width]+=scale*0.25F;}}
            std::memcpy(payload.data(),values.data(),payload.size());}
    }
};
TensorData tensors(const strix::ModelContract& m){
    TensorData d;d.manifest.architecture="qwen4_exp";const auto h=m.hidden_size,e=h*m.hc_count,r=m.hc_lowrank;
    d.add("token_embd.weight",strix::TensorRole::embedding,{h,m.vocab_size},{},0.03F);
    const auto hash=strix::build_ple_hash_contract(m);
    d.add("per_layer_token_embd.weight",strix::TensorRole::ngram_table,{m.ple_head_dim(),hash.minimum_table_rows},{},0.02F,true);
    d.add("blk.0.ple_key.weight",strix::TensorRole::ple_parameter,{m.ple_embed_dim,e},0,0.002F);
    d.add("blk.0.ple_value.weight",strix::TensorRole::ple_parameter,{m.ple_embed_dim,h},0,0.002F);
    for(const auto* name:{"ple_norm_key.weight","ple_norm_query.weight","ple_norm_conv.weight"})d.add("blk.0."+std::string(name),strix::TensorRole::ple_parameter,{h,m.hc_count},0,0.01F);
    d.add("blk.0.ple_conv1d.weight",strix::TensorRole::ple_parameter,{m.ple_conv_kernel_size,e},0,0.05F);
    for(std::uint32_t layer=0;layer<2;++layer){const auto prefix="blk."+std::to_string(layer)+".";
        for(const auto* sub:{"hc_attn","hc_ffn"}){
            d.add(prefix+sub+"_norm.weight",strix::TensorRole::normalization,{e},layer,0.01F);
            d.add(prefix+sub+"_down.weight",strix::TensorRole::dense_projection,{e,r},layer,0.01F);
            d.add(prefix+sub+"_up.weight",strix::TensorRole::dense_projection,{r,e},layer,0.01F);
            d.add(prefix+sub+"_inject.weight",strix::TensorRole::dense_projection,{e,m.hc_count},layer,0.01F);
        }
        d.add(prefix+"ffn_gate_inp.weight",strix::TensorRole::moe_router,{h,m.num_experts},layer,0.02F);
        d.add(prefix+"ffn_gate_exps.weight",strix::TensorRole::moe_expert,{h,m.moe_intermediate_size,m.num_experts},layer,0.015F);
        d.add(prefix+"ffn_up_exps.weight",strix::TensorRole::moe_expert,{h,m.moe_intermediate_size,m.num_experts},layer,0.02F);
        d.add(prefix+"ffn_down_exps.weight",strix::TensorRole::moe_expert,{m.moe_intermediate_size,h,m.num_experts},layer,0.02F);
        d.add(prefix+"shared_expert.gate_proj.weight",strix::TensorRole::dense_projection,{h,m.shared_expert_intermediate_size},layer,0.01F);
        d.add(prefix+"shared_expert.up_proj.weight",strix::TensorRole::dense_projection,{h,m.shared_expert_intermediate_size},layer,0.02F);
        d.add(prefix+"shared_expert.down_proj.weight",strix::TensorRole::dense_projection,{m.shared_expert_intermediate_size,h},layer,0.02F);
        d.add(prefix+"shared_expert_gate.weight",strix::TensorRole::dense_projection,{h,1},layer,0.01F);
    }
    const auto key=m.gdn_key_heads*m.gdn_key_head_dim,value=m.gdn_value_heads*m.gdn_value_head_dim,conv=key*2+value;
    d.add("blk.0.attn_qkv.weight",strix::TensorRole::attention_projection,{h,conv},0,0.03F);
    d.add("blk.0.attn_gate.weight",strix::TensorRole::attention_projection,{h,value},0,0.02F);
    d.add("blk.0.ssm_conv1d.weight",strix::TensorRole::recurrent_state,{m.gdn_conv_kernel,conv},0,0.05F);
    d.add("blk.0.ssm_dt.bias",strix::TensorRole::recurrent_state,{m.gdn_value_heads},0,0.03F);
    d.add("blk.0.ssm_a",strix::TensorRole::recurrent_state,{m.gdn_value_heads},0,-0.25F);
    d.add("blk.0.ssm_beta.weight",strix::TensorRole::recurrent_state,{h,m.gdn_value_heads},0,0.01F);
    d.add("blk.0.ssm_alpha.weight",strix::TensorRole::recurrent_state,{h,m.gdn_value_heads},0,0.01F);
    d.add("blk.0.ssm_norm.weight",strix::TensorRole::recurrent_state,{m.gdn_value_head_dim},0,0.01F);
    d.add("blk.0.ssm_out.weight",strix::TensorRole::recurrent_state,{value,h},0,0.02F);
    d.add("blk.1.attn_q.weight",strix::TensorRole::attention_projection,{h,m.attention_heads*m.head_dim*2},1,0.02F);
    d.add("blk.1.attn_k.weight",strix::TensorRole::attention_projection,{h,m.kv_heads*m.head_dim},1,0.02F);
    d.add("blk.1.attn_v.weight",strix::TensorRole::attention_projection,{h,m.kv_heads*m.head_dim},1,0.03F);
    d.add("blk.1.attn_output.weight",strix::TensorRole::attention_projection,{m.attention_heads*m.head_dim,h},1,0.02F);
    for(const auto* name:{"attn_q_norm.weight","attn_k_norm.weight"})d.add("blk.1."+std::string(name),strix::TensorRole::normalization,{m.head_dim},1,0.01F);
    d.add("blk.1.indexer.q_proj.weight",strix::TensorRole::sparse_index,{h,m.indexer_n_heads*m.indexer_head_dim},1,0.02F);
    d.add("blk.1.indexer.k_proj.weight",strix::TensorRole::sparse_index,{h,m.indexer_head_dim},1,0.02F);
    for(const auto* name:{"indexer.q_norm.weight","indexer.k_norm.weight"})d.add("blk.1."+std::string(name),strix::TensorRole::sparse_index,{m.indexer_head_dim},1,0.01F);
    d.add("output_hc_norm.weight",strix::TensorRole::normalization,{e},{},0.01F);
    d.add("output_hc_down.weight",strix::TensorRole::dense_projection,{e,r},{},0.01F);
    d.add("output_hc_up.weight",strix::TensorRole::dense_projection,{r,e},{},0.01F);
    d.add("output.weight",strix::TensorRole::output_head,{h,m.vocab_size},{},0.02F);
    d.manifest.layout_identity=strix::model_manifest_layout_identity(d.manifest);return d;
}
strix::CompiledSerialExecution compile(const strix::ModelContract& m,bool graph){
    strix::ExecutionBindingRequest request;request.minimum_qualification=strix::kernels::QualificationStatus::experimental;
    request.require_graph_safe=graph;request.device_graph_inputs=graph;
    strix::HipLauncherCatalogConfig config;config.enable_device_graph_inputs=graph;strix::kernels::KernelRegistry registry;
    strix::register_complete_hip_kernel_catalog(m,request,config,registry);
    return strix::compile_serial_execution(m,registry,request,strix::ExecutionBufferRequest{});
}
struct DeviceModel {
    TensorData fixture;
    strix::ModelRuntimeBinding binding;
    std::vector<void*> allocations;
    explicit DeviceModel(TensorData data):fixture(std::move(data)){
        binding.manifest_layout_identity=fixture.manifest.layout_identity;binding.model_content_identity.bytes[0]=91;
        try{for(std::size_t i=0;i<fixture.data.size();++i){void* p=nullptr;check(hipMalloc(&p,fixture.data[i].size()),"complete fixture weight allocation");allocations.push_back(p);
            check(hipMemcpy(p,fixture.data[i].data(),fixture.data[i].size(),hipMemcpyHostToDevice),"complete fixture weight upload");
            binding.tensors.push_back({static_cast<std::uint32_t>(i),strix::RuntimeTensorStorage::resident,reinterpret_cast<std::uintptr_t>(p),fixture.data[i].size(),100+i,0});}}
        catch(...){if(hipDeviceSynchronize()==hipSuccess)for(auto* p:allocations)(void)hipFree(p);throw;}
        strix::validate_model_runtime_binding(fixture.manifest,binding);
    }
    ~DeviceModel(){int current=-1;if(hipGetDevice(&current)==hipSuccess&&current==0&&hipDeviceSynchronize()==hipSuccess)for(auto* p:allocations)(void)hipFree(p);}
};
// Eager control gathers the real resident BF16 table with actual HIP D2H copies;
// captured execution gathers those same resident bytes directly on the device.
class ResidentLookup final:public strix::PleLookupBackend {
public:
    std::uint64_t implementation_identity()const noexcept override{return 0x47525048544c4f4fULL;}
    strix::PleLookupHandle start(const strix::PleLookupRequest& r)override{
        expect(!active&&r.table_runtime&&r.table_manifest&&r.table_manifest->ggml_type==30&&r.row_bytes,"resident fixture lookup request");
        table=r.table_runtime->address;row_bytes=r.row_bytes;rows.assign(r.row_ids.begin(),r.row_ids.end());
        for(auto row:rows){expect(row<r.table_runtime->bytes/row_bytes,"resident fixture lookup bounds");}
        active=true;return 1;
    }
    void finish(strix::PleLookupHandle handle,std::span<std::byte> out)override{
        expect(active&&handle==1&&out.size()>=rows.size()*row_bytes,"resident fixture lookup completion");
        for(std::size_t i=0;i<rows.size();++i){check(hipMemcpy(out.data()+i*row_bytes,reinterpret_cast<const void*>(table+rows[i]*row_bytes),
            static_cast<std::size_t>(row_bytes),hipMemcpyDeviceToHost),"resident table eager gather");}
        active=false;
    }
    void cancel(strix::PleLookupHandle)noexcept override{active=false;}
private:
    bool active=false;std::uintptr_t table=0;std::uint64_t row_bytes=0;std::vector<std::uint64_t> rows;
};
struct Owner final:strix::HipSerialGraphResources {
    std::shared_ptr<DeviceModel> model_weights;
    strix::CompiledSerialExecution compiled;
    strix::SerialSessionStateLayout layout;
    strix::HipRuntimeAllocator allocator;
    strix::AllocationGenerationSource generations;
    strix::RuntimeExecutionMemoryOwner allocation;
    strix::ExecutionMemoryBinding scratch;
    strix::SessionStateMemoryBinding state;
    strix::ExecutionTensorAccessPlan access;
    strix::ExecutionRuntimeResources resources;
    ResidentLookup lookup;
    Owner(const strix::ModelContract& m,std::shared_ptr<DeviceModel> weights,bool graph):model_weights(std::move(weights)),compiled(compile(m,graph)){
        strix::SerialSessionStateLayoutRequest request;request.context_tokens=16;layout=strix::build_serial_session_state_layout(m,request);
        allocation=strix::RuntimeExecutionMemoryOwner::allocate(compiled,layout,allocator,generations);scratch=allocation.scratch();state=allocation.state();
        hipDeviceptr_t native_base=nullptr;std::size_t native_bytes=0;
        check(hipMemGetAddressRange(&native_base,&native_bytes,reinterpret_cast<hipDeviceptr_t>(scratch.arena_base)),"fixture whole scratch backing extent");
        expect(reinterpret_cast<std::uintptr_t>(native_base)==scratch.arena_base,"fixture scratch begins at its full native allocation");scratch.arena_bytes=native_bytes;
        check(hipMemGetAddressRange(&native_base,&native_bytes,reinterpret_cast<hipDeviceptr_t>(state.device_base)),"fixture whole state backing extent");
        expect(reinterpret_cast<std::uintptr_t>(native_base)==state.device_base,"fixture state begins at its full native allocation");state.device_bytes=native_bytes;
        strix::initialize_hip_fresh_runtime_memory(scratch,state);
        const auto& manifest=model_weights->fixture.manifest;const auto index=strix::build_model_tensor_index(manifest);
        const auto requirements=strix::build_serial_execution_tensor_requirements(compiled.execution);
        access=strix::compile_execution_tensor_access_plan(compiled.execution,manifest,index,requirements);
        resources.scratch=&scratch;resources.model_manifest=&manifest;resources.model_weights=&model_weights->binding;
        resources.tensor_access=&access;resources.session_state_layout=&layout;resources.session_state_memory=&state;
        if(!graph)resources.ple_lookup=&lookup;
    }
    const strix::ExecutionRuntimeResources& current()const override{return resources;}
    strix::SerialSessionContext session()const{
        strix::SerialSessionContext s;s.execution_identity=compiled.execution_identity;s.state_layout_identity=layout.layout_identity;
        s.runtime_compatibility_identity=strix::build_runtime_execution_identities(compiled.execution_identity,compiled.arena,resources).compatibility_identity;
        for(const auto& step:compiled.execution.steps)s.state.valid=s.state.valid|step.reads_state|step.writes_state;return s;
    }
    std::vector<std::byte> device_state()const{std::vector<std::byte> bytes(static_cast<std::size_t>(state.device_bytes));
        check(hipMemcpy(bytes.data(),reinterpret_cast<const void*>(state.device_base),bytes.size(),hipMemcpyDeviceToHost),"full persistent device state readback");return bytes;}
    std::vector<std::byte> host_state()const{const auto* begin=reinterpret_cast<const std::byte*>(state.host_base);return {begin,begin+state.host_bytes};}
    std::vector<float> logits()const{const auto address=strix::resolve_execution_buffer(compiled.execution_identity,compiled.arena,scratch,compiled.io.steps.back().writes[0]);
        std::vector<float> values(static_cast<std::size_t>(compiled.vocab_size));check(hipMemcpy(values.data(),reinterpret_cast<const void*>(address.address),values.size()*4,hipMemcpyDeviceToHost),"full logits readback");return values;}
};
void validate_fixture(){
    const auto m=model();const auto data=tensors(m);const auto index=strix::build_model_tensor_index(data.manifest);
    for(bool graph:{false,true}){const auto compiled=compile(m,graph);
        const auto access=strix::compile_execution_tensor_access_plan(compiled.execution,data.manifest,index,strix::build_serial_execution_tensor_requirements(compiled.execution));
        expect(!access.accesses.empty()&&compiled.bindings.missing_steps==0,"complete real graph/eager tensor/variant fixture must resolve");
        if(graph)expect(compiled.device_graph_inputs&&compiled.whole_plan_graph_safe,"complete graph topology must be explicitly compiled");}
    std::cout<<"Complete synthetic native fixture host contracts valid; no GPU work performed.\n";
}
void run(){
    const auto m=model();auto weights=std::make_shared<DeviceModel>(tensors(m));
    auto eager=std::make_shared<Owner>(m,weights,false),captured=std::make_shared<Owner>(m,weights,true);
    auto eager_catalog=std::make_shared<strix::HipLauncherCatalog>(m);
    strix::HipLauncherCatalogConfig config;config.enable_device_graph_inputs=true;auto graph_catalog=std::make_shared<strix::HipLauncherCatalog>(m,config);
    strix::ExecutionLauncherRegistry eager_registry;eager_catalog->register_for_compiled_execution(eager->compiled,eager_registry);
    expect(eager_catalog->audit(eager->compiled,eager_registry).missing_steps==0,"actual eager catalog covers every operator");
    const auto launches=strix::compile_launch_plan(eager->compiled,eager_registry);strix::DirectLaunchDispatcher eager_dispatch(launches);
    strix::GraphMemoryRequest memory;memory.workspace_bytes_per_instance=captured->scratch.arena_bytes+65536;
    memory.executable_bytes_per_instance=4ULL*1024*1024;memory.capture_peak_bytes=4ULL*1024*1024;
    memory.budget_bytes=memory.workspace_bytes_per_instance+16ULL*1024*1024;
    auto insufficient=memory;insufficient.workspace_bytes_per_instance=captured->scratch.arena_bytes;
    rejects([&]{strix::HipSerialGraph denied(captured->compiled,graph_catalog,captured,insufficient);},
        "native graph admission must retain whole scratch plus auxiliary capacity");
    const auto scalar=std::find_if(weights->fixture.manifest.tensors.begin(),weights->fixture.manifest.tensors.end(),
        [](const auto& tensor){return tensor.name=="blk.0.ssm_a";});
    expect(scalar!=weights->fixture.manifest.tensors.end(),"known small native model tensor negative control");
    const auto scalar_index=static_cast<std::size_t>(scalar-weights->fixture.manifest.tensors.begin());
    const auto original_scalar_address=weights->binding.tensors[scalar_index].address;
    for(const auto address:{captured->scratch.arena_base+64,captured->state.device_base+64}){
        weights->binding.tensors[scalar_index].address=address;
        rejects([&]{strix::HipSerialGraph denied(captured->compiled,graph_catalog,captured,memory);},
            "resident model view cannot alias actual writable scratch/state");
    }
    hipDeviceptr_t scalar_base=nullptr;std::size_t scalar_bytes=0;
    check(hipMemGetAddressRange(&scalar_base,&scalar_bytes,reinterpret_cast<hipDeviceptr_t>(original_scalar_address)),"negative control native model extent");
    weights->binding.tensors[scalar_index].address=reinterpret_cast<std::uintptr_t>(scalar_base)+scalar_bytes-4;
    rejects([&]{strix::HipSerialGraph denied(captured->compiled,graph_catalog,captured,memory);},
        "declared model view exceeding native allocation cannot capture");
    weights->binding.tensors[scalar_index].address=original_scalar_address;
    strix::HipSerialGraph graph(captured->compiled,graph_catalog,captured,memory);
    auto es=eager->session(),gs=captured->session();
    expect(eager->device_state()==captured->device_state()&&eager->host_state()==captured->host_state(),"capture must not execute or mutate persistent state");
    const auto hash=strix::build_ple_hash_contract(m);std::vector<std::uint32_t> history(2,1);
    const std::array<std::uint32_t,12> tokens{2,3,2,5,1,4,6,6,7,2,3,5};
    std::optional<strix::SerialPrefixCacheEntry> graph_prefix;
    strix::HipSessionSnapshotTransferBackend snapshot_transfer;
    std::vector<float> first_logits;bool varied=false;std::vector<std::byte> initial=eager->device_state();
    for(std::size_t position=0;position<tokens.size();++position){
        const auto er=strix::execute_serial_token(eager->compiled,es,eager->resources,tokens[position],eager_dispatch);
        check(hipDeviceSynchronize(),"actual complete eager token completion");
        const auto gr=graph.execute(gs,tokens[position]);
        if (position == 3) {
            graph_prefix = strix::SerialPrefixCacheEntry::capture(m,
                std::span<const std::uint32_t>(tokens).first(4), gs, captured->layout,
                captured->state, captured->allocator, captured->generations, snapshot_transfer);
            check(hipDeviceSynchronize(), "graph prefix snapshot completion");
        }
        expect(er.completed_steps==eager->compiled.execution.steps.size()&&gr.completed_steps==captured->compiled.execution.steps.size(),"both real DAGs complete all steps");
        const auto a=eager->logits(),b=captured->logits();expect(a.size()==b.size(),"whole logits dimensions");
        for(std::size_t i=0;i<a.size();++i){expect(std::isfinite(a[i])&&std::bit_cast<std::uint32_t>(a[i])==std::bit_cast<std::uint32_t>(b[i]),"graph/eager whole logits must match bits");}
        if(position==0)first_logits=a;else varied=varied||a!=first_logits;
        expect(eager->device_state()==captured->device_state(),"all KV/recurrent/convolution/index persistent device bytes match");
        expect(eager->host_state()==captured->host_state(),"all host history/position bytes match");
        expect(es.next_position==gs.next_position&&es.committed_tokens==gs.committed_tokens&&es.state.valid==gs.state.valid&&es.state.generations==gs.state.generations&&
            es.health==strix::SerialSessionHealth::ready&&gs.health==strix::SerialSessionHealth::ready,"metadata commits and state generations agree");
        const auto result=strix::hash_ple_token(hash,tokens[position],history);history.assign(result.next_context.begin(),result.next_context.begin()+result.context_count);
        const auto state=strix::resolve_session_state(captured->layout,captured->state,strix::SessionStateComponent::ngram_history);
        expect(std::memcmp(reinterpret_cast<const void*>(state.address),history.data(),history.size()*4)==0,"GPU PLE history agrees with independent CPU hash contract");
        const auto where=strix::resolve_session_state(captured->layout,captured->state,strix::SessionStateComponent::position_history);std::uint64_t recorded=UINT64_MAX;
        std::memcpy(&recorded,reinterpret_cast<const void*>(where.address),8);expect(recorded==position,"host position is current committed absolute position");
    }
    expect(varied&&eager->device_state()!=initial,"fixture executes nonzero token/state work, not constant outputs");
    const auto full_device = captured->device_state(), full_host = captured->host_state();
    const auto full_logits = captured->logits();
    const auto full_metadata = gs;
    expect(graph_prefix && graph_prefix->valid(), "native graph prefix cache must retain recorded lineage");
    graph_prefix->restore(gs, m, captured->layout, captured->state, snapshot_transfer,
        std::span<const std::uint32_t>(tokens).first(4));
    check(hipDeviceSynchronize(), "graph prefix restore completion");
    expect(gs.next_position == 4 && gs.committed_tokens == 4 &&
        gs.health == strix::SerialSessionHealth::ready, "native graph restore publishes exact prefix metadata");
    for (std::size_t position = 4; position < tokens.size(); ++position) {
        (void)graph.execute(gs, tokens[position]);
    }
    const auto replay_logits = captured->logits();
    expect(replay_logits.size() == full_logits.size(), "restored graph logits shape");
    for (std::size_t i = 0; i < full_logits.size(); ++i)
        expect(std::bit_cast<std::uint32_t>(replay_logits[i]) == std::bit_cast<std::uint32_t>(full_logits[i]),
            "restored same-plan graph final logits must match every bit");
    expect(captured->device_state() == full_device && captured->host_state() == full_host,
        "restored graph continuation preserves every KV/recurrent/convolution/index/history byte");
    expect(gs.next_position == full_metadata.next_position && gs.committed_tokens == full_metadata.committed_tokens &&
        gs.state.valid == full_metadata.state.valid && gs.state.generations == full_metadata.state.generations &&
        strix::arithmetic_lineage_matches(gs.arithmetic_lineage, full_metadata.arithmetic_lineage, gs.next_position),
        "restored graph continuation preserves state generations and actual arithmetic history");
    const auto prior=captured->device_state(),host=captured->host_state();const auto next=gs.next_position;
    ++captured->scratch.allocation_generation;rejects([&]{(void)graph.execute(gs,2);},"stale scratch generation rejected before replay");--captured->scratch.allocation_generation;
    ++weights->binding.tensors[0].allocation_generation;rejects([&]{(void)graph.execute(gs,2);},"stale model generation rejected before replay");--weights->binding.tensors[0].allocation_generation;
    rejects([&]{(void)graph.execute(gs,128);},"out-of-vocabulary token cannot replay");
    gs.next_position=16;rejects([&]{(void)graph.execute(gs,2);},"out-of-context position cannot replay");gs.next_position=next;
    expect(!graph.failed()&&gs.health==strix::SerialSessionHealth::ready&&captured->device_state()==prior&&captured->host_state()==host,"deterministic preflight rejects do not execute/poison state");
    // Genuine native nonfinite-model error must poison rather than publish host
    // history/session commits, even though some device state may already change.
    const auto output_index=weights->fixture.manifest.tensors.size()-1;const float nan=std::numeric_limits<float>::quiet_NaN();
    check(hipMemcpy(weights->allocations[output_index],&nan,4,hipMemcpyHostToDevice),"nonfinite actual model weight negative control");
    rejects([&]{(void)graph.execute(gs,2);},"native nonfinite logits must fail graph execution");
    expect(graph.failed()&&gs.health==strix::SerialSessionHealth::poisoned&&gs.next_position==next&&gs.committed_tokens==tokens.size(),"failed native graph poisons without committing token metadata");
    expect(captured->host_state()==host,"failed native graph cannot publish host ngram/position history");
    rejects([&]{(void)graph.execute(gs,2);},"poisoned graph cannot replay");expect(graph.drain(),"completed failed graph retains quiescence proof");
    std::cout<<"Actual complete synthetic serial graph/eager native comparison passed; no model qualification.\n";
}
}
int main(int argc,char** argv)try{
    if(argc==2&&std::string_view(argv[1])=="--validate-fixture"){validate_fixture();return 0;}
    if(argc!=1)throw std::invalid_argument("usage: strix-hip-serial-graph-tests [--validate-fixture]");
    int count=0;if(hipGetDeviceCount(&count)!=hipSuccess||count==0){std::cout<<"SKIP: gfx1151 unavailable\n";return 77;}
    check(hipSetDevice(0),"full graph fixture device");hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0),"full graph fixture properties");
    const std::string architecture=properties.gcnArchName;if(architecture.substr(0,architecture.find(':'))!="gfx1151"){std::cout<<"SKIP: gfx1151 unavailable\n";return 77;}
    run();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
