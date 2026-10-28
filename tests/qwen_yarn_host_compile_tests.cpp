#include "../apps/qwen_prefill_plan.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/qwen_yarn_table.hpp"

#include <fstream>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

// Metadata only: no weights, device allocations, model file or external assets.
ModelManifest canonical_manifest(const ModelContract& model) {
    ModelManifest manifest;
    manifest.architecture = "qwen3next";
    std::uint64_t offset = 256;
    const auto add = [&](std::optional<std::uint32_t> layer, std::string name,
                         TensorRole role, std::vector<std::uint64_t> shape, unsigned type = 8U) {
        ManifestTensor t;
        t.name = (layer ? "blk." + std::to_string(*layer) + "." : "") + name + ".weight";
        t.layer = layer; t.role = role; t.dimensions = std::move(shape); t.ggml_type = type;
        std::uint64_t elements = 1;
        for (auto d : t.dimensions) elements *= d;
        t.byte_size = type == 8U ? elements / 32U * 34U : type == 2U ? elements / 32U * 18U : type == 12U ? elements / 256U * 144U :
            type == 13U ? elements / 256U * 176U : type == 7U ? elements / 32U * 24U : elements * 4U;
        t.absolute_offset = offset;
        offset += (t.byte_size + 255U) / 256U * 256U;
        manifest.total_tensor_bytes += t.byte_size;
        manifest.tensors.push_back(std::move(t));
    };
    add({}, "token_embd", TensorRole::embedding, {2560,model.vocab_size});
    const auto hash = build_ple_hash_contract(model);
    add({}, "per_layer_token_embd", TensorRole::ngram_table, {hash.head_dim,hash.minimum_table_rows},2U);
    for (unsigned l = 0; l < 48U; ++l) {
        if (l+1U==model.ple_layer_1based) {
            add(l,"ple_key",TensorRole::ple_parameter,{2560,10240});
            add(l,"ple_value",TensorRole::ple_parameter,{2560,2560});
            add(l,"ple_norm_key",TensorRole::ple_parameter,{10240},0);
            add(l,"ple_norm_query",TensorRole::ple_parameter,{10240},0);
            add(l,"ple_norm_conv",TensorRole::ple_parameter,{10240},0);
            add(l,"ple_conv1d",TensorRole::ple_parameter,{4,10240},0);
        }
        for (const auto* prefix : {"hc_attn_","hc_ffn_"}) {
            add(l,std::string(prefix)+"norm",TensorRole::normalization,{10240},0);
            add(l,std::string(prefix)+"down",TensorRole::dense_projection,{10240,320});
            add(l,std::string(prefix)+"up",TensorRole::dense_projection,{320,10240});
            add(l,std::string(prefix)+"inject",TensorRole::dense_projection,{10240,4},0);
        }
        add(l,"ffn_gate_inp",TensorRole::moe_router,{2560,512},0);
        const bool q80 = l==2U || l==4U || l==30U || l==46U || l==47U;
        for (const auto* name : {"ffn_gate_exps","ffn_up_exps"})
            add(l,name,TensorRole::moe_expert,{2560,640,512},l==2U ? 13U : 12U);
        add(l,"ffn_down_exps",TensorRole::moe_expert,{640,2560,512},q80 ? 8U : 7U);
        add(l,"ffn_gate_shexp",TensorRole::dense_projection,{2560,640});
        add(l,"ffn_up_shexp",TensorRole::dense_projection,{2560,640});
        add(l,"ffn_down_shexp",TensorRole::dense_projection,{640,2560});
        add(l,"ffn_gate_inp_shexp",TensorRole::dense_projection,{2560},0);
        if ((l+1U)%4U) {
            add(l,"attn_qkv",TensorRole::attention_projection,{2560,10240});
            add(l,"attn_gate",TensorRole::attention_projection,{2560,6144});
            add(l,"ssm_beta",TensorRole::recurrent_state,{2560,48},0);
            add(l,"ssm_alpha",TensorRole::recurrent_state,{2560,48},0);
            add(l,"ssm_out",TensorRole::recurrent_state,{6144,2560});
        } else {
            add(l,"attn_q",TensorRole::attention_projection,{2560,12288});
            add(l,"attn_k",TensorRole::attention_projection,{2560,512});
            add(l,"attn_v",TensorRole::attention_projection,{2560,512});
            add(l,"attn_output",TensorRole::attention_projection,{6144,2560});
            add(l,"attn_q_norm",TensorRole::normalization,{256},0);
            add(l,"attn_k_norm",TensorRole::normalization,{256},0);
            add(l,"indexer.q_proj",TensorRole::sparse_index,{2560,512});
            add(l,"indexer.k_proj",TensorRole::sparse_index,{2560,128});
        }
    }
    add({},"output_hc_norm",TensorRole::normalization,{10240},0);
    add({},"output_hc_down",TensorRole::dense_projection,{10240,320});
    add({},"output_hc_up",TensorRole::dense_projection,{320,10240});
    add({},"output",TensorRole::output_head,{2560,model.vocab_size});
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    return manifest;
}

void descriptors(const ModelContract& model, const HipLauncherCatalogConfig& config,
                 const CompiledSerialExecution& serial, const ModelManifest& manifest,
                 const ExecutionTensorAccessPlan& access, unsigned capacity) {
    // Non-dereferenceable address ranges exercise the real metadata/range
    // preflights. They are never allocated, read, uploaded or dispatched.
    ExecutionMemoryBinding scratch{serial.execution_identity,UINT64_C(1)<<44U,serial.arena.total_bytes,1};
    ModelRuntimeBinding weights;
    weights.manifest_layout_identity = manifest.layout_identity;
    weights.model_content_identity.bytes[0] = 1;
    for (std::size_t i=0;i<manifest.tensors.size();++i) {
        const auto& t = manifest.tensors[i];
        weights.tensors.push_back({static_cast<std::uint32_t>(i),RuntimeTensorStorage::resident,
            (UINT64_C(2)<<44U)+t.absolute_offset,t.byte_size,1,0});
    }
    SerialSessionStateLayoutRequest request;
    request.context_tokens = capacity;
    const auto state = build_serial_session_state_layout(model,request);
    SessionStateMemoryBinding state_memory{state.layout_identity,UINT64_C(3)<<44U,state.device_total_bytes,1,
        UINT64_C(4)<<44U,state.host_total_bytes,1};
    ExecutionRuntimeResources resources;
    resources.scratch = &scratch;
    resources.model_manifest = &manifest;
    resources.model_weights = &weights;
    resources.tensor_access = &access;
    resources.session_state_layout = &state;
    resources.session_state_memory = &state_memory;
    HipLauncherCatalog catalog(model,config);
    ExecutionLauncherRegistry launchers;
    catalog.register_for_compiled_execution(serial,launchers,&resources,{},false);
    require(!catalog.audit(serial,launchers).missing_steps,"runtime catalog coverage");
    const auto plan = compile_launch_plan(serial,launchers);
    require(plan.steps.size()==serial.execution.steps.size(),"runtime descriptor count");
    for (const auto& step : plan.steps) require(step.preflight != nullptr,"runtime deep preflight descriptor");
}

void run(unsigned factor, unsigned capacity, unsigned count, const ModelManifest& manifest) {
    auto model = ModelContract::qwen38_flash_next();
    // The CLI resolves this identity before upload. The opaque CPU owner below
    // is used only when constructing launch descriptors, never dispatched.
    const auto rotary = RotaryDescriptor::yarn(factor);
    Sha256Fast hash;
    generate_qwen_yarn_rows(rotary,0,rotary.effective_max(),[&](auto,auto rows) { hash.update(std::as_bytes(rows)); });
    model.resolved_rotary = RotaryIdentity(rotary,hash.finalize());
    const std::string path = "qwen-yarn-host-" + std::to_string(count) + ".ids";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    { std::ofstream prompt(path); for (unsigned i=0;i<count;++i) prompt << (i%model.vocab_size) << '\n'; }
    std::vector<std::string> args{"decoder","MODEL","--prompt-ids",path,"--rope-yarn",std::to_string(factor),
        "--context",std::to_string(capacity),"--generate","32"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    RequestedOptions request;
    request.options = parse_options(static_cast<int>(argv.size()),argv.data());
    require(request.options.tokens.size()==count && request.environment.empty(),"parsed prompt and clean environment");
    PrefillMemory memory;
    memory.total = memory.available = memory.gtt_limit = UINT64_C(124)<<30U;
    memory.gtt_used = 0;
    // Simulated verified credit isolates host compilation from OS reclaim.
    memory.verify_ple_credit = [](const Options&) { return UINT64_C(28)<<30U; };
    memory.estimate = [&](const Options& o) { return compile_prompt_requirements(o,model,manifest,111334654400ULL); };
    const auto effective = resolve_effective_config(request,model,{},manifest,
        validated_model_content_identity,{true,true,true},memory);
    require(effective.admitted,effective.error);
    validate_effective_options(effective.options,model);
    require(yarn_environment_refusal(effective.environment,factor).empty(),"effective environment classification");
    require(!effective.options.mtp.enabled && effective.admission.at("owner_target_kv_bytes")>0,
        "complete admission inventory without MTP");
    validate_wave_router_model_inputs({});
    const auto layers = moe_q4k_a8::classify_runtime_manifest(model,manifest);
    require(!layers.q4k_with_q51_down_layers.empty(),"grouped Q5_1 down admission");
    moe_q4k_a8::validate_slot_parallel_manifest(model,layers);
    ModelContentIdentity content;
    for (std::size_t i=0;i<content.bytes.size();++i)
        content.bytes[i] = static_cast<unsigned char>(std::stoul(std::string(validated_model_content_identity.substr(i*2U,2U)),nullptr,16));
    moe_q4k_a8::validate_a8_down_content(content);
    moe_q4k_a8::validate_a8_down_manifest(model,layers,true,true);
    moe_q4k_a8::validate_q5k_a8_manifest(model,layers);
    model.qwen_rotary_table = std::make_shared<HipQwenRotaryTable>(model.resolved_rotary,
        std::make_shared<int>(0),rotary.table_bytes());
    auto requests = prompt_compiler_requests(effective.options,model,manifest);
    kernels::KernelRegistry registry;
    register_complete_hip_kernel_catalog(model,requests.binding,requests.catalog,registry);
    auto serial = compile_serial_execution(model,registry,requests.binding,requests.buffers,256);
    const auto access = compile_execution_tensor_access_plan(serial.execution,manifest,build_model_tensor_index(manifest),
        build_serial_execution_tensor_requirements(serial.execution));
    qsa_input_prefetch::validate_tensor_access(qsa_input_prefetch::build_plan(model,manifest),manifest,serial.execution,access);
    shared_expert_prefetch::validate_tensor_access(shared_expert_prefetch::build_plan(model,manifest),manifest,serial.execution,access);
    validate_prompt_tensor_contract(effective.options,manifest,serial,access);
    descriptors(model,requests.catalog,serial,manifest,access,capacity);
    const auto pair = numerical_profile_pair(effective.options);
    require(pair.split(),"production split prefill lineage");
    requests.binding.prefill_arithmetic_identity = numerical_profile_pair_identity(pair);
    requests.catalog.numerical_profile = pair.prefill;
    requests.catalog.fast_prefill_components = pair.fast_components;
    kernels::KernelRegistry prefill_registry;
    register_complete_hip_kernel_catalog(model,requests.binding,requests.catalog,prefill_registry);
    auto prefill = compile_serial_execution(model,prefill_registry,requests.binding,requests.buffers,256);
    validate_sibling_serial_executions(prefill,serial);
    require(effective.host_compilation.at("serial_execution_identity")==prefill.execution_identity &&
        effective.host_compilation.at("serial_kernel_plan_identity")==prefill.kernel_plan_identity,
        "admission and runtime serial identities match across table adoption");
    descriptors(model,requests.catalog,prefill,manifest,access,capacity);
    require(serial.max_position_embeddings==model.effective_max_positions() && model.max_position_embeddings==262144,
        "effective capacity preserves checkpoint geometry");
    for (const auto wrong_capacity : {model.max_position_embeddings,static_cast<std::uint64_t>(capacity)}) {
        auto wrong = prefill;
        wrong.max_position_embeddings = wrong_capacity;
        bool rejected = false;
        try { (void)compile_known_prefix_window(model,wrong,manifest,access,{}); }
        catch (const std::invalid_argument& e) {
            rejected = std::string_view(e.what()).find("complete eager C1 serial source") != std::string_view::npos;
        }
        require(rejected,"native/context capacity cannot replace the resolved rotary maximum");
    }
    SerialSessionStateLayoutRequest state_request;
    state_request.context_tokens = capacity;
    require(build_serial_session_state_layout(model,state_request).context_tokens==capacity,"session capacity");
    auto rows = effective.options.known_prefix_tail_rows;
    rows.insert(rows.begin(),effective.options.known_prefix_prefill_rows);
    for (auto n : rows) {
        const auto plan = compile_known_prefix_prefill(model,prefill,manifest,access,
            {n,true,true,true,true,KnownWindowOutputPolicy::terminal_row,
             effective.options.known_prefill_workspace_budget,effective.options.known_prefill_metadata_budget/2U,
             true,true,true,true,true,false});
        validate_compiled_known_prefix_prefill(plan,prefill);
        validate_known_prefix_window_formats(plan.tile8,prefill,manifest,access);
        if (n==effective.options.known_prefix_prefill_rows)
            require(effective.host_compilation.at("prefill_execution_identity")==plan.prefill_execution_identity &&
                effective.host_compilation.at("prefill_kernel_plan_identity")==plan.prefill_kernel_plan_identity,
                "admission and runtime prefill identities match");
        auto source = prefill;
        source.execution_identity = plan.prefill_execution_identity;
        source.arena = plan.arena.row_view[0];
        source.whole_plan_graph_safe = false;
        descriptors(model,requests.catalog,source,manifest,access,capacity);
        require(plan.tile8.gdn_deferred_rows_steps+plan.tile8.gdn_rows_steps==36U && plan.tile8.qsa_rows_steps==12U &&
            plan.tile8.ple_rows_steps==1U,"all canonical layers admitted");
    }
    std::cout << "PASS host compile f=" << factor << " C=" << capacity << " prompt=" << count
              << " T=" << effective.options.known_prefix_prefill_rows << " tails=" << rows.size()-1U
              << " before device allocation\n";
}
}
int main() {
    try {
        const auto model = ModelContract::qwen38_flash_next();
        require(model.hidden_layers==48U && model.hidden_size==2560U && model.full_attention_layers==12U &&
            model.gdn_layers==36U && model.vocab_size==248320U,"canonical checkpoint geometry");
        const auto manifest = canonical_manifest(model);
        run(2,300000,8192,manifest);
        run(2,300000,299000,manifest);
        run(4,600000,8192,manifest);
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
