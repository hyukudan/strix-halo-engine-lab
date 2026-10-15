#include "strix/lane_chain_hyper_output.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/execution_binding.hpp"
#include "strix/execution_plan.hpp"
#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_ple_launcher.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/kernel_registry.hpp"
#include "strix/model_contract.hpp"
#include "strix/integer_projection_runtime.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool catalog_supported(strix::SerialOp op) {
    using strix::SerialOp;
    switch (op) {
    case SerialOp::token_embedding:
    case SerialOp::position_prepare:
    case SerialOp::hyper_expand:
    case SerialOp::ple:
    case SerialOp::gated_delta_net:
    case SerialOp::indexed_attention:
    case SerialOp::attn_hyper_mix:
    case SerialOp::attn_inject:
    case SerialOp::mlp_hyper_mix:
    case SerialOp::moe_route:
    case SerialOp::moe_combine:
    case SerialOp::mlp_inject:
    case SerialOp::final_hyper_mix:
    case SerialOp::lm_head:
    case SerialOp::shared_expert:
    case SerialOp::moe_experts:
        return true;
    default:
        return false;
    }
}

strix::kernels::KernelVariant placeholder_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::experimental;
    variant.graph_safe = false;
    variant.input_layout = "placeholder";
    variant.output_layout = "placeholder";
    variant.compiler_identity = "placeholder-v1";
    return variant;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    {
        strix::HipLauncherCatalogConfig conflict;
        conflict.virtual_q8_lm_head = true;
        strix::integer_projection::RuntimeConfig integer_config;
        strix::ExecutionBindingRequest conflict_request;
        strix::kernels::KernelRegistry conflict_registry;
        bool rejected = false;
        try { strix::register_complete_hip_kernel_catalog(model, conflict_request,
            conflict, conflict_registry, &integer_config); }
        catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()).find("mutually exclusive") != std::string::npos;
        }
        expect(rejected && conflict_registry.variants().empty(),
            "integer and virtual Q8 catalog reject before registering any variant");
    }

    strix::ExecutionBindingRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.minimum_qualification =
        strix::kernels::QualificationStatus::experimental;

    strix::HipElementwiseLauncherConfig elementwise;
    elementwise.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    elementwise.hc_count =
        static_cast<std::uint32_t>(model.hc_count);
    elementwise.activation =
        strix::HipActivationStorage::bf16;

    strix::HipEmbeddingLauncherConfig embedding;
    embedding.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    embedding.output =
        strix::HipActivationStorage::bf16;

    strix::HipHyperMixLauncherConfig hyper_mix;
    hyper_mix.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    hyper_mix.hc_count =
        static_cast<std::uint32_t>(model.hc_count);
    hyper_mix.lowrank =
        static_cast<std::uint32_t>(model.hc_lowrank);
    hyper_mix.rms_epsilon =
        model.rms_norm_epsilon;
    hyper_mix.activation =
        strix::HipActivationStorage::bf16;

    strix::HipMoeRouteLauncherConfig route;
    route.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    route.expert_count =
        static_cast<std::uint32_t>(model.num_experts);
    route.top_k =
        static_cast<std::uint32_t>(model.num_experts_per_token);
    route.activation =
        strix::HipActivationStorage::bf16;
    route.normalize_selected =
        model.norm_topk_probability;

    strix::kernels::KernelRegistry kernels;
    strix::register_hip_elementwise_kernel_variants(
        model, request, elementwise, kernels);

    strix::HipPleLauncherConfig ple;
    ple.model = model;
    ple.activation =
        strix::HipActivationStorage::bf16;
    strix::register_hip_ple_kernel_variant(
        model, request, ple, kernels);

    strix::HipGdnLauncherConfig gdn;
    gdn.model = model;
    gdn.activation =
        strix::HipActivationStorage::bf16;
    strix::register_hip_gdn_kernel_variant(
        model, request, gdn, kernels);

    strix::HipIndexedAttentionLauncherConfig indexed;
    indexed.model = model;
    indexed.activation =
        strix::HipActivationStorage::bf16;
    strix::register_hip_indexed_attention_kernel_variant(
        model, request, indexed, kernels);
    strix::register_hip_embedding_kernel_variant(
        model, request, embedding, kernels);
    strix::register_hip_hyper_mix_kernel_variant(
        model, request, hyper_mix, kernels);
    strix::register_hip_moe_route_kernel_variant(
        model, request, route, kernels);

    strix::HipMoeExpertsLauncherConfig experts;
    experts.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    experts.intermediate_size =
        static_cast<std::uint32_t>(model.moe_intermediate_size);
    experts.expert_count =
        static_cast<std::uint32_t>(model.num_experts);
    experts.top_k =
        static_cast<std::uint32_t>(model.num_experts_per_token);
    experts.activation =
        strix::HipActivationStorage::bf16;
    strix::register_hip_moe_experts_kernel_variant(
        model, request, experts, kernels);

    strix::HipLmHeadLauncherConfig lm_head;
    lm_head.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    lm_head.vocab_size =
        static_cast<std::uint32_t>(model.vocab_size);
    lm_head.input =
        strix::HipActivationStorage::bf16;
    strix::register_hip_lm_head_kernel_variant(
        model, request, lm_head, kernels);

    strix::HipSharedExpertLauncherConfig shared;
    shared.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    shared.intermediate_size =
        static_cast<std::uint32_t>(
            model.shared_expert_intermediate_size);
    shared.activation =
        strix::HipActivationStorage::bf16;
    strix::register_hip_shared_expert_kernel_variant(
        model, request, shared, kernels);

    const auto execution =
        strix::build_serial_decode_execution_plan(model);

    std::set<std::string> placeholders;
    for (const auto& step : execution.steps) {
        if (catalog_supported(step.op)) {
            continue;
        }
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model, step.op, request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.logical_rows) +
            "." +
            std::to_string(shape.request_concurrency) +
            "." +
            std::to_string(shape.hidden_size) +
            "." +
            std::to_string(shape.auxiliary_size);
        if (!placeholders.insert(key).second) {
            continue;
        }
        kernels.add(
            placeholder_variant(
                "placeholder." + key,
                *family,
                shape));
    }

    const auto bindings =
        strix::bind_serial_execution_plan(
            model,
            execution,
            kernels,
            request);
    expect(
        bindings.missing_steps == 0,
        "fixture binds complete serial plan");


    {
        strix::kernels::KernelRegistry
            complete_registry;
        strix::register_complete_hip_kernel_catalog(
            model,
            request,
            {},
            complete_registry);
        const auto complete_bindings =
            strix::bind_serial_execution_plan(
                model,
                execution,
                complete_registry,
                request);
        for (std::size_t index = 0; index < execution.steps.size(); ++index)
            if (execution.steps[index].op == strix::SerialOp::gated_delta_net)
                expect(complete_bindings.bindings[index].kernel_id.has_value() &&
                    !complete_bindings.bindings[index].kernel_id->ends_with(".virtual256.q8-input"),
                    "default complete catalog selects original GDN id");
        expect(
            complete_bindings.missing_steps == 0 &&
                complete_bindings.bound_steps +
                    complete_bindings.host_or_runtime_steps ==
                    execution.steps.size(),
            "complete catalog helper binds the full serial DAG");
    }

    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x434154414c4f4701ULL;
    compiled.execution =
        execution;
    compiled.bindings =
        bindings;
    compiled.io = strix::build_serial_execution_io_plan(execution);

    strix::ExecutionLauncherRegistry launchers;
    strix::HipLauncherCatalog catalog(
        model,
        {strix::HipActivationStorage::bf16});
    catalog.register_for_compiled_execution(
        compiled,
        launchers);

    const auto coverage =
        catalog.audit(
            compiled,
            launchers);

    std::map<strix::SerialOp, std::uint32_t> missing_by_op;
    for (const auto& missing : coverage.missing) {
        ++missing_by_op[missing.op];
    }

    const auto missing_count = [&](strix::SerialOp op) {
        const auto found = missing_by_op.find(op);
        return found == missing_by_op.end() ? 0U : found->second;
    };

    expect(
        coverage.missing_steps == 0,
        "catalog covers every serial execution step");
    expect(
        missing_count(strix::SerialOp::ple) == 0,
        "PLE launcher is covered");
    expect(
        missing_count(strix::SerialOp::gated_delta_net) == 0,
        "GDN launcher is covered");
    expect(
        missing_count(strix::SerialOp::indexed_attention) == 0,
        "indexed-attention launcher is covered");
    expect(
        missing_count(strix::SerialOp::moe_experts) == 0,
        "routed-expert launcher is covered");
    expect(
        missing_count(strix::SerialOp::shared_expert) == 0,
        "shared-expert launcher is covered");
    expect(
        missing_count(strix::SerialOp::lm_head) == 0,
        "LM-head launcher is covered");
    expect(
        missing_by_op.empty(),
        "HIP launcher catalog has no missing serial operators");

    expect(
        coverage.covered_steps +
                coverage.missing_steps ==
            execution.steps.size(),
        "launcher coverage partitions the full serial DAG");

    {
        auto graph_request = request;
        graph_request.require_graph_safe = true;
        graph_request.device_graph_inputs = true;
        strix::HipLauncherCatalogConfig graph_config;
        graph_config.enable_device_graph_inputs = true;
        graph_config.virtual_q8_gdn_input = true;
        strix::kernels::KernelRegistry graph_kernels;
        strix::register_complete_hip_kernel_catalog(model, graph_request, graph_config, graph_kernels);
        strix::ExecutionBufferRequest buffers;
        const auto graph_compiled = strix::compile_serial_execution(model, graph_kernels, graph_request, buffers);
        for (std::size_t index = 0; index < graph_compiled.execution.steps.size(); ++index)
            if (graph_compiled.execution.steps[index].op == strix::SerialOp::gated_delta_net)
                expect(graph_compiled.bindings.bindings[index].kernel_id.has_value() &&
                    graph_compiled.bindings.bindings[index].kernel_id->ends_with(".virtual256.q8-input"),
                    "flagged complete catalog selects virtual Q8 GDN id");
        expect(graph_compiled.whole_plan_graph_safe && graph_compiled.dynamic_steps == 0,
               "actual device-input catalog binds a complete fixed-shape graph DAG");
        strix::HipLauncherCatalog graph_catalog(model, graph_config);
        strix::ExecutionLauncherRegistry graph_launchers;
        graph_catalog.register_for_compiled_execution(graph_compiled, graph_launchers);
        expect(graph_catalog.audit(graph_compiled, graph_launchers).missing_steps == 0,
               "graph catalog contains every actual operator and device-position bookkeeping consumer");
        auto no_virtual_config = graph_config; no_virtual_config.virtual_q8_gdn_input = false;
        strix::HipLauncherCatalog no_virtual_catalog(model, no_virtual_config);
        bool virtual_mismatch_rejected = false;
        try { strix::ExecutionLauncherRegistry bad; no_virtual_catalog.register_for_compiled_execution(graph_compiled, bad); }
        catch (const std::invalid_argument&) { virtual_mismatch_rejected = true; }
        expect(virtual_mismatch_rejected, "catalog rejects virtual selected GDN id when configuration drops the flag");
        auto prefetch_config = graph_config; prefetch_config.prefetch_q8_gdn_input = true;
        strix::kernels::KernelRegistry prefetch_kernels;
        strix::register_complete_hip_kernel_catalog(model, graph_request, prefetch_config, prefetch_kernels);
        const auto prefetch_compiled = strix::compile_serial_execution(model, prefetch_kernels, graph_request, buffers);
        for (std::size_t index = 0; index < prefetch_compiled.execution.steps.size(); ++index)
            if (prefetch_compiled.execution.steps[index].op == strix::SerialOp::gated_delta_net)
                expect(prefetch_compiled.bindings.bindings[index].kernel_id.has_value() &&
                    prefetch_compiled.bindings.bindings[index].kernel_id->ends_with(".prefetch8.virtual256.q8-input"),
                    "prefetch complete catalog selects explicit prefetch GDN id");
        strix::HipLauncherCatalog prefetch_catalog(model, prefetch_config);
        strix::ExecutionLauncherRegistry prefetch_launchers;
        prefetch_catalog.register_for_compiled_execution(prefetch_compiled, prefetch_launchers);
        expect(prefetch_catalog.audit(prefetch_compiled, prefetch_launchers).missing_steps == 0,
            "prefetch catalog bundle propagates selected flags at its construction site");
        bool prefetch_mismatch_rejected = false;
        try { strix::ExecutionLauncherRegistry bad; graph_catalog.register_for_compiled_execution(prefetch_compiled, bad); }
        catch (const std::invalid_argument&) { prefetch_mismatch_rejected = true; }
        expect(prefetch_mismatch_rejected, "streaming catalog rejects prefetched selected GDN id");
        auto invalid_prefetch_config = prefetch_config; invalid_prefetch_config.virtual_q8_gdn_input = false;
        bool invalid_prefetch_rejected = false;
        try { strix::kernels::KernelRegistry bad;
            strix::register_complete_hip_kernel_catalog(model, graph_request, invalid_prefetch_config, bad); }
        catch (const std::invalid_argument&) { invalid_prefetch_rejected = true; }
        expect(invalid_prefetch_rejected, "catalog rejects prefetch without required virtual input flag");
        bool mismatch_rejected = false;
        try { strix::ExecutionLauncherRegistry bad; catalog.register_for_compiled_execution(graph_compiled, bad); }
        catch (const std::invalid_argument&) { mismatch_rejected = true; }
        expect(mismatch_rejected, "eager catalog cannot label its launchers as a device-input graph");
    }

    {
        // Explicit mixed topology: layer 3 falls back, the other eleven use Q8.
        auto on_request = request;
        on_request.prefetch_q8_qsa_input = true;
        for (std::uint32_t layer = 7; layer < model.hidden_layers; layer += 4)
            on_request.qsa_input_prefetch_layers.push_back(layer);
        strix::HipLauncherCatalogConfig on_config;
        on_config.prefetch_q8_qsa_input = true;
        on_config.qsa_input_prefetch_layers = on_request.qsa_input_prefetch_layers;
        strix::kernels::KernelRegistry mixed_registry;
        strix::register_complete_hip_kernel_catalog(model, on_request, on_config, mixed_registry);
        strix::ExecutionBufferRequest buffers;
        const auto mixed = strix::compile_serial_execution(model, mixed_registry, on_request, buffers);
        const auto off = strix::compile_serial_execution(model, mixed_registry, request, buffers);
        strix::HipLauncherCatalog on_catalog(model, on_config), off_catalog(model);
        strix::ExecutionLauncherRegistry mixed_launchers, off_launchers;
        on_catalog.register_for_compiled_execution(mixed, mixed_launchers);
        off_catalog.register_for_compiled_execution(off, off_launchers);
        unsigned pref = 0, base = 0;
        std::set<std::string> indexed_ids;
        for(const auto& binding : mixed.bindings.bindings) {
            if(binding.op != strix::SerialOp::indexed_attention) continue;
            const auto& id = *binding.kernel_id;
            indexed_ids.insert(id);
            const bool use = id.ends_with(strix::qsa_input_prefetch::kernel_suffix);
            use ? ++pref : ++base;
            expect(use == (*mixed.execution.steps[binding.step_id].layer != 3U),
                   "mixed catalog selects exact admitted layers");
            const auto* variant = mixed_registry.find(id);
            expect(variant && variant->compiler_identity == (use ?
                "hip-gfx1151-indexed-attention-v3-prefetch8-virtual256-q8-qsa-input" :
                "hip-gfx1151-indexed-attention-v2"), "mixed registry compiler identities");
            const auto* selected_launcher = mixed_launchers.find_kernel_launcher(id);
            expect(selected_launcher != nullptr, "mixed bundle registers each indexed id");
            if(!use) {
                const auto* legacy = off_launchers.find_kernel_launcher(id);
                expect(legacy && selected_launcher && legacy->implementation_identity ==
                    selected_launcher->implementation_identity, "fallback identity equals OFF");
            }
        }
        expect(pref == 11 && base == 1 && indexed_ids.size() == 2,
               "mixed catalog has eleven prefetch and one base with two launchers");
        expect(on_catalog.audit(mixed,mixed_launchers).missing_steps == 0,
               "mixed catalog launcher coverage complete");
        const auto rejects = [&](auto&& operation, const char* expected) {
            bool rejected = false;
            try { operation(); } catch(const std::invalid_argument& error) {
                rejected = std::string(error.what()).find(expected) != std::string::npos;
                if(!rejected) std::cerr << error.what() << '\n';
            }
            expect(rejected, expected);
        };
        rejects([&]{strix::ExecutionLauncherRegistry bad;
            off_catalog.register_for_compiled_execution(mixed,bad);}, "binding/config mismatch");
        rejects([&]{strix::ExecutionLauncherRegistry bad;
            on_catalog.register_for_compiled_execution(off,bad);}, "requires eligible bound step");
        auto wrong_config=on_config; wrong_config.qsa_input_prefetch_layers.erase(
            wrong_config.qsa_input_prefetch_layers.begin());
        rejects([&]{strix::kernels::KernelRegistry bad;
            strix::register_complete_hip_kernel_catalog(model,on_request,wrong_config,bad);},
            "QSA input prefetch differs from binding request");
        strix::HipIndexedAttentionLauncherConfig wrong_direct;
        wrong_direct.model=model;
        rejects([&]{strix::kernels::KernelRegistry bad;
            strix::register_hip_indexed_attention_kernel_variant(model,on_request,wrong_direct,bad);},
            "QSA prefetch config differs from binding request");
    }


    {
        auto hc_request=request;hc_request.lane_chain_hyper_output=true;
        for(const auto& step:execution.steps)if(step.op==strix::SerialOp::attn_hyper_mix || step.op==strix::SerialOp::mlp_hyper_mix)hc_request.lane_chain_hyper_output_eligible_steps.push_back(step.id);
        strix::HipLauncherCatalogConfig hc_config;hc_config.parallel_hyper_mix=true;hc_config.staged_hyper_projections=true;hc_config.fused_staged_hyper_mix=true;hc_config.typed_hyper_projection=true;hc_config.prefetch_typed_hyper_rows=true;hc_config.prefetch_q8_hyper_output=true;hc_config.lane_chain_hyper_output=true;
        strix::kernels::KernelRegistry hc_registry;strix::register_complete_hip_kernel_catalog(model,hc_request,hc_config,hc_registry);
        bool has_base=false,has_candidate=false;
        for(const auto& variant:hc_registry.variants())if(variant.family==strix::kernels::OperatorFamily::hyper_mix){if(variant.id.ends_with(strix::lane_chain_hyper_output_suffix))has_candidate=true;else has_base=true;}
        expect(has_base && has_candidate,"complete HC catalog contains base and candidate pair");
        auto bad_config=hc_config;bad_config.lane_chain_hyper_output=false;bool rejected=false;
        try{strix::kernels::KernelRegistry bad;strix::register_complete_hip_kernel_catalog(model,hc_request,bad_config,bad);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"HC catalog rejects config/request opt-in mismatch");
        auto bad_request=hc_request;bad_request.lane_chain_hyper_output=false;rejected=false;
        try{strix::kernels::KernelRegistry bad;strix::register_complete_hip_kernel_catalog(model,bad_request,hc_config,bad);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"HC catalog rejects inverse config/request opt-in mismatch");
        strix::ExecutionBufferRequest hc_buffers;const auto hc_compiled=strix::compile_serial_execution(model,hc_registry,hc_request,hc_buffers);
        expect(hc_compiled.bindings.missing_steps==0,"mixed candidate/base catalog compiles complete plan");
        strix::kernels::KernelRegistry candidate_only;
        for(const auto& variant:hc_registry.variants())if(variant.family!=strix::kernels::OperatorFamily::hyper_mix || variant.id.ends_with(strix::lane_chain_hyper_output_suffix))candidate_only.add(variant);
        rejected=false;try{(void)strix::compile_serial_execution(model,candidate_only,hc_request,hc_buffers);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"compiled serial execution rejects unavailable base suffix admission");
    }

    {
        // Explicit mixed topology: layer 0 falls back, the other forty-seven use Q8.
        auto on_request = request;
        on_request.prefetch_q8_shared_expert = true;
        for (std::uint32_t layer = 1; layer < model.hidden_layers; ++layer)
            on_request.shared_expert_prefetch_layers.push_back(layer);
        strix::HipLauncherCatalogConfig on_config;
        on_config.prefetch_q8_shared_expert = true;
        on_config.shared_expert_prefetch_layers = on_request.shared_expert_prefetch_layers;
        on_config.shared_expert_prefetch_stats =
            std::make_shared<strix::shared_expert_prefetch::RuntimeDispatchStats>();
        strix::kernels::KernelRegistry mixed_registry;
        strix::register_complete_hip_kernel_catalog(model, on_request, on_config, mixed_registry);
        strix::ExecutionBufferRequest buffers;
        const auto mixed = strix::compile_serial_execution(model, mixed_registry, on_request, buffers);
        const auto off = strix::compile_serial_execution(model, mixed_registry, request, buffers);
        strix::HipLauncherCatalog on_catalog(model, on_config), off_catalog(model);
        strix::ExecutionLauncherRegistry mixed_launchers, off_launchers;
        on_catalog.register_for_compiled_execution(mixed, mixed_launchers);
        off_catalog.register_for_compiled_execution(off, off_launchers);
        unsigned pref = 0, base = 0;
        std::set<std::string> shared_ids;
        for(const auto& binding : mixed.bindings.bindings) {
            if(binding.op != strix::SerialOp::shared_expert) continue;
            const auto& id = *binding.kernel_id;
            shared_ids.insert(id);
            const bool use = id.ends_with(strix::shared_expert_prefetch::kernel_suffix);
            use ? ++pref : ++base;
            expect(use == (*mixed.execution.steps[binding.step_id].layer != 0U),
                   "mixed catalog selects exact admitted layers");
            const auto* variant = mixed_registry.find(id);
            expect(variant && variant->compiler_identity == (use ?
                "hip-gfx1151-shared-expert-v2-prefetch8-virtual256-q8" :
                "hip-gfx1151-shared-expert-v1"), "mixed registry compiler identities");
            const auto* selected_launcher = mixed_launchers.find_kernel_launcher(id);
            expect(selected_launcher != nullptr, "mixed bundle registers each indexed id");
            if (selected_launcher) {
                const auto* launcher_config =
                    static_cast<const strix::HipSharedExpertLauncherConfig*>(selected_launcher->user_data);
                expect(launcher_config && launcher_config->prefetch_stats ==
                    on_config.shared_expert_prefetch_stats,
                    "catalog forwards the real shared dispatch statistics owner");
                expect(launcher_config && launcher_config->prefetch_q8 == use,
                    "fallback launcher cannot dispatch shared prefetch");
            }
            if(!use) {
                const auto* legacy = off_launchers.find_kernel_launcher(id);
                expect(legacy && selected_launcher && legacy->implementation_identity ==
                    selected_launcher->implementation_identity, "fallback identity equals OFF");
            }
        }
        expect(pref == 47 && base == 1 && shared_ids.size() == 2,
               "mixed catalog has forty-seven prefetch and one base with two launchers");
        expect(on_catalog.audit(mixed,mixed_launchers).missing_steps == 0,
               "mixed catalog launcher coverage complete");
        const auto rejects = [&](auto&& operation, const char* expected) {
            bool rejected = false;
            try { operation(); } catch(const std::invalid_argument& error) {
                rejected = std::string(error.what()).find(expected) != std::string::npos;
                if(!rejected) std::cerr << error.what() << '\n';
            }
            expect(rejected, expected);
        };
        rejects([&]{strix::ExecutionLauncherRegistry bad;
            off_catalog.register_for_compiled_execution(mixed,bad);}, "layer/config mismatch");
        rejects([&]{strix::ExecutionLauncherRegistry bad;
            on_catalog.register_for_compiled_execution(off,bad);}, "has no eligible step");
        auto wrong_config=on_config; wrong_config.shared_expert_prefetch_layers.erase(
            wrong_config.shared_expert_prefetch_layers.begin());
        rejects([&]{strix::kernels::KernelRegistry bad;
            strix::register_complete_hip_kernel_catalog(model,on_request,wrong_config,bad);},
            "shared prefetch request mismatch");
        strix::HipSharedExpertLauncherConfig wrong_direct;
        wrong_direct.hidden_size=static_cast<std::uint32_t>(model.hidden_size);
        wrong_direct.intermediate_size=static_cast<std::uint32_t>(model.shared_expert_intermediate_size);
        rejects([&]{strix::kernels::KernelRegistry bad;
            strix::register_hip_shared_expert_kernel_variant(model,on_request,wrong_direct,bad);},
            "shared prefetch config/request mismatch");
    }


    if (failures == 0) {
        std::cout
            << "HIP launcher catalog tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
