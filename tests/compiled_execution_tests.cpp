#include "strix/compiled_execution.hpp"

#include <set>
#include <string>
#include <utility>

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape,
    double measured_seconds) {
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
        strix::kernels::QualificationStatus::oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "test";
    variant.output_layout = "test";
    variant.compiler_identity = "test-compiler";
    variant.measured_seconds = measured_seconds;
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request,
    double measured_seconds = 0.0,
    std::string suffix = "") {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);

    std::set<std::string> added;
    for (const auto& step : execution.steps) {
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

        if (!added.insert(key).second) {
            continue;
        }

        registry.add(
            make_variant(
                "test." + key + suffix +
                    (step.op == strix::SerialOp::moe_route && request.ranked_moe_router_topk ? ".ranked.parallel" :
                     step.op == strix::SerialOp::moe_route && request.parallel_moe_router ? ".parallel" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.q4k_a8g32_gate_up ? ".q4k-a8g32-gateup-v2" : ""),
                *family,
                shape,
                measured_seconds));
    }

    return registry;
}

struct NullDispatcher final
    : strix::EagerExecutionDispatcher {
    std::uint32_t kernels = 0;
    std::uint64_t expected_identity = 0;
    std::uint32_t runtime = 0;

    void dispatch_kernel(
        const strix::ExecutionDispatchContext& context) override {
        expect(context.execution_identity == expected_identity && expected_identity != 0,
               "serial kernel dispatch carries compiled execution identity");
        ++kernels;
    }

    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext& context) override {
        expect(context.execution_identity == expected_identity && expected_identity != 0,
               "serial runtime dispatch carries compiled execution identity");
        ++runtime;
    }
};

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::ExecutionBindingRequest binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;

    const auto registry =
        make_registry(
            model,
            binding_request);

    const auto compiled =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            256);

    expect(compiled.bindings.missing_steps == 0,
           "compiled execution is fully bound");
    expect(compiled.kernel_plan_identity != 0 &&
               compiled.execution_identity != 0,
           "compiled execution has stable non-zero identities");
    // Captured from the pinned, pre-router implementation with this fixture.
    expect(compiled.execution_identity == 5195923158154663034ULL &&
               compiled.kernel_plan_identity == 4048923307928001002ULL &&
               compiled.arena.total_bytes == 998400ULL,
           "default serial identities and arena retain the pre-router receipt");
    expect(compiled.arena.total_bytes > 0,
           "compiled execution contains a physical scratch arena");
    expect(!compiled.whole_plan_graph_safe,
           "full decode plan remains non-graph-safe while runtime/dynamic steps exist");
    expect(compiled.dynamic_steps > 0,
           "compiled plan counts dynamic-shape steps");

    const auto compiled_again =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            256);
    expect(compiled_again.kernel_plan_identity ==
               compiled.kernel_plan_identity &&
               compiled_again.execution_identity ==
                   compiled.execution_identity,
           "compiled execution identity is reproducible");

    const auto remeasured_registry =
        make_registry(
            model,
            binding_request,
            9e-6);
    const auto remeasured =
        strix::compile_serial_execution(
            model,
            remeasured_registry,
            binding_request,
            buffer_request,
            256);
    expect(remeasured.execution_identity ==
               compiled.execution_identity,
           "benchmark-time noise does not change compiled execution identity");

    const auto different_registry =
        make_registry(
            model,
            binding_request,
            0.0,
            ".alternate");
    const auto different_kernel_plan =
        strix::compile_serial_execution(
            model,
            different_registry,
            binding_request,
            buffer_request,
            256);
    expect(different_kernel_plan.execution_identity !=
               compiled.execution_identity,
           "kernel identity change invalidates compiled execution identity");

    const auto different_alignment =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            512);
    expect(different_alignment.execution_identity !=
               compiled.execution_identity,
           "arena layout identity participates in compiled execution identity");


    auto different_numerics_model =
        model;
    different_numerics_model.rms_norm_epsilon =
        2e-6F;
    const auto different_numerics =
        strix::compile_serial_execution(
            different_numerics_model,
            registry,
            binding_request,
            buffer_request,
            256);
    expect(
        different_numerics.execution_identity !=
            compiled.execution_identity,
        "RMS epsilon changes compiled execution identity");

    auto different_router_model =
        model;
    different_router_model.norm_topk_probability =
        !model.norm_topk_probability;
    const auto different_router_policy =
        strix::compile_serial_execution(
            different_router_model,
            registry,
            binding_request,
            buffer_request,
            256);
    expect(
        different_router_policy.execution_identity !=
            compiled.execution_identity,
        "router normalization policy changes compiled execution identity");


    auto different_qsa_model =
        model;
    different_qsa_model.indexer_budget =
        model.indexer_budget -
        model.indexer_compress_ratio;
    const auto different_qsa =
        strix::compile_serial_execution(
            different_qsa_model,
            make_registry(
                different_qsa_model,
                binding_request),
            binding_request,
            buffer_request,
            256);
    expect(
        different_qsa.execution_identity !=
            compiled.execution_identity,
        "QSA budget changes compiled execution identity");

    auto different_rope_model =
        model;
    different_rope_model.partial_rotary_factor =
        0.5F;
    const auto different_rope =
        strix::compile_serial_execution(
            different_rope_model,
            make_registry(
                different_rope_model,
                binding_request),
            binding_request,
            buffer_request,
            256);
    expect(
        different_rope.execution_identity !=
            compiled.execution_identity,
        "partial RoPE factor changes compiled execution identity");

    auto different_temporary_request =
        buffer_request;
    different_temporary_request.mlp_intermediate_element_bytes =
        2;
    const auto different_temporary =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            different_temporary_request,
            256);
    expect(
        different_temporary.execution_identity !=
            compiled.execution_identity,
        "MLP temporary storage contract changes compiled execution identity");


    auto different_gdn_norm_model =
        model;
    different_gdn_norm_model.gdn_qk_l2norm =
        !model.gdn_qk_l2norm;
    const auto different_gdn_norm =
        strix::compile_serial_execution(
            different_gdn_norm_model,
            make_registry(
                different_gdn_norm_model,
                binding_request),
            binding_request,
            buffer_request,
            256);
    expect(
        different_gdn_norm.execution_identity !=
            compiled.execution_identity,
        "GDN q/k normalization changes compiled execution identity");


    auto different_gdn_epsilon_model =
        model;
    different_gdn_epsilon_model.gdn_qk_l2_epsilon =
        2e-6F;
    const auto different_gdn_epsilon =
        strix::compile_serial_execution(
            different_gdn_epsilon_model,
            make_registry(
                different_gdn_epsilon_model,
                binding_request),
            binding_request,
            buffer_request,
            256);
    expect(
        different_gdn_epsilon.execution_identity !=
            compiled.execution_identity,
        "GDN q/k epsilon changes compiled execution identity");

    auto different_gdn_conv_model =
        model;
    different_gdn_conv_model.gdn_conv_silu =
        !model.gdn_conv_silu;
    const auto different_gdn_conv =
        strix::compile_serial_execution(
            different_gdn_conv_model,
            make_registry(
                different_gdn_conv_model,
                binding_request),
            binding_request,
            buffer_request,
            256);
    expect(
        different_gdn_conv.execution_identity !=
            compiled.execution_identity,
        "GDN convolution activation changes compiled execution identity");

    strix::SerialSessionStateLayoutRequest state_request;
    state_request.context_tokens = 32;
    const auto state_layout =
        strix::build_serial_session_state_layout(
            model, state_request);

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        static_cast<std::uintptr_t>(0x21000000ULL);
    state_memory.device_bytes =
        state_layout.device_total_bytes;
    state_memory.device_allocation_generation = 1;
    state_memory.host_base =
        static_cast<std::uintptr_t>(0x22000000ULL);
    state_memory.host_bytes =
        state_layout.host_total_bytes;
    state_memory.host_allocation_generation = 1;

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        compiled.execution_identity;
    memory.arena_base =
        static_cast<std::uintptr_t>(0x20000000ULL);
    memory.arena_bytes =
        compiled.arena.total_bytes;
    memory.allocation_generation = 1;

    strix::EagerExecutionRequest run;
    run.logical_rows = 1;
    run.request_concurrency = 1;
    run.state.valid = all_state();

    NullDispatcher dispatcher;
    dispatcher.expected_identity = compiled.execution_identity;
    const auto result =
        strix::execute_compiled_serial_eager(
            compiled,
            memory,
            state_layout,
            state_memory,
            run,
            dispatcher);
    expect(result.completed_steps ==
               compiled.execution.steps.size(),
           "compiled eager execution completes the frozen DAG");
    expect(dispatcher.kernels ==
               compiled.bindings.bound_steps &&
               dispatcher.runtime ==
                   compiled.bindings.host_or_runtime_steps,
           "compiled eager execution dispatches frozen kernel/runtime steps");

    auto wrong_shape = run;
    wrong_shape.logical_rows = 2;
    bool shape_rejected = false;
    try {
        NullDispatcher unused;
        static_cast<void>(
            strix::execute_compiled_serial_eager(
                compiled,
                memory,
                state_layout,
                state_memory,
                wrong_shape,
                unused));
    } catch (const std::invalid_argument&) {
        shape_rejected = true;
    }
    expect(shape_rejected,
           "compiled execution rejects runtime shape mismatch");


    auto no_sparse_request = state_request;
    no_sparse_request.include_sparse_index_candidate = false;
    const auto no_sparse_layout =
        strix::build_serial_session_state_layout(
            model, no_sparse_request);
    auto no_sparse_memory = state_memory;
    no_sparse_memory.layout_identity =
        no_sparse_layout.layout_identity;
    no_sparse_memory.device_bytes =
        no_sparse_layout.device_total_bytes;
    no_sparse_memory.host_bytes =
        no_sparse_layout.host_total_bytes;

    bool sparse_rejected = false;
    try {
        NullDispatcher unused;
        static_cast<void>(
            strix::execute_compiled_serial_eager(
                compiled,
                memory,
                no_sparse_layout,
                no_sparse_memory,
                run,
                unused));
    } catch (const std::invalid_argument&) {
        sparse_rejected = true;
    }
    expect(sparse_rejected,
           "missing sparse state is rejected before compiled dispatch");

    auto graph_request = binding_request;
    graph_request.require_graph_safe = true;
    graph_request.device_graph_inputs = true;
    const auto graph_registry = make_registry(model, graph_request);
    const auto graph_compiled = strix::compile_serial_execution(model, graph_registry, graph_request, buffer_request);
    expect(graph_compiled.whole_plan_graph_safe && graph_compiled.device_graph_inputs &&
               graph_compiled.dynamic_steps == 0 && graph_compiled.non_graph_safe_steps == 0,
           "explicit device-input topology requires graph-safe kernels and fixes maximum launch shapes");
    expect(graph_compiled.execution_identity != compiled.execution_identity,
           "device-input graph cannot reuse eager execution identity");
    graph_request.require_graph_safe = false;
    bool graph_policy_rejected = false;
    try { static_cast<void>(strix::compile_serial_execution(model, graph_registry, graph_request, buffer_request)); }
    catch (const std::invalid_argument&) { graph_policy_rejected = true; }
    expect(graph_policy_rejected, "device-input topology cannot bypass graph-safe kernel selection");

    auto router_binding = binding_request;
    auto router_buffers = buffer_request;
    router_binding.parallel_moe_router = true;
    router_buffers.parallel_moe_router = true;
    const auto router_registry = make_registry(model, router_binding);
    const auto router_compiled = strix::compile_serial_execution(model, router_registry, router_binding, router_buffers);
    expect(router_compiled.execution_identity == 2521722731646212663ULL &&
               router_compiled.kernel_plan_identity == 12954012342789894402ULL &&
               router_compiled.arena.total_bytes == 998400ULL,
           "parallel-only execution preserves fixed identity and workspace receipts");
    expect(router_compiled.execution_identity != compiled.execution_identity &&
               router_compiled.kernel_plan_identity != compiled.kernel_plan_identity,
           "parallel router invalidates serial execution and kernel identities");
    expect(router_compiled.io.router_workspace_slots == 1 && compiled.io.router_workspace_slots == 0,
           "router temporary exists only in opted-in IO");
    std::uint32_t router_uses = 0;
    for (const auto& step : router_compiled.io.steps) {
        if (step.op != strix::SerialOp::moe_route) continue;
        ++router_uses;
        expect(step.temporaries.size() == 1 &&
                   step.temporaries.front().buffer.buffer_class == strix::ExecutionBufferClass::router_workspace &&
                   strix::find_arena_allocation(router_compiled.arena, step.temporaries.front()) != nullptr,
               "every parallel router workspace use has a generation-bound arena allocation");
    }
    expect(router_uses == model.hidden_layers, "all layer routers admit a workspace generation");
    bool router_mismatch = false;
    try { static_cast<void>(strix::compile_serial_execution(model, router_registry, router_binding, buffer_request)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "serial compilation rejects router request/admission mismatch");
    router_mismatch = false;
    try { static_cast<void>(strix::compile_serial_execution(model, registry, router_binding, router_buffers)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "serial compilation rejects legacy kernel under parallel admission");
    router_mismatch = false;
    try { static_cast<void>(strix::compile_serial_execution(model, router_registry, binding_request, buffer_request)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "serial compilation rejects parallel kernel under default admission");
    router_mismatch = false;
    try { static_cast<void>(strix::build_serial_execution_arena_plan(model, router_buffers,
                                   router_compiled.execution, router_compiled.io, 8)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "arena preflight rejects insufficient parallel workspace alignment");
    auto corrupt_router_io = router_compiled.io;
    for (auto& io : corrupt_router_io.steps)
        if (io.op == strix::SerialOp::moe_route) io.temporaries.push_back(io.temporaries.front());
    router_mismatch = false;
    try { strix::validate_serial_execution_io_plan(router_compiled.execution, corrupt_router_io); }
    catch (const std::invalid_argument& error) {
        router_mismatch = std::string(error.what()).find("router workspace IO") != std::string::npos;
    }
    expect(router_mismatch, "IO preflight rejects extra router temporaries with the topology reason");
    router_mismatch = false;
    try { static_cast<void>(strix::build_serial_execution_arena_plan(model, buffer_request,
                                   router_compiled.execution, router_compiled.io)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "arena preflight rejects unrequested router IO");
    auto ranked_binding = router_binding;
    ranked_binding.ranked_moe_router_topk = true;
    const auto ranked_registry = make_registry(model, ranked_binding);
    const auto ranked_compiled = strix::compile_serial_execution(model, ranked_registry, ranked_binding, router_buffers);
    expect(ranked_compiled.execution_identity != router_compiled.execution_identity &&
               ranked_compiled.kernel_plan_identity != router_compiled.kernel_plan_identity &&
               ranked_compiled.arena.total_bytes == router_compiled.arena.total_bytes,
           "ranked router changes identities without changing workspace");
    const auto rejects_ranked = [&](const strix::kernels::KernelRegistry& candidate,
                                    const strix::ExecutionBindingRequest& request,
                                    const strix::ExecutionBufferRequest& buffers) {
        try { (void)strix::compile_serial_execution(model, candidate, request, buffers); }
        catch (const std::invalid_argument&) { return true; } return false;
    };
    expect(rejects_ranked(router_registry, ranked_binding, router_buffers), "ranked admission rejects parallel-only kernel");
    expect(rejects_ranked(ranked_registry, router_binding, router_buffers), "parallel-only admission rejects ranked kernel");
    auto invalid_ranked = binding_request; invalid_ranked.ranked_moe_router_topk = true;
    expect(rejects_ranked(registry, invalid_ranked, buffer_request), "ranked router rejects missing parallel topology");
    router_binding.parallel_moe_router = false;
    router_buffers.parallel_moe_router = false;
    const auto explicit_default = strix::compile_serial_execution(model, registry, router_binding, router_buffers);
    expect(explicit_default.execution_identity == compiled.execution_identity &&
               explicit_default.kernel_plan_identity == compiled.kernel_plan_identity &&
               explicit_default.arena.total_bytes == compiled.arena.total_bytes,
           "explicit disabled router preserves default execution/kernel identities and arena");

    auto q4_binding=binding_request;
    auto q4_buffers=buffer_request;
    q4_binding.q4k_a8g32_gate_up=true; q4_buffers.q4k_a8g32_gate_up=true;
    const auto q4_registry=make_registry(model,q4_binding);
    const auto q4_compiled=strix::compile_serial_execution(model,q4_registry,q4_binding,q4_buffers);
    expect(q4_compiled.execution_identity != compiled.execution_identity &&
        q4_compiled.kernel_plan_identity != compiled.kernel_plan_identity,
        "Q4K A8 opt-in gets distinct compiled and kernel identities");
    expect(rejects_ranked(q4_registry,q4_binding,buffer_request), "Q4K A8 request requires matching buffer admission");
    expect(rejects_ranked(q4_registry,binding_request,q4_buffers), "Q4K A8 buffers require matching request");
    expect(rejects_ranked(registry,q4_binding,q4_buffers), "Q4K A8 admission rejects legacy kernel");
    expect(rejects_ranked(q4_registry,binding_request,buffer_request), "legacy admission rejects Q4K A8 kernel");
    bool q4_alignment_rejected=false;
    try { (void)strix::compile_serial_execution(model,q4_registry,q4_binding,q4_buffers,128); }
    catch(const std::invalid_argument&) { q4_alignment_rejected=true; }
    expect(q4_alignment_rejected, "Q4K A8 rejects insufficient absolute arena alignment");
    q4_binding.q4k_a8g32_gate_up=false; q4_buffers.q4k_a8g32_gate_up=false;
    expect(strix::compile_serial_execution(model,registry,q4_binding,q4_buffers).execution_identity ==
        5195923158154663034ULL, "disabled Q4K A8 preserves pre-edit compiled golden");

    if (failures == 0) {
        std::cout << "compiled execution tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
