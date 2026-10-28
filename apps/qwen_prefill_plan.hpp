#pragma once
#include "qwen_prompt_organization.hpp"
#include "qwen_yarn_owner_inventory.hpp"
namespace strix::decode_cli {
// Compile and size the admitted prompt before HIP setup or state mutation.
inline PrefillRequirements compile_prompt_requirements(const Options& options,
    const ModelContract& model, const ModelManifest& manifest, std::uint64_t model_bytes) {
    auto requests = prompt_compiler_requests(options, model, manifest);
    auto& binding_request = requests.binding;
    auto& buffer_request = requests.buffers;
    auto& catalog_config = requests.catalog;
    kernels::KernelRegistry kernel_registry;
    register_complete_hip_kernel_catalog(model, binding_request, catalog_config, kernel_registry);
    auto compiled =
        strix::compile_serial_execution(
            model,
            kernel_registry,
            binding_request,
            buffer_request,
            256);

    const auto pair = numerical_profile_pair(options);
    if (pair.split()) {
        binding_request.prefill_arithmetic_identity = numerical_profile_pair_identity(pair);
        catalog_config.numerical_profile = pair.prefill;
        catalog_config.fast_prefill_components = pair.fast_components;
        kernels::KernelRegistry prefill_registry;
        register_complete_hip_kernel_catalog(model, binding_request, catalog_config, prefill_registry);
        auto prefill = compile_serial_execution(model, prefill_registry, binding_request, buffer_request, 256);
        validate_sibling_serial_executions(prefill, compiled);
        compiled = std::move(prefill);
    }
    const auto index = build_model_tensor_index(manifest);
    const auto access = compile_execution_tensor_access_plan(compiled.execution, manifest, index,
        build_serial_execution_tensor_requirements(compiled.execution));
    validate_prompt_tensor_contract(options,manifest,compiled,access);
    const auto make_plan = [&](std::uint32_t rows) {
        return compile_known_prefix_prefill(model, compiled, manifest, access,
            {rows,true,true,true,true,KnownWindowOutputPolicy::terminal_row,
             options.rotary.mode() == RotaryMode::yarn ? UINT64_MAX : options.known_prefill_workspace_budget,
             options.rotary.mode() == RotaryMode::yarn ? UINT64_MAX : options.known_prefill_metadata_budget / 2U,
             true,true,true,true,true,false,!options.mtp.enabled && !options.http_serve,true});
    };
    const auto plan = make_plan(options.known_prefix_prefill_rows);
    const auto add = prefill_checked_add;
    const auto mul = prefill_checked_mul;
    SerialSessionStateLayoutRequest state_request;
    state_request.context_tokens = options.context_tokens;
    const auto state = build_serial_session_state_layout(model, state_request);
    PrefillRequirements result;
    result.model_backing = model_bytes;
    result.state_d1 = add(state.device_total_bytes, add(state.host_total_bytes, add(state.host_total_bytes, state.host_alignment - 1U)));
    result.serial_scratch = compiled.arena.total_bytes;
    result.device_base = add(state.device_total_bytes, result.serial_scratch);
    // The serial decode pool (qsa_decode_pool_route) is retained device state from the first decode step on.
    const auto gsq = qpf2k::has_gsq_gate(manifest);
    const auto decode_pool = qsa_decode_pool_route(options, gsq) ? qsa_decode_pool_bytes(options, model, gsq) : 0U;
    result.state_d1 = add(result.state_d1, decode_pool);
    result.device_base = add(result.device_base, decode_pool);
    result.workspace = add(add(plan.arena.physical.total_bytes, 65552U),plan.qsa_score_ring_workspace_bytes);
    const auto map_bytes = mul(plan.logit_row_indices.size(), sizeof(std::uint32_t));
    result.metadata = add(add(plan.host_metadata_payload_bytes, map_bytes), add(plan.host_metadata_payload_bytes, map_bytes));
    result.runtime_payload = add(mul(mul(plan.logical_rows, compiled.execution.steps.size()), sizeof(std::uint32_t)),
        add(map_bytes, add(mul(plan.logit_row_indices.size(), sizeof(ExecutionBufferAddress)), mul(model.vocab_size, 2U * sizeof(float)))));
    YarnPrefillOwnerGeometry geometry;
    geometry.compact_scores = plan.qsa_score_ring_workspace_bytes != 0;
    geometry.decode_pool_bytes = decode_pool;
    const auto collect_geometry = [&](const CompiledKnownPrefixPrefill& item) {
        geometry.ring_bytes = add(geometry.ring_bytes,item.qsa_score_ring_workspace_bytes);
        if (item.terminal_logits_arena)
            for (const auto& allocation : item.arena.physical.allocations)
                if (execution_use_is_logits(allocation.use))
                    geometry.terminal_logits_bytes = add(geometry.terminal_logits_bytes,allocation.bytes);
        geometry.compact_scores = geometry.compact_scores && item.qsa_score_ring_workspace_bytes != 0;
    };
    collect_geometry(plan);
    geometry.primary_compact_bytes = add(geometry.ring_bytes,geometry.terminal_logits_bytes);
    for (const auto rows : options.known_prefix_tail_rows) {
        const auto tail = make_plan(rows);
        collect_geometry(tail);
        result.tails = add(result.tails, add(add(tail.arena.physical.total_bytes, 65552U),tail.qsa_score_ring_workspace_bytes));
        result.metadata = add(result.metadata, add(tail.host_metadata_payload_bytes, tail.host_metadata_payload_bytes));
    }
    if (options.rotary.mode() == RotaryMode::yarn)
        append_yarn_owner_requirements(options,model,state,model_bytes,result,geometry);
    result.organization = {{"serial_execution_identity",compiled.execution_identity},
        {"serial_kernel_plan_identity",compiled.kernel_plan_identity},
        {"prefill_execution_identity",plan.prefill_execution_identity},
        {"prefill_kernel_plan_identity",plan.prefill_kernel_plan_identity},
        {"serial_scratch_bytes",compiled.arena.total_bytes},
        {"prefill_arena_bytes",plan.arena.physical.total_bytes},
        {"prefill_metadata_bytes",plan.host_metadata_payload_bytes},
        {"qsa_decode_pool_bytes",decode_pool},
        {"qsa_decode_pool_key_bytes",qsa_decode_pool_key_bytes(options, gsq)}};
    if(plan.terminal_logits_arena) result.organization.emplace("terminal_logits_arena",1U);
    if(plan.qsa_score_ring_workspace_bytes) {
        result.organization.emplace("qsa_score_ring_workspace_bytes",plan.qsa_score_ring_workspace_bytes);
        result.organization.emplace("qsa_score_ring_rows",plan.qsa_score_ring_rows);
    }
    return result;
}
} // namespace strix::decode_cli
