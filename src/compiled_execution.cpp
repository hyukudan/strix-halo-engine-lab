#include "strix/compiled_execution.hpp"
#include "strix/qsa_selection_plan.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include <algorithm>
#include "strix/shared_expert_prefetch.hpp"
#include "strix/moe_q4k_a8_runtime.hpp"

#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix {

namespace {
GraphExecutionContract serial_graph_contract(const CompiledSerialExecution& compiled) {
    bool safe = compiled.whole_plan_graph_safe && compiled.dynamic_steps == 0 &&
        compiled.non_graph_safe_steps == 0 && !compiled.execution.steps.empty() &&
        compiled.bindings.missing_steps == 0 &&
        compiled.bindings.bindings.size() == compiled.execution.steps.size();
    for (const auto& step : compiled.execution.steps) {
        safe = safe && !step.dynamic_shape && step.graph_safe_candidate;
    }
    for (const auto& binding : compiled.bindings.bindings) {
        safe = safe && ((compiled.device_graph_inputs && binding.host_or_runtime_step &&
            binding.op == SerialOp::position_prepare) ||
            (!binding.host_or_runtime_step && binding.kernel_id.has_value()));
    }
    return {compiled.execution_identity, compiled.kernel_plan_identity, compiled.logical_rows, safe};
}
} // namespace

GraphCapturePlan prepare_compiled_serial_graph_capture(
    const CompiledSerialExecution& compiled, const ExecutionRuntimeResources& resources,
    const GraphDynamicArguments& arguments, const GraphMemoryRequest& memory) {
    return prepare_graph_capture_plan(
        serial_graph_contract(compiled),
        compiled.arena, resources, arguments, memory);
}

GraphEligibility assess_compiled_serial_graph_replay(
    const CompiledSerialExecution& compiled, const GraphCapturePlan& capture,
    const ExecutionRuntimeResources& resources, const GraphDynamicArguments& arguments,
    std::uint64_t position) {
    return assess_graph_replay_eligibility(capture,
        serial_graph_contract(compiled),
        compiled.arena, resources, arguments, position);
}

namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void hash_bool(
    std::uint64_t& hash,
    bool value) noexcept {
    hash_u64(hash, value ? 1ULL : 0ULL);
}

void hash_buffer_use(
    std::uint64_t& hash,
    ExecutionBufferUse use) noexcept {
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            use.buffer.buffer_class));
    hash_u64(hash, use.buffer.slot);
    hash_u64(hash, use.generation);
}

void hash_model(
    std::uint64_t& hash,
    const ModelContract& model) noexcept {
    hash_u64(hash, model.vocab_size);
    hash_u64(hash, model.hidden_size);
    hash_u64(hash, model.max_position_embeddings);
    hash_u64(hash, model.hidden_layers);
    hash_u64(hash, model.full_attention_interval);
    hash_u64(hash, model.ple_layer_1based);
    hash_u64(hash, model.ple_embed_dim);
    hash_u64(hash, model.ple_conv_kernel_size);
    hash_u64(hash, model.ngram_size);
    hash_u64(hash, model.heads_per_ngram);
    hash_u64(hash, model.ngram_vocab_size_base);
    hash_u64(hash, model.ngram_vocab_divisor);
    hash_u64(hash, model.ngram_hash_seed);
    hash_u64(hash, model.ple_eos_token_id);
    hash_u64(hash, model.hc_count);
    hash_u64(hash, model.hc_lowrank);
    hash_u64(hash, model.full_attention_layers);
    hash_u64(hash, model.gdn_layers);
    hash_u64(hash, model.attention_heads);
    hash_u64(hash, model.kv_heads);
    hash_u64(hash, model.head_dim);
    hash_u64(hash, model.mtp_full_attention_layers);
    hash_u64(hash, model.gdn_key_heads);
    hash_u64(hash, model.gdn_key_head_dim);
    hash_u64(hash, model.gdn_value_heads);
    hash_u64(hash, model.gdn_value_head_dim);
    hash_u64(hash, model.gdn_conv_kernel);
    hash_u64(hash, model.indexer_n_heads);
    hash_u64(hash, model.indexer_kv_heads);
    hash_u64(hash, model.indexer_head_dim);
    hash_u64(hash, model.indexer_budget);
    hash_u64(hash, model.indexer_compress_ratio);

    std::uint32_t partial_rotary_bits = 0;
    static_assert(
        sizeof(partial_rotary_bits) ==
        sizeof(model.partial_rotary_factor));
    std::memcpy(
        &partial_rotary_bits,
        &model.partial_rotary_factor,
        sizeof(partial_rotary_bits));
    hash_u64(hash, partial_rotary_bits);

    std::uint32_t rope_theta_bits = 0;
    static_assert(
        sizeof(rope_theta_bits) ==
        sizeof(model.rope_theta));
    std::memcpy(
        &rope_theta_bits,
        &model.rope_theta,
        sizeof(rope_theta_bits));
    hash_u64(hash, rope_theta_bits);

    hash_u64(hash, model.moe_intermediate_size);
    hash_u64(
        hash,
        model.shared_expert_intermediate_size);
    hash_u64(hash, model.num_experts);
    hash_u64(hash, model.num_experts_per_token);

    std::uint32_t rms_epsilon_bits = 0;
    static_assert(
        sizeof(rms_epsilon_bits) ==
        sizeof(model.rms_norm_epsilon));
    std::memcpy(
        &rms_epsilon_bits,
        &model.rms_norm_epsilon,
        sizeof(rms_epsilon_bits));
    hash_u64(hash, rms_epsilon_bits);
    hash_bool(hash, model.norm_topk_probability);
    hash_bool(hash, model.gdn_output_gate_sigmoid);
    hash_bool(hash, model.gdn_qk_l2norm);
    std::uint32_t gdn_qk_epsilon_bits = 0;
    static_assert(sizeof(gdn_qk_epsilon_bits) == sizeof(model.gdn_qk_l2_epsilon));
    std::memcpy(
        &gdn_qk_epsilon_bits,
        &model.gdn_qk_l2_epsilon,
        sizeof(gdn_qk_epsilon_bits));
    hash_u64(hash, gdn_qk_epsilon_bits);
    hash_bool(hash, model.gdn_conv_silu);

    hash_u64(hash, model.bytes_bf16);
    hash_u64(hash, model.bytes_fp32);
}

std::uint64_t build_execution_identity(
    const ModelContract& model,
    const CompiledSerialExecution& compiled,
    const ExecutionBindingRequest& binding_request,
    const ExecutionBufferRequest& buffer_request) noexcept {
    std::uint64_t hash = fnv_offset;

    hash_model(hash, model);
    hash_u64(
        hash,
        compiled.kernel_plan_identity);
    hash_u64(hash, compiled.logical_rows);
    hash_u64(
        hash,
        compiled.request_concurrency);
    hash_bool(
        hash,
        binding_request.require_graph_safe);
    hash_bool(hash, binding_request.device_graph_inputs);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            binding_request.minimum_qualification));

    hash_u64(
        hash,
        buffer_request.activation_element_bytes);
    hash_u64(
        hash,
        buffer_request.control_element_bytes);
    hash_u64(
        hash,
        buffer_request.index_element_bytes);
    hash_u64(
        hash,
        buffer_request.logit_element_bytes);
    hash_u64(
        hash,
        buffer_request.mlp_intermediate_element_bytes);

    if (buffer_request.parallel_moe_router) hash_u64(hash, 0x524f555445525057ULL);
    if (buffer_request.q4k_a8g32_gate_up) hash_u64(hash, 0x51344b4138473332ULL);
    if (buffer_request.grouped_exact_q51_down) hash_u64(hash, 0x5135314752505631ULL);
    if (buffer_request.q51_slot_parallel_down) hash_u64(hash, 0x5135315350415231ULL);
    if (buffer_request.a8_down_q51) hash_u64(hash, 0x5135314138444e31ULL);
    if (buffer_request.a8_down_q80) hash_u64(hash, 0x5138304138444e31ULL);
    if (buffer_request.q5k_a8g32_gate_up) hash_u64(hash, 0x51354b4138475531ULL);
    if (buffer_request.qsa_hierarchical_selection) hash_u64(hash, 0x515341544f503531ULL);
    if (buffer_request.qsa_gqa_heads_per_block != 1) {
        hash_u64(hash, 0x5153414751417631ULL);
        hash_u64(hash, buffer_request.qsa_gqa_heads_per_block);
    }

    hash_u64(hash, compiled.arena.alignment);
    hash_u64(hash, compiled.arena.total_bytes);
    for (const auto& allocation : compiled.arena.allocations) {
        hash_buffer_use(hash, allocation.use);
        hash_u64(hash, allocation.first_step);
        hash_u64(hash, allocation.last_step);
        hash_u64(hash, allocation.offset);
        hash_u64(hash, allocation.bytes);
    }
    hash_u64(hash, 0xfffffffffffffffcULL);

    for (const auto& step :
         compiled.execution.steps) {
        hash_u64(hash, step.id);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                step.op));
        hash_u64(
            hash,
            step.layer.has_value()
                ? static_cast<std::uint64_t>(
                      *step.layer) + 1ULL
                : 0ULL);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                step.reads_state));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                step.writes_state));
        hash_bool(
            hash,
            step.graph_safe_candidate);
        hash_bool(hash, step.dynamic_shape);
        for (const auto dependency :
             step.dependencies) {
            hash_u64(hash, dependency);
        }
        hash_u64(
            hash,
            0xffffffffffffffffULL);
    }

    for (const auto& step : compiled.io.steps) {
        hash_u64(hash, step.step_id);
        for (const auto use : step.reads) {
            hash_buffer_use(hash, use);
        }
        hash_u64(
            hash,
            0xfffffffffffffffeULL);
        for (const auto use : step.writes) {
            hash_buffer_use(hash, use);
        }
        hash_u64(
            hash,
            0xfffffffffffffffdULL);
        for (const auto use : step.temporaries) {
            hash_buffer_use(hash, use);
        }
        hash_u64(
            hash,
            0xfffffffffffffffbULL);
    }

    return hash;
}

}  // namespace

CompiledSerialExecution compile_serial_execution(
    const ModelContract& model,
    const kernels::KernelRegistry& registry,
    const ExecutionBindingRequest& binding_request,
    const ExecutionBufferRequest& buffer_request,
    std::uint64_t arena_alignment) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (binding_request.logical_rows == 0 ||
        binding_request.request_concurrency == 0) {
        throw std::invalid_argument(
            "compiled execution rows and concurrency must be non-zero");
    }
    if (buffer_request.logical_rows !=
        binding_request.logical_rows) {
        throw std::invalid_argument(
            "compiled execution buffer rows must match binding rows");
    }

    if (binding_request.wave8_q8_lm_head && !binding_request.prefetch_q8_lm_head)
        throw std::invalid_argument("wave8 Q8 LM head requires prefetch organization");
    if (binding_request.prefetch_hc_projection_next_tile &&
        (binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1 ||
         model.hidden_size != 2560 || model.hc_count != 4 || model.hc_lowrank != 320 || model.hidden_layers != 48 ||
         model.bytes_bf16 != 2 || model.bytes_fp32 != 4 || buffer_request.activation_element_bytes != 2))
        throw std::invalid_argument("HC next-tile prefetch requires canonical eager C1 execution");
    if ((binding_request.prefetch_q8_lm_head || binding_request.wave8_q8_lm_head || binding_request.wave8_q8_gdn_input ||
         binding_request.wave8_q8_gdn_output || binding_request.wave8_q8_qsa_output) &&
        (binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("Q8 projection organizations require eager C1 execution");
    if (binding_request.wave_moe_router_topk &&
        (!binding_request.parallel_moe_router || binding_request.ranked_moe_router_topk))
        throw std::invalid_argument("wave router requires parallel topology and excludes ranked mode");
    if (binding_request.batched_moe_activation && (!binding_request.q4k_a8g32_gate_up ||
        binding_request.device_graph_inputs || binding_request.require_graph_safe ||
        binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("batched MoE activation requires eager C1 A8 execution");
    if (binding_request.deferred_moe_status &&
        (binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("deferred MoE status supports eager C1 only");
    if (binding_request.ranked_moe_router_topk && !binding_request.parallel_moe_router)
        throw std::invalid_argument("ranked router requires parallel router topology");
    if (binding_request.parallel_moe_router != buffer_request.parallel_moe_router)
        throw std::invalid_argument("router kernel request differs from buffer admission");

    validate_gdn_recurrent_storage(binding_request.gdn_recurrent_storage);
    if (binding_request.gdn_recurrent_storage != GdnRecurrentStorage::key_value &&
        (binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("GDN recurrent storage candidate requires eager C1 execution");
    (void)gdn_recurrent_layout_marker(binding_request.gdn_recurrent_layout);
    if (binding_request.gdn_recurrent_layout != HipGdnRecurrentLayout::baseline &&
        (binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("GDN recurrent LDS layout requires eager C1 execution");
    (void)qsa_gqa_organization_marker(binding_request.qsa_gqa_heads_per_block);
    if (binding_request.qsa_gqa_heads_per_block != buffer_request.qsa_gqa_heads_per_block)
        throw std::invalid_argument("QSA GQA organization differs from buffer admission");
    if (binding_request.qsa_gqa_heads_per_block != 1 &&
        (!binding_request.deferred_moe_status || binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("QSA GQA organization requires eager C1 deferred completion");
    if (binding_request.qsa_hierarchical_selection != buffer_request.qsa_hierarchical_selection)
        throw std::invalid_argument("hierarchical QSA request differs from buffer admission");
    if (binding_request.qsa_hierarchical_selection &&
        (!binding_request.deferred_moe_status || binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("hierarchical QSA requires eager C1 deferred completion");
    if (binding_request.q51_slot_parallel_down != buffer_request.q51_slot_parallel_down)
        throw std::invalid_argument("slot-parallel Q5_1 request differs from buffer admission");
    if (binding_request.q5k_a8g32_gate_up != buffer_request.q5k_a8g32_gate_up)
        throw std::invalid_argument("Q5K A8 gate/up differs from buffer admission");
    if (binding_request.q5k_a8g32_gate_up && (!binding_request.a8_down_q51 || !binding_request.a8_down_q80))
        throw std::invalid_argument("Q5K A8 gate/up requires both A8 down formats");
    if (binding_request.a8_down_q51 != buffer_request.a8_down_q51 ||
        binding_request.a8_down_q80 != buffer_request.a8_down_q80)
        throw std::invalid_argument("A8 down arithmetic differs from buffer admission");
    if ((binding_request.a8_down_q51 || binding_request.a8_down_q80) &&
        (!binding_request.a8_down_q51 || !binding_request.q4k_a8g32_gate_up ||
         !binding_request.q51_slot_parallel_down || !binding_request.batched_moe_activation ||
         !binding_request.deferred_moe_status || binding_request.device_graph_inputs ||
         binding_request.require_graph_safe || binding_request.logical_rows != 1 ||
         binding_request.request_concurrency != 1))
        throw std::invalid_argument("A8 down arithmetic requires eager one-row deferred A8 execution");
    if (binding_request.q51_slot_parallel_down &&
        (!binding_request.grouped_exact_q51_down || !binding_request.q4k_a8g32_gate_up ||
         binding_request.device_graph_inputs || binding_request.require_graph_safe ||
         binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("slot-parallel Q5_1 down requires grouped eager C1 A8 execution");
    if(binding_request.grouped_exact_q51_down != buffer_request.grouped_exact_q51_down)
        throw std::invalid_argument("grouped exact Q5_1 down request differs from buffer admission");
    if(binding_request.grouped_exact_q51_down && !binding_request.q4k_a8g32_gate_up)
        throw std::invalid_argument("grouped exact Q5_1 down requires Q4K A8 slot workspace");
    if (binding_request.q4k_a8g32_gate_up != buffer_request.q4k_a8g32_gate_up)
        throw std::invalid_argument("Q4K A8 kernel request differs from buffer admission");
    if (binding_request.q4k_a8g32_gate_up &&
        (binding_request.logical_rows != 1 || binding_request.request_concurrency != 1))
        throw std::invalid_argument("Q4K A8 runtime supports only C1 one-row execution");

    CompiledSerialExecution compiled;
    if (binding_request.device_graph_inputs && (!binding_request.require_graph_safe ||
        binding_request.logical_rows != 1 || binding_request.request_concurrency != 1)) {
        throw std::invalid_argument("device graph topology requires explicit graph-safe C1 selection");
    }
    compiled.device_graph_inputs = binding_request.device_graph_inputs;
    compiled.logical_rows =
        binding_request.logical_rows;
    compiled.request_concurrency =
        binding_request.request_concurrency;
    compiled.vocab_size = model.vocab_size;
    compiled.max_position_embeddings =
        model.max_position_embeddings;

    compiled.execution =
        build_serial_decode_execution_plan(model);
    if (compiled.device_graph_inputs) {
        // Selected variants must implement fixed maximum launch geometry and
        // device-driven live counts. The eager dynamic topology stays unchanged.
        for (auto& step : compiled.execution.steps) {
            step.dynamic_shape = false;
            step.graph_safe_candidate = true;
        }
    }
    compiled.bindings =
        bind_serial_execution_plan(
            model,
            compiled.execution,
            registry,
            binding_request);

    if (compiled.bindings.missing_steps != 0) {
        throw std::invalid_argument(
            "compiled serial execution requires every kernel step to bind");
    }

    for (const auto& binding : compiled.bindings.bindings) {
        if (binding.kernel_id) {
            constexpr std::string_view marker = ".prefetch-next-tile-v1";
            const auto position = binding.kernel_id->find(marker);
            const bool hc = binding.op == SerialOp::attn_hyper_mix || binding.op == SerialOp::mlp_hyper_mix ||
                            binding.op == SerialOp::final_hyper_mix;
            if ((position != std::string::npos) != (hc && binding_request.prefetch_hc_projection_next_tile))
                throw std::invalid_argument("bound HC next-tile prefetch differs from request");
            if (position != std::string::npos) {
                const std::string_view tail = std::string_view(*binding.kernel_id).substr(position + marker.size());
                if (!tail.empty() && tail != lane_chain_hyper_output_suffix)
                    throw std::invalid_argument("bound HC next-tile marker must precede only the terminal lane-chain suffix");
            }
        }
        if (binding.op == SerialOp::gated_delta_net && binding.kernel_id) {
            const auto marker = gdn_recurrent_storage_marker(GdnRecurrentStorage::value_key);
            const auto first = binding.kernel_id->find(marker);
            const bool selected = binding_request.gdn_recurrent_storage == GdnRecurrentStorage::value_key;
            if ((first != std::string::npos) != selected ||
                (first != std::string::npos && binding.kernel_id->find(marker, first + marker.size()) != std::string::npos))
                throw std::invalid_argument("bound GDN recurrent storage marker differs from request");
        }
        if (binding.op == SerialOp::gated_delta_net && binding.kernel_id) {
            const auto expected = gdn_recurrent_layout_marker(binding_request.gdn_recurrent_layout);
            unsigned found = 0;
            for (const auto layout : {HipGdnRecurrentLayout::stride32, HipGdnRecurrentLayout::stride33, HipGdnRecurrentLayout::transposed}) {
                const auto marker = gdn_recurrent_layout_marker(layout);
                if (binding.kernel_id->find(marker) != std::string::npos) {
                    ++found;
                    if (marker != expected) throw std::invalid_argument("bound GDN recurrent LDS layout differs from request");
                }
            }
            if (found != (expected.empty() ? 0U : 1U))
                throw std::invalid_argument("bound GDN recurrent LDS layout marker is missing or ambiguous");
        }
        if (binding.op == SerialOp::gated_delta_net && binding.kernel_id &&
            (binding.kernel_id->find(q8_projection_organization::gdn_wave8_marker) != std::string::npos) != binding_request.wave8_q8_gdn_input)
            throw std::invalid_argument("bound GDN wave8 organization differs from request");
        if (binding.op == SerialOp::gated_delta_net && binding.kernel_id &&
            (binding.kernel_id->find(q8_projection_organization::gdn_output_wave8_marker) != std::string::npos) != binding_request.wave8_q8_gdn_output)
            throw std::invalid_argument("bound GDN output wave8 organization differs from request");
        if (binding.op == SerialOp::indexed_attention && binding.kernel_id &&
            (binding.kernel_id->find(q8_projection_organization::qsa_output_wave8_marker) != std::string::npos) != binding_request.wave8_q8_qsa_output)
            throw std::invalid_argument("bound QSA output wave8 organization differs from request");
        if (binding.op == SerialOp::indexed_attention && binding.kernel_id) {
            const auto expected = qsa_gqa_organization_marker(binding_request.qsa_gqa_heads_per_block);
            unsigned markers = 0;
            for (const auto heads : {2U, 4U}) {
                const auto marker = qsa_gqa_organization_marker(heads);
                if (binding.kernel_id->find(marker) != std::string::npos) {
                    ++markers;
                    if (marker != expected) throw std::invalid_argument("bound QSA GQA organization differs from request");
                }
            }
            if (markers != (expected.empty() ? 0U : 1U))
                throw std::invalid_argument("bound QSA GQA marker is missing or ambiguous");
        }
        if (binding.op == SerialOp::indexed_attention && binding.kernel_id &&
            (binding.kernel_id->find(qsa_hierarchical_selection_marker) != std::string::npos) != binding_request.qsa_hierarchical_selection)
            throw std::invalid_argument("bound hierarchical QSA selection differs from request");
        if (binding.op == SerialOp::lm_head && binding.kernel_id) {
            const bool p1 = binding.kernel_id->find(q8_projection_organization::lm_prefetch_marker) != std::string::npos;
            const bool p8 = binding.kernel_id->find(q8_projection_organization::lm_wave8_marker) != std::string::npos;
            if ((p1 && p8) || (p1 || p8) != binding_request.prefetch_q8_lm_head || p8 != binding_request.wave8_q8_lm_head)
                throw std::invalid_argument("bound LM prefetch organization differs from request");
        }
        if(binding.op==SerialOp::indexed_attention && binding.kernel_id){const auto& step=compiled.execution.steps[binding.step_id];if(binding.kernel_id->ends_with(qsa_input_prefetch::kernel_suffix)!=qsa_input_prefetch::selected(binding_request.prefetch_q8_qsa_input,binding_request.qsa_input_prefetch_layers,step.layer))throw std::invalid_argument("bound QSA input prefetch differs from admitted topology");}
        if(binding.op==SerialOp::attn_hyper_mix || binding.op==SerialOp::mlp_hyper_mix || binding.op==SerialOp::final_hyper_mix){
            const bool expected=binding_request.lane_chain_hyper_output && std::find(binding_request.lane_chain_hyper_output_eligible_steps.begin(),binding_request.lane_chain_hyper_output_eligible_steps.end(),binding.step_id)!=binding_request.lane_chain_hyper_output_eligible_steps.end();
            if(binding.kernel_id && binding.kernel_id->ends_with(lane_chain_hyper_output_suffix)!=expected)
                throw std::invalid_argument("bound lane-chain HC output differs from admitted topology");
        }
        if (binding.op == SerialOp::moe_experts && binding.kernel_id &&
            (binding.kernel_id->find(".batched-slot-activation-v1") != std::string::npos) != binding_request.batched_moe_activation)
            throw std::invalid_argument("bound batched MoE activation differs from request");
        if (binding.op==SerialOp::moe_experts && binding.kernel_id &&
            (binding.kernel_id->find(".deferred-status-v1")!=std::string::npos)!=binding_request.deferred_moe_status)
            throw std::invalid_argument("bound deferred MoE status mode differs from request");
        if (binding.op == SerialOp::moe_experts && binding.kernel_id) {
            const auto& id = *binding.kernel_id;
            constexpr std::string_view q5k_marker = ".q5k-a8g32-gateup-v1";
            const auto q5k_position = id.find(q5k_marker);
            if ((q5k_position != std::string::npos) != binding_request.q5k_a8g32_gate_up ||
                (q5k_position != std::string::npos &&
                 (id.find(q5k_marker, q5k_position + q5k_marker.size()) != std::string::npos ||
                  std::string_view(id).substr(q5k_position + q5k_marker.size()) !=
                    ".q51-a8g32-down-v1.q80-a8g32-down-v1.mixed-a8-down-v1.q4k-a8g32-gateup-v2")))
                throw std::invalid_argument("bound Q5K A8 gate/up identity differs from request");
            if ((id.find(".q51-a8g32-down-v1") != std::string::npos) != binding_request.a8_down_q51 ||
                (id.find(".q80-a8g32-down-v1") != std::string::npos) != binding_request.a8_down_q80 ||
                id.ends_with(".mixed-a8-down-v1.q4k-a8g32-gateup-v2") !=
                    (binding_request.a8_down_q51 || binding_request.a8_down_q80))
                throw std::invalid_argument("bound A8 down numerical identity differs from request");
        }
        if(binding.op==SerialOp::shared_expert && binding.kernel_id){const auto& step=compiled.execution.steps[binding.step_id];if(binding.kernel_id->ends_with(shared_expert_prefetch::kernel_suffix)!=shared_expert_prefetch::selected(binding_request.prefetch_q8_shared_expert,binding_request.shared_expert_prefetch_layers,step.layer))throw std::invalid_argument("bound shared prefetch differs from admitted topology");}
        if (binding.kernel_id) {
            constexpr std::string_view marker = ".q51-slot-parallel-v1";
            const auto position = binding.kernel_id->find(marker);
            const bool expected = binding.op == SerialOp::moe_experts && binding_request.q51_slot_parallel_down &&
                !binding_request.a8_down_q51;
            if ((position != std::string::npos) != expected)
                throw std::invalid_argument("bound slot-parallel Q5_1 mode differs from request");
            if (position != std::string::npos &&
                std::string_view(*binding.kernel_id).substr(position + marker.size()) != moe_q4k_a8::grouped_q51_kernel_ending)
                throw std::invalid_argument("slot-parallel Q5_1 marker must precede grouped A8 terminal");
        }
        if(binding.op==SerialOp::moe_experts && binding.kernel_id &&
           binding.kernel_id->ends_with(moe_q4k_a8::grouped_q51_kernel_ending)!=(buffer_request.grouped_exact_q51_down && !buffer_request.a8_down_q51))
            throw std::invalid_argument("bound grouped exact Q5_1 down differs from admitted topology");
        if (binding.op == SerialOp::moe_experts && binding.kernel_id.has_value() &&
            binding.kernel_id->ends_with(moe_q4k_a8::runtime_kernel_suffix) != buffer_request.q4k_a8g32_gate_up)
            throw std::invalid_argument("bound Q4K A8 kernel differs from admitted topology");
        if (binding.op == SerialOp::moe_route && binding.kernel_id.has_value() &&
            (binding.kernel_id->ends_with(".parallel") != buffer_request.parallel_moe_router ||
             binding.kernel_id->ends_with(".ranked.parallel") != binding_request.ranked_moe_router_topk ||
             binding.kernel_id->ends_with(".wave.parallel") != binding_request.wave_moe_router_topk))
            throw std::invalid_argument("bound router kernel differs from admitted topology");
    }

    compiled.io =
        build_serial_execution_io_plan(
            compiled.execution, buffer_request.parallel_moe_router);
    compiled.arena =
        build_serial_execution_arena_plan(
            model,
            buffer_request,
            compiled.execution,
            compiled.io,
            arena_alignment);

    std::vector<const kernels::KernelVariant*>
        selected;
    selected.reserve(
        compiled.bindings.bound_steps);

    bool graph_safe = true;

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        const auto& step =
            compiled.execution.steps[index];
        const auto& binding =
            compiled.bindings.bindings[index];

        if (step.dynamic_shape) {
            ++compiled.dynamic_steps;
            graph_safe = false;
        }
        if (!step.graph_safe_candidate) {
            ++compiled.non_graph_safe_steps;
            graph_safe = false;
        }

        if (binding.host_or_runtime_step) {
            graph_safe = graph_safe && compiled.device_graph_inputs && step.op == SerialOp::position_prepare;
            continue;
        }

        if (!binding.kernel_id.has_value()) {
            throw std::logic_error(
                "fully bound execution unexpectedly lacks a kernel id");
        }
        const auto* variant =
            registry.find(*binding.kernel_id);
        if (variant == nullptr) {
            throw std::logic_error(
                "bound kernel id disappeared from registry");
        }
        selected.push_back(variant);

        if (!variant->graph_safe) {
            ++compiled.non_graph_safe_steps;
            graph_safe = false;
        }
    }

    compiled.kernel_plan_identity =
        kernels::plan_identity(
            std::span<
                const kernels::KernelVariant* const>(
                selected.data(),
                selected.size()));

    if (compiled.kernel_plan_identity == 0) {
        throw std::logic_error(
            "compiled kernel plan identity must be non-zero");
    }

    compiled.whole_plan_graph_safe =
        graph_safe;
    compiled.execution_identity =
        build_execution_identity(
            model,
            compiled,
            binding_request,
            buffer_request);

    if (compiled.execution_identity == 0) {
        throw std::logic_error(
            "compiled execution identity must be non-zero");
    }

    return compiled;
}


EagerExecutionResult execute_compiled_serial_eager(
    const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources,
    const EagerExecutionRequest& request,
    EagerExecutionDispatcher& dispatcher) {
    validate_execution_runtime_resources(
        compiled.execution_identity,
        compiled.arena,
        resources,
        request.position);

    if (request.logical_rows !=
            compiled.logical_rows ||
        request.request_concurrency !=
            compiled.request_concurrency) {
        throw std::invalid_argument(
            "runtime shape does not match compiled serial execution");
    }

    return execute_serial_eager(
        compiled.execution,
        compiled.bindings,
        compiled.io,
        compiled.arena,
        request,
        dispatcher,
        resources.scratch,
        resources.session_state_layout,
        resources.session_state_memory,
        &resources,
        compiled.execution_identity);
}

EagerExecutionResult execute_compiled_serial_eager(
    const CompiledSerialExecution& compiled,
    const ExecutionMemoryBinding& memory,
    const SerialSessionStateLayout& session_state_layout,
    const SessionStateMemoryBinding& session_state_memory,
    const EagerExecutionRequest& request,
    EagerExecutionDispatcher& dispatcher) {
    validate_execution_memory_binding(
        compiled.execution_identity,
        compiled.arena,
        memory);
    validate_session_state_memory_binding(
        session_state_layout,
        session_state_memory);

    if (request.logical_rows !=
            compiled.logical_rows ||
        request.request_concurrency !=
            compiled.request_concurrency) {
        throw std::invalid_argument(
            "runtime shape does not match compiled serial execution");
    }

    return execute_serial_eager(
        compiled.execution,
        compiled.bindings,
        compiled.io,
        compiled.arena,
        request,
        dispatcher,
        &memory,
        &session_state_layout,
        &session_state_memory,
        nullptr,
        compiled.execution_identity);
}

}  // namespace strix
