#include "strix/compiled_execution.hpp"

#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix {
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
    hash_u64(hash, model.indexer_kv_heads);
    hash_u64(hash, model.indexer_head_dim);
    hash_u64(hash, model.indexer_compress_ratio);
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

    CompiledSerialExecution compiled;
    compiled.logical_rows =
        binding_request.logical_rows;
    compiled.request_concurrency =
        binding_request.request_concurrency;
    compiled.vocab_size = model.vocab_size;
    compiled.max_position_embeddings =
        model.max_position_embeddings;

    compiled.execution =
        build_serial_decode_execution_plan(model);
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

    compiled.io =
        build_serial_execution_io_plan(
            compiled.execution);
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
            graph_safe = false;
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
        &resources);
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
        &session_state_memory);
}

}  // namespace strix
