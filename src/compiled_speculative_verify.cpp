#include "strix/compiled_speculative_verify.hpp"
#include "strix/moe_q4k_a8_runtime.hpp"

#include <span>
#include <stdexcept>
#include <vector>

namespace strix {

namespace {
GraphExecutionContract verify_graph_contract(const CompiledSpeculativeVerifyExecution& compiled) {
    bool safe = compiled.whole_plan_graph_safe && compiled.dynamic_steps == 0 &&
        !compiled.verify.steps.empty() && compiled.bindings.missing_steps == 0 &&
        compiled.bindings.bindings.size() == compiled.verify.steps.size();
    for (const auto& step : compiled.verify.steps) {
        safe = safe && !step.dynamic_shape && step.graph_safe_candidate &&
            step.mode != SpeculativeVerifyMode::dynamic_row_parallel &&
            step.mode != SpeculativeVerifyMode::runtime_position_window;
    }
    for (const auto& binding : compiled.bindings.bindings) {
        safe = safe && !binding.host_or_runtime_step && binding.kernel_id.has_value();
    }
    return {compiled.execution_identity, compiled.kernel_plan_identity, compiled.candidate_depth, safe};
}
} // namespace

GraphCapturePlan prepare_compiled_verify_graph_capture(
    const CompiledSpeculativeVerifyExecution& compiled, const ExecutionRuntimeResources& resources,
    const GraphDynamicArguments& arguments, const GraphMemoryRequest& memory) {
    return prepare_graph_capture_plan(
        verify_graph_contract(compiled),
        compiled.arena, resources, arguments, memory);
}

GraphEligibility assess_compiled_verify_graph_replay(
    const CompiledSpeculativeVerifyExecution& compiled, const GraphCapturePlan& capture,
    const ExecutionRuntimeResources& resources, const GraphDynamicArguments& arguments,
    std::uint64_t position) {
    return assess_graph_replay_eligibility(capture,
        verify_graph_contract(compiled),
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
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t build_identity(
    const ModelContract& model,
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionBufferRequest& buffer_request) noexcept {
    std::uint64_t hash = fnv_offset;

    hash_u64(hash, model.vocab_size);
    hash_u64(hash, model.hidden_size);
    hash_u64(hash, model.max_position_embeddings);
    hash_u64(hash, model.hidden_layers);
    hash_u64(hash, model.full_attention_interval);
    hash_u64(hash, model.ple_layer_1based);
    hash_u64(hash, model.hc_count);
    hash_u64(hash, model.gdn_layers);
    hash_u64(hash, model.num_experts);
    hash_u64(hash, model.num_experts_per_token);

    hash_u64(hash, compiled.candidate_depth);
    hash_u64(hash, compiled.request_concurrency);
    hash_u64(hash, compiled.kernel_plan_identity);

    hash_u64(hash, buffer_request.activation_element_bytes);
    hash_u64(hash, buffer_request.control_element_bytes);
    hash_u64(hash, buffer_request.index_element_bytes);
    hash_u64(hash, buffer_request.logit_element_bytes);
    hash_u64(hash, buffer_request.mlp_intermediate_element_bytes);

    if (buffer_request.parallel_moe_router) hash_u64(hash, 0x524f555445525057ULL);

    for (const auto& step :
         compiled.verify.steps) {
        hash_u64(hash, step.id);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.op));
        hash_u64(
            hash,
            step.layer.has_value()
                ? static_cast<std::uint64_t>(*step.layer) + 1ULL
                : 0ULL);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.mode));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.reads_state));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.writes_state));
        hash_u64(hash, step.dynamic_shape ? 1ULL : 0ULL);
        hash_u64(hash, step.graph_safe_candidate ? 1ULL : 0ULL);
        for (const auto dependency : step.dependencies) {
            hash_u64(hash, dependency);
        }
        hash_u64(hash, 0xffffffffffffffffULL);
    }

    hash_u64(hash, compiled.arena.alignment);
    hash_u64(hash, compiled.arena.total_bytes);
    for (const auto& allocation :
         compiled.arena.allocations) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                allocation.use.buffer.buffer_class));
        hash_u64(hash, allocation.use.buffer.slot);
        hash_u64(hash, allocation.use.generation);
        hash_u64(hash, allocation.first_step);
        hash_u64(hash, allocation.last_step);
        hash_u64(hash, allocation.offset);
        hash_u64(hash, allocation.bytes);
    }

    return hash;
}

void validate_topology_match(
    const SerialExecutionPlan& topology,
    const SpeculativeVerifyPlan& verify) {
    if (topology.steps.size() !=
        verify.steps.size()) {
        throw std::logic_error(
            "speculative verify topology size differs from serial model topology");
    }

    for (std::size_t index = 0;
         index < topology.steps.size();
         ++index) {
        const auto& serial =
            topology.steps[index];
        const auto& step =
            verify.steps[index];
        if (serial.id != step.id ||
            serial.op != step.op ||
            serial.layer != step.layer ||
            serial.dependencies != step.dependencies ||
            serial.reads_state != step.reads_state ||
            serial.writes_state != step.writes_state) {
            throw std::logic_error(
                "speculative verify topology no longer matches serial operator graph");
        }
    }
}

}  // namespace

CompiledSpeculativeVerifyExecution
compile_speculative_verify_execution(
    const ModelContract& model,
    std::uint32_t candidate_depth,
    const kernels::KernelRegistry& registry,
    const SpeculativeVerifyBindingRequest& binding_request,
    const ExecutionBufferRequest& buffer_request,
    std::uint64_t arena_alignment) {
    if (buffer_request.q4k_a8g32_gate_up)
        throw std::invalid_argument("Q4K A8 runtime does not support speculative verify");
    if (candidate_depth == 0) {
        throw std::invalid_argument(
            "compiled speculative verify requires positive depth");
    }
    if (binding_request.request_concurrency != 1) {
        throw std::invalid_argument(
            "initial speculative verify compilation supports one request only");
    }
    if (buffer_request.logical_rows != candidate_depth) {
        throw std::invalid_argument(
            "verify buffer logical rows must equal candidate depth");
    }

    if (binding_request.ranked_moe_router_topk && !binding_request.parallel_moe_router)
        throw std::invalid_argument("ranked router requires parallel router topology");
    if (binding_request.parallel_moe_router != buffer_request.parallel_moe_router)
        throw std::invalid_argument("router kernel request differs from buffer admission");

    CompiledSpeculativeVerifyExecution compiled;
    compiled.vocab_size =
        model.vocab_size;
    compiled.candidate_depth =
        candidate_depth;
    compiled.request_concurrency =
        binding_request.request_concurrency;
    compiled.topology =
        build_serial_decode_execution_plan(model);
    compiled.verify =
        build_speculative_verify_plan(
            model,
            candidate_depth);

    validate_topology_match(
        compiled.topology,
        compiled.verify);

    compiled.bindings =
        bind_speculative_verify_plan(
            model,
            compiled.verify,
            registry,
            binding_request);
    for (const auto& binding : compiled.bindings.bindings)
        if (binding.op == SerialOp::moe_experts && binding.kernel_id &&
            binding.kernel_id->ends_with(moe_q4k_a8::runtime_kernel_suffix))
            throw std::invalid_argument("Q4K A8 runtime does not support speculative verify");
    if (compiled.bindings.missing_steps != 0) {
        throw std::invalid_argument(
            "compiled speculative verify requires all kernel steps bound");
    }

    for (const auto& binding : compiled.bindings.bindings) {
        if (binding.op == SerialOp::moe_route && binding.kernel_id.has_value() &&
            (binding.kernel_id->ends_with(".parallel") != buffer_request.parallel_moe_router ||
             binding.kernel_id->ends_with(".ranked.parallel") != binding_request.ranked_moe_router_topk))
            throw std::invalid_argument("bound router kernel differs from admitted topology");
    }

    compiled.io =
        build_serial_execution_io_plan(
            compiled.topology, buffer_request.parallel_moe_router);
    compiled.arena =
        build_serial_execution_arena_plan(
            model,
            buffer_request,
            compiled.topology,
            compiled.io,
            arena_alignment);

    std::vector<const kernels::KernelVariant*>
        selected;
    selected.reserve(
        compiled.bindings.bound_steps);

    bool graph_safe = true;

    for (std::size_t index = 0;
         index < compiled.verify.steps.size();
         ++index) {
        const auto& step =
            compiled.verify.steps[index];
        const auto& binding =
            compiled.bindings.bindings[index];

        switch (step.mode) {
        case SpeculativeVerifyMode::sequential_window:
            ++compiled.sequential_window_steps;
            break;
        case SpeculativeVerifyMode::causal_window:
            ++compiled.causal_window_steps;
            break;
        case SpeculativeVerifyMode::dynamic_row_parallel:
            ++compiled.dynamic_steps;
            graph_safe = false;
            break;
        case SpeculativeVerifyMode::runtime_position_window:
            graph_safe = false;
            break;
        case SpeculativeVerifyMode::row_parallel:
            break;
        }

        if (step.dynamic_shape &&
            step.mode !=
                SpeculativeVerifyMode::dynamic_row_parallel) {
            ++compiled.dynamic_steps;
            graph_safe = false;
        }
        if (!step.graph_safe_candidate) {
            graph_safe = false;
        }

        if (binding.host_or_runtime_step) {
            graph_safe = false;
            continue;
        }
        if (!binding.kernel_id.has_value()) {
            throw std::logic_error(
                "fully bound verify step lacks kernel id");
        }

        const auto* variant =
            registry.find(
                *binding.kernel_id);
        if (variant == nullptr ||
            variant->phase !=
                kernels::ExecutionPhase::
                    speculative_verify) {
            throw std::logic_error(
                "verify binding refers to missing or wrong-phase kernel");
        }
        selected.push_back(variant);
        if (!variant->graph_safe) {
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
            "verify kernel plan identity must be non-zero");
    }

    compiled.whole_plan_graph_safe =
        graph_safe;
    compiled.execution_identity =
        build_identity(
            model,
            compiled,
            buffer_request);
    if (compiled.execution_identity == 0) {
        throw std::logic_error(
            "verify execution identity must be non-zero");
    }

    return compiled;
}

}  // namespace strix
