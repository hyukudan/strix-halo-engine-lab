#include "strix/compiled_execution.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/shared_expert_prefetch.hpp"

#include <iostream>
#include <set>
#include <stdexcept>

using namespace strix;
namespace {
int failures = 0;
void expect(bool ok, const char* message) {
    if (!ok) { ++failures; std::cerr << message << '\n'; }
}
kernels::KernelRegistry registry(const ModelContract& model,
                                const ExecutionBindingRequest& request,
                                bool register_alternatives) {
    kernels::KernelRegistry result;
    std::set<std::string> seen;
    for (const auto& step : build_serial_decode_execution_plan(model).steps) {
        const auto family = operator_family_for(step.op);
        if (!family) continue;
        kernels::KernelVariant variant;
        variant.family = *family;
        variant.shape = execution_shape_for(model, step.op, request);
        variant.phase = kernels::ExecutionPhase::decode;
        variant.resources.workgroup_threads = 256;
        variant.numerics = {"bf16", "fp32", "bf16", true, true};
        variant.qualification = kernels::QualificationStatus::experimental;
        variant.input_layout = "host";
        variant.output_layout = "host";
        variant.compiler_identity = "host-contract-only";
        variant.id = "integration." + std::to_string(static_cast<unsigned>(*family)) + "." +
            std::to_string(variant.shape.auxiliary_size);
        if (step.op == SerialOp::moe_route && request.wave_moe_router_topk)
            variant.id += ".wave.parallel";
        if (step.op == SerialOp::moe_experts && request.deferred_moe_status)
            variant.id += ".deferred-status-v1";
        if (!seen.insert(variant.id).second) continue;
        result.add(variant);
        if (!register_alternatives) continue;
        if (step.op == SerialOp::indexed_attention)
            variant.id += qsa_input_prefetch::kernel_suffix;
        else if (step.op == SerialOp::shared_expert)
            variant.id += shared_expert_prefetch::kernel_suffix;
        else if (*family == kernels::OperatorFamily::hyper_mix)
            variant.id += lane_chain_hyper_output_suffix;
        else continue;
        result.add(variant);
    }
    return result;
}
}
int main() {
    const auto model = ModelContract::qwen38_flash_next();
    ExecutionBindingRequest off;
    off.minimum_qualification = kernels::QualificationStatus::experimental;
    expect(!off.wave_moe_router_topk && !off.prefetch_q8_qsa_input &&
           !off.lane_chain_hyper_output && !off.deferred_moe_status &&
           !off.prefetch_q8_shared_expert, "all integrated modes default OFF");
    ExecutionBufferRequest buffers;
    auto alternatives = registry(model, off, true);
    auto base_only = registry(model, off, false);
    const auto original = compile_serial_execution(model, base_only, off, buffers);
    const auto extra = compile_serial_execution(model, alternatives, off, buffers);
    expect(original.execution_identity == extra.execution_identity,
           "additional candidate registrations preserve OFF identity");

    auto on = off;
    on.parallel_moe_router = true;
    on.wave_moe_router_topk = true;
    on.prefetch_q8_qsa_input = true;
    on.qsa_input_prefetch_layers = {7};
    on.prefetch_q8_shared_expert = true;
    on.shared_expert_prefetch_layers = {1};
    on.lane_chain_hyper_output = true;
    on.deferred_moe_status = true;
    for (const auto& step : original.execution.steps)
        if (step.op == SerialOp::final_hyper_mix)
            on.lane_chain_hyper_output_eligible_steps.push_back(step.id);
    auto selected_buffers = buffers;
    selected_buffers.parallel_moe_router = true;
    auto combined_registry = registry(model, on, true);
    const auto combined = compile_serial_execution(model, combined_registry, on, selected_buffers);
    unsigned qsa = 0, shared = 0, hc = 0, router = 0, deferred = 0;
    for (const auto& binding : combined.bindings.bindings) {
        if (!binding.kernel_id) continue;
        const auto& id = *binding.kernel_id;
        const auto& step = combined.execution.steps[binding.step_id];
        if (binding.op == SerialOp::indexed_attention) {
            const bool selected = id.ends_with(qsa_input_prefetch::kernel_suffix);
            expect(selected == (step.layer == 7U), "QSA layer selection remains independent");
            qsa += selected;
        }
        if (binding.op == SerialOp::shared_expert) {
            const bool selected = id.ends_with(shared_expert_prefetch::kernel_suffix);
            expect(selected == (step.layer == 1U), "shared layer selection remains independent");
            shared += selected;
        }
        if (binding.family == kernels::OperatorFamily::hyper_mix) {
            const bool selected = id.ends_with(lane_chain_hyper_output_suffix);
            expect(selected == (binding.op == SerialOp::final_hyper_mix),
                   "HC output step selection remains independent");
            hc += selected;
        }
        if (binding.op == SerialOp::moe_route) router += id.ends_with(".wave.parallel");
        if (binding.op == SerialOp::moe_experts) deferred += id.ends_with(".deferred-status-v1");
    }
    expect(qsa == 1 && shared == 1 && hc == 1 && router == model.hidden_layers &&
           deferred == model.hidden_layers, "all five decoder selections coexist");
    expect(combined.execution_identity != original.execution_identity,
           "combined opt-ins have distinct execution identity");
    auto graph = on;
    graph.require_graph_safe = true;
    try {
        (void)compile_serial_execution(model, combined_registry, graph, selected_buffers);
        expect(false, "combined deferred status must reject graph admission");
    } catch (const std::invalid_argument& error) {
        expect(std::string(error.what()) == "deferred MoE status supports eager C1 only",
               "combined graph rejection preserves specific deferred status contract");
    }
    std::cout << "Spec v1 combined host contracts failures=" << failures << '\n';
    return failures ? 1 : 0;
}
