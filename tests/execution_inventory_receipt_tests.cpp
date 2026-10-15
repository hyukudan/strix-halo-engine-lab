#include "../apps/execution_inventory_receipt.hpp"

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
int calls = 0;
void launcher(const strix::ExecutionDispatchContext&, void*) { ++calls; }
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template <typename Function> bool rejects(Function function) {
    try { function(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

struct Fixture {
    strix::kernels::KernelRegistry variants;
    strix::CompiledSerialExecution compiled;
    strix::CompiledLaunchPlan launches;
    Fixture() {
        const auto model = strix::ModelContract::qwen38_flash_next();
        strix::ExecutionBindingRequest request;
        std::set<std::string> added;
        for (const auto& step : strix::build_serial_decode_execution_plan(model).steps) {
            const auto family = strix::operator_family_for(step.op);
            if (!family) continue;
            const auto shape = strix::execution_shape_for(model, step.op, request);
            const auto key = "fixture.\"\\\n" + std::to_string(static_cast<unsigned>(*family)) + "." +
                std::to_string(shape.hidden_size) + "." + std::to_string(shape.auxiliary_size);
            if (!added.insert(key).second) continue;
            strix::kernels::KernelVariant variant;
            variant.id = key;
            variant.family = *family;
            variant.phase = strix::kernels::ExecutionPhase::decode;
            variant.shape = shape;
            variant.resources.workgroup_threads = 32;
            variant.numerics = {"bf16", "fp32", "bf16", true, true};
            variant.qualification = strix::kernels::QualificationStatus::oracle_matched;
            variant.graph_safe = true;
            variant.input_layout = "row\tmajor";
            variant.output_layout = "fixture";
            variant.compiler_identity = "synthetic-host\"compiler";
            variants.add(variant);
        }
        compiled = strix::compile_serial_execution(model, variants, request, {});
        strix::ExecutionLauncherRegistry registered;
        for (const auto& binding : compiled.bindings.bindings) {
            if (binding.host_or_runtime_step) {
                if (!registered.find_runtime_launcher(binding.op))
                    registered.add_runtime_launcher(binding.op, {"runtime.fixture", 1, launcher, nullptr});
            } else if (!registered.find_kernel_launcher(*binding.kernel_id)) {
                registered.add_kernel_launcher({*binding.kernel_id,
                    static_cast<std::uint64_t>(binding.step_id) + 2U, launcher, nullptr});
            }
        }
        launches = strix::compile_launch_plan(compiled, registered);
    }
    std::string render() const {
        return strix::decode_receipt::execution_inventory_receipt(compiled, launches, variants);
    }
};
} // namespace

int main() {
    const auto environment = strix::decode_receipt::hip_runtime_environment_receipt([](const char* name) -> const char* {
        if (std::string_view(name) == "GPU_MAX_HW_QUEUES") return "1";
        if (std::string_view(name) == "HIP_VISIBLE_DEVICES") return "\"device\\name\n";
        if (std::string_view(name) == "HIP_LAUNCH_BLOCKING") return "";
        return nullptr;
    });
    expect(environment.find("\"GPU_MAX_HW_QUEUES\":\"1\"") != std::string::npos &&
        environment.find("\"HIP_LAUNCH_BLOCKING\":\"\"") != std::string::npos &&
        environment.find("\"HSA_ENABLE_SDMA\":null") != std::string::npos &&
        environment.find("\"effective_hw_queues\":null,\"concurrent_execution_observed\":null") != std::string::npos,
        "requested runtime environment is distinct from effective queues and observed overlap");
    expect(environment.find("device\\\\name\\u000a") != std::string::npos && environment.find('\n') == environment.size() - 1U,
        "runtime environment escapes controls without exposing unrelated variables");
    Fixture fixture;
    const auto text = fixture.render();
    expect(text.starts_with("  \"execution_inventory\":{") && text.ends_with("},\n"),
           "receipt is a reusable comma-terminated JSON fragment");
    expect(text.find("\"evidence_kind\":\"compiled-selected-launchers\"") != std::string::npos &&
           text.find("\"selected_logical_rows\":1,\"selected_request_concurrency\":1") != std::string::npos,
           "receipt exposes only the compiled selected geometry");
    expect(text.find("\"verify_admitted_by_this_receipt\":false") != std::string::npos &&
           text.find("\"prefill_admitted_by_this_receipt\":false") != std::string::npos &&
           text.find("\"native_execution_observed\":null,\"qualification\":null") != std::string::npos,
           "declarations do not become cross-phase admission or native qualification");
    expect(text.find("fixture.\\\"\\\\\\u000a") != std::string::npos &&
           text.find("row\\u0009major") != std::string::npos && text.find('\t') == std::string::npos &&
           text.find('\n') == text.size() - 1U,
           "kernel ids and variant strings escape quotes, backslash and controls");
    expect(text.find("\"runtime_step\":true,\"launcher_bound\":true") != std::string::npos &&
           text.find("\"declared_variant\":null") != std::string::npos &&
           text.find("\"qualification\":\"oracle_matched\"") != std::string::npos,
           "runtime launchers and declared kernel qualification remain distinct");
    expect(calls == 0, "inventory never calls a launcher");

    auto invalid = fixture.launches;
    ++invalid.execution_identity;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(fixture.compiled, invalid, fixture.variants); }),
           "different execution identity rejects");
    invalid = fixture.launches; ++invalid.plan_identity;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(fixture.compiled, invalid, fixture.variants); }),
           "changed launch plan digest rejects using the original compiler hash");
    invalid = fixture.launches; invalid.steps[0].function = nullptr;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(fixture.compiled, invalid, fixture.variants); }),
           "missing callable launcher rejects");
    invalid = fixture.launches; invalid.steps[0].implementation_identity = 0;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(fixture.compiled, invalid, fixture.variants); }),
           "missing implementation identity rejects");
    invalid = fixture.launches; invalid.steps.pop_back();
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(fixture.compiled, invalid, fixture.variants); }),
           "incomplete launch coverage rejects");
    auto compiled = fixture.compiled; ++compiled.execution.steps[0].id;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(compiled, fixture.launches, fixture.variants); }),
           "non-dense step identity rejects");
    compiled = fixture.compiled; compiled.bindings.bindings[0].kernel_id = "unregistered";
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(compiled, fixture.launches, fixture.variants); }),
           "missing declared variant rejects");
    compiled = fixture.compiled; ++compiled.logical_rows;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(compiled, fixture.launches, fixture.variants); }),
           "variant rows cannot be generalized to a different compiled geometry");
    compiled = fixture.compiled; ++compiled.bindings.bound_steps;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(compiled, fixture.launches, fixture.variants); }),
           "binding coverage counters must describe the selected plan");
    compiled = fixture.compiled; ++compiled.kernel_plan_identity;
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(compiled, fixture.launches, fixture.variants); }),
           "changed kernel plan identity rejects");
    strix::kernels::KernelRegistry wrong_phase;
    for (auto variant : fixture.variants.variants()) {
        variant.phase = strix::kernels::ExecutionPhase::prefill;
        wrong_phase.add(variant);
    }
    expect(rejects([&] { strix::decode_receipt::execution_inventory_receipt(fixture.compiled, fixture.launches, wrong_phase); }),
           "a decode receipt cannot use a prefill variant declaration");
    expect(calls == 0, "negative checks never call a launcher");
    return failures ? 1 : 0;
}
