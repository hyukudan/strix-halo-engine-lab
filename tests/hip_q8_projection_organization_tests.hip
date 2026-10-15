#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_lm_head_launcher.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; ++failures; }
}
template<class F> void rejects(F&& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
}

strix::CompiledSerialExecution minimal(strix::SerialOp op, const std::string& id) {
    strix::CompiledSerialExecution compiled;
    compiled.execution_identity = 0x514f5247414e495aULL;
    compiled.logical_rows = 1; compiled.request_concurrency = 1;
    strix::ExecutionStep step; step.id = 0; step.op = op;
    if (op == strix::SerialOp::gated_delta_net) step.layer = 0;
    compiled.execution.steps.push_back(step);
    strix::ExecutionKernelBinding binding;
    binding.step_id = 0; binding.op = op; binding.kernel_id = id;
    compiled.bindings.bindings.push_back(binding);
    return compiled;
}

void test_lm() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    std::array<strix::HipLmHeadLauncherConfig, 4> configs;
    std::array<strix::kernels::KernelVariant, 4> variants;
    std::array<std::uint64_t, 4> hashes{};
    for (unsigned mode = 0; mode < 4; ++mode) {
        auto& config = configs[mode];
        config.hidden_size = static_cast<std::uint32_t>(model.hidden_size);
        config.vocab_size = static_cast<std::uint32_t>(model.vocab_size);
        config.virtual_q8_wave = mode >= 1;
        config.prefetch_q8_wave = mode >= 2;
        config.wave8_q8_wave = mode == 3;
        strix::ExecutionBindingRequest request;
        request.prefetch_q8_lm_head = config.prefetch_q8_wave;
        request.wave8_q8_lm_head = config.wave8_q8_wave;
        strix::kernels::KernelRegistry registry;
        strix::register_hip_lm_head_kernel_variant(model, request, config, registry);
        variants[mode] = registry.variants().front();
        hashes[mode] = strix::make_hip_lm_head_launcher(variants[mode].id, &config).implementation_identity;
        expect(variants[mode].resources.workgroup_threads == (mode == 1 || mode == 2 ? 32U : 256U), "LM declares actual threads");
        expect(variants[mode].resources.lds_bytes == (mode ? 0U : 1024U), "LM declares actual LDS");
        expect(variants[mode].graph_safe == (mode < 2), "only incumbent LM variants claim graph safety");
        expect(variants[mode].numerics.input_precision == "bf16" &&
            variants[mode].numerics.accumulator_precision == "fp32" &&
            variants[mode].numerics.output_precision == "fp32", "LM numerical declarations stay unchanged");
        if (mode >= 2) {
            auto mismatch = request; mismatch.prefetch_q8_lm_head = false;
            rejects([&] { strix::kernels::KernelRegistry bad; strix::register_hip_lm_head_kernel_variant(model, mismatch, config, bad); }, "LM request mismatch rejects");
            for (const bool device : {false, true}) {
                auto graph = request; graph.require_graph_safe = !device; graph.device_graph_inputs = device;
                rejects([&] { strix::kernels::KernelRegistry bad; strix::register_hip_lm_head_kernel_variant(model, graph, config, bad); }, "LM graph request rejects");
            }
            auto f32 = config; f32.input = strix::HipActivationStorage::fp32;
            rejects([&] { (void)strix::make_hip_lm_head_launcher(variants[mode].id, &f32); }, "LM F32 factory rejects");
            rejects([&] { (void)strix::make_hip_lm_head_launcher("test.wrong", &config); }, "LM exact factory id required");
        }
    }
    for (unsigned mode = 0; mode < 4; ++mode) for (unsigned selected = 0; selected < 4; ++selected) {
        auto compiled = minimal(strix::SerialOp::lm_head, variants[selected].id);
        strix::HipLmHeadLauncherBundle bundle(configs[mode]);
        strix::ExecutionLauncherRegistry registry;
        bool accepted = true;
        try { bundle.register_for_compiled_execution(compiled, registry); }
        catch (const std::invalid_argument&) { accepted = false; }
        expect(accepted == (mode == selected), "LM selected-id/config 4x4 matrix is exact");
        if (mode != selected) expect(hashes[mode] != hashes[selected] && variants[mode].id != variants[selected].id &&
            variants[mode].compiler_identity != variants[selected].compiler_identity, "LM modes have distinct identities");
    }
}

void test_gdn() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    std::array<strix::HipGdnLauncherConfig, 4> configs;
    std::array<strix::kernels::KernelVariant, 4> variants;
    std::array<std::uint64_t, 4> hashes{};
    for (unsigned mode = 0; mode < 4; ++mode) {
        auto& config = configs[mode]; config.model = model;
        config.virtual_q8_input_projections = mode >= 1;
        config.prefetch_q8_input_projections = mode >= 2;
        config.wave8_q8_input_projections = mode == 3;
        strix::ExecutionBindingRequest request; request.wave8_q8_gdn_input = mode == 3;
        strix::kernels::KernelRegistry registry;
        strix::register_hip_gdn_kernel_variant(model, request, config, registry);
        variants[mode] = registry.variants().front();
        hashes[mode] = strix::make_hip_gdn_launcher(variants[mode].id, &config).implementation_identity;
        expect(variants[mode].resources.workgroup_threads == 256U && variants[mode].resources.lds_bytes == 16640U &&
            variants[mode].resources.scratch_bytes == variants[0].resources.scratch_bytes,
            "GDN preserves the whole compound recurrence resource bound");
        expect(!variants[mode].graph_safe && variants[mode].numerics.output_precision == "bf16", "GDN eager metadata stays scoped");
        if (mode == 3) {
            auto mismatch = request; mismatch.wave8_q8_gdn_input = false;
            rejects([&] { strix::kernels::KernelRegistry bad; strix::register_hip_gdn_kernel_variant(model, mismatch, config, bad); }, "GDN request mismatch rejects");
            for (const bool device : {false, true}) {
                auto graph = request; graph.require_graph_safe = !device; graph.device_graph_inputs = device;
                rejects([&] { strix::kernels::KernelRegistry bad; strix::register_hip_gdn_kernel_variant(model, graph, config, bad); }, "GDN graph request rejects");
            }
            auto invalid = config; invalid.prefetch_q8_input_projections = false;
            rejects([&] { (void)strix::make_hip_gdn_launcher(variants[mode].id, &invalid); }, "GDN wave8 needs prefetch");
            rejects([&] { (void)strix::make_hip_gdn_launcher("test.wrong", &config); }, "GDN exact factory id required");
        }
    }
    for (unsigned mode = 0; mode < 4; ++mode) for (unsigned selected = 0; selected < 4; ++selected) {
        auto compiled = minimal(strix::SerialOp::gated_delta_net, variants[selected].id);
        strix::HipGdnLauncherBundle bundle(configs[mode]); strix::ExecutionLauncherRegistry registry;
        bool accepted = true;
        try { bundle.register_for_compiled_execution(compiled, registry); }
        catch (const std::invalid_argument&) { accepted = false; }
        expect(accepted == (mode == selected), "GDN selected-id/config 4x4 matrix is exact");
        if (mode != selected) expect(hashes[mode] != hashes[selected] && variants[mode].id != variants[selected].id &&
            variants[mode].compiler_identity != variants[selected].compiler_identity, "GDN modes have distinct identities");
    }
}
}
int main() try { test_lm(); test_gdn(); return failures ? 1 : 0; }
catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
