#include "strix/compiled_execution.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/kernel_registry.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include "strix/model_contract.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::HipHyperMixLauncherConfig geometry() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::HipHyperMixLauncherConfig config;
    config.hidden_size = static_cast<std::uint32_t>(model.hidden_size);
    config.hc_count = static_cast<std::uint32_t>(model.hc_count);
    config.lowrank = static_cast<std::uint32_t>(model.hc_lowrank);
    config.rms_epsilon = model.rms_norm_epsilon;
    return config;
}

strix::HipHyperMixLauncherConfig admitted() {
    auto config = geometry();
    config.parallel = true;
    config.staged_projections = true;
    config.fused_injection = true;
    config.typed_projection = true;
    config.prefetch_rows = true;
    config.prefetch_q8_output = true;
    config.splitk_projection = true;
    return config;
}

strix::ExecutionBindingRequest request_for(bool lane_chain) {
    strix::ExecutionBindingRequest request;
    request.minimum_qualification = strix::kernels::QualificationStatus::experimental;
    request.lane_chain_hyper_output = lane_chain;
    return request;
}

strix::kernels::KernelVariant descriptor(const strix::HipHyperMixLauncherConfig& config) {
    strix::kernels::KernelRegistry registry;
    strix::register_hip_hyper_mix_kernel_variant(
        strix::ModelContract::qwen38_flash_next(), request_for(config.lane_chain_output), config, registry);
    expect(registry.variants().size() == 1, "direct registration creates one descriptor");
    return registry.variants().front();
}

void rejects_config(strix::HipHyperMixLauncherConfig config, std::string_view message) {
    bool rejected = false;
    try {
        static_cast<void>(strix::make_hip_hyper_mix_launcher("invalid.fixture", &config));
    } catch (const std::invalid_argument& error) {
        rejected = std::string_view(error.what()).find(message) != std::string_view::npos;
        if (!rejected) std::cerr << "Unexpected rejection: " << error.what() << '\n';
    }
    expect(rejected, message);
    strix::kernels::KernelRegistry registry;
    try {
        strix::register_hip_hyper_mix_kernel_variant(
            strix::ModelContract::qwen38_flash_next(), request_for(config.lane_chain_output), config, registry);
    } catch (const std::invalid_argument&) {
        // The real config validator must run before adding any descriptor.
    }
    expect(registry.variants().empty(), "invalid config leaves kernel registry empty");
}

void config_and_identity_tests() {
    auto legacy = geometry();
    expect(!legacy.splitk_projection && !legacy.splitk_projection_fp64,
           "both new modes default off");
    const auto old = descriptor(legacy);
    expect(old.id == "hip.hyper_mix.bf16.h2560.hc4.lr320.r1.c1",
           "default legacy kernel ID is unchanged");
    expect(strix::make_hip_hyper_mix_launcher(old.id, &legacy).implementation_identity ==
               11360272515784746702ULL,
           "default legacy implementation hash is unchanged");
    expect(old.resources.workgroup_threads == 256 && old.numerics.accumulator_precision == "fp32",
           "default descriptor retains legacy launch geometry and numerics");

    auto prior_c1 = admitted();
    prior_c1.splitk_projection = false;
    prior_c1.lane_chain_output = true;
    const auto prior_descriptor = descriptor(prior_c1);
    expect(prior_descriptor.id ==
               "hip.hyper_mix.bf16.h2560.hc4.lr320.r1.c1.parallel.staged.fused-injection-v5.typed-q8-f32-v1.prefetch-rows8-v2.prefetch-q8-output-v1.lane-chain-v1",
           "both modes off preserve the qualified C1 kernel ID");
    expect(strix::make_hip_hyper_mix_launcher(prior_descriptor.id, &prior_c1).implementation_identity ==
               14951791474786778603ULL,
           "both modes off preserve the qualified C1 implementation hash");

    auto fp32 = admitted();
    auto fp64 = fp32;
    fp64.splitk_projection_fp64 = true;
    const auto variant32 = descriptor(fp32);
    const auto variant64 = descriptor(fp64);
    expect(variant32.id.ends_with(".splitk-hc-proj-fp32-v1") &&
               variant64.id.ends_with(".splitk-hc-proj-fp64-oracle-v1") && variant32.id != variant64.id,
           "FP32 and FP64 diagnostic descriptors have distinct explicit IDs");
    const auto launcher32 = strix::make_hip_hyper_mix_launcher(variant32.id, &fp32);
    const auto launcher64 = strix::make_hip_hyper_mix_launcher(variant64.id, &fp64);
    expect(launcher32.implementation_identity != launcher64.implementation_identity &&
               launcher32.implementation_identity != 14951791474786778603ULL &&
               launcher64.implementation_identity != 14951791474786778603ULL,
           "FP32, FP64 and prior implementation identities are distinct");
    expect(variant32.resources.workgroup_threads == 512 && variant64.resources.workgroup_threads == 512 &&
               variant32.numerics.accumulator_precision == "fp32" &&
               variant64.numerics.accumulator_precision == "fp64-projection/fp32-epilogue" &&
               variant32.compiler_identity != variant64.compiler_identity,
           "descriptors declare 512 threads and distinguish the FP64 projection epilogue");
    expect(variant32.qualification == strix::kernels::QualificationStatus::experimental &&
               variant64.qualification == strix::kernels::QualificationStatus::experimental,
           "new numerical modes are experimental");

    fp64.splitk_projection = false;
    rejects_config(fp64, "FP64 HC diagnostic requires split-K projection");
    for (unsigned missing = 0; missing < 5; ++missing) {
        auto bad = admitted();
        if (missing == 0) bad.parallel = false;
        if (missing == 1) bad.staged_projections = false;
        if (missing == 2) bad.fused_injection = false;
        if (missing == 3) bad.typed_projection = false;
        if (missing == 4) bad.prefetch_rows = false;
        rejects_config(bad, "split-K HC projection requires");
    }
    auto bad_width = admitted();
    --bad_width.hidden_size;
    rejects_config(bad_width, "split-K HC projection requires");

    for (bool oracle : {false, true}) {
        auto config = admitted();
        config.splitk_projection_fp64 = oracle;
        config.lane_chain_output = true;
        const auto with_c1 = descriptor(config);
        const auto splitk_suffix = oracle ? ".splitk-hc-proj-fp64-oracle-v1" : ".splitk-hc-proj-fp32-v1";
        expect(with_c1.id.ends_with(strix::lane_chain_hyper_output_suffix) &&
                   with_c1.id.find(splitk_suffix) < with_c1.id.size() - std::string_view(strix::lane_chain_hyper_output_suffix).size(),
               "C1 suffix remains last after either projection identity");
        auto wrong_request = request_for(false);
        strix::kernels::KernelRegistry untouched;
        bool rejected = false;
        try {
            strix::register_hip_hyper_mix_kernel_variant(
                strix::ModelContract::qwen38_flash_next(), wrong_request, config, untouched);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && untouched.variants().empty(), "C1 still requires matching binding opt-in");
    }
}

void complete_catalog_tests() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    const auto execution = strix::build_serial_decode_execution_plan(model);
    auto request = request_for(true);
    for (const auto& step : execution.steps) {
        if (step.op == strix::SerialOp::attn_hyper_mix || step.op == strix::SerialOp::mlp_hyper_mix)
            request.lane_chain_hyper_output_eligible_steps.push_back(step.id);
    }
    for (bool oracle : {false, true}) {
        strix::HipLauncherCatalogConfig config;
        config.parallel_hyper_mix = true;
        config.staged_hyper_projections = true;
        config.fused_staged_hyper_mix = true;
        config.typed_hyper_projection = true;
        config.prefetch_typed_hyper_rows = true;
        config.prefetch_q8_hyper_output = true;
        config.lane_chain_hyper_output = true;
        config.splitk_hyper_projection = true;
        config.splitk_hyper_projection_fp64 = oracle;
        strix::kernels::KernelRegistry registry;
        strix::register_complete_hip_kernel_catalog(model, request, config, registry);
        std::set<std::string> hyper_ids;
        unsigned candidate = 0, fallback = 0;
        for (const auto& variant : registry.variants()) {
            if (variant.family != strix::kernels::OperatorFamily::hyper_mix) continue;
            hyper_ids.insert(variant.id);
            if (variant.id.ends_with(strix::lane_chain_hyper_output_suffix)) ++candidate;
            else ++fallback;
            expect(variant.id.find(oracle ? ".splitk-hc-proj-fp64-oracle-v1" : ".splitk-hc-proj-fp32-v1") != std::string::npos,
                   "catalog forwards the projection identity to both C1 and fallback");
        }
        expect(candidate == 1 && fallback == 1 && hyper_ids.size() == 2,
               "catalog retains exact C1/fallback descriptor pair");
        const auto compiled = strix::compile_serial_execution(model, registry, request, {});
        unsigned mixers = 0;
        for (const auto& binding : compiled.bindings.bindings) {
            if (binding.op != strix::SerialOp::attn_hyper_mix && binding.op != strix::SerialOp::mlp_hyper_mix &&
                binding.op != strix::SerialOp::final_hyper_mix) continue;
            ++mixers;
            expect(binding.kernel_id.has_value() && hyper_ids.contains(*binding.kernel_id),
                   "all mixer bindings reference registered exact IDs");
        }
        expect(mixers == 97 && compiled.bindings.missing_steps == 0,
               "complete plan binds all 97 mixers for either numerical mode");
    }
}

// Metadata owners deliberately have no backing allocation. These tests invoke
// admission/registration only, never a launch function. The positive control
// performs the final device-property query on the native test machine.
struct AdmissionFixture {
    strix::HipHyperMixLauncherConfig config = admitted();
    strix::CompiledSerialExecution compiled;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionTensorAccessPlan access;
    strix::ExecutionMemoryBinding scratch;
    strix::SerialSessionStateLayout state_layout;
    strix::SessionStateMemoryBinding state_memory;
    std::vector<std::uint32_t> mixer_steps;

    AdmissionFixture() {
        const auto model = strix::ModelContract::qwen38_flash_next();
        strix::HipLauncherCatalogConfig catalog;
        catalog.parallel_hyper_mix = true;
        catalog.staged_hyper_projections = true;
        catalog.fused_staged_hyper_mix = true;
        catalog.typed_hyper_projection = true;
        catalog.prefetch_typed_hyper_rows = true;
        catalog.prefetch_q8_hyper_output = true;
        catalog.splitk_hyper_projection = true;
        strix::kernels::KernelRegistry registry;
        const auto request = request_for(false);
        strix::register_complete_hip_kernel_catalog(model, request, catalog, registry);
        compiled = strix::compile_serial_execution(model, registry, request, {});
        for (std::uint32_t index = 0; index < 5; ++index) {
            const auto role = index == 4 ? 1U : index;
            strix::ManifestTensor tensor;
            tensor.name = "splitk.fixture." + std::to_string(index);
            tensor.ggml_type = role == 0 || role == 3 ? 0U : 8U;
            tensor.dimensions = role == 0 ? std::vector<std::uint64_t>{10240} :
                role == 1 ? std::vector<std::uint64_t>{10240, 320} :
                role == 2 ? std::vector<std::uint64_t>{320, 10240} :
                            std::vector<std::uint64_t>{10240, 4};
            tensor.byte_size = role == 0 ? 40960U : role == 3 ? 163840U : 3481600U;
            manifest.tensors.push_back(tensor);
            weights.tensors.push_back({index, strix::RuntimeTensorStorage::resident,
                static_cast<std::uintptr_t>(0x10000000ULL * (index + 1ULL)), tensor.byte_size, 1, 0});
        }
        refresh_manifest();
        weights.model_content_identity.bytes[0] = 1;
        access.plan_identity = 0x53504c49544bULL;
        for (const auto& step : compiled.execution.steps) {
            if (step.op != strix::SerialOp::attn_hyper_mix && step.op != strix::SerialOp::mlp_hyper_mix &&
                step.op != strix::SerialOp::final_hyper_mix) continue;
            mixer_steps.push_back(step.id);
            const bool final = step.op == strix::SerialOp::final_hyper_mix;
            for (std::uint32_t slot = 0; slot < (final ? 3U : 4U); ++slot)
                access.accesses.push_back({step.id, step.op, slot, {final && slot == 1 ? 4U : slot}});
        }
        expect(mixer_steps.size() == 97, "admission fixture has 96 mixers and the final 97th");
        scratch = {compiled.execution_identity, static_cast<std::uintptr_t>(0x1000000000ULL),
                   compiled.arena.total_bytes, 1};
        strix::SerialSessionStateLayoutRequest state_request;
        state_request.context_tokens = 8192;
        state_layout = strix::build_serial_session_state_layout(model, state_request);
        state_memory = {state_layout.layout_identity, static_cast<std::uintptr_t>(0x2000000000ULL),
                        state_layout.device_total_bytes, 1, static_cast<std::uintptr_t>(0x3000000000ULL),
                        state_layout.host_total_bytes, 1};
    }

    void refresh_manifest() {
        manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
        weights.manifest_layout_identity = manifest.layout_identity;
        access.manifest_layout_identity = manifest.layout_identity;
    }

    strix::ExecutionRuntimeResources resources() const {
        strix::ExecutionRuntimeResources result;
        result.scratch = &scratch;
        result.model_manifest = &manifest;
        result.model_weights = &weights;
        result.tensor_access = &access;
        result.session_state_layout = &state_layout;
        result.session_state_memory = &state_memory;
        return result;
    }

    strix::ExecutionArenaAllocation& allocation(strix::ExecutionBufferUse use) {
        const auto found = std::find_if(compiled.arena.allocations.begin(), compiled.arena.allocations.end(),
            [&](const auto& item) { return item.use == use; });
        if (found == compiled.arena.allocations.end()) throw std::logic_error("missing fixture arena owner");
        return *found;
    }

    void rejects_before_registration(std::string_view label) {
        const auto runtime = resources();
        strix::ExecutionLauncherRegistry registry;
        strix::HipHyperMixLauncherBundle bundle(config);
        bool rejected = false;
        try { bundle.register_for_compiled_execution(compiled, registry, &runtime); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, label);
        for (const auto& binding : compiled.bindings.bindings)
            if (binding.kernel_id) expect(!registry.find_kernel_launcher(*binding.kernel_id),
                "whole-model admission failure registers no earlier launcher");
    }
};

void whole_model_admission_tests() {
    {
        AdmissionFixture positive;
        const auto runtime = positive.resources();
        strix::preflight_splitk_hyper_projection_model(positive.compiled, runtime, positive.config);
        strix::ExecutionLauncherRegistry registry;
        strix::HipHyperMixLauncherBundle bundle(positive.config);
        bundle.register_for_compiled_execution(positive.compiled, registry, &runtime);
        unsigned mixers = 0;
        for (const auto& binding : positive.compiled.bindings.bindings) {
            if (binding.op != strix::SerialOp::attn_hyper_mix && binding.op != strix::SerialOp::mlp_hyper_mix &&
                binding.op != strix::SerialOp::final_hyper_mix) continue;
            expect(binding.kernel_id && registry.find_kernel_launcher(*binding.kernel_id),
                   "positive metadata control admits every mixer launcher");
            ++mixers;
        }
        expect(mixers == 97, "positive control reaches all 97 mixer bindings");
    }
    {
        AdmissionFixture late;
        --late.manifest.tensors[4].dimensions[0];
        late.refresh_manifest();
        late.rejects_before_registration("wrong-width down tensor at the 97th mixer rejects atomically");
    }
    {
        AdmissionFixture late;
        late.weights.tensors[4].address += 2;
        late.rejects_before_registration("misaligned down owner at the 97th mixer rejects atomically");
    }
    for (unsigned mutation = 0; mutation < 9; ++mutation) {
        AdmissionFixture bad;
        const auto& io = bad.compiled.io.steps[bad.mixer_steps.front()];
        const auto temporary_use = io.temporaries.front();
        const auto injection_use = io.writes[1];
        const auto temporary_offset = bad.allocation(temporary_use).offset;
        if (mutation == 0) {
            auto& temporary = bad.allocation(temporary_use);
            temporary.offset += 4;
            temporary.bytes -= 4;
        }
        if (mutation == 1) bad.weights.tensors[1].address += 2;
        if (mutation == 2) bad.weights.tensors[3].address += 4;
        if (mutation == 3) bad.allocation(temporary_use).bytes = (10240U + 320U) * 4U - 1U;
        if (mutation == 4) {
            // Registered model bytes themselves are valid, but normalization
            // would overwrite the first input coefficient before projection.
            bad.weights.tensors[1].address = bad.scratch.arena_base + temporary_offset;
        }
        if (mutation == 5) bad.allocation(injection_use).offset = temporary_offset + 10240U * 4U;
        if (mutation == 6) bad.allocation(injection_use).bytes = 15;
        if (mutation == 7) bad.allocation(injection_use).offset = temporary_offset;
        if (mutation == 8) bad.allocation(injection_use).offset += 2;
        bad.rejects_before_registration("malformed normalized/down/inject/output owner rejects before registration");
    }
}
} // namespace

int main() {
    try {
        config_and_identity_tests();
        complete_catalog_tests();
        whole_model_admission_tests();
    } catch (const std::exception& error) {
        std::cerr << "Unexpected exception: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
