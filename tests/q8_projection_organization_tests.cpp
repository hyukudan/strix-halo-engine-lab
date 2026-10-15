#include "strix/q8_projection_organization.hpp"
#include "strix/q8_projection_manifest.hpp"
#include "strix/compiled_execution.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <set>
#include <string>
#include <utility>

namespace q8 = strix::q8_projection_organization;
namespace {
int failures = 0;
void expect(bool value, const char* message) {
    if (!value) { std::cerr << message << '\n'; ++failures; }
}
template<class F> void rejects(F&& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
}

strix::kernels::KernelRegistry registry_for(const strix::ModelContract& model,
                                           const strix::ExecutionBindingRequest& request) {
    strix::kernels::KernelRegistry registry;
    std::set<std::string> added;
    for (const auto& step : strix::build_serial_decode_execution_plan(model).steps) {
        const auto family = strix::operator_family_for(step.op);
        if (!family) continue;
        const auto shape = strix::execution_shape_for(model, step.op, request);
        const auto key = std::to_string(static_cast<unsigned>(*family)) + "." +
            std::to_string(shape.auxiliary_size);
        if (!added.insert(key).second) continue;
        strix::kernels::KernelVariant variant;
        variant.id = "test.q8-organization." + key;
        if (step.op == strix::SerialOp::gated_delta_net && request.wave8_q8_gdn_input)
            variant.id += q8::gdn_wave8_marker;
        if (step.op == strix::SerialOp::lm_head && request.prefetch_q8_lm_head)
            variant.id += request.wave8_q8_lm_head ? q8::lm_wave8_marker : q8::lm_prefetch_marker;
        variant.family = *family;
        variant.phase = strix::kernels::ExecutionPhase::decode;
        variant.shape = shape;
        variant.wave = strix::kernels::WaveSize::wave32;
        variant.resources.workgroup_threads = 256;
        variant.numerics.input_precision = "bf16";
        variant.numerics.accumulator_precision = "fp32";
        variant.numerics.output_precision = step.op == strix::SerialOp::lm_head ? "fp32" : "bf16";
        variant.numerics.row_invariant = true;
        variant.numerics.deterministic = true;
        variant.qualification = strix::kernels::QualificationStatus::oracle_matched;
        variant.graph_safe = false;
        variant.input_layout = "row-major";
        variant.output_layout = "row-major";
        variant.compiler_identity = "synthetic-q8-organization-contract";
        registry.add(std::move(variant));
    }
    return registry;
}

void test_compiled_admission() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::ExecutionBindingRequest baseline_request;
    strix::ExecutionBufferRequest buffers;
    buffers.activation_element_bytes = 2;
    const auto baseline_registry = registry_for(model, baseline_request);
    const auto baseline = strix::compile_serial_execution(model, baseline_registry, baseline_request, buffers);
    std::set<std::uint64_t> execution_ids{baseline.execution_identity}, kernel_ids{baseline.kernel_plan_identity};
    for (unsigned mode = 0; mode < 4; ++mode) {
        auto request = baseline_request;
        request.wave8_q8_gdn_input = mode == 0 || mode == 3;
        request.prefetch_q8_lm_head = mode != 0;
        request.wave8_q8_lm_head = mode >= 2;
        const auto registry = registry_for(model, request);
        const auto compiled = strix::compile_serial_execution(model, registry, request, buffers);
        expect(execution_ids.insert(compiled.execution_identity).second &&
            kernel_ids.insert(compiled.kernel_plan_identity).second,
            "each selected organization changes compiled and kernel identity");
        expect(compiled.arena.total_bytes == baseline.arena.total_bytes,
            "Q8 load/geometry overlays do not change arena requirements");
        rejects([&] { (void)strix::compile_serial_execution(model, baseline_registry, request, buffers); },
            "opt-in request rejects incumbent selected ids");
        rejects([&] { (void)strix::compile_serial_execution(model, registry, baseline_request, buffers); },
            "default request rejects candidate selected ids");
        auto graph = request; graph.require_graph_safe = true;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, graph, buffers); },
            "new organizations reject graph-safe admission");
        graph = request; graph.device_graph_inputs = true;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, graph, buffers); },
            "new organizations reject device graph inputs");
        auto parallel = request; parallel.request_concurrency = 2;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, parallel, buffers); },
            "new organizations reject C2");
        auto rows = request; rows.logical_rows = 2;
        auto row_buffers = buffers; row_buffers.logical_rows = 2;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, rows, row_buffers); },
            "new organizations reject R2");
    }
    auto missing_prefetch = baseline_request; missing_prefetch.wave8_q8_lm_head = true;
    rejects([&] { (void)strix::compile_serial_execution(model, baseline_registry, missing_prefetch, buffers); },
        "LM wave8 cannot omit the prefetch dependency");
}

void test_whole_model_alignment_admission() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::ModelManifest manifest;
    manifest.architecture = "qwen3next";
    const auto append = [&](std::string name, std::uint64_t rows, strix::TensorRole role,
                            std::optional<std::uint32_t> layer) {
        strix::ManifestTensor tensor;
        tensor.name = std::move(name); tensor.role = role; tensor.layer = layer;
        tensor.ggml_type = 8; tensor.dimensions = {model.hidden_size, rows};
        tensor.byte_size = (model.hidden_size / 32U) * 34U * rows;
        tensor.absolute_offset = 4096U * (manifest.tensors.size() + 1U);
        manifest.tensors.push_back(std::move(tensor));
    };
    const auto key_width = model.gdn_key_heads * model.gdn_key_head_dim;
    const auto value_width = model.gdn_value_heads * model.gdn_value_head_dim;
    for (std::uint32_t layer = 0; layer < model.hidden_layers; ++layer) {
        if ((layer + 1U) % model.full_attention_interval == 0U) continue;
        const auto prefix = "blk." + std::to_string(layer);
        append(prefix + ".attn_qkv.weight", key_width * 2U + value_width, strix::TensorRole::attention_projection, layer);
        append(prefix + ".attn_gate.weight", value_width, strix::TensorRole::attention_projection, layer);
        append(prefix + ".ssm_beta.weight", model.gdn_value_heads, strix::TensorRole::recurrent_state, layer);
        append(prefix + ".ssm_alpha.weight", model.gdn_value_heads, strix::TensorRole::recurrent_state, layer);
    }
    append("output.weight", model.vocab_size, strix::TensorRole::output_head, std::nullopt);
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
    const auto plan = q8::build_manifest_plan(model, manifest, true, true);
    expect(plan.weights.size() == manifest.tensors.size(), "whole-model plan covers every selected Q8 weight");
    strix::ModelRuntimeBinding binding;
    binding.manifest_layout_identity = manifest.layout_identity;
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i)
        binding.tensors.push_back({static_cast<std::uint32_t>(i), strix::RuntimeTensorStorage::resident,
            0x100000U + i * 4096U, manifest.tensors[i].byte_size, 7, 0});
    q8::validate_resident_weights(plan, binding);
    // A bad last-layer tensor must reject before any model dispatch can start.
    for (std::size_t distance = 1; distance <= 5; ++distance) {
        const auto index = manifest.tensors.size() - distance;
        auto bad = manifest;
        bad.tensors[index].absolute_offset += 8U;
        rejects([&] { (void)q8::build_manifest_plan(model, bad, true, true); },
                "late-layer qkv/z/beta/alpha or head file alignment rejects in whole preflight");
        auto resident = binding;
        resident.tensors[index].address += 8U;
        rejects([&] { q8::validate_resident_weights(plan, resident); },
                "actual late device address rejects before token execution");
        resident = binding; --resident.tensors[index].bytes;
        rejects([&] { q8::validate_resident_weights(plan, resident); }, "short late resident view rejects");
        resident = binding; resident.tensors[index].allocation_generation = 0;
        rejects([&] { q8::validate_resident_weights(plan, resident); }, "missing real owner generation rejects");
    }
    auto duplicate = manifest; duplicate.tensors.push_back(manifest.tensors.front());
    rejects([&] { (void)q8::build_manifest_plan(model, duplicate, false, true); }, "duplicate selected GDN projection rejects");
    // Launcher family matching uses substrings, including noncanonical suffixes.
    // Ambiguity in the last executed GDN layer must fail before state setup.
    for (std::size_t family = 0; family < 4; ++family) {
        auto ambiguous = manifest;
        auto extra = manifest.tensors[manifest.tensors.size() - 5U + family];
        extra.name = family == 0U ? "last.attn_qkv.bias" :
                     family == 1U ? "last.attn_gate_extra.weight" :
                     family == 2U ? "last.ssm_beta_extra.weight" : "last.ssm_alpha_x.weight";
        ambiguous.tensors.push_back(std::move(extra));
        rejects([&] { (void)q8::build_manifest_plan(model, ambiguous, false, true); },
                "late noncanonical duplicate matches launcher and rejects in preflight");
    }
    auto missing = manifest; missing.tensors.erase(missing.tensors.begin());
    rejects([&] { (void)q8::build_manifest_plan(model, missing, false, true); }, "missing selected GDN projection rejects");
    auto invalid = manifest; invalid.tensors.back().ggml_type = 20;
    rejects([&] { (void)q8::build_manifest_plan(model, invalid, true, false); }, "non-Q8 LM head rejects before dispatch");
    auto unaligned_heads = model; unaligned_heads.gdn_key_heads = 1; unaligned_heads.gdn_value_heads = 3;
    rejects([&] { (void)q8::build_manifest_plan(unaligned_heads, manifest, false, true); },
            "unaligned beta/gate workspace slices reject at setup");
    // Default paths do not acquire these extra restrictions or any residency work.
    expect(q8::build_manifest_plan(model, invalid, false, false).weights.empty(), "inactive overlay leaves default admission unchanged");
    q8::validate_resident_weights({}, {});
}
}
int main() {
    test_compiled_admission();
    test_whole_model_alignment_admission();
    // Incumbent modes remain outside the new bounded BF16-only overlay.
    q8::validate_modes(false, false, false, false, 0, 0);
    q8::validate_modes(true, true, false, true, 2560, 248320);
    q8::validate_modes(true, true, true, true, 2560, 10240);
    rejects([] { q8::validate_modes(true, false, true, true, 2560, 19); }, "wave8 needs prefetch");
    rejects([] { q8::validate_modes(false, true, false, true, 2560, 19); }, "prefetch needs virtual organization");
    rejects([] { q8::validate_modes(true, true, false, false, 2560, 19); }, "F32 is outside admitted scope");
    for (const auto width : {0U, 31U, 33U, 10272U})
        rejects([&] { q8::validate_geometry(width, 19); }, "invalid width must reject");
    for (const auto outputs : {0U, 248321U})
        rejects([&] { q8::validate_geometry(32, outputs); }, "invalid output count must reject");
    const q8::View input{0x1000, 64}, weights{0x2000, 34 * 19}, output{0x4000, 4 * 19};
    q8::validate_views(32, 19, input, weights, output);
    q8::validate_views(32, 19, {input.address, 128}, {weights.address, 1024}, {output.address, 128});
    rejects([&] { q8::validate_views(32, 19, input, {weights.address, weights.bytes - 1}, output); }, "short packed weights reject");
    rejects([&] { q8::validate_views(32, 19, {input.address + 2, input.bytes}, weights, output); }, "unaligned input rejects");
    rejects([&] { q8::validate_views(32, 19, input, weights, {input.address, output.bytes}); }, "input/output alias rejects");
    rejects([&] { q8::validate_views(32, 19, input, weights, {weights.address + 16, output.bytes}); }, "weight/output alias rejects");
    rejects([&] { q8::validate_views(32, 19, input, weights,
        {std::numeric_limits<std::uintptr_t>::max() - 15U, 76}); }, "overflow rejects before interval addition");
    return failures ? 1 : 0;
}
