#include "strix/q8_projection_organization.hpp"
#include "strix/q8_projection_manifest.hpp"
#include "strix/compiled_execution.hpp"

#include <cstdint>
#include <algorithm>
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
        if (step.op == strix::SerialOp::gated_delta_net && request.wave8_q8_gdn_output)
            variant.id += q8::gdn_output_wave8_marker;
        if (step.op == strix::SerialOp::indexed_attention && request.wave8_q8_qsa_output)
            variant.id += q8::qsa_output_wave8_marker;
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
    for (unsigned mode = 1; mode < 4; ++mode) {
        auto request = baseline_request;
        request.wave8_q8_gdn_output = (mode & 1U) != 0U;
        request.wave8_q8_qsa_output = (mode & 2U) != 0U;
        const auto registry = registry_for(model, request);
        const auto compiled = strix::compile_serial_execution(model, registry, request, buffers);
        expect(execution_ids.insert(compiled.execution_identity).second &&
            kernel_ids.insert(compiled.kernel_plan_identity).second,
            "each output combination has distinct compiled arithmetic binding identities");
        expect(compiled.arena.total_bytes == baseline.arena.total_bytes,
            "output geometry overlays preserve incumbent scratch geometry");
        rejects([&] { (void)strix::compile_serial_execution(model, baseline_registry, request, buffers); },
            "output request cannot bind unmarked default IDs");
        rejects([&] { (void)strix::compile_serial_execution(model, registry, baseline_request, buffers); },
            "default request cannot bind marked output IDs");
        auto graph = request; graph.require_graph_safe = true;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, graph, buffers); },
            "output organization rejects graph-safe request");
        graph = request; graph.device_graph_inputs = true;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, graph, buffers); },
            "output organization rejects graph-device mode");
        auto concurrent = request; concurrent.request_concurrency = 2;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, concurrent, buffers); },
            "output organization rejects C2");
        auto rows = request; rows.logical_rows = 2;
        auto row_buffers = buffers; row_buffers.logical_rows = 2;
        rejects([&] { (void)strix::compile_serial_execution(model, registry, rows, row_buffers); },
            "output organization rejects R2");
    }
}

strix::ModelManifest output_manifest(const strix::ModelContract& model) {
    strix::ModelManifest manifest;
    manifest.architecture = "qwen3next";
    std::uint64_t offset = 4096;
    for (std::uint32_t layer = 0; layer < model.hidden_layers; ++layer) {
        const bool gdn = (layer + 1U) % model.full_attention_interval != 0U;
        const auto width = gdn ? model.gdn_value_heads * model.gdn_value_head_dim
                               : model.attention_heads * model.head_dim;
        strix::ManifestTensor tensor;
        tensor.name = "blk." + std::to_string(layer) + (gdn ? ".ssm_out.weight" : ".attn_output.weight");
        tensor.layer = layer;
        tensor.role = gdn ? strix::TensorRole::recurrent_state : strix::TensorRole::attention_projection;
        tensor.ggml_type = 8;
        tensor.dimensions = {width, model.hidden_size};
        tensor.byte_size = (width / 32U) * 34U * model.hidden_size;
        tensor.absolute_offset = offset;
        offset += ((tensor.byte_size + 255U) / 256U) * 256U;
        manifest.tensors.push_back(std::move(tensor));
    }
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
    return manifest;
}

strix::ModelRuntimeBinding output_residency(const strix::ModelManifest& manifest) {
    strix::ModelRuntimeBinding resident;
    resident.manifest_layout_identity = manifest.layout_identity;
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i)
        resident.tensors.push_back({static_cast<std::uint32_t>(i), strix::RuntimeTensorStorage::resident,
            static_cast<std::uintptr_t>(0x4000000000ULL + i * 0x2000000ULL),
            manifest.tensors[i].byte_size, 17, 0});
    return resident;
}

void test_output_manifest_and_residency() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    const auto manifest = output_manifest(model);
    const auto gdn = q8::build_manifest_plan(model, manifest, false, false, true, false);
    const auto qsa = q8::build_manifest_plan(model, manifest, false, false, false, true);
    const auto both = q8::build_manifest_plan(model, manifest, false, false, true, true);
    expect(gdn.outputs.size() == 36 && gdn.weights.size() == 36 &&
        qsa.outputs.size() == 12 && qsa.weights.size() == 12 &&
        both.outputs.size() == 48 && both.weights.size() == 48,
        "PhaseA covers all output-only families independently and together");
    for (const auto& output : both.outputs)
        expect(output.width == 6144 && output.rows == 2560 && output.bytes == 16711680,
            "output widths derive from consumers rather than hidden input width");
    q8::validate_resident_weights(both, output_residency(manifest));
    for (const auto index : {46U, 47U}) {
        const auto reject_manifest = [&](auto mutate, const char* text) {
            auto bad = manifest; mutate(bad.tensors[index]);
            bad.layout_identity = strix::model_manifest_layout_identity(bad);
            rejects([&] { (void)q8::build_manifest_plan(model, bad, false, false, true, true); }, text);
        };
        reject_manifest([](auto& t) { t.ggml_type = 0; }, "late nonQ8 output rejects whole opt-in");
        reject_manifest([](auto& t) { t.dimensions[0] = 2560; }, "hidden width cannot replace output consumer width");
        reject_manifest([](auto& t) { t.dimensions[1] = 2561; }, "wrong output row count rejects");
        reject_manifest([](auto& t) { t.absolute_offset += 8; }, "late output file misalignment rejects");
        reject_manifest([](auto& t) { t.absolute_offset = std::numeric_limits<std::uint64_t>::max() - 15U; },
            "late aligned output file end overflow rejects with valid dimensions and byte size");
        reject_manifest([](auto& t) { --t.byte_size; }, "late short output weight rejects");
        reject_manifest([](auto& t) { t.name = "blk.late.unrelated.weight"; }, "missing output family rejects");
        auto extra = manifest;
        auto duplicate = manifest.tensors[index];
        duplicate.name += ".noncanonical-extra";
        extra.tensors.push_back(duplicate);
        extra.layout_identity = strix::model_manifest_layout_identity(extra);
        rejects([&] { (void)q8::build_manifest_plan(model, extra, false, false, true, true); },
            "noncanonical duplicate output matching launcher substrings rejects");
        auto missing = manifest; missing.tensors.erase(missing.tensors.begin() + index);
        missing.layout_identity = strix::model_manifest_layout_identity(missing);
        rejects([&] { (void)q8::build_manifest_plan(model, missing, false, false, true, true); },
            "late missing output rejects before model execution");
    }
    auto aliases = manifest;
    aliases.tensors[46].name = "blk.46.ssm_out_noncanonical.weight";
    aliases.tensors[47].name = "blk.47.attn_output_noncanonical.weight";
    aliases.layout_identity = strix::model_manifest_layout_identity(aliases);
    expect(q8::build_manifest_plan(model, aliases, false, false, true, true).outputs.size() == 48,
        "single noncanonical output aliases follow actual launcher substring matching");
    for (const auto* name : {"blk.46.ssm_norm_out.weight", "blk.46.ssm_dt_out.weight",
                             "blk.46.ssm_norm.ssm_out.weight", "blk.46.ssm_dt.ssm_out.weight"}) {
        auto ignored = manifest;
        auto earlier = manifest.tensors[46];
        earlier.name = name; earlier.ggml_type = 0; earlier.dimensions = {1, 1}; earlier.byte_size = 4;
        ignored.tensors.push_back(std::move(earlier));
        ignored.layout_identity = strix::model_manifest_layout_identity(ignored);
        expect(q8::build_manifest_plan(model, ignored, false, false, true, true).outputs.size() == 48,
            "earlier GDN norm/dt classification wins over out substring");
    }
    auto wide_gdn = model; wide_gdn.gdn_value_heads = 96;
    expect(wide_gdn.validate().empty(), "wide GDN test retains a valid model contract");
    rejects([&] { (void)q8::build_manifest_plan(wide_gdn, output_manifest(wide_gdn), false, false, true, false); },
        "GDN output consumer width beyond10240 rejects");
    auto wide_qsa = model; wide_qsa.attention_heads = 48;
    expect(wide_qsa.validate().empty(), "wide QSA test retains a valid model contract");
    rejects([&] { (void)q8::build_manifest_plan(wide_qsa, output_manifest(wide_qsa), false, false, false, true); },
        "QSA output consumer width beyond10240 rejects");
    auto unaligned = model; unaligned.gdn_key_heads = 1; unaligned.gdn_value_heads = 3;
    expect(unaligned.validate().empty() && q8::gdn_output_core_offset_bytes(unaligned) % 16U != 0U,
        "misaligned core test isolates valid-model workspace alignment");
    rejects([&] { (void)q8::build_manifest_plan(unaligned, output_manifest(unaligned), false, false, true, false); },
        "misaligned FP32 GDN core source rejects");
    const auto resident = output_residency(manifest);
    for (const auto index : {46U, 47U}) {
        for (unsigned failure = 0; failure < 5; ++failure) {
            auto bad = resident;
            if (failure == 0) bad.tensors[index].address += 8;
            if (failure == 1) --bad.tensors[index].bytes;
            if (failure == 2) bad.tensors[index].allocation_generation = 0;
            if (failure == 3) bad.tensors[index].storage = strix::RuntimeTensorStorage::file_mapped;
            if (failure == 4) bad.tensors.erase(bad.tensors.begin() + index);
            rejects([&] { q8::validate_resident_weights(both, bad); },
                "late output resident alignment/extent/generation/storage/presence rejects");
        }
    }
    auto stale = resident; ++stale.manifest_layout_identity;
    rejects([&] { q8::validate_resident_weights(both, stale); }, "resident output manifest identity rejects");
    auto disabled = manifest; disabled.tensors.back().ggml_type = 0;
    expect(q8::build_manifest_plan(model, disabled, false, false, false, false).outputs.empty(),
        "disabled output overlays leave default manifest unrestricted");
}

void test_output_step_admission() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    const auto manifest = output_manifest(model);
    const auto resident = output_residency(manifest);
    const auto plan = q8::build_manifest_plan(model, manifest, false, false, true, true);
    strix::ExecutionBindingRequest request;
    request.wave8_q8_gdn_output = request.wave8_q8_qsa_output = true;
    strix::ExecutionBufferRequest buffers;
    const auto registry = registry_for(model, request);
    const auto compiled = strix::compile_serial_execution(model, registry, request, buffers);
    const strix::ExecutionMemoryBinding memory{compiled.execution_identity,
        static_cast<std::uintptr_t>(0x100000000ULL), compiled.arena.total_bytes + 256U, 9};
    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity = manifest.layout_identity; access.plan_identity = 19;
    for (const auto& step : compiled.execution.steps)
        for (const auto& weight : plan.outputs)
            if (step.op == weight.consumer && step.layer == weight.layer)
                access.accesses.push_back({step.id, step.op, 0, {weight.tensor_index}});
    expect(access.accesses.size() == 48, "PhaseB synthetic slot0 coverage is exactly36GDN+12QSA");
    q8::validate_output_steps(plan, model, compiled, access, memory, resident);
    const auto target_id = access.accesses.front().step_id;
    const auto target_weight = access.accesses.front().tensor_indices.front();
    const auto& target_io = compiled.io.steps[target_id];
    const auto corrupt_arena = [&](auto use, auto mutate) {
        auto copy = compiled;
        bool found = false;
        for (auto& allocation : copy.arena.allocations)
            if (allocation.use == use) { mutate(allocation); found = true; break; }
        expect(found, "corruption targets an actual compiled arena allocation");
        return copy;
    };
    auto bad_access = access; ++bad_access.manifest_layout_identity;
    rejects([&] { q8::validate_output_steps(plan, model, compiled, bad_access, memory, resident); },
        "PhaseB stale tensor access manifest rejects");
    bad_access = access; bad_access.plan_identity = 0;
    rejects([&] { q8::validate_output_steps(plan, model, compiled, bad_access, memory, resident); },
        "PhaseB missing access plan identity rejects");
    bad_access = access; bad_access.accesses.erase(bad_access.accesses.begin());
    rejects([&] { q8::validate_output_steps(plan, model, compiled, bad_access, memory, resident); },
        "PhaseB missing selected slot rejects");
    bad_access = access; bad_access.accesses.front().tensor_indices.push_back(target_weight);
    rejects([&] { q8::validate_output_steps(plan, model, compiled, bad_access, memory, resident); },
        "PhaseB duplicate output tensor index rejects");
    bad_access = access; bad_access.accesses.push_back(bad_access.accesses.front());
    rejects([&] { q8::validate_output_steps(plan, model, compiled, bad_access, memory, resident); },
        "PhaseB duplicate selected access rejects");
    auto bad_compiled = compiled; ++bad_compiled.io.steps[target_id].step_id;
    rejects([&] { q8::validate_output_steps(plan, model, bad_compiled, access, memory, resident); },
        "PhaseB IO step identity rejects");
    bad_compiled = compiled; bad_compiled.io.steps[target_id].op = strix::SerialOp::lm_head;
    rejects([&] { q8::validate_output_steps(plan, model, bad_compiled, access, memory, resident); },
        "PhaseB IO operator identity rejects");
    bad_compiled = compiled; bad_compiled.execution.steps[target_id].op = strix::SerialOp::token_embedding;
    rejects([&] { q8::validate_output_steps(plan, model, bad_compiled, access, memory, resident); },
        "PhaseB output weight without matching compiled step rejects");
    bad_compiled = corrupt_arena(target_io.temporaries.front(), [&](auto& a) {
        a.bytes = q8::gdn_output_core_offset_bytes(model) + 6144U * 4U - 1U;
    });
    rejects([&] { q8::validate_output_steps(plan, model, bad_compiled, access, memory, resident); },
        "PhaseB actual temporary source too short rejects");
    bad_compiled = corrupt_arena(target_io.writes.front(), [](auto& a) { a.offset += 2U; });
    rejects([&] { q8::validate_output_steps(plan, model, bad_compiled, access, memory, resident); },
        "PhaseB destination-only misalignment rejects");
    const auto destination = strix::resolve_execution_buffer(compiled.execution_identity,
        compiled.arena, memory, target_io.writes.front());
    auto bad_resident = resident; bad_resident.tensors[target_weight].address = destination.address;
    rejects([&] { q8::validate_output_steps(plan, model, compiled, access, memory, bad_resident); },
        "PhaseB writable destination aliasing actual weight rejects");
    auto bad_memory = memory; ++bad_memory.execution_identity;
    rejects([&] { q8::validate_output_steps(plan, model, compiled, access, bad_memory, resident); },
        "PhaseB stale scratch execution identity rejects");
    bad_memory = memory; bad_memory.allocation_generation = 0;
    rejects([&] { q8::validate_output_steps(plan, model, compiled, access, bad_memory, resident); },
        "PhaseB scratch allocation without generation rejects");
    const auto qsa_step = access.accesses.back().step_id;
    const auto& qsa_io = compiled.io.steps[qsa_step];
    expect(compiled.execution.steps[qsa_step].op == strix::SerialOp::indexed_attention,
        "last selected output access exercises QSA workspace rather than GDN");
    bad_compiled = corrupt_arena(qsa_io.temporaries.front(), [&](auto& a) {
        a.bytes = strix::build_indexed_attention_workspace_layout(model).attention_output_offset +
                  6144U * 4U - 1U;
    });
    rejects([&] { q8::validate_output_steps(plan, model, bad_compiled, access, memory, resident); },
        "PhaseB QSA-specific output source must fit its actual temporary");
    bad_compiled = corrupt_arena(qsa_io.writes.front(), [](auto& a) { a.offset += 2U; });
    rejects([&] { q8::validate_output_steps(plan, model, bad_compiled, access, memory, resident); },
        "PhaseB QSA destination-only misalignment rejects");
    auto short_plan = plan; short_plan.outputs.pop_back();
    rejects([&] { q8::validate_output_steps(short_plan, model, compiled, access, memory, resident); },
        "PhaseB plan must cover every selected output");
    q8::validate_output_steps({}, model, {}, {}, {}, {});
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
    test_output_manifest_and_residency();
    test_output_step_admission();
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
    const q8::View fp32{0x1000, 128}, bf16{0x4000, 38};
    q8::validate_fp32_bf16_views(32, 19, fp32, weights, bf16);
    q8::validate_fp32_bf16_views(32, 19, fp32, {fp32.address, weights.bytes}, bf16);
    rejects([&] { q8::validate_fp32_bf16_views(32, 19, input, weights, bf16); },
        "new FP32 input guard rejects old BF16 input extent");
    rejects([&] { q8::validate_views(32, 19, fp32, weights, bf16); },
        "unchanged old FP32 output guard rejects the new BF16 extent");
    rejects([&] { q8::validate_fp32_bf16_views(32, 19, fp32, weights, {bf16.address, 37}); },
        "new BF16 output guard requires full output bytes");
    rejects([&] { q8::validate_fp32_bf16_views(32, 19, {fp32.address + 4U, fp32.bytes}, weights, bf16); },
        "new FP32 input guard requires16alignment");
    rejects([&] { q8::validate_fp32_bf16_views(32, 19, fp32, weights, {fp32.address, bf16.bytes}); },
        "new output cannot alias FP32 input");
    rejects([&] { q8::validate_fp32_bf16_views(32, 19, fp32, weights, {weights.address, bf16.bytes}); },
        "new output cannot alias Q8 weights");
    rejects([&] { q8::validate_fp32_bf16_views(32, 19,
        {std::numeric_limits<std::uintptr_t>::max() - 15U, 128}, weights, bf16); },
        "new FP32 input guard rejects address overflow");
    return failures ? 1 : 0;
}
