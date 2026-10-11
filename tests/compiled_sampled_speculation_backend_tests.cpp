#include "strix/compiled_sampled_speculation_backend.hpp"

#include "strix/execution_binding.hpp"

#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

strix::kernels::KernelVariant variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    strix::kernels::ExecutionPhase phase,
    const strix::kernels::ShapeClass& shape) {
    strix::kernels::KernelVariant result;
    result.id = std::move(id);
    result.family = family;
    result.phase = phase;
    result.shape = shape;
    result.wave = strix::kernels::WaveSize::wave32;
    result.resources.workgroup_threads = 256;
    result.numerics.input_precision = "bf16";
    result.numerics.accumulator_precision = "fp32";
    result.numerics.output_precision = "bf16";
    result.numerics.row_invariant = true;
    result.numerics.deterministic = true;
    result.qualification =
        strix::kernels::QualificationStatus::oracle_matched;
    result.graph_safe = true;
    result.input_layout = "fixture";
    result.output_layout = "fixture";
    result.compiler_identity = "fixture";
    return result;
}

strix::kernels::KernelRegistry registry_for(
    const strix::ModelContract& model,
    std::uint32_t rows,
    strix::kernels::ExecutionPhase phase) {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);
    strix::ExecutionBindingRequest request;
    request.logical_rows = rows;
    request.request_concurrency = 1;
    std::set<std::string> unique;
    for (const auto& step : execution.steps) {
        if (step.op == strix::SerialOp::position_prepare) {
            continue;
        }
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model,
                step.op,
                request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.auxiliary_size);
        if (!unique.insert(key).second) {
            continue;
        }
        registry.add(
            variant(
                (phase == strix::kernels::ExecutionPhase::decode
                     ? "decode."
                     : "verify.") +
                    key,
                *family,
                phase,
                shape));
    }
    return registry;
}

struct CountingDispatcher final
    : strix::EagerExecutionDispatcher {
    std::uint32_t calls = 0;
    void dispatch_kernel(
        const strix::ExecutionDispatchContext&) override {
        ++calls;
    }
    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext&) override {
        ++calls;
    }
};

struct ScriptedLogitsReader final
    : strix::TargetLogitsReader {
    std::vector<float> base;
    std::vector<float> verify;
    std::uint32_t calls = 0;

    std::vector<float> read_fp32_logits(
        const strix::ExecutionBufferAddress&,
        std::uint32_t rows,
        std::uint64_t) override {
        ++calls;
        return rows == 1 ? base : verify;
    }
};

struct Fixture {
    strix::ModelContract model =
        strix::ModelContract::qwen38_flash_next();
    strix::CompiledSerialExecution serial;
    strix::CompiledSpeculativeVerifyExecution verify;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::SerialSessionStateLayout state_layout;
    strix::SessionStateMemoryBinding state_memory;
    strix::ExecutionTensorAccessPlan serial_access;
    strix::ExecutionTensorAccessPlan verify_access;
    strix::ExecutionMemoryBinding serial_scratch;
    strix::ExecutionMemoryBinding verify_scratch;
    strix::ExecutionRuntimeResources serial_resources;
    strix::ExecutionRuntimeResources verify_resources;
    strix::SerialSessionContext session;

    Fixture() {
        strix::ExecutionBindingRequest serial_binding;
        serial_binding.logical_rows = 1;
        serial_binding.request_concurrency = 1;
        strix::ExecutionBufferRequest serial_buffers;
        serial_buffers.logical_rows = 1;
        const auto serial_registry =
            registry_for(
                model,
                1,
                strix::kernels::ExecutionPhase::decode);
        serial =
            strix::compile_serial_execution(
                model,
                serial_registry,
                serial_binding,
                serial_buffers,
                256);

        strix::SpeculativeVerifyBindingRequest verify_binding;
        verify_binding.request_concurrency = 1;
        strix::ExecutionBufferRequest verify_buffers;
        verify_buffers.logical_rows = 2;
        const auto verify_registry =
            registry_for(
                model,
                2,
                strix::kernels::ExecutionPhase::speculative_verify);
        verify =
            strix::compile_speculative_verify_execution(
                model,
                2,
                verify_registry,
                verify_binding,
                verify_buffers,
                256);

        strix::ManifestTensor tensor;
        tensor.name = "fixture.weight";
        tensor.dimensions = {64, 64};
        tensor.ggml_type = 1;
        tensor.byte_size = 8192;
        tensor.absolute_offset = 4096;
        tensor.role = strix::TensorRole::dense_projection;
        manifest.architecture = "fixture";
        manifest.tensors.push_back(tensor);
        manifest.total_tensor_bytes = tensor.byte_size;
        manifest.layout_identity =
            strix::model_manifest_layout_identity(manifest);

        weights.manifest_layout_identity =
            manifest.layout_identity;
        weights.model_content_identity.bytes[0] = 0x42;
        weights.tensors.push_back(
            {0,
             strix::RuntimeTensorStorage::resident,
             static_cast<std::uintptr_t>(0x600000000ULL),
             tensor.byte_size,
             1,
             0});

        strix::SerialSessionStateLayoutRequest state_request;
        state_request.context_tokens = 64;
        state_layout =
            strix::build_serial_session_state_layout(
                model,
                state_request);

        state_memory.layout_identity = state_layout.layout_identity;
        state_memory.device_base =
            static_cast<std::uintptr_t>(0x610000000ULL);
        state_memory.device_bytes = state_layout.device_total_bytes;
        state_memory.device_allocation_generation = 2;
        state_memory.host_base =
            static_cast<std::uintptr_t>(0x620000000ULL);
        state_memory.host_bytes = state_layout.host_total_bytes;
        state_memory.host_allocation_generation = 3;

        serial_access.manifest_layout_identity =
            manifest.layout_identity;
        serial_access.plan_identity = 0x7001ULL;
        verify_access.manifest_layout_identity =
            manifest.layout_identity;
        verify_access.plan_identity = 0x7002ULL;

        serial_scratch.execution_identity = serial.execution_identity;
        serial_scratch.arena_base =
            static_cast<std::uintptr_t>(0x630000000ULL);
        serial_scratch.arena_bytes = serial.arena.total_bytes;
        serial_scratch.allocation_generation = 4;

        verify_scratch.execution_identity = verify.execution_identity;
        verify_scratch.arena_base =
            static_cast<std::uintptr_t>(0x640000000ULL);
        verify_scratch.arena_bytes = verify.arena.total_bytes;
        verify_scratch.allocation_generation = 5;

        serial_resources.scratch = &serial_scratch;
        serial_resources.model_manifest = &manifest;
        serial_resources.model_weights = &weights;
        serial_resources.tensor_access = &serial_access;
        serial_resources.session_state_layout = &state_layout;
        serial_resources.session_state_memory = &state_memory;

        verify_resources.scratch = &verify_scratch;
        verify_resources.model_manifest = &manifest;
        verify_resources.model_weights = &weights;
        verify_resources.tensor_access = &verify_access;
        verify_resources.session_state_layout = &state_layout;
        verify_resources.session_state_memory = &state_memory;

        const auto ids =
            strix::build_runtime_execution_identities(
                serial.execution_identity,
                serial.arena,
                serial_resources);
        session.execution_identity = serial.execution_identity;
        session.state_layout_identity = state_layout.layout_identity;
        session.runtime_compatibility_identity = ids.compatibility_identity;
        session.next_position = 10;
        session.committed_tokens = 10;
        session.state.valid = all_state();
        session.state.generations.fill(5);
    }
};

std::vector<float> one_hot_logits(
    std::size_t vocab,
    std::size_t selected) {
    std::vector<float> result(
        vocab,
        -1000.0F);
    result[selected] = 0.0F;
    return result;
}

}  // namespace

int main() {
    Fixture fixture;
    const auto vocab =
        static_cast<std::size_t>(
            fixture.model.vocab_size);
    ScriptedLogitsReader reader;
    reader.base =
        one_hot_logits(vocab, 0);
    const auto first =
        one_hot_logits(vocab, 1);
    const auto second =
        one_hot_logits(vocab, 2);
    reader.verify.reserve(
        first.size() + second.size());
    reader.verify.insert(
        reader.verify.end(),
        first.begin(),
        first.end());
    reader.verify.insert(
        reader.verify.end(),
        second.begin(),
        second.end());

    strix::Fp32SoftmaxTargetDistributionCollector collector(
        reader);
    CountingDispatcher serial_dispatcher;
    CountingDispatcher verify_dispatcher;
    strix::CompiledSampledSpeculationBackend backend(
        fixture.serial,
        fixture.serial_resources,
        serial_dispatcher,
        fixture.verify,
        fixture.verify_resources,
        verify_dispatcher,
        collector);

    const std::vector<std::uint32_t> candidates{0, 1};
    const auto distributions =
        backend.verify_target_distributions(
            fixture.session,
            fixture.serial_resources,
            candidates);

    expect(
        distributions.size() == 3 &&
            distributions[0][0] == 1.0 &&
            distributions[1][1] == 1.0 &&
            distributions[2][2] == 1.0,
        "compiled sampled backend joins serial base distribution with verify suffix rows");
    expect(
        reader.calls == 2 &&
            verify_dispatcher.calls ==
                fixture.verify.topology.steps.size(),
        "sampled backend materializes base and verify logits around one verify dispatch");
    expect(
        fixture.session.next_position == 12 &&
            fixture.session.committed_tokens == 12,
        "sampled backend advances persistent target state through candidate window");

    Fixture zero;
    ScriptedLogitsReader zero_reader;
    zero_reader.base =
        one_hot_logits(
            static_cast<std::size_t>(
                zero.model.vocab_size),
            1);
    strix::Fp32SoftmaxTargetDistributionCollector
        zero_collector(zero_reader);
    CountingDispatcher zero_serial_dispatcher;
    CountingDispatcher zero_verify_dispatcher;
    strix::CompiledSampledSpeculationBackend zero_backend(
        zero.serial,
        zero.serial_resources,
        zero_serial_dispatcher,
        zero.verify,
        zero.verify_resources,
        zero_verify_dispatcher,
        zero_collector);

    const std::vector<std::uint32_t> none;
    const auto base_only =
        zero_backend.verify_target_distributions(
            zero.session,
            zero.serial_resources,
            none);
    expect(
        base_only.size() == 1 &&
            base_only[0][1] == 1.0 &&
            zero_verify_dispatcher.calls == 0 &&
            zero.session.next_position == 10,
        "depth-zero sampled backend reads base distribution without target-state mutation");

    bool bad_logits_rejected = false;
    try {
        Fixture bad;
        ScriptedLogitsReader bad_reader;
        bad_reader.base =
            one_hot_logits(
                static_cast<std::size_t>(
                    bad.model.vocab_size),
                0);
        bad_reader.base[1] =
            std::numeric_limits<float>::quiet_NaN();
        strix::Fp32SoftmaxTargetDistributionCollector
            bad_collector(bad_reader);
        static_cast<void>(
            bad_collector.collect_base_target_distribution(
                bad.serial,
                bad.serial_resources,
                bad.session));
    } catch (const std::invalid_argument&) {
        bad_logits_rejected = true;
    }
    expect(
        bad_logits_rejected,
        "sampled distribution collector rejects non-finite target logits");

    if (failures == 0) {
        std::cout
            << "compiled sampled speculation backend tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
