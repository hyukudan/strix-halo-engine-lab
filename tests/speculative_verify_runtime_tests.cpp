#include "strix/speculative_verify_runtime.hpp"

#include "strix/execution_binding.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <iostream>
#include <stdexcept>

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

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::
            speculative_verify;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::
            oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "verify";
    variant.output_layout = "verify";
    variant.compiler_identity = "fixture";
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    std::uint32_t depth) {
    strix::kernels::KernelRegistry registry;
    const auto verify =
        strix::build_speculative_verify_plan(
            model,
            depth);

    strix::ExecutionBindingRequest shape_request;
    shape_request.logical_rows = depth;
    shape_request.request_concurrency = 1;

    std::set<std::string> unique;
    for (const auto& step :
         verify.steps) {
        if (step.op ==
            strix::SerialOp::position_prepare) {
            continue;
        }
        const auto family =
            strix::operator_family_for(
                step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model,
                step.op,
                shape_request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(
                    *family)) +
            "." +
            std::to_string(
                shape.auxiliary_size);
        if (!unique.insert(key).second) {
            continue;
        }
        registry.add(
            make_variant(
                "verify." + key,
                *family,
                shape));
    }
    return registry;
}

struct RecordingDispatcher final
    : strix::EagerExecutionDispatcher {
    std::uint32_t calls = 0;
    std::optional<std::uint32_t> fail_step;
    std::uint64_t expected_position = 0;
    std::uint64_t expected_identity = 0;
    std::vector<std::uint32_t> expected_tokens;

    void verify(
        const strix::ExecutionDispatchContext& context) {
        expect(context.execution_identity == expected_identity && expected_identity != 0,
               "verify dispatch carries compiled execution identity");
        expect(context.logical_rows ==
                   expected_tokens.size(),
               "verify dispatch carries compiled candidate row count");
        expect(context.input_tokens.size() ==
                   expected_tokens.size(),
               "verify dispatch carries every candidate token");
        expect(std::equal(
                   context.input_tokens.begin(),
                   context.input_tokens.end(),
                   expected_tokens.begin()),
               "verify dispatch preserves candidate token order");
        expect(context.position ==
                   expected_position,
               "verify dispatch carries base candidate position");
    }

    void dispatch_kernel(
        const strix::ExecutionDispatchContext& context) override {
        verify(context);
        if (fail_step.has_value() &&
            *fail_step == context.step_id) {
            throw std::runtime_error(
                "injected verify kernel failure");
        }
        ++calls;
    }

    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext& context) override {
        verify(context);
        ++calls;
    }
};

struct Fixture {
    strix::ModelContract model =
        strix::ModelContract::qwen38_flash_next();
    strix::CompiledSpeculativeVerifyExecution compiled;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionTensorAccessPlan tensor_access;
    strix::SerialSessionStateLayout state_layout;
    strix::SessionStateMemoryBinding state_memory;
    strix::ExecutionMemoryBinding scratch;
    strix::ExecutionRuntimeResources resources;
    strix::SerialSessionContext session;

    Fixture() {
        const auto registry =
            make_registry(model, 4);

        strix::SpeculativeVerifyBindingRequest binding;
        binding.request_concurrency = 1;

        strix::ExecutionBufferRequest buffers;
        buffers.logical_rows = 4;

        compiled =
            strix::compile_speculative_verify_execution(
                model,
                4,
                registry,
                binding,
                buffers,
                256);

        strix::ManifestTensor tensor;
        tensor.name = "fixture.weight";
        tensor.dimensions = {64, 64};
        tensor.ggml_type = 1;
        tensor.byte_size = 8192;
        tensor.absolute_offset = 4096;
        tensor.role =
            strix::TensorRole::dense_projection;
        manifest.architecture = "fixture";
        manifest.tensors.push_back(tensor);
        manifest.total_tensor_bytes =
            tensor.byte_size;
        manifest.layout_identity =
            strix::model_manifest_layout_identity(
                manifest);

        weights.manifest_layout_identity =
            manifest.layout_identity;
        weights.model_content_identity.bytes[0] =
            0x42;
        weights.tensors.push_back(
            {
                0,
                strix::RuntimeTensorStorage::resident,
                static_cast<std::uintptr_t>(
                    0x110000000ULL),
                tensor.byte_size,
                1,
                0,
            });

        tensor_access.manifest_layout_identity =
            manifest.layout_identity;
        tensor_access.plan_identity =
            0x2222ULL;

        strix::SerialSessionStateLayoutRequest state_request;
        state_request.context_tokens = 64;
        state_layout =
            strix::build_serial_session_state_layout(
                model,
                state_request);

        state_memory.layout_identity =
            state_layout.layout_identity;
        state_memory.device_base =
            static_cast<std::uintptr_t>(
                0x120000000ULL);
        state_memory.device_bytes =
            state_layout.device_total_bytes;
        state_memory.device_allocation_generation = 2;
        state_memory.host_base =
            static_cast<std::uintptr_t>(
                0x130000000ULL);
        state_memory.host_bytes =
            state_layout.host_total_bytes;
        state_memory.host_allocation_generation = 3;

        scratch.execution_identity =
            compiled.execution_identity;
        scratch.arena_base =
            static_cast<std::uintptr_t>(
                0x140000000ULL);
        scratch.arena_bytes =
            compiled.arena.total_bytes;
        scratch.allocation_generation = 4;

        resources.scratch = &scratch;
        resources.model_manifest = &manifest;
        resources.model_weights = &weights;
        resources.tensor_access = &tensor_access;
        resources.session_state_layout =
            &state_layout;
        resources.session_state_memory =
            &state_memory;

        session.execution_identity =
            0x9999ULL;
        session.state_layout_identity =
            state_layout.layout_identity;
        session.runtime_compatibility_identity =
            0x8888ULL;
        session.next_position = 10;
        session.committed_tokens = 10;
        session.state.valid = all_state();
        session.state.generations.fill(5);
        for (std::uint64_t position = 0; position < session.next_position; ++position) {
            session.arithmetic_lineage = strix::prepare_serial_row_lineage(session.arithmetic_lineage,
                position, session.execution_identity, compiled.kernel_plan_identity);
        }
    }
};

}  // namespace

int main() {
    {
        Fixture fixture;
        const std::vector<std::uint32_t> candidates{
            10, 20, 30, 40};

        RecordingDispatcher dispatcher;
        dispatcher.expected_identity = fixture.compiled.execution_identity;
        dispatcher.expected_position = 10;
        dispatcher.expected_tokens =
            candidates;

        const auto result =
            strix::execute_compiled_speculative_verify_eager(
                fixture.compiled,
                fixture.resources,
                fixture.session,
                candidates,
                dispatcher);

        expect(result.execution.completed_steps ==
                   fixture.compiled.topology.steps.size(),
               "verify eager execution completes full target DAG");
        expect(fixture.session.next_position == 14 &&
                   fixture.session.committed_tokens == 14 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::ready,
               "successful verify advances target state by candidate depth");
        expect(dispatcher.calls ==
                   fixture.compiled.topology.steps.size(),
               "verify dispatcher sees every kernel/runtime step");
        expect(fixture.session.arithmetic_lineage.history_identity == 0 &&
                   fixture.session.arithmetic_lineage.intervals.empty(),
               "verify window does not fabricate or retain serial arithmetic history");
    }

    {
        Fixture fixture;
        const std::vector<std::uint32_t> short_candidates{
            10, 20};
        RecordingDispatcher dispatcher;
        dispatcher.expected_identity = fixture.compiled.execution_identity;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_compiled_speculative_verify_eager(
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    short_candidates,
                    dispatcher));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected &&
                   dispatcher.calls == 0 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::ready,
               "candidate-depth mismatch fails before dispatch");
        expect(strix::arithmetic_lineage_valid(fixture.session.arithmetic_lineage, 10),
               "verify preflight failure retains the earlier independently recorded history");
    }

    {
        Fixture fixture;
        const std::vector<std::uint32_t> candidates{
            10, 20, 30, 40};

        RecordingDispatcher dispatcher;
        dispatcher.expected_identity = fixture.compiled.execution_identity;
        dispatcher.expected_position = 10;
        dispatcher.expected_tokens =
            candidates;
        const auto gdn =
            std::find_if(
                fixture.compiled.topology.steps.begin(),
                fixture.compiled.topology.steps.end(),
                [](const auto& step) {
                    return step.op ==
                        strix::SerialOp::gated_delta_net;
                });
        if (gdn ==
            fixture.compiled.topology.steps.end()) {
            throw std::runtime_error(
                "fixture verify plan has no GDN step");
        }
        dispatcher.fail_step =
            gdn->id;

        bool failed = false;
        try {
            static_cast<void>(
                strix::execute_compiled_speculative_verify_eager(
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    candidates,
                    dispatcher));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        expect(failed &&
                   fixture.session.next_position == 10 &&
                   fixture.session.committed_tokens == 10 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::poisoned,
               "mid-verify failure poisons target session without publishing metadata advance");
        expect(fixture.session.arithmetic_lineage.tainted &&
                   fixture.session.arithmetic_lineage.history_identity == 0,
               "failed verify mutation invalidates arithmetic history");
    }

    {
        Fixture fixture;
        const std::vector<std::uint32_t> candidates{
            10,
            static_cast<std::uint32_t>(
                fixture.compiled.vocab_size),
            30,
            40};
        RecordingDispatcher dispatcher;
        dispatcher.expected_identity = fixture.compiled.execution_identity;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_compiled_speculative_verify_eager(
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    candidates,
                    dispatcher));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected &&
                   dispatcher.calls == 0 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::ready &&
                   fixture.session.next_position == 10 &&
                   fixture.session.committed_tokens == 10,
               "out-of-vocabulary verify candidate fails before dispatch");
    }

    {
        Fixture fixture;
        fixture.session.next_position = 62;
        fixture.session.committed_tokens = 62;
        const std::vector<std::uint32_t> candidates{
            10, 20, 30, 40};
        RecordingDispatcher dispatcher;
        dispatcher.expected_identity = fixture.compiled.execution_identity;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_compiled_speculative_verify_eager(
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    candidates,
                    dispatcher));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        expect(rejected &&
                   dispatcher.calls == 0 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::ready,
               "verify window beyond allocated state capacity fails before dispatch");
    }

    if (failures == 0) {
        std::cout << "speculative verify runtime tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
