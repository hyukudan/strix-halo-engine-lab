#include "strix/serial_session.hpp"

#include <set>
#include <string>
#include <utility>

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

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::decode;
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
        strix::kernels::QualificationStatus::oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "test";
    variant.output_layout = "test";
    variant.compiler_identity = "test";
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request) {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);

    std::set<std::string> added;
    for (const auto& step : execution.steps) {
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model, step.op, request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.auxiliary_size);
        if (!added.insert(key).second) {
            continue;
        }
        registry.add(
            make_variant(
                "session." + key,
                *family,
                shape));
    }
    return registry;
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

struct SessionDispatcher final
    : strix::EagerExecutionDispatcher {
    std::uint32_t expected_token = 0;
    std::uint64_t expected_position = 0;
    std::uint32_t calls = 0;
    bool fail_on_first_recurrent = false;
    bool failed = false;

    void verify_scalars(
        const strix::ExecutionDispatchContext& context) {
        expect(context.input_token_id == expected_token,
               "dispatch receives session input token");
        expect(context.position == expected_position,
               "dispatch receives session position");
        expect(context.session_state_layout != nullptr &&
                   context.session_state_memory != nullptr,
               "dispatch receives persistent state layout and memory");
    }

    void dispatch_kernel(
        const strix::ExecutionDispatchContext& context) override {
        verify_scalars(context);
        if (context.op ==
                strix::SerialOp::gated_delta_net &&
            context.session_state_layout != nullptr &&
            context.session_state_memory != nullptr) {
            const auto recurrent =
                strix::resolve_session_state(
                    *context.session_state_layout,
                    *context.session_state_memory,
                    strix::SessionStateComponent::recurrent_matrix);
            expect(recurrent.storage ==
                       strix::SessionStateStorage::device &&
                       recurrent.address != 0,
                   "GDN dispatch resolves recurrent device state");
        }
        if (fail_on_first_recurrent &&
            !failed &&
            context.op ==
                strix::SerialOp::gated_delta_net) {
            failed = true;
            throw std::runtime_error(
                "injected recurrent failure");
        }
        ++calls;
    }

    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext& context) override {
        verify_scalars(context);
        if (context.op ==
                strix::SerialOp::position_prepare &&
            context.session_state_layout != nullptr &&
            context.session_state_memory != nullptr) {
            const auto position =
                strix::resolve_session_state(
                    *context.session_state_layout,
                    *context.session_state_memory,
                    strix::SessionStateComponent::position_history);
            expect(position.storage ==
                       strix::SessionStateStorage::host &&
                       position.address != 0,
                   "position step resolves host state");
        }
        ++calls;
    }
};

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::ExecutionBindingRequest binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;
    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;

    const auto registry =
        make_registry(model, binding_request);
    const auto compiled =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            256);

    strix::SerialSessionStateLayoutRequest state_request;
    state_request.context_tokens = 4;
    const auto state_layout =
        strix::build_serial_session_state_layout(
            model, state_request);

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        static_cast<std::uintptr_t>(0x31000000ULL);
    state_memory.device_bytes =
        state_layout.device_total_bytes;
    state_memory.device_allocation_generation = 3;
    state_memory.host_base =
        static_cast<std::uintptr_t>(0x32000000ULL);
    state_memory.host_bytes =
        state_layout.host_total_bytes;
    state_memory.host_allocation_generation = 5;

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        compiled.execution_identity;
    memory.arena_base =
        static_cast<std::uintptr_t>(0x30000000ULL);
    memory.arena_bytes =
        compiled.arena.total_bytes;
    memory.allocation_generation = 1;

    strix::SerialSessionContext session;
    session.execution_identity =
        compiled.execution_identity;
    session.state_layout_identity =
        state_layout.layout_identity;
    session.state.valid = all_state();

    SessionDispatcher first;
    first.expected_token = 123;
    first.expected_position = 0;
    const auto first_result =
        strix::execute_serial_token(
            compiled,
            session,
            memory,
            state_layout,
            state_memory,
            123,
            first);

    expect(first_result.completed_steps ==
               compiled.execution.steps.size(),
           "first serial token executes the full DAG");
    expect(session.next_position == 1 &&
               session.committed_tokens == 1,
           "successful serial token commits position and token count");
    expect(strix::arithmetic_lineage_serial_only(session.arithmetic_lineage,
               compiled.execution_identity, 1),
           "real serial producer publishes its one-row arithmetic lineage after state completion");

    SessionDispatcher second;
    second.expected_token = 456;
    second.expected_position = 1;
    static_cast<void>(
        strix::execute_serial_token(
            compiled,
            session,
            memory,
            state_layout,
            state_memory,
            456,
            second));
    expect(session.next_position == 2 &&
               session.committed_tokens == 2,
           "second serial token advances from committed session state");
    expect(strix::arithmetic_lineage_valid(session.arithmetic_lineage, 2) &&
               session.arithmetic_lineage.intervals.size() == 1 &&
               session.arithmetic_lineage.intervals[0].reduction_plan_identity == compiled.kernel_plan_identity,
           "real consecutive serial rows bind the compiled reduction plan without synthetic chunks");

    const auto before_failure = session;
    SessionDispatcher failing;
    failing.expected_token = 789;
    failing.expected_position = 2;
    failing.fail_on_first_recurrent = true;

    bool failure_observed = false;
    try {
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                session,
                memory,
                state_layout,
                state_memory,
                789,
                failing));
    } catch (const std::runtime_error&) {
        failure_observed = true;
    }
    expect(failure_observed,
           "injected serial kernel failure propagates");
    expect(session.next_position ==
               before_failure.next_position &&
               session.committed_tokens ==
                   before_failure.committed_tokens &&
               session.state.generations ==
                   before_failure.state.generations &&
               session.state.valid ==
                   before_failure.state.valid,
           "failed serial token does not publish committed metadata");
    expect(session.health ==
               strix::SerialSessionHealth::poisoned,
           "mid-execution failure poisons the session until recovery");
    expect(session.arithmetic_lineage.tainted &&
               session.arithmetic_lineage.history_identity == before_failure.arithmetic_lineage.history_identity,
           "failed state mutation taints previous history without publishing a failed row");

    bool poisoned_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                session,
                memory,
                state_layout,
                state_memory,
                1,
                unused));
    } catch (const std::logic_error&) {
        poisoned_rejected = true;
    }
    expect(poisoned_rejected,
           "poisoned session cannot continue decoding");


    auto missing_state = before_failure;
    missing_state.state.valid =
        strix::PersistentState::none;
    bool missing_state_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                missing_state,
                memory,
                state_layout,
                state_memory,
                1,
                unused));
    } catch (const std::invalid_argument&) {
        missing_state_rejected = true;
    }
    expect(missing_state_rejected &&
               missing_state.health ==
                   strix::SerialSessionHealth::ready,
           "preflight state failure does not poison session");

    auto wrong_identity = before_failure;
    ++wrong_identity.execution_identity;
    bool identity_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                wrong_identity,
                memory,
                state_layout,
                state_memory,
                1,
                unused));
    } catch (const std::invalid_argument&) {
        identity_rejected = true;
    }
    expect(identity_rejected,
           "session compiled-plan identity mismatch is rejected");


    auto wrong_state_layout = before_failure;
    ++wrong_state_layout.state_layout_identity;
    bool state_layout_identity_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                wrong_state_layout,
                memory,
                state_layout,
                state_memory,
                1,
                unused));
    } catch (const std::invalid_argument&) {
        state_layout_identity_rejected = true;
    }
    expect(state_layout_identity_rejected,
           "state-layout identity mismatch is rejected");

    auto invalid_token_session = before_failure;
    bool token_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                invalid_token_session,
                memory,
                state_layout,
                state_memory,
                static_cast<std::uint32_t>(
                    compiled.vocab_size),
                unused));
    } catch (const std::invalid_argument&) {
        token_rejected = true;
    }
    expect(token_rejected,
           "out-of-vocabulary input token is rejected before dispatch");

    auto full_context = before_failure;
    full_context.next_position =
        state_layout.context_tokens;
    bool context_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                full_context,
                memory,
                state_layout,
                state_memory,
                1,
                unused));
    } catch (const std::out_of_range&) {
        context_rejected = true;
    }
    expect(context_rejected,
           "allocated session context capacity is enforced before dispatch");


    strix::ModelManifest manifest;
    manifest.architecture = "session-fixture";
    strix::ManifestTensor model_tensor;
    model_tensor.name = "token_embd.weight";
    model_tensor.dimensions = {64, 64};
    model_tensor.ggml_type = 1;
    model_tensor.byte_size = 8192;
    model_tensor.absolute_offset = 4096;
    model_tensor.role = strix::TensorRole::embedding;
    model_tensor.placement.kind =
        strix::PlacementKind::local;
    model_tensor.placement.node = 0;
    manifest.tensors.push_back(model_tensor);
    manifest.total_tensor_bytes =
        model_tensor.byte_size;
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::ModelRuntimeBinding model_weights;
    model_weights.manifest_layout_identity =
        manifest.layout_identity;
    model_weights.model_content_identity.bytes[0] =
        static_cast<std::uint8_t>(0xfeedbeefULL & 0xffULL);
    model_weights.model_content_identity.bytes[1] = 0x5a;
    model_weights.tensors.push_back(
        {
            0,
            strix::RuntimeTensorStorage::resident,
            static_cast<std::uintptr_t>(
                0x33000000ULL),
            model_tensor.byte_size,
            11,
            0,
        });

    strix::ExecutionTensorAccessPlan model_tensor_access;
    model_tensor_access.manifest_layout_identity =
        manifest.layout_identity;
    model_tensor_access.plan_identity =
        0xfaceULL;

    strix::ExecutionRuntimeResources resources;
    resources.scratch = &memory;
    resources.model_manifest = &manifest;
    resources.model_weights = &model_weights;
    resources.tensor_access = &model_tensor_access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &state_memory;

    const auto runtime_ids =
        strix::build_runtime_execution_identities(
            compiled.execution_identity,
            compiled.arena,
            resources);

    auto model_bound = before_failure;
    model_bound.runtime_compatibility_identity =
        runtime_ids.compatibility_identity;

    SessionDispatcher model_bound_dispatcher;
    model_bound_dispatcher.expected_token = 321;
    model_bound_dispatcher.expected_position =
        model_bound.next_position;
    static_cast<void>(
        strix::execute_serial_token(
            compiled,
            model_bound,
            resources,
            321,
            model_bound_dispatcher));
    expect(model_bound.committed_tokens ==
               before_failure.committed_tokens + 1 &&
               model_bound.next_position ==
                   before_failure.next_position + 1,
           "model-bound session executes with complete runtime resources");

    auto changed_weights = model_weights;
    ++changed_weights.model_content_identity.bytes[0];
    auto changed_resources = resources;
    changed_resources.model_weights =
        &changed_weights;

    auto incompatible_session = before_failure;
    incompatible_session.runtime_compatibility_identity =
        runtime_ids.compatibility_identity;
    SessionDispatcher incompatible_dispatcher;
    incompatible_dispatcher.expected_token = 322;
    incompatible_dispatcher.expected_position =
        incompatible_session.next_position;

    bool model_content_rejected = false;
    try {
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                incompatible_session,
                changed_resources,
                322,
                incompatible_dispatcher));
    } catch (const std::invalid_argument&) {
        model_content_rejected = true;
    }
    expect(model_content_rejected &&
               incompatible_dispatcher.calls == 0 &&
               incompatible_session.health ==
                   strix::SerialSessionHealth::ready,
           "model-bound session rejects changed model content before dispatch");

    bool legacy_model_bound_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                incompatible_session,
                memory,
                state_layout,
                state_memory,
                1,
                unused));
    } catch (const std::invalid_argument&) {
        legacy_model_bound_rejected = true;
    }
    expect(legacy_model_bound_rejected,
           "model-bound session cannot bypass content checks through legacy path");

    if (failures == 0) {
        std::cout << "serial session tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
