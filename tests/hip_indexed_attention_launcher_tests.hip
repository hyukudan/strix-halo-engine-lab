#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits +=
        0x7fffU +
        ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        bits >> 16U);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            value) << 16U;
    float output = 0.0F;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

float round_bf16(float value) {
    return bf16_to_float(
        float_to_bf16(value));
}

const strix::ExecutionStep& first_indexed(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::indexed_attention) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no indexed-attention step");
}

std::vector<float> make_matrix(
    std::uint32_t input,
    std::uint32_t output,
    std::uint32_t salt) {
    std::vector<float> result(
        static_cast<std::size_t>(input) *
        output);
    for (std::uint32_t row = 0;
         row < output;
         ++row) {
        for (std::uint32_t feature = 0;
             feature < input;
             ++feature) {
            const auto mixed =
                (row * 17U +
                 feature * 11U +
                 salt) %
                31U;
            result[
                static_cast<std::size_t>(row) *
                    input +
                feature] =
                static_cast<float>(
                    static_cast<int>(mixed) -
                    15) *
                0.00035F;
        }
    }
    return result;
}

std::vector<float> project_bf16(
    const std::vector<std::uint16_t>& input,
    const std::vector<float>& weights,
    std::uint32_t input_width,
    std::uint32_t output_width) {
    std::vector<float> output(output_width);
    for (std::uint32_t row = 0;
         row < output_width;
         ++row) {
        double sum = 0.0;
        for (std::uint32_t feature = 0;
             feature < input_width;
             ++feature) {
            sum +=
                static_cast<double>(
                    bf16_to_float(
                        input[feature])) *
                static_cast<double>(
                    weights[
                        static_cast<std::size_t>(row) *
                            input_width +
                        feature]);
        }
        output[row] =
            round_bf16(
                static_cast<float>(sum));
    }
    return output;
}

void rms_norm_heads(
    std::vector<float>& values,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t stride,
    float epsilon) {
    for (std::uint32_t head = 0;
         head < heads;
         ++head) {
        double sum = 0.0;
        const auto base =
            static_cast<std::size_t>(head) *
            stride;
        for (std::uint32_t feature = 0;
             feature < head_dim;
             ++feature) {
            const auto value =
                values[base + feature];
            sum +=
                static_cast<double>(value) *
                value;
        }
        const auto inv =
            1.0 /
            std::sqrt(
                sum /
                    static_cast<double>(
                        head_dim) +
                epsilon);
        for (std::uint32_t feature = 0;
             feature < head_dim;
             ++feature) {
            values[base + feature] =
                round_bf16(
                    static_cast<float>(
                        values[base + feature] *
                        inv));
        }
    }
}

}  // namespace

int main() try {
    auto model =
        strix::ModelContract::qwen38_flash_next();
    model.hidden_size = 256;
    model.attention_heads = 2;
    model.kv_heads = 1;
    model.indexer_n_heads = 2;
    model.indexer_budget = 2;
    model.indexer_compress_ratio = 2;
    model.max_position_embeddings = 8;

    expect(
        model.validate().empty(),
        "reduced indexed-attention fixture model is valid");

    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            execution);
    const auto& step =
        first_indexed(execution);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.activation_element_bytes = 2;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            buffer_request,
            execution,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc indexed arena");

    strix::ExecutionMemoryBinding scratch;
    scratch.execution_identity =
        0x494e445854455354ULL;
    scratch.arena_base =
        reinterpret_cast<std::uintptr_t>(
            arena_pointer);
    scratch.arena_bytes =
        arena.total_bytes;
    scratch.allocation_generation = 1;

    const auto input =
        strix::resolve_execution_buffer(
            scratch.execution_identity,
            arena,
            scratch,
            step_io.reads[0]);
    const auto output =
        strix::resolve_execution_buffer(
            scratch.execution_identity,
            arena,
            scratch,
            step_io.writes[0]);

    std::vector<std::uint16_t> hidden(
        static_cast<std::size_t>(
            model.hidden_size));
    for (std::size_t index = 0;
         index < hidden.size();
         ++index) {
        hidden[index] =
            float_to_bf16(
                static_cast<float>(
                    static_cast<int>(
                        index % 19U) -
                    9) /
                17.0F);
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input.address),
            hidden.data(),
            hidden.size() *
                sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy indexed hidden input");

    const auto q_width =
        static_cast<std::uint32_t>(
            model.attention_heads *
            model.head_dim *
            2ULL);
    const auto kv_width =
        static_cast<std::uint32_t>(
            model.kv_heads *
            model.head_dim);
    const auto attention_width =
        static_cast<std::uint32_t>(
            model.attention_heads *
            model.head_dim);
    const auto index_q_width =
        static_cast<std::uint32_t>(
            model.indexer_n_heads *
            model.indexer_head_dim);
    const auto index_dim =
        static_cast<std::uint32_t>(
            model.indexer_head_dim);
    const auto hidden_width =
        static_cast<std::uint32_t>(
            model.hidden_size);

    const auto q =
        make_matrix(
            hidden_width,
            q_width,
            1);
    const auto k =
        make_matrix(
            hidden_width,
            kv_width,
            2);
    const auto v =
        make_matrix(
            hidden_width,
            kv_width,
            3);
    const auto o =
        make_matrix(
            attention_width,
            hidden_width,
            4);
    const auto index_q =
        make_matrix(
            hidden_width,
            index_q_width,
            5);
    const auto index_k =
        make_matrix(
            hidden_width,
            index_dim,
            6);
    std::vector<float> q_norm(
        static_cast<std::size_t>(
            model.head_dim),
        0.0F);
    std::vector<float> k_norm =
        q_norm;
    std::vector<float> index_q_norm(
        index_dim,
        0.0F);
    std::vector<float> index_k_norm =
        index_q_norm;

    auto q_expected =
        project_bf16(
            hidden,
            q,
            hidden_width,
            q_width);
    auto k_expected =
        project_bf16(
            hidden,
            k,
            hidden_width,
            kv_width);
    const auto v_expected =
        project_bf16(
            hidden,
            v,
            hidden_width,
            kv_width);
    const auto index_k_expected =
        project_bf16(
            hidden,
            index_k,
            hidden_width,
            index_dim);

    rms_norm_heads(
        q_expected,
        static_cast<std::uint32_t>(
            model.attention_heads),
        static_cast<std::uint32_t>(
            model.head_dim),
        static_cast<std::uint32_t>(
            model.head_dim * 2ULL),
        model.rms_norm_epsilon);
    rms_norm_heads(
        k_expected,
        static_cast<std::uint32_t>(
            model.kv_heads),
        static_cast<std::uint32_t>(
            model.head_dim),
        static_cast<std::uint32_t>(
            model.head_dim),
        model.rms_norm_epsilon);

    // position=0 => partial RoPE is identity.
    std::vector<float> attention_output(
        attention_width);
    const auto kv_group =
        static_cast<std::uint32_t>(
            model.attention_heads /
            model.kv_heads);
    for (std::uint32_t head = 0;
         head < model.attention_heads;
         ++head) {
        const auto kv_head =
            head / kv_group;
        for (std::uint32_t feature = 0;
             feature < model.head_dim;
             ++feature) {
            const auto value =
                round_bf16(
                    v_expected[
                        static_cast<std::size_t>(
                            kv_head) *
                            model.head_dim +
                        feature]);
            const auto gate =
                round_bf16(
                    1.0F /
                    (1.0F +
                     std::exp(
                         -q_expected[
                             static_cast<std::size_t>(
                                 head) *
                                 model.head_dim *
                                 2ULL +
                             model.head_dim +
                             feature])));
            attention_output[
                static_cast<std::size_t>(
                    head) *
                    model.head_dim +
                feature] =
                round_bf16(
                    value * gate);
        }
    }

    std::vector<float> expected_output(
        hidden_width);
    for (std::uint32_t row = 0;
         row < hidden_width;
         ++row) {
        double sum = 0.0;
        for (std::uint32_t feature = 0;
             feature < attention_width;
             ++feature) {
            sum +=
                static_cast<double>(
                    attention_output[feature]) *
                static_cast<double>(
                    o[
                        static_cast<std::size_t>(row) *
                            attention_width +
                        feature]);
        }
        expected_output[row] =
            round_bf16(
                static_cast<float>(
                    sum));
    }

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";

    const auto layer =
        *step.layer;
    const auto add_tensor =
        [&](std::string name,
            strix::TensorRole role,
            std::vector<std::uint64_t> dims,
            const std::vector<float>& values) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.role = role;
            tensor.layer = layer;
            tensor.ggml_type = 0;
            tensor.dimensions =
                std::move(dims);
            tensor.byte_size =
                values.size() *
                sizeof(float);
            manifest.total_tensor_bytes +=
                tensor.byte_size;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add_tensor(
        "blk.3.attn_q.weight",
        strix::TensorRole::attention_projection,
        {hidden_width, q_width},
        q);
    add_tensor(
        "blk.3.attn_k.weight",
        strix::TensorRole::attention_projection,
        {hidden_width, kv_width},
        k);
    add_tensor(
        "blk.3.attn_v.weight",
        strix::TensorRole::attention_projection,
        {hidden_width, kv_width},
        v);
    add_tensor(
        "blk.3.attn_output.weight",
        strix::TensorRole::attention_projection,
        {attention_width, hidden_width},
        o);
    add_tensor(
        "blk.3.attn_q_norm.weight",
        strix::TensorRole::normalization,
        {model.head_dim},
        q_norm);
    add_tensor(
        "blk.3.attn_k_norm.weight",
        strix::TensorRole::normalization,
        {model.head_dim},
        k_norm);
    add_tensor(
        "blk.3.indexer.q_proj.weight",
        strix::TensorRole::sparse_index,
        {hidden_width, index_q_width},
        index_q);
    add_tensor(
        "blk.3.indexer.k_proj.weight",
        strix::TensorRole::sparse_index,
        {hidden_width, index_dim},
        index_k);
    add_tensor(
        "blk.3.indexer.q_norm.weight",
        strix::TensorRole::sparse_index,
        {model.indexer_head_dim},
        index_q_norm);
    add_tensor(
        "blk.3.indexer.k_norm.weight",
        strix::TensorRole::sparse_index,
        {model.indexer_head_dim},
        index_k_norm);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    const std::vector<
        const std::vector<float>*>
        host_weights{
            &q, &k, &v, &o,
            &q_norm, &k_norm,
            &index_q, &index_k,
            &index_q_norm,
            &index_k_norm,
        };
    std::vector<void*> device_weights(
        host_weights.size(),
        nullptr);

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;

    for (std::size_t index = 0;
         index < host_weights.size();
         ++index) {
        const auto bytes =
            host_weights[index]->size() *
            sizeof(float);
        check(
            hipMalloc(
                &device_weights[index],
                bytes),
            "hipMalloc indexed weight");
        auto serialized = *host_weights[index];
        if (manifest.tensors[index].name.find("norm.weight") != std::string::npos)
            for (auto& value : serialized) value += 1.0F;
        check(
            hipMemcpy(
                device_weights[index],
                serialized.data(),
                bytes,
                hipMemcpyHostToDevice),
            "copy indexed weight");
        runtime.tensors.push_back(
            {
                static_cast<std::uint32_t>(
                    index),
                strix::RuntimeTensorStorage::
                    resident,
                reinterpret_cast<
                    std::uintptr_t>(
                    device_weights[index]),
                bytes,
                1,
                0,
            });
    }

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x494e445854455355ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {0, 1, 2, 3},
        });
    access.accesses.push_back(
        {
            step.id,
            step.op,
            1,
            {6, 7, 8, 9},
        });
    access.accesses.push_back(
        {
            step.id,
            step.op,
            2,
            {4, 5},
        });

    const auto state_contract =
        strix::build_indexed_attention_state_layout(
            model);
    constexpr std::uint64_t context_tokens = 8;
    const auto kv_bytes =
        context_tokens *
        state_contract.kv_row_bytes;
    const auto raw_bytes =
        context_tokens *
        state_contract.raw_index_row_bytes;

    strix::SerialSessionStateLayout state_layout;
    state_layout.context_tokens =
        context_tokens;
    state_layout.device_alignment = 256;
    state_layout.host_alignment = 64;
    state_layout.device_total_bytes =
        kv_bytes + raw_bytes;
    state_layout.host_total_bytes = 0;
    state_layout.layout_identity =
        0x494e445853544154ULL;
    state_layout.allocations.push_back(
        {
            strix::SessionStateComponent::
                attention_kv,
            strix::PersistentState::
                attention_kv,
            strix::SessionStateStorage::
                device,
            0,
            kv_bytes,
            false,
        });
    state_layout.allocations.push_back(
        {
            strix::SessionStateComponent::
                sparse_index,
            strix::PersistentState::
                sparse_index,
            strix::SessionStateStorage::
                device,
            kv_bytes,
            raw_bytes,
            false,
        });

    void* device_state = nullptr;
    check(
        hipMalloc(
            &device_state,
            static_cast<std::size_t>(
                state_layout.device_total_bytes)),
        "hipMalloc indexed state");
    check(
        hipMemset(
            device_state,
            0,
            static_cast<std::size_t>(
                state_layout.device_total_bytes)),
        "clear indexed state");

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        reinterpret_cast<std::uintptr_t>(
            device_state);
    state_memory.device_bytes =
        state_layout.device_total_bytes;
    state_memory.device_allocation_generation = 1;

    strix::ExecutionRuntimeResources resources;
    resources.scratch = &scratch;
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &runtime;
    resources.tensor_access =
        &access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &state_memory;

    strix::ExecutionDispatchContext context;
    context.step_id = step.id;
    context.op = step.op;
    context.layer = step.layer;
    context.reads = step_io.reads;
    context.writes = step_io.writes;
    context.temporaries =
        step_io.temporaries;
    context.arena = &arena;
    context.memory = &scratch;
    context.resources = &resources;
    context.session_state_layout =
        &state_layout;
    context.session_state_memory =
        &state_memory;
    context.logical_rows = 1;
    context.request_concurrency = 1;
    context.position = 0;

    strix::HipIndexedAttentionLauncherConfig config;
    config.model = model;
    config.activation =
        strix::HipActivationStorage::bf16;

    auto launcher =
        strix::make_hip_indexed_attention_launcher(
            "test.indexed_attention",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "indexed-attention synchronize");

    std::vector<std::uint16_t> actual(
        hidden_width);
    check(
        hipMemcpy(
            actual.data(),
            reinterpret_cast<
                const void*>(
                    output.address),
            actual.size() *
                sizeof(std::uint16_t),
            hipMemcpyDeviceToHost),
        "copy indexed output");

    float maximum_error = 0.0F;
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    bf16_to_float(
                        actual[index]) -
                    expected_output[index]));
    }
    expect(
        maximum_error <= 4e-3F,
        "position-zero indexed attention matches CPU reference");

    std::vector<std::uint16_t> state_values(
        static_cast<std::size_t>(
            state_layout.device_total_bytes /
            sizeof(std::uint16_t)));
    check(
        hipMemcpy(
            state_values.data(),
            device_state,
            static_cast<std::size_t>(
                state_layout.device_total_bytes),
            hipMemcpyDeviceToHost),
        "copy indexed state");

    const auto* kv_state =
        state_values.data();
    const auto* raw_state =
        reinterpret_cast<
            const std::uint16_t*>(
                reinterpret_cast<
                    const std::byte*>(
                        state_values.data()) +
                kv_bytes);

    float kv_error = 0.0F;
    for (std::uint32_t feature = 0;
         feature < kv_width;
         ++feature) {
        kv_error =
            std::max(
                kv_error,
                std::abs(
                    bf16_to_float(
                        kv_state[feature]) -
                    k_expected[feature]));
        kv_error =
            std::max(
                kv_error,
                std::abs(
                    bf16_to_float(
                        kv_state[
                            kv_width +
                            feature]) -
                    v_expected[feature]));
    }
    expect(
        kv_error <= 3e-3F,
        "indexed attention persists normalized K and projected V");

    float raw_error = 0.0F;
    for (std::uint32_t feature = 0;
         feature < index_dim;
         ++feature) {
        raw_error =
            std::max(
                raw_error,
                std::abs(
                    bf16_to_float(
                        raw_state[feature]) -
                    index_k_expected[
                        feature]));
    }
    expect(
        raw_error <= 2e-3F,
        "indexed attention persists raw index key before norm/RoPE");


    // Capture every indexed-attention kernel, then replay across partial-tail,
    // full-visible and top-k boundaries without changing the graph topology.
    {
        auto graph_config = config;
        graph_config.enable_device_graph_inputs = true;
        auto graph_launcher = strix::make_hip_indexed_attention_launcher("test.indexed.device", &graph_config);
        expect(graph_launcher.implementation_identity != launcher.implementation_identity,
            "indexed device-input capability is bound to launcher identity");
        struct GraphHandles {
            hipStream_t stream = nullptr;
            hipGraph_t graph = nullptr;
            hipGraphExec_t executable = nullptr;
            void* arguments = nullptr;
            void* status = nullptr;
            ~GraphHandles() {
                if (stream && hipStreamQuery(stream) != hipSuccess) return;
                if (executable) static_cast<void>(hipGraphExecDestroy(executable));
                if (graph) static_cast<void>(hipGraphDestroy(graph));
                if (arguments) static_cast<void>(hipFree(arguments));
                if (status) static_cast<void>(hipFree(status));
                if (stream) static_cast<void>(hipStreamDestroy(stream));
            }
        } graph;
        check(hipStreamCreateWithFlags(&graph.stream, hipStreamNonBlocking), "create indexed graph stream");
        check(hipMalloc(&graph.arguments, 16), "allocate indexed arguments");
        check(hipMalloc(&graph.status, 4), "allocate indexed status");
        context.execution_identity = scratch.execution_identity;
        context.position = std::numeric_limits<std::uint64_t>::max();
        strix::HipLaunchContext scope;
        scope.stream = graph.stream; scope.graph_capture = true;
        scope.arguments = strix::bind_graph_inputs(strix::graph_input_layout(1),
            {scratch.execution_identity, reinterpret_cast<std::uintptr_t>(graph.arguments), 16, 1});
        scope.status = {scratch.execution_identity, reinterpret_cast<std::uintptr_t>(graph.status), 4, 1};
        scope.context_tokens = context_tokens;
        scope.vocabulary_size = static_cast<std::uint32_t>(model.vocab_size);
        check(hipStreamBeginCapture(graph.stream, hipStreamCaptureModeThreadLocal), "capture complete indexed attention");
        try {
            const strix::HipLaunchScope owned(scope);
            graph_launcher.function(context, graph_launcher.user_data);
        } catch (...) { static_cast<void>(hipStreamEndCapture(graph.stream, &graph.graph)); throw; }
        check(hipStreamEndCapture(graph.stream, &graph.graph), "finish indexed capture");
        check(hipGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0), "instantiate indexed graph");
        auto set_state = [&](const std::vector<std::uint16_t>& state) {
            check(hipMemcpy(device_state, state.data(), state_layout.device_total_bytes, hipMemcpyHostToDevice), "set indexed graph state");
        };
        auto read_state = [&]() {
            std::vector<std::uint16_t> state(state_layout.device_total_bytes / 2);
            check(hipMemcpy(state.data(), device_state, state_layout.device_total_bytes, hipMemcpyDeviceToHost), "read indexed graph state");
            return state;
        };
        auto control = [&](std::uint64_t position, std::uint32_t status) {
            std::array<std::byte, 16> packet{};
            std::memcpy(packet.data() + 8, &position, 8);
            check(hipMemcpy(graph.arguments, packet.data(), 16, hipMemcpyHostToDevice), "set indexed device position");
            check(hipMemcpy(graph.status, &status, 4, hipMemcpyHostToDevice), "set indexed sticky status");
        };
        auto replay = [&]() {
            check(hipGraphLaunch(graph.executable, graph.stream), "replay complete indexed attention");
            check(hipStreamSynchronize(graph.stream), "complete indexed graph");
        };
        auto read_status = [&]() {
            std::uint32_t result = 0;
            check(hipMemcpy(&result, graph.status, 4, hipMemcpyDeviceToHost), "read indexed status");
            return result;
        };
        std::vector<std::uint16_t> previous(state_layout.device_total_bytes / 2, 0), prior_output;
        set_state(previous);
        bool changed_output = false;
        for (std::uint64_t position = 0; position < context_tokens; ++position) {
            for (unsigned i = 0; i < hidden.size(); ++i) hidden[i] = float_to_bf16(
                static_cast<float>(static_cast<int>((i + position * 3U) % 19U) - 9) / 23.0F);
            check(hipMemcpy(reinterpret_cast<void*>(input.address), hidden.data(), hidden.size() * 2, hipMemcpyHostToDevice), "change indexed graph input");
            context.position = std::numeric_limits<std::uint64_t>::max();
            control(position, 0); replay();
            expect(read_status() == 0, "valid complete indexed graph status");
            const auto current = read_state();
            std::vector<std::uint16_t> graph_output(hidden_width), eager_output(hidden_width);
            check(hipMemcpy(graph_output.data(), reinterpret_cast<void*>(output.address), hidden_width * 2, hipMemcpyDeviceToHost), "read indexed graph output");
            set_state(previous);
            context.position = position;
            auto eager_scope = scope; eager_scope.graph_capture = false;
            { const strix::HipLaunchScope owned(eager_scope); graph_launcher.function(context, graph_launcher.user_data); }
            check(hipStreamSynchronize(graph.stream), "complete indexed eager control");
            check(hipMemcpy(eager_output.data(), reinterpret_cast<void*>(output.address), hidden_width * 2, hipMemcpyDeviceToHost), "read indexed eager output");
            expect(current == read_state() && graph_output == eager_output,
                "indexed graph exactly matches eager output/KV/raw-index across selection boundaries");
            for (const auto value : graph_output) expect(std::isfinite(bf16_to_float(value)), "finite indexed replay output");
            if (!prior_output.empty() && graph_output != prior_output) changed_output = true;
            prior_output = graph_output;
            previous = current; set_state(previous);
        }
        expect(changed_output, "indexed replay fixture is sensitive to changing input/state");
        context.position = std::numeric_limits<std::uint64_t>::max();
        for (const auto position : {context_tokens, std::uint64_t{1} << 32U, std::numeric_limits<std::uint64_t>::max()}) {
            control(position, 0); replay();
            expect(read_status() == 2 && read_state() == previous,
                "invalid indexed device position preserves entire KV/raw-index state");
        }
        control(0, 1); replay();
        expect(read_status() == 1 && read_state() == previous, "indexed upstream status stays sticky without state mutation");
        hidden[0] = 0x7f80;
        check(hipMemcpy(reinterpret_cast<void*>(input.address), hidden.data(), hidden.size() * 2, hipMemcpyHostToDevice), "set nonfinite indexed input");
        control(0, 0); replay();
        expect(read_status() == 4 && read_state() == previous, "nonfinite indexed input fails before state writes");
    }


    // Full operator: changing finite inputs, original projection versus prefetch,
    // independent poisoned destinations, actual eager and captured dispatches.
    {
        struct Q8Block {std::uint16_t scale; std::int8_t values[32];};
        static_assert(sizeof(Q8Block)==34);
        const std::array<unsigned,5> projection_indices{0,1,2,6,7};
        const auto temporary = strix::resolve_execution_buffer(scratch.execution_identity,
            arena,scratch,step_io.temporaries[0]);
        hipStream_t stream=nullptr;
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create QSA comparison stream");
        void* arguments=nullptr; void* status=nullptr;
        check(hipMalloc(&arguments,16),"QSA comparison arguments");
        check(hipMalloc(&status,4),"QSA comparison status");
        context.execution_identity=scratch.execution_identity;
        strix::HipLaunchContext scope;
        scope.stream=stream;
        scope.arguments=strix::bind_graph_inputs(strix::graph_input_layout(1),
            {scratch.execution_identity,reinterpret_cast<std::uintptr_t>(arguments),16,1});
        scope.status={scratch.execution_identity,reinterpret_cast<std::uintptr_t>(status),4,1};
        scope.context_tokens=context_tokens;
        scope.vocabulary_size=static_cast<std::uint32_t>(model.vocab_size);
        const auto read_bytes=[&](std::uintptr_t address,std::uint64_t bytes){
            std::vector<std::byte> result(bytes);
            check(hipMemcpy(result.data(),reinterpret_cast<void*>(address),bytes,hipMemcpyDeviceToHost),
                  "read full QSA comparison bytes");return result;};
        const auto reset=[&](unsigned replay){
            check(hipMemset(reinterpret_cast<void*>(output.address),0xa5,output.bytes),"poison QSA output");
            check(hipMemset(reinterpret_cast<void*>(temporary.address),0xff,temporary.bytes),"poison QSA workspace");
            check(hipMemset(device_state,0xa5,state_layout.device_total_bytes),"poison QSA KV/raw");
            for(unsigned f=0;f<hidden.size();++f)hidden[f]=float_to_bf16(
                static_cast<float>(static_cast<int>((f+replay*7U)%23U)-11)/29.0F);
            check(hipMemcpy(reinterpret_cast<void*>(input.address),hidden.data(),hidden.size()*2,
                            hipMemcpyHostToDevice),"set QSA comparison input");
            std::array<std::byte,16> packet{};std::uint32_t zero=0;
            check(hipMemcpy(arguments,packet.data(),16,hipMemcpyHostToDevice),"set QSA comparison position");
            check(hipMemcpy(status,&zero,4,hipMemcpyHostToDevice),"clear QSA comparison status");
        };
        for(unsigned mixed=0;mixed<2;++mixed){
            for(const auto index:projection_indices){
                auto& tensor=manifest.tensors[index];
                const auto elements=tensor.dimensions[0]*tensor.dimensions[1];
                tensor.ggml_type=8;
                std::vector<Q8Block> blocks(elements/32);
                for(std::size_t b=0;b<blocks.size();++b){blocks[b].scale=0x2400;
                    for(unsigned f=0;f<32;++f)blocks[b].values[f]=static_cast<std::int8_t>(
                        static_cast<int>((b*5+f*3+index)%31U)-15);}
                tensor.byte_size=blocks.size()*sizeof(Q8Block);
                check(hipMemcpy(device_weights[index],blocks.data(),tensor.byte_size,hipMemcpyHostToDevice),
                      "set full QSA Q8 projection");
                if(mixed && index==7){
                    tensor.ggml_type=30;
                    std::vector<std::uint16_t> bf16(elements);
                    for(std::size_t f=0;f<bf16.size();++f)
                        bf16[f]=float_to_bf16(static_cast<float>(blocks[f/32].values[f%32])/64.0F);
                    tensor.byte_size=bf16.size()*2;
                    check(hipMemcpy(device_weights[index],bf16.data(),tensor.byte_size,hipMemcpyHostToDevice),
                          "set full QSA BF16 projection fallback");
                }
                runtime.tensors[index].bytes=tensor.byte_size;
            }
            manifest.layout_identity=strix::model_manifest_layout_identity(manifest);
            runtime.manifest_layout_identity=manifest.layout_identity;
            access.manifest_layout_identity=manifest.layout_identity;
            auto off_config=config;off_config.enable_device_graph_inputs=true;
            auto on_config=off_config;on_config.prefetch_q8_input=true;
            on_config.prefetch_eligible_layers={layer};
            on_config.prefetch_stats=std::make_shared<strix::qsa_input_prefetch::RuntimeDispatchStats>();
            auto off_launch=strix::make_hip_indexed_attention_launcher("test.full.off",&off_config);
            auto on_launch=strix::make_hip_indexed_attention_launcher("test.full.on",&on_config);
            hipGraph_t graphs[2]{};hipGraphExec_t executables[2]{};
            for(unsigned mode=0;mode<2;++mode){
                auto& candidate=mode?on_launch:off_launch;
                scope.graph_capture=true;context.position=std::numeric_limits<std::uint64_t>::max();
                const auto before=on_config.prefetch_stats->dispatched_projections.load();
                check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture QSA ON/OFF");
                try{const strix::HipLaunchScope owned(scope);candidate.function(context,candidate.user_data);}
                catch(...){static_cast<void>(hipStreamEndCapture(stream,&graphs[mode]));throw;}
                check(hipStreamEndCapture(stream,&graphs[mode]),"finish QSA ON/OFF capture");
                std::size_t nodes_count=0;check(hipGraphGetNodes(graphs[mode],nullptr,&nodes_count),"QSA node count");
                std::vector<hipGraphNode_t> nodes(nodes_count);
                check(hipGraphGetNodes(graphs[mode],nodes.data(),&nodes_count),"QSA graph nodes");
                unsigned wide_nodes=0;
                for(auto node:nodes){hipGraphNodeType type;
                    check(hipGraphNodeGetType(node,&type),"QSA node type");
                    if(type==hipGraphNodeTypeKernel){hipKernelNodeParams params{};
                        check(hipGraphKernelNodeGetParams(node,&params),"QSA kernel node params");
                        if(params.blockDim.x==32)++wide_nodes;}}
                expect(wide_nodes==(mode?5U-mixed:0U),"full QSA graph has exact candidate projection nodes");
                expect(on_config.prefetch_stats->dispatched_projections.load()-before==(mode?5U-mixed:0U),
                       "full QSA capture counter proves actual candidate submission");
                check(hipGraphInstantiate(&executables[mode],graphs[mode],nullptr,nullptr,0),"instantiate QSA comparison");
            }
            for(unsigned replay=0;replay<3;++replay){
                std::vector<std::byte> expected_replay_output,expected_state,expected_workspace;
                for(unsigned path=0;path<4;++path){
                    reset(replay);scope.graph_capture=false;context.position=0;
                    if(path<2){auto& candidate=path?on_launch:off_launch;
                        const auto before=on_config.prefetch_stats->dispatched_projections.load();
                        {const strix::HipLaunchScope owned(scope);candidate.function(context,candidate.user_data);}
                        expect(on_config.prefetch_stats->dispatched_projections.load()-before==(path?5U-mixed:0U),
                               "full QSA eager counter matches eligible projections");
                    }else{const auto before=on_config.prefetch_stats->dispatched_projections.load();
                        check(hipGraphLaunch(executables[path-2],stream),"replay QSA comparison");
                        expect(on_config.prefetch_stats->dispatched_projections.load()==before,
                               "QSA device replay does not inflate host capture count");}
                    check(hipStreamSynchronize(stream),"complete QSA comparison");
                    std::uint32_t actual_status=0;
                    check(hipMemcpy(&actual_status,status,4,hipMemcpyDeviceToHost),"read QSA comparison status");
                    expect(actual_status==0,"full QSA candidate completed with valid graph status");
                    std::vector<std::uint16_t> finite_output(hidden_width);
                    check(hipMemcpy(finite_output.data(),reinterpret_cast<void*>(output.address),
                        hidden_width*2,hipMemcpyDeviceToHost),"read finite QSA comparison output");
                    for(auto value:finite_output)expect(std::isfinite(bf16_to_float(value)),
                        "full QSA comparison output is finite");
                    const auto actual_output=read_bytes(output.address,output.bytes);
                    const auto actual_state=read_bytes(state_memory.device_base,state_memory.device_bytes);
                    const auto actual_workspace=read_bytes(temporary.address,temporary.bytes);
                    if(path==0){expected_replay_output=actual_output;expected_state=actual_state;expected_workspace=actual_workspace;}
                    else expect(actual_output==expected_replay_output && actual_state==expected_state &&
                        actual_workspace==expected_workspace,"full QSA ON/OFF eager/graph output/KV/raw/workspace bit equality");
                }
            }
            // Admission failures must enqueue nothing and leave every owner byte intact.
            for(unsigned failure=0;failure<2;++failure){
                reset(0);
                const auto old_address=runtime.tensors[0].address;
                auto bad_arena=arena;
                if(failure==0){
                    const auto* temp_allocation=strix::find_arena_allocation(arena,step_io.temporaries[0]);
                    for(auto& allocation:bad_arena.allocations)
                        if(allocation.use==step_io.writes[0])allocation.offset=temp_allocation->offset;
                    context.arena=&bad_arena;
                }else runtime.tensors[0].address+=2;
                const auto before_arena=read_bytes(scratch.arena_base,scratch.arena_bytes);
                const auto before_state=read_bytes(state_memory.device_base,state_memory.device_bytes);
                const auto before_count=on_config.prefetch_stats->dispatched_projections.load();
                hipGraph_t rejected_graph=nullptr;bool rejected=false;
                scope.graph_capture=true;context.position=std::numeric_limits<std::uint64_t>::max();
                check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture rejected QSA preflight");
                try{const strix::HipLaunchScope owned(scope);on_launch.function(context,on_launch.user_data);}
                catch(const std::invalid_argument& error){rejected=std::string(error.what()).find(failure?
                    "weight alignment/extent invalid":"owner overlap")!=std::string::npos;}
                check(hipStreamEndCapture(stream,&rejected_graph),"end rejected QSA preflight");
                std::size_t nodes=1;check(hipGraphGetNodes(rejected_graph,nullptr,&nodes),"rejected QSA nodes");
                expect(rejected && nodes==0,"QSA preflight rejects precise cause with zero queued nodes");
                expect(before_count==on_config.prefetch_stats->dispatched_projections.load() &&
                    before_arena==read_bytes(scratch.arena_base,scratch.arena_bytes) &&
                    before_state==read_bytes(state_memory.device_base,state_memory.device_bytes),
                    "QSA preflight failure preserves all owners and dispatch count");
                check(hipGraphDestroy(rejected_graph),"destroy rejected QSA graph");
                runtime.tensors[0].address=old_address;context.arena=&arena;
            }
            for(unsigned mode=0;mode<2;++mode){
                check(hipGraphExecDestroy(executables[mode]),"destroy QSA executable");
                check(hipGraphDestroy(graphs[mode]),"destroy QSA graph");}

            // Production output opt-in: independent of input prefetch, with
            // the same real post-attention FP32 source and original Q8 rows.
            // Restore all arena/state bytes and replay positions0..2 per arm.
            const auto original_out = manifest.tensors[3];
            const auto original_out_bytes = runtime.tensors[3].bytes;
            std::vector<Q8Block> out_blocks(static_cast<std::size_t>(attention_width) * hidden_width / 32U);
            for (unsigned edge = 0; edge < 3; ++edge) {
                constexpr std::array<std::uint16_t,8> ordinary{0x0001,0x03ff,0x0400,0x2401,0x3c01,0x8000,0x8400,0xb801};
                constexpr std::array<std::uint16_t,8> nonfinite{0x7e01,0x7c01,0xfe23,0x7c00,0xfc00,0,0x8000,0x3c00};
                for (std::size_t b = 0; b < out_blocks.size(); ++b) {
                    out_blocks[b].scale = edge == 2 ? nonfinite[b % nonfinite.size()] :
                        edge == 1 ? ordinary[b % ordinary.size()] : 0x2400U;
                    for (unsigned f = 0; f < 32; ++f)
                        out_blocks[b].values[f] = static_cast<std::int8_t>(
                            static_cast<int>((b * 11U + f * 7U) % 256U) - 128);
                }
                manifest.tensors[3].ggml_type = 8;
                manifest.tensors[3].byte_size = out_blocks.size() * sizeof(Q8Block);
                runtime.tensors[3].bytes = manifest.tensors[3].byte_size;
                check(hipMemcpy(device_weights[3],out_blocks.data(),manifest.tensors[3].byte_size,
                    hipMemcpyHostToDevice),"set production QSA output Q8 edges");
                manifest.layout_identity=strix::model_manifest_layout_identity(manifest);
                runtime.manifest_layout_identity=manifest.layout_identity;
                access.manifest_layout_identity=manifest.layout_identity;
                for (unsigned input_prefetch = 0; input_prefetch < 2; ++input_prefetch) {
                    auto old_output = config;
                    old_output.prefetch_q8_input = input_prefetch != 0;
                    if (input_prefetch) old_output.prefetch_eligible_layers = {layer};
                    auto new_output = old_output; new_output.wave8_q8_output = true;
                    strix::ExecutionBindingRequest selected;
                    selected.minimum_qualification = strix::kernels::QualificationStatus::experimental;
                    selected.prefetch_q8_qsa_input = old_output.prefetch_q8_input;
                    selected.qsa_input_prefetch_layers = old_output.prefetch_eligible_layers;
                    strix::kernels::KernelRegistry old_registry,new_registry;
                    strix::register_hip_indexed_attention_kernel_variant(model,selected,old_output,old_registry);
                    selected.wave8_q8_qsa_output = true;
                    strix::register_hip_indexed_attention_kernel_variant(model,selected,new_output,new_registry);
                    const auto old_variant = old_registry.variants().front();
                    const auto new_variant = new_registry.variants().front();
                    expect(old_variant.id != new_variant.id && old_variant.compiler_identity != new_variant.compiler_identity,
                        "output opt-in changes selected kernel/compiler identities");
                    auto old_production = strix::make_hip_indexed_attention_launcher(old_variant.id,&old_output);
                    auto new_production = strix::make_hip_indexed_attention_launcher(new_variant.id,&new_output);
                    expect(old_production.implementation_identity != new_production.implementation_identity,
                        "output opt-in changes actual launcher identity");
                    for (unsigned repeat = 0; repeat < 3; ++repeat) {
                        std::array<std::vector<std::byte>,3> expected_arena, expected_persistent;
                        for (unsigned arm = 0; arm < 2; ++arm) {
                            check(hipMemset(arena_pointer,0xa5,arena.total_bytes),"reset complete QSA arena");
                            reset(repeat);
                            scope.graph_capture = false;
                            for (unsigned position = 0; position < 3; ++position) {
                                context.position = position;
                                auto& production = arm ? new_production : old_production;
                                { const strix::HipLaunchScope owned(scope); production.function(context,production.user_data); }
                                check(hipStreamSynchronize(stream),"finish production QSA output comparison");
                                auto actual_arena=read_bytes(scratch.arena_base,scratch.arena_bytes);
                                auto actual_persistent=read_bytes(state_memory.device_base,state_memory.device_bytes);
                                if (!arm) {
                                    expected_arena[position]=std::move(actual_arena);
                                    expected_persistent[position]=std::move(actual_persistent);
                                } else expect(actual_arena == expected_arena[position] &&
                                    actual_persistent == expected_persistent[position],
                                    "production output wave8/input modes preserve full arena and KV/raw bits");
                            }
                        }
                    }
                    // Both a marked scope and an actual stream capture with an
                    // eager scope must reject before any production kernel.
                    for (const bool marked : {false,true}) {
                        reset(0); context.position=0; scope.graph_capture=marked;
                        const auto before_arena=read_bytes(scratch.arena_base,scratch.arena_bytes);
                        const auto before_state=read_bytes(state_memory.device_base,state_memory.device_bytes);
                        hipGraph_t rejected_graph=nullptr; bool rejected=false;
                        check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture output admission rejection");
                        try { const strix::HipLaunchScope owned(scope); new_production.function(context,new_production.user_data); }
                        catch (const std::invalid_argument& error) {
                            rejected=std::string(error.what())=="wave8 QSA output supports eager execution only";
                        }
                        check(hipStreamEndCapture(stream,&rejected_graph),"end output admission rejection");
                        std::size_t nodes=1;check(hipGraphGetNodes(rejected_graph,nullptr,&nodes),"output rejection graph nodes");
                        expect(rejected && nodes==0 && before_arena==read_bytes(scratch.arena_base,scratch.arena_bytes) &&
                            before_state==read_bytes(state_memory.device_base,state_memory.device_bytes),
                            "output capture admission rejects without enqueue or owner mutation");
                        check(hipGraphDestroy(rejected_graph),"destroy output rejection graph");
                    }
                    scope.graph_capture=false;
                }
            }
            manifest.tensors[3]=original_out;runtime.tensors[3].bytes=original_out_bytes;
            check(hipMemcpy(device_weights[3],o.data(),o.size()*sizeof(float),hipMemcpyHostToDevice),
                "restore original QSA output weights");
            manifest.layout_identity=strix::model_manifest_layout_identity(manifest);
            runtime.manifest_layout_identity=manifest.layout_identity;access.manifest_layout_identity=manifest.layout_identity;
        }
        check(hipFree(arguments),"free QSA comparison arguments");
        check(hipFree(status),"free QSA comparison status");
        check(hipStreamDestroy(stream),"destroy QSA comparison stream");
    }

    // Production hierarchical selection: reuse the actual complete QSA
    // launcher, weights and resource descriptors, with independently reset
    // history for each arm. No benchmark or whole-model claim is made here.
    {
        struct Owned {
            strix::HipSetupQuiescenceGuard guard;
            hipStream_t stream = nullptr;
            void* arena = nullptr; void* state = nullptr; void* error = nullptr;
            ~Owned() {
                guard.recover_or_fail_stop();
                if (arena) (void)hipFree(arena);
                if (state) (void)hipFree(state);
                if (error) (void)hipFree(error);
                if (stream) (void)hipStreamDestroy(stream);
            }
        } own;
        auto large_model = model;
        large_model.indexer_budget = 2048; large_model.indexer_compress_ratio = 4;
        large_model.max_position_embeddings = 2056;
        strix::ExecutionBufferRequest large_buffers;
        large_buffers.qsa_hierarchical_selection = true;
        large_buffers.qsa_gqa_heads_per_block = 4;
        const auto large_arena = strix::build_serial_execution_arena_plan(large_model, large_buffers, execution, io, 256);
        strix::ExecutionMemoryBinding large_scratch{0x51534153454c7631ULL, 0, large_arena.total_bytes, 3};
        const auto contract = strix::build_indexed_attention_state_layout(large_model);
        const auto history_kv_bytes = large_model.max_position_embeddings * contract.kv_row_bytes;
        const auto history_raw_bytes = large_model.max_position_embeddings * contract.raw_index_row_bytes;
        auto history_layout = state_layout;
        history_layout.context_tokens = large_model.max_position_embeddings;
        history_layout.device_total_bytes = history_kv_bytes + history_raw_bytes;
        history_layout.layout_identity = 0x51534153454c5354ULL;
        history_layout.allocations[0].bytes = history_kv_bytes;
        history_layout.allocations[1].offset = history_kv_bytes;
        history_layout.allocations[1].bytes = history_raw_bytes;
        own.guard.run([&] {
            check(hipStreamCreateWithFlags(&own.stream, hipStreamNonBlocking), "create hierarchical QSA stream");
            check(hipMalloc(&own.arena, static_cast<std::size_t>(large_scratch.arena_bytes)), "allocate hierarchical arena");
            check(hipMalloc(&own.state, static_cast<std::size_t>(history_layout.device_total_bytes)), "allocate hierarchical history");
            check(hipMalloc(&own.error, sizeof(strix::DeferredMoeFirstError)), "allocate hierarchical D1 record");
        });
        large_scratch.arena_base = reinterpret_cast<std::uintptr_t>(own.arena);
        auto history_memory = state_memory;
        history_memory.layout_identity = history_layout.layout_identity;
        history_memory.device_base = reinterpret_cast<std::uintptr_t>(own.state);
        history_memory.device_bytes = history_layout.device_total_bytes;
        history_memory.device_allocation_generation = 4;
        auto large_resources = resources;
        large_resources.scratch = &large_scratch;
        large_resources.session_state_layout = &history_layout;
        large_resources.session_state_memory = &history_memory;
        auto large_context = context;
        large_context.execution_identity = large_scratch.execution_identity;
        large_context.arena = &large_arena; large_context.memory = &large_scratch;
        large_context.resources = &large_resources;
        large_context.session_state_layout = &history_layout;
        large_context.session_state_memory = &history_memory;
        const auto large_input = strix::resolve_execution_buffer(large_scratch.execution_identity,
            large_arena, large_scratch, step_io.reads[0]);
        const auto large_output = strix::resolve_execution_buffer(large_scratch.execution_identity,
            large_arena, large_scratch, step_io.writes[0]);
        const auto large_temp = strix::resolve_execution_buffer(large_scratch.execution_identity,
            large_arena, large_scratch, step_io.temporaries[0]);
        const auto selected_layout = strix::build_indexed_attention_workspace_layout(large_model, 256, true);
        strix::HipLaunchContext large_scope;
        large_scope.stream = own.stream;
        large_scope.deferred_moe_status = {large_scratch.execution_identity,
            reinterpret_cast<std::uintptr_t>(own.error), 8, 5};
        auto candidate_config = config;
        candidate_config.model = large_model;
        candidate_config.qsa_hierarchical_selection = true;
        candidate_config.selection_stats = std::make_shared<strix::QsaHierarchicalSelectionStats>();
        auto reference_config = candidate_config;
        reference_config.qsa_hierarchical_selection = false;
        strix::ExecutionBindingRequest selected_request;
        selected_request.deferred_moe_status = true;
        selected_request.qsa_hierarchical_selection = true;
        strix::kernels::KernelRegistry selected_kernels;
        strix::register_hip_indexed_attention_kernel_variant(large_model, selected_request, candidate_config, selected_kernels);
        auto candidate = strix::make_hip_indexed_attention_launcher(selected_kernels.variants().front().id, &candidate_config);
        auto reference = strix::make_hip_indexed_attention_launcher("test.hierarchical.reference", &reference_config);
        const auto read = [&](std::uintptr_t address, std::uint64_t bytes) {
            std::vector<std::byte> result(static_cast<std::size_t>(bytes));
            own.guard.run([&] { check(hipMemcpy(result.data(), reinterpret_cast<const void*>(address),
                result.size(), hipMemcpyDeviceToHost), "read hierarchical payload"); });
            return result;
        };
        std::vector<std::uint16_t> history(static_cast<std::size_t>(history_layout.device_total_bytes / 2U));
        const auto reset = [&](unsigned flavor, bool nonfinite, strix::DeferredMoeFirstError prior = {}) {
            for (std::size_t i = 0; i < history.size(); ++i)
                history[i] = flavor ? float_to_bf16(static_cast<float>(static_cast<int>(i % 13U) - 6) / 128.0F) : 0;
            if (nonfinite) history[static_cast<std::size_t>(history_kv_bytes / 2U)] = 0x7fc1U;
            std::vector<std::uint16_t> current(hidden_width);
            for (std::size_t i = 0; i < current.size(); ++i)
                current[i] = float_to_bf16(static_cast<float>(static_cast<int>(i % 11U) - 5) / 32.0F);
            own.guard.run([&] {
                check(hipMemcpy(own.state, history.data(), static_cast<std::size_t>(history_layout.device_total_bytes), hipMemcpyHostToDevice), "reset hierarchical history");
                check(hipMemset(own.arena, 0xa5, static_cast<std::size_t>(large_scratch.arena_bytes)), "poison hierarchical arena");
                check(hipMemcpy(reinterpret_cast<void*>(large_input.address), current.data(), current.size() * 2U, hipMemcpyHostToDevice), "reset hierarchical input");
                check(hipMemcpy(own.error, &prior, sizeof(prior), hipMemcpyHostToDevice), "reset caller-owned D1 between independent fixture arms");
            });
        };
        const auto run = [&](const strix::ExecutionLauncher& selected) {
            own.guard.run([&] { const strix::HipLaunchScope scoped(large_scope);
                selected.function(large_context, selected.user_data); });
        };
        std::vector<std::vector<std::byte>> immutable_weights;
        for (const auto& tensor : runtime.tensors) immutable_weights.push_back(read(tensor.address, tensor.bytes));
        for (const unsigned flavor : {0U, 1U}) {
            for (const std::uint64_t position : {0ULL, 2043ULL, 2047ULL, 2048ULL, 2051ULL, 2052ULL}) {
                large_context.position = position;
                reset(flavor, false); run(reference);
                const auto expected_sparse_output = read(large_output.address, hidden_width * 2U);
                const auto expected_state = read(history_memory.device_base, history_memory.device_bytes);
                const auto visible = position + 1U;
                const auto count = std::min(visible / 4U, std::uint64_t{512}) * 4U + visible % 4U;
                const auto expected_indices = read(large_temp.address + selected_layout.selected_indices_offset, count * 4U);
                for (unsigned repeat = 0; repeat < 2; ++repeat) {
                    reset(flavor, false);
                    const auto prior = candidate_config.selection_stats->submissions.load();
                    run(candidate);
                    expect(expected_sparse_output == read(large_output.address, hidden_width * 2U) &&
                        expected_state == read(history_memory.device_base, history_memory.device_bytes) &&
                        expected_indices == read(large_temp.address + selected_layout.selected_indices_offset, count * 4U),
                        "hierarchical production preserves full attention output, state and chronological indices");
                    expect(candidate_config.selection_stats->submissions.load() - prior == (visible / 4U > 512U ? 1U : 0U),
                        "hierarchical production counter reports only actual sparse candidate submissions");
                    const strix::DeferredMoeFirstError zero{};
                    expect(read(reinterpret_cast<std::uintptr_t>(own.error), 8) ==
                        std::vector<std::byte>(reinterpret_cast<const std::byte*>(&zero), reinterpret_cast<const std::byte*>(&zero) + 8),
                        "finite hierarchical attention leaves D1 success intact");
                }
            }
        }
        // Same production projection/state setup with the shared GQA2/4
        // backend, independently of the hierarchical-selection option.
        for (const auto heads : {2U, 4U}) for (const bool hierarchical : {false, true}) {
            auto grouped_config = candidate_config;
            grouped_config.qsa_gqa_heads_per_block = heads;
            grouped_config.qsa_hierarchical_selection = hierarchical;
            grouped_config.gqa_stats = std::make_shared<strix::QsaHierarchicalSelectionStats>();
            auto grouped_request = selected_request;
            grouped_request.qsa_gqa_heads_per_block = heads;
            grouped_request.qsa_hierarchical_selection = hierarchical;
            strix::kernels::KernelRegistry grouped_kernels;
            strix::register_hip_indexed_attention_kernel_variant(large_model, grouped_request, grouped_config, grouped_kernels);
            auto grouped = strix::make_hip_indexed_attention_launcher(grouped_kernels.variants().front().id, &grouped_config);
            for (const std::uint64_t position : {0ULL, 2047ULL, 2048ULL, 2051ULL, 2052ULL}) {
                large_context.position = position;
                reset(1, false); run(reference);
                const auto expected_sparse_output = read(large_output.address, hidden_width * 2U);
                const auto expected_state = read(history_memory.device_base, history_memory.device_bytes);
                const auto visible = position + 1U;
                const auto count = std::min(visible / 4U, std::uint64_t{512}) * 4U + visible % 4U;
                const auto expected_indices = read(large_temp.address + selected_layout.selected_indices_offset, count * 4U);
                for (unsigned repeat = 0; repeat < 2; ++repeat) {
                    reset(1, false);
                    const auto before = grouped_config.gqa_stats->submissions.load();
                    run(grouped);
                    expect(expected_sparse_output == read(large_output.address, hidden_width * 2U) &&
                        expected_state == read(history_memory.device_base, history_memory.device_bytes) &&
                        expected_indices == read(large_temp.address + selected_layout.selected_indices_offset, count * 4U),
                        "production GQA2/4 preserves output/state and exact selection with either selector");
                    expect(grouped_config.gqa_stats->submissions.load() == before + 1U,
                        "production grouped-attention counter proves actual candidate submission");
                }
            }
            large_context.position = 0;
            reset(0, false);
            const std::uint16_t nonfinite = 0x7fc1U;
            own.guard.run([&] { check(hipMemcpy(reinterpret_cast<void*>(large_input.address), &nonfinite,
                sizeof(nonfinite), hipMemcpyHostToDevice), "inject grouped-attention nonfinite input"); });
            run(grouped);
            strix::DeferredMoeFirstError rejected{};
            const auto error = read(reinterpret_cast<std::uintptr_t>(own.error), 8);
            std::memcpy(&rejected, error.data(), 8);
            expect(rejected.domain == 4U && rejected.bits != 0U,
                "grouped attention rejection uses its own deferred error domain");
        }
        large_context.position = 2051;
        // Exercise both the first and last complete compressed block, so a
        // late producer rejection cannot be lost before sorting or gathering.
        // Inf and distinct NaN payloads must fail even when ReLU would mask the
        // resulting NaN dot product to zero.
        for (const std::uint64_t rejected_row : {0ULL, 2048ULL})
        for (const std::uint16_t rejected_bits : {std::uint16_t{0x7fc1U}, std::uint16_t{0x7f80U}, std::uint16_t{0xff80U}})
        for (const auto prior : {strix::DeferredMoeFirstError{}, strix::DeferredMoeFirstError{1, 8}}) {
            reset(0, false, prior);
            const auto rejected_feature = rejected_row ? large_model.indexer_head_dim - 1U : 0U;
            const auto rejected_address = history_memory.device_base + history_kv_bytes +
                (rejected_row * large_model.indexer_head_dim + rejected_feature) * sizeof(std::uint16_t);
            own.guard.run([&] { check(hipMemcpy(reinterpret_cast<void*>(rejected_address), &rejected_bits,
                sizeof(rejected_bits), hipMemcpyHostToDevice), "inject nonfinite compressed selection history"); });
            run(candidate);
            strix::DeferredMoeFirstError observed{};
            const auto error_bytes = read(reinterpret_cast<std::uintptr_t>(own.error), 8);
            std::memcpy(&observed, error_bytes.data(), 8);
            expect(observed.domain == (prior.domain ? prior.domain : 3U) && observed.bits == (prior.domain ? prior.bits : 1U),
                "nonfinite QSA selection publishes its real domain while preserving an earlier first error");
            const auto count = 2048U;
            expect(read(large_temp.address + selected_layout.selected_indices_offset, count * 4U) == std::vector<std::byte>(count * 4U),
                "rejected sparse selection sanitizes indices before any attention gather");
        }
        // Independent caller reset restores finite success after a rejection;
        // the launcher's producer-local status must not leak between tokens.
        reset(1, false); run(reference);
        const auto recovered_output = read(large_output.address, hidden_width * 2U);
        const auto recovered_state = read(history_memory.device_base, history_memory.device_bytes);
        const auto recovered_indices = read(large_temp.address + selected_layout.selected_indices_offset, 2048U * 4U);
        reset(1, false); run(candidate);
        const strix::DeferredMoeFirstError recovered_error{};
        expect(recovered_output == read(large_output.address, hidden_width * 2U) &&
            recovered_state == read(history_memory.device_base, history_memory.device_bytes) &&
            recovered_indices == read(large_temp.address + selected_layout.selected_indices_offset, 2048U * 4U) &&
            read(reinterpret_cast<std::uintptr_t>(own.error), 8) == std::vector<std::byte>(
                reinterpret_cast<const std::byte*>(&recovered_error), reinterpret_cast<const std::byte*>(&recovered_error) + 8),
            "finite production selection recovers after an independently reset rejection arm");
        for (unsigned mutation = 0; mutation < 5; ++mutation) {
            reset(0, false);
            auto bad_scope = large_scope;
            if (mutation == 0) bad_scope.deferred_moe_status = {};
            if (mutation == 1) bad_scope.deferred_moe_status.allocation_generation = 0;
            if (mutation == 2) bad_scope.deferred_moe_status.arena_base = large_temp.address;
            if (mutation == 3) bad_scope.deferred_moe_status.execution_identity ^= 1U;
            const auto before_arena = read(large_scratch.arena_base, large_scratch.arena_bytes);
            const auto before_state = read(history_memory.device_base, history_memory.device_bytes);
            if (mutation < 4) {
                bool metadata_rejected = false;
                try { const strix::HipLaunchScope scoped(bad_scope); candidate.preflight(large_context, candidate.user_data); }
                catch (const std::invalid_argument&) { metadata_rejected = true; }
                expect(metadata_rejected && before_arena == read(large_scratch.arena_base, large_scratch.arena_bytes) &&
                    before_state == read(history_memory.device_base, history_memory.device_bytes),
                    "actual eager hierarchical metadata rejects without writes");
            }
            hipGraph_t graph = nullptr; bool rejected = false;
            check(hipStreamBeginCapture(own.stream, hipStreamCaptureModeThreadLocal), "capture hierarchical admission rejection");
            try { const strix::HipLaunchScope scoped(bad_scope); candidate.preflight(large_context, candidate.user_data); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(hipStreamEndCapture(own.stream, &graph), "end hierarchical admission rejection");
            std::size_t nodes = 1;
            check(hipGraphGetNodes(graph, nullptr, &nodes), "hierarchical rejection nodes");
            expect(rejected && nodes == 0 && before_arena == read(large_scratch.arena_base, large_scratch.arena_bytes) &&
                before_state == read(history_memory.device_base, history_memory.device_bytes),
                "hierarchical invalid D1/capture metadata rejects before enqueue and owner writes");
            check(hipGraphDestroy(graph), "destroy hierarchical rejected graph");
        }
        for (std::size_t i = 0; i < runtime.tensors.size(); ++i)
            expect(immutable_weights[i] == read(runtime.tensors[i].address, runtime.tensors[i].bytes),
                "production hierarchical selection preserves every weight byte");
    }

    // Compact rollback is a logical-prefix contract. Exercise the REAL QSA
    // reader after speculative writes and poison every inactive future KV row;
    // compare produced outputs/active state, never the unspecified flat tail.
    {
        struct SnapshotOwners {
            strix::HipSetupQuiescenceGuard guard;
            hipStream_t stream = nullptr;
            void* state = nullptr; void* saved = nullptr;
            ~SnapshotOwners() {
                guard.recover_or_fail_stop();
                if (saved) (void)hipFree(saved);
                if (state) (void)hipFree(state);
                if (stream) (void)hipStreamDestroy(stream);
            }
        } own;
        auto snapshot_model = model;
        snapshot_model.gdn_key_heads = snapshot_model.gdn_value_heads = 1;
        snapshot_model.gdn_key_head_dim = snapshot_model.gdn_value_head_dim = 2;
        auto snapshot_manifest = manifest;
        snapshot_manifest.total_tensor_bytes = 0;
        auto snapshot_weights = runtime;
        auto snapshot_access = access;
        for (std::size_t i = 0; i < host_weights.size(); ++i) {
            auto values = *host_weights[i];
            if (snapshot_manifest.tensors[i].name.find("norm.weight") != std::string::npos)
                for (auto& value : values) value += 1.0F;
            snapshot_manifest.tensors[i].ggml_type = 0;
            snapshot_manifest.tensors[i].byte_size = values.size() * sizeof(float);
            snapshot_manifest.total_tensor_bytes += snapshot_manifest.tensors[i].byte_size;
            snapshot_weights.tensors[i].bytes = values.size() * sizeof(float);
            own.guard.run([&] { check(hipMemcpy(device_weights[i], values.data(), values.size() * sizeof(float),
                hipMemcpyHostToDevice), "restore compact QSA F32 fixture weights"); });
        }
        snapshot_manifest.layout_identity = strix::model_manifest_layout_identity(snapshot_manifest);
        snapshot_weights.manifest_layout_identity = snapshot_manifest.layout_identity;
        snapshot_access.manifest_layout_identity = snapshot_manifest.layout_identity;
        const auto layout = strix::build_serial_session_state_layout(snapshot_model, {8,256,64,true});
        std::vector<std::byte> host_bytes(static_cast<std::size_t>(layout.host_total_bytes));
        own.guard.run([&] {
            check(hipStreamCreateWithFlags(&own.stream, hipStreamNonBlocking), "create compact QSA stream");
            check(hipMalloc(&own.state, layout.device_total_bytes), "allocate canonical compact QSA state");
            check(hipMalloc(&own.saved, layout.device_total_bytes), "allocate compact QSA snapshot capacity");
        });
        const strix::SessionStateMemoryBinding memory{layout.layout_identity,
            reinterpret_cast<std::uintptr_t>(own.state),layout.device_total_bytes,31,
            reinterpret_cast<std::uintptr_t>(host_bytes.data()),host_bytes.size(),32};
        auto snapshot_resources = resources;
        snapshot_resources.model_manifest = &snapshot_manifest; snapshot_resources.model_weights = &snapshot_weights;
        snapshot_resources.tensor_access = &snapshot_access;
        snapshot_resources.session_state_layout = &layout; snapshot_resources.session_state_memory = &memory;
        auto row_context = context;
        row_context.resources = &snapshot_resources; row_context.session_state_layout = &layout;
        row_context.session_state_memory = &memory;
        auto snapshot_config = config; snapshot_config.model = snapshot_model;
        const auto production = strix::make_hip_indexed_attention_launcher("test.compact.qsa", &snapshot_config);
        strix::HipLaunchContext scope; scope.stream = own.stream;
        const auto run = [&](std::uint64_t position, unsigned salt) {
            auto values = hidden;
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = float_to_bf16(static_cast<float>(static_cast<int>((i + salt * 3U) % 23U) - 11) / 17.0F);
            own.guard.run([&] {
                check(hipMemcpy(reinterpret_cast<void*>(input.address),values.data(),values.size()*2U,
                    hipMemcpyHostToDevice),"copy compact QSA continuation input");
                row_context.position = position;
                const strix::HipLaunchScope active(scope);
                production.function(row_context,production.user_data);
            });
        };
        const auto* kv = strix::find_session_state_allocation(layout, strix::SessionStateComponent::attention_kv);
        const auto row_bytes = strix::build_indexed_attention_state_layout(snapshot_model).kv_row_bytes;
        for (auto occupied : {1ULL,3ULL,5ULL}) {
            std::vector<std::uint16_t> expected_compact_output;
            std::vector<std::byte> expected_state;
            for (auto policy : {strix::SessionSnapshotPolicy::full_components, strix::SessionSnapshotPolicy::occupied_kv_v1}) {
                std::fill(host_bytes.begin(),host_bytes.end(),std::byte{0});
                own.guard.run([&] { check(hipMemset(own.state,0,layout.device_total_bytes),"reset compact QSA state"); });
                for (std::uint64_t p = 0; p < occupied; ++p) run(p,static_cast<unsigned>(p));
                strix::SerialSessionSnapshotRequest request;
                request.compiled_execution_identity = scratch.execution_identity;
                request.runtime_compatibility_identity = 0x515341;
                request.next_position = request.committed_tokens = occupied;
                request.state.valid = strix::PersistentState::attention_kv | strix::PersistentState::sparse_index;
                request.state.generations.fill(4); request.policy = policy;
                const auto plan = strix::build_serial_session_snapshot_plan(snapshot_model,layout,request);
                const auto restore = strix::build_serial_session_restore_plan(snapshot_model,layout,0x515341,plan);
                const strix::SessionSnapshotMemoryBinding saved{plan.plan_identity,
                    reinterpret_cast<std::uintptr_t>(own.saved),layout.device_total_bytes,0,0,33};
                strix::HipSessionSnapshotTransferBackend transfer;
                strix::capture_serial_session_snapshot(plan,layout,memory,saved,transfer,
                    policy == strix::SessionSnapshotPolicy::occupied_kv_v1 ? &snapshot_model : nullptr);
                run(occupied,77); run(occupied+1U,78); // provisional rows to reject
                strix::SerialSessionContext session;
                session.execution_identity = scratch.execution_identity; session.runtime_compatibility_identity = 0x515341;
                session.state_layout_identity = layout.layout_identity;
                session.next_position = session.committed_tokens = occupied+2U;
                strix::restore_serial_session_snapshot(session,restore,layout,memory,saved,transfer,
                    policy == strix::SessionSnapshotPolicy::occupied_kv_v1 ? &snapshot_model : nullptr);
                // Different current input must overwrite the ENTIRE poisoned
                // row before any selected-key/value read can observe it.
                const std::vector<std::uint16_t> poison(static_cast<std::size_t>((8U-occupied)*row_bytes/2U),0x7fc1U);
                own.guard.run([&] {
                    for (std::uint64_t l=0;l<snapshot_model.full_attention_layers;++l)
                        check(hipMemcpy(reinterpret_cast<void*>(memory.device_base+kv->offset+(l*8U+occupied)*row_bytes),
                            poison.data(),poison.size()*2U,hipMemcpyHostToDevice),"poison inactive future KV rows");
                });
                run(occupied,123);
                std::vector<std::uint16_t> produced(hidden_width);
                std::vector<std::byte> persisted(static_cast<std::size_t>(layout.device_total_bytes));
                own.guard.run([&] {
                    check(hipMemcpy(produced.data(),reinterpret_cast<const void*>(output.address),produced.size()*2U,
                        hipMemcpyDeviceToHost),"read compact QSA output");
                    check(hipMemcpy(persisted.data(),own.state,persisted.size(),hipMemcpyDeviceToHost),"read compact QSA active state");
                });
                for (const auto bits : produced) expect(std::isfinite(bf16_to_float(bits)),
                    "inactive future NaN KV cannot contaminate current QSA output");
                if (policy == strix::SessionSnapshotPolicy::full_components) {
                    expected_compact_output = produced; expected_state = persisted;
                } else {
                    expect(produced == expected_compact_output,"compact QSA rollback continuation equals full snapshot output bits");
                    for (const auto& allocation : layout.allocations) {
                        if (allocation.storage != strix::SessionStateStorage::device) continue;
                        if (allocation.component == strix::SessionStateComponent::attention_kv) {
                            for (std::uint64_t l=0;l<snapshot_model.full_attention_layers;++l) {
                                const auto begin=allocation.offset+l*8U*row_bytes;
                                const auto rows=occupied+(l==strix::indexed_attention_layer_ordinal(snapshot_model,*step.layer)?1U:0U);
                                expect(std::equal(persisted.data()+begin,persisted.data()+begin+rows*row_bytes,
                                    expected_state.data()+begin),"all occupied KV layers equal after actual QSA continuation");
                            }
                        } else expect(std::equal(persisted.data()+allocation.offset,
                            persisted.data()+allocation.offset+allocation.bytes,expected_state.data()+allocation.offset),
                            "QSA compact rollback preserves all non-KV device state");
                    }
                }
            }
        }
    }

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree indexed weight");
    }
    check(
        hipFree(device_state),
        "hipFree indexed state");
    check(
        hipFree(arena_pointer),
        "hipFree indexed arena");

    if (failures == 0) {
        std::cout
            << "HIP indexed-attention launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP indexed-attention launcher tests: "
        << error.what() << '\n';
    return 2;
}
