#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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

void* upload(
    const void* source,
    std::size_t bytes) {
    void* device = nullptr;
    check(
        hipMalloc(
            &device,
            bytes),
        "hipMalloc indexed fixture weight");
    check(
        hipMemcpy(
            device,
            source,
            bytes,
            hipMemcpyHostToDevice),
        "copy indexed fixture weight");
    return device;
}

}  // namespace

int main() try {
    auto model =
        strix::ModelContract::qwen38_flash_next();
    model.hidden_size = 256;
    model.attention_heads = 2;
    model.kv_heads = 1;
    model.indexer_n_heads = 2;

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
        check(
            hipMemcpy(
                device_weights[index],
                host_weights[index]->data(),
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
    constexpr std::uint64_t context_tokens = 4;
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
