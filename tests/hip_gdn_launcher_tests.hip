#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
    std::memcpy(
        &output,
        &bits,
        sizeof(output));
    return output;
}

float round_bf16(float value) {
    return bf16_to_float(
        float_to_bf16(value));
}

float sigmoid(float value) {
    return
        1.0F /
        (1.0F +
         std::exp(-value));
}

float silu(float value) {
    return
        value *
        sigmoid(value);
}

float softplus(float value) {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return std::exp(value);
    }
    return std::log1p(
        std::exp(value));
}

strix::ModelContract fixture_model() {
    auto model =
        strix::ModelContract::qwen38_flash_next();

    model.vocab_size = 128;
    model.ple_eos_token_id = 127;
    model.hidden_size = 128;
    model.max_position_embeddings = 64;
    model.hidden_layers = 2;
    model.full_attention_interval = 2;
    model.ple_layer_1based = 1;
    model.ple_embed_dim = 128;
    model.ple_conv_kernel_size = 4;
    model.ngram_size = 3;
    model.heads_per_ngram = 1;
    model.hc_count = 2;
    model.hc_lowrank = 16;
    model.full_attention_layers = 1;
    model.gdn_layers = 1;
    model.attention_heads = 1;
    model.kv_heads = 1;
    model.head_dim = 128;
    model.mtp_full_attention_layers = 1;
    model.gdn_key_heads = 16;
    model.gdn_key_head_dim = 128;
    model.gdn_value_heads = 48;
    model.gdn_value_head_dim = 128;
    model.gdn_conv_kernel = 4;
    model.indexer_kv_heads = 1;
    model.indexer_head_dim = 128;
    model.indexer_compress_ratio = 4;
    model.moe_intermediate_size = 32;
    model.shared_expert_intermediate_size = 32;
    model.num_experts = 4;
    model.num_experts_per_token = 2;

    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::runtime_error(
            "fixture model invalid: " +
            error);
    }
    return model;
}

const strix::ExecutionStep& gdn_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::gated_delta_net) {
            return step;
        }
    }
    throw std::runtime_error(
        "fixture plan has no GDN step");
}

std::vector<float> sparse_matrix(
    std::uint32_t input,
    std::uint32_t output,
    float scale,
    std::uint32_t stride) {
    std::vector<float> result(
        static_cast<std::size_t>(input) *
        output,
        0.0F);
    for (std::uint32_t row = 0;
         row < output;
         ++row) {
        const auto feature =
            (row * stride + 3U) %
            input;
        result[
            static_cast<std::size_t>(row) *
                input +
            feature] =
            scale *
            static_cast<float>(
                row % 7U + 1U);
    }
    return result;
}

float project_row(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    std::uint32_t input_width,
    std::uint32_t row) {
    double sum = 0.0;
    const auto base =
        static_cast<std::size_t>(
            row) *
        input_width;
    for (std::uint32_t feature = 0;
         feature < input_width;
         ++feature) {
        sum +=
            static_cast<double>(
                input[feature]) *
            static_cast<double>(
                weights[
                    base +
                    feature]);
    }
    return static_cast<float>(
        sum);
}

struct Reference {
    std::vector<float> output;
    std::vector<float> matrix_state;
    std::vector<std::uint16_t> conv_state;
};

Reference cpu_reference(
    const strix::ModelContract& model,
    const std::vector<std::uint16_t>& hidden_bf16,
    const std::vector<float>& qkv_weight,
    const std::vector<float>& z_weight,
    const std::vector<float>& beta_weight,
    const std::vector<float>& alpha_weight,
    const std::vector<float>& conv_weight,
    const std::vector<float>& dt,
    const std::vector<float>& a,
    const std::vector<float>& norm_weight,
    const std::vector<float>& out_weight,
    std::vector<float> matrix_state,
    std::vector<std::uint16_t> conv_state) {
    const auto hidden =
        static_cast<std::uint32_t>(
            model.hidden_size);
    const auto key_heads =
        static_cast<std::uint32_t>(
            model.gdn_key_heads);
    const auto value_heads =
        static_cast<std::uint32_t>(
            model.gdn_value_heads);
    const auto head_dim =
        static_cast<std::uint32_t>(
            model.gdn_key_head_dim);
    const auto key_width =
        key_heads * head_dim;
    const auto value_width =
        value_heads * head_dim;
    const auto conv_width =
        key_width * 2U +
        value_width;
    const auto kernel =
        static_cast<std::uint32_t>(
            model.gdn_conv_kernel);

    std::vector<float> hidden_values(
        hidden);
    for (std::uint32_t index = 0;
         index < hidden;
         ++index) {
        hidden_values[index] =
            bf16_to_float(
                hidden_bf16[index]);
    }

    std::vector<float> qkv(
        conv_width);
    for (std::uint32_t row = 0;
         row < conv_width;
         ++row) {
        qkv[row] =
            round_bf16(
                project_row(
                    hidden_values,
                    qkv_weight,
                    hidden,
                    row));
    }

    std::vector<float> z(
        value_width);
    for (std::uint32_t row = 0;
         row < value_width;
         ++row) {
        z[row] =
            round_bf16(
                project_row(
                    hidden_values,
                    z_weight,
                    hidden,
                    row));
    }

    std::vector<float> beta(
        value_heads);
    std::vector<float> gate(
        value_heads);
    for (std::uint32_t row = 0;
         row < value_heads;
         ++row) {
        beta[row] =
            round_bf16(
                project_row(
                    hidden_values,
                    beta_weight,
                    hidden,
                    row));
        gate[row] =
            round_bf16(
                project_row(
                    hidden_values,
                    alpha_weight,
                    hidden,
                    row));
    }

    for (std::uint32_t channel = 0;
         channel < conv_width;
         ++channel) {
        const auto base =
            static_cast<std::size_t>(
                channel) *
            kernel;
        for (std::uint32_t index = 0;
             index + 1U < kernel;
             ++index) {
            conv_state[
                base +
                index] =
                conv_state[
                    base +
                    index + 1U];
        }
        conv_state[
            base +
            kernel - 1U] =
            float_to_bf16(
                qkv[channel]);

        float sum = 0.0F;
        for (std::uint32_t index = 0;
             index < kernel;
             ++index) {
            sum +=
                bf16_to_float(
                    conv_state[
                        base +
                        index]) *
                conv_weight[
                    base +
                    index];
        }
        qkv[channel] =
            round_bf16(
                silu(sum));
    }

    for (std::uint32_t head = 0;
         head < value_heads;
         ++head) {
        beta[head] =
            round_bf16(
                sigmoid(
                    beta[head]));
        gate[head] =
            a[head] *
            softplus(
                gate[head] +
                dt[head]);
    }

    for (std::uint32_t part = 0;
         part < 2U;
         ++part) {
        for (std::uint32_t head = 0;
             head < key_heads;
             ++head) {
            const auto base =
                static_cast<std::size_t>(
                    part) *
                    key_width +
                static_cast<std::size_t>(
                    head) *
                    head_dim;
            double norm = 0.0;
            for (std::uint32_t index = 0;
                 index < head_dim;
                 ++index) {
                const auto value =
                    qkv[
                        base +
                        index];
                norm +=
                    static_cast<double>(
                        value) *
                    value;
            }
            const float inverse =
                1.0F /
                std::sqrt(
                    static_cast<float>(
                        norm) +
                    model.gdn_qk_l2_epsilon);
            const float q_scale =
                part == 0
                    ? 1.0F /
                          std::sqrt(
                              static_cast<float>(
                                  head_dim))
                    : 1.0F;
            for (std::uint32_t index = 0;
                 index < head_dim;
                 ++index) {
                qkv[
                    base +
                    index] *=
                    inverse *
                    q_scale;
            }
        }
    }

    std::vector<float> core(
        value_width);
    for (std::uint32_t value_head = 0;
         value_head < value_heads;
         ++value_head) {
        const auto key_head =
            value_head %
            key_heads;
        const auto q_base =
            static_cast<std::size_t>(
                key_head) *
            head_dim;
        const auto k_base =
            key_width +
            q_base;
        const auto v_base =
            key_width * 2U +
            static_cast<std::size_t>(
                value_head) *
            head_dim;
        const float decay =
            std::exp(
                gate[value_head]);

        for (std::uint32_t value_index = 0;
             value_index < head_dim;
             ++value_index) {
            float remembered = 0.0F;
            for (std::uint32_t key_index = 0;
                 key_index < head_dim;
                 ++key_index) {
                const auto state_index =
                    (static_cast<std::size_t>(
                         value_head) *
                         head_dim +
                     key_index) *
                        head_dim +
                    value_index;
                matrix_state[state_index] *=
                    decay;
                remembered +=
                    matrix_state[
                        state_index] *
                    qkv[
                        k_base +
                        key_index];
            }

            const float delta =
                (qkv[
                     v_base +
                     value_index] -
                 remembered) *
                beta[value_head];

            float output = 0.0F;
            for (std::uint32_t key_index = 0;
                 key_index < head_dim;
                 ++key_index) {
                const auto state_index =
                    (static_cast<std::size_t>(
                         value_head) *
                         head_dim +
                     key_index) *
                        head_dim +
                    value_index;
                matrix_state[
                    state_index] +=
                    qkv[
                        k_base +
                        key_index] *
                    delta;
                output +=
                    matrix_state[
                        state_index] *
                    qkv[
                        q_base +
                        key_index];
            }
            core[
                static_cast<std::size_t>(
                    value_head) *
                    head_dim +
                value_index] =
                output;
        }
    }

    for (std::uint32_t head = 0;
         head < value_heads;
         ++head) {
        const auto base =
            static_cast<std::size_t>(
                head) *
            head_dim;
        double mean_square = 0.0;
        for (std::uint32_t feature = 0;
             feature < head_dim;
             ++feature) {
            const auto value =
                core[
                    base +
                    feature];
            mean_square +=
                static_cast<double>(
                    value) *
                value;
        }
        mean_square /=
            static_cast<double>(
                head_dim);
        const float inverse =
            1.0F /
            std::sqrt(
                static_cast<float>(
                    mean_square) +
                model.rms_norm_epsilon);

        for (std::uint32_t feature = 0;
             feature < head_dim;
             ++feature) {
            const auto index =
                base +
                feature;
            core[index] =
                core[index] *
                inverse *
                norm_weight[feature] *
                sigmoid(
                    z[index]);
        }
    }

    Reference reference;
    reference.output.resize(
        hidden);
    for (std::uint32_t row = 0;
         row < hidden;
         ++row) {
        const auto value =
            project_row(
                core,
                out_weight,
                value_width,
                row);
        reference.output[row] =
            round_bf16(
                value);
    }
    reference.matrix_state =
        std::move(
            matrix_state);
    reference.conv_state =
        std::move(
            conv_state);
    return reference;
}

}  // namespace

int main() try {
    const auto model =
        fixture_model();
    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        gdn_step(
            plan);
    const auto& step_io =
        io.steps[
            step.id];

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.activation_element_bytes = 2;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            buffer_request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc GDN arena");
    check(
        hipMemset(
            arena_pointer,
            0,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "clear GDN arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x47444e5445535401ULL;
    memory.arena_base =
        reinterpret_cast<
            std::uintptr_t>(
                arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto input =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto output =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    const auto hidden =
        static_cast<std::uint32_t>(
            model.hidden_size);
    const auto key_width =
        static_cast<std::uint32_t>(
            model.gdn_key_heads *
            model.gdn_key_head_dim);
    const auto value_width =
        static_cast<std::uint32_t>(
            model.gdn_value_heads *
            model.gdn_value_head_dim);
    const auto conv_width =
        key_width * 2U +
        value_width;
    const auto value_heads =
        static_cast<std::uint32_t>(
            model.gdn_value_heads);
    const auto kernel =
        static_cast<std::uint32_t>(
            model.gdn_conv_kernel);

    std::vector<std::uint16_t> hidden_values(
        hidden);
    for (std::uint32_t index = 0;
         index < hidden;
         ++index) {
        hidden_values[index] =
            float_to_bf16(
                static_cast<float>(
                    static_cast<int>(
                        index % 19U) -
                    9) /
                23.0F);
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input.address),
            hidden_values.data(),
            hidden_values.size() *
                sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy GDN input");

    const auto qkv =
        sparse_matrix(
            hidden,
            conv_width,
            // Keep recurrence updates large enough that a missed broadcast
            // is visible in state, even after the final BF16 projection.
            0.2500F,
            5U);
    const auto z =
        sparse_matrix(
            hidden,
            value_width,
            0.0015F,
            7U);
    const auto beta =
        sparse_matrix(
            hidden,
            value_heads,
            0.0040F,
            11U);
    const auto alpha =
        sparse_matrix(
            hidden,
            value_heads,
            0.0030F,
            13U);

    std::vector<float> conv(
        static_cast<std::size_t>(
            conv_width) *
        kernel,
        0.0F);
    for (std::uint32_t channel = 0;
         channel < conv_width;
         ++channel) {
        const auto base =
            static_cast<std::size_t>(
                channel) *
            kernel;
        conv[
            base +
            kernel - 1U] =
            0.75F +
            static_cast<float>(
                channel % 5U) *
            0.01F;
        conv[
            base +
            kernel - 2U] =
            0.05F;
    }

    std::vector<float> dt(
        value_heads);
    std::vector<float> a(
        value_heads);
    for (std::uint32_t head = 0;
         head < value_heads;
         ++head) {
        dt[head] =
            0.03F *
            static_cast<float>(
                head + 1U);
        // GGUF stores -exp(A_log), not A_log itself.
        a[head] =
            -0.12F -
            0.03F *
            static_cast<float>(
                head);
    }

    std::vector<float> norm(
        model.gdn_value_head_dim);
    for (std::size_t index = 0;
         index < norm.size();
         ++index) {
        norm[index] =
            0.95F +
            static_cast<float>(
                index % 7U) *
            0.01F;
    }

    const auto out_weight =
        sparse_matrix(
            value_width,
            hidden,
            0.0035F,
            17U);

    strix::SerialSessionStateLayoutRequest
        state_request;
    state_request.context_tokens = 8;
    const auto state_layout =
        strix::build_serial_session_state_layout(
            model,
            state_request);

    void* state_device = nullptr;
    check(
        hipMalloc(
            &state_device,
            static_cast<std::size_t>(
                state_layout.device_total_bytes)),
        "hipMalloc GDN state pool");
    check(
        hipMemset(
            state_device,
            0,
            static_cast<std::size_t>(
                state_layout.device_total_bytes)),
        "clear GDN state pool");
    std::vector<std::byte> host_state(
        static_cast<std::size_t>(
            state_layout.host_total_bytes));

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        reinterpret_cast<
            std::uintptr_t>(
                state_device);
    state_memory.device_bytes =
        state_layout.device_total_bytes;
    state_memory.device_allocation_generation = 1;
    state_memory.host_base =
        reinterpret_cast<
            std::uintptr_t>(
                host_state.data());
    state_memory.host_bytes =
        state_layout.host_total_bytes;
    state_memory.host_allocation_generation = 1;

    const auto matrix_address =
        strix::resolve_gdn_layer_state(
            state_layout,
            state_memory,
            model,
            *step.layer,
            strix::SessionStateComponent::
                recurrent_matrix);
    const auto conv_address =
        strix::resolve_gdn_layer_state(
            state_layout,
            state_memory,
            model,
            *step.layer,
            strix::SessionStateComponent::
                gdn_convolution);

    std::vector<float> matrix_state(
        static_cast<std::size_t>(
            matrix_address.bytes /
            sizeof(float)));
    for (std::size_t index = 0;
         index < matrix_state.size();
         ++index) {
        matrix_state[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 13U) -
                6) *
            1e-5F;
    }

    std::vector<std::uint16_t> conv_state(
        static_cast<std::size_t>(
            conv_address.bytes /
            sizeof(std::uint16_t)));
    for (std::size_t index = 0;
         index < conv_state.size();
         ++index) {
        conv_state[index] =
            float_to_bf16(
                static_cast<float>(
                    static_cast<int>(
                        index % 11U) -
                    5) *
                0.002F);
    }

    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                matrix_address.address),
            matrix_state.data(),
            matrix_state.size() *
                sizeof(float),
            hipMemcpyHostToDevice),
        "copy initial recurrent state");
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                conv_address.address),
            conv_state.data(),
            conv_state.size() *
                sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy initial conv state");

    const auto expected =
        cpu_reference(
            model,
            hidden_values,
            qkv,
            z,
            beta,
            alpha,
            conv,
            dt,
            a,
            norm,
            out_weight,
            matrix_state,
            conv_state);

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";

    const auto add_tensor =
        [&](std::string name,
            strix::TensorRole role,
            std::vector<std::uint64_t> dims,
            const std::vector<float>& values) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.role =
                role;
            tensor.layer =
                step.layer;
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
        "blk.0.attn_qkv.weight",
        strix::TensorRole::
            attention_projection,
        {hidden, conv_width},
        qkv);
    add_tensor(
        "blk.0.attn_gate.weight",
        strix::TensorRole::
            attention_projection,
        {hidden, value_width},
        z);
    add_tensor(
        "blk.0.ssm_conv1d.weight",
        strix::TensorRole::
            recurrent_state,
        {kernel, conv_width},
        conv);
    add_tensor(
        "blk.0.ssm_dt.bias",
        strix::TensorRole::
            recurrent_state,
        {value_heads},
        dt);
    add_tensor(
        "blk.0.ssm_a",
        strix::TensorRole::
            recurrent_state,
        {value_heads},
        a);
    add_tensor(
        "blk.0.ssm_beta.weight",
        strix::TensorRole::
            recurrent_state,
        {hidden, value_heads},
        beta);
    add_tensor(
        "blk.0.ssm_alpha.weight",
        strix::TensorRole::
            recurrent_state,
        {hidden, value_heads},
        alpha);
    add_tensor(
        "blk.0.ssm_norm.weight",
        strix::TensorRole::
            recurrent_state,
        {model.gdn_value_head_dim},
        norm);
    add_tensor(
        "blk.0.ssm_out.weight",
        strix::TensorRole::
            recurrent_state,
        {value_width, hidden},
        out_weight);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    const std::array<
        const std::vector<float>*,
        9>
        source_weights{
            &qkv,
            &z,
            &conv,
            &dt,
            &a,
            &beta,
            &alpha,
            &norm,
            &out_weight,
        };

    std::vector<void*> device_weights(
        source_weights.size(),
        nullptr);
    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;

    for (std::size_t index = 0;
         index <
            source_weights.size();
         ++index) {
        const auto bytes =
            source_weights[index]->size() *
            sizeof(float);
        check(
            hipMalloc(
                &device_weights[index],
                bytes),
            "hipMalloc GDN weight");
        check(
            hipMemcpy(
                device_weights[index],
                source_weights[index]->data(),
                bytes,
                hipMemcpyHostToDevice),
            "copy GDN weight");

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
        0x47444e5445535402ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {2, 3, 4, 5, 6, 7, 8},
        });
    access.accesses.push_back(
        {
            step.id,
            step.op,
            1,
            {0, 1},
        });

    strix::ExecutionRuntimeResources resources;
    resources.scratch =
        &memory;
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
    context.step_id =
        step.id;
    context.op =
        step.op;
    context.layer =
        step.layer;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.temporaries =
        step_io.temporaries;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.session_state_layout =
        &state_layout;
    context.session_state_memory =
        &state_memory;
    context.resources =
        &resources;
    context.reads_state =
        step.reads_state;
    context.writes_state =
        step.writes_state;
    context.logical_rows = 1;
    context.request_concurrency = 1;

    strix::HipGdnLauncherConfig config;
    config.model =
        model;
    config.activation =
        strix::HipActivationStorage::bf16;

    auto launcher =
        strix::make_hip_gdn_launcher(
            "test.gdn",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "GDN synchronize");

    std::vector<std::uint16_t> actual_output(
        hidden);
    check(
        hipMemcpy(
            actual_output.data(),
            reinterpret_cast<const void*>(
                output.address),
            actual_output.size() *
                sizeof(std::uint16_t),
            hipMemcpyDeviceToHost),
        "copy GDN output");

    std::vector<float> actual_matrix(
        matrix_state.size());
    check(
        hipMemcpy(
            actual_matrix.data(),
            reinterpret_cast<const void*>(
                matrix_address.address),
            actual_matrix.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy GDN matrix state");

    std::vector<std::uint16_t> actual_conv(
        conv_state.size());
    check(
        hipMemcpy(
            actual_conv.data(),
            reinterpret_cast<const void*>(
                conv_address.address),
            actual_conv.size() *
                sizeof(std::uint16_t),
            hipMemcpyDeviceToHost),
        "copy GDN conv state");

    float output_error = 0.0F;
    for (std::size_t index = 0;
         index < actual_output.size();
         ++index) {
        output_error =
            std::max(
                output_error,
                std::abs(
                    bf16_to_float(
                        actual_output[index]) -
                    expected.output[index]));
    }

    float matrix_error = 0.0F;
    for (std::size_t index = 0;
         index < actual_matrix.size();
         ++index) {
        matrix_error =
            std::max(
                matrix_error,
                std::abs(
                    actual_matrix[index] -
                    expected.matrix_state[
                        index]));
    }

    expect(
        output_error <= 3e-3F,
        "HIP GDN output matches CPU reference");
    expect(
        matrix_error <= 1e-7F,
        "HIP GDN recurrent matrix matches CPU reference");
    expect(
        actual_conv ==
            expected.conv_state,
        "HIP GDN convolution state matches BF16 CPU reference");

    // Four waves consume each delta broadcast. Check the complete recurrence
    // against the independent scalar equations, then repeat from identical
    // asymmetric state: a shared-memory race must not hide behind BF16 output
    // rounding or an absolute tolerance larger than the state update.
    const auto first_matrix = actual_matrix;
    const auto first_output = actual_output;
    for (unsigned iteration = 0; iteration < 16; ++iteration) {
        check(hipMemcpy(reinterpret_cast<void*>(matrix_address.address), matrix_state.data(),
            matrix_address.bytes, hipMemcpyHostToDevice), "restore recurrence broadcast fixture state");
        check(hipMemcpy(reinterpret_cast<void*>(conv_address.address), conv_state.data(),
            conv_address.bytes, hipMemcpyHostToDevice), "restore recurrence broadcast fixture convolution");
        launcher.function(context, launcher.user_data);
        check(hipDeviceSynchronize(), "drain repeated recurrence broadcast fixture");
        check(hipMemcpy(actual_matrix.data(), reinterpret_cast<const void*>(matrix_address.address),
            matrix_address.bytes, hipMemcpyDeviceToHost), "read repeated recurrence broadcast state");
        check(hipMemcpy(actual_output.data(), reinterpret_cast<const void*>(output.address),
            actual_output.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost), "read repeated recurrence broadcast output");
        expect(std::memcmp(actual_matrix.data(), first_matrix.data(), matrix_address.bytes) == 0 &&
            actual_output == first_output, "repeated identical GDN input/state is bit deterministic");
        float recurrence_error = 0.0F;
        for (std::size_t i = 0; i < actual_matrix.size(); ++i)
            recurrence_error = std::max(recurrence_error, std::fabs(actual_matrix[i] - expected.matrix_state[i]));
        expect(recurrence_error <= 1e-7F, "all waves consume the independently computed GDN delta");
    }

    // Capture the complete existing GDN operator: projections, convolution,
    // recurrence, gated normalization and final projection. No host position
    // scalar controls the captured state stage.
    {
        auto graph_config = config;
        graph_config.enable_device_graph_inputs = true;
        auto graph_launcher = strix::make_hip_gdn_launcher("test.gdn.device", &graph_config);
        expect(graph_launcher.implementation_identity != launcher.implementation_identity,
            "GDN device-input capability is bound to launcher identity");
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
        check(hipStreamCreateWithFlags(&graph.stream, hipStreamNonBlocking), "create GDN graph stream");
        check(hipMalloc(&graph.arguments, 16), "allocate GDN device arguments");
        check(hipMalloc(&graph.status, 4), "allocate GDN graph status");
        context.execution_identity = memory.execution_identity;
        context.position = std::numeric_limits<std::uint64_t>::max();
        strix::HipLaunchContext scope;
        scope.stream = graph.stream; scope.graph_capture = true;
        scope.arguments = strix::bind_graph_inputs(strix::graph_input_layout(1),
            {memory.execution_identity, reinterpret_cast<std::uintptr_t>(graph.arguments), 16, 1});
        scope.status = {memory.execution_identity, reinterpret_cast<std::uintptr_t>(graph.status), 4, 1};
        scope.context_tokens = state_layout.context_tokens;
        scope.vocabulary_size = static_cast<std::uint32_t>(model.vocab_size);
        check(hipStreamBeginCapture(graph.stream, hipStreamCaptureModeThreadLocal), "capture complete GDN");
        try {
            const strix::HipLaunchScope owned(scope);
            graph_launcher.function(context, graph_launcher.user_data);
        } catch (...) { static_cast<void>(hipStreamEndCapture(graph.stream, &graph.graph)); throw; }
        check(hipStreamEndCapture(graph.stream, &graph.graph), "finish GDN capture");
        check(hipGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0), "instantiate GDN graph");
        auto copy_state = [&](const std::vector<float>& matrix, const std::vector<std::uint16_t>& convolution) {
            check(hipMemcpy(reinterpret_cast<void*>(matrix_address.address), matrix.data(), matrix.size() * sizeof(float), hipMemcpyHostToDevice), "set GDN graph matrix");
            check(hipMemcpy(reinterpret_cast<void*>(conv_address.address), convolution.data(), convolution.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice), "set GDN graph convolution");
        };
        auto read_state = [&]() {
            Reference r;
            r.matrix_state.resize(matrix_state.size()); r.conv_state.resize(conv_state.size());
            check(hipMemcpy(r.matrix_state.data(), reinterpret_cast<void*>(matrix_address.address), matrix_address.bytes, hipMemcpyDeviceToHost), "read GDN graph matrix");
            check(hipMemcpy(r.conv_state.data(), reinterpret_cast<void*>(conv_address.address), conv_address.bytes, hipMemcpyDeviceToHost), "read GDN graph convolution");
            return r;
        };
        auto control = [&](std::uint64_t position, std::uint32_t status) {
            std::array<std::byte, 16> packet{};
            std::memcpy(packet.data() + 8, &position, 8);
            check(hipMemcpy(graph.arguments, packet.data(), packet.size(), hipMemcpyHostToDevice), "set dynamic GDN position");
            check(hipMemcpy(graph.status, &status, 4, hipMemcpyHostToDevice), "set sticky GDN status");
        };
        auto replay = [&]() {
            check(hipGraphLaunch(graph.executable, graph.stream), "replay complete GDN");
            check(hipStreamSynchronize(graph.stream), "complete GDN graph");
        };
        auto read_status = [&]() {
            std::uint32_t result = 0;
            check(hipMemcpy(&result, graph.status, 4, hipMemcpyDeviceToHost), "read GDN graph status");
            return result;
        };
        auto previous = Reference{{}, matrix_state, conv_state};
        auto cpu = previous;
        copy_state(previous.matrix_state, previous.conv_state);
        for (unsigned iteration = 0; iteration < 3; ++iteration) {
            for (unsigned i = 0; i < hidden; ++i) hidden_values[i] = float_to_bf16(
                static_cast<float>(static_cast<int>((i + iteration * 3U) % 19U) - 9) / 23.0F);
            check(hipMemcpy(reinterpret_cast<void*>(input.address), hidden_values.data(), hidden_values.size() * 2, hipMemcpyHostToDevice), "change GDN graph input");
            control(iteration, 0); replay();
            expect(read_status() == 0, "valid GDN graph status");
            auto current = read_state();
            std::vector<std::uint16_t> graph_output(hidden), eager_output(hidden);
            check(hipMemcpy(graph_output.data(), reinterpret_cast<void*>(output.address), hidden * 2, hipMemcpyDeviceToHost), "read GDN graph output");
            copy_state(previous.matrix_state, previous.conv_state);
            auto eager_scope = scope; eager_scope.graph_capture = false;
            { const strix::HipLaunchScope owned(eager_scope); graph_launcher.function(context, graph_launcher.user_data); }
            check(hipStreamSynchronize(graph.stream), "complete GDN eager control");
            const auto eager = read_state();
            check(hipMemcpy(eager_output.data(), reinterpret_cast<void*>(output.address), hidden * 2, hipMemcpyDeviceToHost), "read GDN eager control output");
            expect(graph_output == eager_output && current.conv_state == eager.conv_state &&
                std::memcmp(current.matrix_state.data(), eager.matrix_state.data(), matrix_address.bytes) == 0,
                "complete GDN graph replay exactly matches eager state/output bits");
            cpu = cpu_reference(model, hidden_values, qkv, z, beta, alpha, conv, dt, a, norm, out_weight, cpu.matrix_state, cpu.conv_state);
            for (unsigned i = 0; i < hidden; ++i) expect(std::isfinite(bf16_to_float(graph_output[i])) &&
                std::fabs(bf16_to_float(graph_output[i]) - cpu.output[i]) <= 3e-3F, "repeated GDN graph independent CPU output");
            float graph_matrix_error = 0.0F;
            bool graph_matrix_finite = true;
            for (std::size_t i = 0; i < current.matrix_state.size(); ++i) {
                graph_matrix_finite = graph_matrix_finite && std::isfinite(current.matrix_state[i]);
                graph_matrix_error = std::max(graph_matrix_error, std::fabs(current.matrix_state[i] - cpu.matrix_state[i]));
            }
            expect(graph_matrix_finite && graph_matrix_error <= 1e-7F,
                "repeated GDN graph independent CPU matrix");
            expect(current.conv_state == cpu.conv_state, "repeated GDN graph independent CPU convolution");
            previous = current; copy_state(previous.matrix_state, previous.conv_state);
        }
        for (const auto position : {state_layout.context_tokens, std::uint64_t{1} << 32U, std::numeric_limits<std::uint64_t>::max()}) {
            control(position, 0); replay();
            const auto rejected = read_state();
            expect(read_status() == 2 && rejected.conv_state == previous.conv_state &&
                std::memcmp(rejected.matrix_state.data(), previous.matrix_state.data(), matrix_address.bytes) == 0,
                "invalid device GDN position cannot mutate recurrence/convolution");
        }
        control(0, 1); replay();
        auto rejected = read_state();
        expect(read_status() == 1 && rejected.conv_state == previous.conv_state &&
            std::memcmp(rejected.matrix_state.data(), previous.matrix_state.data(), matrix_address.bytes) == 0,
            "upstream GDN status stays sticky and preserves persistent state");
        std::fill(hidden_values.begin(), hidden_values.end(), float_to_bf16(0.5F));
        check(hipMemcpy(reinterpret_cast<void*>(input.address), hidden_values.data(), hidden_values.size() * 2, hipMemcpyHostToDevice), "set finite overflow GDN input");
        auto overflow_qkv = qkv;
        std::fill_n(overflow_qkv.begin(), hidden, std::numeric_limits<float>::max());
        check(hipMemcpy(device_weights[0], overflow_qkv.data(), overflow_qkv.size() * sizeof(float), hipMemcpyHostToDevice), "set GDN overflowing projection weights");
        control(0, 0); replay(); rejected = read_state();
        expect(read_status() == 8 && rejected.conv_state == previous.conv_state &&
            std::memcmp(rejected.matrix_state.data(), previous.matrix_state.data(), matrix_address.bytes) == 0,
            "nonfinite GDN projection rejects before persistent mutation");
        check(hipMemcpy(device_weights[0], qkv.data(), qkv.size() * sizeof(float), hipMemcpyHostToDevice), "restore GDN projection weights");
        hidden_values[0] = 0x7f80;
        check(hipMemcpy(reinterpret_cast<void*>(input.address), hidden_values.data(), hidden_values.size() * 2, hipMemcpyHostToDevice), "set nonfinite GDN graph input");
        control(0, 0); replay(); rejected = read_state();
        expect(read_status() == 4 && rejected.conv_state == previous.conv_state &&
            std::memcmp(rejected.matrix_state.data(), previous.matrix_state.data(), matrix_address.bytes) == 0,
            "nonfinite GDN input rejected before persistent mutation");
    }

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree GDN weight");
    }
    check(
        hipFree(state_device),
        "hipFree GDN state");
    check(
        hipFree(arena_pointer),
        "hipFree GDN arena");

    if (failures == 0) {
        std::cout
            << "HIP GDN launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP GDN launcher tests: "
        << error.what() << '\n';
    return 2;
}
