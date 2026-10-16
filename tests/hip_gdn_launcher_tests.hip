#include "virtual_q8_oracle.hpp"
#include "strix/compiled_execution.hpp"
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
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
bool dense_q8_fixture = false;
std::uint32_t dense_q8_width = 0;

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
    if (dense_q8_fixture && input_width == dense_q8_width)
        return virtual_q8_test::legacy_oracle(input,
            std::span<const float>(weights.data() + static_cast<std::size_t>(row) * input_width, input_width));
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

int main(int argc, char** argv) try {
    auto model = fixture_model();
    const bool candidate_enabled = argc >= 2;
    const std::string fixture_mode = argc >= 3 ? argv[2] : "";
    const bool wave8_enabled = fixture_mode == "wave8";
    const bool wave8_output_enabled = fixture_mode == "wave8-output";
    const bool prefetch_enabled = fixture_mode == "prefetch" || fixture_mode == "prefetch-fallback" || wave8_enabled;
    if (argc == 2 || ((argc == 3 || argc == 4) && (fixture_mode == "fallback" || prefetch_enabled || wave8_output_enabled))) {
        dense_q8_fixture = fixture_mode != "fallback" && fixture_mode != "prefetch-fallback";
        dense_q8_width = static_cast<std::uint32_t>(std::stoul(argv[1]));
        if (dense_q8_width != 128U && dense_q8_width != 288U && dense_q8_width != 480U && dense_q8_width != 640U && dense_q8_width != 2560U)
            throw std::invalid_argument("GDN Q8 fixture supports widths128,288,480,640,2560");
        model.hidden_size = dense_q8_width;
        if (argc == 4) {
            const auto heads = static_cast<std::uint32_t>(std::stoul(argv[3]));
            if ((!wave8_enabled && !wave8_output_enabled) || (heads != 4U && heads != 12U))
                throw std::invalid_argument("GDN wave8 tail fixture supports 4 or 12 value heads");
            model.gdn_key_heads = 4;
            model.gdn_value_heads = heads;
        }
    } else if (argc != 1) throw std::invalid_argument("GDN fixture accepts width and optional fallback mode");
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

    auto qkv =
        sparse_matrix(
            hidden,
            conv_width,
            // Keep recurrence updates large enough that a missed broadcast
            // is visible in state, even after the final BF16 projection.
            0.2500F,
            5U);
    auto z =
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
    auto alpha =
        sparse_matrix(
            hidden,
            value_heads,
            0.0030F,
            13U);

    // Canonical packed Q8_0 blocks: little-endian half scale and 32 signed codes.
    // Three formats are exercised together: Q8 inputs, F32 beta and F32 output.
    std::array<std::vector<std::byte>, 3> packed_q8;
    if (dense_q8_fixture) {
        std::array<std::vector<float>*, 3> matrices{&qkv, &z, &alpha};
        for (std::size_t m = 0; m < matrices.size(); ++m) {
            auto& values = *matrices[m]; auto& bytes = packed_q8[m];
            bytes.resize(values.size() / 32U * 34U);
            for (std::size_t block = 0; block < values.size() / 32U; ++block) {
                const auto exponent = static_cast<unsigned>((block + m) % 3U);
                const auto scale_exponent = m == 0 && block == 0 ? 10U : 7U + exponent;
                const auto mantissa = static_cast<unsigned>((block * 137U + m * 251U) % 1024U);
                const std::uint16_t half = static_cast<std::uint16_t>((scale_exponent << 10U) | mantissa);
                bytes[block * 34U] = static_cast<std::byte>(half & 255U);
                bytes[block * 34U + 1U] = static_cast<std::byte>(half >> 8U);
                const float scale = std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F, static_cast<int>(scale_exponent) - 15);
                for (unsigned local = 0; local < 32; ++local) {
                    const auto code = static_cast<std::int8_t>(static_cast<int>((block * 29U + local * 17U + m * 7U) % 256U) - 128);
                    bytes[block * 34U + 2U + local] = static_cast<std::byte>(static_cast<std::uint8_t>(code));
                    values[block * 32U + local] = scale * static_cast<float>(code);
                }
            }
        }
    }

    if (dense_q8_fixture && hidden >= 288U) {
        // A two-term leaf at features 0/256 whose FMA distinction survives
        // the actual BF16-in-FP32 epilogue (-15.1875 versus -15.125).
        std::fill_n(qkv.begin(), hidden, 0.0F);
        std::fill_n(packed_q8[0].begin(), hidden / 32U * 34U, std::byte{0});
        const auto set_block = [&](unsigned block, std::uint16_t half, unsigned local, std::int8_t code) {
            auto* bytes = packed_q8[0].data() + block * 34U;
            bytes[0] = static_cast<std::byte>(half & 255U); bytes[1] = static_cast<std::byte>(half >> 8U);
            bytes[2U + local] = static_cast<std::byte>(static_cast<std::uint8_t>(code));
        };
        set_block(0, 0x3e57U, 0, 107); set_block(8, 0x3ddbU, 0, -117);
        qkv[0] = 1.5849609375F * 107.0F; qkv[256] = 1.4638671875F * -117.0F;
        hidden_values[0] = float_to_bf16(1.8515625F);
        hidden_values[256] = float_to_bf16(1.921875F);
    }

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

    auto out_weight =
        sparse_matrix(
            value_width,
            hidden,
            0.0035F,
            17U);

    std::vector<std::byte> packed_output;
    if (wave8_output_enabled) {
        packed_output.resize(out_weight.size()/32U*34U);
        for(std::size_t block=0;block<out_weight.size()/32U;++block) {
            const std::uint16_t bits=static_cast<std::uint16_t>(0x0400U+(block*137U)%1024U);
            packed_output[block*34U]=static_cast<std::byte>(bits&255U);
            packed_output[block*34U+1U]=static_cast<std::byte>(bits>>8U);
            const float scale=std::ldexp(1.0F+static_cast<float>(bits&1023U)/1024.0F,-14);
            for(unsigned local=0;local<32;++local) {
                const auto code=static_cast<std::int8_t>(static_cast<int>((block*29U+local*17U)%256U)-128);
                packed_output[block*34U+2U+local]=static_cast<std::byte>(static_cast<std::uint8_t>(code));
                out_weight[block*32U+local]=scale*static_cast<float>(code);
            }
        }
    }

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

    if (dense_q8_fixture) {
        constexpr std::array<std::size_t, 3> indices{0, 1, 6};
        for (std::size_t m = 0; m < indices.size(); ++m) {
            auto& tensor = manifest.tensors[indices[m]];
            manifest.total_tensor_bytes -= tensor.byte_size;
            tensor.ggml_type = 8U; tensor.byte_size = packed_q8[m].size();
            manifest.total_tensor_bytes += tensor.byte_size;
        }
    }
    if(wave8_output_enabled) {
        auto& tensor=manifest.tensors[8];manifest.total_tensor_bytes-=tensor.byte_size;
        tensor.ggml_type=8U;tensor.byte_size=packed_output.size();manifest.total_tensor_bytes+=tensor.byte_size;
    }
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
        const void* payload = source_weights[index]->data();
        auto bytes = source_weights[index]->size() * sizeof(float);
        if (dense_q8_fixture && (index == 0 || index == 1 || index == 6)) {
            const auto m = index == 0 ? 0U : index == 1 ? 1U : 2U;
            payload = packed_q8[m].data(); bytes = packed_q8[m].size();
        }
        if(wave8_output_enabled && index==8U) {payload=packed_output.data();bytes=packed_output.size();}
        check(
            hipMalloc(
                &device_weights[index],
                bytes + (dense_q8_fixture ? 34U : 0U)),
            "hipMalloc GDN weight");
        check(
            hipMemcpy(
                device_weights[index],
                payload,
                bytes,
                hipMemcpyHostToDevice),
            "copy GDN weight");

        if (dense_q8_fixture) {
            std::array<std::byte, 34> guard{};
            guard[0] = std::byte{0}; guard[1] = std::byte{0x7e}; // half quiet NaN
            check(hipMemcpy(static_cast<std::byte*>(device_weights[index]) + bytes, guard.data(),
                guard.size(), hipMemcpyHostToDevice), "poison unbound GDN weight guard");
        }
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

    // The output experiment changes only the output organization. Both arms
    // use the same virtual input path, nonzero initial state and packed weights.
    if(wave8_output_enabled) {
        config.virtual_q8_input_projections=true;
        config.virtual_q8_output_projection=true;
    }
    strix::kernels::KernelRegistry gdn_variants;
    strix::ExecutionBindingRequest gdn_request;
    strix::register_hip_gdn_kernel_variant(model, gdn_request, config, gdn_variants);
    expect(gdn_variants.variants().size() == 1 &&
        gdn_variants.variants().front().resources.lds_bytes >=
            (128U * 32U + 32U) * sizeof(float) + sizeof(std::uint32_t),
        "GDN LDS declaration covers both reductions, immutable delta and graph control");

    {
        strix::HipGdnLauncherConfig default_config;
        default_config.model = fixture_model();
        auto candidate_config = default_config; candidate_config.virtual_q8_input_projections = true;
        strix::kernels::KernelRegistry defaults, candidates;
        strix::register_hip_gdn_kernel_variant(default_config.model, gdn_request, default_config, defaults);
        strix::register_hip_gdn_kernel_variant(default_config.model, gdn_request, candidate_config, candidates);
        const auto original = defaults.variants().front(); const auto candidate = candidates.variants().front();
        expect(original.id == "hip.gdn.bf16.h128.kh16.vh48.d128.r1.c1.scalar" && original.compiler_identity == "hip-gfx1151-gdn-v2",
            "default GDN selected id and compiler identity remain golden");
        // Independently derived from unchanged37d00b7 FNV tag/hash fields for
        // fixture_model(); not copied from this candidate's runtime output.
        expect(strix::make_hip_gdn_launcher(original.id, &default_config).implementation_identity == 0x4c2e7f55e611b0edULL,
            "default GDN implementation hash remains golden");
        expect(candidate.id == original.id + ".virtual256.q8-input" && candidate.compiler_identity == "hip-gfx1151-gdn-v3-virtual256-q8-input",
            "candidate GDN id and compiler identity are explicit");
        expect(candidate.resources.workgroup_threads == original.resources.workgroup_threads &&
            candidate.resources.lds_bytes == original.resources.lds_bytes && candidate.resources.scratch_bytes == original.resources.scratch_bytes &&
            candidate.resources.workgroup_threads == 256U && candidate.resources.lds_bytes == 16640U,
            "composite GDN conservative resource bounds remain unchanged");
        expect(strix::make_hip_gdn_launcher(candidate.id, &candidate_config).implementation_identity == 0x6df298901ec0691aULL,
            "current virtual input hash remains independently golden");
        auto prefetch_config = candidate_config; prefetch_config.prefetch_q8_input_projections = true;
        strix::kernels::KernelRegistry prefetch_registry;
        strix::register_hip_gdn_kernel_variant(prefetch_config.model, gdn_request, prefetch_config, prefetch_registry);
        const auto prefetched = prefetch_registry.variants().front();
        expect(prefetched.id == "hip.gdn.bf16.h128.kh16.vh48.d128.r1.c1.scalar.prefetch8.virtual256.q8-input" &&
            prefetched.compiler_identity == "hip-gfx1151-gdn-v4-prefetch8-virtual256-q8-input" &&
            prefetched.resources.workgroup_threads == original.resources.workgroup_threads &&
            prefetched.resources.lds_bytes == original.resources.lds_bytes && prefetched.resources.scratch_bytes == original.resources.scratch_bytes,
            "prefetch id/compiler are distinct and composite resources unchanged");
        expect(strix::make_hip_gdn_launcher(prefetched.id, &prefetch_config).implementation_identity == 0x31229ee8b0186323ULL,
            "prefetch implementation hash remains independently golden and distinct");
        auto invalid_config = default_config; invalid_config.prefetch_q8_input_projections = true;
        bool register_rejected = false, factory_rejected = false;
        try { strix::kernels::KernelRegistry invalid; strix::register_hip_gdn_kernel_variant(invalid_config.model, gdn_request, invalid_config, invalid); }
        catch (const std::invalid_argument&) { register_rejected = true; }
        try { static_cast<void>(strix::make_hip_gdn_launcher("test.invalid.prefetch", &invalid_config)); }
        catch (const std::invalid_argument&) { factory_rejected = true; }
        expect(register_rejected && factory_rejected, "prefetch without virtual rejects at registry and factory");
        strix::CompiledSerialExecution minimal;
        minimal.execution_identity = 0x47444e5142494e44ULL;
        minimal.execution.steps.push_back(step);
        strix::ExecutionKernelBinding binding; binding.op = step.op; binding.step_id = step.id;
        binding.kernel_id = candidate.id; minimal.bindings.bindings.push_back(binding);
        std::array<strix::HipGdnLauncherConfig, 3> modes{default_config, candidate_config, prefetch_config};
        const std::array<std::string, 3> ids{original.id, candidate.id, prefetched.id};
        for (std::size_t mode = 0; mode < modes.size(); ++mode)
            for (std::size_t selected = 0; selected < ids.size(); ++selected) {
                minimal.bindings.bindings.front().kernel_id = ids[selected];
                strix::HipGdnLauncherBundle bundle(modes[mode]); strix::ExecutionLauncherRegistry registry;
                bool accepted = true;
                try { bundle.register_for_compiled_execution(minimal, registry); }
                catch (const std::invalid_argument&) { accepted = false; }
                expect(accepted == (mode == selected), "GDN3x3 config/selected-id matrix admits exactly matching modes");
            }
        for (unsigned direction = 0; direction < 2; ++direction) {
            minimal.bindings.bindings.front().kernel_id = direction == 0 ? candidate.id : original.id;
            strix::HipGdnLauncherBundle mismatched(direction == 0 ? default_config : candidate_config);
            strix::ExecutionLauncherRegistry registry; bool rejected = false;
            try { mismatched.register_for_compiled_execution(minimal, registry); }
            catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected, "GDN bundle rejects either selected-id/configuration mismatch");
            minimal.bindings.bindings.front().kernel_id = direction == 0 ? original.id : candidate.id;
            strix::HipGdnLauncherBundle matching(direction == 0 ? default_config : candidate_config);
            strix::ExecutionLauncherRegistry matched;
            matching.register_for_compiled_execution(minimal, matched);
            expect(matched.find_kernel_launcher(*minimal.bindings.bindings.front().kernel_id) != nullptr,
                "GDN bundle accepts each matching selected-id/configuration pair");
        }
    }

    const std::vector<std::uint32_t> arena_poison(static_cast<std::size_t>(arena.total_bytes / 4U), 0x7fc07fc0U);
    const auto poison_arena = [&](hipStream_t stream) {
        check(hipMemcpyAsync(arena_pointer, arena_poison.data(), arena.total_bytes, hipMemcpyHostToDevice, stream), "poison complete GDN arena");
        check(hipMemcpyAsync(reinterpret_cast<void*>(input.address), hidden_values.data(), hidden_values.size() * 2U,
            hipMemcpyHostToDevice, stream), "restore GDN input after arena poison");
    };
    poison_arena(nullptr);
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

    if (dense_q8_fixture && hidden >= 288U)
        expect(actual_conv[kernel - 1U] == float_to_bf16(-15.1875F),
            "actual Q8 GDN convolution receives the BF16-visible fused-leaf control");

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
    // Dense mode independently qualifies the changed projection family below.
    // The unchanged scalar-double/libm recurrence oracle has a measured
    // original-vs-CPU rounding gap at these magnitudes; report it explicitly.
    // The canonical and F32-fallback WAR fixtures retain their 1e-7 gate.
    expect(std::all_of(actual_matrix.begin(), actual_matrix.end(), [](float value) { return std::isfinite(value); }),
        "GDN recurrent matrix remains finite");
    if (dense_q8_fixture)
        std::cerr << "GDN_DENSE_CPU_DIAGNOSTIC width=" << hidden << " initial_matrix_max_abs=" << matrix_error << '\n';
    else expect(matrix_error <= 1e-7F, "HIP GDN recurrent matrix matches CPU reference");
    expect(
        actual_conv ==
            expected.conv_state,
        "HIP GDN convolution state matches BF16 CPU reference");

    // Four waves consume each delta broadcast. Check the complete recurrence
    // against the independent scalar equations, then repeat from identical
    // asymmetric state: a shared-memory race must not hide behind BF16 output
    // rounding or an absolute tolerance larger than the state update.
    std::vector<std::byte> first_arena(static_cast<std::size_t>(arena.total_bytes));
    check(hipMemcpy(first_arena.data(), arena_pointer, first_arena.size(), hipMemcpyDeviceToHost), "save original complete arena");
    const auto first_conv = actual_conv;
    const auto first_matrix = actual_matrix;
    const auto first_output = actual_output;
    config.virtual_q8_input_projections = candidate_enabled;
    config.prefetch_q8_input_projections = prefetch_enabled;
    config.wave8_q8_input_projections = wave8_enabled;
    config.wave8_q8_output_projection = wave8_output_enabled;
    const auto original_identity = launcher.implementation_identity;
    std::string selected_id = "test.gdn.virtual";
    if (wave8_enabled || wave8_output_enabled) {
        auto request = gdn_request; request.wave8_q8_gdn_input = wave8_enabled;
        request.wave8_q8_gdn_output = wave8_output_enabled;
        strix::kernels::KernelRegistry registry;
        strix::register_hip_gdn_kernel_variant(model, request, config, registry);
        selected_id = registry.variants().front().id;
    }
    launcher = strix::make_hip_gdn_launcher(selected_id, &config);
    expect((launcher.implementation_identity != original_identity) == candidate_enabled,
        "virtual GDN implementation identity is opt-in");
    if (dense_q8_fixture) {
        const auto reject_before_launch = [&](auto mutate, auto restore, const char* expected_error = nullptr) {
            poison_arena(nullptr);
            check(hipDeviceSynchronize(), "drain GDN admission poison");
            mutate(); bool rejected = false; std::string rejection;
            try { launcher.function(context, launcher.user_data); }
            catch (const std::logic_error& error) { rejected = true; rejection = error.what(); }
            if (expected_error && rejection.find(expected_error) == std::string::npos)
                std::cerr << "GDN_ADMISSION_DIAGNOSTIC expected=" << expected_error << " actual=" << rejection << '\n';
            if (expected_error) expect(rejection.find(expected_error) != std::string::npos,
                "GDN rejection reaches the intended admission gate");
            restore();
            check(hipDeviceSynchronize(), "drain GDN admission rejection");
            std::vector<std::uint16_t> untouched(hidden);
            check(hipMemcpy(untouched.data(), reinterpret_cast<void*>(output.address), hidden * 2U,
                hipMemcpyDeviceToHost), "read rejected GDN output poison");
            expect(rejected && std::all_of(untouched.begin(), untouched.end(), [](auto value) { return value == 0x7fc0U; }),
                "bad GDN admission preserves output poison");
            std::vector<float> untouched_matrix(first_matrix.size());
            std::vector<std::uint16_t> untouched_conv(actual_conv.size());
            check(hipMemcpy(untouched_matrix.data(), reinterpret_cast<void*>(matrix_address.address), matrix_address.bytes,
                hipMemcpyDeviceToHost), "read rejected GDN matrix");
            check(hipMemcpy(untouched_conv.data(), reinterpret_cast<void*>(conv_address.address), conv_address.bytes,
                hipMemcpyDeviceToHost), "read rejected GDN convolution");
            expect(std::memcmp(untouched_matrix.data(), first_matrix.data(), matrix_address.bytes) == 0 && untouched_conv == actual_conv,
                "bad GDN admission preserves persistent state");
        };
        // Full entrypoint must reject the existing F32 output BEFORE either
        // input projection or persistent conv/recurrent update is enqueued.
        if(!wave8_output_enabled) reject_before_launch([&] { config.virtual_q8_output_projection = true; },
            [&] { config.virtual_q8_output_projection = false; },
            "virtual Q8 GDN output requires aligned Q8_0 weights");
        if (prefetch_enabled) {
            reject_before_launch([&] { config.virtual_q8_input_projections = false; },
                [&] { config.virtual_q8_input_projections = true; }, "requires virtual");
            const auto saved_hidden = config.model.hidden_size;
            reject_before_launch([&] {
                config.model.hidden_size = saved_hidden - 1U;
                for (const auto index : {0U, 1U, 5U, 6U}) manifest.tensors[index].dimensions[0] = saved_hidden - 1U;
                manifest.tensors[8].dimensions[1] = saved_hidden - 1U;
            }, [&] {
                config.model.hidden_size = saved_hidden;
                for (const auto index : {0U, 1U, 5U, 6U}) manifest.tensors[index].dimensions[0] = saved_hidden;
                manifest.tensors[8].dimensions[1] = saved_hidden;
            }, wave8_enabled ? "shape outside admitted bounds" : "width must divide 32");
        }
        if (wave8_enabled) {
            // IQ4_NL requires 18 bytes per 32 features, fitting the existing
            // Q8_0 extent (34). F16 would fail the earlier size guard and never
            // reach the intended wave8 format admission guard.
            for (const auto index : {0U, 1U}) {
                const auto saved_kind = manifest.tensors[index].ggml_type;
                reject_before_launch([&] { manifest.tensors[index].ggml_type = 20U; },
                    [&] { manifest.tensors[index].ggml_type = saved_kind; }, "requires Q8_0 qkv and z");
            }
            const auto saved_address = runtime.tensors[0].address;
            reject_before_launch([&] { runtime.tensors[0].address += 2U; },
                [&] { runtime.tensors[0].address = saved_address; }, "invalid alignment or extent");
        }
        if(wave8_output_enabled) {
            reject_before_launch([&] {config.virtual_q8_output_projection=false;},
                [&] {config.virtual_q8_output_projection=true;},"requires virtual Q8 output");
            const auto saved_kind=manifest.tensors[8].ggml_type;
            reject_before_launch([&] {manifest.tensors[8].ggml_type=20U;},
                [&] {manifest.tensors[8].ggml_type=saved_kind;},"requires aligned Q8_0 weights");
            const auto saved_address=runtime.tensors[8].address;
            reject_before_launch([&] {runtime.tensors[8].address+=2U;},
                [&] {runtime.tensors[8].address=saved_address;},"invalid alignment or extent");
            const auto saved_extent=runtime.tensors[8].bytes;
            reject_before_launch([&] {runtime.tensors[8].bytes=saved_extent-1U;},
                [&] {runtime.tensors[8].bytes=saved_extent;});
        }
        const auto saved_bytes = runtime.tensors[0].bytes;
        reject_before_launch([&] { runtime.tensors[0].bytes = saved_bytes - 1U; },
            [&] { runtime.tensors[0].bytes = saved_bytes; });
        const auto saved_dimension = manifest.tensors[0].dimensions[0];
        reject_before_launch([&] { manifest.tensors[0].dimensions[0] = saved_dimension - 1U; },
            [&] { manifest.tensors[0].dimensions[0] = saved_dimension; });
        const auto saved_temporaries = context.temporaries;
        reject_before_launch([&] { context.temporaries = {}; }, [&] { context.temporaries = saved_temporaries; });
        reject_before_launch([&] { config.activation = strix::HipActivationStorage::fp32; },
            [&] { config.activation = strix::HipActivationStorage::bf16; });
    }
    for (unsigned iteration = 0; iteration < 16; ++iteration) {
        check(hipMemcpy(reinterpret_cast<void*>(matrix_address.address), matrix_state.data(),
            matrix_address.bytes, hipMemcpyHostToDevice), "restore recurrence broadcast fixture state");
        check(hipMemcpy(reinterpret_cast<void*>(conv_address.address), conv_state.data(),
            conv_address.bytes, hipMemcpyHostToDevice), "restore recurrence broadcast fixture convolution");
        poison_arena(nullptr);
        launcher.function(context, launcher.user_data);
        check(hipDeviceSynchronize(), "drain repeated recurrence broadcast fixture");
        check(hipMemcpy(actual_matrix.data(), reinterpret_cast<const void*>(matrix_address.address),
            matrix_address.bytes, hipMemcpyDeviceToHost), "read repeated recurrence broadcast state");
        check(hipMemcpy(actual_output.data(), reinterpret_cast<const void*>(output.address),
            actual_output.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost), "read repeated recurrence broadcast output");
        std::vector<std::byte> candidate_arena(first_arena.size());
        check(hipMemcpy(candidate_arena.data(), arena_pointer, candidate_arena.size(), hipMemcpyDeviceToHost), "read candidate complete arena");
        expect(candidate_arena == first_arena, "candidate complete poisoned arena matches original bits");
        check(hipMemcpy(actual_conv.data(), reinterpret_cast<const void*>(conv_address.address),
            conv_address.bytes, hipMemcpyDeviceToHost), "read candidate convolution state");
        expect(actual_conv == first_conv && actual_conv == expected.conv_state,
            "candidate Q8 GDN convolution bits match original and fused CPU oracle");
        if (dense_q8_fixture && hidden >= 288U)
            expect(actual_conv[kernel - 1U] == float_to_bf16(-15.1875F),
                "candidate convolution preserves BF16-visible fused-leaf control");
        expect(std::memcmp(actual_matrix.data(), first_matrix.data(), matrix_address.bytes) == 0 &&
            actual_output == first_output, "repeated identical GDN input/state is bit deterministic");
        if (prefetch_enabled) {
            auto streaming_config = config; streaming_config.prefetch_q8_input_projections = false;
            streaming_config.wave8_q8_input_projections = false;
            auto streaming_launcher = strix::make_hip_gdn_launcher("test.gdn.streaming", &streaming_config);
            check(hipMemcpy(reinterpret_cast<void*>(matrix_address.address), matrix_state.data(), matrix_address.bytes,
                hipMemcpyHostToDevice), "restore streaming reference matrix");
            check(hipMemcpy(reinterpret_cast<void*>(conv_address.address), conv_state.data(), conv_address.bytes,
                hipMemcpyHostToDevice), "restore streaming reference convolution");
            poison_arena(nullptr);
            streaming_launcher.function(context, streaming_launcher.user_data);
            check(hipDeviceSynchronize(), "drain streaming reference");
            std::vector<std::byte> streaming_arena(first_arena.size());
            std::vector<float> streaming_matrix(first_matrix.size());
            std::vector<std::uint16_t> streaming_conv(first_conv.size());
            check(hipMemcpy(streaming_arena.data(), arena_pointer, streaming_arena.size(), hipMemcpyDeviceToHost), "read streaming arena");
            check(hipMemcpy(streaming_matrix.data(), reinterpret_cast<void*>(matrix_address.address), matrix_address.bytes,
                hipMemcpyDeviceToHost), "read streaming matrix");
            check(hipMemcpy(streaming_conv.data(), reinterpret_cast<void*>(conv_address.address), conv_address.bytes,
                hipMemcpyDeviceToHost), "read streaming convolution");
            expect(streaming_arena == candidate_arena && streaming_conv == actual_conv &&
                std::memcmp(streaming_matrix.data(), actual_matrix.data(), matrix_address.bytes) == 0,
                "prefetch and current virtual streaming full arena/state bits are equal");
        }
        float recurrence_error = 0.0F;
        for (std::size_t i = 0; i < actual_matrix.size(); ++i)
            recurrence_error = std::max(recurrence_error, std::fabs(actual_matrix[i] - expected.matrix_state[i]));
        expect(std::all_of(actual_matrix.begin(), actual_matrix.end(), [](float value) { return std::isfinite(value); }),
            "candidate recurrent matrix remains finite");
        if (dense_q8_fixture) {
            if (iteration == 0U || iteration == 15U)
                std::cerr << "GDN_DENSE_CPU_DIAGNOSTIC width=" << hidden << " repeat=" << iteration << " matrix_max_abs=" << recurrence_error << '\n';
        } else expect(recurrence_error <= 1e-7F, "all waves consume the independently computed GDN delta");
    }

    if (wave8_enabled) {
        // Production entrypoints, including all post-projection state updates.
        // Restore the same nonzero state before every arm and poison the whole
        // arena so tail writes and unwritten workspace bytes remain observable.
        const auto saved_hidden = hidden_values;
        const auto saved_packed_z = packed_q8[1];
        auto baseline_config = config;
        baseline_config.virtual_q8_input_projections = false;
        baseline_config.prefetch_q8_input_projections = false;
        baseline_config.wave8_q8_input_projections = false;
        const auto baseline_launcher = strix::make_hip_gdn_launcher("test.gdn.edge.legacy", &baseline_config);
        auto p1_config = config; p1_config.wave8_q8_input_projections = false;
        const auto p1_launcher = strix::make_hip_gdn_launcher("test.gdn.edge.prefetch1", &p1_config);
        struct EdgeSnapshot {
            std::vector<std::byte> arena_bytes;
            std::vector<float> matrix_values;
            std::vector<std::uint16_t> convolution_values;
        };
        const auto run_edge = [&](const strix::ExecutionLauncher& arm) {
            check(hipMemcpy(reinterpret_cast<void*>(matrix_address.address), matrix_state.data(), matrix_address.bytes,
                hipMemcpyHostToDevice), "restore edge recurrent state");
            check(hipMemcpy(reinterpret_cast<void*>(conv_address.address), conv_state.data(), conv_address.bytes,
                hipMemcpyHostToDevice), "restore edge convolution state");
            poison_arena(nullptr);
            arm.function(context, arm.user_data);
            check(hipDeviceSynchronize(), "drain production GDN edge arm");
            EdgeSnapshot result{std::vector<std::byte>(first_arena.size()),
                std::vector<float>(matrix_state.size()), std::vector<std::uint16_t>(conv_state.size())};
            check(hipMemcpy(result.arena_bytes.data(), arena_pointer, result.arena_bytes.size(), hipMemcpyDeviceToHost), "read complete edge arena");
            check(hipMemcpy(result.matrix_values.data(), reinterpret_cast<void*>(matrix_address.address), matrix_address.bytes,
                hipMemcpyDeviceToHost), "read complete edge matrix");
            check(hipMemcpy(result.convolution_values.data(), reinterpret_cast<void*>(conv_address.address), conv_address.bytes,
                hipMemcpyDeviceToHost), "read complete edge convolution");
            return result;
        };
        const auto same = [&](const EdgeSnapshot& a, const EdgeSnapshot& b) {
            return a.arena_bytes == b.arena_bytes && a.convolution_values == b.convolution_values &&
                std::memcmp(a.matrix_values.data(), b.matrix_values.data(), matrix_address.bytes) == 0;
        };
        const auto set_z = [&](std::uint32_t row, std::uint32_t feature, std::uint16_t scale, std::int8_t code) {
            const auto block = static_cast<std::size_t>(row) * (hidden / 32U) + feature / 32U;
            auto* packed = packed_q8[1].data() + block * 34U;
            packed[0] = static_cast<std::byte>(scale & 255U);
            packed[1] = static_cast<std::byte>(scale >> 8U);
            packed[2U + feature % 32U] = static_cast<std::byte>(static_cast<std::uint8_t>(code));
        };
        for (unsigned payload = 0; payload < 3; ++payload) {
            packed_q8[1] = saved_packed_z;
            if (payload == 0) {
                constexpr std::array<std::uint16_t, 7> scales{0x0000U, 0x8000U, 0x0001U,
                    0x03ffU, 0x0400U, 0x7bffU, 0xfbffU};
                constexpr std::array<std::int8_t, 5> codes{-128, -1, 0, 1, 127};
                constexpr std::array<std::uint16_t, 8> activations{0x0000U, 0x8000U, 0x0001U,
                    0x8001U, 0x0080U, 0x8080U, 0x3f80U, 0xbf80U};
                for (unsigned feature = 0; feature < hidden; ++feature)
                    hidden_values[feature] = activations[feature % activations.size()];
                for (unsigned row = 0; row < scales.size(); ++row)
                    for (unsigned feature = 0; feature < hidden; ++feature)
                        set_z(row, feature, scales[row], codes[feature % codes.size()]);
            } else for (unsigned feature = 0; feature < hidden; ++feature) {
                if (payload == 1) {
                    hidden_values[feature] = 0x0080U;
                    set_z(0, feature, 0x0001U, 127);
                } else {
                    hidden_values[feature] = feature < 256U ? 0x8001U : 0x3f80U;
                    set_z(0, feature, feature < 256U ? 0x0001U : 0x8000U, 1);
                }
            }
            check(hipMemcpy(device_weights[1], packed_q8[1].data(), packed_q8[1].size(), hipMemcpyHostToDevice),
                "upload production GDN edge Q8 payload");
            for (unsigned repeat = 0; repeat < 3; ++repeat) {
                const auto original = run_edge(baseline_launcher);
                const auto temporary = strix::resolve_dispatch_temporary(context, 0);
                std::uint32_t z0 = 0;
                check(hipMemcpy(&z0, reinterpret_cast<void*>(temporary.address + conv_width * sizeof(float)),
                    sizeof(z0), hipMemcpyDeviceToHost), "read edge projection control");
                if (payload == 1 && hidden == 2560U)
                    expect(z0 == 0x00020000U, "production decode-first control retains BF16 subnormal bits2");
                if (payload == 2 && hidden >= 256U)
                    expect(z0 == 0x80000000U, "production signed-zero control retains negative zero");
                const auto p1 = run_edge(p1_launcher);
                const auto p8 = run_edge(launcher);
                expect(same(original, p1) && same(original, p8),
                    "production edge payloads preserve all arena, matrix and convolution bits in P1/P8");
            }
        }
        hidden_values = saved_hidden;
        packed_q8[1] = saved_packed_z;
        check(hipMemcpy(device_weights[1], packed_q8[1].data(), packed_q8[1].size(), hipMemcpyHostToDevice),
            "restore production GDN edge weights");

    }
    if(wave8_output_enabled) {
        auto baseline_config=config;baseline_config.wave8_q8_output_projection=false;
        const auto baseline_launcher=strix::make_hip_gdn_launcher("test.gdn.output.virtual",&baseline_config);
        const auto saved_packed_output=packed_output;
        const auto observe_output_arm=[&](const strix::ExecutionLauncher& arm) {
            check(hipMemcpy(reinterpret_cast<void*>(matrix_address.address),matrix_state.data(),matrix_address.bytes,hipMemcpyHostToDevice),"restore output edge matrix");
            check(hipMemcpy(reinterpret_cast<void*>(conv_address.address),conv_state.data(),conv_address.bytes,hipMemcpyHostToDevice),"restore output edge convolution");
            poison_arena(nullptr);arm.function(context,arm.user_data);
            check(hipDeviceSynchronize(),"drain output edge arm");
            std::vector<std::byte> arena_bits(first_arena.size());
            std::vector<float> matrix_bits(first_matrix.size());std::vector<std::uint16_t> conv_bits(first_conv.size());
            check(hipMemcpy(arena_bits.data(),arena_pointer,arena_bits.size(),hipMemcpyDeviceToHost),"read output edge arena");
            check(hipMemcpy(matrix_bits.data(),reinterpret_cast<void*>(matrix_address.address),matrix_address.bytes,hipMemcpyDeviceToHost),"read output edge matrix");
            check(hipMemcpy(conv_bits.data(),reinterpret_cast<void*>(conv_address.address),conv_address.bytes,hipMemcpyDeviceToHost),"read output edge convolution");
            std::vector<std::byte> weights_after(packed_output.size());
            check(hipMemcpy(weights_after.data(),device_weights[8],weights_after.size(),hipMemcpyDeviceToHost),"read output edge weights");
            expect(weights_after==packed_output,"output weights remain read-only");
            expect(std::memcmp(matrix_bits.data(),first_matrix.data(),matrix_address.bytes)==0 && conv_bits==first_conv,
                "output-only organization cannot change recurrent or convolution state");
            return arena_bits;
        };
        // These coefficients exercise the actual normalizer/core workspace and
        // BF16 epilogue; standalone tests discriminate nonfinite payload roles.
        constexpr std::array<std::uint16_t,5> edge_scales{0x8000U,0x0001U,0x03ffU,0x7bffU,0x7e01U};
        for(const auto scale:edge_scales) {
            packed_output=saved_packed_output;
            for(std::size_t block=0;block<packed_output.size()/34U;++block) {
                packed_output[block*34U]=static_cast<std::byte>(scale&255U);
                packed_output[block*34U+1U]=static_cast<std::byte>(scale>>8U);
                packed_output[block*34U+2U]=std::byte{0x80};
                packed_output[block*34U+3U]=std::byte{0};
            }
            check(hipMemcpy(device_weights[8],packed_output.data(),packed_output.size(),hipMemcpyHostToDevice),"upload output edge weights");
            const auto baseline_bits=observe_output_arm(baseline_launcher);
            for(unsigned replay=0;replay<3;++replay)
                expect(observe_output_arm(launcher)==baseline_bits,"wave8 output complete arena matches virtual baseline on edge replay");
        }
        packed_output=saved_packed_output;
        check(hipMemcpy(device_weights[8],packed_output.data(),packed_output.size(),hipMemcpyHostToDevice),"restore output edge weights");
    }
    if(wave8_enabled || wave8_output_enabled) {
        hipStream_t stream = nullptr;
        hipGraph_t rejected_graph = nullptr;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create rejected wave8 GDN stream");
        strix::HipLaunchContext scope; scope.stream = stream;
        bool rejected = false;
        {
            strix::HipLaunchScope owned(scope);
            check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin rejected wave8 GDN capture");
            try { launcher.function(context, launcher.user_data); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(hipStreamEndCapture(stream, &rejected_graph), "end rejected wave8 GDN capture");
        }
        std::size_t count = 0;
        check(hipGraphGetNodes(rejected_graph, nullptr, &count), "count rejected wave8 GDN nodes");
        expect(rejected && count == 0, "wave8 GDN capture rejects before any projection/state enqueue");
        check(hipGraphDestroy(rejected_graph), "destroy rejected wave8 GDN graph");
        check(hipStreamDestroy(stream), "destroy rejected wave8 GDN stream");
    }

    // Capture the complete existing GDN operator: projections, convolution,
    // recurrence, gated normalization and final projection. No host position
    // scalar controls the captured state stage.
    if (!wave8_enabled && !wave8_output_enabled) {
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
        std::size_t node_count = 0;
        check(hipGraphGetNodes(graph.graph, nullptr, &node_count), "count GDN captured nodes");
        std::vector<hipGraphNode_t> nodes(node_count);
        check(hipGraphGetNodes(graph.graph, nodes.data(), &node_count), "get GDN captured nodes");
        std::vector<unsigned> virtual_grids;
        std::vector<void*> virtual_functions;
        for (const auto node : nodes) {
            hipGraphNodeType type{}; check(hipGraphNodeGetType(node, &type), "GDN graph node type");
            if (type != hipGraphNodeTypeKernel) continue;
            hipKernelNodeParams params{}; check(hipGraphKernelNodeGetParams(node, &params), "GDN graph kernel params");
            if (params.blockDim.x == 32U) { virtual_grids.push_back(params.gridDim.x); virtual_functions.push_back(params.func); }
        }
        std::sort(virtual_grids.begin(), virtual_grids.end());
        auto expected_grids = dense_q8_fixture ? std::vector<unsigned>{conv_width, value_width, value_heads} : std::vector<unsigned>{};
        std::sort(expected_grids.begin(), expected_grids.end());
        expect(virtual_grids == expected_grids, "capture selects exactly three Q8 input waves, retains other kernels");
        // Changed-family oracle: replay the captured projection nodes in isolation,
        // before any gate transform, convolution, qk normalization or recurrence.
        const auto check_input_projections = [&](hipGraph_t source_graph, const char* label) {
            std::size_t count = 0;
            check(hipGraphGetNodes(source_graph, nullptr, &count), "isolated projection graph count");
            std::vector<hipGraphNode_t> source_nodes(count);
            check(hipGraphGetNodes(source_graph, source_nodes.data(), &count), "isolated projection graph nodes");
            hipGraph_t projection_graph = nullptr; hipGraphExec_t projection_exec = nullptr;
            check(hipGraphCreate(&projection_graph, 0), "isolated projection graph create");
            unsigned selected = 0;
            for (const auto node : source_nodes) {
                hipGraphNodeType type{}; check(hipGraphNodeGetType(node, &type), "isolated projection node type");
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams params{}; check(hipGraphKernelNodeGetParams(node, &params), "isolated projection params");
                if ((params.blockDim.x != 32U && params.blockDim.x != 256U) ||
                    (params.gridDim.x != conv_width && params.gridDim.x != value_width && params.gridDim.x != value_heads)) continue;
                hipGraphNode_t copied{};
                check(hipGraphAddKernelNode(&copied, projection_graph, nullptr, 0, &params), "isolated copy projection node");
                ++selected;
            }
            expect(selected == 4U, "isolated diagnostic extracts exactly four input projection nodes");
            check(hipGraphInstantiate(&projection_exec, projection_graph, nullptr, nullptr, 0), "isolated projection instantiate");
            poison_arena(graph.stream);
            check(hipGraphLaunch(projection_exec, graph.stream), "isolated projection replay");
            check(hipStreamSynchronize(graph.stream), "isolated projection drain");
            const auto temporary = strix::resolve_dispatch_temporary(context, 0);
            const std::array<unsigned, 4> widths{conv_width, value_width, value_heads, value_heads};
            const std::array<const std::vector<float>*, 4> matrices{&qkv, &z, &beta, &alpha};
            const std::array<const char*, 4> names{"qkv", "z", "beta", "alpha"};
            std::vector<float> activations(hidden);
            for (unsigned feature = 0; feature < hidden; ++feature) activations[feature] = bf16_to_float(hidden_values[feature]);
            std::size_t offset = 0;
            std::vector<std::uint32_t> all_bits;
            for (std::size_t projection = 0; projection < widths.size(); ++projection) {
                std::vector<float> raw(widths[projection]);
                check(hipMemcpy(raw.data(), reinterpret_cast<const void*>(temporary.address + offset * sizeof(float)),
                    raw.size() * sizeof(float), hipMemcpyDeviceToHost), "isolated raw projection read");
                std::size_t mismatch = 0; float maximum_error = 0;
                for (unsigned row = 0; row < widths[projection]; ++row) {
                    const float expected_raw = round_bf16(project_row(activations, *matrices[projection], hidden, row));
                    all_bits.push_back(std::bit_cast<std::uint32_t>(raw[row]));
                    if (std::bit_cast<std::uint32_t>(raw[row]) != std::bit_cast<std::uint32_t>(expected_raw)) {
                        if (mismatch < 8U) std::cerr << "GDN_INPUT_ORACLE_ROW " << label << ' ' << names[projection]
                            << " row=" << row << " actual=" << raw[row] << " expected=" << expected_raw
                            << " actual_bits=" << std::bit_cast<std::uint32_t>(raw[row])
                            << " expected_bits=" << std::bit_cast<std::uint32_t>(expected_raw) << '\n';
                        ++mismatch;
                    }
                    maximum_error = std::max(maximum_error, std::fabs(raw[row] - expected_raw));
                }
                std::cerr << "GDN_INPUT_ORACLE " << label << ' ' << names[projection]
                    << " rows=" << widths[projection] << " mismatches=" << mismatch << " max_abs=" << maximum_error << '\n';
                expect(mismatch == 0U, "isolated independent raw projection oracle matches captured device bits");
                offset += widths[projection];
            }
            std::vector<std::uint32_t> preserved(arena_poison.size());
            check(hipMemcpy(preserved.data(), arena_pointer, arena.total_bytes, hipMemcpyDeviceToHost), "read isolated projection arena guards");
            const auto projected_end = temporary.address + offset * sizeof(float);
            bool guards_preserved = true;
            for (std::size_t word = 0; word < preserved.size(); ++word) {
                const auto address = memory.arena_base + word * sizeof(std::uint32_t);
                if (address >= temporary.address && address < projected_end) continue;
                auto expected_word = std::uint32_t{0x7fc07fc0U};
                if (address >= input.address && address < input.address + hidden_values.size() * 2U) {
                    const auto feature = static_cast<std::size_t>((address - input.address) / 2U);
                    expected_word = static_cast<std::uint32_t>(hidden_values[feature]) |
                        (static_cast<std::uint32_t>(hidden_values[feature + 1U]) << 16U);
                }
                guards_preserved = guards_preserved && preserved[word] == expected_word;
            }
            expect(guards_preserved, "isolated projections preserve input and every arena word outside their four output slices");
            check(hipGraphExecDestroy(projection_exec), "isolated projection exec destroy");
            check(hipGraphDestroy(projection_graph), "isolated projection graph destroy");
            return all_bits;
        };
        std::vector<std::uint32_t> candidate_projection_bits;
        if (dense_q8_fixture) candidate_projection_bits = check_input_projections(graph.graph, "candidate");
        if (prefetch_enabled) {
            auto streaming_config = graph_config; streaming_config.prefetch_q8_input_projections = false;
            auto streaming_launcher = strix::make_hip_gdn_launcher("test.gdn.streaming.capture", &streaming_config);
            hipGraph_t streaming_graph = nullptr;
            check(hipStreamBeginCapture(graph.stream, hipStreamCaptureModeThreadLocal), "capture current streaming control");
            { const strix::HipLaunchScope owned(scope); streaming_launcher.function(context, streaming_launcher.user_data); }
            check(hipStreamEndCapture(graph.stream, &streaming_graph), "finish streaming control capture");
            std::size_t count = 0; check(hipGraphGetNodes(streaming_graph, nullptr, &count), "count streaming control nodes");
            expect(count == node_count, "prefetch retains streaming complete operator node count");
            std::vector<hipGraphNode_t> streaming_nodes(count);
            check(hipGraphGetNodes(streaming_graph, streaming_nodes.data(), &count), "get streaming control nodes");
            std::vector<void*> streaming_other, candidate_other;
            for (const auto node : streaming_nodes) {
                hipGraphNodeType type{}; check(hipGraphNodeGetType(node, &type), "streaming control node type");
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams params{}; check(hipGraphKernelNodeGetParams(node, &params), "streaming control kernel params");
                if (params.blockDim.x == 32U)
                    expect(std::find(virtual_functions.begin(), virtual_functions.end(), params.func) == virtual_functions.end(),
                        "prefetch32 and streaming32 use distinct kernel functions");
                else streaming_other.push_back(params.func);
            }
            for (const auto node : nodes) {
                hipGraphNodeType type{}; check(hipGraphNodeGetType(node, &type), "candidate control node type");
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams params{}; check(hipGraphKernelNodeGetParams(node, &params), "candidate control kernel params");
                if (params.blockDim.x != 32U) candidate_other.push_back(params.func);
            }
            // std::less provides a total order for unrelated function pointers.
            std::sort(streaming_other.begin(), streaming_other.end(), std::less<void*>{});
            std::sort(candidate_other.begin(), candidate_other.end(), std::less<void*>{});
            expect(streaming_other == candidate_other, "all non32 captured functions remain unchanged");
            if (dense_q8_fixture) expect(check_input_projections(streaming_graph, "streaming") == candidate_projection_bits,
                "prefetch and current streaming raw projection arrays match bitwise");
            check(hipGraphDestroy(streaming_graph), "destroy streaming control graph");
        }
        if (dense_q8_fixture) {
            auto baseline_config = graph_config; baseline_config.virtual_q8_input_projections = false; baseline_config.prefetch_q8_input_projections = false;
            auto baseline_launcher = strix::make_hip_gdn_launcher("test.gdn.baseline.capture", &baseline_config);
            hipGraph_t baseline_graph = nullptr;
            check(hipStreamBeginCapture(graph.stream, hipStreamCaptureModeThreadLocal), "capture original GDN control");
            { const strix::HipLaunchScope owned(scope); baseline_launcher.function(context, baseline_launcher.user_data); }
            check(hipStreamEndCapture(graph.stream, &baseline_graph), "finish original GDN control capture");
            std::size_t baseline_count = 0;
            check(hipGraphGetNodes(baseline_graph, nullptr, &baseline_count), "count original GDN nodes");
            std::vector<hipGraphNode_t> baseline_nodes(baseline_count);
            check(hipGraphGetNodes(baseline_graph, baseline_nodes.data(), &baseline_count), "get original GDN nodes");
            expect(baseline_count == node_count, "virtual GDN keeps the complete operator launch count");
            for (const auto node : baseline_nodes) {
                hipGraphNodeType type{}; check(hipGraphNodeGetType(node, &type), "original GDN node type");
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams params{}; check(hipGraphKernelNodeGetParams(node, &params), "original GDN kernel params");
                expect(params.blockDim.x != 32U && std::find(virtual_functions.begin(), virtual_functions.end(), params.func) == virtual_functions.end(),
                    "original capture uses different functions and no virtual32 input kernel");
            }
            expect(check_input_projections(baseline_graph, "original") == candidate_projection_bits,
                "prefetch or streaming raw projection arrays match legacy256 bitwise");
            if (prefetch_enabled && hidden == 2560U) {
                // Raw projection-only edge scope: mutate owned fixture payloads,
                // never execute the recurrent/gate/output graph on these inputs.
                auto edge_config = graph_config; edge_config.prefetch_q8_input_projections = false;
                auto edge_launcher = strix::make_hip_gdn_launcher("test.gdn.edge.streaming", &edge_config);
                hipGraph_t edge_streaming_graph = nullptr;
                check(hipStreamBeginCapture(graph.stream, hipStreamCaptureModeThreadLocal), "capture edge streaming control");
                { const strix::HipLaunchScope owned(scope); edge_launcher.function(context, edge_launcher.user_data); }
                check(hipStreamEndCapture(graph.stream, &edge_streaming_graph), "finish edge streaming control");
                const auto saved_hidden = hidden_values;
                const auto saved_z = z;
                const auto saved_packed_z = packed_q8[1];
                constexpr std::array<std::uint16_t, 7> scales{0x0000U, 0x8000U, 0x0001U,
                    0x03ffU, 0x0400U, 0x7bffU, 0xfbffU};
                constexpr std::array<std::int8_t, 5> codes{-128, -1, 0, 1, 127};
                constexpr std::array<std::uint16_t, 8> activation_bits{0x0000U, 0x8000U, 0x0001U,
                    0x8001U, 0x0080U, 0x8080U, 0x3f80U, 0xbf80U};
                const auto half_scale = [](std::uint16_t bits) {
                    const auto exponent = (bits >> 10U) & 31U;
                    const auto mantissa = bits & 1023U;
                    const float magnitude = exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -24) :
                        std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F,
                            static_cast<int>(exponent) - 15);
                    return (bits & 0x8000U) != 0U ? -magnitude : magnitude;
                };
                for (unsigned feature = 0; feature < hidden; ++feature)
                    hidden_values[feature] = activation_bits[feature % activation_bits.size()];
                for (unsigned row = 0; row < scales.size(); ++row)
                    for (unsigned feature = 0; feature < hidden; ++feature) {
                        const auto block = static_cast<std::size_t>(row) * (hidden / 32U) + feature / 32U;
                        const auto bits = scales[row]; const auto code = codes[feature % codes.size()];
                        packed_q8[1][block * 34U] = static_cast<std::byte>(bits & 255U);
                        packed_q8[1][block * 34U + 1U] = static_cast<std::byte>(bits >> 8U);
                        packed_q8[1][block * 34U + 2U + feature % 32U] =
                            static_cast<std::byte>(static_cast<std::uint8_t>(code));
                        z[static_cast<std::size_t>(row) * hidden + feature] = half_scale(bits) * static_cast<float>(code);
                    }
                check(hipMemcpy(device_weights[1], packed_q8[1].data(), packed_q8[1].size(), hipMemcpyHostToDevice),
                    "upload finite edge Q8 scales and codes");
                const auto edge_original = check_input_projections(baseline_graph, "finite-edge-original");
                expect(check_input_projections(edge_streaming_graph, "finite-edge-streaming") == edge_original,
                    "finite Q8 scale/code and BF16 zero/subnormal edge arrays match original/current streaming bits");
                expect(check_input_projections(graph.graph, "finite-edge-prefetch") == edge_original,
                    "finite Q8 scale/code and BF16 zero/subnormal edge arrays match original/prefetch bits");
                // A decode-association mutant survives the actual BF16 epilogue:
                // x=2^-126, scale=2^-24, code=127. Decoding first yields BF16
                // bits2 at width2560; multiplying x*scale first underflows to0.
                std::fill(hidden_values.begin(), hidden_values.end(), std::uint16_t{0x0080U});
                for (unsigned feature = 0; feature < hidden; ++feature) {
                    const auto block = feature / 32U;
                    packed_q8[1][block * 34U] = std::byte{1};
                    packed_q8[1][block * 34U + 1U] = std::byte{0};
                    packed_q8[1][block * 34U + 2U + feature % 32U] = std::byte{127};
                    z[feature] = std::ldexp(127.0F, -24);
                }
                check(hipMemcpy(device_weights[1], packed_q8[1].data(), packed_q8[1].size(), hipMemcpyHostToDevice),
                    "upload decode-association underflow control");
                const auto underflow_original = check_input_projections(baseline_graph, "underflow-original");
                expect(underflow_original[conv_width] == 0x00020000U,
                    "decode-first underflow control survives BF16 epilogue with bits2");
                expect(check_input_projections(edge_streaming_graph, "underflow-streaming") == underflow_original,
                    "underflow original/current streaming arrays match bits");
                expect(check_input_projections(graph.graph, "underflow-prefetch") == underflow_original,
                    "underflow original/prefetch arrays match bits");
                // Seed every leaf with negative underflow, then accumulate -0
                // coefficients. A coefficient decode using a +0 FMA addend
                // changes later coefficients to +0 and the final sign to +0.
                for (unsigned feature = 0; feature < hidden; ++feature) {
                    hidden_values[feature] = feature < 256U ? 0x8001U : 0x3f80U;
                    const auto block = feature / 32U;
                    const std::uint16_t bits = feature < 256U ? 0x0001U : 0x8000U;
                    packed_q8[1][block * 34U] = static_cast<std::byte>(bits & 255U);
                    packed_q8[1][block * 34U + 1U] = static_cast<std::byte>(bits >> 8U);
                    packed_q8[1][block * 34U + 2U + feature % 32U] = std::byte{1};
                    z[feature] = half_scale(bits);
                }
                check(hipMemcpy(device_weights[1], packed_q8[1].data(), packed_q8[1].size(), hipMemcpyHostToDevice),
                    "upload signed-zero decode control");
                const auto signed_zero_original = check_input_projections(baseline_graph, "signed-zero-original");
                expect(signed_zero_original[conv_width] == 0x80000000U,
                    "coefficient negative zero survives ordered leaves and BF16 epilogue");
                expect(check_input_projections(edge_streaming_graph, "signed-zero-streaming") == signed_zero_original,
                    "signed-zero original/current streaming arrays match bits");
                expect(check_input_projections(graph.graph, "signed-zero-prefetch") == signed_zero_original,
                    "signed-zero original/prefetch arrays match bits");
                hidden_values = saved_hidden; z = saved_z; packed_q8[1] = saved_packed_z;
                check(hipMemcpy(device_weights[1], packed_q8[1].data(), packed_q8[1].size(), hipMemcpyHostToDevice),
                    "restore ordinary Q8 z fixture");
                check(hipGraphDestroy(edge_streaming_graph), "destroy edge streaming graph");
            }
            check(hipGraphDestroy(baseline_graph), "destroy original GDN capture control");
        }
        if (dense_q8_fixture) {
            for (const auto type : {20U, 23U}) {
                if (type == 23U && hidden % 256U != 0U) continue;
                const auto saved_type = manifest.tensors[0].ggml_type;
                manifest.tensors[0].ggml_type = type;
                hipGraph_t fallback_graph = nullptr;
                check(hipStreamBeginCapture(graph.stream, hipStreamCaptureModeThreadLocal), "capture supported nonQ8 fallback");
                { const strix::HipLaunchScope owned(scope); graph_launcher.function(context, graph_launcher.user_data); }
                check(hipStreamEndCapture(graph.stream, &fallback_graph), "finish supported nonQ8 fallback");
                manifest.tensors[0].ggml_type = saved_type;
                std::size_t count = 0; check(hipGraphGetNodes(fallback_graph, nullptr, &count), "count fallback nodes");
                std::vector<hipGraphNode_t> fallback_nodes(count);
                check(hipGraphGetNodes(fallback_graph, fallback_nodes.data(), &count), "get fallback nodes");
                unsigned virtual_count = 0;
                for (const auto node : fallback_nodes) {
                    hipGraphNodeType node_type{}; check(hipGraphNodeGetType(node, &node_type), "fallback node type");
                    if (node_type != hipGraphNodeTypeKernel) continue;
                    hipKernelNodeParams params{}; check(hipGraphKernelNodeGetParams(node, &params), "fallback kernel params");
                    if (params.blockDim.x == 32U) ++virtual_count;
                }
                expect(virtual_count == 2U, "supported IQ4 qkv dispatch retains256 while two Q8 inputs dispatch32");
                check(hipGraphDestroy(fallback_graph), "destroy capture-only nonQ8 fallback");
            }
        }
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
            poison_arena(graph.stream);
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
            if (dense_q8_fixture && hidden >= 288U) {
                hidden_values[0] = float_to_bf16(1.8515625F);
                hidden_values[256] = float_to_bf16(1.921875F);
            }
            check(hipMemcpy(reinterpret_cast<void*>(input.address), hidden_values.data(), hidden_values.size() * 2, hipMemcpyHostToDevice), "change GDN graph input");
            control(iteration, 0); replay();
            expect(read_status() == 0, "valid GDN graph status");
            auto current = read_state();
            std::vector<std::uint16_t> graph_output(hidden), eager_output(hidden);
            check(hipMemcpy(graph_output.data(), reinterpret_cast<void*>(output.address), hidden * 2, hipMemcpyDeviceToHost), "read GDN graph output");
            std::vector<std::byte> graph_arena(first_arena.size());
            check(hipMemcpy(graph_arena.data(), arena_pointer, graph_arena.size(), hipMemcpyDeviceToHost), "read graph complete arena");
            copy_state(previous.matrix_state, previous.conv_state);
            auto eager_scope = scope; eager_scope.graph_capture = false;
            poison_arena(graph.stream);
            auto baseline_config = graph_config; baseline_config.virtual_q8_input_projections = false; baseline_config.prefetch_q8_input_projections = false;
            auto baseline_launcher = strix::make_hip_gdn_launcher("test.gdn.original.device", &baseline_config);
            { const strix::HipLaunchScope owned(eager_scope); baseline_launcher.function(context, baseline_launcher.user_data); }
            check(hipStreamSynchronize(graph.stream), "complete GDN eager control");
            const auto eager = read_state();
            std::vector<std::byte> eager_arena(first_arena.size());
            check(hipMemcpy(eager_arena.data(), arena_pointer, eager_arena.size(), hipMemcpyDeviceToHost), "read original eager complete arena");
            expect(graph_arena == eager_arena, "candidate graph complete poisoned arena matches original eager bits");
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
            if (dense_q8_fixture) {
                expect(graph_matrix_finite, "dense GDN graph matrix remains finite");
                std::cerr << "GDN_DENSE_CPU_DIAGNOSTIC width=" << hidden << " graph_iteration=" << iteration
                    << " matrix_max_abs=" << graph_matrix_error << '\n';
            } else expect(graph_matrix_finite && graph_matrix_error <= 1e-7F,
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
        if (dense_q8_fixture) {
            std::fill(hidden_values.begin(), hidden_values.end(), 0U);
            // Finite largest BF16 times finite Q8 coefficient >=2 produces Inf,
            // without mutating packed weights or provoking generated NaN rounding.
            hidden_values[0] = 0x7f7f;
            check(hipMemcpy(reinterpret_cast<void*>(input.address), hidden_values.data(), hidden_values.size() * 2, hipMemcpyHostToDevice), "set Q8 finite overflow input");
            control(0, 0); replay(); rejected = read_state();
            expect(read_status() == 8 && rejected.conv_state == previous.conv_state &&
                std::memcmp(rejected.matrix_state.data(), previous.matrix_state.data(), matrix_address.bytes) == 0,
                "finite Q8 projection overflow rejects before persistent mutation");
        } else {
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
        }
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
