#include "strix/execution_arena.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
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

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);

void set_iq4_scale(
    IQ4XSBlock& block,
    std::uint32_t subblock,
    std::uint8_t value) {
    const auto low =
        static_cast<std::uint8_t>(
            value & 0x0fU);
    const auto high =
        static_cast<std::uint16_t>(
            (value >> 4U) & 0x03U);

    auto& byte =
        block.scales_l[
            subblock / 2U];
    if ((subblock % 2U) == 0U) {
        byte =
            static_cast<std::uint8_t>(
                (byte & 0xf0U) |
                low);
    } else {
        byte =
            static_cast<std::uint8_t>(
                (byte & 0x0fU) |
                static_cast<std::uint8_t>(
                    low << 4U));
    }

    const auto shift =
        2U * subblock;
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h &
            static_cast<std::uint16_t>(
                ~(0x03U << shift)));
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h |
            static_cast<std::uint16_t>(
                high << shift));
}

std::vector<IQ4XSBlock> make_iq4_rows(
    std::uint32_t rows,
    float base_scale) {
    std::vector<IQ4XSBlock> values(rows);
    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        auto& block = values[row];
        block.d =
            __float2half(
                base_scale *
                static_cast<float>(
                    row + 1U));
        block.scales_h = 0;
        std::fill(
            std::begin(block.scales_l),
            std::end(block.scales_l),
            std::uint8_t{0});

        for (std::uint32_t subblock = 0;
             subblock < 8U;
             ++subblock) {
            set_iq4_scale(
                block,
                subblock,
                33U);
            for (std::uint32_t byte = 0;
                 byte < 16U;
                 ++byte) {
                // code 8 maps to +1; both nibbles are identical.
                block.qs[
                    subblock * 16U +
                    byte] =
                    0x88U;
            }
        }
    }
    return values;
}

std::vector<Q8Block> make_q8_rows(
    std::uint32_t rows,
    float base_scale) {
    std::vector<Q8Block> values(rows);
    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        values[row].scale =
            __float2half(
                base_scale *
                static_cast<float>(
                    row % 17U + 1U));
        std::fill(
            std::begin(values[row].values),
            std::end(values[row].values),
            std::int8_t{1});
    }
    return values;
}

template <typename T>
void upload(
    const std::vector<T>& values,
    void** device) {
    check(
        hipMalloc(
            device,
            values.size() *
                sizeof(T)),
        "hipMalloc hyper-mix tensor");
    check(
        hipMemcpy(
            *device,
            values.data(),
            values.size() *
                sizeof(T),
            hipMemcpyHostToDevice),
        "copy hyper-mix tensor");
}

const strix::ExecutionStep& first_step(
    const strix::SerialExecutionPlan& plan,
    strix::SerialOp op) {
    for (const auto& step : plan.steps) {
        if (step.op == op) {
            return step;
        }
    }
    throw std::runtime_error(
        "requested hyper-mix step not found");
}

float sigmoid(float value) {
    return
        1.0F /
        (1.0F +
         std::exp(-value));
}

struct ReferenceResult {
    std::vector<float> mixed;
    std::vector<float> injection;
};

ReferenceResult reference(
    const std::vector<float>& input,
    const std::vector<float>& norm,
    const std::vector<IQ4XSBlock>& down,
    const std::vector<Q8Block>& up,
    const std::vector<IQ4XSBlock>& inject,
    std::uint32_t hidden,
    std::uint32_t hc,
    std::uint32_t lowrank,
    float epsilon,
    bool with_injection) {
    const auto expanded =
        hidden * hc;
    std::vector<float> normalized(
        expanded);
    for (std::uint32_t stream = 0;
         stream < hc;
         ++stream) {
        float sum = 0.0F;
        for (std::uint32_t feature = 0;
             feature < hidden;
             ++feature) {
            const auto index =
                stream * hidden +
                feature;
            sum +=
                input[index] *
                input[index];
        }
        const float inv =
            1.0F /
            std::sqrt(
                sum /
                    static_cast<float>(
                        hidden) +
                epsilon);
        for (std::uint32_t feature = 0;
             feature < hidden;
             ++feature) {
            const auto index =
                stream * hidden +
                feature;
            normalized[index] =
                input[index] *
                inv *
                (1.0F +
                 norm[index]);
        }
    }

    std::vector<float> low(
        lowrank);
    for (std::uint32_t out = 0;
         out < lowrank;
         ++out) {
        const float weight =
            __half2float(
                down[out].d);
        float sum = 0.0F;
        for (const auto value :
             normalized) {
            sum +=
                value *
                weight;
        }
        const float scaled =
            sum /
            static_cast<float>(hc);
        low[out] =
            scaled *
            sigmoid(scaled);
    }

    ReferenceResult result;
    result.mixed.resize(hidden);
    for (std::uint32_t feature = 0;
         feature < hidden;
         ++feature) {
        float mixed = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc;
             ++stream) {
            const auto output_index =
                stream * hidden +
                feature;
            const float weight =
                __half2float(
                    up[output_index].scale);
            float projected = 0.0F;
            for (const auto value :
                 low) {
                projected +=
                    value *
                    weight;
            }
            mixed +=
                sigmoid(projected) *
                normalized[
                    output_index];
        }
        result.mixed[feature] =
            mixed /
            static_cast<float>(hc);
    }

    if (with_injection) {
        result.injection.resize(hc);
        for (std::uint32_t stream = 0;
             stream < hc;
             ++stream) {
            const float weight =
                __half2float(
                    inject[stream].d);
            float sum = 0.0F;
            for (const auto value :
                 normalized) {
                sum +=
                    value *
                    weight;
            }
            result.injection[stream] =
                2.0F *
                sigmoid(
                    sum /
                    static_cast<float>(
                        hc));
        }
    }
    return result;
}

struct Fixture {
    static constexpr std::uint32_t down_type = 23;
    static constexpr std::uint32_t up_type = 8;
    static constexpr std::uint32_t inject_type = 23;
    static constexpr std::uint32_t hidden = 64;
    static constexpr std::uint32_t hc = 4;
    static constexpr std::uint32_t lowrank = 32;
    static constexpr std::uint32_t expanded =
        hidden * hc;

    std::vector<float> input;
    std::vector<float> norm;
    std::vector<IQ4XSBlock> down;
    std::vector<Q8Block> up;
    std::vector<IQ4XSBlock> inject;

    Fixture()
        : input(expanded),
          norm(expanded),
          down(make_iq4_rows(
              lowrank,
              0.00005F)),
          up(make_q8_rows(
              expanded,
              0.00007F)),
          inject(make_iq4_rows(
              hc,
              0.00003F)) {
        for (std::size_t index = 0;
             index < input.size();
             ++index) {
            input[index] =
                static_cast<float>(
                    static_cast<int>(
                        index % 23U) -
                    11) /
                17.0F;
            norm[index] =
                static_cast<float>(
                    static_cast<int>(
                        index % 5U) -
                    2) *
                0.01F;
        }
        norm[0] = -1.0F;
        norm[1] = -1.25F;
        norm[2] = 0.375F;
    }
};

// Dense geometry deliberately crosses both feature-tile and output-tile
// boundaries without quantized block alignment masking the tail conditions.
struct DenseTailFixture {
    static constexpr std::uint32_t hidden = 67;
    static constexpr std::uint32_t hc = 3;
    static constexpr std::uint32_t lowrank = 139;
    static constexpr std::uint32_t expanded = hidden * hc;
    static constexpr std::uint32_t down_type = 0;
    static constexpr std::uint32_t up_type = 0;
    static constexpr std::uint32_t inject_type = 0;
    std::vector<float> input, norm, down, up, inject;
    DenseTailFixture() : input(expanded), norm(expanded), down(expanded * lowrank),
        up(lowrank * expanded), inject(expanded * hc) {
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(static_cast<int>(index % 29U) - 14) / 19.0F;
            norm[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) / 23.0F;
        }
        for (auto* weights : {&down, &up, &inject})
            for (std::size_t index = 0; index < weights->size(); ++index)
                (*weights)[index] = static_cast<float>(static_cast<int>((index * 7U + index / 11U) % 31U) - 15) / 997.0F;
    }
};

ReferenceResult reference(const std::vector<float>& input, const std::vector<float>& norm,
    const std::vector<float>& down, const std::vector<float>& up, const std::vector<float>& inject,
    std::uint32_t hidden, std::uint32_t hc, std::uint32_t lowrank, float epsilon, bool with_injection) {
    const auto width = hidden * hc;
    std::vector<float> normalized(width), low(lowrank);
    for (std::uint32_t stream = 0; stream < hc; ++stream) {
        double square_sum = 0.0;
        for (std::uint32_t feature = 0; feature < hidden; ++feature) {
            const double value = input[stream * hidden + feature];
            square_sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(square_sum / hidden + epsilon);
        for (std::uint32_t feature = 0; feature < hidden; ++feature) {
            const auto index = stream * hidden + feature;
            normalized[index] = static_cast<float>(input[index] * inverse * (1.0 + norm[index]));
        }
    }
    for (std::uint32_t out = 0; out < lowrank; ++out) {
        double sum = 0.0;
        for (std::uint32_t feature = 0; feature < width; ++feature)
            sum += static_cast<double>(normalized[feature]) * down[out * width + feature];
        const float scaled = static_cast<float>(sum / hc);
        low[out] = scaled * sigmoid(scaled);
    }
    ReferenceResult result;
    result.mixed.resize(hidden);
    for (std::uint32_t feature = 0; feature < hidden; ++feature) {
        double mixed = 0.0;
        for (std::uint32_t stream = 0; stream < hc; ++stream) {
            const auto index = stream * hidden + feature;
            double gate = 0.0;
            for (std::uint32_t inner = 0; inner < lowrank; ++inner)
                gate += static_cast<double>(low[inner]) * up[index * lowrank + inner];
            mixed += normalized[index] * sigmoid(static_cast<float>(gate));
        }
        result.mixed[feature] = static_cast<float>(mixed / hc);
    }
    if (with_injection) {
        result.injection.resize(hc);
        for (std::uint32_t stream = 0; stream < hc; ++stream) {
            double sum = 0.0;
            for (std::uint32_t feature = 0; feature < width; ++feature)
                sum += static_cast<double>(normalized[feature]) * inject[stream * width + feature];
            result.injection[stream] = 2.0F * sigmoid(static_cast<float>(sum / hc));
        }
    }
    return result;
}

template <typename FixtureType = Fixture>
void run_numeric_fixture(
    strix::SerialOp op,
    bool with_injection, bool bf16 = false) {
    using Fixture = FixtureType;
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        first_step(plan, op);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;
    request.activation_element_bytes = bf16 ? 2U : 4U;
    request.control_element_bytes = 4;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc hyper-mix arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x45450000ULL +
        static_cast<std::uint64_t>(
            op);
    memory.arena_base =
        reinterpret_cast<
            std::uintptr_t>(
                arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto input_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto mixed_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    std::optional<
        strix::ExecutionBufferAddress>
        injection_address;
    if (with_injection) {
        injection_address =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena,
                memory,
                step_io.writes[1]);
    }

    Fixture fixture;
    std::vector<std::uint16_t> input_bf16(fixture.input.size());
    if (bf16) {
        for (std::size_t index = 0; index < fixture.input.size(); ++index) {
            int exponent = 0;
            const double value = fixture.input[index];
            const double scaled = std::frexp(std::fabs(value), &exponent) * 256.0;
            auto integer = static_cast<unsigned int>(std::floor(scaled));
            const double fraction = scaled - integer;
            if (fraction > 0.5 || (fraction == 0.5 && (integer & 1U))) ++integer;
            fixture.input[index] = static_cast<float>(std::copysign(std::ldexp(static_cast<double>(integer), exponent - 8), value));
            input_bf16[index] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(fixture.input[index]) >> 16U);
        }
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input_address.address),
            bf16 ? static_cast<const void*>(input_bf16.data()) : fixture.input.data(),
            fixture.input.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)),
            hipMemcpyHostToDevice),
        "copy hyper-mix input");

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";

    const auto add_manifest =
        [&](std::string name,
            std::uint32_t ggml_type,
            std::vector<std::uint64_t> dimensions,
            strix::TensorRole role,
            std::uint64_t bytes) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.ggml_type =
                ggml_type;
            tensor.dimensions =
                std::move(dimensions);
            tensor.role =
                role;
            tensor.byte_size =
                bytes;
            if (op !=
                strix::SerialOp::
                    final_hyper_mix) {
                tensor.layer =
                    step.layer;
            }
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add_manifest(
        with_injection
            ? "blk.0.hc_attn_norm.weight"
            : "output_hc_norm.weight",
        0,
        {Fixture::expanded},
        strix::TensorRole::normalization,
        fixture.norm.size() *
            sizeof(float));
    add_manifest(
        with_injection
            ? "blk.0.hc_attn_down.weight"
            : "output_hc_down.weight",
        Fixture::down_type,
        {
            Fixture::expanded,
            Fixture::lowrank,
        },
        strix::TensorRole::dense_projection,
        fixture.down.size() *
            sizeof(typename decltype(fixture.down)::value_type));
    add_manifest(
        with_injection
            ? "blk.0.hc_attn_up.weight"
            : "output_hc_up.weight",
        Fixture::up_type,
        {
            Fixture::lowrank,
            Fixture::expanded,
        },
        strix::TensorRole::dense_projection,
        fixture.up.size() *
            sizeof(typename decltype(fixture.up)::value_type));
    if (with_injection) {
        add_manifest(
            "blk.0.hc_attn_inject.weight",
            Fixture::inject_type,
            {
                Fixture::expanded,
                Fixture::hc,
            },
            strix::TensorRole::dense_projection,
            fixture.inject.size() *
                sizeof(typename decltype(fixture.inject)::value_type));
    }

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        manifest.tensors.size(),
        nullptr);
    // The independent oracle consumes source zero-centered parameters;
    // GGUF serializes their already-folded multiplicative scales.
    auto gguf_norm = fixture.norm;
    for (auto& value : gguf_norm) value += 1.0F;
    upload(gguf_norm, &device_weights[0]);
    upload(
        fixture.down,
        &device_weights[1]);
    upload(
        fixture.up,
        &device_weights[2]);
    if (with_injection) {
        upload(
            fixture.inject,
            &device_weights[3]);
    }

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    for (std::size_t index = 0;
         index <
            manifest.tensors.size();
         ++index) {
        runtime.tensors.push_back(
            {
                static_cast<
                    std::uint32_t>(
                    index),
                strix::RuntimeTensorStorage::
                    resident,
                reinterpret_cast<
                    std::uintptr_t>(
                    device_weights[index]),
                manifest.tensors[index].
                    byte_size,
                1,
                0,
            });
    }

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x56560000ULL +
        static_cast<std::uint64_t>(
            op);
    for (std::uint32_t slot = 0;
         slot <
            manifest.tensors.size();
         ++slot) {
        access.accesses.push_back(
            {
                step.id,
                step.op,
                slot,
                {slot},
            });
    }

    strix::ExecutionRuntimeResources resources;
    resources.scratch =
        &memory;
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &runtime;
    resources.tensor_access =
        &access;

    strix::ExecutionDispatchContext context;
    context.step_id =
        step.id;
    context.op =
        op;
    context.layer =
        step.layer;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.temporaries = step_io.temporaries;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.resources =
        &resources;
    context.logical_rows = 1;
    context.request_concurrency = 1;

    strix::HipHyperMixLauncherConfig config;
    config.hidden_size =
        Fixture::hidden;
    config.hc_count =
        Fixture::hc;
    config.lowrank =
        Fixture::lowrank;
    config.rms_epsilon =
        1e-6F;
    config.activation = bf16 ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;

    auto launcher =
        strix::make_hip_hyper_mix_launcher(
            "test.hyper_mix",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "hyper-mix synchronize");

    std::vector<float> actual_mixed(
        Fixture::hidden);
    std::vector<std::uint16_t> actual_bf16(Fixture::hidden);
    check(
        hipMemcpy(
            bf16 ? static_cast<void*>(actual_bf16.data()) : actual_mixed.data(),
            reinterpret_cast<
                const void*>(
                mixed_address.address),
            actual_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)),
            hipMemcpyDeviceToHost),
        "copy hyper-mix output");
    if (bf16) for (std::size_t index = 0; index < actual_mixed.size(); ++index)
        actual_mixed[index] = std::bit_cast<float>(static_cast<std::uint32_t>(actual_bf16[index]) << 16U);

    std::vector<float> actual_injection;
    if (with_injection) {
        actual_injection.resize(
            Fixture::hc);
        check(
            hipMemcpy(
                actual_injection.data(),
                reinterpret_cast<
                    const void*>(
                    injection_address->
                        address),
                actual_injection.size() *
                    sizeof(float),
                hipMemcpyDeviceToHost),
            "copy hyper-mix injection");
    }

    const auto expected =
        reference(
            fixture.input,
            fixture.norm,
            fixture.down,
            fixture.up,
            fixture.inject,
            Fixture::hidden,
            Fixture::hc,
            Fixture::lowrank,
            1e-6F,
            with_injection);

    float mixed_error = 0.0F;
    for (std::size_t index = 0;
         index <
            actual_mixed.size();
         ++index) {
        mixed_error =
            std::max(
                mixed_error,
                std::abs(
                    actual_mixed[index] -
                    expected.mixed[index]));
    }
    expect(
        mixed_error <= 2e-3F,
        "fused hyper-mix output matches CPU reference");

    if (with_injection) {
        float injection_error = 0.0F;
        for (std::size_t index = 0;
             index <
                actual_injection.size();
             ++index) {
            injection_error =
                std::max(
                    injection_error,
                    std::abs(
                        actual_injection[index] -
                        expected.injection[index]));
        }
        expect(
            injection_error <= 2e-3F,
            "fused hyper-mix injection weights match CPU reference");
    }

    auto parallel_config = config;
    parallel_config.parallel = true;
    auto parallel_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.parallel", &parallel_config);
    expect(parallel_launcher.implementation_identity != launcher.implementation_identity,
           "parallel hyper-mix has a distinct numerical implementation identity");
    parallel_launcher.function(context, parallel_launcher.user_data);
    check(hipDeviceSynchronize(), "parallel hyper-mix synchronize");
    std::vector<float> parallel_mixed(actual_mixed.size());
    check(hipMemcpy(bf16 ? static_cast<void*>(actual_bf16.data()) : parallel_mixed.data(), reinterpret_cast<void*>(mixed_address.address),
                    parallel_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)), hipMemcpyDeviceToHost), "read parallel hyper-mix output");
    if (bf16) for (std::size_t index = 0; index < parallel_mixed.size(); ++index)
        parallel_mixed[index] = std::bit_cast<float>(static_cast<std::uint32_t>(actual_bf16[index]) << 16U);
    expect(parallel_mixed == actual_mixed, "parallel projections preserve legacy FP32 accumulation results");
    if (with_injection) {
        std::vector<float> parallel_injection(actual_injection.size());
        check(hipMemcpy(parallel_injection.data(), reinterpret_cast<void*>(injection_address->address),
                        parallel_injection.size() * sizeof(float), hipMemcpyDeviceToHost), "read parallel injection");
        expect(parallel_injection == actual_injection, "parallel injection preserves legacy accumulation results");
    }
    auto staged_config = parallel_config;
    staged_config.staged_projections = true;
    auto staged_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.staged", &staged_config);
    expect(staged_launcher.implementation_identity != parallel_launcher.implementation_identity,
           "staged projections have a distinct implementation identity");
    // Preserve immutable baseline bits independently of all later device writes.
    const auto expected_bf16_bits = actual_bf16;
    const auto workspace_address = strix::resolve_dispatch_temporary(context, 0);
    const auto poison_outputs = [&](hipStream_t stream) {
        check(hipMemsetAsync(reinterpret_cast<void*>(workspace_address.address), 0xff,
                             static_cast<std::size_t>(workspace_address.bytes), stream),
              "poison complete admitted hyper workspace");
        check(hipMemsetAsync(reinterpret_cast<void*>(mixed_address.address), 0xff,
                             parallel_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)), stream),
              "poison hyper-mix output");
        if (with_injection)
            check(hipMemsetAsync(reinterpret_cast<void*>(injection_address->address), 0xff,
                                 actual_injection.size() * sizeof(float), stream), "poison hyper injection");
    };
    const auto outputs_match_expected_bits = [&] {
        const auto mixed_bytes = parallel_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float));
        std::vector<std::uint8_t> observed(mixed_bytes);
        check(hipMemcpy(observed.data(), reinterpret_cast<void*>(mixed_address.address),
                        mixed_bytes, hipMemcpyDeviceToHost), "read hyper-mix output bits");
        const void* expected = bf16 ? static_cast<const void*>(expected_bf16_bits.data()) : parallel_mixed.data();
        bool matches = std::memcmp(observed.data(), expected, mixed_bytes) == 0;
        if (with_injection) {
            const auto injection_bytes = actual_injection.size() * sizeof(float);
            std::vector<std::uint8_t> injection(injection_bytes);
            check(hipMemcpy(injection.data(), reinterpret_cast<void*>(injection_address->address),
                            injection_bytes, hipMemcpyDeviceToHost), "read hyper injection bits");
            matches &= std::memcmp(injection.data(), actual_injection.data(), injection_bytes) == 0;
        }
        return matches;
    };
    // Every candidate starts from NaN poison in its entire admitted scratch and
    // outputs. No-op or partial writes cannot inherit prior correct values.
    hipStream_t staged_stream = nullptr;
    hipGraph_t staged_graph = nullptr;
    hipGraphExec_t staged_executable = nullptr;
    check(hipStreamCreateWithFlags(&staged_stream, hipStreamNonBlocking), "create staged stream");
    strix::HipLaunchContext staged_scope;
    staged_scope.stream = staged_stream;
    {
        strix::HipLaunchScope scope(staged_scope);
        poison_outputs(staged_stream);
        check(hipStreamSynchronize(staged_stream), "synchronize staged no-op control");
        expect(!outputs_match_expected_bits(), "poisoned no-op fails staged bit comparison");
        staged_launcher.function(context, staged_launcher.user_data);
        check(hipStreamSynchronize(staged_stream), "synchronize eager staged projection");
        expect(outputs_match_expected_bits(), "eager staged output and injection preserve baseline bits");
        check(hipStreamBeginCapture(staged_stream, hipStreamCaptureModeThreadLocal), "capture staged projection");
        staged_launcher.function(context, staged_launcher.user_data);
        check(hipStreamEndCapture(staged_stream, &staged_graph), "finish staged capture");
    }
    check(hipGraphInstantiate(&staged_executable, staged_graph, nullptr, nullptr, 0), "instantiate staged graph");
    for (unsigned replay = 0; replay < 3; ++replay) {
        poison_outputs(staged_stream);
        check(hipGraphLaunch(staged_executable, staged_stream), "replay staged graph");
        check(hipStreamSynchronize(staged_stream), "synchronize staged graph");
        expect(outputs_match_expected_bits(), "staged replay output and injection preserve baseline bits");
    }
    check(hipGraphExecDestroy(staged_executable), "destroy staged executable");
    check(hipGraphDestroy(staged_graph), "destroy staged graph");
    check(hipStreamDestroy(staged_stream), "destroy staged stream");
    auto fused_config = staged_config;
    fused_config.fused_injection = true;
    auto fused_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.fused.v5", &fused_config);
    expect(fused_launcher.implementation_identity != staged_launcher.implementation_identity,
           "fused injection v5 has a distinct implementation revision");
    hipStream_t fused_stream = nullptr;
    hipGraph_t fused_graph = nullptr;
    hipGraphExec_t fused_executable = nullptr;
    check(hipStreamCreateWithFlags(&fused_stream, hipStreamNonBlocking), "create fused stream");
    strix::HipLaunchContext fused_scope;
    fused_scope.stream = fused_stream;
    {
        strix::HipLaunchScope scope(fused_scope);
        poison_outputs(fused_stream);
        check(hipStreamSynchronize(fused_stream), "synchronize fused no-op control");
        expect(!outputs_match_expected_bits(), "poisoned no-op fails fused bit comparison");
        fused_launcher.function(context, fused_launcher.user_data);
        check(hipStreamSynchronize(fused_stream), "synchronize eager fused injection");
        expect(outputs_match_expected_bits(), "eager fused output and injection preserve baseline bits");
        check(hipStreamBeginCapture(fused_stream, hipStreamCaptureModeThreadLocal), "capture fused injection");
        fused_launcher.function(context, fused_launcher.user_data);
        check(hipStreamEndCapture(fused_stream, &fused_graph), "finish fused capture");
    }
    check(hipGraphInstantiate(&fused_executable, fused_graph, nullptr, nullptr, 0), "instantiate fused graph");
    for (unsigned replay = 0; replay < 3; ++replay) {
        poison_outputs(fused_stream);
        check(hipGraphLaunch(fused_executable, fused_stream), "replay fused graph");
        check(hipStreamSynchronize(fused_stream), "synchronize fused replay");
        expect(outputs_match_expected_bits(), "fused replay output and injection preserve baseline bits");
    }
    check(hipGraphExecDestroy(fused_executable), "destroy fused executable");
    check(hipGraphDestroy(fused_graph), "destroy fused graph");
    check(hipStreamDestroy(fused_stream), "destroy fused stream");
    auto invalid_fused_config = fused_config;
    invalid_fused_config.staged_projections = false;
    bool invalid_fused_rejected = false;
    try { (void)strix::make_hip_hyper_mix_launcher("test.invalid.fused", &invalid_fused_config); }
    catch (const std::invalid_argument&) { invalid_fused_rejected = true; }
    expect(invalid_fused_rejected, "fused injection rejects a nonstaged configuration");
    auto invalid_staged_config = staged_config;
    invalid_staged_config.parallel = false;
    bool invalid_staged_rejected = false;
    try { (void)strix::make_hip_hyper_mix_launcher("test.invalid.staged", &invalid_staged_config); }
    catch (const std::invalid_argument&) { invalid_staged_rejected = true; }
    expect(invalid_staged_rejected, "staged projections reject a nonparallel configuration");
    auto missing_workspace = context;
    missing_workspace.temporaries = {};
    bool rejected = false;
    try { parallel_launcher.function(missing_workspace, parallel_launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "parallel hyper-mix rejects missing admitted scratch");
    rejected = false;
    try { staged_launcher.function(missing_workspace, staged_launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "staged projections reject missing admitted scratch");
    rejected = false;
    try { fused_launcher.function(missing_workspace, fused_launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "fused injection rejects missing admitted scratch");
    for (unsigned malformed = 0; malformed < 3; ++malformed) {
        auto invalid_arena = arena;
        auto invalid_context = context;
        auto invalid_temporaries = step_io.temporaries;
        const auto original_use = invalid_temporaries.front();
        if (malformed == 0)
            invalid_temporaries.front().buffer.buffer_class = strix::ExecutionBufferClass::hidden;
        for (auto& allocation : invalid_arena.allocations) {
            if (allocation.use != original_use) continue;
            allocation.use = invalid_temporaries.front();
            if (malformed == 1) allocation.bytes =
                (static_cast<std::uint64_t>(Fixture::expanded) + Fixture::lowrank) * sizeof(float) - 1U;
            if (malformed == 2) ++allocation.offset;
        }
        invalid_context.arena = &invalid_arena;
        invalid_context.temporaries = invalid_temporaries;
        rejected = false;
        try { staged_launcher.function(invalid_context, staged_launcher.user_data); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "staged projections reject wrong-class, short or unaligned scratch");
        rejected = false;
        try { fused_launcher.function(invalid_context, fused_launcher.user_data); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "fused injection rejects wrong-class, short or unaligned scratch");
    }

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree hyper-mix weight");
    }
    check(
        hipFree(arena_pointer),
        "hipFree hyper-mix arena");
}

void test_catalog_and_bundle() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    strix::ExecutionBindingRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.minimum_qualification =
        strix::kernels::
            QualificationStatus::
                experimental;

    strix::HipHyperMixLauncherConfig config;
    config.hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    config.hc_count =
        static_cast<std::uint32_t>(
            model.hc_count);
    config.lowrank =
        static_cast<std::uint32_t>(
            model.hc_lowrank);
    config.rms_epsilon =
        1e-6F;
    config.activation =
        strix::HipActivationStorage::bf16;

    strix::kernels::KernelRegistry registry;
    strix::register_hip_hyper_mix_kernel_variant(
        model,
        request,
        config,
        registry);
    auto staged_config = config;
    staged_config.parallel = true;
    strix::kernels::KernelRegistry parallel_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, staged_config, parallel_registry);
    expect(parallel_registry.variants().front().resources.lds_bytes >= (256U + 1U) * sizeof(float),
           "parallel LDS declaration includes reduction and shared inverse");
    staged_config.staged_projections = true;
    strix::kernels::KernelRegistry staged_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, staged_config, staged_registry);
    const auto& staged_variant = staged_registry.variants().front();
    expect(staged_variant.resources.lds_bytes >= (128U + 8U * 129U) * sizeof(float) &&
           staged_variant.resources.lds_bytes >= (256U + 1U) * sizeof(float),
           "staged LDS declaration covers projection tiles and normalization");
    expect(staged_variant.resources.scratch_bytes ==
           (model.hidden_size * model.hc_count + model.hc_lowrank) * sizeof(float),
           "staged projections preserve admitted workspace size");
    expect(staged_variant.id != parallel_registry.variants().front().id &&
           staged_variant.compiler_identity != parallel_registry.variants().front().compiler_identity,
           "staged kernel and compiler identities distinguish the numerical path");
    auto fused_config = staged_config;
    fused_config.fused_injection = true;
    strix::kernels::KernelRegistry fused_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, fused_config, fused_registry);
    const auto& fused_variant = fused_registry.variants().front();
    expect(fused_variant.id != staged_variant.id && fused_variant.compiler_identity != staged_variant.compiler_identity,
           "fused injection changes kernel and compiler revision identities");
    expect(fused_variant.resources.lds_bytes == staged_variant.resources.lds_bytes &&
           fused_variant.resources.scratch_bytes == staged_variant.resources.scratch_bytes,
           "fused injection retains staged LDS and admitted global workspace bounds");

    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto bindings =
        strix::bind_serial_execution_plan(
            model,
            execution,
            registry,
            request);

    std::uint32_t expected = 1;
    expected +=
        static_cast<std::uint32_t>(
            model.hidden_layers) *
        2U;

    expect(
        bindings.bound_steps == expected,
        "hyper-mix catalog binds attention, MLP and final mixers");

    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x6767ULL;
    compiled.execution =
        execution;
    compiled.bindings =
        bindings;

    strix::ExecutionLauncherRegistry launchers;
    strix::HipHyperMixLauncherBundle bundle(
        config);
    bundle.register_for_compiled_execution(
        compiled,
        launchers);

    const auto found =
        std::find_if(
            bindings.bindings.begin(),
            bindings.bindings.end(),
            [](const auto& binding) {
                return binding.op ==
                    strix::SerialOp::
                        attn_hyper_mix;
            });
    expect(
        found !=
            bindings.bindings.end() &&
        found->kernel_id.has_value() &&
        launchers.find_kernel_launcher(
            *found->kernel_id) !=
            nullptr,
        "hyper-mix launcher bundle materializes selected mixer id");
}

}  // namespace

int main() try {
    test_catalog_and_bundle();
    run_numeric_fixture(
        strix::SerialOp::attn_hyper_mix,
        true);
    run_numeric_fixture(
        strix::SerialOp::final_hyper_mix,
        false);
    run_numeric_fixture(strix::SerialOp::attn_hyper_mix, true, true);
    run_numeric_fixture(strix::SerialOp::mlp_hyper_mix, true, true);
    run_numeric_fixture(strix::SerialOp::final_hyper_mix, false, true);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::attn_hyper_mix, true);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::final_hyper_mix, false);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::attn_hyper_mix, true, true);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::final_hyper_mix, false, true);

    if (failures == 0) {
        std::cout
            << "HIP hyper-mix launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP hyper-mix launcher tests: "
        << error.what() << '\n';
    return 2;
}
