#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    }
};

void run_numeric_fixture(
    strix::SerialOp op,
    bool with_injection) {
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
    request.activation_element_bytes = 4;
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
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input_address.address),
            fixture.input.data(),
            fixture.input.size() *
                sizeof(float),
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
        23,
        {
            Fixture::expanded,
            Fixture::lowrank,
        },
        strix::TensorRole::dense_projection,
        fixture.down.size() *
            sizeof(IQ4XSBlock));
    add_manifest(
        with_injection
            ? "blk.0.hc_attn_up.weight"
            : "output_hc_up.weight",
        8,
        {
            Fixture::lowrank,
            Fixture::expanded,
        },
        strix::TensorRole::dense_projection,
        fixture.up.size() *
            sizeof(Q8Block));
    if (with_injection) {
        add_manifest(
            "blk.0.hc_attn_inject.weight",
            23,
            {
                Fixture::expanded,
                Fixture::hc,
            },
            strix::TensorRole::dense_projection,
            fixture.inject.size() *
                sizeof(IQ4XSBlock));
    }

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        manifest.tensors.size(),
        nullptr);
    upload(
        fixture.norm,
        &device_weights[0]);
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
    config.activation =
        strix::HipActivationStorage::fp32;

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
    check(
        hipMemcpy(
            actual_mixed.data(),
            reinterpret_cast<
                const void*>(
                mixed_address.address),
            actual_mixed.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy hyper-mix output");

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
