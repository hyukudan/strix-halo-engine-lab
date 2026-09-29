#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/model_contract.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(value) << 16U;
    float output = 0.0F;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

const strix::ExecutionStepIo& find_io(
    const strix::SerialExecutionPlan& plan,
    const strix::SerialExecutionIoPlan& io,
    strix::SerialOp op) {
    for (std::size_t index = 0;
         index < plan.steps.size();
         ++index) {
        if (plan.steps[index].op == op) {
            return io.steps[index];
        }
    }
    throw std::runtime_error("requested operator is not present in serial plan");
}

const strix::ExecutionStep& find_step(
    const strix::SerialExecutionPlan& plan,
    strix::SerialOp op) {
    for (const auto& step : plan.steps) {
        if (step.op == op) {
            return step;
        }
    }
    throw std::runtime_error("requested operator is not present in serial plan");
}

template <typename Storage>
void copy_vector_to_device(
    std::uintptr_t destination,
    const std::vector<Storage>& values) {
    check(
        hipMemcpy(
            reinterpret_cast<void*>(destination),
            values.data(),
            values.size() * sizeof(Storage),
            hipMemcpyHostToDevice),
        "hipMemcpy H2D fixture");
}

template <typename Storage>
std::vector<Storage> copy_vector_from_device(
    std::uintptr_t source,
    std::size_t count) {
    std::vector<Storage> values(count);
    check(
        hipMemcpy(
            values.data(),
            reinterpret_cast<const void*>(source),
            count * sizeof(Storage),
            hipMemcpyDeviceToHost),
        "hipMemcpy D2H fixture");
    return values;
}

strix::ExecutionDispatchContext make_context(
    const strix::ExecutionStep& step,
    const strix::ExecutionStepIo& io,
    const strix::ExecutionArenaPlan& arena,
    const strix::ExecutionMemoryBinding& memory) {
    strix::ExecutionDispatchContext context;
    context.step_id = step.id;
    context.op = step.op;
    context.layer = step.layer;
    context.reads = io.reads;
    context.writes = io.writes;
    context.arena = &arena;
    context.memory = &memory;
    context.logical_rows = 1;
    context.request_concurrency = 1;
    return context;
}



void test_kernel_catalog() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::ExecutionBindingRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.minimum_qualification =
        strix::kernels::QualificationStatus::experimental;

    strix::HipElementwiseLauncherConfig config;
    config.hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    config.hc_count =
        static_cast<std::uint32_t>(
            model.hc_count);
    config.activation =
        strix::HipActivationStorage::bf16;

    strix::kernels::KernelRegistry registry;
    strix::register_hip_elementwise_kernel_variants(
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

    std::uint32_t expected_bound = 0;
    for (std::size_t index = 0;
         index < execution.steps.size();
         ++index) {
        const auto op =
            execution.steps[index].op;
        const bool elementwise =
            op == strix::SerialOp::hyper_expand ||
            op == strix::SerialOp::attn_inject ||
            op == strix::SerialOp::mlp_inject ||
            op == strix::SerialOp::moe_combine;
        if (!elementwise) {
            continue;
        }
        ++expected_bound;
        expect(
            bindings.bindings[index].kernel_id.has_value() &&
                bindings.bindings[index].kernel_id->starts_with(
                    "hip."),
            "HIP elementwise kernel catalog binds the targeted serial operators");
    }
    expect(
        bindings.bound_steps == expected_bound,
        "HIP elementwise catalog binds only implemented families");

    auto wrong = config;
    ++wrong.hidden_size;
    bool geometry_rejected = false;
    try {
        strix::kernels::KernelRegistry unused;
        strix::register_hip_elementwise_kernel_variants(
            model,
            request,
            wrong,
            unused);
    } catch (const std::invalid_argument&) {
        geometry_rejected = true;
    }
    expect(
        geometry_rejected,
        "HIP elementwise kernel catalog rejects model-geometry mismatch");
}

void test_launcher_bundle() {
    strix::CompiledSerialExecution compiled;
    compiled.execution_identity = 0x445566ULL;

    const auto add_step =
        [&](std::uint32_t id,
            strix::SerialOp op,
            std::string kernel_id) {
            strix::ExecutionStep step;
            step.id = id;
            step.op = op;
            compiled.execution.steps.push_back(step);

            strix::ExecutionKernelBinding binding;
            binding.step_id = id;
            binding.op = op;
            binding.family =
                strix::operator_family_for(op);
            binding.kernel_id =
                std::move(kernel_id);
            compiled.bindings.bindings.push_back(
                std::move(binding));
            ++compiled.bindings.bound_steps;
        };

    add_step(
        0,
        strix::SerialOp::hyper_expand,
        "hip.hyper_expand");
    add_step(
        1,
        strix::SerialOp::attn_inject,
        "hip.inject");
    add_step(
        2,
        strix::SerialOp::mlp_inject,
        "hip.inject");
    add_step(
        3,
        strix::SerialOp::moe_combine,
        "hip.moe_combine");

    strix::HipElementwiseLauncherConfig config;
    config.hidden_size = 2560;
    config.hc_count = 4;
    config.activation =
        strix::HipActivationStorage::bf16;

    strix::HipElementwiseLauncherBundle bundle(
        config);
    strix::ExecutionLauncherRegistry registry;
    bundle.register_for_compiled_execution(
        compiled,
        registry);

    expect(
        registry.find_kernel_launcher(
            "hip.hyper_expand") != nullptr &&
        registry.find_kernel_launcher(
            "hip.inject") != nullptr &&
        registry.find_kernel_launcher(
            "hip.moe_combine") != nullptr,
        "HIP elementwise launcher bundle registers selected kernel ids");

    const auto* inject =
        registry.find_kernel_launcher(
            "hip.inject");
    expect(
        inject != nullptr &&
            inject->implementation_identity != 0,
        "HIP elementwise launcher bundle deduplicates compatible inject kernel ids");

    auto collision = compiled;
    collision.bindings.bindings[3].kernel_id =
        "hip.inject";
    strix::ExecutionLauncherRegistry bad_registry;
    bool collision_rejected = false;
    try {
        bundle.register_for_compiled_execution(
            collision,
            bad_registry);
    } catch (const std::invalid_argument&) {
        collision_rejected = true;
    }
    expect(
        collision_rejected,
        "HIP elementwise launcher bundle rejects one id with incompatible semantics");
}

void run_fixture(
    strix::HipActivationStorage storage) {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(model);
    const auto io =
        strix::build_serial_execution_io_plan(plan);

    strix::ExecutionBufferRequest buffers;
    buffers.logical_rows = 1;
    buffers.activation_element_bytes =
        static_cast<std::uint32_t>(storage);
    buffers.control_element_bytes = 4;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model, buffers, plan, io, 256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(arena.total_bytes)),
        "hipMalloc test arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity = 0x12345678ULL;
    memory.arena_base =
        reinterpret_cast<std::uintptr_t>(arena_pointer);
    memory.arena_bytes = arena.total_bytes;
    memory.allocation_generation = 1;

    strix::HipElementwiseLauncherConfig config;
    config.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    config.hc_count =
        static_cast<std::uint32_t>(model.hc_count);
    config.activation = storage;

    const auto hyper_launcher =
        strix::make_hip_hyper_expand_launcher(
            "test.hyper_expand", &config);
    const auto inject_launcher =
        strix::make_hip_inject_launcher(
            "test.inject", &config);
    const auto combine_launcher =
        strix::make_hip_moe_combine_launcher(
            "test.moe_combine", &config);

    expect(
        hyper_launcher.implementation_identity != 0 &&
        inject_launcher.implementation_identity != 0 &&
        combine_launcher.implementation_identity != 0,
        "HIP elementwise launchers have implementation identities");

    const auto activation_count =
        static_cast<std::size_t>(model.hidden_size);
    const auto expanded_count =
        activation_count *
        static_cast<std::size_t>(model.hc_count);

    const auto run_hyper = [&] {
        const auto& step =
            find_step(plan, strix::SerialOp::hyper_expand);
        const auto& step_io =
            find_io(plan, io, strix::SerialOp::hyper_expand);
        auto context =
            make_context(step, step_io, arena, memory);

        const auto input =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.reads[0]);
        const auto output =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.writes[0]);

        std::vector<float> source(activation_count);
        for (std::size_t index = 0;
             index < source.size();
             ++index) {
            source[index] =
                static_cast<float>(
                    static_cast<int>(index % 31U) - 15) /
                13.0F;
        }

        if (storage ==
            strix::HipActivationStorage::fp32) {
            copy_vector_to_device(
                input.address, source);
        } else {
            std::vector<std::uint16_t> packed(source.size());
            std::transform(
                source.begin(), source.end(),
                packed.begin(),
                float_to_bf16);
            copy_vector_to_device(
                input.address, packed);
        }

        hyper_launcher.function(
            context,
            hyper_launcher.user_data);
        check(
            hipDeviceSynchronize(),
            "hyper-expand synchronize");

        std::vector<float> actual(expanded_count);
        if (storage ==
            strix::HipActivationStorage::fp32) {
            actual =
                copy_vector_from_device<float>(
                    output.address,
                    expanded_count);
        } else {
            const auto packed =
                copy_vector_from_device<std::uint16_t>(
                    output.address,
                    expanded_count);
            std::transform(
                packed.begin(), packed.end(),
                actual.begin(),
                bf16_to_float);
        }

        for (std::size_t stream = 0;
             stream < model.hc_count;
             ++stream) {
            for (std::size_t feature = 0;
                 feature < activation_count;
                 ++feature) {
                const auto actual_value =
                    actual[
                        stream * activation_count +
                        feature];
                const auto expected =
                    storage ==
                            strix::HipActivationStorage::fp32
                        ? source[feature]
                        : bf16_to_float(
                              float_to_bf16(
                                  source[feature]));
                if (std::abs(actual_value - expected) >
                    1e-6F) {
                    expect(false,
                           "hyper-expand repeats complete hidden vector");
                    return;
                }
            }
        }
    };

    const auto run_inject = [&] {
        const auto& step =
            find_step(plan, strix::SerialOp::attn_inject);
        const auto& step_io =
            find_io(plan, io, strix::SerialOp::attn_inject);
        auto context =
            make_context(step, step_io, arena, memory);

        const auto hyper =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.reads[0]);
        const auto hidden =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.reads[1]);
        const auto weights =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.reads[2]);
        const auto output =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.writes[0]);

        std::vector<float> hyper_values(expanded_count);
        std::vector<float> hidden_values(activation_count);
        std::vector<float> injection(
            static_cast<std::size_t>(model.hc_count));

        for (std::size_t index = 0;
             index < hyper_values.size();
             ++index) {
            hyper_values[index] =
                static_cast<float>(
                    static_cast<int>(index % 17U) - 8) /
                19.0F;
        }
        for (std::size_t index = 0;
             index < hidden_values.size();
             ++index) {
            hidden_values[index] =
                static_cast<float>(
                    static_cast<int>(index % 23U) - 11) /
                29.0F;
        }
        for (std::size_t index = 0;
             index < injection.size();
             ++index) {
            injection[index] =
                0.25F +
                static_cast<float>(index) * 0.125F;
        }

        const auto upload_activation =
            [&](std::uintptr_t address,
                const std::vector<float>& values) {
                if (storage ==
                    strix::HipActivationStorage::fp32) {
                    copy_vector_to_device(
                        address, values);
                } else {
                    std::vector<std::uint16_t> packed(
                        values.size());
                    std::transform(
                        values.begin(), values.end(),
                        packed.begin(),
                        float_to_bf16);
                    copy_vector_to_device(
                        address, packed);
                }
            };

        upload_activation(
            hyper.address,
            hyper_values);
        upload_activation(
            hidden.address,
            hidden_values);
        copy_vector_to_device(
            weights.address,
            injection);

        inject_launcher.function(
            context,
            inject_launcher.user_data);
        check(
            hipDeviceSynchronize(),
            "inject synchronize");

        std::vector<float> actual(expanded_count);
        if (storage ==
            strix::HipActivationStorage::fp32) {
            actual =
                copy_vector_from_device<float>(
                    output.address,
                    expanded_count);
        } else {
            const auto packed =
                copy_vector_from_device<std::uint16_t>(
                    output.address,
                    expanded_count);
            std::transform(
                packed.begin(), packed.end(),
                actual.begin(),
                bf16_to_float);
        }

        for (std::size_t stream = 0;
             stream < model.hc_count;
             ++stream) {
            for (std::size_t feature = 0;
                 feature < activation_count;
                 ++feature) {
                float base =
                    hyper_values[
                        stream * activation_count +
                        feature];
                float hidden_value =
                    hidden_values[feature];
                if (storage ==
                    strix::HipActivationStorage::bf16) {
                    base =
                        bf16_to_float(
                            float_to_bf16(base));
                    hidden_value =
                        bf16_to_float(
                            float_to_bf16(
                                hidden_value));
                }
                float expected =
                    base +
                    hidden_value *
                        injection[stream];
                if (storage ==
                    strix::HipActivationStorage::bf16) {
                    expected =
                        bf16_to_float(
                            float_to_bf16(
                                expected));
                }

                if (std::abs(
                        actual[
                            stream *
                                activation_count +
                            feature] -
                        expected) >
                    2e-3F) {
                    expect(false,
                           "inject launcher matches hyper residual equation");
                    return;
                }
            }
        }
    };

    const auto run_combine = [&] {
        const auto& step =
            find_step(plan, strix::SerialOp::moe_combine);
        const auto& step_io =
            find_io(plan, io, strix::SerialOp::moe_combine);
        auto context =
            make_context(step, step_io, arena, memory);

        const auto shared =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.reads[0]);
        const auto routed =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.reads[1]);
        const auto output =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena, memory,
                step_io.writes[0]);

        std::vector<float> left(activation_count);
        std::vector<float> right(activation_count);
        for (std::size_t index = 0;
             index < activation_count;
             ++index) {
            left[index] =
                static_cast<float>(
                    static_cast<int>(index % 13U) - 6) /
                11.0F;
            right[index] =
                static_cast<float>(
                    static_cast<int>(index % 19U) - 9) /
                17.0F;
        }

        const auto upload =
            [&](std::uintptr_t address,
                const std::vector<float>& values) {
                if (storage ==
                    strix::HipActivationStorage::fp32) {
                    copy_vector_to_device(
                        address, values);
                } else {
                    std::vector<std::uint16_t> packed(
                        values.size());
                    std::transform(
                        values.begin(), values.end(),
                        packed.begin(),
                        float_to_bf16);
                    copy_vector_to_device(
                        address, packed);
                }
            };
        upload(shared.address, left);
        upload(routed.address, right);

        combine_launcher.function(
            context,
            combine_launcher.user_data);
        check(
            hipDeviceSynchronize(),
            "MoE-combine synchronize");

        std::vector<float> actual(activation_count);
        if (storage ==
            strix::HipActivationStorage::fp32) {
            actual =
                copy_vector_from_device<float>(
                    output.address,
                    activation_count);
        } else {
            const auto packed =
                copy_vector_from_device<std::uint16_t>(
                    output.address,
                    activation_count);
            std::transform(
                packed.begin(), packed.end(),
                actual.begin(),
                bf16_to_float);
        }

        for (std::size_t index = 0;
             index < activation_count;
             ++index) {
            float a = left[index];
            float b = right[index];
            if (storage ==
                strix::HipActivationStorage::bf16) {
                a = bf16_to_float(float_to_bf16(a));
                b = bf16_to_float(float_to_bf16(b));
            }
            float expected = a + b;
            if (storage ==
                strix::HipActivationStorage::bf16) {
                expected =
                    bf16_to_float(
                        float_to_bf16(expected));
            }
            if (std::abs(actual[index] - expected) >
                2e-3F) {
                expect(false,
                       "MoE-combine launcher sums routed and shared rows");
                break;
            }
        }
    };

    run_hyper();
    run_inject();
    run_combine();

    check(
        hipFree(arena_pointer),
        "hipFree test arena");
}

}  // namespace

int main() try {
    test_kernel_catalog();
    test_launcher_bundle();
    run_fixture(
        strix::HipActivationStorage::bf16);
    run_fixture(
        strix::HipActivationStorage::fp32);

    if (failures == 0) {
        std::cout
            << "HIP runtime launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP runtime launcher tests: "
        << error.what() << '\n';
    return 2;
}
