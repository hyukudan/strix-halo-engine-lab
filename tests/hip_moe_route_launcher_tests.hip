#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
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

static_assert(sizeof(Q8Block) == 34);

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
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

const strix::ExecutionStep& first_route_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::moe_route) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no MoE route step");
}

const strix::ExecutionStepIo& io_for(
    const strix::SerialExecutionIoPlan& io,
    std::uint32_t step_id) {
    if (step_id >= io.steps.size()) {
        throw std::out_of_range(
            "step id exceeds IO plan");
    }
    return io.steps[step_id];
}

std::vector<float> make_hidden(
    std::size_t count) {
    std::vector<float> values(
        count,
        0.001F);
    for (std::size_t index = 0;
         index < count;
         ++index) {
        values[index] +=
            static_cast<float>(
                index % 7U) *
            0.00001F;
    }
    return values;
}

std::vector<float> make_f32_weights(
    std::uint32_t experts,
    std::uint32_t hidden) {
    std::vector<float> weights(
        static_cast<std::size_t>(
            experts) *
        hidden);
    for (std::uint32_t expert = 0;
         expert < experts;
         ++expert) {
        const float value =
            0.001F *
            static_cast<float>(
                expert + 1U);
        std::fill(
            weights.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(
                        expert) *
                    hidden),
            weights.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(
                        expert + 1U) *
                    hidden),
            value);
    }
    return weights;
}

std::vector<Q8Block> make_q8_weights(
    std::uint32_t experts,
    std::uint32_t hidden) {
    if (hidden % 32U != 0) {
        throw std::invalid_argument(
            "Q8 fixture hidden width must be divisible by 32");
    }
    const auto blocks_per_row =
        hidden / 32U;
    std::vector<Q8Block> weights(
        static_cast<std::size_t>(
            experts) *
        blocks_per_row);

    for (std::uint32_t expert = 0;
         expert < experts;
         ++expert) {
        const float scale =
            0.001F *
            static_cast<float>(
                expert + 1U);
        for (std::uint32_t block = 0;
             block < blocks_per_row;
             ++block) {
            auto& value =
                weights[
                    static_cast<std::size_t>(
                        expert) *
                        blocks_per_row +
                    block];
            value.scale =
                __float2half(scale);
            std::fill(
                std::begin(value.values),
                std::end(value.values),
                std::int8_t{1});
        }
    }
    return weights;
}

template <typename Activation>
__device__ float fixture_router_activation(const Activation* input, std::uint64_t index) {
    if constexpr (sizeof(Activation) == sizeof(float)) return input[index];
    else {
        return __uint_as_float(static_cast<std::uint32_t>(input[index]) << 16U);
    }
}

template <std::uint32_t Type>
__device__ float fixture_router_coefficient(const void* weights, std::uint32_t width,
                                           std::uint32_t expert, std::uint32_t feature) {
    const auto index = static_cast<std::uint64_t>(expert) * width + feature;
    if constexpr (Type == 0) return static_cast<const float*>(weights)[index];
    else if constexpr (Type == 1) return __half2float(static_cast<const __half*>(weights)[index]);
    else if constexpr (Type == 30) {
        return __uint_as_float(static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(weights)[index]) << 16U);
    } else {
        const auto& block = static_cast<const Q8Block*>(weights)[static_cast<std::uint64_t>(expert) * (width / 32U) + feature / 32U];
        return __half2float(block.scale) * static_cast<float>(block.values[feature % 32U]);
    }
}

// Owned test reference: reproduce the original projection's 256-lane
// ownership, ascending loop and multiply/add expression. This is not the
// production default kernel; its selected outputs are checked separately.
template <typename Activation, std::uint32_t Type>
__global__ void fixture_router_legacy_logits(const Activation* hidden, const void* weights,
                                           std::uint32_t width, std::uint32_t experts, float* logits) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    for (std::uint32_t expert = lane; expert < experts; expert += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t feature = 0; feature < width; ++feature)
            sum += fixture_router_activation(hidden, static_cast<std::uint64_t>(row) * width + feature) *
                   fixture_router_coefficient<Type>(weights, width, expert, feature);
        logits[static_cast<std::uint64_t>(row) * experts + expert] = sum;
    }
}

template <typename Activation>
void run_fixture_legacy_logits(std::uint32_t type, std::uint32_t rows, std::uint32_t width,
                              std::uint32_t experts, const void* hidden, const void* weights, float* logits) {
    switch (type) {
    case 0: hipLaunchKernelGGL(HIP_KERNEL_NAME(fixture_router_legacy_logits<Activation, 0>), dim3(rows), dim3(256),
                              0, nullptr, static_cast<const Activation*>(hidden), weights, width, experts, logits); break;
    case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(fixture_router_legacy_logits<Activation, 1>), dim3(rows), dim3(256),
                              0, nullptr, static_cast<const Activation*>(hidden), weights, width, experts, logits); break;
    case 30: hipLaunchKernelGGL(HIP_KERNEL_NAME(fixture_router_legacy_logits<Activation, 30>), dim3(rows), dim3(256),
                               0, nullptr, static_cast<const Activation*>(hidden), weights, width, experts, logits); break;
    case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(fixture_router_legacy_logits<Activation, 8>), dim3(rows), dim3(256),
                              0, nullptr, static_cast<const Activation*>(hidden), weights, width, experts, logits); break;
    default: throw std::logic_error("unsupported device legacy logits fixture type");
    }
    check(hipGetLastError(), "launch device legacy logits reference");
}

__global__ void router_fixture_denormal_probe(const float* input, const __half* half_input, float* output) {
    // A preserved subnormal input produces a normal result, distinguishing
    // input FTZ independently of output underflow handling.
    output[0] = __builtin_fmaf(input[0], 0x1p126F, input[2]);
    output[1] = __builtin_fmaf(input[1], 0.5F, input[2]);
    output[2] = __half2float(half_input[0]);
}

void qualify_fixture_denormal_scope() {
    const float input[] = {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(), 0.0F};
    __half half_input;
    const std::uint16_t half_bits = 1;
    std::memcpy(&half_input, &half_bits, sizeof(half_bits));
    float* device_input = nullptr; float* device_output = nullptr; __half* device_half = nullptr;
    check(hipMalloc(&device_input, sizeof(input)), "allocate denormal probe inputs");
    check(hipMalloc(&device_output, 3U * sizeof(float)), "allocate denormal probe outputs");
    check(hipMalloc(&device_half, sizeof(half_input)), "allocate denormal probe half");
    check(hipMemcpy(device_input, input, sizeof(input), hipMemcpyHostToDevice), "copy denormal probe inputs");
    check(hipMemcpy(device_half, &half_input, sizeof(half_input), hipMemcpyHostToDevice), "copy denormal probe half");
    hipLaunchKernelGGL(router_fixture_denormal_probe, dim3(1), dim3(1), 0, nullptr, device_input, device_half, device_output);
    check(hipGetLastError(), "launch denormal probe");
    float observed[3]{};
    check(hipMemcpy(observed, device_output, sizeof(observed), hipMemcpyDeviceToHost), "read denormal probe");
    const float expected[] = {std::fma(input[0], 0x1p126F, 0.0F), std::fma(input[1], 0.5F, 0.0F), __half2float(half_input)};
    std::uint32_t bits[3]{};
    std::memcpy(bits, observed, sizeof(bits));
    std::cout << "router denormal runtime probe bits "
              << std::hex << bits[0] << ' ' << bits[1] << ' ' << bits[2] << std::dec << '\n';
    check(hipFree(device_input), "free denormal probe inputs");
    check(hipFree(device_half), "free denormal probe half");
    check(hipFree(device_output), "free denormal probe outputs");
    std::cout << "router denormal observations: FP32 input "
              << (observed[0] == expected[0] ? "gradual" : "non-IEEE/FTZ")
              << "; FP32 output " << (observed[1] == expected[1] ? "gradual" : "non-IEEE/FTZ")
              << "; F16 decode " << (observed[2] == expected[2] ? "gradual" : "non-IEEE/FTZ") << '\n';
    // Observations do not impose a new numerical policy. Finite CPU bit
    // checks conservatively cover normal chains; extremes use the device
    // legacy-style chain and the actual default router's selected outputs.
}

template <typename WeightReader>
void cpu_route_reference(
    const std::vector<float>& hidden,
    std::uint32_t rows,
    std::uint32_t hidden_size,
    std::uint32_t experts,
    std::uint32_t top_k,
    WeightReader&& weight,
    std::vector<std::uint32_t>& indices,
    std::vector<float>& selected_weights,
    bool normalize_selected = true) {
    indices.resize(
        static_cast<std::size_t>(
            rows) *
        top_k);
    selected_weights.resize(
        static_cast<std::size_t>(
            rows) *
        top_k);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        std::vector<float> probabilities(
            experts);
        float maximum =
            -std::numeric_limits<float>::infinity();

        for (std::uint32_t expert = 0;
             expert < experts;
             ++expert) {
            float sum = 0.0F;
            for (std::uint32_t feature = 0;
                 feature < hidden_size;
                 ++feature) {
                sum = std::fma(hidden[static_cast<std::size_t>(row) * hidden_size + feature],
                               weight(expert, feature), sum);
            }
            probabilities[expert] = sum;
            maximum =
                std::max(
                    maximum,
                    sum);
        }

        float denominator = 0.0F;
        for (auto& value :
             probabilities) {
            value =
                std::exp(
                    value - maximum);
            denominator += value;
        }
        for (auto& value :
             probabilities) {
            value /= denominator;
        }

        std::vector<std::uint32_t> order(
            experts);
        std::iota(
            order.begin(),
            order.end(),
            0U);
        std::partial_sort(
            order.begin(),
            order.begin() +
                static_cast<std::ptrdiff_t>(
                    top_k),
            order.end(),
            [&](std::uint32_t left,
                std::uint32_t right) {
                if (probabilities[left] !=
                    probabilities[right]) {
                    return
                        probabilities[left] >
                        probabilities[right];
                }
                return left < right;
            });

        float chosen_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            chosen_sum +=
                probabilities[
                    order[slot]];
        }
        if (!normalize_selected) chosen_sum = 1.0F;

        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            const auto output =
                static_cast<std::size_t>(
                    row) *
                    top_k +
                slot;
            indices[output] =
                order[slot];
            selected_weights[output] =
                probabilities[
                    order[slot]] /
                chosen_sum;
        }
    }
}

void run_case(
    std::uint32_t ggml_type,
    strix::HipActivationStorage activation,
    std::uint32_t expert_count = 512,
    std::uint32_t width = 2560,
    std::uint32_t rows = 1,
    bool normalize_selected = true,
    bool tied = false,
    bool varied = false,
    bool seeded = false,
    bool nan_logits = false,
    bool exceptional_weights = false,
    bool require_distinct_rows = false) {
    auto model =
        strix::ModelContract::
            qwen38_flash_next();
    model.num_experts = expert_count;
    model.hidden_size = width;
    const auto hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    const auto experts =
        static_cast<std::uint32_t>(
            model.num_experts);
    const auto top_k =
        static_cast<std::uint32_t>(
            model.num_experts_per_token);

    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan, true);
    const auto& step =
        first_route_step(plan);
    const auto& step_io =
        io_for(io, step.id);

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = rows;
    buffer_request.parallel_moe_router = true;
    buffer_request.activation_element_bytes =
        static_cast<std::uint32_t>(
            activation);
    buffer_request.control_element_bytes = 4;
    buffer_request.index_element_bytes = 4;
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
        "hipMalloc route arena");

    const auto hidden_address =
        strix::resolve_execution_buffer(
            0x1010ULL,
            arena,
            strix::ExecutionMemoryBinding{
                0x1010ULL,
                reinterpret_cast<
                    std::uintptr_t>(
                    arena_pointer),
                arena.total_bytes,
                1,
            },
            step_io.reads[0]);

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x1010ULL;
    memory.arena_base =
        reinterpret_cast<
            std::uintptr_t>(
                arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto indices_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);
    const auto weights_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[1]);

    auto hidden_float =
        make_hidden(
            static_cast<std::size_t>(
                rows) *
            hidden_size);
    if (tied) std::fill(hidden_float.begin(), hidden_float.end(), 0.0F);
    if (varied)
        for (std::size_t index = 0; index < hidden_float.size(); ++index)
            hidden_float[index] = static_cast<float>(static_cast<int>((index * 11U) % 23U) - 11) / 31.0F;

    // Deterministic bit-pattern generator: mixed exponents and non-BF16 F32
    // significands exercise products that may round differently with contraction.
    std::uint32_t random_state = 0x71d34a29U;
    const auto random_bits = [&] {
        random_state ^= random_state << 13U;
        random_state ^= random_state >> 17U;
        random_state ^= random_state << 5U;
        return random_state;
    };
    const auto mixed_value = [&](std::size_t index, bool weight) {
        const auto random = random_bits();
        std::uint32_t bits = random & 0x80000000U;
        switch (index % 19U) {
        case 0: break;  // signed zero
        case 1: bits |= (random & 0x007fffffU) | 1U; break;  // subnormal
        default:
            // Bound magnitudes to keep logits finite and routes well separated.
            bits |= ((weight ? 118U : 124U) - (random % 8U)) << 23U;
            bits |= (random & 0x007fffffU) | 1U;
            break;
        }
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    if (seeded)
        for (std::size_t index = 0; index < hidden_float.size(); ++index)
            hidden_float[index] = mixed_value(index, false);

    if (require_distinct_rows && rows == 2U)
        for (std::uint32_t feature = 0; feature < hidden_size; ++feature)
            hidden_float[hidden_size + feature] = -hidden_float[feature];
    if (exceptional_weights) {
        expect(rows == 2U, "exceptional fixture has two distinct activation rows");
        hidden_float[0] = 0.0F; hidden_float[hidden_size] = -0.0F;
        hidden_float[1] = 1.0F; hidden_float[hidden_size + 1U] = -1.0F;
    }
    if (nan_logits)
        for (std::uint32_t row = 0; row < rows; ++row)
            hidden_float[static_cast<std::size_t>(row) * hidden_size] = std::numeric_limits<float>::quiet_NaN();

    if (activation ==
        strix::HipActivationStorage::fp32) {
        check(
            hipMemcpy(
                reinterpret_cast<void*>(
                    hidden_address.address),
                hidden_float.data(),
                hidden_float.size() *
                    sizeof(float),
                hipMemcpyHostToDevice),
            "copy FP32 hidden");
    } else {
        std::vector<std::uint16_t> packed(
            hidden_float.size());
        std::transform(
            hidden_float.begin(),
            hidden_float.end(),
            packed.begin(),
            float_to_bf16);
        check(
            hipMemcpy(
                reinterpret_cast<void*>(
                    hidden_address.address),
                packed.data(),
                packed.size() *
                    sizeof(std::uint16_t),
                hipMemcpyHostToDevice),
            "copy BF16 hidden");
    }

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    strix::ManifestTensor router;
    router.name =
        "blk.0.ffn_gate_inp.weight";
    router.dimensions = {
        hidden_size,
        experts,
    };
    router.ggml_type =
        ggml_type;
    router.role =
        strix::TensorRole::moe_router;
    router.layer = 0;

    std::vector<float> f32_weights;
    std::vector<Q8Block> q8_weights;
    void* device_weights = nullptr;

    if (ggml_type == 0 || ggml_type == 1 || ggml_type == 30) {
        f32_weights =
            make_f32_weights(
                experts,
                hidden_size);
        if (varied)
            for (std::uint32_t expert = 0; expert < experts; ++expert)
                for (std::uint32_t feature = 0; feature < hidden_size; ++feature)
                    f32_weights[static_cast<std::size_t>(expert) * hidden_size + feature] =
                        static_cast<float>(static_cast<int>(expert % 23U) - 11) * 0.001F +
                        static_cast<float>(static_cast<int>(feature % 17U) - 8) * 0.0003F +
                        static_cast<float>(expert) * 0.00001F;
        if (seeded) {
            expect(ggml_type == 0, "mixed-exponent fixture uses F32 router weights");
            for (std::size_t index = 0; index < f32_weights.size(); ++index)
                f32_weights[index] = mixed_value(index, true);
        }
        std::vector<std::uint16_t> packed_weights;
        if (ggml_type != 0) {
            packed_weights.resize(f32_weights.size());
            for (std::size_t index = 0; index < f32_weights.size(); ++index) {
                if (ggml_type == 30) {
                    packed_weights[index] = float_to_bf16(f32_weights[index]);
                    f32_weights[index] = bf16_to_float(packed_weights[index]);
                } else {
                    const auto half = __float2half(f32_weights[index]);
                    std::memcpy(&packed_weights[index], &half, sizeof(half));
                    if (varied && index % 41U == 0U)
                        packed_weights[index] = static_cast<std::uint16_t>((index & 1U ? 0x8000U : 0U) |
                            (index % 3U == 0U ? 1U : 0x03ffU));
                    __half decoded;
                    std::memcpy(&decoded, &packed_weights[index], sizeof(decoded));
                    f32_weights[index] = __half2float(decoded);
                }
            }
        }
        if (exceptional_weights) {
            expect(ggml_type == 0, "exceptional router fixture uses F32 coefficients");
            // Exceptions remain row-local in the logits matrix. Hidden feature
            // zero is zero, and feature one has opposite signs in the two rows.
            f32_weights[0U * hidden_size + 1U] = std::numeric_limits<float>::quiet_NaN();
            f32_weights[1U * hidden_size + 1U] = std::numeric_limits<float>::infinity();
            f32_weights[2U * hidden_size + 1U] = -std::numeric_limits<float>::infinity();
            f32_weights[3U * hidden_size + 0U] = std::numeric_limits<float>::infinity();
        }
        router.byte_size =
            f32_weights.size() *
            (ggml_type == 0 ? sizeof(float) : sizeof(std::uint16_t));
        check(
            hipMalloc(
                &device_weights,
                static_cast<std::size_t>(
                    router.byte_size)),
            "hipMalloc F32 router weights");
        check(
            hipMemcpy(
                device_weights,
                ggml_type == 0 ? static_cast<const void*>(f32_weights.data()) : packed_weights.data(),
                static_cast<std::size_t>(
                    router.byte_size),
                hipMemcpyHostToDevice),
            "copy F32 router weights");
    } else if (ggml_type == 8) {
        q8_weights =
            make_q8_weights(
                experts,
                hidden_size);
        if (varied) {
            for (std::size_t index = 0; index < q8_weights.size(); ++index) {
                auto& block = q8_weights[index];
                std::uint16_t scale_bits = 0;
                const auto normal_scale = __float2half(static_cast<float>(static_cast<int>(index % 29U) - 14) / 4096.0F);
                std::memcpy(&scale_bits, &normal_scale, sizeof(scale_bits));
                switch (index % 13U) {
                case 0: scale_bits = 0x0000U; break;
                case 1: scale_bits = 0x8000U; break;
                case 2: scale_bits = 0x0001U; break;
                case 3: scale_bits = 0x83ffU; break;
                default: break;
                }
                std::memcpy(&block.scale, &scale_bits, sizeof(scale_bits));
                for (std::size_t feature = 0; feature < 32U; ++feature) {
                    const auto code = static_cast<int>((index * 71U + feature * 53U) % 256U) - 128;
                    block.values[feature] = static_cast<std::int8_t>(code);
                }
                block.values[0] = -128; block.values[1] = 0; block.values[2] = 127;
            }
        }
        router.byte_size =
            q8_weights.size() *
            sizeof(Q8Block);
        check(
            hipMalloc(
                &device_weights,
                static_cast<std::size_t>(
                    router.byte_size)),
            "hipMalloc Q8 router weights");
        check(
            hipMemcpy(
                device_weights,
                q8_weights.data(),
                static_cast<std::size_t>(
                    router.byte_size),
                hipMemcpyHostToDevice),
            "copy Q8 router weights");
    } else {
        throw std::logic_error(
            "unsupported route test type");
    }

    manifest.total_tensor_bytes =
        router.byte_size;
    manifest.tensors.push_back(
        router);
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    runtime.tensors.push_back(
        {
            0,
            strix::RuntimeTensorStorage::resident,
            reinterpret_cast<
                std::uintptr_t>(
                    device_weights),
            router.byte_size,
            1,
            0,
        });

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x2020ULL;
    access.accesses.push_back(
        {
            step.id,
            strix::SerialOp::moe_route,
            0,
            {0},
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

    strix::ExecutionDispatchContext context;
    context.step_id = step.id;
    context.op = step.op;
    context.layer = step.layer;
    context.reads = step_io.reads;
    context.writes = step_io.writes;
    context.arena = &arena;
    context.memory = &memory;
    context.resources =
        &resources;
    context.logical_rows = rows;
    context.request_concurrency = 1;

    strix::HipMoeRouteLauncherConfig config;
    config.hidden_size =
        hidden_size;
    config.expert_count =
        experts;
    config.top_k =
        top_k;
    config.activation =
        activation;
    config.normalize_selected = normalize_selected;

    auto launcher =
        strix::make_hip_moe_route_launcher(
            "test.moe_route",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "MoE route synchronize");

    std::vector<std::uint32_t> actual_indices(
        static_cast<std::size_t>(
            rows) *
        top_k);
    std::vector<float> actual_weights(
        actual_indices.size());

    check(
        hipMemcpy(
            actual_indices.data(),
            reinterpret_cast<
                const void*>(
                    indices_address.address),
            actual_indices.size() *
                sizeof(std::uint32_t),
            hipMemcpyDeviceToHost),
        "copy route indices");
    check(
        hipMemcpy(
            actual_weights.data(),
            reinterpret_cast<
                const void*>(
                    weights_address.address),
            actual_weights.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy route weights");

    std::vector<float> reference_hidden =
        hidden_float;
    if (activation ==
        strix::HipActivationStorage::bf16) {
        for (auto& value :
             reference_hidden) {
            value =
                bf16_to_float(
                    float_to_bf16(
                        value));
        }
    }

    std::vector<std::uint32_t> expected_indices;
    std::vector<float> expected_weights;

    if (seeded) {
        bool contraction_witness = false;
        for (std::uint32_t row = 0; row < rows && !contraction_witness; ++row) {
            for (std::uint32_t expert = 0; expert < experts && !contraction_witness; ++expert) {
                float fused = 0.0F;
                float separate = 0.0F;
                for (std::uint32_t feature = 0; feature < hidden_size; ++feature) {
                    const auto input = reference_hidden[static_cast<std::size_t>(row) * hidden_size + feature];
                    const auto coefficient = f32_weights[static_cast<std::size_t>(expert) * hidden_size + feature];
                    fused = std::fma(input, coefficient, fused);
                    // Force a rounded product before the separate addition;
                    // this is fixture validation, not the router's CPU oracle.
                    const volatile float product = input * coefficient;
                    separate += product;
                }
                contraction_witness = std::memcmp(&fused, &separate, sizeof(float)) != 0;
            }
        }
        expect(contraction_witness,
               "mixed-exponent fixture distinguishes fused and separate accumulation");
    }

    if (nan_logits || exceptional_weights) {
        expected_indices.assign(static_cast<std::size_t>(rows) * top_k, experts);
        expected_weights.assign(expected_indices.size(), std::numeric_limits<float>::quiet_NaN());
    } else if (ggml_type != 8) {
        cpu_route_reference(
            reference_hidden,
            rows,
            hidden_size,
            experts,
            top_k,
            [&](std::uint32_t expert,
                std::uint32_t feature) {
                return f32_weights[
                    static_cast<
                        std::size_t>(
                        expert) *
                        hidden_size +
                    feature];
            },
            expected_indices,
            expected_weights, normalize_selected);
    } else {
        const auto blocks_per_row =
            hidden_size / 32U;
        cpu_route_reference(
            reference_hidden,
            rows,
            hidden_size,
            experts,
            top_k,
            [&](std::uint32_t expert,
                std::uint32_t feature) {
                const auto& block =
                    q8_weights[
                        static_cast<
                            std::size_t>(
                            expert) *
                            blocks_per_row +
                        feature / 32U];
                return
                    __half2float(
                        block.scale) *
                    static_cast<float>(
                        block.values[
                            feature % 32U]);
            },
            expected_indices,
            expected_weights, normalize_selected);
    }

    const auto read_coefficient = [&](std::uint32_t expert, std::uint32_t feature) {
        if (ggml_type != 8) return f32_weights[static_cast<std::size_t>(expert) * hidden_size + feature];
        const auto& block = q8_weights[static_cast<std::size_t>(expert) * (hidden_size / 32U) + feature / 32U];
        return __half2float(block.scale) * static_cast<float>(block.values[feature % 32U]);
    };
    std::vector<float> expected_logits(static_cast<std::size_t>(rows) * experts);
    std::vector<bool> cpu_oracle_normal_chain(expected_logits.size(), true);
    const auto coefficient_is_half_subnormal = [&](std::uint32_t expert, std::uint32_t feature) {
        std::uint16_t bits = 0;
        if (ggml_type == 1) {
            const auto value = __float2half(f32_weights[static_cast<std::size_t>(expert) * hidden_size + feature]);
            std::memcpy(&bits, &value, sizeof(bits));
        } else if (ggml_type == 8) {
            const auto& block = q8_weights[static_cast<std::size_t>(expert) * (hidden_size / 32U) + feature / 32U];
            std::memcpy(&bits, &block.scale, sizeof(bits));
        }
        return (bits & 0x7c00U) == 0 && (bits & 0x03ffU) != 0;
    };
    for (std::uint32_t row = 0; row < rows; ++row)
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            float sum = 0.0F;
            for (std::uint32_t feature = 0; feature < hidden_size; ++feature) {
                const auto input = reference_hidden[static_cast<std::size_t>(row) * hidden_size + feature];
                const auto coefficient = read_coefficient(expert, feature);
                const auto product = input * coefficient;
                sum = std::fma(input, coefficient, sum);
                if (std::fpclassify(input) == FP_SUBNORMAL || std::fpclassify(coefficient) == FP_SUBNORMAL ||
                    std::fpclassify(product) == FP_SUBNORMAL || std::fpclassify(sum) == FP_SUBNORMAL ||
                    coefficient_is_half_subnormal(expert, feature))
                    cpu_oracle_normal_chain[static_cast<std::size_t>(row) * experts + expert] = false;
            }
            expected_logits[static_cast<std::size_t>(row) * experts + expert] = sum;
        }
    if (require_distinct_rows)
        expect(rows == 2U && !std::equal(actual_indices.begin(), actual_indices.begin() + top_k,
                                        actual_indices.begin() + top_k),
               "row-distinct fixture selects different expert indices in each row");
    if (exceptional_weights) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            const auto offset = static_cast<std::size_t>(row) * experts;
            expect(std::isnan(expected_logits[offset]) && std::isnan(expected_logits[offset + 3U]),
                   "NaN coefficient and zero times infinity stay local to their expert");
            expect(std::isinf(expected_logits[offset + 1U]) && std::isinf(expected_logits[offset + 2U]) &&
                       std::signbit(expected_logits[offset + 1U]) == (row == 1U) &&
                       std::signbit(expected_logits[offset + 2U]) == (row == 0U),
                   "signed infinity logits preserve the distinct row activation signs");
            expect(std::all_of(expected_logits.begin() + static_cast<std::ptrdiff_t>(offset + 4U),
                               expected_logits.begin() + static_cast<std::ptrdiff_t>(offset + experts),
                               [](float value) { return std::isfinite(value); }),
                   "exceptional coefficients do not contaminate unrelated experts");
        }
    }

    const bool route_cpu_qualified = nan_logits || exceptional_weights ||
        std::all_of(cpu_oracle_normal_chain.begin(), cpu_oracle_normal_chain.end(), [](bool value) { return value; });
    if (route_cpu_qualified) expect(
        actual_indices ==
            expected_indices,
        "HIP MoE route selected indices match CPU reference");

    if (!nan_logits && !exceptional_weights)
        expect(std::all_of(actual_weights.begin(), actual_weights.end(), [](float value) { return std::isfinite(value); }),
               "finite route fixture produces finite selected weights");
    if (nan_logits || exceptional_weights)
        expect(std::all_of(actual_weights.begin(), actual_weights.end(),
                           [=](float value) { return normalize_selected ? std::isnan(value) : value == -INFINITY; }),
               "all-NaN logits produce rejected sentinel routes and NaN weights");
    float max_error = 0.0F;
    for (std::size_t index = 0;
         index <
            actual_weights.size();
         ++index) {
        max_error =
            std::max(
                max_error,
                std::abs(
                    actual_weights[index] -
                    expected_weights[index]));
    }
    if (route_cpu_qualified) expect(
        max_error <= 1e-5F,
        "HIP MoE route normalized weights match CPU reference");

    float* reference_logits_device = nullptr;
    check(hipMalloc(&reference_logits_device, expected_logits.size() * sizeof(float)), "allocate legacy logits reference");
    check(hipMemset(reference_logits_device, 0xff, expected_logits.size() * sizeof(float)), "poison legacy logits reference");
    if (activation == strix::HipActivationStorage::fp32)
        run_fixture_legacy_logits<float>(ggml_type, rows, hidden_size, experts,
            reinterpret_cast<const void*>(hidden_address.address), device_weights, reference_logits_device);
    else
        run_fixture_legacy_logits<std::uint16_t>(ggml_type, rows, hidden_size, experts,
            reinterpret_cast<const void*>(hidden_address.address), device_weights, reference_logits_device);
    std::vector<float> device_reference_logits(expected_logits.size());
    check(hipMemcpy(device_reference_logits.data(), reference_logits_device,
                    device_reference_logits.size() * sizeof(float), hipMemcpyDeviceToHost), "read legacy logits reference");
    check(hipFree(reference_logits_device), "free legacy logits reference");
    expect(std::all_of(device_reference_logits.begin(), device_reference_logits.end(), [](float value) {
               std::uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); return bits != 0xffffffffU;
           }), "device legacy-style reference overwrites every poisoned logit");
    const auto strict_entries = std::count(cpu_oracle_normal_chain.begin(), cpu_oracle_normal_chain.end(), true);
    std::cout << "router CPU workspace oracle normal-chain entries " << strict_entries
              << "; extreme-chain entries checked against device legacy-style reference "
              << expected_logits.size() - static_cast<std::size_t>(strict_entries) << '\n';

    context.temporaries = step_io.temporaries;
    const auto workspace_address = strix::resolve_execution_buffer(memory.execution_identity, arena, memory,
                                                                  context.temporaries.front());
    auto parallel_config = config;
    parallel_config.parallel_projection = true;
    auto parallel_launcher = strix::make_hip_moe_route_launcher("test.moe_route.parallel", &parallel_config);
    expect(parallel_launcher.implementation_identity != launcher.implementation_identity,
           "parallel router has a distinct implementation identity");
    const auto verify_parallel = [&] {
        std::vector<std::uint32_t> parallel_indices(actual_indices.size());
        std::vector<float> parallel_weights(actual_weights.size());
        check(hipMemcpy(parallel_indices.data(), reinterpret_cast<void*>(indices_address.address),
                        parallel_indices.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost), "read parallel indices");
        check(hipMemcpy(parallel_weights.data(), reinterpret_cast<void*>(weights_address.address),
                        parallel_weights.size() * sizeof(float), hipMemcpyDeviceToHost), "read parallel weights");
        expect(std::all_of(parallel_indices.begin(), parallel_indices.end(),
                           [](std::uint32_t value) { return value != 0xffffffffU; }),
               "parallel router overwrites every poisoned selected index");
        expect(std::all_of(parallel_weights.begin(), parallel_weights.end(),
                           [](float value) { std::uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); return bits != 0xffffffffU; }),
               "parallel router overwrites every poisoned selected weight");
        std::vector<std::uint32_t> workspace_bits(static_cast<std::size_t>(rows) * experts);
        check(hipMemcpy(workspace_bits.data(), reinterpret_cast<void*>(workspace_address.address),
                        workspace_bits.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost), "read parallel logits workspace");
        expect(std::all_of(workspace_bits.begin(), workspace_bits.end(),
                           [](std::uint32_t bits) { return bits != 0xffffffffU; }),
               "parallel projection overwrites every active logits workspace entry");
        bool logits_match = true;
        for (std::size_t index = 0; index < expected_logits.size(); ++index) {
            float observed = 0.0F;
            std::memcpy(&observed, &workspace_bits[index], sizeof(observed));
            if (std::isnan(expected_logits[index])) logits_match &= std::isnan(observed);
            else if (cpu_oracle_normal_chain[index]) {
                std::uint32_t expected_bits = 0;
                std::memcpy(&expected_bits, &expected_logits[index], sizeof(expected_bits));
                logits_match &= expected_bits == workspace_bits[index];
            }
        }
        expect(logits_match, "normal-chain logits match ascending CPU FMA; exceptional NaNs match class");
        bool device_logits_match = true;
        for (std::size_t index = 0; index < device_reference_logits.size(); ++index) {
            float observed = 0.0F;
            std::memcpy(&observed, &workspace_bits[index], sizeof(observed));
            if (std::isnan(device_reference_logits[index])) device_logits_match &= std::isnan(observed);
            else {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &device_reference_logits[index], sizeof(bits));
                device_logits_match &= bits == workspace_bits[index];
            }
        }
        expect(device_logits_match, "all logits including extremes match the device legacy-style chain (NaNs by class)");
        expect(parallel_indices == actual_indices, "parallel router preserves deterministic selected indices");
        expect(std::memcmp(parallel_weights.data(), actual_weights.data(), parallel_weights.size() * sizeof(float)) == 0,
               "parallel router preserves selected weight bits");
    };
    check(hipMemset(reinterpret_cast<void*>(workspace_address.address), 0xff,
                    workspace_address.bytes), "poison eager parallel logits workspace");
    check(hipMemset(reinterpret_cast<void*>(indices_address.address), 0xff,
                    actual_indices.size() * sizeof(std::uint32_t)), "poison eager parallel indices");
    check(hipMemset(reinterpret_cast<void*>(weights_address.address), 0xff,
                    actual_weights.size() * sizeof(float)), "poison eager parallel weights");
    parallel_launcher.function(context, parallel_launcher.user_data);
    check(hipDeviceSynchronize(), "synchronize eager parallel router");
    verify_parallel();
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create parallel router stream");
    strix::HipLaunchContext launch_context;
    launch_context.stream = stream;
    {
        strix::HipLaunchScope scope(launch_context);
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "capture parallel router");
        parallel_launcher.function(context, parallel_launcher.user_data);
        check(hipStreamEndCapture(stream, &graph), "finish parallel router capture");
    }
    check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate parallel router graph");
    for (unsigned replay = 0; replay < 3; ++replay) {
        // These writes precede each replay in its stream, outside the graph.
        check(hipMemsetAsync(reinterpret_cast<void*>(workspace_address.address), 0xff,
                             workspace_address.bytes, stream), "poison graph parallel logits workspace");
        check(hipMemsetAsync(reinterpret_cast<void*>(indices_address.address), 0xff,
                             actual_indices.size() * sizeof(std::uint32_t), stream), "poison graph parallel indices");
        check(hipMemsetAsync(reinterpret_cast<void*>(weights_address.address), 0xff,
                             actual_weights.size() * sizeof(float), stream), "poison graph parallel weights");
        check(hipGraphLaunch(executable, stream), "replay parallel router graph");
        check(hipStreamSynchronize(stream), "synchronize parallel router graph");
        verify_parallel();
    }
    const auto expect_rejected_workspace = [&](const char* reason, const char* message) {
        bool rejected = false;
        try { parallel_launcher.function(context, parallel_launcher.user_data); }
        catch (const std::invalid_argument& error) { rejected = std::string(error.what()).find(reason) != std::string::npos; }
        expect(rejected, message);
    };
    bool default_extra_rejected = false;
    try { launcher.function(context, launcher.user_data); }
    catch (const std::invalid_argument& error) {
        default_extra_rejected = std::string(error.what()).find("router temporary IO differs") != std::string::npos;
    }
    expect(default_extra_rejected, "default router rejects an extra admitted temporary before launch");
    const auto saved_temporaries = context.temporaries;
    context.temporaries = {};
    expect_rejected_workspace("router temporary IO differs", "parallel router rejects missing workspace");
    context.temporaries = saved_temporaries;
    auto bad_temporaries = std::vector<strix::ExecutionBufferUse>(saved_temporaries.begin(), saved_temporaries.end());
    bad_temporaries[0].buffer.buffer_class = strix::ExecutionBufferClass::hyper_workspace;
    context.temporaries = bad_temporaries;
    expect_rejected_workspace("parallel router temporary must use router_workspace", "parallel router rejects wrong workspace class");
    context.temporaries = saved_temporaries;
    bad_temporaries[0] = saved_temporaries[0];
    ++bad_temporaries[0].generation;
    context.temporaries = bad_temporaries;
    expect_rejected_workspace("parallel router workspace does not belong to this step", "parallel router rejects stale workspace use generation");
    context.temporaries = saved_temporaries;
    auto bad_arena = arena;
    auto* workspace_allocation = const_cast<strix::ExecutionArenaAllocation*>(
        strix::find_arena_allocation(bad_arena, saved_temporaries.front()));
    const auto saved_allocation = *workspace_allocation;
    workspace_allocation->bytes = static_cast<std::uint64_t>(rows) * experts * sizeof(float) - 1U;
    context.arena = &bad_arena;
    expect_rejected_workspace("parallel router requires admitted aligned", "parallel router rejects undersized workspace");
    *workspace_allocation = saved_allocation;
    workspace_allocation->offset += 4U;
    expect_rejected_workspace("parallel router requires admitted aligned", "parallel router rejects unaligned workspace");
    context.arena = &arena;
    const auto saved_generation = memory.allocation_generation;
    memory.allocation_generation = 0;
    expect_rejected_workspace("allocation generation must be non-zero", "parallel router rejects invalid allocation generation");
    memory.allocation_generation = saved_generation;

    check(hipGraphExecDestroy(executable), "destroy parallel router executable");
    check(hipGraphDestroy(graph), "destroy parallel router graph");
    check(hipStreamDestroy(stream), "destroy parallel router stream");

    check(
        hipFree(device_weights),
        "hipFree router weights");
    check(
        hipFree(arena_pointer),
        "hipFree route arena");
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

    strix::HipMoeRouteLauncherConfig config;
    config.hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    config.expert_count =
        static_cast<std::uint32_t>(
            model.num_experts);
    config.top_k =
        static_cast<std::uint32_t>(
            model.num_experts_per_token);
    config.activation =
        strix::HipActivationStorage::bf16;

    strix::kernels::KernelRegistry kernels;
    strix::register_hip_moe_route_kernel_variant(
        model,
        request,
        config,
        kernels);
    auto parallel_config = config;
    parallel_config.parallel_projection = true;
    auto parallel_request = request;
    parallel_request.parallel_moe_router = true;
    strix::kernels::KernelRegistry parallel_kernels;
    strix::register_hip_moe_route_kernel_variant(model, parallel_request, parallel_config, parallel_kernels);
    const auto& parallel_variant = parallel_kernels.variants().front();
    expect(parallel_variant.resources.lds_bytes >= (8U * 260U + 256U) * sizeof(float) &&
           parallel_variant.resources.lds_bytes < 65536U,
           "parallel router declares bounded shared coefficient and reduction storage");
    expect(parallel_variant.id != kernels.variants().front().id &&
           parallel_variant.compiler_identity != kernels.variants().front().compiler_identity &&
           parallel_variant.resources.scratch_bytes == 0,
           "parallel router distinguishes its path and admits temporary storage through the arena");

    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto bindings =
        strix::bind_serial_execution_plan(
            model,
            execution,
            kernels,
            request);

    std::uint32_t route_steps = 0;
    for (std::size_t index = 0;
         index <
            execution.steps.size();
         ++index) {
        if (execution.steps[index].op !=
            strix::SerialOp::moe_route) {
            continue;
        }
        ++route_steps;
        expect(
            bindings.bindings[index].
                kernel_id.has_value(),
            "HIP MoE route catalog binds every route layer");
    }
    expect(
        route_steps ==
            model.hidden_layers &&
        bindings.bound_steps ==
            route_steps,
        "HIP MoE route catalog binds exactly all route steps");

    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x3030ULL;
    compiled.execution =
        execution;
    compiled.bindings =
        bindings;
    compiled.io = strix::build_serial_execution_io_plan(execution);

    strix::ExecutionLauncherRegistry launchers;
    strix::HipMoeRouteLauncherBundle bundle(
        config);
    bundle.register_for_compiled_execution(
        compiled,
        launchers);

    const auto route_it =
        std::find_if(
            bindings.bindings.begin(),
            bindings.bindings.end(),
            [](const auto& binding) {
                return binding.op ==
                    strix::SerialOp::moe_route;
            });
    expect(
        route_it !=
            bindings.bindings.end() &&
        route_it->kernel_id.has_value() &&
        launchers.find_kernel_launcher(
            *route_it->kernel_id) !=
            nullptr,
        "HIP MoE route launcher bundle materializes selected route id");
    auto parallel_compiled = compiled;
    parallel_compiled.bindings = strix::bind_serial_execution_plan(model, execution, parallel_kernels, parallel_request);
    parallel_compiled.io = strix::build_serial_execution_io_plan(execution, true);
    strix::HipMoeRouteLauncherBundle parallel_bundle(parallel_config);
    strix::ExecutionLauncherRegistry parallel_launchers;
    parallel_bundle.register_for_compiled_execution(parallel_compiled, parallel_launchers);
    const auto expect_preflight_rejected = [&](strix::HipMoeRouteLauncherBundle& candidate,
                                               const strix::CompiledSerialExecution& plan, const char* reason) {
        strix::ExecutionLauncherRegistry empty;
        bool rejected = false;
        try { candidate.register_for_compiled_execution(plan, empty); }
        catch (const std::invalid_argument& error) { rejected = std::string(error.what()).find(reason) != std::string::npos; }
        expect(rejected, "router bundle mismatch rejected during registration preflight");
    };
    expect_preflight_rejected(bundle, parallel_compiled, "compiled workspace slots");
    expect_preflight_rejected(parallel_bundle, compiled, "compiled workspace slots");
    auto corrupt = parallel_compiled;
    corrupt.bindings = compiled.bindings;
    expect_preflight_rejected(parallel_bundle, corrupt, "compiled kernel binding");
    corrupt = parallel_compiled;
    for (auto& io : corrupt.io.steps) if (io.op == strix::SerialOp::moe_route) io.temporaries.clear();
    expect_preflight_rejected(parallel_bundle, corrupt, "compiled temporary IO");

}

}  // namespace

int main() try {
    qualify_fixture_denormal_scope();
    test_catalog_and_bundle();
    run_case(
        0,
        strix::HipActivationStorage::fp32);
    run_case(
        8,
        strix::HipActivationStorage::bf16);
    // Exercise partially occupied waves and workgroups as well as a second
    // expert iteration. Inactive lanes still participate in both reductions.
    for (const auto experts : {31U, 32U, 33U, 255U, 256U, 257U, 511U}) {
        run_case(0, strix::HipActivationStorage::fp32, experts);
        run_case(8, strix::HipActivationStorage::bf16, experts);
    }
    for (const auto experts : {33U, 255U, 256U, 257U, 511U, 512U}) {
        run_case(0, strix::HipActivationStorage::fp32, experts, 67, 2, true, false, true);
        run_case(0, strix::HipActivationStorage::bf16, experts, 67, 2, false, false, true);
    }
    run_case(0, strix::HipActivationStorage::fp32, 257, 67, 2, true, true);
    run_case(8, strix::HipActivationStorage::bf16, 511, 96, 2, false, true);
    run_case(1, strix::HipActivationStorage::fp32, 257, 67, 2, true, false, true);
    run_case(30, strix::HipActivationStorage::bf16, 255, 67, 2, false, false, true);

    // Production activation/weight pairing at the full router geometry.
    for (const bool normalize : {false, true}) {
        run_case(0, strix::HipActivationStorage::bf16, 512, 2560, 2, normalize, false, true);
        run_case(0, strix::HipActivationStorage::bf16, 512, 2560, 2, normalize, false, false, true);
        run_case(0, strix::HipActivationStorage::fp32, 512, 2560, 2, normalize, false, false, true);
    }

    run_case(0, strix::HipActivationStorage::bf16, 512, 2560, 2, true, false, false, false, true);
    run_case(0, strix::HipActivationStorage::bf16, 33, 67, 2, false, false, false, false, true);

    // All eight typed paths at canonical geometry and two genuinely distinct rows.
    for (const auto activation : {strix::HipActivationStorage::fp32, strix::HipActivationStorage::bf16}) {
        for (const auto type : {0U, 1U, 30U, 8U})
            run_case(type, activation, 512, 2560, 2, true, false, true, false, false, false, true);
        for (const auto type : {0U, 1U, 30U})
            for (const auto width : {257U, 523U, 2627U})
                run_case(type, activation, 33, width, 2, false, false, true, false, false, false, true);
        for (const auto width : {288U, 2592U})
            run_case(8, activation, 33, width, 2, false, false, true, false, false, false, true);
        run_case(0, activation, 33, 523, 2, true, false, true, false, false, true);
        run_case(0, activation, 33, 523, 2, false, false, true, false, false, true);
    }

    if (failures == 0) {
        std::cout
            << "HIP MoE route launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP MoE route launcher tests: "
        << error.what() << '\n';
    return 2;
}
