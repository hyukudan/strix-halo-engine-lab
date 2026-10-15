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

// Independent serial-insertion selection oracle; extends only fixture-local arrays to32.
__device__ bool fixture_select_better(float a, std::uint32_t ai, float b, std::uint32_t bi) {
    return a > b || (a == b && ai < bi);
}
__global__ void fixture_insertion32_select_kernel(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float logits[512];
    __shared__ float scratch[256];
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x)
        logits[expert] = workspace[static_cast<std::uint64_t>(row) * expert_count + expert];
    __syncthreads();

    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    __syncthreads();

    if (lane == 0) {
        float best_values[32];
        std::uint32_t best_indices[
            32];

        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            best_values[slot] =
                -INFINITY;
            best_indices[slot] =
                expert_count;
        }

        for (std::uint32_t expert = 0;
             expert < expert_count;
             ++expert) {
            const float probability =
                logits[expert];

            for (std::uint32_t slot = 0;
                 slot < top_k;
                 ++slot) {
                if (!fixture_select_better(
                        probability,
                        expert,
                        best_values[slot],
                        best_indices[slot])) {
                    continue;
                }

                for (std::uint32_t move =
                         top_k - 1U;
                     move > slot;
                     --move) {
                    best_values[move] =
                        best_values[
                            move - 1U];
                    best_indices[move] =
                        best_indices[
                            move - 1U];
                }
                best_values[slot] =
                    probability;
                best_indices[slot] =
                    expert;
                break;
            }
        }

        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
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
    bool require_distinct_rows = false, float logit_spread = 0.0F, std::uint32_t requested_top_k = 0, bool wave_only = false, unsigned injected_logits = 0) {
    auto model =
        strix::ModelContract::
            qwen38_flash_next();
    model.num_experts = expert_count;
    if (requested_top_k != 0) model.num_experts_per_token = requested_top_k;
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

    if (logit_spread != 0.0F) {
        std::fill(hidden_float.begin(), hidden_float.end(), 0.0F);
        for (std::uint32_t row = 0; row < rows; ++row) hidden_float[static_cast<std::size_t>(row) * hidden_size] = 1.0F;
    }
    if (injected_logits != 0) {
        std::fill(hidden_float.begin(), hidden_float.end(), 0.0F);
        for (std::uint32_t row = 0; row < rows; ++row) hidden_float[static_cast<std::size_t>(row)*hidden_size] = 1.0F;
    }
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
        if (logit_spread != 0.0F) {
            expect(ggml_type == 0, "spread fixture uses F32 weights");
            std::fill(f32_weights.begin(), f32_weights.end(), 0.0F);
            for (std::uint32_t expert = 1; expert < experts; ++expert)
                f32_weights[static_cast<std::size_t>(expert) * hidden_size] = -logit_spread - static_cast<float>(expert % 3U);
        }
        if (injected_logits != 0) {
            expect(ggml_type == 0, "exact arbitrary-logit injection uses F32 weights");
            std::fill(f32_weights.begin(), f32_weights.end(), 0.0F);
            for (std::uint32_t e = 0; e < experts; ++e) {
                float value = 0.0F;
                switch (injected_logits) {
                case 1: value = 0.25F; break;
                case 2: value = e < 64U ? 1.0F : 0.0F; break;
                case 3: value = static_cast<float>((e / 2U) % 32U); break;
                case 4: value = e == 0 ? 0.0F : -100.0F; break;
                case 5: value = e == 0 ? INFINITY : 0.0F; break;
                case 6: value = -INFINITY; break;
                case 7: value = e < 3U ? static_cast<float>(e) : -INFINITY; break;
                default: value = e == 0 ? std::numeric_limits<float>::quiet_NaN() : 0.0F; break;
                }
                f32_weights[static_cast<std::size_t>(e)*hidden_size] = value;
            }
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
    if (wave_only) {
        config.parallel_projection = true; config.wave_topk = true;
        context.temporaries = step_io.temporaries;
        const auto work = strix::resolve_execution_buffer(memory.execution_identity, arena, memory, context.temporaries.front());
        check(hipMemset(reinterpret_cast<void*>(work.address), 0xff, work.bytes), "poison wave32 eager workspace");
        check(hipMemset(reinterpret_cast<void*>(indices_address.address), 0xff, indices_address.bytes), "poison wave32 eager indices");
        check(hipMemset(reinterpret_cast<void*>(weights_address.address), 0xff, weights_address.bytes), "poison wave32 eager weights");
    }

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

    if (wave_only) {
        const auto work = strix::resolve_execution_buffer(memory.execution_identity, arena, memory, context.temporaries.front());
        // Fresh reference destinations, and actual projection logits from the production launcher.
        check(hipMemset(reinterpret_cast<void*>(indices_address.address), 0xff, indices_address.bytes), "poison insertion32 indices");
        check(hipMemset(reinterpret_cast<void*>(weights_address.address), 0xff, weights_address.bytes), "poison insertion32 weights");
        hipLaunchKernelGGL(fixture_insertion32_select_kernel, dim3(rows), dim3(256), 0, hipStreamLegacy,
            reinterpret_cast<const float*>(work.address), experts, top_k, normalize_selected,
            reinterpret_cast<std::uint32_t*>(indices_address.address), reinterpret_cast<float*>(weights_address.address));
        check(hipGetLastError(), "insertion32 reference launch");
        check(hipDeviceSynchronize(), "insertion32 reference completion");
        std::vector<std::uint32_t> reference_indices(actual_indices.size());
        std::vector<float> reference_weights(actual_weights.size());
        check(hipMemcpy(reference_indices.data(), reinterpret_cast<void*>(indices_address.address), reference_indices.size()*4, hipMemcpyDeviceToHost), "read insertion32 indices");
        check(hipMemcpy(reference_weights.data(), reinterpret_cast<void*>(weights_address.address), reference_weights.size()*4, hipMemcpyDeviceToHost), "read insertion32 weights");
        expect(reference_indices == actual_indices && std::memcmp(reference_weights.data(), actual_weights.data(), actual_weights.size()*4)==0,
               "wave32 eager matches independent device insertion oracle bits including NaNs");
        hipStream_t wave_stream = nullptr; hipGraph_t wave_graph = nullptr; hipGraphExec_t wave_exec = nullptr;
        check(hipStreamCreateWithFlags(&wave_stream, hipStreamNonBlocking), "wave32 stream");
        strix::HipLaunchContext scope_context; scope_context.stream = wave_stream;
        { strix::HipLaunchScope scope(scope_context);
          check(hipStreamBeginCapture(wave_stream, hipStreamCaptureModeThreadLocal), "capture wave32");
          launcher.function(context,launcher.user_data);
          check(hipStreamEndCapture(wave_stream,&wave_graph), "end wave32 capture"); }
        std::size_t nodes=0; check(hipGraphGetNodes(wave_graph,nullptr,&nodes), "wave32 graph count");
        expect(nodes==2, "wave32 graph contains projection and selection only");
        check(hipGraphInstantiate(&wave_exec,wave_graph,nullptr,nullptr,0), "instantiate wave32");
        for (unsigned replay=0; replay<3; ++replay) {
            check(hipMemsetAsync(reinterpret_cast<void*>(work.address),0xff,work.bytes,wave_stream),"poison wave32 replay workspace");
            check(hipMemsetAsync(reinterpret_cast<void*>(indices_address.address),0xff,indices_address.bytes,wave_stream),"poison wave32 replay indices");
            check(hipMemsetAsync(reinterpret_cast<void*>(weights_address.address),0xff,weights_address.bytes,wave_stream),"poison wave32 replay weights");
            check(hipGraphLaunch(wave_exec,wave_stream),"wave32 replay");check(hipStreamSynchronize(wave_stream),"wave32 replay completion");
            check(hipMemcpy(actual_indices.data(),reinterpret_cast<void*>(indices_address.address),actual_indices.size()*4,hipMemcpyDeviceToHost),"read wave32 replay indices");
            check(hipMemcpy(actual_weights.data(),reinterpret_cast<void*>(weights_address.address),actual_weights.size()*4,hipMemcpyDeviceToHost),"read wave32 replay weights");
            expect(reference_indices==actual_indices && std::memcmp(reference_weights.data(),actual_weights.data(),actual_weights.size()*4)==0,
                   "poisoned wave32 captured replay matches insertion oracle bits");
        }
        check(hipGraphExecDestroy(wave_exec),"destroy wave32 exec");check(hipGraphDestroy(wave_graph),"destroy wave32 graph");
        check(hipStreamDestroy(wave_stream),"destroy wave32 stream");check(hipFree(device_weights),"free wave32 weights");
        check(hipFree(arena_pointer),"free wave32 arena");return;
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

    // FMA normal-chain qualification does not establish exp's subnormal policy.
    // Spread fixtures use actual legacy-device outputs, including any FTZ ties.
    const bool route_cpu_qualified = nan_logits || exceptional_weights ||
        (logit_spread == 0.0F && std::all_of(cpu_oracle_normal_chain.begin(), cpu_oracle_normal_chain.end(), [](bool value) { return value; }));
    if (route_cpu_qualified) expect(
        actual_indices ==
            expected_indices,
        "HIP MoE route selected indices match CPU reference");

    if (!nan_logits && !exceptional_weights)
        expect(std::all_of(actual_weights.begin(), actual_weights.end(), [](float value) { return std::isfinite(value); }),
               "finite route fixture produces finite selected weights");
    if (logit_spread != 0.0F && top_k > 1U) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t slot = 1; slot < top_k; ++slot) {
                const auto value = actual_weights[static_cast<std::size_t>(row) * top_k + slot];
                expect(value == 0.0F || std::fpclassify(value) == FP_SUBNORMAL,
                       "spread fixture reaches actual subnormal-or-zero selected tails");
                if (logit_spread >= 120.0F) expect(value == 0.0F, "wide spread reaches underflow-zero index ties");
            }
        }
    }
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
    auto ranked_config = parallel_config;
    ranked_config.ranked_topk = true;
    auto ranked_launcher = strix::make_hip_moe_route_launcher("test.moe_route.ranked.parallel", &ranked_config);
    expect(ranked_launcher.implementation_identity != parallel_launcher.implementation_identity,
           "ranked selection has a distinct implementation identity");
    const auto poison_ranked = [&] {
        check(hipMemsetAsync(reinterpret_cast<void*>(workspace_address.address), 0xff, workspace_address.bytes, stream), "poison ranked workspace");
        check(hipMemsetAsync(reinterpret_cast<void*>(indices_address.address), 0xff, actual_indices.size() * sizeof(std::uint32_t), stream), "poison ranked indices");
        check(hipMemsetAsync(reinterpret_cast<void*>(weights_address.address), 0xff, actual_weights.size() * sizeof(float), stream), "poison ranked weights");
    };
    {
        strix::HipLaunchScope scope(launch_context);
        poison_ranked();
        ranked_launcher.function(context, ranked_launcher.user_data);
    }
    check(hipStreamSynchronize(stream), "synchronize ranked eager router"); verify_parallel();
    hipGraph_t ranked_graph = nullptr; hipGraphExec_t ranked_executable = nullptr;
    {
        strix::HipLaunchScope scope(launch_context);
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "capture ranked router");
        ranked_launcher.function(context, ranked_launcher.user_data);
        check(hipStreamEndCapture(stream, &ranked_graph), "finish ranked router capture");
    }
    const auto captured_functions = [&](hipGraph_t captured) {
        std::size_t count = 0; check(hipGraphGetNodes(captured, nullptr, &count), "count captured router nodes");
        expect(count == 2U, "router capture contains exactly logits and selection nodes");
        std::vector<hipGraphNode_t> nodes(count);
        check(hipGraphGetNodes(captured, nodes.data(), &count), "read captured router nodes");
        std::vector<void*> functions;
        for (const auto node : nodes) {
            std::size_t dependencies = 0;
            check(hipGraphNodeGetDependencies(node, nullptr, &dependencies), "read router node dependencies");
            hipGraphNodeType type; check(hipGraphNodeGetType(node, &type), "read router node type");
            expect(type == hipGraphNodeTypeKernel, "each captured router node is a kernel");
            if (type != hipGraphNodeTypeKernel) continue;
            hipKernelNodeParams params{}; check(hipGraphKernelNodeGetParams(node, &params), "read captured router kernel function");
            if (dependencies == 0U) functions.insert(functions.begin(), params.func);
            else functions.push_back(params.func);
        }
        return functions;
    };
    const auto old_functions = captured_functions(graph), new_functions = captured_functions(ranked_graph);
    unsigned shared_functions = 0;
    for (const auto function : old_functions)
        shared_functions += std::find(new_functions.begin(), new_functions.end(), function) != new_functions.end() ? 1U : 0U;
    expect(old_functions.size() == 2U && new_functions.size() == 2U && shared_functions == 1U,
           "captures retain the identical projection function and a distinct selection function");
    if (old_functions.size() == 2U && new_functions.size() == 2U)
        expect(old_functions[0] == new_functions[0] && old_functions[1] != new_functions[1],
               "actual capture root projection is common and dependent selection function differs");
    check(hipGraphInstantiate(&ranked_executable, ranked_graph, nullptr, nullptr, 0), "instantiate ranked router graph");
    for (unsigned replay = 0; replay < 3; ++replay) {
        poison_ranked();
        check(hipGraphLaunch(ranked_executable, stream), "replay ranked router graph");
        check(hipStreamSynchronize(stream), "synchronize ranked router graph"); verify_parallel();
    }
    auto wave_config = parallel_config; wave_config.wave_topk = true;
    auto wave_launcher = strix::make_hip_moe_route_launcher("test.moe_route.wave.parallel", &wave_config);
    expect(wave_launcher.implementation_identity != ranked_launcher.implementation_identity &&
               wave_launcher.implementation_identity != parallel_launcher.implementation_identity,
           "wave top-k has its own implementation identity");
    { strix::HipLaunchScope scope(launch_context); poison_ranked(); wave_launcher.function(context,wave_launcher.user_data); }
    check(hipStreamSynchronize(stream),"wave eager completion");verify_parallel();
    hipGraph_t wave_graph = nullptr; hipGraphExec_t wave_executable = nullptr;
    { strix::HipLaunchScope scope(launch_context);
      check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture wave router");
      wave_launcher.function(context,wave_launcher.user_data);
      check(hipStreamEndCapture(stream,&wave_graph),"finish wave router capture"); }
    const auto wave_functions = captured_functions(wave_graph);
    if (old_functions.size()==2 && new_functions.size()==2 && wave_functions.size()==2)
        expect(old_functions[0]==wave_functions[0] && old_functions[1]!=wave_functions[1] && new_functions[1]!=wave_functions[1],
               "wave retains original projection and captures a third selection function");
    check(hipGraphInstantiate(&wave_executable,wave_graph,nullptr,nullptr,0),"instantiate wave router");
    for (unsigned replay=0;replay<3;++replay) {
        poison_ranked();check(hipGraphLaunch(wave_executable,stream),"replay wave router");
        check(hipStreamSynchronize(stream),"wave router replay completion");verify_parallel();
    }
    check(hipGraphExecDestroy(wave_executable),"destroy wave executable");check(hipGraphDestroy(wave_graph),"destroy wave graph");
    check(hipGraphExecDestroy(ranked_executable), "destroy ranked router executable");
    check(hipGraphDestroy(ranked_graph), "destroy ranked router graph");
    const auto expect_rejected_workspace = [&](const char* reason, const char* message) {
        for (const auto* candidate : {&parallel_launcher, &ranked_launcher, &wave_launcher}) {
            bool rejected = false;
            try { candidate->function(context, candidate->user_data); }
            catch (const std::invalid_argument& error) { rejected = std::string(error.what()).find(reason) != std::string::npos; }
            expect(rejected, message);
        }
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

    const auto default_receipt = strix::make_hip_moe_route_launcher("receipt.default", &config);
    expect(default_receipt.implementation_identity == 15352140532421398971ULL,
           "default router retains the fixed implementation receipt");

    strix::kernels::KernelRegistry kernels;
    strix::register_hip_moe_route_kernel_variant(
        model,
        request,
        config,
        kernels);
    auto parallel_config = config;
    parallel_config.parallel_projection = true;
    const auto parallel_receipt = strix::make_hip_moe_route_launcher("receipt.parallel", &parallel_config);
    expect(parallel_receipt.implementation_identity == 5065166250755998983ULL,
           "parallel-only router retains the fixed implementation receipt");
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
    auto ranked_config = parallel_config; ranked_config.ranked_topk = true;
    auto ranked_request = parallel_request; ranked_request.ranked_moe_router_topk = true;
    strix::kernels::KernelRegistry ranked_kernels;
    strix::register_hip_moe_route_kernel_variant(model, ranked_request, ranked_config, ranked_kernels);
    const auto& ranked_variant = ranked_kernels.variants().front();
    expect(ranked_variant.id.ends_with(".ranked.parallel") &&
               ranked_variant.compiler_identity == "hip-gfx1151-moe-route-v5-ranked-parallel" &&
               ranked_variant.resources.lds_bytes == parallel_variant.resources.lds_bytes,
           "ranked catalog distinguishes identity within the unchanged resource maximum");
    auto ranked_compiled = parallel_compiled;
    ranked_compiled.bindings = strix::bind_serial_execution_plan(model, execution, ranked_kernels, ranked_request);
    strix::HipMoeRouteLauncherBundle ranked_bundle(ranked_config);
    strix::ExecutionLauncherRegistry ranked_launchers;
    ranked_bundle.register_for_compiled_execution(ranked_compiled, ranked_launchers);
    expect_preflight_rejected(ranked_bundle, parallel_compiled, "compiled kernel binding");
    expect_preflight_rejected(parallel_bundle, ranked_compiled, "compiled kernel binding");
    auto wave_config = parallel_config; wave_config.wave_topk = true;
    auto wave_request = parallel_request; wave_request.wave_moe_router_topk = true;
    strix::kernels::KernelRegistry wave_kernels;
    strix::register_hip_moe_route_kernel_variant(model,wave_request,wave_config,wave_kernels);
    expect(wave_kernels.variants().front().id.ends_with(".wave.parallel") &&
           wave_kernels.variants().front().compiler_identity=="hip-gfx1151-moe-route-v6-wave-parallel",
           "wave catalog suffix and compiler identity distinct");
    auto wave_compiled = parallel_compiled;
    wave_compiled.bindings = strix::bind_serial_execution_plan(model,execution,wave_kernels,wave_request);
    strix::HipMoeRouteLauncherBundle wave_bundle(wave_config);strix::ExecutionLauncherRegistry wave_launchers;
    wave_bundle.register_for_compiled_execution(wave_compiled,wave_launchers);
    expect_preflight_rejected(wave_bundle,ranked_compiled,"compiled kernel binding");
    expect_preflight_rejected(ranked_bundle,wave_compiled,"compiled kernel binding");
    expect_preflight_rejected(wave_bundle,parallel_compiled,"compiled kernel binding");
    for (unsigned invalid=0;invalid<3;++invalid) {
        auto bad=wave_config;
        if(invalid==0)bad.parallel_projection=false;
        if(invalid==1)bad.ranked_topk=true;
        if(invalid==2)bad.top_k=33;
        bool rejected=false;try{(void)strix::make_hip_moe_route_launcher("bad.wave.parallel",&bad);}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"wave config rejects missing parallel, ranked collision and K33");
    }
    bool invalid_ranked = false;
    auto bad_ranked_config = config; bad_ranked_config.ranked_topk = true;
    try { (void)strix::make_hip_moe_route_launcher("invalid.ranked.parallel", &bad_ranked_config); }
    catch (const std::invalid_argument&) { invalid_ranked = true; }
    expect(invalid_ranked, "ranked launcher requires parallel projection");
    corrupt = parallel_compiled;
    for (auto& io : corrupt.io.steps) if (io.op == strix::SerialOp::moe_route) io.temporaries.clear();
    expect_preflight_rejected(parallel_bundle, corrupt, "compiled temporary IO");

}

}  // namespace

int main() try {
    qualify_fixture_denormal_scope();
    for (const auto experts : {1U, 31U, 255U, 256U, 257U, 511U, 512U})
        for (const auto k : {1U, std::min(experts, 16U)})
            for (const bool normalize : {false, true}) {
                run_case(0, strix::HipActivationStorage::fp32, experts, 67, 1, normalize, true, false, false, false, false, false, 0.0F, k);
                run_case(0, strix::HipActivationStorage::fp32, experts, 67, 1, normalize, false, false, false, true, false, false, 0.0F, k);
                for (const float spread : {90.0F, 100.0F, 120.0F})
                    run_case(0, strix::HipActivationStorage::fp32, experts, 67, 1, normalize, false, false, false, false, false, false, spread, k);
            }

    // K32 is launcher-only: its independent literal device insertion oracle
    // has32 fixture slots while every existing production kernel stays16.
    for (const auto experts : {33U,255U,257U,500U,511U,512U})
        for (const auto activation : {strix::HipActivationStorage::bf16,strix::HipActivationStorage::fp32})
            for (bool normalize : {false,true})
                for (unsigned pattern=1;pattern<=8;++pattern)
                    run_case(0,activation,experts,67,2,normalize,false,false,false,false,false,false,0.0F,32,true,pattern);
    // Independent oracle also exercises K1/8/10/16 and arbitrary-logit states,
    // including owner-stride ties, underflow, infinities and partial NaN.
    for (auto k : {1U,8U,10U,16U}) for (unsigned pattern=1;pattern<=8;++pattern)
        run_case(0,strix::HipActivationStorage::bf16,512,67,2,true,false,false,false,false,false,false,0.0F,k,true,pattern);
    for (const auto activation : {strix::HipActivationStorage::bf16,strix::HipActivationStorage::fp32})
        for (auto type : {0U,1U,30U,8U}) for (bool normalize : {false,true})
            for (auto k : {10U,32U})
                run_case(type,activation,512,96,4,normalize,false,false,false,false,false,false,0.0F,k,true);
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
