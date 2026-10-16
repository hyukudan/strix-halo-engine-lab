#include "strix/hip_launch_context.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/q51_slot_parallel_contract.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/hip_setup_quiescence.hpp"
#ifdef STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC
#include "strix/moe_batched_activation_diagnostic.hpp"
#endif
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/hip_q51_grouped_down_probe.hpp"
#endif
#include <array>
#include <limits>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x4d4f454558505255ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

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

struct IQ4NLBlock {
    __half d;
    std::uint8_t qs[16];
};

struct Q51Block {
    __half d;
    __half m;
    std::uint8_t qh[4];
    std::uint8_t qs[16];
};

struct Q4KBlock {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};

struct Q5KBlock {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);
static_assert(sizeof(IQ4NLBlock) == 18);
static_assert(sizeof(Q51Block) == 24);
static_assert(sizeof(Q4KBlock) == 144);
static_assert(sizeof(Q5KBlock) == 176);

__device__ __constant__
std::int8_t iq4_codebook[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

enum class WeightKind {
    f32,
    f16,
    bf16,
    q8_0,
    iq4_nl,
    iq4_xs,
    q5_1,
    q4_k,
    q5_k,
};

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct RoutedTensors {
    TensorPair gate_up;
    TensorPair gate;
    TensorPair up;
    TensorPair down;
    bool fused_gate_up = false;
};

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

void validate_config(
    const HipMoeExpertsLauncherConfig& config) {
    if (config.batched_slot_activation && !config.q4k_a8g32_gate_up)
        throw std::invalid_argument("batched routed activation requires Q4K A8 independent slot workspace");
    if(config.grouped_exact_q51_down && !config.q4k_a8g32_gate_up)
        throw std::invalid_argument("grouped exact Q5_1 down requires Q4K A8 slot workspace");
    if (config.q51_slot_parallel_down && !config.grouped_exact_q51_down)
        throw std::invalid_argument("slot-parallel Q5_1 down requires grouped exact Q5_1 workspace");
    if (config.q4k_a8g32_gate_up) {
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
        throw std::invalid_argument("Q4K A8 runtime capability is not built");
#else
        (void)moe_q4k_a8::build_runtime_workspace(config.hidden_size,
            config.intermediate_size, config.expert_count, config.top_k, config.q51_slot_parallel_down);
#endif
    }
    if (config.hidden_size == 0 ||
        config.intermediate_size == 0 ||
        config.expert_count == 0 ||
        config.top_k == 0 ||
        config.top_k > config.expert_count) {
        throw std::invalid_argument(
            "HIP routed-expert geometry is invalid");
    }
    if (config.activation !=
            HipActivationStorage::bf16 &&
        config.activation !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP routed-expert activation storage is unsupported");
    }
}

const char* storage_name(
    HipActivationStorage storage) noexcept {
    switch (storage) {
    case HipActivationStorage::bf16:
        return "bf16";
    case HipActivationStorage::fp32:
        return "fp32";
    }
    return "unknown";
}

WeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 7: return WeightKind::q5_1;
    case 12: return WeightKind::q4_k;
    case 13: return WeightKind::q5_k;
    case 20: return WeightKind::iq4_nl;
    case 23: return WeightKind::iq4_xs;
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP routed experts support F32, F16, BF16, Q8_0, IQ4_NL, IQ4_XS, Q5_1, Q4_K and Q5_K");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(bits) << 16U);
}

__device__ float load_activation(
    const float* value) {
    return *value;
}

__device__ float load_activation(
    const std::uint16_t* value) {
    return bf16_to_float(*value);
}

__device__ void store_activation(
    float* destination,
    float value) {
    *destination = value;
}

__device__ void store_activation(
    std::uint16_t* destination,
    float value) {
    const auto bits =
        __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    *destination =
        static_cast<std::uint16_t>(
            rounded >> 16U);
}

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
             (4U * (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return static_cast<std::uint8_t>(
        low |
        static_cast<std::uint8_t>(
            high << 4U));
}

template <WeightKind Kind>
__device__ float load_weight(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return static_cast<const float*>(weights)[
        row * row_width + feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(weights);
    return __half2float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(weights);
    return bf16_to_float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row * blocks_per_row +
            feature / 32U];
    return __half2float(block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_nl>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks = static_cast<const IQ4NLBlock*>(weights);
    const auto& block = blocks[row * (row_width / 32U) + feature / 32U];
    const auto within = feature % 32U;
    const auto packed = block.qs[within % 16U];
    const auto code = within < 16U
        ? static_cast<std::uint8_t>(packed & 0x0fU)
        : static_cast<std::uint8_t>(packed >> 4U);
    return __half2float(block.d) * static_cast<float>(iq4_codebook[code]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row * blocks_per_row +
            feature / 256U];

    const auto within =
        feature % 256U;
    const auto subblock =
        within / 32U;
    const auto suboffset =
        within % 32U;
    const auto packed =
        block.qs[
            subblock * 16U +
            suboffset % 16U];
    const auto code =
        suboffset < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    const int local =
        static_cast<int>(
            iq4_local_scale(
                block,
                subblock)) -
        32;
    return __half2float(block.d) *
           static_cast<float>(local) *
           static_cast<float>(
               iq4_codebook[code]);
}

template <>
__device__ float load_weight<WeightKind::q5_1>(
    const void* weights, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    const auto* blocks = static_cast<const Q51Block*>(weights);
    const auto& block = blocks[row * (row_width / 32U) + feature / 32U];
    const auto lane = feature % 32U;
    const auto byte = block.qs[lane % 16U];
    const auto low = lane < 16U ? byte & 15U : byte >> 4U;
    const auto high = (block.qh[lane / 8U] >> (lane % 8U)) & 1U;
    return __half2float(block.d) * static_cast<float>(low | (high << 4U)) +
           __half2float(block.m);
}

// K-block groups carry unsigned six-bit scale and minimum factors in twelve
// bytes. The last four groups use the high bits of the first eight bytes.
__device__ void k_group_factors(
    const std::uint8_t* packed, std::uint32_t group,
    std::uint32_t& scale, std::uint32_t& minimum) {
    if (group < 4U) {
        scale = packed[group] & 63U;
        minimum = packed[group + 4U] & 63U;
    } else {
        scale = (packed[group + 4U] & 15U) | ((static_cast<std::uint32_t>(packed[group - 4U]) >> 6U) << 4U);
        minimum = (static_cast<std::uint32_t>(packed[group + 4U]) >> 4U) | ((static_cast<std::uint32_t>(packed[group]) >> 6U) << 4U);
    }
}

template <typename Block>
__device__ float load_k_weight(
    const Block* blocks, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    const auto& block = blocks[row * (row_width / 256U) + feature / 256U];
    const auto lane = feature % 256U;
    const auto group = lane / 32U;
    const auto byte = block.qs[(group / 2U) * 32U + lane % 32U];
    auto code = group % 2U == 0U ? byte & 15U : byte >> 4U;
    if constexpr (sizeof(Block) == sizeof(Q5KBlock)) {
        code |= ((block.qh[lane % 32U] >> group) & 1U) << 4U;
    }
    std::uint32_t scale = 0, minimum = 0;
    k_group_factors(block.scales, group, scale, minimum);
    return (__half2float(block.d) * static_cast<float>(scale)) * static_cast<float>(code) -
           __half2float(block.dmin) * static_cast<float>(minimum);
}

template <>
__device__ float load_weight<WeightKind::q4_k>(
    const void* weights, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    return load_k_weight(static_cast<const Q4KBlock*>(weights), row_width, row, feature);
}

template <>
__device__ float load_weight<WeightKind::q5_k>(
    const void* weights, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    return load_k_weight(static_cast<const Q5KBlock*>(weights), row_width, row, feature);
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input,
    std::uint64_t output_rows,
    std::uint32_t experts) {
    std::uint64_t row_bytes = 0;
    switch (kind) {
    case WeightKind::f32:
        row_bytes =
            static_cast<std::uint64_t>(input) * 4ULL;
        break;
    case WeightKind::f16:
    case WeightKind::bf16:
        row_bytes =
            static_cast<std::uint64_t>(input) * 2ULL;
        break;
    case WeightKind::q8_0:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 routed-expert input width must be divisible by 32");
        }
        row_bytes =
            static_cast<std::uint64_t>(
                input / 32U) *
            sizeof(Q8Block);
        break;
    case WeightKind::iq4_xs:
        if (input % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS routed-expert input width must be divisible by 256");
        }
        row_bytes =
            static_cast<std::uint64_t>(
                input / 256U) *
            sizeof(IQ4XSBlock);
        break;
    case WeightKind::iq4_nl:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "IQ4_NL routed-expert input width must be divisible by 32");
        }
        row_bytes = static_cast<std::uint64_t>(input / 32U) * sizeof(IQ4NLBlock);
        break;
    case WeightKind::q5_1:
        if (input % 32U != 0) {
            throw std::invalid_argument("Q5_1 routed-expert input width must be divisible by 32");
        }
        row_bytes = static_cast<std::uint64_t>(input / 32U) * sizeof(Q51Block);
        break;
    case WeightKind::q4_k:
    case WeightKind::q5_k:
        if (input % 256U != 0) {
            throw std::invalid_argument("K-quant routed-expert input width must be divisible by 256");
        }
        row_bytes = static_cast<std::uint64_t>(input / 256U) *
            (kind == WeightKind::q4_k ? sizeof(Q4KBlock) : sizeof(Q5KBlock));
        break;
    }
    return row_bytes *
           output_rows *
           experts;
}

template <
    typename Input,
    WeightKind Kind>
__global__ void selected_projection_kernel(
    const Input* input,
    const std::uint32_t* route_indices,
    std::uint32_t slot,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t expert_count,
    float* output, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }
    const auto expert =
        route_indices[slot];
    if (expert >= expert_count) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    const auto row =
        static_cast<std::uint64_t>(expert) *
            output_width +
        out;

    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
        sum +=
            load_activation(
                &input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                row,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] +=
                scratch[
                    threadIdx.x +
                    stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out] =
            scratch[0];
    }
}

__global__ void routed_activation_kernel(
    float* workspace,
    std::uint32_t intermediate_size, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto index =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index >= intermediate_size) {
        return;
    }
    const float gate =
        workspace[index];
    const float up =
        workspace[
            intermediate_size +
            index];
    workspace[index] =
        (gate /
         (1.0F + expf(-gate))) *
        up;
}

#if defined(STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME) || defined(STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC)
// Each y block owns one pre-existing [gate,up] slot. Arithmetic and x indexing
// remain the original routed_activation_kernel's; status is immutable here.
__global__ void batched_routed_activation_kernel(
    float* slots,
    std::uint32_t intermediate_size, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto index =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index >= intermediate_size) {
        return;
    }
    auto* workspace = slots + 2ULL * blockIdx.y * intermediate_size;
    const float gate =
        workspace[index];
    const float up =
        workspace[
            intermediate_size +
            index];
    workspace[index] =
        (gate /
         (1.0F + expf(-gate))) *
        up;
}
#endif

__global__ void validate_routes_kernel(
    const std::uint32_t* routes, const float* weights,
    std::uint32_t count, std::uint32_t experts,
    std::uint32_t* status, std::uint32_t* graph_status) {
    for (std::uint32_t slot = threadIdx.x; slot < count; slot += blockDim.x) {
        if (routes[slot] >= experts || !isfinite(weights[slot])) {
            atomicOr(status, 1U);
            if (graph_status) atomicOr(graph_status, 16U);
        }
    }
}

template <typename Storage>
__global__ void commit_experts_kernel(
    const float* accumulation, Storage* output,
    std::uint32_t count, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) store_activation(&output[index], accumulation[index]);
}

template <
    typename Output,
    WeightKind Kind>
__global__ void selected_down_accumulate_kernel(
    const float* intermediate,
    const std::uint32_t* route_indices,
    const float* route_weights,
    std::uint32_t slot,
    const void* weights,
    std::uint32_t intermediate_size,
    std::uint32_t hidden_size,
    std::uint32_t expert_count,
    Output* output, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= hidden_size) {
        return;
    }
    const auto expert =
        route_indices[slot];
    if (expert >= expert_count) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    const auto row =
        static_cast<std::uint64_t>(expert) *
            hidden_size +
        out;

    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < intermediate_size;
         feature += kThreads) {
        sum +=
            intermediate[feature] *
            load_weight<Kind>(
                weights,
                intermediate_size,
                row,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] +=
                scratch[
                    threadIdx.x +
                    stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float current =
            load_activation(
                &output[out]);
        store_activation(
            &output[out],
            current +
                scratch[0] *
                    route_weights[slot]);
    }
}

bool contains(
    std::string_view text,
    std::string_view needle) {
    return text.find(needle) !=
        std::string_view::npos;
}

TensorPair tensor_pair(
    const ExecutionDispatchContext& context,
    std::uint32_t index) {
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP routed experts require model manifest");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "routed-expert tensor index exceeds manifest");
    }
    return TensorPair{
        &context.resources->
            model_manifest->
            tensors[index],
        &resolve_dispatch_tensor(
            context,
            index),
    };
}

RoutedTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const ExecutionTensorAccess& access,
    const HipMoeExpertsLauncherConfig& config) {
    RoutedTensors result;
    std::uint32_t fused = 0;
    std::uint32_t gate = 0;
    std::uint32_t up = 0;
    std::uint32_t down = 0;

    for (const auto index :
         access.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::moe_expert ||
            tensor.dimensions.size() < 3 ||
            tensor.dimensions[2] !=
                config.expert_count) {
            continue;
        }

        if (tensor.dimensions[0] ==
                config.hidden_size &&
            tensor.dimensions[1] ==
                2ULL *
                    config.intermediate_size &&
            contains(
                tensor.name,
                "gate_up")) {
            result.gate_up = pair;
            ++fused;
            continue;
        }

        if (tensor.dimensions[0] ==
                config.hidden_size &&
            tensor.dimensions[1] ==
                config.intermediate_size) {
            if (contains(
                    tensor.name,
                    "gate") &&
                !contains(
                    tensor.name,
                    "gate_up")) {
                result.gate = pair;
                ++gate;
                continue;
            }
            if (contains(
                    tensor.name,
                    "up")) {
                result.up = pair;
                ++up;
                continue;
            }
        }

        if (tensor.dimensions[0] ==
                config.intermediate_size &&
            tensor.dimensions[1] ==
                config.hidden_size &&
            contains(
                tensor.name,
                "down")) {
            result.down = pair;
            ++down;
        }
    }

    if (down != 1) {
        throw std::invalid_argument(
            "routed expert down tensor is missing or ambiguous");
    }
    if (fused == 1 &&
        gate == 0 &&
        up == 0) {
        result.fused_gate_up = true;
        return result;
    }
    if (fused == 0 &&
        gate == 1 &&
        up == 1) {
        result.fused_gate_up = false;
        return result;
    }

    throw std::invalid_argument(
        "routed expert gate/up tensor family is missing or ambiguous");
}

void validate_tensor(
    const TensorPair& pair,
    std::uint32_t input,
    std::uint64_t output,
    std::uint32_t experts,
    const char* name) {
    if (pair.manifest == nullptr ||
        pair.runtime == nullptr) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor is missing");
    }
    if (!runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor is not GPU-visible");
    }

    const auto required =
        required_weight_bytes(
            weight_kind(
                pair.manifest->ggml_type),
            input,
            output,
            experts);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor binding is undersized");
    }
}

template <typename Input>
void launch_projection_by_kind(
    WeightKind kind,
    const Input* input,
    const std::uint32_t* routes,
    std::uint32_t slot,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t experts,
    float* output, const std::uint32_t* status) {
#define STRIX_PROJECT_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             selected_projection_kernel<                 Input, WeightKind::kind_name>),         dim3(output_width), dim3(kThreads), 0, hip_execution_stream(),         input, routes, slot,         reinterpret_cast<const void*>(weights.address),         input_width, output_width, experts, output, status)
    switch (kind) {
    case WeightKind::f32: STRIX_PROJECT_CASE(f32); break;
    case WeightKind::f16: STRIX_PROJECT_CASE(f16); break;
    case WeightKind::bf16: STRIX_PROJECT_CASE(bf16); break;
    case WeightKind::q8_0: STRIX_PROJECT_CASE(q8_0); break;
    case WeightKind::iq4_nl: STRIX_PROJECT_CASE(iq4_nl); break;
    case WeightKind::iq4_xs: STRIX_PROJECT_CASE(iq4_xs); break;
    case WeightKind::q5_1: STRIX_PROJECT_CASE(q5_1); break;
    case WeightKind::q4_k: STRIX_PROJECT_CASE(q4_k); break;
    case WeightKind::q5_k: STRIX_PROJECT_CASE(q5_k); break;
    }
#undef STRIX_PROJECT_CASE
    check_hip(
        hipGetLastError(),
        "HIP routed-expert projection launch");
}

template <typename Output>
void launch_down_by_kind(
    WeightKind kind,
    const float* intermediate,
    const std::uint32_t* routes,
    const float* route_weights,
    std::uint32_t slot,
    const RuntimeTensorBinding& weights,
    std::uint32_t intermediate_size,
    std::uint32_t hidden_size,
    std::uint32_t experts,
    Output* output, const std::uint32_t* status) {
#define STRIX_DOWN_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             selected_down_accumulate_kernel<                 Output, WeightKind::kind_name>),         dim3(hidden_size), dim3(kThreads), 0, hip_execution_stream(),         intermediate, routes, route_weights, slot,         reinterpret_cast<const void*>(weights.address),         intermediate_size, hidden_size, experts, output, status)
    switch (kind) {
    case WeightKind::f32: STRIX_DOWN_CASE(f32); break;
    case WeightKind::f16: STRIX_DOWN_CASE(f16); break;
    case WeightKind::bf16: STRIX_DOWN_CASE(bf16); break;
    case WeightKind::q8_0: STRIX_DOWN_CASE(q8_0); break;
    case WeightKind::iq4_nl: STRIX_DOWN_CASE(iq4_nl); break;
    case WeightKind::iq4_xs: STRIX_DOWN_CASE(iq4_xs); break;
    case WeightKind::q5_1: STRIX_DOWN_CASE(q5_1); break;
    case WeightKind::q4_k: STRIX_DOWN_CASE(q4_k); break;
    case WeightKind::q5_k: STRIX_DOWN_CASE(q5_k); break;
    }
#undef STRIX_DOWN_CASE
    check_hip(
        hipGetLastError(),
        "HIP routed-expert down launch");
}

__global__ void publish_deferred_moe_first_error(const std::uint32_t* local,
    std::uint32_t* record, std::uint32_t domain) {
    // One thread, one stream: completion of this node precedes the next layer's
    // workspace reset. Existing arithmetic kernels and status words are intact.
    merge_deferred_moe_first_error(*reinterpret_cast<DeferredMoeFirstError*>(record),
        *local, static_cast<DeferredMoeDomain>(domain));
}
void enqueue_deferred_moe_status(const std::uint32_t* local, DeferredMoeDomain domain) {
    hipLaunchKernelGGL(publish_deferred_moe_first_error,dim3(1),dim3(1),0,hip_execution_stream(),
        local,hip_deferred_moe_status(),static_cast<std::uint32_t>(domain));
    check_hip(hipGetLastError(),"publish deferred MoE first-error record");
}

template <typename Storage>
void launch_typed(
    const HipMoeExpertsLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& route_indices,
    const ExecutionBufferAddress& route_weights,
    const ExecutionBufferAddress& output,
    const ExecutionBufferAddress& temporary,
    const RoutedTensors& tensors) {
    const auto* input =
        reinterpret_cast<const Storage*>(
            hidden.address);
    const auto* routes =
        reinterpret_cast<const std::uint32_t*>(
            route_indices.address);
    const auto* weights =
        reinterpret_cast<const float*>(
            route_weights.address);
    auto* workspace =
        reinterpret_cast<float*>(
            temporary.address);
    auto* accumulation = workspace + 2ULL * config.intermediate_size;
    auto* status = reinterpret_cast<std::uint32_t*>(accumulation + config.hidden_size);

    check_hip(
        hipMemsetAsync(
            accumulation,
            0,
            static_cast<std::size_t>(config.hidden_size) * sizeof(float) + sizeof(std::uint32_t),
            hip_execution_stream()),
        "clear routed-expert FP32 accumulation and status");
    hipLaunchKernelGGL(validate_routes_kernel, dim3(1), dim3(kThreads), 0,
        hip_execution_stream(), routes, weights, config.top_k,
        config.expert_count, status, hip_graph_status());
    check_hip(hipGetLastError(), "validate routed-expert routes");

    for (std::uint32_t slot = 0;
         slot < config.top_k;
         ++slot) {
        if (tensors.fused_gate_up) {
            launch_projection_by_kind(
                weight_kind(
                    tensors.gate_up.
                        manifest->ggml_type),
                input,
                routes,
                slot,
                *tensors.gate_up.runtime,
                config.hidden_size,
                2U *
                    config.intermediate_size,
                config.expert_count,
                workspace, status);
        } else {
            launch_projection_by_kind(
                weight_kind(
                    tensors.gate.
                        manifest->ggml_type),
                input,
                routes,
                slot,
                *tensors.gate.runtime,
                config.hidden_size,
                config.intermediate_size,
                config.expert_count,
                workspace, status);
            launch_projection_by_kind(
                weight_kind(
                    tensors.up.
                        manifest->ggml_type),
                input,
                routes,
                slot,
                *tensors.up.runtime,
                config.hidden_size,
                config.intermediate_size,
                config.expert_count,
                workspace +
                    config.intermediate_size, status);
        }

        const auto blocks =
            (config.intermediate_size +
             kThreads - 1U) /
            kThreads;
        hipLaunchKernelGGL(
            routed_activation_kernel,
            dim3(blocks),
            dim3(kThreads),
            0, hip_execution_stream(),
            workspace,
            config.intermediate_size, status);
        check_hip(
            hipGetLastError(),
            "HIP routed-expert activation launch");

        launch_down_by_kind(
            weight_kind(
                tensors.down.manifest->
                    ggml_type),
            workspace,
            routes,
            weights,
            slot,
            *tensors.down.runtime,
            config.intermediate_size,
            config.hidden_size,
            config.expert_count,
            accumulation, status);
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(commit_experts_kernel<Storage>),
        dim3((config.hidden_size + kThreads - 1U) / kThreads), dim3(kThreads),
        0, hip_execution_stream(), accumulation,
        reinterpret_cast<Storage*>(output.address), config.hidden_size, status);
    check_hip(hipGetLastError(), "commit routed-expert output");
    // Graph replay consumes the existing aggregate status at its completion
    // boundary. Eager C1 retains scratch until this stream has drained.
    if (config.deferred_moe_status) {
        enqueue_deferred_moe_status(status,DeferredMoeDomain::legacy);
    } else if (hip_graph_status() == nullptr) {
        std::uint32_t host_status = 0;
        check_hip(hipMemcpyAsync(&host_status, status, sizeof(host_status),
            hipMemcpyDeviceToHost, hip_execution_stream()), "read routed-expert status");
        check_hip(hipStreamSynchronize(hip_execution_stream()), "drain routed-expert validation");
        if (host_status != 0U) throw std::invalid_argument("HIP routed experts rejected invalid routes or nonfinite weights");
    }
}


#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
// Integer classification avoids finite-math assumptions and never mutates the
// graph aggregate except by OR. Each CTA snapshots the sticky status atomically
// before its barrier; bad producers cannot expose indeterminate scratch values.
__global__ void q4k_a8_validate_finite_kernel(const float* values, std::uint32_t count,
    std::uint32_t* status, std::uint32_t* aggregate, bool bf16_commit) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ std::uint32_t initial_status;
    if (threadIdx.x == 0U) initial_status = atomicOr(status,0U);
    __syncthreads();
    if (initial_status != 0U) {
        if (index == 0U && aggregate != nullptr) atomicOr(aggregate, 16U);
        return;
    }
    if (index < count && (__float_as_uint(values[index]) & 0x7fffffffU) >=
        (bf16_commit ? 0x7f7f8000U : 0x7f800000U)) {
        atomicOr(status, 8U);
        if (aggregate != nullptr) atomicOr(aggregate, 16U);
    }
}

// One complete wave owns an output. Eight register leaves represent the
// ORIGINAL256 threads, not a32-leaf reduction. All shuffle participants stay
// active; only complete output waves can return at the output-tail guard.
__global__ void q51_grouped_virtual256_down_probe_kernel(const float* intermediates,
    const std::uint32_t* routes,const float* route_weights,const void* weights,
    std::uint32_t intermediate_size,std::uint32_t hidden_size,std::uint32_t experts,
    std::uint32_t top_k,float* accumulation,const std::uint32_t* status) {
    if(*status!=0U) return; // No writer changes local status during this kernel.
    const auto lane=static_cast<std::uint32_t>(threadIdx.x)%32U;
    const auto wave=static_cast<std::uint32_t>(threadIdx.x)/32U;
    const auto out=static_cast<std::uint32_t>(blockIdx.x)*8U+wave;
    if(out>=hidden_size) return;
    float current=0.0F;
    for(std::uint32_t slot=0;slot<top_k;++slot) {
        const auto expert=routes[slot];
        if(expert>=experts) return; // Same-stream validator already sets bit1.
        const auto row=static_cast<std::uint64_t>(expert)*hidden_size+out;
        const auto input_base=static_cast<std::uint64_t>(slot)*2U*intermediate_size;
        float partial[8]={};
        for(std::uint32_t chunk=0;chunk<intermediate_size;chunk+=256U) {
#pragma unroll
            for(unsigned j=0;j<8;++j) {
                const auto feature=chunk+8U*lane+j;
                if(feature<intermediate_size)
                    partial[j]+=intermediates[input_base+feature]*
                        load_weight<WeightKind::q5_1>(weights,intermediate_size,row,feature);
            }
        }
#pragma unroll
        for(unsigned distance=16;distance>0;distance>>=1U) {
#pragma unroll
            for(unsigned j=0;j<8;++j) {
                const auto neighbor=__shfl_down(partial[j],distance,32);
                if(lane<distance) partial[j]+=neighbor;
            }
        }
        if(lane==0U) {
#pragma unroll
            for(unsigned j=0;j<4;++j) partial[j]+=partial[j+4];
#pragma unroll
            for(unsigned j=0;j<2;++j) partial[j]+=partial[j+2];
            partial[0]+=partial[1];
            current=current+partial[0]*route_weights[slot];
        }
    }
    if(lane==0U) accumulation[out]=current;
}

// Each complete wave owns exactly the incumbent's eight virtual leaves for
// one (slot, output). Store the unweighted dot without arithmetic conversion.
__global__ void q51_slot_parallel_dot_kernel(const float* intermediates,
    const std::uint32_t* routes,const void* weights,
    std::uint32_t intermediate_size,std::uint32_t hidden_size,std::uint32_t experts,
    float* rawdots,const std::uint32_t* status) {
    if(*status!=0U) return;
    const auto lane=static_cast<std::uint32_t>(threadIdx.x)%32U;
    const auto wave=static_cast<std::uint32_t>(threadIdx.x)/32U;
    const auto out=static_cast<std::uint32_t>(blockIdx.x)*8U+wave;
    if(out>=hidden_size) return;
    const auto slot=static_cast<std::uint32_t>(blockIdx.y);
    const auto expert=routes[slot];
    if(expert>=experts) return;
    const auto row=static_cast<std::uint64_t>(expert)*hidden_size+out;
    const auto input_base=static_cast<std::uint64_t>(slot)*2U*intermediate_size;
    float partial[8]={};
    for(std::uint32_t chunk=0;chunk<intermediate_size;chunk+=256U) {
#pragma unroll
        for(unsigned j=0;j<8;++j) {
            const auto feature=chunk+8U*lane+j;
            if(feature<intermediate_size)
                partial[j]+=intermediates[input_base+feature]*
                    load_weight<WeightKind::q5_1>(weights,intermediate_size,row,feature);
        }
    }
#pragma unroll
    for(unsigned distance=16;distance>0;distance>>=1U) {
#pragma unroll
        for(unsigned j=0;j<8;++j) {
            const auto neighbor=__shfl_down(partial[j],distance,32);
            if(lane<distance) partial[j]+=neighbor;
        }
    }
    if(lane==0U) {
#pragma unroll
        for(unsigned j=0;j<4;++j) partial[j]+=partial[j+4];
#pragma unroll
        for(unsigned j=0;j<2;++j) partial[j]+=partial[j+2];
        partial[0]+=partial[1];
        rawdots[static_cast<std::uint64_t>(slot)*hidden_size+out]=partial[0];
    }
}

__global__ void q51_slot_parallel_merge_kernel(const float* rawdots,
    const std::uint32_t* routes,const float* route_weights,
    std::uint32_t hidden_size,std::uint32_t experts,std::uint32_t top_k,
    float* accumulation,const std::uint32_t* status) {
    if(*status!=0U) return;
    const auto out=static_cast<std::uint32_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(out>=hidden_size) return;
    float current=0.0F;
    for(std::uint32_t slot=0;slot<top_k;++slot) {
        if(routes[slot]>=experts) return;
        const float weight=route_weights[slot];
        const float dot=rawdots[static_cast<std::uint64_t>(slot)*hidden_size+out];
        // Actual incumbent merge roles: weight src0, dot src1, accumulator src2.
        // The read/write operand also preserves the positive-zero initial seed.
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dot));
    }
    accumulation[out]=current;
}

void require_q51_slot_eager_stream(hipStream_t stream) {
    if (hip_graph_status()!=nullptr)
        throw std::invalid_argument("slot-parallel Q5_1 down admits eager execution only");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(stream,&capture),"query slot-parallel Q5_1 capture status");
    if (capture!=hipStreamCaptureStatusNone)
        throw std::invalid_argument("slot-parallel Q5_1 down admits eager execution only");
}

struct Q4kA8Range { std::uintptr_t address; std::uint64_t bytes; };
void validate_q4k_a8_ranges(Q4kA8Range temporary, Q4kA8Range output,
    const std::array<Q4kA8Range, 6>& readonly) {
    const auto end = [](Q4kA8Range r) {
        if (r.address == 0 || r.bytes == 0 || r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address)
            throw std::invalid_argument("Q4K A8 invalid runtime address range");
        return r.address + static_cast<std::uintptr_t>(r.bytes);
    };
    const auto disjoint = [&](Q4kA8Range a, Q4kA8Range c) {
        if (a.address < end(c) && c.address < end(a))
            throw std::invalid_argument("Q4K A8 runtime writable range aliases another owner");
    };
    (void)end(temporary); (void)end(output);
    if (temporary.address % 256 != 0)
        throw std::invalid_argument("Q4K A8 runtime temporary requires absolute 256-byte alignment");
    disjoint(temporary,output);
    for (const auto r : readonly) { disjoint(temporary,r); disjoint(output,r); }
}

template <typename Storage>
void launch_q4k_a8_typed(const HipMoeExpertsLauncherConfig& config,
    const ExecutionBufferAddress& hidden, const ExecutionBufferAddress& route_indices,
    const ExecutionBufferAddress& route_weights, const ExecutionBufferAddress& output,
    const ExecutionBufferAddress& temporary, const RoutedTensors& tensors) {
    if (!config.q4k_a8_quiescence)
        throw std::invalid_argument("Q4K A8 runtime requires setup completion guard");
    const auto layout = moe_q4k_a8::build_runtime_workspace(config.hidden_size,
        config.intermediate_size, config.expert_count, config.top_k, config.q51_slot_parallel_down);
    if (temporary.bytes < layout.total_bytes)
        throw std::invalid_argument("Q4K A8 runtime temporary is undersized");
    const auto gate_bytes = required_weight_bytes(WeightKind::q4_k, config.hidden_size,
        config.intermediate_size, config.expert_count);
    // Validate entire runtime owners, not just the exact probe slices. Read-only
    // owners can alias each other; scratch and semantic output cannot alias any.
    validate_q4k_a8_ranges({temporary.address, temporary.bytes},
        {output.address, output.bytes}, {{{hidden.address,hidden.bytes},
        {route_indices.address,route_indices.bytes}, {route_weights.address,route_weights.bytes},
        {tensors.gate.runtime->address,tensors.gate.runtime->bytes}, {tensors.up.runtime->address,tensors.up.runtime->bytes},
        {tensors.down.runtime->address,tensors.down.runtime->bytes}}});
    moe_q4k_a8::Request request;
    request.enabled = true;
    request.approved_numerical_identity = moe_q4k_a8::numerical_identity;
    request.input = config.activation == HipActivationStorage::bf16 ?
        moe_q4k_a8::InputKind::bf16 : moe_q4k_a8::InputKind::fp32;
    request.shape = {config.hidden_size, config.intermediate_size, config.expert_count, config.top_k, 1};
    request.gate_type = tensors.gate.manifest->ggml_type;
    request.up_type = tensors.up.manifest->ggml_type;
    request.fused_gate_up = tensors.fused_gate_up;
    const auto slice = [&](std::uint64_t offset, std::uint64_t bytes) {
        return moe_q4k_a8::DeviceBytes{temporary.address + static_cast<std::uintptr_t>(offset), bytes};
    };
    const moe_q4k_a8::Bindings bindings{
        {hidden.address, static_cast<std::uint64_t>(config.hidden_size) * sizeof(Storage)},
        {tensors.gate.runtime->address,gate_bytes}, {tensors.up.runtime->address,gate_bytes},
        {route_indices.address, static_cast<std::uint64_t>(config.top_k) * sizeof(std::uint32_t)},
        slice(layout.codes_offset,layout.codes_bytes), slice(layout.deltas_offset,layout.deltas_bytes),
        slice(layout.sums_offset,layout.sums_bytes), slice(layout.status_offset,4),
        slice(layout.gate_up_offset,layout.gate_up_bytes)};
    // All admission is complete before even the initial clear is submitted.
    moe_q4k_a8::validate(request,bindings);
    const bool grouped_q51=config.grouped_exact_q51_down && tensors.down.manifest->ggml_type==7U;
    const bool slot_parallel_q51=grouped_q51 && config.q51_slot_parallel_down;
    if(grouped_q51) {
        const q51_grouped_down::Request q51_request{true,q51_grouped_down::numerical_identity,
            {config.hidden_size,config.intermediate_size,config.expert_count,config.top_k}};
        const auto n=q51_grouped_down::required_bytes(q51_request.shape);
        q51_grouped_down::validate(q51_request,{
            {bindings.output.address,n.intermediates},{tensors.down.runtime->address,n.weights},
            {route_indices.address,n.routes},{route_weights.address,n.route_weights},
            {bindings.status.address,n.status},{temporary.address+layout.accumulation_offset,n.accumulation}});
        if (slot_parallel_q51) {
            const q51_slot_parallel::Request parallel_request{true,q51_slot_parallel::numerical_identity,q51_request.shape};
            q51_slot_parallel::validate(parallel_request,{
                {bindings.output.address,n.intermediates},{tensors.down.runtime->address,n.weights},
                {route_indices.address,n.routes},{route_weights.address,n.route_weights},
                {bindings.status.address,n.status},{temporary.address+layout.accumulation_offset,n.accumulation},
                {temporary.address+layout.rawdots_offset,layout.rawdots_bytes}});
        }
    }
    auto* workspace = reinterpret_cast<float*>(bindings.output.address);
    auto* accumulation = reinterpret_cast<float*>(temporary.address + layout.accumulation_offset);
    auto* status = reinterpret_cast<std::uint32_t*>(bindings.status.address);
    const auto* routes = reinterpret_cast<const std::uint32_t*>(route_indices.address);
    const auto* weights = reinterpret_cast<const float*>(route_weights.address);
    auto* aggregate = hip_graph_status();
    check_hip(hipPeekAtLastError(), "Q4K A8 runtime pre-enqueue stale error");
    check_hip(hipMemsetAsync(accumulation,0,
        static_cast<std::size_t>(layout.accumulation_bytes + 4),hip_execution_stream()),
        "clear Q4K A8 runtime accumulation/status");
    hipLaunchKernelGGL(validate_routes_kernel,dim3(1),dim3(kThreads),0,hip_execution_stream(),
        routes,weights,config.top_k,config.expert_count,status,aggregate);
    check_hip(hipGetLastError(),"Q4K A8 route/weight validation");
    moe_q4k_a8::enqueue_probe(request,bindings,hip_execution_stream());
    const auto scan = [&](const float* values,std::uint32_t count,bool bf16_commit=false) {
        hipLaunchKernelGGL(q4k_a8_validate_finite_kernel,
            dim3((count+kThreads-1U)/kThreads),dim3(kThreads),0,hip_execution_stream(),
            values,count,status,aggregate,bf16_commit);
        check_hip(hipGetLastError(),"Q4K A8 finite pipeline validation");
    };
    scan(workspace,2U*config.intermediate_size*config.top_k);
    // Slot buffers are independent. Finish all original SiLU kernels before
    // checking their FP32 values, then preserve ascending down accumulation.
    if (config.batched_slot_activation) {
        hipLaunchKernelGGL(batched_routed_activation_kernel,
            dim3((config.intermediate_size+kThreads-1U)/kThreads,config.top_k),dim3(kThreads),0,
            hip_execution_stream(),workspace,config.intermediate_size,status);
        check_hip(hipGetLastError(),"Q4K A8 batched slot SiLU enqueue");
        if (config.q4k_a8_stats) {
            config.q4k_a8_stats->batched_activation_launches.fetch_add(1,std::memory_order_relaxed);
            config.q4k_a8_stats->batched_activation_slots.fetch_add(config.top_k,std::memory_order_relaxed);
        }
    } else {
        for (std::uint32_t slot=0;slot<config.top_k;++slot) {
            auto* slot_workspace=workspace+2ULL*slot*config.intermediate_size;
            hipLaunchKernelGGL(routed_activation_kernel,
                dim3((config.intermediate_size+kThreads-1U)/kThreads),dim3(kThreads),0,
                hip_execution_stream(),slot_workspace,config.intermediate_size,status);
            check_hip(hipGetLastError(),"Q4K A8 original SiLU enqueue");
        }
    }
    scan(workspace,2U*config.intermediate_size*config.top_k);
    if(slot_parallel_q51) {
        auto* rawdots=reinterpret_cast<float*>(temporary.address+layout.rawdots_offset);
        hipLaunchKernelGGL(q51_slot_parallel_dot_kernel,
            dim3((config.hidden_size+7U)/8U,config.top_k),dim3(kThreads),0,hip_execution_stream(),
            workspace,routes,reinterpret_cast<const void*>(tensors.down.runtime->address),
            config.intermediate_size,config.hidden_size,config.expert_count,rawdots,status);
        check_hip(hipGetLastError(),"native slot-parallel Q5_1 raw dots");
        hipLaunchKernelGGL(q51_slot_parallel_merge_kernel,
            dim3((config.hidden_size+kThreads-1U)/kThreads),dim3(kThreads),0,hip_execution_stream(),
            rawdots,routes,weights,config.hidden_size,config.expert_count,config.top_k,accumulation,status);
        check_hip(hipGetLastError(),"native slot-parallel Q5_1 ordered merge");
        if (config.q4k_a8_stats)
            config.q4k_a8_stats->slot_parallel_q51_submissions.fetch_add(1,std::memory_order_relaxed);
    } else if(grouped_q51) {
        hipLaunchKernelGGL(q51_grouped_virtual256_down_probe_kernel,
            dim3((config.hidden_size+7U)/8U),dim3(kThreads),0,hip_execution_stream(),
            workspace,routes,weights,reinterpret_cast<const void*>(tensors.down.runtime->address),
            config.intermediate_size,config.hidden_size,config.expert_count,config.top_k,accumulation,status);
        check_hip(hipGetLastError(),"native grouped exact Q5_1 down");
    } else {
        for (std::uint32_t slot=0;slot<config.top_k;++slot)
            launch_down_by_kind(weight_kind(tensors.down.manifest->ggml_type),
                workspace+2ULL*slot*config.intermediate_size,routes,weights,slot,*tensors.down.runtime,
                config.intermediate_size,config.hidden_size,config.expert_count,accumulation,status);
    }
    scan(accumulation,config.hidden_size,config.activation == HipActivationStorage::bf16);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(commit_experts_kernel<Storage>),
        dim3((config.hidden_size+kThreads-1U)/kThreads),dim3(kThreads),0,hip_execution_stream(),
        accumulation,reinterpret_cast<Storage*>(output.address),config.hidden_size,status);
    check_hip(hipGetLastError(),"Q4K A8 semantic output commit");
    if (config.deferred_moe_status) {
        enqueue_deferred_moe_status(status,DeferredMoeDomain::q4k_a8);
    } else if (aggregate == nullptr) {
        std::uint32_t host_status = 0;
        try {
            check_hip(hipMemcpyAsync(&host_status,status,sizeof(host_status),hipMemcpyDeviceToHost,
                hip_execution_stream()),"Q4K A8 read status");
            check_hip(hipStreamSynchronize(hip_execution_stream()),"Q4K A8 drain status");
        } catch (...) {
            // host_status is still alive until the original device is drained.
            config.q4k_a8_quiescence->recover_or_fail_stop();
            throw;
        }
        if (host_status != 0)
            throw std::invalid_argument("Q4K A8 rejected status bits (route/weight=1,input=2,delta=4,pipeline=8): " +
                std::to_string(host_status));
    }
    if (config.q4k_a8_stats) config.q4k_a8_stats->quantized_submissions.fetch_add(1,std::memory_order_relaxed);
    if(grouped_q51 && config.q4k_a8_stats)
        config.q4k_a8_stats->grouped_q51_submissions.fetch_add(1,std::memory_order_relaxed);
}
#endif

void launch_moe_experts(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::moe_experts) {
        throw std::invalid_argument(
            "HIP routed-expert launcher attached to wrong operator");
    }
    if (context.reads.size() != 3 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP routed experts expect hidden/routes/weights, one output and one temporary");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP routed-expert v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP routed experts require configuration");
    }

    const auto& config =
        *static_cast<
            HipMoeExpertsLauncherConfig*>(
                user_data);
    validate_config(config);
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    if (config.q51_slot_parallel_down) require_q51_slot_eager_stream(hip_execution_stream());
#endif
    if (config.deferred_moe_status != (hip_deferred_moe_status()!=nullptr))
        throw std::invalid_argument("deferred MoE launcher requires matching owned eager completion scope");

    const auto hidden =
        resolve_dispatch_read(
            context, 0);
    const auto route_indices =
        resolve_dispatch_read(
            context, 1);
    const auto route_weights =
        resolve_dispatch_read(
            context, 2);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);

    const auto activation_bytes =
        static_cast<std::uint32_t>(
            config.activation);
    const auto hidden_required =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        activation_bytes;
    const auto route_index_required =
        static_cast<std::uint64_t>(
            config.top_k) *
        sizeof(std::uint32_t);
    const auto route_weight_required =
        static_cast<std::uint64_t>(
            config.top_k) *
        sizeof(float);
    const auto temporary_required = config.q4k_a8g32_gate_up ?
        moe_q4k_a8::build_runtime_workspace(config.hidden_size, config.intermediate_size,
            config.expert_count, config.top_k, config.q51_slot_parallel_down).total_bytes :
        (2ULL * config.intermediate_size + config.hidden_size) * sizeof(float) + sizeof(std::uint32_t);

    if (hidden.bytes <
            hidden_required ||
        output.bytes <
            hidden_required ||
        route_indices.bytes <
            route_index_required ||
        route_weights.bytes <
            route_weight_required ||
        temporary.bytes <
            temporary_required) {
        throw std::invalid_argument(
            "HIP routed-expert buffers are undersized");
    }

    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto tensors =
        identify_tensors(
            context,
            access,
            config);

    if (tensors.fused_gate_up) {
        validate_tensor(
            tensors.gate_up,
            config.hidden_size,
            2ULL *
                config.intermediate_size,
            config.expert_count,
            "routed gate-up");
    } else {
        validate_tensor(
            tensors.gate,
            config.hidden_size,
            config.intermediate_size,
            config.expert_count,
            "routed gate");
        validate_tensor(
            tensors.up,
            config.hidden_size,
            config.intermediate_size,
            config.expert_count,
            "routed up");
    }
    validate_tensor(
        tensors.down,
        config.intermediate_size,
        config.hidden_size,
        config.expert_count,
        "routed down");

    if (config.deferred_moe_status) {
        const auto& record=current_hip_launch_context()->deferred_moe_status;
        if (!context.memory || record.execution_identity!=context.memory->execution_identity ||
            (context.execution_identity && record.execution_identity!=context.execution_identity))
            throw std::invalid_argument("deferred MoE record belongs to a different execution");
        const auto disjoint=[&](std::uintptr_t address,std::uint64_t bytes) {
            if (bytes>std::numeric_limits<std::uintptr_t>::max()-address ||
                (record.arena_base<address+bytes && address<record.arena_base+record.arena_bytes))
                throw std::invalid_argument("deferred MoE record aliases execution storage");
        };
        disjoint(context.memory->arena_base,context.memory->arena_bytes);
        for (const auto* pair : {&tensors.gate,&tensors.up,&tensors.gate_up,&tensors.down})
            if (pair->runtime) disjoint(pair->runtime->address,pair->runtime->bytes);
        if (context.session_state_memory) {
            disjoint(context.session_state_memory->device_base,context.session_state_memory->device_bytes);
            disjoint(context.session_state_memory->host_base,context.session_state_memory->host_bytes);
        }
    }

    if (config.q4k_a8g32_gate_up) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
        if (tensors.fused_gate_up || tensors.gate.manifest->ggml_type != tensors.up.manifest->ggml_type ||
            (tensors.gate.manifest->ggml_type != 12 && tensors.gate.manifest->ggml_type != 13))
            throw std::invalid_argument("Q4K A8 runtime requires separate paired Q4K or Q5K gate/up");
        if (tensors.gate.manifest->ggml_type == 12) {
            if (config.activation == HipActivationStorage::bf16)
                launch_q4k_a8_typed<std::uint16_t>(config,hidden,route_indices,route_weights,output,temporary,tensors);
            else launch_q4k_a8_typed<float>(config,hidden,route_indices,route_weights,output,temporary,tensors);
            return;
        }
        if (config.q4k_a8_stats) config.q4k_a8_stats->legacy_q5k_submissions.fetch_add(1,std::memory_order_relaxed);
#endif
    }
    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_typed<std::uint16_t>(
            config,
            hidden,
            route_indices,
            route_weights,
            output,
            temporary,
            tensors);
        return;
    case HipActivationStorage::fp32:
        launch_typed<float>(
            config,
            hidden,
            route_indices,
            route_weights,
            output,
            temporary,
            tensors);
        return;
    }
    throw std::logic_error(
        "unreachable routed-expert activation storage");
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipMoeExpertsLauncherConfig& config) {
    return
        "hip.moe_experts." +
        std::string(
            storage_name(
                config.activation)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".m" +
        std::to_string(
            config.intermediate_size) +
        ".e" +
        std::to_string(
            config.expert_count) +
        ".k" +
        std::to_string(
            config.top_k) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) +
        (config.batched_slot_activation ? ".batched-slot-activation-v1" : "") +
        (config.deferred_moe_status ? ".deferred-status-v1" : "") +
        (config.q51_slot_parallel_down ? ".q51-slot-parallel-v1" : "") +
        (config.grouped_exact_q51_down ? ".grouped-exact-q51-down-v1" : "") +
        (config.q4k_a8g32_gate_up ? std::string(moe_q4k_a8::runtime_kernel_suffix) : "");
}

std::uint64_t implementation_identity(
    const HipMoeExpertsLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.intermediate_size);
    hash_u64(hash, config.expert_count);
    hash_u64(hash, config.top_k);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    if (config.q4k_a8g32_gate_up) hash_u64(hash,0x51344b4138473332ULL);
    if (config.grouped_exact_q51_down) hash_u64(hash,0x5135314752505631ULL);
    if (config.q51_slot_parallel_down) hash_u64(hash,0x5135315350415231ULL);
    if (config.deferred_moe_status) hash_u64(hash,0x4445464d4f453031ULL);
    if (config.batched_slot_activation) hash_u64(hash,0x42415453494c5531ULL);
    return hash;
}

void validate_q51_slot_kernel_id(std::string_view id, bool enabled) {
    constexpr std::string_view marker=".q51-slot-parallel-v1";
    const auto position=id.find(marker);
    const bool present=position!=std::string_view::npos;
    const auto ending=std::string(marker)+std::string(moe_q4k_a8::grouped_q51_kernel_ending);
    if (present!=enabled || (enabled && (!id.ends_with(ending) ||
        id.find(marker,position+marker.size())!=std::string_view::npos)))
        throw std::invalid_argument("slot-parallel Q5_1 id differs from config or marker placement");
}

}  // namespace

void register_hip_moe_experts_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipMoeExpertsLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (request.batched_moe_activation != config.batched_slot_activation)
        throw std::invalid_argument("batched routed activation config differs from binding request");
    if (request.batched_moe_activation && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("batched routed activation variant admits eager execution only");
    if(request.grouped_exact_q51_down != config.grouped_exact_q51_down)
        throw std::invalid_argument("grouped exact Q5_1 config differs from binding request");
    if (request.q51_slot_parallel_down != config.q51_slot_parallel_down)
        throw std::invalid_argument("slot-parallel Q5_1 config differs from binding request");
    if (request.q51_slot_parallel_down && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("slot-parallel Q5_1 variant admits eager execution only");
    if (request.deferred_moe_status != config.deferred_moe_status ||
        (config.deferred_moe_status && (request.device_graph_inputs || request.require_graph_safe)))
        throw std::invalid_argument("deferred MoE variant requires matching eager request");
    if (request.q4k_a8g32_gate_up != config.q4k_a8g32_gate_up)
        throw std::invalid_argument("Q4K A8 variant config differs from binding request");
    if (config.hidden_size !=
            model.hidden_size ||
        config.intermediate_size !=
            model.moe_intermediate_size ||
        config.expert_count !=
            model.num_experts ||
        config.top_k !=
            model.num_experts_per_token) {
        throw std::invalid_argument(
            "HIP routed-expert config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP routed-expert v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP routed experts may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::moe_expert;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::moe_experts,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        kThreads * sizeof(float);
    variant.resources.scratch_bytes =
        (2ULL * config.intermediate_size + config.hidden_size) * sizeof(float) + sizeof(std::uint32_t);
    variant.numerics.input_precision =
        storage_name(
            config.activation);
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        storage_name(
            config.activation);
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = !config.deferred_moe_status && !config.q51_slot_parallel_down;
    variant.input_layout =
        "row-major+device-topk";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        "hip-gfx1151-moe-experts-v2";
    if (config.q4k_a8g32_gate_up) {
        variant.resources.lds_bytes = 3328;
        variant.resources.scratch_bytes = moe_q4k_a8::build_runtime_workspace(config.hidden_size,
            config.intermediate_size,config.expert_count,config.top_k,config.q51_slot_parallel_down).total_bytes;
        variant.numerics.accumulator_precision = std::string(moe_q4k_a8::runtime_numerical_identity);
        variant.compiler_identity = "hip-gfx1151-moe-q4k-a8g32-native-v2-probe-strict-rndiv-legacy-down";
    }
    if(config.grouped_exact_q51_down) {
        variant.numerics.accumulator_precision += ";" + std::string(moe_q4k_a8::grouped_q51_numerical_identity);
        variant.compiler_identity += ":grouped-exact-q51-down-v1";
    }
    if(config.deferred_moe_status) variant.compiler_identity += ":deferred-first-error-v1";
    if(config.q51_slot_parallel_down) {
        variant.numerics.accumulator_precision += ";" + std::string(q51_slot_parallel::numerical_identity);
        variant.compiler_identity += ":q51-slot-parallel-rawdot-v1";
    }
    if(config.batched_slot_activation) variant.compiler_identity += ":batched-slot-activation-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_moe_experts_launcher(
    std::string kernel_id_value,
    HipMoeExpertsLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP routed-expert launcher requires id and config");
    }
    validate_config(*config);
    validate_q51_slot_kernel_id(kernel_id_value,config->q51_slot_parallel_down);
    if ((kernel_id_value.find(".batched-slot-activation-v1")!=std::string::npos)!=config->batched_slot_activation)
        throw std::invalid_argument("batched routed activation launcher id differs from config");
    if ((kernel_id_value.find(".deferred-status-v1")!=std::string::npos)!=config->deferred_moe_status)
        throw std::invalid_argument("deferred MoE launcher id differs from config");
    if(kernel_id_value.ends_with(moe_q4k_a8::grouped_q51_kernel_ending)!=config->grouped_exact_q51_down)
        throw std::invalid_argument("grouped exact Q5_1 launcher id differs from config");
    if (kernel_id_value.ends_with(moe_q4k_a8::runtime_kernel_suffix) != config->q4k_a8g32_gate_up)
        throw std::invalid_argument("Q4K A8 launcher id differs from config");
    if (config->q4k_a8g32_gate_up && !config->q4k_a8_quiescence)
        config->q4k_a8_quiescence = std::make_shared<HipSetupQuiescenceGuard>();
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_moe_experts,
        config,
    };
}

void HipMoeExpertsLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP routed-expert bundle requires valid compiled execution");
    }

    std::string registered_id;
    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::moe_experts) {
            continue;
        }
        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP routed-expert step has no kernel binding");
        }
        validate_q51_slot_kernel_id(*binding.kernel_id,config_.q51_slot_parallel_down);
        if ((binding.kernel_id->find(".deferred-status-v1")!=std::string::npos)!=config_.deferred_moe_status)
            throw std::invalid_argument("deferred MoE bundle mode differs from compiled binding");
        if ((binding.kernel_id->find(".batched-slot-activation-v1")!=std::string::npos)!=config_.batched_slot_activation)
            throw std::invalid_argument("batched routed activation bundle differs from compiled selection");
        if(binding.kernel_id->ends_with(moe_q4k_a8::grouped_q51_kernel_ending)!=config_.grouped_exact_q51_down)
            throw std::invalid_argument("grouped exact Q5_1 bundle differs from compiled selection");
        if (binding.kernel_id->ends_with(moe_q4k_a8::runtime_kernel_suffix) != config_.q4k_a8g32_gate_up)
            throw std::invalid_argument("Q4K A8 bundle mode differs from compiled selection");
        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_moe_experts_launcher(
                    registered_id,
                    &config_));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP routed-expert bundle requires one selected kernel id");
        }
    }
}


#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
namespace q51_grouped_down {
namespace {
void enqueue_preflight(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    check_hip(hipPeekAtLastError(),"Q5_1 grouped-down pre-enqueue stale error");
    hipLaunchKernelGGL(validate_routes_kernel,dim3(1),dim3(kThreads),0,stream,
        reinterpret_cast<const std::uint32_t*>(b.routes.address),
        reinterpret_cast<const float*>(b.route_weights.address),request.shape.top_k,
        request.shape.experts,reinterpret_cast<std::uint32_t*>(b.status.address),nullptr);
    check_hip(hipGetLastError(),"Q5_1 grouped-down route validation");
}
}
void enqueue_grouped_probe(const Request& request,const Bindings& b,hipStream_t stream) {
    enqueue_preflight(request,b,stream);
    hipLaunchKernelGGL(q51_grouped_virtual256_down_probe_kernel,
        dim3((request.shape.hidden+7U)/8U),dim3(kThreads),0,stream,
        reinterpret_cast<const float*>(b.intermediates.address),
        reinterpret_cast<const std::uint32_t*>(b.routes.address),
        reinterpret_cast<const float*>(b.route_weights.address),reinterpret_cast<const void*>(b.weights.address),
        request.shape.intermediate,request.shape.hidden,request.shape.experts,request.shape.top_k,
        reinterpret_cast<float*>(b.accumulation.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
    check_hip(hipGetLastError(),"Q5_1 grouped virtual256 down probe");
}
void enqueue_original_receipt(const Request& request,const Bindings& b,hipStream_t stream) {
    enqueue_preflight(request,b,stream);
    for(std::uint32_t slot=0;slot<request.shape.top_k;++slot) {
        const auto address=b.intermediates.address+static_cast<std::uintptr_t>(slot)*2U*request.shape.intermediate*sizeof(float);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(selected_down_accumulate_kernel<float,WeightKind::q5_1>),
            dim3(request.shape.hidden),dim3(kThreads),0,stream,reinterpret_cast<const float*>(address),
            reinterpret_cast<const std::uint32_t*>(b.routes.address),
            reinterpret_cast<const float*>(b.route_weights.address),slot,reinterpret_cast<const void*>(b.weights.address),
            request.shape.intermediate,request.shape.hidden,request.shape.experts,
            reinterpret_cast<float*>(b.accumulation.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
        check_hip(hipGetLastError(),"original Q5_1 down receipt");
    }
}
} // namespace q51_grouped_down

namespace q51_slot_parallel {
void enqueue_slot_parallel_probe(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    require_q51_slot_eager_stream(stream);
    const q51_grouped_down::Request original_request{true,q51_grouped_down::numerical_identity,request.shape};
    const q51_grouped_down::Bindings original{b.intermediates,b.weights,b.routes,b.route_weights,b.status,b.accumulation};
    q51_grouped_down::enqueue_preflight(original_request,original,stream);
    hipLaunchKernelGGL(q51_slot_parallel_dot_kernel,
        dim3((request.shape.hidden+7U)/8U,request.shape.top_k),dim3(kThreads),0,stream,
        reinterpret_cast<const float*>(b.intermediates.address),
        reinterpret_cast<const std::uint32_t*>(b.routes.address),reinterpret_cast<const void*>(b.weights.address),
        request.shape.intermediate,request.shape.hidden,request.shape.experts,
        reinterpret_cast<float*>(b.rawdots.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
    check_hip(hipGetLastError(),"Q5_1 slot-parallel raw dots probe");
    hipLaunchKernelGGL(q51_slot_parallel_merge_kernel,
        dim3((request.shape.hidden+kThreads-1U)/kThreads),dim3(kThreads),0,stream,
        reinterpret_cast<const float*>(b.rawdots.address),
        reinterpret_cast<const std::uint32_t*>(b.routes.address),reinterpret_cast<const float*>(b.route_weights.address),
        request.shape.hidden,request.shape.experts,request.shape.top_k,
        reinterpret_cast<float*>(b.accumulation.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
    check_hip(hipGetLastError(),"Q5_1 slot-parallel ordered merge probe");
}
} // namespace q51_slot_parallel
#endif

#ifdef STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC
namespace moe_batched_activation_diagnostic {
const void* kernel_symbol(Mode mode) {
    switch (mode) {
    case Mode::reference: return reinterpret_cast<const void*>(routed_activation_kernel);
    case Mode::batched: return reinterpret_cast<const void*>(batched_routed_activation_kernel);
    }
    throw std::invalid_argument("invalid routed activation diagnostic mode");
}
void enqueue(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream) {
    const auto launch = validate(shape, bindings, mode);
    if (!stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("routed activation diagnostic requires an explicitly owned stream");
    check_hip(hipPeekAtLastError(), "routed activation diagnostic stale error");
    auto* workspace = reinterpret_cast<float*>(bindings.workspace.address);
    const auto* status = reinterpret_cast<const std::uint32_t*>(bindings.status.address);
    if (mode == Mode::batched) {
        hipLaunchKernelGGL(batched_routed_activation_kernel,
            dim3(launch.grid_x,launch.grid_y),dim3(launch.block_x),0,stream,
            workspace,shape.intermediate,status);
        check_hip(hipGetLastError(), "routed activation diagnostic batched enqueue");
    } else {
        for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
            hipLaunchKernelGGL(routed_activation_kernel,
                dim3(launch.grid_x),dim3(launch.block_x),0,stream,
                workspace+2ULL*slot*shape.intermediate,shape.intermediate,status);
            check_hip(hipGetLastError(), "routed activation diagnostic original enqueue");
        }
    }
}
} // namespace moe_batched_activation_diagnostic
#endif

}  // namespace strix
