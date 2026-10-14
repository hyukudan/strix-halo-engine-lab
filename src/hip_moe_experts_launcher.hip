#include "strix/hip_launch_context.hpp"
#include "strix/hip_moe_experts_launcher.hpp"

#include "strix/dispatch_resources.hpp"

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
    0x4d4f454558505254ULL;
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
        scale = (packed[group + 4U] & 15U) | ((packed[group - 4U] >> 6U) << 4U);
        minimum = (packed[group + 4U] >> 4U) | ((packed[group] >> 6U) << 4U);
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
    float* output) {
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
    std::uint32_t intermediate_size) {
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
    Output* output) {
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
    float* output) {
#define STRIX_PROJECT_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             selected_projection_kernel<                 Input, WeightKind::kind_name>),         dim3(output_width), dim3(kThreads), 0, hip_execution_stream(),         input, routes, slot,         reinterpret_cast<const void*>(weights.address),         input_width, output_width, experts, output)
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
    Output* output) {
#define STRIX_DOWN_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             selected_down_accumulate_kernel<                 Output, WeightKind::kind_name>),         dim3(hidden_size), dim3(kThreads), 0, hip_execution_stream(),         intermediate, routes, route_weights, slot,         reinterpret_cast<const void*>(weights.address),         intermediate_size, hidden_size, experts, output)
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

    check_hip(
        hipMemsetAsync(
            reinterpret_cast<void*>(
                output.address),
            0,
            static_cast<std::size_t>(
                output.bytes), hip_execution_stream()),
        "clear routed-expert output");

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
                workspace);
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
                workspace);
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
                    config.intermediate_size);
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
            config.intermediate_size);
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
            reinterpret_cast<Storage*>(
                output.address));
    }
}

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
    const auto temporary_required =
        static_cast<std::uint64_t>(
            config.intermediate_size) *
        2ULL *
        sizeof(float);

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
            request.request_concurrency);
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
    return hash;
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
        static_cast<std::uint64_t>(
            config.intermediate_size) *
        2ULL *
        sizeof(float);
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
    variant.graph_safe = true;
    variant.input_layout =
        "row-major+device-topk";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        "hip-gfx1151-moe-experts-v1";
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

}  // namespace strix
