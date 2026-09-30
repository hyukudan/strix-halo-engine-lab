#include "strix/hip_ple_launcher.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/ple_workspace.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x504c454c41554e31ULL;
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

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);

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
    iq4_xs,
};

enum class SmallWeightKind {
    f32,
    f16,
    bf16,
};

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct PleTensors {
    TensorPair table;
    TensorPair key;
    TensorPair value;
    TensorPair norm_key;
    TensorPair norm_query;
    TensorPair norm_conv;
    TensorPair conv;
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
    const HipPleLauncherConfig& config) {
    if (const auto error =
            config.model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "HIP PLE v1 targets canonical BF16 activations only");
    }
    if (config.model.ple_head_count() != 16 ||
        config.model.ple_head_dim() != 160 ||
        config.model.ple_conv_kernel_size != 4 ||
        config.model.ngram_size != 3 ||
        config.model.hc_count != 4) {
        throw std::invalid_argument(
            "HIP PLE v1 requires pinned Qwen4Exp PLE geometry");
    }
}

WeightKind weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 23: return WeightKind::iq4_xs;
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP PLE projection supports F32, F16, BF16, Q8_0 and IQ4_XS");
    }
}

SmallWeightKind small_weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return SmallWeightKind::f32;
    case 1: return SmallWeightKind::f16;
    case 30: return SmallWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP PLE norm/conv supports F32, F16 and BF16");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

__device__ std::uint16_t float_to_bf16(
    float value) {
    const auto bits =
        __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        rounded >> 16U);
}

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto packed =
        block.scales_l[
            subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (packed >>
             (4U *
              (subblock % 2U))) &
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
    return static_cast<const float*>(
        weights)[
        row * row_width +
        feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return __half2float(
        static_cast<const __half*>(
            weights)[
            row * row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return bf16_to_float(
        static_cast<
            const std::uint16_t*>(
                weights)[
            row * row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
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

__device__ float load_small(
    const void* values,
    SmallWeightKind kind,
    std::uint64_t index) {
    switch (kind) {
    case SmallWeightKind::f32:
        return static_cast<
            const float*>(values)[index];
    case SmallWeightKind::f16:
        return __half2float(
            static_cast<
                const __half*>(values)[index]);
    case SmallWeightKind::bf16:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    values)[index]);
    }
    return 0.0F;
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input_width,
    std::uint64_t output_rows) {
    switch (kind) {
    case WeightKind::f32:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 2ULL;
    case WeightKind::q8_0:
        if (input_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 PLE projection width must divide 32");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 32U) *
            output_rows *
            sizeof(Q8Block);
    case WeightKind::iq4_xs:
        if (input_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS PLE projection width must divide 256");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 256U) *
            output_rows *
            sizeof(IQ4XSBlock);
    }
    throw std::logic_error(
        "unreachable PLE weight kind");
}

std::uint64_t small_weight_bytes(
    SmallWeightKind kind,
    std::uint64_t elements) {
    switch (kind) {
    case SmallWeightKind::f32:
        return elements * 4ULL;
    case SmallWeightKind::f16:
    case SmallWeightKind::bf16:
        return elements * 2ULL;
    }
    throw std::logic_error(
        "unreachable PLE small weight kind");
}

template <WeightKind Kind>
__global__ void project_bf16_to_fp32_kernel(
    const std::uint16_t* input,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
        sum +=
            bf16_to_float(
                input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                out,
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

__global__ void normalize_key_and_gate_kernel(
    float* key,
    const std::uint16_t* query,
    const void* key_norm,
    SmallWeightKind key_norm_kind,
    const void* query_norm,
    SmallWeightKind query_norm_kind,
    float* gates,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    float epsilon) {
    const auto stream =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (stream >= hc_count) {
        return;
    }

    __shared__ float key_reduce[kThreads];
    __shared__ float query_reduce[kThreads];
    __shared__ float inverse_key;
    __shared__ float inverse_query;

    float key_square = 0.0F;
    float query_square = 0.0F;
    const auto base =
        static_cast<std::uint64_t>(
            stream) *
        hidden_size;

    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const float k =
            key[base + feature];
        const float q =
            bf16_to_float(
                query[base + feature]);
        key_square += k * k;
        query_square += q * q;
    }

    key_reduce[lane] =
        key_square;
    query_reduce[lane] =
        query_square;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            key_reduce[lane] +=
                key_reduce[
                    lane + stride];
            query_reduce[lane] +=
                query_reduce[
                    lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        inverse_key =
            rsqrtf(
                key_reduce[0] /
                    static_cast<float>(
                        hidden_size) +
                epsilon);
        inverse_query =
            rsqrtf(
                query_reduce[0] /
                    static_cast<float>(
                        hidden_size) +
                epsilon);
    }
    __syncthreads();

    float dot = 0.0F;
    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const auto index =
            base + feature;
        const float normalized_key =
            key[index] *
            inverse_key *
            (1.0F +
             load_small(
                 key_norm,
                 key_norm_kind,
                 index));
        key[index] =
            normalized_key;

        const float normalized_query =
            bf16_to_float(
                query[index]) *
            inverse_query *
            (1.0F +
             load_small(
                 query_norm,
                 query_norm_kind,
                 index));
        dot +=
            normalized_key *
            normalized_query;
    }

    key_reduce[lane] = dot;
    __syncthreads();
    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            key_reduce[lane] +=
                key_reduce[
                    lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        const float score =
            key_reduce[0] /
            sqrtf(
                static_cast<float>(
                    hidden_size));
        const float magnitude =
            sqrtf(
                fmaxf(
                    fabsf(score),
                    1e-6F));
        const float signed_root =
            score > 0.0F
                ? magnitude
                : score < 0.0F
                    ? -magnitude
                    : 0.0F;
        gates[stream] =
            1.0F /
            (1.0F +
             expf(-signed_root));
    }
}

__global__ void gated_value_kernel(
    const float* value,
    const float* gates,
    float* gated,
    std::uint32_t hidden_size,
    std::uint32_t hc_count) {
    const auto index =
        static_cast<std::uint64_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    const auto total =
        static_cast<std::uint64_t>(
            hidden_size) *
        hc_count;
    if (index >= total) {
        return;
    }
    const auto stream =
        static_cast<std::uint32_t>(
            index /
            hidden_size);
    const auto feature =
        static_cast<std::uint32_t>(
            index %
            hidden_size);
    gated[index] =
        value[feature] *
        gates[stream];
}

__global__ void grouped_norm_kernel(
    const float* input,
    const void* norm_weight,
    SmallWeightKind norm_kind,
    float* output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    float epsilon) {
    const auto stream =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (stream >= hc_count) {
        return;
    }

    __shared__ float reduction[kThreads];
    __shared__ float inverse_rms;

    const auto base =
        static_cast<std::uint64_t>(
            stream) *
        hidden_size;
    float square = 0.0F;
    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const float value =
            input[base + feature];
        square +=
            value * value;
    }
    reduction[lane] = square;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            reduction[lane] +=
                reduction[
                    lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        inverse_rms =
            rsqrtf(
                reduction[0] /
                    static_cast<float>(
                        hidden_size) +
                epsilon);
    }
    __syncthreads();

    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const auto index =
            base + feature;
        output[index] =
            input[index] *
            inverse_rms *
            (1.0F +
             load_small(
                 norm_weight,
                 norm_kind,
                 index));
    }
}

__global__ void conv_residual_kernel(
    const std::uint16_t* hidden,
    const float* gated,
    const float* normalized,
    std::uint16_t* conv_state,
    const void* conv_weight,
    SmallWeightKind conv_kind,
    std::uint16_t* output,
    std::uint32_t channels,
    std::uint32_t history,
    std::uint32_t kernel_size,
    std::uint32_t dilation) {
    const auto channel =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (channel >= channels) {
        return;
    }

    auto* state =
        conv_state +
        static_cast<std::uint64_t>(
            channel) *
        history;

    float conv = 0.0F;
    for (std::uint32_t tap = 0;
         tap < kernel_size;
         ++tap) {
        const auto lag =
            (kernel_size - 1U - tap) *
            dilation;
        const float value =
            lag == 0
                ? normalized[channel]
                : bf16_to_float(
                      state[
                          history -
                          lag]);
        conv +=
            value *
            load_small(
                conv_weight,
                conv_kind,
                static_cast<std::uint64_t>(
                    channel) *
                    kernel_size +
                tap);
    }

    for (std::uint32_t index = 0;
         index + 1U < history;
         ++index) {
        state[index] =
            state[index + 1U];
    }
    state[history - 1U] =
        float_to_bf16(
            normalized[channel]);

    const float silu =
        conv /
        (1.0F +
         expf(-conv));
    output[channel] =
        float_to_bf16(
            bf16_to_float(
                hidden[channel]) +
            gated[channel] +
            silu);
}

template <WeightKind Kind>
void launch_projection(
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            project_bf16_to_fp32_kernel<
                Kind>),
        dim3(output_width),
        dim3(kThreads),
        0, 0,
        input,
        reinterpret_cast<
            const void*>(
                weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP PLE projection launch");
}

void launch_projection_by_kind(
    WeightKind kind,
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    switch (kind) {
    case WeightKind::f32:
        launch_projection<
            WeightKind::f32>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::f16:
        launch_projection<
            WeightKind::f16>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::bf16:
        launch_projection<
            WeightKind::bf16>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::q8_0:
        launch_projection<
            WeightKind::q8_0>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::iq4_xs:
        launch_projection<
            WeightKind::iq4_xs>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
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
        context.resources->
            model_manifest == nullptr) {
        throw std::invalid_argument(
            "HIP PLE requires model manifest");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "PLE tensor index exceeds manifest");
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

PleTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const HipPleLauncherConfig& config) {
    PleTensors result;

    const auto& table_access =
        resolve_dispatch_tensor_access(
            context, 0);
    if (table_access.tensor_indices.size() !=
        1U) {
        throw std::invalid_argument(
            "HIP PLE requires exactly one n-gram table");
    }
    result.table =
        tensor_pair(
            context,
            table_access.tensor_indices[0]);

    const auto& parameter_access =
        resolve_dispatch_tensor_access(
            context, 1);

    std::uint32_t key = 0;
    std::uint32_t value = 0;
    std::uint32_t norm_key = 0;
    std::uint32_t norm_query = 0;
    std::uint32_t norm_conv = 0;
    std::uint32_t conv = 0;

    const auto hc_hidden =
        config.model.hidden_size *
        config.model.hc_count;

    for (const auto index :
         parameter_access.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::ple_parameter) {
            continue;
        }

        if (contains(
                tensor.name,
                "ple_norm_key")) {
            result.norm_key = pair;
            ++norm_key;
        } else if (contains(
                       tensor.name,
                       "ple_norm_query")) {
            result.norm_query = pair;
            ++norm_query;
        } else if (contains(
                       tensor.name,
                       "ple_norm_conv")) {
            result.norm_conv = pair;
            ++norm_conv;
        } else if (contains(
                       tensor.name,
                       "ple_conv1d")) {
            result.conv = pair;
            ++conv;
        } else if (contains(
                       tensor.name,
                       "ple_key")) {
            result.key = pair;
            ++key;
        } else if (contains(
                       tensor.name,
                       "ple_value")) {
            result.value = pair;
            ++value;
        }
    }

    if (key != 1 ||
        value != 1 ||
        norm_key != 1 ||
        norm_query != 1 ||
        norm_conv != 1 ||
        conv != 1) {
        throw std::invalid_argument(
            "PLE parameter tensor family is missing or ambiguous");
    }

    if (result.table.manifest->role !=
            TensorRole::ngram_table ||
        result.table.manifest->
            dimensions.size() < 2 ||
        result.table.manifest->
            dimensions[0] !=
            config.model.ple_head_dim()) {
        throw std::invalid_argument(
            "PLE table geometry does not match contract");
    }
    if (result.key.manifest->
            dimensions.size() < 2 ||
        result.key.manifest->
            dimensions[0] !=
            config.model.ple_embed_dim ||
        result.key.manifest->
            dimensions[1] !=
            hc_hidden ||
        result.value.manifest->
            dimensions.size() < 2 ||
        result.value.manifest->
            dimensions[0] !=
            config.model.ple_embed_dim ||
        result.value.manifest->
            dimensions[1] !=
            config.model.hidden_size) {
        throw std::invalid_argument(
            "PLE projection geometry does not match contract");
    }

    return result;
}

void validate_projection(
    const TensorPair& pair,
    std::uint32_t input,
    std::uint64_t output,
    const char* name) {
    if (pair.runtime == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " is not GPU-visible");
    }
    const auto required =
        required_weight_bytes(
            weight_kind(
                pair.manifest->ggml_type),
            input,
            output);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " binding is undersized");
    }
}

void validate_small(
    const TensorPair& pair,
    std::uint64_t elements,
    const char* name) {
    if (pair.runtime == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " is not GPU-visible");
    }
    const auto required =
        small_weight_bytes(
            small_weight_kind(
                pair.manifest->ggml_type),
            elements);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " binding is undersized");
    }
}

void launch_ple(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::ple) {
        throw std::invalid_argument(
            "HIP PLE launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP PLE expects one hyper read/write and one workspace");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP PLE v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr ||
        context.resources == nullptr ||
        context.resources->ple_lookup ==
            nullptr) {
        throw std::invalid_argument(
            "HIP PLE requires config and lookup backend");
    }

    const auto& config =
        *static_cast<
            HipPleLauncherConfig*>(
                user_data);
    validate_config(config);

    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);
    const auto workspace_layout =
        build_ple_workspace_layout(
            config.model);

    const auto expanded =
        config.model.hidden_size *
        config.model.hc_count;
    const auto expanded_bytes =
        expanded *
        config.model.bytes_bf16;

    if (input.bytes <
            expanded_bytes ||
        output.bytes <
            expanded_bytes ||
        temporary.bytes <
            workspace_layout.total_bytes) {
        throw std::invalid_argument(
            "HIP PLE buffers are undersized");
    }

    const auto tensors =
        identify_tensors(
            context,
            config);

    validate_projection(
        tensors.key,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        expanded,
        "PLE key projection");
    validate_projection(
        tensors.value,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        config.model.hidden_size,
        "PLE value projection");

    validate_small(
        tensors.norm_key,
        expanded,
        "PLE key norm");
    validate_small(
        tensors.norm_query,
        expanded,
        "PLE query norm");
    validate_small(
        tensors.norm_conv,
        expanded,
        "PLE conv norm");
    validate_small(
        tensors.conv,
        config.model.ple_conv_kernel_size *
            expanded,
        "PLE convolution");

    const auto ngram_state =
        resolve_dispatch_state(
            context,
            SessionStateComponent::ngram_history);
    const auto conv_state =
        resolve_dispatch_state(
            context,
            SessionStateComponent::ple_convolution);

    if (ngram_state.storage !=
            SessionStateStorage::host ||
        ngram_state.address == 0 ||
        ngram_state.bytes <
            (config.model.ngram_size - 1ULL) *
                sizeof(std::uint32_t) ||
        conv_state.storage !=
            SessionStateStorage::device ||
        conv_state.address == 0 ||
        conv_state.bytes <
            static_cast<std::uint64_t>(
                expanded) *
                (config.model.ple_conv_kernel_size - 1ULL) *
                config.model.ngram_size *
                config.model.bytes_bf16) {
        throw std::invalid_argument(
            "HIP PLE persistent state binding is invalid");
    }

    const auto hash_contract =
        build_ple_hash_contract(
            config.model);
    std::array<
        std::uint32_t,
        PleHashContract::maximum_ngram - 1>
        previous{};
    const auto context_count =
        static_cast<std::size_t>(
            config.model.ngram_size - 1ULL);

    if (context.position == 0) {
        std::fill_n(
            previous.begin(),
            context_count,
            static_cast<std::uint32_t>(
                config.model.ple_eos_token_id));
    } else {
        std::memcpy(
            previous.data(),
            reinterpret_cast<
                const void*>(
                    ngram_state.address),
            context_count *
                sizeof(std::uint32_t));
    }

    const auto hash =
        hash_ple_token(
            hash_contract,
            context.input_token_id,
            std::span<
                const std::uint32_t>(
                    previous.data(),
                    context_count));

    PleLookupRequest lookup;
    lookup.row_ids =
        std::span<
            const std::uint64_t>(
                hash.row_ids.data(),
                hash.row_count);
    lookup.row_bytes =
        hash_contract.row_bytes;
    lookup.table_manifest =
        tensors.table.manifest;
    lookup.table_runtime =
        tensors.table.runtime;

    alignas(64)
    std::array<
        std::byte,
        PleHashContract::maximum_heads * 512>
        host_staging{};

    const auto lookup_bytes =
        static_cast<std::uint64_t>(
            hash.row_count) *
        hash_contract.row_bytes;
    if (lookup_bytes >
        host_staging.size()) {
        throw std::overflow_error(
            "PLE lookup exceeds fixed host staging");
    }

    auto& backend =
        *context.resources->ple_lookup;
    PleLookupHandle handle = 0;
    try {
        handle =
            backend.start(
                lookup);
        backend.finish(
            handle,
            std::span<std::byte>(
                host_staging.data(),
                static_cast<std::size_t>(
                    lookup_bytes)));
        handle = 0;
    } catch (...) {
        if (handle != 0) {
            backend.cancel(
                handle);
        }
        throw;
    }

    auto* workspace =
        reinterpret_cast<std::byte*>(
            temporary.address);
    auto* gathered =
        reinterpret_cast<std::uint16_t*>(
            workspace +
            workspace_layout.gathered_offset);
    auto* key =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.key_offset);
    auto* value =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.value_offset);
    auto* gated =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.gated_offset);
    auto* normalized =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.normalized_offset);
    auto* gates =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.gate_offset);

    check_hip(
        hipMemcpy(
            gathered,
            host_staging.data(),
            static_cast<std::size_t>(
                lookup_bytes),
            hipMemcpyHostToDevice),
        "copy PLE gathered embedding");

    launch_projection_by_kind(
        weight_kind(
            tensors.key.manifest->
                ggml_type),
        gathered,
        *tensors.key.runtime,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        static_cast<std::uint32_t>(
            expanded),
        key);

    launch_projection_by_kind(
        weight_kind(
            tensors.value.manifest->
                ggml_type),
        gathered,
        *tensors.value.runtime,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        value);

    hipLaunchKernelGGL(
        normalize_key_and_gate_kernel,
        dim3(
            static_cast<std::uint32_t>(
                config.model.hc_count)),
        dim3(kThreads),
        0, 0,
        key,
        reinterpret_cast<
            const std::uint16_t*>(
                input.address),
        reinterpret_cast<
            const void*>(
                tensors.norm_key.
                    runtime->address),
        small_weight_kind(
            tensors.norm_key.
                manifest->ggml_type),
        reinterpret_cast<
            const void*>(
                tensors.norm_query.
                    runtime->address),
        small_weight_kind(
            tensors.norm_query.
                manifest->ggml_type),
        gates,
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        static_cast<std::uint32_t>(
            config.model.hc_count),
        config.model.rms_norm_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP PLE key/query gate launch");

    const auto value_blocks =
        (expanded +
         kThreads - 1ULL) /
        kThreads;
    hipLaunchKernelGGL(
        gated_value_kernel,
        dim3(
            static_cast<std::uint32_t>(
                value_blocks)),
        dim3(kThreads),
        0, 0,
        value,
        gates,
        gated,
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        static_cast<std::uint32_t>(
            config.model.hc_count));
    check_hip(
        hipGetLastError(),
        "HIP PLE gated value launch");

    hipLaunchKernelGGL(
        grouped_norm_kernel,
        dim3(
            static_cast<std::uint32_t>(
                config.model.hc_count)),
        dim3(kThreads),
        0, 0,
        gated,
        reinterpret_cast<
            const void*>(
                tensors.norm_conv.
                    runtime->address),
        small_weight_kind(
            tensors.norm_conv.
                manifest->ggml_type),
        normalized,
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        static_cast<std::uint32_t>(
            config.model.hc_count),
        config.model.rms_norm_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP PLE conv norm launch");

    const auto history =
        static_cast<std::uint32_t>(
            (config.model.ple_conv_kernel_size -
             1ULL) *
            config.model.ngram_size);
    const auto conv_blocks =
        (expanded +
         kThreads - 1ULL) /
        kThreads;

    hipLaunchKernelGGL(
        conv_residual_kernel,
        dim3(
            static_cast<std::uint32_t>(
                conv_blocks)),
        dim3(kThreads),
        0, 0,
        reinterpret_cast<
            const std::uint16_t*>(
                input.address),
        gated,
        normalized,
        reinterpret_cast<
            std::uint16_t*>(
                conv_state.address),
        reinterpret_cast<
            const void*>(
                tensors.conv.runtime->
                    address),
        small_weight_kind(
            tensors.conv.manifest->
                ggml_type),
        reinterpret_cast<
            std::uint16_t*>(
                output.address),
        static_cast<std::uint32_t>(
            expanded),
        history,
        static_cast<std::uint32_t>(
            config.model.ple_conv_kernel_size),
        static_cast<std::uint32_t>(
            config.model.ngram_size));
    check_hip(
        hipGetLastError(),
        "HIP PLE convolution launch");

    // PLE v1 deliberately synchronizes here: host n-gram history must not
    // advance until the device convolution/state update has succeeded.
    check_hip(
        hipDeviceSynchronize(),
        "HIP PLE synchronize");

    std::memcpy(
        reinterpret_cast<void*>(
            ngram_state.address),
        hash.next_context.data(),
        static_cast<std::size_t>(
            hash.context_count) *
            sizeof(std::uint32_t));
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipPleLauncherConfig& config) {
    return
        "hip.ple." +
        std::string(
            config.activation ==
                    HipActivationStorage::bf16
                ? "bf16"
                : "fp32") +
        ".h" +
        std::to_string(
            config.model.hidden_size) +
        ".hc" +
        std::to_string(
            config.model.hc_count) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency);
}

std::uint64_t implementation_identity(
    const HipPleLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.hc_count);
    hash_u64(
        hash,
        config.model.ple_embed_dim);
    hash_u64(
        hash,
        config.model.ple_conv_kernel_size);
    hash_u64(
        hash,
        config.model.ngram_size);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    return hash;
}

}  // namespace

void register_hip_ple_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipPleLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (config.model.hidden_size !=
            model.hidden_size ||
        config.model.ple_embed_dim !=
            model.ple_embed_dim ||
        config.model.ngram_size !=
            model.ngram_size) {
        throw std::invalid_argument(
            "HIP PLE config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP PLE v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP PLE may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            sparse_lookup;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::ple,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        2U *
        kThreads *
        sizeof(float);
    variant.resources.scratch_bytes =
        build_ple_workspace_layout(
            model).total_bytes;
    variant.numerics.input_precision =
        "bf16";
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    // Host hash/lookup and explicit state-commit synchronization make v1
    // unsuitable for graph capture.
    variant.graph_safe = false;
    variant.input_layout =
        "hc-stream-major+file-backed-ngram";
    variant.output_layout =
        "hc-stream-major";
    variant.compiler_identity =
        "hip-gfx1151-ple-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_ple_launcher(
    std::string kernel_id_value,
    HipPleLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP PLE launcher requires id and config");
    }
    validate_config(*config);
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_ple,
        config,
    };
}

void HipPleLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP PLE bundle requires valid compiled execution");
    }

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::ple) {
            continue;
        }
        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP PLE step has no kernel binding");
        }
        registry.add_kernel_launcher(
            make_hip_ple_launcher(
                *binding.kernel_id,
                &config_));
        return;
    }

    throw std::invalid_argument(
        "compiled execution has no PLE step");
}

}  // namespace strix
