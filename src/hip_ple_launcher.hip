#include "strix/hip_launch_context.hpp"
#include "strix/hip_ple_launcher.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/ple_workspace.hpp"
#include "strix/graph_ple.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x504c454c41554e32ULL;
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

// Same 256-leaf DAG as project_bf16_to_fp32_kernel<q8_0> inside one wave per row:
// leaf t=8*lane+local accumulates features t, t+256, ... in order, then the
// LDS tree (strides 128..8 are lane shuffles 16..1, strides 4..1 are local).
__global__ __launch_bounds__(256) void project_wave8_q8_bf16_to_fp32_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) + static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint32_t chunk = 0; chunk < input_width; chunk += kThreads) {
        const auto feature = chunk + 8U * lane;
        if (feature < input_width) {
            const auto& block = blocks[feature / 32U];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += bf16_to_float(input[feature + local]) *
                    (__half2float(block.scale) * static_cast<float>(block.values[feature % 32U + local]));
        }
    }
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto partner = __shfl_down(partial[local], distance, 32);
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4]; partial[1] += partial[5];
        partial[2] += partial[6]; partial[3] += partial[7];
        partial[0] += partial[2]; partial[1] += partial[3];
        partial[0] += partial[1];
        output[row] = partial[0];
    }
}

// project_wave8_q8_bf16_to_fp32_kernel for Rows inputs at once: each Q8 block is decoded once and
// applied to every row with that row's own leaf accumulation and tree (bit-identical per row).
template <unsigned Rows>
__global__ __launch_bounds__(256) void project_wave8_q8_bf16_to_fp32_rows_kernel(
    const std::uint16_t* input, std::uint32_t input_stride, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output, std::uint32_t output_stride) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[Rows][8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) + static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint32_t chunk = 0; chunk < input_width; chunk += kThreads) {
        const auto feature = chunk + 8U * lane;
        if (feature < input_width) {
            const auto& block = blocks[feature / 32U];
            float weight[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                weight[local] = __half2float(block.scale) * static_cast<float>(block.values[feature % 32U + local]);
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r)
#pragma unroll
                for (unsigned local = 0; local < 8; ++local)
                    partial[r][local] += bf16_to_float(input[r * input_stride + feature + local]) * weight[local];
        }
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
#pragma unroll
        for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                const auto partner = __shfl_down(partial[r][local], distance, 32);
                if (lane < distance) partial[r][local] += partner;
            }
        }
        if (lane == 0) {
            auto* p = partial[r];
            p[0] += p[4]; p[1] += p[5];
            p[2] += p[6]; p[3] += p[7];
            p[0] += p[2]; p[1] += p[3];
            p[0] += p[1];
            output[static_cast<std::uint64_t>(r) * output_stride + row] = p[0];
        }
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
            load_small(
                 key_norm,
                 key_norm_kind,
                 index);
        key[index] =
            normalized_key;

        const float normalized_query =
            bf16_to_float(
                query[index]) *
            inverse_query *
            load_small(
                 query_norm,
                 query_norm_kind,
                 index);
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
            load_small(
                 norm_weight,
                 norm_kind,
                 index);
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
    std::uint32_t dilation, std::uint32_t* graph_status = nullptr) {
    if (graph_status && atomicAdd(graph_status, 0U) != 0U) return;
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
        0, hip_execution_stream(),
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
        if (input_width % 32U == 0U) {
            hipLaunchKernelGGL(project_wave8_q8_bf16_to_fp32_kernel, dim3((output_width + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), input, reinterpret_cast<const void*>(weights.address), input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP PLE wave8 Q8 projection launch");
            return;
        }
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

    const auto hc_hidden=config.model.hidden_size*config.model.hc_count;
    if(const auto* cached=context.prepared_tensor_views) {
        if(!context.resources) throw std::invalid_argument("prepared PLE roles require retained resources");
        cached->validate_stage_scope(context.resources->distributed_stage,context.step_id);
        cached->validate_source_handles(context.resources->model_manifest,context.resources->model_weights,context.resources->tensor_access);
        const auto& r=cached->ple_roles(context.execution_identity,context.step_id);
        result={tensor_pair(context,r.table),tensor_pair(context,r.key),tensor_pair(context,r.value),tensor_pair(context,r.norm_key),
            tensor_pair(context,r.norm_query),tensor_pair(context,r.norm_conv),tensor_pair(context,r.conv)};
    } else {
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

struct DevicePleHash {
    std::uint32_t ngram, heads_per_ngram, heads, width, eos;
    std::uint64_t multipliers[8], offsets[32], vocabularies[32];
};
__global__ void graph_ple_gather_kernel(DevicePleHash h, const std::uint16_t* table,
    const std::uint32_t* token, const std::uint64_t* position, const std::uint32_t* previous,
    std::uint16_t* gathered, std::uint32_t vocabulary, std::uint64_t context,
    std::uint32_t* status) {
    if (atomicAdd(status, 0U)) return;
    if (*token >= vocabulary || *position >= context) { atomicOr(status, 16U); return; }
    std::uint64_t values[8]; values[0] = *token;
    bool cut = false;
    for (std::uint32_t shift = 1; shift < h.ngram; ++shift) {
        const auto value = *position == 0 ? h.eos : previous[h.ngram - 1U - shift];
        if (value >= vocabulary) { atomicOr(status, 16U); return; }
        cut = cut || value == h.eos; values[shift] = cut ? h.eos : value;
    }
    const auto element = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element >= h.heads * h.width) return;
    const auto head = element / h.width;
    const auto ngram = head / h.heads_per_ngram + 2U;
    auto mixed = values[0] * h.multipliers[0];
    for (std::uint32_t shift = 1; shift < ngram; ++shift) mixed ^= values[shift] * h.multipliers[shift];
    const auto row = mixed % h.vocabularies[head] + h.offsets[head];
    gathered[element] = table[row * h.width + element % h.width];
}
__global__ void graph_ple_history_kernel(const std::uint32_t* token, const std::uint64_t* position,
    const std::uint32_t* previous, std::uint32_t* next, std::uint32_t count,
    std::uint32_t eos, std::uint32_t* status) {
    if (threadIdx.x || atomicAdd(status, 0U)) return;
    for (std::uint32_t i = 0; i + 1U < count; ++i) next[i] = *position == 0 ? eos : previous[i + 1U];
    next[count - 1U] = *token;
}

struct PreparedPleLaunch {
    const HipPleLauncherConfig* config;
    ExecutionBufferAddress input, output, temporary;
    PleWorkspaceLayout workspace_layout;
    std::uint64_t expanded;
    PleTensors tensors;
    SessionStateAddress ngram_state, conv_state;
    PleHashContract hash_contract;
};
PreparedPleLaunch prepare_ple_launch(const ExecutionDispatchContext& context, void* user_data) {
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
        ((!current_hip_launch_context() || !current_hip_launch_context()->graph_capture) &&
        context.resources->ple_lookup == nullptr)) {
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

    return {&config, input, output, temporary, workspace_layout, expanded, tensors,
        ngram_state, conv_state, hash_contract};
}
void preflight_ple(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_ple_launch(context, user_data);
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known window PLE preflight is eager only");
    // P0 uses the existing synchronous lookup/upload boundary. Never start or
    // finish the single pending backend lookup during metadata admission.
    if (prepared.config->upload_pool || prepared.config->enable_owned_event_upload)
        throw std::invalid_argument("known window PLE requires synchronous upload without an event pool");
    if (!dynamic_cast<FileMappedPleLookupBackend*>(context.resources->ple_lookup))
        throw std::invalid_argument("known window PLE currently admits the synchronous file-mapped backend only");
    const auto& table = *prepared.tensors.table.manifest;
    const auto& binding = *prepared.tensors.table.runtime;
    if (table.dimensions.size() != 2 || !table.dimensions[0] ||
        table.dimensions[1] < prepared.hash_contract.minimum_table_rows ||
        (table.ggml_type != 30U && table.ggml_type != 20U) ||
        table.dimensions[0] > std::numeric_limits<std::uint64_t>::max() / 2U ||
        table.dimensions[0] * 2U != prepared.hash_contract.row_bytes ||
        binding.storage != RuntimeTensorStorage::file_mapped || !binding.address ||
        binding.bytes > std::numeric_limits<std::uintptr_t>::max() - binding.address)
        throw std::invalid_argument("known window PLE table metadata or mapping is incompatible");
    auto stored_row_bytes = prepared.hash_contract.row_bytes;
    if (table.ggml_type == 20U) {
        if (table.dimensions[0] % iq4_nl::block_elements != 0U)
            throw std::invalid_argument("known window PLE IQ4_NL rows require complete blocks");
        stored_row_bytes = table.dimensions[0] / iq4_nl::block_elements * iq4_nl::block_bytes;
    }
    if (!stored_row_bytes || table.dimensions[1] > std::numeric_limits<std::uint64_t>::max() / stored_row_bytes ||
        table.byte_size < table.dimensions[1] * stored_row_bytes || binding.bytes < table.byte_size)
        throw std::invalid_argument("known window PLE declared table rows exceed the retained mapping");
    if (context.resources->ple_hash_contract) {
        const auto& a = *context.resources->ple_hash_contract;
        const auto& b = prepared.hash_contract;
        if (a.ngram_size != b.ngram_size || a.heads_per_ngram != b.heads_per_ngram ||
            a.head_count != b.head_count || a.head_dim != b.head_dim || a.eos_token_id != b.eos_token_id ||
            a.row_bytes != b.row_bytes || a.multipliers != b.multipliers || a.head_offsets != b.head_offsets ||
            a.head_vocab_sizes != b.head_vocab_sizes || a.minimum_table_rows != b.minimum_table_rows)
            throw std::invalid_argument("known window PLE prefetch hash contract differs from the selected launcher");
    }
    const auto bytes = static_cast<std::uint64_t>(prepared.hash_contract.head_count) * prepared.hash_contract.row_bytes;
    if (bytes != prepared.workspace_layout.gathered_bytes || bytes > PleHashContract::maximum_heads * 512ULL)
        throw std::invalid_argument("PLE gathered upload does not match its exact workspace region");
}

// key/value: the row's key and value projections when a window computed them already for all its rows.
void enqueue_ple_device(const PreparedPleLaunch& prepared, std::uint16_t* gathered, float* key_rows = nullptr,
    float* value_rows = nullptr) {
    const auto& config = *prepared.config;
    const auto& input = prepared.input;
    const auto& output = prepared.output;
    const auto& temporary = prepared.temporary;
    const auto& workspace_layout = prepared.workspace_layout;
    const auto expanded = prepared.expanded;
    const auto& tensors = prepared.tensors;
    const auto& conv_state = prepared.conv_state;
    auto* workspace =
        reinterpret_cast<std::byte*>(
            temporary.address);
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

    if (key_rows) {
        key = key_rows; value = value_rows;
    } else {
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
    }

    hipLaunchKernelGGL(
        normalize_key_and_gate_kernel,
        dim3(
            static_cast<std::uint32_t>(
                config.model.hc_count)),
        dim3(kThreads),
        0, hip_execution_stream(),
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
        0, hip_execution_stream(),
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
        0, hip_execution_stream(),
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
        0, hip_execution_stream(),
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
            config.model.ngram_size), hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP PLE convolution launch");

}

void launch_ple(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_ple_launch(context, user_data);
    const auto& config = *prepared.config;
    const auto& temporary = prepared.temporary;
    const auto& workspace_layout = prepared.workspace_layout;
    const auto& tensors = prepared.tensors;
    const auto& ngram_state = prepared.ngram_state;
    const auto& hash_contract = prepared.hash_contract;

    alignas(64)
    std::array<
        std::byte,
        PleHashContract::maximum_heads * 512>
        host_staging{};

    PleHashResult hash;
    const auto* graph = current_hip_launch_context();
    if (graph && graph->graph_capture) {
        if (!config.enable_resident_graph || config.upload_pool || config.enable_owned_event_upload ||
            tensors.table.manifest->ggml_type != 30)
            throw std::invalid_argument("captured PLE requires opt-in resident BF16 table without host upload");
        hash.row_count = hash_contract.head_count;
    } else {
    auto& backend =
        *context.resources->ple_lookup;

    if (context.resources->ple_prefetch != nullptr &&
        context.resources->ple_prefetch->active) {
        const auto prefetched_bytes =
            context.resources->ple_prefetch->
                lookup_bytes;
        if (prefetched_bytes >
            host_staging.size()) {
            throw std::overflow_error(
                "PLE prefetched lookup exceeds fixed host staging");
        }
        hash =
            finish_ple_lookup_prefetch(
                context.input_token_id,
                context.position,
                backend,
                *context.resources->ple_prefetch,
                std::span<std::byte>(
                    host_staging.data(),
                    static_cast<std::size_t>(
                        prefetched_bytes)));
    } else {
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
                reinterpret_cast<const void*>(
                    ngram_state.address),
                context_count *
                    sizeof(std::uint32_t));
        }

        hash =
            hash_ple_token(
                hash_contract,
                context.input_token_id,
                std::span<const std::uint32_t>(
                    previous.data(),
                    context_count));

        PleLookupRequest lookup;
        lookup.row_ids =
            std::span<const std::uint64_t>(
                hash.row_ids.data(),
                hash.row_count);
        lookup.row_bytes =
            hash_contract.row_bytes;
        lookup.table_manifest =
            tensors.table.manifest;
        lookup.table_runtime =
            tensors.table.runtime;

        const auto sync_lookup_bytes =
            static_cast<std::uint64_t>(
                hash.row_count) *
            hash_contract.row_bytes;
        if (sync_lookup_bytes >
            host_staging.size()) {
            throw std::overflow_error(
                "PLE lookup exceeds fixed host staging");
        }

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
                        sync_lookup_bytes)));
            handle = 0;
        } catch (...) {
            if (handle != 0) {
                backend.cancel(handle);
            }
            throw;
        }
    }

    }

    const auto lookup_bytes =
        static_cast<std::uint64_t>(
            hash.row_count) *
        hash_contract.row_bytes;

    auto* workspace =
        reinterpret_cast<std::byte*>(
            temporary.address);
    auto* gathered =
        reinterpret_cast<std::uint16_t*>(
            workspace +
            workspace_layout.gathered_offset);

    if (lookup_bytes != workspace_layout.gathered_bytes || lookup_bytes > host_staging.size()) {
        throw std::invalid_argument("PLE gathered upload does not match its exact workspace region");
    }
    if (graph && graph->graph_capture) {
        auto gathered_view = temporary;
        gathered_view.address = reinterpret_cast<std::uintptr_t>(gathered);
        gathered_view.bytes = workspace_layout.gathered_bytes;
        validate_graph_ple_bindings(hash_contract, graph->vocabulary_size, graph->context_tokens,
            *tensors.table.runtime, graph->arguments, graph->status, gathered_view,
            graph->ngram_input, graph->ngram_output);
        DevicePleHash h{};
        h.ngram = hash_contract.ngram_size; h.heads_per_ngram = hash_contract.heads_per_ngram;
        h.heads = hash_contract.head_count; h.width = hash_contract.head_dim; h.eos = hash_contract.eos_token_id;
        std::copy_n(hash_contract.multipliers.begin(), 8, h.multipliers);
        std::copy_n(hash_contract.head_offsets.begin(), 32, h.offsets);
        std::copy_n(hash_contract.head_vocab_sizes.begin(), 32, h.vocabularies);
        hipLaunchKernelGGL(graph_ple_gather_kernel, dim3((h.heads * h.width + 255U) / 256U), dim3(256),
            0, graph->stream, h, reinterpret_cast<const std::uint16_t*>(tensors.table.runtime->address),
            reinterpret_cast<const std::uint32_t*>(graph->arguments.memory.arena_base + graph->arguments.token_offset),
            hip_graph_position(), reinterpret_cast<const std::uint32_t*>(graph->ngram_input.arena_base),
            gathered, graph->vocabulary_size, graph->context_tokens, hip_graph_status());
        check_hip(hipGetLastError(), "resident PLE graph hash/gather");
    } else {
    // The synchronous host upload uses the backend/default copy stream. An
    // explicit nonblocking eager stream must first release reused arena views.
    // This also keeps stack staging lifetime independent of async copy errors.
    if (graph) check_hip(hipStreamSynchronize(graph->stream), "PLE eager upload arena fence");
    if (config.upload_pool != nullptr) {
        auto destination = temporary;
        destination.address = reinterpret_cast<std::uintptr_t>(gathered);
        destination.bytes = workspace_layout.gathered_bytes;
        const auto ticket = config.upload_pool->start(
            std::span<const std::byte>(host_staging.data(), static_cast<std::size_t>(lookup_bytes)),
            destination);
        // Preserve the eager synchronous boundary: dependent projections only
        // start after the copy's event proves the retained host input released.
        config.upload_pool->wait(ticket, destination);
    } else {
        check_hip(hipMemcpy(gathered, host_staging.data(), static_cast<std::size_t>(lookup_bytes),
                            hipMemcpyHostToDevice), "copy PLE gathered embedding");
    }
    }

    enqueue_ple_device(prepared, gathered);

    if (graph && graph->graph_capture) {
        hipLaunchKernelGGL(graph_ple_history_kernel, dim3(1), dim3(1), 0, graph->stream,
            reinterpret_cast<const std::uint32_t*>(graph->arguments.memory.arena_base + graph->arguments.token_offset),
            hip_graph_position(), reinterpret_cast<const std::uint32_t*>(graph->ngram_input.arena_base),
            reinterpret_cast<std::uint32_t*>(graph->ngram_output.arena_base), hash_contract.ngram_size - 1U,
            hash_contract.eos_token_id, hip_graph_status());
        check_hip(hipGetLastError(), "resident PLE graph next history");
        return; // Host history publishes only after whole-graph completion/status.
    }

    // PLE v1 deliberately synchronizes here: host n-gram history must not
    // advance until the device convolution/state update has succeeded.
    check_hip(
        hipStreamSynchronize(hip_execution_stream()),
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
            request.request_concurrency) + (config.enable_resident_graph ? ".resident-graph" : "");
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
    hash_u64(hash, config.enable_resident_graph ? 1 : 0);
    if (config.upload_pool != nullptr || config.enable_owned_event_upload) {
        hash_u64(hash, 0x484f535455504c31ULL);
    }
    return hash;
}

}  // namespace
void testing::enqueue_ple_q8_projection(bool wave8,const std::uint16_t* input,const void* q8,std::uint32_t input_width,
    std::uint32_t output_width,float* output,void* stream) {
    const auto s=static_cast<hipStream_t>(stream);
    if(wave8) hipLaunchKernelGGL(project_wave8_q8_bf16_to_fp32_kernel,dim3((output_width+7U)/8U),dim3(256),0,s,input,q8,input_width,output_width,output);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_bf16_to_fp32_kernel<WeightKind::q8_0>),dim3(output_width),dim3(kThreads),0,s,
        input,q8,input_width,output_width,output);
    check_hip(hipGetLastError(),"PLE Q8 projection test launch");
}

struct HipPleLauncherBundle::OwnedUploads {
    AllocationGenerationSource generations;
    HipHostUploadBackend backend{1};
    HostUploadPool pool{backend, generations,
        HostUploadPolicy{1, HipPleLauncherConfig::owned_event_upload_maximum_bytes,
                        HipPleLauncherConfig::owned_event_upload_maximum_bytes, 4096}};
};

HipPleLauncherBundle::HipPleLauncherBundle(HipPleLauncherConfig config)
    : config_(std::move(config)) {
    if (config_.enable_owned_event_upload) {
        if (config_.upload_pool != nullptr) {
            throw std::invalid_argument("PLE must select either an owned or injected upload pool");
        }
        uploads_ = std::make_unique<OwnedUploads>();
        config_.upload_pool = &uploads_->pool;
    }
}

HipPleLauncherBundle::~HipPleLauncherBundle() = default;

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
    variant.graph_safe = config.enable_resident_graph;
    variant.input_layout =
        "hc-stream-major+file-backed-ngram";
    variant.output_layout =
        "hc-stream-major";
    variant.compiler_identity =
        "hip-gfx1151-ple-v2";
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
    if (config->enable_owned_event_upload && config->upload_pool == nullptr) {
        throw std::invalid_argument("PLE owned event upload requires a launcher bundle");
    }
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_ple,
        config,
        &preflight_ple,
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

struct HipPleKnownWindow::Impl {
    HipPleLauncherConfig config;
    std::uint32_t max_rows = 0, rows = 0, slot = 0;
    bool prefill = false;
    std::uint64_t stride = 0, conv_bytes = 0;
    std::array<std::byte*, 2> staging{};
    std::array<hipEvent_t, 2> released{};
    std::array<bool, 2> recorded{};
    std::byte* snapshots = nullptr;
    float* projections = nullptr;   // every row's key, then every row's value
    std::uintptr_t conv_address = 0, ngram_address = 0;
    std::vector<PleHashResult> hashes;
    ~Impl() {
        for (auto* p : staging) if (p) (void)hipHostFree(p);
        for (auto e : released) if (e) (void)hipEventDestroy(e);
        if (snapshots) (void)hipFree(snapshots);
        if (projections) (void)hipFree(projections);
    }
    // Key and value projections of all rows, each Q8 weight block read once (the per-row
    // wave8 arithmetic). False when a projection is not wave8 Q8: rows project one by one.
    bool project_rows(const PreparedPleLaunch& p, const std::uint16_t* gathered, std::uint32_t count, std::uint32_t first = 0U) const {
        const auto width = static_cast<std::uint32_t>(config.model.ple_embed_dim);
        if (weight_kind(p.tensors.key.manifest->ggml_type) != WeightKind::q8_0 ||
            weight_kind(p.tensors.value.manifest->ggml_type) != WeightKind::q8_0 || width % 32U) return false;
        const auto expanded = static_cast<std::uint32_t>(p.expanded), hidden = static_cast<std::uint32_t>(config.model.hidden_size);
        const auto in_stride = static_cast<std::uint32_t>(stride / sizeof(std::uint16_t));
        float* keys = projections + static_cast<std::uint64_t>(first) * expanded;
        float* values = projections + static_cast<std::uint64_t>(max_rows) * expanded + static_cast<std::uint64_t>(first) * hidden;
        const auto launch = [&](auto rows) {
            constexpr unsigned R = decltype(rows)::value;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<R>), dim3((expanded + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), gathered, in_stride, reinterpret_cast<const void*>(p.tensors.key.runtime->address), width,
                expanded, keys, expanded);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<R>), dim3((hidden + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), gathered, in_stride, reinterpret_cast<const void*>(p.tensors.value.runtime->address), width,
                hidden, values, hidden);
        };
        switch (count) {
        case 1: launch(std::integral_constant<unsigned, 1>{}); break;
        case 2: launch(std::integral_constant<unsigned, 2>{}); break;
        case 3: launch(std::integral_constant<unsigned, 3>{}); break;
        case 4: launch(std::integral_constant<unsigned, 4>{}); break;
        case 5: launch(std::integral_constant<unsigned, 5>{}); break;
        case 6: launch(std::integral_constant<unsigned, 6>{}); break;
        case 7: launch(std::integral_constant<unsigned, 7>{}); break;
        case 8: {
            // Eight register-resident rows run 3x slower than two four-row passes (same rows' arithmetic).
            launch(std::integral_constant<unsigned, 4>{});
            const auto* upper = gathered + 4U * in_stride;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<4>), dim3((expanded + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), upper, in_stride, reinterpret_cast<const void*>(p.tensors.key.runtime->address), width,
                expanded, keys + 4ULL * expanded, expanded);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<4>), dim3((hidden + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), upper, in_stride, reinterpret_cast<const void*>(p.tensors.value.runtime->address), width,
                hidden, values + 4ULL * hidden, hidden);
            break;
        }
        default: return false;
        }
        check_hip(hipGetLastError(), "known-window PLE rows projection launch");
        return true;
    }
    std::vector<PreparedPleLaunch> prepare(std::span<const ExecutionDispatchContext> spans) {
        if (spans.empty() || spans.size() > max_rows)
            throw std::invalid_argument("known-window PLE row count exceeds its staging");
        if (!spans[0].resources || !spans[0].resources->ple_lookup)
            throw std::invalid_argument("known-window PLE requires a host lookup backend");
        std::vector<PreparedPleLaunch> out;
        out.reserve(spans.size());
        // Window rows hold per-row copies of one resource set (only the scratch binding differs, and
        // each row resolves its own arena views below): the same sources resolve the same tensors.
        const auto same_sources = [](const ExecutionRuntimeResources* a, const ExecutionRuntimeResources* b) {
            return a == b || (a && b && a->model_manifest == b->model_manifest && a->model_weights == b->model_weights &&
                a->tensor_access == b->tensor_access && a->session_state_layout == b->session_state_layout &&
                a->session_state_memory == b->session_state_memory && a->ple_lookup == b->ple_lookup &&
                a->ple_hash_contract == b->ple_hash_contract && a->distributed_stage == b->distributed_stage);
        };
        for (std::size_t r = 0; r < spans.size(); ++r) {
            const bool shared = r > 0U && (prefill || (same_sources(spans[r].resources, spans[0].resources) &&
                spans[r].prepared_tensor_views == spans[0].prepared_tensor_views));
            if (shared) {
                // Rows of one step share config, tensors, layout, hash contract and session state with
                // row 0 (checked below); only the row's own arena views differ, resolved and size-checked here.
                const auto& row = spans[r];
                const auto& head = spans[0];
                if (row.op != head.op || row.step_id != head.step_id || row.layer != head.layer ||
                    row.kernel_id != head.kernel_id || row.execution_identity != head.execution_identity ||
                    !same_sources(row.resources, head.resources) || row.logical_rows != 1U || row.request_concurrency != 1U ||
                    row.reads.size() != 1U || row.writes.size() != 1U || row.temporaries.size() != 1U)
                    throw std::invalid_argument("prefill PLE rows differ from their first row");
                auto copy = out.front();
                copy.input = resolve_dispatch_read(row, 0);
                copy.output = resolve_dispatch_write(row, 0);
                copy.temporary = resolve_dispatch_temporary(row, 0);
                copy.ngram_state = resolve_dispatch_state(row, SessionStateComponent::ngram_history);
                copy.conv_state = resolve_dispatch_state(row, SessionStateComponent::ple_convolution);
                if (copy.input.bytes < out.front().input.bytes || copy.output.bytes < out.front().output.bytes ||
                    copy.temporary.bytes < out.front().temporary.bytes)
                    throw std::invalid_argument("prefill PLE row buffers are undersized");
                out.push_back(copy);
            } else out.push_back(prepare_ple_launch(spans[r], &config));
            const auto& a = out.front(); const auto& b = out.back();
            if (spans[r].position != spans[0].position + r || b.conv_state.address != a.conv_state.address ||
                b.conv_state.bytes != a.conv_state.bytes || b.ngram_state.address != a.ngram_state.address ||
                b.tensors.table.runtime != a.tensors.table.runtime ||
                spans[r].resources->ple_lookup != spans[0].resources->ple_lookup)
                throw std::invalid_argument("known-window PLE rows do not share one ordered session");
        }
        if (out[0].conv_state.bytes < conv_bytes)
            throw std::invalid_argument("known-window PLE convolution state is undersized");
        return out;
    }
};

HipPleKnownWindow::HipPleKnownWindow(const ModelContract& model, HipActivationStorage activation, std::uint32_t max_rows,
    bool prefill)
    : impl_(std::make_unique<Impl>()) {
    auto& w = *impl_;
    if (!max_rows || max_rows > (prefill ? 2048U : 8U))
        throw std::invalid_argument("known-window PLE supports one to eight rows (prefill: up to 2048)");
    w.prefill = prefill;
    w.config.model = model; w.config.activation = activation; validate_config(w.config);
    w.max_rows = max_rows;
    w.stride = PleHashContract::maximum_heads * 512ULL;
    w.conv_bytes = static_cast<std::uint64_t>(model.hidden_size * model.hc_count) *
        (model.ple_conv_kernel_size - 1ULL) * model.ngram_size * model.bytes_bf16;
    for (std::size_t i = 0; i < 2; ++i) {
        void* p = nullptr;
        check_hip(hipHostMalloc(&p, w.stride * max_rows, hipHostMallocMapped | hipHostMallocCoherent), "known-window PLE staging");
        w.staging[i] = static_cast<std::byte*>(p);
        check_hip(hipEventCreateWithFlags(&w.released[i], hipEventDisableTiming), "known-window PLE staging event");
    }
    void* snap = nullptr;
    check_hip(hipMalloc(&snap, w.conv_bytes * (prefill ? 1U : max_rows)), "known-window PLE convolution snapshots");
    w.snapshots = static_cast<std::byte*>(snap);
    void* projections = nullptr;
    check_hip(hipMalloc(&projections, sizeof(float) * max_rows * model.hidden_size * (model.hc_count + 1ULL)),
        "known-window PLE row projections");
    w.projections = static_cast<float*>(projections);
    w.hashes.resize(max_rows);
}
HipPleKnownWindow::~HipPleKnownWindow() = default;
std::uint32_t HipPleKnownWindow::last_rows() const noexcept { return impl_->rows; }

void HipPleKnownWindow::preflight(std::span<const ExecutionDispatchContext> rows) const {
    static_cast<void>(impl_->prepare(rows));
}

void HipPleKnownWindow::launch(std::span<const ExecutionDispatchContext> rows) {
    auto& w = *impl_;
    static const bool timing = [] { const char* v = std::getenv("STRIX_PREFILL_TIMING"); return v && v[0] == '1'; }();
    auto mark = std::chrono::steady_clock::now();
    const auto lap = [&](const char* name) {
        if (!timing || !w.prefill) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "prefill ple %-8s %8.2f ms\n", name, std::chrono::duration<double, std::milli>(now - mark).count());
        mark = now;
    };
    const auto prepared = w.prepare(rows);
    lap("prepare");
    const auto prepared_at = std::chrono::steady_clock::now();
    const auto& contract = prepared[0].hash_contract;
    auto& backend = *rows[0].resources->ple_lookup;
    auto* prefetch = rows[0].resources->ple_prefetch;
    const auto context_count = static_cast<std::size_t>(w.config.model.ngram_size - 1ULL);
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> previous{};
    if (rows[0].position == 0)
        std::fill_n(previous.begin(), context_count, static_cast<std::uint32_t>(w.config.model.ple_eos_token_id));
    else
        std::memcpy(previous.data(), reinterpret_cast<const void*>(prepared[0].ngram_state.address),
            context_count * sizeof(std::uint32_t));
    w.slot ^= 1U;
    // The slot was last read by the window before the previous one.
    if (w.recorded[w.slot]) check_hip(hipEventSynchronize(w.released[w.slot]), "known-window PLE staging reuse");
    auto* staging = w.staging[w.slot];
    // Prefill blocks on the file-mapped backend: the n-gram hashes chain row to row, but each row's
    // gather (IQ4_NL decode to BF16) is independent, so the rows are hashed in order on this thread and
    // gathered by up to 8 threads, each with its own stateless file-mapped backend.
    static const std::uint64_t mapped_identity = FileMappedPleLookupBackend{}.implementation_identity();
    const bool parallel = w.prefill && rows.size() >= 32U && backend.implementation_identity() == mapped_identity;
    std::vector<std::size_t> deferred;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        auto* destination = staging + r * w.stride;
        PleHashResult hash;
        if (parallel && !(r == 0 && prefetch && prefetch->active)) {
            hash = hash_ple_token(contract, rows[r].input_token_id,
                std::span<const std::uint32_t>(previous.data(), context_count));
            if (static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes > w.stride)
                throw std::overflow_error("PLE lookup exceeds window staging");
            deferred.push_back(r);
        } else if (r == 0 && prefetch && prefetch->active) {
            if (prefetch->lookup_bytes > w.stride) throw std::overflow_error("PLE prefetched lookup exceeds window staging");
            hash = finish_ple_lookup_prefetch(rows[0].input_token_id, rows[0].position, backend, *prefetch,
                std::span<std::byte>(destination, static_cast<std::size_t>(prefetch->lookup_bytes)));
        } else {
            hash = hash_ple_token(contract, rows[r].input_token_id,
                std::span<const std::uint32_t>(previous.data(), context_count));
            const auto bytes = static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes;
            if (bytes > w.stride) throw std::overflow_error("PLE lookup exceeds window staging");
            PleLookupRequest lookup;
            lookup.row_ids = std::span<const std::uint64_t>(hash.row_ids.data(), hash.row_count);
            lookup.row_bytes = contract.row_bytes;
            lookup.table_manifest = prepared[r].tensors.table.manifest;
            lookup.table_runtime = prepared[r].tensors.table.runtime;
            PleLookupHandle handle = 0;
            try {
                handle = backend.start(lookup);
                backend.finish(handle, std::span<std::byte>(destination, static_cast<std::size_t>(bytes)));
                handle = 0;
            } catch (...) { if (handle) backend.cancel(handle); throw; }
        }
        if (static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes != prepared[r].workspace_layout.gathered_bytes)
            throw std::invalid_argument("PLE window lookup does not match its gathered extent");
        std::copy_n(hash.next_context.begin(), context_count, previous.begin());
        w.hashes[r] = hash;
    }
    lap("hash");
    if (!deferred.empty()) {
        const auto workers = static_cast<std::size_t>(std::min(8U, std::max(1U, std::thread::hardware_concurrency())));
        std::vector<std::exception_ptr> errors(workers);
        const auto gather = [&](std::size_t worker) {
            try {
                FileMappedPleLookupBackend local;
                for (std::size_t i = worker; i < deferred.size(); i += workers) {
                    const auto r = deferred[i];
                    const auto& hash = w.hashes[r];
                    PleLookupRequest lookup;
                    lookup.row_ids = std::span<const std::uint64_t>(hash.row_ids.data(), hash.row_count);
                    lookup.row_bytes = contract.row_bytes;
                    lookup.table_manifest = prepared[r].tensors.table.manifest;
                    lookup.table_runtime = prepared[r].tensors.table.runtime;
                    const auto handle = local.start(lookup);
                    try {
                        local.finish(handle, std::span<std::byte>(staging + r * w.stride,
                            static_cast<std::size_t>(static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes)));
                    } catch (...) { local.cancel(handle); throw; }
                }
            } catch (...) { errors[worker] = std::current_exception(); }
        };
        std::vector<std::thread> threads;
        threads.reserve(workers - 1U);
        for (std::size_t t = 1; t < workers; ++t) threads.emplace_back(gather, t);
        gather(0);
        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
    }
    lap("gather");
    if (!w.prefill) {
        // STRIX_CYCLE_TIMING=1: host time of a window's PLE prepare + hash + gather, printed at exit.
        struct Totals { double ms = 0, prep = 0; unsigned long long n = 0; bool on = std::getenv("STRIX_CYCLE_TIMING") != nullptr;
            ~Totals() { if (on && n) std::fprintf(stderr, "window PLE host prepare+gather: %llu x %.4f ms (prepare %.4f ms)\n", n,
                ms / static_cast<double>(n), prep / static_cast<double>(n)); } };
        static Totals totals;
        if (totals.on) {
            totals.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mark).count();
            totals.prep += std::chrono::duration<double, std::milli>(prepared_at - mark).count(); ++totals.n;
        }
    }
    void* device = nullptr;
    check_hip(hipHostGetDevicePointer(&device, staging, 0), "known-window PLE staging device view");
    const auto stream = hip_execution_stream();
    bool projected = true;
    if (w.prefill) {
        // Eight-row projection groups over all rows (the window path's arithmetic per row).
        for (std::uint32_t first = 0; first < rows.size() && projected; first += 8U) {
            const auto count = std::min<std::uint32_t>(8U, static_cast<std::uint32_t>(rows.size()) - first);
            projected = w.project_rows(prepared[first], reinterpret_cast<const std::uint16_t*>(static_cast<std::byte*>(device) +
                first * w.stride), count, first);
        }
    } else projected = w.project_rows(prepared[0], static_cast<const std::uint16_t*>(device), static_cast<std::uint32_t>(rows.size()));
    const auto expanded = prepared[0].expanded, hidden = w.config.model.hidden_size;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        auto* gathered = reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(device) + r * w.stride);
        if (projected)
            enqueue_ple_device(prepared[r], gathered, w.projections + r * expanded,
                w.projections + w.max_rows * expanded + r * hidden);
        else enqueue_ple_device(prepared[r], gathered);
        if (w.prefill) continue;
        check_hip(hipMemcpyAsync(w.snapshots + r * w.conv_bytes, reinterpret_cast<const void*>(prepared[r].conv_state.address),
            w.conv_bytes, hipMemcpyDeviceToDevice, stream), "known-window PLE convolution snapshot");
    }
    check_hip(hipEventRecord(w.released[w.slot], stream), "known-window PLE staging release");
    w.recorded[w.slot] = true;
    w.rows = static_cast<std::uint32_t>(rows.size());
    w.conv_address = prepared[0].conv_state.address; w.ngram_address = prepared[0].ngram_state.address;
    if (w.prefill) {
        // Every prefill row is kept: the convolution state already is the last row's; publish its history.
        const auto& hash = w.hashes[rows.size() - 1U];
        std::memcpy(reinterpret_cast<void*>(w.ngram_address), hash.next_context.data(),
            static_cast<std::size_t>(hash.context_count) * sizeof(std::uint32_t));
        w.rows = 0;
    }
}

void HipPleKnownWindow::commit(std::uint32_t n_keep, void* stream) {
    if (impl_->prefill) throw std::logic_error("prefill PLE rows publish their history in launch()");
    auto& w = *impl_;
    if (!w.rows || !n_keep || n_keep > w.rows) throw std::invalid_argument("known-window PLE commit outside the last window");
    if (n_keep < w.rows)
        check_hip(hipMemcpyAsync(reinterpret_cast<void*>(w.conv_address), w.snapshots + (n_keep - 1U) * w.conv_bytes,
            w.conv_bytes, hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream)), "known-window PLE convolution commit");
    const auto& hash = w.hashes[n_keep - 1U];
    std::memcpy(reinterpret_cast<void*>(w.ngram_address), hash.next_context.data(),
        static_cast<std::size_t>(hash.context_count) * sizeof(std::uint32_t));
    w.rows = 0;
}

}  // namespace strix
