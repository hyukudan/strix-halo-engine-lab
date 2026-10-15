#include "strix/hip_launch_context.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"

#include "strix/dispatch_resources.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaxHc = 8;
constexpr std::uint32_t kMaxLowrank = 512;
constexpr std::uint32_t kStagedThreads = 128;
constexpr std::uint32_t kStagedOutputs = 8;
constexpr std::uint32_t kStagedPitch = kStagedThreads + 1;
constexpr std::uint64_t kLauncherTag =
    0x48595045524d4959ULL;
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

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) &
            0xffULL;
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

void validate_config(
    const HipHyperMixLauncherConfig& config) {
    if (config.staged_projections && !config.parallel)
        throw std::invalid_argument("staged hyper projections require parallel hyper-mix");
    if (config.fused_injection && !config.staged_projections)
        throw std::invalid_argument("fused injection requires staged hyper projections");
    if (config.typed_projection && !config.fused_injection)
        throw std::invalid_argument("typed hyper projection requires fused staged execution");
    if (config.hidden_size == 0 ||
        config.hc_count <= 1 ||
        config.hc_count > kMaxHc ||
        config.lowrank == 0 ||
        config.lowrank > kMaxLowrank ||
        !std::isfinite(config.rms_epsilon) ||
        config.rms_epsilon <= 0.0F) {
        throw std::invalid_argument(
            "HIP hyper-mix geometry is invalid");
    }
    if (config.activation !=
            HipActivationStorage::bf16 &&
        config.activation !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP hyper-mix activation storage is unsupported");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

template <typename Storage>
__device__ float load_activation(
    const Storage* values,
    std::uint64_t index);

template <>
__device__ float load_activation<float>(
    const float* values,
    std::uint64_t index) {
    return values[index];
}

template <>
__device__ float load_activation<std::uint16_t>(
    const std::uint16_t* values,
    std::uint64_t index) {
    return bf16_to_float(
        values[index]);
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
    return
        static_cast<std::uint8_t>(
            low |
            static_cast<std::uint8_t>(
                high << 4U));
}

__device__ float load_linear_weight(
    const void* weights,
    std::uint32_t ggml_type,
    std::uint32_t row_width,
    std::uint32_t output_index,
    std::uint32_t input_index) {
    const auto flat =
        static_cast<std::uint64_t>(
            output_index) *
            row_width +
        input_index;

    switch (ggml_type) {
    case 0:
        return static_cast<
            const float*>(
                weights)[flat];

    case 1:
        return __half2float(
            static_cast<
                const __half*>(
                    weights)[flat]);

    case 30:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    weights)[flat]);

    case 8: {
        const auto blocks_per_row =
            row_width / 32U;
        const auto& block =
            static_cast<
                const Q8Block*>(
                    weights)[
                static_cast<
                    std::uint64_t>(
                    output_index) *
                    blocks_per_row +
                input_index / 32U];
        return
            __half2float(
                block.scale) *
            static_cast<float>(
                block.values[
                    input_index % 32U]);
    }

    case 23: {
        const auto blocks_per_row =
            row_width / 256U;
        const auto& block =
            static_cast<
                const IQ4XSBlock*>(
                    weights)[
                static_cast<
                    std::uint64_t>(
                    output_index) *
                    blocks_per_row +
                input_index / 256U];
        const auto within =
            input_index % 256U;
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
                ? static_cast<
                      std::uint8_t>(
                      packed & 0x0fU)
                : static_cast<
                      std::uint8_t>(
                      packed >> 4U);
        const int local =
            static_cast<int>(
                iq4_local_scale(
                    block,
                    subblock)) -
            32;
        return
            __half2float(block.d) *
            static_cast<float>(local) *
            static_cast<float>(
                iq4_codebook[code]);
    }

    default:
        return NAN;
    }
}

__device__ float load_norm_weight(
    const void* weights,
    std::uint32_t ggml_type,
    std::uint32_t index) {
    switch (ggml_type) {
    case 0:
        return static_cast<
            const float*>(
                weights)[index];
    case 1:
        return __half2float(
            static_cast<
                const __half*>(
                    weights)[index]);
    case 30:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    weights)[index]);
    default:
        return NAN;
    }
}


template <typename Storage>
__device__ float normalized_input_value(
    const Storage* expanded_input,
    const void* norm_weight,
    std::uint32_t norm_type,
    const float* inverse_rms,
    std::uint32_t row,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t stream,
    std::uint32_t feature) {
    const auto expanded_width =
        hidden_size * hc_count;
    const auto local_index =
        stream * hidden_size +
        feature;
    const auto source_index =
        static_cast<std::uint64_t>(
            row) *
            expanded_width +
        local_index;
    const float source =
        load_activation(
            expanded_input,
            source_index);
    const float weight =
        load_norm_weight(
            norm_weight,
            norm_type,
            local_index);
    return
        source *
        inverse_rms[stream] *
        weight;
}

__device__ float sigmoid(
    float value) {
    return
        1.0F /
        (1.0F +
         expf(-value));
}

template <typename Storage>
__global__ __launch_bounds__(kThreads)
void hyper_mix_kernel(
    const Storage* expanded_input,
    const void* norm_weight,
    std::uint32_t norm_type,
    const void* down_weight,
    std::uint32_t down_type,
    const void* up_weight,
    std::uint32_t up_type,
    const void* inject_weight,
    std::uint32_t inject_type,
    bool produce_injection,
    Storage* mixed_output,
    float* injection_output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t lowrank,
    float epsilon) {
    const auto row =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    const auto expanded_width =
        hidden_size * hc_count;

    __shared__ float reduction[
        kMaxHc * kThreads];
    __shared__ float inverse_rms[
        kMaxHc];
    __shared__ float lowrank_values[
        kMaxLowrank];

    float local_square[kMaxHc] = {};
    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += blockDim.x) {
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            const auto index =
                static_cast<std::uint64_t>(
                    row) *
                    expanded_width +
                static_cast<std::uint64_t>(
                    stream) *
                    hidden_size +
                feature;
            const float value =
                load_activation(
                    expanded_input,
                    index);
            local_square[stream] +=
                value * value;
        }
    }

    for (std::uint32_t stream = 0;
         stream < hc_count;
         ++stream) {
        reduction[
            stream * kThreads +
            lane] =
            local_square[stream];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            for (std::uint32_t stream = 0;
                 stream < hc_count;
                 ++stream) {
                reduction[
                    stream * kThreads +
                    lane] +=
                    reduction[
                        stream * kThreads +
                        lane +
                        stride];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            inverse_rms[stream] =
                rsqrtf(
                    reduction[
                        stream * kThreads] /
                        static_cast<float>(
                            hidden_size) +
                    epsilon);
        }
    }
    __syncthreads();

    for (std::uint32_t output = lane;
         output < lowrank;
         output += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            for (std::uint32_t feature = 0;
                 feature < hidden_size;
                 ++feature) {
                const auto input_index =
                    stream *
                        hidden_size +
                    feature;
                sum +=
                    normalized_input_value(
                        expanded_input,
                        norm_weight,
                        norm_type,
                        inverse_rms,
                        row,
                        hidden_size,
                        hc_count,
                        stream,
                        feature) *
                    load_linear_weight(
                        down_weight,
                        down_type,
                        expanded_width,
                        output,
                        input_index);
            }
        }
        const float scaled =
            sum /
            static_cast<float>(
                hc_count);
        lowrank_values[output] =
            scaled *
            sigmoid(scaled);
    }
    __syncthreads();

    if (produce_injection &&
        lane < hc_count) {
        float sum = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            for (std::uint32_t feature = 0;
                 feature < hidden_size;
                 ++feature) {
                const auto input_index =
                    stream *
                        hidden_size +
                    feature;
                sum +=
                    normalized_input_value(
                        expanded_input,
                        norm_weight,
                        norm_type,
                        inverse_rms,
                        row,
                        hidden_size,
                        hc_count,
                        stream,
                        feature) *
                    load_linear_weight(
                        inject_weight,
                        inject_type,
                        expanded_width,
                        lane,
                        input_index);
            }
        }
        injection_output[
            static_cast<std::uint64_t>(
                row) *
                hc_count +
            lane] =
            2.0F *
            sigmoid(
                sum /
                static_cast<float>(
                    hc_count));
    }

    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += blockDim.x) {
        float mixed = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            const auto output_index =
                stream *
                    hidden_size +
                feature;
            float up = 0.0F;
            for (std::uint32_t inner = 0;
                 inner < lowrank;
                 ++inner) {
                up +=
                    lowrank_values[inner] *
                    load_linear_weight(
                        up_weight,
                        up_type,
                        lowrank,
                        output_index,
                        inner);
            }
            mixed +=
                sigmoid(up) *
                normalized_input_value(
                    expanded_input,
                    norm_weight,
                    norm_type,
                    inverse_rms,
                    row,
                    hidden_size,
                    hc_count,
                    stream,
                    feature);
        }

        store_activation(
            &mixed_output[
                static_cast<
                    std::uint64_t>(
                    row) *
                    hidden_size +
                feature],
            mixed /
                static_cast<float>(
                    hc_count));
    }
}

bool supported_linear_type(
    std::uint32_t ggml_type) noexcept {
    return ggml_type == 0 ||
           ggml_type == 1 ||
           ggml_type == 8 ||
           ggml_type == 23 ||
           ggml_type == 30;
}

// Preserve the legacy per-output accumulation order while distributing the
// independent projection outputs across workgroups. Normalized input is
// materialized once in the admitted, generation-owned execution scratch.
template <typename Storage>
__global__ void prepare_hyper_norm_kernel(
    const Storage* input, const void* norm, std::uint32_t norm_type,
    std::uint32_t hidden, std::uint32_t streams, float epsilon, float* normalized) {
    const auto stream = blockIdx.x;
    const auto row = blockIdx.y;
    const auto lane = threadIdx.x;
    const auto width = hidden * streams;
    const auto base = static_cast<std::uint64_t>(row) * width + stream * hidden;
    __shared__ float reduction[kThreads];
    __shared__ float inverse;
    float sum = 0.0F;
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads) {
        const float value = load_activation(input, base + feature);
        sum += value * value;
    }
    reduction[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    if (lane == 0) inverse = rsqrtf(reduction[0] / static_cast<float>(hidden) + epsilon);
    __syncthreads();
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads)
        normalized[base + feature] = load_activation(input, base + feature) * inverse *
            load_norm_weight(norm, norm_type, stream * hidden + feature);
}

__global__ void parallel_hyper_lowrank_kernel(
    const float* normalized, const void* down, std::uint32_t kind,
    std::uint32_t width, std::uint32_t streams, std::uint32_t lowrank, float* low) {
    const auto output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= lowrank) return;
    const auto row = blockIdx.y;
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < width; ++feature)
        sum += normalized[static_cast<std::uint64_t>(row) * width + feature] *
            load_linear_weight(down, kind, width, output, feature);
    const float scaled = sum / static_cast<float>(streams);
    low[static_cast<std::uint64_t>(row) * lowrank + output] = scaled * sigmoid(scaled);
}

__global__ void parallel_hyper_injection_kernel(
    const float* normalized, const void* inject, std::uint32_t kind,
    std::uint32_t width, std::uint32_t streams, float* output) {
    const auto stream = threadIdx.x;
    if (stream >= streams) return;
    const auto row = blockIdx.x;
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < width; ++feature)
        sum += normalized[static_cast<std::uint64_t>(row) * width + feature] *
            load_linear_weight(inject, kind, width, stream, feature);
    output[static_cast<std::uint64_t>(row) * streams + stream] =
        2.0F * sigmoid(sum / static_cast<float>(streams));
}

template <typename Storage>
__global__ void parallel_hyper_output_kernel(
    const float* normalized, const float* low, const void* up, std::uint32_t kind,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank, Storage* output) {
    const auto feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= hidden) return;
    const auto row = blockIdx.y;
    const auto width = hidden * streams;
    float mixed = 0.0F;
    for (std::uint32_t stream = 0; stream < streams; ++stream) {
        const auto index = stream * hidden + feature;
        float gate = 0.0F;
        for (std::uint32_t inner = 0; inner < lowrank; ++inner)
            gate += low[static_cast<std::uint64_t>(row) * lowrank + inner] *
                load_linear_weight(up, kind, lowrank, index, inner);
        mixed += normalized[static_cast<std::uint64_t>(row) * width + index] * sigmoid(gate);
    }
    store_activation(&output[static_cast<std::uint64_t>(row) * hidden + feature],
                     mixed / static_cast<float>(streams));
}

// Loader lanes fetch contiguous features of each coefficient row. Only the
// first eight lanes own accumulation chains; neither the feature order nor
// contraction expression changes. Padding avoids adjacent row bank aliasing.
// Every lane participates in both barriers, including partial output tiles.
template <bool Injection>
__global__ void staged_hyper_projection_kernel(
    const float* normalized, const void* weights, std::uint32_t kind,
    std::uint32_t width, std::uint32_t streams, std::uint32_t outputs, float* output) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    const auto lane = threadIdx.x;
    const auto first_output = blockIdx.x * kStagedOutputs;
    const auto row = blockIdx.y;
    const auto chain = first_output + lane;
    const bool active = lane < kStagedOutputs && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < count) {
            values[lane] = normalized[static_cast<std::uint64_t>(row) * width + tile + lane];
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                if (first_output + local < outputs)
                    coefficients[local][lane] = load_linear_weight(
                        weights, kind, width, first_output + local, tile + lane);
            }
        }
        __syncthreads();
        if (active)
            for (std::uint32_t inner = 0; inner < count; ++inner)
                sum += values[inner] * coefficients[lane][inner];
        // Consumers in other waves must finish before any loader refills LDS.
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        output[static_cast<std::uint64_t>(row) * outputs + chain] =
            Injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

// The injection role is uniform for an entire workgroup. Its tile and chain
// arithmetic match v4; the extra workgroups share the lowrank launch/stream.
__global__ void fused_staged_hyper_projection_kernel(
    const float* normalized, const void* down, std::uint32_t down_kind,
    std::uint32_t width, std::uint32_t streams, std::uint32_t lowrank, float* low,
    const void* inject, std::uint32_t inject_kind, float* injection_output,
    std::uint32_t lowrank_blocks) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    const auto lane = threadIdx.x;
    const bool injection = blockIdx.x >= lowrank_blocks;
    const auto role_block = injection ? blockIdx.x - lowrank_blocks : blockIdx.x;
    const auto first_output = role_block * kStagedOutputs;
    const auto row = blockIdx.y;
    const auto chain = first_output + lane;
    const auto outputs = injection ? streams : lowrank;
    const auto* weights = injection ? inject : down;
    const auto kind = injection ? inject_kind : down_kind;
    auto* output = injection ? injection_output : low;
    const bool active = lane < kStagedOutputs && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < count) {
            values[lane] = normalized[static_cast<std::uint64_t>(row) * width + tile + lane];
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                if (first_output + local < outputs)
                    coefficients[local][lane] = load_linear_weight(
                        weights, kind, width, first_output + local, tile + lane);
            }
        }
        __syncthreads();
        if (active)
            for (std::uint32_t inner = 0; inner < count; ++inner)
                sum += values[inner] * coefficients[lane][inner];
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        output[static_cast<std::uint64_t>(row) * outputs + chain] =
            injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

template <std::uint32_t Kind>
__device__ float typed_projection_weight(const void* weights, std::uint32_t width,
    std::uint32_t output, std::uint32_t feature) {
    static_assert(Kind == 8U || Kind == 0U);
    if constexpr (Kind == 8U) {
        const auto& block = static_cast<const Q8Block*>(weights)[
            static_cast<std::uint64_t>(output) * (width / 32U) + feature / 32U];
        return __half2float(block.scale) * static_cast<float>(block.values[feature % 32U]);
    } else {
        return static_cast<const float*>(weights)[static_cast<std::uint64_t>(output) * width + feature];
    }
}

template <std::uint32_t Kind, bool Injection>
__device__ void typed_projection_role(const float* normalized, const void* weights,
    std::uint32_t width, std::uint32_t streams, std::uint32_t outputs,
    std::uint32_t role_block, float* output, float* values,
    float (*coefficients)[kStagedPitch]) {
    const auto lane = threadIdx.x;
    const auto first_output = role_block * kStagedOutputs;
    const auto row = blockIdx.y;
    const auto chain = first_output + lane;
    const bool active = lane < kStagedOutputs && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < count) {
            values[lane] = normalized[static_cast<std::uint64_t>(row) * width + tile + lane];
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                if (first_output + local < outputs)
                    coefficients[local][lane] = typed_projection_weight<Kind>(
                        weights, width, first_output + local, tile + lane);
            }
        }
        __syncthreads();
        if (active)
            for (std::uint32_t inner = 0; inner < count; ++inner)
                sum += values[inner] * coefficients[lane][inner];
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        output[static_cast<std::uint64_t>(row) * outputs + chain] =
            Injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

__global__ void typed_fused_hyper_projection_kernel(
    const float* normalized, const void* down, std::uint32_t width,
    std::uint32_t streams, std::uint32_t lowrank, float* low,
    const void* inject, float* injection_output, std::uint32_t lowrank_blocks) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    // Role is block-uniform: every lane enters one complete producer/consumer loop.
    if (blockIdx.x < lowrank_blocks)
        typed_projection_role<8U, false>(normalized, down, width, streams, lowrank,
            blockIdx.x, low, values, coefficients);
    else
        typed_projection_role<0U, true>(normalized, inject, width, streams, streams,
            blockIdx.x - lowrank_blocks, injection_output, values, coefficients);
}

template <typename Storage>
__global__ void staged_hyper_output_kernel(
    const float* normalized, const float* low, const void* up, std::uint32_t kind,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank, Storage* output) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    const auto lane = threadIdx.x;
    const auto first_feature = blockIdx.x * kStagedOutputs;
    const auto feature = first_feature + lane;
    const auto row = blockIdx.y;
    const bool active = lane < kStagedOutputs && feature < hidden;
    const auto width = hidden * streams;
    float mixed = 0.0F;
    for (std::uint32_t stream = 0; stream < streams; ++stream) {
        float gate = 0.0F;
        for (std::uint32_t tile = 0; tile < lowrank; tile += kStagedThreads) {
            const auto remaining = lowrank - tile;
            const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
            if (lane < count) {
                values[lane] = low[static_cast<std::uint64_t>(row) * lowrank + tile + lane];
                for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                    if (first_feature + local < hidden)
                        coefficients[local][lane] = load_linear_weight(
                            up, kind, lowrank, stream * hidden + first_feature + local, tile + lane);
                }
            }
            __syncthreads();
            if (active)
                for (std::uint32_t inner = 0; inner < count; ++inner)
                    gate += values[inner] * coefficients[lane][inner];
            __syncthreads();
        }
        if (active)
            mixed += normalized[static_cast<std::uint64_t>(row) * width + stream * hidden + feature] * sigmoid(gate);
    }
    if (active)
        store_activation(&output[static_cast<std::uint64_t>(row) * hidden + feature],
                         mixed / static_cast<float>(streams));
}

bool supported_norm_type(
    std::uint32_t ggml_type) noexcept {
    return ggml_type == 0 ||
           ggml_type == 1 ||
           ggml_type == 30;
}

std::uint64_t linear_bytes(
    std::uint32_t ggml_type,
    std::uint32_t row_width,
    std::uint64_t output_rows) {
    switch (ggml_type) {
    case 0:
        return
            static_cast<std::uint64_t>(
                row_width) *
            output_rows *
            4ULL;
    case 1:
    case 30:
        return
            static_cast<std::uint64_t>(
                row_width) *
            output_rows *
            2ULL;
    case 8:
        if (row_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 hyper-mix row width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 32U) *
            output_rows *
            sizeof(Q8Block);
    case 23:
        if (row_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS hyper-mix row width must be divisible by 256");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 256U) *
            output_rows *
            sizeof(IQ4XSBlock);
    default:
        throw std::invalid_argument(
            "unsupported hyper-mix linear weight type");
    }
}

std::uint64_t norm_bytes(
    std::uint32_t ggml_type,
    std::uint32_t elements) {
    switch (ggml_type) {
    case 0:
        return
            static_cast<std::uint64_t>(
                elements) *
            4ULL;
    case 1:
    case 30:
        return
            static_cast<std::uint64_t>(
                elements) *
            2ULL;
    default:
        throw std::invalid_argument(
            "hyper-mix norm supports only F32, F16 and BF16");
    }
}

const ManifestTensor& manifest_tensor(
    const ExecutionDispatchContext& context,
    const ExecutionTensorAccess& access) {
    if (access.tensor_indices.size() != 1U ||
        context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP hyper-mix requires one tensor per binding slot");
    }
    const auto index =
        access.tensor_indices.front();
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP hyper-mix tensor index exceeds manifest");
    }
    return
        context.resources->
            model_manifest->
            tensors[index];
}

const RuntimeTensorBinding& runtime_tensor(
    const ExecutionDispatchContext& context,
    const ExecutionTensorAccess& access) {
    if (access.tensor_indices.size() != 1U) {
        throw std::invalid_argument(
            "HIP hyper-mix runtime tensor slot is not unique");
    }
    const auto& tensor =
        resolve_dispatch_tensor(
            context,
            access.tensor_indices.front());
    if (!runtime_tensor_gpu_visible(tensor.storage) ||
        tensor.address == 0) {
        throw std::invalid_argument(
            "HIP hyper-mix requires resident or repacked GPU-visible weights");
    }
    return tensor;
}

template <typename Storage>
void launch_typed(
    const ExecutionDispatchContext& context,
    const HipHyperMixLauncherConfig& config,
    const ExecutionBufferAddress& input,
    const ExecutionBufferAddress& output,
    const ExecutionBufferAddress* injection_output,
    const ManifestTensor& norm_manifest,
    const RuntimeTensorBinding& norm_runtime,
    const ManifestTensor& down_manifest,
    const RuntimeTensorBinding& down_runtime,
    const ManifestTensor& up_manifest,
    const RuntimeTensorBinding& up_runtime,
    const ManifestTensor* inject_manifest,
    const RuntimeTensorBinding* inject_runtime) {
    if (config.parallel) {
        if (context.temporaries.size() != 1)
            throw std::invalid_argument("parallel hyper-mix requires one admitted workspace");
        const auto temporary = resolve_dispatch_temporary(context, 0);
        if (temporary.use.buffer.buffer_class != ExecutionBufferClass::hyper_workspace)
            throw std::invalid_argument("parallel hyper-mix requires its typed workspace binding");
        const auto width = static_cast<std::uint64_t>(config.hidden_size) * config.hc_count;
        const auto required = static_cast<std::uint64_t>(context.logical_rows) * (width + config.lowrank) * sizeof(float);
        if (temporary.bytes < required || temporary.address % alignof(float) != 0)
            throw std::invalid_argument("parallel hyper-mix workspace is undersized or unaligned");
        auto* normalized = reinterpret_cast<float*>(temporary.address);
        auto* low = normalized + static_cast<std::uint64_t>(context.logical_rows) * width;
        constexpr std::uint32_t projection_threads = 32;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare_hyper_norm_kernel<Storage>),
            dim3(config.hc_count, context.logical_rows), dim3(kThreads), 0, hip_execution_stream(),
            reinterpret_cast<const Storage*>(input.address), reinterpret_cast<const void*>(norm_runtime.address),
            norm_manifest.ggml_type, config.hidden_size, config.hc_count, config.rms_epsilon, normalized);
        check_hip(hipGetLastError(), "parallel hyper-mix normalization launch");
        if (config.staged_projections) {
            if (config.fused_injection) {
                const auto lowrank_blocks = (config.lowrank + kStagedOutputs - 1U) / kStagedOutputs;
                const auto injection_blocks = injection_output ? (config.hc_count + kStagedOutputs - 1U) / kStagedOutputs : 0U;
                if (config.typed_projection) {
                    hipLaunchKernelGGL(typed_fused_hyper_projection_kernel,
                        dim3(lowrank_blocks + injection_blocks, context.logical_rows), dim3(kStagedThreads),
                        0, hip_execution_stream(), normalized, reinterpret_cast<const void*>(down_runtime.address),
                        static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low,
                        inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                        injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, lowrank_blocks);
                } else {
                hipLaunchKernelGGL(fused_staged_hyper_projection_kernel,
                    dim3(lowrank_blocks + injection_blocks, context.logical_rows), dim3(kStagedThreads),
                    0, hip_execution_stream(), normalized,
                    reinterpret_cast<const void*>(down_runtime.address), down_manifest.ggml_type,
                    static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low,
                    inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                    inject_manifest ? inject_manifest->ggml_type : 0U,
                    injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, lowrank_blocks);
                }
                check_hip(hipGetLastError(), "fused staged hyper-mix projection launch");
                hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_output_kernel<Storage>),
                    dim3((config.hidden_size + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                    dim3(kStagedThreads), 0, hip_execution_stream(), normalized, low,
                    reinterpret_cast<const void*>(up_runtime.address), up_manifest.ggml_type,
                    config.hidden_size, config.hc_count, config.lowrank, reinterpret_cast<Storage*>(output.address));
                check_hip(hipGetLastError(), "fused staged hyper-mix output launch");
                return;
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_projection_kernel<false>),
                dim3((config.lowrank + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                dim3(kStagedThreads), 0, hip_execution_stream(), normalized,
                reinterpret_cast<const void*>(down_runtime.address), down_manifest.ggml_type,
                static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low);
            check_hip(hipGetLastError(), "staged hyper-mix lowrank launch");
            if (injection_output) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_projection_kernel<true>),
                    dim3((config.hc_count + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                    dim3(kStagedThreads), 0, hip_execution_stream(), normalized,
                    reinterpret_cast<const void*>(inject_runtime->address), inject_manifest->ggml_type,
                    static_cast<std::uint32_t>(width), config.hc_count, config.hc_count,
                    reinterpret_cast<float*>(injection_output->address));
                check_hip(hipGetLastError(), "staged hyper-mix injection launch");
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_output_kernel<Storage>),
                dim3((config.hidden_size + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                dim3(kStagedThreads), 0, hip_execution_stream(), normalized, low,
                reinterpret_cast<const void*>(up_runtime.address), up_manifest.ggml_type,
                config.hidden_size, config.hc_count, config.lowrank, reinterpret_cast<Storage*>(output.address));
            check_hip(hipGetLastError(), "staged hyper-mix output launch");
            return;
        }
        hipLaunchKernelGGL(parallel_hyper_lowrank_kernel,
            dim3((config.lowrank + projection_threads - 1U) / projection_threads, context.logical_rows),
            dim3(projection_threads), 0, hip_execution_stream(), normalized,
            reinterpret_cast<const void*>(down_runtime.address), down_manifest.ggml_type,
            static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low);
        check_hip(hipGetLastError(), "parallel hyper-mix lowrank launch");
        if (injection_output) {
            hipLaunchKernelGGL(parallel_hyper_injection_kernel, dim3(context.logical_rows),
                dim3(projection_threads), 0, hip_execution_stream(), normalized,
                reinterpret_cast<const void*>(inject_runtime->address), inject_manifest->ggml_type,
                static_cast<std::uint32_t>(width), config.hc_count, reinterpret_cast<float*>(injection_output->address));
            check_hip(hipGetLastError(), "parallel hyper-mix injection launch");
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_hyper_output_kernel<Storage>),
            dim3((config.hidden_size + projection_threads - 1U) / projection_threads, context.logical_rows),
            dim3(projection_threads), 0, hip_execution_stream(), normalized, low,
            reinterpret_cast<const void*>(up_runtime.address), up_manifest.ggml_type,
            config.hidden_size, config.hc_count, config.lowrank, reinterpret_cast<Storage*>(output.address));
        check_hip(hipGetLastError(), "parallel hyper-mix output launch");
        return;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            hyper_mix_kernel<Storage>),
        dim3(context.logical_rows),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<
            const Storage*>(
                input.address),
        reinterpret_cast<const void*>(
            norm_runtime.address),
        norm_manifest.ggml_type,
        reinterpret_cast<const void*>(
            down_runtime.address),
        down_manifest.ggml_type,
        reinterpret_cast<const void*>(
            up_runtime.address),
        up_manifest.ggml_type,
        inject_runtime != nullptr
            ? reinterpret_cast<
                  const void*>(
                  inject_runtime->address)
            : nullptr,
        inject_manifest != nullptr
            ? inject_manifest->ggml_type
            : 0U,
        injection_output != nullptr,
        reinterpret_cast<Storage*>(
            output.address),
        injection_output != nullptr
            ? reinterpret_cast<float*>(
                  injection_output->address)
            : nullptr,
        config.hidden_size,
        config.hc_count,
        config.lowrank,
        config.rms_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP hyper-mix kernel launch");
}

void launch_hyper_mix(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const bool final_mix =
        context.op ==
        SerialOp::final_hyper_mix;
    if (context.op !=
            SerialOp::attn_hyper_mix &&
        context.op !=
            SerialOp::mlp_hyper_mix &&
        !final_mix) {
        throw std::invalid_argument(
            "HIP hyper-mix launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() !=
            (final_mix ? 1U : 2U)) {
        throw std::invalid_argument(
            "HIP hyper-mix scratch IO does not match operator form");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP hyper-mix requires configuration");
    }

    const auto& config =
        *static_cast<
            HipHyperMixLauncherConfig*>(
                user_data);
    validate_config(config);

    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    std::optional<
        ExecutionBufferAddress>
        injection_output;
    if (!final_mix) {
        injection_output =
            resolve_dispatch_write(
                context, 1);
    }

    const auto& norm_access =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto& down_access =
        resolve_dispatch_tensor_access(
            context, 1);
    const auto& up_access =
        resolve_dispatch_tensor_access(
            context, 2);

    const auto& norm_manifest =
        manifest_tensor(
            context,
            norm_access);
    const auto& down_manifest =
        manifest_tensor(
            context,
            down_access);
    const auto& up_manifest =
        manifest_tensor(
            context,
            up_access);

    const auto& norm_runtime =
        runtime_tensor(
            context,
            norm_access);
    const auto& down_runtime =
        runtime_tensor(
            context,
            down_access);
    const auto& up_runtime =
        runtime_tensor(
            context,
            up_access);

    const ManifestTensor*
        inject_manifest = nullptr;
    const RuntimeTensorBinding*
        inject_runtime = nullptr;
    if (!final_mix) {
        const auto& access =
            resolve_dispatch_tensor_access(
                context, 3);
        inject_manifest =
            &manifest_tensor(
                context,
                access);
        inject_runtime =
            &runtime_tensor(
                context,
                access);
    }

    if (config.typed_projection && (down_manifest.ggml_type != 8U ||
        (inject_manifest != nullptr && inject_manifest->ggml_type != 0U)))
        throw std::invalid_argument("typed hyper projection requires Q8_0 down and F32 injection");

    const auto expanded_width =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        config.hc_count;
    const auto activation_bytes =
        static_cast<std::uint32_t>(
            config.activation);

    if (input.bytes <
            static_cast<std::uint64_t>(
                context.logical_rows) *
                expanded_width *
                activation_bytes ||
        output.bytes <
            static_cast<std::uint64_t>(
                context.logical_rows) *
                config.hidden_size *
                activation_bytes) {
        throw std::invalid_argument(
            "HIP hyper-mix activation buffers are undersized");
    }
    if (injection_output.has_value() &&
        injection_output->bytes <
            static_cast<std::uint64_t>(
                context.logical_rows) *
                config.hc_count *
                sizeof(float)) {
        throw std::invalid_argument(
            "HIP hyper-mix injection buffer is undersized");
    }

    if (!supported_norm_type(
            norm_manifest.ggml_type) ||
        !supported_linear_type(
            down_manifest.ggml_type) ||
        !supported_linear_type(
            up_manifest.ggml_type) ||
        (inject_manifest != nullptr &&
         !supported_linear_type(
             inject_manifest->
                 ggml_type))) {
        throw std::invalid_argument(
            "HIP hyper-mix contains unsupported weight type");
    }

    if (norm_manifest.dimensions.empty() ||
        norm_manifest.dimensions[0] !=
            expanded_width ||
        down_manifest.dimensions.size() < 2 ||
        down_manifest.dimensions[0] !=
            expanded_width ||
        down_manifest.dimensions[1] !=
            config.lowrank ||
        up_manifest.dimensions.size() < 2 ||
        up_manifest.dimensions[0] !=
            config.lowrank ||
        up_manifest.dimensions[1] !=
            expanded_width) {
        throw std::invalid_argument(
            "HIP hyper-mix tensor geometry does not match contract");
    }
    if (inject_manifest != nullptr &&
        (inject_manifest->dimensions.size() < 2 ||
         inject_manifest->dimensions[0] !=
             expanded_width ||
         inject_manifest->dimensions[1] !=
             config.hc_count)) {
        throw std::invalid_argument(
            "HIP hyper-mix inject tensor geometry does not match contract");
    }

    const auto norm_required =
        norm_bytes(
            norm_manifest.ggml_type,
            static_cast<
                std::uint32_t>(
                expanded_width));
    const auto down_required =
        linear_bytes(
            down_manifest.ggml_type,
            static_cast<
                std::uint32_t>(
                expanded_width),
            config.lowrank);
    const auto up_required =
        linear_bytes(
            up_manifest.ggml_type,
            config.lowrank,
            expanded_width);

    if (norm_runtime.bytes <
            norm_required ||
        down_runtime.bytes <
            down_required ||
        up_runtime.bytes <
            up_required) {
        throw std::invalid_argument(
            "HIP hyper-mix runtime weight binding is undersized");
    }
    if (inject_manifest != nullptr) {
        const auto inject_required =
            linear_bytes(
                inject_manifest->ggml_type,
                static_cast<
                    std::uint32_t>(
                    expanded_width),
                config.hc_count);
        if (inject_runtime->bytes <
            inject_required) {
            throw std::invalid_argument(
                "HIP hyper-mix inject runtime binding is undersized");
        }
    }

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_typed<
            std::uint16_t>(
                context,
                config,
                input,
                output,
                injection_output
                    ? &*injection_output
                    : nullptr,
                norm_manifest,
                norm_runtime,
                down_manifest,
                down_runtime,
                up_manifest,
                up_runtime,
                inject_manifest,
                inject_runtime);
        return;
    case HipActivationStorage::fp32:
        launch_typed<float>(
            context,
            config,
            input,
            output,
            injection_output
                ? &*injection_output
                : nullptr,
            norm_manifest,
            norm_runtime,
            down_manifest,
            down_runtime,
            up_manifest,
            up_runtime,
            inject_manifest,
            inject_runtime);
        return;
    }
    throw std::logic_error(
        "unreachable hyper-mix activation storage");
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipHyperMixLauncherConfig& config) {
    return
        "hip.hyper_mix." +
        std::string(
            storage_name(
                config.activation)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".hc" +
        std::to_string(
            config.hc_count) +
        ".lr" +
        std::to_string(
            config.lowrank) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (config.parallel ? ".parallel" : "") +
        (config.staged_projections ? ".staged" : "") + (config.fused_injection ? ".fused-injection-v5" : "") + (config.typed_projection ? ".typed-q8-f32-v1" : "");
}

std::uint64_t implementation_identity(
    const HipHyperMixLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.hc_count);
    hash_u64(hash, config.lowrank);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    std::uint32_t epsilon_bits = 0;
    static_assert(
        sizeof(epsilon_bits) ==
        sizeof(config.rms_epsilon));
    std::memcpy(
        &epsilon_bits,
        &config.rms_epsilon,
        sizeof(epsilon_bits));
    hash_u64(hash, epsilon_bits);
    hash_u64(hash, config.parallel ? 1U : 0U);
    if (config.staged_projections) hash_u64(hash, 0x535441474544ULL);
    if (config.fused_injection) hash_u64(hash, 0x4655534544484305ULL);
    if (config.typed_projection) hash_u64(hash, 0x5459504544484301ULL);
    return hash;
}

}  // namespace

void register_hip_hyper_mix_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipHyperMixLauncherConfig& config,
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
        config.hc_count !=
            model.hc_count ||
        config.lowrank !=
            model.hc_lowrank) {
        throw std::invalid_argument(
            "HIP hyper-mix config does not match model contract");
    }
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "HIP hyper-mix variant requires non-zero rows and concurrency");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP hyper-mix may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            hyper_mix;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::attn_hyper_mix,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        (kMaxHc * kThreads +
         kMaxHc +
         kMaxLowrank) *
        sizeof(float);
    if (config.parallel) {
        // Conservative 256-byte-rounded declarations include the shared
        // normalization inverse as well as the reduction array.
        variant.resources.lds_bytes = ((kThreads + 1U) * sizeof(float) + 255U) & ~255ULL;
        if (config.staged_projections)
            variant.resources.lds_bytes = ((kStagedThreads + kStagedOutputs * kStagedPitch) * sizeof(float) + 255U) & ~255ULL;
        variant.resources.scratch_bytes = static_cast<std::uint64_t>(request.logical_rows) *
            (static_cast<std::uint64_t>(config.hidden_size) * config.hc_count + config.lowrank) * sizeof(float);
    }
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
        "hc-stream-major";
    variant.output_layout =
        "row-major+fp32-injection";
    variant.compiler_identity =
        config.typed_projection ? "hip-gfx1151-hyper-mix-v6-typed-projection" :
        config.fused_injection ? "hip-gfx1151-hyper-mix-v5-fused-injection" :
        config.staged_projections ? "hip-gfx1151-hyper-mix-v4-staged" :
        config.parallel ? "hip-gfx1151-hyper-mix-v3-parallel" : "hip-gfx1151-hyper-mix-v2";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_hyper_mix_launcher(
    std::string kernel_id_value,
    HipHyperMixLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP hyper-mix launcher requires id and config");
    }
    validate_config(*config);
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_hyper_mix,
        config,
    };
}

void preflight_typed_hyper_projection_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources) {
    if (compiled.execution_identity == 0 || resources.model_manifest == nullptr || resources.tensor_access == nullptr)
        throw std::invalid_argument("typed hyper preflight requires a complete compiled model binding");
    std::uint32_t count = 0;
    for (const auto& step : compiled.execution.steps) {
        if (step.op != SerialOp::attn_hyper_mix && step.op != SerialOp::mlp_hyper_mix && step.op != SerialOp::final_hyper_mix) continue;
        ExecutionDispatchContext context;
        context.execution_identity = compiled.execution_identity;
        context.step_id = step.id;
        context.op = step.op;
        context.resources = &resources;
        const auto& down = manifest_tensor(context, resolve_dispatch_tensor_access(context, 1));
        if (down.ggml_type != 8U) throw std::invalid_argument("typed hyper model contains a non-Q8_0 down projection");
        if (step.op != SerialOp::final_hyper_mix) {
            const auto& inject = manifest_tensor(context, resolve_dispatch_tensor_access(context, 3));
            if (inject.ggml_type != 0U) throw std::invalid_argument("typed hyper model contains a non-F32 injection projection");
        }
        ++count;
    }
    if (count == 0U) throw std::invalid_argument("typed hyper preflight found no mixer bindings");
}

void HipHyperMixLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry,
    const ExecutionRuntimeResources* resources) {
    if (config_.typed_projection) {
        if (resources == nullptr) throw std::invalid_argument("typed hyper bundle requires complete model preflight resources");
        preflight_typed_hyper_projection_model(compiled, *resources);
    }
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP hyper-mix bundle requires valid compiled execution");
    }

    // Fail atomically for the typed selection before registering its first id.
    if (config_.typed_projection) {
        ExecutionBindingRequest expected;
        expected.logical_rows = compiled.logical_rows;
        expected.request_concurrency = compiled.request_concurrency;
        const auto selected_id = kernel_id(expected, config_);
        for (std::size_t index = 0; index < compiled.execution.steps.size(); ++index) {
            const auto op = compiled.execution.steps[index].op;
            if (op != SerialOp::attn_hyper_mix && op != SerialOp::mlp_hyper_mix &&
                op != SerialOp::final_hyper_mix) continue;
            const auto& binding = compiled.bindings.bindings[index];
            if (binding.host_or_runtime_step || !binding.kernel_id.has_value() ||
                *binding.kernel_id != selected_id)
                throw std::invalid_argument("typed hyper bundle differs from selected kernel identity");
        }
    }

    std::string registered_id;
    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        const auto op =
            compiled.execution.steps[index].op;
        if (op !=
                SerialOp::attn_hyper_mix &&
            op !=
                SerialOp::mlp_hyper_mix &&
            op !=
                SerialOp::final_hyper_mix) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP hyper-mix step has no kernel binding");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_hyper_mix_launcher(
                    registered_id,
                    &config_));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP hyper-mix bundle requires one selected kernel id for all mixer steps");
        }
    }
}

}  // namespace strix
