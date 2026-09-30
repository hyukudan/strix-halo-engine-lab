#include "strix/hip_gdn_launcher.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/memory_plan.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kProjectionThreads = 256;
constexpr std::uint32_t kRecurrentThreads = 128;
constexpr std::uint32_t kValueTile = 32;
constexpr std::uint64_t kLauncherTag =
    0x47444e4c41554e31ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4NLBlock {
    __half d;
    std::uint8_t qs[16];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4NLBlock) == 18);
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
    iq4_nl,
    iq4_xs,
};

enum class SmallWeightKind : std::uint32_t {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
};

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct GdnTensors {
    TensorPair qkv;
    TensorPair z;
    TensorPair beta;
    TensorPair alpha;
    TensorPair conv;
    TensorPair dt;
    TensorPair a;
    TensorPair norm;
    TensorPair out;
};

struct Workspace {
    float* qkv = nullptr;
    float* z = nullptr;
    float* beta = nullptr;
    float* gate = nullptr;
    float* core = nullptr;
    std::uint64_t elements = 0;
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

void hash_float(
    std::uint64_t& hash,
    float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(
        sizeof(bits) == sizeof(value));
    std::memcpy(
        &bits,
        &value,
        sizeof(bits));
    hash_u64(hash, bits);
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

std::uint32_t checked_u32(
    std::uint64_t value,
    const char* name) {
    if (value == 0 ||
        value >
            std::numeric_limits<
                std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) +
            " does not fit uint32");
    }
    return static_cast<std::uint32_t>(
        value);
}

void validate_config(
    const HipGdnLauncherConfig& config) {
    if (const auto error =
            config.model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "HIP GDN v1 targets canonical BF16 activations only");
    }
    if (!config.model.gdn_qk_l2norm ||
        !config.model.gdn_conv_silu ||
        !config.model.gdn_output_gate_sigmoid) {
        throw std::invalid_argument(
            "HIP GDN v1 requires pinned Qwen4Exp GDN semantics");
    }
    if (config.model.gdn_key_head_dim !=
            kRecurrentThreads ||
        config.model.gdn_value_head_dim !=
            kRecurrentThreads ||
        config.model.gdn_key_head_dim !=
            config.model.gdn_value_head_dim ||
        config.model.gdn_value_heads %
                config.model.gdn_key_heads !=
            0 ||
        config.model.gdn_value_head_dim %
                kValueTile !=
            0) {
        throw std::invalid_argument(
            "HIP GDN v1 requires 128-wide equal/tileable key and value heads");
    }
}

WeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 20: return WeightKind::iq4_nl;
    case 23: return WeightKind::iq4_xs;
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP GDN projection supports F32, F16, BF16, Q8_0, IQ4_NL and IQ4_XS");
    }
}

SmallWeightKind small_weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return SmallWeightKind::f32;
    case 1: return SmallWeightKind::f16;
    case 30: return SmallWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP GDN small tensors support F32, F16 and BF16");
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

__device__ float round_bf16(
    float value) {
    return bf16_to_float(
        float_to_bf16(value));
}

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
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
        row *
            row_width +
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
            row *
                row_width +
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
            row *
                row_width +
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
__device__ float load_weight<WeightKind::iq4_nl>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4NLBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
            feature / 32U];
    const auto within =
        feature % 32U;
    const auto packed =
        block.qs[
            within % 16U];
    const auto code =
        within < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    return __half2float(block.d) *
           static_cast<float>(
               iq4_codebook[code]);
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
                const __half*>(
                    values)[index]);
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
    case WeightKind::iq4_nl:
        if (input_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0/IQ4_NL GDN projection width must divide 32");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 32U) *
            output_rows *
            (kind == WeightKind::q8_0
                 ? sizeof(Q8Block)
                 : sizeof(IQ4NLBlock));
    case WeightKind::iq4_xs:
        if (input_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS GDN projection width must divide 256");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 256U) *
            output_rows *
            sizeof(IQ4XSBlock);
    }
    throw std::logic_error(
        "unreachable GDN weight kind");
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
        "unreachable GDN small weight kind");
}

__global__ void clear_bf16_kernel(
    std::uint16_t* output,
    std::uint32_t elements) {
    const auto index =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index < elements) {
        output[index] = 0;
    }
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

    __shared__ float scratch[kProjectionThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kProjectionThreads) {
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
             kProjectionThreads / 2U;
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
            round_bf16(
                scratch[0]);
    }
}

template <WeightKind Kind>
__global__ void project_fp32_to_bf16_kernel(
    const float* input,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[kProjectionThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kProjectionThreads) {
        sum +=
            input[feature] *
            load_weight<Kind>(
                weights,
                input_width,
                out,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kProjectionThreads / 2U;
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
            float_to_bf16(
                scratch[0]);
    }
}

__global__ void conv_update_kernel(
    float* qkv,
    std::uint16_t* conv_state,
    const void* conv_weights,
    SmallWeightKind weight_kind_value,
    std::uint32_t channels,
    std::uint32_t kernel_size) {
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
            kernel_size;

    for (std::uint32_t index = 0;
         index + 1U < kernel_size;
         ++index) {
        state[index] =
            state[index + 1U];
    }
    state[kernel_size - 1U] =
        float_to_bf16(
            qkv[channel]);

    float sum = 0.0F;
    for (std::uint32_t index = 0;
         index < kernel_size;
         ++index) {
        sum +=
            bf16_to_float(
                state[index]) *
            load_small(
                conv_weights,
                weight_kind_value,
                static_cast<std::uint64_t>(
                    channel) *
                    kernel_size +
                index);
    }
    const float silu =
        sum /
        (1.0F +
         expf(-sum));
    qkv[channel] =
        round_bf16(
            silu);
}

__device__ float softplus(
    float value) {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return expf(value);
    }
    return log1pf(
        expf(value));
}

__global__ void transform_beta_gate_kernel(
    float* beta,
    float* gate,
    const void* dt,
    SmallWeightKind dt_kind,
    const void* a,
    SmallWeightKind a_kind,
    std::uint32_t heads) {
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (head >= heads) {
        return;
    }

    beta[head] =
        round_bf16(
            1.0F /
            (1.0F +
             expf(-beta[head])));

    const float biased =
        gate[head] +
        load_small(
            dt,
            dt_kind,
            head);
    gate[head] =
        load_small(
            a,
            a_kind,
            head) *
        softplus(
            biased);
}

__global__ void normalize_qk_kernel(
    float* qkv,
    std::uint32_t key_heads,
    std::uint32_t head_dim,
    float epsilon) {
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const bool is_key =
        block >= key_heads;
    const auto head =
        is_key
            ? block - key_heads
            : block;
    if (head >= key_heads ||
        threadIdx.x >= head_dim) {
        return;
    }

    __shared__ float scratch[kRecurrentThreads];

    const auto base =
        (is_key
             ? static_cast<std::uint64_t>(
                   key_heads) *
                   head_dim
             : 0ULL) +
        static_cast<std::uint64_t>(
            head) *
            head_dim;

    const float value =
        qkv[
            base +
            threadIdx.x];
    scratch[threadIdx.x] =
        value * value;
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
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

    const float inverse =
        rsqrtf(
            scratch[0] +
            epsilon);
    const float q_scale =
        is_key
            ? 1.0F
            : rsqrtf(
                  static_cast<float>(
                      head_dim));
    qkv[
        base +
        threadIdx.x] =
        value *
        inverse *
        q_scale;
}

__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_step_kernel(
    const float* qkv,
    const float* beta,
    const float* gate,
    float* state,
    float* core,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim) {
    const auto tile_count =
        head_dim /
        kValueTile;
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto value_head =
        block /
        tile_count;
    const auto tile =
        block %
        tile_count;
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    if (value_head >= value_heads ||
        lane >= head_dim) {
        return;
    }

    const auto key_head =
        value_head %
        key_heads;
    const auto value_base =
        tile *
        kValueTile;

    const auto q_base =
        static_cast<std::uint64_t>(
            key_head) *
        head_dim;
    const auto k_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim +
        q_base;
    const auto v_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim *
            2ULL +
        static_cast<std::uint64_t>(
            value_head) *
            head_dim;

    __shared__ float reduction[
        kRecurrentThreads *
        kValueTile];

    float retained[kValueTile];
#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        retained[column] =
            state[state_index] *
            expf(
                gate[value_head]);
        reduction[
            static_cast<std::size_t>(
                lane) *
                kValueTile +
            column] =
            retained[column] *
            qkv[
                k_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            const auto value_index =
                value_base +
                column;
            reduction[column] =
                (qkv[
                     v_base +
                     value_index] -
                 reduction[column]) *
                beta[value_head];
        }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto delta =
            reduction[column];
        retained[column] +=
            qkv[
                k_base +
                lane] *
            delta;
        reduction[
            static_cast<std::size_t>(
                lane) *
                kValueTile +
            column] =
            retained[column] *
            qkv[
                q_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            core[
                static_cast<std::uint64_t>(
                    value_head) *
                    head_dim +
                value_base +
                column] =
                reduction[column];
        }
    }

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        state[state_index] =
            retained[column];
    }
}

__global__ void gated_rms_norm_kernel(
    float* core,
    const float* z,
    const void* norm_weights,
    SmallWeightKind norm_kind,
    std::uint32_t heads,
    std::uint32_t head_dim,
    float epsilon) {
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (head >= heads ||
        threadIdx.x >= head_dim) {
        return;
    }

    __shared__ float scratch[kRecurrentThreads];
    const auto index =
        static_cast<std::uint64_t>(
            head) *
            head_dim +
        threadIdx.x;
    const float value =
        core[index];
    scratch[threadIdx.x] =
        value * value;
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
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

    const float inv =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    const float gate_value =
        1.0F /
        (1.0F +
         expf(-z[index]));
    core[index] =
        value *
        inv *
        load_small(
            norm_weights,
            norm_kind,
            threadIdx.x) *
        gate_value;
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
        dim3(kProjectionThreads),
        0,
        0,
        input,
        reinterpret_cast<const void*>(
            weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP GDN input projection launch");
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
        launch_projection<WeightKind::f32>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::f16:
        launch_projection<WeightKind::f16>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::bf16:
        launch_projection<WeightKind::bf16>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::q8_0:
        launch_projection<WeightKind::q8_0>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::iq4_nl:
        launch_projection<WeightKind::iq4_nl>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::iq4_xs:
        launch_projection<WeightKind::iq4_xs>(
            input, weights, input_width, output_width, output);
        return;
    }
}

template <WeightKind Kind>
void launch_output_projection(
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            project_fp32_to_bf16_kernel<
                Kind>),
        dim3(output_width),
        dim3(kProjectionThreads),
        0,
        0,
        input,
        reinterpret_cast<const void*>(
            weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP GDN output projection launch");
}

void launch_output_by_kind(
    WeightKind kind,
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output) {
    switch (kind) {
    case WeightKind::f32:
        launch_output_projection<WeightKind::f32>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::f16:
        launch_output_projection<WeightKind::f16>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::bf16:
        launch_output_projection<WeightKind::bf16>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::q8_0:
        launch_output_projection<WeightKind::q8_0>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::iq4_nl:
        launch_output_projection<WeightKind::iq4_nl>(
            input, weights, input_width, output_width, output);
        return;
    case WeightKind::iq4_xs:
        launch_output_projection<WeightKind::iq4_xs>(
            input, weights, input_width, output_width, output);
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
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP GDN requires model manifest resources");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP GDN tensor index exceeds manifest");
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

void assign_unique(
    TensorPair& destination,
    std::uint32_t& count,
    TensorPair candidate) {
    destination =
        candidate;
    ++count;
}

GdnTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const HipGdnLauncherConfig& config) {
    const auto& recurrent =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto& attention =
        resolve_dispatch_tensor_access(
            context, 1);

    GdnTensors result;
    std::uint32_t qkv_count = 0;
    std::uint32_t z_count = 0;
    std::uint32_t beta_count = 0;
    std::uint32_t alpha_count = 0;
    std::uint32_t conv_count = 0;
    std::uint32_t dt_count = 0;
    std::uint32_t a_count = 0;
    std::uint32_t norm_count = 0;
    std::uint32_t out_count = 0;

    for (const auto index :
         attention.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::attention_projection) {
            continue;
        }
        if (contains(
                tensor.name,
                "attn_qkv")) {
            assign_unique(
                result.qkv,
                qkv_count,
                pair);
        } else if (contains(
                       tensor.name,
                       "attn_gate")) {
            assign_unique(
                result.z,
                z_count,
                pair);
        }
    }

    for (const auto index :
         recurrent.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::recurrent_state) {
            continue;
        }
        const auto name =
            std::string_view(
                tensor.name);
        if (contains(
                name,
                "ssm_conv1d")) {
            assign_unique(
                result.conv,
                conv_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_beta")) {
            assign_unique(
                result.beta,
                beta_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_alpha")) {
            assign_unique(
                result.alpha,
                alpha_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_dt")) {
            assign_unique(
                result.dt,
                dt_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_norm")) {
            assign_unique(
                result.norm,
                norm_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_out")) {
            assign_unique(
                result.out,
                out_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_a") &&
                   !contains(
                       name,
                       "ssm_alpha")) {
            assign_unique(
                result.a,
                a_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_in")) {
            throw std::invalid_argument(
                "HIP GDN v1 does not support legacy fused ssm_in export");
        }
    }

    if (qkv_count != 1 ||
        z_count != 1 ||
        beta_count != 1 ||
        alpha_count != 1 ||
        conv_count != 1 ||
        dt_count != 1 ||
        a_count != 1 ||
        norm_count != 1 ||
        out_count != 1) {
        throw std::invalid_argument(
            "HIP GDN tensor family is missing or ambiguous");
    }

    const auto hidden =
        config.model.hidden_size;
    const auto key_width =
        config.model.gdn_key_heads *
        config.model.gdn_key_head_dim;
    const auto value_width =
        config.model.gdn_value_heads *
        config.model.gdn_value_head_dim;
    const auto conv_width =
        key_width * 2ULL +
        value_width;

    const auto expect_matrix =
        [&](const TensorPair& pair,
            std::uint64_t input,
            std::uint64_t output,
            const char* name) {
            if (pair.manifest->
                    dimensions.size() < 2 ||
                pair.manifest->
                    dimensions[0] !=
                    input ||
                pair.manifest->
                    dimensions[1] !=
                    output) {
                throw std::invalid_argument(
                    std::string(name) +
                    " GDN tensor geometry mismatch");
            }
        };
    const auto expect_vector =
        [&](const TensorPair& pair,
            std::uint64_t elements,
            const char* name) {
            if (pair.manifest->
                    dimensions.empty() ||
                pair.manifest->
                    dimensions[0] !=
                    elements) {
                throw std::invalid_argument(
                    std::string(name) +
                    " GDN vector geometry mismatch");
            }
        };

    expect_matrix(
        result.qkv,
        hidden,
        conv_width,
        "qkv");
    expect_matrix(
        result.z,
        hidden,
        value_width,
        "z");
    expect_matrix(
        result.beta,
        hidden,
        config.model.gdn_value_heads,
        "beta");
    expect_matrix(
        result.alpha,
        hidden,
        config.model.gdn_value_heads,
        "alpha");
    expect_matrix(
        result.conv,
        config.model.gdn_conv_kernel,
        conv_width,
        "conv");
    expect_vector(
        result.dt,
        config.model.gdn_value_heads,
        "dt");
    expect_vector(
        result.a,
        config.model.gdn_value_heads,
        "A");
    expect_vector(
        result.norm,
        config.model.gdn_value_head_dim,
        "norm");
    expect_matrix(
        result.out,
        value_width,
        hidden,
        "out");

    return result;
}

void validate_projection_tensor(
    const TensorPair& pair,
    std::uint32_t input_width,
    std::uint64_t output_rows,
    const char* name) {
    if (pair.runtime == nullptr ||
        pair.manifest == nullptr ||
        (pair.runtime->storage !=
             RuntimeTensorStorage::resident &&
         pair.runtime->storage !=
             RuntimeTensorStorage::repacked_cache) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN projection is not GPU-visible");
    }
    const auto required =
        required_weight_bytes(
            weight_kind(
                pair.manifest->
                    ggml_type),
            input_width,
            output_rows);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN projection binding is undersized");
    }
}

void validate_small_tensor(
    const TensorPair& pair,
    std::uint64_t elements,
    const char* name) {
    if (pair.runtime == nullptr ||
        pair.manifest == nullptr ||
        (pair.runtime->storage !=
             RuntimeTensorStorage::resident &&
         pair.runtime->storage !=
             RuntimeTensorStorage::repacked_cache) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN tensor is not GPU-visible");
    }
    const auto required =
        small_weight_bytes(
            small_weight_kind(
                pair.manifest->
                    ggml_type),
            elements);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN tensor binding is undersized");
    }
}

Workspace workspace_for(
    const ExecutionBufferAddress& temporary,
    const ModelContract& model) {
    const auto key_width =
        model.gdn_key_heads *
        model.gdn_key_head_dim;
    const auto value_width =
        model.gdn_value_heads *
        model.gdn_value_head_dim;
    const auto conv_width =
        key_width * 2ULL +
        value_width;
    const auto elements =
        conv_width +
        value_width +
        model.gdn_value_heads * 2ULL +
        value_width;

    if (temporary.address == 0 ||
        temporary.bytes <
            elements *
                sizeof(float)) {
        throw std::invalid_argument(
            "HIP GDN temporary workspace is undersized");
    }

    auto* base =
        reinterpret_cast<float*>(
            temporary.address);
    Workspace result;
    result.qkv = base;
    result.z =
        result.qkv +
        conv_width;
    result.beta =
        result.z +
        value_width;
    result.gate =
        result.beta +
        model.gdn_value_heads;
    result.core =
        result.gate +
        model.gdn_value_heads;
    result.elements =
        elements;
    return result;
}

void launch_gdn(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::gated_delta_net) {
        throw std::invalid_argument(
            "HIP GDN launcher attached to wrong operator");
    }
    if (!context.layer.has_value() ||
        context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP GDN expects layer id, one read/write and one temporary");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP GDN v1 supports only C1 one-row decode");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP GDN requires configuration");
    }

    const auto& config =
        *static_cast<
            HipGdnLauncherConfig*>(
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
    const auto workspace =
        workspace_for(
            temporary,
            config.model);
    const auto tensors =
        identify_tensors(
            context,
            config);

    const auto hidden =
        checked_u32(
            config.model.hidden_size,
            "hidden_size");
    const auto key_heads =
        checked_u32(
            config.model.gdn_key_heads,
            "gdn_key_heads");
    const auto value_heads =
        checked_u32(
            config.model.gdn_value_heads,
            "gdn_value_heads");
    const auto head_dim =
        checked_u32(
            config.model.gdn_key_head_dim,
            "gdn_head_dim");
    const auto key_width =
        checked_u32(
            config.model.gdn_key_heads *
                config.model.gdn_key_head_dim,
            "gdn_key_width");
    const auto value_width =
        checked_u32(
            config.model.gdn_value_heads *
                config.model.gdn_value_head_dim,
            "gdn_value_width");
    const auto conv_width =
        checked_u32(
            static_cast<std::uint64_t>(
                key_width) *
                2ULL +
            value_width,
            "gdn_conv_width");
    const auto kernel =
        checked_u32(
            config.model.gdn_conv_kernel,
            "gdn_conv_kernel");

    const auto hidden_bytes =
        static_cast<std::uint64_t>(
            hidden) *
        sizeof(std::uint16_t);
    if (input.bytes <
            hidden_bytes ||
        output.bytes <
            hidden_bytes) {
        throw std::invalid_argument(
            "HIP GDN hidden buffers are undersized");
    }

    validate_projection_tensor(
        tensors.qkv,
        hidden,
        conv_width,
        "qkv");
    validate_projection_tensor(
        tensors.z,
        hidden,
        value_width,
        "z");
    validate_projection_tensor(
        tensors.beta,
        hidden,
        value_heads,
        "beta");
    validate_projection_tensor(
        tensors.alpha,
        hidden,
        value_heads,
        "alpha");
    validate_projection_tensor(
        tensors.out,
        value_width,
        hidden,
        "out");
    validate_small_tensor(
        tensors.conv,
        static_cast<std::uint64_t>(
            conv_width) *
            kernel,
        "conv");
    validate_small_tensor(
        tensors.dt,
        value_heads,
        "dt");
    validate_small_tensor(
        tensors.a,
        value_heads,
        "A");
    validate_small_tensor(
        tensors.norm,
        head_dim,
        "norm");

    const auto matrix_state =
        resolve_dispatch_gdn_layer_state(
            context,
            config.model,
            SessionStateComponent::
                recurrent_matrix);
    const auto conv_state =
        resolve_dispatch_gdn_layer_state(
            context,
            config.model,
            SessionStateComponent::
                gdn_convolution);
    if (matrix_state.storage !=
            SessionStateStorage::device ||
        conv_state.storage !=
            SessionStateStorage::device ||
        matrix_state.bytes <
            gdn_matrix_state_bytes_per_layer(
                config.model) ||
        conv_state.bytes <
            gdn_conv_state_bytes_per_layer(
                config.model)) {
        throw std::invalid_argument(
            "HIP GDN persistent state bindings are invalid");
    }

    const auto* hidden_input =
        reinterpret_cast<
            const std::uint16_t*>(
                input.address);

    launch_projection_by_kind(
        weight_kind(
            tensors.qkv.manifest->
                ggml_type),
        hidden_input,
        *tensors.qkv.runtime,
        hidden,
        conv_width,
        workspace.qkv);
    launch_projection_by_kind(
        weight_kind(
            tensors.z.manifest->
                ggml_type),
        hidden_input,
        *tensors.z.runtime,
        hidden,
        value_width,
        workspace.z);
    launch_projection_by_kind(
        weight_kind(
            tensors.beta.manifest->
                ggml_type),
        hidden_input,
        *tensors.beta.runtime,
        hidden,
        value_heads,
        workspace.beta);
    launch_projection_by_kind(
        weight_kind(
            tensors.alpha.manifest->
                ggml_type),
        hidden_input,
        *tensors.alpha.runtime,
        hidden,
        value_heads,
        workspace.gate);

    const auto conv_blocks =
        (conv_width +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    hipLaunchKernelGGL(
        conv_update_kernel,
        dim3(conv_blocks),
        dim3(kProjectionThreads),
        0,
        0,
        workspace.qkv,
        reinterpret_cast<
            std::uint16_t*>(
                conv_state.address),
        reinterpret_cast<const void*>(
            tensors.conv.runtime->
                address),
        small_weight_kind(
            tensors.conv.manifest->
                ggml_type),
        conv_width,
        kernel);
    check_hip(
        hipGetLastError(),
        "HIP GDN causal convolution launch");

    const auto parameter_blocks =
        (value_heads +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    hipLaunchKernelGGL(
        transform_beta_gate_kernel,
        dim3(parameter_blocks),
        dim3(kProjectionThreads),
        0,
        0,
        workspace.beta,
        workspace.gate,
        reinterpret_cast<const void*>(
            tensors.dt.runtime->
                address),
        small_weight_kind(
            tensors.dt.manifest->
                ggml_type),
        reinterpret_cast<const void*>(
            tensors.a.runtime->
                address),
        small_weight_kind(
            tensors.a.manifest->
                ggml_type),
        value_heads);
    check_hip(
        hipGetLastError(),
        "HIP GDN beta/gate transform launch");

    hipLaunchKernelGGL(
        normalize_qk_kernel,
        dim3(key_heads * 2U),
        dim3(kRecurrentThreads),
        0,
        0,
        workspace.qkv,
        key_heads,
        head_dim,
        config.model.
            gdn_qk_l2_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP GDN q/k normalization launch");

    const auto tiles =
        head_dim /
        kValueTile;
    hipLaunchKernelGGL(
        recurrent_step_kernel,
        dim3(value_heads * tiles),
        dim3(kRecurrentThreads),
        0,
        0,
        workspace.qkv,
        workspace.beta,
        workspace.gate,
        reinterpret_cast<float*>(
            matrix_state.address),
        workspace.core,
        key_heads,
        value_heads,
        head_dim);
    check_hip(
        hipGetLastError(),
        "HIP GDN recurrent update launch");

    hipLaunchKernelGGL(
        gated_rms_norm_kernel,
        dim3(value_heads),
        dim3(kRecurrentThreads),
        0,
        0,
        workspace.core,
        workspace.z,
        reinterpret_cast<const void*>(
            tensors.norm.runtime->
                address),
        small_weight_kind(
            tensors.norm.manifest->
                ggml_type),
        value_heads,
        head_dim,
        config.model.
            rms_norm_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP GDN gated RMSNorm launch");

    launch_output_by_kind(
        weight_kind(
            tensors.out.manifest->
                ggml_type),
        workspace.core,
        *tensors.out.runtime,
        value_width,
        hidden,
        reinterpret_cast<
            std::uint16_t*>(
                output.address));
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config) {
    return
        "hip.gdn.bf16.h" +
        std::to_string(
            config.model.hidden_size) +
        ".kh" +
        std::to_string(
            config.model.gdn_key_heads) +
        ".vh" +
        std::to_string(
            config.model.gdn_value_heads) +
        ".d" +
        std::to_string(
            config.model.gdn_key_head_dim) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency);
}

std::uint64_t implementation_identity(
    const HipGdnLauncherConfig& config) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.gdn_key_heads);
    hash_u64(
        hash,
        config.model.gdn_key_head_dim);
    hash_u64(
        hash,
        config.model.gdn_value_heads);
    hash_u64(
        hash,
        config.model.gdn_value_head_dim);
    hash_u64(
        hash,
        config.model.gdn_conv_kernel);
    hash_float(
        hash,
        config.model.rms_norm_epsilon);
    hash_float(
        hash,
        config.model.gdn_qk_l2_epsilon);
    hash_u64(
        hash,
        config.model.gdn_output_gate_sigmoid
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_qk_l2norm
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_conv_silu
            ? 1ULL
            : 0ULL);
    return hash;
}

}  // namespace

void register_hip_gdn_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    validate_config(config);
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.model.hidden_size !=
            model.hidden_size ||
        config.model.gdn_key_heads !=
            model.gdn_key_heads ||
        config.model.gdn_key_head_dim !=
            model.gdn_key_head_dim ||
        config.model.gdn_value_heads !=
            model.gdn_value_heads ||
        config.model.gdn_value_head_dim !=
            model.gdn_value_head_dim ||
        config.model.gdn_conv_kernel !=
            model.gdn_conv_kernel ||
        config.model.rms_norm_epsilon !=
            model.rms_norm_epsilon ||
        config.model.gdn_qk_l2_epsilon !=
            model.gdn_qk_l2_epsilon ||
        config.model.gdn_output_gate_sigmoid !=
            model.gdn_output_gate_sigmoid ||
        config.model.gdn_qk_l2norm !=
            model.gdn_qk_l2norm ||
        config.model.gdn_conv_silu !=
            model.gdn_conv_silu) {
        throw std::invalid_argument(
            "HIP GDN config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP GDN v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::
                target_qualified ||
        qualification ==
            kernels::QualificationStatus::
                production ||
        qualification ==
            kernels::QualificationStatus::
                rejected) {
        throw std::invalid_argument(
            "unmeasured HIP GDN may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            recurrent;
    variant.phase =
        kernels::ExecutionPhase::
            decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::gated_delta_net,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kProjectionThreads;
    variant.resources.lds_bytes =
        kRecurrentThreads *
        kValueTile *
        sizeof(float);
    variant.resources.scratch_bytes =
        build_serial_execution_buffer_plan(
            model,
            ExecutionBufferRequest{
                1,
                2,
                4,
                4,
                4,
                4,
            }).gdn_workspace_bytes;
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
    variant.graph_safe = false;
    variant.input_layout =
        "row-major";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        "hip-gfx1151-gdn-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_gdn_launcher(
    std::string kernel_id_value,
    HipGdnLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP GDN launcher requires id and config");
    }
    validate_config(*config);
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_gdn,
        config,
    };
}

void HipGdnLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP GDN bundle requires valid compiled execution");
    }

    std::string registered_id;
    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::gated_delta_net) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP GDN step has no kernel binding");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_gdn_launcher(
                    registered_id,
                    &config_));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP GDN bundle requires one selected kernel id");
        }
    }
}

}  // namespace strix
