#include "strix/hip_launch_context.hpp"
#include "strix/hip_shared_expert_launcher.hpp"

#include "strix/dispatch_resources.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
constexpr std::uint64_t kLauncherTag =
    0x5348455850455254ULL;
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

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
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
    const HipSharedExpertLauncherConfig& config) {
    if (config.hidden_size == 0 ||
        config.intermediate_size == 0) {
        throw std::invalid_argument(
            "HIP shared-expert geometry must be non-zero");
    }
    shared_expert_prefetch::validate_layers(static_cast<std::uint64_t>(UINT32_MAX)+1U,config.prefetch_q8,config.prefetch_eligible_layers);
    if (config.activation !=
            HipActivationStorage::bf16 &&
        config.activation !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP shared-expert activation storage is unsupported");
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
    case 20: return WeightKind::iq4_nl;
    case 23: return WeightKind::iq4_xs;
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP shared expert supports F32, F16, BF16, Q8_0, IQ4_NL and IQ4_XS");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
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
    std::uint32_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    return static_cast<const float*>(weights)[
        static_cast<std::uint64_t>(row) *
            row_width +
        feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(weights);
    return __half2float(
        values[
            static_cast<std::uint64_t>(row) *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(weights);
    return bf16_to_float(
        values[
            static_cast<std::uint64_t>(row) *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
                blocks_per_row +
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
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4NLBlock*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
                blocks_per_row +
            feature / 32U];
    const auto within =
        feature % 32U;
    const auto packed =
        block.qs[within % 16U];
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
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
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

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input,
    std::uint64_t output) {
    switch (kind) {
    case WeightKind::f32:
        return
            static_cast<std::uint64_t>(
                input) *
            output * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                input) *
            output * 2ULL;
    case WeightKind::q8_0:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 shared-expert input width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                input / 32U) *
            output *
            sizeof(Q8Block);
    case WeightKind::iq4_nl:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "IQ4_NL shared-expert input width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                input / 32U) *
            output *
            sizeof(IQ4NLBlock);
    case WeightKind::iq4_xs:
        if (input % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS shared-expert input width must be divisible by 256");
        }
        return
            static_cast<std::uint64_t>(
                input / 256U) *
            output *
            sizeof(IQ4XSBlock);
    }
    throw std::logic_error(
        "unreachable shared-expert weight kind");
}

template <
    typename Input,
    WeightKind Kind>
__global__ void projection_to_fp32_kernel(
    const Input* input,
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
            load_activation(
                &input[feature]) *
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

__global__ void gated_activation_kernel(
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
    typename Input,
    WeightKind Kind>
__global__ void scalar_projection_kernel(
    const Input* input,
    const void* weights,
    std::uint32_t input_width,
    float* output) {
    __shared__ float scratch[kThreads];
    float sum = 0.0F;
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
                0,
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
        *output =
            1.0F /
            (1.0F +
             expf(-scratch[0]));
    }
}

template <
    typename Output,
    WeightKind Kind>
__global__ void down_projection_kernel(
    const float* intermediate,
    const float* shared_gate,
    const void* weights,
    std::uint32_t intermediate_size,
    std::uint32_t hidden_size,
    Output* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= hidden_size) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
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
        store_activation(
            &output[out],
            scratch[0] *
                *shared_gate);
    }
}

// Exact virtual T256 leaves. New kernels alone use this helper; all legacy
// device functions remain literal. Compiler contraction and denormal modes are
// inherited from this same translation unit and require linked ISA qualification.
template<typename Input,bool Prefetch>
__device__ __forceinline__ float shared_q8_virtual256_dot(const Input* input,const void* weights,std::uint32_t width,std::uint32_t row){
    const unsigned lane=threadIdx.x&31U;float partial[8]{};
    const auto* blocks=static_cast<const Q8Block*>(weights)+static_cast<std::uint64_t>(row)*(width/32U);
    for(std::uint64_t chunk=0;chunk<width;chunk+=256U){
        if constexpr(Prefetch){const auto base=chunk+8U*lane;if(base<width){const auto feature=static_cast<std::uint32_t>(base);const auto& block=blocks[feature/32U];Input values[8];std::int8_t codes[8];
#pragma unroll
            for(unsigned local=0;local<8;++local){values[local]=input[feature+local];codes[local]=block.values[feature%32U+local];}
            const float scale=__half2float(block.scale);
#pragma unroll
            for(unsigned local=0;local<8;++local)partial[local]+=load_activation(&values[local])*(scale*static_cast<float>(codes[local]));
        }}else{
#pragma unroll
            for(unsigned local=0;local<8;++local){const auto feature=chunk+8U*lane+local;if(feature<width)partial[local]+=load_activation(&input[feature])*load_weight<WeightKind::q8_0>(weights,width,row,static_cast<std::uint32_t>(feature));}
        }
    }
#pragma unroll
    for(unsigned distance=16;distance>0;distance/=2){
#pragma unroll
        for(unsigned local=0;local<8;++local){const float partner=__shfl_down(partial[local],distance,32);if(lane<distance)partial[local]+=partner;}
    }
    if(lane==0){partial[0]+=partial[4];partial[1]+=partial[5];partial[2]+=partial[6];partial[3]+=partial[7];partial[0]+=partial[2];partial[1]+=partial[3];partial[0]+=partial[1];}
    return partial[0];
}
template<typename Input,bool Prefetch,unsigned RowsPerBlock>
__global__ void shared_q8_virtual_projection_kernel(const Input* input,const void* weights,std::uint32_t width,std::uint32_t outputs,float* output){
    const auto row=static_cast<std::uint32_t>(blockIdx.x)*RowsPerBlock+threadIdx.x/32U;if(row>=outputs)return;
    const float sum=shared_q8_virtual256_dot<Input,Prefetch>(input,weights,width,row);if((threadIdx.x&31U)==0)output[row]=sum;
}
template<typename Output,bool Prefetch,unsigned RowsPerBlock>
__global__ void shared_q8_virtual_down_kernel(const float* input,const float* shared_gate,const void* weights,std::uint32_t width,std::uint32_t outputs,Output* output){
    const auto row=static_cast<std::uint32_t>(blockIdx.x)*RowsPerBlock+threadIdx.x/32U;if(row>=outputs)return;
    const float sum=shared_q8_virtual256_dot<float,Prefetch>(input,weights,width,row);if((threadIdx.x&31U)==0)store_activation(&output[row],sum * *shared_gate);
}

bool contains(
    std::string_view text,
    std::string_view needle) {
    return text.find(needle) !=
        std::string_view::npos;
}

TensorPair tensor_pair(
    const ExecutionDispatchContext& context,
    std::uint32_t tensor_index) {
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP shared expert requires model manifest");
    }
    if (tensor_index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "shared-expert tensor index exceeds manifest");
    }
    return TensorPair{
        &context.resources->
            model_manifest->
            tensors[tensor_index],
        &resolve_dispatch_tensor(
            context,
            tensor_index),
    };
}

struct SharedTensors {
    TensorPair gate;
    TensorPair up;
    TensorPair down;
    TensorPair shared_gate;
};

SharedTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const ExecutionTensorAccess& access,
    const HipSharedExpertLauncherConfig& config) {
    SharedTensors result;
    std::uint32_t gate_matches = 0;
    std::uint32_t up_matches = 0;
    std::uint32_t down_matches = 0;
    std::uint32_t scalar_matches = 0;

    for (const auto index :
         access.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::dense_projection) {
            continue;
        }

        const bool scalar_geometry =
            tensor.dimensions.size() >= 1 &&
            tensor.dimensions[0] ==
                config.hidden_size &&
            (tensor.dimensions.size() == 1 ||
             (tensor.dimensions.size() >= 2 &&
              tensor.dimensions[1] == 1));
        if (scalar_geometry &&
            (contains(
                 tensor.name,
                 "gate_inp") ||
             contains(
                 tensor.name,
                 "shared_expert_gate") ||
             contains(
                 tensor.name,
                 "shexp_gate_inp"))) {
            result.shared_gate = pair;
            ++scalar_matches;
            continue;
        }

        if (tensor.dimensions.size() < 2) {
            continue;
        }

        if (tensor.dimensions[0] ==
                config.intermediate_size &&
            tensor.dimensions[1] ==
                config.hidden_size &&
            contains(
                tensor.name,
                "down")) {
            result.down = pair;
            ++down_matches;
            continue;
        }

        if (tensor.dimensions[0] ==
                config.hidden_size &&
            tensor.dimensions[1] ==
                config.intermediate_size) {
            if (contains(
                    tensor.name,
                    "up")) {
                result.up = pair;
                ++up_matches;
                continue;
            }
            if (contains(
                    tensor.name,
                    "gate")) {
                result.gate = pair;
                ++gate_matches;
                continue;
            }
        }
    }

    if (gate_matches != 1 ||
        up_matches != 1 ||
        down_matches != 1 ||
        scalar_matches != 1) {
        throw std::invalid_argument(
            "shared-expert tensor family is missing or ambiguous");
    }
    return result;
}

void validate_tensor(
    const TensorPair& pair,
    std::uint32_t input,
    std::uint64_t output,
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
    const auto kind =
        weight_kind(
            pair.manifest->ggml_type);
    const auto required =
        required_weight_bytes(
            kind,
            input,
            output);
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
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output,bool prefetch=false,shared_expert_prefetch::RuntimeDispatchStats* stats=nullptr) {
    if(prefetch && kind==WeightKind::q8_0 && input_width%32U==0){hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_projection_kernel<Input,true,8>),dim3(output_width/8U+(output_width%8U!=0U)),dim3(256),0,hip_execution_stream(),input,reinterpret_cast<const void*>(weights.address),input_width,output_width,output);check_hip(hipGetLastError(),"HIP shared prefetch8 gate/up launch");if(stats)stats->dispatched_projections.fetch_add(1,std::memory_order_relaxed);return;}
#define STRIX_LAUNCH_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             projection_to_fp32_kernel<                 Input, WeightKind::kind_name>),         dim3(output_width),         dim3(kThreads),         0, hip_execution_stream(),         input,         reinterpret_cast<const void*>(weights.address),         input_width,         output_width,         output)
    switch (kind) {
    case WeightKind::f32: STRIX_LAUNCH_CASE(f32); break;
    case WeightKind::f16: STRIX_LAUNCH_CASE(f16); break;
    case WeightKind::bf16: STRIX_LAUNCH_CASE(bf16); break;
    case WeightKind::q8_0: STRIX_LAUNCH_CASE(q8_0); break;
    case WeightKind::iq4_nl: STRIX_LAUNCH_CASE(iq4_nl); break;
    case WeightKind::iq4_xs: STRIX_LAUNCH_CASE(iq4_xs); break;
    }
#undef STRIX_LAUNCH_CASE
    check_hip(
        hipGetLastError(),
        "HIP shared-expert projection launch");
}

template <typename Input>
void launch_scalar_by_kind(
    WeightKind kind,
    const Input* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    float* output) {
#define STRIX_SCALAR_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             scalar_projection_kernel<                 Input, WeightKind::kind_name>),         dim3(1), dim3(kThreads), 0, hip_execution_stream(),         input,         reinterpret_cast<const void*>(weights.address),         input_width, output)
    switch (kind) {
    case WeightKind::f32: STRIX_SCALAR_CASE(f32); break;
    case WeightKind::f16: STRIX_SCALAR_CASE(f16); break;
    case WeightKind::bf16: STRIX_SCALAR_CASE(bf16); break;
    case WeightKind::q8_0: STRIX_SCALAR_CASE(q8_0); break;
    case WeightKind::iq4_nl: STRIX_SCALAR_CASE(iq4_nl); break;
    case WeightKind::iq4_xs: STRIX_SCALAR_CASE(iq4_xs); break;
    }
#undef STRIX_SCALAR_CASE
    check_hip(
        hipGetLastError(),
        "HIP shared-expert gate launch");
}

template <typename Output>
void launch_down_by_kind(
    WeightKind kind,
    const float* intermediate,
    const float* shared_gate,
    const RuntimeTensorBinding& weights,
    std::uint32_t intermediate_size,
    std::uint32_t hidden_size,
    Output* output,bool prefetch=false,shared_expert_prefetch::RuntimeDispatchStats* stats=nullptr) {
    if(prefetch && kind==WeightKind::q8_0 && intermediate_size%32U==0){hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_down_kernel<Output,true,8>),dim3(hidden_size/8U+(hidden_size%8U!=0U)),dim3(256),0,hip_execution_stream(),intermediate,shared_gate,reinterpret_cast<const void*>(weights.address),intermediate_size,hidden_size,output);check_hip(hipGetLastError(),"HIP shared prefetch8 down launch");if(stats)stats->dispatched_projections.fetch_add(1,std::memory_order_relaxed);return;}
#define STRIX_DOWN_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             down_projection_kernel<                 Output, WeightKind::kind_name>),         dim3(hidden_size),         dim3(kThreads),         0, hip_execution_stream(),         intermediate,         shared_gate,         reinterpret_cast<const void*>(weights.address),         intermediate_size,         hidden_size,         output)
    switch (kind) {
    case WeightKind::f32: STRIX_DOWN_CASE(f32); break;
    case WeightKind::f16: STRIX_DOWN_CASE(f16); break;
    case WeightKind::bf16: STRIX_DOWN_CASE(bf16); break;
    case WeightKind::q8_0: STRIX_DOWN_CASE(q8_0); break;
    case WeightKind::iq4_nl: STRIX_DOWN_CASE(iq4_nl); break;
    case WeightKind::iq4_xs: STRIX_DOWN_CASE(iq4_xs); break;
    }
#undef STRIX_DOWN_CASE
    check_hip(
        hipGetLastError(),
        "HIP shared-expert down launch");
}

template <typename Storage>
void launch_typed(
    const HipSharedExpertLauncherConfig& config,
    const ExecutionBufferAddress& input,
    const ExecutionBufferAddress& output,
    const ExecutionBufferAddress& temporary,
    const SharedTensors& tensors) {
    auto* workspace =
        reinterpret_cast<float*>(
            temporary.address);
    auto* gate =
        workspace;
    auto* up =
        workspace +
        config.intermediate_size;
    auto* shared_gate =
        up;

    const auto* hidden =
        reinterpret_cast<const Storage*>(
            input.address);

    launch_projection_by_kind(
        weight_kind(
            tensors.gate.manifest->
                ggml_type),
        hidden,
        *tensors.gate.runtime,
        config.hidden_size,
        config.intermediate_size,
        gate,config.prefetch_q8 && shared_expert_prefetch::eligible(*tensors.gate.manifest),config.prefetch_stats.get());

    launch_projection_by_kind(
        weight_kind(
            tensors.up.manifest->
                ggml_type),
        hidden,
        *tensors.up.runtime,
        config.hidden_size,
        config.intermediate_size,
        up,config.prefetch_q8 && shared_expert_prefetch::eligible(*tensors.up.manifest),config.prefetch_stats.get());

    const auto blocks =
        (config.intermediate_size +
         kThreads - 1U) /
        kThreads;
    hipLaunchKernelGGL(
        gated_activation_kernel,
        dim3(blocks),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace,
        config.intermediate_size);
    check_hip(
        hipGetLastError(),
        "HIP shared-expert activation launch");

    launch_scalar_by_kind(
        weight_kind(
            tensors.shared_gate.
                manifest->ggml_type),
        hidden,
        *tensors.shared_gate.runtime,
        config.hidden_size,
        shared_gate);

    launch_down_by_kind(
        weight_kind(
            tensors.down.manifest->
                ggml_type),
        gate,
        shared_gate,
        *tensors.down.runtime,
        config.intermediate_size,
        config.hidden_size,
        reinterpret_cast<Storage*>(
            output.address),config.prefetch_q8 && shared_expert_prefetch::eligible(*tensors.down.manifest),config.prefetch_stats.get());
}

void launch_shared_expert(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::shared_expert) {
        throw std::invalid_argument(
            "HIP shared-expert launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP shared expert expects one read, one write and one temporary");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP shared-expert v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP shared expert requires configuration");
    }

    const auto& config =
        *static_cast<
            HipSharedExpertLauncherConfig*>(
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

    const auto activation_bytes =
        static_cast<std::uint32_t>(
            config.activation);
    const auto hidden_required =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        activation_bytes;
    const auto temp_required =
        static_cast<std::uint64_t>(
            config.intermediate_size) *
        2ULL *
        sizeof(float);

    if (input.bytes <
            hidden_required ||
        output.bytes <
            hidden_required ||
        temporary.bytes <
            temp_required) {
        throw std::invalid_argument(
            "HIP shared-expert buffers are undersized");
    }

    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto tensors =
        identify_tensors(
            context,
            access,
            config);

    validate_tensor(
        tensors.gate,
        config.hidden_size,
        config.intermediate_size,
        "shared gate projection");
    validate_tensor(
        tensors.up,
        config.hidden_size,
        config.intermediate_size,
        "shared up projection");
    validate_tensor(
        tensors.down,
        config.intermediate_size,
        config.hidden_size,
        "shared down projection");
    validate_tensor(
        tensors.shared_gate,
        config.hidden_size,
        1,
        "shared expert gate");

    if(config.prefetch_q8){
        if(!shared_expert_prefetch::selected(true,config.prefetch_eligible_layers,context.layer))throw std::invalid_argument("shared prefetch layer differs from admitted plan");
        unsigned eligible=0;for(const auto* tensor:{&tensors.gate,&tensors.up,&tensors.down})if(shared_expert_prefetch::eligible(*tensor->manifest)){++eligible;const auto required=shared_expert_prefetch::required_weight_bytes(tensor->manifest->dimensions[0],tensor->manifest->dimensions[1]);if(tensor->manifest->absolute_offset%16U!=0 || tensor->runtime->address%16U!=0 || tensor->manifest->byte_size<required || tensor->runtime->bytes<required)throw std::invalid_argument("shared prefetch weight alignment/extent invalid");}
        if(eligible==0)throw std::invalid_argument("shared prefetch step has no eligible matrix");
        struct Range{std::uintptr_t address;std::uint64_t bytes;};
        std::array<Range,5> reads{};reads[0]={input.address,input.bytes};unsigned index=1;for(const auto* tensor:{&tensors.gate,&tensors.up,&tensors.down,&tensors.shared_gate})reads[index++]={tensor->runtime->address,tensor->runtime->bytes};
        const std::array<Range,2> writes{{{temporary.address,temporary.bytes},{output.address,output.bytes}}};
        const auto valid=[](Range v){return v.address!=0 && v.bytes!=0 && v.bytes<=UINTPTR_MAX-v.address;};const auto overlap=[](Range a,Range b){return a.address<b.address+b.bytes && b.address<a.address+a.bytes;};
        if(input.address%activation_bytes || output.address%activation_bytes || temporary.address%4U)throw std::invalid_argument("shared prefetch activation owner alignment invalid");
        for(auto v:reads)if(!valid(v))throw std::invalid_argument("shared prefetch read owner invalid");
        for(unsigned i=0;i<writes.size();++i){if(!valid(writes[i]))throw std::invalid_argument("shared prefetch write owner invalid");for(auto v:reads)if(overlap(writes[i],v))throw std::invalid_argument("shared prefetch writable/read alias");for(unsigned j=0;j<i;++j)if(overlap(writes[i],writes[j]))throw std::invalid_argument("shared prefetch writable alias");}
    }

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_typed<std::uint16_t>(
            config,
            input,
            output,
            temporary,
            tensors);
        return;
    case HipActivationStorage::fp32:
        launch_typed<float>(
            config,
            input,
            output,
            temporary,
            tensors);
        return;
    }
    throw std::logic_error(
        "unreachable shared-expert activation storage");
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipSharedExpertLauncherConfig& config) {
    return
        "hip.shared_expert." +
        std::string(
            storage_name(
                config.activation)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".m" +
        std::to_string(
            config.intermediate_size) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (config.prefetch_q8?std::string(shared_expert_prefetch::kernel_suffix):"");
}

std::uint64_t implementation_identity(
    const HipSharedExpertLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.intermediate_size);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    if(config.prefetch_q8){hash_u64(hash,0x5348505246563031ULL);for(auto layer:config.prefetch_eligible_layers)hash_u64(hash,layer);}
    return hash;
}

}  // namespace

namespace {
template<typename Storage>
void shared_projection_probe_typed(HipSharedQ8ProjectionRole role,HipSharedQ8ProjectionMode mode,HipSharedQ8ProjectionView input,HipSharedQ8ProjectionView weights,HipSharedQ8ProjectionView gate,HipSharedQ8ProjectionView output,unsigned width,unsigned outputs,hipStream_t stream){
    const auto* w=reinterpret_cast<const void*>(weights.address);const auto* g=reinterpret_cast<const float*>(gate.address);
    if(role==HipSharedQ8ProjectionRole::gate_up){const auto* in=reinterpret_cast<const Storage*>(input.address);auto* out=reinterpret_cast<float*>(output.address);
        if(mode==HipSharedQ8ProjectionMode::legacy256)hipLaunchKernelGGL(HIP_KERNEL_NAME(projection_to_fp32_kernel<Storage,WeightKind::q8_0>),dim3(outputs),dim3(256),0,stream,in,w,width,outputs,out);
        else if(mode==HipSharedQ8ProjectionMode::virtual256)hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_projection_kernel<Storage,false,1>),dim3(outputs),dim3(32),0,stream,in,w,width,outputs,out);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_projection_kernel<Storage,true,8>),dim3(outputs/8U+(outputs%8U!=0U)),dim3(256),0,stream,in,w,width,outputs,out);
    }else{const auto* in=reinterpret_cast<const float*>(input.address);auto* out=reinterpret_cast<Storage*>(output.address);
        if(mode==HipSharedQ8ProjectionMode::legacy256)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_projection_kernel<Storage,WeightKind::q8_0>),dim3(outputs),dim3(256),0,stream,in,g,w,width,outputs,out);
        else if(mode==HipSharedQ8ProjectionMode::virtual256)hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_down_kernel<Storage,false,1>),dim3(outputs),dim3(32),0,stream,in,g,w,width,outputs,out);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_down_kernel<Storage,true,8>),dim3(outputs/8U+(outputs%8U!=0U)),dim3(256),0,stream,in,g,w,width,outputs,out);
    }
}
}
void launch_hip_shared_q8_projection_probe(HipSharedQ8ProjectionRole role,HipSharedQ8ProjectionMode mode,HipActivationStorage storage,HipSharedQ8ProjectionView input,HipSharedQ8ProjectionView weights,HipSharedQ8ProjectionView gate,HipSharedQ8ProjectionView output,std::uint32_t width,std::uint32_t outputs,void* stream){
    const bool down=role==HipSharedQ8ProjectionRole::down;const auto weight_bytes=shared_expert_prefetch::required_weight_bytes(width,outputs);const auto element_bytes=static_cast<std::uint64_t>(storage);
    const auto valid=[](HipSharedQ8ProjectionView v,std::uint64_t n){return v.address!=0 && v.address%16U==0 && v.bytes==n && v.bytes<=UINTPTR_MAX-v.address;};
    if((role!=HipSharedQ8ProjectionRole::gate_up && !down) || (mode!=HipSharedQ8ProjectionMode::legacy256 && mode!=HipSharedQ8ProjectionMode::virtual256 && mode!=HipSharedQ8ProjectionMode::prefetch8) || (storage!=HipActivationStorage::bf16 && storage!=HipActivationStorage::fp32) || !valid(input,static_cast<std::uint64_t>(width)*(down?4U:element_bytes)) || !valid(weights,weight_bytes) || !valid(output,static_cast<std::uint64_t>(outputs)*(down?element_bytes:4U)) || (down?!valid(gate,4U):(gate.address!=0 || gate.bytes!=0)))throw std::invalid_argument("shared projection probe exact views/role invalid");
    const auto overlap=[](HipSharedQ8ProjectionView x,HipSharedQ8ProjectionView y){return x.address<y.address+y.bytes && y.address<x.address+x.bytes;};if(overlap(output,input)||overlap(output,weights)||(down && overlap(output,gate)))throw std::invalid_argument("shared projection probe writable alias");
    check_hip(hipPeekAtLastError(),"shared projection probe stale error");
    if(storage==HipActivationStorage::bf16)shared_projection_probe_typed<std::uint16_t>(role,mode,input,weights,gate,output,width,outputs,static_cast<hipStream_t>(stream));else shared_projection_probe_typed<float>(role,mode,input,weights,gate,output,width,outputs,static_cast<hipStream_t>(stream));check_hip(hipGetLastError(),"shared projection probe launch");
}

void register_hip_shared_expert_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipSharedExpertLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if(config.prefetch_q8!=request.prefetch_q8_shared_expert || config.prefetch_eligible_layers!=request.shared_expert_prefetch_layers)throw std::invalid_argument("shared prefetch config/request mismatch");
    shared_expert_prefetch::validate_layers(model.hidden_layers,config.prefetch_q8,config.prefetch_eligible_layers);
    if(config.prefetch_q8 && qualification!=kernels::QualificationStatus::experimental)throw std::invalid_argument("shared prefetch variant remains experimental");
    if (config.hidden_size !=
            model.hidden_size ||
        config.intermediate_size !=
            model.shared_expert_intermediate_size) {
        throw std::invalid_argument(
            "HIP shared-expert config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP shared-expert v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP shared expert may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::shared_expert;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::shared_expert,
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
        "row-major";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        config.prefetch_q8?"hip-gfx1151-shared-expert-v2-prefetch8-virtual256-q8":"hip-gfx1151-shared-expert-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_shared_expert_launcher(
    std::string kernel_id_value,
    HipSharedExpertLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP shared-expert launcher requires id and config");
    }
    validate_config(*config);
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_shared_expert,
        config,
    };
}

void HipSharedExpertLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP shared-expert bundle requires valid compiled execution");
    }

    ExecutionBindingRequest shape;shape.logical_rows=compiled.logical_rows;shape.request_concurrency=compiled.request_concurrency;
    if(config_.prefetch_q8 && std::none_of(compiled.bindings.bindings.begin(),compiled.bindings.bindings.end(),
        [](const auto& binding){return binding.op==SerialOp::shared_expert && binding.kernel_id &&
            binding.kernel_id->ends_with(shared_expert_prefetch::kernel_suffix);}))
        throw std::invalid_argument("shared prefetch bundle has no eligible step");
    std::vector<std::string> registered;bool saw_prefetch=false;
    for(std::size_t i=0;i<compiled.execution.steps.size();++i){if(compiled.execution.steps[i].op!=SerialOp::shared_expert)continue;const auto& binding=compiled.bindings.bindings[i];if(binding.host_or_runtime_step || !binding.kernel_id || binding.kernel_id->empty())throw std::invalid_argument("HIP shared step has no kernel binding");const bool prefetch=binding.kernel_id->ends_with(shared_expert_prefetch::kernel_suffix);if(prefetch!=shared_expert_prefetch::selected(config_.prefetch_q8,config_.prefetch_eligible_layers,compiled.execution.steps[i].layer))throw std::invalid_argument("shared prefetch bundle layer/config mismatch");auto* selected=prefetch?&config_:&fallback_config_;if(*binding.kernel_id!=kernel_id(shape,*selected))throw std::invalid_argument("HIP shared bundle permits only its base/prefetch pair");if(std::find(registered.begin(),registered.end(),*binding.kernel_id)==registered.end()){registry.add_kernel_launcher(make_hip_shared_expert_launcher(*binding.kernel_id,selected));registered.push_back(*binding.kernel_id);}saw_prefetch|=prefetch;}
    if(config_.prefetch_q8 && !saw_prefetch)throw std::invalid_argument("shared prefetch bundle has no eligible step");
}

}  // namespace strix
