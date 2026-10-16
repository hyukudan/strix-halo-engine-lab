#include "strix/hip_mtp_shared_expert.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_shared_serving_cohort.hpp"
#include "strix/shared_expert_rows_diagnostic_contract.hpp"
#if defined(STRIX_BUILD_SHARED_EXPERT_ROWS_DIAGNOSTIC)
#include "strix/hip_shared_expert_rows_diagnostic.hpp"
#endif

#include "strix/dispatch_resources.hpp"
#include "strix/prepared_tensor_views.hpp"

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
// The legacy LDS tree emits upper/partner as v_add_f32 src0 and lower/self as
// src1. Commuting them preserves finite sums but can select a different NaN
// payload. Keep that actual operand order in every new virtual-tree addition.
__device__ __forceinline__ float shared_q8_legacy_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(upper), "v"(lower));
    return result;
}

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
        for(unsigned local=0;local<8;++local){const float partner=__shfl_down(partial[local],distance,32);if(lane<distance)partial[local]=shared_q8_legacy_tree_add(partial[local],partner);}
    }
    if(lane==0){partial[0]=shared_q8_legacy_tree_add(partial[0],partial[4]);partial[1]=shared_q8_legacy_tree_add(partial[1],partial[5]);partial[2]=shared_q8_legacy_tree_add(partial[2],partial[6]);partial[3]=shared_q8_legacy_tree_add(partial[3],partial[7]);partial[0]=shared_q8_legacy_tree_add(partial[0],partial[2]);partial[1]=shared_q8_legacy_tree_add(partial[1],partial[3]);partial[0]=shared_q8_legacy_tree_add(partial[0],partial[1]);}
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
    const float sum=shared_q8_virtual256_dot<float,Prefetch>(input,weights,width,row);if((threadIdx.x&31U)==0){
        // Match the legacy emitted multiply: gate src0, reduction src1.
        // Both operands may be NaNs, so commuting them can change payload bits.
        float gated;
        asm volatile("v_mul_f32 %0, %1, %2" : "=v"(gated) : "v"(*shared_gate), "v"(sum));
        store_activation(&output[row],gated);
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
    if(const auto* cached=context.prepared_tensor_views) {
        if(!context.resources) throw std::invalid_argument("prepared shared roles require retained resources");
        cached->validate_stage_scope(context.resources->distributed_stage,context.step_id);
        cached->validate_source_handles(context.resources->model_manifest,context.resources->model_weights,context.resources->tensor_access);
        const auto& roles=cached->shared_roles(context.execution_identity,context.step_id,config.hidden_size,config.intermediate_size);
        return {tensor_pair(context,roles.gate),tensor_pair(context,roles.up),tensor_pair(context,roles.down),tensor_pair(context,roles.shared_gate)};
    }
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

struct PreparedSharedExpert {
    const HipSharedExpertLauncherConfig& config;
    ExecutionBufferAddress input, output, temporary;
    SharedTensors tensors;
};
PreparedSharedExpert prepare_shared_expert(
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

    return {config, input, output, temporary, tensors};
}

void preflight_shared_expert(const ExecutionDispatchContext& context, void* user_data) {
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window launcher preflight requires eager scope");
    static_cast<void>(prepare_shared_expert(context, user_data));
}

void launch_shared_expert(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const auto prepared = prepare_shared_expert(context, user_data);
    const auto& config = prepared.config;
    const auto& input = prepared.input;
    const auto& output = prepared.output;
    const auto& temporary = prepared.temporary;
    const auto& tensors = prepared.tensors;

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
        &preflight_shared_expert,
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

namespace strix {
namespace {
// Each wave handles one output feature and all Rows independent input vectors.
// A Q8 block's eight codes and scale are loaded/decoded once per lane/chunk,
// then reused across Rows accumulators. No cross-row arithmetic/reduction.
template <typename Input, unsigned Rows>
__device__ __forceinline__ void shared_rows_q8_dot(
    const Input* input, std::uint64_t input_stride, const void* weights,
    std::uint32_t width, std::uint32_t out, float (&result)[Rows]) {
    const unsigned lane = threadIdx.x & 31U;
    float partial[Rows][8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(out) * (width / 32U);
    for (std::uint64_t chunk = 0; chunk < width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        if (base < width) {
            const auto& block = blocks[base / 32U];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local)
                codes[local] = block.values[base % 32U + local];
            const float scale = __half2float(block.scale);
            float coefficient[8];
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local)
                coefficient[local] = scale * static_cast<float>(codes[local]);
#pragma unroll
            for (unsigned row = 0; row < Rows; ++row) {
                Input values[8];
#pragma unroll
                for (unsigned local = 0; local < 8U; ++local)
                    values[local] = input[row * input_stride + base + local];
#pragma unroll
                for (unsigned local = 0; local < 8U; ++local)
                    partial[row][local] += load_activation(&values[local]) * coefficient[local];
            }
        }
    }
#pragma unroll
    for (unsigned distance = 16U; distance; distance /= 2U) {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local) {
                const float partner = __shfl_down(partial[row][local], distance, 32);
                if (lane < distance)
                    partial[row][local] = shared_q8_legacy_tree_add(partial[row][local], partner);
            }
        }
    }
    if (lane == 0U) {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            auto* p = partial[row];
            p[0] = shared_q8_legacy_tree_add(p[0], p[4]);
            p[1] = shared_q8_legacy_tree_add(p[1], p[5]);
            p[2] = shared_q8_legacy_tree_add(p[2], p[6]);
            p[3] = shared_q8_legacy_tree_add(p[3], p[7]);
            p[0] = shared_q8_legacy_tree_add(p[0], p[2]);
            p[1] = shared_q8_legacy_tree_add(p[1], p[3]);
            result[row] = shared_q8_legacy_tree_add(p[0], p[1]);
        }
    }
}
template <unsigned Rows, bool Down>
__global__ void shared_rows_q8_projection_kernel(
    const void* input, std::uint64_t input_stride_bytes, const void* weights,
    std::uint32_t width, std::uint32_t outputs, void* output,
    std::uint64_t output_stride_bytes, const float* shared_gate,
    std::uint64_t gate_stride_floats) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * 8U + threadIdx.x / 32U;
    if (out >= outputs) return; // wave-uniform; kernel has no block barriers
    float sums[Rows];
    if constexpr (Down)
        shared_rows_q8_dot<float, Rows>(static_cast<const float*>(input), input_stride_bytes / 4U,
                                       weights, width, out, sums);
    else
        shared_rows_q8_dot<std::uint16_t, Rows>(static_cast<const std::uint16_t*>(input), input_stride_bytes / 2U,
                                               weights, width, out, sums);
    if ((threadIdx.x & 31U) == 0U) {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            auto* destination = static_cast<unsigned char*>(output) + row * output_stride_bytes;
            if constexpr (Down) {
                float gated;
                asm volatile("v_mul_f32 %0, %1, %2" : "=v"(gated)
                             : "v"(shared_gate[row * gate_stride_floats]), "v"(sums[row]));
                store_activation(reinterpret_cast<std::uint16_t*>(destination) + out, gated);
            } else reinterpret_cast<float*>(destination)[out] = sums[row];
        }
    }
}
} // namespace
} // namespace strix

#if defined(STRIX_BUILD_SHARED_EXPERT_ROWS_DIAGNOSTIC)
namespace strix {
namespace {
template <unsigned Rows>
void enqueue_shared_rows_projection(bool down, shared_expert_rows_diagnostic::Shape s,
    shared_expert_rows_diagnostic::Span input, shared_expert_rows_diagnostic::Span weights,
    shared_expert_rows_diagnostic::Span workspace, shared_expert_rows_diagnostic::Span output,
    hipStream_t stream) {
    const auto* gate = reinterpret_cast<const float*>(workspace.address) + s.intermediate;
    if (down) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_rows_q8_projection_kernel<Rows, true>),
            dim3((s.hidden + 7U) / 8U), dim3(256), 0, stream,
            reinterpret_cast<const void*>(workspace.address), s.workspace_stride_bytes,
            reinterpret_cast<const void*>(weights.address), s.intermediate, s.hidden,
            reinterpret_cast<void*>(output.address), s.output_stride_bytes, gate, s.workspace_stride_bytes / 4U);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_rows_q8_projection_kernel<Rows, false>),
            dim3((s.intermediate + 7U) / 8U), dim3(256), 0, stream,
            reinterpret_cast<const void*>(input.address), s.input_stride_bytes,
            reinterpret_cast<const void*>(weights.address), s.hidden, s.intermediate,
            reinterpret_cast<void*>(output.address), s.workspace_stride_bytes, nullptr, 0ULL);
    }
    check_hip(hipGetLastError(), "shared multirow Q8 projection launch");
}
} // namespace

namespace shared_expert_rows_diagnostic {
const void* projection_symbol(Arm arm, std::uint32_t rows, bool down) {
    if (rows != 2U && rows != 4U && rows != 8U)
        throw std::invalid_argument("shared rows symbol geometry is invalid");
    if (arm == Arm::reference_rows) {
        if (down) return reinterpret_cast<const void*>(shared_q8_virtual_down_kernel<std::uint16_t, true, 8>);
        return reinterpret_cast<const void*>(shared_q8_virtual_projection_kernel<std::uint16_t, true, 8>);
    }
    if (arm != Arm::shared_q8_rows) throw std::invalid_argument("shared rows symbol arm is invalid");
#define STRIX_SHARED_ROWS_SYMBOL(R) \
    if (rows == R) { \
        if (down) return reinterpret_cast<const void*>(shared_rows_q8_projection_kernel<R, true>); \
        return reinterpret_cast<const void*>(shared_rows_q8_projection_kernel<R, false>); \
    }
    STRIX_SHARED_ROWS_SYMBOL(2U)
    STRIX_SHARED_ROWS_SYMBOL(4U)
    STRIX_SHARED_ROWS_SYMBOL(8U)
#undef STRIX_SHARED_ROWS_SYMBOL
    throw std::logic_error("unreachable shared rows symbol");
}
void enqueue(Shape s, Span input, Weights weights, Span workspace, Span output, Arm arm, hipStream_t stream) {
    (void)validate(s, input, weights, workspace, output, arm);
    if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread)
        throw std::invalid_argument("shared rows diagnostic requires explicit owned stream");
    check_hip(hipPeekAtLastError(), "shared rows diagnostic stale launch error");
    const auto candidate_projection = [&](bool down, Span weight, Span target) {
        switch (s.rows) {
        case 2: enqueue_shared_rows_projection<2>(down, s, input, weight, workspace, target, stream); break;
        case 4: enqueue_shared_rows_projection<4>(down, s, input, weight, workspace, target, stream); break;
        case 8: enqueue_shared_rows_projection<8>(down, s, input, weight, workspace, target, stream); break;
        default: throw std::logic_error("unreachable shared rows launch");
        }
    };
    if (arm == Arm::shared_q8_rows) {
        candidate_projection(false, weights.gate_q8, workspace);
        auto up = workspace; up.address += static_cast<std::uint64_t>(s.intermediate) * 4U;
        candidate_projection(false, weights.up_q8, up);
    }
    for (std::uint32_t row = 0; row < s.rows; ++row) {
        const auto* hidden = reinterpret_cast<const std::uint16_t*>(input.address + row * s.input_stride_bytes);
        auto* gate = reinterpret_cast<float*>(workspace.address + row * s.workspace_stride_bytes);
        auto* up = gate + s.intermediate;
        if (arm == Arm::reference_rows) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_projection_kernel<std::uint16_t, true, 8>),
                dim3((s.intermediate + 7U) / 8U), dim3(256), 0, stream,
                hidden, reinterpret_cast<const void*>(weights.gate_q8.address), s.hidden, s.intermediate, gate);
            check_hip(hipGetLastError(), "shared rows reference gate launch");
            hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_projection_kernel<std::uint16_t, true, 8>),
                dim3((s.intermediate + 7U) / 8U), dim3(256), 0, stream,
                hidden, reinterpret_cast<const void*>(weights.up_q8.address), s.hidden, s.intermediate, up);
            check_hip(hipGetLastError(), "shared rows reference up launch");
        }
        hipLaunchKernelGGL(gated_activation_kernel, dim3((s.intermediate + 255U) / 256U),
            dim3(256), 0, stream, gate, s.intermediate);
        check_hip(hipGetLastError(), "shared rows activation launch");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(scalar_projection_kernel<std::uint16_t, WeightKind::f32>),
            dim3(1), dim3(256), 0, stream, hidden,
            reinterpret_cast<const void*>(weights.shared_gate_f32.address), s.hidden, up);
        check_hip(hipGetLastError(), "shared rows scalar launch");
        if (arm == Arm::reference_rows) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_q8_virtual_down_kernel<std::uint16_t, true, 8>),
                dim3((s.hidden + 7U) / 8U), dim3(256), 0, stream, gate, up,
                reinterpret_cast<const void*>(weights.down_q8.address), s.intermediate, s.hidden,
                reinterpret_cast<std::uint16_t*>(output.address + row * s.output_stride_bytes));
            check_hip(hipGetLastError(), "shared rows reference down launch");
        }
    }
    if (arm == Arm::shared_q8_rows) candidate_projection(true, weights.down_q8, output);
}
} // namespace shared_expert_rows_diagnostic
} // namespace strix
#endif

namespace strix {
namespace {
using KnownSharedSpan = shared_expert_rows_diagnostic::Span;
struct KnownSharedR2Views {
    shared_expert_rows_diagnostic::Shape shape;
    KnownSharedSpan input, workspace, output;
    shared_expert_rows_diagnostic::Weights weights;
};

bool known_shared_overlap(KnownSharedSpan left, KnownSharedSpan right) {
    return left.address < right.address + right.bytes && right.address < left.address + left.bytes;
}
void known_shared_range(KnownSharedSpan view) {
    if (!view.address || !view.bytes || view.bytes > UINTPTR_MAX - view.address)
        throw std::invalid_argument("known-window shared R2 owner range is invalid");
}
std::uint64_t known_shared_stride(const ExecutionBufferAddress& first,
                                  const ExecutionBufferAddress& second) {
    if (first.bytes != second.bytes || first.bytes > UINT64_MAX - 255U || second.address <= first.address)
        throw std::invalid_argument("known-window shared R2 row extent/order differs");
    const auto expected = (first.bytes + 255U) / 256U * 256U;
    const auto stride = static_cast<std::uint64_t>(second.address - first.address);
    if (stride != expected)
        throw std::invalid_argument("known-window shared R2 row stride differs from serial allocation");
    return stride;
}
const ExecutionMemoryBinding& known_shared_memory(const ExecutionDispatchContext& row) {
    if (!row.resources || !row.resources->scratch || !row.arena)
        throw std::invalid_argument("known-window shared R2 requires retained row resources");
    const auto& memory = *row.resources->scratch;
    validate_execution_memory_binding(row.execution_identity, *row.arena, memory);
    if (row.memory && row.memory != row.resources->scratch)
        throw std::invalid_argument("known-window shared R2 scratch aliases another memory binding");
    return memory;
}
KnownSharedR2Views known_shared_r2_views(std::span<const ExecutionDispatchContext, 2> rows,
                                       const HipSharedExpertLauncherConfig& config) {
    validate_config(config);
    if (config.activation != HipActivationStorage::bf16 || config.hidden_size != 2560U ||
        config.intermediate_size != 640U)
        throw std::invalid_argument("known-window shared R2 requires canonical BF16 geometry");
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window shared R2 is eager only");
    const auto& first = rows[0];
    const auto& second = rows[1];
    if (!first.execution_identity || first.execution_identity != second.execution_identity ||
        first.op != SerialOp::shared_expert || second.op != first.op || first.step_id != second.step_id ||
        !first.layer || first.layer != second.layer || first.position == UINT64_MAX ||
        second.position != first.position + 1U || !first.kernel_id || first.kernel_id != second.kernel_id)
        throw std::invalid_argument("known-window shared R2 row identity/position differs");
    ExecutionBindingRequest serial_shape;
    if (*first.kernel_id != kernel_id(serial_shape, config) ||
        (config.prefetch_q8 && !shared_expert_prefetch::selected(true, config.prefetch_eligible_layers, first.layer)))
        throw std::invalid_argument("known-window shared R2 serial base/prefetch ID differs");
    if (!first.resources || !second.resources ||
        first.resources->scratch != second.resources->scratch ||
        first.resources->model_manifest != second.resources->model_manifest ||
        first.resources->model_weights != second.resources->model_weights ||
        first.resources->tensor_access != second.resources->tensor_access)
        throw std::invalid_argument("known-window shared R2 weight/access owners differ");
    std::array<ExecutionBufferAddress, 2> input{}, output{}, workspace{};
    std::array<SharedTensors, 2> tensors{};
    const ExecutionMemoryBinding* shared_memory = nullptr;
    for (unsigned index = 0; index < 2U; ++index) {
        const auto& row = rows[index];
        if (row.prepared_tensor_views || row.prepared_arena_views ||
            row.logical_rows != 1U || row.request_concurrency != 1U || row.reads.size() != 1U ||
            row.writes.size() != 1U || row.temporaries.size() != 1U ||
            row.reads[0] != first.reads[0] || row.writes[0] != first.writes[0] ||
            row.temporaries[0] != first.temporaries[0])
            throw std::invalid_argument("known-window shared R2 requires matching C1 row IO");
        const auto& memory = known_shared_memory(row);
        if (shared_memory && (memory.arena_base != shared_memory->arena_base ||
            memory.arena_bytes != shared_memory->arena_bytes ||
            memory.allocation_generation != shared_memory->allocation_generation))
            throw std::invalid_argument("known-window shared R2 rows use different allocations");
        shared_memory = &memory;
        if (!row.resources->model_manifest || !row.resources->model_weights || !row.resources->tensor_access)
            throw std::invalid_argument("known-window shared R2 model resources are missing");
        validate_model_runtime_binding(*row.resources->model_manifest, *row.resources->model_weights);
        input[index] = resolve_dispatch_read(row, 0);
        output[index] = resolve_dispatch_write(row, 0);
        workspace[index] = resolve_dispatch_temporary(row, 0);
        for (const auto view : {input[index], output[index], workspace[index]}) {
            known_shared_range({view.address, view.bytes});
            if (view.allocation_generation != memory.allocation_generation)
                throw std::invalid_argument("known-window shared R2 resolved allocation is stale");
        }
        const auto& access = resolve_dispatch_tensor_access(row, 0);
        tensors[index] = identify_tensors(row, access, config);
        const auto validate_weight = [&](const TensorPair& tensor, std::uint32_t width,
                                         std::uint32_t outputs, std::uint32_t type) {
            if (!tensor.manifest || !tensor.runtime || tensor.manifest->ggml_type != type ||
                tensor.manifest->layer != row.layer || !tensor.runtime->allocation_generation ||
                tensor.manifest->absolute_offset % 16U || tensor.runtime->address % 16U)
                throw std::invalid_argument("known-window shared R2 tensor type/layer/owner differs");
            const auto& dims = tensor.manifest->dimensions;
            if (dims.empty() || dims[0] != width ||
                (outputs == 1U ? !(dims.size() == 1U || (dims.size() == 2U && dims[1] == 1U)) :
                                  !(dims.size() == 2U && dims[1] == outputs)))
                throw std::invalid_argument("known-window shared R2 tensor dimensions differ");
            validate_tensor(tensor, width, outputs, "known-window shared R2");
            const auto required = required_weight_bytes(weight_kind(type), width, outputs);
            if (tensor.manifest->byte_size != required)
                throw std::invalid_argument("known-window shared R2 tensor byte extent differs");
            known_shared_range({tensor.runtime->address, tensor.runtime->bytes});
        };
        validate_weight(tensors[index].gate, config.hidden_size, config.intermediate_size, 8U);
        validate_weight(tensors[index].up, config.hidden_size, config.intermediate_size, 8U);
        validate_weight(tensors[index].down, config.intermediate_size, config.hidden_size, 8U);
        validate_weight(tensors[index].shared_gate, config.hidden_size, 1U, 0U);
    }
    const auto pairs = [](const SharedTensors& value) {
        return std::array{value.gate, value.up, value.down, value.shared_gate};
    };
    const auto row0_weights = pairs(tensors[0]);
    const auto row1_weights = pairs(tensors[1]);
    std::array<KnownSharedSpan, 6> reads{};
    reads[0] = {input[0].address, input[0].bytes}; reads[1] = {input[1].address, input[1].bytes};
    for (unsigned index = 0; index < 4U; ++index) {
        const auto& left = *row0_weights[index].runtime;
        const auto& right = *row1_weights[index].runtime;
        // Prepared row caches may own separate immutable binding snapshots.
        // Compare their source tensor and complete physical binding, rather
        // than requiring the snapshot objects themselves to share an address.
        if (row0_weights[index].manifest != row1_weights[index].manifest ||
            left.tensor_index != right.tensor_index || left.storage != right.storage ||
            left.address != right.address || left.bytes != right.bytes ||
            left.allocation_generation != right.allocation_generation || left.owner_node != right.owner_node)
            throw std::invalid_argument("known-window shared R2 resolved weight snapshots differ");
        reads[index + 2U] = {row0_weights[index].runtime->address, row0_weights[index].runtime->bytes};
    }
    const std::array<KnownSharedSpan, 4> writes{{
        {workspace[0].address, workspace[0].bytes}, {workspace[1].address, workspace[1].bytes},
        {output[0].address, output[0].bytes}, {output[1].address, output[1].bytes}}};
    for (unsigned index = 0; index < writes.size(); ++index) {
        for (const auto read : reads) if (known_shared_overlap(writes[index], read))
            throw std::invalid_argument("known-window shared R2 cross-row writable/read alias");
        for (unsigned previous = 0; previous < index; ++previous)
            if (known_shared_overlap(writes[index], writes[previous]))
                throw std::invalid_argument("known-window shared R2 cross-row writable alias");
    }
    KnownSharedR2Views result;
    result.shape = {config.hidden_size, config.intermediate_size, 2U,
        known_shared_stride(input[0], input[1]), known_shared_stride(output[0], output[1]),
        known_shared_stride(workspace[0], workspace[1])};
    const auto layout = shared_expert_rows_diagnostic::layout(
        result.shape, shared_expert_rows_diagnostic::Arm::shared_q8_rows);
    if (input[0].bytes < static_cast<std::uint64_t>(config.hidden_size) * 2U ||
        output[0].bytes < static_cast<std::uint64_t>(config.hidden_size) * 2U ||
        workspace[0].bytes < static_cast<std::uint64_t>(config.intermediate_size) * 8U)
        throw std::invalid_argument("known-window shared R2 row payload is undersized");
    result.input = {input[0].address, layout.input_bytes};
    result.output = {output[0].address, layout.output_bytes};
    result.workspace = {workspace[0].address, layout.workspace_bytes};
    result.weights = {{tensors[0].gate.runtime->address, layout.gate_up_weight_bytes},
        {tensors[0].up.runtime->address, layout.gate_up_weight_bytes},
        {tensors[0].down.runtime->address, layout.down_weight_bytes},
        {tensors[0].shared_gate.runtime->address, layout.scalar_weight_bytes}};
    (void)shared_expert_rows_diagnostic::validate(result.shape, result.input, result.weights,
        result.workspace, result.output, shared_expert_rows_diagnostic::Arm::shared_q8_rows);
    return result;
}
void known_shared_r2_projection(bool down, const KnownSharedR2Views& views,
    KnownSharedSpan weight, KnownSharedSpan target, hipStream_t stream,
    const HipSharedExpertLauncherConfig& config) {
    const auto& shape = views.shape;
    if (down) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_rows_q8_projection_kernel<2U, true>),
            dim3((shape.hidden + 7U) / 8U), dim3(256), 0, stream,
            reinterpret_cast<const void*>(views.workspace.address), shape.workspace_stride_bytes,
            reinterpret_cast<const void*>(weight.address), shape.intermediate, shape.hidden,
            reinterpret_cast<void*>(target.address), shape.output_stride_bytes,
            reinterpret_cast<const float*>(views.workspace.address) + shape.intermediate,
            shape.workspace_stride_bytes / 4U);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_rows_q8_projection_kernel<2U, false>),
            dim3((shape.intermediate + 7U) / 8U), dim3(256), 0, stream,
            reinterpret_cast<const void*>(views.input.address), shape.input_stride_bytes,
            reinterpret_cast<const void*>(weight.address), shape.hidden, shape.intermediate,
            reinterpret_cast<void*>(target.address), shape.workspace_stride_bytes, nullptr, 0ULL);
    }
    check_hip(hipGetLastError(), "known-window shared R2 projection launch");
    if (config.prefetch_q8 && config.prefetch_stats)
        config.prefetch_stats->dispatched_projections.fetch_add(1, std::memory_order_relaxed);
}
} // namespace

void shared_expert_known_window_r2_preflight(std::span<const ExecutionDispatchContext, 2> rows,
    const HipSharedExpertLauncherConfig& config) {
    (void)known_shared_r2_views(rows, config);
}
void shared_expert_known_window_r2_launch(std::span<const ExecutionDispatchContext, 2> rows,
    const HipSharedExpertLauncherConfig& config) {
    const auto views = known_shared_r2_views(rows, config);
    const auto* scope = current_hip_launch_context();
    if (!scope || scope->graph_capture || !scope->stream || scope->stream == hipStreamLegacy ||
        scope->stream == hipStreamPerThread)
        throw std::invalid_argument("known-window shared R2 requires explicit eager owned stream");
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(scope->stream, &capture), "known-window shared R2 capture query");
    if (capture != hipStreamCaptureStatusNone)
        throw std::invalid_argument("known-window shared R2 cannot run during capture");
    check_hip(hipPeekAtLastError(), "known-window shared R2 stale launch error");
    known_shared_r2_projection(false, views, views.weights.gate_q8, views.workspace, scope->stream, config);
    auto up_target = views.workspace;
    up_target.address += static_cast<std::uint64_t>(config.intermediate_size) * 4U;
    known_shared_r2_projection(false, views, views.weights.up_q8, up_target, scope->stream, config);
    for (unsigned row = 0; row < 2U; ++row) {
        auto* workspace = reinterpret_cast<float*>(views.workspace.address + row * views.shape.workspace_stride_bytes);
        const auto* hidden = reinterpret_cast<const std::uint16_t*>(views.input.address + row * views.shape.input_stride_bytes);
        hipLaunchKernelGGL(gated_activation_kernel, dim3((config.intermediate_size + 255U) / 256U),
            dim3(256), 0, scope->stream, workspace, config.intermediate_size);
        check_hip(hipGetLastError(), "known-window shared R2 activation launch");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(scalar_projection_kernel<std::uint16_t, WeightKind::f32>),
            dim3(1), dim3(256), 0, scope->stream, hidden,
            reinterpret_cast<const void*>(views.weights.shared_gate_f32.address), config.hidden_size,
            workspace + config.intermediate_size);
        check_hip(hipGetLastError(), "known-window shared R2 scalar launch");
    }
    known_shared_r2_projection(true, views, views.weights.down_q8, views.output, scope->stream, config);
}

namespace {
KnownSharedR2Views known_shared_rows_views(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config) {
    if (rows.size() != 4U && rows.size() != 8U)
        throw std::invalid_argument("known-window shared rows count must be R4 or R8");
    const auto& first = rows.front();
    if (first.position > UINT64_MAX - (rows.size() - 1U))
        throw std::invalid_argument("known-window shared rows positions overflow");
    // Retain the complete original admission, including exact tensor metadata,
    // for every row. No pair is submitted while a later pair is unvalidated.
    std::array<KnownSharedR2Views, 4> pairs{};
    for (std::size_t pair = 0; pair < rows.size() / 2U; ++pair)
        pairs[pair] = known_shared_r2_views(
            std::span<const ExecutionDispatchContext, 2>(rows.data() + pair * 2U, 2U), config);
    std::array<ExecutionBufferAddress, 8> inputs{}, outputs{}, workspaces{};
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        if (row.execution_identity != first.execution_identity || row.step_id != first.step_id ||
            row.op != first.op || row.layer != first.layer || row.kernel_id != first.kernel_id ||
            row.position != first.position + r || row.reads_state != first.reads_state ||
            row.writes_state != first.writes_state || row.reads[0] != first.reads[0] ||
            row.writes[0] != first.writes[0] || row.temporaries[0] != first.temporaries[0])
            throw std::invalid_argument("known-window shared rows identity/position/IO differs");
        const auto& a = *first.resources; const auto& b = *row.resources;
        if (a.scratch != b.scratch || a.model_manifest != b.model_manifest || a.model_weights != b.model_weights ||
            a.tensor_access != b.tensor_access || a.session_state_layout != b.session_state_layout ||
            a.session_state_memory != b.session_state_memory || a.ple_lookup != b.ple_lookup ||
            a.ple_hash_contract != b.ple_hash_contract || a.ple_prefetch != b.ple_prefetch ||
            row.session_state_layout != first.session_state_layout || row.session_state_memory != first.session_state_memory)
            throw std::invalid_argument("known-window shared rows retained resource owners differ");
        const auto& memory = known_shared_memory(row);
        if (memory.arena_bytes > UINTPTR_MAX - memory.arena_base)
            throw std::invalid_argument("known-window shared rows scratch extent overflows");
        if (memory.execution_identity != first.resources->scratch->execution_identity ||
            memory.arena_base != first.resources->scratch->arena_base ||
            memory.arena_bytes != first.resources->scratch->arena_bytes ||
            memory.allocation_generation != first.resources->scratch->allocation_generation)
            throw std::invalid_argument("known-window shared rows physical allocation differs");
        inputs[r] = resolve_dispatch_read(row, 0);
        outputs[r] = resolve_dispatch_write(row, 0);
        workspaces[r] = resolve_dispatch_temporary(row, 0);
        for (const auto view : {inputs[r], outputs[r], workspaces[r]}) {
            if (view.address < memory.arena_base || view.address - memory.arena_base > memory.arena_bytes ||
                view.bytes > memory.arena_bytes - (view.address - memory.arena_base))
                throw std::invalid_argument("known-window shared rows resolved view exceeds physical scratch");
        }
        if (r && (known_shared_stride(inputs[r - 1U], inputs[r]) != pairs[0].shape.input_stride_bytes ||
            known_shared_stride(outputs[r - 1U], outputs[r]) != pairs[0].shape.output_stride_bytes ||
            known_shared_stride(workspaces[r - 1U], workspaces[r]) != pairs[0].shape.workspace_stride_bytes))
            throw std::invalid_argument("known-window shared rows stripe differs");
    }
    const auto weights = [](const KnownSharedR2Views& p) {
        return std::array{p.weights.gate_q8, p.weights.up_q8, p.weights.down_q8, p.weights.shared_gate_f32};
    };
    const auto first_weights = weights(pairs[0]);
    for (std::size_t pair = 1; pair < rows.size() / 2U; ++pair) {
        const auto other = weights(pairs[pair]);
        for (std::size_t w = 0; w < first_weights.size(); ++w)
            if (other[w].address != first_weights[w].address || other[w].bytes != first_weights[w].bytes)
                throw std::invalid_argument("known-window shared rows selected weights differ");
    }
    std::array<KnownSharedSpan, 16> writes{};
    for (std::size_t r = 0; r < rows.size(); ++r) {
        writes[2U * r] = {outputs[r].address, outputs[r].bytes};
        writes[2U * r + 1U] = {workspaces[r].address, workspaces[r].bytes};
    }
    for (std::size_t w = 0; w < 2U * rows.size(); ++w) {
        for (std::size_t r = 0; r < rows.size(); ++r)
            if (known_shared_overlap(writes[w], {inputs[r].address, inputs[r].bytes}))
                throw std::invalid_argument("known-window shared rows cross-row writable/input alias");
        for (const auto weight : first_weights) if (known_shared_overlap(writes[w], weight))
            throw std::invalid_argument("known-window shared rows writable/weight alias");
        for (std::size_t previous = 0; previous < w; ++previous)
            if (known_shared_overlap(writes[w], writes[previous]))
                throw std::invalid_argument("known-window shared rows cross-row writable alias");
    }
    auto result = pairs[0];
    result.shape.rows = static_cast<std::uint32_t>(rows.size());
    const auto layout = shared_expert_rows_diagnostic::layout(result.shape,
        shared_expert_rows_diagnostic::Arm::shared_q8_rows);
    result.input.bytes = layout.input_bytes; result.output.bytes = layout.output_bytes;
    result.workspace.bytes = layout.workspace_bytes;
    (void)shared_expert_rows_diagnostic::validate(result.shape, result.input, result.weights,
        result.workspace, result.output, shared_expert_rows_diagnostic::Arm::shared_q8_rows);
    return result;
}
hipStream_t known_shared_rows_stream() {
    const auto* scope = current_hip_launch_context();
    if (!scope || scope->graph_capture || !scope->stream || scope->stream == hipStreamLegacy ||
        scope->stream == hipStreamPerThread)
        throw std::invalid_argument("known-window shared rows requires explicit eager owned stream");
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(scope->stream, &capture), "known-window shared rows capture query");
    if (capture != hipStreamCaptureStatusNone)
        throw std::invalid_argument("known-window shared rows cannot run during capture");
    return scope->stream;
}
template <unsigned Rows>
void known_shared_rows_projection(bool down, const KnownSharedR2Views& views,
    KnownSharedSpan weight, KnownSharedSpan target, hipStream_t stream,
    const HipSharedExpertLauncherConfig& config) {
    const auto& s = views.shape;
    if (down) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_rows_q8_projection_kernel<Rows, true>),
            dim3((s.hidden + 7U) / 8U), dim3(256), 0, stream,
            reinterpret_cast<const void*>(views.workspace.address), s.workspace_stride_bytes,
            reinterpret_cast<const void*>(weight.address), s.intermediate, s.hidden,
            reinterpret_cast<void*>(target.address), s.output_stride_bytes,
            reinterpret_cast<const float*>(views.workspace.address) + s.intermediate, s.workspace_stride_bytes / 4U);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(shared_rows_q8_projection_kernel<Rows, false>),
            dim3((s.intermediate + 7U) / 8U), dim3(256), 0, stream,
            reinterpret_cast<const void*>(views.input.address), s.input_stride_bytes,
            reinterpret_cast<const void*>(weight.address), s.hidden, s.intermediate,
            reinterpret_cast<void*>(target.address), s.workspace_stride_bytes, nullptr, 0ULL);
    }
    check_hip(hipGetLastError(), "known-window shared rows projection launch");
    if (config.prefetch_q8 && config.prefetch_stats)
        config.prefetch_stats->dispatched_projections.fetch_add(1, std::memory_order_relaxed);
}
template <unsigned Rows>
void known_shared_rows_enqueue(const KnownSharedR2Views& views, hipStream_t stream,
    const HipSharedExpertLauncherConfig& config) {
    known_shared_rows_projection<Rows>(false, views, views.weights.gate_q8, views.workspace, stream, config);
    auto up = views.workspace; up.address += static_cast<std::uint64_t>(config.intermediate_size) * 4U;
    known_shared_rows_projection<Rows>(false, views, views.weights.up_q8, up, stream, config);
    for (unsigned row = 0; row < Rows; ++row) {
        auto* workspace = reinterpret_cast<float*>(views.workspace.address + row * views.shape.workspace_stride_bytes);
        const auto* input = reinterpret_cast<const std::uint16_t*>(views.input.address + row * views.shape.input_stride_bytes);
        hipLaunchKernelGGL(gated_activation_kernel, dim3((config.intermediate_size + 255U) / 256U),
            dim3(256), 0, stream, workspace, config.intermediate_size);
        check_hip(hipGetLastError(), "known-window shared rows activation launch");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(scalar_projection_kernel<std::uint16_t, WeightKind::f32>),
            dim3(1), dim3(256), 0, stream, input, reinterpret_cast<const void*>(views.weights.shared_gate_f32.address),
            config.hidden_size, workspace + config.intermediate_size);
        check_hip(hipGetLastError(), "known-window shared rows scalar launch");
    }
    known_shared_rows_projection<Rows>(true, views, views.weights.down_q8, views.output, stream, config);
}
} // namespace

void shared_expert_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config) {
    if (rows.size() == 2U) {
        shared_expert_known_window_r2_preflight(std::span<const ExecutionDispatchContext, 2>(rows.data(), 2), config);
        return;
    }
    static_cast<void>(known_shared_rows_views(rows, config));
    static_cast<void>(known_shared_rows_stream());
}
void shared_expert_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config) {
    if (rows.size() == 2U) {
        shared_expert_known_window_r2_launch(std::span<const ExecutionDispatchContext, 2>(rows.data(), 2), config);
        return;
    }
    const auto views = known_shared_rows_views(rows, config);
    const auto stream = known_shared_rows_stream();
    check_hip(hipPeekAtLastError(), "known-window shared rows stale launch error");
    if (rows.size() == 4U) known_shared_rows_enqueue<4U>(views, stream, config);
    else known_shared_rows_enqueue<8U>(views, stream, config);
}
} // namespace strix

namespace strix {
namespace {
struct ServingSharedPointers {
    std::uintptr_t input[4]{}, output[4]{}, workspace[4]{};
};
struct ServingSharedPrepared { KnownSharedR2Views views; ServingSharedPointers pointers; };
ServingSharedPrepared prepare_serving_shared(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config,HipSharedServingStaging staging) {
    if ((rows.size()!=2U && rows.size()!=4U) || rows.size()>staging.capacity_rows ||
        config.hidden_size!=2560U || config.intermediate_size!=640U || config.activation!=HipActivationStorage::bf16 ||
        !staging.address || staging.address%256U || !staging.allocation_generation ||
        staging.bytes!=hip_shared_serving_staging_bytes(staging.capacity_rows) || staging.bytes>UINTPTR_MAX-staging.address)
        throw std::invalid_argument("shared serving cohort has invalid canonical rows or staging");
    ServingSharedPrepared result;
    const auto range=[](KnownSharedSpan v) {
        if(!v.address || !v.bytes || v.bytes>UINTPTR_MAX-v.address)
            throw std::invalid_argument("shared serving cohort has an invalid retained range");
    };
    std::array<KnownSharedSpan,12> writes{}, reads{};
    std::array<RuntimeTensorBinding,4> weights{};
    const auto& first=rows.front();
    for(std::size_t row=0;row<rows.size();++row) {
        const auto& context=rows[row];
        if(context.logical_rows!=1U || context.request_concurrency!=1U || !context.execution_identity ||
            context.execution_identity!=first.execution_identity || context.step_id!=first.step_id || !context.layer || context.layer!=first.layer ||
            !context.kernel_id || context.kernel_id!=first.kernel_id || context.reads_state!=PersistentState::none ||
            context.writes_state!=PersistentState::none || !context.resources || !context.resources->model_weights ||
            !context.resources->model_weights->model_content_identity.valid() ||
            context.resources->model_weights->model_content_identity!=first.resources->model_weights->model_content_identity)
            throw std::invalid_argument("shared serving cohort original C1 step or model differs");
        constexpr std::string_view base="hip.shared_expert.bf16.h2560.m640.r1.c1";
        const auto actual=*context.kernel_id;
        const bool exact=config.prefetch_q8 ? actual.size()==base.size()+shared_expert_prefetch::kernel_suffix.size() &&
            actual.starts_with(base) && actual.ends_with(shared_expert_prefetch::kernel_suffix) : actual==base;
        if(!exact)
            throw std::invalid_argument("shared serving cohort requires the exact selected serial shared ID");
        const auto prepared=prepare_shared_expert(context,const_cast<HipSharedExpertLauncherConfig*>(&config));
        const std::array<TensorPair,4> tensors{prepared.tensors.gate,prepared.tensors.up,prepared.tensors.down,prepared.tensors.shared_gate};
        for(std::size_t i=0;i<tensors.size();++i) {
            const auto& t=tensors[i]; const auto& w=*t.runtime;
            if(t.manifest->ggml_type!=(i==3U?0U:8U) || !w.allocation_generation || w.address%16U ||
                t.manifest->absolute_offset%16U)
                throw std::invalid_argument("shared serving cohort requires aligned Q8 matrices and F32 scalar");
            const auto& dims=t.manifest->dimensions;
            const auto width=i==2U?640U:2560U;
            const auto outputs=i==3U?1U:(i==2U?2560U:640U);
            if(t.manifest->layer!=context.layer || dims.empty() || dims[0]!=width ||
                (i==3U ? !(dims.size()==1U || (dims.size()==2U && dims[1]==1U)) : !(dims.size()==2U && dims[1]==outputs)) ||
                t.manifest->byte_size<required_weight_bytes(weight_kind(t.manifest->ggml_type),width,outputs))
                throw std::invalid_argument("shared serving cohort tensor shape or source extent differs");
            range({w.address,w.bytes});
            if(!row) weights[i]=w;
            else if(w.tensor_index!=weights[i].tensor_index || w.storage!=weights[i].storage || w.address!=weights[i].address ||
                w.bytes!=weights[i].bytes || w.allocation_generation!=weights[i].allocation_generation || w.owner_node!=weights[i].owner_node)
                throw std::invalid_argument("shared serving cohort actual immutable weight owners differ");
        }
        const KnownSharedSpan input{prepared.input.address,prepared.input.bytes};
        const KnownSharedSpan output{prepared.output.address,prepared.output.bytes};
        const KnownSharedSpan work{prepared.temporary.address,prepared.temporary.bytes};
        range(input); range(output); range(work);
        if(input.address%16U || output.address%16U || work.address%16U)
            throw std::invalid_argument("shared serving cohort row view is misaligned");
        result.pointers.input[row]=input.address; result.pointers.output[row]=output.address; result.pointers.workspace[row]=work.address;
        reads[row]=input; writes[2U*row]=output; writes[2U*row+1U]=work;
    }
    for(std::size_t i=0;i<4U;++i) reads[rows.size()+i]={weights[i].address,weights[i].bytes};
    const KnownSharedSpan pool{staging.address,staging.bytes};
    for(std::size_t i=0;i<rows.size()+4U;++i) {
        if(known_shared_overlap(pool,reads[i])) throw std::invalid_argument("shared serving staging aliases a request input or weight");
        for(std::size_t j=0;j<2U*rows.size();++j)
            if(known_shared_overlap(writes[j],reads[i])) throw std::invalid_argument("shared serving cohort writable/read alias");
    }
    for(std::size_t i=0;i<2U*rows.size();++i) {
        if(known_shared_overlap(pool,writes[i])) throw std::invalid_argument("shared serving staging aliases request writes");
        for(std::size_t j=0;j<i;++j) if(known_shared_overlap(writes[i],writes[j]))
            throw std::invalid_argument("shared serving cohort writable owners alias");
    }
    auto& view=result.views;
    view.shape={2560U,640U,static_cast<std::uint32_t>(rows.size()),5120U,5120U,5120U};
    view.input={staging.address,rows.size()*5120U};
    view.workspace={staging.address+staging.capacity_rows*5120ULL,rows.size()*5120U};
    view.output={staging.address+staging.capacity_rows*10240ULL,rows.size()*5120U};
    view.weights={{weights[0].address,weights[0].bytes},{weights[1].address,weights[1].bytes},
        {weights[2].address,weights[2].bytes},{weights[3].address,weights[3].bytes}};
    return result;
}
__global__ void serving_shared_gather(ServingSharedPointers pointers,std::uint16_t* packed) {
    const auto feature=static_cast<std::uint32_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto row=static_cast<std::uint32_t>(blockIdx.y);
    if(feature<2560U) packed[row*2560U+feature]=reinterpret_cast<const std::uint16_t*>(pointers.input[row])[feature];
}
__global__ void serving_shared_scatter(ServingSharedPointers pointers,const unsigned char* output,const unsigned char* work) {
    const auto byte=static_cast<std::uint32_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto row=static_cast<std::uint32_t>(blockIdx.y);
    if(byte<5120U) {
        reinterpret_cast<unsigned char*>(pointers.output[row])[byte]=output[row*5120U+byte];
        reinterpret_cast<unsigned char*>(pointers.workspace[row])[byte]=work[row*5120U+byte];
    }
}
} // namespace
void shared_expert_serving_cohort_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config,HipSharedServingStaging staging) {
    (void)prepare_serving_shared(rows,config,staging);
}
void shared_expert_serving_cohort_launch(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config,HipSharedServingStaging staging) {
    const auto prepared=prepare_serving_shared(rows,config,staging);
    const auto stream=known_shared_rows_stream();
    const auto* scope=current_hip_launch_context();
    if(scope->deferred_moe_status.arena_base) throw std::invalid_argument("shared serving cohort cannot publish a request D1 record");
    check_hip(hipPeekAtLastError(),"shared serving cohort stale error");
    hipLaunchKernelGGL(serving_shared_gather,dim3(10U,static_cast<unsigned>(rows.size())),dim3(256),0,stream,
        prepared.pointers,reinterpret_cast<std::uint16_t*>(prepared.views.input.address));
    check_hip(hipGetLastError(),"shared serving input gather");
    if(rows.size()==2U) known_shared_rows_enqueue<2U>(prepared.views,stream,config);
    else known_shared_rows_enqueue<4U>(prepared.views,stream,config);
    hipLaunchKernelGGL(serving_shared_scatter,dim3(20U,static_cast<unsigned>(rows.size())),dim3(256),0,stream,
        prepared.pointers,reinterpret_cast<const unsigned char*>(prepared.views.output.address),
        reinterpret_cast<const unsigned char*>(prepared.views.workspace.address));
    check_hip(hipGetLastError(),"shared serving output/workspace scatter");
}
} // namespace strix

namespace strix {
namespace {
void mtp_shared_prepare(const MtpNativeContext& c){
    mtp_native::context(c);
    if(c.model->hidden_size!=2560U || c.model->shared_expert_intermediate_size!=640U)
        throw std::invalid_argument("MTP shared expert geometry is unsupported");
    const auto input=mtp_native::buffer(c,HipMtpBuffer::hidden0,5120U,2U);
    const auto output=mtp_native::buffer(c,HipMtpBuffer::hidden1,5120U,2U);
    const auto workspace=mtp_native::buffer(c,HipMtpBuffer::moe_workspace,5120U,4U);
    mtp_native::disjoint({input,output,workspace});
    const auto& t=*c.tensors;const std::array tensors{t.shared_gate,t.shared_up,t.shared_down,t.shared_scalar_gate};
    for(std::size_t i=0;i<tensors.size();++i){const auto pair=tensors[i];const bool scalar=i==3U,down=i==2U;
        const std::vector<std::uint64_t> shape=scalar?std::vector<std::uint64_t>{2560U}:
            std::vector<std::uint64_t>{down?640U:2560U,down?2560U:640U};
        const std::uint64_t bytes=scalar?10240U:static_cast<std::uint64_t>(shape[0]/32U)*34U*shape[1];
        if(!pair.manifest || !pair.runtime || pair.manifest->ggml_type!=(scalar?0U:8U) || pair.manifest->dimensions!=shape ||
           pair.manifest->byte_size!=bytes || !runtime_tensor_gpu_visible(pair.runtime->storage) || !pair.runtime->address ||
           !pair.runtime->allocation_generation || pair.runtime->bytes<bytes || pair.runtime->address%16U ||
           pair.runtime->bytes>std::numeric_limits<std::uintptr_t>::max()-pair.runtime->address)
            throw std::invalid_argument("MTP shared expert tensor type, shape, extent or owner is invalid");
        const MtpDeviceView weights{pair.runtime->address,pair.runtime->bytes,pair.runtime->allocation_generation};
        mtp_native::disjoint({weights,output,workspace});
    }
}
}
void preflight_hip_mtp_shared_expert(const MtpNativeContext& c){mtp_shared_prepare(c);}
void launch_hip_mtp_shared_expert(const MtpNativeContext& c,void* raw_stream){
    mtp_shared_prepare(c);
    if(!raw_stream || raw_stream!=c.resources->stream())throw std::invalid_argument("MTP shared expert stream differs from actual owner");
    const auto stream=static_cast<hipStream_t>(raw_stream);hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(stream,&capture),"MTP shared capture query");
    if(capture!=hipStreamCaptureStatusNone)throw std::invalid_argument("MTP shared expert is eager only");
    check_hip(hipPeekAtLastError(),"MTP shared prior HIP error");
    const auto& owner=*c.resources;const auto& t=*c.tensors;
    const auto* input=reinterpret_cast<const std::uint16_t*>(owner.view(HipMtpBuffer::hidden0).address);
    auto* workspace=reinterpret_cast<float*>(owner.view(HipMtpBuffer::moe_workspace).address);
    auto* up=workspace+640U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(projection_to_fp32_kernel<std::uint16_t,WeightKind::q8_0>),dim3(640U),dim3(kThreads),0,stream,
        input,reinterpret_cast<const void*>(t.shared_gate.runtime->address),2560U,640U,workspace);
    check_hip(hipGetLastError(),"MTP shared Q8 gate projection");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(projection_to_fp32_kernel<std::uint16_t,WeightKind::q8_0>),dim3(640U),dim3(kThreads),0,stream,
        input,reinterpret_cast<const void*>(t.shared_up.runtime->address),2560U,640U,up);
    check_hip(hipGetLastError(),"MTP shared Q8 up projection");
    hipLaunchKernelGGL(gated_activation_kernel,dim3(3U),dim3(kThreads),0,stream,workspace,640U);
    check_hip(hipGetLastError(),"MTP shared activation");
    // Scalar replaces up[0] ONLY after activation consumed the complete up row.
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scalar_projection_kernel<std::uint16_t,WeightKind::f32>),dim3(1U),dim3(kThreads),0,stream,
        input,reinterpret_cast<const void*>(t.shared_scalar_gate.runtime->address),2560U,up);
    check_hip(hipGetLastError(),"MTP shared F32 scalar gate");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(down_projection_kernel<std::uint16_t,WeightKind::q8_0>),dim3(2560U),dim3(kThreads),0,stream,
        workspace,up,reinterpret_cast<const void*>(t.shared_down.runtime->address),640U,2560U,
        reinterpret_cast<std::uint16_t*>(owner.view(HipMtpBuffer::hidden1).address));
    check_hip(hipGetLastError(),"MTP shared Q8 down output");
}
} // namespace strix
