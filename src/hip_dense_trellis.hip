#include "strix/hip_dense_trellis.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdexcept>

// Wave Hadamard and split reduction organization adapted from the related
// project-owned strix_trellis transforms/backend at 9c28680. This operator uses
// FP32 transformed inputs and explicit FP32 FMAs, a distinct numerical recipe.
namespace strix {
namespace {
#pragma clang fp contract(off)
constexpr unsigned splits = 4U;
struct ContiguousRows {
    const std::uint16_t* input;
    float* output;
    std::uint64_t input_stride, output_stride;
};
__device__ const std::uint16_t* input_row(const ContiguousRows& views, unsigned row) { return views.input + row * views.input_stride; }
__device__ const std::uint16_t* input_row(const HipDenseTrellisRows& views, unsigned row) { return views.input[row]; }
__device__ void store(const ContiguousRows& views, unsigned row, unsigned column, float value) {
    views.output[row * views.output_stride + column] = value;
}
__device__ void store(const HipDenseTrellisRows& views, unsigned row, unsigned column, float value) {
    const auto bits = __float_as_uint(value);
    views.output[row][column] = __uint_as_float((bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U);
}
__device__ float scale(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }
__device__ void transform(float (&v)[4]) {
    const unsigned lane = threadIdx.x & 31U;
#pragma unroll
    for (unsigned bit = 1; bit < 32U; bit <<= 1U) {
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const float peer = __shfl_xor(v[j], static_cast<int>(bit), 32);
            v[j] = lane & bit ? peer - v[j] : v[j] + peer;
        }
    }
#pragma unroll
    for (unsigned bit = 1; bit < 4U; bit <<= 1U)
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) if (!(j & bit)) {
            const float a = v[j], b = v[j + bit];
            v[j] = a + b; v[j + bit] = a - b;
        }
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) v[j] *= 0x1.6a09e6p-4F;
}
template <class Views>
__global__ void input_transform(HipDenseTrellisWeights weights, Views views, float* transformed, unsigned rows) {
    const unsigned group = blockIdx.x * 4U + threadIdx.x / 32U;
    if (group >= rows * (weights.inputs / 128U)) return;
    const unsigned row = group / (weights.inputs / 128U), begin = group % (weights.inputs / 128U) * 128U;
    const unsigned lane = threadIdx.x & 31U;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        const auto i = begin + lane + 32U * j;
        v[j] = __uint_as_float(static_cast<unsigned>(input_row(views, row)[i]) << 16U) * scale(weights.input_scales[i]);
    }
    transform(v);
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) transformed[static_cast<std::uint64_t>(row) * weights.inputs + begin + lane + 32U * j] = v[j];
}
template <unsigned Bits>
__device__ float coefficient(const std::uint32_t* tile, unsigned position) {
    // Bits [end-16,end) on the MSB-first circular stream, including the tail at
    // position zero. All arithmetic below the final conversion is exact dyadic.
    constexpr unsigned ring = 256U * Bits, words = ring / 32U;
    const unsigned start = ((position + 1U) * Bits + ring - 16U) % ring;
    const unsigned word = start / 32U, shift = start % 32U;
    const std::uint64_t pair = static_cast<std::uint64_t>(tile[word]) << 32U | tile[(word + 1U) % words];
    const unsigned state = static_cast<unsigned>(pair >> (48U - shift)) & 65535U;
    const unsigned p = state * 0x83dcd12dU;
    const unsigned bytes = (p & 255U) + ((p >> 8U) & 255U) + ((p >> 16U) & 255U) + (p >> 24U);
    const int numerator = static_cast<int>((1024U + bytes) * 1774U) - 1329 * 2048;
    return __half2float(__float2half_rn(static_cast<float>(numerator) * 0x1p-18F));
}
template <unsigned Bits> struct Window {
    static constexpr unsigned words = 8U * Bits;
    unsigned first, second, third, offset;
    __device__ explicit Window(unsigned lane) {
        int begin = static_cast<int>((8U * lane + 1U) * Bits) - 16;
        if (begin < 0) begin += static_cast<int>(256U * Bits);
        first = static_cast<unsigned>(begin) / 32U;
        offset = static_cast<unsigned>(begin) & 31U;
        second = first + 1U < words ? first + 1U : first + 1U - words;
        third = first + 2U < words ? first + 2U : first + 2U - words;
    }
    __device__ std::uint64_t load(const std::uint32_t* tile) const {
        const auto high = static_cast<std::uint64_t>(tile[first]) << 32U | tile[second];
        // For K5/K6 these offsets are respectively 5/13/21/29 and 6/22.
        return (high << offset) | (static_cast<std::uint64_t>(tile[third]) >> (32U - offset));
    }
};
__device__ float fast_coefficient(unsigned state) {
    const unsigned p = state * 0x83dcd12dU;
    const int bytes = __builtin_amdgcn_sudot4(false, static_cast<int>(p), false, 0x01010101, 0, false);
    const int numerator = (1024 + bytes) * 1774 - 1329 * 2048;
    return __half2float(__float2half_rn(static_cast<float>(numerator) * 0x1p-18F));
}
__device__ float2 paired_coefficients(unsigned first, unsigned second) {
    // Packed codec organization follows the related project-owned
    // strix_trellis optimized codebook helper; no projection arithmetic is adopted.
    // 1024 + byte_sum is exactly representable as F16: its bits are 0x6400
    // + byte_sum. F16 FMA with 1774/2^18 and -1329/128 rounds the same exact
    // dyadic numerator as coefficient(), once. Only the codec uses packed F16;
    // activations and both independent accumulation chains remain FP32.
    const auto byte_sum = [](unsigned state) {
        return static_cast<unsigned>(__builtin_amdgcn_sudot4(false, static_cast<int>(state * 0x83dcd12dU),
            false, 0x01010101, 0, false));
    };
    const auto values = __halves2half2(__ushort_as_half(static_cast<unsigned short>(0x6400U + byte_sum(first))),
        __ushort_as_half(static_cast<unsigned short>(0x6400U + byte_sum(second))));
    return __half22float2(__hfma2(values,
        __halves2half2(__ushort_as_half(0x1eeeU), __ushort_as_half(0x1eeeU)),
        __halves2half2(__ushort_as_half(0xc931U), __ushort_as_half(0xc931U))));
}
template <unsigned Bits>
__global__ void decode_window(HipDenseTrellisWeights weights, std::uint16_t* output) {
    const auto at = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(weights.inputs) * weights.outputs;
    if (at >= elements / 8U) return;
    const auto tile = at / 32U, tile_n = tile % (weights.outputs / 16U), tile_i = tile / (weights.outputs / 16U);
    const unsigned lane = static_cast<unsigned>(at % 32U);
    const auto window = Window<Bits>(lane).load(weights.tiles + tile * (8U * Bits));
#pragma unroll
    for (unsigned fragment = 0; fragment < 8U; ++fragment) {
        const auto state = static_cast<unsigned>(window >> (48U - fragment * Bits)) & 65535U;
        const unsigned row = 2U * (lane & 3U) + (fragment & 1U) + 8U * ((fragment / 2U) & 1U);
        const unsigned col = lane / 4U + 8U * (fragment / 4U);
        output[(tile_i * 16U + row) * weights.outputs + tile_n * 16U + col] = __half_as_ushort(__float2half_rn(fast_coefficient(state)));
    }
}
template <unsigned Bits>
__global__ void decode(HipDenseTrellisWeights weights, std::uint16_t* output) {
    const auto at = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<std::uint64_t>(weights.inputs) * weights.outputs) return;
    const auto tile_n = at / 256U % (weights.outputs / 16U), tile_i = at / 256U / (weights.outputs / 16U);
    const unsigned p = static_cast<unsigned>(at % 256U), lane = p / 8U, fragment = p % 8U;
    const auto* tile = weights.tiles + (at / 256U) * (8U * Bits);
    const unsigned row = 2U * (lane & 3U) + (fragment & 1U) + 8U * ((fragment / 2U) & 1U);
    const unsigned col = lane / 4U + 8U * (fragment / 4U);
    output[(tile_i * 16U + row) * weights.outputs + tile_n * 16U + col] = __half_as_ushort(__float2half_rn(coefficient<Bits>(tile, p)));
}
template <unsigned Bits, unsigned Rows>
__global__ void core(HipDenseTrellisWeights weights, const float* input, float* partial, unsigned rows) {
    const unsigned tile_n = blockIdx.x * 4U + threadIdx.x / 32U;
    if (tile_n >= weights.outputs / 16U) return;
    const unsigned lane = threadIdx.x & 31U, first_row = blockIdx.z * Rows;
    const unsigned begin = (weights.inputs / 16U) * blockIdx.y / splits;
    const unsigned end = (weights.inputs / 16U) * (blockIdx.y + 1U) / splits;
    float sum[Rows][2]{};
    for (unsigned tile_i = begin; tile_i < end; ++tile_i) {
        const auto* tile = weights.tiles + (static_cast<std::uint64_t>(tile_i) * (weights.outputs / 16U) + tile_n) * (8U * Bits);
#pragma unroll
        for (unsigned fragment = 0; fragment < 8U; ++fragment) {
            const float q = coefficient<Bits>(tile, 8U * lane + fragment);
            const unsigned feature = tile_i * 16U + 2U * (lane & 3U) + (fragment & 1U) + 8U * ((fragment / 2U) & 1U);
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r)
                if (first_row + r < rows)
                    sum[r][fragment / 4U] = __builtin_fmaf(q, input[static_cast<std::uint64_t>(first_row + r) * weights.inputs + feature], sum[r][fragment / 4U]);
        }
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r)
#pragma unroll
        for (unsigned h = 0; h < 2U; ++h) {
            float v = sum[r][h];
            v += __shfl_down(v, 2U, 4); v += __shfl_down(v, 1U, 4);
            if (!(lane & 3U) && first_row + r < rows)
                partial[(static_cast<std::uint64_t>(blockIdx.y) * rows + first_row + r) * weights.outputs + tile_n * 16U + lane / 4U + h * 8U] = v;
        }
}
template <unsigned Bits, unsigned Rows, bool CompleteRows, unsigned Waves = 8U>
__global__ void core_staged(HipDenseTrellisWeights weights, const float* input, float* partial, unsigned rows) {
    // Exactly forty 16-feature tiles in each of four partitions of I=2560.
    // Every wave in the CTA reuses this activation slice for different outputs.
    __shared__ float2 stage[Rows][320U];
    const unsigned first_row = blockIdx.z * Rows;
    const unsigned begin = 40U * blockIdx.y, end = begin + 40U;
    for (unsigned i = threadIdx.x; i < Rows * 640U; i += blockDim.x) {
        const unsigned r = i / 640U, feature = i % 640U;
        reinterpret_cast<float*>(stage[r])[feature] = (CompleteRows || first_row + r < rows) ? input[static_cast<std::uint64_t>(first_row + r) * 2560U + begin * 16U + feature] : 0.0F;
    }
    __syncthreads();
    const unsigned tile_n = blockIdx.x * Waves + threadIdx.x / 32U;
    if (tile_n >= weights.outputs / 16U) return;
    const unsigned lane = threadIdx.x & 31U;
    const Window<Bits> reader(lane);
    float sum[Rows][2]{};
    for (unsigned batch = begin; batch < end; batch += 2U) {
        // Expose independent packed loads before either dependent decode chain.
        std::uint64_t windows[2];
#pragma unroll
        for (unsigned u = 0; u < 2U; ++u)
            windows[u] = reader.load(weights.tiles + (static_cast<std::uint64_t>(batch + u) * (weights.outputs / 16U) + tile_n) * (8U * Bits));
#pragma unroll
        for (unsigned u = 0; u < 2U; ++u) {
            const unsigned tile_i = batch + u;
            const auto window = windows[u];
            float2 q[4];
#pragma unroll
            for (unsigned fragment = 0; fragment < 4U; ++fragment) {
                q[fragment] = paired_coefficients(static_cast<unsigned>(window >> (48U - fragment * Bits)) & 65535U,
                    static_cast<unsigned>(window >> (48U - (fragment + 4U) * Bits)) & 65535U);
            }
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) if (CompleteRows || first_row + r < rows) {
#pragma unroll
                for (unsigned pair = 0; pair < 2U; ++pair) {
                    const auto x = stage[r][(tile_i - begin) * 8U + (lane & 3U) + 4U * pair];
                    sum[r][0] = __builtin_fmaf(q[2U * pair].x, x.x, sum[r][0]);
                    sum[r][1] = __builtin_fmaf(q[2U * pair].y, x.x, sum[r][1]);
                    sum[r][0] = __builtin_fmaf(q[2U * pair + 1U].x, x.y, sum[r][0]);
                    sum[r][1] = __builtin_fmaf(q[2U * pair + 1U].y, x.y, sum[r][1]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r)
#pragma unroll
        for (unsigned h = 0; h < 2U; ++h) {
            float v = sum[r][h];
            v += __shfl_down(v, 2U, 4); v += __shfl_down(v, 1U, 4);
            if (!(lane & 3U) && (CompleteRows || first_row + r < rows))
                partial[(static_cast<std::uint64_t>(blockIdx.y) * rows + first_row + r) * weights.outputs + tile_n * 16U + lane / 4U + h * 8U] = v;
        }
}
template <class Views>
__global__ void output_transform(HipDenseTrellisWeights weights, const float* partial, Views views, unsigned rows) {
    const unsigned group = blockIdx.x * 4U + threadIdx.x / 32U;
    if (group >= rows * (weights.outputs / 128U)) return;
    const unsigned row = group / (weights.outputs / 128U), begin = group % (weights.outputs / 128U) * 128U;
    const unsigned lane = threadIdx.x & 31U;
    float v[4]{};
#pragma unroll
    for (unsigned split = 0; split < splits; ++split)
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j)
            v[j] += partial[(static_cast<std::uint64_t>(split) * rows + row) * weights.outputs + begin + lane + 32U * j];
    transform(v);
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        const unsigned i = begin + lane + 32U * j;
        store(views, row, i, v[j] * scale(weights.output_scales[i]));
    }
}
template <unsigned Bits> void project(HipDenseTrellisWeights weights, const float* input,
    float* partial, unsigned rows, hipStream_t stream, bool optimized) {
    const auto run = [&]<unsigned R>() {
        if (optimized && weights.inputs == 2560U) {
            // The host proves all row tiles complete for R1/2/4/8 and multiples
            // of eight. Elide row predicates without changing partial writers.
            const auto staged = [&]<bool Complete>() {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(core_staged<Bits, R, Complete>), dim3((weights.outputs / 16U + 7U) / 8U, splits, (rows + R - 1U) / R),
                    dim3(256U), 0U, stream, weights, input, partial, rows);
            };
            if (rows % R == 0U) staged.template operator()<true>();
            else staged.template operator()<false>();
        }
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(core<Bits, R>), dim3((weights.outputs / 16U + 3U) / 4U, splits, (rows + R - 1U) / R),
                dim3(128U), 0U, stream, weights, input, partial, rows);
    };
    if (rows == 1U) run.template operator()<1U>();
    else if (rows == 2U) run.template operator()<2U>();
    else if (rows <= 4U) run.template operator()<4U>();
    else run.template operator()<8U>();
}
void validate(const HipDenseTrellisWeights& weights, unsigned rows, const HipDenseTrellisWorkspace& workspace) {
    if (!weights.tiles || !weights.input_scales || !weights.output_scales || !weights.inputs || !weights.outputs ||
        weights.inputs % 128U || weights.outputs % 128U || (weights.bits != 5U && weights.bits != 6U) ||
        !rows || rows > 64U ||
        !workspace.transformed || !workspace.partials ||
        workspace.transformed_bytes < static_cast<std::uint64_t>(rows) * weights.inputs * 4U ||
        workspace.partial_bytes < static_cast<std::uint64_t>(splits) * rows * weights.outputs * 4U)
        throw std::invalid_argument("dense trellis projection descriptor or workspace differs");
}
void check_launch() {
    const auto error = hipGetLastError();
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
template <class Views> void submit(const HipDenseTrellisWeights& weights, const Views& views, unsigned rows,
    const HipDenseTrellisWorkspace& workspace, hipStream_t stream, bool optimized = true) {
    hipLaunchKernelGGL(input_transform<Views>, dim3((rows * (weights.inputs / 128U) + 3U) / 4U), dim3(128U), 0U, stream,
        weights, views, workspace.transformed, rows);
    check_launch();
    if (weights.bits == 5U) project<5U>(weights, workspace.transformed, workspace.partials, rows, stream, optimized);
    else project<6U>(weights, workspace.transformed, workspace.partials, rows, stream, optimized);
    check_launch();
    hipLaunchKernelGGL(output_transform<Views>, dim3((rows * (weights.outputs / 128U) + 3U) / 4U), dim3(128U), 0U, stream,
        weights, workspace.partials, views, rows);
    check_launch();
}
} // namespace
void launch_hip_dense_trellis(const HipDenseTrellisWeights& weights, const std::uint16_t* input,
    std::uint64_t input_stride, float* output, std::uint64_t output_stride, std::uint32_t rows,
    const HipDenseTrellisWorkspace& workspace, void* raw_stream) {
    validate(weights, rows, workspace);
    if (!input || !output || input_stride < weights.inputs || output_stride < weights.outputs)
        throw std::invalid_argument("dense trellis contiguous rows differ");
    submit(weights, ContiguousRows{input, output, input_stride, output_stride}, rows, workspace, static_cast<hipStream_t>(raw_stream));
}
void launch_hip_dense_trellis_rows(const HipDenseTrellisWeights& weights, const HipDenseTrellisRows& views,
    std::uint32_t rows, const HipDenseTrellisWorkspace& workspace, void* raw_stream) {
    validate(weights, rows, workspace);
    if (rows > views.input.size()) throw std::invalid_argument("dense trellis row views exceed eight");
    for (unsigned r = 0; r < rows; ++r)
        if (!views.input[r] || !views.output[r]) throw std::invalid_argument("dense trellis row view is null");
    submit(weights, views, rows, workspace, static_cast<hipStream_t>(raw_stream));
}
void launch_hip_dense_trellis_scalar_reference(const HipDenseTrellisWeights& weights, const std::uint16_t* input,
    std::uint64_t input_stride, float* output, std::uint64_t output_stride, std::uint32_t rows,
    const HipDenseTrellisWorkspace& workspace, void* raw_stream) {
    validate(weights, rows, workspace);
    if (!input || !output || input_stride < weights.inputs || output_stride < weights.outputs)
        throw std::invalid_argument("dense trellis reference rows differ");
    submit(weights, ContiguousRows{input, output, input_stride, output_stride}, rows, workspace, static_cast<hipStream_t>(raw_stream), false);
}
void launch_hip_dense_trellis_decode(const HipDenseTrellisWeights& weights,
    std::uint16_t* output, std::uint64_t output_bytes, void* raw_stream) {
    const auto count = static_cast<std::uint64_t>(weights.inputs) * weights.outputs;
    if (!weights.tiles || !output || !weights.inputs || !weights.outputs || weights.inputs % 128U || weights.outputs % 128U ||
        (weights.bits != 5U && weights.bits != 6U) || output_bytes < count * 2U || count > (1ULL << 34U))
        throw std::invalid_argument("dense trellis decode descriptor differs");
    const auto stream = static_cast<hipStream_t>(raw_stream);
    const auto blocks = static_cast<unsigned>((count / 8U + 255U) / 256U);
    if (weights.bits == 5U) hipLaunchKernelGGL(decode_window<5U>, dim3(blocks), dim3(256U), 0U, stream, weights, output);
    else hipLaunchKernelGGL(decode_window<6U>, dim3(blocks), dim3(256U), 0U, stream, weights, output);
    const auto error = hipGetLastError();
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
} // namespace strix
