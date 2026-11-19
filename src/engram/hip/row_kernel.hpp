#pragma once
// Engram row preparation, HIP kernels (the one kernel family of this operation on HIP; contract and host reference:
// include/strix/engram/row_prep.hpp). Engines bind it through RowPrepView + enqueue_row_prep; the form, the row scale
// and the row width are compile-time policies of the family. Compile with the engine's strict arithmetic flags
// (-ffp-contract=off -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt, wave32): the head_rms_bf16 reduction
// follows ds41::chunked_dot_of and the host reference bitwise.
//
// decode: one thread per value (the DeepSeek V4.1 kernel as qualified, moved here unchanged for e8m0_per32 x 256).
// head_rms_bf16: one wave per row (8 rows per 256-thread block); lane l holds values l, l + 32, ... in registers.
#include "strix/engram/row_prep.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strix::engram::hip {

inline constexpr unsigned row_prep_wave = 32U, row_prep_threads = 256U, row_prep_rows_per_block = row_prep_threads / row_prep_wave;

template <RowScale Scale, std::size_t Elements>
__device__ __forceinline__ float row_value_device(const unsigned char* __restrict__ row, unsigned j) {
    const float v = ds41::codec::e4m3(row[j]);
    if constexpr (Scale == RowScale::e8m0_per32) {
        return v * ds41::codec::e8m0(row[static_cast<unsigned>(Elements) + j / 32U]);
    } else if constexpr (Scale == RowScale::f32_per_row) {
        float s;
        __builtin_memcpy(&s, row + Elements, sizeof(float));
        return v * s;
    } else {
        return v;
    }
}

template <RowScale Scale, std::size_t Elements>
__global__ __launch_bounds__(256) void row_decode_kernel(const unsigned char* __restrict__ rows, unsigned long long row_count,
                                                         float* __restrict__ out) {
    constexpr std::size_t bytes = RowFormat{static_cast<std::uint32_t>(Elements), Scale}.bytes();
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * 256U + threadIdx.x;
    if (index >= row_count * Elements) return;
    const auto* row = rows + index / Elements * bytes;
    const unsigned j = static_cast<unsigned>(index % Elements);
    out[index] = row_value_device<Scale, Elements>(row, j);
}

template <RowScale Scale, unsigned Elements>
__global__ __launch_bounds__(256) void row_head_rms_kernel(const unsigned char* __restrict__ rows, unsigned row_count, unsigned heads,
                                                           const float* __restrict__ weight, float eps, float* __restrict__ out) {
    static_assert(Elements > 0U && Elements <= row_prep_max_elements, "head_rms_bf16: one lane-dot chunk");
    constexpr unsigned per = (Elements + row_prep_wave - 1U) / row_prep_wave;
    constexpr std::size_t bytes = RowFormat{Elements, Scale}.bytes();
    const unsigned lane = threadIdx.x % row_prep_wave, r = blockIdx.x * row_prep_rows_per_block + threadIdx.x / row_prep_wave;
    if (r >= row_count) return;  // uniform per wave
    const unsigned char* row = rows + static_cast<std::size_t>(r) * bytes;
    float x[per];
    float acc = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < per; ++i) {
        const unsigned j = lane + row_prep_wave * i;
        x[i] = 0.0F;
        if (j < Elements) {
            x[i] = ds41::round_to_bf16(row_value_device<Scale, Elements>(row, j));
            acc = acc + x[i] * x[i];
        }
    }
    for (int d = 16; d; d >>= 1) acc = acc + __shfl_xor(acc, d, static_cast<int>(row_prep_wave));
    const float ss = 0.0F + acc;  // chunked_dot_of: total = 0 + chunk 0
    const float rr = 1.0F / ds41::declared_sqrt(ss / static_cast<float>(Elements) + eps);
    const float* w = weight + static_cast<std::size_t>(r % heads) * Elements;
    float* o = out + static_cast<std::size_t>(r) * Elements;
#pragma unroll
    for (unsigned i = 0; i < per; ++i) {
        const unsigned j = lane + row_prep_wave * i;
        if (j < Elements) o[j] = ds41::round_to_bf16(w[j] * ds41::round_to_bf16(x[i] * rr));
    }
}

// The form, scale and width are template arguments so an engine instantiates only what it binds (the view must match).
template <RowPrepForm Form, RowScale Scale, std::uint32_t Elements>
void enqueue_row_prep(const RowPrepView& v, hipStream_t stream) {
    if (v.form != Form || v.fmt.scale != Scale || v.fmt.elements != Elements) throw std::invalid_argument("Engram row prep view");
    if (const char* why = row_prep_invalid(v)) throw std::invalid_argument(why);
    if (v.row_count == 0U) return;
    if constexpr (Form == RowPrepForm::decode) {
        const auto values = static_cast<unsigned long long>(v.row_count) * Elements;
        hipLaunchKernelGGL((row_decode_kernel<Scale, Elements>), dim3(static_cast<unsigned>((values + 255U) / 256U)), dim3(256), 0,
                           stream, v.rows, static_cast<unsigned long long>(v.row_count), v.out);
    } else {
        if (v.row_count > 0xFFFFFFFFULL - row_prep_rows_per_block) throw std::invalid_argument("Engram row prep: row_count");
        const auto rows = static_cast<unsigned>(v.row_count);
        hipLaunchKernelGGL((row_head_rms_kernel<Scale, Elements>), dim3((rows + row_prep_rows_per_block - 1U) / row_prep_rows_per_block),
                           dim3(row_prep_threads), 0, stream, v.rows, rows, v.heads, v.norm_weight, v.eps, v.out);
    }
}

}  // namespace strix::engram::hip
