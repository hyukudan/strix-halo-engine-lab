#pragma once
// Shared HIP KV shadow writer (contract include/strix/attn/cache_shadow.hpp). Converts committed FP32 master rows
// [from, from + rows) of K and V into the shadow form at the master's index. Bits and failure word ==
// cache_shadow::mirror_f16_ref / mirror_i8g16_ref (the failure word may gather the same bits from several threads).
// Readers that convert newer rows on the fly call the same cache_shadow::F16 / Int8G16 / encode steps.
//   mirror_f16_kernel    one thread per 4 elements of [rows][width] (K and V); block 256, grid ceil(rows*width/4/256).
//                        width % 4 == 0, master rows 16-byte aligned.
//   mirror_i8g16_kernel  one thread per 16-feature group; blockIdx.y: 0 = K, 1 = V; block 256,
//                        grid (ceil(rows*width/16/256), 2). width % 16 == 0. Tagged: the K plane checks each row once.
#include <hip/hip_runtime.h>

#include "strix/attn/cache_shadow.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::cache_shadow {
namespace cs = strix::attn::cache_shadow;

// Flag steps of the failure word, shared by the writer and by readers that convert rows the shadow does not hold yet.
__device__ inline void note_nonfinite(float x, unsigned* failure) {
    if (!cs::finite(x))
        atomicOr(failure, cs::flag_nonfinite);
}
// Int8G16 scale bits of a group maximum; an infinite or NaN scale raises flag_overflow.
__device__ inline std::uint16_t scale_flagged(float amax, unsigned* failure) {
    const auto bits = cs::Int8G16::scale_bits(amax);
    if (cs::half_special(bits))
        atomicOr(failure, cs::flag_overflow);
    return bits;
}
// F16 encode + decode of one value (a reader's conversion of a master value); an infinite or NaN half raises
// flag_overflow.
__device__ inline float f16_rounded(float x, unsigned* failure) {
    const auto h = cs::F16::encode(x);
    if (cs::half_special(h))
        atomicOr(failure, cs::flag_overflow);
    return cs::F16::decode(h);
}

template <class Tags = cs::Untagged>
__global__ void mirror_f16_kernel(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from,
                                  unsigned rows, unsigned width, std::uint16_t* hk, std::uint16_t* hv,
                                  unsigned* failure) {
    const auto index = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4U;
    if (index >= std::size_t(rows) * width)
        return;
    const auto p = from + index / width;
    if constexpr (Tags::checked)
        if (index % width == 0 && tags[p] != p)
            atomicOr(failure, cs::flag_tag);
    const auto at = std::size_t(p) * width + index % width;
    const float4 a = *reinterpret_cast<const float4*>(k + at), b = *reinterpret_cast<const float4*>(v + at);
    const float ka[4] = {a.x, a.y, a.z, a.w}, va[4] = {b.x, b.y, b.z, b.w};
    bool overflow = false;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto x = cs::F16::encode(ka[j]), y = cs::F16::encode(va[j]);
        overflow |= cs::half_special(x) || cs::half_special(y);
        hk[at + j] = x;
        hv[at + j] = y;
    }
    if (overflow)
        atomicOr(failure, cs::flag_overflow);
}

template <class Tags = cs::Untagged>
__global__ void mirror_i8g16_kernel(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from,
                                    unsigned rows, unsigned width, std::int8_t* ck, std::int8_t* cv, std::uint16_t* sk,
                                    std::uint16_t* sv, unsigned* failure) {
    constexpr unsigned G = cs::group;
    const auto group_index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned groups = width / G;
    if (group_index >= std::size_t(rows) * groups)
        return;
    const bool value = blockIdx.y != 0;
    const auto p = from + group_index / groups;
    if constexpr (Tags::checked)
        if (!value && group_index % groups == 0 && tags[p] != p)
            atomicOr(failure, cs::flag_tag);
    const auto at = std::size_t(p) * width + (group_index % groups) * G;
    const float* src = (value ? v : k) + at;
    float x[G];
    float amax = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < G; j += 4) {
        const float4 f = *reinterpret_cast<const float4*>(src + j);
        x[j] = f.x; x[j + 1] = f.y; x[j + 2] = f.z; x[j + 3] = f.w;
    }
#pragma unroll
    for (unsigned j = 0; j < G; ++j) {
        note_nonfinite(x[j], failure);
        amax = cs::Int8G16::max_step(amax, cs::Int8G16::magnitude(x[j]));
    }
    const auto bits = scale_flagged(amax, failure);
    const float d = cs::Int8G16::scale_value(bits);
    std::int8_t* codes = (value ? cv : ck) + at;
#pragma unroll
    for (unsigned j = 0; j < G; ++j)
        codes[j] = std::int8_t(cs::Int8G16::code(x[j], d));
    (value ? sv : sk)[std::size_t(p) * groups + group_index % groups] = bits;
}

}  // namespace strix::attn_core::cache_shadow
