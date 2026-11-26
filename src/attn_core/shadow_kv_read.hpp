#pragma once
// Shared HIP KV-format trait of shadow readers (contract include/strix/attn/cache_shadow.hpp: Form, ShadowRead; writer
// src/attn_core/cache_shadow.hpp). A decode attention that reads K / V through a reduced shadow calls these steps; the
// form is a template trait (Form::f16 | Form::kv8), never a model-local copy. Values are those of the shared encode:
//   load_shadow     Count consecutive elements [at, at + Count) of the K (or V) row p of a shadow row (p < mirrored).
//   convert_master  the same Count elements converted in registers from the FP32 master row (rows the shadow does not
//                   hold yet), raising the writer's failure flags; equal to the values the writer stores for that row.
//   fetch_raw / decode_raw  load_shadow split into loads issued ahead of their use and the decode (same values).
// Count 16 = one lane's own group; Count 8 = half a group held by two consecutive lanes (lane bit 0): convert_master
// combines their maxima by one xor shuffle, so every lane of the wave must call it with the same row.
#include <hip/hip_runtime.h>

#include "attn_core/cache_shadow.hpp"
#include "strix/attn/cache_shadow.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::shadow_kv {
namespace cs = strix::attn::cache_shadow;
namespace sc = strix::attn_core::cache_shadow;
using Form = cs::Form;
using ShadowRead = cs::ShadowRead;

__device__ inline float half_value(unsigned bits16) { return cs::F16::decode((unsigned short)bits16); }

template <unsigned Count, Form Format>
__device__ inline void load_shadow(const ShadowRead& m, std::uint64_t p, bool value, unsigned at, float (&out)[Count]) {
    static_assert(Count == 8U || Count == 16U);
    if constexpr (Format == Form::f16) {
        const auto* src = static_cast<const std::uint16_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
        if constexpr (Count == 16U) {
#pragma unroll
            for (unsigned c = 0; c < 2; ++c) {
                const uint4 u = *reinterpret_cast<const uint4*>(src + 8U * c);
                const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    out[8U * c + 2U * i] = half_value(w[i] & 0xffffU);
                    out[8U * c + 2U * i + 1U] = half_value(w[i] >> 16U);
                }
            }
        } else {
            const uint4 u = *reinterpret_cast<const uint4*>(src);
            const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                out[2U * i] = half_value(w[i] & 0xffffU);
                out[2U * i + 1U] = half_value(w[i] >> 16U);
            }
        }
    } else {
        const auto* codes = static_cast<const std::int8_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
        const float d = cs::Int8G16::scale_value((value ? m.v_scale : m.k_scale)[std::size_t(p) * (m.width / cs::group) +
                                                                                 at / cs::group]);
        unsigned w[Count / 4U];
        if constexpr (Count == 16U) {
            const uint4 u = *reinterpret_cast<const uint4*>(codes);
            w[0] = u.x; w[1] = u.y; w[2] = u.z; w[3] = u.w;
        } else {
            const uint2 u = *reinterpret_cast<const uint2*>(codes);
            w[0] = u.x; w[1] = u.y;
        }
#pragma unroll
        for (unsigned c = 0; c < Count / 4U; ++c)
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                out[4U * c + i] = cs::Int8G16::decode(int(std::int8_t((w[c] >> (8U * i)) & 0xffU)), d);
    }
}

// row = the master row's element `at` (16-byte aligned FP32).
template <unsigned Count, Form Format>
__device__ inline void convert_master(const float* row, float (&out)[Count], unsigned* failure) {
    static_assert(Count == 8U || Count == 16U);
#pragma unroll
    for (unsigned c = 0; c < Count / 4U; ++c) {
        const float4 f = *reinterpret_cast<const float4*>(row + 4U * c);
        out[4U * c] = f.x;
        out[4U * c + 1U] = f.y;
        out[4U * c + 2U] = f.z;
        out[4U * c + 3U] = f.w;
    }
    if constexpr (Format == Form::f16) {
#pragma unroll
        for (unsigned i = 0; i < Count; ++i)
            out[i] = sc::f16_rounded(out[i], failure);
    } else {
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < Count; ++i) {
            sc::note_nonfinite(out[i], failure);
            amax = cs::Int8G16::max_step(amax, cs::Int8G16::magnitude(out[i]));
        }
        if constexpr (Count == 8U) {
            const float other = __shfl_xor(amax, 1, 32);
            amax = cs::Int8G16::max_step(amax, other);
        }
        const float d = cs::Int8G16::scale_value(sc::scale_flagged(amax, failure));
#pragma unroll
        for (unsigned i = 0; i < Count; ++i)
            out[i] = cs::Int8G16::decode(cs::Int8G16::code(out[i], d), d);
    }
}

// Raw shadow operands of one lane: Count 16 = 32 B of halves or 16 B of codes + scale, Count 8 = 16 B of halves or
// 8 B of codes + scale.
template <unsigned Count, Form Format>
struct RawKv {
    uint4 a, b;
    unsigned short scale;
};
template <unsigned Count, Form Format>
__device__ inline RawKv<Count, Format> fetch_raw(const ShadowRead& m, std::uint64_t p, bool value, unsigned at) {
    RawKv<Count, Format> raw{};
    if constexpr (Format == Form::f16) {
        const auto* src = static_cast<const std::uint16_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
        raw.a = *reinterpret_cast<const uint4*>(src);
        if constexpr (Count == 16U)
            raw.b = *reinterpret_cast<const uint4*>(src + 8U);
    } else {
        const auto* codes = static_cast<const std::int8_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
        if constexpr (Count == 16U)
            raw.a = *reinterpret_cast<const uint4*>(codes);
        else {
            const uint2 u = *reinterpret_cast<const uint2*>(codes);
            raw.a.x = u.x;
            raw.a.y = u.y;
        }
        raw.scale = (value ? m.v_scale : m.k_scale)[std::size_t(p) * (m.width / cs::group) + at / cs::group];
    }
    return raw;
}
template <unsigned Count, Form Format>
__device__ inline void decode_raw(const RawKv<Count, Format>& raw, float (&out)[Count]) {
    if constexpr (Format == Form::f16) {
        const unsigned w[8] = {raw.a.x, raw.a.y, raw.a.z, raw.a.w, raw.b.x, raw.b.y, raw.b.z, raw.b.w};
#pragma unroll
        for (unsigned i = 0; i < Count / 2U; ++i) {
            out[2U * i] = half_value(w[i] & 0xffffU);
            out[2U * i + 1U] = half_value(w[i] >> 16U);
        }
    } else {
        const float d = cs::Int8G16::scale_value(raw.scale);
        const unsigned w[4] = {raw.a.x, raw.a.y, raw.a.z, raw.a.w};
#pragma unroll
        for (unsigned c = 0; c < Count / 4U; ++c)
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                out[4U * c + i] = cs::Int8G16::decode(int(std::int8_t((w[c] >> (8U * i)) & 0xffU)), d);
    }
}

}  // namespace strix::attn_core::shadow_kv
