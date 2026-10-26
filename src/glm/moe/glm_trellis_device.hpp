#pragma once
// GLM MoE kernels: device building blocks of the strix-trellis 5acf090 EXL3 linear (mul1 codebook), copied from the
// MiniMax copy (src/minimax/moe/mmx_trellis_device.hpp, qualified bitwise against strix-trellis there) and kept
// operation for operation, so the fused kernels of src/glm/moe/ reproduce strix_gemv's bits row by row:
//   input   t = RN16(H32(RN16(RN32(x * suh))))                       (transforms.hpp input_transform)
//   core    per output tile column, per split s: lane chains of v_dot2_f32_f16 over the split's k-tiles
//           (pairs 0,1 -> sum 0, pairs 2,3 -> sum 1), then v += shfl_down(v, 2, 4); v += shfl_down(v, 1, 4)
//   finish  acc = ((+0 + p_0) + p_1) + ... + p_{S-1}; y = RN32(H32(acc) * svh)   (output_transform, alpha 1, no bias)
// The split count S is 5acf090's measured_vector_splits for <= 8 rows; for the GLM shapes (K4 / K6) no measured
// special case applies (gemv_policy.hpp), so S = vector_splits below.
// Compile with -ffp-contract=off (engine flags).
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace strix::glm::moe::dev {

using Half2 = _Float16 __attribute__((ext_vector_type(2)));

// strix-trellis 5acf090 split policy for <= 8 rows (gemv_policy.hpp measured_vector_splits; no GLM shape is special-cased).
__host__ __device__ constexpr unsigned vector_splits(unsigned input, unsigned output) {
    return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
}

// RN32 boundary before a later RN16 conversion (hadamard.hpp materialize_fp32).
__device__ __forceinline__ float materialize(float v) {
    asm volatile("" : "+v"(v));
    return v;
}

// H32 over 128 values held as lane + 32 j (hadamard.hpp hadamard128): strides 1..16 across lanes, 32 and 64 in
// registers, then one multiply by RN32(1/sqrt(128)).
__device__ __forceinline__ void hadamard128(float (&v)[4], unsigned lane) {
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float peer = __shfl_xor(v[j], stride, 32);
            v[j] = (lane & static_cast<unsigned>(stride)) ? peer - v[j] : v[j] + peer;
        }
    }
#pragma unroll
    for (int stride = 1; stride < 4; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (!(j & stride)) {
                const float l = v[j], r = v[j + stride];
                v[j] = l + r;
                v[j + stride] = l - r;
            }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = materialize(v[j] * 0x1.6a09e6p-4F);
}

__device__ __forceinline__ float half_bits(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// One wave: input transform of the 128 channels [begin, begin + 128) of one row.
__device__ __forceinline__ void transform_block(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin,
                                                unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[begin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}
// Same transform, writing the 128 results at out[obegin ...] (an LDS chunk that starts at channel begin - obegin).
__device__ __forceinline__ void transform_block_at(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin,
                                                   unsigned obegin, unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[obegin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}

// ---------------------------------------------------------------- mul1 decode (packed_decoder.hpp, qualified bitwise
// against trellis_optimized.hpp mul1_half2)
template <std::uint32_t Multiplier> __device__ __forceinline__ std::uint32_t hash16(std::uint32_t state) {
    constexpr std::uint32_t adjusted = Multiplier - 1U;
    constexpr std::uint32_t upper = (adjusted & 0xffff0000U) | 1U;
    constexpr std::uint32_t lower = adjusted & 65535U;
    std::uint32_t high, result;
    asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel:[0,0] op_sel_hi:[0,1]" : "=v"(high) : "v"(state), "s"(upper));
    asm("v_mad_u32_u16 %0, %1, %2, %1" : "=v"(result) : "v"(high), "s"(lower));
    return result;
}
__device__ __forceinline__ Half2 mul1_pair(std::uint32_t first, std::uint32_t second) {
    const std::uint32_t a = hash16<0x83dcd12dU>(first), b = hash16<0x83dcd12dU>(second);
    std::uint32_t low, both;
    constexpr std::uint32_t seed = 0x64006400U;
    asm("v_sad_u8 %0, %1, 0, %2" : "=v"(low) : "v"(a), "s"(seed));
    asm("v_sad_hi_u8 %0, %1, 0, %2" : "=v"(both) : "v"(b), "v"(low));
    const __half2 d = __hfma2(__builtin_bit_cast(__half2, both), __builtin_bit_cast(__half2, 0x1eee1eeeU),
                              __builtin_bit_cast(__half2, 0xc931c931U));
    return __builtin_bit_cast(Half2, d);
}

// Per-lane constants of the aligned 16x16 tile reader (trellis_optimized.hpp AlignedLane<Bits, true>): the lane owns
// eight consecutive states starting at bit ((lane * 8 + 1) * Bits - 16) of the tile's Bits * 8-word ring.
template <int Bits> struct LaneGeom {
    std::uint32_t oa, ob, oc;  // byte offsets of the three source words inside the tile
    std::uint32_t align;       // 32 - shift (shift is never 0 for integer Bits in 1..8)
    __device__ explicit LaneGeom(unsigned lane) {
        constexpr int count = Bits * 8;
        const int start = (static_cast<int>(lane) * 8 + 1) * Bits + 256 * Bits - 16;
        const int index = start / 32 - count, shift = start & 31;
        const int ia = index < 0 ? index + count : index;
        const int ib = index + 1 >= count ? index + 1 - count : index + 1;
        const int ic = index + 2 >= count ? index + 2 - count : index + 2;
        oa = static_cast<std::uint32_t>(ia) * 4U;
        ob = static_cast<std::uint32_t>(ib) * 4U;
        oc = static_cast<std::uint32_t>(ic) * 4U;
        align = 32U - static_cast<std::uint32_t>(shift);
    }
};

// Explicit address spaces: pointers read from memory (the Exl3 tables) are generic to the compiler, which would emit
// flat loads (64-bit lane addresses, and counted on the LDS counter too).
using GlobalU32 = const std::uint32_t __attribute__((address_space(1)));
using LocalU32 = const std::uint32_t __attribute__((address_space(3)));
// Wave-uniform copy of a pointer (scalar registers), so global loads take the saddr form.
template <class T> __device__ __forceinline__ const T* uniform(const T* p) {
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    const auto lo = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(static_cast<std::uint32_t>(v))));
    const auto hi = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(static_cast<std::uint32_t>(v >> 32U))));
    return reinterpret_cast<const T*>((static_cast<std::uintptr_t>(hi) << 32U) | lo);
}
__device__ __forceinline__ std::uint32_t load_u32(const unsigned char* base, std::uint32_t offset) {
    asm("" : "+v"(offset));  // keep a 32-bit VGPR offset on a wave-uniform base (saddr form)
    return *(GlobalU32*)(base + offset);
}

template <int Bits> struct TileStates {
    std::uint32_t high, low, tail;
    TileStates() = default;
    __device__ __forceinline__ TileStates(const unsigned char* tile, const LaneGeom<Bits>& g) {
        const std::uint32_t a = load_u32(tile, g.oa), b = load_u32(tile, g.ob), c = load_u32(tile, g.oc);
        high = __builtin_amdgcn_alignbit(a, b, g.align);  // (a << shift) | (b >> (32 - shift))
        low = __builtin_amdgcn_alignbit(b, c, g.align);
        tail = c << (32U - g.align);
    }
    template <int J> __device__ __forceinline__ std::uint32_t state() const {
        constexpr int offset = J * Bits;
        if constexpr (offset <= 16) return (high >> (16 - offset)) & 65535U;
        else if constexpr (offset < 32) return ((high << (offset - 16)) | (low >> (48 - offset))) & 65535U;
        else if constexpr (offset <= 48) return (low >> (48 - offset)) & 65535U;
        else return ((low << (offset - 48)) | (tail >> (80 - offset))) & 65535U;
    }
};

// The 4-lane reduction of one sum; lane (lane & 3) == 0 holds output tile_col * 16 + lane / 4 + h * 8.
__device__ __forceinline__ float reduce4(float v) {
    v += __shfl_down(v, 2, 4);
    v += __shfl_down(v, 1, 4);
    return v;
}

// Prefetched form: the raw words and activation pairs of the next D k-tiles stay in flight in a register ring (the
// activations ride in the same ring because vector-memory counters retire in issue order). Same arithmetic, same bits.
template <int Bits, int R> struct Slot {
    std::uint32_t a, b, c;
    Half2 act[R][2];
};

template <int Bits, int R>
__device__ __forceinline__ void compute_slot(const Slot<Bits, R>& s, const LaneGeom<Bits>& g, unsigned nrows,
                                             float (&sums)[R][2]) {
    TileStates<Bits> t;
    t.high = __builtin_amdgcn_alignbit(s.a, s.b, g.align);
    t.low = __builtin_amdgcn_alignbit(s.b, s.c, g.align);
    t.tail = s.c << (32U - g.align);
    const Half2 w0 = mul1_pair(t.template state<0>(), t.template state<1>());
    const Half2 w1 = mul1_pair(t.template state<2>(), t.template state<3>());
    const Half2 w2 = mul1_pair(t.template state<4>(), t.template state<5>());
    const Half2 w3 = mul1_pair(t.template state<6>(), t.template state<7>());
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) {
            sums[r][0] = __builtin_amdgcn_fdot2(w0, s.act[r][0], sums[r][0], false);
            sums[r][0] = __builtin_amdgcn_fdot2(w1, s.act[r][1], sums[r][0], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w2, s.act[r][0], sums[r][1], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w3, s.act[r][1], sums[r][1], false);
        }
}

}  // namespace strix::glm::moe::dev
