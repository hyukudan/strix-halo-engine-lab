#pragma once

#include "../hip_arch_caps.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace strix::hip::arch {

#if defined(__HIP_DEVICE_COMPILE__)
template <class Result, class... Args>
__device__ __forceinline__ Result unavailable(Args...) {
    __builtin_trap();
}
#endif

#if defined(__HIP_DEVICE_COMPILE__) && STRIX_HIP_ARCH_RDNA4
// The existing selectors are the identity within a row. Each lane reads its
// peer in the other 16-lane row. Preserve the inactive-source behavior.
template <bool FetchInactive, bool BoundZero>
__device__ __forceinline__ unsigned exchange16(unsigned old, unsigned source) {
    if constexpr (FetchInactive) {
        return unavailable<unsigned>(old, source);
    } else {
        const auto peer = static_cast<unsigned>(__lane_id()) ^ 16U;
        const auto active = static_cast<unsigned>(__ballot(1));
        const unsigned value = __shfl_xor(source, 16, 32);
        return (active & (1U << peer)) != 0U ? value : (BoundZero ? 0U : old);
    }
}
#endif

// With clamping disabled the instruction adds modulo 2^32. Form every byte
// product in int, then perform all additions in unsigned arithmetic.
template <bool SignedA, bool SignedB>
#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__
#endif
constexpr unsigned dot4(unsigned a, unsigned b, unsigned acc) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        const unsigned aa = a >> shift;
        const unsigned bb = b >> shift;
        const int av = static_cast<int>(aa & (SignedA ? 127U : 255U)) -
                       static_cast<int>(SignedA ? aa & 128U : 0U);
        const int bv = static_cast<int>(bb & (SignedB ? 127U : 255U)) -
                       static_cast<int>(SignedB ? bb & 128U : 0U);
        acc += static_cast<unsigned>(av * bv);
    }
    return acc;
}

}  // namespace strix::hip::arch

// Keep native expansions identical, including immediate operands. The host
// pass and the existing CPU decoder shim parse the original expressions.
#define STRIX_HIP_TRAP_ACC(c, ...) ::strix::hip::arch::unavailable<decltype(c)>(c, __VA_ARGS__)
#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HAS_WMMA_F16_W32
#define STRIX_HIP_WMMA_F16 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
#else
#define STRIX_HIP_WMMA_F16(a, b, c) STRIX_HIP_TRAP_ACC(c, a, b)
#endif
#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HAS_WMMA_BF16_W32
#define STRIX_HIP_WMMA_BF16 __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32
#else
#define STRIX_HIP_WMMA_BF16(a, b, c) STRIX_HIP_TRAP_ACC(c, a, b)
#endif
#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HAS_WMMA_IU4_W32
#define STRIX_HIP_WMMA_IU4 __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32
#else
#define STRIX_HIP_WMMA_IU4(sa, a, sb, b, c, clamp) STRIX_HIP_TRAP_ACC(c, sa, a, sb, b, clamp)
#endif
#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HAS_WMMA_IU8_W32
#define STRIX_HIP_WMMA_IU8 __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32
#else
#define STRIX_HIP_WMMA_IU8(sa, a, sb, b, c, clamp) STRIX_HIP_TRAP_ACC(c, sa, a, sb, b, clamp)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HAS_PERMLANEX16
#define STRIX_HIP_EXCHANGE16(old, src, fi, bc) \
    __builtin_amdgcn_permlanex16(old, src, 0x76543210U, 0xfedcba98U, fi, bc)
#elif STRIX_HIP_ARCH_RDNA4
#define STRIX_HIP_EXCHANGE16(old, src, fi, bc) ::strix::hip::arch::exchange16<fi, bc>(old, src)
#else
#define STRIX_HIP_EXCHANGE16(old, src, fi, bc) ::strix::hip::arch::unavailable<unsigned>(old, src, fi, bc)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HAS_SUDOT4
#define STRIX_HIP_SUDOT4 __builtin_amdgcn_sudot4
#elif STRIX_HIP_ARCH_RDNA4
#define STRIX_HIP_SUDOT4(sa, a, sb, b, c, clamp) \
    ((clamp) ? ::strix::hip::arch::unavailable<int>(a, b, c) : \
     __builtin_bit_cast(int, (::strix::hip::arch::dot4<sa, sb>( \
         static_cast<unsigned>(a), static_cast<unsigned>(b), static_cast<unsigned>(c)))))
#else
#define STRIX_HIP_SUDOT4(sa, a, sb, b, c, clamp) ::strix::hip::arch::unavailable<int>(sa, a, sb, b, c, clamp)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HAS_UDOT4
#define STRIX_HIP_UDOT4 __builtin_amdgcn_udot4
#elif STRIX_HIP_ARCH_RDNA4
#define STRIX_HIP_UDOT4(a, b, c, clamp) \
    ((clamp) ? ::strix::hip::arch::unavailable<unsigned>(a, b, c) : ::strix::hip::arch::dot4<false, false>(a, b, c))
#else
#define STRIX_HIP_UDOT4(a, b, c, clamp) ::strix::hip::arch::unavailable<unsigned>(a, b, c, clamp)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HIP_ARCH_RDNA3_LAYOUT
#define STRIX_HIP_RDNA3_ASM(...) __VA_ARGS__
#define STRIX_HIP_PERMLANE16 __builtin_amdgcn_permlane16
#define STRIX_HIP_PERMLANE64 __builtin_amdgcn_permlane64
#define STRIX_HIP_FDOT2 __builtin_amdgcn_fdot2
#define STRIX_HIP_BUFFER_RSRC __builtin_amdgcn_make_buffer_rsrc
#define STRIX_HIP_RAW_LOAD_B16 __builtin_amdgcn_raw_buffer_load_b16
#define STRIX_HIP_RAW_LOAD_B32 __builtin_amdgcn_raw_buffer_load_b32
#define STRIX_HIP_RAW_LOAD_B128 __builtin_amdgcn_raw_buffer_load_b128
#else
#define STRIX_HIP_RDNA3_ASM(...) do { \
    if constexpr (false) { __VA_ARGS__; } \
    __builtin_trap(); \
} while (false)
#define STRIX_HIP_PERMLANE16(...) ::strix::hip::arch::unavailable<unsigned>(__VA_ARGS__)
#define STRIX_HIP_PERMLANE64(...) ::strix::hip::arch::unavailable<int>(__VA_ARGS__)
#define STRIX_HIP_FDOT2(...) ::strix::hip::arch::unavailable<float>(__VA_ARGS__)
#define STRIX_HIP_BUFFER_RSRC(...) ::strix::hip::arch::unavailable<__amdgpu_buffer_rsrc_t>(__VA_ARGS__)
#define STRIX_HIP_RAW_LOAD_B16(...) ::strix::hip::arch::unavailable<short>(__VA_ARGS__)
#define STRIX_HIP_RAW_LOAD_B32(...) ::strix::hip::arch::unavailable<int>(__VA_ARGS__)
typedef int StrixRawBuffer128 __attribute__((ext_vector_type(4)));
#define STRIX_HIP_RAW_LOAD_B128(...) ::strix::hip::arch::unavailable<StrixRawBuffer128>(__VA_ARGS__)
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || STRIX_HIP_TARGET_WAVE_SIZE == 32
#define STRIX_HIP_BALLOT32 __builtin_amdgcn_ballot_w32
#else
#define STRIX_HIP_BALLOT32(...) ::strix::hip::arch::unavailable<unsigned>(__VA_ARGS__)
#endif
