#pragma once

#include <hip/hip_runtime.h>

#ifndef STRIX_DPP_REDUCE
#define STRIX_DPP_REDUCE 1
#endif
#if STRIX_DPP_REDUCE != 0 && STRIX_DPP_REDUCE != 1
#error "STRIX_DPP_REDUCE must be 0 or 1"
#endif

namespace strix::hip_wave32_dpp {

// Shared form of the project-owned lane_xor/lane_down helpers. Exchange bits
// only: callers retain their original arithmetic and tree order.
//
// xor32: lane L reads L^D, within each logical group of 32. The DPP route
// is limited to gfx1151 wave32; wave64 and other ISAs retain HIP shuffles
// verbatim. The supported gfx1151 HIP execution contract is wave32.
// Like HIP's ds_bpermute shuffle, an inactive source supplies zero bits.
// Use zero fallback/BOUND_ZERO, not the old=value/BOUND_OFF form used by
// full-wave-only helpers: that form would return self for a masked source.
// FI is false. See AMD's ISA and cross-lane operation descriptions:
// https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/
// https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
template <unsigned D, class T>
__device__ __forceinline__ T xor32(T value) {
    static_assert(D == 1 || D == 2 || D == 4 || D == 8 || D == 16);
    static_assert(sizeof(T) == sizeof(int) || sizeof(T) == 2 * sizeof(int));
#if STRIX_DPP_REDUCE && defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    if constexpr (sizeof(T) == sizeof(int)) {
        if (__builtin_amdgcn_wavefrontsize() != 32) return __shfl_xor(value, D, 32);
        int bits;
        __builtin_memcpy(&bits, &value, sizeof(bits));
        if constexpr (D == 16)
            bits = static_cast<int>(__builtin_amdgcn_permlanex16(
                0U, static_cast<unsigned>(bits), 0x76543210U, 0xfedcba98U, false, true));
        else
            bits = __builtin_amdgcn_update_dpp(0, bits, 0x160 + D, 0xf, 0xf, true);
        T result;
        __builtin_memcpy(&result, &bits, sizeof(bits));
        return result;
    } else {
        // Keep FP64 diagnostic accumulators on their original two-word shuffle.
        return __shfl_xor(value, D, 32);
    }
#else
    return __shfl_xor(value, D, 32);
#endif
}

// Only lanes (L & 31) < D may consume this result. They read L+D, exactly
// as __shfl_down(value,D,32). For D<16 all those reads stay inside row zero;
// for D=16 they cross to row one. Upper lanes must NOT consume the result:
// row-local DPP does not implement shfl_down's cross-row and width-clamp
// behavior there. Do not use this helper in an unguarded all-lane tree.
// Source inactivity is handled identically to xor32. Whole inactive waves,
// partial blocks do not fetch stale VGPR data; wave64 uses the HIP fallback.
template <unsigned D, class T>
__device__ __forceinline__ T down_tree32(T value) {
    static_assert(D == 1 || D == 2 || D == 4 || D == 8 || D == 16);
    static_assert(sizeof(T) == sizeof(int) || sizeof(T) == 2 * sizeof(int));
#if STRIX_DPP_REDUCE && defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    if constexpr (sizeof(T) == sizeof(int)) {
        if (__builtin_amdgcn_wavefrontsize() != 32) return __shfl_down(value, D, 32);
        int bits;
        __builtin_memcpy(&bits, &value, sizeof(bits));
        if constexpr (D == 16)
            bits = static_cast<int>(__builtin_amdgcn_permlanex16(
                0U, static_cast<unsigned>(bits), 0x76543210U, 0xfedcba98U, false, true));
        else
            // ISA row_shl:D (0x100+D) reads the higher-numbered source lane.
            // row_shr:D would read L-D and reverse the required exchange.
            bits = __builtin_amdgcn_update_dpp(0, bits, 0x100 + D, 0xf, 0xf, true);
        T result;
        __builtin_memcpy(&result, &bits, sizeof(bits));
        return result;
    } else {
        return __shfl_down(value, D, 32);
    }
#else
    return __shfl_down(value, D, 32);
#endif
}

// Callers use only the five power-of-two distances admitted above. Keeping
// the fallback outside the switch restores the original dynamic HIP shuffle
// directly in OFF/wave64/FP64/other-ISA builds, without a dispatch tree.
// On the DPP route each case selects an immediate; no impossible default
// ds_bpermute path is emitted when a compiler leaves a loop rolled.
template <class T>
__device__ __forceinline__ T xor32(T value, unsigned distance) {
#if STRIX_DPP_REDUCE && defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    if constexpr (sizeof(T) == sizeof(int)) {
        if (__builtin_amdgcn_wavefrontsize() == 32) {
            switch (distance) {
            case 1: return xor32<1>(value);
            case 2: return xor32<2>(value);
            case 4: return xor32<4>(value);
            case 8: return xor32<8>(value);
            case 16: return xor32<16>(value);
            default: __builtin_unreachable();
            }
        }
    }
#endif
    return __shfl_xor(value, static_cast<int>(distance), 32);
}

// Loop counters are usually int: forward them without a sign-conversion warning at every caller.
template <class T>
__device__ __forceinline__ T xor32(T value, int distance) {
    return xor32(value, static_cast<unsigned>(distance));
}
template <class T>
__device__ __forceinline__ T down_tree32(T value, unsigned distance) {
#if STRIX_DPP_REDUCE && defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    if constexpr (sizeof(T) == sizeof(int)) {
        if (__builtin_amdgcn_wavefrontsize() == 32) {
            switch (distance) {
            case 1: return down_tree32<1>(value);
            case 2: return down_tree32<2>(value);
            case 4: return down_tree32<4>(value);
            case 8: return down_tree32<8>(value);
            case 16: return down_tree32<16>(value);
            default: __builtin_unreachable();
            }
        }
    }
#endif
    return __shfl_down(value, distance, 32);
}

} // namespace strix::hip_wave32_dpp
