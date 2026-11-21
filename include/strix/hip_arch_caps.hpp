#pragma once

// Instruction/fragment availability, not tuning or numerical qualification.
// Keep target detection in one place. Unknown targets fail closed.
// Indirection so host tests can model builtin availability without touching
// the compiler-reserved __has_builtin.
#ifndef STRIX_HAS_BUILTIN
#ifdef __has_builtin
#define STRIX_HAS_BUILTIN(x) __has_builtin(x)
#else
#define STRIX_HAS_BUILTIN(x) 0
#endif
#endif

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define STRIX_HIP_ARCH_RDNA3 1
#else
#define STRIX_HIP_ARCH_RDNA3 0
#endif
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define STRIX_HIP_ARCH_RDNA35 1
#else
#define STRIX_HIP_ARCH_RDNA35 0
#endif
#define STRIX_HIP_ARCH_RDNA3_LAYOUT (STRIX_HIP_ARCH_RDNA3 || STRIX_HIP_ARCH_RDNA35)
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRIX_HIP_ARCH_RDNA4 1
#else
#define STRIX_HIP_ARCH_RDNA4 0
#endif
#if defined(__gfx942__) || defined(__gfx950__)
#define STRIX_HIP_ARCH_CDNA 1
#else
#define STRIX_HIP_ARCH_CDNA 0
#endif
#if STRIX_HIP_ARCH_RDNA3_LAYOUT || STRIX_HIP_ARCH_RDNA4
#define STRIX_HIP_TARGET_WAVE_SIZE 32
#elif STRIX_HIP_ARCH_CDNA
#define STRIX_HIP_TARGET_WAVE_SIZE 64
#else
#define STRIX_HIP_TARGET_WAVE_SIZE 0
#endif

// The packed IU4/IU8 operand ABI used by integer_projection is RDNA 3.5 only.
#define STRIX_HAS_RDNA35_WMMA (STRIX_HIP_ARCH_RDNA35 && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32) && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32))
#define STRIX_HAS_WMMA_F16_W32 (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32))
#define STRIX_HAS_WMMA_BF16_W32 (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32))
#define STRIX_HAS_WMMA_IU8_W32 (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32))
#define STRIX_HAS_WMMA_IU4_W32 (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32))
#define STRIX_HAS_PERMLANEX16 (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_permlanex16))
#define STRIX_HAS_DPP_ROW (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_update_dpp))
#define STRIX_HAS_SUDOT4 (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_sudot4))
#define STRIX_HAS_UDOT4 (STRIX_HIP_ARCH_RDNA3_LAYOUT && \
    STRIX_HAS_BUILTIN(__builtin_amdgcn_udot4))
