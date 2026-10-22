#pragma once
// GCC math builtins used by the declared-arithmetic headers, for MSVC hosts (cl.exe, and nvcc with cl.exe as host
// compiler, whose front end does not know them either). Each maps to the correctly rounded C library / CUDA math
// function of the same operation (rintf: round to nearest even in the default mode; fmaf: single rounding; sqrtf:
// IEEE root), so host and device results are the same bits as with the builtins. __builtin_huge_valf and
// __builtin_bit_cast are MSVC intrinsics already. With any other compiler this header is empty and the builtins
// are compiled as before.
#if defined(_MSC_VER) && !defined(__clang__)
#include <cmath>

#if defined(__CUDACC__)
#define STRIX_MSVC_BUILTIN_HD __host__ __device__ __forceinline__
#else
#define STRIX_MSVC_BUILTIN_HD __forceinline
#endif

STRIX_MSVC_BUILTIN_HD float __builtin_rintf(float x) { return ::rintf(x); }
STRIX_MSVC_BUILTIN_HD float __builtin_fmaf(float a, float b, float c) { return ::fmaf(a, b, c); }
STRIX_MSVC_BUILTIN_HD float __builtin_sqrtf(float x) { return ::sqrtf(x); }
STRIX_MSVC_BUILTIN_HD constexpr float __builtin_inff() { return __builtin_bit_cast(float, 0x7f800000U); }

#undef STRIX_MSVC_BUILTIN_HD
#endif
