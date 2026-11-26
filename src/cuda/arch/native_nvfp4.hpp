#pragma once

#include "strix/cuda/arch/native_images.hpp"

#include <cuda_runtime_api.h>

#include <atomic>

namespace strix::cuda::arch {

// The build can link portable and native libraries in the same process.
[[maybe_unused]] static cudaError_t native_nvfp4_status() {
#if defined(STRIX_CUDA_NATIVE_NVFP4)
    constexpr NativeNvfp4Images images{
#if defined(STRIX_CUDA_NVFP4_SM120)
        true,
#else
        false,
#endif
#if defined(STRIX_CUDA_NVFP4_SM121)
        true,
#else
        false,
#endif
    };
    static std::atomic<int> cache[64]{};  // 0 unknown, 1 supported, 2 unsupported
    int device = 0;
    if (const auto e = cudaGetDevice(&device); e != cudaSuccess) return e;
    if (device >= 0 && device < 64) {
        const int value = cache[device].load(std::memory_order_relaxed);
        if (value != 0) return value == 1 ? cudaSuccess : cudaErrorNotSupported;
    }
    int major = 0, minor = 0;
    if (const auto e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); e != cudaSuccess) return e;
    if (const auto e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device); e != cudaSuccess) return e;
    const bool supported = images.supports(major, minor);
    if (device >= 0 && device < 64) cache[device].store(supported ? 1 : 2, std::memory_order_relaxed);
    return supported ? cudaSuccess : cudaErrorNotSupported;
#else
    return cudaErrorNotSupported;
#endif
}

}  // namespace strix::cuda::arch
