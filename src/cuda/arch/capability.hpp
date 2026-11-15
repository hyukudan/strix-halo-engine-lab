// CUDA backend, architecture layer: what the current device can do, asked once per device ordinal. Kernels choose
// an architecture path through these queries (capability), never by model. The specialisations themselves live in
// src/cuda/arch/<sm> (see src/cuda/arch/sm120/README.md).
#pragma once

#include <cuda_runtime.h>

namespace strix::cuda::arch {

// True when the current device's compute capability major is >= Major. Cached per device ordinal (0..63); a failed
// API call or an ordinal outside the cache answers false (fail closed: the caller keeps its portable path).
template <int Major>
inline bool current_device_at_least() {
    static int cap[64] = {};  // 0 unknown, 1 yes, 2 no
    int dev = -1, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (cap[dev] == 0) cap[dev] = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= Major ? 1 : 2;
    return cap[dev] == 1;
}

// Programmatic dependent launch (cudaLaunchAttributeProgrammaticStreamSerialization): compute capability 9.0+.
inline bool has_pdl() { return current_device_at_least<9>(); }
// Thread block clusters (cudaLaunchAttributeClusterDimension): compute capability 9.0+.
inline bool has_clusters() { return current_device_at_least<9>(); }

}  // namespace strix::cuda::arch
