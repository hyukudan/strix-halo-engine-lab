#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
#include "strix/device/decode_geometry.hpp"  // the launch plan of elements(), shared by both device languages

#include <cstddef>
#include <limits>

namespace strix::core::pointwise {

#if defined(__HIPCC__)
using Status = hipError_t;
using Stream = hipStream_t;
inline constexpr auto invalid = hipErrorInvalidValue;
inline Status last_error() { return hipGetLastError(); }
#else
using Status = cudaError_t;
using Stream = cudaStream_t;
inline constexpr auto invalid = cudaErrorInvalidValue;
inline Status last_error() { return cudaGetLastError(); }
#endif

// The single-grid form preserves the legacy CUDA add launch geometry. Wider
// views use a bounded grid with grid-stride indexing.
template <bool GridStride, class Index, class Transform>
__device__ __forceinline__ void visit_elements(Index size, const Transform& transform) {
    const Index start = Index{blockIdx.x} * blockDim.x + threadIdx.x;
    if constexpr (GridStride) {
        const Index stride = Index{gridDim.x} * blockDim.x;
        for (Index i = start; i < size; i += stride) transform(i);
    } else if (start < size) {
        transform(start);
    }
}

template <class Transform>
__global__ void elements_kernel(std::size_t size, Transform transform) {
    visit_elements<true>(size, transform);
}

template <class Transform>
Status elements(std::size_t size, Transform transform, Stream stream) {
    if (size == 0 || size > std::numeric_limits<std::size_t>::max() / sizeof(float)) return invalid;
    const device::LaunchGeometry geometry = device::decode_geometry::pointwise(size);
    elements_kernel<<<geometry.grid<dim3>(), geometry.block<dim3>(), 0, stream>>>(size, transform);
    return last_error();
}

struct Add {
    const float* lhs;
    const float* rhs;
    float* output;

    template <class Index>
    __device__ void operator()(Index i) const { output[i] = __fadd_rn(lhs[i], rhs[i]); }
};

}  // namespace strix::core::pointwise
