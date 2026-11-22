#pragma once

#include "cuda/core/moe_microscaled_r1.cuh"
#include "moe_core/microscaled_r1.hpp"

// Fixture-only binding to the existing gate/up and down kernels. Production
// can call the same schedule in the CUDA branch of detail::dot_rows.
namespace strix::moe_core::microscaled_r1::detail {

template <>
__device__ __forceinline__ void dot_rows<packed::MatrixFormat::nvfp4, 1U>(
    const packed::PackedMatrix& matrix, unsigned col, unsigned stride, unsigned live,
    const float* input, unsigned width, unsigned lane, float (&out)[2]) {
    cuda_body::dot_rows<packed::MatrixFormat::nvfp4, 1U>(matrix, col, stride, live, input, width, lane, out);
}

template <>
__device__ __forceinline__ void dot_rows<packed::MatrixFormat::mxfp8, 1U>(
    const packed::PackedMatrix& matrix, unsigned col, unsigned stride, unsigned live,
    const float* input, unsigned width, unsigned lane, float (&out)[2]) {
    cuda_body::dot_rows<packed::MatrixFormat::mxfp8, 1U>(matrix, col, stride, live, input, width, lane, out);
}

template <>
__device__ __forceinline__ void dot_rows<packed::MatrixFormat::nvfp4, 1U>(
    const packed::PackedMatrix& matrix, unsigned col, unsigned stride, unsigned live,
    const std::uint16_t* input, unsigned width, unsigned lane, float (&out)[1]) {
    cuda_body::dot_rows<packed::MatrixFormat::nvfp4, 1U>(matrix, col, stride, live, input, width, lane, out);
}

template <>
__device__ __forceinline__ void dot_rows<packed::MatrixFormat::mxfp8, 1U>(
    const packed::PackedMatrix& matrix, unsigned col, unsigned stride, unsigned live,
    const std::uint16_t* input, unsigned width, unsigned lane, float (&out)[1]) {
    cuda_body::dot_rows<packed::MatrixFormat::mxfp8, 1U>(matrix, col, stride, live, input, width, lane, out);
}

} // namespace strix::moe_core::microscaled_r1::detail
