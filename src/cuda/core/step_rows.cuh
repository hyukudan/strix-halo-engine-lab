#pragma once

#include "../../core/dense_row_plan.hpp"
#include "strix/exec/descriptor.hpp"

#include <cuda_runtime.h>

namespace strix::cuda::core {

template <unsigned Rows>
struct StepRows {
    static constexpr unsigned width = Rows;
    const exec::StepBlock* descriptor;

    __device__ bool valid(unsigned row) const {
        return row < Rows && exec::step_row_valid(*descriptor, row);
    }

    __device__ unsigned position(unsigned row) const {
        return exec::step_row(*descriptor, row).position;
    }
};

template <class F>
auto dispatch_step_rows(unsigned width, F launch) {
    switch (width) {
    case 1: return launch.template operator()<1>();
    case 2: return launch.template operator()<2>();
    case 4: return launch.template operator()<4>();
    case 8: return launch.template operator()<8>();
    case 16: return launch.template operator()<16>();
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace strix::cuda::core

namespace strix::cuda_core {

// Native entry points for descriptor traits of translation-unit-local families.
// dense_bf16_fp32_step: grid y = width / core::dense_step_row_tile(width) row tiles.
cudaError_t dense_bf16_fp32_step(const float* input, unsigned in,
    std::span<const core::Bf16RowProjection> projections,
    const exec::StepBlock* descriptor, unsigned width, dim3 grid, cudaStream_t stream);

cudaError_t dense_bf16_mma3_step(const float* input, unsigned in,
    std::span<const core::Bf16RowProjection> projections, bool round,
    const exec::StepBlock* descriptor, dim3 grid, cudaStream_t stream);

cudaError_t rmsnorm_bf16_step(const float* input, const std::uint16_t* gain,
    float* output, unsigned width, float epsilon, const exec::StepBlock* descriptor,
    dim3 grid, cudaStream_t stream);

}  // namespace strix::cuda_core
