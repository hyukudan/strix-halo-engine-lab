#pragma once

#include "core/row_transforms.hpp"

#include <cuda_runtime_api.h>
#include <cstdint>

namespace strix::cuda_core {

enum class Bf16Fp32Store { round, softcap };

template <Bf16Fp32Store Kind>
struct Bf16Store {
    float cap;
    __device__ float operator()(float value) const {
        if constexpr (Kind == Bf16Fp32Store::softcap)
            core::row_transforms::Bf16Softcap{&value, cap}(0);
        else
            core::row_transforms::Bf16Round{&value}(0);
        return value;
    }
};

// Same FP32 R1 reduction as dense_rows_multi. Output stays a float image.
cudaError_t dense_bf16_fp32_r1_epilogue(const std::uint16_t* weights, const float* input, float* output,
                                       unsigned in, unsigned out, Bf16Fp32Store store, float cap,
                                       cudaStream_t stream);

}  // namespace strix::cuda_core
