#pragma once

// Shared helpers of the CUDA Mini dense fixtures (mini_dense_cuda.cu, mini_operand_class_cuda.cu).

#include "inkling2/inkling2_mini_attention_prepare_cuda.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace mini_dense_fixture {

inline void require(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::cerr << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template <class T>
struct Buffer {
    explicit Buffer(std::size_t count, T value = {}) : size(count) {
        require(cudaMallocManaged(&data, count * sizeof(T)));
        std::fill_n(data, count, value);
    }
    ~Buffer() { require(cudaFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data = nullptr;
    std::size_t size = 0;
};

// Three-plane reference of one BF16 projection on tall rows (the exact BF16 split, hi / mid / lo into one accumulator).
inline void three_plane_reference(strix::cuda_core::BigCtx* ctx, const std::uint16_t* weights, unsigned out, unsigned width, unsigned rows,
                           const float* input, float* result, strix::cuda_core::BigEpi epilogue = strix::cuda_core::BigEpi::none) {
    const strix::cuda_core::DenseWeight matrix{weights, strix::cuda_core::WeightFormat::bf16, out, width};
    if (!strix::cuda_core::dense_big_supported(matrix)) {
        std::cerr << "tall reference: matrix not supported by the big route\n";
        std::exit(1);
    }
    strix::cuda_core::BigAct act{};
    require(strix::cuda_core::dense_big_split(input, rows, width, strix::cuda_core::WeightFormat::bf16, 3, &act, nullptr, ctx));
    require(strix::cuda_core::dense_big_gemm(matrix, act, result, nullptr, epilogue, nullptr, ctx));
    // Callers read the managed result on the host (and Windows has no concurrent managed access): finish here.
    require(cudaDeviceSynchronize());
}

inline void fill_bf16_weights(Buffer<std::uint16_t>& weight, std::uint32_t& seed) {
    for (std::size_t i = 0; i < weight.size; ++i) {
        seed = 1664525U * seed + 1013904223U;
        weight.data[i] = static_cast<std::uint16_t>(0x3C00U | ((seed >> 16U) & 0x8000U) | ((seed >> 12U) & 0x01FFU));
    }
}

inline bool same_floats(const float* a, const float* b, std::size_t count) { return !std::memcmp(a, b, count * sizeof(float)); }

}  // namespace mini_dense_fixture
