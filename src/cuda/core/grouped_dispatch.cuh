#pragma once

#include "cuda/core/grouped_microscaled.cuh"
#include "moe_core/grouped_dispatch.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "strix/cuda_core/dense.hpp"
#endif

namespace strix::cuda::core::grouped_dispatch {

struct CudaBackend {
    static constexpr unsigned rows_per_tile = grouped_microscaled::kRowsPerTile;
    static constexpr auto fits = grouped_microscaled::fits;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto fits_matrix_tile = grouped_microscaled::fits_tile<Format>;

#if defined(__CUDACC__) && !defined(__HIPCC__)
    using Status = cudaError_t;
    using Stream = cudaStream_t;
    static constexpr auto success = cudaSuccess;
    static constexpr auto invalid_value = cudaErrorInvalidValue;
    static constexpr auto gather = moe_core::grouped_rows::gather;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto launch_projection = grouped_microscaled::launch<Format>;

    template<unsigned Experts, unsigned TopK>
    static Status materialize(const moe_core::device_grouped::Args& args,
                              std::span<unsigned> scratch, Stream stream) {
        namespace plan = moe_core::device_grouped;
        using Planner = plan::Planner<Experts, TopK>;
        const auto status = Planner::enqueue(args, args.rows, Experts, TopK, args.rows_per_tile,
                                             scratch.data(), stream);
        if (status != success) return status;
        constexpr unsigned threads = 256U;
        const unsigned blocks = (args.rows * TopK - 1U) / threads + 1U;
        plan::materialize<TopK><<<blocks, threads, 0, stream>>>(args);
        return cudaGetLastError();
    }
#endif
};

#if defined(__CUDACC__) && !defined(__HIPCC__)
// Preserve FP32 router projection accumulation and partition rows in input order.
template<class Config>
inline cudaError_t router_logits(const cuda_core::DenseWeight& router, unsigned rows,
                                 std::span<const float> input, std::span<float> logits, cudaStream_t stream) {
    const auto input_elements = std::size_t{rows} * Config::hidden;
    const auto output_elements = std::size_t{rows} * Config::experts;
    if (rows == 0U || rows > Config::max_rows || !router.data ||
        router.in != Config::hidden || router.out != Config::experts ||
        (router.format != cuda_core::WeightFormat::bf16 && router.format != cuda_core::WeightFormat::f32) ||
        input.size() < input_elements || logits.size() < output_elements) return cudaErrorInvalidValue;
    for (unsigned first = 0; first < rows;) {
        const unsigned count = std::min(cuda_core::kMaxRows, rows - first);
        const auto status = cuda_core::dense_rows(
            router, input.data() + std::size_t{first} * Config::hidden,
            logits.data() + std::size_t{first} * Config::experts, count, nullptr, stream, cuda_core::DenseArith::fp32);
        if (status != cudaSuccess) return status;
        first += count;
    }
    return cudaSuccess;
}
#endif

}  // namespace strix::cuda::core::grouped_dispatch
