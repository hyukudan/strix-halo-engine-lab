#pragma once

#include "cuda/core/grouped_microscaled.cuh"
#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "inkling2_mini_schedule.hpp"
#include "moe_core/expert_activation.hpp"
#include "moe_core/grouped_rows.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "cuda/core/route_wave.cuh"
#include "strix/cuda_core/dense.hpp"
#endif

namespace strix::inkling2 {

namespace mini_cuda_moe_detail {

namespace grouped = cuda::core::grouped_microscaled;
using weight::microscaled::MatrixFormat;

inline bool fits_tile(unsigned physical_layer, MiniExpertProjection projection, const grouped::TileRows& tile) {
    const auto geometry = mini_expert_geometry(projection);
    if (physical_layer >= kPhysicalLayers || geometry.inputs == 0 || tile.weights.row_order != mini_expert_row_order(projection)) return false;
    if (mini_expert_format(physical_layer) == MatrixFormat::nvfp4) {
        return grouped::fits_tile<MatrixFormat::nvfp4>(tile, geometry.inputs, geometry.outputs);
    }
    return grouped::fits_tile<MatrixFormat::mxfp8>(tile, geometry.inputs, geometry.outputs);
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
struct Router {
    static constexpr auto policy = kMiniRoute;
    using Math = detail::MiniMoeMath;
};

inline cudaError_t project(unsigned physical_layer, MiniExpertProjection projection,
                           const grouped::TileRows* tiles, unsigned count, cudaStream_t stream,
                           const unsigned* tile_count = nullptr, const unsigned* failure = nullptr) {
    const auto geometry = mini_expert_geometry(projection);
    if (physical_layer >= kPhysicalLayers || geometry.inputs == 0) return cudaErrorInvalidValue;
    const grouped::Args args{tiles, count, geometry.inputs, geometry.outputs, tile_count, failure};
    if (mini_expert_format(physical_layer) == MatrixFormat::nvfp4) return grouped::launch<MatrixFormat::nvfp4>(args, stream);
    return grouped::launch<MatrixFormat::mxfp8>(args, stream);
}
#endif

}  // namespace mini_cuda_moe_detail

struct MiniCudaMoeBackend {
    static constexpr unsigned rows_per_tile = mini_cuda_moe_detail::grouped::kRowsPerTile;
    static constexpr auto fits = mini_cuda_moe_detail::grouped::fits;
    static constexpr auto fits_tile = mini_cuda_moe_detail::fits_tile;

#if defined(__CUDACC__) && !defined(__HIPCC__)
    using Status = cudaError_t;
    using Stream = cudaStream_t;
    static constexpr auto success = cudaSuccess;
    static constexpr auto invalid_value = cudaErrorInvalidValue;
    static constexpr auto gather = moe_core::grouped_rows::gather;
    static constexpr auto combine = moe_core::grouped_rows::combine;
    static constexpr auto activate = moe_core::expert_activation::launch;
    static constexpr auto project = mini_cuda_moe_detail::project;
#endif
};

#if defined(__CUDACC__) && !defined(__HIPCC__)

// Router logits stay FP32, including the projection accumulation. Reuse
// the common dense decode GEMV in ordered blocks of at most kMaxRows.
// The caller may instead materialize FP32 logits with a larger-M producer.
inline cudaError_t enqueue_mini_cuda_router_logits(
    const cuda_core::DenseWeight& router, unsigned rows, std::span<const float> input,
    std::span<float> logits, cudaStream_t stream) {
    const auto input_elements = std::size_t{rows} * kHiddenWidth;
    const auto output_elements = std::size_t{rows} * kExperts;
    if (rows == 0 || rows > kMaxContext || !router.data ||
        router.in != kHiddenWidth || router.out != kExperts ||
        (router.format != cuda_core::WeightFormat::bf16 && router.format != cuda_core::WeightFormat::f32) ||
        input.size() < input_elements || logits.size() < output_elements) return cudaErrorInvalidValue;
    for (unsigned first = 0; first < rows;) {
        const unsigned count = std::min(cuda_core::kMaxRows, rows - first);
        const auto status = cuda_core::dense_rows(
            router, input.data() + std::size_t{first} * kHiddenWidth,
            logits.data() + std::size_t{first} * kExperts, count, nullptr, stream, cuda_core::DenseArith::fp32);
        if (status != cudaSuccess) return status;
        first += count;
    }
    return cudaSuccess;
}

using MiniCudaResidentMoePlan = moe_core::device_grouped::Planner<kExperts, kExpertsPerToken>;

// Run selection from precomputed device logits, build expert-major CSR,
// materialize the live descriptors and execute all grouped stages on stream.
// The device failure flag distinguishes a rejected route from an enqueue
// error. Invalid route contents leave output unwritten.
inline cudaError_t enqueue_mini_cuda_resident_moe(
    unsigned physical_layer, unsigned rows, std::span<const float> input, std::span<const float> logits,
    std::span<const float> selection_bias, std::span<float> output, MiniMoeResidentRouteBuffers route,
    std::span<const MiniMoeExpert> matrices, moe_core::device_grouped::CsrTables csr,
    moe_core::device_grouped::Output planned, MiniMoeWorkspace workspace, cudaStream_t stream) {
    namespace plan = moe_core::device_grouped;
    plan::Args args{};
    if (input.size() < std::size_t{rows} * kHiddenWidth || output.size() < std::size_t{rows} * kHiddenWidth ||
        logits.size() < std::size_t{rows} * kExperts || selection_bias.size() != kExperts ||
        !prepare_mini_resident_moe<MiniCudaMoeBackend>(physical_layer, rows, route, matrices, csr, planned, workspace, args) ||
        route.scratch.size() < MiniCudaResidentMoePlan::scratch_bytes(rows) / sizeof(unsigned)) return cudaErrorInvalidValue;
    auto status = cuda::core::route_wave::launch<mini_cuda_moe_detail::Router>(
        logits.data(), selection_bias.data(), route.ids.data(), route.weights.data(), route.status.data(), rows, stream);
    if (status != cudaSuccess) return status;
    status = MiniCudaResidentMoePlan::enqueue(args, rows, kExperts, kExpertsPerToken, args.rows_per_tile,
                                             route.scratch.data(), stream);
    if (status != cudaSuccess) return status;

    constexpr unsigned threads = 256U;
    const unsigned blocks = (rows * kExpertsPerToken - 1U) / threads + 1U;
    plan::materialize<kExpertsPerToken><<<blocks, threads, 0, stream>>>(args);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    return enqueue_planned_mini_moe<MiniCudaMoeBackend>(physical_layer, args, input, output, workspace, stream);
}

#endif

}  // namespace strix::inkling2
