#pragma once

#include "core/bf16_lm_head.hpp"
#include "strix/weight/resident_slices.hpp"

#include <limits>

#if defined(__HIPCC__)
#include "core/dense_linear.hpp"
#else
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#endif

namespace strix::core {

#if defined(__HIPCC__)
namespace detail {
template <unsigned Bucket>
void launch_q8_tight_rows(dense_linear::View weights, const float* input, float* output,
                         unsigned rows, row_transforms::Stream stream) {
    constexpr unsigned tile = Bucket <= 8U ? 2U : 4U;
    constexpr unsigned block_outputs = 4U * tile;
    dense_linear::kernel<Bucket, tile, weight_decode::Format::q8_0, dense_linear::Epilogue::store_tight_rows>
        <<<(weights.outputs + block_outputs - 1U) / block_outputs, 128, 0, stream>>>(
            weights, weights, input, output, rows);
}
}  // namespace detail
#endif

// Bind compact resident weights to the native dense family. Only live output
// rows are projected. The caller owns the activation/output BF16 boundaries.
// CUDA callers can supply a reserved scratch context for aligned tall shapes.
inline row_transforms::Status enqueue_resident_q8_rows(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, float* output, row_transforms::Stream stream
#if !defined(__HIPCC__)
    , cuda_core::BigCtx* context = nullptr
#endif
) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (matrix.storage != weight::kQ8WeightStorage || !input || !output || input == output ||
        reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
        !matrix.bytes.data() || reinterpret_cast<std::uintptr_t>(matrix.bytes.data()) % 16U != 0U ||
        rows == 0 || live_outputs == 0 || live_outputs > matrix.rows ||
        live_outputs > std::numeric_limits<unsigned>::max() - 31U ||
        live_outputs > max_floats / rows || matrix.columns == 0 || matrix.columns > max_floats / rows ||
        matrix.columns > std::numeric_limits<unsigned>::max() - 31U || matrix.columns % weight::Q8Storage::elements ||
        matrix.row_bytes != weight::Q8Storage::row_bytes(matrix.columns) ||
        matrix.rows > matrix.bytes.size() / matrix.row_bytes)
        return row_transforms::invalid;

    const auto width = static_cast<unsigned>(matrix.columns);
#if defined(__HIPCC__)
    const dense_linear::View weights{matrix.bytes.data(), width, live_outputs, static_cast<std::size_t>(matrix.row_bytes)};
    return for_each_dense_chunk<gemv_rows::max_rows>(rows, hipSuccess, [&](unsigned first, unsigned count) {
        const auto* x = input + std::size_t{first} * width;
        auto* y = output + std::size_t{first} * live_outputs;
        if (count == 1U) detail::launch_q8_tight_rows<1>(weights, x, y, count, stream);
        else if (count <= 2U) detail::launch_q8_tight_rows<2>(weights, x, y, count, stream);
        else if (count <= 4U) detail::launch_q8_tight_rows<4>(weights, x, y, count, stream);
        else if (count <= 8U) detail::launch_q8_tight_rows<8>(weights, x, y, count, stream);
        else detail::launch_q8_tight_rows<16>(weights, x, y, count, stream);
        return hipGetLastError();
    });
#else
    const cuda_core::DenseWeight weights{matrix.bytes.data(), cuda_core::WeightFormat::q8_0, live_outputs, width};
    if (context && rows > cuda_core::kMaxRows && cuda_core::dense_big_supported(weights)) {
        cuda_core::BigAct activation{};
        const auto status = cuda_core::dense_big_split(input, rows, width, weights.format, 3, &activation, stream, context);
        if (status != cudaSuccess) return status;
        return cuda_core::dense_big_gemm(weights, activation, output, nullptr, cuda_core::BigEpi::none, stream, context);
    }
    return for_each_dense_chunk<cuda_core::kMaxRows>(rows, cudaSuccess, [&](unsigned first, unsigned count) {
        return cuda_core::dense_rows(weights, input + std::size_t{first} * width,
            output + std::size_t{first} * live_outputs, count, nullptr, stream, cuda_core::DenseArith::mma3);
    });
#endif
}

inline row_transforms::Status enqueue_resident_q8_softcap_head(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, std::span<float> logits, float cap, row_transforms::Stream stream) {
    if (live_outputs == 0 || rows > logits.size() / live_outputs || !(cap > 0.0F))
        return row_transforms::invalid;
    const auto status = enqueue_resident_q8_rows(matrix, live_outputs, input, rows, logits.data(), stream);
#if defined(__HIPCC__)
    if (status != hipSuccess) return status;
#else
    if (status != cudaSuccess) return status;
#endif
    return enqueue_bf16_head_softcap(rows, live_outputs, logits.data(), cap, stream);
}

}  // namespace strix::core
