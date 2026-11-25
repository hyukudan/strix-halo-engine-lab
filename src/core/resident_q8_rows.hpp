#pragma once

#include "core/bf16_lm_head.hpp"
#include "core/resident_row_plan.hpp"

#if defined(__HIPCC__)
#include "device/hip_current_device_arch.hpp"

#include "core/dense_linear.hpp"
#include "core/f16_tile_gemm_small.hpp"
#else
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#endif

namespace strix::core {

#if defined(__HIPCC__)
namespace detail {
template <unsigned Bucket, weight_decode::Format Format = weight_decode::Format::q8_0>
void launch_q8_tight_rows(dense_linear::View weights, const float* input, float* output,
                         unsigned rows, row_transforms::Stream stream) {
    constexpr unsigned tile = Bucket <= 8U ? 2U : 4U;
    constexpr unsigned block_outputs = 4U * tile;
    dense_linear::kernel<Bucket, tile, Format, dense_linear::Epilogue::store_tight_rows>
        <<<(weights.outputs + block_outputs - 1U) / block_outputs, 128, 0, stream>>>(
            weights, weights, input, output, rows);
}
// The low-bit route plan bound to the current device (strix/device/lowbit_rows.hpp: the device gate binds the points
// its declaration states). When no plan is bound, the FP16 form uses the large tile at every row count.
inline LowBitRowPlan current_lowbit_row_plan() {
    int dev = 0;
    return hipGetDevice(&dev) == hipSuccess ? bound_lowbit_row_plan(dev) : LowBitRowPlan{};
}
// Lever "lowbit.rows-select": this launch runs the FP32 sweep (core/resident_row_plan.hpp kLowBitSweepCells).
inline bool lowbit_rows_sweep(LowBitArithmetic arithmetic, weight::FormatId storage, unsigned outputs, unsigned width,
                              unsigned rows) {
    if (arithmetic != LowBitArithmetic::f16_rows_select) return false;
    const unsigned bits = storage == weight::kQ4G64WeightStorage ? 4U : storage == weight::kQ6G64WeightStorage ? 6U : 0U;
    return lowbit_sweep_cell(strix::hip::arch::current_device_arch_name(), bits, outputs, width, rows);
}
// Low-bit projections sharing one input: one launch of the planned route (both routes compute the same bits).
template <class Storage, unsigned Capacity>
row_transforms::Status launch_lowbit_group(const float* input, unsigned rows, unsigned width,
                                           const f16_tile_gemm::Q8ProjectionGroup<Capacity>& group,
                                           row_transforms::Stream stream) {
    const auto plan = current_lowbit_row_plan();
    if (plan_lowbit_rows(rows, plan) == LowBitRoute::small_tile)
        return f16_tile_gemm::small_m::gemm_lowbit<Storage>(
            input, rows, width, group, stream, lowbit_wide_block(rows, group.first_panel[group.count], plan));
    return f16_tile_gemm::gemm_lowbit_projections<Storage>(input, rows, width, group, stream);
}
template <weight_decode::Format Format>
row_transforms::Status launch_lowbit_rows(dense_linear::View weights, const float* input, float* output,
                                         unsigned rows, row_transforms::Stream stream) {
    if (rows == 1U) launch_q8_tight_rows<1, Format>(weights, input, output, rows, stream);
    else if (rows <= 2U) launch_q8_tight_rows<2, Format>(weights, input, output, rows, stream);
    else if (rows <= 4U) launch_q8_tight_rows<4, Format>(weights, input, output, rows, stream);
    else if (rows <= 8U) launch_q8_tight_rows<8, Format>(weights, input, output, rows, stream);
    else launch_q8_tight_rows<16, Format>(weights, input, output, rows, stream);
    return hipGetLastError();
}
}  // namespace detail
#endif

// HIP low-bit resident rows. Selecting f16_group_scale explicitly enables the two WMMA schedules; the candidate
// f16_rows_select runs them too, except in the measured sweep cells (lever "lowbit.rows-select").
inline row_transforms::Status enqueue_resident_lowbit_rows(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, float* output, row_transforms::Stream stream,
    LowBitArithmetic arithmetic = LowBitArithmetic::fp32_sweep) {
    if (!valid_resident_lowbit_rows(matrix, live_outputs, input, rows, output, arithmetic)) return row_transforms::invalid;
#if defined(__HIPCC__)
    const auto width = static_cast<unsigned>(matrix.columns);
    if (arithmetic == LowBitArithmetic::fp32_sweep ||
        detail::lowbit_rows_sweep(arithmetic, matrix.storage, live_outputs, width, rows)) {
        const dense_linear::View weights{matrix.bytes.data(), width, live_outputs, static_cast<std::size_t>(matrix.row_bytes)};
        return matrix.storage == weight::kQ4G64WeightStorage
                   ? detail::launch_lowbit_rows<weight_decode::Format::q4_g64>(weights, input, output, rows, stream)
                   : detail::launch_lowbit_rows<weight_decode::Format::q6_g64>(weights, input, output, rows, stream);
    }
    f16_tile_gemm::Q8ProjectionGroup<1> group{};
    group.append({matrix.bytes.data(), live_outputs, output});
    return matrix.storage == weight::kQ4G64WeightStorage
               ? detail::launch_lowbit_group<weight::Q4G64Storage>(input, rows, width, group, stream)
               : detail::launch_lowbit_group<weight::Q6G64Storage>(input, rows, width, group, stream);
#else
    (void)stream;
    return row_transforms::invalid;
#endif
}

// Bind compact resident weights to the native dense family. Only live output
// rows are projected. The caller owns the activation/output BF16 boundaries.
// CUDA callers can supply a reserved scratch context for aligned tall shapes.
inline row_transforms::Status enqueue_resident_q8_rows(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, float* output, row_transforms::Stream stream
#if !defined(__HIPCC__)
    , cuda_core::BigCtx* context = nullptr
#endif
    , LowBitArithmetic arithmetic = LowBitArithmetic::fp32_sweep
) {
    if (matrix.storage == weight::kQ4G64WeightStorage || matrix.storage == weight::kQ6G64WeightStorage)
        return enqueue_resident_lowbit_rows(matrix, live_outputs, input, rows, output, stream, arithmetic);
    if (!valid_resident_q8_rows(matrix, live_outputs, input, rows, output)) return row_transforms::invalid;

    const auto width = static_cast<unsigned>(matrix.columns);
#if defined(__HIPCC__)
    if (use_resident_q8_tile(rows)) {
        f16_tile_gemm::Q8ProjectionGroup<1> group{};
        group.append({matrix.bytes.data(), live_outputs, output});
        return f16_tile_gemm::gemm_q8_projections(input, rows, width, group, stream);
    }
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

// The grouped contract of a compact list, checked before anything is enqueued: one matrix row per output and the
// storage's own validator per descriptor (low-bit storages on HIP only: the CUDA dense family has no decoder for
// them), and the whole list's byte ranges. enqueue_resident_q8_projections runs exactly this check; a mixed list
// (dispatch_resident_rows) runs it for every compact run before its first launch.
template <unsigned Capacity>
inline bool resident_q8_projections_admitted(const float* input, unsigned width, unsigned rows,
                                             std::span<const ResidentRowProjection> projections,
                                             [[maybe_unused]] LowBitArithmetic arithmetic = LowBitArithmetic::fp32_sweep) {
    if (projections.empty() || projections.size() > Capacity) return false;
    arithmetic = grouped_lowbit_arithmetic(arithmetic);
    for (const auto& projection : projections) {
        const auto* matrix = std::get_if<weight::ResidentMatrixView>(&projection.weights);
        if (!matrix || matrix->columns != width || matrix->rows != projection.outputs) return false;
        const bool lowbit = matrix->storage == weight::kQ4G64WeightStorage || matrix->storage == weight::kQ6G64WeightStorage;
#if defined(__HIPCC__)
        if (lowbit ? !lowbit_f16_form(arithmetic) ||
                         !valid_resident_lowbit_rows(*matrix, projection.outputs, input, rows, projection.output, arithmetic)
                   : !valid_resident_q8_rows(*matrix, projection.outputs, input, rows, projection.output))
            return false;
#else
        if (lowbit || !valid_resident_q8_rows(*matrix, projection.outputs, input, rows, projection.output)) return false;
#endif
    }
    return resident_list_ranges_disjoint(input, width, rows, projections);
}

// Resident linear projections have a common input and no stored output padding.
// The standalone matrix entry also supports projecting a live prefix (e.g. the LM head).
template <unsigned Capacity>
inline row_transforms::Status enqueue_resident_q8_projections(
    const float* input, unsigned width, unsigned rows, std::span<const ResidentRowProjection> projections,
    row_transforms::Stream stream
#if !defined(__HIPCC__)
    , cuda_core::BigCtx* context = nullptr
#endif
    , LowBitArithmetic arithmetic = LowBitArithmetic::fp32_sweep
) {
    // Every descriptor is checked before submission. A live prefix uses the standalone entry.
    if (!resident_q8_projections_admitted<Capacity>(input, width, rows, projections, arithmetic))
        return row_transforms::invalid;
#if defined(__HIPCC__)
    constexpr auto success = hipSuccess;
    // Runs of consecutive projections of one storage, in any order of storages: a low-bit run is one launch of its
    // route, a Q8_0 run above the sweep one tile launch, a Q8_0 run at or below the sweep one sweep per projection.
    for (std::size_t i = 0; i < projections.size();) {
        const auto storage = std::get<weight::ResidentMatrixView>(projections[i].weights).storage;
        std::size_t end = i + 1U;
        while (end < projections.size() && std::get<weight::ResidentMatrixView>(projections[end].weights).storage == storage)
            ++end;
        const auto run = projections.subspan(i, end - i);
        i = end;
        const bool q4 = storage == weight::kQ4G64WeightStorage;
        const bool tile = q4 || storage == weight::kQ6G64WeightStorage || use_resident_q8_tile(rows);
        if (tile) {
            f16_tile_gemm::Q8ProjectionGroup<Capacity> group{};
            for (const auto& projection : run) {
                const auto& matrix = std::get<weight::ResidentMatrixView>(projection.weights);
                group.append({matrix.bytes.data(), projection.outputs, projection.output});
            }
            row_transforms::Status status;
            if (q4) status = detail::launch_lowbit_group<weight::Q4G64Storage>(input, rows, width, group, stream);
            else if (storage == weight::kQ6G64WeightStorage)
                status = detail::launch_lowbit_group<weight::Q6G64Storage>(input, rows, width, group, stream);
            else status = f16_tile_gemm::gemm_q8_projections(input, rows, width, group, stream);
            if (status != success) return status;
        } else {
            for (const auto& projection : run) {
                const auto status = enqueue_resident_q8_rows(std::get<weight::ResidentMatrixView>(projection.weights),
                                                             projection.outputs, input, rows, projection.output, stream);
                if (status != success) return status;
            }
        }
    }
    return success;
#else
    constexpr auto success = cudaSuccess;
    for (const auto& projection : projections) {
        const auto status = enqueue_resident_q8_rows(std::get<weight::ResidentMatrixView>(projection.weights),
                                                     projection.outputs, input, rows, projection.output, stream,
                                                     context);
        if (status != success) return status;
    }
    return success;
#endif
}

inline row_transforms::Status enqueue_resident_q8_softcap_head(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, std::span<float> logits, float cap, row_transforms::Stream stream,
    LowBitArithmetic arithmetic = LowBitArithmetic::fp32_sweep) {
    if (live_outputs == 0 || rows > logits.size() / live_outputs || !(cap > 0.0F))
        return row_transforms::invalid;
    const auto status = enqueue_resident_q8_rows(matrix, live_outputs, input, rows, logits.data(), stream
#if !defined(__HIPCC__)
                                               , nullptr
#endif
                                               , arithmetic);
#if defined(__HIPCC__)
    if (status != hipSuccess) return status;
#else
    if (status != cudaSuccess) return status;
#endif
    return enqueue_bf16_head_softcap(rows, live_outputs, logits.data(), cap, stream);
}

// Rows of one launch of the resident Q8_0 rows family that keep every row on its one-row arithmetic. HIP: the tight
// rows sweep up to gemv_rows::max_rows (each row's accumulators visit their chunks in the same ascending order for any
// row bucket; more rows switch to the F16 tile). CUDA: one row (the per-row identity of its multi-row dense rows is
// not established).
#if defined(__HIPCC__)
inline constexpr unsigned kResidentQ8RowBitsRows = gemv_rows::max_rows;
#else
inline constexpr unsigned kResidentQ8RowBitsRows = 1U;
#endif

// Softcapped score rows of a compact Q8_0 head at a row pitch (cohort lanes, verify windows), the Q8_0 form of
// enqueue_bf16_softcap_head_rows: row r lands at head::logit_row_offset(r, pitch). Each launch projects `pitch`
// outputs per row (the live outputs plus stored pad rows nobody reads; live <= pitch <= stored head rows) for at most
// kResidentQ8RowBitsRows rows, so the head is read once per launch instead of once per row and every live score keeps
// the bits of the one-row head.
inline row_transforms::Status enqueue_resident_q8_softcap_head_rows(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, unsigned pitch, const float* input,
    unsigned rows, std::span<float> logits, float cap, row_transforms::Stream stream,
    LowBitArithmetic arithmetic = LowBitArithmetic::fp32_sweep) {
    if (matrix.columns == 0U || !(cap > 0.0F) ||
        !head::pitched_rows_fit(rows, live_outputs, pitch, matrix.rows, logits.size()))
        return row_transforms::invalid;
#if defined(__HIPCC__)
    constexpr auto success = hipSuccess;
#else
    constexpr auto success = cudaSuccess;
#endif
    for (unsigned first = 0; first < rows;) {
        const auto launch = static_cast<unsigned>(head::logit_rows_launch(rows, first, kResidentQ8RowBitsRows).rows);
        const auto status = enqueue_resident_q8_softcap_head(
            matrix, pitch, input + std::size_t{first} * matrix.columns, launch,
            logits.subspan(head::logit_row_offset(first, pitch), std::size_t{launch} * pitch), cap, stream, arithmetic);
        if (status != success) return status;
        first += launch;
    }
    return success;
}

}  // namespace strix::core
