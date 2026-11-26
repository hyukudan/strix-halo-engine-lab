#pragma once

#include "row_plan.hpp"
#include "cuda/core/dense/bf16_row_abi.hpp"
#include "core/dense_row_plan.hpp"
#include "cuda/core/lookup_cuda.cuh"
#include "strix/core/rows.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/weight/registry.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <stdexcept>

namespace strix::cuda_core {

namespace detail {

// Activation planes of the tall BF16 split: mma3 3 (exact), mma2 2, mma1 1
// (FAST); 0 for fp32, which has no tall MMA path. A BF16-exact operand needs one
// plane under every class (core::exact_bf16_planes: same bits as the request).
constexpr int bf16_tall_terms(DenseArith arithmetic, core::Bf16Operand operand = core::Bf16Operand::fp32) {
    const int planes = core::exact_bf16_planes(operand);
    switch (arithmetic) {
    case DenseArith::mma3: return planes;
    case DenseArith::mma2: return planes < 2 ? planes : 2;
    case DenseArith::mma1: return 1;
    case DenseArith::fp32: break;
    }
    return 0;
}

}  // namespace detail

// One short MMA3 dispatch across independent groups, with at most eight
// projections in total. Outputs must be disjoint from all other operands.
// Keep each original group's unsupported shapes on its incumbent FP32 route.
inline cudaError_t enqueue_bf16_dense_row_groups(std::span<const Bf16DenseRowGroup> groups, unsigned rows,
                                                 const device::Profile& profile, cudaStream_t stream) {
    namespace plan_rows = core::rows;
    if (rows == 0 || rows > kMaxRows || groups.empty() || groups.size() > plan_rows::kMaxJobs || profile.units == 0)
        return cudaErrorInvalidValue;

    std::array<plan_rows::Job, plan_rows::kMaxJobs> jobs;
    std::size_t total = 0, job_count = 0;
    for (const auto& group : groups) {
        if (!detail::valid_bf16_dense_row_group(group, rows, true)) return cudaErrorInvalidValue;
        total += group.projections.size();
        if (total > plan_rows::kMaxJobs) return cudaErrorInvalidValue;
        for (const auto& projection : group.projections) {
            if (!detail::bf16_mma3_tile(group.input_width, projection.outputs)) continue;
            const auto epilogue = projection.round_bf16 ? kRowsRoundBf16 : plan_rows::ChainId{0};
            jobs[job_count++] = {{projection.weights, weight::kBf16WeightStorage}, projection.outputs,
                                 group.input_width, {group.input}, {projection.output}, epilogue, {}};
        }
    }

    const auto enqueue_fp32 = [&] {
        for (const auto& group : groups) {
            std::array<DenseWeight, kMaxBf16RowProjections> weights;
            std::array<float*, kMaxBf16RowProjections> outputs;
            int count = 0;
            for (const auto& projection : group.projections) {
                if (detail::bf16_mma3_tile(group.input_width, projection.outputs)) continue;
                weights[count] = {projection.weights, WeightFormat::bf16, projection.outputs, group.input_width};
                outputs[count++] = projection.output;
            }
            if (count == 0) continue;
            const auto status = dense_rows_multi(weights.data(), outputs.data(), count, group.input, rows, nullptr,
                                                 stream, DenseArith::fp32);
            if (status != cudaSuccess) return status;
        }
        return cudaSuccess;
    };

    if (job_count == 0) return enqueue_fp32();
    try {
        const auto plan = plan_rows::plan({std::span(jobs).first(job_count), rows, 0, plan_rows::Packing::fewest_launches},
                                          profile);
        if (const auto status = enqueue_fp32(); status != cudaSuccess) return status;
        return dense_mma3_rows_grouped(plan, stream);
    } catch (const std::invalid_argument&) {
        return cudaErrorInvalidValue;
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    }
}

// Tall MMA projections share a split activation with the arithmetic's term
// count (mma3 3, mma2 2, mma1 1), one plane when the caller declares the input
// BF16-exact (same bits; the rounded epilogue then runs on the one-plane chain).
// Unsupported MMA shapes can use the same FP32 fallback across a grid of row
// windows. A caller with a device profile may bind short MMA3 groups to the
// shared rows planner. `planes_report`, when given, is reset to 0 on entry and, after a successful dispatch, receives the
// activation planes of the tall split that actually ran: the term count when at least one projection took the tall MMA
// path (a dispatch mixing it with the FP32 grid or short rows reports the split it ran), 0 when none did (every
// projection took the FP32 grid, the short-row path or the device-profile rows planner, which split no activation here)
// and after a failure. A caller never reads a stale or requested count.
inline cudaError_t enqueue_bf16_dense_rows(const float* input, unsigned input_width, unsigned rows,
                                           std::span<const core::Bf16RowProjection> projections,
                                           cudaStream_t stream, DenseArith arithmetic, BigCtx* context,
                                           const device::Profile* profile = nullptr,
                                           core::Bf16Operand operand = core::Bf16Operand::fp32,
                                           int* planes_report = nullptr) {
    if (planes_report) *planes_report = 0;
    const Bf16DenseRowGroup group{input, input_width, projections};
    if (profile && arithmetic == DenseArith::mma3 && rows <= kMaxRows)
        return enqueue_bf16_dense_row_groups(std::span(&group, 1), rows, *profile, stream);
    if (!detail::valid_bf16_dense_row_group(group, rows, arithmetic == DenseArith::mma3)) return cudaErrorInvalidValue;

    const bool round_bf16 = projections.front().round_bf16;
    std::array<DenseWeight, kMaxBf16RowProjections> short_weights{};
    std::array<float*, kMaxBf16RowProjections> short_outputs{};
    unsigned short_count = 0;
    BigAct activation{};
    const int terms = detail::bf16_tall_terms(arithmetic, operand);
    int split_planes = 0;
    for (const auto& projection : projections) {
        const DenseWeight weight{projection.weights, WeightFormat::bf16, projection.outputs, input_width};
        constexpr core::Bf16RowLimits limits{kMaxRows, kFp32GridMaxRows, kCompactBf16Outputs, 32U, 16U};
        const auto path = core::plan_bf16_row_projection(input_width, weight.out, rows, context && terms > 0, limits, terms);
        if (path == core::Bf16RowPath::big || path == core::Bf16RowPath::big_compact) {
            if (activation.planes == nullptr) {
                const auto status = dense_big_split(input, rows, input_width, WeightFormat::bf16, terms,
                                                    &activation, stream, context);
                if (status != cudaSuccess) return status;
            }
            split_planes = std::max(split_planes, bf16_split_planes(path, terms));
            const auto epilogue = round_bf16 ? BigEpi::round_bf16 : BigEpi::none;
            const auto status = path == core::Bf16RowPath::big_compact
                ? dense_big_bf16_compact(weight, activation, projection.output, epilogue, stream, context)
                : dense_big_gemm(weight, activation, projection.output, nullptr, epilogue, stream, context);
            if (status != cudaSuccess) return status;
        } else if (path == core::Bf16RowPath::fp32_grid) {
            const auto status = dense_fp32_rows_grid(weight, input, projection.output, rows, stream);
            if (status != cudaSuccess) return status;
        } else {
            short_weights[short_count] = weight;
            short_outputs[short_count++] = projection.output;
        }
    }
    cudaError_t status = cudaSuccess;
    if (short_count != 0) status = core::for_each_dense_chunk<kMaxRows>(rows, cudaSuccess, [&](unsigned first, unsigned count) {
        std::array<float*, kMaxBf16RowProjections> outputs;
        for (unsigned i = 0; i < short_count; ++i)
            outputs[i] = short_outputs[i] + std::size_t{first} * short_weights[i].out;
        if (round_bf16)
            return dense_bf16_mma3_rows_round(short_weights.data(), outputs.data(), static_cast<int>(short_count),
                                              input + std::size_t{first} * input_width, count, stream);
        return dense_rows_multi(short_weights.data(), outputs.data(), static_cast<int>(short_count),
                                input + std::size_t{first} * input_width, count, nullptr, stream, arithmetic);
    });
    if (status == cudaSuccess && planes_report) *planes_report = split_planes;
    return status;
}

// Rows padded to the tall tile (the plane a one-term split stores is [rows_pad][input_width]).
[[nodiscard]] constexpr unsigned bf16_plane_rows_pad(unsigned rows) { return (rows + 127U) / 128U * 128U; }

// True when every projection of a BF16-exact operand takes the tall one-plane path (big; the compact tile is
// three-term only), the path enqueue_bf16_dense_rows_plane runs. A caller that produces the operand plane itself
// (no split) admits first.
[[nodiscard]] inline bool bf16_plane_rows_admitted(unsigned input_width, unsigned rows,
                                                   std::span<const core::Bf16RowProjection> projections,
                                                   const BigCtx* context) {
    if (!context || rows == 0 || rows > std::numeric_limits<unsigned>::max() - 127U ||
        input_width == 0U || input_width % 32U != 0 || projections.empty() ||
        projections.size() > kMaxBf16RowProjections) return false;
    constexpr auto max_plane_elements = std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
    if (input_width > max_plane_elements / bf16_plane_rows_pad(rows)) return false;
    const bool round_bf16 = projections.front().round_bf16;
    for (const auto& projection : projections) {
        if (!projection.weights || !projection.output || projection.round_bf16 != round_bf16 ||
            (round_bf16 && projection.round_separate) || !projection.contiguous(input_width)) return false;
        if (plan_bf16_row_projection(input_width, projection.outputs, rows, kMaxRows, true, 1) != Bf16RowPath::big)
            return false;
    }
    return true;
}

// enqueue_bf16_dense_rows with Bf16Operand::bf16_exact, reading the caller's operand plane instead of splitting an
// FP32 input: plane = [bf16_plane_rows_pad(rows)][input_width] u16, row r = the BF16 encodings of the BF16-exact
// finite values of input row r (the bits the one-term split stores), 16-byte aligned. Rows rows..rows_pad-1 must be
// writable: they are zero-filled here, as the split stores them. Same kernels, same epilogues, same bits; a shape the
// planner sends elsewhere is refused (cudaErrorInvalidValue) before any launch.
inline cudaError_t enqueue_bf16_dense_rows_plane(std::uint16_t* plane, unsigned input_width, unsigned rows,
                                                 std::span<const core::Bf16RowProjection> projections,
                                                 cudaStream_t stream, BigCtx* context) {
    if (!plane || reinterpret_cast<std::uintptr_t>(plane) % 16U != 0U ||
        !bf16_plane_rows_admitted(input_width, rows, projections, context)) return cudaErrorInvalidValue;
    const unsigned rows_pad = bf16_plane_rows_pad(rows);
    if (rows_pad > rows) {
        const auto status = cudaMemsetAsync(plane + std::size_t{rows} * input_width, 0,
                                            std::size_t{rows_pad - rows} * input_width * sizeof(std::uint16_t), stream);
        if (status != cudaSuccess) return status;
    }
    BigAct activation{};
    activation.planes = plane;
    activation.rows = rows;
    activation.rows_pad = rows_pad;
    activation.in = input_width;
    activation.terms = 1;
    const auto epilogue = projections.front().round_bf16 ? BigEpi::round_bf16 : BigEpi::none;
    for (const auto& projection : projections) {
        const DenseWeight weight{projection.weights, WeightFormat::bf16, projection.outputs, input_width};
        const auto status = dense_big_gemm(weight, activation, projection.output, nullptr, epilogue, stream, context);
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

// Device token IDs must already be validated. The gather widens BF16 and
// the in-place RMSNorm applies BF16 gain before its single BF16 output round.
inline cudaError_t enqueue_bf16_embeddings_rmsnorm(const std::uint16_t* table, unsigned vocabulary,
                                                   unsigned width, const std::uint16_t* gain, float epsilon,
                                                   const std::uint32_t* ids, float* output, unsigned rows,
                                                   cudaStream_t stream) {
    if (!table || !gain || !ids || !output || rows == 0 || rows > 65535U || vocabulary == 0 ||
        width == 0 || width > 8192U || !(epsilon > 0.0F)) return cudaErrorInvalidValue;
    namespace lookup = ::strix::cuda::core::lookup;
    lookup::View view{{table, vocabulary, width}, lookup::Form::bf16_to_f32, output, rows};
    view.programmatic = true;
    if (const auto status = lookup::enqueue(view, lookup::TokenIds{ids}, stream); status != cudaSuccess) return status;
    const GroupedView layout{rows, 1U, width};
    return rmsnorm_bf16(output, layout, gain, output, layout, epsilon, stream);
}

}  // namespace strix::cuda_core
