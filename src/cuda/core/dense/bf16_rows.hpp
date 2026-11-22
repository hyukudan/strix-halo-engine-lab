#pragma once

#include "core/dense_row_plan.hpp"
#include "cuda/core/lookup_cuda.cuh"
#include "row_plan.hpp"
#include "strix/core/rows.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <limits>
#include <new>
#include <stdexcept>

namespace strix::cuda_core {

inline constexpr std::size_t kMaxBf16RowProjections = 4;

struct Bf16DenseRowGroup {
    const float* input;
    unsigned input_width;
    std::span<const core::Bf16RowProjection> projections;
};

namespace detail {

inline bool bf16_mma3_tile(unsigned in, unsigned out) {
    return in % core::rows::kBlockIn == 0 && out % core::rows::kTileOut == 0;
}

inline bool valid_bf16_dense_row_group(const Bf16DenseRowGroup& group, unsigned rows, DenseArith arithmetic) {
    const auto& [input, input_width, projections] = group;
    const auto aligned = [](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0;
    };
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!aligned(input) || input_width == 0 || input_width % 8U != 0 || rows == 0 ||
        input_width > max_floats / rows || projections.empty() || projections.size() > kMaxBf16RowProjections)
        return false;

    const bool round_bf16 = projections.front().round_bf16;
    if (round_bf16 && arithmetic != DenseArith::mma3) return false;
    std::uint64_t total_outputs = 0;
    for (const auto& projection : projections) {
        if (projection.round_bf16 != round_bf16 ||
            (round_bf16 && !bf16_mma3_tile(input_width, projection.outputs)) ||
            projection.outputs == 0 || projection.outputs > max_floats / rows ||
            !aligned(projection.weights) || !aligned(projection.output) || projection.output == input ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / input_width)
            return false;
        total_outputs += projection.outputs;
    }
    return total_outputs <= std::numeric_limits<unsigned>::max();
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
        if (!detail::valid_bf16_dense_row_group(group, rows, DenseArith::mma3)) return cudaErrorInvalidValue;
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

// Tall MMA3 projections share a split activation. Unsupported MMA3 shapes can
// use the same FP32 fallback across a grid of row windows. A caller with a
// device profile may bind short MMA3 groups to the shared rows planner.
inline cudaError_t enqueue_bf16_dense_rows(const float* input, unsigned input_width, unsigned rows,
                                           std::span<const core::Bf16RowProjection> projections,
                                           cudaStream_t stream, DenseArith arithmetic, BigCtx* context,
                                           const device::Profile* profile = nullptr) {
    const Bf16DenseRowGroup group{input, input_width, projections};
    if (profile && arithmetic == DenseArith::mma3 && rows <= kMaxRows)
        return enqueue_bf16_dense_row_groups(std::span(&group, 1), rows, *profile, stream);
    if (!detail::valid_bf16_dense_row_group(group, rows, arithmetic)) return cudaErrorInvalidValue;

    const bool round_bf16 = projections.front().round_bf16;
    std::array<DenseWeight, kMaxBf16RowProjections> short_weights{};
    std::array<float*, kMaxBf16RowProjections> short_outputs{};
    unsigned short_count = 0;
    BigAct activation{};
    for (const auto& projection : projections) {
        const DenseWeight weight{projection.weights, WeightFormat::bf16, projection.outputs, input_width};
        const auto path = plan_bf16_row_projection(input_width, weight.out, rows, kMaxRows,
                                                  context && arithmetic == DenseArith::mma3);
        if (path == Bf16RowPath::big || path == Bf16RowPath::big_compact) {
            if (activation.planes == nullptr) {
                const auto status = dense_big_split(input, rows, input_width, WeightFormat::bf16, 3,
                                                    &activation, stream, context);
                if (status != cudaSuccess) return status;
            }
            const auto epilogue = round_bf16 ? BigEpi::round_bf16 : BigEpi::none;
            const auto status = path == Bf16RowPath::big_compact
                ? dense_big_bf16_compact(weight, activation, projection.output, epilogue, stream, context)
                : dense_big_gemm(weight, activation, projection.output, nullptr, epilogue, stream, context);
            if (status != cudaSuccess) return status;
        } else if (path == Bf16RowPath::fp32_grid) {
            const auto status = dense_fp32_rows_grid(weight, input, projection.output, rows, stream);
            if (status != cudaSuccess) return status;
        } else {
            short_weights[short_count] = weight;
            short_outputs[short_count++] = projection.output;
        }
    }
    if (short_count == 0) return cudaSuccess;
    return core::for_each_dense_chunk<kMaxRows>(rows, cudaSuccess, [&](unsigned first, unsigned count) {
        std::array<float*, kMaxBf16RowProjections> outputs;
        for (unsigned i = 0; i < short_count; ++i)
            outputs[i] = short_outputs[i] + std::size_t{first} * short_weights[i].out;
        if (round_bf16)
            return dense_bf16_mma3_rows_round(short_weights.data(), outputs.data(), static_cast<int>(short_count),
                                              input + std::size_t{first} * input_width, count, stream);
        return dense_rows_multi(short_weights.data(), outputs.data(), static_cast<int>(short_count),
                                input + std::size_t{first} * input_width, count, nullptr, stream, arithmetic);
    });
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
