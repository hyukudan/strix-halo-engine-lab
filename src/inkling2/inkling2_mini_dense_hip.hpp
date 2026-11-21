#pragma once

#include "inkling2_mini_config.hpp"
#include "core/dense_row_plan.hpp"
#include "ops/norm/rms_bf16_rows.hpp"

#if defined(__HIPCC__)
#include "core/bf16_lm_head.hpp"
#include "core/fma_bf16_rows.hpp"
#include "core/hip_bf16_embedding_rows.hpp"
#include "strix/hip_generation_output.hpp"

#include <stdexcept>
#endif

namespace strix::inkling2 {

using MiniHipDenseProjection = core::Bf16RowProjection;
using MiniHipRmsRows = ops::norm::Bf16RmsRows;
using MiniHipRmsReduction = ops::norm::RmsRowsReduction;

#if defined(__HIPCC__)

inline hipError_t enqueue_mini_hip_dense_rows(const float* input, unsigned input_width, unsigned rows,
                                              std::span<const MiniHipDenseProjection> projections,
                                              hipStream_t stream) {
    return core::fma_rows::enqueue_bf16_rows(input, input_width, rows, projections, kVocabulary, stream);
}

inline hipError_t enqueue_mini_hip_rms(const MiniHipRmsRows& args, hipStream_t stream,
                                      MiniHipRmsReduction reduction = MiniHipRmsReduction::parallel) {
    if (args.rows > kMaxContext) return hipErrorInvalidValue;
    return ops::norm::enqueue_rms_bf16_rows(args, 1.0e-6F, reduction, stream);
}

inline void enqueue_mini_hip_embeddings(std::span<const std::uint16_t> table,
                                        std::span<const std::uint32_t> ids,
                                        std::span<float> output, hipStream_t stream) {
    if (ids.size() > kMaxContext) throw std::invalid_argument("Inkling 2 Mini embedding exceeds context");
    core::enqueue_hip_bf16_embedding_rows(table, kVocabulary, kHiddenWidth, ids, output, stream);
}

// The checkpoint's padded vocabulary rows are excluded from the projection.
inline hipError_t enqueue_mini_hip_head(const float* input, unsigned rows,
                                       std::span<const std::uint16_t> weights,
                                       std::span<float> logits, hipStream_t stream) {
    return core::enqueue_bf16_softcap_head(rows, {kHiddenWidth, kVocabulary, kOutputVocabulary, kLogitSoftcap},
        weights, logits, stream, [&](std::span<const MiniHipDenseProjection> projections) {
            return enqueue_mini_hip_dense_rows(input, kHiddenWidth, rows, projections, stream);
        });
}

inline void enqueue_mini_hip_greedy(std::span<const ExecutionBufferAddress> logits,
                                   ExecutionBufferAddress partials, HipGreedyRowsReply* replies,
                                   hipStream_t stream) {
    enqueue_hip_greedy_rows(logits, kOutputVocabulary, partials, replies, stream);
}

#endif

}  // namespace strix::inkling2
