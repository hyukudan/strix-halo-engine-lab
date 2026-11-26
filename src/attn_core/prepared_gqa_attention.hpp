#pragma once

#include "strix/exec/descriptor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/pointwise.hpp"
#include "strix/attn/cache_rows.hpp"
#endif

namespace strix::attn_core {

struct PreparedGqaAttentionRows {
    const float* queries;          // [row][query head][dim] FP32
    const float* keys;             // [row][KV head][dim] FP32
    const float* values;           // [row][KV head][dim] FP32
    const float* relative_logits;  // [row][query head][backward distance] FP32, natural units
    const float* sinks;            // [row][query head] FP32, natural units
    float* output;                 // [row][query head][dim] FP32
    unsigned first_position;
    unsigned count;
};

struct PreparedStepFinite {
    const exec::StepBlock* step;
    PreparedGqaAttentionRows rows;
    unsigned heads, kv_heads, width, relative_extent;
    unsigned* failure;  // One sticky failure bit per row, initialized by the caller.
};

struct FiniteScan {
    unsigned stride;
    std::size_t elements;
};

// Each query head visits its full image, including masked relative logits.
constexpr FiniteScan prepared_finite_scan(unsigned rows, unsigned heads, unsigned width, unsigned relative_extent) {
    const unsigned stride = std::max(width, relative_extent);
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (stride == 0U || heads == 0U || rows > limit / stride / heads) return {};
    return {stride, std::size_t{rows} * heads * stride};
}

// Rebind the prepared buffers to consecutive launches with a smaller query
// window. Zero keeps the original window. Buffer row strides are unchanged.
template <class Shape, class Status, class Launch>
inline Status for_each_prepared_gqa_window(const PreparedGqaAttentionRows& rows, unsigned window_rows,
                                           Status success, Launch launch) {
    if (window_rows == 0U || window_rows >= rows.count) return launch(rows);
    for (unsigned first = 0; first < rows.count;) {
        const auto count = std::min(window_rows, rows.count - first);
        const auto query = std::size_t{first} * Shape::kQueryWidth;
        const auto kv = std::size_t{first} * Shape::kKvWidth;
        const auto heads = std::size_t{first} * Shape::kQueryHeads;
        const auto status = launch(PreparedGqaAttentionRows{
            rows.queries + query, rows.keys + kv, rows.values + kv,
            rows.relative_logits ? rows.relative_logits + heads * Shape::kRelativeExtent : nullptr,
            rows.sinks ? rows.sinks + heads : nullptr, rows.output + query, rows.first_position + first, count,
        });
        if (status != success) return status;
        first += count;
    }
    return success;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace detail {

struct FiniteAttentionOperands {
    PreparedGqaAttentionRows rows;
    unsigned heads;
    unsigned kv_heads;
    unsigned width;
    unsigned bias_extent;
    unsigned stride;
    unsigned* failure;

    __device__ void operator()(std::size_t i) const {
        namespace cr = strix::attn::cache_rows;
        const auto query_head = i / stride;
        const auto column = i % stride;
        bool valid = column != 0U || !rows.sinks || cr::finite(rows.sinks[query_head]);
        if (column < width) {
            const auto q = query_head * width + column;
            valid = valid && cr::encodes_finite(rows.queries[q]) && cr::finite(rows.output[q]);
            const auto head = query_head % heads;
            if (head < kv_heads) {
                const auto kv = (query_head / heads * kv_heads + head) * width + column;
                valid = valid && cr::encodes_finite(rows.keys[kv]) && cr::encodes_finite(rows.values[kv]);
            }
        }
        if (column < bias_extent) valid = valid && cr::finite(rows.relative_logits[query_head * bias_extent + column]);
        if (!valid) atomicOr(failure, 1U);
    }
};

#if defined(__CUDACC__) && !defined(__HIPCC__)
template <unsigned Rows>
struct FiniteStepOperands {
    const exec::StepBlock* step;
    FiniteAttentionOperands operands;

    __device__ void operator()(std::size_t i) const {
        const auto row = static_cast<unsigned>(i / (std::size_t{operands.heads} * operands.stride));
        if (row >= Rows || !exec::step_row_valid(*step, row)) return;
        auto check = operands;
        check.failure += row;
        check(i);
    }
};
#endif

}  // namespace detail

// Check all operands, including relative logits omitted by the causal mask.
// The caller has already admitted the attention geometry and enqueued the output.
template <class Stream>
inline auto enqueue_validate_prepared_finite(const PreparedGqaAttentionRows& rows, unsigned heads,
                                              unsigned kv_heads, unsigned width, unsigned bias_extent,
                                              unsigned* failure, Stream stream) {
    const auto scan = prepared_finite_scan(rows.count, heads, width, bias_extent);
    if (scan.elements == 0U) return core::pointwise::invalid;
    return core::pointwise::elements(scan.elements,
        detail::FiniteAttentionOperands{rows, heads, kv_heads, width, bias_extent, scan.stride, failure}, stream);
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
template <unsigned Rows>
inline core::pointwise::Status enqueue_prepared_step_finite(const PreparedStepFinite& args,
                                                          core::pointwise::Stream stream) {
    const auto scan = prepared_finite_scan(Rows, args.heads, args.width, args.relative_extent);
    if (!args.step || !args.failure || args.rows.count != Rows || scan.elements == 0U ||
        !args.heads || !args.kv_heads || args.kv_heads > args.heads || !args.width ||
        !args.rows.queries || !args.rows.keys || !args.rows.values || !args.rows.output ||
        (args.relative_extent && !args.rows.relative_logits)) return core::pointwise::invalid;
    return core::pointwise::elements(scan.elements, detail::FiniteStepOperands<Rows>{args.step,
        {args.rows, args.heads, args.kv_heads, args.width, args.relative_extent, scan.stride, args.failure}}, stream);
}
#endif
#endif

}  // namespace strix::attn_core
