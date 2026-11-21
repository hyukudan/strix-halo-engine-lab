#pragma once

#include "dense_flash_prefill.hpp"
#include "masks.hpp"
#include "relative_bias.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::attn_core::dense {

struct PreparedBf16Args {
    const float* queries;              // [query row][query head][dim]
    float* output;                     // [query row][query head][dim]
    const std::uint16_t* keys;         // [cache slot][KV head][dim]
    const std::uint16_t* values;       // [cache slot][KV head][dim]
    const std::uint16_t* zero_padding; // One zero BF16 head row for masked tile loads.
    unsigned* failure;
    unsigned rows;
    unsigned kv_heads;
    unsigned first_position;
    unsigned window;                   // Zero means full causal attention.
    unsigned cache_slots;
    bool wrapped;                      // Wrapped slot = absolute position % cache_slots.
    RelativeBiasRows relative_bias;
    AttentionSinkRows sinks;
};

// FAST attention over prepared operands. The binding supplies kScaleLog2.
// All rows visible to the chunk must be resident before launch. Align Q, K, V,
// zero padding and any Split-K workspace to 16 bytes.
template <unsigned Dim, unsigned Group, bool WithBias, bool WithSink, unsigned Split = 1, unsigned Positions = 16>
struct PreparedBf16Policy {
    static_assert(Dim > 0 && Group > 0 && Positions > 0);
    using Args = PreparedBf16Args;
    using Shape = dense::Shape<Dim, Group, Split, Positions>;
    static constexpr auto kOp = tile::Operand::bf16;
    static constexpr bool kValueRows = true;
    static constexpr bool kRebaseSink = WithSink;

    __host__ __device__ static bool valid(const Args& a) {
        constexpr unsigned last_tile_end = UINT_MAX - 15U;
        if (!a.queries || !a.output || !a.keys || !a.values || !a.zero_padding || !a.failure ||
            a.rows == 0 || a.rows > UINT_MAX - (Positions - 1U) || a.kv_heads == 0 ||
            a.kv_heads > UINT_MAX / Group || a.first_position > last_tile_end ||
            a.rows > last_tile_end - a.first_position) return false;
        if constexpr (WithBias) {
            if (a.relative_bias.extent != 0 &&
                (!a.relative_bias.logits || a.relative_bias.heads != a.kv_heads * Group)) return false;
        }
        if constexpr (WithSink) {
            if (!a.sinks.logits || a.sinks.heads != a.kv_heads * Group) return false;
        }
        const unsigned end = a.first_position + a.rows;
        const unsigned first_visible = mask::window_lo(a.first_position, a.window);
        return a.wrapped ? a.window != 0 && a.cache_slots >= end - first_visible : a.cache_slots >= end;
    }

    __host__ __device__ static unsigned rows(const Args& a) { return a.rows; }
    __device__ static unsigned position(const Args& a) { return a.first_position; }
    __device__ static unsigned window(const Args& a) { return a.window; }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
    __device__ static const float* query(const Args& a, unsigned row, unsigned head) {
        return a.queries + query_offset(a, row, head);
    }
    __device__ static float* out(const Args& a, unsigned row, unsigned head) {
        return a.output + query_offset(a, row, head);
    }
    __device__ static const std::uint16_t* key(const Args& a, unsigned pos, unsigned head) {
        return cache_row(a, a.keys, pos, head);
    }
    __device__ static const std::uint16_t* value(const Args& a, unsigned pos, unsigned head) {
        return cache_row(a, a.values, pos, head);
    }
    __device__ static float score_bias(const Args& a, unsigned row, unsigned head, unsigned p, unsigned k)
        requires(WithBias) {
        return a.relative_bias.score_bias(row, head, p, k);
    }
    __device__ static float sink(const Args& a, unsigned row, unsigned head) requires(WithSink) {
        return a.sinks.sink(row, head);
    }

private:
    __device__ static std::size_t query_offset(const Args& a, unsigned row, unsigned head) {
        return (std::size_t(row) * a.kv_heads * Group + head) * Dim;
    }
    __device__ static const std::uint16_t* cache_row(const Args& a, const std::uint16_t* cache,
                                                    unsigned pos, unsigned head) {
        if (pos < mask::window_lo(a.first_position, a.window) || pos >= a.first_position + a.rows) {
            return a.zero_padding;
        }
        const unsigned slot = a.wrapped ? pos % a.cache_slots : pos;
        return cache + (std::size_t(slot) * a.kv_heads + head) * Dim;
    }
};

template <class Policy>
bool can_launch_prepared_bf16(const PreparedBf16Args& args, std::size_t workspace_size, unsigned splits) {
    if (!Policy::valid(args) || splits == 0) return false;
    using Shape = typename Policy::Shape;
    const unsigned heads = args.kv_heads * Shape::group;
    return splits == 1U || workspace_size / splits / heads / (Shape::dim + 2U) >= args.rows;
}

template <class Policy>
hipError_t launch_prepared_bf16(const PreparedBf16Args& args, hipStream_t stream,
                                std::span<float> workspace = {}, unsigned splits = 1) {
    if (!can_launch_prepared_bf16<Policy>(args, workspace.size(), splits)) return hipErrorInvalidValue;
    using Shape = typename Policy::Shape;
    const unsigned heads = args.kv_heads * Shape::group;
    return launch<Policy, true>(args, heads, stream, workspace.data(), splits);
}

}  // namespace strix::attn_core::dense
