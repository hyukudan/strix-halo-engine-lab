#pragma once

#include "cache_rows.hpp"
#include "dense_flash_prefill.hpp"
#include "masks.hpp"
#include "prepared_bf16_lanes_plan.hpp"
#include "prepared_gqa_attention.hpp"
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

struct PreparedBf16Cache {
    std::uint16_t* keys;    // [slot][KV head][dim] BF16
    std::uint16_t* values;  // [slot][KV head][dim] BF16
    const std::uint16_t* zero_padding;  // One zero BF16 head row for masked tile loads.
    unsigned slots;
    bool wrapped;
};

// FAST attention over prepared operands. The binding supplies kScaleLog2.
// All rows visible to the chunk must be resident before launch. Align Q, K, V,
// zero padding and any Split-K workspace to 16 bytes.
template <unsigned Dim, unsigned Group, bool WithBias, bool WithSink, unsigned Split = 1, unsigned Positions = 16,
          bool Bidirectional = false>
struct PreparedBf16Policy {
    static_assert(Dim > 0 && Group > 0 && Positions > 0);
    static_assert(!Bidirectional || (!WithBias && !WithSink));
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
        const unsigned first_visible = visible_begin(a);
        return a.wrapped ? a.window != 0 && a.cache_slots >= end - first_visible : a.cache_slots >= end;
    }

    __host__ __device__ static unsigned rows(const Args& a) { return a.rows; }
    __device__ static unsigned position(const Args& a) { return a.first_position; }
    __device__ static unsigned window(const Args& a) { return a.window; }
    __host__ __device__ static mask::Range key_range(const Args& a) requires(Bidirectional) {
        return mask::block_range(a.first_position, a.rows, a.window);
    }
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
    __host__ __device__ static unsigned visible_begin(const Args& a) {
        if constexpr (Bidirectional) return key_range(a).begin;
        else return mask::window_lo(a.first_position, a.window);
    }

    __device__ static std::size_t query_offset(const Args& a, unsigned row, unsigned head) {
        return (std::size_t(row) * a.kv_heads * Group + head) * Dim;
    }
    __device__ static const std::uint16_t* cache_row(const Args& a, const std::uint16_t* cache,
                                                    unsigned pos, unsigned head) {
        if (pos < visible_begin(a) || pos >= a.first_position + a.rows) {
            return a.zero_padding;
        }
        const unsigned slot = a.wrapped ? pos % a.cache_slots : pos;
        return cache + (std::size_t(slot) * a.kv_heads + head) * Dim;
    }
};

// The relative bias projected inside the flash (dense::DenseProjectedBias) instead of read from a
// [row][head][distance] image. Base is a PreparedBf16Policy without the image bias (WithBias false); its descriptor
// is the incumbent one plus the projection (PreparedBf16Args and the policies that use it are unchanged). Same
// logits and the same failure bit as the image and its producer (dense_flash_prefill.hpp).
struct PreparedBf16ProjectedArgs : PreparedBf16Args {
    RelativeBiasProjection relative_projection;
};

template <class Base, unsigned Width, unsigned MaxExtent>
struct PreparedBf16ProjectedBias : Base {
    static_assert(!dense::DenseScoreBias<Base>, "the base policy reads no bias image");
    using Args = PreparedBf16ProjectedArgs;
    static constexpr unsigned kBiasWidth = Width, kBiasMaxExtent = MaxExtent;

    __host__ __device__ static bool valid(const Args& a) {
        const RelativeBiasProjection& p = a.relative_projection;
        return Base::valid(a) && (p.extent == 0U ||
            (p.features && p.weights && p.extent <= MaxExtent && p.extent % 8U == 0U &&
             p.heads == a.kv_heads * Base::Shape::group &&
             reinterpret_cast<std::uintptr_t>(p.features) % 16U == 0U &&
             reinterpret_cast<std::uintptr_t>(p.weights) % 16U == 0U));
    }
    __device__ static RelativeBiasProjection bias_projection(const Args& a) { return a.relative_projection; }
};

// Replay keeps the eager tile partition and merge order. The captured grid
// covers the maximum visible prefix and excess split blocks do no work.
template <class Policy>
struct PreparedBf16StepPolicy : Policy {
    struct Args : PreparedBf16Args {
        const std::uint32_t* device_position;
    };

    __device__ static unsigned bind_device_step(Args& args) {
        args.first_position = *args.device_position;
        return small_row_key_splits(args.first_position, args.rows, args.window);
    }
};

template <class Policy>
bool can_launch_prepared_bf16(const typename Policy::Args& args, std::size_t workspace_size, unsigned splits) {
    if (!Policy::valid(args) || splits == 0) return false;
    using Shape = typename Policy::Shape;
    const unsigned heads = args.kv_heads * Shape::group;
    return splits == 1U || workspace_size / splits / heads / (Shape::dim + 2U) >= args.rows;
}

template <class Policy>
hipError_t launch_prepared_bf16(const typename Policy::Args& args, hipStream_t stream,
                                std::span<float> workspace = {}, unsigned splits = 1) {
    if (!can_launch_prepared_bf16<Policy>(args, workspace.size(), splits)) return hipErrorInvalidValue;
    using Shape = typename Policy::Shape;
    const unsigned heads = args.kv_heads * Shape::group;
    return launch<Policy, true>(args, heads, stream, workspace.data(), splits);
}

template <class Policy>
PreparedBf16Args bind_prepared_bf16(unsigned kv_heads, unsigned window, unsigned relative_extent,
                                    const PreparedGqaAttentionRows& rows, const PreparedBf16Cache& cache,
                                    unsigned* failure) {
    const unsigned heads = kv_heads * Policy::Shape::group;
    return {rows.queries, rows.output, cache.keys, cache.values, cache.zero_padding, failure,
            rows.count, kv_heads, rows.first_position, window, cache.slots, cache.wrapped,
            {rows.relative_logits, heads, relative_extent}, {rows.sinks, heads}};
}

// The K/V append and attention share the same cache view and stream. The caller
// owns the committed prefix and must discard or restore the append on failure.
template <class Policy, bool ProducersCheckBiasAndSinks = false>
hipError_t append_prepared_bf16(unsigned kv_heads, unsigned window, unsigned relative_extent,
                                 const PreparedGqaAttentionRows& rows, const PreparedBf16Cache& cache,
                                 unsigned* failure, hipStream_t stream, std::span<float> workspace,
                                 unsigned splits, const std::uint32_t* device_position = nullptr) {
    const auto attention = bind_prepared_bf16<Policy>(kv_heads, window, relative_extent, rows, cache, failure);
    if (!can_launch_prepared_bf16<Policy>(attention, workspace.size(), splits)) return hipErrorInvalidValue;

    const unsigned heads = kv_heads * Policy::Shape::group;
    const unsigned bound = window == 0U ? cache.slots : std::min(window, cache.slots);
    const unsigned max_splits = device_position ? small_row_key_splits(bound - 1U, 1U, window) : splits;
    if (device_position && (rows.count != 1U ||
        !can_launch_prepared_bf16<Policy>(attention, workspace.size(), max_splits))) return hipErrorInvalidValue;

    constexpr unsigned dim = Policy::Shape::dim;
    const unsigned width = kv_heads * dim;
    const attn::cache_rows::Rows layout{width, cache.wrapped ? cache.slots : 0U};
    const auto status = attn_core::cache_rows::store_pair(
        {rows.keys, width, device_position, rows.first_position, rows.count, layout, cache.keys, failure},
        {rows.values, width, device_position, rows.first_position, rows.count, layout, cache.values, failure}, stream);
    if (status != hipSuccess) return status;
    using Step = PreparedBf16StepPolicy<Policy>;
    const auto launched = device_position
        ? launch<Step, true>({attention, device_position}, heads, stream, workspace.data(), max_splits)
        : launch_prepared_bf16<Policy>(attention, stream, workspace, splits);
    if (launched != hipSuccess) return launched;
    // Cache stores check K/V. Flash checks Q and output. Unchecked external
    // producers also need a scan for relative logits and sinks hidden by masks.
    if constexpr (!ProducersCheckBiasAndSinks) {
        return enqueue_validate_prepared_finite(rows, kv_heads * Policy::Shape::group, kv_heads, dim,
                                                relative_extent, failure, stream);
    }
    return hipSuccess;
}

// Decode lanes (prepared_bf16_lanes_plan.hpp): lane l = blockIdx.x binds its one-row arguments and runs the single-row
// flash body with row block 0 and its own split count; work-groups above that count exit, as the step replay does.
struct PreparedBf16LaneTable {
    PreparedBf16Lane lane[kMaxPreparedBf16Lanes];
};

template <class Policy>
__global__ __launch_bounds__(Policy::Shape::threads) __attribute__((amdgpu_waves_per_eu(waves_per_eu<Policy>())))
void flash_lanes(PreparedBf16Args base, PreparedBf16LaneTable table, float* ws, std::size_t lane_ws) {
    using Shape = typename Policy::Shape;
    const unsigned index = blockIdx.x;
    const auto a = prepared_bf16_lane_args(base, table.lane[index], index, base.kv_heads * Shape::group, Shape::dim);
    const unsigned splits = lane_key_splits(a.first_position, a.window);
    if (blockIdx.z >= splits) return;
    flash_body<Policy>(a, splits > 1U ? ws + index * lane_ws : nullptr, splits, 0U, 0U);
}

template <class Policy>
__global__ __launch_bounds__(Policy::Shape::dim / 4U)
void combine_lanes(PreparedBf16Args base, PreparedBf16LaneTable table, const float* ws, std::size_t lane_ws) {
    using Shape = typename Policy::Shape;
    const unsigned index = blockIdx.x;
    const auto a = prepared_bf16_lane_args(base, table.lane[index], index, base.kv_heads * Shape::group, Shape::dim);
    const unsigned splits = lane_key_splits(a.first_position, a.window);
    if (splits <= 1U) return;
    combine_body<Policy>(a, ws + index * lane_ws, splits, 0U, blockIdx.y, gridDim.y);
}

// append_prepared_bf16<Policy, true> for every lane, in one store, one flash and (when a lane splits) one combine
// launch per kMaxPreparedBf16Lanes lanes. `image` is the packed lane image: lane l at row l of queries, keys, values,
// relative logits, sinks and output (image.count == lanes.size(); image.first_position is unused: every lane is
// bound and checked at its own first_position). The producers
// checked the relative logits and sinks (ProducersCheckBiasAndSinks). Each lane owns lane_workspace_floats of the
// Split-K workspace. Lanes write distinct caches, or one sequence's consecutive rows on one cache whose ring holds
// them (sequence_lane_caches). Every lane's output and cache bits are those of its one-lane
// append (prepared_bf16_lanes_plan.hpp): the cohort width changes the launch count, not the arithmetic.
template <class Policy>
hipError_t append_prepared_bf16_lanes(unsigned kv_heads, unsigned window, unsigned relative_extent,
                                       const PreparedGqaAttentionRows& image, std::span<const PreparedBf16Lane> lanes,
                                       const std::uint16_t* zero_padding, unsigned* failure, hipStream_t stream,
                                       std::span<float> workspace) {
    using Shape = typename Policy::Shape;
    constexpr unsigned dim = Shape::dim;
    const unsigned heads = kv_heads * Shape::group;
    const unsigned width = kv_heads * dim;
    if (lanes.empty()) return hipSuccess;
    if (image.count != lanes.size() || !sequence_lane_caches(lanes, window)) return hipErrorInvalidValue;
    const std::size_t lane_ws = lane_workspace_floats(heads, dim);
    const unsigned per_launch = lanes_per_launch(workspace.size(), heads, dim);
    const unsigned max_splits = max_lane_key_splits(lanes, window);
    const unsigned step = max_splits > 1U ? per_launch : kMaxPreparedBf16Lanes;
    if (step == 0U) return hipErrorInvalidValue;

    PreparedGqaAttentionRows first = image;
    first.count = 1U;
    const auto bind_base = [&](std::size_t lane) {
        first.queries = image.queries + lane * heads * dim;
        first.output = image.output + lane * heads * dim;
        first.relative_logits = image.relative_logits ? image.relative_logits + lane * heads * relative_extent : nullptr;
        first.sinks = image.sinks ? image.sinks + lane * heads : nullptr;
        const auto& l = lanes[lane];
        first.first_position = l.first_position;  // Admission checks this lane's slot, not the image's.
        return bind_prepared_bf16<Policy>(kv_heads, window, relative_extent, first,
                                          {l.keys, l.values, zero_padding, l.cache_slots, l.wrapped != 0U}, failure);
    };
    // Admit every lane before any store.
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
        const unsigned splits = lane_key_splits(lanes[lane].first_position, window);
        if (!can_launch_prepared_bf16<Policy>(bind_base(lane), splits > 1U ? lane_ws : 0U, splits))
            return hipErrorInvalidValue;
    }
    for (std::size_t begin = 0; begin < lanes.size(); begin += step) {
        const auto count = static_cast<unsigned>(std::min<std::size_t>(step, lanes.size() - begin));
        const auto chunk = lanes.subspan(begin, count);
        PreparedBf16LaneTable table{};
        attn_core::cache_rows::LaneStores stores{};
        for (unsigned l = 0; l < count; ++l) {
            table.lane[l] = chunk[l];
            stores.lane[l] = {chunk[l].keys, chunk[l].values, chunk[l].first_position,
                              {width, chunk[l].wrapped ? chunk[l].cache_slots : 0U}};
        }
        const auto base = bind_base(begin);
        auto status = attn_core::cache_rows::store_lanes(image.keys + begin * width, image.values + begin * width,
                                                         width, stores, count, failure, stream);
        if (status != hipSuccess) return status;
        const unsigned splits = max_lane_key_splits(chunk, window);
        float* ws = splits > 1U ? workspace.data() : nullptr;
        STRIX_HIP_LAUNCH_GGL(Rdna3Layout, flash_lanes<Policy>, dim3(count, kv_heads, splits), dim3(Shape::threads), 0,
                             stream, base, table, ws, lane_ws);
        if ((status = hipGetLastError()) != hipSuccess) return status;
        if (splits > 1U) {
            STRIX_HIP_LAUNCH_GGL(Rdna3Layout, combine_lanes<Policy>, dim3(count, heads), dim3(dim / 4U), 0, stream,
                                 base, table, static_cast<const float*>(ws), lane_ws);
            if ((status = hipGetLastError()) != hipSuccess) return status;
        }
    }
    return hipSuccess;
}

// append_prepared_bf16 for a projected-bias policy: prefill rows with one split (no key-split workspace, no
// device position). The bias has no image: the flash computes and flags it, so the finite scan skips it.
template <class Policy, bool ProducersCheckBiasAndSinks = false>
hipError_t append_prepared_bf16_projected(unsigned kv_heads, unsigned window, const PreparedGqaAttentionRows& rows,
                                           const PreparedBf16Cache& cache, RelativeBiasProjection projection,
                                           unsigned* failure, hipStream_t stream) {
    static_assert(DenseProjectedBias<Policy>);
    PreparedBf16ProjectedArgs attention{bind_prepared_bf16<Policy>(kv_heads, window, 0U, rows, cache, failure), {}};
    attention.relative_projection = projection;
    if (!Policy::valid(attention)) return hipErrorInvalidValue;
    constexpr unsigned dim = Policy::Shape::dim;
    const unsigned width = kv_heads * dim;
    const attn::cache_rows::Rows layout{width, cache.wrapped ? cache.slots : 0U};
    const auto status = attn_core::cache_rows::store_pair(
        {rows.keys, width, nullptr, rows.first_position, rows.count, layout, cache.keys, failure},
        {rows.values, width, nullptr, rows.first_position, rows.count, layout, cache.values, failure}, stream);
    if (status != hipSuccess) return status;
    const auto launched = launch<Policy, true>(attention, kv_heads * Policy::Shape::group, stream, nullptr, 1U);
    if (launched != hipSuccess) return launched;
    if constexpr (!ProducersCheckBiasAndSinks) {
        return enqueue_validate_prepared_finite(rows, kv_heads * Policy::Shape::group, kv_heads, dim, 0U, failure,
                                                stream);
    }
    return hipSuccess;
}

}  // namespace strix::attn_core::dense
